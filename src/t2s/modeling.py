"""Model loading, (Q)LoRA supervised fine-tuning with prompt-masked loss, and batched generation."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def device_info() -> dict:
    import torch

    cuda = torch.cuda.is_available()
    return {"cuda": cuda, "name": torch.cuda.get_device_name(0) if cuda else "cpu",
            "bf16": bool(cuda and torch.cuda.is_bf16_supported()),
            "mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if cuda else 0}


def compute_dtype():
    import torch

    info = device_info()
    if not info["cuda"]:
        return torch.float32
    return torch.bfloat16 if info["bf16"] else torch.float16  # T4 -> fp16


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def disable_incompatible_torchao():
    """Kaggle images ship torchao 0.10. peft >= 0.21 raises ImportError while creating LoRA layers on a
    non-quantised model if it finds a torchao older than 0.16. We never use torchao, so tell peft it's unavailable."""
    import sys

    try:
        from peft.import_utils import is_torchao_available
        is_torchao_available()
        return False
    except ImportError:
        pass
    import peft.tuners.lora  # noqa: F401  (make sure the modules that imported the check are loaded)

    for name, mod in list(sys.modules.items()):
        if name.startswith("peft") and mod is not None and hasattr(mod, "is_torchao_available"):
            setattr(mod, "is_torchao_available", lambda: False)
    print("note: incompatible torchao found -> disabled for peft (not needed here)")
    return True


def load_model(name: str, load_4bit: bool = True, adapter: str | None = None, merge: bool = False):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = compute_dtype()
    kw = {"dtype": dtype}
    if torch.cuda.is_available():
        kw["device_map"] = {"": 0}
    if load_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    if adapter:
        disable_incompatible_torchao()
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        if merge and not load_4bit:
            model = model.merge_and_unload()
    model.eval()
    return model


# ------------------------------------------------------------------------------- data for SFT
def tokenize_sft(tok, messages: list[dict], max_len: int):
    """Tokenise a chat; loss only on the assistant answer (prompt tokens get label -100)."""
    prompt_text = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full_text = tok.apply_chat_template(messages, tokenize=False)
    if not full_text.startswith(prompt_text):
        raise ValueError("chat template: prompt is not a prefix of the full conversation")
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tok(full_text[len(prompt_text):], add_special_tokens=False)["input_ids"]
    ids = prompt_ids + answer_ids
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "labels": [-100] * len(prompt_ids) + answer_ids}


class SFTDataset:
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.records[i]


@dataclass
class Collator:
    """LEFT-pads so that every sequence ends at the same position. The supervised SQL answer then sits in
    the last few positions of the batch, which lets the loss compute logits only there (see AnswerOnlyTrainer)."""
    pad_id: int

    def __call__(self, batch):
        import torch

        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, L - n:] = torch.tensor(b["input_ids"])
            labels[i, L - n:] = torch.tensor(b["labels"])
            att[i, L - n:] = 1
        pos = (att.cumsum(-1) - 1).clamp(min=0)  # correct positions despite left padding
        return {"input_ids": ids, "labels": labels, "attention_mask": att, "position_ids": pos}


def answer_only_loss(model, inputs, num_items_in_batch=None):
    """Causal-LM loss computed ONLY over the trailing answer window.
    A full forward pass would materialise logits of shape batch x seq_len x 152k vocab (several GB in fp32 for
    1.5k-token prompts - the cause of OOM on 16 GB GPUs). With left padding all answers end at the last position,
    so we ask the model for the last K+1 logits only (`logits_to_keep`), where K covers the longest answer."""
    import torch
    import torch.nn.functional as F

    labels = inputs["labels"]
    has = (labels != -100).any(0).nonzero()
    K = int(labels.shape[1] - has.min().item()) if len(has) else 1
    out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                position_ids=inputs.get("position_ids"), logits_to_keep=K + 1, use_cache=False)
    logits = out.logits[:, :-1, :].float()           # predictions for the last K tokens
    target = labels[:, -K:]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-100,
                           reduction="sum")
    n = num_items_in_batch if num_items_in_batch is not None else (target != -100).sum()
    if torch.is_tensor(n):
        n = n.to(loss.device)
    return loss / n


def _trainer_class():
    from transformers import Trainer

    class AnswerOnlyTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = answer_only_loss(model, inputs, num_items_in_batch)
            return (loss, None) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            import torch

            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                loss = answer_only_loss(model, inputs)
            return loss.detach(), None, None

    return AnswerOnlyTrainer


def build_sft_records(tok, chats: list[list[dict]], max_len: int):
    recs, dropped = [], 0
    for m in chats:
        r = tokenize_sft(tok, m, max_len)
        if r is None:
            dropped += 1
        else:
            recs.append(r)
    return recs, dropped


# ------------------------------------------------------------------------------- training
def train_lora(model_name, train_records, eval_records, tok, output_dir, *, load_4bit=True, r=16, alpha=None,
               dropout=0.05, lr=2e-4, epochs=1.0, batch_size=2, grad_accum=8, max_steps=-1, logging_steps=10,
               eval_steps=100, save_steps=100, seed=42, resume=True):
    """QLoRA (4-bit NF4 base + LoRA on all linear layers). Checkpoints go to `output_dir` (put it on Drive
    so a Colab disconnect does not lose progress; training resumes from the last checkpoint)."""
    import glob

    import torch
    disable_incompatible_torchao()
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import TrainingArguments, set_seed

    set_seed(seed)
    model = load_model(model_name, load_4bit=load_4bit)
    model.config.use_cache = False
    if load_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    cfg = LoraConfig(r=r, lora_alpha=alpha or 2 * r, lora_dropout=dropout, target_modules=LORA_TARGETS,
                     bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(train_records) / (batch_size * grad_accum))
    total = max_steps if max_steps > 0 else int(steps_per_epoch * epochs)
    info = device_info()
    args = TrainingArguments(
        output_dir=output_dir, per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum, num_train_epochs=epochs, max_steps=max_steps, learning_rate=lr,
        lr_scheduler_type="cosine", warmup_steps=max(1, int(0.03 * total)), weight_decay=0.0,
        logging_steps=logging_steps, eval_strategy="steps", eval_steps=eval_steps, save_strategy="steps",
        save_steps=save_steps, save_total_limit=2, fp16=info["cuda"] and not info["bf16"], bf16=info["bf16"],
        optim="paged_adamw_8bit" if (load_4bit and info["cuda"]) else "adamw_torch", gradient_checkpointing=False,
        train_sampling_strategy="group_by_length", report_to="none", seed=seed, remove_unused_columns=False, dataloader_num_workers=0,
        max_grad_norm=1.0,
    )
    Trainer = _trainer_class()
    trainer = Trainer(model=model, args=args, train_dataset=SFTDataset(train_records),
                      eval_dataset=SFTDataset(eval_records), data_collator=Collator(tok.pad_token_id))
    ckpt = resume and sorted(glob.glob(f"{output_dir}/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
    trainer.train(resume_from_checkpoint=ckpt[-1] if ckpt else None)
    return trainer


# ------------------------------------------------------------------------------- generation
def generate(model, tok, chats: list[list[dict]], batch_size: int = 16, max_new_tokens: int = 200,
             progress: bool = True, cache_path: str | None = None) -> list[str]:
    """Greedy batched generation; sorts prompts by length to minimise padding; halves the batch on OOM.
    `cache_path` (a .jsonl file) makes it resumable: finished outputs are appended after every batch and
    skipped on the next call, so a crashed kernel loses at most one batch."""
    import json
    import os

    import torch

    tok.padding_side = "left"
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in chats]
    out = [None] * len(texts)
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partially written last line
                if r["i"] < len(out) and r.get("n") == len(texts):
                    out[r["i"]] = r["o"]
        print(f"resuming: {sum(o is not None for o in out)}/{len(texts)} already generated")
    todo = [j for j in np.argsort([-len(t) for t in texts]) if out[j] is None]
    eos = [tok.eos_token_id]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end != tok.unk_token_id and im_end not in eos:
        eos.append(im_end)
    i, bs = 0, batch_size
    try:
        from tqdm.auto import tqdm
        bar = tqdm(total=len(todo), disable=not progress)
    except ImportError:
        bar = None
    cache = open(cache_path, "a") if cache_path else None
    while i < len(todo):
        idx = todo[i:i + bs]
        enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        try:
            with torch.inference_mode():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                     top_p=None, top_k=None, pad_token_id=tok.pad_token_id, eos_token_id=eos,
                                     stop_strings=[";"], tokenizer=tok)  # a SQL statement ends at ';'
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print("OOM -> batch size", bs)
            continue
        dec = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for j, d in zip(idx, dec):
            out[j] = d
            if cache:
                cache.write(json.dumps({"i": int(j), "n": len(texts), "o": d}) + "\n")
        if cache:
            cache.flush()
        i += len(idx)
        if bar:
            bar.update(len(idx))
    if bar:
        bar.close()
    if cache:
        cache.close()
    return out


def pick_fewshot(train_examples, k: int = 3, seed: int = 0, exclude_dbs: set | None = None):
    """k demonstrations from distinct training databases (dev DBs are disjoint from train DBs in Spider)."""
    rng = random.Random(seed)
    pool = [e for e in train_examples if not exclude_dbs or e.db_id not in exclude_dbs]
    rng.shuffle(pool)
    chosen, seen = [], set()
    for e in pool:
        if e.db_id not in seen and 40 < len(e.query) < 160:
            chosen.append(e)
            seen.add(e.db_id)
        if len(chosen) == k:
            break
    return chosen
