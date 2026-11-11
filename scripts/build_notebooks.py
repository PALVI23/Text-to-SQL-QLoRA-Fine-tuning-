"""Generates the Colab notebooks in ../notebooks.   Run: python scripts/build_notebooks.py"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

SETUP = '''# ---------- Setup: run this cell first ----------
import os, sys, subprocess, json, time, warnings
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less GPU memory fragmentation
warnings.filterwarnings("ignore")
SMOKE = os.environ.get("T2S_SMOKE") == "1"      # local CPU smoke test (ignore on Colab/Kaggle)
ON_KAGGLE = os.path.exists("/kaggle/input")
try:
    import google.colab  # noqa: F401
    ON_COLAB = True
except ImportError:
    ON_COLAB = False
from pathlib import Path
if ON_KAGGLE:
    # Code is attached as a Kaggle Dataset (read-only) -> copy it to the writable /kaggle/working
    import glob, shutil, zipfile
    PROJECT_DIR = "/kaggle/working/text2sql"

    def _walk(top="/kaggle/input"):
        return os.walk(top, followlinks=True)   # Kaggle may mount inputs via symlinks

    def _version(code_root):
        import re as _re
        m = _re.search(r"__version__\\s*=\\s*(\\d+)", (Path(code_root) / "src/t2s/__init__.py").read_text())
        return int(m.group(1)) if m else 0

    def _find_code():
        zips, found = [], []
        for root, dirs, files in _walk():
            if root.endswith("src/t2s") and "modeling.py" in files and not root.startswith(PROJECT_DIR):
                found.append(Path(root).parents[1])
        if found:   # several attached versions -> use the newest code
            found.sort(key=_version, reverse=True)
            if len(found) > 1:
                print("several code copies found:", [(str(f), _version(f)) for f in found])
            return found[0]
        for root, dirs, files in _walk():
            zips += [Path(root) / f for f in files if f.lower().endswith(".zip") and "text2sql" in f.lower()]
        for z in zips:   # dataset kept as an un-extracted zip -> extract it ourselves
            tmp = Path("/kaggle/tmp/code"); tmp.mkdir(parents=True, exist_ok=True)
            zipfile.ZipFile(z).extractall(tmp)
            for root, dirs, files in os.walk(tmp):
                if root.endswith("src/t2s") and "modeling.py" in files:
                    return Path(root).parents[1]
        return None

    code_root = _find_code()
    if code_root is None:
        print("Could not find the text2sql code. Contents of /kaggle/input:")
        for root, dirs, files in _walk():
            depth = root.count(os.sep) - 2
            if depth <= 4:
                print("  " * depth + os.path.basename(root) + "/", (files[:5] if files else ""))
        raise RuntimeError("Code dataset not attached. Right panel -> Input -> '+ Add Input' -> Your Work -> Datasets -> "
                           "text2sql-code -> Add. Then run again. (Send the listing above if it is attached.)")
    print("code found at", code_root, "| code version", _version(code_root))
    # Kaggle images ship torchao 0.10, which makes peft refuse to load LoRA adapters on fp16 models. We don't need it.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], capture_output=True)
    shutil.copytree(code_root, PROJECT_DIR, dirs_exist_ok=True,   # always refresh the code
                    ignore=shutil.ignore_patterns("artifacts", "results", "__pycache__"))
    for m in [m for m in list(sys.modules) if m == "t2s" or m.startswith("t2s.")]:
        del sys.modules[m]   # pick up new code if the kernel was not restarted
    # Resume support: if a previous run's Output is attached as an input, reuse its results/artifacts
    for root, dirs, files in _walk():
        if os.path.basename(root) == "results" and os.path.basename(os.path.dirname(root)) == "text2sql":
            prev_root = Path(root).parent
            for sub_dir in ["results", "artifacts", "reports"]:
                if (prev_root / sub_dir).exists():
                    shutil.copytree(prev_root / sub_dir, f"{PROJECT_DIR}/{sub_dir}", dirs_exist_ok=True)
            print("restored previous results from", prev_root)
    if not SMOKE:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{PROJECT_DIR}/requirements-colab.txt"], check=True)
elif SMOKE:
    PROJECT_DIR = os.environ["T2S_PROJECT_DIR"]
elif not ON_COLAB:
    # Local machine (VS Code / Jupyter): the notebook lives in text2sql/notebooks/
    PROJECT_DIR = str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd())
else:
    from google.colab import drive
    drive.mount("/content/drive")
    PROJECT_DIR = "/content/drive/MyDrive/masters-projects/text2sql"   # <- change if needed
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{PROJECT_DIR}/requirements-colab.txt"], check=True)
sys.path.insert(0, f"{PROJECT_DIR}/src")
ART, FIG, RES = f"{PROJECT_DIR}/artifacts", f"{PROJECT_DIR}/reports/figures", f"{PROJECT_DIR}/results"
for d in (ART, FIG, RES): os.makedirs(d, exist_ok=True)

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid"); plt.rcParams["figure.dpi"] = 110
from t2s.spider import download_spider, find_spider_root, load_split, db_path, serialize_schema, schema_stats
from t2s import modeling as MD, evaluate as EV
from t2s.analysis import hardness, HARDNESS_ORDER

# ---- data ----
if SMOKE:
    from t2s.smoke import make_fake_spider
    SPIDER = find_spider_root(make_fake_spider(f"{PROJECT_DIR}/data"))
else:
    if ON_KAGGLE:
        attached = [os.path.join(r, "dev.json") for r, d, f in os.walk("/kaggle/input", followlinks=True)
                    if "dev.json" in f and os.path.isdir(os.path.join(r, "database"))]   # Spider attached as input?
        SPIDER = find_spider_root(Path(attached[0]).parent) if attached else download_spider("/kaggle/working/spider")
    elif ON_COLAB:
        SPIDER = download_spider("/content/spider", drive_zip="/content/drive/MyDrive/masters-projects/spider_data.zip")
    else:
        SPIDER = download_spider(f"{PROJECT_DIR}/data/spider", drive_zip=f"{PROJECT_DIR}/data/spider_data.zip")
train_all, dev = load_split(SPIDER, "train"), load_split(SPIDER, "dev")

# ---- model ----
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
if SMOKE:
    from t2s.smoke import make_tiny_model
    tiny = f"{ART}/tiny-model"
    if not os.path.exists(f"{tiny}/config.json"):
        make_tiny_model(tiny, [e.question + " " + e.query for e in train_all] + [serialize_schema(db_path(SPIDER, e.db_id), "ddl_rows") for e in train_all[:20] + dev])
    BASE_MODEL = tiny
DEV_SUBSET = sorted(np.random.RandomState(0).choice(len(dev), min(300, len(dev)), replace=False))  # fixed subset for ablations
SEED = 42
def savefig(name):
    plt.tight_layout(); plt.savefig(f"{FIG}/{name}.png", dpi=160, bbox_inches="tight"); plt.show()
def save_json(o, name):
    json.dump(o, open(f"{ART}/{name}.json", "w"), indent=2, default=float)
def load_json(name, default=None):
    p = f"{ART}/{name}.json"; return json.load(open(p)) if os.path.exists(p) else default
def cached(name, fn):
    """Run fn() once and store its DataFrame as results/<name>.parquet; later calls (after a crash) just load it."""
    p = f"{RES}/{name}.parquet"
    if os.path.exists(p):
        df = pd.read_parquet(p); print(f"[cached] {name}: EX={df.exec_match.mean():.4f}"); return df
    df = fn(); df.to_parquet(p, index=False); print(f"[done] {name}: EX={df.exec_match.mean():.4f}"); return df
def gen_cache(name):
    return f"{RES}/gen_{name}.jsonl"
print(f"train={len(train_all)}  dev={len(dev)}  model={BASE_MODEL}  device={MD.device_info()}")'''


FREE_GPU = ("code", '''# Free GPU memory held by models from earlier steps (matters when running everything in one session)
import gc, torch
for _name in ["model", "m7", "trainer", "m"]:
    globals().pop(_name, None)
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    print(f"GPU memory in use: {torch.cuda.memory_allocated()/1e9:.2f} GB")''')


def nb(cells, gpu=True):
    book = nbf.v4.new_notebook()
    book.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
    if gpu:
        book.metadata |= {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}}
    for kind, src in cells:
        src = src.strip("\n")
        book.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    return book


NB1 = [
("md", """# 01 · Spider data analysis and prompting baselines
**Project:** Fine-tuning a small open LLM for text-to-SQL, with execution-based evaluation.
**Base model:** `Qwen2.5-Coder-1.5B-Instruct`. **Benchmark:** Spider 1.0 (cross-database; dev DBs never appear in training).

This notebook profiles the data and measures how far **prompting alone** gets:
schema-format ablation (zero-shot) → few-shot → full-dev baselines. Runtime: **T4 GPU**, about 35–50 min.

> Spider 1.0 is downloaded automatically by the setup cell (official Google Drive file, with a Hugging Face mirror as fallback).
> Only if both fail: put the zip manually at `MyDrive/masters-projects/spider_data.zip`."""),
("code", SETUP),
("md", "## 1. Dataset profile"),
("code", '''dev_h = pd.Series([hardness(e) for e in dev]).value_counts().reindex(HARDNESS_ORDER)
print("train DBs:", len({e.db_id for e in train_all}), " dev DBs:", len({e.db_id for e in dev}),
      " overlap:", len({e.db_id for e in train_all} & {e.db_id for e in dev}))
st = pd.DataFrame([schema_stats(db_path(SPIDER, d)) | {"db_id": d} for d in sorted({e.db_id for e in dev})])
display(st.describe().round(1))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
dev_h.plot.bar(ax=ax[0], color="#4C72B0"); ax[0].set_title("Dev hardness (official Spider levels)")
st.n_tables.plot.hist(ax=ax[1], bins=15, color="#55A868"); ax[1].set_title("Tables per dev database")
savefig("01_data_profile")'''),
("code", '''# Prompt length (tokens) under the three schema formats -> drives memory/speed and the max length for training
tok = MD.load_tokenizer(BASE_MODEL)
lens = {}
for fmt in ["compact", "ddl", "ddl_rows"]:
    chats = EV.make_chats(train_all[::10], SPIDER, fmt, with_answer=True)
    lens[fmt] = [len(tok(tok.apply_chat_template(c, tokenize=False), add_special_tokens=False)["input_ids"]) for c in chats]
L = pd.DataFrame(lens); display(L.describe(percentiles=[.5, .9, .95, .99]).round(0))
L.plot.hist(bins=40, alpha=.6, figsize=(7, 3.5)); plt.xlabel("tokens per training example"); savefig("01_prompt_lengths")'''),
("md", """## 2. Zero-shot: which schema format works best?
Same model and question; only the schema format changes (fixed 300-question dev subset):
* `compact`: `table(col1, col2*)` plus foreign keys
* `ddl`: `CREATE TABLE` statements with types, primary keys and foreign keys
* `ddl_rows`: DDL plus 3 example rows per table (Rajkumar et al., 2022)"""),
("code", '''model = MD.load_model(BASE_MODEL, load_4bit=False)     # 1.5B in fp16 = ~3 GB; faster than 4-bit for inference
sub = [dev[i] for i in DEV_SUBSET]
fmt_results = {}
for fmt in ["compact", "ddl", "ddl_rows"]:
    fmt_results[f"zero-shot [{fmt}]"] = cached(f"subset_zero_{fmt}", lambda: EV.score(
        sub, MD.generate(model, tok, EV.make_chats(sub, SPIDER, fmt), batch_size=16, cache_path=gen_cache(f"subset_zero_{fmt}")), SPIDER))
fmt_table = EV.compare_table(fmt_results); fmt_table.round(3)'''),
("code", '''BEST_FMT = fmt_table.exec_acc.idxmax().split("[")[1].rstrip("]")
save_json({"best_format": BEST_FMT, "table": fmt_table.reset_index().to_dict(orient="records")}, "schema_format_ablation")
print("best schema format:", BEST_FMT)'''),
("md", "## 3. Few-shot (3 demonstrations from training databases)"),
("code", '''shots = MD.pick_fewshot(train_all, k=3, seed=SEED)
for s in shots: print(f"[{s.db_id}] {s.question}\\n   {s.query}")
fmt_results["3-shot"] = cached("subset_3shot", lambda: EV.score(sub, MD.generate(
    model, tok, EV.make_chats(sub, SPIDER, BEST_FMT, fewshot=shots), batch_size=8, cache_path=gen_cache("subset_3shot")), SPIDER))
EV.compare_table(fmt_results).round(3)'''),
("md", """## 4. Full-dev baselines (1,034 questions)
Zero-shot on the full dev set. 3-shot on the full dev set is optional (`FULL_DEV_3SHOT`): its long prompts make it the
slowest step (~75 min on a T4), and the 300-question subset already measures it."""),
("code", '''FULL_DEV_3SHOT = False
full = {"zero-shot": cached("dev_zero-shot", lambda: EV.score(dev, MD.generate(
    model, tok, EV.make_chats(dev, SPIDER, BEST_FMT), batch_size=16, cache_path=gen_cache("dev_zero-shot")), SPIDER))}
if FULL_DEV_3SHOT:
    full["3-shot"] = cached("dev_3-shot", lambda: EV.score(dev, MD.generate(
        model, tok, EV.make_chats(dev, SPIDER, BEST_FMT, fewshot=shots), batch_size=8, cache_path=gen_cache("dev_3-shot")), SPIDER))
save_json({"fewshot_examples": [vars(s) | {"sql": None} for s in shots]}, "fewshot")
EV.compare_table(full).round(4)'''),
("md", """## 5. (Optional) Reference: a 5× larger model, zero-shot
`Qwen2.5-Coder-7B-Instruct` in 4-bit on the 300-question subset. It answers: *can fine-tuning a 1.5B model match a 7B one?*"""),
("code", '''RUN_7B = not SMOKE        # set False to skip (~20-30 min on a T4)
if RUN_7B and not os.path.exists(f"{RES}/subset_7b_zero_shot.parquet"):
    import gc, torch
    del model; gc.collect(); torch.cuda.empty_cache()
    big = "Qwen/Qwen2.5-Coder-7B-Instruct"
    m7, tok7 = MD.load_model(big, load_4bit=True), MD.load_tokenizer(big)
    r7 = cached("subset_7b_zero_shot", lambda: EV.score(sub, MD.generate(
        m7, tok7, EV.make_chats(sub, SPIDER, BEST_FMT), batch_size=8, cache_path=gen_cache("subset_7b")), SPIDER))
    del m7; gc.collect(); torch.cuda.empty_cache()'''),
("code", '''pd.concat({k: v for k, v in fmt_results.items()}).reset_index(level=0).rename(columns={"level_0": "system"}) \\
  .to_parquet(f"{RES}/subset_prompting.parquet", index=False)
print("saved.")'''),
]

NB2 = [
("md", """# 02 · QLoRA fine-tuning
4-bit NF4 base model + LoRA adapters on every linear layer (q, k, v, o, gate, up, down). The loss covers **only the SQL answer tokens**; prompt tokens are masked.
Training data: Spider `train_spider` + `train_others`, about 8.6k examples from 146 databases. A held-out set of training databases is used for validation loss.

Checkpoints are saved to Drive every 100 steps. If Colab disconnects, re-run the notebook and it resumes from the last checkpoint.
Runtime: **T4 GPU**, about 60–90 min for 1 epoch."""),
("code", SETUP),
FREE_GPU,
("code", '''BEST_FMT = (load_json("schema_format_ablation") or {}).get("best_format", "ddl")
MAX_LEN = 256 if SMOKE else 1536
R = 16
RUN_NAME = f"qlora-r{R}-{BEST_FMT}"
OUT = f"{ART}/{RUN_NAME}"
print("schema format:", BEST_FMT, "| run:", RUN_NAME)'''),
("md", "## 1. Build the SFT dataset (validation = held-out *databases*, mimicking the cross-database setting)"),
("code", '''rng = np.random.RandomState(SEED)
train_dbs = sorted({e.db_id for e in train_all})
val_dbs = set(rng.choice(train_dbs, max(1, len(train_dbs) // 20), replace=False))
tr_ex = [e for e in train_all if e.db_id not in val_dbs]
va_ex = [e for e in train_all if e.db_id in val_dbs][:300]
tok = MD.load_tokenizer(BASE_MODEL)
tr_rec, dropped = MD.build_sft_records(tok, EV.make_chats(tr_ex, SPIDER, BEST_FMT, with_answer=True), MAX_LEN)
va_rec, _ = MD.build_sft_records(tok, EV.make_chats(va_ex, SPIDER, BEST_FMT, with_answer=True), MAX_LEN)
print(f"train {len(tr_rec)} (dropped {dropped} longer than {MAX_LEN} tokens) | val {len(va_rec)} from {len(val_dbs)} DBs")
r0 = tr_rec[0]; ans = [t for t, l in zip(r0["input_ids"], r0["labels"]) if l != -100]
print("supervised tokens of example 0 ->", repr(tok.decode(ans)))'''),
("md", "## 2. Train"),
("code", '''ADAPTER = f"{ART}/adapter-{RUN_NAME}"
ALREADY_TRAINED = os.path.exists(f"{ADAPTER}/adapter_config.json") and load_json("main_run") is not None
if ALREADY_TRAINED:
    print("Adapter already trained (restored from a previous run) -> skipping training:", ADAPTER)
else:
  t0 = time.time()
  trainer = MD.train_lora(BASE_MODEL, tr_rec, va_rec, tok, OUT, load_4bit=not SMOKE, r=R, lr=2e-4, epochs=1,
                        batch_size=2, grad_accum=8, eval_steps=100 if not SMOKE else 2, save_steps=100 if not SMOKE else 2,
                        logging_steps=10 if not SMOKE else 1, max_steps=-1 if not SMOKE else 4, seed=SEED)
  train_minutes = (time.time() - t0) / 60
  import torch
  peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
  trainer.model.save_pretrained(ADAPTER); tok.save_pretrained(ADAPTER)
  n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
  n_total = sum(p.numel() for p in trainer.model.parameters())
  print(f"done in {train_minutes:.1f} min, peak GPU mem {peak_gb:.1f} GB, trainable params {n_trainable/1e6:.1f}M")'''),
("code", '''if ALREADY_TRAINED:
    hist = pd.read_csv(f"{RES}/train_log_{RUN_NAME}.csv")
else:
    hist = pd.DataFrame(trainer.state.log_history)
fig, ax = plt.subplots(figsize=(7, 4))
h = hist.dropna(subset=["loss"]); ax.plot(h.step, h.loss, label="train loss", alpha=.8)
if "eval_loss" in hist:
    e = hist.dropna(subset=["eval_loss"]); ax.plot(e.step, e.eval_loss, "o-", label="val loss (held-out DBs)")
ax.set_xlabel("step"); ax.set_ylabel("cross-entropy (answer tokens)"); ax.legend(); ax.set_title("QLoRA training")
savefig("02_loss_curve")
hist.to_csv(f"{RES}/train_log_{RUN_NAME}.csv", index=False)
if not ALREADY_TRAINED:
  save_json({"run": RUN_NAME, "adapter": f"{ART}/adapter-{RUN_NAME}", "schema_format": BEST_FMT, "r": R, "max_len": MAX_LEN,
             "n_train": len(tr_rec), "n_dropped": dropped, "train_minutes": train_minutes, "peak_gpu_gb": peak_gb,
             "trainable_params": n_trainable, "total_params": n_total, "gpu": MD.device_info()["name"]}, "main_run")'''),
]

NB3 = [
("md", """# 03 · Evaluation and error analysis
Fine-tuned model vs. the prompting baselines on the **full Spider dev set**: execution accuracy overall and by hardness,
a paired significance test, an automatic error taxonomy, and qualitative examples. Runtime: **T4 GPU**, about 15 min."""),
("code", SETUP),
FREE_GPU,
("code", '''run = load_json("main_run"); BEST_FMT = run["schema_format"]
tok = MD.load_tokenizer(BASE_MODEL)
model = MD.load_model(BASE_MODEL, load_4bit=False, adapter=run["adapter"], merge=True)   # merged fp16 = fast inference
ft = cached("dev_finetuned", lambda: EV.score(dev, MD.generate(
    model, tok, EV.make_chats(dev, SPIDER, BEST_FMT), batch_size=16, cache_path=gen_cache("dev_finetuned")), SPIDER))'''),
("md", "## 1. Headline comparison"),
("code", '''res = {"zero-shot": pd.read_parquet(f"{RES}/dev_zero-shot.parquet")}
if os.path.exists(f"{RES}/dev_3-shot.parquet"):
    res["3-shot"] = pd.read_parquet(f"{RES}/dev_3-shot.parquet")
res["QLoRA fine-tuned"] = ft
table = EV.compare_table(res)
table.to_csv(f"{RES}/headline_table.csv"); table.round(4)'''),
("code", '''sig = {f"fine-tuned vs {k}": EV.mcnemar(res[k].exec_match, ft.exec_match) for k in res if k != "QLoRA fine-tuned"}
pd.DataFrame(sig).T'''),
("md", "### Same 300-question subset: zero-shot vs 3-shot vs 7B vs fine-tuned"),
("code", '''subset = {"1.5B zero-shot": res["zero-shot"].iloc[DEV_SUBSET]}
if os.path.exists(f"{RES}/subset_3shot.parquet"):
    subset["1.5B 3-shot"] = pd.read_parquet(f"{RES}/subset_3shot.parquet")
if os.path.exists(f"{RES}/subset_7b_zero_shot.parquet"):
    subset["7B zero-shot (4-bit)"] = pd.read_parquet(f"{RES}/subset_7b_zero_shot.parquet")
subset["1.5B QLoRA fine-tuned"] = ft.iloc[DEV_SUBSET]
ref = EV.compare_table(subset); ref.to_csv(f"{RES}/subset_comparison.csv"); display(ref.round(4))
if "1.5B 3-shot" in subset:
    sig["fine-tuned vs 3-shot (subset)"] = EV.mcnemar(subset["1.5B 3-shot"].exec_match.reset_index(drop=True),
                                                      subset["1.5B QLoRA fine-tuned"].exec_match.reset_index(drop=True))
pd.DataFrame(sig).T'''),
("code", '''acc = table[[f"acc_{h}" for h in HARDNESS_ORDER]].T; acc.index = HARDNESS_ORDER
ax = acc.plot.bar(figsize=(8, 4), rot=0, color=["#BBBBBB", "#8172B3", "#C44E52"][-acc.shape[1]:])
ax.set_ylabel("execution accuracy"); ax.set_title("Spider dev: execution accuracy by hardness"); ax.set_ylim(0, 1)
for c in ax.containers: ax.bar_label(c, fmt="%.2f", fontsize=8)
savefig("03_accuracy_by_hardness")'''),
("md", "## 2. Error taxonomy: what kinds of mistakes does fine-tuning fix?"),
("code", '''err = pd.DataFrame({k: v.error_category.value_counts() for k, v in res.items()}).fillna(0).astype(int)
err = err.loc[err.sum(axis=1).sort_values().index]
err.to_csv(f"{RES}/error_taxonomy.csv")
ax = err.plot.barh(figsize=(9, 5.5), color=["#BBBBBB", "#8172B3", "#C44E52"][-err.shape[1]:])
ax.set_xlabel("# failed dev questions"); ax.set_title("Error categories (lower is better)")
savefig("03_error_taxonomy"); err'''),
("code", '''BASE_NAME = "3-shot" if "3-shot" in res else "zero-shot"
base = res[BASE_NAME]
fixed = ft[(~base.exec_match.values) & ft.exec_match.values]
broken = ft[(base.exec_match.values) & ~ft.exec_match.values]
print(f"fixed by fine-tuning: {len(fixed)}   broken by fine-tuning: {len(broken)}\\n")
for title, df in [("FIXED", fixed), ("BROKEN", broken)]:
    print(f"======== {title} (examples) ========")
    for i in df.sample(min(4, len(df)), random_state=0).index:
        print(f"[{dev[i].db_id} | {ft.hardness[i]}] {dev[i].question}\\n  gold : {ft.gold[i]}\\n  {BASE_NAME}: {base.pred[i]}"
              f"\\n  FT   : {ft.pred[i]}\\n")'''),
("code", '''# Accuracy vs. schema size: do larger databases hurt?
ft["n_tables"] = [schema_stats(db_path(SPIDER, d))["n_tables"] for d in ft.db_id]
bins = pd.cut(ft.n_tables, [0, 3, 5, 8, 100], labels=["1-3", "4-5", "6-8", "9+"])
by_size = pd.DataFrame({k: v.exec_match.groupby(bins.values).mean() for k, v in res.items()})
by_size.plot(marker="o", figsize=(6, 3.5)); plt.ylabel("execution accuracy"); plt.xlabel("tables in database")
savefig("03_accuracy_by_schema_size"); by_size.round(3)'''),
("md", "## 3. Auto-generated results summary"),
("code", '''abl = pd.read_csv(f"{RES}/ablations.csv") if os.path.exists(f"{RES}/ablations.csv") else None
fmt = load_json("schema_format_ablation")
lines = ["# Results summary (auto-generated by notebook 03)", "",
 f"* Base model: {BASE_MODEL}; adapter: LoRA r={run['r']} on all linear layers, 4-bit NF4 base (QLoRA)",
 f"* Trainable parameters: {run['trainable_params']/1e6:.1f}M of {run['total_params']/1e6:.0f}M; trained "
 f"{run['train_minutes']:.0f} min on {run['gpu']} (peak {run['peak_gpu_gb']:.1f} GB)",
 f"* **Spider dev (1,034) execution accuracy: zero-shot {table.loc['zero-shot','exec_acc']:.1%} → fine-tuned "
 f"{table.loc['QLoRA fine-tuned','exec_acc']:.1%}**",
 "* 300-question subset: " + ", ".join(f"{k} {v:.1%}" for k, v in ref.exec_acc.items()),
 f"* Execution-error rate: {table.loc['zero-shot','exec_error_rate']:.1%} → {table.loc['QLoRA fine-tuned','exec_error_rate']:.1%}",
 "* McNemar p-values: " + ", ".join(f"{k}: {v['p_value']:.2e}" for k, v in sig.items()),
 "", "## Headline", table.round(4).to_markdown(),
 "", "## Schema-format ablation (zero-shot, 300-question subset)", pd.DataFrame(fmt["table"]).round(4).to_markdown(index=False),
 "", "## Error taxonomy", err.to_markdown(),
 "", "## By schema size", by_size.round(3).to_markdown()]
lines += ["", "## Same 300-question subset", ref.round(4).to_markdown()]
if abl is not None:
    lines += ["", "## Ablations (300-question subset)", abl.round(4).to_markdown(index=False)]
open(f"{PROJECT_DIR}/reports/results_summary.md", "w").write("\\n".join(lines)); print("\\n".join(lines[:8]))'''),
]

NB4 = [
("md", """# 04 · Ablations: training-data size and LoRA rank
Each run trains for 1 epoch on a fraction of the data and is evaluated on the fixed 300-question dev subset.
Runtime: **T4 GPU**, about 60–80 min in total. Run it after notebook 03, then re-run the last cell of 03 to add the table to the summary."""),
("code", SETUP),
("code", '''run = load_json("main_run"); BEST_FMT, MAX_LEN = run["schema_format"], run["max_len"]
tok = MD.load_tokenizer(BASE_MODEL)
rng = np.random.RandomState(SEED)
chats = EV.make_chats(train_all, SPIDER, BEST_FMT, with_answer=True)
all_rec, _ = MD.build_sft_records(tok, chats, MAX_LEN)
rng.shuffle(all_rec)
sub = [dev[i] for i in DEV_SUBSET]
CONFIGS = [  # (label, data fraction, rank) - all at 10% data: on a T4 a full epoch took ~7 h, so 10% ~ 40 min each
    ("10% data, r=16", 0.10, 16), ("10% data, r=4", 0.10, 4), ("10% data, r=64", 0.10, 64)]
if SMOKE: CONFIGS = CONFIGS[:2]'''),
("code", '''import gc, torch
rows = []
for label, frac, r in CONFIGS:
    name = f"abl-f{int(frac*100)}-r{r}"
    if os.path.exists(f"{RES}/{name}.parquet"):
        df = pd.read_parquet(f"{RES}/{name}.parquet")
    else:
        recs = all_rec[: max(8, int(len(all_rec) * frac))]
        t0 = time.time()
        trainer = MD.train_lora(BASE_MODEL, recs, recs[:16], tok, f"{ART}/{name}", load_4bit=not SMOKE, r=r,
                                epochs=1, eval_steps=10**6, save_steps=50 if not SMOKE else 2, logging_steps=10,
                                max_steps=-1 if not SMOKE else 2, seed=SEED)
        trainer.model.save_pretrained(f"{ART}/adapter-{name}"); mins = (time.time() - t0) / 60
        del trainer; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
        m = MD.load_model(BASE_MODEL, load_4bit=False, adapter=f"{ART}/adapter-{name}", merge=True)
        df = EV.score(sub, MD.generate(m, tok, EV.make_chats(sub, SPIDER, BEST_FMT), batch_size=16), SPIDER)
        df["train_minutes"] = mins; df.to_parquet(f"{RES}/{name}.parquet", index=False)
        del m; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    rows.append({"config": label, "data_frac": frac, "rank": r, "exec_acc": df.exec_match.mean(),
                 "train_minutes": df.get("train_minutes", pd.Series([np.nan])).iloc[0]})
    print(rows[-1])'''),
("code", '''ft = pd.read_parquet(f"{RES}/dev_finetuned.parquet").iloc[DEV_SUBSET]
zs = pd.read_parquet(f"{RES}/dev_zero-shot.parquet").iloc[DEV_SUBSET]
rows += [{"config": f"100% data, r={run['r']} (main)", "data_frac": 1.0, "rank": run["r"], "exec_acc": ft.exec_match.mean(),
          "train_minutes": run["train_minutes"]},
         {"config": "no fine-tuning (zero-shot)", "data_frac": 0.0, "rank": 0, "exec_acc": zs.exec_match.mean(), "train_minutes": 0}]
abl = pd.DataFrame(rows); abl.to_csv(f"{RES}/ablations.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
d = abl[(abl["rank"].isin([run["r"], 0]))].sort_values("data_frac")
ax[0].plot(d.data_frac * 100, d.exec_acc, "o-"); ax[0].set_xlabel("% of training data"); ax[0].set_ylabel("EX (subset)")
ax[0].set_title("Data scaling")
rk = abl[abl.data_frac == 0.10].sort_values("rank")
ax[1].plot(rk["rank"], rk.exec_acc, "o-"); ax[1].set_xscale("log", base=2); ax[1].set_xlabel("LoRA rank (10% data)")
ax[1].set_title("LoRA rank"); savefig("04_ablations"); abl.round(4)'''),
]

NB5 = [
("md", """# 05 · Demo app and model publishing
Gradio app: choose a database, ask a question in English, and see the generated SQL plus its executed result. Optionally publish the adapter to the Hugging Face Hub."""),
("code", SETUP),
("code", '''sys.path.insert(0, f"{PROJECT_DIR}/app")
from app import build_demo
run = load_json("main_run")
demo = build_demo(BASE_MODEL, run["adapter"], SPIDER, run["schema_format"], sorted({e.db_id for e in dev}))
demo.launch(share=not SMOKE, prevent_thread_lock=SMOKE)'''),
("md", """## Publish the adapter (optional)
Add a Hugging Face *write* token as a Colab secret named `HF_TOKEN` (key icon in the left sidebar), then set `HF_REPO`."""),
("code", '''PUBLISH = False
HF_REPO = "your-hf-username/qwen2.5-coder-1.5b-spider-qlora"
if PUBLISH and not SMOKE:
    from google.colab import userdata
    from huggingface_hub import login
    from peft import PeftModel
    login(userdata.get("HF_TOKEN"))
    m = MD.load_model(BASE_MODEL, load_4bit=False, adapter=run["adapter"])
    m.push_to_hub(HF_REPO); MD.load_tokenizer(BASE_MODEL).push_to_hub(HF_REPO)
    print("https://huggingface.co/" + HF_REPO)'''),
]


KAGGLE_INTRO = ("md", """# Text-to-SQL with QLoRA: full pipeline (Kaggle edition)
Runs notebooks 01 → 02 → 03 (and optionally 04) in one go: about 3.5 h on a Kaggle T4 (4.5 h with ablations).

**Kaggle settings (right panel):** Accelerator = **GPU T4 x2** · Internet = **On** · Input = your `text2sql` code dataset.
Run it with **Save Version → Save & Run All (Commit)**. It then runs in the background (you can close the browser)
and every file under `/kaggle/working` is kept as the notebook's **Output**. Download `text2sql/reports/` when it finishes.""")
RUN_ABLATIONS_CELL = ("code", "RUN_ABLATIONS = False   # True adds ~2.5 h on a T4 (3 extra 10%-data trainings + evals)")
ABL_GUARD = "if RUN_ABLATIONS:\n"


def _body(cells):
    """Drop each notebook's own title+setup cells (first two)."""
    return cells[2:]


def kaggle_pipeline():
    cells = [KAGGLE_INTRO, ("code", SETUP), RUN_ABLATIONS_CELL]
    for title, part in [("Part 1: data and prompting baselines", NB1), ("Part 2: QLoRA fine-tuning", NB2),
                        ("Part 3: evaluation and error analysis", NB3)]:
        cells.append(("md", f"# {title}"))
        cells += _body(part)
    cells.append(("md", "# Part 4: ablations (optional)"))
    for kind, src in _body(NB4):
        if kind == "code":
            src = ABL_GUARD + "\n".join("    " + l for l in src.strip("\n").splitlines())
        cells.append((kind, src))
    cells.append(("md", "# Final summary (includes ablations if they ran)"))
    cells.append(NB3[-1])
    cells.append(("code", 'import shutil; shutil.make_archive("/kaggle/working/text2sql_reports", "zip", f"{PROJECT_DIR}/reports") '
                          'if ON_KAGGLE else None\nprint("Download: Output -> text2sql_reports.zip")'))
    return cells


def main():
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for name, cells in [("01_data_and_prompting_baselines", NB1), ("02_qlora_finetune", NB2),
                        ("03_evaluation_error_analysis", NB3), ("04_ablations", NB4), ("05_demo_app", NB5),
                        ("KAGGLE_full_pipeline", kaggle_pipeline())]:
        nbf.write(nb(cells), out / f"{name}.ipynb")
        print("wrote", name)


if __name__ == "__main__":
    main()
