# Text-to-SQL: QLoRA fine-tuning of a small open LLM, with execution-based evaluation

Fine-tunes **Qwen2.5-Coder-1.5B-Instruct** with **QLoRA** (4-bit NF4 base and LoRA adapters on every linear layer) to turn
English questions into SQLite queries over databases it has never seen (the **Spider 1.0** cross-database benchmark).
The emphasis is on **rigorous evaluation**: predicted queries are *executed* against the real databases and their results compared with the gold query's results.

## What's inside
1. **Data profiling**: Spider's official hardness levels (re-implemented from `evaluation.py`), schema sizes, token budgets.
2. **Prompting baselines**: a zero-shot ablation over three schema serialisations (compact / DDL / DDL + sample rows), then a 3-shot baseline.
3. **QLoRA SFT** with a loss on answer tokens only, validation on *held-out databases*, and checkpoint resume (Colab-proof).
4. **Evaluation**: execution accuracy overall and by hardness, McNemar's paired significance test, accuracy vs. schema size, and an optional comparison with a 5× larger model (Qwen2.5-Coder-7B, zero-shot).
5. **Error taxonomy**: automatic classification of each failure (schema hallucination, join structure, aggregation, grouping, filter values, …), showing which error types fine-tuning fixes.
6. **Ablations**: training-data size (10/30/100%) and LoRA rank (4/16/64).
7. **Demo**: a Gradio app (question → SQL → executed result) and an optional push of the adapter to the Hugging Face Hub.

## Results (Spider dev; full write-up in [`reports/report.md`](reports/report.md))

| System (n = 1,034 unseen-database questions) | Execution acc. | Easy | Medium | Hard | Extra | Exec. errors |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B zero-shot | 61.0% | 79.8% | 64.1% | 43.1% | 43.4% | 19.4% |
| **+ QLoRA fine-tuning (this work)** | **68.2%** | 80.2% | **72.7%** | **58.1%** | **48.8%** | **11.4%** |

* **+7.2 points** overall (McNemar p = 3×10⁻⁷), and **+15 points on hard questions**. Trained 1.2% of parameters (18.5M) on one free T4 GPU.
* **Schema hallucinations halved** (queries referencing non-existent columns: 185 → 98). Some failures move "downstream", from crashing to running but giving the wrong answer.
* **Schema format matters:** `CREATE TABLE` + example rows beats a compact `table(columns)` listing by **+10 points** zero-shot.
* **Honest limits:** on a 300-question subset, fine-tuning (66.7%) is not significantly better than 3-shot prompting (63.3%, p = 0.31), and a 5× larger 7B model used zero-shot reaches 81.3%.

## Layout
```
src/t2s/      spider.py (data + schema serialisation)  prompts.py  execution.py (EX metric)
              analysis.py (hardness + error taxonomy)  modeling.py (QLoRA training, batched generation)
              evaluate.py  smoke.py (tiny model + mini-Spider for CPU tests)
notebooks/    01_data_and_prompting_baselines  02_qlora_finetune  03_evaluation_error_analysis  04_ablations  05_demo_app
tests/        unit tests (pytest)
scripts/      build_notebooks.py, smoke_test.py (runs every notebook on CPU with a tiny random model)
```

## How to run (Google Colab, T4 GPU)
1. Upload the `text2sql` folder to `MyDrive/masters-projects/text2sql`.
2. Download Spider 1.0 from https://yale-lily.github.io/spider and save it as `MyDrive/masters-projects/spider_data.zip`.
3. *Runtime → Change runtime type → T4 GPU*. Run notebooks 01 → 02 → 03 (→ 04 → re-run the last cell of 03) → 05.
   Approximate GPU time: 45 + 75 + 15 + 70 min. Everything is saved to Drive, so sessions can be split across days.

## Evaluation protocol
* **Execution accuracy (EX):** a prediction counts as correct if its result equals the gold query's result. Rows are compared as a multiset unless the gold query has `ORDER BY`; small float differences and column permutations are tolerated. Each query has a 10 s timeout and runs on a read-only connection.
  This is a simplified version of the official test-suite metric, and the numbers are not directly comparable to the leaderboard.
* Spider's dev databases are disjoint from its training databases, so this measures generalisation to unseen schemas.
