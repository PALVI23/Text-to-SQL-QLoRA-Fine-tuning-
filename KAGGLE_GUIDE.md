# Project 3 on Kaggle: start-to-finish guide

All earlier problems are handled in this version:
| Problem you hit | Fix in the code |
|---|---|
| Spider 1.0 not on the website | Downloaded automatically (official Google Drive file, Hugging Face mirror as fallback) |
| Kernel died mid-generation | Every step saves to disk; generation saves after every batch; re-runs skip finished work |
| Full-dev 3-shot too slow | Off by default (3-shot is measured on the 300-question subset) |
| CUDA out of memory in training | Loss is computed only on the SQL answer tokens; GPU is freed between stages; micro-batch 2 × 8 |
| Kaggle token / kaggle.json | Not needed for this project |

## A. One-time setup (10 min)
1. **Phone-verify Kaggle:** profile picture → *Settings* → **Phone verification**. This is required for GPU and internet access.
2. **Upload the code:** **+ Create → New Dataset** → drag in `text2sql-kaggle-final.zip` (don't unzip it) → title `text2sql-code` → keep it **Private** → **Create**.
   (If the dataset already exists: open it → **New Version** → upload the zip → **Create**.)
3. **Create the notebook:** **+ Create → New Notebook** → **File → Import Notebook** → choose `KAGGLE_full_pipeline.ipynb`
   (unzip a copy of the zip on your laptop; it is in `text2sql/notebooks/`).
4. **Right-hand panel** (click ▸ at the top right if it's hidden):
   * **Session options → Accelerator:** `GPU T4 x2`
   * **Session options → Internet:** `On`
   * **Input → + Add Input → Your Work → Datasets →** `text2sql-code` → **Add**

## B. Quick check (5 min, then stop)
5. Click the first code cell → **Shift+Enter**. Wait 2–4 min. You should see:
   ```
   Spider check: dev questions = 1034 (expected 1034), databases = ...
   train=8659  dev=1034  model=Qwen/Qwen2.5-Coder-1.5B-Instruct  device={'cuda': True, 'name': 'Tesla T4', ...}
   ```
6. Choose your options in the next cell:
   * `RUN_ABLATIONS = True` gives better results for your report (+~90 min). Set it to `False` if you're short on GPU hours.
7. **Stop the session** (⏻ at the top) so you don't spend GPU hours while idle.

## C. Full run in the background (about 5–6 h with ablations, about 4 h without)
8. **Save Version** (top right) → **Save & Run All (Commit)** → **Save**.
9. Close the tab. To check on it: notebook page → **Versions**, or *View Active Events* at the bottom left. It's fine to look at the log while it runs.
10. When it finishes: open the version → **Output** tab → download **`text2sql_reports.zip`** and send it to me.
    (Optional: also download `text2sql/artifacts/adapter-qlora-r16-*`. That's your fine-tuned model.)

## D. If the run fails (crash, timeout, error)
11. Open the failed version → **Logs** → copy the last ~30 lines and send them to me.
12. To resume instead of starting over: in the notebook editor, **+ Add Input → Your Work → Notebooks** → select **this notebook**
    (its latest version's output) → **Add**. Then do **Save & Run All (Commit)** again.
    The log will show `restored previous results` and `[cached] ...`; finished steps are skipped and training continues from its last checkpoint.

## What the run does (and roughly how long it takes on a T4)
| Part | What | Time |
|---|---|---|
| 1 | Data profile; zero-shot with 3 schema formats (300 questions); 3-shot (300); zero-shot on the full dev set (1,034); 7B reference (300) | ~1.5–2 h |
| 2 | QLoRA fine-tuning, 1 epoch (~530 steps) | ~1.5 h |
| 3 | Evaluate the fine-tuned model on all 1,034 questions; tables, error taxonomy, figures | ~30 min |
| 4 | Ablations: 10%/30% data, LoRA rank 4/64 (optional) | ~1.5 h |
| end | Writes `reports/results_summary.md` + `figures/` and zips them as `text2sql_reports.zip` | – |

GPU budget: Kaggle gives about 30 GPU hours a week, and this run uses about 4–6.
