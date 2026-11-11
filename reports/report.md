# Teaching a Small Open LLM to Write SQL: QLoRA Fine-Tuning of Qwen2.5-Coder-1.5B on Spider, with Execution-Based Error Analysis

**Author:** _Your Name_ · **Code:** _github.com/your-username/text2sql_ · **Hardware:** one free NVIDIA T4 (Kaggle)

## Abstract
I fine-tune Qwen2.5-Coder-1.5B-Instruct for text-to-SQL with QLoRA (4-bit NF4 base and rank-16 LoRA on every linear layer,
1.2% of parameters trained) on the cross-database Spider benchmark, using a single free T4 GPU. Predictions are evaluated by
**executing** them against the real SQLite databases. On all 1,034 dev questions, whose databases never appear in training, execution accuracy rises
from **61.0% (zero-shot) to 68.2%** (+7.2 points; McNemar p = 3×10⁻⁷). The gains come from medium (+8.5) and hard
(+15.0) questions. An automatic error taxonomy shows *why*: fine-tuning **halves schema hallucinations** (references to
non-existent columns: 185 → 98) and cuts the execution-error rate from 19.4% to 11.4%. Some failures change type rather than disappear:
queries that used to crash now run but return wrong results. I report two limits honestly: on a 300-question subset, fine-tuning
is not significantly better than 3-shot prompting (66.7% vs 63.3%, p = 0.31), and a 5× larger model (7B) used zero-shot is still far
ahead (81.3%).

## 1. Setup
* **Data:** Spider 1.0. 8,659 training questions over 146 databases, and 1,034 dev questions over 20 *different* databases, so the test is
  generalisation to unseen schemas. Difficulty labels come from a re-implementation of Spider's official hardness rules.
* **Prompt:** a system instruction, the database schema, and the question. The model answers with SQL only.
* **Metric: execution accuracy (EX).** Predicted and gold SQL are run on a read-only connection with a 10 s timeout. Results are compared as
  multisets (order-sensitive only when the gold query has ORDER BY), tolerating column permutation and small float differences.
  This is a simplified version of the official test-suite metric, so the numbers are not directly comparable to the leaderboard.
* **Fine-tuning:** QLoRA with a 4-bit NF4 base (double quantisation) and LoRA r = 16, α = 32 on q/k/v/o/gate/up/down. 18.5M trainable parameters,
  1 epoch, LR 2e-4 with cosine schedule, effective batch 16 (2 × 8 accumulation). The loss covers **answer tokens only**. Validation
  loss is measured on 7 held-out *training databases* to mimic the cross-database setting. Training took 417 min on a T4 with peak memory 5.0 GB.
* **Engineering for a free GPU.** The naive training loss materialised logits of size batch × 1,536 × 152k vocabulary (≈7 GB) and ran out of memory.
  With left padding all answers end at the same position, so logits are computed only for the answer window (`logits_to_keep`).
  The loss is numerically identical (verified) and the memory is a small fraction. Generation is batched and length-sorted,
  every step is cached and resumable after Kaggle crashes, and training resumes from checkpoints.

## 2. Prompting baselines
**Schema format matters more than examples** (zero-shot, 300-question subset):

| Schema serialisation | EX | Execution-error rate |
|---|---|---|
| `table(col, …)` + foreign keys (compact) | 51.3% | 28.0% |
| `CREATE TABLE` DDL (types, keys) | 60.0% | 22.7% |
| DDL + 3 example rows per table | **61.3%** | **18.3%** |

Adding types and keys gives +8.7 points. Example rows add only +1.3 points, but they reduce execution errors further, plausibly because
the model sees real column values. DDL + rows was used for everything else. On the same subset, **3-shot prompting** reaches 63.3%
(+1.7 over zero-shot).

## 3. Fine-tuning results
| System (Spider dev, n = 1,034) | EX | Easy (248) | Medium (446) | Hard (174) | Extra (166) | Exec. errors |
|---|---|---|---|---|---|---|
| Zero-shot | 61.0% | 79.8% | 64.1% | 43.1% | 43.4% | 19.4% |
| **QLoRA fine-tuned** | **68.2%** | 80.2% | **72.7%** | **58.1%** | **48.8%** | **11.4%** |
| Δ | **+7.2** | +0.4 | +8.5 | **+15.0** | +5.4 | −8.0 |

* **Significance:** McNemar's paired test on the 1,034 questions gives p = 3.2 × 10⁻⁷.
* **Where the gains come from:** easy questions were already handled well (80%). Fine-tuning helps most on *hard* questions
  (+15 points), the multi-condition, join and grouping queries where knowing Spider's SQL conventions matters most.
* **Schema size:** fine-tuning improves accuracy for every database size, most for 4–5-table databases (70.6% → 82.9%).
  Large schemas (6+ tables) remain hardest (about 53–59%).

### Same 300-question subset, including a larger model
| System | EX | Hard | Exec. errors |
|---|---|---|---|
| 1.5B zero-shot | 61.7% | 44.4% | 18.3% |
| 1.5B 3-shot | 63.3% | 55.6% | 18.3% |
| **1.5B QLoRA fine-tuned** | **66.7%** | **64.8%** | **10.3%** |
| 7B zero-shot (4-bit) | 81.3% | 77.8% | 0.7% |

* Fine-tuned vs 3-shot: +3.4 points, **not significant** on 300 questions (McNemar p = 0.31). The full-dev 3-shot run was skipped
  for compute reasons, so the comparison relies on this subset.
* **A 5× larger model (7.6B parameters) used zero-shot is still 14.6 points better.** Fine-tuning closed about a quarter of the 1.5B → 7B gap. The 7B model almost
  never produces queries that fail to execute (0.7%), which suggests better schema grounding comes with scale.

## 4. Error analysis: what fine-tuning fixed, and what it didn't
Every failure is classified automatically: execution errors by SQLite error type, and wrong results by comparing the
structure of the predicted and gold SQL.

| Error category (failed dev questions) | Zero-shot | Fine-tuned | Change |
|---|---|---|---|
| **Unknown column (schema hallucination)** | **185** | **98** | **−87 (−47%)** |
| Nesting / set operation (INTERSECT, EXCEPT, sub-queries) | 59 | 47 | −12 |
| Filter condition / values | 14 | 7 | −7 |
| Grouping / HAVING | 10 | 6 | −4 |
| Join structure | 64 | 72 | +8 |
| Other: column choice / semantics | 26 | 44 | +18 |
| All other categories | 45 | 55 | +10 |
| **Total failures** | **403** | **329** | **−74** |

1. **The biggest single effect is grounding.** Zero-shot, almost half of all failures (185 / 403) reference a column that doesn't
   exist. Fine-tuning halves this.
2. **Errors move "downstream".** Some queries that used to crash now *execute* but return the wrong answer: join-structure and
   column-choice errors increase. Fine-tuning taught the model to use the schema correctly more often than it taught it the right
   *reasoning*. This is why execution accuracy, not just execution success, is the right metric.
3. **Set operations and nesting improve** (−20%), consistent with the large gain on hard questions.

## 5. Limitations and next steps
* **About 22% of training examples were dropped** (1,880 of 8,390) because their prompts exceeded 1,536 tokens. These are the largest
  schemas, which may explain the weaker results on 6+-table databases. Schema pruning (including only relevant tables) would keep them.
* One epoch with default hyper-parameters; no tuning. The planned data-size and LoRA-rank ablations were not run because of GPU time
  (one epoch took 7 h on a free T4).
* The EX metric is a simplified version of the official test-suite evaluation, so the numbers are not directly comparable to the leaderboard.
* Greedy decoding only. Self-consistency (sampling several queries and voting on execution results) and execution-guided repair
  (re-prompting with the SQLite error) are natural next steps, because 11% of fine-tuned predictions still fail to execute.
