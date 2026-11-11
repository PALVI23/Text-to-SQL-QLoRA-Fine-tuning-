"""Glue: build prompts for a list of examples, score predictions, summarise."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .analysis import HARDNESS_ORDER, categorize_error, hardness
from .execution import execution_match
from .prompts import build_messages, extract_sql, normalize_gold
from .spider import db_path, serialize_schema


def make_chats(examples, root, schema_fmt="ddl", fewshot=None, with_answer=False):
    shots = [(serialize_schema(db_path(root, e.db_id), "compact"), e.question, normalize_gold(e.query))
             for e in (fewshot or [])]
    return [build_messages(serialize_schema(db_path(root, e.db_id), schema_fmt), e.question,
                           normalize_gold(e.query) if with_answer else None, shots) for e in examples]


def score(examples, raw_outputs, root, timeout_s=10.0, workers=8) -> pd.DataFrame:
    preds = [extract_sql(o or "") for o in raw_outputs]

    def one(args):
        e, p = args
        r = execution_match(db_path(root, e.db_id), e.query, p, timeout_s)
        return r

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, zip(examples, preds)))
    rows = []
    for e, raw, p, r in zip(examples, raw_outputs, preds, res):
        rows.append({"db_id": e.db_id, "question": e.question, "gold": normalize_gold(e.query), "pred": p,
                     "raw_output": raw, "hardness": hardness(e), **r,
                     "error_category": None if r["exec_match"] else categorize_error(e.query, p, r["pred_error"])})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, name: str = "") -> dict:
    out = {"system": name, "n": len(df), "exec_acc": df.exec_match.mean(),
           "exec_error_rate": df.pred_error.notna().mean()}
    for h in HARDNESS_ORDER:
        sub = df[df.hardness == h]
        out[f"acc_{h}"] = sub.exec_match.mean() if len(sub) else float("nan")
    return out


def compare_table(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame([summarize(df, k) for k, df in results.items()]).set_index("system")


def mcnemar(a: pd.Series, b: pd.Series) -> dict:
    """Paired significance test between two systems on the same examples (exact binomial McNemar)."""
    from scipy.stats import binomtest

    a, b = a.astype(bool).values, b.astype(bool).values
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"only_b_correct": n01, "only_a_correct": n10, "p_value": float(p)}
