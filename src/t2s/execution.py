"""Execution-based evaluation: run predicted and gold SQL on the database and compare results."""
from __future__ import annotations

import re
import sqlite3
import time
from collections import Counter
from pathlib import Path


def execute(db: str | Path, sql: str, timeout_s: float = 10.0, max_rows: int = 100_000):
    """Returns (rows, error). Read-only connection, with a wall-clock timeout via the progress handler."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, check_same_thread=False)
    conn.text_factory = lambda b: b.decode(errors="ignore")
    start = time.time()
    conn.set_progress_handler(lambda: 1 if time.time() - start > timeout_s else 0, 10_000)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows)
        return rows, None
    except sqlite3.OperationalError as e:
        msg = str(e)
        return None, ("timeout" if "interrupted" in msg else msg)
    except Exception as e:  # noqa: BLE001 - any failure counts as an execution error
        return None, f"{type(e).__name__}: {e}"
    finally:
        conn.close()


def _norm(v):
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, bytes):
        return v.decode(errors="ignore")
    return v


def _canon_row(r):
    return tuple(sorted((repr(_norm(v)) for v in r)))


def results_match(gold_rows, pred_rows, ordered: bool) -> bool:
    """Order-sensitive only if the gold query has ORDER BY; tolerant to column permutation."""
    if gold_rows is None or pred_rows is None:
        return False
    if len(gold_rows) != len(pred_rows):
        return False
    g = [tuple(_norm(v) for v in r) for r in gold_rows]
    p = [tuple(_norm(v) for v in r) for r in pred_rows]
    if ordered:
        return g == p or [_canon_row(r) for r in g] == [_canon_row(r) for r in p]
    return Counter(g) == Counter(p) or Counter(map(_canon_row, g)) == Counter(map(_canon_row, p))


def has_order_by(sql: str) -> bool:
    return re.search(r"\border\s+by\b", sql, re.I) is not None


def execution_match(db, gold: str, pred: str, timeout_s: float = 10.0) -> dict:
    g_rows, g_err = execute(db, gold, timeout_s)
    if not pred.strip():
        return {"exec_match": False, "pred_error": "empty prediction", "gold_error": g_err}
    p_rows, p_err = execute(db, pred, timeout_s)
    return {"exec_match": results_match(g_rows, p_rows, has_order_by(gold)) if not p_err else False,
            "pred_error": p_err, "gold_error": g_err}
