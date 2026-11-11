"""Spider hardness levels (re-implementation of the official evaluation.py rules) and an automatic
error taxonomy for failed predictions."""
from __future__ import annotations

import re

WHERE_OPS = ("not", "between", "=", ">", "<", ">=", "<=", "!=", "in", "like", "is", "exists")
AGG_NONE = 0
HARDNESS_ORDER = ["easy", "medium", "hard", "extra"]


# ------------------------------------------------------------------ official hardness (needs parsed sql)
def _nested(sql):
    nested = []
    for cu in sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]:
        if isinstance(cu[3], dict):
            nested.append(cu[3])
        if isinstance(cu[4], dict):
            nested.append(cu[4])
    for k in ("intersect", "except", "union"):
        if sql.get(k) is not None:
            nested.append(sql[k])
    return nested


def _has_agg(unit):
    return unit[0] != AGG_NONE


def _count_agg(units):
    return len([u for u in units if _has_agg(u)])


def _comp1(sql):
    c = int(len(sql["where"]) > 0) + int(len(sql["groupBy"]) > 0) + int(len(sql["orderBy"]) > 0)
    c += int(sql["limit"] is not None)
    if len(sql["from"]["table_units"]) > 0:
        c += len(sql["from"]["table_units"]) - 1
    ao = sql["from"]["conds"][1::2] + sql["where"][1::2] + sql["having"][1::2]
    c += len([t for t in ao if t == "or"])
    cus = sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]
    c += len([cu for cu in cus if cu[1] == WHERE_OPS.index("like")])
    return c


def _others(sql):
    c = 0
    agg = _count_agg(sql["select"][1]) + _count_agg(sql["where"][::2]) + _count_agg(sql["groupBy"])
    if len(sql["orderBy"]) > 0:
        agg += _count_agg([u[1] for u in sql["orderBy"][1] if u[1]] + [u[2] for u in sql["orderBy"][1] if u[2]])
    agg += _count_agg(sql["having"])
    c += int(agg > 1) + int(len(sql["select"][1]) > 1) + int(len(sql["where"]) > 1) + int(len(sql["groupBy"]) > 1)
    return c


def spider_hardness(sql: dict) -> str:
    c1, c2, co = _comp1(sql), len(_nested(sql)), _others(sql)
    if c1 <= 1 and co == 0 and c2 == 0:
        return "easy"
    if (co <= 2 and c1 <= 1 and c2 == 0) or (c1 <= 2 and co < 2 and c2 == 0):
        return "medium"
    if (co > 2 and c1 <= 2 and c2 == 0) or (2 < c1 <= 3 and co <= 2 and c2 == 0) or (c1 <= 1 and co == 0 and c2 <= 1):
        return "hard"
    return "extra"


def hardness(example) -> str:
    if example.sql:
        try:
            return spider_hardness(example.sql)
        except (KeyError, IndexError, TypeError):
            pass
    return heuristic_hardness(example.query)


def heuristic_hardness(sql: str) -> str:
    """Fallback when the parsed SQL is unavailable (approximate)."""
    f = sql_features(sql)
    c1 = f["where"] + f["group"] + f["order"] + f["limit"] + f["joins"] + f["or"] + f["like"]
    c2 = f["nested"] + f["setop"]
    co = int(f["n_aggs"] > 1) + int(f["n_select_cols"] > 1)
    if c1 <= 1 and co == 0 and c2 == 0:
        return "easy"
    if c1 <= 2 and co < 2 and c2 == 0:
        return "medium"
    if c2 <= 1 and c1 <= 3:
        return "hard"
    return "extra"


# ------------------------------------------------------------------ error taxonomy
def sql_features(sql: str) -> dict:
    s = " " + re.sub(r"'[^']*'|\"[^\"]*\"", "'v'", sql.lower()) + " "
    sel = re.search(r"select (.*?) from ", s)
    sel_cols = len([c for c in sel.group(1).split(",")]) if sel else 0
    return {
        "joins": len(re.findall(r"\bjoin\b", s)),
        "where": int(" where " in s),
        "group": int(" group by " in s),
        "having": int(" having " in s),
        "order": int(" order by " in s),
        "limit": int(" limit " in s),
        "distinct": int(" distinct " in s),
        "or": len(re.findall(r"\bor\b", s)),
        "like": len(re.findall(r"\blike\b", s)),
        "nested": max(0, len(re.findall(r"\bselect\b", s)) - 1 - len(re.findall(r"\b(union|intersect|except)\b", s))),
        "setop": len(re.findall(r"\b(union|intersect|except)\b", s)),
        "aggs": tuple(sorted(set(re.findall(r"\b(count|sum|avg|min|max)\s*\(", s)))),
        "n_aggs": len(re.findall(r"\b(count|sum|avg|min|max)\s*\(", s)),
        "n_select_cols": sel_cols,
        "literals": tuple(sorted(re.findall(r"'[^']*'|\b\d+(?:\.\d+)?\b", sql.lower()))),
    }


def categorize_error(gold: str, pred: str, pred_error: str | None) -> str:
    if pred_error:
        e = pred_error.lower()
        if e == "empty prediction":
            return "empty / unparsable output"
        if "no such column" in e:
            return "exec: unknown column (schema hallucination)"
        if "no such table" in e:
            return "exec: unknown table (schema hallucination)"
        if "ambiguous column" in e:
            return "exec: ambiguous column (join aliasing)"
        if "syntax error" in e or "incomplete input" in e or "unrecognized token" in e:
            return "exec: syntax error"
        if e == "timeout":
            return "exec: timeout"
        return "exec: other error"
    g, p = sql_features(gold), sql_features(pred)
    if (g["nested"] > 0) != (p["nested"] > 0) or g["setop"] != p["setop"]:
        return "wrong: nesting / set operation"
    if g["joins"] != p["joins"]:
        return "wrong: join structure"
    if g["group"] != p["group"] or g["having"] != p["having"]:
        return "wrong: grouping / having"
    if g["aggs"] != p["aggs"]:
        return "wrong: aggregation"
    if g["order"] != p["order"] or g["limit"] != p["limit"]:
        return "wrong: ordering / limit"
    if g["distinct"] != p["distinct"]:
        return "wrong: distinct"
    if g["where"] != p["where"] or g["literals"] != p["literals"]:
        return "wrong: filter condition / values"
    if g["n_select_cols"] != p["n_select_cols"]:
        return "wrong: selected columns"
    return "wrong: other (column choice / semantics)"
