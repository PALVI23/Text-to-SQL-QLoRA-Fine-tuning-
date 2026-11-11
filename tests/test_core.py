import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2s.analysis import categorize_error, spider_hardness  # noqa: E402
from t2s.execution import execution_match, results_match  # noqa: E402
from t2s.prompts import extract_sql  # noqa: E402
from t2s.smoke import make_fake_spider  # noqa: E402
from t2s.spider import db_path, find_spider_root, load_split, serialize_schema  # noqa: E402


def test_extract_sql():
    assert extract_sql("```sql\nSELECT a FROM t;\n```") == "SELECT a FROM t"
    assert extract_sql("Here is the query: SELECT count(*) FROM x; -- done") == "SELECT count(*) FROM x"
    assert extract_sql("```\nselect 1") == "select 1"
    assert extract_sql("") == ""


def test_results_match():
    assert results_match([(1, "a"), (2, "b")], [(2, "b"), (1, "a")], ordered=False)
    assert not results_match([(1,), (2,)], [(2,), (1,)], ordered=True)
    assert results_match([(1, "a")], [("a", 1)], ordered=False)  # column permutation
    assert results_match([(0.1 + 0.2,)], [(0.3,)], ordered=False)
    assert not results_match([(1,), (1,)], [(1,)], ordered=False)  # multiset, not set


def test_execution_and_schema(tmp_path):
    root = find_spider_root(make_fake_spider(tmp_path))
    dev = load_split(root, "dev")
    db = db_path(root, dev[0].db_id)
    assert execution_match(db, dev[0].query, dev[0].query)["exec_match"]
    r = execution_match(db, dev[0].query, "SELECT nonexistent FROM student")
    assert not r["exec_match"] and "no such column" in r["pred_error"]
    assert categorize_error(dev[0].query, "SELECT nonexistent FROM student", r["pred_error"]).startswith("exec: unknown column")
    for fmt in ["compact", "ddl", "ddl_rows"]:
        s = serialize_schema(db, fmt)
        assert "student" in s
    assert "example rows" in serialize_schema(db, "ddl_rows")
    assert "Foreign keys" in serialize_schema(db_path(root, "concert_hall"), "compact")


def test_error_taxonomy():
    assert categorize_error("SELECT count(*) FROM a", "SELECT max(x) FROM a", None) == "wrong: aggregation"
    assert categorize_error("SELECT a FROM t JOIN s ON t.id=s.id", "SELECT a FROM t", None) == "wrong: join structure"
    assert categorize_error("SELECT a FROM t WHERE b = 'x'", "SELECT a FROM t WHERE b = 'y'", None) == \
        "wrong: filter condition / values"


def _cu(op=2, nested=None):
    return [False, op, [0, [0, 1, False], None], nested if nested is not None else 1.0, None]


def _sql(**kw):
    base = {"select": [False, [[0, [0, [0, 1, False], None]]]], "from": {"table_units": [["table_unit", 0]], "conds": []},
            "where": [], "groupBy": [], "having": [], "orderBy": [], "limit": None,
            "intersect": None, "union": None, "except": None}
    base.update(kw)
    return base


def test_official_hardness():
    assert spider_hardness(_sql()) == "easy"
    assert spider_hardness(_sql(where=[_cu()])) == "easy"
    assert spider_hardness(_sql(where=[_cu(), "and", _cu()])) == "medium"
    # join of 3 tables + where + order + limit -> comp1 = 2+1+1+1 = 5 -> extra
    assert spider_hardness(_sql(**{"from": {"table_units": [["table_unit", 0]] * 3, "conds": []}},
                                where=[_cu()], orderBy=["desc", [[0, [0, 1, False], None]]], limit=1)) == "extra"
    # one nested query, nothing else -> hard
    assert spider_hardness(_sql(where=[_cu(op=8, nested=_sql())])) == "hard"
