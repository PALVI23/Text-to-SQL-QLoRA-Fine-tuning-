"""Spider 1.0 loading and schema serialisation (read straight from the SQLite files)."""
from __future__ import annotations

import json
import re
import sqlite3
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass
class Example:
    db_id: str
    question: str
    query: str
    sql: dict | None = None  # Spider's parsed SQL (used for the official hardness levels)
    split: str = ""


def find_spider_root(base: str | Path) -> Path:
    """Spider zips ship as spider/, spider_data/ or nested; locate the folder containing dev.json."""
    base = Path(base)
    for p in [base, *base.glob("*"), *base.glob("*/*")]:
        if (p / "dev.json").exists() and (p / "database").exists():
            return p
    raise FileNotFoundError(f"No Spider folder (with dev.json and database/) under {base}")


def extract_spider_zip(zip_path: str | Path, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    try:
        return find_spider_root(out_dir)
    except FileNotFoundError:
        pass
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    return find_spider_root(out_dir)


SPIDER_GDRIVE_ID = "1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0"   # official Spider 1.0 zip (id used by the HF loader script)
SPIDER_HF_MIRROR = ("HAL-9001/spider-databases", "spider_data.zip")  # community mirror (fallback)


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_spider(out_dir: str | Path, drive_zip: str | Path | None = None) -> Path:
    """Get Spider 1.0 (questions + SQLite databases), trying in order:
    1. a zip you put in Google Drive (`drive_zip`), 2. the official Google Drive file via gdown,
    3. a Hugging Face mirror. Validates the result (dev = 1,034 questions, 200 databases)."""
    import subprocess
    import sys

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        return validate_spider(find_spider_root(out_dir))
    except FileNotFoundError:
        pass
    attempts = []
    if drive_zip and Path(drive_zip).exists():
        attempts.append(("your Drive copy", lambda: Path(drive_zip)))

    def gdrive():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
        import gdown

        z = out_dir / "spider_download.zip"
        gdown.download(id=SPIDER_GDRIVE_ID, output=str(z), quiet=False)
        return z

    def hf_mirror():
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(SPIDER_HF_MIRROR[0], SPIDER_HF_MIRROR[1], repo_type="dataset",
                                    local_dir=str(out_dir / "_hf")))

    attempts += [("official Google Drive (gdown)", gdrive), ("Hugging Face mirror", hf_mirror)]
    for name, fn in attempts:
        try:
            print(f"Trying {name} ...")
            z = fn()
            if not z or not Path(z).exists() or not zipfile.is_zipfile(z):
                raise RuntimeError("download did not produce a zip file")
            print(f"  sha256 = {_sha256(Path(z))}")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(out_dir)
            root = validate_spider(find_spider_root(out_dir))
            print(f"  OK: Spider found at {root}")
            return root
        except Exception as e:  # noqa: BLE001 - try the next source
            print(f"  failed: {type(e).__name__}: {str(e)[:200]}")
    raise RuntimeError(
        "Could not download Spider automatically. Manual route: on a laptop open "
        f"https://drive.google.com/uc?id={SPIDER_GDRIVE_ID} (or the 'Spider 1.0' download on "
        "https://yale-lily.github.io/spider), save the zip as MyDrive/masters-projects/spider_data.zip and re-run.")


def validate_spider(root: Path) -> Path:
    n_dev = len(json.load(open(root / "dev.json")))
    n_db = len([p for p in (root / "database").iterdir() if p.is_dir()])
    print(f"Spider check: dev questions = {n_dev} (expected 1034), databases = {n_db} (expected 166-206)")
    if n_dev < 1000 or n_db < 160:
        print("  WARNING: this does not look like the full Spider 1.0 release.")
    return root


def load_split(root: Path, split: str) -> list[Example]:
    files = {"train": ["train_spider.json", "train_others.json"], "dev": ["dev.json"]}[split]
    out = []
    for f in files:
        if (root / f).exists():
            for r in json.load(open(root / f)):
                out.append(Example(r["db_id"], r["question"].strip(), r["query"].strip(), r.get("sql"), split))
    return out


def db_path(root: Path, db_id: str) -> Path:
    for sub in ["database", "test_database"]:
        p = Path(root) / sub / db_id / f"{db_id}.sqlite"
        if p.exists():
            return p
    raise FileNotFoundError(db_id)


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    conn.text_factory = lambda b: b.decode(errors="ignore")
    return conn


@lru_cache(maxsize=512)
def _schema_info(path: str):
    conn = _connect(Path(path))
    cur = conn.cursor()
    tables = [(n, s) for n, s in cur.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()]
    info = []
    for name, ddl in tables:
        cols = cur.execute(f'PRAGMA table_info("{name}")').fetchall()  # cid, name, type, notnull, dflt, pk
        fks = cur.execute(f'PRAGMA foreign_key_list("{name}")').fetchall()  # id, seq, table, from, to, ...
        try:
            rows = cur.execute(f'SELECT * FROM "{name}" LIMIT 3').fetchall()
        except sqlite3.Error:
            rows = []
        info.append({"name": name, "ddl": ddl or "", "cols": [(c[1], c[2], c[5]) for c in cols],
                     "fks": [(f[3], f[2], f[4]) for f in fks], "rows": rows})
    conn.close()
    return info


def _clean_ddl(ddl: str) -> str:
    ddl = re.sub(r"--[^\n]*", "", ddl)          # drop SQL comments
    ddl = re.sub(r"\s+", " ", ddl).strip()
    return ddl.replace("( ", "(").replace(" )", ")")


def _fmt_val(v, maxlen=40):
    s = repr(v) if not isinstance(v, str) else "'" + v.replace("\n", " ")[:maxlen] + "'"
    return s


def serialize_schema(path: str | Path, fmt: str = "ddl") -> str:
    """fmt:
    compact  - table(col, col, ...) + foreign keys          (fewest tokens)
    ddl      - CREATE TABLE statements (types, PK, FK)
    ddl_rows - CREATE TABLE + 3 example rows per table      (Rajkumar et al., 2022 style)
    """
    info = _schema_info(str(path))
    parts = []
    if fmt == "compact":
        for t in info:
            pk = {c for c, _, ispk in t["cols"] if ispk}
            cols = ", ".join(c + ("*" if c in pk else "") for c, _, _ in t["cols"])
            parts.append(f"{t['name']}({cols})")
        fks = [f"{t['name']}.{a} = {tb}.{b}" for t in info for a, tb, b in t["fks"]]
        if fks:
            parts.append("Foreign keys: " + "; ".join(fks))
        return "\n".join(parts)
    for t in info:
        ddl = _clean_ddl(t["ddl"]) if t["ddl"] else f"CREATE TABLE {t['name']} (" + ", ".join(
            f"{c} {ty}" for c, ty, _ in t["cols"]) + ")"
        parts.append(ddl + ";")
        if fmt == "ddl_rows" and t["rows"]:
            header = " | ".join(c for c, _, _ in t["cols"])
            rows = "\n".join(" | ".join(_fmt_val(v) for v in r) for r in t["rows"])
            parts.append(f"/* 3 example rows from {t['name']}:\n{header}\n{rows}\n*/")
    return "\n".join(parts)


def schema_stats(path: str | Path) -> dict:
    info = _schema_info(str(path))
    return {"n_tables": len(info), "n_columns": sum(len(t["cols"]) for t in info),
            "n_fks": sum(len(t["fks"]) for t in info)}
