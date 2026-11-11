"""Gradio demo: natural-language question -> SQL (fine-tuned model) -> executed result table."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from t2s import modeling as MD  # noqa: E402
from t2s.execution import execute  # noqa: E402
from t2s.prompts import build_messages, extract_sql  # noqa: E402
from t2s.spider import _schema_info, db_path, serialize_schema  # noqa: E402


def build_demo(base_model, adapter, spider_root, schema_fmt, db_ids):
    import gradio as gr

    tok = MD.load_tokenizer(base_model)
    model = MD.load_model(base_model, load_4bit=False, adapter=adapter, merge=True)

    def show_schema(db_id):
        return serialize_schema(db_path(spider_root, db_id), "compact")

    def ask(db_id, question):
        db = db_path(spider_root, db_id)
        msgs = build_messages(serialize_schema(db, schema_fmt), question)
        raw = MD.generate(model, tok, [msgs], batch_size=1, progress=False)[0]
        sql = extract_sql(raw)
        rows, err = execute(db, sql)
        if err:
            return sql, pd.DataFrame({"error": [err]})
        cols = None
        try:
            import sqlite3
            cur = sqlite3.connect(f"file:{db}?mode=ro", uri=True).execute(sql)
            cols = [d[0] for d in cur.description]
        except Exception:  # noqa: BLE001
            pass
        return sql, pd.DataFrame(rows[:200], columns=cols if cols and len(cols) == len(rows[0]) else None) if rows else pd.DataFrame()

    with gr.Blocks(title="Text-to-SQL demo") as demo:
        gr.Markdown("# Text-to-SQL with a QLoRA-fine-tuned Qwen2.5-Coder-1.5B\nPick a Spider database and ask a question.")
        with gr.Row():
            db = gr.Dropdown(db_ids, value=db_ids[0], label="Database")
            q = gr.Textbox(label="Question", value="How many singers are there?")
        schema = gr.Textbox(label="Schema", lines=6)
        btn = gr.Button("Generate SQL and run", variant="primary")
        sql = gr.Code(label="Generated SQL", language="sql")
        table = gr.Dataframe(label="Result")
        db.change(show_schema, db, schema)
        demo.load(show_schema, db, schema)
        btn.click(ask, [db, q], [sql, table])
    return demo
