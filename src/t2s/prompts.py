"""Chat prompts and SQL extraction."""
from __future__ import annotations

import re

SYSTEM = ("You are an expert SQLite developer. Given a database schema and a question, write a single "
          "SQLite query that answers the question. Return only the SQL query, with no explanation.")


def user_turn(schema: str, question: str) -> str:
    return f"### Database schema\n{schema}\n\n### Question\n{question}"


def build_messages(schema: str, question: str, answer: str | None = None, fewshot: list[tuple] | None = None):
    """fewshot: list of (schema, question, sql) demonstrations, given as previous chat turns."""
    msgs = [{"role": "system", "content": SYSTEM}]
    for s, q, a in fewshot or []:
        msgs += [{"role": "user", "content": user_turn(s, q)}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": user_turn(schema, question)})
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


_FENCE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.S | re.I)


def extract_sql(text: str) -> str:
    """Pull one SQL statement out of a model response (handles code fences and chatter)."""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    elif "```" in text:  # unterminated fence
        text = text.split("```", 1)[1]
        text = re.sub(r"^(sql|sqlite)\s", "", text.strip(), flags=re.I)
    # start at the first SELECT/WITH if the model added a preamble
    m = re.search(r"\b(SELECT|WITH)\b", text, re.I)
    if m:
        text = text[m.start():]
    text = text.split(";")[0]
    return re.sub(r"\s+", " ", text).strip()


def normalize_gold(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).strip()
