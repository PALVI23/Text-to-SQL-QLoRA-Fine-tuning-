"""Fixtures for CPU smoke tests: a miniature Spider-format dataset (real SQLite files) and a tiny,
randomly initialised Qwen2-architecture model with a chat template. Lets the full train/generate/eval
pipeline run in about a minute without a GPU or downloads."""
from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path

DBS = {
    "concert_hall": {
        "ddl": ["CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, country TEXT, age INTEGER)",
                "CREATE TABLE concert (concert_id INTEGER PRIMARY KEY, title TEXT, year INTEGER, singer_id INTEGER, "
                "FOREIGN KEY (singer_id) REFERENCES singer(singer_id))"],
        "rows": {"singer": [(i, f"Singer{i}", random.Random(i).choice(["France", "USA", "India"]), 20 + i) for i in range(1, 11)],
                 "concert": [(i, f"Show{i}", 2010 + i % 5, 1 + i % 10) for i in range(1, 21)]},
        "qs": [("How many singers are there?", "SELECT count(*) FROM singer"),
               ("List the names of singers from France.", "SELECT name FROM singer WHERE country = 'France'"),
               ("What is the average age of singers?", "SELECT avg(age) FROM singer"),
               ("Show concert titles in 2012.", "SELECT title FROM concert WHERE year = 2012"),
               ("Count concerts per year.", "SELECT year, count(*) FROM concert GROUP BY year"),
               ("Name of the oldest singer?", "SELECT name FROM singer ORDER BY age DESC LIMIT 1"),
               ("Titles of concerts by singers from India.",
                "SELECT T1.title FROM concert AS T1 JOIN singer AS T2 ON T1.singer_id = T2.singer_id WHERE T2.country = 'India'")],
    },
    "bookshop": {
        "ddl": ["CREATE TABLE book (book_id INTEGER PRIMARY KEY, title TEXT, price REAL, genre TEXT)",
                "CREATE TABLE sale (sale_id INTEGER PRIMARY KEY, book_id INTEGER, qty INTEGER, "
                "FOREIGN KEY (book_id) REFERENCES book(book_id))"],
        "rows": {"book": [(i, f"Book{i}", 5.0 + i, ["fiction", "science"][i % 2]) for i in range(1, 13)],
                 "sale": [(i, 1 + i % 12, i % 4 + 1) for i in range(1, 30)]},
        "qs": [("How many books are there?", "SELECT count(*) FROM book"),
               ("What is the maximum price?", "SELECT max(price) FROM book"),
               ("List titles of fiction books.", "SELECT title FROM book WHERE genre = 'fiction'"),
               ("Total quantity sold?", "SELECT sum(qty) FROM sale"),
               ("Number of books per genre.", "SELECT genre, count(*) FROM book GROUP BY genre"),
               ("Titles of books that were never sold.",
                "SELECT title FROM book WHERE book_id NOT IN (SELECT book_id FROM sale)")],
    },
    "school": {
        "ddl": ["CREATE TABLE student (student_id INTEGER PRIMARY KEY, name TEXT, grade INTEGER, city TEXT)"],
        "rows": {"student": [(i, f"Stu{i}", 6 + i % 6, ["Pune", "Delhi", "Bangalore"][i % 3]) for i in range(1, 25)]},
        "qs": [("How many students are there?", "SELECT count(*) FROM student"),
               ("Names of students in grade 8.", "SELECT name FROM student WHERE grade = 8"),
               ("Which city has the most students?",
                "SELECT city FROM student GROUP BY city ORDER BY count(*) DESC LIMIT 1"),
               ("Distinct cities of students.", "SELECT DISTINCT city FROM student")],
    },
}
TRAIN_DBS, DEV_DBS = ["concert_hall", "bookshop"], ["school"]


def make_fake_spider(root: str | Path) -> Path:
    root = Path(root) / "spider_data"
    for db, spec in DBS.items():
        d = root / "database" / db
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{db}.sqlite"
        if p.exists():
            p.unlink()
        conn = sqlite3.connect(p)
        for ddl in spec["ddl"]:
            conn.execute(ddl)
        for t, rows in spec["rows"].items():
            conn.executemany(f"INSERT INTO {t} VALUES ({','.join('?' * len(rows[0]))})", rows)
        conn.commit()
        conn.close()

    def items(dbs, reps):
        return [{"db_id": db, "question": q, "query": s} for db in dbs for q, s in DBS[db]["qs"]] * reps

    json.dump(items(TRAIN_DBS, 6), open(root / "train_spider.json", "w"))
    json.dump(items(DEV_DBS, 1), open(root / "dev.json", "w"))
    json.dump([], open(root / "tables.json", "w"))
    return root


QWEN_STYLE_TEMPLATE = (
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")


def make_tiny_model(out_dir: str | Path, corpus: list[str]) -> str:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    out_dir = Path(out_dir)
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "[UNK]"]
    tk = Tokenizer(models.BPE(unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=600, special_tokens=specials,
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                  unk_token="[UNK]")
    tok.chat_template = QWEN_STYLE_TEMPLATE
    tok.save_pretrained(out_dir)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, tie_word_embeddings=True)
    Qwen2ForCausalLM(cfg).save_pretrained(out_dir)
    return str(out_dir)
