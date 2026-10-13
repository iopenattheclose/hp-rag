"""
Roles are seeded from hp_lexicon.ROLES, so that file stays the single source of truth.

Tables
  roles / role_books   what each role may see (books + max clearance)
  users                who is who (demo users seeded)
  chunks               every chunk of every strategy, with inherited clearance/characters/topics
  audit_log            one row per question: who, what, which chunks, allowed/denied
  gold_questions       hand-labelled eval questions (+ which role should/shouldn't answer)
  eval_runs / eval_results   metrics per run and per question

The views chapters / segments / sentences (over the parquet files) are kept so
the DuckDB UI still works.  No FOREIGN KEY constraints on purpose: DuckDB's FK
handling makes bulk reloads (delete + insert) painful, and the gateway checks roles itself.
"""
from __future__ import annotations

import argparse

import duckdb

import hp_lexicon as lx

DESCRIPTIONS = {
    "muggle": "Sees the plot, not the secrets. Books 1-7, clearance 0.",
    "first_year": "Book 1 only, may learn spells and potions (clearance 1).",
    "professor": "All books, spells and potions, but no Dark Arts specifics (clearance 1).",
    "restricted_section": "Full library access, all books, clearance 2.",
}

DEMO_USERS = [
    ("vernon", "muggle"),
    ("harry", "first_year"),
    ("mcgonagall", "professor"),
    ("hermione", "restricted_section"),
]

DDL = """
CREATE TABLE IF NOT EXISTS roles (
    role_id        VARCHAR PRIMARY KEY,
    description    VARCHAR,
    max_clearance  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS role_books (
    role_id  VARCHAR NOT NULL,
    book     INTEGER NOT NULL,
    PRIMARY KEY (role_id, book)
);
CREATE TABLE IF NOT EXISTS users (
    user_id     VARCHAR PRIMARY KEY,
    role_id     VARCHAR NOT NULL,
    created_at  TIMESTAMP DEFAULT current_timestamp
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id         VARCHAR PRIMARY KEY,
    strategy         VARCHAR NOT NULL,
    book             INTEGER NOT NULL,
    chapter_number   INTEGER NOT NULL,
    chapter_title    VARCHAR,
    chunk_index      INTEGER NOT NULL,
    char_start       INTEGER NOT NULL,
    char_end         INTEGER NOT NULL,
    text             VARCHAR NOT NULL,      -- what is embedded
    context_text     VARCHAR NOT NULL,      -- what the LLM is shown
    n_words          INTEGER NOT NULL,
    clearance        INTEGER NOT NULL,      -- max over overlapping segments
    characters       VARCHAR[],
    topics           VARCHAR[],
    clearance_terms  VARCHAR[],
    seg_ids          VARCHAR[]
);
CREATE INDEX IF NOT EXISTS idx_chunks_strategy ON chunks(strategy);
CREATE SEQUENCE IF NOT EXISTS audit_seq;
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id        BIGINT DEFAULT nextval('audit_seq') PRIMARY KEY,
    ts              TIMESTAMP DEFAULT current_timestamp,
    user_id         VARCHAR,
    role_id         VARCHAR,
    strategy        VARCHAR,
    question        VARCHAR,
    filter_applied  VARCHAR,                -- the exact book/clearance filter injected
    retrieved_ids   VARCHAR[],
    blocked_ids     VARCHAR[],              -- failed the second DuckDB check
    outcome         VARCHAR,                -- answered | refused | error
    guard_trips     VARCHAR[],              -- e.g. similarity_cutoff, output_guard
    answer          VARCHAR,
    latency_ms      INTEGER
);
CREATE TABLE IF NOT EXISTS gold_questions (
    q_id             VARCHAR PRIMARY KEY,
    question         VARCHAR NOT NULL,
    reference_answer VARCHAR,
    gold_chunk_seg_ids VARCHAR[],           -- segments that contain the answer
    min_clearance    INTEGER NOT NULL,      -- hand-labelled: lowest clearance that should answer it
    books            INTEGER[],             -- books where the answer lives
    kind             VARCHAR                -- factual | restricted | out_of_scope | trap
);
CREATE SEQUENCE IF NOT EXISTS eval_run_seq;
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id      BIGINT DEFAULT nextval('eval_run_seq') PRIMARY KEY,
    ts          TIMESTAMP DEFAULT current_timestamp,
    strategy    VARCHAR,
    embed_model VARCHAR,
    llm_model   VARCHAR,
    fault_injection BOOLEAN DEFAULT false, -- true = RBAC filter deliberately disabled
    notes       VARCHAR
);
CREATE TABLE IF NOT EXISTS eval_results (
    run_id        BIGINT NOT NULL,
    q_id          VARCHAR NOT NULL,
    role_id       VARCHAR NOT NULL,
    recall_at_k   DOUBLE,
    mrr           DOUBLE,
    leaked        BOOLEAN,                  -- gold-labelled restricted content reached the prompt/answer
    false_deny    BOOLEAN,                  -- allowed question was refused
    faithfulness  DOUBLE,
    latency_ms    INTEGER,
    PRIMARY KEY (run_id, q_id, role_id)
);
"""

VIEWS = {
    "chapters": "data/processed/chapters.parquet",
    "segments": "data/processed/segments.parquet",
    "sentences": "data/processed/sentences.parquet",
}


def connect(db: str = "hp.duckdb") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(db)
    con.execute(DDL)
    return con


def seed(con):
    for role, cfg in lx.ROLES.items():
        con.execute("INSERT OR REPLACE INTO roles VALUES (?, ?, ?)",
                    [role, DESCRIPTIONS.get(role, ""), cfg["max_clearance"]])
        con.execute("DELETE FROM role_books WHERE role_id = ?", [role])
        for b in cfg["books"]:
            con.execute("INSERT INTO role_books VALUES (?, ?)", [role, b])
    # drop roles that no longer exist in hp_lexicon (keeps the table honest)
    keep = list(lx.ROLES)
    ph = ",".join("?" * len(keep))
    con.execute(f"DELETE FROM roles WHERE role_id NOT IN ({ph})", keep)
    con.execute(f"DELETE FROM role_books WHERE role_id NOT IN ({ph})", keep)
    for user, role in DEMO_USERS:
        con.execute("INSERT OR REPLACE INTO users (user_id, role_id) VALUES (?, ?)", [user, role])


def make_views(con):
    from pathlib import Path
    for name, path in VIEWS.items():
        if Path(path).exists():
            # a table of the same name would block this; they are views by design
            con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{path}')")


def load_chunks(con, strategy: str, rows: list[dict]):
    """Replace all chunks of one strategy."""
    import pandas as pd
    con.execute("DELETE FROM chunks WHERE strategy = ?", [strategy])
    df = pd.DataFrame(rows)
    con.register("chunk_df", df)
    con.execute("""
        INSERT INTO chunks
        SELECT chunk_id, strategy, book, chapter_number, chapter_title, chunk_index,
               char_start, char_end, text, context_text, n_words, clearance,
               CAST(characters AS VARCHAR[]), CAST(topics AS VARCHAR[]),
               CAST(clearance_terms AS VARCHAR[]), CAST(seg_ids AS VARCHAR[])
        FROM chunk_df
    """)
    con.unregister("chunk_df")


def get_role(con, user_id: str):
    """Role row for a user, or None.  The gateway must FAIL CLOSED on None."""
    r = con.execute("""
        SELECT r.role_id, r.max_clearance,
               list(rb.book ORDER BY rb.book) AS books
        FROM users u JOIN roles r USING (role_id) JOIN role_books rb USING (role_id)
        WHERE u.user_id = ? GROUP BY r.role_id, r.max_clearance
    """, [user_id]).fetchone()
    if not r:
        return None
    return {"role_id": r[0], "max_clearance": r[1], "books": list(r[2])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="hp.duckdb")
    args = ap.parse_args()
    con = connect(args.db)
    seed(con)
    make_views(con)
    print("Tables:", [r[0] for r in con.execute("SHOW TABLES").fetchall()])
    print(con.execute("""SELECT u.user_id, r.role_id, r.max_clearance,
                         list(rb.book ORDER BY rb.book) AS books
                         FROM users u JOIN roles r USING(role_id) JOIN role_books rb USING(role_id)
                         GROUP BY ALL ORDER BY 1""").fetchall())


if __name__ == "__main__":
    main()