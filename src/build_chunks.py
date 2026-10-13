"""Phase 2, step 1: run the chunkers and write the chunk tables.

Reads  data/processed/chapters.parquet (or .jsonl) and segments.parquet (or .jsonl)
Writes data/processed/chunks_<strategy>.{parquet,jsonl}
       data/processed/chunk_report.txt
Then (unless --no-db) loads everything into hp.duckdb via db_schema.load_chunks().

Usage:  python build_chunks.py [--strategies fixed recursive sentence_window] [--no-db]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from build_metadata import load_chapters, write_table
from chunkers import STRATEGIES, SegmentIndex, inherit


def load_rows(path: Path):
    return load_chapters(path)          # parquet if present, else the .jsonl twin


def chunk_book(strategy, chapters, seg_index):
    rows = []
    fn = STRATEGIES[strategy]
    for ch in chapters:
        b, n, text = ch["book"], ch["chapter_number"], ch["text"]
        idx = seg_index[(b, n)]
        for i, (a, z, body, ctx) in enumerate(fn(text)):
            if not body.strip():
                continue
            inh = inherit(idx, a, z)
            context = ctx if ctx is not None else body
            rows.append({
                "chunk_id": f"{strategy}:b{b}_c{n:02d}_{i:05d}",
                "strategy": strategy, "book": b, "chapter_number": n,
                "chapter_title": ch["chapter_title"], "chunk_index": i,
                "char_start": a, "char_end": z,
                "text": body,                 # what gets embedded
                "context_text": context,      # what the LLM is shown (== text except sentence_window)
                "n_words": len(body.split()),
                **inh,
            })
    return rows


def report(all_rows, out: Path):
    lines = ["=== Chunks per strategy ===",
             f"{'strategy':<17}{'chunks':>8}{'avg_w':>8}{'min_w':>7}{'max_w':>7}"
             f"{'clr0%':>8}{'clr1%':>8}{'clr2%':>8}"]
    for strat, rows in all_rows.items():
        w = [r["n_words"] for r in rows]
        c = Counter(r["clearance"] for r in rows)
        t = len(rows)
        lines.append(f"{strat:<17}{t:>8}{sum(w)/t:>8.0f}{min(w):>7}{max(w):>7}"
                     f"{100*c[0]/t:>8.1f}{100*c[1]/t:>8.1f}{100*c[2]/t:>8.1f}")
    lines += ["", "(Higher clr1/clr2 % = more content locked away from low roles: that is the",
              " over-restriction cost of big chunks.)"]
    txt = "\n".join(lines)
    print(txt)
    (out / "chunk_report.txt").write_text(txt, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters", default="data/processed/chapters.parquet")
    ap.add_argument("--segments", default="data/processed/segments.parquet")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    ap.add_argument("--no-db", action="store_true")
    ap.add_argument("--db", default="hp.duckdb")
    args = ap.parse_args()
    out = Path(args.out_dir)

    chapters = load_rows(Path(args.chapters))
    by_ch = defaultdict(list)
    for s in load_rows(Path(args.segments)):
        by_ch[(s["book"], s["chapter_number"])].append(s)
    seg_index = {k: SegmentIndex(v) for k, v in by_ch.items()}

    all_rows = {}
    for strat in args.strategies:
        rows = chunk_book(strat, chapters, seg_index)
        # inherited lists may be numpy arrays after a parquet round trip -> plain lists for json
        write_table(rows, out / f"chunks_{strat}")
        all_rows[strat] = rows
        print(f"{strat}: {len(rows):,} chunks")
    report(all_rows, out)

    if not args.no_db:
        import db_schema
        con = db_schema.connect(args.db)
        for strat, rows in all_rows.items():
            db_schema.load_chunks(con, strat, rows)
        print(f"Loaded chunks into {args.db}")


if __name__ == "__main__":
    main()