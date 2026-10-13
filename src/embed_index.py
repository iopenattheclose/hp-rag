"""Phase 2, step 2: embed chunks and upsert them into Qdrant.

One Qdrant collection per (strategy, embedding model):  hp_<strategy>__<model_slug>
Each point = vector + payload (book, clearance are payload-indexed for the RBAC pre-filter).

Prereqs:   uv pip install qdrant-client sentence-transformers
           Qdrant running in Docker (already done), hp.duckdb with chunks (build_chunks.py)
Setup:     export EMBED_MODEL_PATH=/path/to/all-MiniLM-L6-v2     # add to ~/.zshrc to keep it
Usage:     python embed_index.py --strategies fixed --limit 200
           python embed_index.py --strategies fixed recursive
           python embed_index.py --strategies sentence_window
           (--model overrides the env var)
Safe to stop and re-run: points already in the collection are skipped.
Env:       QDRANT_URL (default http://localhost:6333), QDRANT_API_KEY (optional)
"""
from __future__ import annotations

import argparse
import os
import time
import uuid

import duckdb
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

from embedding import Embedder, collection_name

NS = uuid.UUID("6f1d3c0a-0000-4000-8000-00000000a11c")      # fixed namespace -> stable ids


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(NS, chunk_id))


def existing_ids(client, coll) -> set[str]:
    seen, offset = set(), None
    while True:
        pts, offset = client.scroll(coll, limit=5000, offset=offset,
                                    with_payload=False, with_vectors=False)
        seen.update(str(p.id) for p in pts)
        if offset is None:
            return seen


def ensure_collection(client, coll, dim):
    if client.collection_exists(coll):
        have = client.get_collection(coll).config.params.vectors.size
        if have != dim:
            raise SystemExit(f"{coll} has vector size {have}, this model gives {dim}. "
                             f"Different model -> use a different collection.")
        return
    client.create_collection(coll, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
    # the RBAC pre-filter needs these; without them filtering is slow
    client.create_payload_index(coll, "book", PayloadSchemaType.INTEGER)
    client.create_payload_index(coll, "clearance", PayloadSchemaType.INTEGER)
    client.create_payload_index(coll, "chunk_id", PayloadSchemaType.KEYWORD)


def truncation_report(emb, rows, strategy):
    """Warn if chunks are longer than the model can read (the tail is silently dropped)."""
    if not emb.max_tokens:
        return
    sample = rows[:: max(1, len(rows) // 400)][:400]
    counts = emb.count_tokens([r["text"] for r in sample])
    over = sum(1 for c in counts if c > emb.max_tokens)
    print(f"  token check ({strategy}): avg {sum(counts)/len(counts):.0f}, max {max(counts)}, "
          f"model limit {emb.max_tokens}; {100*over/len(counts):.0f}% of sampled chunks are truncated")
    if over / len(counts) > 0.1:
        print("  !! many chunks exceed the model's limit: the end of each chunk is NOT embedded. "
              "Use a longer-context model or smaller chunks.")


def run(strategy, args, con, client, emb):
    coll = collection_name(strategy, args.model)
    q = ("SELECT chunk_id, book, chapter_number, chapter_title, chunk_index, char_start, char_end, "
         "text, context_text, n_words, clearance, characters, topics, seg_ids "
         "FROM chunks WHERE strategy = ? ORDER BY book, chapter_number, chunk_index")
    cur = con.execute(q, [strategy])
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print(f"{strategy}: no chunks in DuckDB - run build_chunks.py first")
        return

    ensure_collection(client, coll, emb.dim)
    truncation_report(emb, rows, strategy)
    done = existing_ids(client, coll)
    todo = [r for r in rows if point_id(r["chunk_id"]) not in done]
    print(f"{strategy}: {len(rows):,} chunks, {len(done):,} already in {coll}, "
          f"{len(todo):,} to do  (dim={emb.dim})")

    t0 = time.time()
    for i in range(0, len(todo), args.batch):
        batch = todo[i:i + args.batch]
        vecs = emb.embed([r["text"] for r in batch])
        pts = [
            PointStruct(
                id=point_id(r["chunk_id"]), vector=v,
                payload={
                    "chunk_id": r["chunk_id"], "strategy": strategy,
                    "book": r["book"], "chapter_number": r["chapter_number"],
                    "chapter_title": r["chapter_title"], "chunk_index": r["chunk_index"],
                    "clearance": r["clearance"],
                    "char_start": r["char_start"], "char_end": r["char_end"],
                    "n_words": r["n_words"],
                    "characters": r["characters"], "topics": r["topics"], "seg_ids": r["seg_ids"],
                    "text": r["text"], "context_text": r["context_text"],
                },
            )
            for r, v in zip(batch, vecs)
        ]
        client.upsert(coll, points=pts, wait=True)
        n = i + len(batch)
        if (i // args.batch) % 10 == 0 or n == len(todo):
            rate = n / (time.time() - t0)
            eta = (len(todo) - n) / rate / 60
            print(f"  {n:,}/{len(todo):,}  {rate:.1f} chunks/s  ETA {eta:.1f} min")
    print(f"{strategy}: done -> collection {coll} has {client.count(coll).count:,} points")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="hp.duckdb")
    ap.add_argument("--strategies", nargs="+", default=["fixed", "recursive", "sentence_window"])
    ap.add_argument("--model", default=os.environ.get("EMBED_MODEL_PATH"),
                    help="local folder or HF id of the embedding model (default: env EMBED_MODEL_PATH)")
    ap.add_argument("--backend", default="st", choices=["st", "ollama"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="only the first N chunks (smoke test)")
    args = ap.parse_args()
    if not args.model:
        raise SystemExit("No embedding model: set EMBED_MODEL_PATH (e.g. export "
                         "EMBED_MODEL_PATH=/path/to/all-MiniLM-L6-v2) or pass --model.")

    emb = Embedder(args.backend, args.model, args.batch)
    con = duckdb.connect(args.db, read_only=True)
    client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
                          api_key=os.environ.get("QDRANT_API_KEY"))
    for s in args.strategies:
        run(s, args, con, client, emb)


if __name__ == "__main__":
    main()