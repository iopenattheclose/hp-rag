"""Hybrid retrieval with RBAC:  dense (Qdrant) + BM25  ->  RRF fusion  ->  cross-encoder rerank.

RBAC is applied BEFORE every stage:
  dense   Qdrant pre-filter  book IN role.books AND clearance <= role.max_clearance
  BM25    disallowed chunks are masked out before ranking
  rerank  only sees fused candidates, which are already allowed
and every final hit is re-verified against DuckDB (fail closed).

Usage:
  python retrieve.py --user vernon --strategy recursive --q "Who told Harry the password to the prefects' bathroom?"
  python retrieve.py ... --target recursive:b4_c23_00039      # where does this chunk rank at each stage?
  python retrieve.py ... --no-bm25 --no-rerank                # dense only (the old behaviour)
Env: EMBED_MODEL_PATH, RERANK_MODEL (default BAAI/bge-reranker-base), QDRANT_URL, QDRANT_API_KEY
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from bm25 import BM25
from embedding import Embedder, collection_name

RRF_K = 60


def rrf(rank_lists: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal rank fusion over lists of doc indices (best first)."""
    score: dict[int, float] = {}
    for lst in rank_lists:
        for r, d in enumerate(lst, 1):
            score[d] = score.get(d, 0.0) + 1.0 / (k + r)
    return sorted(score.items(), key=lambda x: -x[1])


class Retriever:
    def __init__(self, strategy: str, model: str, backend: str = "st", db: str = "hp.duckdb",
                 rerank_model: str | None = None):
        import duckdb
        from qdrant_client import QdrantClient
        self.strategy, self.model = strategy, model
        self.con = duckdb.connect(db, read_only=True)
        rows = self.con.execute(
            "SELECT chunk_id, book, chapter_number, chapter_title, clearance, context_text, "
            "char_start, char_end FROM chunks WHERE strategy = ? ORDER BY chunk_id", [strategy]).fetchall()
        if not rows:
            raise SystemExit(f"no chunks for strategy {strategy} in DuckDB")
        self.ids = [r[0] for r in rows]
        self.idx = {c: i for i, c in enumerate(self.ids)}
        self.book = np.array([r[1] for r in rows])
        self.chapter = [(r[2], r[3]) for r in rows]
        self.clr = np.array([r[4] for r in rows])
        self.ctx = [r[5] for r in rows]
        self.span = [(r[6], r[7]) for r in rows]
        print(f"[retriever] building BM25 over {len(rows):,} chunks ...")
        self.bm25 = BM25(self.ctx)
        self.emb = Embedder(backend, model)
        self.coll = collection_name(strategy, model)
        self.client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
                                   api_key=os.environ.get("QDRANT_API_KEY"))
        self.rerank_name = rerank_model or os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base")
        self._ce = None

    def _reranker(self):
        if self._ce is None:
            from sentence_transformers import CrossEncoder
            import torch
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            self._ce = CrossEncoder(self.rerank_name, device=dev, max_length=512)
            print(f"[retriever] reranker {self.rerank_name} on {dev}")
        return self._ce

    def _dedupe(self, order: list[int], max_overlap: float = 0.3) -> list[int]:
        """Drop a hit if >max_overlap of it is already covered by a better-ranked hit
        (same book + chapter). Overlapping chunks would otherwise waste top-k slots."""
        kept: list[int] = []
        for i in order:
            a0, a1 = self.span[i]
            dup = False
            for j in kept:
                if self.book[j] != self.book[i] or self.chapter[j][0] != self.chapter[i][0]:
                    continue
                b0, b1 = self.span[j]
                ov = min(a1, b1) - max(a0, b0)
                if ov > 0 and ov / max(1, min(a1 - a0, b1 - b0)) > max_overlap:
                    dup = True
                    break
            if not dup:
                kept.append(i)
        return kept

    def allowed_mask(self, role: dict) -> np.ndarray:
        return np.isin(self.book, role["books"]) & (self.clr <= role["max_clearance"])

    def search(self, role: dict | None, query: str, k: int = 8, n_dense: int = 50, n_bm25: int = 50,
               n_rerank: int = 40, use_bm25: bool = True, use_rerank: bool = True,
               dedupe: bool = True):
        if role is None:
            raise PermissionError("unknown role -> refuse (fail closed)")
        from qdrant_client.models import FieldCondition, Filter, MatchAny, Range
        allowed = self.allowed_mask(role)
        flt = Filter(must=[FieldCondition(key="book", match=MatchAny(any=role["books"])),
                           FieldCondition(key="clearance", range=Range(lte=role["max_clearance"]))])
        blocked: list[str] = []

        def keep(i: int) -> bool:               # belt and braces: never trust one layer
            if allowed[i]:
                return True
            blocked.append(self.ids[i])
            return False

        vec = self.emb.embed([query])[0]
        pts = self.client.query_points(self.coll, query=vec, query_filter=flt, limit=n_dense,
                                       with_payload=["chunk_id"]).points
        dense = [i for i in (self.idx[p.payload["chunk_id"]] for p in pts) if keep(i)]
        lexical = [i for i, _ in self.bm25.search(query, allowed=allowed, k=n_bm25)] if use_bm25 else []

        fused = rrf([dense, lexical] if use_bm25 else [dense])
        cand = [i for i, _ in fused[:n_rerank] if keep(i)]
        ranks = {
            "dense": {i: r for r, i in enumerate(dense, 1)},
            "bm25": {i: r for r, i in enumerate(lexical, 1)},
            "fused": {i: r for r, i in enumerate((i for i, _ in fused), 1)},
        }
        final = cand
        rr_scores: dict[int, float] = {}
        if use_rerank and cand:
            sc = self._reranker().predict([(query, self.ctx[i]) for i in cand], show_progress_bar=False)
            rr_scores = {i: float(s) for i, s in zip(cand, sc)}
            final = sorted(cand, key=lambda i: -rr_scores[i])
            ranks["rerank"] = {i: r for r, i in enumerate(final, 1)}
        if dedupe:
            final = self._dedupe(final)
        hits = [{"idx": i, "chunk_id": self.ids[i], "book": int(self.book[i]), "clearance": int(self.clr[i]),
                 "chapter_number": self.chapter[i][0],
                 "chapter": self.chapter[i][1], "context_text": self.ctx[i],
                 "dense_rank": ranks["dense"].get(i), "bm25_rank": ranks["bm25"].get(i),
                 "fused_rank": ranks["fused"].get(i), "rerank_score": rr_scores.get(i)}
                for i in final[:k]]
        self._verify(hits, role)
        return hits, ranks, blocked

    def _verify(self, hits, role):
        """Second check against DuckDB. Anything not allowed here is a bug -> raise."""
        ids = [h["chunk_id"] for h in hits]
        if not ids:
            return
        ph = ",".join("?" * len(ids))
        books = ",".join(str(int(b)) for b in role["books"])
        ok = self.con.execute(
            f"SELECT count(*) FROM chunks WHERE chunk_id IN ({ph}) AND clearance <= ? AND book IN ({books})",
            ids + [role["max_clearance"]]).fetchone()[0]
        if ok != len(ids):
            raise RuntimeError(f"RBAC verification failed: {ok}/{len(ids)} hits allowed")


def main():
    import db_schema
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--q", required=True)
    ap.add_argument("--strategy", default="recursive")
    ap.add_argument("--model", default=os.environ.get("EMBED_MODEL_PATH"))
    ap.add_argument("--backend", default="st", choices=["st", "ollama"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--no-bm25", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--target", help="chunk_id to track through the stages")
    ap.add_argument("--db", default="hp.duckdb")
    args = ap.parse_args()
    if not args.model:
        raise SystemExit("set EMBED_MODEL_PATH or pass --model")

    r = Retriever(args.strategy, args.model, args.backend, args.db)
    role = db_schema.get_role(r.con, args.user)
    hits, ranks, blocked = r.search(role, args.q, k=args.k, use_bm25=not args.no_bm25,
                                    use_rerank=not args.no_rerank)
    print(f"\nuser={args.user} role={role['role_id']} max_clearance={role['max_clearance']}  "
          f"bm25={'off' if args.no_bm25 else 'on'} rerank={'off' if args.no_rerank else 'on'}\n")
    for n, h in enumerate(hits, 1):
        rs = "" if h["rerank_score"] is None else f"rerank={h['rerank_score']:.3f}  "
        print(f"#{n}  {rs}dense={h['dense_rank']} bm25={h['bm25_rank']} fused={h['fused_rank']}  "
              f"{h['chunk_id']} book={h['book']} clr={h['clearance']}  {h['chapter']}")
        print("    ", h["context_text"][:220].replace("\n", " "), "...\n")
    if blocked:
        print("!! blocked by second check:", blocked)
    if args.target:
        i = r.idx.get(args.target)
        if i is None:
            print("target not found")
        else:
            print(f"target {args.target}: " + "  ".join(
                f"{s}={ranks[s].get(i, '-')}" for s in ("dense", "bm25", "fused", "rerank") if s in ranks))


if __name__ == "__main__":
    main()