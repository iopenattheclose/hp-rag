"""Small in-memory BM25 (numpy only).  Used for hybrid retrieval in retrieve.py.

RBAC: search() takes an `allowed` boolean mask.  Disallowed chunks are removed
BEFORE ranking, so they can never appear in the results (pre-filter, same as Qdrant).
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

_TOK = re.compile(r"[a-z0-9]+")
STOP = set("""a an and are as at be but by for from had has have he her him his i if in is it its
me my not of on or our she so that the their them they this to was we were what when where which
who whom why will with you your do did does""".split())


def tokenize(text: str) -> list[str]:
    # apostrophes split words ("prefects'" -> "prefects"); hyphens split too ("pine-fresh")
    t = text.lower().replace("’", " ").replace("'", " ")
    return [w for w in _TOK.findall(t) if w not in STOP]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.n = k1, len(texts)
        docs, tfs = defaultdict(list), defaultdict(list)
        dl = np.zeros(self.n, dtype=np.float32)
        for i, t in enumerate(texts):
            toks = tokenize(t)
            dl[i] = len(toks)
            for term, c in Counter(toks).items():
                docs[term].append(i)
                tfs[term].append(c)
        self.post = {t: (np.asarray(docs[t], dtype=np.int32), np.asarray(tfs[t], dtype=np.float32))
                     for t in docs}
        avg = float(dl.mean()) or 1.0
        self.norm = (k1 * (1 - b + b * dl / avg)).astype(np.float32)

    def search(self, query: str, allowed: np.ndarray | None = None, k: int = 50):
        """-> [(doc_index, score)] best first; only docs matching >=1 query term and `allowed`."""
        scores = np.zeros(self.n, dtype=np.float32)
        hit = np.zeros(self.n, dtype=bool)
        for term in set(tokenize(query)):
            p = self.post.get(term)
            if p is None:
                continue
            d, tf = p
            df = len(d)
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            scores[d] += idf * tf * (self.k1 + 1) / (tf + self.norm[d])
            hit[d] = True
        if allowed is not None:
            hit &= allowed
        idx = np.flatnonzero(hit)
        if idx.size == 0:
            return []
        top = idx[np.argsort(-scores[idx])[:k]]
        return [(int(i), float(scores[i])) for i in top]