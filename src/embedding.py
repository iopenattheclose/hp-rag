"""Embedding backends, one interface.  Used by embed_index.py and search_test.py.

  --backend st      sentence-transformers (default). Runs on the Mac GPU (MPS). Works with a
                    local folder or a HF id, e.g. all-MiniLM-L6-v2, bge-small-en-v1.5, bge-m3.
  --backend ollama  Ollama's /api/embed, if you ever use it.

The collection name includes the model, because vector sizes differ per model and you must
never mix two models in one collection:  hp_<strategy>__<model_slug>
"""
from __future__ import annotations

import os
import re


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", os.path.basename(model.rstrip("/"))).strip("_").lower()


def collection_name(strategy: str, model: str) -> str:
    return f"hp_{strategy}__{slug(model)}"


class Embedder:
    def __init__(self, backend: str, model: str, batch: int = 64):
        self.backend, self.model, self.batch = backend, model, batch
        self.slug = slug(model)
        self.max_tokens = None
        if backend == "st":
            from sentence_transformers import SentenceTransformer
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._m = SentenceTransformer(model, device=device)
            self.max_tokens = self._m.max_seq_length
            print(f"[embedder] sentence-transformers  model={model}  device={device}  "
                  f"max_seq_length={self.max_tokens}")
        elif backend == "ollama":
            import httpx
            self._http = httpx
            self._url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        else:
            raise ValueError(f"unknown backend {backend}")
        self.dim = len(self.embed(["dimension probe"])[0])

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "st":
            v = self._m.encode(texts, batch_size=self.batch, normalize_embeddings=True,
                               show_progress_bar=False)
            return v.tolist()
        r = self._http.post(f"{self._url}/api/embed", json={"model": self.model, "input": texts},
                            timeout=300)
        r.raise_for_status()
        return r.json()["embeddings"]

    def count_tokens(self, texts: list[str]) -> list[int] | None:
        """Token counts (st backend only) - used to warn about silent truncation."""
        if self.backend != "st":
            return None
        tok = self._m.tokenizer
        return [len(tok(t, add_special_tokens=True)["input_ids"]) for t in texts]