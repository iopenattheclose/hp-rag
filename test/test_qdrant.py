"""Qdrant smoke test: basic CRUD + filtering, then collection-level JWT RBAC.

Run:  export QDRANT_API_KEY=your-admin-key && python test_qdrant.py

Note: current Qdrant (1.16+) JWT RBAC scopes access per COLLECTION and access
level (r / rw / prw). Row-level payload filters inside a token were removed,
so each role gets its own collection containing only what that role may see.
"""
import os
import time

import jwt  # pip install pyjwt
from qdrant_client import QdrantClient, models

URL = "http://localhost:6333"
ADMIN_KEY = os.environ["QDRANT_API_KEY"]
ALL = "smoke_all"          # everything (Restricted Section pass)
BOOK1 = "smoke_book1"      # only book 1 (First-year student)

admin = QdrantClient(url=URL, api_key=ADMIN_KEY)

DOCS = [
    (1, [0.9, 0.1, 0.0, 0.0], {"book": 1, "clearance": 0, "text": "Harry meets Hagrid."}),
    (2, [0.8, 0.2, 0.1, 0.0], {"book": 1, "clearance": 1, "text": "The Sorting Hat sorts Harry."}),
    (3, [0.1, 0.1, 0.9, 0.2], {"book": 7, "clearance": 2, "text": "Horcrux discussion (restricted)."}),
]


def make(name, docs):
    if admin.collection_exists(name):
        admin.delete_collection(name)
    admin.create_collection(
        name, vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE)
    )
    admin.create_payload_index(name, "book", models.PayloadSchemaType.INTEGER)
    admin.upsert(name, points=[
        models.PointStruct(id=i, vector=v, payload=p) for i, v, p in docs
    ])


# ---- 1. Basic test: build collections, search, filter ----------------------
make(ALL, DOCS)
make(BOOK1, [d for d in DOCS if d[2]["book"] == 1])
print("Points:", {c: admin.count(c).count for c in (ALL, BOOK1)})

q = [0.1, 0.1, 0.9, 0.2]  # looks like the book 7 chunk
hits = admin.query_points(ALL, query=q, limit=3).points
print("\nAdmin search on full collection (book 7 chunk should rank first):")
for h in hits:
    print(f"  id={h.id} score={h.score:.3f} book={h.payload['book']} {h.payload['text']}")
assert hits[0].payload["book"] == 7

only_book1 = models.Filter(must=[
    models.FieldCondition(key="book", match=models.MatchValue(value=1))
])
hits = admin.query_points(ALL, query=q, query_filter=only_book1, limit=3).points
print("\nSame query with a book==1 filter (book 7 must NOT appear):")
for h in hits:
    print(f"  id={h.id} book={h.payload['book']} {h.payload['text']}")
assert all(h.payload["book"] == 1 for h in hits)

# ---- 2. RBAC test: read-only token for the book-1 collection only ----------
claims = {
    "exp": int(time.time()) + 3600,
    "access": [{"collection": BOOK1, "access": "r"}],
}
token = jwt.encode(claims, ADMIN_KEY, algorithm="HS256")
first_year = QdrantClient(url=URL, api_key=token)

# 2a. Can read its own collection, and only sees book 1
hits = first_year.query_points(BOOK1, query=q, limit=5).points
books_seen = sorted({h.payload["book"] for h in hits})
print("\nFirst-year token, search on its collection, books seen:", books_seen)
assert books_seen == [1], "RBAC LEAK: saw other books!"

# 2b. Cannot read the full collection
try:
    first_year.query_points(ALL, query=q, limit=5)
    raise SystemExit("RBAC FAIL: token could read the full collection!")
except SystemExit:
    raise
except Exception as e:
    print("Read of full collection correctly rejected:", type(e).__name__)

# 2c. Cannot write
try:
    first_year.upsert(BOOK1, points=[
        models.PointStruct(id=99, vector=[0, 0, 0, 1], payload={"book": 1})
    ])
    raise SystemExit("RBAC FAIL: read-only token was able to write!")
except SystemExit:
    raise
except Exception as e:
    print("Write correctly rejected:", type(e).__name__)

# ---- cleanup ---------------------------------------------------------------
for c in (ALL, BOOK1):
    admin.delete_collection(c)
print("\nAll checks passed. Qdrant + collection-level RBAC are working.")