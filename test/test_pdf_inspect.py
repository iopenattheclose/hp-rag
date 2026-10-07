"""Step 1: see what pdf-inspector returns for one HP PDF.

Run:  python inspect_pdf.py data/raw/book1.pdf
Then paste the printed output back so the real parser can be written
against the actual result structure (not guessed).
"""
import json
import sys
from pathlib import Path

import pdf_inspector  # pip install pdf-inspector

path = Path(sys.argv[1])
res = pdf_inspector.process_pdf(str(path))


def short(v, n=300):
    s = repr(v)
    return s if len(s) <= n else s[:n] + f"... [{len(s)} chars]"


print("TYPE:", type(res))
if isinstance(res, dict):
    items = list(res.items())
else:
    names = [a for a in dir(res) if not a.startswith("_")]
    print("ATTRIBUTES:", names)
    items = []
    for a in names:
        try:
            v = getattr(res, a)
        except Exception as e:  # noqa: BLE001
            v = f"<error: {e}>"
        if not callable(v):
            items.append((a, v))

print("\n--- FIELDS ---")
for k, v in items:
    kind = type(v).__name__
    extra = f" len={len(v)}" if hasattr(v, "__len__") else ""
    print(f"{k}: <{kind}{extra}> {short(v)}")

# Save a full dump so we can look at specific pages later
out = Path("data/processed") / f"inspect_{path.stem}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(dict(items), default=str, indent=2)[:5_000_000])
print(f"\nFull dump saved to {out}")