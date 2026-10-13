"""Phase 1, step 6: build the metadata tables from the parsed chapters.

Reads  data/processed/chapters.parquet (or chapters.jsonl)
Writes data/processed/sentences.{parquet,jsonl}   one row per sentence (with char offsets)
       data/processed/segments.{parquet,jsonl}    ~150-word tagged units: THE metadata table
       data/processed/review/*.csv                files to eyeball the tagging

Why segments?  Tagging every sentence with an LLM is slow, and each of the 7
chunking strategies would otherwise need its own tagging.  Instead we tag these
small stable segments ONCE (characters, topics, clearance).  Any chunk produced
later by any chunker looks up the segments it overlaps (by char offsets) and
inherits max(clearance), union(characters), union(topics).

Usage:  python build_metadata.py [--in data/processed/chapters.parquet]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

if __package__:
    from . import hp_lexicon as lx
else:
    import hp_lexicon as lx

SCENE = "<<SCENE_BREAK>>"
ABBR = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|St|Prof|Mt|Jr|Sr|Capt|Col|Gen|vs)\.$")
# sentence boundary: end punctuation, optional closing quotes, space, then a capital/quote
BOUNDARY = re.compile(r"(?<=[.!?…])([\"”’)]*)\s+(?=[\"“‘(]?[A-Z0-9])")

TARGET_WORDS = 150     # a segment is closed once it reaches this size ...
HARD_MAX_WORDS = 260   # ... or would exceed this
MIN_TAIL_WORDS = 40    # a shorter leftover is merged into the previous segment


# ----------------------------------------------------------------------------
def split_sentences(text: str, base: int = 0):
    """Return [(start, end)] char spans (absolute = base + local)."""
    spans, start = [], 0
    for m in BOUNDARY.finditer(text):
        if not m.group(1) and ABBR.search(text[max(0, m.start() - 8):m.start()]):
            continue                                   # "Mr. Dursley" is not a boundary
        spans.append((start, m.end(1)))
        start = m.end()
    spans.append((start, len(text)))
    return [(base + a, base + b) for a, b in spans if text[a:b].strip()]


def scene_parts(text: str):
    """Split a chapter at scene-break markers -> [(scene_idx, start, end)]."""
    parts, pos, idx = [], 0, 0
    for m in re.finditer(re.escape(SCENE), text):
        parts.append((idx, pos, m.start()))
        pos, idx = m.end(), idx + 1
    parts.append((idx, pos, len(text)))
    return [p for p in parts if text[p[1]:p[2]].strip()]


def group_sentences(sents, text):
    """Greedy-group consecutive sentences into ~TARGET_WORDS segments."""
    groups, cur, words = [], [], 0
    for s in sents:
        w = len(text[s[0]:s[1]].split())
        if cur and (words >= TARGET_WORDS or words + w > HARD_MAX_WORDS):
            groups.append(cur)
            cur, words = [], 0
        cur.append(s)
        words += w
    if cur:
        if groups and words < MIN_TAIL_WORDS:
            groups[-1].extend(cur)
        else:
            groups.append(cur)
    return groups


# ----------------------------------------------------------------------------
def tag(text: str):
    """Characters, topics and clearance for one segment of text."""
    chars = Counter()
    for name, rx in lx.CHAR_RES.items():
        n = len(rx.findall(text))
        if n:
            chars[name] = n
    topics, terms, clearance = [], [], 0
    for topic, rx in lx.TOPIC_RES.items():
        scan = lx.TOPIC_EXCLUDE[topic].sub(" ", text) if topic in lx.TOPIC_EXCLUDE else text
        hits = [m.group(0).lower() for m in rx.finditer(scan)]
        if not hits:
            continue
        topics.append(topic)
        level = lx.TOPIC_CLEARANCE.get(topic, 0)
        if level > 0:
            terms += sorted(set(hits))
        clearance = max(clearance, level)
    return [c for c, _ in chars.most_common()], topics, clearance, sorted(set(terms))


def fill_gaps(segs):
    """Raise short runs of segments that sit between two segments of level L (same scene)."""
    n_filled = 0
    cfg = lx.GAP_FILL
    for level in cfg["levels"]:
        anchors = [i for i, s in enumerate(segs) if s["clearance"] >= level]   # keyword-based only
        for a, z in zip(anchors, anchors[1:]):
            gap = z - a - 1
            if not 1 <= gap <= cfg["max_gap"]:
                continue
            if len({segs[k]["scene_index"] for k in range(a, z + 1)}) != 1:
                continue                                  # never fill across a scene break
            for k in range(a + 1, z):
                if segs[k]["clearance"] < level:
                    segs[k]["clearance"] = level
                    segs[k]["clearance_source"] = "gap_fill"
                    n_filled += 1
    return n_filled


def build(chapters):
    sent_rows, seg_rows = [], []
    for ch in chapters:
        b, n, text = ch["book"], ch["chapter_number"], ch["text"]
        ch_start = len(seg_rows)
        seg_i = 0
        sent_i = 0
        for scene_idx, a, z in scene_parts(text):
            sents = split_sentences(text[a:z], a)
            for grp in group_sentences(sents, text):
                start, end = grp[0][0], grp[-1][1]
                seg_text = text[start:end]
                seg_id = f"b{b}_c{n:02d}_s{seg_i:03d}"
                chars, topics, clr, terms = tag(seg_text)
                seg_rows.append({
                    "seg_id": seg_id, "book": b, "chapter_number": n,
                    "chapter_title": ch["chapter_title"], "scene_index": scene_idx,
                    "seg_index": seg_i, "char_start": start, "char_end": end,
                    "n_words": len(seg_text.split()), "text": seg_text,
                    "characters": chars, "topics": topics,
                    "clearance": clr, "clearance_terms": terms,
                    "clearance_source": "keyword" if clr else "none",
                })
                for s0, s1 in grp:
                    sent_rows.append({
                        "seg_id": seg_id, "book": b, "chapter_number": n,
                        "sent_index": sent_i, "char_start": s0, "char_end": s1,
                        "text": text[s0:s1],
                    })
                    sent_i += 1
                seg_i += 1
        fill_gaps(seg_rows[ch_start:])          # same dict objects -> updates in place
    return sent_rows, seg_rows


# ----------------------------------------------------------------------------
def load_chapters(path: Path):
    if path.suffix == ".parquet" and path.exists():
        import pandas as pd
        return pd.read_parquet(path).to_dict("records")
    jl = path.with_suffix(".jsonl")
    return [json.loads(l) for l in open(jl, encoding="utf-8")]


def write_table(rows, out_base: Path):
    with open(out_base.with_suffix(".jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    try:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(out_base.with_suffix(".parquet"), index=False)
    except Exception as e:  # noqa: BLE001
        print(f"  (parquet skipped for {out_base.name}: {str(e)[:60]})")


def report(segs, out: Path):
    lines = ["", "=== Segments per book and clearance ==="]
    lines.append(f"{'book':>4} {'segments':>8} {'words':>9} {'clr0':>7} {'clr1':>7} {'clr2':>7}")
    for b in sorted({s["book"] for s in segs}):
        bs = [s for s in segs if s["book"] == b]
        c = Counter(s["clearance"] for s in bs)
        lines.append(f"{b:>4} {len(bs):>8} {sum(s['n_words'] for s in bs):>9,} "
                     f"{c[0]:>7} {c[1]:>7} {c[2]:>7}")
    lines += ["", "=== What each role would see (segments) ==="]
    for role, cfg in lx.ROLES.items():
        n = sum(1 for s in segs if s["book"] in cfg["books"]
                and s["clearance"] <= cfg["max_clearance"])
        lines.append(f"  {role:<20} {n:>6} of {len(segs)}  ({100*n/len(segs):.0f}%)")
    chars = Counter(c for s in segs for c in s["characters"])
    topics = Counter(t for s in segs for t in s["topics"])
    lines += ["", "Top characters: " + ", ".join(f"{k} {v}" for k, v in chars.most_common(12)),
              "Topics: " + ", ".join(f"{k} {v}" for k, v in topics.most_common())]
    print("\n".join(lines))
    (out / "review" / "tagging_report.txt").write_text("\n".join(lines), encoding="utf-8")


def write_review_csv(segs, out: Path):
    import csv
    random.seed(7)
    rv = out / "review"
    rv.mkdir(exist_ok=True)
    for level in (2, 1, 0):
        pool = [s for s in segs if s["clearance"] == level]
        sample = random.sample(pool, min(60, len(pool)))
        with open(rv / f"sample_clearance{level}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["seg_id", "clearance", "terms_that_triggered", "your_verdict",
                        "text_first_300_chars"])
            for s in sample:
                w.writerow([s["seg_id"], s["clearance"], "|".join(s["clearance_terms"]),
                            "", s["text"][:300]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/chapters.parquet")
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()
    out = Path(args.out_dir)
    (out / "review").mkdir(parents=True, exist_ok=True)

    chapters = load_chapters(Path(args.inp))
    sents, segs = build(chapters)
    write_table(sents, out / "sentences")
    write_table(segs, out / "segments")
    write_review_csv(segs, out)
    print(f"chapters={len(chapters)}  sentences={len(sents):,}  segments={len(segs):,}")
    report(segs, out)
    print(f"\nWrote {out/'segments.parquet'}, {out/'sentences.parquet'} and {out/'review'}/")


if __name__ == "__main__":
    main()