"""Chunking strategies 1-3: fixed-size, recursive, sentence-window.

Every chunker works on ONE chapter's text (with <<SCENE_BREAK>> markers) and
returns spans [(char_start, char_end, text, context_text)] where the char offsets
point into the ORIGINAL chapter text.  Offsets are what lets a chunk inherit
clearance / characters / topics from the tagged segments (see inherit()).

SECURITY NOTE: the span used for inheritance must cover everything the LLM can
read.  For sentence-window that is the WINDOW, not the single embedded sentence.
"""
from __future__ import annotations

import re
from bisect import bisect_left
from collections import Counter

from build_metadata import SCENE, split_sentences

WORD = re.compile(r"\S+")


def clean(text: str) -> str:
    """Display/embedding text: scene markers become a blank line."""
    return re.sub(r"\s*" + re.escape(SCENE) + r"\s*", "\n\n", text).strip()


# ----------------------------------------------------------------------------
# 1. Fixed-size: N words, M words overlap, blind to sentences and scenes
# ----------------------------------------------------------------------------
def fixed_size(text: str, size: int = 200, overlap: int = 40):
    words = [m for m in WORD.finditer(text) if m.group(0) != SCENE]
    step = max(1, size - overlap)
    out, i = [], 0
    while i < len(words):
        w = words[i:i + size]
        a, z = w[0].start(), w[-1].end()
        out.append((a, z, clean(text[a:z]), None))
        if i + size >= len(words):
            break
        i += step
    return out


# ----------------------------------------------------------------------------
# 2. Recursive: split on the biggest natural boundary that fits, then merge
# ----------------------------------------------------------------------------
_SEPS = [
    re.compile(re.escape(SCENE)),                       # scene break
    re.compile(r"\n\s*\n"),                             # paragraph
    re.compile(r"(?<=[.!?…])[\"”’)]*\s+(?=[\"“‘(]?[A-Z0-9])"),   # sentence
    re.compile(r"\s+"),                                 # word
]


def _pieces(text, a, z, level):
    """Split text[a:z] with separator `level` -> [(start, end)] (separators dropped)."""
    out, pos = [], a
    for m in _SEPS[level].finditer(text, a, z):
        if m.start() > pos:
            out.append((pos, m.start()))
        pos = m.end()
    if pos < z:
        out.append((pos, z))
    return [(s, e) for s, e in out if text[s:e].strip()]


def _nw(text, a, z):
    return len(text[a:z].split())


def recursive(text: str, max_words: int = 220, overlap: int = 30):
    spans: list[tuple[int, int]] = []

    def split(a, z, level):
        if _nw(text, a, z) <= max_words or level >= len(_SEPS):
            spans.append((a, z))
            return
        parts = _pieces(text, a, z, level)
        if len(parts) <= 1:                 # separator did nothing -> go finer
            return split(a, z, level + 1)
        # merge consecutive parts up to max_words; oversized parts recurse
        cur = None
        for s, e in parts:
            if _nw(text, s, e) > max_words:
                if cur:
                    spans.append(cur)
                    cur = None
                split(s, e, level + 1)
            elif cur is None:
                cur = (s, e)
            elif _nw(text, cur[0], e) <= max_words:
                cur = (cur[0], e)
            else:
                spans.append(cur)
                cur = (s, e)
        if cur:
            spans.append(cur)

    split(0, len(text), 0)

    out = []
    for i, (a, z) in enumerate(spans):
        a2 = a
        if i > 0 and overlap and text[a:z].strip():
            # prepend the last `overlap` words of the previous chunk, but never cross a scene break
            pa, pz = spans[i - 1]
            gap = text[pz:a]
            if SCENE not in gap:
                ws = [m for m in WORD.finditer(text, pa, pz)]
                tail = ws[-overlap:]
                if tail:
                    a2 = min(a, tail[0].start())
        body = clean(text[a2:z])
        if body:
            out.append((a2, z, body, None))
    return out


# ----------------------------------------------------------------------------
# 3. Sentence-window: embed ONE sentence, hand the LLM the sentence +/- k
# ----------------------------------------------------------------------------
def sentence_window(text: str, window: int = 2, min_words: int = 4):
    sents = []
    pos = 0
    for scene_part in re.split(re.escape(SCENE), text):
        base = text.index(scene_part, pos) if scene_part else pos
        for s, e in split_sentences(scene_part, base):
            sents.append((s, e, base))
        pos = base + len(scene_part)
    # scene id per sentence, so a window never crosses a scene break
    scene_of, sid, last = [], 0, 0
    for s, e, _ in sents:
        if SCENE in text[last:s]:
            sid += 1
        scene_of.append(sid)
        last = e

    # fold very short sentences ("Harry nodded.") into the next one in the same scene
    merged = []
    for i, (s, e, _) in enumerate(sents):
        if merged and merged[-1][3] == scene_of[i] and len(text[merged[-1][0]:merged[-1][1]].split()) < min_words:
            merged[-1] = (merged[-1][0], e, None, scene_of[i])
        else:
            merged.append((s, e, None, scene_of[i]))

    out = []
    for i, (s, e, _, sc) in enumerate(merged):
        lo, hi = i, i
        while lo > 0 and i - lo < window and merged[lo - 1][3] == sc:
            lo -= 1
        while hi < len(merged) - 1 and hi - i < window and merged[hi + 1][3] == sc:
            hi += 1
        wa, wz = merged[lo][0], merged[hi][1]
        out.append((wa, wz, clean(text[s:e]), clean(text[wa:wz])))
    return out


STRATEGIES = {
    "fixed": lambda t: fixed_size(t, size=200, overlap=40),
    "recursive": lambda t: recursive(t, max_words=220, overlap=30),
    "sentence_window": lambda t: sentence_window(t, window=2),
}


# ----------------------------------------------------------------------------
# Inheritance: chunk -> overlapping segments -> clearance / characters / topics
# ----------------------------------------------------------------------------
class SegmentIndex:
    """Segments of one chapter, sorted by char_start."""

    def __init__(self, segs):
        self.segs = sorted(segs, key=lambda s: s["char_start"])
        self.ends = [s["char_end"] for s in self.segs]

    def overlapping(self, a, z):
        i = bisect_left(self.ends, a + 1)          # first seg with char_end > a
        out = []
        while i < len(self.segs) and self.segs[i]["char_start"] < z:
            out.append(self.segs[i])
            i += 1
        return out


def inherit(index: SegmentIndex, a: int, z: int):
    segs = index.overlapping(a, z)
    chars = Counter()
    topics, terms, seg_ids = set(), set(), []
    clr = 0
    for s in segs:
        clr = max(clr, int(s["clearance"]))
        chars.update(list(s["characters"]))
        topics.update(list(s["topics"]))
        terms.update(list(s["clearance_terms"]))
        seg_ids.append(s["seg_id"])
    return {
        "clearance": clr,
        "characters": [c for c, _ in chars.most_common()],
        "topics": sorted(topics),
        "clearance_terms": sorted(terms),
        "seg_ids": seg_ids,
    }