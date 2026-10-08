"""Phase 1: parse the seven HP PDFs into clean per-chapter text.

Uses Firecrawl's pdf-inspector for extraction, then cleans the markdown it
returns.  Layout assumptions (from the real book-1 output):

  * pages are separated by a line containing two U+FFFD glyphs ("� �")
  * every body page starts with a running header: "#### CHAPTER <WORD>" on one
    page and "#### <CHAPTER TITLE>" on the next
  * a chapter-opening page has a spaced header: "#### C H A P T E R O N E"
  * the first letter of each chapter is a drop cap that is extracted OUT of
    order as a line starting with "# " (e.g. "# Mproud to say ...")
  * front matter (title page, copyright, contents) comes before the first
    opener; a colophon page with no header follows the last chapter
  * page numbers do not appear in the extracted text at all

Usage:
    python parse_books.py                       # all PDFs in data/raw
    python parse_books.py data/raw/book1.pdf    # one PDF
    python parse_books.py --from-json dump.json # test on a saved inspect dump
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

EXPECTED_CHAPTERS = {1: 17, 2: 18, 3: 22, 4: 37, 5: 38, 6: 30, 7: 36}
APPROX_WORDS = {1: 77_000, 2: 85_000, 3: 107_000, 4: 190_000, 5: 257_000,
                6: 169_000, 7: 198_000}
BOOK_KEYWORDS = [  # filename -> book number
    (1, ["sorcerer", "philosopher", "book1", "book_1", "book-1"]),
    (2, ["chamber", "book2", "book_2", "book-2"]),
    (3, ["prisoner", "azkaban", "book3", "book_3", "book-3"]),
    (4, ["goblet", "book4", "book_4", "book-4"]),
    (5, ["order", "phoenix", "book5", "book_5", "book-5"]),
    (6, ["half", "prince", "book6", "book_6", "book-6"]),
    (7, ["deathly", "hallows", "book7", "book_7", "book-7"]),
]

# hyphenated words that must keep their hyphen when split across lines
KEEP_HYPHEN = {"color-coding"}

EPILOGUE_BOOKS = {7}   # book 7 ends with an unnumbered "Epilogue: Nineteen Years Later"

SCENE = "<<SCENE_BREAK>>"
PAGE = "<<PB>>"
PAGE_SEP = re.compile(r"\n+� ?�\n+")
DROPCAP = re.compile(r"^# (\S+)(.*)$")  # exactly one '#'
CONTENTS_LINE = re.compile(
    r"^(?:#+\s*)?(?:[A-Z]+(?:-[A-Z]+)?\s+)?(?P<title>[A-Za-z].*?)\s+·\s+\d+\s*$")

UNITS = ["", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT",
         "NINE", "TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN",
         "SIXTEEN", "SEVENTEEN", "EIGHTEEN", "NINETEEN"]
TENS = {20: "TWENTY", 30: "THIRTY", 40: "FORTY"}


def number_word(n: int) -> str:
    if n < 20:
        return UNITS[n]
    t, u = (n // 10) * 10, n % 10
    return TENS[t] + (UNITS[u] if u else "")


WORD2NUM = {number_word(n): n for n in range(1, 41)}


def norm_letters(s: str) -> str:
    return re.sub(r"[^A-Z]", "", s.upper())


def book_number_from_name(name: str) -> int | None:
    low = re.sub(r"[^a-z0-9_\-]", "", name.lower())
    for n, keys in BOOK_KEYWORDS:
        if any(k in low for k in keys):
            return n
    return None


# ----------------------------------------------------------------------------
def extract_markdown(pdf_path: Path) -> str:
    import pdf_inspector  # pip install pdf-inspector

    res = pdf_inspector.process_pdf(str(pdf_path))
    md = res["markdown"] if isinstance(res, dict) else getattr(res, "markdown")
    return md


def contents_titles(front: str) -> list[str]:
    titles = []
    for line in front.split("\n"):
        mt = CONTENTS_LINE.match(line.strip())
        if mt:
            titles.append(mt.group("title").strip())
    return titles


def header_kind(line: str, title_keys: list[str]):
    """Classify a line as a header, ignoring spacing, dashes and '#' level.

    Returns (kind, n): ("num", chapter number), ("title", chapter index),
    ("epilogue", None), ("booktitle", None) or (None, None).
    "C H A P T E R T W E N T Y - O N E", "CHAPTER TWENTY-ONE" and
    "C H A P T E R T W E N T Y – O N E" all normalise to CHAPTERTWENTYONE.
    """
    body = line.lstrip("#").strip()
    if not body or len(body) > 80:
        return None, None
    key = norm_letters(body)
    if key.startswith("CHAPTER") and key[7:] in WORD2NUM:
        return "num", WORD2NUM[key[7:]]
    if key == "EPILOGUE":
        return "epilogue", None
    if line.startswith("#"):
        if key.startswith("HARRYPOTTER") and len(key) < 60:
            return "booktitle", None
        for idx, tk in enumerate(title_keys):
            if key == tk or (len(key) >= 5 and
                             difflib.SequenceMatcher(None, key, tk).ratio() >= 0.92):
                return "title", idx + 1
    return None, None


def contents_end(lines: list[str]):
    """Index of the last 'Title · 123' contents line near the start of the book."""
    last = None
    for i, l in enumerate(lines[: max(300, len(lines) // 8)]):
        if CONTENTS_LINE.match(l.strip()):
            last = i
    return last


def split_chapters(md: str, book: int, log: list[str]):
    md = PAGE_SEP.sub("\n" + PAGE + "\n", md)
    lines = [l.rstrip() for l in md.split("\n")]
    expected = EXPECTED_CHAPTERS[book]
    last_num = expected + (1 if book in EPILOGUE_BOOKS else 0)

    cend = contents_end(lines)
    start_at = cend + 1 if cend is not None else 0
    titles = contents_titles("\n".join(lines[:start_at]))
    if len(titles) != last_num:
        log.append(f"WARN contents titles parsed={len(titles)} expected={last_num}")
    title_keys = [norm_letters(t) for t in titles]

    # --- chapter 1 starts at the first header for it after the contents -----
    first = None
    for i in range(start_at, len(lines)):
        kind, n = header_kind(lines[i], title_keys)
        if kind in ("num", "title") and n == 1:
            first = i
            break
    if first is None:
        raise SystemExit("Could not find the start of chapter 1 - layout differs "
                         "from book 1. Send me ~100 lines around 'CHAPTER ONE'.")
    body = lines[first:]

    # --- drop the colophon / back matter after the final chapter page -------
    last_hdr = None
    for i, l in enumerate(body):
        if header_kind(l, title_keys)[0] is not None:
            last_hdr = i
    end = len(body)
    if last_hdr is not None:
        nxt = next((j for j in range(last_hdr, len(body)) if body[j] == PAGE), None)
        if nxt is not None:
            dropped = [l for l in body[nxt:] if l.strip() and l != PAGE]
            if dropped:
                log.append("dropped trailing back matter: " + " ".join(dropped)[:140])
            end = nxt
    body = body[:end]

    # --- walk lines: a new chapter starts at the first header for the NEXT
    #     chapter number; repeats of the current number are running headers ---
    chapters: list[dict] = []
    cur, next_n = None, 1
    for l in body:
        if l == PAGE:
            if cur is not None:
                cur["lines"].append(PAGE)
            continue
        if not l.strip():
            continue
        kind, n = header_kind(l, title_keys)
        if kind is not None:
            start = None
            if kind in ("num", "title") and n == next_n and next_n <= last_num:
                start = n
            elif kind == "num" and n is not None and n > next_n and n <= last_num:
                log.append(f"WARN chapter(s) {next_n}..{n - 1} had no header; "
                           f"starting chapter {n} at {l!r}")
                start = n
            elif kind == "epilogue" and book in EPILOGUE_BOOKS and next_n == last_num:
                start = last_num
            if start is not None:
                cur = {"num": start, "lines": [], "opener": l}
                chapters.append(cur)
                next_n = start + 1
            continue                      # every header line is dropped from the text
        if cur is not None:
            cur["lines"].append(l)

    got = [c["num"] for c in chapters]
    missing = [n for n in range(1, last_num + 1) if n not in got]
    for n in missing[:5]:                 # show what the header looked like
        want = "CHAPTER" + number_word(n) if n <= expected else "EPILOGUE"
        hits = [(i, l) for i, l in enumerate(lines)
                if want in norm_letters(l) and len(l) < 90][:3]
        log.append(f"DEBUG missing chapter {n}: lines containing {want!r}: {hits}")
    return chapters, titles


def fix_dropcap(lines: list[str], log: list[str], tag: str) -> list[str]:
    """Re-attach the drop-cap letter that pdf-inspector extracted out of order."""
    for i, l in enumerate(lines[:12]):
        mt = DROPCAP.match(l)
        if not mt:
            continue
        tok, rest = mt.group(1), mt.group(2)
        if len(tok) == 1:                      # "# T longest-ever ..."
            letter, tail = tok, rest.lstrip()
        else:                                  # "# Mproud to say ..."
            letter, tail = tok[0], tok[1:] + rest
        out = lines[:i] + [tail] + lines[i + 1:]
        first_text = next(j for j, x in enumerate(out) if x != PAGE)
        out[first_text] = letter + out[first_text]
        return out
    log.append(f"WARN {tag}: no drop-cap line found")
    return lines


def join_text(lines: list[str]) -> str:
    # join lines/pages with a single space, but keep line-break hyphens joinable
    parts, buf = [], []
    for l in lines:
        buf.append(" " if l == PAGE else l)
    text = "\n".join(buf)
    return re.sub(r"\s*\n\s*", " ", text)


def clean_text(text: str) -> str:
    text = text.replace("�", "").replace(" ", " ")
    text = re.sub(r"[​­﻿]", "", text)
    text = re.sub(r"(?:(?<=\s)|^)\\?\*(?:\s\\?\*){2}(?=\s|$)", f" {SCENE} ", text)
    text = re.sub(r"(?<!\w)#{1,6} ", "", text)       # stray heading marks
    text = text.replace("\\*", "").replace("*", "").replace("\\", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fix_hyphens(texts: list[str], log: list[str]) -> list[str]:
    """Rejoin 'gar- dens' style line-break splits using the book's own vocabulary."""
    pat = re.compile(r"\b([A-Za-z]+)- ([a-z]+)\b")
    # vocabulary from text WITHOUT the split fragments (else "commen"/"tary" count)
    big = pat.sub(" ", " ".join(texts))
    vocab = set(re.findall(r"\b[a-z]{2,}\b", big.lower()))
    hyph_vocab = set(re.findall(r"\b[a-z]+(?:-[a-z]+)+\b", big.lower()))
    uncertain: list[str] = []

    def repl(m: re.Match) -> str:
        a, b = m.group(1), m.group(2)
        joined, hyph = (a + b).lower(), f"{a}-{b}".lower()
        if m.start() > 0 and m.string[m.start() - 1] == "-":
            return f"{a}-{b}"              # inside a chain: You-Know- oh
        if joined in vocab:
            return a + b
        if hyph in hyph_vocab or hyph in KEEP_HYPHEN:
            return f"{a}-{b}"
        if len(a) >= 4 and len(b) >= 4 and a.lower() in vocab and b.lower() in vocab:
            return f"{a}-{b}"              # two real words -> probably a compound
        uncertain.append(f"{a}- {b}")
        return a + b                       # default: line-break split

    out = [pat.sub(repl, t) for t in texts]
    if uncertain:
        log.append(f"hyphen joins not confirmed by vocabulary ({len(uncertain)}): "
                   + ", ".join(sorted(set(uncertain))[:25]))
    return out


# ----------------------------------------------------------------------------
def parse_book(md: str, book: int):
    log: list[str] = []
    chapters, titles = split_chapters(md, book, log)

    raw_texts = []
    for ch in chapters:
        lines = fix_dropcap(ch["lines"], log, f"b{book} ch{ch['num']}")
        raw_texts.append(clean_text(join_text(lines)))
    texts = fix_hyphens(raw_texts, log)

    rows = []
    for ch, text in zip(chapters, texts):
        n = ch["num"]
        title = titles[n - 1] if len(titles) >= n else ""
        if n > EXPECTED_CHAPTERS[book]:                    # epilogue
            title = "Epilogue: " + (title or "Nineteen Years Later")
        rows.append({
            "book": book, "chapter_number": n, "chapter_title": title,
            "text": text, "n_words": len(re.findall(r"\b[\w’'-]+\b", text)),
            "n_scene_breaks": text.count(SCENE),
        })
    return rows, log


def validate(rows, book: int, log: list[str]) -> bool:
    ok = True
    nums = [r["chapter_number"] for r in rows]
    exp = EXPECTED_CHAPTERS[book] + (1 if book in EPILOGUE_BOOKS else 0)
    if nums != list(range(1, exp + 1)):
        ok = False
        log.append(f"FAIL chapter numbers {nums} != 1..{exp}")
    words = sum(r["n_words"] for r in rows)
    target = APPROX_WORDS[book]
    if abs(words - target) / target > 0.12:
        ok = False
        log.append(f"FAIL word count {words:,} is >12% from ~{target:,}")
    for r in rows:
        if r["n_words"] < 800:
            ok = False
            log.append(f"FAIL ch{r['chapter_number']} only {r['n_words']} words")
        if re.search(r"[a-z]- [a-z]", r["text"]):
            log.append(f"note ch{r['chapter_number']}: leftover 'x- y' hyphen split")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="*")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--from-json", help="inspect JSON dump (testing)")
    ap.add_argument("--book", type=int, help="book number for --from-json")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "markdown").mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[int, str, str]] = []  # (book, label, markdown)
    if args.from_json:
        md = json.load(open(args.from_json))["markdown"]
        jobs.append((args.book or 1, args.from_json, md))
    else:
        paths = [Path(p) for p in args.pdfs] or sorted(Path(args.raw_dir).glob("*.pdf"))
        if not paths:
            sys.exit(f"No PDFs found in {args.raw_dir}")
        for p in paths:
            b = book_number_from_name(p.name)
            if b is None:
                print(f"SKIP {p.name}: cannot tell which book this is "
                      "(rename it, e.g. book3.pdf)")
                continue
            cache = out / "markdown" / f"book{b}.md"
            if cache.exists():
                md = cache.read_text()
            else:
                md = extract_markdown(p)
                cache.write_text(md)
            jobs.append((b, p.name, md))

    all_rows, report, all_ok = [], [], True
    for b, label, md in sorted(jobs):
        rows, log = parse_book(md, b)
        ok = validate(rows, b, log)
        all_ok &= ok
        words = sum(r["n_words"] for r in rows)
        report.append(f"\n=== Book {b}  ({label}) ===  "
                      f"{'OK' if ok else 'CHECK'}  chapters={len(rows)}/"
                      f"{EXPECTED_CHAPTERS[b]}  words={words:,}  "
                      f"scene_breaks={sum(r['n_scene_breaks'] for r in rows)}")
        for r in rows:
            report.append(f"  ch{r['chapter_number']:>2} {r['chapter_title'][:34]:<34} "
                          f"{r['n_words']:>6}w | {r['text'][:70]!r} ... {r['text'][-40:]!r}")
        report += [f"  ! {x}" for x in log]
        all_rows += rows

    (out / "parse_report.txt").write_text("\n".join(report))
    print("\n".join(report))

    try:
        import pandas as pd
        pd.DataFrame(all_rows).to_parquet(out / "chapters.parquet", index=False)
        print(f"\nWrote {out/'chapters.parquet'}")
    except Exception as e:  # noqa: BLE001
        print(f"\n(parquet skipped: {e})")
    with open(out / "chapters.jsonl", "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {out/'chapters.jsonl'}  |  report: {out/'parse_report.txt'}")
    if not all_ok:
        print("\nSome checks failed - see lines marked FAIL above.")


if __name__ == "__main__":
    main()