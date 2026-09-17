"""Fetch public-domain human text for training (opt-in internet use).

Project Gutenberg books are public domain, so they are a clean-license
source of human prose. Intended workflow — temp file, deleted once the
model does well::

    python main.py --mode train --fetch-human 60

which downloads ~60 paragraph samples into a temp dir, trains the critic
on them, and deletes the temp dir when accuracy clears --fetch-min-acc
(keeps it otherwise, printing the path so you can inspect or retry).

Standalone use (keep the files)::

    python fetch_corpus.py --out data/temp_human --samples 60

Note: classics are 19th-century prose — great for learning generic AI-vs-
human patterns, but your own modern writing (--human-dir) teaches the
critic *your* voice. Combine both when you can.
"""
from __future__ import annotations

import argparse
import logging
import re
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"

# Varied public-domain works (Austen, Dickens, Twain, Shelley, ...).
GUTENBERG_DEFAULT_IDS = [1342, 11, 74, 76, 84, 98, 43, 1400, 1661, 2701]

MIN_CHUNK = 60
MAX_CHUNK = 1500


def download_book_text(book_id: int, timeout: int = 30) -> str:
    """Download one Gutenberg ebook as text (raises on network failure)."""
    url = GUTENBERG_URL.format(book_id=book_id)
    req = urllib.request.Request(url, headers={"User-Agent": "humaize-local-trainer/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    logger.info("fetch_corpus: downloaded book %d (%d chars)", book_id, len(raw))
    return raw


def strip_gutenberg_boilerplate(raw: str) -> str:
    """Cut the license header/footer; fall back to trimming 3% each end."""
    lines = raw.splitlines()
    start, end = 0, len(lines)
    for i, line in enumerate(lines):
        if "START OF" in line and "PROJECT GUTENBERG" in line:
            start = i + 1
            break
    for i in range(len(lines) - 1, -1, -1):
        if "END OF" in lines[i] and "PROJECT GUTENBERG" in lines[i]:
            end = i
            break
    else:
        end = len(lines)
    if start >= end:  # markers missing — assume mostly-body with thin edges
        cut = max(1, len(lines) // 33)
        start, end = cut, len(lines) - cut
    return "\n".join(lines[start:end]).strip()


def _is_heading(chunk: str) -> bool:
    alpha = [c for c in chunk if c.isalpha()]
    if not alpha:
        return True
    upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
    if len(chunk) < 120 and upper_ratio > 0.5:
        return True
    # Table-of-contents blocks slip past the case check (mixed-case titles).
    return chunk.upper().count("CHAPTER") >= 3


def prepare_book_samples(raw: str, max_samples: int = 0) -> list[str]:
    """Boilerplate-strip + paragraph-chunk one book into training samples."""
    body = strip_gutenberg_boilerplate(raw)
    chunks = [c.strip() for c in re.split(r"\n\s*\n", body)]
    samples = [
        c for c in chunks
        if MIN_CHUNK <= len(c) <= MAX_CHUNK and not _is_heading(c)
    ]
    if max_samples > 0:
        # Even spread across the book (beginning/middle/end styles differ).
        step = max(1, len(samples) // max_samples)
        samples = samples[::step][:max_samples]
    return samples


def fetch_temp_human_samples(
    dest_dir: str | Path,
    n_samples: int = 60,
    book_ids: list[int] | None = None,
    per_book_cap: int = 25,
) -> tuple[int, int]:
    """Download books until ~n_samples paragraphs land in dest_dir.

    Writes one `<book_id>.txt` per book (paragraphs blank-line separated,
    ready for --split-paragraphs). Skips failed downloads with a warning.
    Returns (books_saved, samples_saved). Raises RuntimeError if nothing
    could be fetched (e.g. no connection).
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    ids = list(book_ids) if book_ids else list(GUTENBERG_DEFAULT_IDS)
    books_saved, samples_saved = 0, 0
    i = 0
    while samples_saved < n_samples and i < len(ids):
        book_id = ids[i]
        i += 1
        try:
            raw = download_book_text(book_id)
        except Exception as exc:
            logger.warning("fetch_corpus: book %d failed (%s); skipping", book_id, exc)
            continue
        need = min(per_book_cap, n_samples - samples_saved)
        samples = prepare_book_samples(raw, max_samples=need)
        if not samples:
            continue
        (dest / f"book_{book_id}.txt").write_text("\n\n".join(samples), encoding="utf-8")
        books_saved += 1
        samples_saved += len(samples)
    if samples_saved == 0:
        raise RuntimeError(
            "fetch_corpus: downloaded nothing — check your connection or "
            "pass --book-ids with different Gutenberg ebook numbers."
        )
    logger.info("fetch_corpus: saved %d samples from %d books to %s",
                samples_saved, books_saved, dest)
    return books_saved, samples_saved


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch public-domain human training text")
    ap.add_argument("--out", required=True, help="Directory to write book_*.txt files")
    ap.add_argument("--samples", type=int, default=60, help="Target paragraph count")
    ap.add_argument("--book-ids", default=None, help="Comma-separated Gutenberg ebook IDs")
    args = ap.parse_args(argv)
    ids = [int(x) for x in args.book_ids.split(",")] if args.book_ids else None
    books, samples = fetch_temp_human_samples(args.out, args.samples, ids)
    print(f"saved {samples} samples from {books} books to {args.out}")
    print(f"train with: python main.py --mode train --human-dir {args.out} --split-paragraphs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
