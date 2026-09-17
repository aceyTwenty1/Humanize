"""Dataset loading + built-in demo corpus.

Expected on-disk format (CSV or JSONL) with columns/keys:
    text,label    where label: 1 = human, 0 = AI

Plain-text import: point --human-dir / --ai-dir at your own writing — one
`.txt`/`.md` file or a folder of them — and each file trains as one sample
with the matching label (no CSV wrangling needed).

No network access. If no dataset path is given, a small built-in paired
demo corpus is returned so `main.py` runs end-to-end offline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Label = Literal[0, 1]

PLAIN_EXTENSIONS = {".txt", ".md", ".markdown"}


@dataclass
class TextSample:
    text: str
    label: int  # 1 = human, 0 = AI

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError(f"label must be 0 (AI) or 1 (human), got {self.label}")
        self.text = self.text.strip()
        if not self.text:
            raise ValueError("Empty text sample")


DEMO_SAMPLES: list[TextSample] = [
    # --- human (label 1): irregular rhythm, asides, fragments, hedging ---
    TextSample("I dunno — yesterday's meeting ran way over, and honestly half of it could've been an email. We just sat there nodding.", 1),
    TextSample("My grandma's soup? You can't really write down the recipe. A pinch of this, taste, adjust. It never comes out the same twice, and that's the point.", 1),
    TextSample("Ugh. Train delayed again. Third time this week. Anyway — I finally finished the report on the ride, so, silver linings, I guess.", 1),
    TextSample("We hiked up before sunrise, freezing, couldn't feel our fingers. But then the light hit the valley and nobody said anything for a while. Worth it.", 1),
    TextSample("Look, I'm no expert on taxes, but last year I messed up the filing and it took three calls and a lot of coffee to sort out. Learn from my mistakes.", 1),
    TextSample("The kid asked why the sky is blue and I gave some half-remembered physics answer. He stared at me, then asked if birds know. Fair question, honestly.", 1),
    TextSample("Been trying to fix this leaky faucet since Saturday. Watched four videos, bought the wrong washer twice. It still drips. It mocks me at night.", 1),
    TextSample("Coffee first. Then emails. Then, if I'm lucky, actual work before lunch derails everything. That's just how Tuesdays go.", 1),
    TextSample("She laughed mid-sentence, apologized, started over — that kind of nervous laugh when you're telling a story that mattered to you. I liked her immediately.", 1),
    TextSample("Not gonna lie, the first draft was terrible. Like, really bad. But you keep chipping at it, and somewhere around version six it starts breathing.", 1),
    # --- AI-like (label 0): uniform, formal, enumerative, low burstiness ---
    TextSample("In conclusion, it is important to note that effective communication is essential for organizational success and productivity enhancement.", 0),
    TextSample("Furthermore, the implementation of robust methodologies facilitates the optimization of workflows and the maximization of stakeholder value.", 0),
    TextSample("This article will explore five key strategies. First, prioritize tasks. Second, delegate effectively. Third, monitor progress. Fourth, evaluate outcomes. Fifth, iterate continuously.", 0),
    TextSample("It is widely acknowledged that artificial intelligence is transforming industries by leveraging large-scale datasets and advanced computational paradigms.", 0),
    TextSample("In today's fast-paced world, individuals must utilize cutting-edge tools to remain competitive. Moreover, adaptability is paramount for sustained growth.", 0),
    TextSample("The following analysis delves into the multifaceted dimensions of sustainability, encompassing environmental, economic, and social considerations.", 0),
    TextSample("Additionally, it should be emphasized that comprehensive planning is crucial. Subsequently, execution must be aligned with overarching strategic objectives.", 0),
    TextSample("Overall, the utilization of data-driven decision-making processes enables organizations to achieve optimal outcomes and mitigate potential risks.", 0),
    TextSample("In summary, this report provides a comprehensive overview of best practices. It is recommended that stakeholders adhere to established guidelines.", 0),
    TextSample("Moreover, leveraging synergies across departments fosters a culture of innovation. Consequently, enterprises can unlock unprecedented opportunities for expansion.", 0),
]


def load_paired_dataset(path: str | Path | None = None) -> list[TextSample]:
    """Load human/AI pairs from CSV/JSONL, else return the demo corpus."""
    if path is None:
        return list(DEMO_SAMPLES)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    import logging as _logging

    _log = _logging.getLogger(__name__)
    samples: list[TextSample] = []
    skipped = 0
    if p.suffix.lower() == ".csv":
        import csv

        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "text" not in (reader.fieldnames or []) or "label" not in (reader.fieldnames or []):
                raise ValueError("CSV must have 'text' and 'label' columns")
            for row in reader:
                try:
                    samples.append(TextSample(row["text"], int(row["label"])))
                except (ValueError, KeyError):
                    skipped += 1
                    continue
    elif p.suffix.lower() in (".jsonl", ".json"):
        import json

        with open(p, encoding="utf-8") as f:
            first = f.read(1)
            f.seek(0)
            if p.suffix.lower() == ".json" or (first == "["):
                items = json.load(f)
            else:
                items = [json.loads(line) for line in f if line.strip()]
            for it in items:
                try:
                    samples.append(TextSample(str(it["text"]), int(it["label"])))
                except (ValueError, KeyError, TypeError):
                    skipped += 1
                    continue
    else:
        raise ValueError(f"Unsupported dataset extension: {p.suffix} (use .csv or .jsonl)")
    if len(samples) < 4:
        raise ValueError(f"Need >= 4 samples, got {len(samples)}")
    if skipped:
        _log.warning("load_paired_dataset: skipped %d malformed rows in %s", skipped, p)
    return samples


def texts_and_labels(samples: list[TextSample]) -> tuple[list[str], list[int]]:
    return [s.text for s in samples], [s.label for s in samples]


def _read_text_file(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    return text or None


def load_plain_texts(
    path: str | Path,
    label: int,
    split_paragraphs: bool = False,
    min_chars: int = 40,
) -> list[TextSample]:
    """Import your own writing as training samples with a fixed label.

    Args:
        path: One text file or a folder of `.txt`/`.md` files (folders are
            searched recursively; other extensions are ignored there, while a
            directly-passed file is read regardless of extension).
        label: 1 = human-written, 0 = AI-written.
        split_paragraphs: If True, split each file on blank lines so one
            long document yields many samples; chunks below `min_chars`
            are skipped (headings/fragments add noise, not signal).
        min_chars: Minimum chunk length when splitting.

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    import logging as _logging

    _log = _logging.getLogger(__name__)
    if label not in (0, 1):
        raise ValueError(f"label must be 0 (AI) or 1 (human), got {label}")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Text import path not found: {p}")
    if p.is_file():
        files = [p]
    else:
        files = sorted(
            f for f in p.rglob("*")
            if f.is_file() and f.suffix.lower() in PLAIN_EXTENSIONS
        )
    samples: list[TextSample] = []
    skipped = 0
    for f in files:
        text = _read_text_file(f)
        if text is None:
            skipped += 1
            continue
        chunks = (
            [c.strip() for c in re.split(r"\n\s*\n", text) if len(c.strip()) >= min_chars]
            if split_paragraphs
            else [text]
        )
        if not chunks:
            skipped += 1
            continue
        for chunk in chunks:
            try:
                samples.append(TextSample(chunk, label))
            except ValueError:
                skipped += 1
    if skipped:
        _log.warning("load_plain_texts: skipped %d empty/unreadable file(s) under %s", skipped, p)
    _log.info("load_plain_texts: imported %d samples (label=%d) from %s", len(samples), label, p)
    return samples


def load_training_corpus(
    data: str | Path | None = None,
    human_dir: str | Path | None = None,
    ai_dir: str | Path | None = None,
    split_paragraphs: bool = False,
    include_demo: bool = True,
) -> list[TextSample]:
    """Assemble the full training set from every provided source.

    Typical use — your writing as the human voice, built-in AI samples as
    the contrast (so both classes exist without extra work)::

        load_training_corpus(human_dir="my_writing/")

    Sources merge in order (CSV/JSONL via `data`, built-in demo, imports).
    See finalize_corpus for dedupe/validation.
    """
    samples: list[TextSample] = []
    if data is not None:
        samples.extend(load_paired_dataset(data))
    elif include_demo:
        samples.extend(load_paired_dataset(None))
    if human_dir is not None:
        samples.extend(load_plain_texts(human_dir, 1, split_paragraphs))
    if ai_dir is not None:
        samples.extend(load_plain_texts(ai_dir, 0, split_paragraphs))
    return finalize_corpus(samples)


def finalize_corpus(samples: list[TextSample]) -> list[TextSample]:
    """Dedupe exact-duplicate texts and require >= 4 samples across classes."""
    seen: set[str] = set()
    unique: list[TextSample] = []
    for s in samples:
        if s.text not in seen:
            seen.add(s.text)
            unique.append(s)
    if len(unique) < 4:
        raise ValueError(
            f"Need >= 4 training samples, got {len(unique)}. "
            "Add --human-dir / --ai-dir (or drop --no-demo)."
        )
    labels = {s.label for s in unique}
    if labels != {0, 1}:
        have = "human-only" if labels == {1} else "AI-only"
        raise ValueError(
            f"Training data is {have} — need both classes (1=human, 0=AI). "
            "Pair --human-dir with built-in AI samples (drop --no-demo) or add --ai-dir."
        )
    return unique
