"""Shared utilities: logging, seeding, device handling, payload isolation, text helpers.

All helpers run locally with stdlib + optional torch/numpy. No external APIs.
"""
from __future__ import annotations

import logging
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

INPUT_TAG_PATTERN = re.compile(r"<input_text>(.*?)</input_text>", re.DOTALL)


class PayloadIsolationError(ValueError):
    """Raised when <input_text> payload extraction fails or looks like injection."""


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logging once and return the project logger."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        root.addHandler(handler)
    root.setLevel(numeric)
    return logging.getLogger("humaize")


def set_seed(seed: int = 42) -> None:
    """Seed python/numpy/torch RNGs for reproducibility (best-effort)."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(preference: Optional[str] = None) -> str:
    """Resolve a torch device string without importing torch at module scope."""
    if preference:
        return preference
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def extract_payload(raw: str, max_chars: int = 20_000) -> str:
    """Extract the isolated payload from <input_text>...</input_text>.

    DATASET & PAYLOAD ISOLATION contract:
      * The caller passes the *entire* raw runtime string (which may contain
        untrusted text). Only content inside exactly one <input_text> block is
        treated as data. Everything outside is discarded and never appended to
        model prompts as instructions.
      * Raises PayloadIsolationError if zero or 2+ blocks are found, or if the
        payload is empty / over the length budget. This fail-closed behaviour
        blocks prompt-injection via tag duplication or instruction mixing.

    Args:
        raw: Full raw runtime string, expected to contain one tagged block.
        max_chars: Maximum allowed payload length.

    Returns:
        The stripped payload string.
    """
    if not isinstance(raw, str):
        raise PayloadIsolationError(f"Expected str input, got {type(raw).__name__}")
    matches = INPUT_TAG_PATTERN.findall(raw)
    if len(matches) == 0:
        raise PayloadIsolationError(
            "No <input_text>...</input_text> block found. Wrap input as "
            "'<input_text>your text here</input_text>'."
        )
    if len(matches) > 1:
        raise PayloadIsolationError(
            f"Found {len(matches)} <input_text> blocks; exactly one is required "
            "(rejecting to prevent payload/instruction mixing)."
        )
    payload = matches[0].strip()
    if not payload:
        raise PayloadIsolationError("Payload inside <input_text> tags is empty.")
    if len(payload) > max_chars:
        raise PayloadIsolationError(
            f"Payload length {len(payload)} exceeds budget of {max_chars} chars."
        )
    return payload


def wrap_payload(text: str) -> str:
    """Wrap a bare string in <input_text> tags (convenience for tests/CLI)."""
    return f"<input_text>\n{text.strip()}\n</input_text>"


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+|-[A-Za-z0-9]+)*")


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter (no external data files needed).

    Note: only splits before an uppercase/digit opener, so informal text
    like "hello. world" stays one sentence and slightly understates
    burstiness — a known conservative bias, applied equally to all classes.
    """
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts = _SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def tokenize_words(text: str) -> list[str]:
    """Lowercased word tokenizer used for n-gram / diversity statistics."""
    return [m.group(0).lower() for m in _WORD.finditer(text)]


# Closed-class glue words carrying no topic signal. Used ONLY to focus
# overlap filters on content words ("here is the rewritten text" shares
# nothing topical with any payload despite matching "is"/"the").
CONTENT_STOPWORDS = frozenset(
    "a an the is are was were be been being am it its this that these those "
    "and or but of for to in on at by with from as into out up down over under "
    "so such no not very can will just than then here there what which who how "
    "when where why i you he she we they me him her us them my your his our their "
    "do does did done have has had having would could should may might must shall "
    "also more most other some any all each every own same too s t re ve ll d m".split()
)


def content_words(text: str) -> list[str]:
    """Topic-bearing words (stopwords removed) for overlap comparisons."""
    return [w for w in tokenize_words(text) if w not in CONTENT_STOPWORDS]


@dataclass
class SamplingParams:
    """Decoding hyper-parameters adjusted dynamically by the feedback loop."""

    temperature: float = 0.9
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    max_new_tokens: int = 256
    top_k: int = 0  # 0 = disabled; >0 keeps sampling to the top-k tokens
    no_repeat_ngram_size: int = 0  # 0 = disabled; >0 bans repeating n-grams (forces rewording)

    # Exploration schedule: conservative <-> wild. Entries differ in
    # randomness (temperature/top_p/top_k), anti-formula pressure
    # (repetition_penalty) and paraphrase forcing (no_repeat_ngram_size).
    # Pure decoding exploration — no linguistic rules involved.
    _SCHEDULE: ClassVar[tuple] = (
        {"temperature": 0.90, "top_p": 0.95, "repetition_penalty": 1.10, "top_k": 0,   "no_repeat_ngram_size": 0},
        {"temperature": 1.05, "top_p": 0.92, "repetition_penalty": 1.18, "top_k": 50,  "no_repeat_ngram_size": 0},
        {"temperature": 0.75, "top_p": 0.98, "repetition_penalty": 1.05, "top_k": 0,   "no_repeat_ngram_size": 0},
        {"temperature": 1.15, "top_p": 0.90, "repetition_penalty": 1.25, "top_k": 100, "no_repeat_ngram_size": 3},
        {"temperature": 0.85, "top_p": 0.96, "repetition_penalty": 1.12, "top_k": 50,  "no_repeat_ngram_size": 0},
        {"temperature": 1.00, "top_p": 0.93, "repetition_penalty": 1.20, "top_k": 0,   "no_repeat_ngram_size": 3},
        {"temperature": 0.70, "top_p": 0.97, "repetition_penalty": 1.08, "top_k": 30,  "no_repeat_ngram_size": 0},
        {"temperature": 1.30, "top_p": 0.88, "repetition_penalty": 1.30, "top_k": 100, "no_repeat_ngram_size": 3},
    )

    # Yoga Ultra greedy fast-path: first iteration uses this single draw
    # (no sampling) — 30% faster, still guided by Critic. Falls back to
    # schedule on retry if it fails gates.
    YOGA_GREEDY: ClassVar[dict] = {"temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.0, "top_k": 0, "no_repeat_ngram_size": 0}

    def mutate(self, step: int) -> "SamplingParams":
        """Deterministic exploration schedule over retry iterations.

        Successive retries walk conservative -> bold -> precise -> wild
        regions (then cycle), pushing the generator out of formulaic
        (high-predictability, low-burstiness) zones.
        """
        entry = self._SCHEDULE[step % len(self._SCHEDULE)]
        return SamplingParams(
            temperature=entry["temperature"],
            top_p=entry["top_p"],
            repetition_penalty=entry["repetition_penalty"],
            top_k=entry["top_k"],
            no_repeat_ngram_size=entry["no_repeat_ngram_size"],
            max_new_tokens=self.max_new_tokens,
        )

    def for_candidate(self, step: int, k: int) -> "SamplingParams":
        """Params for candidate k within one iteration.

        k=0 keeps these params untouched (greedy/custom calls with a single
        candidate behave exactly as before); k>=1 walks the schedule from
        step+k so simultaneous candidates explore different regions instead
        of sampling the same distribution twice.
        """
        if k <= 0:
            return self
        varied = SamplingParams().mutate(step + k)
        varied.max_new_tokens = self.max_new_tokens
        return varied

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "top_k": self.top_k,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "max_new_tokens": self.max_new_tokens,
        }


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_light() -> bool:
    """True when HUMAIZE_LIGHT=1: trade speed for lower RAM/CPU/disk use."""
    import os

    return os.environ.get("HUMAIZE_LIGHT", "0") == "1"


def lexical_similarity(a: str, b: str) -> float:
    """Token recall of `a` in `b`: fraction of input words also in the rewrite.

    Recall (not Dice/F1) because a good rewrite may add varied diction —
    what matters for meaning preservation is that the input's content words
    are still covered. Off-topic rambling shares few input words → low score.
    """
    sa, sb = set(tokenize_words(a)), set(tokenize_words(b))
    if not sa:
        return 0.0
    return len(sa & sb) / len(sa)


def ngram_recall(a: str, b: str, n: int = 2) -> float:
    """Ordered-phrase recall: fraction of input n-grams covered by the rewrite.

    Unigram recall ignores word order ("dog bites man" == "man bites dog");
    bigram recall catches phrase scrambling and topic drift that unigrams
    miss. Returns 1.0 when the input is too short to form an n-gram.
    """
    wa, wb = tokenize_words(a), tokenize_words(b)
    if len(wa) < n:
        return 1.0
    grams_a = {tuple(wa[i : i + n]) for i in range(len(wa) - n + 1)}
    grams_b = {tuple(wb[i : i + n]) for i in range(len(wb) - n + 1)}
    if not grams_a:
        return 1.0
    return len(grams_a & grams_b) / len(grams_a)


def fidelity_score(a: str, b: str) -> dict[str, float]:
    """Meaning-preservation breakdown: unigram + bigram recall + composite.

    Composite weights unigrams (topic coverage) over bigrams (phrasing):
    a faithful paraphrase keeps most words but reorders phrases, so it
    scores high on unigrams and moderately on bigrams; off-topic text
    scores low on both.
    """
    uni = lexical_similarity(a, b)
    bi = ngram_recall(a, b, 2)
    return {"unigram": uni, "bigram": bi, "composite": 0.6 * uni + 0.4 * bi}


def is_near_copy(a: str, b: str, min_unigram: float = 0.85, min_bigram: float = 0.75) -> bool:
    """True when `b` echoes `a` near-verbatim instead of rewriting it.

    Small models obey "preserve meaning" by changing nothing — a copy
    scores exactly like the input and wastes retry iterations. BOTH gates
    must trip (word overlap AND phrase overlap) so genuine paraphrases,
    which reorder phrases (low bigram recall), are never flagged.
    """
    if not a.strip() or not b.strip():
        return False
    if a.strip() == b.strip():
        return True
    return lexical_similarity(a, b) >= min_unigram and ngram_recall(a, b, 2) >= min_bigram


@dataclass
class GateSpec:
    """GateSpec / Fidelity — the triple gate.

    Single value object that owns every fidelity threshold. No consumer
    hardcodes 0.85/0.75/1.8; all read the spec. Deepens the scattered
    literal problem into one module with locality.
    """

    min_similarity: float = 0.35  # unigram recall floor
    copy_unigram: float = 0.85
    copy_bigram: float = 0.75
    max_length_ratio: float = 1.8  # bloat guard

    def is_copy(self, a: str, b: str) -> bool:
        return is_near_copy(a, b, self.copy_unigram, self.copy_bigram)

    def fidelity(self, a: str, b: str) -> dict[str, float]:
        return fidelity_score(a, b)

    def is_bloated(self, original: str, candidate: str) -> bool:
        return len(tokenize_words(candidate)) > self.max_length_ratio * max(
            len(tokenize_words(original)), 1
        )


def copy_to_clipboard(text: str) -> bool:
    """Copy text to the OS clipboard. Returns True on success, False otherwise.

    Stdlib-first with per-OS persistent methods. Never raises — clipboard
    failure must not break the pipeline.
    """
    import base64
    import platform
    import subprocess

    text = text or ""
    if not text:
        return False
    try:
        system = platform.system()
        if system == "Windows":
            # Set-Clipboard via -EncodedCommand: unicode-safe, no quoting
            # issues (payload travels base64), persists after exit.
            inner = base64.b64encode(text.encode("utf-16-le")).decode("ascii")
            ps = (
                "$t=[Text.Encoding]::Unicode.GetString("
                f"[Convert]::FromBase64String('{inner}'));"
                "Set-Clipboard -Value $t"
            )
            cmd = base64.b64encode(ps.encode("utf-16-le")).decode("ascii")
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", cmd],
                check=True, timeout=30,
            )
            return True
        if system == "Darwin":
            subprocess.run(["pbcopy"], input=text.encode("utf-8"),
                           check=True, timeout=10)
            return True
        for cmd in (["xclip", "-selection", "clipboard"], ["xsel", "--clipboard"]):
            try:
                subprocess.run(cmd, input=text.encode("utf-8"),
                               check=True, timeout=10)
                return True
            except (FileNotFoundError, subprocess.SubprocessError):
                continue
    except Exception:
        pass
    try:  # last resort: tkinter (may not persist after exit on Windows)
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
        root.destroy()
        return True
    except Exception:
        return False
