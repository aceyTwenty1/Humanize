"""MODULE 1 — Feature Extractor & Pattern Analyzer.

Computes *features* (not verdicts) locally with PyTorch + HF transformers:

  * token log-probabilities + average perplexity (causal LM scoring)
  * LM surprise variance (std of token log-probs — uniform vs bursty)
  * burstiness variance (std / CV / range of sentence-length distribution)
  * n-gram entropy (Shannon entropy of bigram/trigram distributions)
  * repetition rate (duplicate-bigram fraction — formulaic looping signal)
  * passive-voice density (surface-form auxiliary+participle rate — a *feature*
    fed to the learned classifier, never a hardcoded decision rule)
  * vocabulary predictability (mean top-1 predictive probability / confidence)
  * word/punctuation shape (mean word length, punctuation density,
    contraction rate — surface style markers, weights learned downstream)
  * detector-inspired style signals (each a *feature column* whose weight is
    learned from paired data — never a verdict rule):
      - long-word ratio + nominalization rate ("sophisticated clarity":
        revolutionize / implementation / sustainability)
      - sentence-opener diversity + short-sentence rate ("creative grammar":
        uniform SVO vs varied/fragmented rhythm)
      - hyphenated-compound + ALL-CAPS-token rates ("mechanical precision":
        decades-long / DNA-style technical density)
      - intensifier rate ("speculative focus": completely / very / highly)
      - formal-transition rate (moreover / furthermore / in conclusion)
  * second-wave trace signals (same learned-column contract):
      - AI-cliché rate (delve / tapestry / digital age / synergy)
      - hedge rate (maybe / seems / likely) vs booster rate
        (clearly / fundamentally / crucially)
      - first/second-person pronoun rates (human voice vs address)
      - question / exclamation / em-dash rates (interactive rhythm)
      - enumeration rate (First, / 1. list scaffolding)
      - Flesch-Kincaid grade (readability level)

The verdict ("human-like or not") is learned downstream by
:class:`classifier.LocalPatternClassifier`, never hardcoded here.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Optional

from config import AnalyzerConfig
from utils import resolve_device, split_sentences, tokenize_words

logger = logging.getLogger(__name__)

# Surface auxiliary + (adverbs) + past-participle-ish form. Used ONLY as a
# numeric feature column; the classification threshold is learned from data.
_PASSIVE_RE = re.compile(
    r"\b(am|is|are|was|were|be|been|being)\b"
    r"(?:\s+\w+ly)?"          # optional adverb, e.g. "was quickly taken"
    r"\s+\w+(?:ed|en|wn|d)\b",
    re.IGNORECASE,
)

# Contraction surface form ("don't", "could've"). Feature column only —
# the classifier learns its weight from data, like every other stat here.
_CONTRACTION_RE = re.compile(
    r"[A-Za-z]+\'(?:re|ve|ll|d|m|s|t)\b", re.IGNORECASE
)
_PUNCT_RE = re.compile(r"[.,;:!?—–\-'\"()\[\]…/]")
# Nominal (Latinate) style marker: implementation, sustainability, ...
# Feature column only — the learned weight decides its importance.
_NOMINAL_RE = re.compile(r"\w+(?:tion|sion|ment|ence|ance|ity)s?\b", re.IGNORECASE)
_CAPS_RE = re.compile(r"\b[A-Z]{2,}\b")
# Closed formal-connector and intensifier lists. Again: features, not rules —
# human prose uses these too; the classifier learns the distributional gap.
_TRANSITIONS_1 = frozenset(
    "moreover furthermore additionally consequently therefore thus hence "
    "nevertheless nonetheless however meanwhile subsequently accordingly "
    "conversely firstly secondly thirdly lastly overall".split()
)
_TRANSITIONS_2 = frozenset({
    ("in", "conclusion"), ("in", "summary"), ("in", "contrast"),
})
# Known AI clichés / hedge / booster vocabularies. Feature columns only:
# human prose uses these too — the classifier learns the distributional gap.
_CLICHE_WORDS = frozenset(
    "delve delves delved delving tapestry tapestries realm realms seamless "
    "seamlessly robust holistic synergy synergies paradigm paradigms leverage "
    "leverages leveraged leveraging cutting-edge fast-paced ever-evolving "
    "game-changer deep-dive state-of-the-art groundbreaking".split()
)
_CLICHE_BIGRAMS = frozenset({
    ("digital", "age"), ("vast", "amounts"), ("game", "changer"), ("deep", "dive"),
})
_HEDGES = frozenset(
    "maybe perhaps possibly might tend tends somewhat rather seem seems seemed "
    "appear appears appeared likely generally typically arguably presumably".split()
)
_BOOSTERS = frozenset(
    "clearly obviously undoubtedly crucially essentially fundamentally vitally "
    "paramount undeniably certainly definitely".split()
)
_FIRST_PERSON = frozenset({"i", "me", "my", "mine", "we", "us", "our", "ours"})
_SECOND_PERSON = frozenset({"you", "your", "yours", "yourself", "yourselves"})
_ENUM_START_RE = re.compile(
    r"^(?:first|second|third|fourth|fifth|firstly|secondly|thirdly|\d+[.)])\b",
    re.IGNORECASE,
)
_INTENSIFIERS = frozenset(
    "very highly completely extremely totally absolutely entirely deeply "
    "greatly profoundly truly really utterly exceptionally remarkably "
    "incredibly vastly widely fully".split()
)

STAT_FEATURE_NAMES: list[str] = [
    "perplexity",
    "mean_log_prob",
    "vocab_predictability",
    "logprob_std",
    "burstiness_var",
    "burstiness_cv",
    "bigram_entropy",
    "trigram_entropy",
    "passive_density",
    "mean_sentence_len",
    "lexical_diversity",
    "mean_word_len",
    "repetition_rate",
    "punctuation_density",
    "contraction_rate",
    "sentence_len_range",
    "long_word_ratio",
    "nominalization_rate",
    "sentence_opener_diversity",
    "short_sentence_rate",
    "hyphenated_rate",
    "caps_token_rate",
    "intensifier_rate",
    "transition_rate",
    "ai_cliche_rate",
    "hedge_rate",
    "booster_rate",
    "first_person_rate",
    "second_person_rate",
    "question_rate",
    "exclamation_rate",
    "emdash_rate",
    "enumeration_rate",
    "flesch_kincaid_grade",
]


@dataclass
class AnalysisResult:
    """Full per-text analysis output."""

    text: str
    token_log_probs: list[float] = field(default_factory=list)
    perplexity: float = float("nan")
    mean_log_prob: float = float("nan")
    vocab_predictability: float = float("nan")  # mean top-1 prob, higher = more predictable
    logprob_std: float = float("nan")  # std of token log-probs (surprise variance)
    burstiness_var: float = 0.0  # std of sentence word-counts
    burstiness_cv: float = 0.0  # std / mean (scale-free)
    bigram_entropy: float = 0.0
    trigram_entropy: float = 0.0
    passive_density: float = 0.0  # passive-like constructions per sentence
    mean_sentence_len: float = 0.0
    lexical_diversity: float = 0.0  # unique / total words
    mean_word_len: float = 0.0  # avg chars per word
    repetition_rate: float = 0.0  # 1 - unique_bigrams/total_bigrams
    punctuation_density: float = 0.0  # punctuation marks per word
    contraction_rate: float = 0.0  # contractions per word
    sentence_len_range: float = 0.0  # max - min sentence word-counts
    long_word_ratio: float = 0.0  # words with >= 9 chars / total
    nominalization_rate: float = 0.0  # -tion/-ment/-ence/... words / total
    sentence_opener_diversity: float = 0.0  # unique first-words / sentences
    short_sentence_rate: float = 0.0  # sentences with < 6 words / total
    hyphenated_rate: float = 0.0  # hyphenated compounds / total words
    caps_token_rate: float = 0.0  # ALL-CAPS tokens (len>=2) / total words
    intensifier_rate: float = 0.0  # very/completely/... per word
    transition_rate: float = 0.0  # formal connectors per sentence
    ai_cliche_rate: float = 0.0  # delve/tapestry/digital age per word
    hedge_rate: float = 0.0  # maybe/seems/likely per word
    booster_rate: float = 0.0  # clearly/fundamentally per word
    first_person_rate: float = 0.0  # i/we per word
    second_person_rate: float = 0.0  # you per word
    question_rate: float = 0.0  # ? sentences / total
    exclamation_rate: float = 0.0  # ! sentences / total
    emdash_rate: float = 0.0  # em-dashes per word
    enumeration_rate: float = 0.0  # First,/1. openers per sentence
    flesch_kincaid_grade: float = 0.0  # readability grade level
    num_tokens: int = 0
    num_sentences: int = 0

    def to_feature_dict(self) -> dict[str, float]:
        d = asdict(self)
        d.pop("text", None)
        d.pop("token_log_probs", None)
        return {k: float(d.get(k, 0.0)) for k in STAT_FEATURE_NAMES if k in d}

    def to_feature_vector(self) -> list[float]:
        import math as _m

        vec: list[float] = []
        for k in STAT_FEATURE_NAMES:
            v = self.to_feature_dict()[k]
            if k == "perplexity" and (_m.isnan(v) or _m.isinf(v)):
                v = 500.0  # cap for downstream scaler when LM is unavailable
            if _m.isnan(v) or _m.isinf(v):
                v = 0.0
            vec.append(float(v))
        # log-compress perplexity so one column does not dominate the scaler
        vec[0] = float(_m.log1p(max(vec[0], 0.0)))
        return vec


class PatternAnalyzer:
    """Local statistical feature extractor backed by a small causal LM.

    Args:
        config: AnalyzerConfig with model_name / device / max_length.
        load_model: If False, skip LM loading (stats-only mode for fast
            classifier training / CI). LM features return NaN and the
            classifier must be trained with ``use_stats_features`` on the
            non-LM subset — see :meth:`cheap_stats_vector`.
    """

    def __init__(self, config: Optional[AnalyzerConfig] = None, load_model: bool = True) -> None:
        self.config = config or AnalyzerConfig()
        self.device = resolve_device(self.config.device)
        self._model = None
        self._tokenizer = None
        self.has_lm = False
        if load_model:
            try:
                self._load_lm()
            except Exception as exc:  # offline / no weights -> stats-only mode
                logger.warning(
                    "PatternAnalyzer: could not load LM '%s' (%s). "
                    "Falling back to stats-only mode.",
                    self.config.model_name,
                    exc,
                )

    # ------------------------------------------------------------------ setup
    def _load_lm(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        import torch

        name = self.config.model_name
        logger.info("PatternAnalyzer: loading local LM '%s' on %s", name, self.device)
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=False, local_files_only=False)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(
            name, trust_remote_code=False, local_files_only=False
        )
        mdl.to(self.device)
        mdl.eval()
        self._model = mdl
        self._tokenizer = tok
        self.has_lm = True

    @property
    def model_name(self) -> str:
        return self.config.model_name

    # ------------------------------------------------------------- LM scoring
    def token_log_probs(self, text: str) -> list[float]:
        """Per-token log P(token_i | prefix) under the local LM (no grad)."""
        if not self.has_lm:
            return []
        import torch

        assert self._model is not None and self._tokenizer is not None
        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )
        input_ids = enc["input_ids"].to(self.device)
        if input_ids.shape[1] < 2:
            return []
        with torch.no_grad():
            out = self._model(input_ids)
            log_probs = torch.log_softmax(out.logits, dim=-1)
            # log P(x_{t} | x_{<t}) for t >= 1
            gathered = log_probs[:, :-1, :].gather(
                2, input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
        return gathered[0].detach().float().cpu().tolist()

    def perplexity(self, text: str) -> float:
        """Average perplexity with sliding window for long inputs."""
        if not self.has_lm:
            return float("nan")
        import torch

        assert self._model is not None and self._tokenizer is not None
        enc = self._tokenizer(text, return_tensors="pt", truncation=False)
        ids = enc["input_ids"][0]
        max_len, stride = self.config.max_length, self.config.stride
        if len(ids) == 0:
            return float("nan")
        nll_sum, n_tokens = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(ids), stride):
                window = ids[start : start + max_len].unsqueeze(0).to(self.device)
                if window.shape[1] < 2:
                    continue
                logits = self._model(window).logits
                lp = torch.log_softmax(logits, dim=-1)
                tgt = window[:, 1:]
                nll = -lp[:, :-1, :].gather(2, tgt.unsqueeze(-1)).squeeze(-1)
                # only score the strided (non-overlapping) tail to avoid double count
                take = nll[:, -min(stride, nll.shape[1]) :] if start > 0 else nll
                nll_sum += take.sum().item()
                n_tokens += take.numel()
        if n_tokens == 0:
            return float("nan")
        return math.exp(nll_sum / n_tokens)

    def vocabulary_predictability(self, text: str) -> float:
        """Mean top-1 next-token probability (model confidence).

        High values => every token was highly predictable => formulaic/AI-like
        *tendency* (again, just a feature — the classifier learns the weight).
        """
        if not self.has_lm:
            return float("nan")
        import torch

        assert self._model is not None and self._tokenizer is not None
        enc = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.config.max_length
        )
        ids = enc["input_ids"].to(self.device)
        if ids.shape[1] < 2:
            return float("nan")
        with torch.no_grad():
            probs = torch.softmax(self._model(ids).logits[:, :-1, :], dim=-1)
            top1, _ = probs.max(dim=-1)
        return float(top1.mean().item())

    # ------------------------------------------------------- surface features
    @staticmethod
    def burstiness(text: str) -> tuple[float, float, float]:
        """Return (std, cv, mean) of sentence word-count distribution."""
        import numpy as np

        sentences = split_sentences(text)
        if len(sentences) < 2:
            words = tokenize_words(text)
            mean = float(len(words)) if words else 0.0
            return 0.0, 0.0, mean
        lens = np.array([max(len(tokenize_words(s)), 1) for s in sentences], dtype=float)
        std = float(lens.std())
        mean = float(lens.mean())
        cv = float(std / mean) if mean > 0 else 0.0
        return std, cv, mean

    @staticmethod
    def ngram_entropy(text: str, n: int = 2) -> float:
        """Shannon entropy (bits) of the n-gram frequency distribution."""
        words = tokenize_words(text)
        if len(words) < n:
            return 0.0
        grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        counts = Counter(grams)
        total = sum(counts.values())
        return float(-sum((c / total) * math.log2(c / total) for c in counts.values()))

    @staticmethod
    def passive_voice_density(text: str) -> float:
        """Passive-like constructions per sentence (feature only, not a rule)."""
        sentences = split_sentences(text)
        if not sentences:
            return 0.0
        hits = sum(1 for s in sentences if _PASSIVE_RE.search(s))
        return hits / len(sentences)

    @staticmethod
    def lexical_diversity(text: str) -> float:
        words = tokenize_words(text)
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    @staticmethod
    def mean_word_len(text: str) -> float:
        words = tokenize_words(text)
        if not words:
            return 0.0
        return sum(len(w) for w in words) / len(words)

    @staticmethod
    def repetition_rate(text: str) -> float:
        """Duplicate-bigram fraction (1 = loops the same pairs, 0 = all fresh)."""
        words = tokenize_words(text)
        if len(words) < 3:
            return 0.0
        grams = [tuple(words[i : i + 2]) for i in range(len(words) - 1)]
        return 1.0 - len(set(grams)) / len(grams)

    @staticmethod
    def punctuation_density(text: str) -> float:
        words = tokenize_words(text)
        if not words:
            return 0.0
        return len(_PUNCT_RE.findall(text)) / len(words)

    @staticmethod
    def contraction_rate(text: str) -> float:
        words = tokenize_words(text)
        if not words:
            return 0.0
        return len(_CONTRACTION_RE.findall(text)) / len(words)

    @staticmethod
    def sentence_len_range(text: str) -> float:
        sentences = split_sentences(text)
        if len(sentences) < 2:
            return 0.0
        lens = [len(tokenize_words(s)) for s in sentences]
        return float(max(lens) - min(lens))

    @staticmethod
    def long_word_ratio(text: str, min_len: int = 9) -> float:
        words = tokenize_words(text)
        if not words:
            return 0.0
        return sum(1 for w in words if len(w) >= min_len) / len(words)

    @staticmethod
    def nominalization_rate(text: str) -> float:
        words = tokenize_words(text)
        if not words:
            return 0.0
        return sum(1 for w in words if _NOMINAL_RE.fullmatch(w)) / len(words)

    @staticmethod
    def sentence_opener_diversity(text: str) -> float:
        """Unique sentence first-words / sentence count (structural variety).

        Formulaic text opens every sentence the same way ("First, ... Second,
        ..."); varied prose doesn't. <2 sentences -> 0.0 (no signal).
        """
        sentences = split_sentences(text)
        if len(sentences) < 2:
            return 0.0
        openers = set()
        for s in sentences:
            words = tokenize_words(s)
            if words:
                openers.add(words[0])
        return len(openers) / len(sentences)

    @staticmethod
    def short_sentence_rate(text: str, max_len: int = 6) -> float:
        sentences = split_sentences(text)
        if not sentences:
            return 0.0
        return sum(1 for s in sentences if len(tokenize_words(s)) < max_len) / len(sentences)

    @staticmethod
    def hyphenated_rate(text: str) -> float:
        words = tokenize_words(text)
        if not words:
            return 0.0
        return sum(1 for w in words if "-" in w) / len(words)

    @staticmethod
    def caps_token_rate(text: str) -> float:
        words = tokenize_words(text)
        if not words:
            return 0.0
        return len(_CAPS_RE.findall(text)) / len(words)

    @staticmethod
    def intensifier_rate(text: str) -> float:
        words = tokenize_words(text)
        if not words:
            return 0.0
        return sum(1 for w in words if w in _INTENSIFIERS) / len(words)

    @staticmethod
    def transition_rate(text: str) -> float:
        """Formal connectors (moreover / in conclusion / ...) per sentence."""
        sentences = split_sentences(text)
        if not sentences:
            return 0.0
        hits = 0
        for s in sentences:
            words = tokenize_words(s)
            if any(w in _TRANSITIONS_1 for w in words):
                hits += 1
                continue
            if any(tuple(words[i : i + 2]) in _TRANSITIONS_2 for i in range(len(words) - 1)):
                hits += 1
        return hits / len(sentences)

    @staticmethod
    def ai_cliche_rate(text: str) -> float:
        words = tokenize_words(text)
        if not words:
            return 0.0
        hits = sum(1 for w in words if w in _CLICHE_WORDS)
        hits += sum(
            1 for i in range(len(words) - 1)
            if (words[i], words[i + 1]) in _CLICHE_BIGRAMS
        )
        return hits / len(words)

    @staticmethod
    def _lexicon_rate(text: str, lexicon: frozenset) -> float:
        words = tokenize_words(text)
        if not words:
            return 0.0
        return sum(1 for w in words if w in lexicon) / len(words)

    @staticmethod
    def hedge_rate(text: str) -> float:
        return PatternAnalyzer._lexicon_rate(text, _HEDGES)

    @staticmethod
    def booster_rate(text: str) -> float:
        return PatternAnalyzer._lexicon_rate(text, _BOOSTERS)

    @staticmethod
    def first_person_rate(text: str) -> float:
        return PatternAnalyzer._lexicon_rate(text, _FIRST_PERSON)

    @staticmethod
    def second_person_rate(text: str) -> float:
        return PatternAnalyzer._lexicon_rate(text, _SECOND_PERSON)

    @staticmethod
    def question_rate(text: str) -> float:
        sentences = split_sentences(text)
        if not sentences:
            return 0.0
        return sum(1 for s in sentences if s.rstrip().endswith("?")) / len(sentences)

    @staticmethod
    def exclamation_rate(text: str) -> float:
        sentences = split_sentences(text)
        if not sentences:
            return 0.0
        return sum(1 for s in sentences if s.rstrip().endswith("!")) / len(sentences)

    @staticmethod
    def emdash_rate(text: str) -> float:
        words = tokenize_words(text)
        if not words:
            return 0.0
        return (text.count("—") + text.count("--")) / len(words)

    @staticmethod
    def enumeration_rate(text: str) -> float:
        """List-scaffold openers (First, / Secondly, / 1.) per sentence."""
        sentences = split_sentences(text)
        if not sentences:
            return 0.0
        return sum(1 for s in sentences if _ENUM_START_RE.match(s.strip())) / len(sentences)

    @staticmethod
    def _syllable_count(word: str) -> int:
        word = word.lower()
        count, prev_vowel = 0, False
        for ch in word:
            is_vowel = ch in "aeiouy"
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith("e") and count > 1:
            count -= 1
        return max(count, 1)

    @staticmethod
    def flesch_kincaid_grade(text: str) -> float:
        """US grade level from words/sentence and syllables/word (heuristic)."""
        sentences = split_sentences(text)
        words = tokenize_words(text)
        if not sentences or not words:
            return 0.0
        wps = len(words) / len(sentences)
        spw = sum(PatternAnalyzer._syllable_count(w) for w in words) / len(words)
        return 0.39 * wps + 11.8 * spw - 15.59

    # ------------------------------------------------------------- main entry
    def analyze(self, text: str) -> AnalysisResult:
        """Compute the full feature set for one text (LM + surface stats)."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("analyze() requires a non-empty string")
        text = text.strip()

        lps = self.token_log_probs(text)
        if lps:
            import numpy as np

            mean_lp = float(np.mean(lps))
            lp_std = float(np.std(lps))
            ppl = float(self.perplexity(text))
            pred = float(self.vocabulary_predictability(text))
            n_tok = len(lps) + 1
        else:
            mean_lp, lp_std, ppl, pred, n_tok = (
                float("nan"), float("nan"), float("nan"), float("nan"), 0
            )

        std, cv, mean_len = self.burstiness(text)
        sentences = split_sentences(text)
        sent_lens = [len(tokenize_words(s)) for s in sentences]
        len_range = float(max(sent_lens) - min(sent_lens)) if len(sent_lens) >= 2 else 0.0
        return AnalysisResult(
            text=text[:2000],
            token_log_probs=lps,
            perplexity=ppl,
            mean_log_prob=mean_lp,
            vocab_predictability=pred,
            logprob_std=lp_std,
            burstiness_var=std,
            burstiness_cv=cv,
            bigram_entropy=self.ngram_entropy(text, 2),
            trigram_entropy=self.ngram_entropy(text, 3),
            passive_density=self.passive_voice_density(text),
            mean_sentence_len=mean_len,
            lexical_diversity=self.lexical_diversity(text),
            mean_word_len=self.mean_word_len(text),
            repetition_rate=self.repetition_rate(text),
            punctuation_density=self.punctuation_density(text),
            contraction_rate=self.contraction_rate(text),
            sentence_len_range=len_range,
            long_word_ratio=self.long_word_ratio(text),
            nominalization_rate=self.nominalization_rate(text),
            sentence_opener_diversity=self.sentence_opener_diversity(text),
            short_sentence_rate=self.short_sentence_rate(text),
            hyphenated_rate=self.hyphenated_rate(text),
            caps_token_rate=self.caps_token_rate(text),
            intensifier_rate=self.intensifier_rate(text),
            transition_rate=self.transition_rate(text),
            ai_cliche_rate=self.ai_cliche_rate(text),
            hedge_rate=self.hedge_rate(text),
            booster_rate=self.booster_rate(text),
            first_person_rate=self.first_person_rate(text),
            second_person_rate=self.second_person_rate(text),
            question_rate=self.question_rate(text),
            exclamation_rate=self.exclamation_rate(text),
            emdash_rate=self.emdash_rate(text),
            enumeration_rate=self.enumeration_rate(text),
            flesch_kincaid_grade=self.flesch_kincaid_grade(text),
            num_tokens=n_tok,
            num_sentences=len(sentences),
        )

    def analyze_batch(self, texts: list[str]) -> list[AnalysisResult]:
        return [self.analyze(t) for t in texts]

    def feature_matrix(self, texts: list[str]) -> "object":
        """Stacked (n, 34) feature matrix for classifier training (numpy)."""
        import numpy as np

        return np.array([self.analyze(t).to_feature_vector() for t in texts], dtype=float)

    # ------------------------------------------- LM-free stats (fast training)
    @staticmethod
    def cheap_stats_vector(text: str) -> list[float]:
        """30-dim LM-free stats vector (no model download needed).

        Order: [burst_std, burst_cv, bigram_H, trigram_H, passive_dens,
                mean_sent_len, lexical_diversity, mean_word_len,
                repetition_rate, punctuation_density, contraction_rate,
                sentence_len_range, long_word_ratio, nominalization_rate,
                opener_diversity, short_sentence_rate, hyphenated_rate,
                caps_token_rate, intensifier_rate, transition_rate,
                ai_cliche_rate, hedge_rate, booster_rate, first_person_rate,
                second_person_rate, question_rate, exclamation_rate,
                emdash_rate, enumeration_rate, flesch_kincaid_grade].
        Used to train the discriminator quickly; LM columns (ppl etc.) are
        appended when the scorer model is available.
        """
        std, cv, mean_len = PatternAnalyzer.burstiness(text)
        sentences = split_sentences(text)
        sent_lens = [len(tokenize_words(s)) for s in sentences]
        len_range = float(max(sent_lens) - min(sent_lens)) if len(sent_lens) >= 2 else 0.0
        return [
            std,
            cv,
            PatternAnalyzer.ngram_entropy(text, 2),
            PatternAnalyzer.ngram_entropy(text, 3),
            PatternAnalyzer.passive_voice_density(text),
            mean_len,
            PatternAnalyzer.lexical_diversity(text),
            PatternAnalyzer.mean_word_len(text),
            PatternAnalyzer.repetition_rate(text),
            PatternAnalyzer.punctuation_density(text),
            PatternAnalyzer.contraction_rate(text),
            len_range,
            PatternAnalyzer.long_word_ratio(text),
            PatternAnalyzer.nominalization_rate(text),
            PatternAnalyzer.sentence_opener_diversity(text),
            PatternAnalyzer.short_sentence_rate(text),
            PatternAnalyzer.hyphenated_rate(text),
            PatternAnalyzer.caps_token_rate(text),
            PatternAnalyzer.intensifier_rate(text),
            PatternAnalyzer.transition_rate(text),
            PatternAnalyzer.ai_cliche_rate(text),
            PatternAnalyzer.hedge_rate(text),
            PatternAnalyzer.booster_rate(text),
            PatternAnalyzer.first_person_rate(text),
            PatternAnalyzer.second_person_rate(text),
            PatternAnalyzer.question_rate(text),
            PatternAnalyzer.exclamation_rate(text),
            PatternAnalyzer.emdash_rate(text),
            PatternAnalyzer.enumeration_rate(text),
            PatternAnalyzer.flesch_kincaid_grade(text),
        ]

    CHEAP_FEATURE_NAMES: list[str] = [
        "burstiness_var",
        "burstiness_cv",
        "bigram_entropy",
        "trigram_entropy",
        "passive_density",
        "mean_sentence_len",
        "lexical_diversity",
        "mean_word_len",
        "repetition_rate",
        "punctuation_density",
        "contraction_rate",
        "sentence_len_range",
        "long_word_ratio",
        "nominalization_rate",
        "sentence_opener_diversity",
        "short_sentence_rate",
        "hyphenated_rate",
        "caps_token_rate",
        "intensifier_rate",
        "transition_rate",
        "ai_cliche_rate",
        "hedge_rate",
        "booster_rate",
        "first_person_rate",
        "second_person_rate",
        "question_rate",
        "exclamation_rate",
        "emdash_rate",
        "enumeration_rate",
        "flesch_kincaid_grade",
    ]


if __name__ == "__main__":  # smoke test: python analyzer.py
    import json

    a = PatternAnalyzer(load_model=False)
    demo = "I dunno — yesterday ran way over. Honestly half of it could've been an email!"
    r = a.analyze(demo)
    print(json.dumps(r.to_feature_dict(), indent=2))
