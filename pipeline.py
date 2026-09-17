"""MODULE 5a — Runtime Inference & Isolation Pipeline.

Orchestration:  raw tagged input -> isolate payload -> analyze -> generate
candidates -> Critic validation -> adaptive retry loop.

Isolation contract (fail-closed):
  * Input MUST contain exactly one <input_text>...</input_text> block.
  * Only the inner payload is ever sent to the analyzer / generator.
  * Outer text (even if it looks like instructions) is discarded and logged,
    never appended to model prompts. The generator additionally instructs the
    model to treat tagged content as DATA.

Feedback loop:
  * Score each candidate with LocalPatternClassifier (Human-Likeness).
  * If best < target (default 0.85), mutate SamplingParams (temperature /
    top_p / repetition_penalty schedule — no linguistic rules) and retry.
  * Return the best candidate seen across all iterations with full telemetry.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from config import PipelineConfig
from utils import (
    PayloadIsolationError,
    SamplingParams,
    extract_payload,
    fidelity_score,
    is_near_copy,
    lexical_similarity,
    split_sentences,
    tokenize_words,
)

logger = logging.getLogger(__name__)

# Long inputs are rewritten sentence-by-sentence above this size: small
# local models hold one sentence in mind far better than a paragraph.
CHUNK_MIN_WORDS = 40
# Verbosity guard: a rewrite may not bloat past this multiple of the input
# word count. Padding with extra clauses games burstiness/variety scores
# while drifting from the original meaning — reject and retry instead.
MAX_LENGTH_RATIO = 1.8

# Appended to the generator guidance once the model has echoed the input:
# names the failure mode so the next iteration rewords instead of copying.
COPY_WARNING = (
    "The previous attempt copied the input nearly verbatim, which is a "
    "failure. Substantially reword and restructure while keeping the meaning."
)

try:
    from analyzer import AnalysisResult, PatternAnalyzer
    from classifier import LocalPatternClassifier
    from generator import RewriteCandidate, TextRewriter
except ImportError:  # pragma: no cover
    from src.humaize.analyzer import AnalysisResult, PatternAnalyzer  # type: ignore
    from src.humaize.classifier import LocalPatternClassifier  # type: ignore
    from src.humaize.generator import RewriteCandidate, TextRewriter  # type: ignore


@dataclass
class ScoredCandidate:
    text: str
    score: float
    params: SamplingParams
    iteration: int
    similarity: float = 0.0  # unigram recall with the input payload
    bigram_similarity: float = 0.0  # ordered-phrase recall (word order kept?)
    fidelity: float = 0.0  # 0.6*unigram + 0.4*bigram composite
    is_copy: bool = False  # near-verbatim echo, not a rewrite
    is_bloated: bool = False  # padded past MAX_LENGTH_RATIO, not a rewrite
    chunks_kept: int = 0  # chunked mode: sentences kept original (drifted)


@dataclass
class PipelineResult:
    best_text: str
    best_score: float
    target_score: float
    criteria_met: bool
    iterations_used: int
    original_features: dict[str, float]
    best_features: dict[str, float] | None
    candidates: list[ScoredCandidate] = field(default_factory=list)
    elapsed_s: float = 0.0
    best_similarity: float = 0.0
    min_similarity: float = 0.0
    explanation: str = ""  # detected AI patterns (learned, shown to user)
    explanation_details: str = ""  # per-feature direction + token evidence
    best_bigram_similarity: float = 0.0
    best_fidelity: float = 0.0
    fixed_patterns: str = ""  # flagged in input, no longer flagged in rewrite
    remaining_patterns: str = ""  # still flagged after rewriting
    copies_rejected: int = 0  # near-verbatim echoes refused as rewrites
    best_is_copy: bool = False  # True only when NOTHING but echoes was generated
    chunks_total: int = 0  # chunked mode: input sentence count (0 = one-shot)
    chunks_kept_original: int = 0  # ... of which kept verbatim (drift guard)
    polish_passes_used: int = 0  # refinement rounds applied to the winner


class HumanizationPipeline:
    """Runtime engine wiring analyzer -> rewriter -> critic with retries."""

    def __init__(
        self,
        analyzer: PatternAnalyzer,
        rewriter: TextRewriter,
        critic: LocalPatternClassifier,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.analyzer = analyzer
        self.rewriter = rewriter
        self.critic = critic
        self.config = config or PipelineConfig()

    # ------------------------------------------------------------- isolation
    def isolate(self, raw_input: str) -> str:
        """Extract payload; fail closed on missing/duplicated tags."""
        payload = extract_payload(raw_input, max_chars=self.config.max_chars)
        outside_len = max(len(raw_input) - len(payload), 0)
        if outside_len > 0:
            logger.debug("HumanizationPipeline: discarded %d chars outside <input_text>", outside_len)
        return payload

    def _rewrite_chunked(
        self,
        sentences: list[str],
        guidance: str | None,
        params: SamplingParams,
        sim_floor: float,
    ) -> tuple[RewriteCandidate, int, int]:
        """Rewrite a long input sentence-by-sentence, then rejoin.

        Small models drift on paragraphs but handle single sentences. Each
        chunk is rewritten independently with the shared guidance; a chunk
        rewrite that echoes, bloats, or drifts below the floor keeps the
        ORIGINAL sentence (perfect fidelity beats rambling). Returns the
        joined candidate plus kept-original and chunk-echo counts.
        """
        parts: list[str] = []
        kept, echoes = 0, 0
        for sent in sentences:
            try:
                cands = self.rewriter.rewrite(
                    sent, params=params, num_candidates=1, guidance=guidance
                )
            except Exception as exc:
                logger.warning("HumanizationPipeline: chunk generation failed (%s)", exc)
                cands = []
            pick: str | None = None
            for c in cands:
                if is_near_copy(sent, c.text):
                    echoes += 1
                    continue
                if lexical_similarity(sent, c.text) < sim_floor:
                    continue
                if len(tokenize_words(c.text)) > MAX_LENGTH_RATIO * max(len(tokenize_words(sent)), 1):
                    continue  # padded chunk: bloat, not a rewrite
                pick = c.text
                break
            if pick is None:
                kept += 1
                parts.append(sent)
            else:
                parts.append(pick)
        return RewriteCandidate(text=" ".join(parts), params=params), kept, echoes

    # ------------------------------------------------------------------- run
    def run(
        self,
        raw_input: str,
        target_score: Optional[float] = None,
        max_iters: Optional[int] = None,
        num_candidates: Optional[int] = None,
        min_similarity: Optional[float] = None,
        polish_passes: Optional[int] = None,
    ) -> PipelineResult:
        """End-to-end humanization with adaptive resampling loop.

        A candidate must pass BOTH gates: human-likeness >= target AND
        lexical similarity to the input >= min_similarity. Off-topic
        rambling — however "human" it scores — is rejected and retried.
        """
        t0 = time.time()
        target = target_score if target_score is not None else self.config.target_score
        iters = max_iters if max_iters is not None else self.config.max_iters
        n_cand = num_candidates if num_candidates is not None else self.config.num_candidates
        sim_floor = min_similarity if min_similarity is not None else self.config.min_similarity

        payload = self.isolate(raw_input)
        orig_analysis = self.analyzer.analyze(payload)
        orig_features = orig_analysis.to_feature_dict()
        try:
            orig_score = float(self.critic.human_likeness_score(payload))
        except Exception:
            orig_score = float("nan")
        logger.info("HumanizationPipeline: input score=%.3f target=%.2f", orig_score, target)

        # Attribute: which LEARNED patterns make this input recognizable as
        # AI? The resulting guidance targets the rewrite at those patterns.
        guidance: str | None = None
        explanation = ""
        explanation_details = ""
        orig_flagged: set[str] = set()
        try:
            expl = self.critic.explain(payload)
            explanation = expl.summary()
            explanation_details = expl.details()
            orig_flagged = {f.name for f in expl.flagged}
            guidance = expl.guidance() or None
            logger.info("HumanizationPipeline: detected patterns: %s", explanation)
        except Exception as exc:
            logger.info("HumanizationPipeline: no pattern explanation (%s)", exc)

        base_params = SamplingParams()
        scored: list[ScoredCandidate] = []
        best: Optional[ScoredCandidate] = None
        copies_rejected = 0

        # Long multi-sentence inputs go sentence-by-sentence (small models
        # drift on paragraphs); short ones take the direct path.
        sentences = split_sentences(payload)
        chunked = len(sentences) > 1 and len(tokenize_words(payload)) > CHUNK_MIN_WORDS
        if chunked:
            logger.info("HumanizationPipeline: long input (%d words, %d sentences) — "
                        "rewriting sentence-by-sentence", len(tokenize_words(payload)), len(sentences))

        for it in range(max(iters, 1)):
            params = base_params.mutate(it)
            logger.info(
                "HumanizationPipeline: iter %d/%d (temp=%.2f top_p=%.2f rep=%.2f)",
                it + 1, iters, params.temperature, params.top_p, params.repetition_penalty,
            )
            # Candidate k=0 uses the iteration params; k>=1 walks the
            # schedule so simultaneous candidates explore different
            # decoding regions instead of sampling one distribution twice.
            # Guidance escalation is recomputed per candidate so a copy is
            # answered with the warning on the very next draw.
            for k in range(max(n_cand, 1)):
                pk = params.for_candidate(it, k)
                if copies_rejected:
                    gk = ((guidance + "\n") if guidance else "") + COPY_WARNING
                else:
                    gk = guidance
                try:
                    if chunked:
                        ck, kept, chunk_echoes = self._rewrite_chunked(
                            sentences, gk, pk, sim_floor)
                        copies_rejected += chunk_echoes
                        cands = [ck]
                        chunk_kept = kept
                    else:
                        cands = self.rewriter.rewrite(
                            payload, params=pk, num_candidates=1, guidance=gk
                        )
                        chunk_kept = 0
                except Exception as exc:
                    logger.warning("HumanizationPipeline: generation failed on iter %d (%s)", it, exc)
                    continue
                for c in cands:
                    s = float(self.critic.human_likeness_score(c.text))
                    sim = lexical_similarity(payload, c.text)
                    fid = fidelity_score(payload, c.text)
                    copy = is_near_copy(payload, c.text)
                    bloated = (len(tokenize_words(c.text))
                               > MAX_LENGTH_RATIO * max(len(tokenize_words(payload)), 1))
                    sc = ScoredCandidate(text=c.text, score=s, params=pk,
                                         iteration=it, similarity=sim,
                                         bigram_similarity=fid["bigram"],
                                         fidelity=fid["composite"],
                                         is_copy=copy,
                                         is_bloated=bloated,
                                         chunks_kept=chunk_kept if chunked else 0)
                    scored.append(sc)
                    if bloated:
                        logger.info("HumanizationPipeline: iter %d bloated candidate "
                                    "rejected (%.1fx input length)", it + 1,
                                    len(tokenize_words(c.text)) / max(len(tokenize_words(payload)), 1))
                        continue  # verbosity hack: padding is not humanizing
                    if copy:
                        copies_rejected += 1
                        logger.info("HumanizationPipeline: iter %d near-verbatim copy "
                                    "rejected (score=%.3f)", it + 1, s)
                        continue  # an echo is not a rewrite, however it scores
                    if sim < sim_floor:
                        continue  # on-topic gate: rambling is rejected, not rewarded
                    if best is None or s > best.score:
                        best = sc
            iter_cands = [s for s in scored if s.iteration == it]
            passing = [s for s in iter_cands if s.similarity >= sim_floor]
            logger.info("HumanizationPipeline: iter %d best=%.3f sim=%.2f (global best=%.3f)",
                        it + 1,
                        max((s.score for s in passing), default=0.0),
                        max((s.similarity for s in iter_cands), default=0.0),
                        best.score if best else 0.0)
            if best is not None and best.score >= target:
                break  # criteria met — stop early

        if best is None:
            # No acceptable rewrite. Prefer an attempted (changed, lean)
            # rewrite over an echo, padding, or rambling — NOT met either way.
            changed = [s for s in scored if not s.is_copy and not s.is_bloated]
            if changed:
                best = max(changed, key=lambda s: s.score)
                logger.warning(
                    "HumanizationPipeline: no candidate passed the gates; "
                    "returning best changed attempt (score=%.3f sim=%.2f) as not-met.",
                    best.score, best.similarity)
            else:
                fallback = max(scored, key=lambda s: s.similarity, default=None)
                if fallback is None:
                    raise RuntimeError("HumanizationPipeline: no candidates generated in any iteration")
                logger.warning(
                    "HumanizationPipeline: model only echoed the input (%d copies); "
                    "returning echo as not-met.", copies_rejected)
                best = fallback
        # Polish pass: the winner may still carry AI traces the first rewrite
        # missed. Re-explain the CURRENT best, rewrite it once more against
        # the surviving patterns, and keep the polished version only if it
        # scores higher without losing fidelity. Never polishes echoes/padding.
        polish_budget = polish_passes if polish_passes is not None else self.config.polish_passes
        polish_used = 0
        if (best is not None and not best.is_copy and not best.is_bloated
                and not (best.score >= target and best.similarity >= sim_floor)):
            for _ in range(max(polish_budget, 0)):
                try:
                    g2 = self.critic.explain(best.text).guidance() or None
                except Exception:
                    break
                try:
                    cands2 = self.rewriter.rewrite(
                        best.text, params=SamplingParams(temperature=1.0, top_p=0.95),
                        num_candidates=1, guidance=g2)
                except Exception as exc:
                    logger.warning("HumanizationPipeline: polish generation failed (%s)", exc)
                    break
                improved = False
                for c in cands2:
                    s2 = float(self.critic.human_likeness_score(c.text))
                    sim2 = lexical_similarity(payload, c.text)
                    fid2 = fidelity_score(payload, c.text)
                    if (s2 > best.score and sim2 >= sim_floor
                            and not is_near_copy(payload, c.text)
                            and len(tokenize_words(c.text))
                            <= MAX_LENGTH_RATIO * max(len(tokenize_words(payload)), 1)):
                        logger.info("HumanizationPipeline: polish improved %.3f -> %.3f",
                                    best.score, s2)
                        best = ScoredCandidate(text=c.text, score=s2, params=c.params,
                                               iteration=best.iteration, similarity=sim2,
                                               bigram_similarity=fid2["bigram"],
                                               fidelity=fid2["composite"])
                        improved = True
                        polish_used += 1
                if not improved:
                    break
        try:
            best_features: dict[str, float] | None = self.analyzer.analyze(best.text).to_feature_dict()
        except Exception:
            best_features = None
        # In-depth close-out: which detected patterns did the rewrite actually
        # fix, and which survived? Compares flagged sets before vs after.
        fixed, remaining = "", ""
        try:
            after = self.critic.explain(best.text)
            after_names = {f.name for f in after.flagged}
            fixed_names = sorted(orig_flagged - after_names)
            remaining_names = sorted(orig_flagged & after_names)
            fixed = ", ".join(fixed_names) if fixed_names else (
                "none — rewrite kept the same flagged patterns" if orig_flagged else "")
            remaining = ", ".join(remaining_names)
        except Exception as exc:
            logger.debug("HumanizationPipeline: post-rewrite explain skipped (%s)", exc)
        result = PipelineResult(
            best_text=best.text,
            best_score=best.score,
            target_score=target,
            criteria_met=bool(best.score >= target and best.similarity >= sim_floor
                              and not best.is_copy and not best.is_bloated),
            iterations_used=(max(s.iteration for s in scored) + 1) if scored else 0,
            original_features=orig_features,
            best_features=best_features,
            candidates=scored,
            elapsed_s=time.time() - t0,
            best_similarity=best.similarity,
            min_similarity=sim_floor,
            explanation=explanation,
            explanation_details=explanation_details,
            best_bigram_similarity=best.bigram_similarity,
            best_fidelity=best.fidelity,
            fixed_patterns=fixed,
            remaining_patterns=remaining,
            copies_rejected=copies_rejected,
            best_is_copy=best.is_copy,
            chunks_total=len(sentences) if chunked else 0,
            chunks_kept_original=best.chunks_kept,
            polish_passes_used=polish_used,
        )
        logger.info(
            "HumanizationPipeline: done best=%.3f met=%s iters=%d (%.1fs)",
            result.best_score, result.criteria_met, result.iterations_used, result.elapsed_s,
        )
        return result

    def run_batch(self, raw_inputs: list[str], **kwargs) -> list[PipelineResult | BaseException]:
        out: list[PipelineResult | BaseException] = []
        for raw in raw_inputs:
            try:
                out.append(self.run(raw, **kwargs))
            except Exception as exc:  # isolation errors included per-item
                out.append(exc)
        return out


if __name__ == "__main__":  # manual smoke test (needs trained critic)
    print("HumanizationPipeline module OK. Run via: python main.py --mode infer --fast")
