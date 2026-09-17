#!/usr/bin/env python3
"""
Humanize quality & speed benchmark.

Usage:
  python bench.py                    # runs default suite
  python bench.py --quick            # fast sanity check
  python bench.py --text "your text" --effort standard
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import AppConfig, apply_light_mode
from main import build_runtime
from classifier import LocalPatternClassifier
from analyzer import PatternAnalyzer
from main import run_training
from utils import wrap_payload, is_near_copy, lexical_similarity

EFFORT_PRESETS = {
  "quick":    {"max_iters": 2, "num_candidates": 1, "polish_passes": 0, "min_similarity": 0.35},
  "standard": {"max_iters": 4, "num_candidates": 1, "polish_passes": 1, "min_similarity": 0.35},
  "deep":     {"max_iters": 6, "num_candidates": 2, "polish_passes": 2, "min_similarity": 0.35},
  "max":      {"max_iters": 8, "num_candidates": 2, "polish_passes": 2, "min_similarity": 0.30},
}

TEST_CASES = [
    ("ai_formal", "In conclusion, it is important to note that effective communication is essential for organizational success. Furthermore, robust methodologies facilitate optimization."),
    ("ai_listicle", "This article explores five key strategies. First, prioritize tasks. Second, delegate effectively. Third, monitor progress. Fourth, evaluate outcomes. Fifth, iterate continuously."),
    ("ai_speculative", "AI will completely revolutionize healthcare by leveraging vast amounts of data and advanced computational paradigms."),
    ("ai_corporate", "It is widely acknowledged that artificial intelligence is transforming industries. Moreover, leveraging synergies across departments fosters a culture of innovation."),
    ("human_casual", "I dunno — yesterday's meeting ran way over, and honestly half of it could've been an email."),
    ("human_story", "We hiked up before sunrise, freezing, couldn't feel our fingers. But then the light hit the valley and nobody said anything for a while. Worth it."),
    ("human_opinion", "Look, I'm no expert on taxes, but last year I messed up the filing and it took three calls and a lot of coffee to sort out."),
]

def load_pipeline(effort: str, model_id: str | None, fast: bool, light: bool):
    from config import AppConfig, apply_light_mode
    from main import build_runtime
    from classifier import LocalPatternClassifier
    from analyzer import PatternAnalyzer
    from main import run_training

    cfg = AppConfig()
    if light:
        from config import apply_light_mode
        cfg = apply_light_mode(cfg)

    preset = EFFORT_PRESETS[effort]
    cpath = Path("artifacts/classifier.joblib")
    if cpath.exists():
        critic = LocalPatternClassifier.load(cpath, analyzer=None)
        try:
            from analyzer import PatternAnalyzer as PA
            cheap = len(PA.CHEAP_FEATURE_NAMES)
            need_lm = critic._stat_dim != cheap and critic._stat_dim != 0
        except Exception:
            need_lm = False
        critic.analyzer = PatternAnalyzer(config=cfg.analyzer, load_model=need_lm)
    else:
        critic = run_training(cfg, None, fast)

    pipe = build_runtime(cfg, critic, fast, analyzer_load_model=need_lm)
    return pipe, preset, cfg.pipeline.target_score

def run_one(pipe, text: str, preset: dict, target: float) -> dict:
    raw = text if "<input_text>" in text else wrap_payload(text)
    t0 = time.time()
    res = pipe.run(raw, target_score=target, **preset)
    dt = time.time() - t0
    payload = text
    try:
        from utils import extract_payload
        payload = extract_payload(raw)
    except Exception:
        pass

    return {
        "time_sec": round(dt, 2),
        "score": round(float(res.best_score), 3),
        "sim": round(float(res.best_similarity), 3),
        "bigram": round(float(res.best_bigram_similarity), 3),
        "met": bool(res.criteria_met),
        "is_copy": is_near_copy(payload, res.best_text),
        "text": res.best_text[:120] + ("…" if len(res.best_text) > 120 else ""),
    }

def main():
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--effort", default="standard", choices=list(EFFORT_PRESETS))
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--text", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fast", action="store_true", default=True)
    ap.add_argument("--light", action="store_true", default=True)
    args = ap.parse_args()

    os.environ.setdefault("HUMAIZE_FAST", "1" if args.fast else "0")
    os.environ.setdefault("HUMAIZE_LIGHT", "1" if args.light else "0")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    fast = args.fast
    light = args.light

    print(f"Loading pipeline (effort={args.effort}, fast={fast}, light={light})…")
    t_load = time.time()
    pipe, preset, target = load_pipeline(args.effort, args.model_id, fast, light)
    print(f"Pipeline loaded in {time.time() - t_load:.1f}s")

    cases = [("custom", args.text)] if args.text else TEST_CASES
    if args.quick:
        cases = cases[:2]

    print(f"\n{'case':<15} {'time':>6} {'score':>6} {'sim':>6} {'bigram':>7} {'met':>4} {'copy':>4}  output")
    print("-" * 100)

    totals = {"time": 0.0, "score": 0.0, "sim": 0.0, "bigram": 0.0, "met": 0, "copies": 0}
    for name, text in cases:
        if not text:
            continue
        r = run_one(pipe, text, preset, target)
        totals["time"] += r["time_sec"]
        totals["score"] += r["score"]
        totals["sim"] += r["sim"]
        totals["bigram"] += r["bigram"]
        totals["met"] += 1 if r["met"] else 0
        totals["copies"] += 1 if r["is_copy"] else 0
        print(f"{name:<15} {r['time_sec']:>6.1f}s {r['score']:>6.3f} {r['sim']:>6.3f} {r['bigram']:>7.3f} {str(r['met']):>4} {str(r['is_copy']):>4}  {r['text']}")

    n = len(cases)
    print("-" * 100)
    print(f"{'AVG':<15} {totals['time']/n:>6.1f}s {totals['score']/n:>6.3f} {totals['sim']/n:>6.3f} {totals['bigram']/n:>7.3f} {totals['met']/n*100:>3.0f}% {totals['copies']/n*100:>3.0f}%")

if __name__ == "__main__":
    main()