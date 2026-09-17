#!/usr/bin/env python3
"""
Persistent humanize server — loads model once, then processes JSON lines on stdin.

Protocol:
  stdin:  {"text": "...", "effort": "standard", "model_id": "..."}
  stdout: {"ok": true, "score": 0.82, "sim": 0.71, "bigram": 0.65, "text": "...", "met": true}
          {"ok": false, "error": "..."}
  stderr: debug logs (ignored by client)
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import AppConfig, apply_light_mode
from main import build_runtime
from classifier import LocalPatternClassifier
from analyzer import PatternAnalyzer
from main import run_training
from utils import wrap_payload, is_near_copy

EFFORT_PRESETS = {
  "quick":    {"max_iters": 2, "num_candidates": 1, "polish_passes": 0, "min_similarity": 0.35},
  "standard": {"max_iters": 4, "num_candidates": 1, "polish_passes": 1, "min_similarity": 0.35},
  "deep":     {"max_iters": 6, "num_candidates": 2, "polish_passes": 2, "min_similarity": 0.35},
  "max":      {"max_iters": 8, "num_candidates": 2, "polish_passes": 2, "min_similarity": 0.30},
}

def log(*a):
    print(*a, file=sys.stderr, flush=True)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    args = ap.parse_args()

    import os
    os.environ.setdefault("HUMAIZE_FAST", "1")
    os.environ.setdefault("HUMAIZE_LIGHT", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    cfg = AppConfig.from_yaml(args.config) if Path(args.config).exists() else AppConfig()
    cfg = apply_light_mode(cfg)

    if args.model_id:
        low = args.model_id.lower()
        if any(t in low for t in ("7b","8b","13b","14b","70b")):
            cfg.generator.model_name = args.model_id
            use_fast = False
        else:
            cfg.generator.fallback_model_name = args.model_id
            use_fast = True
    else:
        use_fast = True

    # ---- load critic + pipeline ONCE ----
    log("loading critic…")
    cpath = Path(cfg.classifier.save_path)
    if cpath.exists():
        critic = LocalPatternClassifier.load(cpath, analyzer=None)
        try:
            cheap = len(__import__('analyzer', fromlist=['PatternAnalyzer']).PatternAnalyzer.CHEAP_FEATURE_NAMES)
            need_lm = critic._stat_dim != cheap and critic._stat_dim != 0
        except Exception:
            need_lm = False
        critic.analyzer = PatternAnalyzer(config=cfg.analyzer, load_model=need_lm)
    else:
        from main import run_training
        critic = run_training(cfg, None, use_fast)

    pipe = build_runtime(cfg, critic, use_fast)
    log("server ready")

    # ---- event loop ----
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"ok": False, "error": "bad json"}), flush=True)
            continue

        effort = req.get("effort", "standard")
        preset = EFFORT_PRESETS.get(effort, EFFORT_PRESETS["standard"])
        raw = req.get("text", "").strip()
        if not raw:
            print(json.dumps({"ok": False, "error": "empty text"}), flush=True)
            continue

        if "<input_text>" not in raw:
            from utils import wrap_payload
            raw = wrap_payload(raw)

        try:
            res = pipe.run(raw, target_score=cfg.pipeline.target_score,
                           max_iters=preset["max_iters"],
                           num_candidates=preset["num_candidates"],
                           min_similarity=preset["min_similarity"],
                           polish_passes=preset["polish_passes"])
        except Exception as e:
            print(json.dumps({"ok": False, "error": str(e)}), flush=True)
            continue

        from utils import is_near_copy
        payload = raw
        try:
            from utils import extract_payload
            payload = extract_payload(raw)
        except Exception:
            pass

        if is_near_copy(payload, res.best_text):
            print(json.dumps({"ok": False, "error": "model echoed input", "score": res.best_score, "sim": res.best_similarity}), flush=True)
            continue

        out = {
            "ok": True,
            "score": round(float(res.best_score), 3),
            "sim": round(float(res.best_similarity), 3),
            "bigram": round(float(res.best_bigram_similarity), 3),
            "text": res.best_text,
            "met": bool(res.criteria_met),
        }
        print(json.dumps(out, ensure_ascii=False), flush=True)

    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        pass