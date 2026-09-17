#!/usr/bin/env python3
"""Bridge for the Ink TUI — one-shot humanize via the real pipeline.

Called as:  python tui/bridge.py --effort standard --text "your text"
Prints single JSON line: {"ok":true,"score":0.82,"sim":0.71,"text":"..."}
or {"ok":false,"error":"..."}.

Loads the critic+generator on first call via existing Python stack
(cli.py / main.py pattern). No new deps. Keep --fast --light for CPU.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import AppConfig, apply_light_mode
from main import build_runtime  # noqa: E402
from classifier import LocalPatternClassifier  # noqa: E402
from analyzer import PatternAnalyzer  # noqa: E402
from main import run_training  # noqa: E402
from utils import wrap_payload  # noqa: E402

EFFORT_PRESETS = {
  "quick":    {"max_iters": 2, "num_candidates": 1, "polish_passes": 0, "min_similarity": 0.35},
  "standard": {"max_iters": 4, "num_candidates": 1, "polish_passes": 1, "min_similarity": 0.35},
  "deep":     {"max_iters": 6, "num_candidates": 2, "polish_passes": 2, "min_similarity": 0.35},
  "max":      {"max_iters": 8, "num_candidates": 2, "polish_passes": 2, "min_similarity": 0.30},
}

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--effort", default="standard", choices=list(EFFORT_PRESETS))
  ap.add_argument("--model-id", default=None)
  ap.add_argument("--text", required=True)
  ap.add_argument("--config", default=str(ROOT / "config.yaml"))
  args = ap.parse_args()

  cfg = AppConfig.from_yaml(args.config) if Path(args.config).exists() else AppConfig()
  preset = EFFORT_PRESETS[args.effort]
  # keep parity with tui default path: fast+light
  import os
  os.environ.setdefault("HUMAIZE_FAST","1")
  os.environ.setdefault("HUMAIZE_LIGHT","1")
  os.environ.setdefault("OMP_NUM_THREADS","2")
  os.environ.setdefault("MKL_NUM_THREADS","2")
  os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
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

  # load or train critic (aligned to artifact dims)
  cpath = Path(cfg.classifier.save_path)
  if cpath.exists():
    critic = LocalPatternClassifier.load(cpath, analyzer=None)
    try:
      cheap = len(PatternAnalyzer.CHEAP_FEATURE_NAMES)
      need_lm = critic._stat_dim != cheap and critic._stat_dim != 0
    except Exception:
      need_lm = False
    critic.analyzer = PatternAnalyzer(config=cfg.analyzer, load_model=need_lm)
  else:
    critic = run_training(cfg, None, use_fast)

  pipe = build_runtime(cfg, critic, use_fast)

  raw = args.text.strip()
  if "<input_text>" not in raw:
    raw = wrap_payload(raw)

  try:
    res = pipe.run(raw, target_score=cfg.pipeline.target_score,
                   max_iters=preset["max_iters"],
                   num_candidates=preset["num_candidates"],
                   min_similarity=preset["min_similarity"],
                   polish_passes=preset["polish_passes"])
  except Exception as e:
    print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
    return 0

  # guard: never return input verbatim as "rewrite"
  from utils import is_near_copy, lexical_similarity
  payload = raw
  try:
    from utils import extract_payload
    payload = extract_payload(raw)
  except Exception:
    pass

  if is_near_copy(payload, res.best_text):
    print(json.dumps({"ok": False, "error": "model echoed input — try longer text or /effort deep", "score": res.best_score, "sim": res.best_similarity}, ensure_ascii=False))
    return 0

  out = {
    "ok": True,
    "score": round(float(res.best_score), 3),
    "sim": round(float(res.best_similarity), 3),
    "bigram": round(float(res.best_bigram_similarity), 3),
    "text": res.best_text,
    "met": bool(res.criteria_met),
  }
  print(json.dumps(out, ensure_ascii=False))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
