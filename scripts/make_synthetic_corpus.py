"""Generate a synthetic AI-labeled training corpus using ONLY local models.

Takes N local open-weights models x M prompts, generates continuations
offline (no APIs), and writes `data/synthetic.csv` with columns
text,label (+ source_model,prompt_id metadata) ready for:

    python main.py --mode train --data data/synthetic.csv

Human rows (label=1) are carried over from the built-in demo corpus
(`data_loader.DEMO_SAMPLES`) and/or an optional --human-data CSV, so the
output is already paired human vs AI.

Usage:
    python scripts/make_synthetic_corpus.py --models gpt2 Qwen/Qwen2.5-0.5B-Instruct --samples-per-prompt 2
    python scripts/make_synthetic_corpus.py --dry-run   # no model downloads; writes human rows only
"""
from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_loader import DEMO_SAMPLES, load_paired_dataset  # noqa: E402
from utils import set_seed, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

# Diverse, neutral prompt seeds — topics only, no style instructions that
# would collapse output diversity.
PROMPT_SEEDS: list[str] = [
    "Write a short paragraph about morning routines in a busy city.",
    "Explain why public libraries still matter.",
    "Describe a memorable train journey.",
    "Summarize the benefits of walking every day.",
    "Write a product review for a comfortable pair of shoes.",
    "Explain how photosynthesis works in simple terms.",
    "Describe your favorite meal and how it is prepared.",
    "Write a brief email apologizing for a late delivery.",
    "Discuss the pros and cons of remote work.",
    "Tell a short story about a lost dog finding its way home.",
    "Explain the water cycle to a ten-year-old.",
    "Write an opinion paragraph about social media and attention spans.",
    "Describe the atmosphere of a small-town market on Saturday.",
    "Summarize a recent book you enjoyed.",
    "Write instructions for brewing a good cup of tea.",
    "Discuss whether college degrees are still worth it.",
    "Describe a rainy afternoon from a window.",
    "Write a paragraph about the future of electric cars.",
    "Explain inflation in plain language.",
    "Tell a story about learning to swim.",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local multi-model synthetic corpus builder")
    p.add_argument("--models", nargs="*", default=["gpt2"],
                   help="Local HF model IDs (must be cached/downloadable once, then offline)")
    p.add_argument("--prompts", default=None, help="Text file with one prompt per line (else built-ins)")
    p.add_argument("--num-prompts", type=int, default=10, help="How many seed prompts to use")
    p.add_argument("--samples-per-prompt", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--output", default="data/synthetic.csv")
    p.add_argument("--human-data", default=None, help="Optional CSV/JSONL with extra human rows")
    p.add_argument("--include-demo-humans", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="Write human rows only (no generation)")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def load_prompts(path: str | None, num: int, seed: int) -> list[str]:
    if path:
        lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines()]
        prompts = [ln for ln in lines if ln]
    else:
        prompts = list(PROMPT_SEEDS)
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts[:max(num, 1)]


def generate_for_model(
    model_id: str,
    prompts: list[str],
    samples_per_prompt: int,
    max_new_tokens: int,
    seed: int,
    device: str | None,
) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from utils import resolve_device

    dev = resolve_device(device)
    logger.info("Loading local model '%s' on %s", model_id, dev)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True,
        torch_dtype=torch.float16 if dev == "cuda" else torch.float32,
    )
    mdl.to(dev)
    mdl.eval()

    rows: list[dict] = []
    rng = random.Random(seed)
    temps = [0.7, 0.9, 1.05, 1.2]
    pid = 0
    for prompt in prompts:
        for k in range(samples_per_prompt):
            t = temps[(pid + k) % len(temps)]
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(dev)
            with torch.no_grad():
                out = mdl.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=t, top_p=0.95, repetition_penalty=1.08,
                    pad_token_id=tok.eos_token_id,
                )
            text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            # strip prompt echo + whitespace collapse
            text = " ".join(text.split())
            if len(text) < 40:  # skip degenerate generations
                continue
            rows.append({"text": text, "label": 0, "source_model": model_id,
                         "prompt_id": pid, "temperature": t})
        pid += 1
    del mdl
    try:
        import torch as _t
        if dev == "cuda":
            _t.cuda.empty_cache()
    except Exception:
        pass
    logger.info("Model '%s': %d AI rows", model_id, len(rows))
    return rows


def main(argv: list[str] | None = None) -> int:
    setup_logging("INFO")
    args = parse_args(argv)
    set_seed(args.seed)
    prompts = load_prompts(args.prompts, args.num_prompts, args.seed)
    logger.info("%d prompts x %d samples x %d models", len(prompts),
                args.samples_per_prompt, len(args.models))

    rows: list[dict] = []
    # --- human rows (label 1): learned discriminator needs both classes ---
    if args.include_demo_humans:
        rows += [{"text": s.text, "label": 1, "source_model": "human-demo",
                  "prompt_id": -1, "temperature": ""} for s in DEMO_SAMPLES if s.label == 1]
    if args.human_data:
        for s in load_paired_dataset(args.human_data):
            if s.label == 1:
                rows.append({"text": s.text, "label": 1, "source_model": "human-file",
                             "prompt_id": -1, "temperature": ""})

    # --- AI rows (label 0) from local models ---
    if not args.dry_run:
        for mid in args.models:
            try:
                rows += generate_for_model(mid, prompts, args.samples_per_prompt,
                                           args.max_new_tokens, args.seed, args.device)
            except Exception as exc:
                logger.error("Skipping model '%s' (failed to load/generate: %s)", mid, exc)
    else:
        logger.info("dry-run: skipping generation")

    # dedupe on normalized text, keep first occurrence
    seen, uniq = set(), []
    for r in rows:
        key = " ".join(r["text"].lower().split())
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "source_model", "prompt_id", "temperature"])
        w.writeheader()
        w.writerows(uniq)
    n_ai = sum(1 for r in uniq if r["label"] == 0)
    n_hu = len(uniq) - n_ai
    logger.info("Wrote %s (%d AI / %d human, %d total)", out, n_ai, n_hu, len(uniq))
    print(f"Wrote {out} ({n_ai} AI / {n_hu} human). Train with: python main.py --mode train --data {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
