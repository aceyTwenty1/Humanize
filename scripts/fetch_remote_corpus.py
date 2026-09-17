"""Fetch AI-labeled training rows from REMOTE chat APIs (opt-in rule break).

Scope note: the runtime pipeline (analyzer -> rewriter -> critic) stays fully
local ML. This script ONLY collects offline training data in bulk from hosted
models you have API access to, then writes the same CSV schema as
scripts/make_synthetic_corpus.py:

    python scripts/fetch_remote_corpus.py --models <ids...> --num-prompts 10 --output data/remote.csv
    python main.py --mode train --data data/remote.csv

Provider: any OpenAI-compatible chat endpoint (default OpenRouter, which hosts
many free-tier models). Key via --api-key or OPENROUTER_API_KEY env var.
Uses ONLY stdlib (urllib) so no new dependencies.

Free-model IDs change often; override with --models. Defaults below were
valid OpenRouter :free slugs at time of writing — drop any that 404/429.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_loader import DEMO_SAMPLES, load_paired_dataset  # noqa: E402
from utils import set_seed, setup_logging  # noqa: E402

try:
    from scripts.make_synthetic_corpus import PROMPT_SEEDS
except ImportError:  # direct-script fallback
    PROMPT_SEEDS = ["Write a short paragraph about morning routines in a busy city."]

logger = logging.getLogger(__name__)

# 15 OpenRouter free-tier slugs (override with --models as availability shifts).
DEFAULT_MODELS: list[str] = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
    "qwen/qwen3-235b-a22b:free",
    "deepseek/deepseek-r1:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "microsoft/phi-4-reasoning:free",
    "qwen/qwen3-30b-a3b:free",
    "meta-llama/llama-4-maverick:free",
    "google/gemini-2.0-flash-exp:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "qwen/qwen2.5-vl-72b-instruct:free",
    "mistralai/mistral-nemo:free",
    "moonshotai/kimi-vl-a3b-thinking:free",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
    "tngtech/deepseek-r1t-chimera:free",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Remote multi-model corpus fetcher (training data only)")
    p.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    p.add_argument("--num-prompts", type=int, default=10)
    p.add_argument("--prompts", default=None, help="File with one prompt per line (else built-ins)")
    p.add_argument("--samples-per-prompt", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=180)
    p.add_argument("--output", default="data/remote.csv")
    p.add_argument("--human-data", default=None)
    p.add_argument("--include-demo-humans", action="store_true", default=True)
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1/chat/completions")
    p.add_argument("--api-key", default=None)
    p.add_argument("--referer", default="humaize-local", help="HTTP-Referer sent to OpenRouter")
    p.add_argument("--delay", type=float, default=2.0, help="Seconds between calls (rate-limit courtesy)")
    p.add_argument("--timeout", type=int, default=90)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="Write human rows only (no API calls)")
    return p.parse_args(argv)


def _post(url: str, key: str, referer: str, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}",
                 "HTTP-Referer": referer},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_one(base_url: str, key: str, referer: str, model: str, prompt: str,
              max_tokens: int, timeout: int, retries: int) -> str | None:
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.9}
    for attempt in range(retries + 1):
        try:
            data = _post(base_url, key, referer, payload, timeout)
            text = data["choices"][0]["message"]["content"].strip()
            return " ".join(text.split())
        except Exception as exc:
            logger.warning("model=%s attempt=%d failed: %s", model, attempt, exc)
            time.sleep(2 ** attempt * 2)
    return None


def main(argv: list[str] | None = None) -> int:
    setup_logging("INFO")
    args = parse_args(argv)
    set_seed(args.seed)
    key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not key and not args.dry_run:
        print("ERROR: no API key. Set OPENROUTER_API_KEY or pass --api-key (or use --dry-run).")
        return 2

    if args.prompts:
        lines = [ln.strip() for ln in Path(args.prompts).read_text(encoding="utf-8").splitlines()]
        seeds = [ln for ln in lines if ln]
    else:
        seeds = list(PROMPT_SEEDS)
    rng = random.Random(args.seed)
    rng.shuffle(seeds)
    prompts = seeds[:max(args.num_prompts, 1)]

    rows: list[dict] = []
    if args.include_demo_humans:
        rows += [{"text": s.text, "label": 1, "source_model": "human-demo",
                  "prompt_id": -1, "temperature": ""} for s in DEMO_SAMPLES if s.label == 1]
    if args.human_data:
        for s in load_paired_dataset(args.human_data):
            if s.label == 1:
                rows.append({"text": s.text, "label": 1, "source_model": "human-file",
                             "prompt_id": -1, "temperature": ""})

    if not args.dry_run:
        pid = 0
        for prompt in prompts:
            for model in args.models:
                for _ in range(args.samples_per_prompt):
                    text = fetch_one(args.base_url, key, args.referer, model, prompt,
                                     args.max_tokens, args.timeout, args.retries)
                    time.sleep(args.delay)
                    if text and len(text) >= 40:
                        rows.append({"text": text, "label": 0, "source_model": model,
                                     "prompt_id": pid, "temperature": 0.9})
            pid += 1
    else:
        logger.info("dry-run: no API calls")

    seen, uniq = set(), []
    for r in rows:
        k = " ".join(r["text"].lower().split())
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "source_model", "prompt_id", "temperature"])
        w.writeheader()
        w.writerows(uniq)
    n_ai = sum(1 for r in uniq if r["label"] == 0)
    print(f"Wrote {out} ({n_ai} AI / {len(uniq) - n_ai} human). Train: python main.py --mode train --data {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
