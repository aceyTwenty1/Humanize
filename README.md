# Humaize — Local Human-Like Rewriting (Yoga 9i Optimized)

Fully **local** ML pipeline that learns AI-writing patterns from your data and rewrites text to sound human. No cloud, no API keys, no hardcoded rules — pattern detection is **learned** (`TF-IDF + 30 stat features → LogReg/XGBoost → Human-Likeness ∈ [0,1]`).

Optimized for **Yoga 9i 15ITL (i7 EVO / Iris Xe / 16GB)** — runs in `~900MB` RAM, `~3s` per rewrite after warmup, entirely on CPU. First run downloads models once (~700MB), then forever offline.

## Basic Use — 3 steps

```bash
# 1. Install (light = Yoga profile)
pip install -r requirements-light.txt

# 2. Train once (demo data or your own — see below)
python main.py --mode train --fast --light
# or double-click: train.bat  (drag a folder/file onto it to train on your writing)

# 3. Humanize — pick one:
python cli.py --fast --light                          # chat (see Chat section)
python cli.py --fast --light --text "paste text here" # one-shot
python main.py --mode infer --fast --light --input "<input_text>Your text here.</input_text>"
echo "<input_text>Your text</input_text>" > in.txt && python main.py --mode infer --fast --light --input-file in.txt
```

Output shows: `score` vs `target 0.85`, `fidelity` (unigram/bigram), which **learned** patterns were fixed, and the rewrite (also copied to clipboard).

## How it works

```
<input_text> payload → isolate (fail-closed) → PatternAnalyzer (30 stats) → CriticBundle (Human-Likeness) → explain/guidance → TextRewriter (SmolLM2-360M, chat template) → GateSpec (sim 0.35 / copy 0.85·0.75 / bloat 1.8) → retry 1× → output
```

| Module | File | Role |
|---|---|---|
| Feature Extractor | `analyzer.py → PatternAnalyzer` | 30 LM-free stats (burstiness, entropy, repetition, passive, clichés, hedges/boosters, ?/!/— , FK grade) — 34 with optional LM |
| Critic Bundle | `classifier.py → CriticBundle` | Paired `(critic, analyzer)` with matched `stat_dim`. Single `load(path)` — no caller guesses `--fast`. `explain()` gives raise/lower directions |
| Gate Spec | `utils.GateSpec` via `config.PipelineConfig` | The triple gate: `min_similarity`, `is_near_copy`, `MAX_LENGTH_RATIO` — one spec, all gates read it |
| Resource Profile | `config.apply_light_mode → ResourceProfile` | Single owner of "what light/Yoga means" — pure, testable caps (see Yoga section) |
| Generator | `generator.py → TextRewriter` | Local `SmolLM2-360M` (~700MB, Iris Xe optional via OpenVINO) or `Qwen-7B 4-bit` on GPU; `RewriteSanitizer` 5-step chain |
| Runtime | `pipeline.py → HumanizationPipeline` | Isolation + explain → targeted generate → GateSpec validate + sentence-chunk for long inputs |
| Entry point | `main.py` | `train` / `infer` / `full` (+ RL) |

Deep modules: `CriticBundle`, `GateSpec`, `ResourceProfile`, `RewriteSanitizer` — see `CONTEXT.md`.

## Payload isolation

Input **must** be `<input_text>...</input_text>`. Only inner text reaches models; outer text is discarded (0 or 2+ blocks → error). Prompt also tells the model tagged text is DATA.

## Yoga 9i profile (`--light`)

`HUMAIZE_LIGHT=1` or `--light` activates the pure `ResourceProfile`:

| Cap | Normal | Yoga Ultra (`--light`) |
|---|---|---|
| Analyzer window | 1024 | **128** (no LM) |
| TF-IDF | 5000 + xgboost | **800 + logreg** |
| Generator prompt | 6000 chars / 256 tok | **2000 / 64 tok** |
| Pipeline | 4 iters + 2 polish | **1 iter / 0 polish**, greedy first try |
| Threads | auto | **2** |
| Peak RAM | ~2.5GB (7B) | **~0.9GB** |
| Speed (360M) | ~8s | **~3s after warmup** |

Pure function — `apply_light_mode` returns a new `AppConfig`, never mutates. Long docs still sentence-chunk at 40 words.

Optional Iris Xe boost (2-3×, same seam):
```bash
pip install optimum-intel openvino --quiet  # auto-detected on next load
```

## Chat (Kilo-style)

```bash
python cli.py --fast --light            # menus on start
python cli.py --fast --light --effort deep  # quick/standard/deep/max
python cli.py --fast --light --model-id HuggingFaceTB/SmolLM2-1.7B-Instruct
```

Inside: `/effort`, `/model`, `/target 0.85`, `/iters`, `/sim`, `/polish`, `/analyze <text>`, `/save [path]`, `/status`, `/m` multiline. Models reload in place, critic kept. `chat.bat` = same auto `--fast --light`.

`--light` on 360M is instruction-tuned — raw `gpt2` would ramble.

## Documents with tables

Via `dochumanize.py` (stdlib `.docx`, no deps):

- Body paras → full pipeline one by one.
- Table cells → line-batched (one generation per batch, per-line `GateSpec` check + individual fallback).
- Short labels/numbers/dates/headings kept as-is; nested tables + `w:sdt` wrappers handled.
- `cli.py --input-file doc.txt` or `python -c "from dochumanize import parse_docx, humanize_document, write_docx; ..."`

Output: `<name>_humanized.docx` next to original. PDFs → convert to `.docx` first.

## Training on your own writing

No CSV needed — point at your files:
```bash
python main.py --mode train --fast --light --human-dir my_writing/   # folder or .txt/.md
python main.py --mode train --fast --light --human-dir my/ --split-paragraphs  # one doc → many samples
train.bat  # drag folder onto it
```

| Flag | Effect |
|---|---|
| `--ai-dir ai/` | AI samples (label 0) |
| `--human-dir hw/` | Your writing (label 1) + built-in AI contrast |
| `--split-paragraphs` | Split files on blank lines |
| `--no-demo` | Use only your data |
| `--data pairs.csv` | Classic `text,label` CSV/JSONL (combines) |

15+ varied samples per side beats 100 identical ones.

## Temp internet corpus

```bash
python main.py --mode train --fast --light --fetch-human 60
```

Gutenberg public-domain → temp dir → auto-deleted if `acc ≥ 0.7` (`--fetch-min-acc`), else kept + path printed. `--fetch-keep` keeps always. Standalone: `python fetch_corpus.py --out data/tmp --samples 60`

## Production (GPU, 7B 4-bit)

```bash
python main.py --mode full --data data/pairs.csv          # no --fast/--light
python main.py --mode infer --input-file input.txt
```
Needs `bitsandbytes` + CUDA. `HUMAIZE_FAST=1` forces fallback to 360M.

## MLP critic

```bash
python main.py --mode train --fast --light --model-type mlp
```
`mlp_hidden_layers` / `mlp_max_iter` in `config.yaml`. Trees win <100 samples; MLP needs more data.

## Longer training

```bash
python main.py --mode train --fast --light --trees 500 --epochs 3
```
`--trees` = boosting rounds, `--epochs` = RL passes (RL skipped under `--fast` on CPU). Demo memorizes — grow corpus first. `train_long.bat` = 10-min Yoga session (400 samples + MLP + RL).

## TUI (optional)

```bash
cd tui && npm install && npm run dev -- --no-wizard --effort quick
# or: tui\bridge.py / server.py for Node ↔ Python IPC
```

## Tests

```bash
python -m pytest tests/ -q          # 41 tests, ~20s
HUMAIZE_FAST=1 python analyzer.py   # smoke
HUMAIZE_FAST=1 python classifier.py
```
