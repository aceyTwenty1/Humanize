# Humaize — Autonomous Local ML System for Human-Like Rewriting

Fully **local** pipeline (no external APIs / GPT dependencies) that learns AI-writing
patterns from data and rewrites text to sound human. Pattern detection is **learned**
(TF-IDF + statistical features → gradient-boosted discriminator used as an RL reward
model), never a hardcoded rule list.

## Architecture

| Module | File | Role |
|---|---|---|
| Feature Extractor | `analyzer.py` → `PatternAnalyzer` | LM perplexity / log-probs + surprise variance (local HF model) + burstiness (var/CV/range), n-gram entropy, repetition rate, passive density, vocab predictability, word/punctuation shape, detector-inspired signals (nominalizations, transitions, intensifiers, openers, hyphen/caps density) plus second-wave traces: AI clichés, hedges/boosters, pronoun voice, ?/!/— rhythm, enumerations, readability grade (34 features, 30 LM-free) |
| Classifier / Reward | `classifier.py` → `LocalPatternClassifier` | XGBoost (fallback HistGB/LogReg) on TF-IDF + stats → Human-Likeness ∈ [0,1]; RL reward model; `explain()` attributes verdicts to learned class centroids + token weights with raise/lower direction, `details()` breakdown, `global_importances()` corpus-wide signals |
| Generator | `generator.py` → `TextRewriter` | Local Qwen-2.5-7B-Instruct / Mistral-7B in 4-bit QLoRA (`bitsandbytes`+`peft`); CPU fallback SmolLM2-360M-Instruct via chat templates; accepts learned-pattern `guidance` for targeted edits |
| Adversarial RL | `train_rl.py` → `AdversarialTrainer` | TRL PPO / DPO: Actor generates → Critic scores → policy update; offline fallbacks included |
| Runtime | `pipeline.py` → `HumanizationPipeline` | `<input_text>` isolation + explain → targeted generate → validate (score ≥ target AND input-overlap ≥ floor, echoes/padding rejected) + adaptive resampling loop + polish passes on the winner; long inputs rewrite sentence-by-sentence; reports unigram/bigram/composite fidelity plus fixed-vs-remaining pattern report |
| Entry point | `main.py` | End-to-end train + RL + inference |

Supporting: `config.py` + `config.yaml`, `data_loader.py`, `utils.py` (isolation, `SamplingParams` schedule).

## Payload isolation

Runtime input **must** be wrapped: `<input_text>...</input_text>`. Only the inner
payload reaches the models; outer text is discarded (fail-closed on 0 or 2+ blocks).
The rewrite prompt additionally instructs the model to treat tagged content as DATA.

## Quickstart (offline demo, CPU-safe)

```bash
pip install -r requirements.txt
python main.py --mode full --fast
python main.py --mode infer --fast --input "<input_text>Your text here.</input_text>"
```

## Lightweight mode (less RAM/CPU/disk, slower)

```bash
pip install -r requirements-light.txt   # skips GPU/RL stack
python main.py --mode train --fast --light
python cli.py --fast --light
```

## Desktop GUI (claymorphism)

```bash
gui.bat                                  # double-click: pastel puffy UI
python gui.py --fast --light --effort deep
```

Tkinter only — no new dependencies, fully offline. Input card, effort
chips (Quick/Standard/Deep/Max), model dropdown (reloads live, critic
kept), goal slider, one big HUMANIZE button, score pill, rewrite + pattern
breakdown. Heavy work runs in threads so the window never freezes; first
run downloads the model once, then offline.

### Documents with tables (GUI)

**Upload doc** accepts `.txt`/`.md` (loaded into the input box) and
`.docx`. For Word files, **Humanize file** rewrites the content while the
tables keep their grid, widths, and styles byte-for-byte:

- Body paragraphs go through the full pipeline one by one.
- Table cells are rewritten in line-batches (one generation per batch,
  per-line validation, individual fallback) — a 20-cell table costs a few
  generations, not twenty.
- Short labels, numbers, dates, and headings are kept as-is (nothing to
  de-humanize, everything to break); tables nested in cells are processed
  as their own blocks, and content in Word's structured-document wrappers
  is found too.
- Run-level formatting inside a rewritten paragraph flattens to plain
  text; paragraph-level formatting is preserved.

Output saves next to the original as `<name>_humanized.docx`, with a
per-unit report (before → after scores) in the details pane. Logic lives
in `dochumanize.py` (stdlib-only `.docx` handling); PDFs need converting
to `.docx` first.

## Chat (Kilo-style selectors)
```bash
python cli.py --fast --light            # interactive chat, menus on start
python cli.py --fast --effort deep      # skip menu: deep search preset
python cli.py --fast --model-id HuggingFaceTB/SmolLM2-1.7B-Instruct
```

On start the chat asks for an **effort level** (`quick` → `max`: controls
retries, candidates/iter, polish passes, similarity floor) and a **model**
(SmolLM2-360M/1.7B on CPU, Qwen/Mistral-7B on GPU). Inside the chat,
`/effort [name]` and `/model [n|hf-id]` switch live — models reload in
place, the critic is kept. Manual `/iters`/`/num`/`/sim`/`/polish` tweaks
mark effort as `custom` (shown in `/status`).

`--light` switches the fallback generator to SmolLM2-360M-Instruct (~700MB,
instruction-tuned — raw base models like gpt2 can't follow rewrite instructions
and ramble off-topic), caps the analyzer window (512), TF-IDF (2000 features)
and RL batches (1), limits CPU threads, and enables gradient checkpointing in
RL training.

## Production (GPU, 7B 4-bit)

```bash
python main.py --mode full --data data/pairs.csv
python main.py --mode train --data data/pairs.csv
python main.py --mode infer --input-file input.txt
```

Dataset format (CSV or JSONL): `text,label` with `1=human, 0=AI`.

## Training on your own writing

Point the trainer at your own text — no CSV wrangling. One file or a folder
of `.txt`/`.md` files; each file becomes one sample:

```bash
python main.py --mode train --fast --human-dir my_writing/
```

Your files train as human (label 1), paired with the built-in AI samples as
contrast — so both classes exist with zero extra work. Options:

| Flag | Effect |
|---|---|
| `--ai-dir ai_samples/` | AI examples with label 0 (instead of / alongside built-ins) |
| `--split-paragraphs` | Split each file on blank lines → many samples per document |
| `--no-demo` | Exclude the built-in corpus (use only your data) |
| `--data pairs.csv` | Classic CSV/JSONL path (combines with the above) |

Tip: 15+ samples per side with varied topics beats 100 near-identical ones —
the critic learns *your* voice against AI patterns, and `explain()` then
targets rewrites at exactly what makes new text sound unlike you.

## Temp internet corpus (fetch → train → auto-delete)

Need human examples fast? Pull public-domain books (Project Gutenberg,
clean license) into a temp dir for one training run:

```bash
python main.py --mode train --fast --fetch-human 60
```

This downloads ~60 paragraphs, trains, and **deletes the temp files when
accuracy clears `--fetch-min-acc`** (default 0.7). If the score falls short,
the files are kept and their path printed so you can inspect or retry.
`--fetch-keep` always keeps them; `--book-ids 1342,11` picks specific books;
`python fetch_corpus.py --out data/temp_human --samples 60` fetches standalone.

Heads-up: classics are 19th-century prose — good for generic patterns, but
pair with `--human-dir` (your modern voice) for best results.

## Neural-net critic (MLP)

Beyond trees (`xgboost`/`histgb`) and linear (`logreg`), the critic offers a
feed-forward neural network:

```bash
python main.py --mode train --fast --model-type mlp
```

Tune via `classifier.mlp_hidden_layers` / `mlp_max_iter` in `config.yaml`.
Rule of thumb: trees win under ~100 samples; the MLP pulls ahead with more
data. Token-level evidence (`explain()` word hits, `global_importances()`)
is unavailable for the MLP — per-feature directions still work.

## Longer training

Two knobs (flags or `config.yaml`):

```bash
python main.py --mode train --fast --trees 500 --epochs 3
```

- `--trees N` — boosting rounds for the `xgboost`/`histgb` critic
  (`classifier.xgb_estimators` / `gb_max_iter`). More rounds = finer
  patterns, but on the 20-sample demo it just memorizes — longer critic
  training pays off once `--human-dir` / `--fetch-human` grows the corpus.
- `--epochs N` — adversarial RL passes over the prompts (`rl.num_epochs`,
  default 1). Note RL weight updates are skipped under `--fast` on CPU, so
  longer RL matters in full (GPU) mode; on CPU, longer *critic* training is
  where the gains are.

## Tests

```bash
python -m pytest tests/ -q
HUMAIZE_FAST=1 python analyzer.py
HUMAIZE_FAST=1 python classifier.py
```
