"""Humaize chat CLI — Kilo Code-style interactive terminal.

A chat-sidebar-like REPL for the local humanizer: you type/paste text, it
streams the pipeline steps (analyze -> generate -> score -> retry) and
renders the rewrite with a score bar and feature deltas.

    python cli.py --fast                 # interactive chat (CPU-safe models)
    python cli.py --fast --text " paste text here "
    python cli.py --fast --input-file input.txt

Slash commands (inside chat):
    /help              this help            /target 0.85       goal score
    /iters 4           max retries              /num 2        candidates/iter
    /sim 0.35          min input overlap        /analyze <text>    score only
    /save [path]       save rewrite             /model             backend status
    /quit              exit
Multiline: type /m, paste lines, end with a lone '.' line.
Bare text (no tags) is auto-wrapped in <input_text>; tagged input is used
as-is to preserve the isolation contract.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None  # type: ignore

from config import AppConfig, apply_light_mode  # noqa: E402
from main import build_runtime, run_training  # noqa: E402
from classifier import LocalPatternClassifier  # noqa: E402
from analyzer import PatternAnalyzer  # noqa: E402
from utils import extract_payload, wrap_payload, PayloadIsolationError, copy_to_clipboard  # noqa: E402

HELP = """\
/help              this help        /quit, /q, exit     leave chat
/effort [name]     effort preset: quick|standard|deep|max (menu if empty)
/model [n|id]      switch generator model (menu if empty)
/target 0.85       goal score       /iters 4            max retries
/num 2             candidates/iter  /sim 0.35           min input overlap
/polish 2          refinement passes on winner (0=off)
/analyze <text>    score only       /save [path]        save rewrite
/status            backend status   /m                  multiline input
Just type text to humanize it. <input_text> tags optional (auto-added)."""


EFFORT_PRESETS = {
    "quick": {"max_iters": 2, "num_candidates": 1, "polish_passes": 0, "min_similarity": 0.35,
              "desc": "1 draft + 1 retry, no polish — fastest"},
    "standard": {"max_iters": 4, "num_candidates": 1, "polish_passes": 1, "min_similarity": 0.35,
                 "desc": "balanced search + 1 polish pass (default)"},
    "deep": {"max_iters": 6, "num_candidates": 2, "polish_passes": 2, "min_similarity": 0.35,
             "desc": "wide search, 2 candidates/iter + 2 polish passes"},
    "max": {"max_iters": 8, "num_candidates": 2, "polish_passes": 2, "min_similarity": 0.30,
            "desc": "everything: 8 iters, relaxed similarity floor for hard inputs"},
}

LOCAL_MODELS = [
    {"id": "HuggingFaceTB/SmolLM2-360M-Instruct", "label": "SmolLM2-360M — CPU, fast, light (default)",
     "kind": "small"},
    {"id": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "label": "SmolLM2-1.7B — CPU, slower, better rewrites",
     "kind": "small"},
    {"id": "Qwen/Qwen2.5-7B-Instruct", "label": "Qwen2.5-7B — GPU 4-bit, best quality",
     "kind": "big"},
    {"id": "mistralai/Mistral-7B-Instruct-v0.3", "label": "Mistral-7B — GPU 4-bit, alternative",
     "kind": "big"},
]


def resolve_effort(name: str) -> dict:
    """Preset values for an effort level (raises ValueError on unknown)."""
    key = (name or "").strip().lower()
    if key not in EFFORT_PRESETS:
        raise ValueError(f"Unknown effort '{name}' (use: {', '.join(EFFORT_PRESETS)})")
    return EFFORT_PRESETS[key]


def guess_model_kind(model_id: str) -> str:
    """Heuristic: 7B+ models need the GPU path, small ones run on CPU."""
    low = model_id.lower()
    return "big" if any(t in low for t in ("7b", "8b", "13b", "14b", "70b")) else "small"


def choose(title: str, options: list[str], default: int = 0) -> int | None:
    """Numbered menu (Enter = default, q = cancel). Returns index or None."""
    _print(f"{title} (Enter = default [{default + 1}])", "cyan")
    for i, opt in enumerate(options):
        _print(f"  {i + 1}) {opt}")
    try:
        raw = input("select › ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None
    if raw in ("",):
        return default
    if raw in ("q", "quit", "exit", "cancel"):
        return None
    try:
        idx = int(raw) - 1
        return idx if 0 <= idx < len(options) else None
    except ValueError:
        return None


def _print(msg: str = "", style: str = "") -> None:
    if HAS_RICH:
        console.print(msg, style=style, markup=False, highlight=False)
    else:
        print(msg)


def _rule(title: str = "") -> None:
    if HAS_RICH:
        console.rule(title)
    else:
        print(f"--- {title} ---")


def _bubble(role: str, text: str) -> None:
    if HAS_RICH:
        title = "you" if role == "you" else "humaize"
        border = "cyan" if role == "you" else "green"
        console.print(Panel(text, title=title, border_style=border, expand=False))
    else:
        print(f"[{role}] {text}")


def score_bar(score: float, target: float, width: int = 24) -> str:
    fill = max(0, min(width, round(score * width)))
    bar = "█" * fill + "░" * (width - fill)
    mark = "✓ PASS" if score >= target else "✗ retry"
    return f"[{bar}] {score:.3f} / {target:.2f}  {mark}"


def feature_deltas(orig: dict, new: dict | None) -> str:
    keys = ["perplexity", "logprob_std", "burstiness_var", "sentence_len_range",
            "sentence_opener_diversity", "bigram_entropy", "repetition_rate",
            "passive_density", "nominalization_rate", "transition_rate",
            "intensifier_rate", "ai_cliche_rate", "hedge_rate", "booster_rate",
            "lexical_diversity", "mean_word_len", "flesch_kincaid_grade",
            "contraction_rate", "punctuation_density"]
    lines = []
    for k in keys:
        if k not in orig and (not new or k not in new):
            continue
        a = orig.get(k, float("nan"))
        b = (new or {}).get(k, float("nan"))
        try:
            line = f"  {k:<26} {a:>9.3f}  →  {b:>9.3f}"
        except (TypeError, ValueError):
            line = f"  {k:<18} {a}  →  {b}"
        lines.append(line)
    return "\n".join(lines)


class Chat:
    def __init__(self, cfg: AppConfig, fast: bool,
                 effort: str | None = None, model_id: str | None = None,
                 prompt_select: bool = False) -> None:
        self.cfg = cfg
        self.fast = fast
        self.target = cfg.pipeline.target_score
        self.max_iters = cfg.pipeline.max_iters
        self.num = cfg.pipeline.num_candidates
        self.min_sim = cfg.pipeline.min_similarity
        self.polish = cfg.pipeline.polish_passes
        self.effort = "custom"
        self.model_label = ""
        if effort:
            resolve_effort(effort)  # validate; values already applied to cfg by main()
            self.effort = effort.strip().lower()
        if model_id:
            self._apply_model_id(model_id)
        self.last_rewrite = ""
        _print("loading local critic + models (first run downloads once, then offline)...", "yellow")
        cpath = Path(cfg.classifier.save_path)
        if cpath.exists():
            critic = LocalPatternClassifier.load(cpath, analyzer=None)
            # Match analyzer to the artifact's expected stat dims: the demo
            # critic was trained LM-free (30 dims); a full critic has 34.
            # Mismatch would crash the next score() call, so align instead of
            # trusting the --fast flag blindly.
            try:
                cheap = len(PatternAnalyzer.CHEAP_FEATURE_NAMES)
                need_lm = critic._stat_dim != cheap and critic._stat_dim != 0
            except Exception:
                need_lm = not fast
            critic.analyzer = PatternAnalyzer(config=cfg.analyzer, load_model=need_lm)
            if need_lm and not critic.analyzer.has_lm:
                _print("full LM requested by the saved critic but not available — "
                       "retrain with matching --fast setting fixes this", "yellow")
        else:
            _print("no saved critic — quick training on demo corpus...", "yellow")
            critic = run_training(cfg, None, fast)
        self.pipe = build_runtime(cfg, critic, fast)
        if not self.model_label:
            self.model_label = getattr(self.pipe.rewriter, "active_model_name", "?") or "?"
        if prompt_select:
            self.interactive_setup()
        _print("ready.", "green")

    # ---------------------------------------------------------- effort/model
    def set_effort(self, name: str, silent: bool = False) -> None:
        preset = resolve_effort(name)
        self.max_iters = preset["max_iters"]
        self.num = preset["num_candidates"]
        self.polish = preset["polish_passes"]
        self.min_sim = preset["min_similarity"]
        self.effort = name.strip().lower()
        if not silent:
            _print(f"effort={self.effort}: iters={self.max_iters} cand={self.num} "
                   f"polish={self.polish} min_sim={self.min_sim} — {preset['desc']}", "green")

    def _apply_model_id(self, model_id: str) -> None:
        kind = guess_model_kind(model_id)
        if kind == "big":
            self.cfg.generator.model_name = model_id
            self.fast = False
        else:
            self.cfg.generator.fallback_model_name = model_id
            self.fast = True
        self.model_label = model_id

    def set_model(self, model_id: str) -> None:
        """Switch generator model live (reloads; first use downloads once)."""
        known = {m["id"]: m for m in LOCAL_MODELS}
        if model_id not in known:
            _print(f"custom model id — guessing requirements from the name.", "yellow")
        kind = known[model_id]["kind"] if model_id in known else guess_model_kind(model_id)
        if kind == "big":
            _print("big model selected: needs a CUDA GPU + model download; "
                   "on CPU-only machines this will be very slow or fail.", "yellow")
        _print(f"loading {model_id} ...", "yellow")
        self._apply_model_id(model_id)
        self.pipe = build_runtime(self.cfg, self.pipe.critic, self.fast)
        _print(f"model={self.pipe.rewriter.active_model_name}", "green")

    def effort_menu(self) -> None:
        names = list(EFFORT_PRESETS)
        idx = choose("Effort level", [f"{n:<8} — {EFFORT_PRESETS[n]['desc']}" for n in names],
                     default=names.index("standard"))
        if idx is not None:
            self.set_effort(names[idx])

    def model_menu(self) -> None:
        ids = [m["id"] for m in LOCAL_MODELS]
        try:
            current = ids.index(self.pipe.rewriter.active_model_name)
        except ValueError:
            current = 0
        idx = choose("Generator model (downloads once, then offline)",
                     [m["label"] for m in LOCAL_MODELS], default=current)
        if idx is not None and ids[idx] != getattr(self.pipe.rewriter, "active_model_name", ""):
            self.set_model(ids[idx])
        elif idx is not None:
            _print("already on that model.", "dim")

    def interactive_setup(self) -> None:
        _rule("setup")
        self.effort_menu()
        self.model_menu()
        _print("")

    def status(self) -> str:
        rw = self.pipe.rewriter
        model = getattr(rw, "active_model_name", "?") or "?"
        return (f"model={model} critic={self.pipe.critic.backend} effort={self.effort} "
                f"target={self.target} iters={self.max_iters} cand={self.num} "
                f"polish={self.polish} min_sim={self.min_sim}")

    def do_analyze(self, raw: str) -> None:
        try:
            payload = extract_payload(raw, self.cfg.pipeline.max_chars)
        except PayloadIsolationError:
            payload = raw.strip()
            _print("(no <input_text> tags — scoring raw text as-is)", "yellow")
        a = self.pipe.analyzer.analyze(payload)
        s = float(self.pipe.critic.human_likeness_score(payload))
        body = f"score-only analysis\n{score_bar(s, self.target)}\nfeatures:\n{feature_deltas(a.to_feature_dict(), None)}"
        try:
            body += f"\ndetected patterns:\n{self.pipe.critic.explain(payload).details()}"
        except Exception:
            pass
        try:
            top = self.pipe.critic.global_importances(5)
            if top:
                body += "\nstrongest learned signals overall: " + ", ".join(
                    f"{n} ({w:.2f})" for n, w in top)
        except Exception:
            pass
        _bubble("humaize", body)

    def do_rewrite(self, raw: str) -> None:
        if "<input_text>" not in raw:
            raw = wrap_payload(raw)
            _print("(wrapped in <input_text> for isolation)", "dim")
        _bubble("you", extract_payload(raw)[:600] + ("…" if len(raw) > 600 else ""))
        _print("⚙ analyze → ✎ generate → ✓ validate", "dim")
        res = self.pipe.run(raw, target_score=self.target,
                            max_iters=self.max_iters, num_candidates=self.num,
                            min_similarity=self.min_sim, polish_passes=self.polish)
        for sc in res.candidates:
            p = sc.params
            flag = "" if sc.similarity >= res.min_similarity else "  ✗ off-topic, rejected"
            _print(f"  iter {sc.iteration + 1}  temp={p.temperature:.2f} top_p={p.top_p:.2f} "
                   f"rep={p.repetition_penalty:.2f} tk={p.top_k} nr={p.no_repeat_ngram_size}  →  "
                   f"{sc.score:.3f} sim={sc.similarity:.2f} bigram={sc.bigram_similarity:.2f}{flag}")
        self.last_rewrite = res.best_text
        if res.explanation_details:
            _print(f"detected AI patterns:\n{res.explanation_details}", "yellow")
        elif res.explanation:
            _print(f"detected AI patterns: {res.explanation}", "yellow")
        _bubble("humaize", res.best_text)
        if copy_to_clipboard(res.best_text):
            _print("⧉ rewrite copied to clipboard", "dim")
        else:
            _print("⧉ clipboard unavailable — use /save to write to a file", "yellow")
        _print(score_bar(res.best_score, res.target_score),
               "green" if res.criteria_met else "yellow")
        _print(f"fidelity to input: unigram {res.best_similarity:.2f} · "
               f"bigram {res.best_bigram_similarity:.2f} · "
               f"composite {res.best_fidelity:.2f} (floor {res.min_similarity:.2f})",
               "dim")
        if res.fixed_patterns:
            _print(f"patterns fixed by rewrite: {res.fixed_patterns}", "green")
        if res.remaining_patterns:
            _print(f"patterns still present: {res.remaining_patterns}", "yellow")
        if res.copies_rejected:
            _print(f"rejected {res.copies_rejected} near-verbatim echo(es) — model tried "
                   f"to return your input unchanged", "yellow")
        if res.chunks_total:
            done = res.chunks_total - res.chunks_kept_original
            _print(f"long input: sentence-by-sentence ({done}/{res.chunks_total} reworded, "
                   f"{res.chunks_kept_original} kept original to preserve meaning)", "dim")
        if res.polish_passes_used:
            _print(f"polish pass applied x{res.polish_passes_used} (re-attacked surviving traces)", "dim")
        if res.best_is_copy:
            _print("warning: model only echoed your input — try again with /iters 6, "
                   "or a longer input text", "red")
        _print(f"feature drift (orig → rewrite):\n{feature_deltas(res.original_features, res.best_features)}", "dim")
        _print(f"⌁ {res.iterations_used} iter(s) · {res.elapsed_s:.1f}s · {datetime.now():%H:%M}", "dim")

    def multiline(self) -> str:
        _print("multiline mode — end with a lone '.' line", "dim")
        lines = []
        while True:
            try:
                ln = input("... ")
            except EOFError:
                break
            if ln.strip() == ".":
                break
            lines.append(ln)
        return "\n".join(lines)

    def loop(self) -> int:
        _rule("humaize chat")
        _print(self.status(), "dim")
        _print("Type text to humanize, /m for multiline, /help for commands.\n")
        while True:
            try:
                line = input("you › ").strip()
            except (EOFError, KeyboardInterrupt):
                _print("\nbye.")
                return 0
            if not line:
                continue
            low = line.lower()
            if low in ("/quit", "/q", "exit"):
                _print("bye.")
                return 0
            if low == "/help":
                _print(HELP); continue
            if low == "/m":
                self.do_rewrite(self.multiline()); continue
            if low.startswith("/target"):
                try:
                    self.target = float(line.split()[1]); _print(f"target={self.target}")
                    self.effort = "custom"
                except (IndexError, ValueError):
                    _print("usage: /target 0.85")
                continue
            if low.startswith("/iters"):
                try:
                    self.max_iters = int(line.split()[1]); _print(f"max_iters={self.max_iters}")
                    self.effort = "custom"
                except (IndexError, ValueError):
                    _print("usage: /iters 4")
                continue
            if low.startswith("/num"):
                try:
                    self.num = int(line.split()[1]); _print(f"num_candidates={self.num}")
                    self.effort = "custom"
                except (IndexError, ValueError):
                    _print("usage: /num 2")
                continue
            if low.startswith("/sim"):
                try:
                    self.min_sim = float(line.split()[1]); _print(f"min_similarity={self.min_sim}")
                    self.effort = "custom"
                except (IndexError, ValueError):
                    _print("usage: /sim 0.35")
                continue
            if low.startswith("/polish"):
                try:
                    self.polish = int(line.split()[1]); _print(f"polish_passes={self.polish}")
                    self.effort = "custom"
                except (IndexError, ValueError):
                    _print("usage: /polish 2  (0 = off)")
                continue
            if low.startswith("/analyze"):
                self.do_analyze(line[len("/analyze"):].strip() or "(empty)")
                continue
            if low.startswith("/save"):
                parts = line.split(maxsplit=1)
                path = Path(parts[1]) if len(parts) > 1 else Path("rewrite_out.txt")
                if not self.last_rewrite:
                    _print("nothing to save yet.")
                else:
                    path.write_text(self.last_rewrite, encoding="utf-8")
                    _print(f"saved → {path}")
                continue
            if low == "/model":
                _print(self.status()); continue
            if low == "/status":
                _print(self.status()); continue
            if low.startswith("/effort"):
                parts = line.split(maxsplit=1)
                if len(parts) > 1:
                    try:
                        self.set_effort(parts[1])
                    except ValueError as exc:
                        _print(str(exc), "red")
                else:
                    self.effort_menu()
                continue
            if low.startswith("/model"):
                parts = line.split(maxsplit=1)
                if len(parts) > 1:
                    arg = parts[1].strip()
                    ids = [m["id"] for m in LOCAL_MODELS]
                    try:
                        self.set_model(ids[int(arg) - 1])
                    except (ValueError, IndexError):
                        self.set_model(arg)  # treat as a custom HF model id
                else:
                    self.model_menu()
                continue
            if line.startswith("/") and low not in ("/m",):
                _print("unknown command — /help"); continue
            try:
                self.do_rewrite(line)
            except PayloadIsolationError as exc:
                _print(f"isolation rejected input: {exc}", "red")
            except Exception as exc:
                _print(f"error: {exc}", "red")
            _print("")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Humaize chat CLI")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--light", action="store_true", help="Lightweight mode: less RAM/CPU/disk, slower")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--text", default=None, help="one-shot rewrite, then exit")
    ap.add_argument("--input-file", default=None)
    ap.add_argument("--target", type=float, default=None)
    ap.add_argument("--max-iters", type=int, default=None)
    ap.add_argument("--min-sim", type=float, default=None)
    ap.add_argument("--no-polish", action="store_true", help="Disable refinement passes")
    ap.add_argument("--effort", default=None, choices=["quick", "standard", "deep", "max"],
                    help="Effort preset (iters/candidates/polish); prompts in chat if omitted")
    ap.add_argument("--model-id", default=None,
                    help="Generator HF model id (overrides menu); prompts in chat if omitted")
    args = ap.parse_args(argv)

    cfg = AppConfig.from_yaml(args.config) if Path(args.config).exists() else AppConfig()
    if args.effort:
        preset = resolve_effort(args.effort)
        cfg.pipeline.max_iters = preset["max_iters"]
        cfg.pipeline.num_candidates = preset["num_candidates"]
        cfg.pipeline.polish_passes = preset["polish_passes"]
        cfg.pipeline.min_similarity = preset["min_similarity"]
    if args.target is not None:
        cfg.pipeline.target_score = args.target
    if args.max_iters is not None:
        cfg.pipeline.max_iters = args.max_iters
    if args.min_sim is not None:
        cfg.pipeline.min_similarity = args.min_sim
    if args.no_polish:
        cfg.pipeline.polish_passes = 0

    if args.fast:
        os.environ["HUMAIZE_FAST"] = "1"
    if args.light:
        os.environ["HUMAIZE_LIGHT"] = "1"
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if os.environ.get("HUMAIZE_LIGHT", "0") == "1":
        cfg = apply_light_mode(cfg)
    fast = args.fast or os.environ.get("HUMAIZE_FAST", "0") == "1"

    one_shot = bool(args.text or args.input_file)
    chat = Chat(cfg, fast, effort=args.effort, model_id=args.model_id,
                prompt_select=not one_shot and args.effort is None and args.model_id is None)
    if one_shot:
        raw = args.text or Path(args.input_file).read_text(encoding="utf-8")
        try:
            chat.do_rewrite(raw)
        except PayloadIsolationError as exc:
            _print(f"isolation rejected input: {exc}", "red")
            return 2
        except Exception as exc:
            _print(f"error: {exc}", "red")
            return 1
        return 0
    return chat.loop()


if __name__ == "__main__":
    raise SystemExit(main())
