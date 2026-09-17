"""Humaize desktop GUI — claymorphism edition.

Puffy pastel cards, chunky 3D buttons, zero new dependencies (tkinter only,
fully offline like the rest of the project).

    python gui.py --fast --light          # CPU-safe (default path via gui.bat)
    python gui.py --fast --effort deep --model-id HuggingFaceTB/SmolLM2-1.7B-Instruct

Layout: input card -> effort chips + model menu + target slider ->
HUMANIZE button -> score bar -> output card -> pattern-details card.
Heavy work (model load, rewrite, model switch) runs in worker threads;
the UI stays responsive and polls a queue.
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cli import EFFORT_PRESETS, LOCAL_MODELS, resolve_effort  # noqa: E402
from config import AppConfig, apply_light_mode  # noqa: E402
from dochumanize import (  # noqa: E402
    eligible_units,
    humanize_document,
    parse_docx,
    plain_preview,
    write_docx,
)
from main import build_runtime, run_training  # noqa: E402
from analyzer import PatternAnalyzer  # noqa: E402
from classifier import LocalPatternClassifier  # noqa: E402
from utils import copy_to_clipboard, wrap_payload  # noqa: E402

# ---------------------------------------------------------------- palette ---
BG = "#E8EBF3"        # app background (soft lavender-gray)
CARD = "#EFF1F8"      # card face (near-white)
INK = "#4A4E69"       # text (slate)
MUTED = "#8A8FA8"     # secondary text
SHADOW_DARK = "#C2C8DA"
SHADOW_LIGHT = "#FFFFFF"
CORAL = "#FFB3C1"     # primary button
CORAL_DARK = "#E58AA0"
MINT = "#B5EAD7"      # copy button
SKY = "#A8D8FF"       # selected chip
CHIP = "#E4E7F2"      # idle chip
GREEN = "#5FBF7A"
AMBER = "#E0A83C"
RED = "#DE6B6B"

FONT_TITLE = ("Segoe UI", 20, "bold")
FONT_H = ("Segoe UI", 11, "bold")
FONT_B = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)


def hex_mix(a: str, b: str, t: float) -> str:
    """Blend two #rrggbb colors (pure helper, unit-tested)."""
    ah = tuple(int(a[i:i + 2], 16) for i in (1, 3, 5))
    bh = tuple(int(b[i:i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ah, bh))


def round_rect(cv: tk.Canvas, x0, y0, x1, y1, r, **kw):
    """Smooth rounded rectangle on a canvas (clay building block)."""
    pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
           x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
    return cv.create_polygon(pts, smooth=True, splinesteps=24, **kw)


class ClayCard(tk.Frame):
    """Puffy card: dark offset shadow + light offset shadow + face + gloss."""

    def __init__(self, master, title: str = "", **kw):
        super().__init__(master, bg=BG, **kw)
        self.cv = tk.Canvas(self, bg=BG, highlightthickness=0, bd=0)
        self.cv.pack(fill="both", expand=True)
        self.body = tk.Frame(self.cv, bg=CARD)
        self._pad = 16
        self._win = self.cv.create_window(self._pad, self._pad, window=self.body, anchor="nw")
        if title:
            lbl = tk.Label(self.body, text=title, font=FONT_H, bg=CARD, fg=INK, anchor="w")
            lbl.pack(fill="x", padx=6, pady=(2, 6))
            self._title = lbl
        self.cv.bind("<Configure>", lambda _e: self._draw())
        self.body.bind("<Configure>", lambda _e: self._fit())

    def _draw(self):
        w, h = self.cv.winfo_width(), self.cv.winfo_height()
        if w < 10 or h < 10:
            return
        self.cv.delete("clay")
        r = 24
        round_rect(self.cv, 7, 10, w - 3, h - 3, r, fill=SHADOW_DARK, tags="clay")
        round_rect(self.cv, 1, 1, w - 9, h - 9, r, fill=SHADOW_LIGHT, tags="clay")
        round_rect(self.cv, 4, 5, w - 6, h - 6, r, fill=CARD, tags="clay")
        # top gloss: faint white arc kissing the top edge
        self.cv.create_arc(20, -h * 0.85, w - 20, h * 0.45, start=200, extent=140,
                           style="arc", outline="#FFFFFF", width=6, tags="clay")
        self.cv.tag_raise(self._win)
        self.cv.itemconfig(self._win, width=max(w - 2 * self._pad, 50))

    def _fit(self):
        try:
            h = self.body.winfo_reqheight() + 2 * self._pad + 6
            if abs(self.cv.winfo_height() - h) > 2:
                self.cv.configure(height=h)
        except Exception:
            pass


class ClayButton(tk.Canvas):
    """Chunky pill button with hover/press clay physics."""

    def __init__(self, master, text, command, fill=CORAL, dark=CORAL_DARK,
                 width=200, height=48, font=FONT_H, **kw):
        super().__init__(master, width=width, height=height, bg=BG,
                         highlightthickness=0, bd=0, **kw)
        self._fill, self._dark = fill, dark
        self._cmd, self._font = command, font
        self._text, self._state = text, "normal"
        self._selected = False
        self._bw, self._bh = width, height
        self.bind("<Enter>", lambda _e: self._paint("hover"))
        self.bind("<Leave>", lambda _e: self._paint("normal"))
        self.bind("<Button-1>", lambda _e: self._paint("pressed"))
        self.bind("<ButtonRelease-1>", self._click)
        self._paint("normal")

    def _paint(self, state):
        self._state = state
        self.delete("b")
        w, h, r = self._bw, self._bh, 22
        dy = 3 if state == "pressed" else 0
        face = self._fill() if callable(self._fill) else self._fill
        if self._selected:
            # inset look: inner shadow on top-left
            round_rect(self, 2, 2 + dy, w - 2, h - 2 + dy, r, fill=self._dark, tags="b")
            round_rect(self, 4, 5 + dy, w - 4, h - 4 + dy, r - 2, fill=hex_mix(face, "#000000", 0.08),
                       tags="b")
        else:
            round_rect(self, 2, 5 + dy, w - 2, h - 1 + dy, r, fill=self._dark, tags="b")
            round_rect(self, 2, 2 + dy, w - 2, h - 4 + dy, r, fill=face, tags="b")
        self.create_text(w / 2, h / 2 + dy, text=self._text, font=self._font,
                         fill=INK, tags="b")

    def _click(self, _event):
        self._paint("hover")
        if self._cmd:
            self._cmd()

    def set_selected(self, value: bool):
        self._selected = value
        self._paint("normal")

    def set_text(self, text: str):
        self._text = text
        self._paint("normal")


class ProgressBar(tk.Canvas):
    """Slim clay status bar: sliding chunk while busy, fill fraction for docs."""

    def __init__(self, master, **kw):
        super().__init__(master, height=14, bg=BG, highlightthickness=0, bd=0, **kw)
        self._mode = "idle"  # idle | busy | frac
        self._frac, self._pos = 0.0, 0.0
        self.bind("<Configure>", lambda _e: self._paint())

    def busy(self):
        if self._mode != "busy":
            self._mode = "busy"
            self._tick()

    def set_fraction(self, frac: float):
        self._mode = "frac"
        self._frac = max(0.0, min(1.0, frac))
        self._paint()

    def clear(self):
        self._mode = "idle"
        self._paint()

    def _tick(self):
        if self._mode != "busy":
            return
        self._pos = (self._pos + 0.05) % 1.5
        self._paint()
        self.after(90, self._tick)

    def _paint(self):
        self.delete("p")
        w = self.winfo_width()
        if w < 10:
            self.after(100, self._paint)  # layout not ready yet — retry
            return
        h = 14
        round_rect(self, 2, 2, w - 2, h - 2, 6, fill="#D5DAE9", tags="p")
        if self._mode == "busy":
            cw = max(w * 0.25, 40)
            x = (self._pos - 0.25) * w
            round_rect(self, x, 2, x + cw, h - 2, 6, fill=SKY, tags="p")
        elif self._mode == "frac" and self._frac > 0:
            round_rect(self, 2, 2, 2 + max((w - 4) * self._frac, 12), h - 2, 6,
                       fill=SKY, tags="p")


class ScoreBar(tk.Canvas):
    """Rounded progress pill: amber while trying, green on PASS."""

    def __init__(self, master, **kw):
        super().__init__(master, height=30, bg=CARD, highlightthickness=0, bd=0, **kw)
        self.bind("<Configure>", lambda _e: self._paint())
        self._score, self._target, self._met = 0.0, 0.85, False

    def set(self, score: float, target: float, met: bool):
        self._score, self._target, self._met = score, target, met
        self._paint()

    def _paint(self):
        self.delete("s")
        w = self.winfo_width()
        if w < 10:
            return
        h = 30
        round_rect(self, 2, 4, w - 2, h - 4, 13, fill="#DDE1EE", tags="s")
        fill = max(int((w - 4) * max(0.0, min(1.0, self._score))), 26)
        color = GREEN if self._met else (AMBER if self._score >= 0.4 else RED)
        round_rect(self, 2, 4, 2 + fill, h - 4, 13, fill=color, tags="s")
        self.create_text(w / 2, h / 2,
                         text=f"{self._score:.3f} / {self._target:.2f}  "
                              f"{'PASS' if self._met else 'retry'}",
                         font=FONT_H, fill="#FFFFFF" if self._score > 0.35 else INK, tags="s")


class App(tk.Tk):
    def __init__(self, cfg: AppConfig, fast: bool,
                 effort: str = "standard", model_id: str | None = None):
        super().__init__()
        self.cfg, self.fast = cfg, fast
        self.title("humaize — local humanizer")
        self.configure(bg=BG)
        self.geometry("880x980")
        self.minsize(640, 700)
        self.jobs: queue.Queue = queue.Queue()
        self.busy = True
        self.pipe = None
        self.critic = None
        self.effort = effort
        self._build_widgets()
        self._apply_effort(effort)
        if model_id:
            self._pending_model = model_id
        else:
            self._pending_model = None
        self._submit("load", None)
        self.after(150, self._pump)

    # ------------------------------------------------------------------ UI ---
    def _build_widgets(self):
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=22, pady=(16, 4))
        tk.Label(top, text="humaize", font=FONT_TITLE, bg=BG, fg=INK).pack(side="left")
        tk.Label(top, text="local AI-pattern humanizer", font=FONT_B, bg=BG, fg=MUTED).pack(
            side="left", padx=(10, 0), pady=(10, 0))

        in_card = ClayCard(self, title="Your text")
        in_card.pack(fill="x", padx=18, pady=6)
        self.input_text = tk.Text(in_card.body, height=4, font=FONT_B, bg="#FFFFFF", fg=INK,
                                  relief="flat", highlightthickness=0, wrap="word",
                                  padx=8, pady=8)
        self.input_text.pack(fill="x", padx=6, pady=(0, 4))

        ctl = ClayCard(self, title="Effort  ·  Model  ·  Goal")
        ctl.pack(fill="x", padx=18, pady=6)
        row = tk.Frame(ctl.body, bg=CARD)
        row.pack(fill="x", padx=6, pady=(0, 2))
        self.chips: dict[str, ClayButton] = {}
        for name in EFFORT_PRESETS:
            btn = ClayButton(row, text=name.capitalize(), width=104, height=40,
                             fill=lambda n=name: SKY if n == self.effort else CHIP,
                             dark=SHADOW_DARK, font=FONT_B,
                             command=lambda n=name: self.on_effort(n))
            btn.pack(side="left", padx=4)
            self.chips[name] = btn
        row2 = tk.Frame(ctl.body, bg=CARD)
        row2.pack(fill="x", padx=6, pady=(6, 2))
        tk.Label(row2, text="Model", font=FONT_B, bg=CARD, fg=INK).pack(side="left")
        self.model_labels = [m["label"] for m in LOCAL_MODELS]
        self.model_ids = [m["id"] for m in LOCAL_MODELS]
        self.model_var = tk.StringVar(value=self.model_labels[0])
        menu = tk.OptionMenu(row2, self.model_var, *self.model_labels,
                             command=lambda _v: self.on_model_pick())
        menu.configure(bg=CHIP, fg=INK, font=FONT_B, relief="flat",
                       highlightthickness=0, activebackground=SKY)
        menu.pack(side="left", padx=8)
        tk.Label(row2, text="Goal", font=FONT_B, bg=CARD, fg=INK).pack(side="left", padx=(8, 0))
        self.target_var = tk.DoubleVar(value=self.cfg.pipeline.target_score)
        tk.Scale(row2, from_=0.5, to=0.95, resolution=0.05, orient="horizontal",
                 variable=self.target_var, bg=CARD, fg=INK, font=FONT_SMALL,
                 highlightthickness=0, troughcolor="#DDE1EE", length=130).pack(side="left")

        brow = tk.Frame(self, bg=BG)
        brow.pack(fill="x", padx=18, pady=8)
        self.go_btn = ClayButton(brow, text="HUMANIZE", command=self.on_humanize,
                                 width=220, height=52)
        self.go_btn.pack(side="left", padx=4)
        self.copy_btn = ClayButton(brow, text="Copy result", command=self.on_copy,
                                   fill=MINT, dark="#7FBFA4", width=170, height=52, font=FONT_B)
        self.copy_btn.pack(side="left", padx=4)
        self.clear_btn = ClayButton(brow, text="Clear", command=self.on_clear,
                                    fill=CHIP, dark=SHADOW_DARK, width=120, height=52, font=FONT_B)
        self.clear_btn.pack(side="left", padx=4)
        self.upload_btn = ClayButton(brow, text="Upload doc", command=self.on_upload,
                                     fill=SKY, dark="#8FB4E8", width=150, height=52, font=FONT_B)
        self.upload_btn.pack(side="left", padx=4)

        self.doc_bar = tk.Frame(self, bg=BG)
        self.doc_label_var = tk.StringVar(value="")
        tk.Label(self.doc_bar, textvariable=self.doc_label_var, font=FONT_B,
                 bg=BG, fg=INK, anchor="w").pack(side="left", padx=(4, 8))
        self.doc_go_btn = ClayButton(self.doc_bar, text="Humanize file", command=self.on_humanize_file,
                                     fill=MINT, dark="#7FBFA4", width=170, height=44, font=FONT_B)
        self.doc_go_btn.pack(side="left", padx=4)
        self.doc_close_btn = ClayButton(self.doc_bar, text="Close file", command=self.on_close_doc,
                                        fill=CHIP, dark=SHADOW_DARK, width=130, height=44, font=FONT_B)
        self.doc_close_btn.pack(side="left", padx=4)
        self.doc = None

        score_card = ClayCard(self, title="Human-likeness")
        score_card.pack(fill="x", padx=18, pady=6)
        self.score = ScoreBar(score_card.body)
        self.score.pack(fill="x", padx=6, pady=(0, 4))

        out_card = ClayCard(self, title="Rewrite")
        out_card.pack(fill="both", expand=True, padx=18, pady=6)
        self.output_text = tk.Text(out_card.body, height=5, font=FONT_B, bg="#FFFFFF", fg=INK,
                                   relief="flat", highlightthickness=0, wrap="word",
                                   padx=8, pady=8, state="disabled")
        self.output_text.pack(fill="both", expand=True, padx=6, pady=(0, 2))
        self.detail_text = tk.Text(out_card.body, height=4, font=FONT_SMALL, bg=CARD, fg=MUTED,
                                   relief="flat", highlightthickness=0, wrap="word",
                                   padx=8, pady=4)
        self.detail_text.pack(fill="x", padx=6, pady=(0, 4))

        self.status_var = tk.StringVar(value="starting…")
        tk.Label(self, textvariable=self.status_var, font=FONT_SMALL, bg=BG, fg=MUTED,
                 anchor="w").pack(fill="x", padx=24, pady=(0, 2))
        self.prog = ProgressBar(self)
        self.prog.pack(fill="x", padx=24, pady=(0, 12))

    # --------------------------------------------------------------- actions -
    def _set_status(self, msg: str):
        self.status_var.set(msg)

    def on_effort(self, name: str):
        try:
            self._apply_effort(name)
            self._set_status(f"effort={name}: "
                             f"{EFFORT_PRESETS[name]['desc']}")
        except ValueError as exc:
            messagebox.showerror("Effort", str(exc))

    def _apply_effort(self, name: str):
        preset = resolve_effort(name)
        self.effort = name.strip().lower()
        self.effort_preset = dict(preset)
        for n, btn in self.chips.items():
            btn.set_selected(n == self.effort)

    def on_model_pick(self):
        idx = self.model_labels.index(self.model_var.get())
        self._submit("model", self.model_ids[idx])

    def on_clear(self):
        self.input_text.delete("1.0", "end")
        self.on_close_doc()

    def on_close_doc(self):
        self.doc = None
        self.doc_bar.pack_forget()
        self.go_btn.set_text("HUMANIZE")

    def _run_kwargs(self) -> dict:
        p = self.effort_preset
        return {"target_score": float(self.target_var.get()),
                "max_iters": p["max_iters"],
                "num_candidates": p["num_candidates"],
                "min_similarity": p["min_similarity"],
                "polish_passes": p["polish_passes"]}

    def on_upload(self):
        path = filedialog.askopenfilename(
            title="Upload document",
            filetypes=[("Word documents", "*.docx"), ("Text/Markdown", "*.txt *.md"),
                       ("All files", "*.*")])
        if not path:
            return
        lower = path.lower()
        if lower.endswith((".txt", ".md", ".markdown")):
            try:
                text = Path(path).read_text(encoding="utf-8")
            except Exception as exc:
                messagebox.showerror("Upload", f"Could not read file:\n{exc}")
                return
            self.on_close_doc()
            self.input_text.delete("1.0", "end")
            self.input_text.insert("end", text.strip())
            self._set_status(f"loaded {Path(path).name} into the input box — press HUMANIZE")
        elif lower.endswith(".docx"):
            try:
                doc = parse_docx(path)
            except Exception as exc:
                self._set_status(f"upload failed: {exc}")
                messagebox.showerror("Upload", f"Could not parse document:\n{exc}")
                return
            self.doc = doc
            units = eligible_units(doc)
            self.doc_label_var.set(
                f"{doc.source.name} — {doc.n_paragraphs} paragraphs, "
                f"{doc.n_tables} table(s), {len(units)} text units to humanize "
                f"(short labels kept as-is)")
            self.doc_bar.pack(fill="x", padx=18, pady=(0, 4))
            self.go_btn.set_text("HUMANIZE FILE")
            self._set_status("document loaded — press HUMANIZE FILE "
                             "(tables keep their structure; cell text is rewritten)")
        else:
            messagebox.showinfo("Upload", "Supported: .docx (tables preserved), .txt, .md\n"
                                          "Tip: convert PDFs to .docx first.")

    def on_humanize_file(self):
        if self.pipe is None:
            self._set_status("models still loading — try again in a moment")
            return
        if self.busy or self.doc is None:
            return
        self.busy = True
        self._set_status("humanizing document…")
        self._submit("doc", None)

    def on_copy(self):
        text = self.output_text.get("1.0", "end").strip()
        if copy_to_clipboard(text):
            self._set_status("rewrite copied to clipboard")
        else:
            self._set_status("clipboard unavailable")

    def on_humanize(self):
        if self.pipe is None:
            self._set_status("models still loading — try again in a moment")
            return
        if self.doc is not None:
            self.on_humanize_file()  # a document is loaded: humanize the file
            return
        if self.busy:
            return
        raw = self.input_text.get("1.0", "end").strip()
        if not raw:
            messagebox.showinfo("humaize", "Type or paste some text first.")
            return
        try:
            wrapped = wrap_payload(raw)
        except Exception as exc:
            messagebox.showerror("humaize", str(exc))
            return
        self.busy = True
        self._set_status("analyze → generate → validate…")
        self._submit("humanize", wrapped)

    # --------------------------------------------------------------- workers -
    def _submit(self, kind: str, payload):
        self.prog.busy()
        threading.Thread(target=self._work, args=(kind, payload), daemon=True).start()

    def _work(self, kind: str, payload):
        try:
            if kind == "load":
                cpath = Path(self.cfg.classifier.save_path)
                if cpath.exists():
                    critic = LocalPatternClassifier.load(cpath, analyzer=None)
                    try:
                        cheap = len(PatternAnalyzer.CHEAP_FEATURE_NAMES)
                        need_lm = critic._stat_dim != cheap and critic._stat_dim != 0
                    except Exception:
                        need_lm = not self.fast
                    critic.analyzer = PatternAnalyzer(config=self.cfg.analyzer,
                                                     load_model=need_lm)
                else:
                    critic = run_training(self.cfg, None, self.fast)
                pipe = build_runtime(self.cfg, critic, self.fast)
                self.jobs.put(("loaded", pipe))
                if self._pending_model:
                    self._work("model", self._pending_model)
                    self._pending_model = None
            elif kind == "model":
                model_id = payload
                kind_guess = "big" if any(t in model_id.lower()
                                          for t in ("7b", "8b", "13b", "14b", "70b")) else "small"
                if kind_guess == "big":
                    self.cfg.generator.model_name = model_id
                    self.fast = False
                else:
                    self.cfg.generator.fallback_model_name = model_id
                    self.fast = True
                pipe = build_runtime(self.cfg, self.critic, self.fast)
                self.jobs.put(("model_ready", pipe))
            elif kind == "humanize":
                assert self.pipe is not None
                res = self.pipe.run(payload, **self._run_kwargs())
                self.jobs.put(("result", res))
            elif kind == "doc":
                assert self.pipe is not None and self.doc is not None
                doc, kwargs = self.doc, self._run_kwargs()

                def prog(done: int, total: int, label: str):
                    self.jobs.put(("doc_progress", (done, total, label)))

                report = humanize_document(self.pipe, doc, kwargs, progress=prog)
                out = doc.source.with_name(doc.source.stem + "_humanized.docx")
                write_docx(doc, out)
                self.jobs.put(("doc_result", (out, report)))
        except Exception as exc:  # never kill the UI thread; report instead
            self.jobs.put(("error", f"{kind}: {exc}"))

    def _pump(self):
        try:
            while True:
                kind, payload = self.jobs.get_nowait()
                if kind == "loaded":
                    self.pipe = payload
                    self.critic = payload.critic
                    model = getattr(payload.rewriter, "active_model_name", "?") or "?"
                    self._set_status(f"ready — model={model} critic={payload.critic.backend}")
                    self.prog.clear()
                    self.busy = False
                elif kind == "model_ready":
                    self.pipe = payload
                    model = getattr(payload.rewriter, "active_model_name", "?") or "?"
                    try:
                        self.model_var.set(next(
                            l for l, i in zip(self.model_labels, self.model_ids) if i == model))
                    except StopIteration:
                        pass
                    self._set_status(f"model switched — {model}")
                    self.prog.clear()
                    self.busy = False
                elif kind == "result":
                    self._show(payload)
                    self.prog.clear()
                    self.busy = False
                elif kind == "doc_progress":
                    done, total, label = payload
                    self.prog.set_fraction(done / total if total else 0)
                    self._set_status(f"humanizing document… unit {done}/{total} — {label}")
                elif kind == "doc_result":
                    self._show_doc(*payload)
                    self.prog.clear()
                    self.busy = False
                elif kind == "error":
                    self._set_status(f"error: {payload}")
                    self.prog.clear()
                    self.busy = False
        except queue.Empty:
            pass
        self.after(150, self._pump)

    def _show(self, res):
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.insert("end", res.best_text)
        self.output_text.configure(state="disabled")
        self.score.set(res.best_score, res.target_score, res.criteria_met)
        bits = []
        if res.explanation_details:
            bits.append(res.explanation_details)
        bits.append(f"fidelity: unigram {res.best_similarity:.2f} · "
                    f"bigram {res.best_bigram_similarity:.2f} · "
                    f"composite {res.best_fidelity:.2f} (floor {res.min_similarity:.2f})")
        if res.fixed_patterns:
            bits.append(f"fixed: {res.fixed_patterns}")
        if res.remaining_patterns:
            bits.append(f"still present: {res.remaining_patterns}")
        if res.copies_rejected:
            bits.append(f"rejected {res.copies_rejected} echo(es)")
        if res.chunks_total:
            bits.append(f"sentence-by-sentence {res.chunks_total - res.chunks_kept_original}/"
                        f"{res.chunks_total} reworded")
        if res.polish_passes_used:
            bits.append(f"polish x{res.polish_passes_used}")
        if res.best_is_copy:
            bits.append("WARNING: model only echoed your input — try again or Deep effort")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("end", "\n".join(bits))
        state = "PASS" if res.criteria_met else "needs work"
        self._set_status(f"done in {res.elapsed_s:.1f}s · {res.iterations_used} iter(s) — {state}")

    def _show_doc(self, out_path, report: list):
        assert self.doc is not None
        rewritten = sum(1 for r in report if r["action"] == "rewritten")
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.insert("end", plain_preview(self.doc))
        self.output_text.configure(state="disabled")
        self.score.set(1.0 if rewritten else 0.0, 1.0, bool(rewritten))
        lines = [f"saved → {out_path}", f"{rewritten}/{len(report)} units rewritten"]
        for r in report:
            if r["action"] == "rewritten":
                lines.append(f"  • {r['label']}: {r.get('before', '?')} → {r.get('after', '?')} "
                             f"(sim {r.get('sim', '?')})")
            else:
                lines.append(f"  • {r['label']}: {r['action']}")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("end", "\n".join(lines))
        self._set_status(f"document done — {rewritten}/{len(report)} units rewritten → {out_path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Humaize claymorphism GUI")
    ap.add_argument("--fast", action="store_true", help="Small CPU-safe models")
    ap.add_argument("--light", action="store_true", help="Less RAM/CPU/disk (slower)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--effort", default="standard", choices=list(EFFORT_PRESETS))
    ap.add_argument("--model-id", default=None, help="Generator HF model id")
    args = ap.parse_args(argv)

    cfg = AppConfig.from_yaml(args.config) if Path(args.config).exists() else AppConfig()
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

    try:
        app = App(cfg, fast, effort=args.effort, model_id=args.model_id)
    except tk.TclError as exc:
        print(f"GUI needs a display: {exc}")
        return 2
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
