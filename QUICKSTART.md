# Humaize — Quickstart (Yoga 9i)

3 commands. Local, offline after first download (~700MB).

```bash
# 1. Install
pip install -r requirements-light.txt

# 2. Train (uses built-in demo — or drag your folder onto train.bat)
python main.py --mode train --fast --light

# 3. Humanize
python cli.py --fast --light --text "Your AI-sounding text here."
# — or —
echo "<input_text>Your text</input_text>" > in.txt && python main.py --mode infer --fast --light --input-file in.txt
# — or double-click: chat.bat  (interactive)  /  train.bat  (drag folder to train on your writing)
```

That's it. First run downloads `SmolLM2-360M` once, then `~3s` per rewrite, `~0.9GB` RAM on Yoga 9i.

Need tables? `python cli.py --fast --light --input-file doc.txt` or `dochumanize.py` for `.docx` (keeps tables).

Full docs → `README.md` • Domain seams → `CONTEXT.md`
