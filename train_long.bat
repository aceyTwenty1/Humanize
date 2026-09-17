@echo off
REM Humaize LONG training — a ~10-minute session. Double-click and walk away.
REM
REM What it does (all local, CPU-safe):
REM   1. Fetches ~400 public-domain human paragraphs (temp dir, auto-deleted
REM      if accuracy clears 0.6, otherwise kept + path printed).
REM   2. Trains the MLP neural-net critic on demo + fetched text.
REM   3. Runs the adversarial RL loop on the small fallback model
REM      (--rl-cpu; this is the slow, valuable part: generations on CPU).
REM   4. Runs the two inference demos so you can see the result.
REM Wall time varies with machine/network — roughly 8-15 minutes here.

cd /d "%~dp0"

python main.py --mode full --fast --light --fetch-human 400 --model-type mlp --mlp-iters 800 --epochs 2 --rl-cpu --fetch-min-acc 0.6

pause
