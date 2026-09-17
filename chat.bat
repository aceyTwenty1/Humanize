@echo off
REM Humaize chat shortcut — double-click to open the CLI.
REM Extra args are forwarded, e.g.: chat.bat --text "hello"
REM --fast = small CPU-safe models. --light = even less RAM/CPU/disk (slower).
REM For full GPU quality, remove the --fast and --light flags below.

cd /d "%~dp0"
python cli.py --fast --light %*

pause
