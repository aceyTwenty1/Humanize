@echo off
REM Humaize training shortcut — double-click to start.
REM
REM Drag-and-drop:
REM   *.csv / *.json / *.jsonl  -> labeled dataset file (text,label)
REM   folder  -or-  *.txt / *.md -> your own writing, trained as HUMAN
REM   extra flags, e.g.:  train.bat --model-type mlp --trees 500
REM
REM Optional tweaks — from cmd, e.g.:  set MODEL=mlp&& train.bat
REM   MODEL=mlp        critic backend: auto (default), xgboost, histgb, logreg, mlp
REM   TREES=500        boosting rounds (longer critic training)
REM   EPOCHS=3         RL epochs (full GPU mode only; skipped under --fast)
REM   FETCH=60         fetch ~60 public-domain human samples to a temp dir
REM                    for this run (auto-deleted if accuracy is good)
REM
REM --fast = small CPU-safe models. --light = even less RAM/CPU/disk (slower).
REM For full GPU training, remove the --fast and --light flags below.

cd /d "%~dp0"

set EXTRA=
if not "%MODEL%"=="" set EXTRA=%EXTRA% --model-type %MODEL%
if not "%TREES%"=="" set EXTRA=%EXTRA% --trees %TREES%
if not "%EPOCHS%"=="" set EXTRA=%EXTRA% --epochs %EPOCHS%
if not "%FETCH%"=="" set EXTRA=%EXTRA% --fetch-human %FETCH%

set ARG1=%~1
if "%ARG1:~0,2%"=="--" (
    python main.py --mode train --fast --light %* %EXTRA%
    goto done
)

if "%~1"=="" (
    python main.py --mode train --fast --light %EXTRA%
    goto done
)

if exist "%~1\*" (
    echo Folder detected - training on your writing as HUMAN.
    python main.py --mode train --fast --light --human-dir "%~1" %EXTRA%
    goto done
)

if /i "%~x1"==".csv" goto dataset
if /i "%~x1"==".json" goto dataset
if /i "%~x1"==".jsonl" goto dataset

echo Text file detected - training on your writing as HUMAN.
python main.py --mode train --fast --light --human-dir "%~1" %EXTRA%
goto done

:dataset
python main.py --mode train --fast --light --data "%~1" %EXTRA%
goto done

:done
pause
