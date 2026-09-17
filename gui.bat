@echo off
REM Humaize claymorphism GUI — double-click to open (no console window).
REM Optional flags, e.g.:  gui.bat --effort deep
REM   --effort quick|standard|deep|max   (default standard)
REM   --model-id <hf-id>                 (default SmolLM2-360M)
REM Remove --fast --light for full GPU mode.
REM Errors still appear inside the GUI status bar.

cd /d "%~dp0"

where pythonw >nul 2>nul
if %errorlevel%==0 (
    start "" pythonw gui.py --fast --light %*
) else (
    start "" /min python gui.py --fast --light %*
)
