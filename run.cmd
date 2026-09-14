@echo off
REM ===================================================================
REM  SIH 26055 - RF Environment Simulator
REM  One-click setup and run for Windows.
REM
REM  Double-click this file, or run:  run.cmd
REM
REM  It will, in order:
REM    1. find or install Python 3.12
REM    2. create the .venv virtual environment
REM    3. install every pinned dependency
REM    4. run the full test suite
REM    5. print the comparison table
REM
REM  Safe to run again: steps already done are skipped.
REM ===================================================================

setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo.
echo  ==========================================================
echo   SIH 26055  RF Environment Simulator
echo   One-click setup
echo  ==========================================================
echo.

REM ---------- 1. locate Python 3.12 ----------
set "PY="
py -3.12 -c "import sys" >nul 2>&1 && set "PY=py -3.12"

if not defined PY (
    echo  [1/5] Python 3.12 not found. Installing it with winget...
    echo        ^(this needs an internet connection and may take a few minutes^)
    echo.
    where winget >nul 2>&1
    if errorlevel 1 (
        echo  ERROR: winget is not available on this machine.
        echo.
        echo  Please install Python 3.12 manually from:
        echo      https://www.python.org/downloads/release/python-31210/
        echo  Tick "Add python.exe to PATH" during installation, then run this file again.
        echo.
        pause
        exit /b 1
    )
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements --silent
    py -3.12 -c "import sys" >nul 2>&1 && set "PY=py -3.12"
    if not defined PY (
        echo.
        echo  Python 3.12 was installed but this window cannot see it yet.
        echo  Close this window, open a new terminal, and run run.cmd again.
        echo.
        pause
        exit /b 1
    )
    echo  [1/5] Python 3.12 installed.
) else (
    echo  [1/5] Python 3.12 found.
)

REM ---------- 2. virtual environment ----------
if exist ".venv\Scripts\python.exe" (
    echo  [2/5] Virtual environment already exists, reusing it.
) else (
    echo  [2/5] Creating the virtual environment in .venv ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo  ERROR: could not create the virtual environment.
        pause
        exit /b 1
    )
)

set "VPY=%CD%\.venv\Scripts\python.exe"

REM ---------- 3. dependencies ----------
REM numba is installed first so pip pins a compatible numpy, as the build plan requires.
"%VPY%" -c "import numba, numpy, gymnasium, scipy, pandas, pytest" >nul 2>&1
if errorlevel 1 (
    echo  [3/5] Installing dependencies ^(numpy, numba, gymnasium, scipy, pandas, pytest^)...
    echo        First run downloads about 100 MB and takes a few minutes.
    "%VPY%" -m pip install --upgrade pip --quiet
    "%VPY%" -m pip install --quiet -r requirements.txt
    if errorlevel 1 (
        echo.
        echo  ERROR: dependency installation failed. Scroll up for the reason.
        echo  A common cause on Windows is long-path support being disabled:
        echo      https://pip.pypa.io/warnings/enable-long-paths
        echo.
        pause
        exit /b 1
    )
    echo  [3/5] Dependencies installed.
) else (
    echo  [3/5] Dependencies already installed.
)

REM ---------- 4. tests ----------
echo.
echo  [4/5] Running the test suite ^(88 unit and contract tests^)...
echo.
"%VPY%" -m pytest -q
if errorlevel 1 (
    echo.
    echo  ERROR: tests failed. The simulator is not in a good state.
    pause
    exit /b 1
)

echo.
echo  [4/5] Running the black-box physics checks ^(13 cases^)...
echo.
"%VPY%" -m pytest tests/verify_sim.py -q -s
if errorlevel 1 (
    echo.
    echo  ERROR: physics verification failed.
    pause
    exit /b 1
)

REM ---------- 5. the demo ----------
echo.
echo  [5/5] Running the comparison table. This is the headline result.
echo.
"%VPY%" -m eval.run_comparison --scenarios scenarios/train_default.json --seeds 0-4 --schedulers round_robin weighted_priority clairvoyant --quiet

echo.
echo  ==========================================================
echo   Everything works.
echo.
echo   To see the live ASCII waterfall:
echo       .venv\Scripts\python.exe -m eval.show_waterfall
echo.
echo   To re-run just the comparison table:
echo       .venv\Scripts\python.exe -m eval.run_comparison --scenarios scenarios/train_default.json --seeds 0-4 --schedulers round_robin weighted_priority clairvoyant
echo.
echo   Read README.md for the full guide.
echo  ==========================================================
echo.
pause
