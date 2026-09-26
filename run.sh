#!/usr/bin/env bash
# ===================================================================
#  SIH 26055 - RF Environment Simulator
#  One-click setup and run for Linux and macOS.
#
#      chmod +x run.sh && ./run.sh
#
#  It will, in order:
#    1. find Python 3.11 or 3.12
#    2. create the .venv virtual environment
#    3. install every pinned dependency
#    4. run the full test suite
#    5. print the comparison table
#
#  Safe to run again: steps already done are skipped.
# ===================================================================
set -euo pipefail
cd "$(dirname "$0")"

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
fail() { printf '\033[31mERROR: %s\033[0m\n' "$1" >&2; exit 1; }

echo
bold "=========================================================="
bold " SIH 26055  RF Environment Simulator"
bold " One-click setup"
bold "=========================================================="
echo

# ---------- 1. locate a supported Python ----------
PY=""
for c in python3.12 python3.11 python3 python; do
  if command -v "$c" >/dev/null 2>&1; then
    if "$c" -c 'import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] < (3,13) else 1)' 2>/dev/null; then
      PY="$c"; break
    fi
  fi
done

if [ -z "$PY" ]; then
  echo "Python 3.11 or 3.12 was not found. Numba has no wheels for 3.13+, so one of these is required."
  echo
  echo "  Ubuntu / Debian :  sudo apt install python3.12 python3.12-venv"
  echo "  Fedora          :  sudo dnf install python3.12"
  echo "  macOS (Homebrew):  brew install python@3.12"
  echo
  fail "no supported Python interpreter"
fi
echo "[1/5] Using $($PY --version) at $(command -v $PY)"

# ---------- 2. virtual environment ----------
if [ -x ".venv/bin/python" ]; then
  echo "[2/5] Virtual environment already exists, reusing it."
else
  echo "[2/5] Creating the virtual environment in .venv ..."
  "$PY" -m venv .venv || fail "could not create the virtual environment (on Debian/Ubuntu you may need python3-venv)"
fi
VPY="$PWD/.venv/bin/python"

# ---------- 3. dependencies ----------
# numba is installed first so pip pins a compatible numpy, as the build plan requires.
if "$VPY" -c "import numba, numpy, gymnasium, scipy, pandas, pytest" >/dev/null 2>&1; then
  echo "[3/5] Dependencies already installed."
else
  echo "[3/5] Installing dependencies (numpy, numba, gymnasium, scipy, pandas, pytest)..."
  echo "      First run downloads about 100 MB and takes a few minutes."
  "$VPY" -m pip install --upgrade pip --quiet
  "$VPY" -m pip install --quiet -r requirements.txt || fail "dependency installation failed (scroll up for the reason)"
  echo "[3/5] Dependencies installed."
fi

# ---------- 4. tests ----------
echo
echo "[4/5] Running the test suite (88 unit and contract tests)..."
echo
"$VPY" -m pytest -q || fail "tests failed; the simulator is not in a good state"

echo
echo "[4/5] Running the black-box physics checks (13 cases)..."
echo
"$VPY" -m pytest tests/verify_sim.py -q -s || fail "physics verification failed"

# ---------- 5. the demo ----------
echo
echo "[5/5] Running the comparison table. This is the headline result."
echo
"$VPY" -m eval.run_comparison \
    --scenarios scenarios/train_default.json \
    --seeds 0-4 \
    --schedulers round_robin weighted_priority clairvoyant \
    --quiet

echo
bold "=========================================================="
bold " Everything works."
echo
echo " To see the live ASCII waterfall:"
echo "     .venv/bin/python -m eval.show_waterfall"
echo
echo " To re-run just the comparison table:"
echo "     .venv/bin/python -m eval.run_comparison \\"
echo "         --scenarios scenarios/train_default.json --seeds 0-4 \\"
echo "         --schedulers round_robin weighted_priority clairvoyant"
echo
echo " Read README.md for the full guide."
bold "=========================================================="
echo
