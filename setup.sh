#!/usr/bin/env bash
# ============================================================
#  OpenFlo one-time setup (Linux / macOS).
#  Creates a local .venv and installs OpenFlo + every
#  dependency, so ./openflo-gui.sh can launch. Safe to re-run
#  (idempotent: re-uses the venv, re-installs/updates deps).
#
#  Run:  ./setup.sh        (chmod +x once if needed)
# ============================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Locate a Python 3. pip enforces the >=3.11 floor on install.
PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then
    echo "[OpenFlo setup] No Python found. Install Python 3.11+ and re-run." >&2
    exit 1
fi
echo "[OpenFlo setup] Using Python: $PY ($("$PY" --version 2>&1))"

if [ ! -x "$ROOT/.venv/bin/python" ]; then
    echo "[OpenFlo setup] Creating virtual environment in .venv ..."
    "$PY" -m venv "$ROOT/.venv"
fi

VPY="$ROOT/.venv/bin/python"
echo "[OpenFlo setup] Upgrading pip ..."
"$VPY" -m pip install --upgrade pip
echo "[OpenFlo setup] Installing OpenFlo + dependencies (this can take a few minutes) ..."
"$VPY" -m pip install -e ".[gui]"

echo
echo "[OpenFlo setup] Done. Launch OpenFlo with:  ./openflo-gui.sh"
