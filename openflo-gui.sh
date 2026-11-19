#!/usr/bin/env bash
# ============================================================
#  Quick-launch the OpenFlo gate-editor GUI on Linux / macOS.
#  Run:  ./openflo-gui.sh   (chmod +x once if needed)
#  Uses the project .venv if present, else system python3.
# ============================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Make the src-layout package importable even without `pip install`.
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

# First launch with no venv? Bootstrap it (installs deps), then continue.
if [ ! -x "$ROOT/.venv/bin/python" ] && [ -f "$ROOT/setup.sh" ]; then
    echo "[OpenFlo] First launch - installing dependencies, please wait..."
    bash "$ROOT/setup.sh"
fi

if [ -x "$ROOT/.venv/bin/python" ]; then
    PY="$ROOT/.venv/bin/python"
else
    PY="$(command -v python3 || command -v python || true)"
fi

if [ -z "${PY:-}" ]; then
    echo "No Python found. Install Python 3.11+ or create the venv." >&2
    exit 1
fi

# From a terminal: run inline so output/errors are visible. Launched from a
# file manager (no tty): redirect to a log so a failure isn't silent.
if [ -t 1 ]; then
    exec "$PY" -m openflo.gui "$@"
else
    exec "$PY" -m openflo.gui "$@" >>"${TMPDIR:-/tmp}/openflo-gui.log" 2>&1
fi
