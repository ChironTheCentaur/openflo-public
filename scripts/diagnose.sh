#!/usr/bin/env bash
# OpenFlo install health check (macOS / Linux). Run from a terminal.
# Reports whether dependencies, engines, and core behaviour are within norms.
# Use this when the GUI won't start or behaves oddly.
set -u
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
echo "Running OpenFlo diagnostics with $PY ..."
echo
"$PY" -m openflo.diagnostics "$@"
rc=$?
echo
echo "(exit code $rc: 0 = healthy, non-zero = issues found)"
exit "$rc"
