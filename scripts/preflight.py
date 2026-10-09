#!/usr/bin/env python
"""Pre-push gate that mirrors CI, so a green run here means green CI.

CI (.github/workflows/ci.yml) runs on BOTH ubuntu-latest and windows-latest, and
its bare `pyright` step type-checks against each runner's OWN OS. Running pyright
once locally only covers your OS, so platform-specific branches — ctypes.windll
(Windows-only) or os.gettid / os.PRIO_PROCESS / os.setpriority (Unix-only) — slip
past a local check and fail CI. This runs pyright for BOTH platforms (the gap
that caused the "passes local, fails CI" streak), plus ruff and the test suite.

    python scripts/preflight.py            # ruff + pyright(Linux,Windows) + tests
    python scripts/preflight.py --quick    # lint + type-check only (skip tests)

Exits non-zero if any step fails. Run from the repo root.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

PY = sys.executable
# Headless + UTF-8 so the Tk GUI tests run without a display (CI uses Xvfb on
# Linux; on Windows/locally the Agg backend stands in).
ENV = {**os.environ, "MPLBACKEND": "Agg", "PYTHONUTF8": "1"}


def _step(title, cmd):
    print(f"\n=== {title} ===\n  $ {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    rc = subprocess.run(cmd, env=ENV).returncode
    print(f"  -> {'OK' if rc == 0 else 'FAIL'} ({time.perf_counter() - t0:.0f}s)",
          flush=True)
    return rc == 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="lint + type-check only (skip the test suite)")
    args = ap.parse_args(argv)

    if not os.path.isdir(".git"):
        sys.exit("Run from the repo root (no .git here).")

    steps = [
        ("Lint (ruff)", [PY, "-m", "ruff", "check", "."]),
        ("Type check — Linux (CI parity)",
         [PY, "-m", "pyright", "--pythonplatform", "Linux", "src"]),
        ("Type check — Windows (CI parity)",
         [PY, "-m", "pyright", "--pythonplatform", "Windows", "src"]),
    ]
    if not args.quick:
        steps.append(("Tests (pytest)", [PY, "-m", "pytest", "-q"]))

    results = [(title, _step(title, cmd)) for title, cmd in steps]

    print("\n===== preflight summary =====")
    for title, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {title}")
    if all(p for _, p in results):
        print("All green — safe to push (matches CI).")
        return 0
    print("FAILURES above — fix before pushing.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
