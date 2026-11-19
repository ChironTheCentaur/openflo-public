#!/usr/bin/env python
"""Regenerate the import-cost figures quoted in openflo/__init__.py and
openflo/pipeline.py.

Both files defer heavy dependencies — __init__ through a PEP 562 __getattr__,
pipeline through a module-level __getattr__ — and both justify that with a
number. Numbers in comments rot, so this makes them reproducible.

MEASURE BY WALL CLOCK, minus bare interpreter startup. An earlier attempt used
`python -X importtime` and took max() of the cumulative column, which is the
largest single import TREE, not the total: adding a second heavy module to the
command then appeared to cost nothing at all, and the resulting "saving" was
off by a factor of two. If you change this script, keep the subtraction and
keep the median — process startup is noisy enough to swamp small differences.

Run:  python scripts/bench_import_cost.py [--repeats N]
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time

CASES = {
    'import openflo': 'import openflo',
    'import numpy, pandas': 'import numpy, pandas',
    'import openflo.pipeline': 'import openflo.pipeline',
    'import matplotlib.pyplot': 'import matplotlib.pyplot',
    'openflo.pipeline + pyplot': 'import openflo.pipeline, matplotlib.pyplot',
}


def wall(code, repeats):
    xs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        subprocess.run([sys.executable, '-c', code], capture_output=True)
        xs.append((time.perf_counter() - t0) * 1000)
    return statistics.median(xs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--repeats', type=int, default=7)
    args = ap.parse_args()

    base = wall('pass', args.repeats)
    print(f'  bare interpreter startup: {base:6.1f} ms  (subtracted below)\n')

    res = {}
    for label, code in CASES.items():
        res[label] = wall(code, args.repeats) - base
        print(f'  {label:30} {res[label]:7.1f} ms')

    saved = res['openflo.pipeline + pyplot'] - res['import openflo.pipeline']
    print(f'\n  deferring matplotlib.pyplot saves : {saved:7.1f} ms')
    print(f'  `import openflo` against the stack: '
          f'{res["import openflo"]:.0f} ms vs '
          f'{res["import numpy, pandas"]:.0f} ms for numpy+pandas')

    # The property, not the timing: tests/test_import_is_lazy.py pins this too,
    # but a benchmark that silently measured an eager import would be
    # meaningless, so check it here as well.
    r = subprocess.run(
        [sys.executable, '-c',
         'import openflo, sys; '
         'print(",".join(m for m in ("numpy","pandas","scipy","matplotlib") '
         'if m in sys.modules))'],
        capture_output=True, text=True)
    leaked = r.stdout.strip()
    print(f'  heavy modules pulled in by `import openflo`: {leaked or "none"}')
    if leaked:
        print('  WARNING: the laziness these numbers describe is gone.')


if __name__ == '__main__':
    main()
