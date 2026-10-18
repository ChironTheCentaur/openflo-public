"""`import openflo` must not drag in the scientific stack.

openflo/__init__.py exposes its public surface through PEP 562 `__getattr__`
so that importing the package is cheap. Measured on this machine: `import
openflo` is ~81 ms, while numpy + pandas alone is ~330 ms and
`import openflo.pipeline` is ~506 ms.

The timing is not what this test pins — timings rot, and they differ by
machine and by dependency version. The PROPERTY is what matters and what can
regress silently: one stray top-level `import numpy` in __init__.py, or in
anything it imports, and the laziness is gone with nothing to say so. The CLI
still starts, the tests still pass, and every entry point just got slower.

Run in a subprocess because this test session has already imported half the
world; `sys.modules` in-process would answer the wrong question entirely.
"""
from __future__ import annotations

import subprocess
import sys

# Deferred on purpose. numpy is the one that costs the most and pulls the rest.
HEAVY = ('numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'umap',
         'phenograph', 'igraph', 'numba')


def _modules_after(code):
    """Import something in a FRESH interpreter, return what got loaded."""
    r = subprocess.run(
        [sys.executable, '-c',
         f'{code}\nimport sys, json\n'
         f'print(json.dumps(sorted(m for m in sys.modules)))'],
        capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-2000:]
    import json
    return set(json.loads(r.stdout.splitlines()[-1]))


def test_importing_openflo_does_not_load_the_scientific_stack():
    loaded = _modules_after('import openflo')
    leaked = sorted(m for m in HEAVY if m in loaded)
    assert not leaked, (
        f'`import openflo` pulled in {leaked}. The PEP 562 lazy surface in '
        f'openflo/__init__.py exists to avoid exactly this — something now '
        f'imports it at module level, and every entry point pays for it.')


def test_the_public_surface_still_works_through_the_lazy_path():
    """Guards the opposite failure: laziness achieved by exporting nothing.

    A __init__.py that imported nothing would pass the test above and be
    useless, so check that a deferred name actually resolves and that
    reaching it does load its module."""
    loaded = _modules_after(
        'import openflo\n'
        'fn = openflo.fit_mesf_calibration\n'
        'assert callable(fn), fn')
    assert 'numpy' in loaded, (
        'touching a public name did not load its module — the lazy surface '
        'resolved to something that is not the real implementation')


def test_every_advertised_name_resolves():
    """All of them, not a sample.

    A registry is only pinned if every entry is exercised — the same lesson
    the QC denylist taught, where three entries sat unprotected because the
    fixture never produced their columns.

    Touching all 66 names imports the whole scientific stack, which is exactly
    what the laziness tests above must not do, so this runs in its own
    subprocess.
    """
    code = "\n".join([
        "import openflo",
        "bad = []",
        "for name in sorted(openflo._PUBLIC):",
        "    try:",
        "        getattr(openflo, name)",
        "    except Exception as e:",
        "        bad.append(name + ' -> ' + openflo._PUBLIC[name] + ': '",
        "                   + type(e).__name__ + ': ' + str(e)[:80])",
        "print(len(openflo._PUBLIC))",
        "print('|'.join(bad))",
    ])
    r = subprocess.run([sys.executable, '-c', code], capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]

    lines = r.stdout.strip().splitlines()
    count = int(lines[0])
    broken = [b for b in (lines[1] if len(lines) > 1 else '').split('|') if b]

    # Guards the vacuous pass: an empty or tiny registry would report no
    # breakage while checking nothing.
    assert count > 50, (
        f'_PUBLIC advertises only {count} names — either the public surface '
        f'shrank unexpectedly or this test is reading the wrong object')
    assert not broken, (
        'names advertised by openflo._PUBLIC that do not resolve:\n  '
        + '\n  '.join(broken))
