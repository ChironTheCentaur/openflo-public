"""Two threads running numba parallel kernels at once must not kill the app.

numba's ``workqueue`` threading layer -- the one a Mac gets, having neither
OpenMP nor TBB -- calls ``abort()`` when a second thread launches a parallel
kernel while one is running. OpenFlo does exactly that: the embedding warm-up
compiles UMAP in a background thread while a real embedding may be running.
openflo/numba_threads.py makes those calls take turns under ``workqueue``.

The crash is reproduced here on any OS by forcing ``workqueue`` in a fresh
interpreter. Unfixed, that interpreter dies of SIGABRT; fixed, it finishes,
and every embedding it made from two threads equals the one made serially.

The second half pins coverage: every call into a numba-parallel library, and
every ``parallel=True`` kernel of OpenFlo's own, must go through the lock, so a
new one cannot reintroduce the crash unnoticed.
"""
from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / 'src' / 'openflo'

_CONCURRENT = textwrap.dedent('''
    import threading

    import numpy as np
    import pandas as pd

    from openflo import dr_compare, warmup
    from openflo.pipeline import FlowSample

    CH = ['FL1-A', 'FL2-A', 'FL3-A', 'FL4-A']
    # Above UMAP's 4096-row switch to NN-descent, whose kernels are parallel.
    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(c, 1.0, (1700, len(CH))) for c in (0, 6, 12)])

    def via_pipeline():
        s = FlowSample.from_dataframe(pd.DataFrame(X, columns=CH), name='s1')
        s.run_umap(channels=CH, n_neighbors=10, random_state=7)
        return np.asarray(s.umap_coords)

    def via_compare():
        return dr_compare._embed_one('umap', X, 3, n_neighbors=10,
                                     n_epochs=30)

    ROUNDS = 3
    got = {'pipeline': [], 'compare': []}
    errors = []

    def loop(key, fn, warm):
        try:
            if warm:
                warmup._warm()        # the real background compile
            for _ in range(ROUNDS):
                got[key].append(fn())
        except BaseException as e:    # noqa: BLE001
            errors.append(repr(e))

    ts = [threading.Thread(target=loop, args=('pipeline', via_pipeline, True)),
          threading.Thread(target=loop, args=('compare', via_compare, False))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors, errors
    import numba
    assert numba.threading_layer() == 'workqueue', numba.threading_layer()

    want = {'pipeline': via_pipeline(), 'compare': via_compare()}
    for key, runs in got.items():
        assert len(runs) == ROUNDS, (key, len(runs))
        for r in runs:
            assert r.shape == (len(X), 2), r.shape
            assert np.isfinite(r).all()
            assert np.array_equal(r, want[key]), key
    print('CONCURRENT-OK')
''')


def test_parallel_kernels_from_two_threads_under_workqueue():
    pytest.importorskip('umap', reason='umap-learn not installed')
    env = dict(os.environ, NUMBA_THREADING_LAYER='workqueue')
    r = subprocess.run([sys.executable, '-c', _CONCURRENT], env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (
        f'returncode {r.returncode} (-6 is SIGABRT: numba workqueue saw two '
        f'threads in parallel kernels at once)\n{r.stderr[-3000:]}')
    assert 'CONCURRENT-OK' in r.stdout, r.stdout[-2000:]


# --- coverage guard ---------------------------------------------------------

# Constructors of libraries whose fit runs numba parallel=True kernels.
_PARALLEL_LIBS = {'UMAP', 'TRIMAP', 'PaCMAP', 'NNDescent'}
_JITS = {'jit', 'njit', 'vectorize', 'guvectorize'}


def _name(node):
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _is_lock(node):
    """``parallel_kernels()`` (a with-item) or ``serialized`` (a decorator)."""
    if isinstance(node, ast.Call):
        node = node.func
    return _name(node) in ('parallel_kernels', 'serialized')


def _unguarded(source, filename='<src>'):
    """``file:line what`` for each numba-parallel call site not under the
    lock: a library constructor in a function with no ``parallel_kernels()``
    block and no ``@serialized``, or a ``parallel=True`` jit anywhere but in
    the decorator list of a ``@serialized`` function."""
    tree = ast.parse(source, filename)
    bad, ok_jits = [], set()
    funcs = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for fn in funcs:
        if any(_is_lock(d) for d in fn.decorator_list):
            ok_jits.update(id(d) for d in fn.decorator_list)
    for fn in funcs:
        locked = any(_is_lock(d) for d in fn.decorator_list) or any(
            isinstance(n, (ast.With, ast.AsyncWith))
            and any(_is_lock(i.context_expr) for i in n.items)
            for n in ast.walk(fn))
        for n in ast.walk(fn):
            if (isinstance(n, ast.Call) and _name(n.func) in _PARALLEL_LIBS
                    and not locked):
                bad.append(f'{filename}:{n.lineno} {_name(n.func)}(...)')
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and _name(n.func) in _JITS
                and id(n) not in ok_jits
                and any(k.arg == 'parallel' and isinstance(k.value, ast.Constant)
                        and k.value.value is True for k in n.keywords)):
            bad.append(f'{filename}:{n.lineno} {_name(n.func)}(parallel=True)')
    return bad


def _sources():
    return {p.name: p.read_text(encoding='utf-8') for p in SRC.rglob('*.py')}


def test_every_numba_parallel_call_site_takes_the_lock():
    bad = [b for name, src in _sources().items()
           for b in _unguarded(src, name)]
    assert not bad, (
        'numba-parallel code reachable without openflo.numba_threads: under '
        'the workqueue layer (macOS) two threads here abort the process. Wrap '
        'the call in `with numba_threads.parallel_kernels():` or decorate the '
        'kernel with `@numba_threads.serialized`:\n  ' + '\n  '.join(bad))


def test_the_guard_finds_the_known_call_sites():
    """Guards the guard: one that recognised nothing would pass vacuously."""
    found = sum(1 for src in _sources().values()
                for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Call) and _name(n.func) in _PARALLEL_LIBS)
    # warmup, three in pipeline, three in dr_compare.
    assert found >= 7, found


def test_the_guard_flags_unwrapped_code():
    assert _unguarded('def f(X):\n    return umap.UMAP().fit_transform(X)\n')
    assert _unguarded('@numba.njit(parallel=True)\ndef k(a):\n    pass\n')
    assert _unguarded('k = numba.njit(parallel=True)(f)\n')
    assert not _unguarded(
        'def f(X):\n    r = umap.UMAP()\n'
        '    with numba_threads.parallel_kernels():\n'
        '        return r.fit_transform(X)\n')
    assert not _unguarded(
        '@numba_threads.serialized\n@njit(parallel=True)\n'
        'def k(a):\n    pass\n')
    assert not _unguarded('@numba.njit\ndef k(a):\n    pass\n')


def test_no_lock_when_the_layer_is_threadsafe():
    """A thread-safe layer must not pay for the lock (and workqueue must)."""
    code = ('from openflo import numba_threads as nt\n'
            'import numba\n'
            'cm = nt.parallel_kernels()\n'
            'print(numba.threading_layer(), cm is nt._LOCK)\n')
    out = {}
    for layer in ('workqueue', 'omp'):
        env = dict(os.environ, NUMBA_THREADING_LAYER=layer)
        r = subprocess.run([sys.executable, '-c', code], env=env,
                           capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            continue                      # layer not on this machine
        out[layer] = r.stdout.split()
    assert out.get('workqueue') == ['workqueue', 'True'], out
    if 'omp' in out:
        assert out['omp'] == ['omp', 'False'], out
