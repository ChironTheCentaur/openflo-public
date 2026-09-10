"""The embedding warm-up must be invisible except in the clock.

It exists to move ~18s of numba compilation off the critical path, into the
window where clustering is running. Compilation cannot change a result — the
same function is called with the same inputs either way — so the only way this
module could corrupt a run is by disturbing shared state on its way there.
The global numpy RNG is the one piece of shared state it could plausibly
touch, and a run that draws from it would then depend on whether a background
thread happened to get there first: irreproducible, and irreproducible in a
way that changes between machines and between runs on the same machine.

So that is what these pin, along with "never raises" and "never wastes work".
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from openflo import warmup


@pytest.fixture(autouse=True)
def _reset():
    """The module is deliberately once-per-process; tests need it fresh."""
    warmup._done = False
    if warmup._started.locked():
        warmup._started.release()
    yield
    warmup._done = False
    if warmup._started.locked():
        warmup._started.release()


@pytest.mark.parametrize('cfg,expected', [
    (None, False),
    ({}, False),
    ({'umap': False, 'tsne': True}, False),   # t-SNE is not numba-backed
    ({'umap': True}, True),
    ({'trimap': True}, True),
    ({'pacmap': True}, True),
])
def test_only_warms_when_an_embedding_is_coming(cfg, expected):
    assert warmup.wants_embedding(cfg) is expected


def test_no_thread_started_when_nothing_would_use_it():
    """A run with no embedding must not burn a core compiling kernels it will
    never call."""
    assert warmup.start({'tsne': True}) is None
    assert warmup.start(None) is None


def test_second_call_in_a_process_is_a_no_op():
    first = warmup.start({'umap': True}, force=True)
    second = warmup.start({'umap': True}, force=True)
    assert first is not None
    assert second is None, 'warm-up started twice in one process'
    first.join(timeout=180)


def test_the_global_numpy_rng_is_untouched():
    """The load-bearing one. If the warm-up drew from np.random.* it would
    advance the global stream, and any later code that uses it — in this
    process, during the same run — would silently get different numbers
    depending on thread timing.

    Sabotage check: change `rng.normal` to `np.random.normal` in warmup._warm
    and this test goes red.
    """
    np.random.seed(12345)
    before = np.random.get_state()

    t = warmup.start({'umap': True})
    assert t is not None
    t.join(timeout=300)
    assert not t.is_alive(), 'warm-up did not finish in time'

    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1]), (
        'the warm-up advanced the global numpy RNG; a run drawing from it '
        'would depend on background-thread timing')
    assert before[2:] == after[2:]


def test_failure_is_swallowed(monkeypatch):
    """Its only job is to make something faster. Failing at that must cost
    time, never the run."""
    import builtins
    real_import = builtins.__import__

    def _boom(name, *a, **kw):
        if name == 'umap':
            raise ImportError('simulated: umap-learn not installed')
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, '__import__', _boom)
    t = warmup.start({'umap': True})
    assert t is not None
    t.join(timeout=60)
    assert not t.is_alive()
    assert warmup._done is False        # it failed, and said so quietly


def test_numba_thread_masking_is_thread_local():
    """An assumption the warm-up rests on, pinned so it cannot change quietly.

    UMAP brackets every fit with:

        self._original_n_threads = numba.get_num_threads()
        numba.set_num_threads(self.n_jobs)      # 1, when random_state is set
        ...
        numba.set_num_threads(self._original_n_threads)

    If that mask were process-global, a warm-up thread would throttle the whole
    process to one thread mid-compile, and two overlapping fits would race over
    a single variable — the losing interleaving leaves the process pinned at 1
    thread for good, which would look like an unexplained permanent slowdown
    rather than a bug.

    Measured on numba 0.65.1: the mask is thread-local, so neither can happen.
    If this test ever fails, the warm-up must join before any real embedding
    starts (see run_umap) rather than running alongside it.
    """
    import threading

    import numba

    baseline = numba.get_num_threads()
    if baseline < 2:
        pytest.skip('needs more than one numba thread to distinguish')

    observed = {}
    holding = threading.Event()
    release = threading.Event()

    def worker():
        numba.set_num_threads(1)
        observed['worker'] = numba.get_num_threads()
        holding.set()
        release.wait(timeout=10)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    assert holding.wait(timeout=10), 'worker never set its thread count'
    observed['main_while_worker_holds_one'] = numba.get_num_threads()
    release.set()
    t.join(timeout=10)

    assert observed['worker'] == 1
    assert observed['main_while_worker_holds_one'] == baseline, (
        'numba.set_num_threads is process-global in this version: a background '
        'warm-up now throttles the whole process and can race UMAP\'s '
        'save/restore. The warm-up must be joined before any real embedding.')
    assert numba.get_num_threads() == baseline


def test_quitting_during_a_warm_up_exits_cleanly(tmp_path):
    """The realistic interruption: EmbeddingDialog opens, the warm-up starts,
    and the user closes the app a second later — or hits Cancel, which
    taskkills the whole process tree.

    A daemon thread caught mid-execution by interpreter shutdown is a classic
    source of hangs and shutdown tracebacks, and this one is running a numba
    compile. So: the process must exit promptly, with status 0, and print no
    traceback at the user.
    """
    import subprocess
    import time

    script = tmp_path / 'quit_during_warmup.py'
    script.write_text(
        'import sys, time, warnings\n'
        'warnings.filterwarnings("ignore")\n'
        f'sys.path.insert(0, {str(_repo_src())!r})\n'
        'from openflo import warmup\n'
        't = warmup.start({"umap": True})\n'
        'assert t is not None\n'
        'time.sleep(1.0)\n'
        'print("EXIT", t.is_alive())\n',
        encoding='utf-8')

    start = time.perf_counter()
    r = subprocess.run([sys.executable, str(script)], capture_output=True,
                       text=True, timeout=180,
                       env={**os.environ, 'MPLBACKEND': 'Agg',
                            'PYTHONUTF8': '1'})
    elapsed = time.perf_counter() - start

    assert r.returncode == 0, f'exited {r.returncode}\n{r.stderr[-1500:]}'
    assert 'EXIT True' in r.stdout, (
        'the warm-up finished before the process quit, so this ran the easy '
        f'case and proved nothing: {r.stdout!r}')
    assert 'Traceback' not in r.stderr, (
        f'shutdown printed a traceback at the user:\n{r.stderr[-1500:]}')
    assert elapsed < 60, (
        f'took {elapsed:.1f}s to quit — the warm-up is blocking exit; it must '
        f'be a daemon thread')


def _repo_src():
    import pathlib
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / 'src' / 'openflo').is_dir():
            return str(parent / 'src')
    raise AssertionError('repo src not found')
