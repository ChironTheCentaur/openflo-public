"""Pipeline Workspace run path, end to end through a REAL child process.

WorkspacePanel._run -> _pump_jobs -> _launch_next -> _finish_prep ->
`python -m openflo.workspace job.pkl result.pkl` (_subprocess_main ->
compute_run) -> _service_current -> _finalize_current is the code behind the
Workspace's Run button. The suite covers compute_run in-process and the
staging state machine with hand-built records, but never launches the child,
reads its result back, retries a crash, or cancels a live child.

The driver methods are bound to a minimal Tk-free runner (the pattern
tests/test_workspace_prep_offthread.py uses); `after` is pumped by hand.
Only prepare_unit (which reads the editor) is replaced, by a prep holding a
frame with a KNOWN answer: sample s1 = 300 events of population A, sample s2
= 200 events of population B, four markers, separated by > 15 SD.
"""
from __future__ import annotations

import collections
import os
import subprocess
import sys
import time
import types

import numpy as np
import pandas as pd
import pytest

from openflo import workspace as W

P = W.WorkspacePanel


class _Runner:
    def __init__(self, out_dir):
        self._cur = None
        self._prep = None
        self._jobs = collections.deque()
        self._err = self._ok = self._idx = 0
        self._total = 0
        self._cancel_requested = False
        self._editor = object()
        self._out_dir = out_dir
        self.running = None
        self._after = []
        self.log = []
        self.status_var = types.SimpleNamespace(
            set=self.log.append, get=lambda: self.log[-1] if self.log else '')

    def after(self, _ms, fn):
        self._after.append(fn)

    def _set_running(self, flag):
        self.running = flag

    _pump_jobs = P._pump_jobs
    _launch_next = P._launch_next
    _finish_prep = P._finish_prep
    _read_stdout = staticmethod(P._read_stdout)
    _absorb_stdout = P._absorb_stdout
    _service_current = P._service_current
    _finalize_current = P._finalize_current
    _kill_current = P._kill_current
    _finish = P._finish
    _cancel = P._cancel
    _running = P._running

    def queue(self, cfg, label='unit'):
        self._jobs.append({'unit': {'members': [], 'label': label},
                           'label': label, 'attempt': 1, 'cfg': dict(cfg)})
        self._total = len(self._jobs)
        self._after.append(self._pump_jobs)

    def pump(self, timeout, on_tick=None):
        end = time.monotonic() + timeout
        while self._after and time.monotonic() < end:
            fn = self._after.pop(0)
            fn()
            if on_tick:
                on_tick(self)
            time.sleep(0.05)
        return not self._after


def _prep(label, *_a):
    rng = np.random.default_rng(3)
    a = rng.normal(0, 0.1, (300, 4))
    b = rng.normal(2, 0.1, (200, 4))
    df = pd.DataFrame(np.vstack([a, b]), columns=['M1', 'M2', 'M3', 'M4'])
    df['__group__'] = pd.Categorical(['(ungrouped)'] * 500)
    df['__sample__'] = pd.Categorical(['s1'] * 300 + ['s2'] * 200)
    return {'data': df, 'channels': ['M1', 'M2', 'M3', 'M4'], 'label': label,
            'note': '2 sample(s)', 'n_events': 500, 'color_by': '__sample__'}


CFG = {'method': 'flowsom', 'n_metaclusters': 2, 'seed': 42, 'k': 15,
       'max_events': 0, 'umap': False, 'tsne': False, 'phate': False,
       'trimap': False, 'pacmap': False}


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setattr(W, 'prepare_unit',
                        lambda editor, members, label, cfg: _prep(label))
    return _Runner(str(tmp_path / 'run'))


def test_run_button_path_produces_known_clusters(runner):
    runner.queue(CFG, 'unitA')
    assert runner.pump(timeout=180), 'driver did not finish'
    assert runner.running is False
    assert runner.log[-1].startswith('Done: 1 ok'), runner.log
    assert any('running… (separate process)' in m for m in runner.log)
    # progress lines streamed from the child's stdout reached the status bar
    assert any('flowsom clustering 500 cells on 4 markers' in m
               for m in runner.log), runner.log
    assert any('✓ 2 clusters, 500 ev' in m for m in runner.log), runner.log
    out = runner._out_dir
    freq = pd.read_csv(os.path.join(out, 'unitA_clusters.csv'))
    assert sorted(freq['count']) == [200, 300]
    ct = pd.read_csv(os.path.join(out, 'unitA_cluster_by_sample.csv'),
                     index_col=0)
    # each cluster is 100 % one sample and 0 % the other
    assert sorted(map(tuple, ct[['s1', 's2']].to_numpy().tolist())) == \
        [(0.0, 100.0), (100.0, 0.0)]
    ev = pd.read_csv(os.path.join(out, 'unitA_events.csv'))
    assert len(ev) == 500 and 'cluster' in ev
    # the per-event cluster agrees with the sample of origin
    assert ev.groupby('__sample__')['cluster'].nunique().tolist() == [1, 1]
    # the job temp dir is removed after the result is read back
    assert runner._cur is None and runner._prep is None


def _fake_child(cmd):
    real = subprocess.Popen

    def popen(args, *a, **k):
        if isinstance(args, list) and args[1:3] == ['-m', 'openflo.workspace']:
            args = [sys.executable, '-c', cmd]
        return real(args, *a, **k)
    return popen


def test_a_crashing_child_is_retried_once_at_fewer_events(runner, monkeypatch):
    monkeypatch.setattr(subprocess, 'Popen',
                        _fake_child('import sys; print("boom"); sys.exit(3)'))
    runner.queue(dict(CFG, max_events=5000), 'unitC')
    assert runner.pump(timeout=60)
    joined = '\n'.join(runner.log)
    assert 'crashed (exit 3) — boom — retrying once at max_events=2500' in joined
    assert '✗ no result (exit 3) — boom' in joined
    assert runner.log[-1].startswith('Done: 0 ok, 1 failed')


def test_cancel_kills_the_running_child_and_its_tree(runner, monkeypatch,
                                                     tmp_path):
    """Cancel must take down the child AND what it spawned (PhenoGraph's
    pool / Louvain binaries) — the reason _kill_current uses taskkill /T."""
    psutil = pytest.importorskip('psutil')
    pidfile = tmp_path / 'grandchild.pid'
    child = ('import subprocess, sys, time\n'
             'g = subprocess.Popen([sys.executable, "-c", '
             '"import time; time.sleep(120)"])\n'
             f'open({str(pidfile)!r}, "w").write(str(g.pid))\n'
             'time.sleep(120)\n')
    monkeypatch.setattr(subprocess, 'Popen', _fake_child(child))
    runner.queue(CFG, 'unitK')
    seen = {}

    def tick(r):
        if r._cur is not None and 'proc' not in seen and pidfile.exists() \
                and pidfile.read_text():
            seen['proc'] = r._cur['proc']
            seen['gpid'] = int(pidfile.read_text())
            r._cancel()
    t0 = time.monotonic()
    try:
        assert runner.pump(timeout=60, on_tick=tick)
        assert 'proc' in seen, 'child never started'
        assert runner.log[-1].startswith('Cancelled.')
        assert seen['proc'].poll() is not None, 'child still alive after cancel'
        time.sleep(1.0)
        g = seen['gpid']
        alive = psutil.pid_exists(g) and psutil.Process(g).status() != \
            psutil.STATUS_ZOMBIE
        assert not alive, 'grandchild survived cancel (tree not killed)'
        assert time.monotonic() - t0 < 30
    finally:
        g = seen.get('gpid')
        if g and psutil.pid_exists(g):
            try:
                psutil.Process(g).kill()
            except psutil.Error:
                pass
