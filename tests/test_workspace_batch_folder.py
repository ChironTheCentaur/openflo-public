"""Pipeline Workspace > Batch folder… (PipelineWorkspaceView._batch_folder,
2 % covered): "Run the current settings over every .fcs in a folder".

Known answer: two files, each 60 % population A / 40 % population B
(well separated); FlowSOM with 2 metaclusters must report exactly those
counts per file, and every file must get its clusters CSV and per-event CSV.
The run bar's "max ev" setting ("caps memory/time") must apply here too.
"""
from __future__ import annotations

import time
import types

import numpy as np
import pandas as pd

from openflo import workspace as W

CH = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A']
MK = ['', '', 'CD901b', 'CD902', 'CD903']
N = 1500


def _write(path, seed):
    import flowio
    rng = np.random.default_rng(seed)
    n1 = int(N * .6)
    ev = np.column_stack([
        rng.lognormal(10, .2, N), rng.lognormal(9, .2, N),
        np.r_[rng.normal(20000, 1500, n1), rng.normal(150, 40, N - n1)],
        np.r_[rng.normal(150, 40, n1), rng.normal(20000, 1500, N - n1)],
        rng.normal(800, 100, N)]).astype(np.float32)
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.flatten().tolist(), CH, opt_channel_names=MK)


class _View:
    def __init__(self, cfg):
        self.model = types.SimpleNamespace(run_cfg=cfg)
        self._editor = None
        self.queued, self.log = [], []

    def _sync_run_cfg(self):
        pass

    def after(self, _ms, fn):
        self.queued.append(fn)

    def _status(self, msg):
        self.log.append(msg)

    _batch_folder = W.PipelineWorkspaceView._batch_folder


def _run(tmp_path, monkeypatch, cfg):
    import tkinter.filedialog as fd
    import tkinter.messagebox as mb
    indir, outdir = tmp_path / 'in', tmp_path / 'out'
    indir.mkdir()
    outdir.mkdir()
    for i in (1, 2):
        _write(str(indir / f's{i}.fcs'), i)
    answers = [str(indir), str(outdir)]
    monkeypatch.setattr(fd, 'askdirectory', lambda **k: answers.pop(0))
    monkeypatch.setattr(mb, 'askyesno', lambda *a, **k: True)
    monkeypatch.setattr(W, 'ResultsViewer', lambda *a, **k: None)
    v = _View(dict(W.default_run_cfg(), **cfg))
    v._batch_folder()
    end = time.monotonic() + 120
    while time.monotonic() < end and not any(m.startswith('Batch done')
                                             for m in v.log):
        while v.queued:
            v.queued.pop(0)()
        time.sleep(0.05)
    return v, outdir


CFG = {'method': 'flowsom', 'n_metaclusters': 2, 'umap': False,
       'trimap': False, 'max_events': 0}


def test_batch_folder_clusters_every_file(tmp_path, monkeypatch):
    v, out = _run(tmp_path, monkeypatch, CFG)
    assert v.log[-1].startswith('Batch done: 2 ok, 0 failed'), v.log
    for i in (1, 2):
        lab = W.run_label({'sample': f's{i}'})
        freq = pd.read_csv(out / f'{lab}_clusters.csv')
        assert sorted(freq['count']) == [600, 900]
        ev = pd.read_csv(out / f'{lab}_events.csv')
        assert len(ev) == N
        # loaded like the editor does: compensated + logicle (0..~1 scale),
        # not raw detector values (~20000)
        assert ev['FL1-A'].max() < 1.5


def test_batch_folder_honours_max_events(tmp_path, monkeypatch):
    """Batch folder… built its own frame and never applied the cap, while
    compute_run assumes the caller already did, so 'max ev' was ignored."""
    v, out = _run(tmp_path, monkeypatch, dict(CFG, max_events=500))
    assert v.log[-1].startswith('Batch done: 2 ok, 0 failed'), v.log
    for i in (1, 2):
        lab = W.run_label({'sample': f's{i}'})
        assert int(pd.read_csv(out / f'{lab}_clusters.csv')['count'].sum()) == 500
        ev = pd.read_csv(out / f'{lab}_events.csv')
        assert len(ev) == 500
        # The capped frame still carries its transform record: the events
        # CSV's fluor columns are read back as logicle, not taken as linear.
        from openflo.pipeline import read_transforms_sidecar
        rec = read_transforms_sidecar(str(out / f'{lab}_events.csv'),
                                      list(ev.columns), len(ev))
        assert (rec or {}).get('FL1-A', {}).get('method') == 'logicle', rec
