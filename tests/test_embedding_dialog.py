"""Analyze -> Compare embeddings... runs the ticked method on the chosen
subsample and shows a faithful layout.

ui_embedding.EmbeddingDialog._run and editor_compute._start_embedding were
never executed by the suite (EmbeddingDialog is in test_dialog_tooltips'
_UNCONSTRUCTIBLE set). Three blobs 8 SD apart give a known answer: in a
faithful 2-D layout every point's nearest neighbour is in its own blob.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(__file__))

from test_gui_smoke import _editor_or_skip  # noqa: E402


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def _sync(work, on_done=None, on_error=None, busy_msg=None):
    try:
        r = work()
    except Exception as exc:                      # noqa: BLE001
        if on_error:
            on_error(exc)
        return
    if on_done:
        on_done(r)


def test_compare_embeddings_dialog_runs_tsne_on_the_subsample(monkeypatch):
    import tkinter as tk
    from tkinter import ttk

    from sklearn.neighbors import NearestNeighbors

    import openflo.dr_compare as drc
    import openflo.warmup as warmup
    from openflo.pipeline import FlowSample
    monkeypatch.setattr(warmup, 'start', lambda *a, **k: None)  # no numba compile
    # Probing every backend imports umap/phate/trimap/pacmap (~15 s); the
    # dialog only needs to know t-SNE is there.
    monkeypatch.setattr(drc, 'available_methods', lambda: ['tsne'])
    seen = {}
    real = drc.run_embeddings

    def spy(X, **kw):
        seen['kw'] = kw
        seen['res'] = real(X, **kw)
        return seen['res']
    monkeypatch.setattr(drc, 'run_embeddings', spy)

    rng = np.random.default_rng(0)
    truth = np.repeat([0, 1, 2], 300)
    X = np.vstack([rng.normal(c, 1.0, (300, 4)) for c in (0.0, 8.0, 16.0)])
    df = pd.DataFrame(X, columns=[f'M{i}-A' for i in range(4)])
    df['cluster'] = truth
    root, ed, _gui = _editor_or_skip()
    try:
        s = FlowSample.from_dataframe(df, name='blobs')
        ed._samples['blobs'] = s
        ed._sample_order.append('blobs')
        ed._sample_trial['blobs'] = 'T'
        ed._sample_gates['blobs'] = {}
        ed._sample_gate_order['blobs'] = []
        ed._active_sample = 'blobs'
        ed.run_async = _sync
        ed._open_dr_compare()
        dlg = next(w for w in ed.winfo_children() if isinstance(w, tk.Toplevel)
                   and w.title() == 'Compare embeddings')
        for cb in (w for w in _walk(dlg) if isinstance(w, ttk.Checkbutton)):
            want = str(cb.cget('text')).startswith('TSNE')
            if bool(ed.getvar(cb.cget('variable'))) != want:
                cb.invoke()
        next(w for w in _walk(dlg) if isinstance(w, ttk.Spinbox)).set('300')
        next(w for w in _walk(dlg) if isinstance(w, ttk.Button)
             and str(w.cget('text')) == 'Run').invoke()

        assert seen['kw']['methods'] == ('tsne',)
        assert seen['kw']['max_points'] == 300
        res = seen['res']
        idx = res['index']
        assert len(idx) == 300 and np.all(np.diff(idx) > 0)
        xy = res['coords']['tsne']
        assert xy.shape == (300, 2) and np.isfinite(xy).all()
        nn = NearestNeighbors(n_neighbors=2).fit(xy)
        j = nn.kneighbors(xy, return_distance=False)[:, 1]
        assert np.mean(truth[idx][j] == truth[idx]) >= 0.98
        assert 'tsne' in ed.status_var.get()
        assert any(isinstance(w, tk.Toplevel) and
                   w.title().startswith('Embedding comparison')
                   for w in ed.winfo_children())
        assert not ed._dr_running
    finally:
        root.destroy()
