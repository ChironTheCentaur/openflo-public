"""Sample QC window (Analysis -> Sample QC) — values, not just guards.

tests/test_ui_sample_qc.py checks the empty-selection guards; nothing checks
that the window ranks a constructed batch effect correctly or that its CSV /
AnnData exports carry the computed numbers.

Ground truth: four samples, two batches. Batch B is batch A shifted by +1.0
in CD3 and CD4 (on the display scale), so every within-batch distance must be
smaller than every cross-batch distance, and MDS must separate the batches
along one axis.
"""
from __future__ import annotations

import importlib
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.conftest import gui_unavailable


def _window_or_skip():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        gui_unavailable(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    gui = importlib.import_module('openflo.gui')
    gui.messagebox.askyesno = lambda *a, **k: True
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    rng = np.random.default_rng(3)
    for _i, (nm, shift, trial) in enumerate((('a1', 0.0, 'A'), ('a2', 0.0, 'A'),
                                            ('b1', 1.0, 'B'),
                                            ('b2', 1.0, 'B'))):
        df = pd.DataFrame({'FSC-A': rng.normal(5e4, 5e3, 2000),
                           'CD3': rng.normal(1.0 + shift, 0.2, 2000),
                           'CD4': rng.normal(2.0 + shift, 0.2, 2000)})
        ed._samples[nm] = SimpleNamespace(
            name=nm, path=None, data=df, fluor_channels=['CD3', 'CD4'],
            channel_labels={c: c for c in df.columns})
        ed._sample_order.append(nm)
        ed._sample_plot_enabled[nm] = True
        ed._sample_trial[nm] = trial
        ed._sample_colors[nm] = '#1f77b4'
    ed._trial_order[:] = ['A', 'B']
    from openflo.ui_sample_qc import SampleQCWindow
    win = SampleQCWindow(ed)
    win.withdraw()
    return root, ed, win


def test_sample_qc_separates_constructed_batches_and_exports(tmp_path,
                                                             monkeypatch):
    root, ed, win = _window_or_skip()
    try:
        win._compute()
        names, D, xy = list(win._names), np.asarray(win._D), win._xy
        assert names == ['a1', 'a2', 'b1', 'b2']
        within = [D[0, 1], D[2, 3]]
        cross = [D[i, j] for i in (0, 1) for j in (2, 3)]
        assert max(within) < 0.2 * min(cross)
        np.testing.assert_allclose(D, D.T)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)
        # the MDS axis that separates the batches
        gap = np.abs(xy[:2].mean(0) - xy[2:].mean(0)).max()
        spread = max(np.abs(xy[0] - xy[1]).max(), np.abs(xy[2] - xy[3]).max())
        assert gap > 5 * spread

        import openflo.ui_sample_qc as mod
        out = tmp_path / 'd.csv'
        monkeypatch.setattr(mod, 'ask_csv_path', lambda *a, **k: str(out))
        monkeypatch.setattr(mod.messagebox, 'showinfo', lambda *a, **k: None)
        win._export_csv()
        back = pd.read_csv(out, index_col=0)
        assert list(back.index) == names and list(back.columns) == names
        np.testing.assert_allclose(back.to_numpy(), D)
    finally:
        root.destroy()


def test_sample_qc_anndata_export_carries_every_event(tmp_path, monkeypatch):
    anndata = pytest.importorskip('anndata')
    root, ed, win = _window_or_skip()
    try:
        import openflo.ui_sample_qc as mod
        out = tmp_path / 'x.h5ad'
        monkeypatch.setattr(mod.filedialog, 'asksaveasfilename',
                            lambda *a, **k: str(out))
        monkeypatch.setattr(mod.messagebox, 'showinfo', lambda *a, **k: None)
        win._export_h5ad()
        ad = anndata.read_h5ad(out)
        assert ad.shape == (8000, 2)
        assert list(ad.var_names) == ['CD3', 'CD4']
        x = np.asarray(ad.X)
        expect = np.vstack([ed._samples[n].data[['CD3', 'CD4']].to_numpy()
                            for n in ('a1', 'a2', 'b1', 'b2')])
        np.testing.assert_allclose(np.sort(x, axis=0), np.sort(expect, axis=0),
                                   rtol=1e-6)
    finally:
        root.destroy()
