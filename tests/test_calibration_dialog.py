"""MESF / ABC calibration dialog (Tools -> Calibration…) — the action path.

calibration.py is tested directly; CalibrationDialog's Detect / Fit / Apply
handlers had 5-28% coverage. Ground truth: six bead populations at known
linear MFI (CV 3%) each assigned MESF = 3 x MFI, and a 'cells' sample whose
every event sits at MFI 2000, so its MESF must be 6000.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.conftest import gui_unavailable

PEAKS = np.array([50, 300, 1500, 6000, 25000, 100000], float)
K = 3.0


def _editor_or_skip():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
    except ImportError:
        gui_unavailable("tkinter not available — headless environment")
    try:
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        gui_unavailable(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    import importlib
    gui = importlib.import_module('openflo.gui')
    gui.messagebox.askyesno = lambda *a, **k: True
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    return root, ed


def _frames(seed=3):
    rng = np.random.default_rng(seed)
    beads = np.concatenate([rng.normal(p, 0.03 * p, 800) for p in PEAKS])
    cells = np.full(500, 2000.0)

    def df(v):
        return pd.DataFrame({'FSC-A': np.full(len(v), 7e4),
                             'SSC-A': np.full(len(v), 7e4), 'FL4-A': v})
    return df(beads), df(cells)


def _register(ed, name, df):
    s = SimpleNamespace(name=name, path=None, data=df,
                        fluor_channels=['FL4-A'],
                        channel_labels={c: c for c in df.columns})
    ed._samples[name] = s
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    if 'T' not in ed._trial_order:
        ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = True
    if len(ed._samples) == 1:
        ed._channels = list(df.columns)
        ed._channel_labels = {c: c for c in df.columns}
        ed._populate_channel_combos()


def _run_dialog(ed):
    from openflo.ui_calibration import CalibrationDialog
    dlg = CalibrationDialog(ed)
    dlg.bead_var.set('beads')
    dlg._sync_channels()
    dlg._detect()
    shown = [ln.strip() for ln in
             dlg.txt.get('1.0', 'end').strip().splitlines()]
    dlg.txt.delete('1.0', 'end')
    dlg.txt.insert('1.0', '\n'.join(f"{s}\t{K * p:g}"
                                    for s, p in zip(shown, PEAKS, strict=True)))
    dlg._fit()
    dlg.apply_var.set(dlg.chan_var.get())
    dlg._apply()
    return dlg, shown


def test_calibration_dialog_detect_fit_apply_on_linear_channel():
    root, ed = _editor_or_skip()
    try:
        audits = []
        ed._audit = lambda ev, **k: audits.append((ev, k))
        beads, cells = _frames()
        _register(ed, 'beads', beads)
        _register(ed, 'cells', cells)
        dlg, shown = _run_dialog(ed)
        assert len(shown) == 6
        np.testing.assert_allclose([float(s) for s in shown], PEAKS,
                                   rtol=0.02)
        assert dlg._cal is not None and dlg._cal['r2'] > 0.999
        assert dlg._cal['slope'] == pytest.approx(K, rel=0.01)
        got = ed._samples['cells'].data['MESF:FL4-A'].to_numpy(float)
        np.testing.assert_allclose(got, 6000.0, rtol=0.005)
        assert "Applied to 2 sample(s) → 'MESF:FL4-A'" in dlg.result.cget(
            'text')
        assert audits and audits[-1][0] == 'calibration'
        assert audits[-1][1]['n_samples'] == 2
    finally:
        root.destroy()


def test_calibration_dialog_refuses_degenerate_assignment():
    root, ed = _editor_or_skip()
    try:
        beads, _ = _frames()
        _register(ed, 'beads', beads)
        from openflo.ui_calibration import CalibrationDialog
        dlg = CalibrationDialog(ed)
        dlg.txt.insert('1.0', '100\t500\n1000, 500\n10000 500\nnot a row\n')
        assert dlg._parse() == [(100.0, 500.0), (1000.0, 500.0),
                                (10000.0, 500.0)]
        dlg._fit()
        assert dlg._cal is None
        assert 'degenerate' in dlg.result.cget('text')
        dlg._apply()
        assert dlg.result.cget('text') == "Fit a calibration first."
        assert 'MESF:FL4-A' not in ed._samples['beads'].data.columns
    finally:
        root.destroy()


def test_calibration_on_loader_processed_samples_gives_true_mesf(tmp_path):
    """The loader logicle-transforms FL4-A; peaks, fit and MESF column must
    still come from the linear values (MESF is linear in linear MFI)."""
    from openflo.fcs_export import write_fcs
    root, ed = _editor_or_skip()
    try:
        beads, cells = _frames()
        for nm, df in (('beads', beads), ('cells', cells)):
            write_fcs(df, str(tmp_path / f'{nm}.fcs'))
            ed._load_worker(nm, str(tmp_path / f'{nm}.fcs'))
        for _ in range(100):
            root.update()
            if len(ed._samples) == 2:
                break
        assert len(ed._samples) == 2
        dlg, shown = _run_dialog(ed)
        np.testing.assert_allclose([float(s) for s in shown], PEAKS,
                                   rtol=0.02)
        assert dlg._cal is not None and dlg._cal['r2'] > 0.999
        assert dlg._cal['slope'] == pytest.approx(K, rel=0.01)
        got = ed._samples['cells'].data['MESF:FL4-A'].to_numpy(float)
        assert np.median(got) == pytest.approx(6000.0, rel=0.05)
    finally:
        root.destroy()
