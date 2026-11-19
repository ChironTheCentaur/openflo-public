"""MESF / ABC calibration dialog (Tools -> Calibration…) — the action path.

calibration.py is tested directly; CalibrationDialog's Detect / Fit / Apply
handlers had 5-28% coverage. Ground truth: six bead populations at known
linear MFI (CV 3%) each assigned MESF = 3 x MFI, and a 'cells' sample whose
every event sits at MFI 2000, so its MESF must be 6000.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests._tk_wait import pump_until
from tests.conftest import editor_or_skip

PEAKS = np.array([50, 300, 1500, 6000, 25000, 100000], float)
K = 3.0


def _editor_or_skip():
    return editor_or_skip()


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
    # These checks are of the straight-line fit; log-log is the default
    # (Complaints 25) and is checked in test_loglog_mesf.
    dlg.model_var.set('linear')
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
    # These samples carry no acquisition keywords, so the batch is the
    # beads alone; the whole (All) tab is every sample, as before D5.
    dlg.scope_var.set('tab')
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
        # Bead peaks: 800 events at CV 3% each; detection lands within 0.25% of the
        # true MFI on this seed, so 2% is a loose bound.
        np.testing.assert_allclose([float(s) for s in shown], PEAKS,
                                   rtol=0.02)
        assert dlg._cal is not None and dlg._cal['r2'] > 0.999
        # Slope fitted to six detected peaks (each off by < 0.25%); measured 0.1% off.
        assert dlg._cal['slope'] == pytest.approx(K, rel=0.01)
        got = ed._samples['cells'].data['MESF:FL4-A'].to_numpy(float)
        np.testing.assert_allclose(got, 6000.0, rtol=0.005)
        assert ("Applied to 2 sample(s) in this tab → 'MESF:FL4-A'"
                in dlg.result.cget('text'))
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
        assert pump_until(root, lambda: len(ed._samples) == 2)
        dlg, shown = _run_dialog(ed)
        # Same beads through the loader's logicle transform: peaks within 0.25% here.
        np.testing.assert_allclose([float(s) for s in shown], PEAKS,
                                   rtol=0.02)
        assert dlg._cal is not None and dlg._cal['r2'] > 0.999
        # Slope from the six detected peaks; measured 0.1% off K.
        assert dlg._cal['slope'] == pytest.approx(K, rel=0.01)
        got = ed._samples['cells'].data['MESF:FL4-A'].to_numpy(float)
        # Cells sit exactly at MFI 2000; after the loader round trip and the fit the
        # median is within 0.03% of 6000, so 5% is far looser than needed.
        assert np.median(got) == pytest.approx(6000.0, rel=0.05)
    finally:
        root.destroy()


def test_calibration_fit_names_the_lines_it_leaves_out():
    """COMP-5: a mistyped line ('1x000') or a non-positive one on log-log
    was dropped without a word; the fit ran on the rest with R² = 1.0000.
    The fit now names each line it left out. Generated values: known =
    5·MFI exactly, so the fit on the remaining lines is still exact."""
    root, ed = _editor_or_skip()
    try:
        beads, _ = _frames()
        _register(ed, 'beads', beads)
        from openflo.ui_calibration import CalibrationDialog
        dlg = CalibrationDialog(ed)
        dlg.model_var.set('linear')
        dlg.txt.insert('1.0', '100\t500\n1x000\t5000\n10000\t50000\n'
                              '400\n\n20000 nan\n')
        dlg._fit()
        text = dlg.result.cget('text')
        assert text.endswith(
            "  Left out: line 2: “1x000” is not a number; line 4 has no "
            "value; line 6: “nan” is not a number.")
        assert dlg._cal['n'] == 2 and dlg._cal['slope'] == pytest.approx(5.0)
        dlg.model_var.set('loglog')
        dlg.txt.delete('1.0', 'end')
        dlg.txt.insert('1.0', '100\t500\n-5\t25\n1000\t5000\n10000\t0\n')
        dlg._fit()
        text = dlg.result.cget('text')
        assert "n=2" in text and text.endswith(
            "  Left out: line 2: log-log needs values above 0; line 4: "
            "log-log needs values above 0.")
        # Nothing left out -> no note.
        dlg.txt.delete('1.0', 'end')
        dlg.txt.insert('1.0', '100\t500\n1000\t5000\n')
        dlg._fit()
        assert 'Left out' not in dlg.result.cget('text')
        dlg.destroy()
    finally:
        root.destroy()


def test_switching_the_model_after_a_fit_refits_with_it():
    """Fit with Linear, then click Log-log: the result line and Add MESF
    column use the log-log fit, not the old straight line."""
    root, ed = _editor_or_skip()
    try:
        beads, cells = _frames()
        _register(ed, 'beads', beads)
        _register(ed, 'cells', cells)
        from openflo.ui_calibration import CalibrationDialog
        dlg = CalibrationDialog(ed)
        # Planted on a power law: value = 3 * MFI ** 1.1, exact.
        mfi = PEAKS
        val = K * mfi ** 1.1
        dlg.txt.insert('1.0', '\n'.join(f"{m:g}\t{v:.10g}"
                                        for m, v in zip(mfi, val,
                                                        strict=True)))
        buttons = {w.cget('text'): w for w in _walk(dlg)
                   if w.winfo_class() == 'TRadiobutton'}
        buttons['Linear'].invoke()
        dlg._fit()
        assert dlg._cal['model'] == 'linear'
        assert dlg.result.cget('text').startswith('value = ')
        buttons['Log-log'].invoke()
        assert dlg.model_var.get() == 'loglog'
        assert dlg._cal['model'] == 'loglog'
        # Worked here: log10(value) = 1.1 log10(MFI) + log10(3).
        assert dlg._cal['slope'] == pytest.approx(1.1, rel=1e-9)
        assert dlg._cal['intercept'] == pytest.approx(np.log10(K), rel=1e-9)
        assert dlg.result.cget('text').startswith('log value = 1.1')
        dlg.apply_var.set(dlg.chan_var.get())
        dlg.scope_var.set('tab')
        dlg._apply()
        got = ed._samples['cells'].data['MESF:FL4-A'].to_numpy(float)
        np.testing.assert_allclose(got, K * 2000.0 ** 1.1, rtol=1e-9)
        # A switch the lines can't be fitted with leaves no fit to apply.
        dlg.txt.delete('1.0', 'end')
        dlg.txt.insert('1.0', '100\t300\n')
        buttons['Linear'].invoke()
        assert dlg._cal is None
        dlg._apply()
        assert dlg.result.cget('text') == "Fit a calibration first."
    finally:
        root.destroy()


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)
