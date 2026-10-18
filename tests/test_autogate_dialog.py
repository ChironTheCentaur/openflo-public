"""Auto-gate, end to end through its dialog.

Coverage before: AutoGateDialog.__init__ ran 1/60 statements and _apply
1/25; AutoGateMixin._auto_gate 1/10, _auto_gate_gmm 1/25,
_auto_gate_threshold 1/17. The pipeline routines are tested
(tests/test_autogate.py), but nothing checked that the dialog's options reach
them or that proposals land under the selected parent.

Path: _auto_gate -> AutoGateDialog -> _apply(opts) -> _run_auto_gate ->
_auto_gate_{singlet,gmm,threshold} -> _add_gate_multi.

Ground truth: synthetic events with known labels -- two Gaussian populations
on CD3 x CD4, and singlets (FSC-A ~ FSC-H) plus 15% doublets (FSC-A ~ 1.9 FSC-H).
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.conftest import gui_unavailable


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


def _data(seed=11):
    rng = np.random.default_rng(seed)
    na, nb = 6000, 4000
    a = rng.multivariate_normal([1000, 1200], [[150 ** 2, 15000], [15000, 200 ** 2]], na)
    b = rng.multivariate_normal([5000, 4000], [[400 ** 2, 0], [0, 300 ** 2]], nb)
    xy = np.vstack([a, b])
    lab = np.r_[np.zeros(na, int), np.ones(nb, int)]
    n = na + nb
    h = rng.uniform(2e4, 1.5e5, n)
    doublet = rng.random(n) < 0.15
    ratio = np.where(doublet, rng.normal(1.9, 0.1, n), rng.normal(1.0, 0.04, n))
    df = pd.DataFrame({'FSC-A': h * ratio, 'FSC-H': h,
                       'SSC-A': rng.uniform(1e4, 1e5, n),
                       'CD3': xy[:, 0], 'CD4': xy[:, 1]})
    return df, lab, doublet


def _setup(ed):
    df, lab, doublet = _data()
    cols = list(df.columns)
    ed._samples['s1'] = SimpleNamespace(
        name='s1', path=r'C:\exp\s1.fcs', data=df, fluor_channels=['CD3', 'CD4'],
        channel_labels={c: c for c in cols})
    ed._sample_order.append('s1')
    ed._sample_colors['s1'] = '#1f77b4'
    ed._sample_trial['s1'] = 'T'
    ed._trial_order.append('T')
    ed._sample_plot_enabled['s1'] = True
    ed._channels = cols
    ed._channel_labels = {c: c for c in cols}
    ed._populate_channel_combos()
    ed._set_active_sample('s1')
    return df, lab, doublet


def _run(ed, method, **vars_):
    from openflo.ui_autogate import AutoGateDialog
    ed._auto_gate()
    (dlg,) = [w for w in ed.winfo_children() if isinstance(w, AutoGateDialog)]
    dlg.method_var.set(method)
    dlg._sync()
    for k, v in vars_.items():
        getattr(dlg, k).set(v)
    before = set(ed._gates)
    dlg._apply()
    assert not dlg.winfo_exists()                     # dialog closes on Propose
    return [g for g in ed._gates if g not in before]


def test_autogate_singlet_from_dialog_keeps_singlets_drops_doublets():
    import openflo.pipeline as fp
    root, ed = _editor_or_skip()
    try:
        df, _lab, doublet = _setup(ed)
        assert ed._find_area_height_channels() == ('FSC-A', 'FSC-H')
        (gid,) = _run(ed, 'singlet', k_var='3.0')
        g = ed._gates[gid]
        assert (g['kind'], g['x_channel'], g['y_channel'], g['name']) == (
            'polygon', 'FSC-A', 'FSC-H', 'Singlets')
        m = fp.gate_to_mask(g, df)
        assert m[~doublet].mean() > 0.99
        assert m[doublet].mean() < 0.01
        # The band width the user typed reaches the fit: k=1 keeps far fewer.
        (gid1,) = _run(ed, 'singlet', k_var='1.0')
        assert fp.gate_to_mask(ed._gates[gid1], df)[~doublet].mean() < 0.8
    finally:
        root.destroy()


def test_autogate_gmm_from_dialog_finds_both_populations_at_requested_coverage():
    import openflo.pipeline as fp
    root, ed = _editor_or_skip()
    try:
        df, lab, _d = _setup(ed)
        ed.x_combo.set('CD3')
        ed.y_combo.set('CD4')
        new = _run(ed, 'gmm', cov_var='90', kmax_var='6', minw_var='2')
        assert len(new) == 2
        # (fraction of A inside, fraction of B inside) per proposed ellipse,
        # sorted so B's ellipse (A fraction 0) comes first.
        covers = sorted(
            (fp.gate_to_mask(ed._gates[g], df)[lab == 0].mean(),
             fp.gate_to_mask(ed._gates[g], df)[lab == 1].mean()) for g in new)
        # One ellipse per population, each enclosing ~90% of it and none of
        # the other.
        assert covers[0][0] == 0.0 and covers[0][1] == pytest.approx(0.90, abs=0.02)
        assert covers[1][1] == 0.0 and covers[1][0] == pytest.approx(0.90, abs=0.02)
        assert all(ed._gates[g]['parent_id'] is None for g in new)
    finally:
        root.destroy()


def test_autogate_gmm_fits_inside_and_attaches_to_the_selected_parent():
    import openflo.pipeline as fp
    root, ed = _editor_or_skip()
    try:
        df, lab, _d = _setup(ed)
        ed.x_combo.set('CD3')
        ed.y_combo.set('CD4')
        pid = ed._add_gate({'kind': 'rect', 'x_channel': 'CD3', 'y_channel': 'CD4',
                            'x0': 0, 'x1': 2500, 'y0': 0, 'y1': 2500,
                            'name': 'A-box', 'parent_id': None})
        ed._refresh_gate_list()
        ed.gate_tv.selection_set(ed._gate_iid('s1', pid))
        (gid,) = _run(ed, 'gmm', cov_var='90')
        assert ed._gates[gid]['parent_id'] == pid
        m = np.asarray(fp.cumulative_gate_mask(ed._gates, gid, df))
        assert m[lab == 0].mean() == pytest.approx(0.90, abs=0.02)
        assert m[lab == 1].sum() == 0
    finally:
        root.destroy()


def test_autogate_threshold_from_dialog_splits_the_bimodal_channel():
    import openflo.pipeline as fp
    root, ed = _editor_or_skip()
    try:
        df, lab, _d = _setup(ed)
        ed.x_combo.set('CD3')
        (gid,) = _run(ed, 'threshold')
        g = ed._gates[gid]
        assert (g['kind'], g['channel']) == ('threshold', 'CD3')
        assert 1600 < g['value'] < 4000
        m = fp.gate_to_mask(g, df)
        assert (m == (lab == 1)).all()
        # Linear channel: the value is the intensity, written as the axis
        # writes it ('1.73K'; it read '1.73e+03').
        assert ed.status_var.get() == (
            f"Auto threshold on CD3 = {g['value'] / 1000:.3g}K.")
    finally:
        root.destroy()


def test_autogate_threshold_status_gives_the_linear_cut_on_logicle():
    """The cut is a coordinate on the stored scale. On a logicle channel the
    status said 'Auto threshold on CD3 = 0.507' -- read as an intensity of
    0.5 for a cut at 1,667 -- so it now says the intensity the cut sits at,
    written as the axis and the gate list write it ('1.67K')."""
    import openflo.pipeline as fp
    root, ed = _editor_or_skip()
    try:
        df, lab, _d = _setup(ed)
        s = fp.FlowSample.from_dataframe(df, name='s1')
        s.fluor_channels = ['CD3', 'CD4']
        s.apply_transform()                 # as the editor's loader stores it
        ed._samples['s1'] = s
        ed.x_combo.set('CD3')
        (gid,) = _run(ed, 'threshold')
        g = ed._gates[gid]
        lin = float(fp.inverse_transform_values(np.array([g['value']]))[0])
        assert 1600 < lin < 4000
        assert (fp.gate_to_mask(g, s.data) == (lab == 1)).all()
        assert ed.status_var.get() == (
            f"Auto threshold on CD3 = {lin / 1000:.3g}K.")
        assert ed._gate_text('s1', g) == f"T  CD3 >= {lin / 1000:.3g}K"
    finally:
        root.destroy()
