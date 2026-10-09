"""FMO gating dialog (Gate tools -> FMO gating…) — the Apply action.

gating_helpers.fmo_threshold(_gate) is tested directly; FMOGatingDialog._apply
was 44% covered. Ground truth: an FMO control whose CD3 distribution is known,
so the gate value must be exactly np.percentile(FMO CD3, pct).
"""
from __future__ import annotations

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


def _register(ed, name, df):
    s = SimpleNamespace(name=name, path=None, data=df,
                        fluor_channels=['CD3', 'CD4'],
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


def _data(seed=4):
    rng = np.random.default_rng(seed)
    fmo = pd.DataFrame({
        'FSC-A': np.r_[rng.normal(0.6, 0.05, 9000),
                       rng.normal(0.1, 0.03, 1000)],
        'CD3': np.r_[rng.normal(0.2, 0.05, 9000),
                     rng.normal(0.6, 0.05, 1000)],         # bright debris
        'CD4': rng.normal(0.2, 0.05, 10000)})
    stained = pd.DataFrame({
        'FSC-A': rng.normal(0.6, 0.05, 10000),
        'CD3': np.r_[rng.normal(0.2, 0.05, 5000),
                     rng.normal(0.65, 0.03, 5000)],
        'CD4': rng.normal(0.2, 0.05, 10000)})
    return fmo, stained


def _setup(ed):
    fmo, stained = _data()
    _register(ed, 'stained', stained)
    _register(ed, 'fmo_cd3', fmo)
    ed._set_active_sample('stained')
    return fmo, stained


def test_fmo_dialog_places_gate_at_fmo_percentile():
    root, ed = _editor_or_skip()
    try:
        fmo, _ = _setup(ed)
        from openflo.ui_fmo import FMOGatingDialog
        dlg = FMOGatingDialog(ed)
        dlg._pct.set('99.5')
        dlg._map['CD3'].set('fmo_cd3')              # CD4 left '(none)'
        dlg._apply()
        gates = [g for g in ed._gates.values() if g.get('kind') == 'threshold']
        assert len(gates) == 1
        assert gates[0]['channel'] == 'CD3'
        assert gates[0]['value'] == pytest.approx(
            float(np.percentile(fmo['CD3'], 99.5)), abs=1e-12)
        assert ed.status_var.get() == (
            "Added 1 FMO threshold gate(s) at the 99.5th percentile to "
            "stained.")
    finally:
        root.destroy()


def test_fmo_dialog_rejects_out_of_range_percentile():
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        from openflo.ui_fmo import FMOGatingDialog
        dlg = FMOGatingDialog(ed)
        dlg._pct.set('150')
        dlg._map['CD3'].set('fmo_cd3')
        dlg._apply()
        assert not ed._gates
        assert ed.status_var.get() == "FMO percentile must be between 0 and 100."
    finally:
        root.destroy()


def test_fmo_dialog_without_mapping_adds_nothing():
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        from openflo.ui_fmo import FMOGatingDialog
        dlg = FMOGatingDialog(ed)
        dlg._apply()
        assert not ed._gates
        assert ed.status_var.get() == "No FMO mappings chosen — nothing added."
    finally:
        root.destroy()


def test_fmo_gate_follows_selected_parent_gate():
    """With a parent gate selected (here 'cells', excluding bright debris),
    the FMO gate goes under it and its cutoff is the FMO percentile INSIDE
    that parent. Taken over every FMO event, the debris set the cutoff at
    0.662 instead of 0.317 and called 16.8% of a 50%-positive sample CD3+."""
    root, ed = _editor_or_skip()
    try:
        fmo, _ = _setup(ed)
        pid = ed._add_gate({'kind': 'threshold', 'channel': 'FSC-A',
                            'value': 0.3, 'op': '>=', 'parent_id': None,
                            'name': 'cells'})
        ed._refresh_gate_list()
        ed.gate_tv.selection_set(ed._gate_iid('stained', pid))
        root.update()
        assert ed._selected_gate_id() == pid
        from openflo.ui_fmo import FMOGatingDialog
        dlg = FMOGatingDialog(ed)
        dlg._map['CD3'].set('fmo_cd3')
        dlg._apply()
        g = [g for g in ed._gates.values() if g.get('channel') == 'CD3'][0]
        assert g['parent_id'] == pid
        inside = fmo['FSC-A'] >= 0.3
        assert g['value'] == pytest.approx(
            float(np.percentile(fmo.loc[inside, 'CD3'], 99)), abs=1e-9)
        assert ed.status_var.get() == (
            "Added 1 FMO threshold gate(s) at the 99th percentile to "
            "stained under cells.")
    finally:
        root.destroy()
