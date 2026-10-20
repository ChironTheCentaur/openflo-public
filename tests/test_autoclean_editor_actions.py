"""Auto-clean gate — the editor actions (Auto-clean button, right-click
Debris method / bead reference, Edit auto-clean parameters…).

The pure masks are tested in tests/test_autoclean_methods.py, test_qc.py and
test_autoclean_counts.py; the editor layer that edits a recipe and attributes
removed events per method (_autoclean_set_param / _set_debris_mode /
_method_masks / _edit_autoclean_params) had 3-58% coverage.

Ground truth (linear scatter, as the editor holds it):
  'dirty': 8000 cells FSC-A ~ N(100k, 8k), A/H = 1.33; 1000 debris FSC-A
           ~ U(2k, 15k); 1000 doublets FSC-A ~ N(190k, 10k), A/H = 2.5.
           Event order in Time is shuffled (anomalies interspersed).
  'clean': 10000 cells only.
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.conftest import gui_unavailable


def _editor_or_skip():
    os.environ.setdefault('MPLBACKEND', 'Agg')
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


def _frames(seed=9, doublets=True):
    rng = np.random.default_rng(seed)

    def cells(n):
        a = rng.normal(100_000, 8_000, n)
        return (a, a / 1.33 * (1 + rng.normal(0, 0.02, n)),
                rng.normal(50_000, 8_000, n))
    ca, ch, cs = cells(8000)
    da = rng.uniform(2_000, 15_000, 1000)
    parts_a, parts_h, parts_s = [ca, da], [ch, da / 1.33], [
        cs, rng.uniform(1_000, 10_000, 1000)]
    truth = ['cell'] * 8000 + ['debris'] * 1000
    if doublets:
        wa = rng.normal(190_000, 10_000, 1000)
        parts_a.append(wa)
        parts_h.append(wa / 2.5)
        parts_s.append(rng.normal(50_000, 8_000, 1000))
        truth += ['doublet'] * 1000
    n = len(truth)
    dirty = pd.DataFrame({'FSC-A': np.concatenate(parts_a),
                          'FSC-H': np.concatenate(parts_h),
                          'SSC-A': np.concatenate(parts_s),
                          'CD3-A': rng.normal(1000, 100, n),
                          'Time': rng.permutation(n).astype(float)})
    ca, ch, cs = cells(10000)
    clean = pd.DataFrame({'FSC-A': ca, 'FSC-H': ch, 'SSC-A': cs,
                          'CD3-A': rng.normal(1000, 100, 10000),
                          'Time': np.arange(10000, dtype=float)})
    return dirty, clean, np.array(truth)


def _register(ed, name, df):
    ed._samples[name] = SimpleNamespace(
        name=name, path=None, data=df, fluor_channels=['CD3-A'],
        channel_labels={c: c for c in df.columns})
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


def _ac_gid(ed, name):
    return next(g for g, d in ed._sample_gates[name].items()
                if d.get('kind') == 'autoclean')


def _setup(ed, doublets=True):
    dirty, clean, truth = _frames(doublets=doublets)
    _register(ed, 'dirty', dirty)
    _register(ed, 'clean', clean)
    ed._target_samples = lambda mode='selected': ['dirty', 'clean']
    ed._set_active_sample('dirty')
    ed._create_autoclean_gate()
    return dirty, truth


def test_autoclean_removes_constructed_doublets_only_and_attributes_them():
    root, ed = _editor_or_skip()
    try:
        dirty, truth = _setup(ed)
        assert "Added 'autocleaned sample' to 2 samples." in \
            ed.status_var.get()
        total, drop, per, _r = ed._autoclean_counts('dirty', _ac_gid(ed, 'dirty'))
        assert total == 10000 and per['doublets'] == 1000
        rm = ed._autoclean_method_masks('dirty')['doublets'].to_numpy()
        np.testing.assert_array_equal(rm, truth == 'doublet')
        _t, _d, per_c, _r = ed._autoclean_counts('clean', _ac_gid(ed, 'clean'))
        assert per_c['doublets'] == 0 and per_c['debris'] == 0
        assert per_c['margin'] == 0
        # a second press does not stack a second recipe
        ed._create_autoclean_gate()
        assert 'already on' in ed.status_var.get()
        assert sum(d.get('kind') == 'autoclean'
                   for d in ed._sample_gates['dirty'].values()) == 1
    finally:
        root.destroy()


def test_set_param_manual_floor_recomputes_and_clears():
    root, ed = _editor_or_skip()
    try:
        dirty, truth = _setup(ed)
        gid = _ac_gid(ed, 'dirty')
        ed._autoclean_set_param('dirty', gid, 'debris', min_fsc=50_000.0)
        assert ed._autoclean_counts('dirty', gid)[2]['debris'] == 1000
        rm = ed._autoclean_method_masks('dirty')['debris'].to_numpy()
        np.testing.assert_array_equal(rm, truth == 'debris')
        ed._autoclean_set_param('dirty', gid, 'debris', min_fsc=None)
        _g, m = ed._autoclean_method('dirty', gid, 'debris')
        assert 'min_fsc' not in m['params']
    finally:
        root.destroy()


def test_debris_mode_switch_and_bead_reference():
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        gid = _ac_gid(ed, 'dirty')
        ed._autoclean_set_debris_mode('dirty', gid, 'valley')
        _g, m = ed._autoclean_method('dirty', gid, 'debris')
        assert m['params']['mode'] == 'valley'
        msg = ed.status_var.get()
        assert msg.startswith("Debris → valley") and 'small real cells' in msg
        ed._autoclean_set_debris_mode('dirty', gid, 'bead')
        assert 'no bead file is loaded' in ed.status_var.get()
        assert 'nothing is cut' in ed.status_var.get()
        assert 'bead_fsc' not in m['params']
        # load 8 µm beads at FSC-A 60k -> 4 µm floor = 30k FSC-A
        rng = np.random.default_rng(1)
        _register(ed, 'beads_8um', pd.DataFrame({
            'FSC-A': rng.normal(60_000, 1_000, 2000),
            'FSC-H': rng.normal(45_000, 800, 2000),
            'SSC-A': rng.normal(20_000, 500, 2000),
            'CD3-A': rng.normal(100, 10, 2000),
            'Time': np.arange(2000, dtype=float)}))
        ed._autoclean_set_debris_mode('dirty', gid, 'bead')
        assert ed.status_var.get() == "Debris → beads ‘beads_8um’."
        bead_fsc = float(np.median(ed._samples['beads_8um'].data['FSC-A']))
        assert m['params']['bead_fsc'] == pytest.approx(bead_fsc)
        floor = 4.0 * bead_fsc / 8.0
        expect = int((ed._samples['dirty'].data['FSC-A'] < floor).sum())
        assert ed._autoclean_counts('dirty', gid)[2]['debris'] == expect == 1000
    finally:
        root.destroy()


def _walk(w):
    for c in w.winfo_children():
        yield c
        yield from _walk(c)


def test_the_gui_says_what_the_debris_method_really_does():
    """With no bead file the debris method cuts nothing; the create status,
    the gate-list tag and its reason must say so, and must follow the mode
    the recipe actually holds. Every test passed with the old, now false,
    texts put back ("debris uses the auto-valley cut", "[beads→valley: no
    bead file]"), and a typed 'Valley' cut nothing under a [valley] tag."""
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        status = ed.status_var.get()
        assert 'debris cuts nothing' in status and 'auto-valley' not in status
        gid = _ac_gid(ed, 'dirty')
        _g, m = ed._autoclean_method('dirty', gid, 'debris')

        def row(mode=None):
            if mode is not None:
                m['params']['mode'] = mode
                ed._autoclean_invalidate('dirty', gid)
            ed._refresh_gate_list()
            return (ed.gate_tv.item(ed._method_iid('dirty', gid, 'debris'),
                                    'text'),
                    ed._autoclean_counts('dirty', gid)[2]['debris'])

        text, n = row()
        assert '[beads: no bead file, nothing cut]' in text and n == 0
        assert 'no bead size anchor' in text
        text, n = row('Valley')               # as the dialog once stored it
        assert '[valley]' in text and n >= 950
        text, n = row('vally')
        assert '[valley]' not in text and 'unknown mode' in text and n == 0
    finally:
        root.destroy()


def test_params_dialog_offers_the_debris_mode_as_a_fixed_choice():
    """Free text let 'Valley' through. The mode is a read-only choice of
    bead or valley, shown normalised for an old session; an old session's
    unknown mode is refused until one is chosen."""
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        gid = _ac_gid(ed, 'dirty')
        _g, m = ed._autoclean_method('dirty', gid, 'debris')
        ed.wait_window = lambda *a, **k: None      # don't block the test

        def open_dialog(stored):
            m['params']['mode'] = stored            # as an old session holds it
            ed._edit_autoclean_params('dirty', gid)
            dlg = [w for w in ed.winfo_children()
                   if isinstance(w, tk.Toplevel)
                   and w.title() == "Auto-clean parameters"][-1]
            deb = next(f for f in _walk(dlg) if isinstance(f, ttk.Labelframe)
                       and str(f.cget('text')).startswith('Debris'))
            box = next(w for w in _walk(deb) if isinstance(w, ttk.Combobox))
            apply_btn = next(w for w in _walk(dlg)
                             if isinstance(w, ttk.Button)
                             and str(w.cget('text')) == 'Apply')
            return dlg, box, apply_btn

        dlg, box, apply_btn = open_dialog('bead')
        assert str(box.cget('state')) == 'readonly'
        assert tuple(box.cget('values')) == ('bead', 'valley')
        box.set('valley')
        apply_btn.invoke()
        assert m['params']['mode'] == 'valley' and not dlg.winfo_exists()

        dlg, box, apply_btn = open_dialog('Valley')
        assert box.get() == 'valley'
        dlg.destroy()

        dlg, box, apply_btn = open_dialog('vally')
        apply_btn.invoke()
        assert m['params']['mode'] == 'vally' and dlg.winfo_exists()
        errs = [dlg.getvar(str(w.cget('textvariable'))) for w in _walk(dlg)
                if isinstance(w, ttk.Label) and str(w.cget('textvariable'))]
        assert any("'mode'" in e for e in errs), errs
        box.set('bead')
        apply_btn.invoke()
        assert m['params']['mode'] == 'bead'
    finally:
        root.destroy()


def test_edit_autoclean_params_dialog_writes_recipe():
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        gid = _ac_gid(ed, 'dirty')
        ed.wait_window = lambda *a, **k: None      # don't block the test
        ed._edit_autoclean_params('dirty', gid)
        dlg = next(w for w in ed.winfo_children()
                   if isinstance(w, tk.Toplevel)
                   and w.title() == "Auto-clean parameters")
        frames = {str(f.cget('text')): f for f in _walk(dlg)
                  if isinstance(f, ttk.Labelframe)}
        dbl = frames['Doublets (FSC-A/FSC-H)']
        entries = [w for w in _walk(dbl) if isinstance(w, ttk.Entry)]
        assert len(entries) == 1                   # tol
        entries[0].delete(0, 'end')
        entries[0].insert(0, '0.1')
        drift = frames['Signal drift']
        chk = next(w for w in _walk(drift) if isinstance(w, ttk.Checkbutton))
        chk.invoke()                               # disable drift
        apply_btn = next(w for w in _walk(dlg) if isinstance(w, ttk.Button)
                         and str(w.cget('text')) == 'Apply')
        apply_btn.invoke()
        _g, m = ed._autoclean_method('dirty', gid, 'doublets')
        assert m['params']['tol'] == pytest.approx(0.1)
        _g, d = ed._autoclean_method('dirty', gid, 'drift')
        assert d['enabled'] is False
        assert 'drift' not in ed._autoclean_method_masks('dirty')
        assert ed.status_var.get() == "Auto-clean parameters updated."
    finally:
        root.destroy()


def test_valley_debris_cut_survives_a_doublet_population():
    """The diffuse debris plateau touches the low edge of the histogram
    window, so it was never a peak, and the valley found was the one between
    cells and doublets, above the median: 0 of 1000 debris removed and the
    diagnostic said 'FSC-A is unimodal'."""
    root, ed = _editor_or_skip()
    try:
        _dirty, truth = _setup(ed, doublets=True)
        gid = _ac_gid(ed, 'dirty')
        ed._autoclean_set_debris_mode('dirty', gid, 'valley')
        rm = ed._autoclean_method_masks('dirty')['debris'].to_numpy()
        assert (rm & (truth == 'debris')).sum() >= 950
        assert (rm & (truth == 'cell')).sum() <= 40
    finally:
        root.destroy()


@pytest.mark.parametrize('dist,n', [('normal', 200), ('uniform', 1000)])
def test_bimodal_valley_rarely_splits_unimodal_data(dist, n):
    """At the sizes the granular rescue feeds it (the low-FSC subset, >= 50
    events) a histogram of ~1 event a bin made noise peaks: normal n=200
    was split in 43/100 seeds, uniform n=1000 in 71/100, and a false split
    kept non-granular debris as 'granulocytes'."""
    from openflo.pipeline import _bimodal_valley
    hits = 0
    for seed in range(100):
        rng = np.random.default_rng(seed)
        v = (rng.normal(5000, 1000, n) if dist == 'normal'
             else rng.uniform(1000, 10000, n))
        hits += _bimodal_valley(v) is not None
    assert hits <= 5, f'{hits}/100 unimodal samples split'
