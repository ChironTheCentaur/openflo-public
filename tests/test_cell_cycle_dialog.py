"""Analyze -> Cell cycle... runs end to end and reports the right phases.

The pure model (analyze_dna / assign_phase) is covered by test_cell_cycle.py,
but the dialog path that users actually click never ran under the suite:
editor_analysis._open_cell_cycle_dialog -> do_run -> _run_cell_cycle ->
FlowSample.cell_cycle -> phase populations + CellCycleWindow.

Known answer: 6000 G1 events at 50,000, 1000 S events uniform between the
peaks, 2000 G2/M events at 100,000 (2x G1).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from test_gui_smoke import _editor_or_skip  # noqa: E402

from openflo.async_task import wait_for_job  # noqa: E402
from tests._fcs_writer import write_float_fcs
from tests._tk_wait import settle

N_G1, N_S, N_G2 = 6000, 1000, 2000


def _dna(seed=0):
    rng = np.random.default_rng(seed)
    v = np.concatenate([rng.normal(50_000, 2_500, N_G1),
                        rng.uniform(57_500, 92_500, N_S),
                        rng.normal(100_000, 4_000, N_G2)])
    return v[rng.permutation(v.size)]


def _register(ed, s):
    name = s.name
    ed._samples[name] = s
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = True
    ed._sample_gates[name] = {}
    ed._sample_gate_order[name] = []
    ed._sample_gate_seq[name] = 0
    ed._channels = list(s.data.columns)
    ed._channel_labels = dict(s.channel_labels)
    ed._populate_channel_combos()
    ed._active_sample = name


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def _dialog(ed, title):
    import tkinter as tk
    return next(w for w in ed.winfo_children()
                if isinstance(w, tk.Toplevel) and w.title().startswith(title))


def _run_dialog(ed, k=None):
    from tkinter import ttk
    ed._open_cell_cycle_dialog()
    dlg = _dialog(ed, 'Cell cycle')
    combo = next(w for w in _walk(dlg) if isinstance(w, ttk.Combobox))
    if k is not None:
        # The doublet-cut spinbox is the one ranging 0.5..5.0.
        spin = next(w for w in _walk(dlg) if isinstance(w, ttk.Spinbox)
                    and float(w.cget('from')) == 0.5)
        spin.set(str(k))
    chosen = combo.get()
    run = next(w for w in _walk(dlg) if isinstance(w, ttk.Button)
               and str(w.cget('text')) == 'Run')
    run.invoke()
    assert wait_for_job(ed._job_strip)     # the model runs off the Tk thread
    return chosen


def test_dialog_runs_the_model_and_imports_phases():
    import openflo.pipeline as fp
    root, ed, _gui = _editor_or_skip()
    try:
        dna = _dna()
        s = fp.FlowSample.from_dataframe(
            pd.DataFrame({'FSC-A': np.full(dna.size, 8e4), 'V450-A': dna}),
            name='cc', labels={'V450-A': 'DAPI'})
        _register(ed, s)
        chosen = _run_dialog(ed, k=2.0)
        assert 'V450-A' in chosen               # DNA dye auto-detected
        res = s.cell_cycle_result
        assert res and res['ok'] and res['channel'] == 'V450-A'
        # The dialog's k reaches the model.
        assert res['g1_hi'] == pytest.approx(
            res['boundaries']['g1_mean'] + 2.0 * res['g1_sd'])
        ref = fp.analyze_dna(dna, k=2.0)
        # The dialog fits Dean-Jett-Fox by default; the boundaries result
        # it started from is kept beside it.
        assert res['method'] == 'Dean-Jett-Fox'
        for key in ('pct_g1', 'pct_s', 'pct_g2m'):
            assert res['boundaries'][key] == pytest.approx(ref[key])
        assert 15 < res['pct_g2m'] < 30                # true G2/M share 22.2 %
        names = {g['name'] for g in ed._sample_gates['cc'].values()
                 if g.get('kind') == 'category'}
        assert {'G1', 'S', 'G2M'} <= names
        assert _dialog(ed, 'Cell cycle')               # result window opened
    finally:
        root.destroy()


def test_dialog_on_an_editor_loaded_fcs_finds_g2m(tmp_path):
    """The loader logicle-transforms the DNA channel; the model must still
    run on the linear values (G2 = 2 x G1), or every G2/M cell is called S."""

    import openflo.pipeline as fp
    dna = _dna()
    n = dna.size
    rng = np.random.default_rng(1)
    ev = np.column_stack([rng.normal(8e4, 8e3, n), rng.normal(4e4, 5e3, n),
                          dna, np.linspace(0, 100, n)]).astype(np.float32)
    path = tmp_path / 'dna.fcs'
    write_float_fcs(path, ev, ['FSC-A', 'SSC-A', 'V450-A', 'Time'],
                    opt_channel_names=['', '', 'DAPI', ''])
    # Exactly what editor_loadpool._load_worker does to a dropped file.
    s = fp.FlowSample(str(path))
    s.run_qc()
    s.auto_compensate()
    s.apply_transform()
    root, ed, _gui = _editor_or_skip()
    try:
        _register(ed, s)
        _run_dialog(ed)
        res = s.cell_cycle_result
        assert res and res['ok']
        assert 15 < res['pct_g2m'] < 30, (
            f"G1 {res['pct_g1']:.1f} / S {res['pct_s']:.1f} / "
            f"G2M {res['pct_g2m']:.1f} — true G2/M share is 22.2 %")
    finally:
        root.destroy()


def test_a_sample_skipped_in_a_multi_sample_run_is_named_at_the_end():
    """A run over every sample sets each failure in the status bar, and the
    final line replaced it: a sample whose DNA channel is of unknown scale
    (an older OpenFlo CSV) vanished from the report. The final line must
    name it and say why."""

    import openflo.pipeline as fp
    from openflo.channel_transforms import UNKNOWN_SPEC
    root, ed, _gui = _editor_or_skip()
    try:
        for name, seed in (('cc', 0), ('old', 1)):
            dna = _dna(seed)
            s = fp.FlowSample.from_dataframe(
                pd.DataFrame({'FSC-A': np.full(dna.size, 8e4),
                              'V450-A': dna}),
                name=name, labels={'V450-A': 'DAPI'})
            if name == 'cc':
                _register(ed, s)
                continue
            # as restored from a 2.6.1 sidecar, in the same trial
            s.data_transforms = {'V450-A': dict(UNKNOWN_SPEC)}
            ed._samples[name] = s
            ed._sample_order.append(name)
            ed._sample_trial[name] = 'T'
            ed._sample_plot_enabled[name] = True
            ed._sample_gates[name] = {}
            ed._sample_gate_order[name] = []
            ed._sample_gate_seq[name] = 0
        ed._active_sample = 'cc'
        ed._run_cell_cycle('V450-A', all_samples=True, k=2.0)
        assert wait_for_job(ed._job_strip)
        settle(root, 1.0)                    # let anything deferred run
        msg = ed.status_var.get()
        assert 'done for 1 sample' in msg, msg
        assert 'old' in msg and 'unknown' in msg, msg
    finally:
        root.destroy()


def test_dna_channel_list_offers_only_intensity_channels():
    """The DNA-content list offered cluster labels and embedding
    coordinates (leiden, cluster, UMAP1, ...) next to the measured
    channels; picking one modelled cluster ids as DNA content."""
    from tkinter import ttk

    import openflo.pipeline as fp
    root, ed, _gui = _editor_or_skip()
    try:
        n = 2000
        rng = np.random.default_rng(2)
        df = pd.DataFrame({
            'FSC-A': rng.normal(8e4, 8e3, n), 'SSC-A': rng.normal(4e4, 5e3, n),
            'FL1-A': _dna()[:n], 'Time': np.linspace(0, 100, n),
            'cluster': rng.integers(0, 5, n), 'leiden': rng.integers(0, 5, n),
            'flowsom': rng.integers(0, 9, n),
            'flowsom_meta': rng.integers(0, 3, n),
            'UMAP1': rng.normal(0, 3, n), 'UMAP2': rng.normal(0, 3, n),
            'TSNE1': rng.normal(0, 9, n), 'TSNE2': rng.normal(0, 9, n)})
        s = fp.FlowSample.from_dataframe(df, name='lab')
        _register(ed, s)
        ed._open_cell_cycle_dialog()
        dlg = _dialog(ed, 'Cell cycle')
        combo = next(w for w in _walk(dlg) if isinstance(w, ttk.Combobox))
        offered = list(combo.cget('values'))
        assert offered == [ed._fmt_channel(c)
                           for c in ('FSC-A', 'SSC-A', 'FL1-A')]
        # Nothing is auto-detected here; the first offered channel is set,
        # not a label column.
        assert combo.get() == ed._fmt_channel('FSC-A')
        dlg.destroy()
    finally:
        root.destroy()


def test_a_channel_with_no_dna_peaks_says_so():
    """On a scatter channel (one bell-shaped peak) Cell cycle reported
    "done" with G1 100 %, S 0 %, G2/M 0 % and added the phases as
    populations. It must say no usable G1/G2 peaks were found, and why."""
    import openflo.pipeline as fp
    root, ed, _gui = _editor_or_skip()
    try:
        n = 9000
        rng = np.random.default_rng(6)
        s = fp.FlowSample.from_dataframe(
            pd.DataFrame({'FSC-A': rng.normal(8e4, 8e3, n),
                          'SSC-A': rng.normal(35_000, 7_000, n),
                          'FL1-A': _dna()}),
            name='bell')
        _register(ed, s)
        ed._run_cell_cycle('SSC-A', all_samples=False, method='boundaries')
        assert wait_for_job(ed._job_strip)
        settle(root, 0.5)
        msg = ed.status_var.get()
        assert 'no usable G1/G2 DNA peaks' in msg, msg
        assert 'too wide' in msg and 'done' not in msg, msg
        assert not s.cell_cycle_result['ok']
        assert not any(g.get('kind') == 'category'
                       for g in ed._sample_gates['bell'].values())
        # The DNA channel of the same sample still fits.
        ed._run_cell_cycle('FL1-A', all_samples=False, method='boundaries')
        assert wait_for_job(ed._job_strip)
        settle(root, 0.5)
        assert 'done for 1 sample' in ed.status_var.get()
        assert s.cell_cycle_result['ok']
    finally:
        root.destroy()
