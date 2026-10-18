"""Tools → Normalize batches (CytoNorm)…: the dialog over batch_normalize.

Generated data: two acquisition days, each with a reference sample (the
same material) and a study sample; day 2 reads 30 % brighter on FL1-A and
FL2-A. Normalizing through the references must bring day 2's reference onto
day 1's, carry the study sample with it, and keep the study sample's real
difference. Negative partners: a batch with no reference (or two) blocks
the run, a set-aside sample is left out of the fit, and with "apply" off
nothing loaded changes.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')

from tests.conftest import gui_unavailable  # noqa: E402

CH = ['FL1-A', 'FL2-A']


def _sample(name, scale, frac_bright, date, rng):
    # Two populations (1.0 and 1.5); the reference holds both, as a CytoNorm
    # reference must, the study sample more of the bright one.
    n_b = int(3000 * frac_bright)
    base = np.r_[rng.normal(1.0, 0.1, (3000 - n_b, 2)),
                 rng.normal(1.5, 0.1, (n_b, 2))]
    df = pd.DataFrame(np.c_[rng.normal(5, 1, (3000, 2)), base * scale],
                      columns=['FSC-A', 'SSC-A'] + CH)
    return SimpleNamespace(name=name, path=f'/data/{name}.fcs', data=df,
                           fluor_channels=list(CH),
                           channel_labels={c: c for c in df.columns},
                           metadata={'date': date})


@pytest.fixture
def ed():
    try:
        import tkinter as tk
    except ImportError:
        gui_unavailable("tkinter not available — headless environment")
    try:
        root = tk.Tk()
    except Exception as e:                              # noqa: BLE001
        gui_unavailable(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    import importlib
    gui = importlib.import_module('openflo.gui')
    editor = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                      on_save=None, primary=False)
    editor.withdraw()
    rng = np.random.default_rng(11)
    spec = [('d1_ref', 1.0, 0.5, '01-JAN-2026', 'Run1'),
            ('d1_study', 1.0, 0.8, '01-JAN-2026', 'Run1'),
            ('d2_ref', 1.3, 0.5, '08-JAN-2026', 'Run2'),
            ('d2_study', 1.3, 0.8, '08-JAN-2026', 'Run2')]
    for name, scale, frac, date, trial in spec:
        editor._samples[name] = _sample(name, scale, frac, date, rng)
        editor._sample_order.append(name)
        editor._sample_trial[name] = trial
        editor._sample_gates.setdefault(name, {})
        editor._sample_gate_order.setdefault(name, [])
    yield editor
    root.destroy()


def _median(ed, name, ch='FL1-A'):
    return float(np.median(ed._samples[name].data[ch]))


def test_plan_finds_one_reference_per_batch_by_keyword(ed):
    from openflo.ui_batch_normalize import BY_KEYWORD, plan_batches
    p = plan_batches(ed, BY_KEYWORD, '$DATE', 'controls', 'ref')
    assert p['error'] == ''
    assert p['batch_of']['d2_study'] == '08-JAN-2026'
    assert p['references'] == {'01-JAN-2026': 'd1_ref',
                               '08-JAN-2026': 'd2_ref'}
    assert p['channels'] == CH


def test_plan_refuses_a_batch_without_exactly_one_reference(ed):
    from openflo.ui_batch_normalize import BY_TRIAL, plan_batches
    p = plan_batches(ed, BY_TRIAL, mode='controls', ref_token='d1_')
    assert 'Run1: 2 matches' in p['error'] and 'Run2: 0 matches' in p['error']
    one = plan_batches(ed, BY_TRIAL, mode='controls', ref_token='')
    assert one['error']
    # A single batch cannot be normalized.
    ed._sample_trial.update({'d2_ref': 'Run1', 'd2_study': 'Run1'})
    assert 'at least 2' in plan_batches(ed, BY_TRIAL)['error']


def test_set_aside_and_comp_controls_stay_out_of_the_fit(ed):
    from openflo.ui_batch_normalize import BY_TRIAL, plan_batches
    ed._sample_set_aside['d1_study'] = 'clogged'
    p = plan_batches(ed, BY_TRIAL, mode='goal')
    assert 'd1_study' not in p['batch_of'] and p['left_out'] == ['d1_study']


def _bright(ed, name, cut=1.25, ch='FL1-A'):
    """Fraction of events past a fixed gate: the biology being compared."""
    return float((ed._samples[name].data[ch] > cut).mean())


def _peak(ed, name, ch='FL1-A'):
    """Where the dim population sits (its median)."""
    v = ed._samples[name].data[ch]
    return float(np.median(v[v < np.median(v)]))


def test_reference_normalization_aligns_days_and_keeps_biology(ed):
    from openflo.ui_batch_normalize import BatchNormalizeWindow
    # Before: day 2 reads 30 % brighter, so a fixed gate over-counts it.
    assert _peak(ed, 'd2_ref') / _peak(ed, 'd1_ref') == pytest.approx(
        1.3, rel=0.03)
    assert _bright(ed, 'd2_ref') > _bright(ed, 'd1_ref') + 0.3
    w = BatchNormalizeWindow(ed)
    w.by_var.set('FCS keyword')
    w.mode_var.set('controls')
    w.ref_var.set('ref')
    w.meta_var.set(1)
    w._run(sync=True)
    assert w._result is not None, w.status_var.get()
    # Day 2's reference now reads as day 1's...
    assert _peak(ed, 'd2_ref') == pytest.approx(_peak(ed, 'd1_ref'),
                                                rel=0.02)
    assert _bright(ed, 'd2_ref') == pytest.approx(_bright(ed, 'd1_ref'),
                                                  abs=0.03)
    # ...and its study sample came along: the same gate counts the same
    # bright fraction on both days, still more than the reference has.
    # (Both days move onto the goal, the mean of the two, so the fixed
    # gate's fractions are not the raw 0.5 and 0.8.)
    assert _bright(ed, 'd2_study') == pytest.approx(_bright(ed, 'd1_study'),
                                                    abs=0.03)
    assert _bright(ed, 'd2_study') > _bright(ed, 'd2_ref') + 0.15
    assert 'CytoNorm' in w._result['methods']
    assert 'Van Gassen' in w.text.get('1.0', 'end')
    entry = ed._audit_log.entries()[-1]
    assert entry['action'] == 'cytonorm'
    assert entry['details']['references'] == {'01-JAN-2026': 'd1_ref',
                                              '08-JAN-2026': 'd2_ref'}
    # Gate undo cannot reverse it, and says where to go instead.
    assert 'Restore original values' in ed._undo_stack[-1]['barrier']
    w.destroy()


def test_restore_puts_back_the_original_values_without_reloading(ed):
    from openflo.ui_batch_normalize import BatchNormalizeWindow
    raw = {n: ed._samples[n].data for n in ed._sample_order}
    w = BatchNormalizeWindow(ed)
    assert str(w.restore_btn.cget('state')) == 'disabled'
    w.mode_var.set('goal')
    w.meta_var.set(1)
    w._run(sync=True)
    w._run(sync=True)                    # a second run stacks on the first
    assert ed._samples['d2_ref'].data is not raw['d2_ref']
    assert str(w.restore_btn.cget('state')) == 'normal'
    w._restore()
    # Back to the values from before the FIRST normalization.
    for n, df in raw.items():
        assert ed._samples[n].data is df
    assert ed._audit_log.entries()[-1]['action'] == 'cytonorm.restored'
    assert str(w.restore_btn.cget('state')) == 'disabled'
    # Nothing left to restore: a second restore changes nothing.
    assert ed._restore_batch_norm() == 0
    w.destroy()


def test_without_apply_nothing_loaded_changes(ed, tmp_path):
    from openflo import ui_batch_normalize
    from openflo.ui_batch_normalize import BatchNormalizeWindow
    seen = {}

    def fake_run(editor, plan, mode, nmeta, out_dir=None):
        seen['out_dir'] = out_dir
        return real(editor, plan, mode, nmeta, None)
    real = ui_batch_normalize.run_normalization
    orig = {n: ed._samples[n].data for n in ed._sample_order}
    w = BatchNormalizeWindow(ed)
    w.mode_var.set('goal')
    w.meta_var.set(1)
    w.apply_var.set(False)
    w.copies_var.set(True)
    w.dir_var.set(str(tmp_path))
    ui_batch_normalize.run_normalization = fake_run
    try:
        w._run(sync=True)
    finally:
        ui_batch_normalize.run_normalization = real
    assert seen['out_dir'] == str(tmp_path)
    assert w._result is not None
    for n, df in orig.items():
        assert ed._samples[n].data is df
    # Neither destination chosen: nothing runs.
    w._result = None
    w.copies_var.set(False)
    w._run(sync=True)
    assert w._result is None and 'Choose where' in w.status_var.get()
    w.destroy()
