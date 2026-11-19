"""Which samples "Add MESF column" writes to (scope audit D5).

The bead fit was written as MESF:<channel> into every loaded sample with
the channel, whatever instrument, date, voltages or tab, and replaced a
MESF column a sample had from its own beads, though the window's own tip
says settings must match. Now "Apply to:" offers this batch (default: the
samples acquired like the bead sample, acquisition.batch_of, limited to the
active tab, as compensation does), every sample in this tab, or the bead
sample only; and it asks (default No) before replacing a MESF column, No
applying nothing.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openflo.view_tabs import ViewTab
from tests.conftest import editor_or_skip

COLS = ('FSC-A', 'SSC-A', 'FL1-A', 'FL2-A')
DAY1 = {'$CYT': 'InstrA', '$DATE': '01-JAN-2026', '$P3N': 'FL1-A',
        '$P3V': '500'}
OTHER = {'$CYT': 'InstrB', '$DATE': '02-JAN-2026', '$P3N': 'FL1-A',
         '$P3V': '650'}
CAL = {'slope': 2.0, 'intercept': 0.0, 'r2': 0.99, 'n': 6,
       'model': 'linear'}


def _add(ed, name, meta, value, cols=COLS):
    df = pd.DataFrame({c: np.full(40, value) for c in cols})
    ed._samples[name] = SimpleNamespace(
        name=name, path=f'{name}.fcs', data=df, metadata=meta,
        data_transforms={},
        fluor_channels=[c for c in cols if c.startswith('FL')],
        channel_labels={c: c for c in cols})
    ed._sample_order.append(name)
    ed._sample_trial[name] = 'Day1'
    if 'Day1' not in ed._trial_order:
        ed._trial_order.append('Day1')
    ed._sample_plot_enabled[name] = True
    ed._sample_gates.setdefault(name, {})
    ed._sample_gate_order.setdefault(name, [])
    ed._sample_gate_seq.setdefault(name, 0)


class _Ask:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def __call__(self, *a, **k):
        self.calls.append((a, k))
        return self.answer


@pytest.fixture
def env(monkeypatch):
    from tkinter import messagebox
    root, ed = editor_or_skip()
    _add(ed, 'SampleBeads', dict(DAY1), 100.0)
    _add(ed, 'SampleDay1', dict(DAY1), 200.0)
    _add(ed, 'SampleOther', dict(OTHER), 300.0)
    _add(ed, 'SampleNoKw', {}, 400.0)
    ed._channels = list(COLS)
    ed._channel_labels = {c: c for c in COLS}
    ed._populate_channel_combos()
    audits = []
    ed._audit = lambda ev, **k: audits.append((ev, k))
    ask = _Ask(False)
    monkeypatch.setattr(messagebox, 'askyesno', ask)
    yield SimpleNamespace(root=root, ed=ed, ask=ask, audits=audits)
    root.destroy()


def _dialog(ed, bead='SampleBeads'):
    from openflo.ui_calibration import CalibrationDialog
    dlg = CalibrationDialog(ed)
    dlg.bead_var.set(bead)
    dlg._sync_channels()
    dlg.apply_var.set('FL1-A')
    dlg._cal = dict(CAL)
    return dlg


def _with_mesf(ed):
    return [n for n in ed._sample_order
            if 'MESF:FL1-A' in ed._samples[n].data.columns]


def test_by_default_only_the_bead_samples_batch_gets_the_column(env):
    dlg = _dialog(env.ed)
    assert dlg.scope_var.get() == 'batch'
    assert dlg._batch_var.get() == "Batch: 2 samples: SampleBeads, SampleDay1"
    assert 'SampleOther: another instrument' in dlg._batch_tip_text
    assert 'SampleNoKw: no acquisition keywords' in dlg._batch_tip_text
    dlg._apply()
    assert _with_mesf(env.ed) == ['SampleBeads', 'SampleDay1']
    np.testing.assert_allclose(
        env.ed._samples['SampleDay1'].data['MESF:FL1-A'], 400.0)
    text = dlg.result.cget('text')
    assert "Applied to 2 sample(s) in this batch → 'MESF:FL1-A'" in text
    assert 'SampleBeads, SampleDay1' in text
    dlg.destroy()


def test_the_batch_leaves_out_samples_the_tab_hides(env):
    ts = env.ed._tabs()
    ts.tabs.append(ViewTab('Tab 2', members={'SampleBeads': None,
                                             'SampleOther': None}))
    ts.current = 1
    dlg = _dialog(env.ed)
    assert 'SampleDay1: not shown in this tab' in dlg._batch_tip_text
    assert dlg.scope_var.get() == 'bead'       # nobody else in the batch
    dlg._apply()
    assert _with_mesf(env.ed) == ['SampleBeads']
    dlg.destroy()


def test_every_sample_in_this_tab(env):
    dlg = _dialog(env.ed)
    dlg.scope_var.set('tab')
    dlg._apply()
    assert _with_mesf(env.ed) == list(env.ed._sample_order)
    assert 'in this tab' in dlg.result.cget('text')
    dlg.destroy()


def test_every_sample_in_this_tab_stays_in_the_tab(env):
    ts = env.ed._tabs()
    ts.tabs.append(ViewTab('Tab 2', members={'SampleDay1': None,
                                             'SampleOther': None}))
    ts.current = 1
    dlg = _dialog(env.ed)
    dlg.scope_var.set('tab')
    dlg._apply()
    assert _with_mesf(env.ed) == ['SampleDay1', 'SampleOther']
    dlg.destroy()


def test_bead_sample_only(env):
    dlg = _dialog(env.ed)
    dlg.scope_var.set('bead')
    dlg._apply()
    assert _with_mesf(env.ed) == ['SampleBeads']
    dlg.destroy()


def test_no_batch_means_bead_only_but_a_tab_pick_is_kept(env):
    dlg = _dialog(env.ed, bead='SampleNoKw')
    assert dlg.scope_var.get() == 'bead'
    assert 'disabled' in dlg._scope_rbs['batch'].state()
    assert 'no other sample' in dlg._batch_var.get()
    dlg.scope_var.set('tab')
    dlg._refresh_scope()
    assert dlg.scope_var.get() == 'tab'
    dlg.destroy()


def test_a_users_pick_survives_a_redraw(env):
    dlg = _dialog(env.ed)
    dlg.scope_var.set('bead')
    dlg.bead_var.set('SampleDay1')
    dlg._sync_channels()                       # redraws the batch
    assert dlg.scope_var.get() == 'bead'
    assert not dlg._scope_rbs['batch'].instate(['disabled'])
    dlg.destroy()


def test_replacing_an_existing_column_asks_default_no(env):
    env.ed._samples['SampleDay1'].data['MESF:FL1-A'] = 7.0
    dlg = _dialog(env.ed)
    dlg._apply()
    assert len(env.ask.calls) == 1
    (_t, text), kw = env.ask.calls[0]
    assert "Replace 'MESF:FL1-A' in 1 sample(s) (SampleDay1)" in text
    assert kw.get('default') == 'no'
    # No: nothing applied anywhere, the old column kept.
    assert _with_mesf(env.ed) == ['SampleDay1']
    np.testing.assert_allclose(
        env.ed._samples['SampleDay1'].data['MESF:FL1-A'], 7.0)
    assert 'Nothing applied' in dlg.result.cget('text')
    assert env.audits == []
    env.ask.answer = True
    dlg._apply()
    np.testing.assert_allclose(
        env.ed._samples['SampleDay1'].data['MESF:FL1-A'], 400.0)
    assert _with_mesf(env.ed) == ['SampleBeads', 'SampleDay1']
    dlg.destroy()


def test_a_column_outside_the_scope_is_not_asked_about(env):
    env.ed._samples['SampleOther'].data['MESF:FL1-A'] = 7.0
    dlg = _dialog(env.ed)
    dlg._apply()
    assert env.ask.calls == []
    np.testing.assert_allclose(
        env.ed._samples['SampleOther'].data['MESF:FL1-A'], 7.0)
    dlg.destroy()


def test_a_sample_of_unknown_scale_is_neither_asked_about_nor_written(
        env, monkeypatch):
    from openflo import channel_transforms
    real = channel_transforms.linear_values

    def linear_values(sample, ch):
        if sample.name == 'SampleOther':
            raise channel_transforms.UnknownScaleError('planted')
        return real(sample, ch)
    monkeypatch.setattr(channel_transforms, 'linear_values', linear_values)
    env.ed._samples['SampleOther'].data['MESF:FL1-A'] = 7.0
    dlg = _dialog(env.ed)
    dlg.scope_var.set('tab')
    dlg._apply()
    assert env.ask.calls == []
    text = dlg.result.cget('text')
    assert 'Applied to 3 sample(s) in this tab' in text
    assert 'scale unknown: SampleOther' in text
    assert env.audits[-1][1]['n_samples'] == 3
    assert 'SampleOther' not in env.audits[-1][1]['samples']
    np.testing.assert_allclose(
        env.ed._samples['SampleOther'].data['MESF:FL1-A'], 7.0)
    dlg.destroy()


def test_the_audit_names_the_samples_and_scope(env):
    dlg = _dialog(env.ed)
    dlg._apply()
    ev, rec = env.audits[-1]
    assert ev == 'calibration'
    assert rec['samples'] == ['SampleBeads', 'SampleDay1']
    assert rec['n_samples'] == 2
    assert rec['scope'] == 'batch'
    assert rec['bead_sample'] == 'SampleBeads'
    dlg.destroy()


def test_the_batch_follows_the_target_channel(env):
    """Voltages are compared on the channel written to: FL2-A has none in
    these files, so no other sample is in its batch."""
    dlg = _dialog(env.ed)
    dlg.apply_var.set('FL2-A')
    dlg.apply_cb.event_generate('<<ComboboxSelected>>')
    dlg.update()
    assert 'no other sample' in dlg._batch_var.get()
    assert dlg.scope_var.get() == 'bead'
    dlg.destroy()


def test_a_bead_sample_the_tab_hides_still_gets_its_batch_column(env):
    ts = env.ed._tabs()
    ts.tabs.append(ViewTab('Tab 2', members={'SampleDay1': None}))
    ts.current = 1
    dlg = _dialog(env.ed)
    dlg._apply()
    assert _with_mesf(env.ed) == ['SampleBeads', 'SampleDay1']
    dlg.destroy()


def test_picking_another_bead_sample_redraws_the_batch(env):
    dlg = _dialog(env.ed)
    assert dlg._batch_var.get().startswith('Batch: 2 samples')
    dlg.bead_var.set('SampleNoKw')
    dlg._sync_channels()
    assert 'no other sample' in dlg._batch_var.get()
    assert dlg.scope_var.get() == 'bead'
    dlg.destroy()


# Gaps a review found (scope audit D5 review, N1).

def test_tab_scope_skips_a_sample_without_the_channel(env):
    """A tab sample lacking the target channel is skipped, not crashed on."""
    _add(env.ed, 'SampleNoFl1', {}, 300.0, cols=('FSC-A', 'SSC-A', 'FL2-A'))
    dlg = _dialog(env.ed)
    dlg.scope_var.set('tab')
    dlg._apply()
    assert 'SampleNoFl1' not in _with_mesf(env.ed)
    assert len(_with_mesf(env.ed)) == 4
    assert 'Applied to 4 sample(s) in this tab' in dlg.result.cget('text')
    dlg.destroy()


def test_the_batch_choice_comes_back_after_a_bead_sample_without_one(env):
    dlg = _dialog(env.ed, bead='SampleNoKw')
    assert dlg._scope_rbs['batch'].instate(['disabled'])
    assert dlg.scope_var.get() == 'bead'
    dlg.bead_var.set('SampleBeads')
    dlg._sync_channels()
    assert not dlg._scope_rbs['batch'].instate(['disabled'])
    assert dlg.scope_var.get() == 'batch'       # no pick was made
    dlg.destroy()


def test_no_target_with_the_channel_applies_and_audits_nothing(env):
    _add(env.ed, 'SampleNoFl1', {}, 300.0, cols=('FSC-A', 'SSC-A', 'FL2-A'))
    dlg = _dialog(env.ed, bead='SampleNoFl1')
    dlg.apply_var.set('FL1-A')
    dlg.scope_var.set('bead')
    dlg._apply()
    assert _with_mesf(env.ed) == []
    assert 'No sample' in dlg.result.cget('text')
    assert env.audits == []
    dlg.destroy()
