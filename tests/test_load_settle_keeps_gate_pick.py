"""Rows picked while files are still landing stay picked.

When a load burst ends (a session reopening, a folder of files dropped),
the settle pass highlights the active sample, and the tree is rebuilt for
the new rows. The pass used to highlight the active sample whatever the
user had picked meanwhile, and the rebuild kept only the first picked row,
so "New tab from selection" took the active sample alone instead of the
gates, samples or trial the user had picked. A session reopen also put the
saved sample back on every retry pass, over a row picked on another one.

The load itself only ever selects the active sample's row, so anything
else selected is the user's. The pass still highlights the active sample
when nothing is picked, and when the pick no longer matches the active
sample (it was switched without the tree, as a restore does). Synthetic
samples and generic channels only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from openflo.tree_ids import gate_iid, sample_iid, trial_iid
from tests.test_view_tabs import _editor_or_skip

COLS = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A']


def _sample(name, seed):
    from openflo.pipeline import FlowSample
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.uniform(1, 1000, (400, len(COLS))), columns=COLS)
    return FlowSample.from_dataframe(df, name=name, path=f'{name}.fcs')


def _gate(ed, name, value, channel='FL1-A'):
    return ed._add_gate({'kind': 'threshold', 'name': name,
                         'channel': channel, 'op': '>', 'value': value,
                         'parent_id': None}, audit=False)


def _two_samples_gated(ed):
    """s1 (active) holds 'Pos', s2 holds 'Hi'; the first burst settled."""
    ed._on_loaded('s1', _sample('s1', 1))
    ed._on_loaded('s2', _sample('s2', 2))
    ed._on_load_settled()
    ed.update()
    ed._set_active_sample('s2')
    hi = _gate(ed, 'Hi', 700.0)
    ed._set_active_sample('s1')
    pos = _gate(ed, 'Pos', 500.0)
    ed._refresh_gate_list()
    return pos, hi


def _late_arrival_settles(ed):
    """Another file lands and its burst settles, the timer's pass run
    directly (the app runs it 120 ms after the last arrival). The new
    sample row makes the pass rebuild the tree."""
    name = f's{len(ed._samples) + 1}'
    ed._on_loaded(name, _sample(name, len(ed._samples) + 1))
    pending = getattr(ed, '_load_settle_after', None)
    if pending:
        ed.after_cancel(pending)
        ed._load_settle_after = None
    ed._on_load_settled()
    ed.update()


def _pick(ed, rows, run_event=True):
    """Select ``rows``; with ``run_event`` the select event runs, as for a
    click (picking a gate makes its sample active)."""
    ed.gate_tv.selection_set(rows)
    if run_event:
        ed.update()
    assert set(ed.gate_tv.selection()) == set(rows)


def test_a_gate_picked_during_the_load_stays_picked_for_a_new_tab():
    root, ed = _editor_or_skip()
    try:
        pos, _hi = _two_samples_gated(ed)
        _pick(ed, [gate_iid('s1', pos)])
        _late_arrival_settles(ed)
        assert ed.gate_tv.selection() == (gate_iid('s1', pos),)
        assert ed._active_sample == 's1'
        ed._new_tab_from_selection()
        assert ed._tabs().active.members == {'s1': [pos]}
    finally:
        root.destroy()


def test_two_gates_stay_picked_through_the_rebuild_and_quietly():
    root, ed = _editor_or_skip()
    try:
        pos, _hi = _two_samples_gated(ed)
        lo = _gate(ed, 'Lo', 100.0, channel='FL2-A')
        assert lo != pos
        ed._refresh_gate_list()
        rows = [gate_iid('s1', pos), gate_iid('s1', lo)]
        _pick(ed, rows)
        # The rebuild's re-select is an echo: the gate readout is not redone
        # over the load's status line.
        readouts = []
        ed._show_gate_count = lambda *a: readouts.append(a)
        _late_arrival_settles(ed)
        assert set(ed.gate_tv.selection()) == set(rows)
        assert readouts == []
        assert ed.status_var.get().startswith('3 sample(s) loaded')
        ed._new_tab_from_selection()
        assert sorted(ed._tabs().active.members['s1']) == sorted([pos, lo])
    finally:
        root.destroy()


def test_a_rebuild_keeps_the_picked_rows_that_are_still_there():
    root, ed = _editor_or_skip()
    try:
        pos, _hi = _two_samples_gated(ed)
        lo = _gate(ed, 'Lo', 100.0, channel='FL2-A')
        ed._refresh_gate_list()
        _pick(ed, [gate_iid('s1', pos), gate_iid('s1', lo)])
        ed._gates.pop(lo)
        ed._gate_id_order.remove(lo)
        ed._refresh_gate_list()
        assert ed.gate_tv.selection() == (gate_iid('s1', pos),)
    finally:
        root.destroy()


def test_a_trial_two_samples_or_a_presort_row_stay_picked():
    root, ed = _editor_or_skip()
    try:
        _two_samples_gated(ed)
        _pick(ed, [trial_iid(ed._trial_for('s1'))])
        _late_arrival_settles(ed)
        assert ed.gate_tv.selection() == (trial_iid(ed._trial_for('s1')),)
        ed._new_tab_from_selection()
        assert set(ed._tabs().active.members) >= {'s1', 's2'}
        ed._switch_tab(0)

        _pick(ed, [sample_iid('s1'), sample_iid('s2')])
        _late_arrival_settles(ed)
        assert set(ed.gate_tv.selection()) == {sample_iid('s1'),
                                               sample_iid('s2')}

        ed._tabs().active.presort = [('s1', None)]
        ed._refresh_gate_list()
        _pick(ed, ['PRE:0'])
        _late_arrival_settles(ed)
        assert ed.gate_tv.selection() == ('PRE:0',)
    finally:
        root.destroy()


def test_nothing_picked_gives_the_active_sample():
    root, ed = _editor_or_skip()
    try:
        _two_samples_gated(ed)
        ed.gate_tv.selection_set(())
        ed.update()
        _late_arrival_settles(ed)
        assert ed.gate_tv.selection() == (sample_iid('s1'),)
    finally:
        root.destroy()


def test_a_pick_on_another_sample_gives_way_to_the_active_sample():
    """The active sample switched without the tree (a restore does this):
    the highlight follows it, as gates and compensation act on it."""
    root, ed = _editor_or_skip()
    try:
        pos, hi = _two_samples_gated(ed)
        for rows in ([gate_iid('s2', hi)],
                     [gate_iid('s1', pos), gate_iid('s2', hi)],
                     [sample_iid('s2')]):
            _pick(ed, rows, run_event=False)
            _late_arrival_settles(ed)
            assert ed.gate_tv.selection() == (sample_iid('s1'),), rows
            assert ed._active_sample == 's1'
    finally:
        root.destroy()


def test_a_session_reopen_puts_the_saved_sample_back_once():
    """Reopening with s1 saved active: the first restore pass highlights
    s1; a gate the user then picks on s2 survives the later passes."""
    root, ed = _editor_or_skip()
    try:
        _pos, hi = _two_samples_gated(ed)
        ed._set_active_sample('s2')
        ed._session_restore = {'x': None, 'y': None, 'color': None,
                               'active': 's1', 'plot_enabled': {}}
        ed._check_fixes_on_open = lambda: None
        ed._loading.add('still_coming')          # more files to land
        ed._apply_session_restore(_tries=29)     # a retry pass follows
        assert ed._active_sample == 's1'
        assert ed.gate_tv.selection() == (sample_iid('s1'),)
        ed.update()
        _pick(ed, [gate_iid('s2', hi)])
        assert ed._active_sample == 's2'
        ed._loading.discard('still_coming')
        ed._apply_session_restore(_tries=30)     # the last pass
        ed.update()
        assert ed.gate_tv.selection() == (gate_iid('s2', hi),)
        assert ed._active_sample == 's2'
    finally:
        root.destroy()


def _reopening(ed, active):
    ed._session_restore = {'x': None, 'y': None, 'color': None,
                           'active': active, 'plot_enabled': {}}
    ed._check_fixes_on_open = lambda: None


def test_a_header_picked_during_a_reopen_survives_the_last_pass():
    root, ed = _editor_or_skip()
    try:
        _two_samples_gated(ed)
        _reopening(ed, 's1')
        ed._loading.add('still_coming')
        ed._apply_session_restore(_tries=29)
        assert ed.gate_tv.selection() == (sample_iid('s1'),)
        ed.update()
        header = trial_iid(ed._trial_for('s1'))
        _pick(ed, [header])
        ed._loading.discard('still_coming')
        ed._apply_session_restore(_tries=30)     # the last pass
        ed.update()
        assert ed.gate_tv.selection() == (header,)
    finally:
        root.destroy()


def test_a_saved_sample_that_never_loads_leaves_picks_alone():
    """Its file is missing: nothing to put back, so a header picked while
    the others load stays picked through every pass."""
    root, ed = _editor_or_skip()
    try:
        _two_samples_gated(ed)
        _reopening(ed, 'gone')
        header = trial_iid(ed._trial_for('s1'))
        _pick(ed, [header])
        ed._loading.add('still_coming')
        ed._apply_session_restore(_tries=29)
        ed.update()
        assert ed.gate_tv.selection() == (header,)
        ed._loading.discard('still_coming')
        ed._apply_session_restore(_tries=30)
        ed.update()
        assert ed.gate_tv.selection() == (header,)
    finally:
        root.destroy()


def test_a_saved_sample_still_loading_is_put_back_when_it_lands():
    root, ed = _editor_or_skip()
    try:
        _two_samples_gated(ed)
        _reopening(ed, 'late')
        ed._loading.add('late')
        ed._apply_session_restore(_tries=29)
        assert ed._active_sample == 's1'
        ed._on_loaded('late', _sample('late', 9))
        ed._loading.discard('late')
        ed._apply_session_restore(_tries=30)
        ed.update()
        assert ed._active_sample == 'late'
        assert ed.gate_tv.selection() == (sample_iid('late'),)
    finally:
        root.destroy()
