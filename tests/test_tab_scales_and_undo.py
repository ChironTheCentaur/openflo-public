"""Each tray tab keeps its own axis scales and ranges, and Undo says what it
reverts and keeps the gates of samples loaded after its checkpoint.

* T4 -- an axis scale or range set in one tab carried into every other tab
  and into the session's saved scales (the tab's view stored no scale for a
  channel on its default, and showing a tab skipped that "no scale").
* T3 -- Undo goes back to the tab its checkpoint was taken in, and can
  revert a gate on a sample the tab does not show; its status said only
  "Undo.", so the user was not told which tab or gate.
* U1 -- Undo past a sample load (loads are not undo steps) dropped that
  sample's gates and left the active sample's gates detached from the
  ones the session saves, so a gate drawn next was lost.

Synthetic samples and generic channels only.
"""
from __future__ import annotations

import pytest

from openflo.view_tabs import TabSet, ViewTab
from tests.conftest import editor_or_skip, load_menu_sample


@pytest.fixture
def editor():
    """s1 and s2 loaded (each with gate Pos on FL1-A); tab A holds s1."""
    root, ed = editor_or_skip(accept_confirms=False)
    try:
        load_menu_sample(ed, 's1', 1)
        load_menu_sample(ed, 's2', 2)
        ed._set_active_sample('s1')
        ed._refresh_gate_list()
        ed._clear_undo_pending()
        ed.tab_a = ed._tabs().new_tab('A', members={'s1': None})
        ed._rebuild_tab_strip()
        ed.update()
        yield ed
    finally:
        root.destroy()


# ── T4: axis scales and ranges stay in their tab ─────────────────────────

def test_a_scale_set_in_a_tab_stays_in_that_tab(editor):
    ed = editor
    assert 'FL1-A' not in ed._channel_scale      # All: the default
    ed._switch_tab(ed.tab_a)
    ed._set_axis_config('FL1-A', 'linear', (10.0, 900.0))
    ed._switch_tab(0)
    assert 'FL1-A' not in ed._channel_scale
    assert 'FL1-A' not in ed._channel_range
    assert ed._scale_of('FL1-A') == 'logicle'
    # ... nor in the session's top-level (the All tab's) scales.
    assert 'FL1-A' not in ed._session_state()['channel_scale']
    # Tab A still has its own.
    ed._switch_tab(ed.tab_a)
    assert ed._channel_scale['FL1-A'] == 'linear'
    assert ed._channel_range['FL1-A'] == pytest.approx((10.0, 900.0))


def test_all_keeps_its_own_scale_when_a_tab_sets_another(editor):
    ed = editor
    ed._set_axis_config('FL2-A', 'asinh', None)        # in All
    ed._switch_tab(ed.tab_a)
    assert ed._channel_scale['FL2-A'] == 'asinh'     # a new tab inherits
    ed._set_axis_config('FL2-A', 'linear', None)
    ed._switch_tab(0)
    assert ed._channel_scale['FL2-A'] == 'asinh'


def test_tab_scales_are_saved_and_read_back():
    t = ViewTab('A', members={'s1': None},
                channel_scale={'FL1-A': 'linear'},
                channel_range={'FL1-A': (1.0, 2.0), 'FL2-A': ['x', 3]})
    assert t.channel_range == {'FL1-A': (1.0, 2.0)}  # a corrupt one skipped
    back = TabSet.from_dict(TabSet([ViewTab('All', None), t]).to_dict())
    assert back.tabs[1].channel_scale == {'FL1-A': 'linear'}
    assert back.tabs[1].channel_range == {'FL1-A': (1.0, 2.0)}
    # A tab saved before tabs kept their own scales has none: it takes
    # the ones in use when first shown.
    old = t.to_dict()
    del old['channel_scale'], old['channel_range']
    assert ViewTab.from_dict(old).channel_scale is None
    assert ViewTab.from_dict(old).channel_range is None
    # Corrupt (not a mapping) is read as none, not an error.
    old.update(channel_scale=['linear'], channel_range='x')
    assert ViewTab.from_dict(old).channel_scale is None
    assert ViewTab.from_dict(old).channel_range is None


def test_duplicating_a_tab_copies_its_scales(editor):
    ed = editor
    ed._switch_tab(ed.tab_a)
    ed._set_axis_config('FL1-A', 'linear', None)
    ed._duplicate_tab()
    assert ed._tabs().active.name == 'A copy'
    assert ed._channel_scale['FL1-A'] == 'linear'
    ed._switch_tab(0)
    assert 'FL1-A' not in ed._channel_scale


# ── T3: Undo names the tab and the gates it reverts ──────────────────────

def test_undo_from_another_tab_says_where_and_what(editor):
    ed = editor
    pos = ed._sample_gate_order['s1'][0]
    ed._checkpoint()
    ed._clear_undo_pending()
    ed._sample_gates['s1'][pos]['enabled'] = False        # in All
    ed._switch_tab(ed.tab_a)
    ed._undo()
    assert ed._tabs().active.name == 'All'
    msg = ed.status_var.get()
    assert "back in tab 'All'" in msg
    assert "switched on 'Pos' (s1)" in msg
    assert ed._sample_gates['s1'][pos]['enabled'] is True
    ed._redo()
    assert "switched off 'Pos' (s1)" in ed.status_var.get()
    assert "back in tab" not in ed.status_var.get()


def test_undo_of_a_gate_on_a_sample_the_tab_hides_says_so(editor):
    ed = editor
    ed._switch_tab(ed.tab_a)                        # shows s1 only
    ed._checkpoint()
    ed._clear_undo_pending()
    ed._set_active_sample('s2')
    gid = ed._add_gate({'kind': 'threshold', 'name': 'Hidden',
                        'channel': 'FL2-A', 'op': '>', 'value': 1.0,
                        'parent_id': None}, audit=False)
    ed._set_active_sample('s1')
    ed._undo()
    assert gid not in ed._sample_gates['s2']
    assert ("removed 'Hidden' (s2, not shown in this tab)"
            in ed.status_var.get())


def test_undo_status_does_not_call_a_loaded_samples_gates_removed(editor):
    ed = editor
    ed._checkpoint()
    ed._clear_undo_pending()
    ed._add_gate({'kind': 'threshold', 'name': 'Undone', 'channel': 'FL2-A',
                  'op': '>', 'value': 1.0, 'parent_id': None}, audit=False)
    ed._suspend_undo = True                    # as a load runs
    try:
        load_menu_sample(ed, 's3', 3)
    finally:
        ed._suspend_undo = False
    ed._undo()
    msg = ed.status_var.get()
    assert "removed 'Undone' (s1)" in msg
    assert '(s3' not in msg                    # s3 keeps its gate Pos
    assert [g['name'] for g in ed._sample_gates['s3'].values()] == ['Pos']


def test_the_status_detail_never_stops_undo(editor, monkeypatch):
    import openflo.editor_undo as eu
    ed = editor
    pos = ed._sample_gate_order['s1'][0]
    ed._checkpoint()
    ed._clear_undo_pending()
    ed._sample_gates['s1'][pos]['enabled'] = False

    def boom(*_a):
        raise ValueError('ambiguous truth value')
    monkeypatch.setattr(eu, '_gate_changes', boom)
    ed._undo()
    assert ed._sample_gates['s1'][pos]['enabled'] is True
    assert ed.status_var.get().startswith('Undo. (')


def test_change_list_is_capped_and_counts_hidden():
    from openflo.editor_undo import _describe_changes, _gate_changes
    before = {'a': {'g1': {'name': 'P1'}, 'g2': {'name': 'P2'}},
              'b': {'g1': {'name': 'Q', 'enabled': True}}}
    after = {'a': {}, 'b': {'g1': {'name': 'Q', 'enabled': False}},
             'c': {'g1': {'name': 'R'}}}
    ch = _gate_changes(before, after)
    assert ch == [('a', 'removed', 'P1'), ('a', 'removed', 'P2'),
                  ('b', 'switched off', 'Q'), ('c', 'restored', 'R')]
    text = _describe_changes(ch, lambda n: n == 'a')
    assert text == ("removed 'P1' (a); removed 'P2' (a); 2 more gate "
                    "change(s), 2 on samples not shown in this tab.")
    assert _describe_changes([], lambda n: True) == ''
    # Hidden samples among the named ones are marked, not counted again.
    text = _describe_changes(ch, lambda n: n != 'a')
    assert text == ("removed 'P1' (a, not shown in this tab); removed 'P2' "
                    "(a, not shown in this tab); 2 more gate change(s).")


# ── U1: Undo past a sample load ──────────────────────────────────────────

def test_undo_keeps_a_loaded_sample_that_is_not_active(editor):
    ed = editor
    # One checkpoint before the load: the gate below joins it (as within
    # one Tk event), so Undo steps back past the load.
    depth = len(ed._undo_stack)
    ed._checkpoint()
    ed._suspend_undo = True                    # as a load runs
    try:
        load_menu_sample(ed, 's3', 3)
        ed._cluster_labels['s3'] = {'leiden': 'labels'}
    finally:
        ed._suspend_undo = False
    ed._set_active_sample('s1')
    ed._add_gate({'kind': 'threshold', 'name': 'Undone', 'channel': 'FL2-A',
                  'op': '>', 'value': 1.0, 'parent_id': None}, audit=False)
    assert len(ed._undo_stack) == depth + 1
    ed._undo()
    assert ed._active_sample == 's1'
    assert [g['name'] for g in ed._sample_gates['s3'].values()] == ['Pos']
    assert ed._cluster_labels['s3'] == {'leiden': 'labels'}
    assert 'Undone' not in [g['name'] for g in ed._sample_gates['s1'].values()]


def test_undo_past_a_load_keeps_the_loaded_sample_and_live_gates(editor):
    ed = editor
    ed._checkpoint()
    ed._clear_undo_pending()
    ed._set_active_sample('s1')
    ed._add_gate({'kind': 'threshold', 'name': 'Undone', 'channel': 'FL2-A',
                  'op': '>', 'value': 1.0, 'parent_id': None}, audit=False)
    ed._suspend_undo = True                    # as a load runs
    try:
        load_menu_sample(ed, 's3', 3)          # active now, with gate Pos
        ed._cluster_labels['s3'] = {'leiden': 'labels'}
    finally:
        ed._suspend_undo = False
    ed._undo()
    assert ed._active_sample == 's3'
    names = [g['name'] for g in ed._sample_gates['s3'].values()]
    assert names == ['Pos']
    assert 'Undone' not in [g['name'] for g in ed._sample_gates['s1'].values()]
    assert ed._gates is ed._sample_gates['s3']
    assert ed._gate_id_order is ed._sample_gate_order['s3']
    assert ed._cluster_labels['s3'] == {'leiden': 'labels'}
    # Redo reapplies the undone gate and keeps s3's.
    ed._redo()
    assert 'Undone' in [g['name'] for g in ed._sample_gates['s1'].values()]
    assert [g['name'] for g in ed._sample_gates['s3'].values()] == ['Pos']
    assert ed._gates is ed._sample_gates['s3']
    ed._undo()
    # A gate drawn now is in what the session saves.
    ed._clear_undo_pending()
    ed._add_gate({'kind': 'threshold', 'name': 'After', 'channel': 'FL2-A',
                  'op': '>', 'value': 1.0, 'parent_id': None}, audit=False)
    saved = ed._session_state()
    assert [g['name'] for g in saved['sample_gates']['s3']] == ['Pos',
                                                               'After']
