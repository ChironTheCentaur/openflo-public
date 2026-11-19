"""Dragging sample rows onto a day or a Comps/Samples subgroup in the editor
tree (``DnDMixin._regroup_dragged_samples``), run on a stand-in editor with
no window: the real tree-id parser, the real grouping rules and a status line
that records what it is told.

Each test checks the resulting day, side, day order and status text against
values worked out by hand, so a slip in the selection, the side override,
the counts or the message shows up here.
"""
from __future__ import annotations

import pytest

from openflo import tree_ids
from openflo.editor_dnd import DnDMixin
from openflo.editor_grouping import GroupingMixin


class _Status:
    def __init__(self):
        self.texts = []

    def set(self, text):
        self.texts.append(text)


class _Editor(DnDMixin, GroupingMixin):
    """Just the state and calls the drag regroup uses."""

    _parse_iid = staticmethod(tree_ids.parse_iid)

    def __init__(self, samples, trial_order=None, overrides=None,
                 selection=()):
        self._samples = {n: object() for n in samples}
        self._sample_order = list(samples)
        self._sample_trial = dict(samples)
        self._trial_order = (list(trial_order) if trial_order is not None
                             else list(dict.fromkeys(samples.values())))
        self._sample_is_comp = dict(overrides or {})
        self._press_selection = selection
        self.status_var = _Status()
        self.calls = []

    def _refresh_gate_list(self):
        self.calls.append('refresh')

    def _schedule_replot(self, delay):
        self.calls.append(('replot', delay))


S, T, SG = tree_ids.sample_iid, tree_ids.trial_iid, tree_ids.subgroup_iid


def _drag(ed, src, tgt):
    ed._regroup_dragged_samples(src, tgt)


def test_one_sample_to_another_day_drops_the_emptied_day():
    ed = _Editor({'S1': 'Day 1', 'S2': 'Day 2'})
    _drag(ed, S('S1'), T('Day 2'))
    assert ed._sample_trial == {'S1': 'Day 2', 'S2': 'Day 2'}
    assert ed._trial_order == ['Day 2']
    assert ed._sample_is_comp == {}
    assert ed.calls == ['refresh', ('replot', 0)]
    assert ed.status_var.texts == ['Moved 1 sample(s) → Day 2.']


def test_a_new_day_is_added_to_the_order_once():
    ed = _Editor({'S1': 'Day 1', 'S2': 'Day 1'},
                 selection=(S('S1'), S('S2')))
    _drag(ed, S('S2'), T('Day 9'))
    assert ed._sample_trial == {'S1': 'Day 9', 'S2': 'Day 9'}
    assert ed._trial_order == ['Day 9']
    assert ed.status_var.texts == ['Moved 2 sample(s) → Day 9.']


def test_selection_counts_only_when_the_dragged_row_is_in_it():
    ed = _Editor({'S1': 'Day 1', 'S2': 'Day 1', 'S3': 'Day 2'},
                 selection=(S('S1'), S('S2')))
    _drag(ed, S('S3'), T('Day 3'))     # S3 is not in the selection
    assert ed._sample_trial == {'S1': 'Day 1', 'S2': 'Day 1', 'S3': 'Day 3'}
    assert ed._trial_order == ['Day 1', 'Day 3']
    assert ed.status_var.texts == ['Moved 1 sample(s) → Day 3.']


def test_the_count_leaves_out_samples_already_there():
    ed = _Editor({'S1': 'Day 1', 'S2': 'Day 2', 'S3': 'Day 2'},
                 selection=(S('S1'), S('S2'), S('S3')))
    _drag(ed, S('S1'), T('Day 2'))
    assert ed.status_var.texts == ['Moved 1 sample(s) → Day 2.']


def test_a_sample_with_no_day_yet_is_in_the_default_one():
    # _trial_for gives 'Trial' to a sample with no stored day, so dropping it
    # on 'Trial' moves nothing.
    ed = _Editor({'S1': 'Trial', 'S2': 'Day 2'})
    del ed._sample_trial['S1']
    _drag(ed, S('S1'), T('Trial'))
    assert 'S1' not in ed._sample_trial
    assert ed.calls == [] and ed.status_var.texts == []


def test_a_day_drop_keeps_the_side_override():
    ed = _Editor({'S1': 'Day 1', 'S2': 'Day 2'}, overrides={'S1': True})
    _drag(ed, S('S1'), T('Day 2'))
    assert ed._sample_is_comp == {'S1': True}
    assert ed.status_var.texts == ['Moved 1 sample(s) → Day 2.']


def test_an_unloaded_samples_day_does_not_keep_a_day_alive():
    ed = _Editor({'S1': 'Day 1', 'S2': 'Day 2'})
    ed._sample_trial['Gone'] = 'Day 1'     # remembered, no longer loaded
    _drag(ed, S('S1'), T('Day 2'))
    assert ed._trial_order == ['Day 2']


def test_a_one_row_selection_is_just_the_dragged_row():
    ed = _Editor({'S1': 'Day 1', 'S2': 'Day 1'}, selection=(S('S1'),))
    _drag(ed, S('S1'), T('Day 2'))
    assert ed._sample_trial == {'S1': 'Day 2', 'S2': 'Day 1'}
    assert ed._trial_order == ['Day 1', 'Day 2']


def test_selection_skips_repeats_gates_and_unloaded_samples():
    ed = _Editor({'S1': 'Day 1', 'S2': 'Day 1', 'S3': 'Day 2'},
                 selection=(S('S2'), tree_ids.gate_iid('S1', 'g1'), S('S2'),
                            S('Gone'), '', S('S1')))
    _drag(ed, S('S2'), T('Day 2'))
    assert ed._sample_trial == {'S1': 'Day 2', 'S2': 'Day 2', 'S3': 'Day 2'}
    assert ed._trial_order == ['Day 2']
    # The status names the day the samples went to, counted once each.
    assert ed.status_var.texts == ['Moved 2 sample(s) → Day 2.']


@pytest.mark.parametrize('src', ['', tree_ids.gate_iid('S1', 'g1'),
                                 S('Gone')])
def test_nothing_to_move_does_nothing(src):
    ed = _Editor({'S1': 'Day 1'})
    _drag(ed, src, T('Day 2'))
    assert ed._sample_trial == {'S1': 'Day 1'}
    assert ed._trial_order == ['Day 1']
    assert ed.calls == [] and ed.status_var.texts == []


@pytest.mark.parametrize('tgt', ['nonsense', S('Gone')])
def test_a_target_with_no_day_or_side_does_nothing(tgt):
    ed = _Editor({'S1': 'Day 1', 'S2': 'Day 2'})
    _drag(ed, S('S1'), tgt)
    assert ed._sample_trial == {'S1': 'Day 1', 'S2': 'Day 2'}
    assert ed.calls == [] and ed.status_var.texts == []


def test_dropping_on_the_day_it_is_already_in_says_nothing():
    ed = _Editor({'S1': 'Day 1', 'S2': 'Day 2'}, trial_order=['Day 1',
                                                              'Old', 'Day 2'])
    _drag(ed, S('S1'), T('Day 1'))
    assert ed._trial_order == ['Day 1', 'Old', 'Day 2']  # not pruned
    assert ed.calls == [] and ed.status_var.texts == []


def test_sample_to_comps_subgroup_sets_the_override():
    ed = _Editor({'S1': 'Day 1', 'S2': 'Day 2'})
    _drag(ed, S('S1'), SG('comp', 'Day 2'))
    assert ed._sample_trial == {'S1': 'Day 2', 'S2': 'Day 2'}
    assert ed._sample_is_comp == {'S1': True}
    assert ed.status_var.texts == ['Moved 1 sample(s) → Day 2 / Comps.']


def test_side_change_alone_counts_as_a_move():
    ed = _Editor({'S1': 'Day 1'}, overrides={'S1': True})
    _drag(ed, S('S1'), SG('samp', 'Day 1'))
    assert ed._sample_is_comp == {'S1': False}
    assert ed.status_var.texts == ['Moved 1 sample(s) → Day 1 / Samples.']


def test_dropping_on_the_side_its_name_gives_records_the_choice():
    ed = _Editor({'S1': 'Day 1', 'S2': 'Day 1'},
                 selection=(S('S1'), S('S2')))
    _drag(ed, S('S1'), SG('samp', 'Day 1'))
    assert ed._sample_is_comp == {'S1': False, 'S2': False}
    assert ed.calls == []
    assert ed.status_var.texts == [
        '2 sample(s) already in Samples; now kept there by your choice, '
        'not by name.']


def test_a_choice_already_recorded_is_not_counted_again():
    ed = _Editor({'S1': 'Day 1', 'S2': 'Day 1'}, overrides={'S1': True},
                 selection=(S('S1'), S('S2')))
    ed._sample_is_comp['S2'] = True
    _drag(ed, S('S1'), SG('comp', 'Day 1'))
    assert ed._sample_is_comp == {'S1': True, 'S2': True}
    assert ed.calls == [] and ed.status_var.texts == []


def test_moved_and_recorded_together_reports_the_move():
    # S1 moves day and keeps its name-given side (recorded); S2 already in
    # Day 2 changes side. Both count as moved; the side is recorded for S1.
    ed = _Editor({'S1': 'Day 1', 'S2': 'Day 2'}, overrides={'S2': True},
                 selection=(S('S1'), S('S2')))
    _drag(ed, S('S1'), SG('samp', 'Day 2'))
    assert ed._sample_trial == {'S1': 'Day 2', 'S2': 'Day 2'}
    assert ed._sample_is_comp == {'S1': False, 'S2': False}
    assert ed._trial_order == ['Day 2']
    assert ed.status_var.texts == ['Moved 2 sample(s) → Day 2 / Samples.']


def test_drop_on_a_sample_row_takes_that_samples_day_and_side():
    ed = _Editor({'S1': 'Day 1', 'S2': 'Day 2'}, overrides={'S2': True})
    _drag(ed, S('S1'), S('S2'))
    assert ed._sample_trial['S1'] == 'Day 2'
    assert ed._sample_is_comp == {'S2': True, 'S1': True}
    assert ed.status_var.texts == ['Moved 1 sample(s) → Day 2 / Comps.']


def test_on_a_sample_row_a_name_given_side_is_not_recorded():
    # Only a subgroup drop records a side the sample already has.
    ed = _Editor({'S1': 'Day 1', 'S2': 'Day 1'})
    _drag(ed, S('S1'), tree_ids.gate_iid('S2', 'g1'))
    assert ed._sample_is_comp == {}
    assert ed.calls == [] and ed.status_var.texts == []


def test_day_order_keeps_its_order_and_drops_every_empty_day():
    ed = _Editor({'S1': 'Day 3', 'S2': 'Day 1', 'S3': 'Day 2'},
                 trial_order=['Day 3', 'Stale', 'Day 1', 'Day 2'])
    _drag(ed, S('S3'), T('Day 1'))
    assert ed._trial_order == ['Day 3', 'Day 1']
