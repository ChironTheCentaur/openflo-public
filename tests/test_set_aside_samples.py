"""Samples set aside with a reason (item 13): such a sample is in no group
under any factor, is greyed in the tree and audited, keeps its reason and
note through save and reload, and is named in group statistics and the
methods text; the sample beside it stays in its group.

All data are generated here (synthetic sample names, generic FL channels).
"""
from __future__ import annotations

import os

import tests._four_sample_editor as shared

ed = shared.ed


def test_set_aside_sample_is_in_no_group_under_any_factor(ed):
    try:
        assert ed._set_sample_aside(['synthetic_a2'], 'clogged') == 1
        for factor, group in (('Trial / day', 'Day1'),
                              ('Comp vs Samples', 'Samples'),
                              ('Name token', 'synthetic')):
            assert ed._sample_group_label('synthetic_a2', factor,
                                          ['synthetic']) is None
            # The sample beside it is still in its own group.
            assert ed._sample_group_label('synthetic_a1', factor,
                                          ['synthetic']) == group
        assert ed._set_aside_lines() == ['synthetic_a2: clogged']
        # An empty reason sets nothing aside.
        assert ed._set_sample_aside(['synthetic_b1'], '  ') == 0
        assert 'synthetic_b1' not in ed._sample_set_aside
    finally:
        ed._return_sample(['synthetic_a2'])
    assert ed._sample_group_label('synthetic_a2', 'Trial / day') == 'Day1'


def test_set_aside_is_greyed_in_the_tree_and_audited(ed):
    try:
        ed._set_sample_aside(['synthetic_b2'], 'too few events')
        ed._set_sample_note('synthetic_b2', 'rerun next week')
        text = ed.gate_tv.item(ed._sample_iid('synthetic_b2'), 'text')
        assert 'set aside: too few events' in text and '✎' in text
        assert text.startswith('⊘')
        other = ed.gate_tv.item(ed._sample_iid('synthetic_b1'), 'text')
        assert 'set aside' not in other and other.startswith('■')
        acts = [e['action'] for e in ed._audit_log.entries()]
        assert 'sample.set_aside' in acts
    finally:
        ed._return_sample(['synthetic_b2'])
        ed._set_sample_note('synthetic_b2', '')
    assert 'synthetic_b2' not in ed._sample_notes


def test_set_aside_and_note_survive_save_and_reload(ed):
    try:
        ed._set_sample_aside(['synthetic_b1'], 'bad stain')
        ed._set_sample_note('synthetic_b1', 'check lot')
        entry = next(e for e in ed._session_state()['samples']
                     if e['name'] == 'synthetic_b1')
        assert entry['set_aside'] == 'bad stain'
        assert entry['note'] == 'check lot'
        plain = next(e for e in ed._session_state()['samples']
                     if e['name'] == 'synthetic_a1')
        assert 'set_aside' not in plain and 'note' not in plain
        # Reload: the loader restores both from the saved entry.
        sample = ed._samples['synthetic_b1']
        ed._sample_set_aside.clear()
        ed._sample_notes.clear()
        pkey = os.path.normcase(os.path.abspath(sample.path))
        ed._pending_sample_meta[pkey] = entry
        ed._on_loaded('synthetic_b1', sample)
        assert ed._sample_set_aside == {'synthetic_b1': 'bad stain'}
        assert ed._sample_notes == {'synthetic_b1': 'check lot'}
    finally:
        ed._sample_set_aside.clear()
        ed._sample_notes.clear()


def test_group_stats_and_methods_name_the_set_aside_sample(ed):
    from openflo.ui_group_stats import GroupStatsWindow
    from openflo.ui_methods import MethodsWindow
    try:
        ed._set_sample_aside(['synthetic_a2'], 'clogged')
        w = GroupStatsWindow(ed)
        w._ch.set('FL1-A')
        w._run()
        text = w._txt.get('1.0', 'end')
        w.destroy()
        assert 'Set aside: synthetic_a2: clogged' in text
        assert 'Day1: n=1' in text and 'Day2: n=2' in text
        m = MethodsWindow(ed)
        assert ('Samples set aside from group comparisons, with the reason: '
                'synthetic_a2 (clogged).') in m._para
        m.destroy()
    finally:
        ed._return_sample(['synthetic_a2'])
    w = GroupStatsWindow(ed)
    w._ch.set('FL1-A')
    w._run()
    text = w._txt.get('1.0', 'end')
    w.destroy()
    assert 'Day1: n=2' in text and 'Set aside' not in text
