"""Edit > Group samples: samples go into groups by an FCS keyword or by a
sample-to-condition table (CSV or tab separated).

Every check runs on generated data with synthetic sample names, and is
proved to fail on the old behaviour by a mutation check (see
docs/test-assurance/).
"""
from __future__ import annotations

import pytest

import tests._sample_editor as shared
from openflo import sample_meta as sm
from tests._sample_editor import _s, _tag

editor = shared.editor
virtual_screen = shared.virtual_screen
no_dialogs = shared.no_dialogs
engines_known = shared.engines_known


def test_keywords_are_read_without_dollar_and_in_capitals():
    s = _s('x', **{'$FIL': 'x.fcs', 'tube name': ' ctrl ', 'EMPTY': ''})
    assert sm.keywords(s) == {'FIL': 'x.fcs', 'TUBE NAME': 'ctrl'}


def test_grouping_keywords_offer_only_what_splits_the_samples():
    samples = [
        _s('s1', **{'TUBE NAME': 'ctrl', 'WELL ID': 'A1', '$P1N': 'FL1-A',
                    '$DATE': '01-JAN-2026', 'CONDITION': 'x', '$TOT': '900'}),
        _s('s2', **{'TUBE NAME': 'stim', 'WELL ID': 'A2', '$P1N': 'FL1-A',
                    '$DATE': '02-JAN-2026', 'CONDITION': 'x', '$TOT': '500'}),
        _s('s3', **{'TUBE NAME': 'ctrl', 'WELL ID': 'A3', '$P1N': 'FL1-A',
                    '$DATE': '03-JAN-2026', 'CONDITION': 'x', '$TOT': '900'}),
    ]
    assert sm.grouping_keywords(samples) == [('TUBE NAME', 2)]
    assert sm.groups_by_keyword(samples, '$tube name') == {
        's1': 'ctrl', 's2': 'stim', 's3': 'ctrl'}
    samples.append(_s('s4'))
    assert sm.groups_by_keyword(samples, 'TUBE NAME')['s4'] == \
        "No TUBE NAME"


def test_condition_table_matches_names_and_file_names():
    names = ['Sample_A', 'sample_b', 'Sample_C']
    text = ("file,donor,condition\n"
            "sample_a.fcs,d1,ctrl\n"
            "SAMPLE_B,d2,stim\n"
            "nobody.fcs,d3,stim\n")
    res = sm.read_condition_table(text, names)
    assert res['groups'] == {'Sample_A': 'ctrl', 'sample_b': 'stim'}
    assert res['annotations']['Sample_A'] == {'donor': 'd1'}
    assert res['unmatched_rows'] == ['nobody.fcs']
    assert res['missing'] == ['Sample_C']
    tabbed = "sample\tgroup\nSample_C\tlate\n"
    assert sm.read_condition_table(tabbed, names)['groups'] == {
        'Sample_C': 'late'}
    with pytest.raises(ValueError):
        sm.read_condition_table("just one column\nSample_A\n", names)


def test_group_samples_by_keyword(editor):
    from openflo.ui_sample_groups import GroupSamplesDialog
    root, ed = editor
    _tag(ed, alpha_1={'TUBE NAME': 'ctrl'}, alpha_2={'TUBE NAME': 'stim'},
         beta_1={'TUBE NAME': 'ctrl'})
    ed._note_saved()
    dlg = GroupSamplesDialog(ed)
    root.update()
    assert [k for k, _n in dlg.choices] == ['TUBE NAME']
    dlg.keyword_combo.current(0)
    dlg.keyword_combo.event_generate('<<ComboboxSelected>>')
    root.update()
    shown = {dlg.tree.item(i, 'values')[0]: dlg.tree.item(i, 'values')[1]
             for i in dlg.tree.get_children()}
    assert shown == {'alpha_1': 'ctrl', 'alpha_2': 'stim', 'beta_1': 'ctrl'}
    dlg.apply()
    root.update()
    assert {n: ed._trial_for(n) for n in ed._sample_order} == shown
    assert sorted(ed._trial_order) == ['ctrl', 'stim']    # Day1 emptied
    assert ed.title().startswith('*')
    assert ed._audit_log.entries()[-1]['action'] == 'samples.group'


def test_group_samples_from_a_table_keeps_the_rest(editor):
    from openflo.ui_sample_groups import GroupSamplesDialog
    root, ed = editor
    dlg = GroupSamplesDialog(ed)
    res = dlg.use_table_text("sample,condition\nalpha_1.fcs,treated\n"
                             "gamma_9,treated\n", 'conditions.csv')
    assert res['unmatched_rows'] == ['gamma_9']
    assert 'gamma_9' in dlg.note_var.get()
    dlg.apply()
    root.update()
    assert ed._trial_for('alpha_1') == 'treated'
    assert ed._trial_for('alpha_2') == 'Day1'           # not in the table
    assert ed._trial_order == ['Day1', 'treated']
