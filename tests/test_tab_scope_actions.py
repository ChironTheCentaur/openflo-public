"""Actions taken in a tray tab reach only what that tab shows.

A tab is a view: its trial rows list only the trial's samples the tab
holds, and it shows only the gates it was given. Several actions still used
the global lists, so from a tab they reached samples or gates the user
could not see (scope audit T1, T5, T6, G7, D2):

* T1 -- "Clear" on a trial row cleared the trial's hidden samples too;
* T5 -- "Clear" on a sample row cleared gates the tab hides (and a shown
  gate a hidden one sits under takes that hidden gate with it);
* T6 -- the trial and header check boxes, and trial-row Copy to tab / Set
  aside (the selection's samples), acted on hidden members;
* G7 -- "every sample" actions (Clear all, Copy gates to, template targets,
  typed coordinates, combination gates, Re-place on each sample) listed
  every loaded sample;
* D2 -- "Downsample to smallest" took its display floor from samples in
  other tabs, and "Display + data", which cuts the data of every sample in
  every tab, did so without a word from a tab.

On the All tab every one of these still reaches every sample. Synthetic
samples and generic channels only; dialogs are recorded, not shown.
"""
from __future__ import annotations

from tkinter import ttk

import pytest

from tests.conftest import editor_or_skip, load_menu_sample


def _gate(ed, name, channel, parent=None):
    return ed._add_gate({'kind': 'threshold', 'name': name, 'channel': channel,
                         'op': '>', 'value': 300.0, 'parent_id': parent},
                        audit=False)


@pytest.fixture
def editor():
    """s1, s2 (trial T) and s3, s4 (trial U), each with root gates Pos
    (FL1-A) and Q (FL2-A); s4 also has Child and Other under Pos.
    Tab A = {s1, s3}: it hides s2 of T and s4 of U. It is opened last."""
    root, ed = editor_or_skip(accept_confirms=False)
    try:
        ids = {}
        for i, (n, tr) in enumerate((('s1', 'T'), ('s2', 'T'), ('s3', 'U'),
                                     ('s4', 'U'))):
            pos = load_menu_sample(ed, n, i, trial=tr)
            ids[n] = {'Pos': pos, 'Q': _gate(ed, 'Q', 'FL2-A')}
        ed._set_active_sample('s4')
        ids['s4']['Child'] = _gate(ed, 'Child', 'FL2-A', ids['s4']['Pos'])
        ids['s4']['Other'] = _gate(ed, 'Other', 'FL1-A', ids['s4']['Pos'])
        ed._set_active_sample('s1')
        ed._refresh_gate_list()
        ed._clear_undo_pending()
        ed.ids = ids
        ts = ed._tabs()
        ed.tab_a = ts.new_tab('A', members={'s1': None, 's3': None})
        ed._rebuild_tab_strip()
        ed._switch_tab(ed.tab_a)
        ed.update()
        yield ed
    finally:
        root.destroy()


def _open_tab(ed, name, members):
    idx = ed._tabs().new_tab(name, members=members)
    ed._rebuild_tab_strip()
    ed._switch_tab(idx)
    ed.update()


def _asks(monkeypatch, answer):
    """Record every askyesno (title, message, default) and answer it."""
    from tkinter import messagebox
    asked = []

    def ask(title=None, message=None, **k):
        asked.append((title, message, k.get('default')))
        return answer
    monkeypatch.setattr(messagebox, 'askyesno', ask)
    return asked


def _names(ed, sample):
    return sorted(g['name'] for g in ed._sample_gates[sample].values())


def _select(ed, iid):
    ed.gate_tv.selection_set(iid)
    ed.update()


# ── T1 / T5: Clear on a trial or sample row ─────────────────────────────
def _audits(ed, monkeypatch):
    """Record every audit event ``(action, details)``."""
    got = []
    monkeypatch.setattr(ed, '_audit', lambda a, **d: got.append((a, d)))
    return got


def test_clearing_a_trial_row_in_a_tab_leaves_its_hidden_members(
        editor, monkeypatch):
    from openflo.tree_ids import trial_iid
    ed = editor
    audits = _audits(ed, monkeypatch)
    _select(ed, trial_iid('T'))
    ed._clear_selected_gate()
    assert ed._sample_gates['s1'] == {}
    assert _names(ed, 's2') == ['Pos', 'Q']               # hidden: kept
    # Only s1 was edited: no "kept" count for the hidden sample's gates,
    # and the audit trail doesn't name s2.
    assert ed.status_var.get() == "Cleared 2 gate(s) — Ctrl+Z to undo."
    assert audits == [('gate.remove', {'n_gates': 2, 'samples': ['s1']})]


def test_clearing_a_trial_row_on_all_clears_every_member(editor):
    from openflo.tree_ids import trial_iid
    ed = editor
    ed._switch_tab(0)
    _select(ed, trial_iid('T'))
    ed._clear_selected_gate()
    assert ed._sample_gates['s1'] == {} and ed._sample_gates['s2'] == {}


def test_clearing_a_sample_row_keeps_the_gates_the_tab_hides(editor):
    ed = editor
    _open_tab(ed, 'B', {'s3': [ed.ids['s3']['Pos']]})
    _select(ed, ed._sample_iid('s3'))
    ed._clear_selected_gate()
    assert _names(ed, 's3') == ['Q']
    status = ed.status_var.get()
    assert status == ("Cleared 1 gate(s); kept 1 gate(s) this tab doesn't "
                      "show, or that those gates need — Ctrl+Z to undo.")
    ed._undo()
    assert _names(ed, 's3') == ['Pos', 'Q']


def test_a_shown_gate_a_hidden_gate_sits_under_stays(editor):
    """Tab C shows s4's Child, so also its parent Pos; Other (hidden) sits
    under Pos, so clearing s4 there removes Child alone."""
    ed = editor
    _open_tab(ed, 'C', {'s4': [ed.ids['s4']['Child']]})
    _select(ed, ed._sample_iid('s4'))
    ed._clear_selected_gate()
    assert _names(ed, 's4') == ['Other', 'Pos', 'Q']
    # Other and Q are hidden; Pos is held for Other.
    assert "; kept 3 gate(s) this tab doesn't show" in ed.status_var.get()


def _with_hidden_boolean(ed):
    """s3 gains Both = Pos AND Q; Tab B then shows only s3's Pos, so Both
    (and Q) are hidden there while Both combines the shown Pos."""
    ids = ed.ids['s3']
    ed._switch_tab(0)
    ed._set_active_sample('s3')
    ids['Both'] = ed._add_gate({'kind': 'boolean', 'op': 'and',
                                'operands': [ids['Pos'], ids['Q']],
                                'name': 'Both', 'parent_id': None},
                               audit=False)
    _open_tab(ed, 'B', {'s3': [ids['Pos']]})


def test_a_shown_operand_of_a_hidden_boolean_stays(editor):
    """Removing Pos would leave the hidden Both selecting nothing, with
    nothing on screen to say so."""
    from openflo.gate_eval import broken_booleans
    ed = editor
    _with_hidden_boolean(ed)
    _select(ed, ed._sample_iid('s3'))
    ed._clear_selected_gate()
    assert _names(ed, 's3') == ['Both', 'Pos', 'Q']
    assert broken_booleans(ed._sample_gates['s3']) == ([], [])
    assert ed.status_var.get().startswith("No gates to clear; kept 3")


def test_clear_all_keeps_a_shown_operand_of_a_hidden_boolean(
        editor, monkeypatch):
    from openflo.gate_eval import broken_booleans
    ed = editor
    _with_hidden_boolean(ed)
    _asks(monkeypatch, True)
    ed._clear_all()
    assert _names(ed, 's3') == ['Both', 'Pos', 'Q']
    assert broken_booleans(ed._sample_gates['s3']) == ([], [])


# ── G7: Clear all ────────────────────────────────────────────────────────
def test_clear_all_in_a_tab_clears_only_what_it_shows(editor, monkeypatch):
    ed = editor
    asked = _asks(monkeypatch, True)
    ed._clear_all()
    assert len(asked) == 1
    assert "from 2 sample(s) in this tab" in asked[0][1]
    assert ed._sample_gates['s1'] == {} and ed._sample_gates['s3'] == {}
    assert _names(ed, 's2') == ['Pos', 'Q']
    assert _names(ed, 's4') == ['Child', 'Other', 'Pos', 'Q']
    assert "from the samples this tab shows" in ed.status_var.get()


def test_clear_all_in_a_tab_keeps_gates_it_hides(editor, monkeypatch):
    ed = editor
    _open_tab(ed, 'C', {'s4': [ed.ids['s4']['Child']]})
    _asks(monkeypatch, True)
    ed._clear_all()
    assert _names(ed, 's4') == ['Other', 'Pos', 'Q']


def test_clear_all_on_the_all_tab_clears_every_sample(editor, monkeypatch):
    ed = editor
    ed._switch_tab(0)
    asked = _asks(monkeypatch, True)
    ed._clear_all()
    assert "from 4 sample(s)?" in asked[0][1]
    assert all(ed._sample_gates[n] == {} for n in ('s1', 's2', 's3', 's4'))


# ── G7: the other "every sample" actions ─────────────────────────────────
def _checkbutton_texts(dlg):
    from tests.test_boolean_gate_tools import _walk
    return [w.cget('text') for w in _walk(dlg)
            if isinstance(w, ttk.Checkbutton)]


def test_copy_gates_to_lists_only_the_tabs_samples(editor):
    from tests.test_boolean_gate_tools import _toplevel
    ed = editor
    ed._open_copy_gates_dialog()
    dlg = _toplevel(ed, 'Copy gates')
    texts = ' '.join(_checkbutton_texts(dlg))
    dlg.destroy()
    assert 's3' in texts
    assert 's2' not in texts and 's4' not in texts


def test_template_targets_are_the_tabs_samples(editor, monkeypatch):
    from tests.test_boolean_gate_tools import _toplevel
    ed = editor
    monkeypatch.setattr(ed, 'wait_window', lambda *_a: None)
    ed._ask_template_apply()
    dlg = _toplevel(ed, 'Apply template to…')
    texts = [t for t in _checkbutton_texts(dlg) if t.startswith('s')]
    dlg.destroy()
    assert texts == ['s1  (active)', 's3']


def test_typed_coordinates_reach_only_the_tabs_samples(editor):
    ed = editor
    ids = ed.ids
    before = ed._sample_gates['s2'][ids['s2']['Pos']]['value']
    n = ed._apply_gate_coordinates('s1', ids['s1']['Pos'], {'value': '2K'},
                                   every_sample=True)
    assert n == 2
    assert (ed._sample_gates['s3'][ids['s3']['Pos']]['value']
            == ed._sample_gates['s1'][ids['s1']['Pos']]['value'])
    assert ed._sample_gates['s2'][ids['s2']['Pos']]['value'] == before


def test_combination_gates_reach_only_the_tabs_samples(editor):
    ed = editor
    ids = ed.ids['s1']
    built = ed._make_combinations([ids['Pos'], ids['Q']], every_sample=True,
                                  show=False)
    assert sorted(built) == ['s1', 's3']
    assert _names(ed, 's2') == ['Pos', 'Q']


def test_replace_on_each_sample_proposes_only_the_tabs_samples(editor):
    ed = editor
    got = {p['sample'] for p in ed._refit_proposals('s1', ed.ids['s1']['Pos'])}
    assert got == {'s1', 's3'}
    ed._switch_tab(0)
    got = {p['sample'] for p in ed._refit_proposals('s1', ed.ids['s1']['Pos'])}
    assert got == {'s1', 's2', 's3', 's4'}


# ── T6: check boxes and the selection's samples ──────────────────────────
def test_the_trial_check_box_acts_on_the_members_the_tab_shows(editor):
    from openflo.tree_ids import trial_iid
    ed = editor
    ed._sample_plot_enabled['s1'] = False         # the only shown member: off
    ed._sample_plot_enabled['s2'] = True          # hidden in this tab
    ed._refresh_gate_list()
    assert ed.gate_tv.set(trial_iid('T'), ed.gate_tv['columns'][0]) == '☐'
    ed._press_selection = ()
    ed._handle_checkbox_click(trial_iid('T'))
    assert ed._sample_plot_enabled['s1'] is True   # the click turns it on
    assert ed._sample_plot_enabled['s2'] is True   # hidden: untouched
    ed._handle_checkbox_click(trial_iid('T'))
    assert ed._sample_plot_enabled['s1'] is False
    assert ed._sample_plot_enabled['s2'] is True


def test_the_header_check_box_acts_on_the_tabs_samples(editor):
    ed = editor
    ed._toggle_all_sample_plots()
    assert {n: ed._sample_plot_enabled[n] for n in ('s1', 's2', 's3', 's4')} \
        == {'s1': False, 's2': True, 's3': False, 's4': True}


def test_a_selected_trial_row_stands_for_its_shown_members(editor):
    from openflo.tree_ids import trial_iid
    ed = editor
    _select(ed, trial_iid('T'))
    assert ed._target_samples('selected') == ['s1']     # Set aside etc.
    assert ed._selected_tab_rows() == [('s1', None)]    # Copy to tab
    ed._switch_tab(0)
    _select(ed, trial_iid('T'))
    assert ed._target_samples('selected') == ['s1', 's2']


# ── D2: downsample to smallest ───────────────────────────────────────────
def _shrink(ed, name, n):
    ed._samples[name].data = ed._samples[name].data.iloc[:n].copy()


def test_the_display_floor_comes_from_the_tabs_samples(editor):
    ed = editor
    _shrink(ed, 's2', 50)                          # tiny, hidden in tab A
    assert ed._smallest_loaded_sample_size() == 300
    ed._switch_tab(0)
    assert ed._smallest_loaded_sample_size() == 50


def test_the_data_floor_is_the_same_in_every_tab(editor):
    ed = editor
    _shrink(ed, 's2', 50)
    ed._sample_plot_enabled['s2'] = False          # off in tab A only
    assert ed._data_floor_sample() == (50, 's2')
    ed._switch_tab(0)
    assert ed._data_floor_sample() == (50, 's2')


def _set_ds_mode(ed, mode):
    ed._ds_mode_var.set(mode)
    ed._on_ds_mode_changed()


def _turn_on_display_and_data(ed):
    _set_ds_mode(ed, 'Display + data')


def test_display_and_data_in_a_tab_asks_and_no_trims_nothing(
        editor, monkeypatch):
    from tkinter import messagebox
    ed = editor
    _shrink(ed, 's2', 50)
    _set_ds_mode(ed, 'Off')
    asked = _asks(monkeypatch, False)
    _turn_on_display_and_data(ed)
    assert len(asked) == 1
    _title, msg, default = asked[0]
    assert default == messagebox.NO
    assert "3 loaded sample(s) in every tab" in msg
    assert "'s2', which this tab doesn't show" in msg
    assert "1 of them this tab doesn't show" in msg   # s4
    assert [len(ed._samples[n].data) for n in ('s1', 's2', 's3', 's4')] \
        == [300, 50, 300, 300]
    # No puts the mode back as it was.
    assert ed.ds_propagate_var.get() is False
    assert ed.ds_display_var.get() is False
    assert ed._ds_mode_var.get() == 'Off'


def test_no_to_display_and_data_from_display_only_keeps_display_only(
        editor, monkeypatch):
    ed = editor
    _shrink(ed, 's2', 50)
    _set_ds_mode(ed, 'Display only')
    _asks(monkeypatch, False)
    _turn_on_display_and_data(ed)
    assert ed._ds_mode_var.get() == 'Display only'
    assert ed.ds_display_var.get() is True
    assert ed.ds_propagate_var.get() is False


def test_display_and_data_in_a_tab_yes_trims_every_sample(
        editor, monkeypatch):
    ed = editor
    _shrink(ed, 's2', 50)
    _asks(monkeypatch, True)
    _turn_on_display_and_data(ed)
    assert {len(ed._samples[n].data) for n in ('s1', 's2', 's3', 's4')} \
        == {50}


def test_display_and_data_on_the_all_tab_does_not_ask(editor, monkeypatch):
    ed = editor
    ed._switch_tab(0)
    _shrink(ed, 's2', 50)
    asked = _asks(monkeypatch, True)
    _turn_on_display_and_data(ed)
    assert asked == []
    assert {len(ed._samples[n].data) for n in ('s1', 's2', 's3', 's4')} \
        == {50}


def test_a_subgroup_row_stands_for_its_shown_members(editor):
    ed = editor
    assert ed._row_members(('subgroup', 'samp', 'T')) == ['s1']
    ed._switch_tab(0)
    assert ed._row_members(('subgroup', 'samp', 'T')) == ['s1', 's2']


def test_dragging_a_trial_row_carries_only_its_shown_members(editor):
    """A trial row dragged onto another tab, the layout page or a Pipeline
    Workspace carries the members it lists, not the ones the tab hides."""
    from types import SimpleNamespace

    from openflo.tree_ids import trial_iid
    ed = editor
    ed._press_selection = ()
    ev = SimpleNamespace(x_root=0, y_root=0)
    got = {}
    ed._tab_at = lambda *_a: 0
    ed._drop_rows_on_tab = lambda idx, rows: got.setdefault('tab', rows)
    ed._maybe_drop_to_tab(trial_iid('T'), ev)
    ed._layout_window_open = lambda: object()
    ed._drop_to_layout = lambda t, *_a: got.setdefault('layout', t)
    ed._maybe_drop_to_layout(trial_iid('T'), ev)
    ed._workspace_panel = SimpleNamespace(
        point_in_target=lambda *_a: 'tree',
        drop_at=lambda _ed, nodes, *_a: got.setdefault('workspace', nodes))
    ed._maybe_drop_to_workspace(trial_iid('T'), ev)
    assert got == {'tab': [('s1', None)], 'layout': [('s1', None)],
                   'workspace': [('sample', 's1')]}
