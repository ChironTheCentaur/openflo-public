"""Remove from tab vs Remove from session in the Samples & Gates tray.

On a tab other than All, a sample can be taken off that tab only (it stays
loaded, in All and in any other tab that has it), or removed from the session
(gone from every tab). Both are offered side by side in the sample and trial
right-click menus (the sample's under Remove ▸) and as tray buttons, each
with help text. Synthetic samples
and generic channels only; no menu or window is shown (tk_popup is replaced
by a recorder) and the editor stays withdrawn.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.conftest import (
    all_menus,
    editor_or_skip,
    invoke_menu_path,
    load_menu_sample,
    menu_labels,
    right_click_row,
    submenu,
)

TAB_ONLY = "Remove from this tab"          # gate rows: Tabs ▸
SAMPLE_REMOVE = "Remove"                    # sample rows: Remove ▸ ...
SAMPLE_TAB_ONLY = "From this tab"
SESSION = "From session (all tabs)…"
TRIAL_TAB_ONLY = "Remove trial from this tab"
TRIAL_SESSION = "Remove trial from session (samples + gates)…"


@pytest.fixture
def three_samples():
    """An editor with s1, s2, s3 loaded (each with one gate) and two extra
    tabs, A = {s1, s2} and B = {s1}; tab A is open."""
    root, ed = editor_or_skip(accept_confirms=False)
    try:
        gids = {n: load_menu_sample(ed, n, i)
                for i, n in enumerate(('s1', 's2', 's3'))}
        ed._refresh_gate_list()
        ts = ed._tabs()
        a = ts.new_tab('A', members={'s1': None, 's2': None})
        ts.new_tab('B', members={'s1': None})
        ed._rebuild_tab_strip()
        ed._switch_tab(a)
        yield ed, gids
    finally:
        root.destroy()


def _all_labels(menu):
    """Every entry's label in ``menu`` and all its submenus."""
    return [lbl for m in all_menus(menu) for lbl in menu_labels(m)]


def _labels_index(menu, label):
    end = menu.index('end')
    for i in range(end + 1):
        if menu.type(i) != 'separator' and menu.entrycget(i, 'label') == label:
            return i
    pytest.fail(f'{label!r} not in the menu')


def test_sample_menu_on_a_tab_offers_both_removals_side_by_side(
        three_samples, monkeypatch, menu_popups):
    from openflo.tree_ids import sample_iid
    ed, _ = three_samples
    menu = right_click_row(ed, monkeypatch, menu_popups, sample_iid('s1'))
    assert menu_labels(submenu(menu, SAMPLE_REMOVE)) == [
        "Clear gates (keep sample)", SAMPLE_TAB_ONLY, SESSION]
    everything = _all_labels(menu)
    assert everything.count(SAMPLE_TAB_ONLY) == 1   # not again elsewhere
    assert TAB_ONLY not in everything               # nor in the old form
    assert "Remove sample" not in everything        # the old, ambiguous label


def test_sample_menu_on_the_all_tab_offers_only_session_removal(
        three_samples, monkeypatch, menu_popups):
    from openflo.tree_ids import sample_iid
    ed, _ = three_samples
    ed._switch_tab(0)
    menu = right_click_row(ed, monkeypatch, menu_popups, sample_iid('s1'))
    labels = menu_labels(submenu(menu, SAMPLE_REMOVE))
    assert SAMPLE_TAB_ONLY not in labels and SESSION in labels


def test_remove_from_this_tab_keeps_the_sample_loaded_everywhere_else(
        three_samples, monkeypatch, menu_popups):
    from openflo.tree_ids import sample_iid
    ed, gids = three_samples
    ts = ed._tabs()
    menu = right_click_row(ed, monkeypatch, menu_popups, sample_iid('s1'))
    invoke_menu_path(menu, SAMPLE_REMOVE, SAMPLE_TAB_ONLY)
    assert set(ts.active.members) == {'s2'}            # off tab A
    assert 's1' in ed._samples                         # still loaded
    assert gids['s1'] in ed._sample_gates['s1']        # with its gate
    assert ts.tabs[0].has_sample('s1')                 # still in All
    assert ts.tabs[2].members == {'s1': None}          # and in tab B
    assert "still loaded" in ed.status_var.get()


def test_remove_from_session_unloads_the_sample_from_every_tab(
        three_samples, monkeypatch, menu_popups):
    from tkinter import messagebox

    from openflo.tree_ids import sample_iid
    ed, _ = three_samples
    ts = ed._tabs()
    asked = []
    monkeypatch.setattr(messagebox, 'askyesno',
                        lambda *a, **k: asked.append(a) or True)
    menu = right_click_row(ed, monkeypatch, menu_popups, sample_iid('s1'))
    invoke_menu_path(menu, SAMPLE_REMOVE, SESSION)
    assert len(asked) == 1                     # it asks first
    assert 's1' not in ed._samples
    assert 's1' not in ts.tabs[1].members and 's1' not in ts.tabs[2].members
    assert 's2' in ed._samples and 's3' in ed._samples
    assert "from the session" in ed.status_var.get()


def test_trial_menu_on_a_tab_offers_both_removals(
        three_samples, monkeypatch, menu_popups):
    from openflo.tree_ids import trial_iid
    ed, _ = three_samples
    ts = ed._tabs()
    menu = right_click_row(ed, monkeypatch, menu_popups, trial_iid('T'))
    labels = menu_labels(menu)
    assert labels.count(TRIAL_TAB_ONLY) == 1 and TAB_ONLY not in labels
    assert labels.index(TRIAL_SESSION) == labels.index(TRIAL_TAB_ONLY) + 1
    invoke_menu_path(menu, TRIAL_TAB_ONLY)
    assert ts.active.members == {}                     # tab A emptied
    assert set(ed._samples) == {'s1', 's2', 's3'}      # nothing unloaded
    # Counts the two samples the tab held, not the trial's three.
    assert "Removed 2 sample(s) from tab 'A'" in ed.status_var.get()


def test_gate_menu_on_a_tab_still_has_remove_from_this_tab(
        three_samples, monkeypatch, menu_popups):
    from openflo.tree_ids import gate_iid
    ed, gids = three_samples
    menu = right_click_row(ed, monkeypatch, menu_popups,
                           gate_iid('s1', gids['s1']))
    assert TAB_ONLY in menu_labels(submenu(menu, "Tabs"))
    assert _all_labels(menu).count(TAB_ONLY) == 1


def test_a_gate_row_reports_only_what_was_removed(
        three_samples, monkeypatch, menu_popups):
    from openflo.tree_ids import gate_iid
    ed, gids = three_samples
    ts = ed._tabs()
    # Tab A shows s1 with every gate: one gate can't come off, and the
    # status says so instead of claiming a removal.
    menu = right_click_row(ed, monkeypatch, menu_popups, gate_iid('s1', gids['s1']))
    invoke_menu_path(menu, "Tabs", TAB_ONLY)
    assert ts.active.members == {'s1': None, 's2': None}
    msg = ed.status_var.get()
    assert msg.startswith("Nothing removed") and "Removed" not in msg
    # A tab that lists that gate by itself does lose it.
    ts.active.members['s1'] = [gids['s1']]
    ed._refresh_gate_list()
    menu = right_click_row(ed, monkeypatch, menu_popups, gate_iid('s1', gids['s1']))
    invoke_menu_path(menu, "Tabs", TAB_ONLY)
    assert 's1' not in ts.active.members and 's1' in ed._samples
    assert "Removed 1 sample(s) from tab 'A'" in ed.status_var.get()


def _gate(ed, sample, name, parent=None):
    ed._set_active_sample(sample)
    return ed._add_gate({'kind': 'threshold', 'name': name,
                         'channel': 'FL2-A', 'op': '>', 'value': 300.0,
                         'parent_id': parent}, audit=False)


def test_a_listed_gate_is_reported_by_what_the_tree_still_shows(
        three_samples, monkeypatch, menu_popups):
    from openflo.tree_ids import gate_iid
    ed, gids = three_samples
    ts = ed._tabs()
    pos = gids['s1']
    hi = _gate(ed, 's1', 'Hi', parent=pos)       # child of Pos
    other = _gate(ed, 's1', 'Other')             # a second root gate
    # Pos and its child listed: the child stays shown as part of Pos's
    # subtree, so taking it off the list removes nothing from the tree.
    ts.active.members['s1'] = [pos, hi]
    ed._refresh_gate_list()
    menu = right_click_row(ed, monkeypatch, menu_popups, gate_iid('s1', hi))
    invoke_menu_path(menu, "Tabs", TAB_ONLY)
    assert ed.gate_tv.exists(gate_iid('s1', hi))
    assert ed.status_var.get().startswith("Nothing removed")
    # A gate only shown as Pos's parent-or-child, never listed: same.
    ts.active.members['s1'] = [pos]
    ed._refresh_gate_list()
    menu = right_click_row(ed, monkeypatch, menu_popups, gate_iid('s1', hi))
    invoke_menu_path(menu, "Tabs", TAB_ONLY)
    assert ed.gate_tv.exists(gate_iid('s1', hi))
    assert ed.status_var.get().startswith("Nothing removed")
    # A separate listed root gate does come off, and only it.
    ts.active.members['s1'] = [pos, other]
    ed._refresh_gate_list()
    menu = right_click_row(ed, monkeypatch, menu_popups, gate_iid('s1', other))
    invoke_menu_path(menu, "Tabs", TAB_ONLY)
    assert not ed.gate_tv.exists(gate_iid('s1', other))
    assert ed.gate_tv.exists(gate_iid('s1', hi))
    assert ts.active.members['s1'] == [pos]
    assert "Removed 1 gate(s) from tab 'A'" in ed.status_var.get()


def test_remove_from_tab_on_all_changes_nothing(three_samples):
    from openflo.tree_ids import sample_iid
    ed, _ = three_samples
    ts = ed._tabs()
    ed._switch_tab(0)
    ed.gate_tv.selection_set(sample_iid('s1'))
    before = [t.to_dict() for t in ts.tabs]
    ed._remove_rows_from_tab()
    assert [t.to_dict() for t in ts.tabs] == before
    assert 's1' in ed._samples
    assert ed.status_var.get().startswith("The All tab always holds")


def test_button_state_follows_a_loaded_session(three_samples):
    ed, _ = three_samples
    ts = ed._tabs()
    btn = ed._tab_remove_btn
    saved = ts.to_dict()
    # Saved with tab A open: loads with the button on.
    ed._switch_tab(0)
    assert btn.instate(['disabled'])
    ed._restore_tabs_state(saved)
    assert ed._tabs().current == 1 and not btn.instate(['disabled'])
    # Saved with All open: loads with it off.
    saved['current'] = 0
    ed._restore_tabs_state(saved)
    assert ed._tabs().active.is_all and btn.instate(['disabled'])


def test_remove_from_tab_button_is_off_on_all_and_works_on_a_tab(
        three_samples):
    from openflo.tree_ids import sample_iid
    ed, _ = three_samples
    ts = ed._tabs()
    btn = ed._tab_remove_btn
    assert btn.winfo_manager() == 'pack'               # it is in the tray
    assert 'Remove' in [str(w.cget('text')) for w in btn.master.winfo_children()
                        if w.winfo_class() == 'TButton']
    assert not btn.instate(['disabled'])               # tab A is open
    ed._switch_tab(0)
    assert btn.instate(['disabled'])                   # All tab
    ed._switch_tab(1)
    assert not btn.instate(['disabled'])
    ed.gate_tv.selection_set(sample_iid('s2'))
    btn.invoke()
    assert set(ts.active.members) == {'s1'}
    assert 's2' in ed._samples and ts.tabs[0].has_sample('s2')
    # Deleting every extra tab leaves All, and the button goes off again.
    while len(ts.tabs) > 1:
        ed._switch_tab(len(ts.tabs) - 1)
        ed._delete_tab()
    assert ts.active.is_all and btn.instate(['disabled'])


def test_every_remove_entry_has_help_and_the_menu_shows_it(
        three_samples, monkeypatch, menu_popups):
    from openflo.tree_ids import sample_iid, trial_iid
    ed, _ = three_samples
    for label in (TAB_ONLY, SAMPLE_TAB_ONLY, SESSION, TRIAL_TAB_ONLY,
                  TRIAL_SESSION):
        assert ed._MENU_ITEM_HELP.get(label), f'no help for {label!r}'
    assert "can't be undone" in ed._MENU_ITEM_HELP[SESSION]
    assert "stay loaded" in ed._MENU_ITEM_HELP[TAB_ONLY]
    assert "stay loaded" in ed._MENU_ITEM_HELP[SAMPLE_TAB_ONLY]
    # Hovering an entry of the right-click menu, or of its Remove submenu,
    # puts its help in the status bar, as the menubar does.
    top = right_click_row(ed, monkeypatch, menu_popups, sample_iid('s1'))
    assert '<<MenuSelect>>' in top.bind()
    menu = submenu(top, SAMPLE_REMOVE)
    assert '<<MenuSelect>>' in menu.bind()
    for label in (SAMPLE_TAB_ONLY, SESSION):
        idx = _labels_index(menu, label)
        monkeypatch.setattr(menu, 'index',
                            lambda what, _i=idx: _i if what == 'active'
                            else type(menu).index(menu, what))
        ed._tooltips_enabled.set(True)
        ed._on_menu_select(SimpleNamespace(widget=menu))
        assert ed.status_var.get() == ed._MENU_ITEM_HELP[label]
    menu = right_click_row(ed, monkeypatch, menu_popups, trial_iid('T'))
    assert '<<MenuSelect>>' in menu.bind()


def test_both_buttons_have_hover_tips(monkeypatch):
    import tkinter as tk

    import openflo.gui as gui
    tips = {}
    real = gui.ViewGateEditorWindow._tip

    def record(self, w, text):
        try:
            tips[str(w.cget('text'))] = text
        except tk.TclError:                           # a widget with no text option is not a button
            pass
        return real(self, w, text)
    monkeypatch.setattr(gui.ViewGateEditorWindow, '_tip', record)
    root2, ed2 = editor_or_skip(accept_confirms=False)
    try:
        assert "every tab" in tips['Remove']
        assert "Remove from tab" in tips['Remove']
        assert "stay loaded" in tips['Remove from tab']
        assert ed2._tab_remove_btn.instate(['disabled'])  # opens on All
    finally:
        root2.destroy()


def test_a_command_status_outlasts_the_menu_closing(
        three_samples, monkeypatch, menu_popups):
    """Hover an entry (its help shows), pick it, then the menu unmaps: the
    command's result stays, whichever order the platform sends these in.
    Closing without picking still restores the earlier text."""
    from openflo.tree_ids import sample_iid
    ed, _ = three_samples
    ed._tooltips_enabled.set(True)
    menu = submenu(right_click_row(ed, monkeypatch, menu_popups,
                                   sample_iid('s1')), SAMPLE_REMOVE)
    idx = _labels_index(menu, SAMPLE_TAB_ONLY)
    monkeypatch.setattr(menu, 'index', lambda w, _i=idx: _i if w == 'active'
                        else type(menu).index(menu, w))
    ed.status_var.set('before')
    ed._on_menu_select(SimpleNamespace(widget=menu))
    ed._on_menu_unmap()                       # closed without a pick
    assert ed.status_var.get() == 'before'
    ed._on_menu_select(SimpleNamespace(widget=menu))
    menu.invoke(idx)                          # command first...
    ed._on_menu_unmap()                       # ...then the unmap
    assert "still loaded" in ed.status_var.get()

