"""The sample and gate right-click menus, regrouped into submenus.

A short top level (Copy sample, Paste, Copy for… ▸, Move to another window,
New gate ▸, Sample info ▸, Remove ▸ for a sample row; Copy / Cut / Paste,
Edit ▸, New gate ▸, Backgate, Export, Tabs ▸, Delete for a gate row), with
every action of the old flat menus still reachable. Copy sample copies the
sample with its gates; Paste puts it in another window or on another tab of
this one. Synthetic samples and generic channels only; no menu or window is
shown (tk_popup is replaced by a recorder) and the editor stays withdrawn.
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

SAMPLE_TOP = ["Copy sample (Ctrl+C)", "Paste (Ctrl+V)", "Copy for…",
              "Move to another window", "New gate", "Sample info", "Remove"]
GATE_TOP = ["Copy gate (Ctrl+C)", "Cut gate (Ctrl+X)", "Paste (Ctrl+V)",
            "Edit", "New gate", "Backgate (show on plot)",
            "Export population as FCS…", "Tabs", "Delete gate (cascade)"]


@pytest.fixture
def editor():
    """s1, s2, s3 loaded (one gate each) and two extra tabs, A = {s1, s2}
    and B = {s1}; tab A is open."""
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


def _record(monkeypatch, ed, names):
    """Replace the editor's handlers ``names`` with recorders of
    ``(name, args)``."""
    calls = []
    for n in names:
        monkeypatch.setattr(
            ed, n, lambda *a, _n=n, **k: calls.append((_n, a)))
    return calls


# ── the menus' shape ────────────────────────────────────────────────────

def test_sample_menu_top_level_is_short(editor, monkeypatch, menu_popups):
    from openflo.tree_ids import sample_iid
    ed, _ = editor
    menu = right_click_row(ed, monkeypatch, menu_popups, sample_iid('s1'))
    assert menu_labels(menu) == SAMPLE_TOP
    assert menu_labels(submenu(menu, "Copy for…")) == [
        "Another window, with gates",
        "Another window, without gates (file path)", "Another tab",
        "Gates only, to other samples…", "On/off state, to other tabs"]
    assert menu_labels(submenu(submenu(menu, "Copy for…"), "Another tab")) == [
        "B", "New tab…"]
    assert menu_labels(submenu(menu, "New gate")) == ["Boolean gate…",
                                               "Combination gates…"]
    assert menu_labels(submenu(menu, "Sample info")) == [
        "Sample color…", "Add note…", "Set aside from comparisons…"]


def test_gate_menu_top_level_is_short(editor, monkeypatch, menu_popups):
    from openflo.tree_ids import gate_iid
    ed, gids = editor
    menu = right_click_row(ed, monkeypatch, menu_popups, gate_iid('s1', gids['s1']))
    assert menu_labels(menu) == GATE_TOP
    assert menu_labels(submenu(menu, "Edit")) == [
        "Coordinates…", "Re-place on each sample…", "Gate color…"]
    assert menu_labels(submenu(menu, "Tabs")) == [
        "Copy to tab", "Apply to other tabs", "Remove from this tab"]


# ── nothing lost: every old action is still reachable ───────────────────

# The handler each entry runs, and what it is handed: with s1 and s2
# selected and s1 right-clicked, the selection-wide entries get both
# samples, the per-sample ones s1, the tab ones the tab's index (B is 2).
BOTH = (['s1', 's2'],)
SAMPLE_PATHS = {
    ("Copy sample (Ctrl+C)",): ('_copy_samples_for_paste', BOTH),
    ("Copy for…", "Another window, with gates"):
        ('_copy_samples_for_paste', BOTH),
    ("Copy for…", "Another window, without gates (file path)"):
        ('_copy_sample_paths', BOTH),
    ("Copy for…", "Another tab", "B"): ('_copy_rows_to_tab', (2,)),
    ("Copy for…", "Another tab", "New tab…"): ('_copy_rows_to_new_tab', ()),
    ("Copy for…", "Gates only, to other samples…"):
        ('_open_copy_gates_dialog', ()),
    ("Copy for…", "On/off state, to other tabs", "All tabs"):
        ('_apply_rows_to_all_tabs', ()),
    ("Copy for…", "On/off state, to other tabs", "Choose tabs…"):
        ('_ask_apply_rows_to_tabs', ()),
    ("Copy for…", "On/off state, to other tabs",
     "Within a tab (choose samples)…"): ('_ask_apply_rows_within_tab', ()),
    ("Move to another window",): ('_send_samples_transfer', BOTH),
    ("New gate", "Boolean gate…"): ('_open_boolean_dialog', ()),
    ("New gate", "Combination gates…"): ('_open_combination_dialog', ()),
    ("Sample info", "Sample color…"): ('_pick_sample_color', ('s1',)),
    ("Sample info", "Add note…"): ('_prompt_sample_note', ('s1',)),
    ("Sample info", "Set aside from comparisons…"):
        ('_prompt_set_aside', BOTH),
    ("Remove", "Clear gates (keep sample)"): ('_clear_selected_gate', ()),
    ("Remove", "From this tab"): ('_remove_rows_from_tab', ()),
    ("Remove", "From session (all tabs)…"): ('_remove_selected', ()),
}


def test_every_sample_action_is_still_reachable(editor, monkeypatch, menu_popups):
    from openflo.tree_ids import sample_iid
    ed, _ = editor
    calls = _record(monkeypatch, ed, {h for h, _a in SAMPLE_PATHS.values()}
                    | {'_return_sample', '_on_paste'})
    both = (sample_iid('s1'), sample_iid('s2'))
    for path, want in SAMPLE_PATHS.items():
        ed.gate_tv.selection_set(both)
        menu = right_click_row(ed, monkeypatch, menu_popups,
                               sample_iid('s1'), select=False)
        calls.clear()
        invoke_menu_path(menu, *path)
        assert calls == [want], (path, calls)
    # A set-aside sample offers Return and Change reason instead.
    ed._sample_set_aside['s1'] = 'test'
    ed.gate_tv.selection_set(both)
    menu = right_click_row(ed, monkeypatch, menu_popups, sample_iid('s1'),
                           select=False)
    assert menu_labels(submenu(menu, "Sample info"))[2:] == [
        "Return to comparisons", "Change set-aside reason…"]
    calls.clear()
    invoke_menu_path(menu, "Sample info", "Return to comparisons")
    invoke_menu_path(menu, "Sample info", "Change set-aside reason…")
    assert calls == [('_return_sample', BOTH), ('_prompt_set_aside', BOTH)]


GATE_PATHS = {
    ("Copy gate (Ctrl+C)",): '_on_copy',
    ("Cut gate (Ctrl+X)",): '_on_cut',
    ("Edit", "Coordinates…"): '_edit_gate_coordinates',
    ("Edit", "Re-place on each sample…"): '_refit_gate_per_sample',
    ("Edit", "Gate color…"): '_pick_gate_color',
    ("New gate", "Boolean gate…"): '_open_boolean_dialog',
    ("New gate", "Combination gates…"): '_open_combination_dialog',
    ("Backgate (show on plot)",): '_backgate_selected',
    ("Export population as FCS…",): '_export_population_fcs',
    ("Tabs", "Copy to tab", "B"): '_copy_rows_to_tab',
    ("Tabs", "Copy to tab", "New tab…"): '_copy_rows_to_new_tab',
    ("Tabs", "Apply to other tabs", "All tabs"): '_apply_rows_to_all_tabs',
    ("Tabs", "Apply to other tabs", "Choose tabs…"):
        '_ask_apply_rows_to_tabs',
    ("Tabs", "Apply to other tabs", "Within a tab (choose samples)…"):
        '_ask_apply_rows_within_tab',
    ("Tabs", "Remove from this tab"): '_remove_rows_from_tab',
    ("Delete gate (cascade)",): '_clear_selected_gate',
}


def test_every_gate_action_is_still_reachable(editor, monkeypatch, menu_popups):
    from openflo.tree_ids import gate_iid
    ed, gids = editor
    calls = _record(monkeypatch, ed, set(GATE_PATHS.values())
                    | {'_clear_backgate', '_edit_autoclean_params'})
    for path, handler in GATE_PATHS.items():
        menu = right_click_row(ed, monkeypatch, menu_popups,
                            gate_iid('s1', gids['s1']))
        calls.clear()
        invoke_menu_path(menu, *path)
        assert [c[0] for c in calls] == [handler], (path, calls)
    # With backgating on, Clear backgating sits beside Backgate.
    ed._backgate = {'s1': [gids['s1']]}
    menu = right_click_row(ed, monkeypatch, menu_popups, gate_iid('s1', gids['s1']))
    top = menu_labels(menu)
    assert top[top.index("Backgate (show on plot)") + 1] == "Clear backgating"
    calls.clear()
    invoke_menu_path(menu, "Clear backgating")
    assert calls == [('_clear_backgate', ())]


def test_auto_clean_gate_menu_keeps_its_parameters_and_removal(
        editor, monkeypatch, menu_popups):
    from openflo.tree_ids import gate_iid
    ed, _ = editor
    ed._switch_tab(0)                                 # s3 shows on All
    ac = ed._add_gate({'kind': 'autoclean', 'name': 'Clean',
                       'methods': [], 'parent_id': None}, audit=False)
    ed._refresh_gate_list()
    calls = _record(monkeypatch, ed, {'_edit_autoclean_params',
                                      '_clear_selected_gate'})
    menu = right_click_row(ed, monkeypatch, menu_popups, gate_iid('s3', ac))
    assert menu_labels(submenu(menu, "Edit"))[0] == "Auto-clean parameters…"
    assert menu_labels(menu)[-1] == "Remove auto-clean gate"
    invoke_menu_path(menu, "Edit", "Auto-clean parameters…")
    invoke_menu_path(menu, "Remove auto-clean gate")
    assert [c[0] for c in calls] == ['_edit_autoclean_params',
                                     '_clear_selected_gate']


def test_every_entry_has_help_and_every_submenu_shows_it(
        editor, monkeypatch, menu_popups):
    from openflo.tree_ids import gate_iid, sample_iid
    ed, gids = editor
    tab_names = {t.name for t in ed._tabs().tabs}
    for iid in (sample_iid('s1'), gate_iid('s1', gids['s1'])):
        top = right_click_row(ed, monkeypatch, menu_popups, iid)
        for m in all_menus(top):
            assert '<<MenuSelect>>' in m.bind()
            for label in menu_labels(m):
                if label in tab_names:
                    continue                  # a tab's own name
                assert ed._MENU_ITEM_HELP.get(label), f'no help: {label!r}'
    # Hovering a submenu entry shows its help in the status bar.
    menu = submenu(right_click_row(ed, monkeypatch, menu_popups, sample_iid('s1')),
                "Copy for…")
    monkeypatch.setattr(menu, 'index', lambda what: 1 if what == 'active'
                        else type(menu).index(menu, what))
    ed._tooltips_enabled.set(True)
    ed._on_menu_select(SimpleNamespace(widget=menu))
    assert ed.status_var.get() == ed._MENU_ITEM_HELP[
        "Another window, without gates (file path)"]


# ── what Copy sample and Paste do ───────────────────────────────────────

def test_copy_sample_copies_it_with_its_gates(editor):
    from openflo.tree_ids import sample_iid
    ed, gids = editor
    ed.gate_tv.selection_set(sample_iid('s2'))
    ed._on_copy()                                     # Ctrl+C on the row
    bundle = ed._read_transfer_bundle()
    assert bundle is not None, 'Ctrl+C on a sample did not copy the sample'
    [entry] = bundle['samples']
    assert entry['name'] == 's2' and entry['path'] == 's2.fcs'
    assert [g['id'] for g in entry['gates']] == [gids['s2']]
    assert "Paste" in ed.status_var.get()


def test_copy_without_gates_copies_only_the_paths(editor, monkeypatch, menu_popups):
    from openflo.tree_ids import sample_iid
    ed, _ = editor
    ed._switch_tab(0)
    ed.gate_tv.selection_set((sample_iid('s1'), sample_iid('s3')))
    monkeypatch.setattr(ed.gate_tv, 'identify_row',
                        lambda _y: sample_iid('s1'))
    ed._on_right_click(SimpleNamespace(x=1, y=1, x_root=0, y_root=0))
    invoke_menu_path(menu_popups[-1], "Copy for…",
            "Another window, without gates (file path)")
    assert ed.clipboard_get() == 's1.fcs\ns3.fcs'
    assert ed._read_transfer_bundle() is None
    assert "without gates" in ed.status_var.get()


def test_paste_a_copied_sample_on_another_tab_adds_it_there(
        editor, monkeypatch, menu_popups):
    from openflo.tree_ids import sample_iid
    ed, gids = editor
    ts = ed._tabs()
    menu = right_click_row(ed, monkeypatch, menu_popups, sample_iid('s2'))
    invoke_menu_path(menu, "Copy sample (Ctrl+C)")             # from tab A
    ed._switch_tab(2)                                  # tab B = {s1}
    ed._sample_plot_enabled['s2'] = False              # shown off: kept so
    loaded = set(ed._samples)
    menu = right_click_row(ed, monkeypatch, menu_popups, sample_iid('s1'))
    invoke_menu_path(menu, "Paste (Ctrl+V)")
    assert ts.tabs[2].members == {'s1': None, 's2': None}   # every gate
    assert ts.tabs[2].plot_enabled['s2'] is False
    assert set(ed._samples) == loaded                  # nothing loaded twice
    assert list(ed._sample_gates['s2']) == [gids['s2']]
    assert "Added 1 sample(s), with their gates, to tab 'B'" in \
        ed.status_var.get()
    # On the All tab it is already there: nothing changes, and it says so.
    ed._switch_tab(0)
    before = {i: dict(t.members or {}) for i, t in enumerate(ts.tabs)}
    ed._on_paste()
    assert {i: dict(t.members or {}) for i, t in enumerate(ts.tabs)} == before
    assert "already loaded here" in ed.status_var.get()


def test_paste_loads_the_samples_that_are_not_here_yet(
        editor, monkeypatch, menu_popups):
    import json
    ed, _ = editor
    calls = _record(monkeypatch, ed, {'_paste_samples_transfer'})
    bundle = ed._sample_transfer_bundle(['s3'])
    bundle['samples'].append({'name': 'elsewhere', 'path': 'elsewhere.fcs',
                              'gates': [], 'trial': ''})
    ed.clipboard_clear()
    ed.clipboard_append(json.dumps(bundle))
    ed._on_paste()                                    # tab A = {s1, s2}
    assert 's3' in ed._tabs().active.members          # here: onto the tab
    assert calls == [('_paste_samples_transfer', ())]  # the rest: loaded


def test_paste_takes_whatever_was_copied_last(editor, monkeypatch, menu_popups):
    from openflo.tree_ids import gate_iid, sample_iid
    ed, gids = editor
    ts = ed._tabs()
    # Gates copied, then a sample: Paste takes the sample.
    ed.gate_tv.selection_set(gate_iid('s1', gids['s1']))
    ed._on_copy()
    ed._copy_samples_transfer(['s3'])        # e.g. from another window
    ed.gate_tv.selection_set(sample_iid('s2'))
    ed._set_active_sample('s2')
    ed._on_paste()
    assert 's3' in ts.active.members
    assert len(ed._sample_gates['s2']) == 1           # no gate pasted
    # A sample copied, then gates: Paste takes the gates.
    ed.gate_tv.selection_set(gate_iid('s1', gids['s1']))
    ed._on_copy()
    ed.gate_tv.selection_set(sample_iid('s2'))
    ed._set_active_sample('s2')
    ed._on_paste()
    assert len(ed._sample_gates['s2']) == 2           # the gate pasted
    # The same sample copied again after the gates (the clipboard text is
    # what it was when the gates were copied): Paste takes the sample.
    ts.active.remove('s3')
    ed._copy_samples_for_paste(['s3'])
    ed.gate_tv.selection_set(gate_iid('s1', gids['s1']))
    ed._on_copy()
    ed._copy_samples_for_paste(['s3'])
    ed.gate_tv.selection_set(sample_iid('s2'))
    ed._on_paste()
    assert 's3' in ts.active.members
    assert len(ed._sample_gates['s2']) == 2           # no gate pasted


# ── review fixes: a failed copy, a paste that changes nothing, a move ───

def test_a_failed_sample_copy_keeps_the_copied_gates(editor):
    from openflo.pipeline import FlowSample
    from openflo.tree_ids import gate_iid, sample_iid
    ed, gids = editor
    ed._copy_samples_for_paste(['s3'])               # a bundle on the board
    ed.gate_tv.selection_set(gate_iid('s1', gids['s1']))
    ed._on_copy()                                    # then gates
    # A sample with no raw .fcs (CSV / processed) can't be copied.
    ed._samples['s2'] = FlowSample.from_dataframe(
        ed._samples['s2'].data, name='s2', path='s2.csv')
    ed.gate_tv.selection_set(sample_iid('s2'))
    ed._on_copy()
    assert "no raw FCS" in ed.status_var.get()
    assert ed._clip_kind == 'gate_tree'               # the gates kept
    ed._set_active_sample('s2')
    ed._on_paste()
    assert len(ed._sample_gates['s2']) == 2           # and they paste
    assert 's3' not in ed._tabs().active.members


def test_paste_says_so_when_the_tab_already_shows_the_sample(editor):
    ed, _ = editor
    ts = ed._tabs()
    ed._copy_samples_for_paste(['s2'])
    before = dict(ts.active.members)                  # A = {s1, s2}
    ed._on_paste()
    assert ts.active.members == before
    assert "Added" not in ed.status_var.get()
    assert "already on tab 'A' with every gate" in ed.status_var.get()
    # A tab showing only some of its gates gets them all.
    ts.active.members['s2'] = []
    ed._on_paste()
    assert ts.active.members['s2'] is None
    assert "Added 1 sample(s)" in ed.status_var.get()


def test_a_move_from_another_window_takes_nothing_already_open_here(
        editor, monkeypatch, tmp_path):
    import json
    import os
    ed, _ = editor
    queued = []
    monkeypatch.setattr(ed, '_queue_fcs_loads', lambda paths: queued.extend(paths))
    new = tmp_path / 'new.fcs'
    new.write_bytes(b'')                              # a file to load
    # s3 is open here, read from a file that is on disk too.
    s3 = tmp_path / 's3.fcs'
    s3.write_bytes(b'')
    ed._samples['s3'].path = str(s3)
    ed._path_to_name[os.path.normcase(os.path.abspath(str(s3)))] = 's3'
    bundle = ed._sample_transfer_bundle(['s3'])
    bundle['move_id'] = 'other-window-1'
    bundle['move_pid'] = os.getpid() + 1              # another process
    bundle['samples'].append({'name': 'new', 'path': str(new),
                              'gates': [], 'trial': ''})
    ed.clipboard_clear()
    ed.clipboard_append(json.dumps(bundle))
    ed._on_paste()                                    # tab A = {s1, s2}
    assert 's3' not in ed._tabs().active.members     # not put on the tab
    assert queued == [str(new)]                       # only new loads
    key = os.path.normcase(os.path.abspath(str(new)))
    assert ed._inbound_move['pending'] == {key}       # s3 never claimed
    msg = ed.status_var.get()
    assert msg.count("Already open here (kept, NOT moved): s3.fcs") == 1
