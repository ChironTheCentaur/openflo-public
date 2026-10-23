"""Tabs of the Samples & Gates tray (beta): the tab model, and in a real
editor that a tab shows and plots only its members, restores its own view
and zoom, takes copied rows, applies rows to other tabs, and survives a
session save. Synthetic samples and generic channels only."""
from __future__ import annotations

import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from openflo import view_tabs as vt  # noqa: E402
from tests.conftest import gui_unavailable  # noqa: E402

COLS = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A']


def _gates():
    return {'g1': {'parent_id': None}, 'g2': {'parent_id': 'g1'},
            'g3': {'parent_id': 'g2'}, 'g4': {'parent_id': None}}


def test_all_tab_holds_everything_and_stays_first():
    ts = vt.TabSet()
    assert ts.names() == ['All'] and ts.active.is_all
    assert ts.active.shown_gates('any', _gates()) == set(_gates())
    idx = ts.new_tab('Mine')
    ts.remove(0)                                  # All can't be deleted
    assert ts.tabs[0].is_all and len(ts.tabs) == 2
    ts.remove(idx)
    assert ts.names() == ['All']
    # A file whose first tab isn't All gets one back in front.
    back = vt.TabSet.from_dict({'tabs': [{'name': 'x', 'members': {}}]})
    assert back.names() == ['All', 'x']


def test_a_copied_gate_brings_its_ancestors_and_subtree():
    t = vt.ViewTab('t', members={})
    t.add('s1', 'g2')
    assert t.shown_gates('s1', _gates()) == {'g1', 'g2', 'g3'}
    assert t.shown_gates('s2', _gates()) == set()
    t.add('s1')                                   # the whole sample
    assert t.shown_gates('s1', _gates()) == set(_gates())
    t.add('s1', 'g4')                             # already every gate
    assert t.members == {'s1': None}
    t.remove('s1')
    assert not t.has_sample('s1')


def test_names_stay_unique_and_round_trip():
    ts = vt.TabSet()
    a = ts.new_tab('Day')
    b = ts.new_tab('Day')
    assert ts.tabs[b].name == 'Day 2'
    ts.rename(b, 'Day')                           # taken: made unique
    assert ts.tabs[b].name == 'Day 2'
    ts.tabs[a].add('s1', 'g2')
    ts.tabs[a].view = {'x': 'FL1-A', 'xlim': [0.0, 1.0]}
    ts.tabs[a].plot_enabled = {'s1': False}
    ts.current = a
    dup = ts.duplicate(0, samples=['s1', 's2'])   # a copy of All
    assert ts.tabs[dup].members == {'s1': None, 's2': None}
    back = vt.TabSet.from_dict(ts.to_dict())
    assert back.to_dict() == ts.to_dict()


def test_apply_rows_adds_and_switches():
    t1, t2 = vt.ViewTab('a', members={}), vt.ViewTab('b', members={'s1': None})
    vt.apply_rows([t1, t2], [('s1', None), ('s1', 'g2')],
                  {'s1': False}, {('s1', 'g2'): False})
    assert t1.has_sample('s1') and t1.plot_enabled['s1'] is False
    assert t1.gate_enabled['s1']['g2'] is False
    assert t2.members == {'s1': None}             # still every gate
    assert t2.gate_enabled['s1']['g2'] is False


# ── in the editor ────────────────────────────────────────────────────────

def _editor_or_skip():
    try:
        import tkinter as tk
    except ImportError:
        gui_unavailable('tkinter not available')
    try:
        root = tk.Tk()
    except Exception as e:                            # noqa: BLE001
        gui_unavailable(f'Tk cannot initialise without a display: {e}')
    root.withdraw()
    import importlib
    gui = importlib.import_module('openflo.gui')
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    return root, ed


def _load(ed, name, seed):
    from openflo.pipeline import FlowSample
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.uniform(1, 1000, (400, len(COLS))), columns=COLS)
    ed._samples[name] = FlowSample.from_dataframe(df, name=name,
                                                  path=f'{name}.fcs')
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    if 'T' not in ed._trial_order:
        ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = True
    ed._channels = list(COLS)
    ed._channel_labels = {c: c for c in COLS}
    ed._populate_channel_combos()
    ed._set_active_sample(name)
    return ed._add_gate({'kind': 'threshold', 'name': 'Pos',
                         'channel': 'FL1-A', 'op': '>', 'value': 500.0,
                         'parent_id': None}, audit=False)


def _tab_point(ed, idx):
    """Screen point in the middle of tab ``idx`` on the strip, found with
    the editor's own hit test over the whole notebook: where the tabs sit
    is the theme's call (Aqua centres them and pads them down from the
    top), so no fixed row or edge is assumed."""
    nb = ed._tab_nb
    x0, y0 = nb.winfo_rootx(), nb.winfo_rooty()
    hits = [(x0 + x, y0 + y) for y in range(0, nb.winfo_height(), 2)
            for x in range(0, nb.winfo_width(), 2)
            if ed._tab_at(x0 + x, y0 + y) == idx]
    assert hits, f'tab {idx} not found on the strip'
    xs, ys = sorted(p[0] for p in hits), sorted(p[1] for p in hits)
    return xs[len(xs) // 2], ys[len(ys) // 2]


def test_a_tab_is_found_wherever_the_theme_puts_it():
    root, ed = _editor_or_skip()
    try:
        from tkinter import ttk
        _load(ed, 's1', 1)
        ed._refresh_gate_list()
        b = ed._tabs().new_tab('B', members={'s1': None})
        # Aqua's notebook metrics: tabs centred, the row padded 8 px down.
        style = ttk.Style(ed)
        style.configure('Probe.TNotebook', tabposition='n',
                        tabmargins=(10, 0), padding=(18, 8, 18, 17))
        ed._tab_nb.configure(style='Probe.TNotebook')
        ed._rebuild_tab_strip()
        ed.deiconify()
        ed.update()
        nb = ed._tab_nb
        x, y = _tab_point(ed, b)
        assert y - nb.winfo_rooty() >= 8
        assert ed._tab_at(x, y) == b
        assert ed._tab_at(x, nb.winfo_rooty() + 2) is None
    finally:
        root.destroy()


def _tree_rows(ed):
    out = []

    def walk(p=''):
        for iid in ed.gate_tv.get_children(p):
            out.append(iid)
            walk(iid)
    walk()
    return out


def test_a_tab_shows_plots_and_restores_only_its_own(monkeypatch):
    root, ed = _editor_or_skip()
    try:
        from openflo.tree_ids import gate_iid, sample_iid
        _load(ed, 's1', 1)
        g2 = _load(ed, 's2', 2)
        ed._refresh_gate_list()
        ed.x_combo.set('FSC-A')
        ed.y_combo.set('SSC-A')
        ed.mode_var.set('dot')
        ed._replot()
        # New tab from s2 selected.
        ed.gate_tv.selection_set(sample_iid('s2'))
        ed.update()
        ed._new_tab_from_selection()
        ts = ed._tabs()
        assert ts.current == 1 and ts.active.members == {'s2': None}
        rows = _tree_rows(ed)
        assert sample_iid('s2') in rows and sample_iid('s1') not in rows
        assert gate_iid('s2', g2) in rows
        assert ed._selected_samples() == ['s2']
        # This tab's own view, zoomed.
        ed.x_combo.set('FL1-A')
        ed.y_combo.set('FL2-A')
        ed._replot()
        ed.ax.set_xlim(100.0, 200.0)
        ed._remember_zoom()                 # what a wheel / box zoom does
        ed._switch_tab(0)
        assert ed._selected_samples() == ['s1', 's2']
        assert ed.x_combo.get() == ed._fmt_channel('FSC-A')
        assert sample_iid('s1') in _tree_rows(ed)
        assert ed.ax.get_xlim() != pytest.approx((100.0, 200.0))
        ed._switch_tab(1)
        assert ed.x_combo.get() == ed._fmt_channel('FL1-A')
        assert ed.ax.get_xlim() == pytest.approx((100.0, 200.0))
        # ...and it stays through later redraws (a gate edit, a toggle).
        ed._replot()
        ed._replot()
        assert ed.ax.get_xlim() == pytest.approx((100.0, 200.0))
        # Back and forth again: still kept.
        ed._switch_tab(0)
        ed._switch_tab(1)
        ed._replot()
        assert ed.ax.get_xlim() == pytest.approx((100.0, 200.0))
        # Reset forgets it.
        ed._reset_plot_view()
        ed._replot()
        assert ed.ax.get_xlim() != pytest.approx((100.0, 200.0))
        # On/off is per tab: switch s2 off here, it stays on in All.
        ed._sample_plot_enabled['s2'] = False
        ed._switch_tab(0)
        assert ed._sample_plot_enabled['s2'] is True
        ed._switch_tab(1)
        assert ed._sample_plot_enabled['s2'] is False
    finally:
        root.destroy()


def test_copy_to_tab_apply_to_tabs_and_the_menu():
    root, ed = _editor_or_skip()
    try:
        import tkinter as tk

        from openflo.tree_ids import gate_iid
        g1 = _load(ed, 's1', 1)
        _load(ed, 's2', 2)
        ed._refresh_gate_list()
        ts = ed._tabs()
        a = ts.new_tab('A')
        b = ts.new_tab('B')
        ed._rebuild_tab_strip()
        ed.gate_tv.selection_set(gate_iid('s1', g1))
        ed._copy_rows_to_tab(a)
        assert ts.tabs[a].members == {'s1': [g1]}
        # Switch the gate off in All and push that to every tab.
        ed._sample_gates['s1'][g1]['enabled'] = False
        ed._apply_rows_to_all_tabs()
        assert ts.tabs[a].gate_enabled['s1'][g1] is False
        assert ts.tabs[b].gate_enabled['s1'][g1] is False
        assert ts.tabs[b].members == {'s1': [g1]}
        # The tree's right-click menu carries the tab entries.
        menu = tk.Menu(ed, tearoff=0)
        ed._add_tab_menu_items(menu)
        labels = [menu.entrycget(i, 'label') for i in range(menu.index('end') + 1)
                  if menu.type(i) != 'separator']
        assert 'Copy to tab' in labels and 'Apply to other tabs' in labels
    finally:
        root.destroy()


def test_tabs_and_layout_are_saved_in_the_session():
    root, ed = _editor_or_skip()
    try:
        from openflo import layout_page as lp
        from openflo.session_format import SESSION_KEYS, migrate_session
        from openflo.tree_ids import sample_iid
        _load(ed, 's1', 1)
        _load(ed, 's2', 2)
        ed._refresh_gate_list()
        ed.gate_tv.selection_set(sample_iid('s2'))
        ed._new_tab_from_selection()
        ed._layout_doc = lp.LayoutDoc()
        ed._layout_doc.pages[0].add(lp.text_item('x'))
        state = ed._session_state()
        assert {'view_tabs', 'layout'} <= set(SESSION_KEYS) <= set(state)
        assert [t['name'] for t in state['view_tabs']['tabs']] == ['All',
                                                                  'Tab 1']
        assert state['view_tabs']['current'] == 1
        # A fresh editor reads them back.
        ed2_root, ed2 = _editor_or_skip()
        try:
            data, _notes = migrate_session(dict(state))
            ed2._restore_tabs_state(data['view_tabs'])
            assert ed2._tabs().names() == ['All', 'Tab 1']
            assert ed2._tabs().active.members == {'s2': None}
        finally:
            ed2_root.destroy()
        # A version-1 session (no tabs, no layout) still opens.
        v1 = {k: state[k] for k in state if k not in ('view_tabs', 'layout')}
        v1['version'] = 1
        data, notes = migrate_session(v1)
        assert notes and data['view_tabs'] is None and data['layout'] is None
    finally:
        root.destroy()


# ── v2 sessions with cluster populations ─────────────────────────────────

@pytest.mark.parametrize('method,col', [('leiden', 'leiden'),
                                        ('flowsom', 'flowsom_meta')])
def test_v2_session_with_label_populations_saves_tabs_and_layout(
        tmp_path, monkeypatch, method, col):
    import json

    from openflo import layout_page as lp
    from openflo.layout_render import CAT_PREFIX, SampleContext
    from openflo.session_format import SESSION_VERSION
    from openflo.tree_ids import gate_iid
    from tests.test_compensation_editor_apply import _load as _load_fcs
    from tests.test_compensation_editor_apply import _write_mixed
    from tests.test_session_restore_transforms import _restore
    from tests.test_transform_records import (
        _new_editor,
        capture_warnings,
        isolate_home,
    )
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                            # noqa: BLE001
        gui_unavailable(f'Tk cannot initialise: {e}')
    root.withdraw()
    try:
        ed = _new_editor(root)
        _write_mixed(tmp_path / 'a.fcs')
        _load_fcs(root, ed, [('a', tmp_path / 'a.fcs')])
        ed._set_active_sample('a')
        ed._run_clustering(method=method, all_samples=False, k=15, grid=4,
                           n_meta=3, embedding='none', resolution=0.3)
        gates = ed._sample_gates['a']
        cat = next(gid for gid, g in gates.items()
                   if g.get('kind') == 'category' and g.get('channel') == col)
        # A tab holding one cluster population, and a layout plotting it.
        ed._refresh_gate_list()
        ed.gate_tv.selection_set(gate_iid('a', cat))
        ed._new_tab_from_selection()
        key = SampleContext(ed, 'a').keys[cat]
        ed._layout_doc = lp.LayoutDoc()
        ed._layout_doc.pages[0].add(lp.plot_item(
            population=key, x='FLA-A', color=f'{CAT_PREFIX}{col}'))
        path = tmp_path / f'{method}.flowsession'
        ed._write_session(str(path))
        saved = json.loads(path.read_text(encoding='utf-8'))
        assert saved['version'] == SESSION_VERSION
        assert saved['view_tabs']['tabs'][1]['members'] == {'a': [cat]}
        ed2 = _restore(root, path, raw_first=True)
        assert ed2._tabs().names() == ['All', 'Tab 1']
        assert ed2._tabs().tabs[1].members == {'a': [cat]}
        item = next(iter(ed2._layout_doc.pages[0].items.values()))
        assert item['populations'] == [key]
        assert SampleContext(ed2, 'a').keys[cat] == key
    finally:
        root.destroy()


def test_apply_within_one_tab_touches_only_the_chosen_samples():
    tab = vt.ViewTab('B', members={'s1': None, 's2': None, 's3': ['h1']})
    keys = {('s1', 'g1'): 'n:pos', ('s2', 'k1'): 'n:pos', ('s3', 'h2'): 'n:pos'}
    by = {(s, k): g for (s, g), k in keys.items()}
    n = vt.apply_within(tab, [('s1', 'g1'), ('s1', None)], ['s3'],
                        {'s1': False}, {('s1', 'g1'): False},
                        lambda s, g: keys.get((s, g)),
                        lambda s, k: by.get((s, k)))
    assert n == 2
    assert tab.gate_enabled == {'s3': {'h2': False}}
    assert tab.plot_enabled == {'s3': False}
    assert tab.members['s3'] == ['h1', 'h2']        # now shown there
    assert 's2' not in tab.gate_enabled


def test_apply_within_a_tab_from_the_editor():
    root, ed = _editor_or_skip()
    try:
        from openflo.tree_ids import gate_iid
        g1 = _load(ed, 's1', 1)
        g2 = _load(ed, 's2', 2)
        g3 = _load(ed, 's3', 3)
        ed._refresh_gate_list()
        ts = ed._tabs()
        b = ts.new_tab('B', members={'s1': None, 's2': None, 's3': None})
        ed._rebuild_tab_strip()
        ed._sample_gates['s1'][g1]['enabled'] = False
        ed.gate_tv.selection_set(gate_iid('s1', g1))
        assert ed._apply_rows_within_tab(b, ['s3']) == 1
        assert ts.tabs[b].gate_enabled == {'s3': {g3: False}}
        ed._switch_tab(b)
        assert ed._sample_gates['s3'][g3]['enabled'] is False
        assert ed._sample_gates['s2'][g2]['enabled'] is True
        # The dialog lists that tab's samples and filters them by Find.
        import tkinter as tk
        ed._switch_tab(0)
        ed.gate_tv.selection_set(gate_iid('s1', g1))
        before = set(ed.winfo_children())
        ed._ask_apply_rows_within_tab()
        (dlg,) = [w for w in ed.winfo_children()
                  if w not in before and isinstance(w, tk.Toplevel)]

        def find(w, cls):
            for c in w.winfo_children():
                if isinstance(c, cls):
                    yield c
                yield from find(c, cls)
        (lb,) = list(find(dlg, tk.Listbox))
        assert lb.get(0, 'end') == ('s1', 's2', 's3')
        from tkinter import ttk
        entry = next(e for e in find(dlg, ttk.Entry)
                     if not isinstance(e, ttk.Combobox))
        entry.insert(0, '3')
        dlg.update()
        assert lb.get(0, 'end') == ('s3',)
        dlg.destroy()
        menu = tk.Menu(ed, tearoff=0)
        ed._add_tab_menu_items(menu)
        ap = menu.nametowidget(menu.entrycget(
            [menu.entrycget(i, 'label') if menu.type(i) != 'separator'
             else '' for i in range(menu.index('end') + 1)].index(
                'Apply to other tabs'), 'menu'))
        labels = [ap.entrycget(i, 'label') for i in range(ap.index('end') + 1)]
        assert 'Within a tab (choose samples)…' in labels
    finally:
        root.destroy()


# ── round 3: dropping rows onto a tab, the Presort zone ──────────────────

def test_drop_goes_to_presort_when_arranged_else_to_the_top():
    t = vt.ViewTab('B', members={'s1': None, 's2': None})
    trial = {'s1': 'D1', 's2': 'D1', 's3': 'D1', 's4': 'D2'}.get
    order = ['s1', 's2', 's3', 's4']
    assert not t.is_grouped(trial, order)
    assert t.drop([('s3', None)], grouped=False) == 'top'
    assert t.samples(order)[0] == 's3' and t.presort == []
    # Now it has its own order: arranged, so the next drop waits.
    assert t.is_grouped(trial, order)
    assert t.drop([('s4', 'g1')], grouped=True) == 'presort'
    assert t.presort == [('s4', 'g1')] and not t.has_sample('s4')
    t.place([('s4', 'g1')], before='s1')
    assert t.presort == [] and t.members['s4'] == ['g1']
    assert t.samples(order) == ['s3', 's4', 's1', 's2']
    # Samples from two groups also count as arranged.
    u = vt.ViewTab('C', members={'s1': None, 's4': None})
    assert u.is_grouped(trial, order)
    back = vt.ViewTab.from_dict(t.to_dict())
    assert back.order == t.order and back.presort == t.presort


def test_dragging_rows_onto_a_tab_and_placing_from_presort():
    root, ed = _editor_or_skip()
    try:
        from types import SimpleNamespace

        from openflo.tree_ids import sample_iid
        for i, n in enumerate(('s1', 's2', 's3', 's4'), 1):
            _load(ed, n, i)
        ed._sample_trial['s4'] = 'D2'
        ed._trial_order.append('D2')
        ed._refresh_gate_list()
        ts = ed._tabs()
        b = ts.new_tab('B', members={'s1': None, 's2': None})
        ed._rebuild_tab_strip()
        ed.deiconify()
        ed.update()
        # Find tab B on the strip and drop s3 there: B isn't arranged, so
        # s3 goes to the top of it.
        xb, y = _tab_point(ed, b)
        ed._press_selection = ()
        ev = SimpleNamespace(x_root=xb, y_root=y)
        assert ed._maybe_drop_to_tab(sample_iid('s3'), ev)
        assert ts.tabs[b].samples(ed._sample_order)[0] == 's3'
        # B has its own order now: the next drop waits in its Presort zone.
        assert ed._maybe_drop_to_tab(sample_iid('s4'), ev)
        assert ts.tabs[b].presort == [('s4', None)]
        ed._switch_tab(b)
        assert ed.gate_tv.exists('PRE:zone')
        assert not ed.gate_tv.exists(sample_iid('s4'))
        # Dragged from the zone onto s1: placed before s1, zone gone.
        ed._place_presort_row('PRE:0', sample_iid('s1'))
        assert ts.tabs[b].samples(ed._sample_order)[:3] == ['s3', 's4', 's1']
        assert not ed.gate_tv.exists('PRE:zone')
        assert ed.gate_tv.exists(sample_iid('s4'))
    finally:
        root.destroy()


def test_dragging_back_from_the_pipeline_to_the_tray():
    root, ed = _editor_or_skip()
    try:
        from types import SimpleNamespace

        from openflo.tree_ids import sample_iid
        from openflo.workspace import PipelineWorkspaceView
        _load(ed, 's1', 1)
        g2 = _load(ed, 's2', 2)
        _load(ed, 's3', 3)
        ed._refresh_gate_list()
        ts = ed._tabs()
        b = ts.new_tab('B', members={'s1': None})
        ed._rebuild_tab_strip()
        ed._switch_tab(b)
        ed.deiconify()
        ed.update()
        # A Pipeline item (s2's gate) dropped on the tree joins tab B.
        tv = ed.gate_tv
        x, y = tv.winfo_rootx() + 20, tv.winfo_rooty() + 30
        view = SimpleNamespace(
            _editor=ed, _press_row='M1',
            _selected_targets=lambda: [('item', 'M1')],
            model=SimpleNamespace(item=lambda m: {'sample': 's2',
                                                  'gate_id': g2}))
        view._dragged_rows = lambda: PipelineWorkspaceView._dragged_rows(view)
        ev = SimpleNamespace(x_root=x, y_root=y)
        assert PipelineWorkspaceView._drop_to_tray(view, ev)
        assert ts.tabs[b].members['s2'] == [g2]
        assert tv.exists(sample_iid('s2'))
        # Onto a tab on the strip: it lands there (All takes it as is).
        x0, ty = _tab_point(ed, b)
        assert ed._drop_from_workspace([('s3', None)], x0, ty)
        assert ts.tabs[b].has_sample('s3')
        # Anywhere else: not consumed.
        assert not ed._drop_from_workspace([('s3', None)], -5000, -5000)
    finally:
        root.destroy()
