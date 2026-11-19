"""The Pipeline Workspace view is built one section at a time.

``PipelineWorkspaceView._build`` used to make the whole view in one method
of 268 lines. It now calls one builder per part, in the order the widgets
were always made (Tk names widgets and packs them by creation order):
toolbar, run-bar variables, parameter rows, embedding rows, option rows,
run actions, tree, tree bindings. Between the embedding rows and the option
rows ``_build`` still syncs the [embed] boxes and starts the background
engine check, and it sets the drag state last.

These tests pin that order on a recording stand-in, what each builder
makes, and that every Tk variable a control shows outlives the builder
that made the control: a ``tk.Variable`` held only by a builder's local is
unset by Tcl when the builder returns, and the control then shows nothing
and its traces stop, with no error. All data is generated.
"""
from __future__ import annotations

import gc
import inspect
from types import SimpleNamespace

import pytest

from tests.conftest import gui_unavailable

BUILDERS = (
    '_build_toolbar', '_build_run_vars', '_build_param_rows',
    '_build_embed_rows', '_build_option_rows', '_build_run_actions',
    '_build_tree', '_build_tree_bindings',
)
# What _build calls, in order: the builders with the two mid-build calls
# between the embedding rows and the option rows.
BUILD_CALLS = (
    '_build_toolbar', '_build_run_vars', '_build_param_rows',
    '_build_embed_rows', '_sync_embed_boxes', '_check_embed_engines',
    '_build_option_rows', '_build_run_actions', '_build_tree',
    '_build_tree_bindings',
)
DRAG_STATE = ('_press_row', '_press_col', '_press_x', '_press_y',
              '_drag_active', '_drag_threshold', '_drop_hi')
RUN_VARS = ('_method_var', '_res_var', '_k_var', '_max_var', '_seed_var',
            '_umap_var', '_tsne_var', '_phate_var', '_trimap_var',
            '_pacmap_var', '_concat_var', '_reproducible_var',
            '_fast_graph_var', '_events_csv_var')
# A run_cfg with every value off its default, so a variable that lost its
# value (or was never set from the cfg) shows.
CFG = {'method': 'leiden', 'resolution': 0.5, 'k': 12, 'max_events': 2500,
       'seed': 'pilot run 3', 'umap': False, 'tsne': True, 'phate': True,
       'trimap': True, 'pacmap': True, 'concatenate': True,
       'reproducible': False, 'fast_graph': True, 'events_csv': True}
NOT_INSTALLED = 'not installed'


def _view_class():
    try:
        from openflo.workspace import PipelineWorkspaceView
    except ImportError as exc:
        gui_unavailable(f"the GUI cannot be imported: {exc}")
    return PipelineWorkspaceView


# ── the split itself ───────────────────────────────────────────────────────

def test_build_is_a_short_list_of_builders_in_order():
    cls = _view_class()
    assert [n for n in BUILDERS if not callable(cls.__dict__.get(n))] == []
    src = inspect.getsource(cls._build)
    assert len(src.splitlines()) < 60
    pos = [src.find(f'self.{n}(') for n in BUILD_CALLS]
    assert -1 not in pos, dict(zip(BUILD_CALLS, pos, strict=True))
    assert pos == sorted(pos)
    # The drag state is the last thing _build sets.
    last = max(src.find(f'self.{a}') for a in DRAG_STATE)
    assert min(src.find(f'self.{a}') for a in DRAG_STATE) > pos[-1]
    assert src[last:].count('self.') == 1


def test_no_builder_name_is_defined_twice_in_the_class_tree():
    cls = _view_class()
    for name in BUILDERS:
        owners = [k.__name__ for k in cls.__mro__ if name in vars(k)]
        assert owners == ['PipelineWorkspaceView'], (name, owners)


# ── fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def root(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('USERPROFILE', str(home))
    try:
        import tkinter as tk
    except ImportError:
        gui_unavailable("tkinter not available")
    try:
        r = tk.Tk()
    except Exception as e:  # noqa: BLE001 - any Tk start failure means no GUI
        gui_unavailable(f"Tk cannot initialise without a display: {e}")
    r.withdraw()
    yield r
    try:
        r.destroy()
    except Exception:  # noqa: BLE001 - teardown; the root may already be gone
        pass


@pytest.fixture
def engines(monkeypatch):
    """[embed] engines: PHATE not installed, TriMap and PaCMAP working.
    The background check (check_engines) runs at once and reports what
    ``after_check`` holds; run_async delivers its result at the next idle."""
    import openflo.async_task as at
    import openflo.capabilities as cap
    state = {'now': {'phate': NOT_INSTALLED, 'trimap': None, 'pacmap': None},
             'after_check': {}, 'checked': []}
    monkeypatch.setattr(cap, '_IMPORT_RESULT', dict(state['now']))

    def check(mods=None):
        state['checked'].append(list(mods or []))
        cap._IMPORT_RESULT.update(state['after_check'])
        return {}

    def run_async(widget, work, on_done=None, **_kw):
        result = work()
        if on_done is not None:
            widget.after_idle(lambda: on_done(result))
    monkeypatch.setattr(cap, 'check_engines', check)
    monkeypatch.setattr(at, 'run_async', run_async)
    return state


def _editor(tips):
    def _tip(widget, text):
        tips[str(widget)] = text
    return SimpleNamespace(_tip=_tip, _samples={}, _sample_order=[])


def _model():
    from openflo.workspace_run import WorkspaceModel
    m = WorkspaceModel('Pipeline')
    m.run_cfg.update(CFG)
    g = m.new_group('Group A')
    m.add_item({'sample': 'S01', 'trial': 'TrialA', 'path': None,
                'gate_id': None, 'gate_path': '', 'comp_matrix': [[1.0]]}, g)
    m.add_item({'sample': 'S02', 'trial': 'TrialA', 'path': None,
                'gate_id': 'g1', 'gate_path': 'FL1-A+',
                'comp_matrix': [[1.0]]})
    return m


def _view(root, tips=None, chosen=None):
    cls = _view_class()
    v = cls(root, _model(), editor=_editor({} if tips is None else tips),
            on_embed_chosen=(chosen.append if chosen is not None else None))
    v.pack(fill='both', expand=True)
    return v


def _texts(frame):
    out = []
    for w in frame.winfo_children():
        if w.winfo_class() in ('TEntry', 'TCombobox'):
            out.append((w.winfo_class(), None))   # 'text' would read -textvariable
            continue
        try:
            out.append((w.winfo_class(), str(w.cget('text'))))
        except Exception:  # noqa: BLE001 - a widget without a text option
            out.append((w.winfo_class(), None))
    return out


# ── call order, on a recording stand-in ────────────────────────────────────

def test_build_calls_each_part_in_order_and_sets_drag_state_last(
        root, engines):
    """The ordered list of calls and attribute sets, which a snapshot taken
    after the build cannot see: the [embed] boxes are synced and the engine
    check started when the embedding rows exist and the option rows do not
    yet; the drag state is set after the last builder returns."""
    cls = _view_class()
    log: list = []

    class Recording(cls):
        def __setattr__(self, name, value):
            log.append(('set', name))
            super().__setattr__(name, value)

    names = BUILD_CALLS
    for name in names:
        real = getattr(cls, name)

        def rec(self, *a, _real=real, _name=name):
            log.append(('call', _name, a))
            if _name == '_sync_embed_boxes':
                # The embedding rows exist, the option rows do not.
                assert list(self._embed_boxes) == [
                    'umap', 'tsne', 'phate', 'trimap', 'pacmap']
                assert not hasattr(self, 'btn_run')
            out = _real(self, *a)
            log.append(('return', _name))
            return out
        setattr(Recording, name, rec)

    v = Recording(root, _model(), editor=_editor({}))
    calls = [e[1] for e in log if e[0] == 'call']
    assert calls[:len(names)] == list(names)
    # _build_param_rows' result is what _build_option_rows gets.
    from openflo.tool_window import FAST_GRAPH_TIP
    opt = next(e for e in log if e[:2] == ('call', '_build_option_rows'))
    assert opt[2][-1] == FAST_GRAPH_TIP
    # Each builder is passed the toolbar / run bar it fills.
    tb = next(e for e in log if e[:2] == ('call', '_build_toolbar'))
    assert tb[2][0].winfo_parent() == str(v) and tb[2][1] is v._editor
    for n in ('_build_param_rows', '_build_embed_rows', '_build_option_rows',
              '_build_run_actions'):
        e = next(e for e in log if e[:2] == ('call', n))
        assert e[2][0] is v.btn_run.master.master, n
    # The drag state comes after the last builder returns, as one block,
    # and nothing is set after it during the build.
    end = log.index(('return', '_build_tree_bindings'))
    assert tuple(e[1] for e in log[end:] if e[0] == 'set') == DRAG_STATE
    before = [e[1] for e in log[:end] if e[0] == 'set']
    assert not set(DRAG_STATE) & set(before), before


def test_mid_build_sync_runs_before_the_option_rows(root, engines):
    """The sync's effect is in place before the option rows are made: a
    box whose engine is missing is disabled when _build_option_rows runs."""
    cls = _view_class()
    seen = {}

    class Probe(cls):
        def _build_option_rows(self, *a):
            seen['phate'] = self._embed_boxes['phate'][0].instate(['disabled'])
            seen['checked'] = list(engines['checked'])
            return super()._build_option_rows(*a)

    Probe(root, _model(), editor=_editor({}))
    assert seen == {'phate': True,
                    'checked': [['phate', 'trimap', 'pacmap']]}


# ── what each builder makes ────────────────────────────────────────────────

def test_toolbar_menus_and_tips(root, engines):
    """Import / Edit, + Group and Group menus, each with its tip on the
    editor that _menubtn closes over."""
    tips: dict = {}
    v = _view(root, tips)
    bar = v.nametowidget(v.grid_slaves(row=0, column=0)[0])
    assert _texts(bar) == [('TMenubutton', 'Import / Edit'),
                           ('TButton', '＋ Group'),
                           ('TMenubutton', 'Group ▾')]
    want = {
        'Import / Edit': ['Import all samples', 'Import selected sample(s)',
                          None, 'Remove selected', 'Clear workspace'],
        'Group ▾': ['Group selected', 'Group params…', None,
                    'Set comp / beads…', 'Set FMOs…', 'Markers…']}
    for mb in bar.winfo_children():
        if mb.winfo_class() != 'TMenubutton':
            continue
        menu = mb.nametowidget(mb.cget('menu'))
        got = [None if menu.type(i) == 'separator'
               else menu.entrycget(i, 'label')
               for i in range(menu.index('end') + 1)]
        assert got == want[mb.cget('text')]
        assert str(mb) in tips
        assert mb.pack_info()['side'] == 'left'
    first, plus, group = bar.winfo_children()
    assert first.pack_info()['padx'] == (0, 3)
    assert group.pack_info()['padx'] == (3, 0)
    assert tips[str(first)].startswith("Import loaded samples")
    assert tips[str(group)].startswith("Group the selection")
    assert tips[str(plus)].startswith("Create an empty group")


def test_toolbar_menu_entries_call_the_view(root, engines, monkeypatch):
    """Each menu entry runs its own view method (the commands bound when the
    menu was built)."""
    cls = _view_class()
    called = []
    names = ('_import_all', '_import_selected', '_remove_selected', '_clear',
             '_group_selected', '_group_params', '_set_comp_picker',
             '_set_fmo_picker', '_pick_markers', '_new_group')
    for n in names:
        monkeypatch.setattr(cls, n, lambda self, _n=n: called.append(_n))
    v = _view(root)
    bar = v.nametowidget(v.grid_slaves(row=0, column=0)[0])
    for w in bar.winfo_children():
        if w.winfo_class() == 'TButton':
            w.invoke()
            continue
        menu = w.nametowidget(w.cget('menu'))
        for i in range(menu.index('end') + 1):
            if menu.type(i) == 'command':
                menu.invoke(i)
    assert called == ['_import_all', '_import_selected', '_remove_selected',
                      '_clear', '_new_group', '_group_selected',
                      '_group_params', '_set_comp_picker', '_set_fmo_picker',
                      '_pick_markers']


def test_run_bar_rows_in_pack_order(root, engines):
    """Parameter rows, the two embedding rows, the divider, the option rows
    and the actions, packed in that order in the run bar."""
    v = _view(root)
    runbar = v.btn_run.master.master
    assert runbar.grid_info()['row'] == 1
    rows = runbar.pack_slaves()
    got = [_texts(r) if r.winfo_class() == 'TFrame' else r.winfo_class()
           for r in rows]
    assert got[0][0] == ('TLabel', 'Method')
    assert got[0][1][0] == 'TCombobox'
    assert got[1] == [('TLabel', 'Neighbors'), ('TEntry', None),
                      ('TLabel', 'Resolution'), ('TEntry', None)]
    assert got[2] == [('TLabel', 'Max events'), ('TEntry', None),
                      ('TLabel', 'Seed'), ('TEntry', None)]
    assert got[3] == [('TLabel', 'embed:'), ('TCheckbutton', 'UMAP'),
                      ('TCheckbutton', 't-SNE'), ('TCheckbutton', 'PHATE')]
    assert got[4] == [('TCheckbutton', 'TriMap'), ('TCheckbutton', 'PaCMAP')]
    assert got[5] == 'TSeparator'
    assert got[6] == [('TCheckbutton', 'Concatenate'),
                      ('TCheckbutton', 'Reproducible'),
                      ('TCheckbutton', 'Fast graph')]
    assert got[7] == [('TCheckbutton', 'Also write events as CSV')]
    assert got[8] == [('TButton', '▶ Run'), ('TButton', 'Cancel'),
                      ('TButton', 'Results…')]
    assert len(got) == 9


def test_run_vars_hold_the_models_run_cfg(root, engines):
    """Each run-bar variable starts from run_cfg; an [embed] engine that is
    not installed starts unticked."""
    v = _view(root)
    got = {n: getattr(v, n).get() for n in RUN_VARS}
    assert got == {
        '_method_var': 'leiden', '_res_var': '0.5', '_k_var': '12',
        '_max_var': '2500', '_seed_var': 'pilot run 3', '_umap_var': False,
        '_tsne_var': True, '_phate_var': False, '_trimap_var': True,
        '_pacmap_var': True, '_concat_var': True,
        '_reproducible_var': False, '_fast_graph_var': True,
        '_events_csv_var': True}


def test_tk_variables_outlive_the_builders(root, engines):
    """Every variable a run-bar control shows is a live Tcl variable after
    the build (and a garbage collection), is the one the view keeps, and
    still holds run_cfg's value. A variable kept only as a builder's local
    would be unset here: the control would show nothing and a click would
    not reach the view."""
    v = _view(root)
    gc.collect()
    root.update()
    kept = {str(getattr(v, n)): n for n in RUN_VARS}
    shown = {}

    def walk(w):
        for c in w.winfo_children():
            for opt in ('variable', 'textvariable'):
                try:
                    name = str(c.cget(opt))
                except Exception:  # noqa: BLE001 - option not on this widget
                    continue
                if name:
                    shown[str(c)] = name
            walk(c)
    walk(v)
    # The method dropdown shows a display variable that named_choice keeps
    # on the combobox and syncs with _method_var by traces.
    combo = next(c for row in v.btn_run.master.master.winfo_children()
                 for c in row.winfo_children()
                 if c.winfo_class() == 'TCombobox')
    method = str(v._method_var)
    assert sorted(set(shown.values())) == sorted(
        (set(kept) - {method}) | {str(combo.shown_var)})
    for name in list(kept) + [str(combo.shown_var)]:
        assert root.tk.getboolean(root.tk.call('info', 'exists', name)), name
    assert root.tk.call('trace', 'info', 'variable', method)
    assert combo.get() == 'Leiden'
    v._method_var.set('flowsom')
    assert combo.get() == 'FlowSOM'
    assert root.tk.globalgetvar(str(v._concat_var)) in (1, True, '1')
    # A click on a box reaches the variable the view reads.
    box = next(c for c in v._embed_boxes['tsne'][0].master.winfo_children()
               if str(c.cget('text')) == 't-SNE')
    box.invoke()
    assert v._tsne_var.get() is False
    concat = next(c for row in v.btn_run.master.master.winfo_children()
                  for c in row.winfo_children()
                  if c.winfo_class() == 'TCheckbutton'
                  and str(c.cget('text')) == 'Concatenate')
    concat.invoke()
    assert v._concat_var.get() is False


def test_param_rows_tips_and_fast_graph_tip(root, engines):
    from openflo.tool_window import CLUSTER_RESOLUTION_TIP, FAST_GRAPH_TIP
    from openflo.workspace import EVENTS_CSV_TIP
    tips: dict = {}
    v = _view(root, tips)
    runbar = v.btn_run.master.master
    by_text = {}
    for row in runbar.pack_slaves():
        for c in row.winfo_children():
            try:
                by_text.setdefault(str(c.cget('text')), c)
            except Exception:  # noqa: BLE001 - no text option
                pass
    assert tips[str(by_text['Fast graph'])] == FAST_GRAPH_TIP
    assert tips[str(by_text['Also write events as CSV'])] == EVENTS_CSV_TIP
    k_row = runbar.pack_slaves()[1]
    res_entry = k_row.winfo_children()[3]
    assert str(res_entry.cget('textvariable')) == str(v._res_var)
    assert tips[str(res_entry)].startswith(CLUSTER_RESOLUTION_TIP)
    assert str(k_row.winfo_children()[1].cget('textvariable')) == str(v._k_var)


def test_embed_rows_boxes_tips_and_choices(root, engines):
    """Each box reports its own key when clicked and resolves its own tip at
    hover time; the missing engine's box is disabled and unticked."""
    tips: dict = {}
    chosen: list = []
    v = _view(root, tips, chosen)
    assert list(v._embed_boxes) == ['umap', 'tsne', 'phate', 'trimap',
                                    'pacmap']
    rows = [cb.master for cb, _v in v._embed_boxes.values()]
    assert rows[0] is rows[1] is rows[2] and rows[3] is rows[4]
    assert rows[0] is not rows[3]
    for key, (cb, var) in v._embed_boxes.items():
        assert str(cb.cget('variable')) == str(var)
        assert var is getattr(v, f'_{key}_var')
        tip = tips[str(cb)]
        assert callable(tip)
        cb.invoke()
    assert chosen == ['umap', 'tsne', 'trimap', 'pacmap']   # PHATE disabled
    assert v._embed_boxes['phate'][0].instate(['disabled'])
    assert "can't run here" in tips[str(v._embed_boxes['phate'][0])]()
    assert tips[str(v._embed_boxes['tsne'][0])]() == "Compute a t-SNE embedding."
    assert tips[str(v._embed_boxes['trimap'][0])]().startswith(
        "Compute a TriMap embedding")


def test_engine_check_result_updates_the_boxes(root, engines):
    """The background check started mid-build syncs the boxes again when it
    finishes: an engine found broken is disabled and unticked."""
    engines['after_check'] = {'trimap': 'fails to load (ImportError: x)'}
    v = _view(root)
    assert not v._embed_boxes['trimap'][0].instate(['disabled'])
    assert v._trimap_var.get() is True
    root.update()
    assert v._embed_boxes['trimap'][0].instate(['disabled'])
    assert v._trimap_var.get() is False


def test_run_actions(root, engines):
    tips: dict = {}
    v = _view(root, tips)
    assert _texts(v.btn_run.master) == [('TButton', '▶ Run'),
                                        ('TButton', 'Cancel'),
                                        ('TButton', 'Results…')]
    assert v.btn_cancel.instate(['disabled'])
    assert not v.btn_run.instate(['disabled'])
    for b in v.btn_run.master.winfo_children():
        assert str(b) in tips


def test_tree_columns_and_tags(root, engines):
    from openflo.theme import current_palette
    v = _view(root)
    t = v.tree
    assert t.master.grid_info()['row'] == 2
    assert tuple(t['columns']) == ('on', 'comp', 'fmo')
    assert t.heading('#0')['text'] == 'Samples / populations'
    assert t.heading('comp')['text'] == 'Comp / beads'
    assert t.column('#0')['width'] == 140 and t.column('fmo')['width'] == 58
    pal = current_palette()
    assert t.tag_configure('off')['foreground'] == pal['muted']
    assert t.tag_configure('warn')['foreground'] == pal['error']
    assert t.tag_configure('drop_target')['background'] == pal['highlight']
    sb = [c for c in t.master.winfo_children() if c.winfo_class() == 'TScrollbar']
    assert len(sb) == 1 and sb[0].grid_info()['column'] == 1
    # The tree was rendered over the model after the build.
    assert len(t.get_children('')) == 2


def test_tree_bindings_reach_the_view(root, engines, monkeypatch):
    cls = _view_class()
    seen = []
    for n in ('_on_press', '_on_motion', '_on_release', '_remove_selected'):
        monkeypatch.setattr(cls, n, lambda self, *a, _n=n: seen.append(_n))
    v = _view(root)
    root.deiconify()
    root.update()
    for seq in ('<Button-1>', '<B1-Motion>', '<ButtonRelease-1>'):
        assert v.tree.bind(seq)
    v.tree.event_generate('<Button-1>', x=5, y=5)
    v.tree.event_generate('<B1-Motion>', x=6, y=6)
    v.tree.event_generate('<ButtonRelease-1>', x=6, y=6)
    v.tree.focus_force()
    root.update()
    v.tree.event_generate('<Delete>')
    assert seen == ['_on_press', '_on_motion', '_on_release',
                    '_remove_selected']


def test_drag_state_after_build_and_a_drag(root, engines):
    """The drag state starts cleared, and a press / move / release from an
    item row onto the group row moves the item into the group."""
    v = _view(root)
    assert (v._press_row, v._press_col, v._press_x, v._press_y,
            v._drag_active, v._drag_threshold, v._drop_hi) == (
        None, None, 0, 0, False, 5, None)
    root.deiconify()
    root.geometry('700x800+0+0')
    root.update()
    rows = list(v.tree.get_children(''))
    gid = next(r for r in rows if r.startswith('G'))
    mid = next(r for r in rows if r.startswith('M'))
    b1, b2 = v.tree.bbox(mid, '#0'), v.tree.bbox(gid, '#0')
    assert b1 and b2
    v.tree.event_generate('<ButtonPress-1>', x=b1[0] + 20, y=b1[1] + 5)
    assert v._press_row == mid
    v.tree.event_generate('<B1-Motion>', x=b2[0] + 20, y=b2[1] + 5)
    assert v._drag_active is True and v._drop_hi == gid
    v.tree.event_generate('<ButtonRelease-1>', x=b2[0] + 20, y=b2[1] + 5)
    assert v.model.item_group(mid) == gid
    assert v._drag_active is False
