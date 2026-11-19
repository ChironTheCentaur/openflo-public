"""The editor window sets its state up one step at a time.

``ViewGateEditorWindow.__init__`` used to set about a hundred attributes in
one method of about 310 lines. It now calls one ``_init_<step>`` method per
kind of state (window size, samples, gate stores, loading, plot, backgating
overlay, display variables, undo), in the order the attributes were always
set, before ``_build``. The callback error handler is installed right after
the window size, before any state step.

These tests pin the step list and its order, what each step sets, and that
``tests/_editor_stub.py`` (which test stubs are built from) follows the steps,
so the stub keeps every attribute ``__init__`` sets before ``_build``.
"""
from __future__ import annotations

import ast
import inspect
import json
import queue
import threading
from types import SimpleNamespace

import pytest

from tests._editor_stub import _collect, _init_assignments
from tests.conftest import gui_unavailable

INIT_STEPS = (
    '_init_window_size', '_init_sample_state', '_init_gate_state',
    '_init_load_state', '_init_plot_state', '_init_overlay_state',
    '_init_display_vars', '_init_undo_state',
)

# What each step sets, in order. The window-size step sets no attribute.
STEP_STATE = {
    '_init_window_size': (),
    '_init_sample_state': (
        '_samples', '_sample_order', '_sample_colors', '_sample_trial',
        '_trial_order', '_sample_is_comp', '_sample_set_aside',
        '_sample_notes', '_channels', '_channel_labels',
        '_explicit_channel_labels', '_channel_transform'),
    '_init_gate_state': (
        '_sample_gates', '_sample_gate_seq', '_sample_gate_order'),
    '_init_load_state': (
        '_pending_sample_gates', '_pending_wsp_trial', '_pending_wsp_axes',
        '_restore_lost_operands', '_restore_built_on', '_pending_sample_meta',
        '_pending_sample_prep', '_autoclean_cache', '_autoclean_count_cache',
        '_autoclean_method_cache', '_loading', '_path_to_name',
        '_name_to_path', '_load_queue', '_load_seq', '_load_pool',
        '_load_pool_started', '_load_gen', '_load_epoch', '_load_job_epochs',
        '_load_stop', '_load_total', '_load_done'),
    '_init_plot_state': (
        '_channel_scale', '_channel_range', '_default_channel_scale',
        '_active_sample', '_gates', '_gate_id_seq', '_gate_id_order',
        '_vlines', '_hlines', '_shape_artists', '_drag_state',
        '_quad_set_seq', '_drag_translate_ctx', '_drag_poly_vertices',
        '_replot_after_id', '_cbar', '_audit_log', '_audit_window',
        '_last_unmix_qc', '_last_unmix_spectra', '_last_unmix_detectors',
        '_selector', '_cluster_labels'),
    '_init_overlay_state': (
        '_backgate', '_gate_density_full', '_backgate_hidden',
        '_backgate_legend_pick', '_backgate_legend_anchor',
        '_backgate_legend_collapsed', '_backgate_legend_bbox',
        '_backgate_legend_header', '_backgate_legend_artists',
        '_backgate_legend_rows', '_legend_drag'),
    '_init_display_vars': (
        '_theme_var', '_chrome_var', '_plot_bg_var', '_gate_labels_var',
        '_fmo_hist_var', '_density_cmap_var', '_group_palette_var',
        '_spawn_corner', '_tooltips_enabled', '_legend_placement',
        '_legend_pos_cache', '_legend_pending'),
    '_init_undo_state': (
        '_undo_stack', '_redo_stack', '_undo_pending', '_suspend_undo',
        '_UNDO_MAX'),
}

# Set in __init__ itself, before the state steps.
INIT_OWN_STATE = ('_defer_plot_canvas', '_error_count', 'fcs_dir',
                  'labels_str', 'on_save', '_primary', '_app')

# Steps that only set plain attributes (no Tk call): runnable on a stub.
PLAIN_STEPS = ('_init_sample_state', '_init_gate_state', '_init_load_state',
               '_init_plot_state', '_init_overlay_state', '_init_undo_state')


def _editor_class():
    try:
        from openflo.gui import ViewGateEditorWindow
    except ImportError as exc:
        gui_unavailable(f"the GUI cannot be imported: {exc}")
    return ViewGateEditorWindow


def _self_calls(func):
    """Names of the ``self.<name>(...)`` statements of `func`, in order."""
    tree = ast.parse(inspect.cleandoc('\n' + inspect.getsource(func)))
    body = tree.body[0].body
    return [st.value.func.attr for st in body
            if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call)
            and isinstance(st.value.func, ast.Attribute)
            and isinstance(st.value.func.value, ast.Name)
            and st.value.func.value.id == 'self']


def test_init_is_a_short_list_of_state_steps_in_order():
    cls = _editor_class()
    missing = [n for n in INIT_STEPS if not callable(cls.__dict__.get(n))]
    assert missing == []
    assert len(inspect.getsource(cls.__init__).splitlines()) < 80
    calls = _self_calls(cls.__init__)
    steps = [c for c in calls if c.startswith('_init_')]
    assert steps == list(INIT_STEPS)
    # Every step runs before _build, which builds on their state.
    assert calls.index('_build') > calls.index(INIT_STEPS[-1])


def test_error_handler_is_installed_before_any_state_step():
    calls = _self_calls(_editor_class().__init__)
    handler = calls.index('_install_callback_error_handler')
    assert calls.index('_init_window_size') < handler
    assert handler < min(calls.index(n) for n in INIT_STEPS[1:])


def test_no_step_name_is_defined_twice_in_the_class_tree():
    """A step must not shadow (or be shadowed by) a mixin method."""
    cls = _editor_class()
    for name in INIT_STEPS:
        owners = [k.__name__ for k in cls.__mro__ if name in vars(k)]
        assert owners == ['ViewGateEditorWindow'], (name, owners)


@pytest.mark.parametrize('step', PLAIN_STEPS)
def test_each_plain_step_sets_its_state_in_order(step):
    stub = SimpleNamespace()
    getattr(_editor_class(), step)(stub)
    assert tuple(vars(stub)) == STEP_STATE[step]


def test_plain_steps_start_empty():
    cls = _editor_class()
    stub = SimpleNamespace()
    for step in PLAIN_STEPS:
        getattr(cls, step)(stub)
    assert stub._samples == {} and stub._sample_gates == {}
    assert isinstance(stub._load_queue, queue.PriorityQueue)
    assert isinstance(stub._load_stop, threading.Event)
    assert not stub._load_stop.is_set()
    assert (stub._load_seq, stub._load_total, stub._load_done) == (0, 0, 0)
    assert stub._load_pool_started is False and stub._loading == set()
    assert stub._default_channel_scale == 'logicle'
    assert stub._active_sample is None and stub._gates == {}
    assert type(stub._audit_log).__name__ == 'AuditLog'
    assert stub._backgate_legend_anchor == (0.020, 0.965)
    assert stub._backgate_legend_collapsed is False
    assert stub._undo_stack == [] and stub._redo_stack == []
    assert stub._undo_stack is not stub._redo_stack
    assert stub._undo_pending is False and stub._suspend_undo is False
    assert stub._UNDO_MAX == 100


def test_stub_collects_init_and_every_step_in_order():
    names = [name for name, _, _ in _init_assignments()]
    expected = list(INIT_OWN_STATE)
    for step in INIT_STEPS:
        expected += STEP_STATE[step]
    assert names == expected


def _scan(src):
    """What the stub's scan collects from the ``__init__`` of class W in
    `src`, as names."""
    cls = ast.parse(src).body[0]
    methods = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}
    out = []
    _collect(methods['__init__'].body, methods, {}, out)
    return [name for name, _, _ in out]


def test_stub_scan_follows_steps_in_call_order_and_stops_at_build():
    src = (
        "class W:\n"
        "    def __init__(self):\n"
        "        self.a = 1\n"
        "        self._init_two()\n"
        "        self.b: int = 2\n"
        "        self._init_one()\n"
        "        other.c = 3\n"
        "        other._init_nested()\n"
        "        self.helper()\n"
        "        self._build()\n"
        "        self.after_build = 4\n"
        "    def _init_one(self):\n"
        "        self.one = 1\n"
        "        self._init_nested()\n"
        "    def _init_nested(self):\n"
        "        self.nested = 1\n"
        "    def _init_two(self):\n"
        "        x = self.two = y = 2\n"
        "    def helper(self):\n"
        "        self.from_helper = 1\n"
    )
    assert _scan(src) == ['a', 'two', 'b', 'one', 'nested']


def test_stub_scan_stops_at_build_inside_a_step():
    src = (
        "class W:\n"
        "    def __init__(self):\n"
        "        self._init_one()\n"
        "        self.after = 1\n"
        "    def _init_one(self):\n"
        "        self.one = 1\n"
        "        self._build()\n"
        "        self.late = 1\n"
    )
    assert _scan(src) == ['one']


@pytest.fixture
def tk_root(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    (home / '.openflo').mkdir(parents=True)
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('USERPROFILE', str(home))
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        gui_unavailable(f"Tk cannot start: {exc}")
    root.withdraw()
    yield root, home
    root.destroy()


class _Stop(Exception):
    pass


def test_state_set_before_build_is_what_the_stub_lists(tk_root, monkeypatch):
    """On a real window: the attributes set before ``_build`` are exactly
    the stub's names, in the same order."""
    import tkinter as tk
    root, _ = tk_root
    cls = _editor_class()
    seen = []

    def stop(self):
        seen.extend(vars(self))
        raise _Stop

    monkeypatch.setattr(cls, '_build', stop)
    with pytest.raises(_Stop):
        cls(root, fcs_dir=None, labels_str='', on_save=None, primary=False)
    plain = set(vars(tk.Toplevel(root))) | {'_tclCommands'}
    assert [n for n in seen if n not in plain] == [
        name for name, _, _ in _init_assignments()]


@pytest.mark.parametrize('prefs', [
    {},
    {'theme': 'midnight', 'gate_labels': False, 'fmo_on_histograms': False,
     'density_cmap': 'viridis', 'group_palette': 'classic',
     'spawn_corner': 'top-left', 'tooltips': False,
     'legend_placement': 'best'},
])
def test_display_vars_follow_the_prefs(tk_root, prefs):
    root, home = tk_root
    (home / '.openflo' / 'prefs.json').write_text(json.dumps(prefs))
    ed = _editor_class()(root, fcs_dir=None, labels_str='', on_save=None,
                         primary=False, defer_plot_canvas=True)
    ed.withdraw()
    flipped = bool(prefs)
    assert ed._theme_var.get() == ('midnight' if flipped else 'light')
    assert ed._chrome_var.get() == ('dark' if flipped else 'light')
    assert ed._plot_bg_var.get() == ('dark' if flipped else 'light')
    assert ed._gate_labels_var.get() is not flipped
    assert ed._fmo_hist_var.get() is not flipped
    assert ed._density_cmap_var.get() == ('viridis' if flipped else 'turbo')
    assert ed._group_palette_var.get() == (
        'classic' if flipped else 'okabe_ito')
    assert ed._spawn_corner.get() == ('top-left' if flipped else 'off')
    assert ed._tooltips_enabled.get() is not flipped
    assert ed._legend_placement.get() == ('best' if flipped else 'cached')
    assert ed._legend_pos_cache == {} and ed._legend_pending is None
    assert ed.title() == "Untitled session — OpenFlo"
    # The handler is on the root, and the close button saves the session.
    assert root.report_callback_exception == ed._report_callback_exception
    assert ed.protocol('WM_DELETE_WINDOW')
