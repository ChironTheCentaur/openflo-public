"""Every control in the editor's inline dialogs carries hover help.

``tests/test_dialog_tooltips.py`` checks the ``ui_*`` dialog classes. Many
other windows are built inline, as a ``tk.Toplevel(self)`` inside an editor
method or the Pipeline Workspace, and nothing checked them: the copy-gates,
boolean, clear-all, close and transform dialogs, the cluster and cell-cycle
dialogs and the rest offered no hover help at all. These are the windows an
inexperienced user meets least often, so they are the ones that need it most.

Each dialog is opened here through its real method, on generated data
(channels FL1-A, FL2-A..., samples s1, s2), and every control a user can act
on must have an ``<Enter>`` binding: the tooltip's hook. The rules are the
existing test's (INTERACTIVE, SELF_EVIDENT), imported, so the two cannot
disagree about what needs a tip.

Deterministic by construction: only the binding is checked. Nothing moves the
pointer or waits for a tooltip to appear, and nothing waits for a person:
``wait_window`` and ``grab_set`` return at once, and message boxes and file
pickers are patched, so no test can hang a Windows CI job.
"""
from __future__ import annotations

import tkinter as tk

import numpy as np
import pandas as pd
import pytest

from tests.test_dialog_tooltips import (
    INTERACTIVE,
    SELF_EVIDENT,
    _editor_or_skip,
    _label,
    _walk,
)

COLS = ['FSC-A', 'FSC-H', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A']


# ── helpers ─────────────────────────────────────────────────────────────────

@pytest.fixture
def ed(monkeypatch):
    """A real editor with nothing that waits for a person: modal waits and
    grabs return at once, and any message box answers without showing."""
    from tkinter import filedialog, messagebox
    monkeypatch.setattr(tk.Misc, 'wait_window', lambda *a, **k: None)
    monkeypatch.setattr(tk.Misc, 'grab_set', lambda *a, **k: None)
    monkeypatch.setattr(tk.Misc, 'grab_set_global', lambda *a, **k: None)
    for name in ('showinfo', 'showwarning', 'showerror'):
        monkeypatch.setattr(messagebox, name, lambda *a, **k: 'ok')
    for name in ('askyesno', 'askokcancel', 'askyesnocancel'):
        monkeypatch.setattr(messagebox, name, lambda *a, **k: False)
    for name in ('askopenfilename', 'askopenfilenames', 'asksaveasfilename',
                 'askdirectory'):
        monkeypatch.setattr(filedialog, name, lambda *a, **k: '')
    # Background engine checks (Help > Environment) must not start threads.
    from openflo import async_task
    monkeypatch.setattr(async_task, 'run_async', lambda *a, **k: None)
    root, editor = _editor_or_skip()
    try:
        yield editor
    finally:
        try:
            root.destroy()
        except Exception:                                # noqa: BLE001 - teardown; the root may already be destroyed, and a teardown error must not hide the test's own result
            pass


def _frame(seed, n=600):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.uniform(1, 1000, (n, len(COLS))), columns=COLS)
    df['FSC-H'] = df['FSC-A'] * rng.uniform(0.9, 1.1, n)
    df['leiden'] = np.arange(n) % 3
    df['cluster'] = (np.arange(n) // 7) % 3
    return df


def _load(ed, name, seed):
    """Load a generated sample ``name`` and give it two threshold gates."""
    from openflo.pipeline import FlowSample
    s = FlowSample.from_dataframe(_frame(seed), name=name, path=f'{name}.fcs')
    ed._samples[name] = s
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    if 'T' not in ed._trial_order:
        ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = True
    ed._channels = list(s.data.columns)
    ed._channel_labels = {c: c for c in s.data.columns}
    ed._populate_channel_combos()
    ed._set_active_sample(name)
    return [ed._add_gate({'kind': 'threshold', 'name': f'Pos{i}',
                          'channel': ch, 'op': '>', 'value': 500.0,
                          'parent_id': None}, audit=False)
            for i, ch in enumerate(('FL1-A', 'FL2-A'))]


def _loaded(ed):
    """Two samples, s1 active, each with two gates. Returns s1's gate ids."""
    _load(ed, 's2', 2)
    ids = _load(ed, 's1', 1)
    ed._refresh_gate_list()
    return ids


def _toplevels(root):
    return [w for w in _walk(root) if isinstance(w, tk.Toplevel)]


def _opened(ed, open_it):
    """Run ``open_it`` and return the one window it opened."""
    root = ed.master
    before = {str(w) for w in _toplevels(root)}
    open_it()
    root.update_idletasks()
    new = [w for w in _toplevels(root) if str(w) not in before]
    assert len(new) == 1, (f'expected one new window, got '
                           f'{[w.title() for w in new]}')
    return new[0]


_LABELLED = {'TButton', 'TMenubutton', 'TCheckbutton', 'TRadiobutton',
             'Button', 'Checkbutton', 'Menubutton'}


def _describe(w):
    """A control as a person would find it: its own text, or for a field
    (whose ``text`` is only its variable's name) the label before it."""
    cls = w.winfo_class()
    if cls in _LABELLED:
        return f'{cls} {_label(w)!r}'
    try:
        sibs = w.master.winfo_children()
        before = sibs[:sibs.index(w)]
    except Exception:                                    # noqa: BLE001 - a widget with no master or no siblings has no label before it
        before = []
    for s in reversed(before):
        if s.winfo_class() in ('TLabel', 'Label') and _label(s).strip():
            return f'{cls} after {_label(s).strip()!r}'
    return f'{cls} {w}'


def _missing(win):
    """Every actionable control under ``win`` without a hover tip."""
    out = []
    for w in _walk(win):
        if w.winfo_class() not in INTERACTIVE:
            continue
        if w.winfo_class() == 'TButton' and _label(w) in SELF_EVIDENT:
            continue
        try:
            bound = w.bind() or ()
        except Exception:                                # noqa: BLE001 - skip widgets whose bindings cannot be read
            continue
        if '<Enter>' not in bound:
            out.append(_describe(w))
    return out


def _check(name, win):
    missing = _missing(win)
    assert not missing, (
        f'{name}: {len(missing)} control(s) have no hover tooltip. Attach one '
        f'with self._tip / openflo.ui_tips.tip: {missing[:8]}')


# ── the dialogs ─────────────────────────────────────────────────────────────

def _autoclean(ed):
    from openflo.autoclean import default_autoclean_methods
    from openflo.gaussian_cleanup import default_method
    from openflo.ui_mass_cytometry import beads_method
    methods = default_autoclean_methods()
    methods += [default_method(['FL1-A']), beads_method(['FL3-A'])]
    gid = ed._add_gate({'kind': 'autoclean', 'name': 'Clean',
                        'methods': methods, 'parent_id': None}, audit=False)
    return lambda: ed._edit_autoclean_params('s1', gid)


def _tabs(ed, within):
    from openflo.tree_ids import gate_iid
    ids = _loaded(ed)
    ts = ed._tabs()
    ts.new_tab('A', members={'s1': None, 's2': None})
    ed._rebuild_tab_strip()
    ed.gate_tv.selection_set(gate_iid('s1', ids[0]))
    return (ed._ask_apply_rows_within_tab if within
            else ed._ask_apply_rows_to_tabs)


def _figure(ed):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(3, 2))
    fig.add_subplot(111).plot([0, 1], [0, 1])
    return lambda: ed._show_figure_preview(fig, ['s1'])


def _update(ed):
    from openflo import editor_update
    return lambda: editor_update.ask_update_or_cancel(ed, 'Update', 'text')


def _choice(ed):
    return lambda: ed._ask_choice(
        'Session changed on disk', 'text',
        [('Save as a copy', 'copy'), ('Overwrite', 'overwrite'),
         ('Cancel', 'cancel')],
        tips=['Copy tip.', 'Overwrite tip.', None])


def _annotate(ed, col):
    ed._import_populations('leiden')
    ed._import_populations('cluster')
    if col == 'cluster':
        return ed._annotate_clusters
    return lambda: ed._annotate_populations(col)


DIALOGS = {
    'template apply': lambda ed: ed._ask_template_apply,
    'figure preview': _figure,
    'close choice': lambda ed: ed._ask_close_choice,
    'gate coordinates': lambda ed: (
        lambda: ed._edit_gate_coordinates('s1', ed._sample_gate_order['s1'][0])),
    'auto-clean parameters': _autoclean,
    'apply to tabs': lambda ed: _tabs(ed, within=False),
    'apply within a tab': lambda ed: _tabs(ed, within=True),
    'update offer': _update,
    'channel transforms': lambda ed: ed._open_transform_editor,
    'copy gates': lambda ed: ed._open_copy_gates_dialog,
    'boolean gate': lambda ed: ed._open_boolean_dialog,
    'clear all gates': lambda ed: (lambda: ed._ask_clear_all(4, 2, 1)),
    'environment': lambda ed: ed._show_environment,
    'about': lambda ed: ed._show_about,
    'keyboard shortcuts': lambda ed: ed._show_shortcuts,
    'diagnostics': lambda ed: (
        lambda: ed._show_diagnostics_result('[FAIL] generated', 1)),
    'session choice': _choice,
    'combination gates': lambda ed: ed._open_combination_dialog,
    'annotate populations': lambda ed: _annotate(ed, 'leiden'),
    'annotate clusters': lambda ed: _annotate(ed, 'cluster'),
    'cell cycle': lambda ed: ed._open_cell_cycle_dialog,
    'cluster': lambda ed: ed._open_cluster_dialog,
}

# Dialogs that need the samples' gates in place before they are opened.
_NEEDS_SAMPLES = set(DIALOGS) - {
    'close choice', 'update offer', 'session choice', 'environment', 'about',
    'keyboard shortcuts', 'diagnostics', 'figure preview', 'apply to tabs',
    'apply within a tab'}


@pytest.mark.parametrize('name', sorted(DIALOGS))
def test_inline_dialog_controls_have_hover_help(ed, name):
    if name in _NEEDS_SAMPLES:
        _loaded(ed)
    win = _opened(ed, DIALOGS[name](ed))
    _check(name, win)


# ── the Pipeline Workspace ──────────────────────────────────────────────────

def _workspace(ed):
    _loaded(ed)
    ed._open_pipeline_workspace()
    ed.update_idletasks()
    panel = getattr(ed, '_workspace_panel', None)
    assert panel is not None, (
        'the Pipeline Workspace panel was not built: _open_pipeline_workspace '
        'left _workspace_panel unset, so the checks below cannot run')
    return panel, panel._active_view()


def test_workspace_panel_controls_have_hover_help(ed):
    panel, _view = _workspace(ed)
    # "On top" is packed only while the panel floats; check it all the same.
    _check('workspace panel', panel)
    assert '<Enter>' in panel._ontop_cb.bind()


def test_workspace_group_params_has_hover_help(ed):
    _panel, view = _workspace(ed)
    view._new_group()
    gid = next(iter(view.model.groups))
    view.tree.selection_set(gid)
    _check('group params', _opened(ed, view._group_params))


def test_workspace_markers_has_hover_help(ed):
    _panel, view = _workspace(ed)
    _check('markers', _opened(ed, view._pick_markers))


def test_workspace_results_viewer_has_hover_help(ed, tmp_path):
    from openflo.workspace import ResultsViewer
    (tmp_path / 'gen_events.csv').write_text('FL1-A,cluster\n1,0\n')
    (tmp_path / 'gen_clusters.csv').write_text('cluster,n\n0,1\n')
    win = ResultsViewer(ed, str(tmp_path), editor=ed)
    win.update_idletasks()
    _check('results viewer', win)
