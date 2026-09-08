"""The gate tree must not be rebuilt on every drag tick.

``_refresh_gate_list`` rebuilds rows for EVERY loaded sample, so its cost
scales with the workspace, not with the gate being dragged: ~8 ms at 20
samples x 30 gates, which at drag rate is roughly half of wall-clock. Both the
line-gate drag and the histogram slider now defer it to end-of-drag, matching
what the other drag kinds already did.

These tests count calls rather than timing anything, so they are stable in CI.
"""
import os
from types import SimpleNamespace

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

COLS = ('FSC-A', 'SSC-A', 'CD3', 'CD4')


def _editor_or_skip():
    try:
        import tkinter as tk
    except ImportError:
        pytest.skip('tkinter not available')
    try:
        root = tk.Tk()
        root.withdraw()
    except Exception as e:                       # noqa: BLE001
        pytest.skip(str(e))
    import importlib
    gui = importlib.import_module('openflo.gui')
    gui.messagebox.askyesno = lambda *a, **k: True
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    df = pd.DataFrame({c: np.linspace(0, 1, 200) for c in COLS})
    ed._samples['s1'] = SimpleNamespace(
        name='s1', path=r'C:\e\s1.fcs', data=df,
        fluor_channels=[c for c in COLS if c not in ('FSC-A', 'SSC-A')],
        channel_labels={c: c for c in COLS})
    ed._sample_order.append('s1')
    ed._sample_colors['s1'] = '#1f77b4'
    ed._sample_trial['s1'] = 'T'
    ed._trial_order.append('T')
    ed._sample_plot_enabled['s1'] = True
    ed._sample_gates['s1'] = {}
    ed._channels = list(COLS)
    ed._channel_labels = {c: c for c in COLS}
    ed._populate_channel_combos()
    return root, ed


def _count_refreshes(ed, monkeypatch):
    calls = []
    monkeypatch.setattr(type(ed), '_refresh_gate_list',
                        lambda self, *a, **k: calls.append(1), raising=True)
    return calls


def test_slider_drag_does_not_rebuild_the_tree(monkeypatch):
    """Continuous ttk.Scale callbacks must not each rebuild the tree."""
    root, ed = _editor_or_skip()
    try:
        ed._slider_channel = 'CD3'
        ed.slider_kind_var.set('threshold')
        calls = _count_refreshes(ed, monkeypatch)
        for v in np.linspace(0.1, 0.9, 25):      # simulate a drag
            ed.slider_lo.set(float(v))
            ed._on_slider_lo()
        assert not calls, (f'tree rebuilt {len(calls)}x during a 25-tick drag '
                           '— it must be deferred to <ButtonRelease-1>')
        ed._on_slider_release()                  # end of drag
        assert len(calls) == 1, 'release must refresh the tree exactly once'
    finally:
        root.destroy()


def test_slider_still_commits_the_gate_every_tick(monkeypatch):
    """Deferring the TREE refresh must not defer the gate itself — the plot
    has to keep tracking the slider."""
    root, ed = _editor_or_skip()
    try:
        ed._slider_channel = 'CD3'
        ed.slider_kind_var.set('threshold')
        monkeypatch.setattr(type(ed), '_refresh_gate_list',
                            lambda self, *a, **k: None)
        ed.slider_lo.set(0.25)
        ed._on_slider_lo()
        gid = ed._slider_gate_id
        assert gid, 'no gate created by the slider'
        first = ed._gates[gid]['value']
        ed.slider_lo.set(0.75)
        ed._on_slider_lo()
        assert ed._gates[gid]['value'] != first, \
            'gate value did not follow the slider'
    finally:
        root.destroy()


def test_line_drag_motion_does_not_rebuild_the_tree():
    """The v/h line-drag motion handler was the lone drag kind that refreshed
    the tree per mousemove; _on_release covers it."""
    import inspect

    from openflo import editor_gating
    src = inspect.getsource(editor_gating.GatingMixin._on_motion)
    # The 1-D line branch ends at its `return`; no refresh may appear before it.
    line_branch = src.split("if kind in ('v', 'h')", 1)
    assert len(line_branch) == 2, 'line-drag branch not found — test is stale'
    body = line_branch[1].split('return', 1)[0]
    assert '_refresh_gate_list' not in body, (
        'line-gate drag rebuilds the whole gate tree on every mousemove; '
        'let _on_release refresh once at end-of-drag instead')
