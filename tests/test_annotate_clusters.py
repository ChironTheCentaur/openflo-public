"""editor_analysis._annotate_clusters — the "Annotate clusters…" dialog that
names the active sample's clusters (2% covered: only its no-sample guard ran).

Driven through its own entries and Apply button: a typed name must land in
_cluster_labels AND on the matching cluster gate; clearing an entry must
restore the shared default name ('Cluster N' / 'Unclustered (noise)') and
drop the stored label; Cancel must change nothing.
"""
import os
import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd

from tests.conftest import gui_unavailable


def _editor_or_skip():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        gui_unavailable(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    import importlib
    gui = importlib.import_module('openflo.gui')
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    return root, ed


def _clustered(ed):
    from openflo.pipeline import FlowSample
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(60, 2)), columns=['CD1', 'CD2'])
    df['cluster'] = np.repeat([0, 1, 2, -1], 15)
    s = FlowSample.from_dataframe(df, name='s1', path=r'C:\exp\s1.fcs')
    ed._samples['s1'] = s
    ed._sample_order.append('s1')
    ed._sample_colors['s1'] = '#1f77b4'
    ed._sample_trial['s1'] = 'T'
    ed._trial_order.append('T')
    ed._sample_plot_enabled['s1'] = True
    ed._channels = ['CD1', 'CD2', 'cluster']
    ed._channel_labels = {c: c for c in ed._channels}
    ed._populate_channel_combos()
    ed._set_active_sample('s1')
    ed._import_clusters()
    return s


def _open(ed):
    ed._annotate_clusters()
    dlg = next(w for w in ed.winfo_children() if isinstance(w, tk.Toplevel)
               and w.title().startswith('Annotate clusters'))
    rows, buttons, stack = {}, {}, [dlg]
    while stack:
        x = stack.pop()
        if isinstance(x, ttk.Entry):
            lbl = next(c for c in x.master.winfo_children()
                       if isinstance(c, ttk.Label))
            rows[lbl.cget('text')] = x
        elif isinstance(x, ttk.Button):
            buttons[x.cget('text')] = x
        stack.extend(x.winfo_children())
    return dlg, rows, buttons


def _gate_names(ed):
    return {g['cluster_id']: g['name'] for g in ed._sample_gates['s1'].values()
            if g.get('kind') == 'cluster'}


def test_apply_names_clusters_and_their_gates():
    root, ed = _editor_or_skip()
    try:
        _clustered(ed)
        assert _gate_names(ed) == {0: 'Cluster 0', 1: 'Cluster 1',
                                   2: 'Cluster 2', -1: 'Unclustered (noise)'}
        dlg, rows, buttons = _open(ed)
        assert set(rows) == {'Cluster 0', 'Cluster 1', 'Cluster 2',
                             'Unclustered (noise)'}
        rows['Cluster 1'].delete(0, 'end')
        rows['Cluster 1'].insert(0, 'CD4 T cells')
        buttons['Apply'].invoke()
        assert not dlg.winfo_exists()
        assert ed._cluster_labels['s1'][1] == 'CD4 T cells'
        assert _gate_names(ed)[1] == 'CD4 T cells'
        assert _gate_names(ed)[0] == 'Cluster 0'
        assert ed._cluster_display_name('s1', 1) == 'CD4 T cells'
        assert "Updated cluster names for 's1'" in ed.status_var.get()

        # Re-open: the entry is pre-filled with the stored name; clearing it
        # restores the default name and forgets the label.
        dlg, rows, buttons = _open(ed)
        assert rows['Cluster 1'].get() == 'CD4 T cells'
        rows['Cluster 1'].delete(0, 'end')
        buttons['Apply'].invoke()
        assert 1 not in ed._cluster_labels['s1']
        assert _gate_names(ed)[1] == 'Cluster 1'
    finally:
        root.destroy()


def test_cancel_changes_nothing():
    root, ed = _editor_or_skip()
    try:
        _clustered(ed)
        before = _gate_names(ed)
        dlg, rows, buttons = _open(ed)
        rows['Cluster 0'].delete(0, 'end')
        rows['Cluster 0'].insert(0, 'B cells')
        buttons['Cancel'].invoke()
        assert not dlg.winfo_exists()
        assert _gate_names(ed) == before
        assert ed._cluster_labels.get('s1', {}).get(0) is None
    finally:
        root.destroy()


def test_unclustered_sample_is_refused():
    from openflo.pipeline import FlowSample
    root, ed = _editor_or_skip()
    try:
        ed._samples['s1'] = FlowSample.from_dataframe(
            pd.DataFrame({'CD1': [1.0, 2.0]}), name='s1')
        ed._sample_order.append('s1')
        ed._active_sample = 's1'
        ed._annotate_clusters()
        assert not any(w.title().startswith('Annotate clusters')
                       for w in ed.winfo_children()
                       if isinstance(w, tk.Toplevel))
        assert "has no 'cluster' column" in ed.status_var.get()
    finally:
        root.destroy()
