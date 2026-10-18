"""Populations -> Annotate <column>... rename dialogs write the typed names.

editor_analysis._annotate_populations (generic label column) and
_annotate_clusters (PhenoGraph 'cluster') build a Toplevel of Entry rows and an
Apply button whose do_apply never ran under the suite (0 % covered). Here the
entries are typed and Apply is clicked; the gate names, the cluster-label
store and the blank-entry fallback are checked.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(__file__))

from test_gui_smoke import _editor_or_skip  # noqa: E402


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def _setup():
    from openflo.pipeline import FlowSample
    root, ed, _gui = _editor_or_skip()
    rng = np.random.default_rng(0)
    df = pd.DataFrame({'CD3-A': rng.normal(0, 1, 90),
                       'leiden': np.repeat([0, 1, 2], 30),
                       'cluster': np.repeat([2, 0, 1], 30)})
    s = FlowSample.from_dataframe(df, name='s1')
    ed._samples['s1'] = s
    ed._sample_order.append('s1')
    ed._sample_colors['s1'] = '#1f77b4'
    ed._sample_trial['s1'] = 'T'
    ed._trial_order.append('T')
    ed._sample_plot_enabled['s1'] = True
    ed._sample_gates['s1'] = {}
    ed._sample_gate_order['s1'] = []
    ed._sample_gate_seq['s1'] = 0
    ed._channels = list(df.columns)
    ed._channel_labels = {c: c for c in df.columns}
    ed._active_sample = 's1'
    ed._import_populations('leiden')
    ed._import_populations('cluster')
    return root, ed


def _fill_and_apply(ed, title_prefix, texts):
    import tkinter as tk
    from tkinter import ttk
    dlg = next(w for w in ed.winfo_children()
               if isinstance(w, tk.Toplevel) and w.title().startswith(title_prefix))
    entries = [w for w in _walk(dlg) if isinstance(w, ttk.Entry)]
    assert len(entries) == len(texts)
    for e, t in zip(entries, texts, strict=True):
        e.delete(0, 'end')
        e.insert(0, t)
    next(w for w in _walk(dlg) if isinstance(w, ttk.Button)
         and str(w.cget('text')) == 'Apply').invoke()


def test_rename_label_column_populations():
    root, ed = _setup()
    try:
        ed._annotate_populations('leiden')
        _fill_and_apply(ed, 'Annotate', ['T cells', '  ', 'B cells'])
        names = {int(g['value']): g['name']
                 for g in ed._sample_gates['s1'].values()
                 if g.get('kind') == 'category' and g.get('channel') == 'leiden'}
        # A blank entry falls back to '<column> <value>', never an empty name.
        assert names == {0: 'T cells', 1: 'leiden 1', 2: 'B cells'}
        cl = {g['cluster_id']: g['name'] for g in ed._sample_gates['s1'].values()
              if g.get('kind') == 'cluster'}
        assert cl == {0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2'}
    finally:
        root.destroy()


def test_rename_phenograph_clusters():
    root, ed = _setup()
    try:
        ed._annotate_clusters()
        _fill_and_apply(ed, 'Annotate clusters', ['Mono', '', 'NK'])
        assert ed._cluster_labels['s1'] == {0: 'Mono', 2: 'NK'}
        cl = {g['cluster_id']: g['name'] for g in ed._sample_gates['s1'].values()
              if g.get('kind') == 'cluster'}
        assert cl == {0: 'Mono', 1: 'Cluster 1', 2: 'NK'}
        assert ed._cluster_display_name('s1', 2) == 'NK'
        # Clearing a name later restores the default.
        ed._annotate_clusters()
        _fill_and_apply(ed, 'Annotate clusters', ['', '', 'NK'])
        assert 0 not in ed._cluster_labels['s1']
        cl = {g['cluster_id']: g['name'] for g in ed._sample_gates['s1'].values()
              if g.get('kind') == 'cluster'}
        assert cl[0] == 'Cluster 0'
    finally:
        root.destroy()
