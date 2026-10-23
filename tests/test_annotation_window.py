"""Analyze -> Annotate populations names clusters by phenotype, and only the
clusters it was asked about.

annotate.py's pure functions are unit-tested, but the window that drives them
(ui_annotation.PopulationAnnotationWindow: detector->antibody relabel, MEM,
reference matching, write-back through editor._apply_population_names) never
ran under the suite. Four synthetic blobs with known phenotypes give a known
answer under the window's default reference table.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(__file__))

from test_gui_smoke import _editor_or_skip  # noqa: E402

MARKERS = ['CD3-A', 'CD4-A', 'CD8-A', 'CD19-A']
PHENO = {0: (1, 1, 0, 0), 1: (1, 0, 1, 0), 2: (0, 0, 0, 1), 3: (0, 0, 0, 0)}
EXPECT = {0: 'CD4 T', 1: 'CD8 T', 2: 'B cell', 3: 'unknown'}


def _sample(with_cluster_col):
    from openflo.pipeline import FlowSample
    rng = np.random.default_rng(0)
    blocks, pops = [], []
    for p, ph in PHENO.items():
        blocks.append(np.column_stack([rng.normal(8 if on else 1, 0.5, 300)
                                       for on in ph]))
        pops += [p] * 300
    df = pd.DataFrame(np.vstack(blocks), columns=MARKERS)
    df['leiden'] = np.array(pops)
    if with_cluster_col:
        # A PhenoGraph run on the same sample numbers the same cells differently.
        df['cluster'] = (np.array(pops) + 1) % 4
    return FlowSample.from_dataframe(
        df, name='pbmc', labels={m: m.split('-')[0] for m in MARKERS})


def _editor_with(s):
    root, ed, _gui = _editor_or_skip()
    name = s.name
    ed._samples[name] = s
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = True
    ed._sample_gates[name] = {}
    ed._sample_gate_order[name] = []
    ed._sample_gate_seq[name] = 0
    ed._channels = list(s.data.columns)
    ed._channel_labels = dict(s.channel_labels)
    ed._active_sample = name
    return root, ed


def _annotate_leiden(ed):
    from openflo.ui_annotation import PopulationAnnotationWindow
    w = PopulationAnnotationWindow(ed, 'pbmc')
    w.withdraw()
    w.col_var.set('leiden')
    w._compute()
    w._apply()
    return w


def test_window_names_each_cluster_by_its_phenotype():
    root, ed = _editor_with(_sample(with_cluster_col=False))
    try:
        ed._import_populations('leiden')
        w = _annotate_leiden(ed)
        # MEM is computed on antibody names, not detector names.
        assert list(w._mem.columns) == ['CD3', 'CD4', 'CD8', 'CD19']
        assigned = {int(i): w._tv.set(i, 'name') for i in w._tv.get_children()}
        assert float(w.state_thr_var.get()) == 3.0   # at the default
        assert assigned == EXPECT
        leiden = {int(g['value']): g['name']
                  for g in ed._sample_gates['pbmc'].values()
                  if g.get('kind') == 'category' and g.get('channel') == 'leiden'}
        assert leiden[0] == 'CD4 T' and leiden[1] == 'CD8 T'
        assert leiden[2] == 'B cell'
        assert leiden[3] == 'leiden 3'          # 'unknown' is not written back
    finally:
        root.destroy()


def _named_with_states(ed, mem, state_thr=None):
    """Run Assign names on a planted MEM table (pops x markers)."""
    from openflo.ui_annotation import PopulationAnnotationWindow
    w = PopulationAnnotationWindow(ed, 'pbmc')
    w.withdraw()
    w.col_var.set('leiden')
    w._compute()
    w._mem = mem
    if state_thr is not None:
        w.state_thr_var.set(state_thr)
    w._apply()
    return w, {int(i): w._tv.set(i, 'name') for i in w._tv.get_children()}


def _weak_cd19_mem(cd19):
    # Population 2 is clearly CD3- with a weak CD19 score; the others carry
    # no markers. Under the default table that is 'B cell' only if CD19 counts.
    mem = pd.DataFrame(0.0, index=[0, 1, 2, 3],
                       columns=['CD3', 'CD4', 'CD8', 'CD19'])
    mem.loc[2, 'CD3'] = -8.0
    mem.loc[2, 'CD19'] = cd19
    return mem


def test_state_threshold_defaults_to_population_states_and_is_separate():
    """Reference matching uses its own State threshold, 3 like
    annotate.population_states, not the MEM label threshold (2): a CD19 score
    of 2.5 does not make a CD3- population a B cell."""
    import inspect

    from openflo.annotate import population_states
    root, ed = _editor_with(_sample(with_cluster_col=False))
    try:
        ed._import_populations('leiden')
        w, named = _named_with_states(ed, _weak_cd19_mem(2.5))
        dflt = inspect.signature(population_states).parameters['threshold']
        assert float(w.state_thr_var.get()) == dflt.default == 3.0
        assert float(w.thr_var.get()) == 2.0     # MEM label field unchanged
        assert named[2] == 'unknown'
        # The added field keeps the window within 1024x768.
        w.update_idletasks()
        assert w.winfo_reqwidth() <= 1024 and w.winfo_reqheight() <= 768
        # The same population with CD19 above 3 is a B cell.
        _w, named = _named_with_states(ed, _weak_cd19_mem(4.0))
        assert named[2] == 'B cell'
    finally:
        root.destroy()


def test_state_threshold_below_two_is_used_as_given():
    """No floor at 2: a State threshold of 1.5 lets a CD19 score of 1.5
    count, so the population is named B cell."""
    root, ed = _editor_with(_sample(with_cluster_col=False))
    try:
        ed._import_populations('leiden')
        _w, named = _named_with_states(ed, _weak_cd19_mem(1.5), '1.5')
        assert named[2] == 'B cell'
    finally:
        root.destroy()


def test_annotating_leiden_leaves_phenograph_clusters_alone():
    # Naming Leiden populations must not relabel the PhenoGraph clusters: the
    # two id spaces partition the same cells differently.
    root, ed = _editor_with(_sample(with_cluster_col=True))
    try:
        ed._import_populations('cluster')
        ed._import_populations('leiden')
        _annotate_leiden(ed)
        cluster_names = {g['cluster_id']: g['name']
                         for g in ed._sample_gates['pbmc'].values()
                         if g.get('kind') == 'cluster'}
        assert cluster_names == {c: f'Cluster {c}' for c in range(4)}, (
            f'PhenoGraph clusters renamed by a Leiden annotation: {cluster_names}')
        assert ed._cluster_display_name('pbmc', 0) == 'Cluster 0'
    finally:
        root.destroy()
