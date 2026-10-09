"""Confirmed clustering defects, each pinned as a STRICT xfail.

Each test states the behaviour that should hold and currently does not. They
report "xfailed" today; when a defect is fixed its test XPASSes, which
strict=True turns into a failure, so the marker gets removed and the test
starts guarding the fix. Repro scripts for each sit next to this file.
"""
import os
import random
import tkinter as tk

import numpy as np
import pandas as pd
import pytest

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
    ed.run_async = lambda work, on_done=None, on_error=None, busy_msg=None: (
        on_done(work()))
    return root, ed


def _add(ed, df, name='s1'):
    from openflo.pipeline import FlowSample
    s = FlowSample.from_dataframe(df, name=name, path=rf'C:\exp\{name}.fcs')
    ed._samples[name] = s
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = True
    ed._channels = list(df.columns)
    ed._channel_labels = {c: c for c in df.columns}
    ed._populate_channel_combos()
    ed._set_active_sample(name)
    return s


def _blobs(sizes=(150, 100, 60), seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(8.0 * i, 0.4, (n, 3)) for i, n in enumerate(sizes)])
    return pd.DataFrame(X, columns=['CD1', 'CD2', 'CD3'])


def test_cluster_dialog_defaults_to_reproducible():
    # Pressing Run with the dialog defaults must cluster reproducibly, as the
    # library, the CLI and the workspace do.
    import inspect

    from openflo.pipeline import FlowSample
    root, ed = _editor_or_skip()
    try:
        _add(ed, _blobs())
        got = {}
        ed._run_clustering = lambda **kw: got.update(kw)
        ed._open_cluster_dialog()
        dlg = next(w for w in ed.winfo_children()
                   if isinstance(w, tk.Toplevel) and w.title() == 'Cluster')
        stack = [dlg]
        while stack:
            w = stack.pop()
            if getattr(w, 'cget', None) and w.winfo_class() == 'TButton' \
                    and w.cget('text') == 'Run':
                w.invoke()
                break
            stack.extend(w.winfo_children())
        lib = inspect.signature(FlowSample.cluster).parameters['reproducible']
        assert got['reproducible'] is lib.default is True
    finally:
        root.destroy()


def test_flowsom_that_wrote_nothing_is_not_reported_as_done():
    # run_flowsom skips a sample with fewer events than grid x grid; the run
    # must then not be reported as done, audited, or put in the Methods text.
    root, ed = _editor_or_skip()
    try:
        audited = []
        ed._audit = lambda action, **d: audited.append(action)
        s = _add(ed, _blobs(sizes=(30, 20, 10)))      # 60 events < 10x10 grid
        ed._run_clustering(method='flowsom', all_samples=False, k=30, grid=10,
                           n_meta=10, embedding='none')
        if 'flowsom_meta' in s.data.columns:          # precondition, not the defect
            pytest.fail('precondition: FlowSOM on 60 events was expected to write nothing')
        assert 'done' not in ed.status_var.get()
        assert 'cluster' not in audited
    finally:
        root.destroy()


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="gui.py:1349 LABEL_COLUMNS has no "
                   "'leiden' entry: its -1 bucket imports as 'leiden -1' in a "
                   "palette colour, and the Populations menu omits Leiden")
def test_leiden_noise_bucket_is_named_and_grey():
    from openflo.pipeline import NOISE_CLUSTER_COLOR
    root, ed = _editor_or_skip()
    try:
        df = _blobs()
        df.iloc[[0, 1], 0] = np.nan
        _add(ed, df)
        ed._run_clustering(method='leiden', all_samples=False, k=15, grid=10,
                           n_meta=10, embedding='none', resolution=0.5)
        noise = [g for g in ed._sample_gates['s1'].values()
                 if g.get('kind') == 'category' and g.get('value') == -1]
        if not noise:                                 # precondition, not the defect
            pytest.fail('precondition: no Leiden noise population was imported')
        assert noise[0]['color'] == NOISE_CLUSTER_COLOR
        assert '-1' not in noise[0]['name']
        assert 'leiden' in ed._label_columns_present()
    finally:
        root.destroy()


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='pipeline.py:5710 run_louvain replaces '
                   "igraph's default RNG (the random module) with a fresh "
                   'random.Random(), so random.seed() no longer controls igraph')
def test_run_louvain_leaves_igraph_rng_as_found():
    import igraph as ig

    from openflo.pipeline import FlowSample

    def edges():
        random.seed(123)
        return ig.Graph.Erdos_Renyi(n=40, p=0.2).get_edgelist()
    ig.set_random_number_generator(random)              # igraph's default
    if edges() != edges():                              # precondition, not the defect
        pytest.fail('precondition: random.seed() does not control igraph here')
    rng = np.random.default_rng(0)
    s = FlowSample.from_dataframe(
        pd.DataFrame(rng.normal(size=(100, 3)), columns=['A', 'B', 'C']))
    s.run_louvain(channels=['A', 'B', 'C'], n_neighbors=10, restarts=1)
    try:
        assert edges() == edges()
    finally:
        ig.set_random_number_generator(random)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='pipeline.py:2881 flowsom_layout ignores '
                   'its seed argument (layout_fruchterman_reingold(seed=None))')
def test_flowsom_layout_seed_gives_the_same_layout():
    from openflo.pipeline import flowsom_layout, flowsom_mst
    edges, _ = flowsom_mst(np.random.default_rng(1).normal(size=(25, 4)))
    assert np.allclose(flowsom_layout(25, edges, seed=42),
                       flowsom_layout(25, edges, seed=42))
