"""Clustering and embedding defaults that must agree across the API, the
Cluster dialog, the Pipeline panel and the CLI.

* ``channels=None`` takes one column per marker everywhere: each detector's
  area column, its height only when it has no area, never its width
  (pipeline.default_cluster_channels).
* ``run_leiden`` / ``run_louvain`` use 30 neighbours, as the app and CLI do.
* The Cluster dialog embeds on the channels it clustered, passes its event
  cap to FlowSOM and names the real caps in its status line.
* Compare embeddings: seed 42, UMAP 30 / 0.3 given to UMAP only.
* The Pipeline panel says how many of a unit's events it clustered.

All data are synthetic, on detector names FL1-A, FL1-H, ...
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openflo import pipeline
from openflo.pipeline import FlowSample, default_cluster_channels

# ── the shared helper ─────────────────────────────────────────────────────────


def test_one_channel_per_marker_area_else_height_never_width():
    names = ['FL1-A', 'FL1-H', 'FL1-W',          # all three: area only
             'FL2-H',                            # height only: kept
             'FL3 Area', 'FL3-Height', 'FL3 H',  # spelled-out area wins
             'FL4-W',                            # width only: dropped
             'fl6-a', 'FL6-H',                   # case does not matter
             'FL7_A', 'FL7_H',                   # underscore separator
             'Ch176']                            # no pulse suffix: kept
    assert default_cluster_channels(names) == [
        'FL1-A', 'FL2-H', 'FL3 Area', 'fl6-a', 'FL7_A', 'Ch176']


def test_height_spellings_are_heights_of_the_same_detector():
    # The two old helpers disagreed on '-Height': one dropped it, one kept it.
    assert default_cluster_channels(['FL3-A', 'FL3-Height']) == ['FL3-A']
    assert default_cluster_channels(['FL3 A', 'FL3 Height']) == ['FL3 A']
    assert default_cluster_channels(['FL3-Height', 'FL4-A']) == [
        'FL3-Height', 'FL4-A']


def test_order_is_kept_and_an_empty_pick_falls_back_to_every_channel():
    assert default_cluster_channels(['FL2-H', 'FL1-A', 'FL1-H']) == [
        'FL2-H', 'FL1-A']
    assert default_cluster_channels(['FL1-W', 'FL2-W']) == ['FL1-W', 'FL2-W']
    assert default_cluster_channels([]) == []
    assert default_cluster_channels(None) == []


def test_workspace_and_cli_use_the_shared_helper():
    from openflo import cli
    from openflo.workspace import proper_run_channels
    fl = ['FL1-A', 'FL1-H', 'FL1-W', 'FL2-H', 'FL3-A', 'FL3-Height']
    assert proper_run_channels(SimpleNamespace(fluor_channels=fl)) == \
        default_cluster_channels(fl) == ['FL1-A', 'FL2-H', 'FL3-A']
    assert not hasattr(cli, '_proper_markers')       # the CLI copy is gone
    assert 'default_cluster_channels' in inspect.getsource(cli._fit_cytonorm)


# ── every clusterer and embedding defaults to it ──────────────────────────────

def _three_blobs_on_fl2_height(n=100, seed=0):
    """Three populations separated ONLY on FL2-H (a marker with no area
    column). FL1-A is noise; FL1-H and FL1-W are non-finite everywhere, so a
    run that uses them has no finite event left, and a run that drops FL2-H
    sees only noise."""
    rng = np.random.default_rng(seed)
    truth = np.repeat([0, 1, 2], n)
    df = pd.DataFrame({
        'FL1-A': rng.normal(0.0, 0.3, 3 * n),
        'FL1-H': np.nan,
        'FL1-W': np.nan,
        'FL2-H': np.concatenate([rng.normal(c, 0.3, n) for c in (0, 10, 20)]),
    })
    s = FlowSample.from_dataframe(df, name='sample_a')
    assert s.fluor_channels == ['FL1-A', 'FL1-H', 'FL1-W', 'FL2-H']
    return s, truth


def _pure(labels, truth):
    from sklearn.metrics import homogeneity_score
    labels = np.asarray(labels)
    assert (labels >= 0).all(), 'events were left unclustered'
    return homogeneity_score(truth, labels) > 0.95


@pytest.mark.parametrize('run, col', [
    (lambda s: s.run_leiden(resolution=0.3), 'leiden'),
    (lambda s: s.run_louvain(restarts=1), 'louvain'),
    (lambda s: s.cluster(k=15), 'cluster'),
    (lambda s: s.run_flowsom(grid=(3, 3), n_metaclusters=3), 'flowsom_meta'),
])
def test_clusterers_default_to_one_channel_per_marker(run, col):
    s, truth = _three_blobs_on_fl2_height()
    run(s)
    assert col in s.data.columns
    assert _pure(s.data[col], truth)


def test_flowsom_records_the_default_channels():
    s, _ = _three_blobs_on_fl2_height()
    s.run_flowsom(grid=(3, 3), n_metaclusters=3)
    assert s.flowsom_result['channels'] == ['FL1-A', 'FL2-H']


def test_embeddings_default_to_one_channel_per_marker():
    s, _ = _three_blobs_on_fl2_height()
    _X, sub, avail = s._embedding_input(None, None, 0)
    assert avail == ['FL1-A', 'FL2-H'] and sub.all()
    s.run_tsne(perplexity=10.0)
    assert np.isfinite(s.data[['TSNE1', 'TSNE2']].to_numpy()).all()


# ── Leiden / Louvain neighbours ───────────────────────────────────────────────

def test_leiden_and_louvain_default_to_30_neighbours(monkeypatch):
    seen = []
    real = pipeline._snn_jaccard_graph

    def spy(X, k, prune=False):
        seen.append(k)
        return real(X, k, prune=prune)
    monkeypatch.setattr(pipeline, '_snn_jaccard_graph', spy)
    s, _ = _three_blobs_on_fl2_height()
    s.run_leiden()
    s.run_louvain(restarts=1)
    assert seen == [30, 30]
    for fn in (FlowSample.run_leiden, FlowSample.run_louvain):
        assert inspect.signature(fn).parameters['n_neighbors'].default == 30


# ── the Cluster dialog ────────────────────────────────────────────────────────

def _gui():
    from tests.test_cluster_dialog import _add, _dialog, _editor_or_skip, _walk
    return _add, _dialog, _editor_or_skip, _walk


def _hw_frame(n=60, seed=1):
    rng = np.random.default_rng(seed)
    base = np.concatenate([rng.normal(c, 0.3, n) for c in (0, 8, 16)])
    return pd.DataFrame({'FL1-A': base, 'FL1-H': base * 0.9,
                         'FL1-W': rng.normal(1, 0.1, 3 * n),
                         'FL2-H': base[::-1].copy()})


def test_dialog_lists_every_channel_but_ticks_one_per_marker():
    from tkinter import ttk
    _add, _dialog, _editor_or_skip, _walk = _gui()
    root, ed = _editor_or_skip()
    try:
        _add(ed, _hw_frame())
        got = []
        ed._run_clustering = lambda **kw: got.append(kw)
        ed._open_cluster_dialog()
        dlg = _dialog(ed, 'Cluster')
        lb = ed._cluster_marker_list
        assert lb.get(0, 'end') == ('FL1-A', 'FL1-H', 'FL1-W', 'FL2-H')
        assert lb.curselection() == (0, 3)
        run = next(w for w in _walk(dlg) if isinstance(w, ttk.Button)
                   and w.cget('text') == 'Run')
        run.invoke()
        assert got[-1]['channels'] is None        # the default, as pre-ticked
        ed._open_cluster_dialog()
        dlg = _dialog(ed, 'Cluster')
        lb = ed._cluster_marker_list
        lb.selection_clear(3)
        lb.selection_set(1)
        next(w for w in _walk(dlg) if isinstance(w, ttk.Button)
             and w.cget('text') == 'Run').invoke()
        assert got[-1]['channels'] == ['FL1-A', 'FL1-H']
    finally:
        root.destroy()


def _spied_editor():
    _add, _dialog, _editor_or_skip, _walk = _gui()
    root, ed = _editor_or_skip()
    s = _add(ed, _hw_frame())
    calls = {}
    real_flowsom = type(s).run_flowsom

    def flowsom(self, **kw):
        calls['flowsom'] = kw
        return real_flowsom(self, **kw)

    def tsne(self, **kw):
        calls['tsne'] = kw
        self.data['TSNE1'] = 0.0
        self.data['TSNE2'] = 0.0
    s.run_flowsom = flowsom.__get__(s)
    s.run_tsne = tsne.__get__(s)
    busy = []
    run_async = ed.run_async

    def spy_async(work, on_done=None, on_error=None, busy_msg=None):
        busy.append(busy_msg)
        return run_async(work, on_done, on_error, busy_msg)
    ed.run_async = spy_async
    return root, ed, s, calls, busy


def test_dialog_embeds_on_the_channels_it_clustered():
    from openflo.result_stamp import column_stamps
    root, ed, s, calls, _busy = _spied_editor()
    try:
        ed._run_clustering(method='flowsom', all_samples=False, k=30, grid=3,
                           n_meta=3, embedding='t-SNE',
                           channels=['FL1-A', 'FL2-H'])
        assert calls['tsne']['channels'] == ['FL1-A', 'FL2-H']
        # None: both resolve to the same default, and the stamps say which.
        ed._run_clustering(method='flowsom', all_samples=False, k=30, grid=3,
                           n_meta=3, embedding='t-SNE', channels=None)
        assert calls['tsne']['channels'] is None
        stamps = column_stamps(s)
        assert stamps['TSNE1']['inputs'] == ['FL1-A', 'FL2-H']
        assert stamps['flowsom_meta']['inputs'] == ['FL1-A', 'FL2-H']
    finally:
        root.destroy()


def test_dialog_passes_its_cap_to_flowsom_and_names_the_real_caps():
    root, ed, s, calls, busy = _spied_editor()
    try:
        kw = dict(all_samples=False, k=15, grid=3, n_meta=3, embedding='none')
        ed._run_clustering(method='flowsom', **kw)
        assert calls['flowsom']['max_events'] == 50_000
        assert '50,000' in busy[-1] and 'full data' not in busy[-1]
        ed._run_clustering(method='flowsom', downsample='Custom…',
                           custom_n='150', **kw)
        assert calls['flowsom']['max_events'] == 150
        assert '150 ev/sample' in busy[-1]
        ed._run_clustering(method='leiden', **kw)
        assert '200,000' in busy[-1] and 'full data' not in busy[-1]
        ed._run_clustering(method='phenograph', **kw)
        assert 'full data' in busy[-1]           # PhenoGraph has no cap
    finally:
        root.destroy()


def test_compare_embeddings_window_uses_one_channel_per_marker(monkeypatch):
    import openflo.dr_compare as drc
    import openflo.ui_embedding as ue
    _add, _dialog, _editor_or_skip, _walk = _gui()
    monkeypatch.setattr(drc, 'available_methods', lambda: ['tsne'])
    seen = {}
    monkeypatch.setattr(ue, 'EmbeddingDialog',
                        lambda *a: seen.setdefault('args', a))
    root, ed = _editor_or_skip()
    try:
        _add(ed, _hw_frame())
        ed._open_dr_compare()
        assert seen['args'][-1] == ['FL1-A', 'FL2-H']
    finally:
        root.destroy()


# ── Compare embeddings: per-method settings ───────────────────────────────────

def test_run_embeddings_seed_42_and_umap_settings_only_for_umap(monkeypatch):
    import openflo.dr_compare as drc
    got = {}

    def fake(method, X, seed, **kw):
        got[method] = (seed, kw)
        return np.zeros((len(X), 2))
    monkeypatch.setattr(drc, '_embed_one', fake)
    monkeypatch.setattr(drc, '_backend_importable', lambda m: True)
    X = np.random.default_rng(0).normal(size=(40, 3))
    out = drc.run_embeddings(X, methods=('umap', 'tsne', 'phate'))
    assert not out['skipped']
    assert got['umap'] == (42, {'n_neighbors': 30, 'min_dist': 0.3})
    assert got['tsne'] == (42, {})
    assert got['phate'] == (42, {})
    drc.run_embeddings(X, methods=('umap', 'tsne'), seed=7,
                       method_kw={'umap': {'n_neighbors': 15},
                                  'tsne': {'perplexity': 5.0}})
    assert got['umap'] == (7, {'n_neighbors': 15, 'min_dist': 0.3})
    assert got['tsne'] == (7, {'perplexity': 5.0})


# ── Pipeline panel ────────────────────────────────────────────────────────────

def _ws_sample(n=300, seed=0):
    rng = np.random.default_rng(seed)
    cols = {}
    for m in ('FL1', 'FL2', 'FL3'):
        base = rng.normal(size=n)
        cols[f'{m}-A'] = base + rng.normal(scale=0.1, size=n)
        cols[f'{m}-H'] = base * 0.9
    return FlowSample.from_dataframe(pd.DataFrame(cols), name='sample_a')


def test_panel_result_says_clustered_n_of_m_events(tmp_path):
    from openflo import workspace as W
    editor = SimpleNamespace(_samples={'sample_a': _ws_sample()})
    item = {'sample': 'sample_a', 'path': None, 'gate_id': None, 'gates': {}}
    cfg = {'k': 10, 'max_events': 100, 'seed': 42, 'umap': False,
           'trimap': False}
    res = W.run_workspace_item(editor, item, cfg, str(tmp_path))
    assert res['ok'], res['error']
    assert (res['n_events'], res['n_total']) == (100, 300)
    assert W.run_events_text(res) == 'clustered 100 of 300 events'
    full = W.run_workspace_item(editor, item, dict(cfg, max_events=0),
                                str(tmp_path / 'full'))
    assert (full['n_events'], full['n_total']) == (300, 300)
    assert W.run_events_text(full) == '300 ev'
    assert W.DEFAULT_RUN_CFG['max_events'] == 5000


def test_panel_status_line_shows_the_subsample(tmp_path):
    import collections
    import pickle
    import queue

    from openflo import workspace as W
    res = dict(W._blank_result('G', 100), ok=True, n_clusters=2, n_total=300)
    result = tmp_path / 'result.pkl'
    with open(result, 'wb') as fh:
        pickle.dump(res, fh)
    (tmp_path / 'job').mkdir()
    lines = []
    view = SimpleNamespace(
        status_var=SimpleNamespace(set=lines.append), _total=1, _ok=0,
        _err=0, _jobs=collections.deque(), _warnings=[], _prep=None,
        _cancel_requested=False, _out_dir=str(tmp_path),
        _absorb_stdout=lambda cur: None, _finish=lambda *_a: None,
        _cur={'job': {'label': 'G', 'attempt': 1, 'cfg': {}},
              'proc': SimpleNamespace(poll=lambda: 0),
              'result_path': str(result), 'jobdir': str(tmp_path / 'job'),
              'n': 1, 'msgq': queue.Queue()})
    W.WorkspacePanel._finalize_current(view)
    assert 'clustered 100 of 300 events' in lines[-1], lines


def test_hand_built_panel_config_runs_reproducible(tmp_path, monkeypatch):
    from openflo import workspace as W
    seen = {}
    real = FlowSample.cluster

    def spy(self, **kw):
        seen.update(kw)
        return real(self, **kw)
    monkeypatch.setattr(FlowSample, 'cluster', spy)
    editor = SimpleNamespace(_samples={'sample_a': _ws_sample()})
    item = {'sample': 'sample_a', 'path': None, 'gate_id': None, 'gates': {}}
    cfg = {'k': 10, 'max_events': 0, 'seed': 42, 'umap': False}   # no key
    res = W.run_workspace_item(editor, item, cfg, str(tmp_path))
    assert res['ok'], res['error']
    assert seen['reproducible'] is True


def test_hand_built_panel_config_ticks_reproducible():
    import os
    import tkinter as tk

    from openflo import workspace as W
    from tests.conftest import gui_unavailable
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        root = tk.Tk()
    except Exception as e:                        # noqa: BLE001
        gui_unavailable(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    try:
        model = W.WorkspaceModel()
        model.run_cfg = {'method': 'phenograph'}      # built by hand: no key
        view = W.PipelineWorkspaceView(root, model)
        assert view._reproducible_var.get() is True
        view._reproducible_var.set(False)
        model.run_cfg = {'method': 'phenograph'}
        view._load_cfg_into_widgets()
        assert view._reproducible_var.get() is True
    finally:
        root.destroy()


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_cli_group_umap_uses_the_api_sample_size(monkeypatch):
    from openflo import cli

    class Stop(Exception):
        pass
    assert inspect.signature(FlowSample.run_umap).parameters[
        'sample_n'].default == 100_000
    seen = {}

    def spy(self, **kw):
        seen.update(kw)
        raise Stop
    monkeypatch.setattr(FlowSample, 'run_umap', spy)
    s = _ws_sample()
    with pytest.raises(Stop):
        cli.run_group_umap([s], 'group_a', 'unused', random_state=42)
    assert 'sample_n' not in seen                 # run_umap's own 100,000


def test_cli_k_help_names_leiden_too():
    from openflo.cli_args import build_parser
    act = next(a for a in build_parser()._actions if '--k' in a.option_strings)
    assert 'Leiden' in act.help and 'PhenoGraph' in act.help
