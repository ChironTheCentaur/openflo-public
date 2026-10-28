"""PhenoGraph's resolution: FlowSample.cluster(resolution=...), and the
Cluster dialog, openflo-run, the workspace, FlowExperiment and the Methods
paragraph that pass it on or record it.

PhenoGraph over-splits data with few markers. The fixture is four
well-separated populations in three channels (300 events each, centres 10
apart, sd 1): at the standard resolution 1.0 PhenoGraph returns more than
four clusters, each still inside one population, and at 0.4 it returns the
four populations. The truth is the population each event was drawn from.

Only PhenoGraph's reproducible route (its seeded Leiden backend) has a
resolution; its Louvain runs the Blondel binaries, which take none. Those
cases are checked with phenograph.cluster stubbed, since Louvain is not
seed-reproducible and would make the test flaky.
"""
from __future__ import annotations

import contextlib
import io
import logging
import tempfile
import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd
import pytest

from openflo.pipeline import FlowSample

# openflo-run's fixtures: a trial folder of two groups, and a stubbed run.
from tests.test_cli_run import stubbed as stubbed  # noqa: F401
from tests.test_cli_run import two_groups as two_groups  # noqa: F401

CH = ['FL1-A', 'FL2-A', 'FL3-A']
CENTRES = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], float)


def _blobs(n_per=300, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([c + rng.normal(0, 1, (n_per, 3)) for c in CENTRES])
    truth = np.repeat(np.arange(len(CENTRES)), n_per)
    return pd.DataFrame(X, columns=CH), truth


def _sample(name='SampleA', seed=0):
    df, truth = _blobs(seed=seed)
    s = FlowSample.from_dataframe(df, name=name)
    s.fluor_channels = list(CH)
    return s, truth


def _cluster(s, **kw):
    with contextlib.redirect_stdout(io.StringIO()):   # phenograph is chatty
        s.cluster(channels=CH, k=30, **kw)
    return s.data['cluster'].to_numpy()


def _pure(labels, truth):
    """True when every cluster holds events of one population only."""
    return all(len(set(truth[labels == c])) == 1 for c in np.unique(labels))


def _matches_truth(labels, truth):
    """True when the clusters ARE the populations (a 1:1 relabelling)."""
    pairs = set(zip(labels.tolist(), truth.tolist(), strict=True))
    return (len(pairs) == len(set(truth.tolist())) == len(set(labels.tolist())))


# ── FlowSample.cluster ──────────────────────────────────────────────────────

def test_resolution_1_gives_exactly_the_labels_of_the_code_before_it():
    """The old code path called phenograph.cluster(X, k, n_jobs,
    clustering_algo='leiden', seed) with no resolution. The default, and an
    explicit 1.0, must give those labels bit for bit."""
    import phenograph
    df, _ = _blobs()
    with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp), \
            contextlib.redirect_stdout(io.StringIO()):
        old, _, _ = phenograph.cluster(df[CH].to_numpy(float), k=30,
                                       n_jobs=1, clustering_algo='leiden',
                                       seed=42)
    s_default, _ = _sample()
    s_one, _ = _sample()
    assert np.array_equal(_cluster(s_default), old)
    assert np.array_equal(_cluster(s_one, resolution=1.0), old)
    assert s_one.cluster_settings['resolution'] == 1.0
    assert s_one.cluster_settings['route'] == 'leiden'


def test_lower_resolution_merges_an_over_split_result_into_the_populations():
    s, truth = _sample()
    standard = _cluster(s, resolution=1.0)
    # The over-split this setting is for: more clusters than populations,
    # every one of them inside a single population.
    assert len(np.unique(standard)) > 4
    assert _pure(standard, truth)
    low = _cluster(s, resolution=0.4)
    assert len(np.unique(low)) == 4
    assert _matches_truth(low, truth)
    assert s.cluster_settings['resolution'] == 0.4
    # Higher splits further still.
    assert len(np.unique(_cluster(s, resolution=2.0))) > len(np.unique(standard))


_UNSET = object()


class _FakePhenograph:
    """Records the arguments FlowSample.cluster passes to phenograph.cluster
    (only those it passes). Its signature has the two FlowSample.cluster
    looks for before it takes the seeded Leiden route."""

    def __init__(self):
        self.calls = []

    def __call__(self, data, clustering_algo=_UNSET, seed=_UNSET, **kw):
        if clustering_algo is not _UNSET:
            kw['clustering_algo'] = clustering_algo
        if seed is not _UNSET:
            kw['seed'] = seed
        self.calls.append(kw)
        return np.zeros(len(data), dtype=int), None, 0.5


@pytest.fixture
def fake_pg(monkeypatch):
    import phenograph
    fake = _FakePhenograph()
    monkeypatch.setattr(phenograph, 'cluster', fake)
    return fake


def test_the_leiden_route_gets_the_resolution_and_1_adds_nothing(fake_pg):
    s, _ = _sample()
    s.cluster(channels=CH, k=30, resolution=0.4)
    s.cluster(channels=CH, k=30)
    lowered, default = fake_pg.calls
    assert lowered == {'k': 30, 'n_jobs': 1, 'clustering_algo': 'leiden',
                       'seed': 42, 'resolution_parameter': 0.4}
    # 1.0 sends exactly the old arguments: nothing about resolution at all,
    # so phenograph's own default (1) is what runs.
    assert default == {'k': 30, 'n_jobs': 1, 'clustering_algo': 'leiden',
                       'seed': 42}


def test_louvain_route_takes_no_resolution_and_says_so(fake_pg, caplog):
    s, _ = _sample()
    with caplog.at_level(logging.WARNING, logger='openflo.pipeline'):
        s.cluster(channels=CH, k=30, resolution=0.4, reproducible=False,
                  use_gpu=False)
    (call,) = fake_pg.calls
    assert call == {'k': 30, 'n_jobs': 1}     # Louvain: no resolution
    assert s.cluster_settings['resolution'] is None
    assert s.cluster_settings['route'] == 'louvain'
    assert any('resolution 0.40 not applied' in r.getMessage()
               for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='openflo.pipeline'):
        s.cluster(channels=CH, k=30, reproducible=False, use_gpu=False)
    assert not any('not applied' in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize('bad', [0, -0.5, float('nan'), float('inf')])
def test_a_resolution_that_is_not_above_zero_is_refused(fake_pg, bad):
    s, _ = _sample()
    with pytest.raises(ValueError, match='resolution'):
        s.cluster(channels=CH, k=30, resolution=bad)
    assert fake_pg.calls == []


# ── FlowExperiment (Python API) ─────────────────────────────────────────────

def test_run_all_and_cluster_jointly_use_the_resolution():
    from openflo.pipeline import FlowExperiment
    exp = FlowExperiment.__new__(FlowExperiment)
    exp.cluster_params = exp.joint_clustering = None
    samples = [_sample('Donor1', seed=1), _sample('Donor2', seed=2)]
    exp.samples = {s.name: s for s, _ in samples}
    with contextlib.redirect_stdout(io.StringIO()):
        exp.run_all(qc=False, compensate=False, transform=False, k=30,
                    resolution=0.4)
    assert exp.cluster_params['resolution'] == 0.4
    for s, truth in samples:
        assert _matches_truth(s.data['cluster'].to_numpy(), truth)
    with contextlib.redirect_stdout(io.StringIO()):
        labels = exp.cluster_jointly()
    assert exp.joint_clustering['resolution'] == 0.4
    for (_s, truth), lab in zip(samples, labels.values(), strict=True):
        assert _matches_truth(np.asarray(lab), truth)


# ── openflo-run ─────────────────────────────────────────────────────────────

def test_cli_flag_is_parsed_and_checked():
    from openflo.cli_args import build_parser
    ap = build_parser()
    assert ap.parse_args([]).resolution == 1.0
    assert ap.parse_args(['--resolution', '0.4']).resolution == 0.4
    for bad in ('0', '-1', 'nan', 'abc'):
        with contextlib.redirect_stderr(io.StringIO()), \
                pytest.raises(SystemExit):
            ap.parse_args(['--resolution', bad])


def test_cli_worker_passes_the_resolution_to_phenograph(monkeypatch):
    from openflo.cli import _cluster_by_method
    seen = []
    monkeypatch.setattr(FlowSample, 'cluster',
                        lambda self, **kw: seen.append(kw))
    s, _ = _sample()
    task = {'k': 30, 'n_jobs': 1, 'random_state': 42}
    _cluster_by_method(s, dict(task, method='phenograph', resolution=0.4))
    _cluster_by_method(s, dict(task, method='phenograph'))
    assert seen[0]['resolution'] == 0.4
    assert seen[1]['resolution'] == 1.0


def test_openflo_run_resolution_reaches_every_phenograph_call(
        two_groups, tmp_path, stubbed, monkeypatch):
    """--resolution reaches each sample's clustering and the shared one."""
    import json

    from tests.test_cli_run import _main
    trial, groups, _ = two_groups
    seen = []
    orig = FlowSample.cluster

    def spy(self, **kw):
        seen.append(kw)
        return orig(self, **kw)
    monkeypatch.setattr(FlowSample, 'cluster', spy)
    _main(monkeypatch, '--trials', trial, '--groups', json.dumps(groups),
          '--fmo-sets', '{}', '-q', '--resolution', '0.4',
          '--out', str(tmp_path / 'a'))
    assert len(seen) == 5                    # 4 samples + 1 shared
    assert [kw['resolution'] for kw in seen] == [0.4] * 5


# ── workspace ──────────────────────────────────────────────────────────────

def test_workspace_run_applies_and_records_the_resolution(tmp_path):
    from openflo import workspace as W
    df, truth = _blobs()
    out = {}
    for res in (1.0, 0.4):
        cfg = dict(W.default_run_cfg(), method='phenograph', umap=False,
                   trimap=False, pacmap=False, phate=False, tsne=False,
                   max_events=0, reproducible=True, resolution=res)
        prep = {'data': df.copy(), 'channels': list(CH), 'label': 'SampleA',
                'n_events': len(df)}
        with contextlib.redirect_stdout(io.StringIO()):
            out[res] = W.compute_run(prep, cfg, str(tmp_path / str(res)))
        assert out[res]['ok'], out[res]
        assert out[res]['resolution'] == res
    assert out[1.0]['n_clusters'] > 4
    assert out[0.4]['n_clusters'] == 4


# ── Methods paragraph ──────────────────────────────────────────────────────

def test_methods_paragraph_reports_the_resolution_used():
    from openflo.provenance import methods_paragraph
    text = methods_paragraph([{'action': 'cluster', 'details': {
        'method': 'phenograph', 'k': 30, 'resolution': 0.4}}])
    assert 'PhenoGraph (Levine et al., 2015; k = 30, resolution 0.4)' in text
    # Louvain (resolution None) and entries from before: no resolution claim.
    for details in ({'method': 'phenograph', 'k': 30, 'resolution': None},
                    {'method': 'phenograph'}):
        text = methods_paragraph([{'action': 'cluster', 'details': details}])
        assert 'resolution' not in text and 'Levine et al., 2015' in text


# ── the Cluster dialog ─────────────────────────────────────────────────────

def _editor_or_skip():
    from tests.test_cluster_dialog import _editor_or_skip as make
    return make()


def _add(ed, df, name='SampleA'):
    from tests.test_cluster_dialog import _add as add
    return add(ed, df, name)


def _walk(w):
    stack = [w]
    while stack:
        x = stack.pop()
        yield x
        stack.extend(x.winfo_children())


def _open(ed):
    ed._open_cluster_dialog()
    return next(w for w in ed.winfo_children()
                if isinstance(w, tk.Toplevel) and w.title() == 'Cluster')


def _press(dlg, text):
    next(w for w in _walk(dlg)
         if isinstance(w, (ttk.Button, ttk.Radiobutton, ttk.Checkbutton))
         and w.cget('text') == text).invoke()


def test_dialog_shows_a_resolution_box_for_phenograph_with_a_tip(monkeypatch):
    from openflo.tool_window import CLUSTER_RESOLUTION_TIP
    root, ed = _editor_or_skip()
    try:
        _add(ed, _blobs()[0])
        tipped = []
        ed._tip = lambda widget, text: tipped.append((widget, text))
        got = {}
        ed._run_clustering = lambda **kw: got.update(kw)
        dlg = _open(ed)
        lbl, spin = ed._cluster_method_rows['resolution']
        # PhenoGraph is the default method: its resolution box is shown,
        # usable, pre-filled with 1.0 and explained on hover.
        assert spin.grid_info() and lbl.grid_info()
        assert str(spin.cget('state')) == 'normal'
        assert float(spin.get()) == 1.0
        assert (spin, CLUSTER_RESOLUTION_TIP) in tipped
        for words in ('1.0 is the standard', '0.3-0.5', 'over-split',
                      'more, smaller clusters'):
            assert words in CLUSTER_RESOLUTION_TIP
        assert not ed._cluster_resolution_note.grid_info()
        # Out of range is refused with a message, not silently replaced.
        from openflo import dialog_inputs
        errors = []
        monkeypatch.setattr(dialog_inputs.messagebox, 'showerror',
                            lambda *a, **k: errors.append(a))
        spin.delete(0, 'end')
        spin.insert(0, '9')
        _press(dlg, 'Run')
        assert got == {} and dlg.winfo_exists()
        assert len(errors) == 1 and 'Resolution' in errors[0][1]
        spin.delete(0, 'end')
        spin.insert(0, '0.4')
        _press(dlg, 'Run')
        assert got['method'] == 'phenograph'
        assert got['resolution'] == 0.4 and got['reproducible'] is True
    finally:
        root.destroy()


def test_dialog_greys_out_resolution_for_phenograph_louvain():
    root, ed = _editor_or_skip()
    try:
        _add(ed, _blobs()[0])
        got = {}
        ed._run_clustering = lambda **kw: got.update(kw)
        dlg = _open(ed)
        _, spin = ed._cluster_method_rows['resolution']
        spin.delete(0, 'end')
        spin.insert(0, '0.4')
        _press(dlg, 'Reproducible (seeded PhenoGraph)')     # untick
        assert str(spin.cget('state')) == 'disabled'
        assert ed._cluster_resolution_note.grid_info()
        assert 'needs Reproducible' in ed._cluster_resolution_note.cget('text')
        # Leiden always has a resolution, whatever the PhenoGraph box says.
        _press(dlg, 'Leiden')
        assert str(spin.cget('state')) == 'normal'
        assert not ed._cluster_resolution_note.grid_info()
        _press(dlg, 'PhenoGraph')
        assert str(spin.cget('state')) == 'disabled'
        _press(dlg, 'Run')
        # Not applied, so not sent: the run is the standard one.
        assert got['reproducible'] is False and got['resolution'] == 1.0
    finally:
        root.destroy()


def test_dialog_run_clusters_at_the_resolution_and_audits_it():
    root, ed = _editor_or_skip()
    try:
        df, truth = _blobs()
        s = _add(ed, df)
        with contextlib.redirect_stdout(io.StringIO()):
            ed._run_clustering(method='phenograph', all_samples=False, k=30,
                               grid=10, n_meta=10, embedding='none',
                               resolution=0.4, reproducible=True)
        assert _matches_truth(s.data['cluster'].to_numpy(), truth)
        (entry,) = [d for a, d in ed._audited if a == 'cluster']
        assert (entry['method'], entry['k'], entry['resolution'],
                entry['reproducible']) == ('phenograph', 30, 0.4, True)
    finally:
        root.destroy()


def test_session_keeps_the_resolution_through_the_methods_text():
    """The audit trail is what a saved session keeps and what the Methods
    window reads: the run's entry there names the resolution."""
    from openflo.provenance import methods_paragraph
    root, ed = _editor_or_skip()
    try:
        _add(ed, _blobs()[0])
        with contextlib.redirect_stdout(io.StringIO()):
            ed._run_clustering(method='leiden', all_samples=False, k=15,
                               grid=10, n_meta=10, embedding='none',
                               resolution=0.4)
        entries = [{'action': a, 'details': d} for a, d in ed._audited]
        assert 'resolution 0.4' in methods_paragraph(entries)
    finally:
        root.destroy()
