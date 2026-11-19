"""The helpers FlowSample.cluster is built from, path by path.

FlowSample.cluster was split into small helpers (resolution check,
sub-sample pick, GPU admission, PhenoGraph arguments, the CPU run, the
settings record, sub-sample label expansion, label storage). Each test
below pins one branch with an expected value worked out by hand or from an
independent computation, so a wrong branch, a swapped comparison or a
dropped log line fails. PhenoGraph and the GPU are replaced by stand-ins:
real PhenoGraph Louvain is not seed-reproducible, and the suite has no GPU.
All data are synthetic, on channels FL1-A..FL3-A.
"""
from __future__ import annotations

import logging
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

import openflo.flow_sample as P
from openflo.exceptions import ClusteringError
from openflo.pipeline import FlowSample

CH = ['FL1-A', 'FL2-A', 'FL3-A']
LOGGER = 'openflo.pipeline'


def _msgs(caplog, level):
    return [r.getMessage() for r in caplog.records if r.levelno == level]


def _seeded_cluster(calls, labels=None, q=0.4, raises=None):
    """A phenograph.cluster stand-in that accepts the seeded-Leiden args."""
    def cluster(X, k=30, n_jobs=1, clustering_algo='louvain', seed=None,
                resolution_parameter=1.0):
        calls.append({'X': X.copy(), 'k': k, 'n_jobs': n_jobs,
                      'clustering_algo': clustering_algo, 'seed': seed,
                      'resolution_parameter': resolution_parameter,
                      'cwd': os.getcwd()})
        if raises is not None:
            raise raises
        lab = (np.zeros(len(X), dtype=int) if labels is None
               else labels(X))
        return lab, None, q
    return cluster


@pytest.fixture
def fake_pg(monkeypatch):
    """Install a stand-in `phenograph` module; return (module, calls)."""
    calls = []
    mod = types.ModuleType('phenograph')
    mod.cluster = _seeded_cluster(calls)
    monkeypatch.setitem(sys.modules, 'phenograph', mod)
    return mod, calls


# ── _checked_resolution ─────────────────────────────────────────────────────

def test_resolution_is_returned_as_float():
    assert P._checked_resolution('1.5') == 1.5
    assert type(P._checked_resolution(2)) is float


@pytest.mark.parametrize('bad', [0, -0.5, float('nan'), float('inf')])
def test_resolution_not_finite_and_positive_is_refused(bad):
    with pytest.raises(ValueError, match='resolution must be > 0'):
        P._checked_resolution(bad)


# ── _cluster_subsample ──────────────────────────────────────────────────────

@pytest.mark.parametrize('cap', [None, 0, 10, 50])
def test_no_subsample_when_cap_unset_or_not_exceeded(cap):
    Xc = np.arange(30.0).reshape(10, 3)
    idx, Xs = P._cluster_subsample(Xc, cap, 42)
    assert idx is None
    assert Xs is Xc


def test_subsample_is_the_sorted_seeded_choice():
    Xc = np.arange(60.0).reshape(20, 3)
    idx, Xs = P._cluster_subsample(Xc, 7, 3)
    expected = np.sort(np.random.default_rng(3).choice(20, 7, replace=False))
    assert idx is not None
    np.testing.assert_array_equal(idx, expected)
    np.testing.assert_array_equal(Xs, Xc[expected])
    assert np.all(np.diff(idx) > 0)


# ── _should_try_gpu ─────────────────────────────────────────────────────────

@pytest.fixture
def vram(monkeypatch):
    """GPU kit present; free VRAM set per test via the returned dict."""
    state = {'free': None, 'asked': 0}

    def free():
        state['asked'] += 1
        return state['free']
    monkeypatch.setattr(P, 'GPU_CLUSTERING_AVAILABLE', True)
    monkeypatch.setattr(P, '_vram_free_gb', free)
    return state


def test_reproducible_never_tries_gpu_nor_asks_vram(vram):
    assert P._should_try_gpu(True, True, 1.0) is False
    assert vram['asked'] == 0


def test_use_gpu_false_never_tries_gpu(vram):
    assert P._should_try_gpu(False, False, 1.0) is False
    assert vram['asked'] == 0


def test_no_gpu_kit_never_tries_gpu(vram, monkeypatch):
    monkeypatch.setattr(P, 'GPU_CLUSTERING_AVAILABLE', False)
    assert P._should_try_gpu(False, 'auto', 1.0) is False


def test_forced_gpu_ignores_low_vram(vram):
    vram['free'] = 0.1
    assert P._should_try_gpu(False, True, 1.0) is True


def test_auto_admits_unknown_vram(vram):
    vram['free'] = None
    assert P._should_try_gpu(False, 'auto', 1.0) is True


def test_auto_admits_vram_at_the_threshold(vram):
    vram['free'] = 1.0
    assert P._should_try_gpu(False, 'auto', 1.0) is True


def test_auto_refuses_low_vram_and_says_so(vram, caplog):
    vram['free'] = 0.4
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert P._should_try_gpu(False, 'auto', 1.0) is False
    assert _msgs(caplog, logging.INFO) == [
        '  [VRAM admission] 0.4 GB < 1.0 GB — clustering on CPU']


# ── _phenograph_kwargs ──────────────────────────────────────────────────────

def test_louvain_kwargs_when_not_reproducible():
    kw, route = P._phenograph_kwargs(_seeded_cluster([]), 15, 2, False, 7, 0.5)
    assert (kw, route) == ({'k': 15, 'n_jobs': 2}, 'louvain')


def test_seeded_leiden_kwargs_send_no_default_resolution():
    kw, route = P._phenograph_kwargs(
        _seeded_cluster([]), 15, 2, True, np.int64(7), 1.0)
    assert route == 'leiden'
    assert kw == {'k': 15, 'n_jobs': 2, 'clustering_algo': 'leiden',
                  'seed': 7}
    assert type(kw['seed']) is int


def test_seeded_leiden_kwargs_pass_other_resolution():
    kw, route = P._phenograph_kwargs(_seeded_cluster([]), 30, 1, True, 0, 0.5)
    assert route == 'leiden'
    assert kw['resolution_parameter'] == 0.5


@pytest.mark.parametrize('sig', ['no_seed', 'no_algo'])
def test_old_phenograph_falls_back_to_louvain_with_a_warning(sig, caplog):
    if sig == 'no_seed':
        def fn(X, k=30, n_jobs=1, clustering_algo='louvain'):
            pass
    else:
        def fn(X, k=30, n_jobs=1, seed=None):
            pass
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        kw, route = P._phenograph_kwargs(fn, 30, 1, True, 5, 0.5)
    assert (kw, route) == ({'k': 30, 'n_jobs': 1}, 'louvain')
    assert len(_msgs(caplog, logging.WARNING)) == 1
    assert 'seeded Leiden' in _msgs(caplog, logging.WARNING)[0]


# ── _run_phenograph_cpu ─────────────────────────────────────────────────────

def test_cpu_run_returns_labels_q_and_route_from_a_scratch_dir(fake_pg):
    mod, calls = fake_pg
    mod.cluster = _seeded_cluster(calls, labels=lambda X: np.arange(len(X)),
                                  q=0.61)
    X = np.ones((4, 3))
    start = os.getcwd()
    comm, q, route = P._run_phenograph_cpu(X, 9, 1, True, 3, 1.0, 'shape')
    np.testing.assert_array_equal(comm, [0, 1, 2, 3])
    assert (q, route) == (0.61, 'leiden')
    assert calls[0]['k'] == 9 and calls[0]['seed'] == 3
    assert calls[0]['cwd'] != start
    assert os.getcwd() == start


def test_cpu_run_without_phenograph_raises_clustering_error(monkeypatch):
    monkeypatch.setitem(sys.modules, 'phenograph', None)
    with pytest.raises(ClusteringError, match='phenograph is required') as ei:
        P._run_phenograph_cpu(np.ones((3, 3)), 5, 1, False, 0, 1.0, 's')
    assert isinstance(ei.value.__cause__, ImportError)


def test_cpu_run_failure_is_wrapped(fake_pg):
    mod, calls = fake_pg
    mod.cluster = _seeded_cluster(calls, raises=RuntimeError('graph broke'))
    with pytest.raises(ClusteringError,
                       match='Phenograph CPU clustering failed: graph broke'):
        P._run_phenograph_cpu(np.ones((3, 3)), 5, 1, False, 0, 1.0, 's')


# ── _phenograph_settings ────────────────────────────────────────────────────

def test_settings_record_resolution_only_for_leiden(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        lei = P._phenograph_settings(np.int64(15), 'leiden', 'CPU', 4, 0.5)
    assert lei == {'method': 'phenograph', 'k': 15, 'route': 'leiden',
                   'backend': 'CPU', 'random_state': 4, 'resolution': 0.5}
    assert type(lei['k']) is int
    assert _msgs(caplog, logging.WARNING) == []


def test_settings_warn_when_louvain_cannot_apply_resolution(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        lou = P._phenograph_settings(30, 'louvain', 'GPU', 1, 0.5)
    assert lou['resolution'] is None and lou['backend'] == 'GPU'
    assert _msgs(caplog, logging.WARNING) == [
        "  [!] resolution 0.50 not applied: PhenoGraph's Louvain has no "
        "resolution; it needs reproducible (seeded Leiden) clustering."]


def test_settings_no_warning_for_louvain_at_default(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        P._phenograph_settings(30, 'louvain', 'CPU', 1, 1.0)
    assert _msgs(caplog, logging.WARNING) == []


# ── _expand_subsample_labels ────────────────────────────────────────────────

def _line(xs):
    return np.array([[x, 0.0, 0.0] for x in xs])


def test_rest_takes_the_label_of_the_nearest_subsample_event(caplog):
    Xc = _line([0, 1, 2, 10, 11, 12])
    sub_idx = np.array([0, 4])
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = P._expand_subsample_labels(Xc, Xc[sub_idx], sub_idx,
                                         np.array([5, 7]))
    np.testing.assert_array_equal(out, [5, 5, 5, 7, 7, 7])
    assert any('Assigning 4 remaining events' in m
               for m in _msgs(caplog, logging.INFO))


def test_noise_subsample_events_are_not_used_as_neighbours():
    Xc = _line([0, 1, 9, 10])
    sub_idx = np.array([0, 3])
    out = P._expand_subsample_labels(Xc, Xc[sub_idx], sub_idx,
                                     np.array([-1, 2]))
    # Event 1 is nearest the noise event 0, but noise is not a reference.
    np.testing.assert_array_equal(out, [-1, 2, 2, 2])


def test_all_noise_subsample_leaves_the_rest_unassigned(caplog):
    Xc = _line([0, 1, 2])
    sub_idx = np.array([1])
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        out = P._expand_subsample_labels(Xc, Xc[sub_idx], sub_idx,
                                         np.array([-1]))
    np.testing.assert_array_equal(out, [-1, -1, -1])
    assert len(_msgs(caplog, logging.WARNING)) == 1
    assert 'marked as noise' in _msgs(caplog, logging.WARNING)[0]


# ── FlowSample._store_cluster_labels / _try_cluster_gpu ─────────────────────

def _sample(n=6):
    df = pd.DataFrame(np.arange(n * 3, dtype=float).reshape(n, 3) / 10,
                      columns=CH)
    s = FlowSample.from_dataframe(df, name='SampleA')
    s.fluor_channels = list(CH)
    return s


def test_store_writes_labels_where_masked_and_minus_one_elsewhere():
    s = _sample(5)
    s._store_cluster_labels(np.array([True, False, True, True, False]),
                            np.array([3, 0, 4]))
    assert s.data['cluster'].tolist() == [3, -1, 0, 4, -1]
    np.testing.assert_array_equal(s.clusters, [3, -1, 0, 4, -1])


def test_gpu_success_returns_its_labels(monkeypatch):
    monkeypatch.setattr(FlowSample, '_cluster_gpu',
                        lambda self, X, k: (np.full(len(X), k), 0.7))
    lab, q = _sample()._try_cluster_gpu(np.ones((3, 3)), 4, 's')
    np.testing.assert_array_equal(lab, [4, 4, 4])
    assert q == 0.7


def test_gpu_failure_returns_none_and_warns(monkeypatch, caplog):
    def boom(self, X, k):
        raise MemoryError('out of VRAM')
    monkeypatch.setattr(FlowSample, '_cluster_gpu', boom)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _sample()._try_cluster_gpu(np.ones((3, 3)), 4, 's') == (None,
                                                                      -1.0)
    assert _msgs(caplog, logging.WARNING) == [
        '  [!] GPU clustering failed (MemoryError: out of VRAM) — CPU '
        'fallback']


# ── FlowSample.cluster end to end (stand-ins) ───────────────────────────────

def test_gpu_result_is_kept_and_recorded(fake_pg, vram, monkeypatch):
    _, calls = fake_pg
    monkeypatch.setattr(FlowSample, '_cluster_gpu',
                        lambda self, X, k: (np.arange(len(X)) % 2, 0.8))
    s = _sample(6)
    assert s.cluster(channels=CH, reproducible=False, use_gpu=True) is s
    assert calls == []
    assert s.cluster_settings['backend'] == 'GPU'
    assert s.cluster_settings['route'] == 'louvain'
    assert s.data['cluster'].tolist() == [0, 1, 0, 1, 0, 1]


def test_gpu_failure_falls_back_to_cpu(fake_pg, vram, monkeypatch):
    _, calls = fake_pg
    def boom(self, X, k):
        raise RuntimeError('cuda')
    monkeypatch.setattr(FlowSample, '_cluster_gpu', boom)
    s = _sample(6)
    s.cluster(channels=CH, reproducible=False, use_gpu=True, k=12)
    assert len(calls) == 1 and calls[0]['k'] == 12
    assert s.cluster_settings['backend'] == 'CPU'


def test_cluster_end_to_end_labels_rows_and_logs(fake_pg, caplog):
    mod, calls = fake_pg
    mod.cluster = _seeded_cluster(
        calls, labels=lambda X: np.array([1, -1, 0][:len(X)]), q=0.33)
    df = pd.DataFrame({'FL1-A': [0.0, 1.0, np.nan, 2.0, 3.0],
                       'FL2-A': [0.0, 0.0, 0.0, np.inf, 0.0],
                       'FL3-A': [1.0, 1.0, 1.0, 1.0, 1.0]})
    s = FlowSample.from_dataframe(df, name='SampleA')
    s.fluor_channels = list(CH)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        s.cluster(channels=CH, k=5, random_state=9, resolution=0.5)
    # Rows 2 (NaN) and 3 (inf) are not clustered; noise (-1) is not a
    # cluster in the count.
    assert s.data['cluster'].tolist() == [1, -1, -1, -1, 0]
    assert calls[0]['X'].shape == (3, 3)
    assert calls[0]['resolution_parameter'] == 0.5
    assert s.cluster_settings == {
        'method': 'phenograph', 'k': 5, 'route': 'leiden', 'backend': 'CPU',
        'random_state': 9, 'resolution': 0.5}
    assert '  → 2 clusters, Q=0.330 [CPU, resolution 0.5]' in _msgs(
        caplog, logging.INFO)


def test_cluster_subsample_end_to_end(fake_pg, caplog):
    mod, calls = fake_pg
    mod.cluster = _seeded_cluster(
        calls, labels=lambda X: (X[:, 0] > 5).astype(int))
    df = pd.DataFrame({'FL1-A': [0.0, 1, 2, 3, 10, 11, 12, 13],
                       'FL2-A': [0.0] * 8, 'FL3-A': [0.0] * 8})
    s = FlowSample.from_dataframe(df, name='SampleA')
    s.fluor_channels = list(CH)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        s.cluster(channels=CH, max_events=4, random_state=0)
    assert '  Phenograph: 4 (subsample of 8) × 3, k=30 …' in _msgs(
        caplog, logging.INFO)
    assert calls[0]['X'].shape == (4, 3)
    assert s.data['cluster'].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
