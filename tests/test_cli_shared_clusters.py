"""cli._shared_cluster_labels on in-memory samples: the ONE clustering every
sample of an openflo-run shares, so a group comparison's cluster ids mean the
same population in every sample (test_cli_run.py covers the run wiring).

The clustering backends are stubbed with a rule whose answer is known (an
event's cluster is its brightest channel), so the pooling and the labelling
of events left out of the pool are checked exactly.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

import openflo.cli as cli
from openflo.pipeline import FlowSample

CH = ['A', 'B', 'C']


def _sample(name, X, chans=CH, labels=None):
    s = FlowSample.__new__(FlowSample)
    s.name = name
    s.data = pd.DataFrame(np.asarray(X, dtype=float), columns=pd.Index(chans))
    s.fluor_channels = list(chans)
    s.scatter_channels = []
    s.channel_labels = {c: c for c in chans} | dict(labels or {})
    s.clusters = None
    return s


def _blobs(n, seed, d=3):
    return np.random.default_rng(seed).normal(0.0, 1.0, (n, d))


def _brightest(df):
    """The stub clustering: an event's cluster is its brightest channel."""
    return np.argmax(df.to_numpy(dtype=float), axis=1)


@pytest.fixture
def pooled(monkeypatch):
    """Stub PhenoGraph, Leiden and FlowSOM with _brightest; record every
    frame they are asked to cluster."""
    seen = []

    def stub(col):
        def run(self, **kw):
            seen.append(self.data.copy())
            self.data[col] = _brightest(self.data[self.fluor_channels])
            return self
        return run
    monkeypatch.setattr(FlowSample, 'cluster', stub('cluster'))
    monkeypatch.setattr(FlowSample, 'run_leiden', stub('leiden'))
    monkeypatch.setattr(FlowSample, 'run_flowsom', stub('flowsom_meta'))
    return seen


def test_unpooled_events_take_the_nearest_pooled_label_on_every_core(
        pooled, monkeypatch):
    """Events left out of the pool were labelled by a single-threaded
    scikit-learn KD-tree in the main process: 3 x 300k events with a 60k
    pool took 144 s for that step alone. A scipy KDTree queried on every
    core took 10 s on the same data. The answer must still be each event's
    exact nearest pooled event."""
    import scipy.spatial
    workers = []

    class Spy(scipy.spatial.KDTree):
        def query(self, x, *a, **kw):
            workers.append(kw.get('workers'))
            return super().query(x, *a, **kw)
    monkeypatch.setattr(scipy.spatial, 'KDTree', Spy)
    samples = [_sample(f's{i}', _blobs(400, i)) for i in range(3)]
    labels, *_ = cli._shared_cluster_labels(
        samples, cli._cluster_settings(15, 1, 150, 1.0, 42))
    pool = pooled[-1][CH].to_numpy()
    assert len(pool) == 150
    plab = _brightest(pooled[-1][CH])
    for s, lab in zip(samples, labels, strict=True):
        X = s.data[CH].to_numpy()
        nearest = ((X[:, None, :] - pool[None]) ** 2).sum(-1).argmin(1)
        assert (lab == plab[nearest]).all()
    assert workers and set(workers) == {-1}


def test_pool_takes_an_equal_share_from_every_sample(pooled):
    """The pool was a uniform draw over all events, so a sample's share was
    its share of the events: two 400-event samples took 96 of a
    100-event pool and two 20-event samples 3 and 1 events,
    too few for a population only they hold to form a cluster. Each sample
    now gets an equal share; a sample smaller than its share gives all of
    its events and the remainder is split among the rest: 30/30/20/20."""
    sizes = [400, 400, 20, 20]
    samples = []
    for i, n in enumerate(sizes):
        X = _blobs(n, i)
        X[:, 0] = 1000 * i            # channel A tells which sample it is
        samples.append(_sample(f's{i}', X))
    cli._shared_cluster_labels(samples, cli._cluster_settings(
        15, 1, 100, 1.0, 42))
    origin = (pooled[-1]['A'].to_numpy() // 1000).astype(int)
    assert np.bincount(origin, minlength=4).tolist() == [30, 30, 20, 20]


@pytest.mark.parametrize('sizes, cap, quota', [
    ([10, 10, 10], 30, [10, 10, 10]),      # everything fits: every event
    ([10, 10, 10], 7, [3, 2, 2]),          # remainder to the first samples
    ([1000, 5, 0, 60], 100, [48, 5, 0, 47]),
])
def test_equal_quota(sizes, cap, quota):
    assert cli._equal_quota(sizes, cap) == quota
    assert sum(quota) == min(cap, sum(sizes))


def test_channels_align_by_label_with_area_height_kept_apart():
    """CD901b and CD902 swap fluors between the panels; CD901b-H pairs with
    CD901b-H, not CD901b-A. FL5-A is labelled in one panel only, so it is
    matched by detector, and the run says so."""
    a = _sample('a', _blobs(5, 0, 4), ['FL1-A', 'FL2-A', 'FL5-A', 'FL1-H'],
                {'FL1-A': 'CD901b', 'FL2-A': 'CD902', 'FL1-H': 'CD901b'})
    b = _sample('b', _blobs(5, 1, 4), ['FL1-A', 'FL2-A', 'FL5-A', 'FL2-H'],
                {'FL2-A': 'CD901b', 'FL1-A': 'CD902', 'FL5-A': 'CD903',
                 'FL2-H': 'CD901b'})
    channels, columns, notes, _ = cli._align_shared_channels([a, b],
                                                            ['a', 'b'])
    assert channels == ['FL1-A', 'FL2-A', 'FL5-A', 'FL1-H']
    assert columns[1] == ['FL2-A', 'FL1-A', 'FL5-A', 'FL2-H']
    assert notes == ['CD901b: FL1-A in a; FL2-A in b',
                     'CD902: FL2-A in a; FL1-A in b',
                     'CD901b: FL1-H in a; FL2-H in b',
                     'FL5-A: matched by detector; no label in a; CD903 in b']


def test_channels_never_reuse_a_detector_or_pair_two_labels():
    """b's B already holds CD901b, so a's unlabelled B cannot also take it;
    and CD3 / CD4 on one detector are two markers, never one column."""
    a = _sample('a', _blobs(5, 0), ['A', 'B', 'C'], {'A': 'CD901b', 'C': 'CD3'})
    b = _sample('b', _blobs(5, 1), ['A', 'B', 'C'], {'B': 'CD901b', 'C': 'CD4'})
    channels, columns, notes, _ = cli._align_shared_channels([a, b],
                                                            ['a', 'b'])
    assert (channels, columns) == (['A'], [['A'], ['B']])
    assert notes == ['CD901b: A in a; B in b',
                     'C: different markers, not pooled; CD3 in a; CD4 in b',
                     'B: no matching channel in b',
                     'A: no matching channel in a']


def test_labels_ignore_punctuation_and_a_trailing_fluor_name():
    """'Ly-6G' / 'Ly6G', 'Sca-1' / 'Sca1' and 'CD901b FL1' / 'CD901b' on one
    detector were 'different markers, not pooled': folding only case and
    spacing dropped a marker over punctuation in 207 of 4000 fuzzed panels,
    and could leave no column, and so no comparison, at all."""
    a = _sample('a', _blobs(5, 0), ['FL2-A', 'FL5-A', 'FL1-A'],
                {'FL2-A': 'Ly-6G', 'FL5-A': 'Sca-1', 'FL1-A': 'CD901b FL1'})
    b = _sample('b', _blobs(5, 1), ['FL2-A', 'FL5-A', 'FL1-A'],
                {'FL2-A': 'Ly6G', 'FL5-A': 'Sca1', 'FL1-A': 'CD901b'})
    channels, columns, notes, display = cli._align_shared_channels(
        [a, b], ['a', 'b'])
    assert channels == ['FL2-A', 'FL5-A', 'FL1-A']
    assert columns[1] == channels and notes == []
    assert display == {'FL2-A': 'Ly-6G', 'FL5-A': 'Sca-1',
                       'FL1-A': 'CD901b FL1'}


def test_greek_letters_keep_markers_apart():
    """The label fold kept only [a-z0-9], so Greek letters vanished:
    'TCRβ' and 'TCRγδ' both became 'tcr', 'CD8α' and 'CD8β' both 'cd8',
    'IFN-γ' and 'IFN-α' one marker. With the panels swapped between two
    samples, each detector was paired with the OTHER marker, silently.
    'Ly-6G' / 'Ly6G' must still match, and 'CD8α FL1' must keep its α
    when the fluor word is dropped."""
    dets = ['FL4-A', 'FL5-A', 'FL2-A', 'FL1-A', 'FL6-A', 'FL7-A',
            'FL3-A']
    a = _sample('a', _blobs(5, 0, 7), dets,
                {'FL4-A': 'TCRβ', 'FL5-A': 'TCRγδ', 'FL2-A': 'CD8α',
                 'FL1-A': 'CD8β FL1', 'FL6-A': 'IFN-γ',
                 'FL7-A': 'IFN-α', 'FL3-A': 'Ly-6G'})
    b = _sample('b', _blobs(5, 1, 7), dets,
                {'FL4-A': 'TCRγδ', 'FL5-A': 'TCRβ', 'FL2-A': 'CD8β',
                 'FL1-A': 'CD8α', 'FL6-A': 'IFN-α', 'FL7-A': 'IFN-γ',
                 'FL3-A': 'Ly6G'})
    channels, columns, _, display = cli._align_shared_channels(
        [a, b], ['a', 'b'])
    assert channels == dets
    assert columns[1] == ['FL5-A', 'FL4-A', 'FL1-A', 'FL2-A', 'FL7-A',
                          'FL6-A', 'FL3-A']
    assert len(set(display.values())) == len(dets)


def test_a_label_that_is_the_detectors_fluor_is_no_label():
    """$PnS 'FL6' on FL6-A names the fluor, not a marker. Read as a
    marker it clashed with b's 'CD3' on the same detector, and the channel
    was dropped."""
    a = _sample('a', _blobs(5, 0, 2), ['FL6-A', 'FL5-A'],
                {'FL6-A': 'FL6', 'FL5-A': 'CD4'})
    b = _sample('b', _blobs(5, 1, 2), ['FL6-A', 'FL5-A'],
                {'FL6-A': 'CD3', 'FL5-A': 'CD4'})
    channels, _, notes, display = cli._align_shared_channels(
        [a, b], ['a', 'b'])
    assert channels == ['FL6-A', 'FL5-A']
    assert notes == ['FL6-A: matched by detector; no label in a; CD3 in b']
    assert display['FL6-A'] == 'CD3'


def test_duplicate_labels_pair_by_detector_and_labels_ignore_case():
    """Both panels carry 'Dump' on FL1-A and FL6-A. One label kept one
    detector: b lists FL6-A first, so a's FL1-A was paired with b's
    FL6-A and the other two were dropped. 'CD901b' and 'CD901B ' (or 'CD3'
    and 'cd3') on one detector were two different markers."""
    a = _sample('a', _blobs(5, 0, 4), ['FL1-A', 'FL6-A', 'FL5-A', 'FL2-A'],
                {'FL1-A': 'Dump', 'FL6-A': 'Dump', 'FL5-A': 'CD901b',
                 'FL2-A': 'CD3'})
    b = _sample('b', _blobs(5, 1, 4), ['FL6-A', 'FL1-A', 'FL5-A', 'FL2-A'],
                {'FL1-A': 'Dump', 'FL6-A': 'Dump', 'FL5-A': 'CD901B ',
                 'FL2-A': 'cd3'})
    out = cli._align_shared_channels([a, b], ['a', 'b'])
    channels, columns, notes = out[:3]
    assert channels == ['FL1-A', 'FL6-A', 'FL5-A', 'FL2-A']
    assert columns[1] == channels
    assert notes == []
    assert out[3] == {'FL1-A': 'Dump (FL1-A)', 'FL6-A': 'Dump (FL6-A)',
                      'FL5-A': 'CD901b', 'FL2-A': 'CD3'}


@pytest.mark.parametrize('order', list(itertools.permutations(range(3))))
def test_two_labels_on_one_detector_are_never_pooled_in_any_order(order):
    """s1 has CD4 on FL4-A, s2 CD8, s0 no label there. With s0 first the
    detector fallback pooled CD4 and CD8 in one column; with s1 first FL4-A
    was left out. The channel set depended on the order of the groups."""
    spec = [('s0', {}), ('s1', {'FL4-A': 'CD4'}), ('s2', {'FL4-A': 'CD8'})]
    every = [_sample(nm, _blobs(5, i, 2), ['FL4-A', 'FL5-A'], lab)
             for i, (nm, lab) in enumerate(spec)]
    samples = [every[i] for i in order]
    channels, _, notes = cli._align_shared_channels(
        samples, [s.name for s in samples])[:3]
    assert channels == ['FL5-A']
    conflict = [n for n in notes if n.startswith('FL4-A')]
    assert len(conflict) == 1
    assert all(w in conflict[0] for w in ('CD4 in s1', 'CD8 in s2',
                                          'no label in s0'))


@pytest.mark.parametrize('order', list(itertools.permutations(range(3))))
def test_unlabelled_detector_is_not_its_label_when_the_label_is_elsewhere(
        order):
    """s0 carries CD4 on FL5-A, so its unlabelled FL4-A is some other
    marker; s1's FL4-A is CD4. Matching FL4-A by detector would pool CD4
    with that unknown marker."""
    spec = [('s0', ['FL4-A', 'FL5-A', 'FL2-A'], {'FL5-A': 'CD4'}),
            ('s1', ['FL4-A', 'FL2-A'], {'FL4-A': 'CD4'}),
            ('s2', ['FL4-A', 'FL2-A'], {})]
    every = [_sample(nm, _blobs(5, i, len(ch)), ch, lab)
             for i, (nm, ch, lab) in enumerate(spec)]
    samples = [every[i] for i in order]
    channels, columns, _ = cli._align_shared_channels(
        samples, [s.name for s in samples])[:3]
    assert channels == ['FL2-A']
    assert all(c == ['FL2-A'] for c in columns)


def test_flowsom_labels_every_event_through_the_trained_som(monkeypatch):
    """An event left out of the FlowSOM pool took the label of its nearest
    POOLED EVENT, not FlowSOM's own answer: the metacluster of its
    best-matching SOM node. Near a metacluster boundary the two differ.
    Every event now goes through the SOM the pool trained (real FlowSOM,
    small data)."""
    from openflo.pipeline import _som_assign, _som_metacluster
    trained = []
    real = FlowSample.run_flowsom

    def spy(self, **kw):
        real(self, **kw)
        trained.append(self.flowsom_result)
        return self
    monkeypatch.setattr(FlowSample, 'run_flowsom', spy)
    samples = [_sample(f's{i}', _blobs(3000, i)) for i in range(3)]
    labels, *_ = cli._shared_cluster_labels(samples, cli._cluster_settings(
        15, 1, 600, 1.0, 42, {'method': 'flowsom', 'n_metaclusters': 6}))
    W = trained[-1]['weights']
    meta = _som_metacluster(W, 6)
    for s, lab in zip(samples, labels, strict=True):
        want = meta[_som_assign(s.data[CH].to_numpy(), W)]
        assert (lab == want).all()


@pytest.mark.parametrize('pool, n_jobs', [(50_000, -1), (1_000_000, 1)])
def test_every_core_only_when_free_ram_covers_the_pool(monkeypatch, pool,
                                                       n_jobs):
    """Each PhenoGraph Jaccard child takes about 163 MB + 240 B per pooled
    event (starmap sends it the kNN index); the gate assumed a flat
    0.25 GB a core, so with 10 GB free and 24 cores it let a 1M-event pool
    start children needing about 12 GB."""
    import os
    import types

    import psutil
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: types.SimpleNamespace(
        available=10 * 1024 ** 3, total=64 * 1024 ** 3, percent=0.0))
    monkeypatch.setattr(os, 'cpu_count', lambda: 24)
    assert cli._shared_n_jobs(1, pool) == n_jobs


@pytest.mark.parametrize('method, cap', [('leiden', 200_000),
                                         ('flowsom', 50_000)])
def test_pool_stops_at_the_methods_own_cap(pooled, method, cap):
    """With no --max-events the pool held as many events as the largest
    sample. Leiden partitions at most 200,000 events (FlowSOM trains on at
    most 50,000) and 1-NN-assigns the rest of whatever it is given, so a
    bigger pool only added a second 1-NN pass, the slow step above."""
    samples = [_sample(f's{i}', _blobs(cap + 50_000, i)) for i in range(2)]
    cli._shared_cluster_labels(samples, cli._cluster_settings(
        15, 1, None, 1.0, 42, {'method': method}))
    assert len(pooled[-1]) == cap
