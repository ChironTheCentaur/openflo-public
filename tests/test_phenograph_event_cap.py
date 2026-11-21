"""PhenoGraph's default event cap, the warning before a graph clustering of
more events than that, and the Cluster dialog's Cancel.

PhenoGraph used to cluster every event of a sample: about a million events
needed several GB on one core (about 9 GB at k=30, more at a higher k)
and froze an 8 GB laptop while it swapped.
It now caps at 200,000 like Leiden (the rest take their nearest clustered
event's cluster), warns before a run above that, and can be canceled.

Generated data and generic channel names only. PhenoGraph itself is
stubbed where a test only needs to see how many events reach it.
"""
import tkinter.messagebox

import numpy as np
import pandas as pd
import pytest

from openflo import cluster_caps
from openflo import flow_sample as fs
from openflo.pipeline import FlowSample

CAP = 200_000
CH = ['FL1-A', 'FL2-A', 'FL3-A']


def _two_blobs(n, seed=0):
    """`n` events in two far-apart blobs; blob 1 has FL1-A > 0."""
    rng = np.random.default_rng(seed)
    side = np.where(np.arange(n) % 2 == 0, -20.0, 20.0)
    X = rng.normal(0.0, 1.0, (n, 3))
    X[:, 0] += side
    return pd.DataFrame(X, columns=CH)


@pytest.fixture
def fake_phenograph(monkeypatch):
    """Replace the PhenoGraph call with one that labels by FL1-A's sign and
    records how many events it was given."""
    seen = []

    def fake(X_cluster, k, n_jobs, reproducible, random_state, resolution,
             shape):
        seen.append(len(X_cluster))
        return (X_cluster[:, 0] > 0).astype(int), 0.5, 'leiden'
    monkeypatch.setattr(fs, '_run_phenograph_cpu', fake)
    return seen


# ── the library default ────────────────────────────────────────────────────

def test_cluster_default_caps_at_200k_and_assigns_the_rest(fake_phenograph):
    df = _two_blobs(CAP + 600)
    s = FlowSample.from_dataframe(df, name='big')
    s.cluster(channels=CH, use_gpu=False)
    assert fake_phenograph == [CAP]
    lab = s.data['cluster'].to_numpy()
    # Every event has a cluster, the unclustered ones by their nearest
    # clustered event: the blob it sits in.
    assert (lab >= 0).all()
    assert (lab == (df['FL1-A'].to_numpy() > 0)).all()


def test_cluster_none_still_clusters_every_event(fake_phenograph):
    s = FlowSample.from_dataframe(_two_blobs(CAP + 600), name='big')
    s.cluster(channels=CH, use_gpu=False, max_events=None)
    assert fake_phenograph == [CAP + 600]


def test_cluster_below_the_cap_is_unchanged(fake_phenograph):
    s = FlowSample.from_dataframe(_two_blobs(5_000), name='small')
    s.cluster(channels=CH, use_gpu=False)
    assert fake_phenograph == [5_000]


# ── checkpoint (Cancel) inside FlowSample.cluster ───────────────────────────

class _Stop(Exception):
    pass


@pytest.mark.parametrize('stop_at', [1, 2])
def test_checkpoint_stops_before_anything_is_written(fake_phenograph,
                                                     stop_at):
    s = FlowSample.from_dataframe(_two_blobs(1_000), name='s')
    calls = []

    def checkpoint():
        calls.append(len(fake_phenograph))
        if len(calls) == stop_at:
            raise _Stop()
    with pytest.raises(_Stop):
        s.cluster(channels=CH, use_gpu=False, checkpoint=checkpoint)
    assert 'cluster' not in s.data.columns
    assert getattr(s, 'cluster_settings', None) is None
    # The first stop comes before PhenoGraph runs, the second after it.
    assert fake_phenograph == ([] if stop_at == 1 else [1_000])


def test_checkpoint_that_never_raises_changes_nothing(fake_phenograph):
    s = FlowSample.from_dataframe(_two_blobs(1_000), name='s')
    calls = []
    s.cluster(channels=CH, use_gpu=False,
              checkpoint=lambda: calls.append(1))
    assert len(calls) == 2 and 'cluster' in s.data.columns


# ── openflo-run: --max-events 0 is each method's own cap ────────────────────

def test_cli_phenograph_without_max_events_uses_the_cap():
    from openflo.shared_clustering import cluster_by_method
    got = {}

    class S:
        name = 's'
        data = pd.DataFrame({'a': np.zeros(100)})

        def cluster(self, **kw):
            got.update(kw)
    cluster_by_method(S(), {'method': 'phenograph', 'k': 30, 'n_jobs': 1,
                            'max_events': None})
    assert got['max_events'] == CAP
    cluster_by_method(S(), {'method': 'phenograph', 'k': 30, 'n_jobs': 1,
                            'max_events': 500_000})
    assert got['max_events'] == 500_000


def test_shared_pool_for_phenograph_is_capped_like_leiden():
    from openflo.shared_clustering import shared_pool_cap
    for method in ('phenograph', 'leiden'):
        assert shared_pool_cap([900_000, 10], {'method': method}) == CAP
    assert shared_pool_cap([900_000], {'method': 'flowsom'}) == 50_000
    assert shared_pool_cap([900_000], {'method': 'phenograph',
                                       'max_events': 700_000}) == 700_000


# ── the warning text and memory estimate ────────────────────────────────────

def test_memory_estimate_grows_with_events():
    small = cluster_caps.clustering_memory_gb('phenograph', 100_000)
    big = cluster_caps.clustering_memory_gb('phenograph', 1_000_000)
    assert 0 < small < big
    dense = cluster_caps.clustering_memory_gb('leiden', CAP, fast_graph=False)
    fast = cluster_caps.clustering_memory_gb('leiden', CAP)
    assert fast < 4 < 15 < dense      # measured 2.38 GB fast, ~21 GB dense
    # More neighbors need more memory, for every method.
    for method in ('phenograph', 'leiden'):
        gb = [cluster_caps.clustering_memory_gb(method, CAP, k)
              for k in (5, 30, 60, 100, 200)]
        assert gb == sorted(gb) and gb[-1] > 2.5 * gb[1]


def test_memory_estimate_covers_every_measured_point():
    """The estimate is at or above every measured peak (12 channels, k up
    to 200), and not wildly above it."""
    assert len(cluster_caps.MEMORY_MEASURED) >= 25
    for key, n, k, measured in cluster_caps.MEMORY_MEASURED:
        method = 'phenograph' if key == 'phenograph' else 'leiden'
        est = cluster_caps.clustering_memory_gb(
            method, n, k, fast_graph=key != 'leiden-dense')
        assert 1.0 <= est / measured <= 1.8, (key, n, k, measured, est)


def test_memory_estimate_pins_the_review_cases():
    """The round-1 review's measurements: PhenoGraph 100k at k=100 took
    2.71 GB (the k-blind estimate said 1.07); dense Leiden 50k at k=60 ran
    out of 11.9 GB."""
    est = cluster_caps.clustering_memory_gb
    assert 2.71 <= est('phenograph', 100_000, 100) < 3.5
    assert est('leiden', 50_000, 60, fast_graph=False) > 11.9
    # Default PhenoGraph at k=100 on a laptop with 5 GB free is asked about.
    assert cluster_caps.large_run_warning('phenograph', CAP, 5.0,
                                          k=100) is not None
    assert cluster_caps.large_run_warning('phenograph', CAP, 5.0,
                                          k=30) is None
    assert cluster_caps.clustering_memory_gb('flowsom', 10**6) is None
    assert cluster_caps.clustering_memory_gb('phenograph', 0) is None


def test_warning_only_above_a_graph_methods_cap():
    warn = cluster_caps.large_run_warning
    assert warn('phenograph', CAP, 64.0) is None
    assert warn('leiden', CAP, 64.0) is None
    assert warn('flowsom', 10**7, 1.0) is None
    text = warn('phenograph', CAP + 1, 64.0)
    gb = cluster_caps.clustering_memory_gb('phenograph', CAP + 1)
    assert text is not None and f'{gb:.1f} GB' in text
    assert '64.0 GB is free' in text and 'more memory than is free' not in text
    assert 'Leiden' in (warn('leiden', CAP + 1, None) or '')


def test_warning_says_when_it_needs_more_than_is_free():
    text = cluster_caps.large_run_warning('phenograph', 1_000_000, 2.0)
    assert text is not None and 'more memory than is free' in text
    assert '2.0 GB is free' in text


def test_warning_within_the_cap_when_free_memory_is_short():
    warn = cluster_caps.large_run_warning
    # Leiden's own default, 200,000 events, fits in 8 GB on its default
    # fast graph, but the dense graph needs far more.
    assert warn('leiden', CAP, 8.0) is None
    text = warn('leiden', CAP, 8.0, fast_graph=False)
    assert text is not None and 'more memory than is free' in text
    assert 'more than its default' not in text and 'FlowSOM' in text
    need = cluster_caps.clustering_memory_gb('phenograph', CAP)
    assert warn('phenograph', CAP, need - 0.01) is not None
    assert warn('phenograph', CAP, need + 0.01) is None
    assert warn('phenograph', CAP, None) is None     # free memory unknown


# ── the Cluster dialog run ─────────────────────────────────────────────────

def test_method_max_events():
    from openflo.editor_analysis import method_max_events as mme
    assert mme('phenograph', None, False, 10**6) == CAP
    assert mme('leiden', None, False, 10**6) == CAP
    assert mme('flowsom', None, False, 10**6) == 50_000
    assert mme('phenograph', None, True, 10**6) is None
    assert mme('leiden', None, True, 10**6) is None
    assert mme('flowsom', None, True, 10**6) == 10**6
    assert mme('phenograph', 1_500, False, 10**6) == 1_500
    assert mme('phenograph', 1_500, True, 10**6) == 1_500


def _editor():
    from tests.test_cluster_dialog import _editor_or_skip
    return _editor_or_skip()


def _spy_cluster(s, record, before=None):
    """Record each FlowSample.cluster call's keywords on `s`, then run the
    real call (on PhenoGraph stubbed by fake_phenograph)."""
    real = s.cluster

    def spy(**kw):
        record.append(kw)
        if before is not None:
            before()
        return real(**kw)
    s.cluster = spy


@pytest.fixture
def asks(monkeypatch):
    """Answers for the large-run warning; records each question asked."""
    state = {'answer': False, 'asked': [], 'free': 64.0}

    def ask(title, text, **kw):
        state['asked'].append((title, text))
        return state['answer']
    monkeypatch.setattr(tkinter.messagebox, 'askyesno', ask)
    monkeypatch.setattr(cluster_caps, 'available_memory_gb',
                        lambda: state['free'])
    return state


def _run(ed, downsample, method='phenograph', custom_n=None, **kw):
    ed._run_clustering(method=method, all_samples=True, k=15, grid=4,
                       n_meta=3, embedding='none', downsample=downsample,
                       custom_n=custom_n, **kw)


def test_default_run_caps_phenograph_without_asking(fake_phenograph, asks):
    from tests.test_cluster_dialog import _add
    root, ed = _editor()
    try:
        s = _add(ed, _two_blobs(CAP + 600), 'big')
        calls = []
        _spy_cluster(s, calls)
        _run(ed, 'Default')
        assert asks['asked'] == []
        assert calls[0]['max_events'] == CAP and fake_phenograph == [CAP]
        assert (s.data['cluster'] >= 0).all()
        assert not ed._clustering_busy
    finally:
        ed.destroy()
        root.destroy()


@pytest.mark.parametrize('mode,custom_n', [('All events', None),
                                           ('Custom…', CAP + 100)])
def test_large_run_asks_first_and_no_stops_it(fake_phenograph, asks, mode,
                                              custom_n):
    from tests.test_cluster_dialog import _add
    root, ed = _editor()
    try:
        s = _add(ed, _two_blobs(CAP + 600), 'big')
        calls = []
        _spy_cluster(s, calls)
        _run(ed, mode, custom_n=custom_n)
        assert len(asks['asked']) == 1
        title, text = asks['asked'][0]
        assert title == 'Large clustering run' and ' GB of memory' in text
        n = CAP + 600 if custom_n is None else custom_n
        assert f'{n:,} events' in text
        assert calls == [] and 'cluster' not in s.data.columns
        assert not getattr(ed, '_clustering_busy', False)
        assert ed.status_var.get().startswith('Clustering not started')
        # Yes runs it on every event (or the number typed).
        asks['answer'] = True
        _run(ed, mode, custom_n=custom_n)
        assert calls[-1]['max_events'] == custom_n
        assert fake_phenograph == [n]
    finally:
        ed.destroy()
        root.destroy()


def test_no_question_when_the_events_are_within_the_cap(fake_phenograph,
                                                        asks):
    from tests.test_cluster_dialog import _add
    root, ed = _editor()
    try:
        _add(ed, _two_blobs(3_000), 'small')
        _run(ed, 'All events')
        _run(ed, 'Custom…', custom_n=CAP + 100)   # more than the sample has
        assert asks['asked'] == [] and fake_phenograph == [3_000, 3_000]
    finally:
        ed.destroy()
        root.destroy()


def test_default_leiden_asks_when_free_memory_is_short(asks):
    from tests.test_cluster_dialog import _add
    root, ed = _editor()
    try:
        s = _add(ed, _two_blobs(CAP + 600), 'big')
        got = []
        s.run_leiden = lambda **kw: got.append(kw)
        asks['free'] = 8.0
        # The default fast graph fits in 8 GB: no question.
        _run(ed, 'Default', method='leiden')
        assert asks['asked'] == [] and got[-1]['fast_graph'] is True
        # The dense graph does not.
        _run(ed, 'Default', method='leiden', fast_graph=False)
        assert len(asks['asked']) == 1 and len(got) == 1
        assert f'{CAP:,} events' in asks['asked'][0][1]
        # Nor does the fast graph in 1 GB.
        asks['free'] = 1.0
        _run(ed, 'Default', method='leiden')
        assert len(asks['asked']) == 2 and len(got) == 1
        asks['free'] = 64.0
        _run(ed, 'Default', method='leiden', fast_graph=False)
        assert len(asks['asked']) == 2 and got[-1]['max_events'] == CAP
        assert got[-1]['fast_graph'] is False
    finally:
        ed.destroy()
        root.destroy()


def test_flowsom_all_events_trains_on_every_event_without_asking(asks):
    from tests.test_cluster_dialog import _add
    root, ed = _editor()
    try:
        s = _add(ed, _two_blobs(CAP + 600), 'big')
        got = []
        s.run_flowsom = lambda **kw: got.append(kw)
        _run(ed, 'All events', method='flowsom')
        assert asks['asked'] == [] and got[0]['max_events'] == CAP + 600
        _run(ed, 'Default', method='flowsom')
        assert got[1]['max_events'] == 50_000
    finally:
        ed.destroy()
        root.destroy()


def test_cancel_between_samples_keeps_the_finished_ones(fake_phenograph):
    from tests.test_cluster_dialog import _add
    root, ed = _editor()
    try:
        a = _add(ed, _two_blobs(1_000, seed=1), 'a')
        b = _add(ed, _two_blobs(1_000, seed=2), 'b')
        shown = []
        calls_a, calls_b = [], []

        def press_cancel():
            shown.append(ed._cluster_cancel_btn.winfo_manager())
            ed._cancel_clustering()
        _spy_cluster(a, calls_a)
        _spy_cluster(b, calls_b)
        # Cancel is pressed while sample a clusters, after its last
        # checkpoint: a finishes, b never starts.
        real_store = a._store_cluster_labels

        def store(*args):
            press_cancel()
            return real_store(*args)
        a._store_cluster_labels = store
        _run(ed, 'Default')
        assert shown == ['grid']                     # Cancel was on show
        assert 'cluster' in a.data.columns and calls_b == []
        assert 'cluster' not in b.data.columns
        status = ed.status_var.get()
        assert status.startswith('Clustering canceled after 1 of 2'), status
        assert not ed._clustering_busy
        assert ed._cluster_cancel_btn.winfo_manager() == ''
        assert ed._audited and ed._audited[-1][1]['samples'] == ['a']
    finally:
        ed.destroy()
        root.destroy()


def test_cancel_during_phenograph_writes_nothing(fake_phenograph):
    from tests.test_cluster_dialog import _add
    root, ed = _editor()
    try:
        s = _add(ed, _two_blobs(1_000), 's')
        calls = []
        _spy_cluster(s, calls, before=ed._cancel_clustering)
        _run(ed, 'Default')
        assert len(calls) == 1 and fake_phenograph == []
        assert 'cluster' not in s.data.columns
        assert ed.status_var.get() == ('Clustering canceled. No sample was '
                                       'changed.')
        assert not ed._clustering_busy and not ed._audited
        assert ed._cluster_cancel_btn.winfo_manager() == ''
    finally:
        ed.destroy()
        root.destroy()


def test_cancel_when_no_run_is_going_does_nothing():
    root, ed = _editor()
    try:
        import threading
        # The flag a finished run left behind is not set by a late Cancel.
        left = ed._clustering_cancel = threading.Event()
        ed._clustering_busy = False
        ed.status_var.set('idle')
        ed._cancel_clustering()
        assert ed.status_var.get() == 'idle' and not left.is_set()
    finally:
        ed.destroy()
        root.destroy()


def test_clustering_runs_off_the_tk_thread(fake_phenograph, monkeypatch):
    """The run goes through run_async (a worker thread), so the window keeps
    answering while it works."""
    from tests.test_cluster_dialog import _add
    root, ed = _editor()
    try:
        _add(ed, _two_blobs(1_000), 's')
        got = []
        ed.run_async = lambda work, **kw: got.append((work, kw))
        _run(ed, 'Default')
        assert len(got) == 1 and fake_phenograph == []
        assert ed._clustering_busy
        assert ed._cluster_cancel_btn.winfo_manager() == 'grid'
    finally:
        ed.destroy()
        root.destroy()


# ── the Workspace (Pipeline) run ───────────────────────────────────────────

@pytest.mark.parametrize('method', ['phenograph', 'leiden'])
def test_workspace_graph_run_caps_even_at_max_events_0(method):
    """'max ev' 0 hands the whole unit on; PhenoGraph and Leiden still
    partition at most their cap, and a smaller unit is clustered whole."""
    from openflo.workspace_run import _cluster_sample
    got = []

    class S:
        def cluster(self, **kw):
            got.append(kw)

        def run_leiden(self, **kw):
            got.append(kw)
    for n in (CAP + 600, 3_000):
        _cluster_sample(S(), method, CH, {'max_events': 0}, 30, n, 42)
    assert [kw['max_events'] for kw in got] == [CAP, 3_000]
    if method == 'leiden':
        assert all(kw['fast_graph'] is True for kw in got)


def test_workspace_leiden_keeps_a_saved_dense_graph():
    from openflo.workspace_run import _cluster_sample
    got = []

    class S:
        def run_leiden(self, **kw):
            got.append(kw)
    _cluster_sample(S(), 'leiden', CH, {'fast_graph': False}, 30, 3_000, 42)
    assert got[0]['fast_graph'] is False


# ── Cancel and the question: the edges (review round 1) ────────────────────

def _store_then(s, fn):
    """Call `fn` just before `s` writes its labels (after its last
    checkpoint), then write them."""
    real = s._store_cluster_labels

    def store(*args):
        fn()
        return real(*args)
    s._store_cluster_labels = store


def test_cancel_skips_the_embedding_and_keeps_the_clusters(fake_phenograph):
    from tests.test_cluster_dialog import _add
    root, ed = _editor()
    try:
        s = _add(ed, _two_blobs(1_000), 's')
        embedded = []
        s.run_umap = lambda **kw: embedded.append(kw)
        _store_then(s, ed._cancel_clustering)
        ed._run_clustering(method='phenograph', all_samples=False, k=15,
                           grid=4, n_meta=3, embedding='UMAP',
                           downsample='Default')
        assert embedded == [] and 'cluster' in s.data.columns
        assert ed.status_var.get().startswith(
            'Clustering canceled after 1 of 1')
    finally:
        ed.destroy()
        root.destroy()


def test_a_capped_run_embeds_on_the_same_cap(fake_phenograph):
    from tests.test_cluster_dialog import _add
    root, ed = _editor()
    try:
        s = _add(ed, _two_blobs(3_000), 's')
        seen = []

        def umap(**kw):
            seen.append(kw)
            s.data['UMAP1'] = 0.0
            s.data['UMAP2'] = 0.0
        s.run_umap = umap
        ed._run_clustering(method='phenograph', all_samples=False, k=15,
                           grid=4, n_meta=3, embedding='UMAP',
                           downsample='Custom…', custom_n=1_500)
        assert seen[0]['sample_n'] == 1_500
        ed._run_clustering(method='phenograph', all_samples=False, k=15,
                           grid=4, n_meta=3, embedding='UMAP',
                           downsample='Default')
        assert 'sample_n' not in seen[1]
    finally:
        ed.destroy()
        root.destroy()


def test_pressing_cancel_disables_it_and_says_so_then_the_next_run_has_it_back(
        fake_phenograph):
    from tests.test_cluster_dialog import _add
    root, ed = _editor()
    try:
        s = _add(ed, _two_blobs(1_000), 's')
        btn = ed._cluster_cancel_btn
        seen = []

        def press():
            ed._cancel_clustering()
            seen.append((str(btn.cget('state')), ed.status_var.get()))
        _store_then(s, press)
        _run(ed, 'Default')
        assert seen == [('disabled', 'Canceling clustering: it stops after '
                         'the step now running…')]
        # The next run starts with Cancel on show and pressable again.
        s._store_cluster_labels = type(s)._store_cluster_labels.__get__(s)
        states = []
        calls = []
        _spy_cluster(s, calls, before=lambda: states.append(
            (btn.winfo_manager(), str(btn.cget('state')))))
        _run(ed, 'Default')
        assert states == [('grid', 'normal')]
        assert ed.status_var.get().startswith('phenograph')
    finally:
        ed.destroy()
        root.destroy()


def test_a_failed_run_hides_cancel_and_ends_busy(fake_phenograph):
    from tests.test_cluster_dialog import _add
    root, ed = _editor()
    try:
        s = _add(ed, _two_blobs(1_000), 's')

        def boom(**kw):
            raise RuntimeError('planted failure')
        s.cluster = boom
        _run(ed, 'Default')
        assert ed.status_var.get() == 'Clustering failed: planted failure'
        assert not ed._clustering_busy
        assert ed._cluster_cancel_btn.winfo_manager() == ''
    finally:
        ed.destroy()
        root.destroy()


def test_the_question_counts_the_biggest_sample(fake_phenograph, asks):
    from tests.test_cluster_dialog import _add
    root, ed = _editor()
    try:
        _add(ed, _two_blobs(3_000, seed=1), 'small')    # active, first
        _add(ed, _two_blobs(CAP + 600, seed=2), 'big')
        ed._set_active_sample('small')
        _run(ed, 'All events')
        assert len(asks['asked']) == 1
        assert f'{CAP + 600:,} events' in asks['asked'][0][1]
    finally:
        ed.destroy()
        root.destroy()


def test_a_canceled_run_leaves_an_unreached_samples_earlier_labels(
        fake_phenograph):
    from tests.test_cluster_dialog import _add
    root, ed = _editor()
    try:
        a = _add(ed, _two_blobs(1_000, seed=1), 'a')
        b = _add(ed, _two_blobs(1_000, seed=2), 'b')
        earlier = np.arange(len(b.data)) % 3          # an earlier run's
        b.data['cluster'] = earlier
        _store_then(a, ed._cancel_clustering)
        _run(ed, 'Default')
        assert (b.data['cluster'].to_numpy() == earlier).all()
        audit = [d for act, d in ed._audited if act == 'cluster']
        assert len(audit) == 1 and audit[0]['samples'] == ['a']
        assert ed.status_var.get().startswith(
            'Clustering canceled after 1 of 2')
    finally:
        ed.destroy()
        root.destroy()


def test_the_question_uses_the_runs_k(fake_phenograph, asks):
    """Default PhenoGraph on 200,000 events needs about 2 GB at k=30 and
    about 6 GB at k=100: with 5 GB free only the k=100 run is asked about."""
    from tests.test_cluster_dialog import _add
    root, ed = _editor()
    try:
        _add(ed, _two_blobs(CAP + 600), 'big')
        asks['free'] = 5.0
        for k in (30, 100):
            ed._run_clustering(method='phenograph', all_samples=True, k=k,
                               grid=4, n_meta=3, embedding='none',
                               downsample='Default')
        assert len(asks['asked']) == 1
        need = cluster_caps.clustering_memory_gb('phenograph', CAP, 100)
        assert f'{need:.1f} GB' in asks['asked'][0][1]
        assert fake_phenograph == [CAP]           # only the k=30 run ran
    finally:
        ed.destroy()
        root.destroy()
