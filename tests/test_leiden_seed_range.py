"""A word or phrase Seed, a negative seed or one at or above
2**31 made every Leiden run, and every reproducible PhenoGraph run, fail with
OverflowError, because leidenalg takes a signed 32-bit seed while
seeds.resolve_seed returns [0, 2**32). The seed is now folded into [0, 2**31)
only where it is handed to leidenalg (seeds.signed32_seed). A seed below 2**31
reaches leidenalg unchanged, and FlowSOM, the subsample draws and the recorded
settings still see the seed resolve_seed gives, so no run that worked before
changes its answer.

Generated blobs on generic channels; the expected numbers are recomputed here.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
import pytest

from openflo.pipeline import FlowSample
from openflo.seeds import resolve_seed, signed32_seed

CH = ['FL1-A', 'FL2-A']


def _blobs(n=150, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.vstack([rng.normal(m, 0.6, (n, 2))
                     for m in ([0, 0], [10, 10], [0, 10])])
    return pd.DataFrame(pts, columns=CH), np.repeat([0, 1, 2], n)


def _sample(df):
    return FlowSample.from_dataframe(df.copy(), name='sample_a')


def _pure(labels, truth):
    for c in np.unique(labels):
        if np.bincount(truth[labels == c], minlength=3).max() != (
                labels == c).sum():
            return False
    return True


def test_resolve_seed_is_unchanged_and_the_fold_is_identity_below_2_31():
    # resolve_seed itself is untouched: FlowSOM and numpy keep these seeds.
    assert resolve_seed(42) == 42
    assert resolve_seed('7') == 7
    assert resolve_seed(-7) == 2 ** 32 - 7 == 4294967289
    assert resolve_seed(2 ** 31) == 2147483648
    assert resolve_seed('pilot run 3') == 2505760063
    for s in (0, 1, 42, 7, 123456789, 2 ** 31 - 1):
        assert signed32_seed(s) == s
    assert signed32_seed(2 ** 31) == 0
    assert signed32_seed(2505760063) == 2505760063 - 2 ** 31 == 358276415
    assert signed32_seed(4294967289) == 2147483641
    assert signed32_seed(-7) == 2 ** 31 - 7


@pytest.mark.parametrize('seed', ['pilot run 3', '-7', 2 ** 31])
def test_leiden_runs_with_a_seed_at_or_above_2_31(seed, monkeypatch):
    import leidenalg
    seen = []
    orig = leidenalg.find_partition

    def spy(*a, **kw):
        seen.append(kw['seed'])
        return orig(*a, **kw)
    monkeypatch.setattr(leidenalg, 'find_partition', spy)
    df, truth = _blobs()
    rs = resolve_seed(seed)
    assert rs >= 2 ** 31
    s = _sample(df)
    s.run_leiden(channels=CH, random_state=rs)
    lab = s.data['leiden'].to_numpy()
    assert seen == [rs - 2 ** 31]
    assert (lab >= 0).all() and len(np.unique(lab)) >= 3 and _pure(lab, truth)
    # The folded run is the run seeded with rs - 2**31 directly.
    s2 = _sample(df)
    s2.run_leiden(channels=CH, random_state=rs - 2 ** 31)
    assert np.array_equal(lab, s2.data['leiden'].to_numpy())


def test_leiden_hands_a_seed_below_2_31_to_leidenalg_unchanged(monkeypatch):
    import leidenalg
    seen = []
    orig = leidenalg.find_partition

    def spy(*a, **kw):
        seen.append(kw['seed'])
        return orig(*a, **kw)
    monkeypatch.setattr(leidenalg, 'find_partition', spy)
    df, _ = _blobs()
    for rs in (42, 7, 2 ** 31 - 1):
        _sample(df).run_leiden(channels=CH, random_state=rs)
    assert seen == [42, 7, 2 ** 31 - 1]


def test_reproducible_phenograph_runs_with_a_negative_or_phrase_seed():
    df, truth = _blobs()
    for seed in ('-7', 'pilot run 3'):
        rs = resolve_seed(seed)
        s = _sample(df)
        with contextlib.redirect_stdout(io.StringIO()):
            s.cluster(channels=CH, k=15, max_events=None, random_state=rs,
                      reproducible=True, use_gpu=False)
        assert s.cluster_settings['route'] == 'leiden'
        # The settings record the seed the user's setting resolved to.
        assert s.cluster_settings['random_state'] == rs
        lab = s.data['cluster'].to_numpy()
        assert (lab >= 0).all() and _pure(lab, truth)


def test_phenograph_kwargs_fold_only_the_leiden_seed():
    from openflo.flow_sample import _phenograph_kwargs

    def fake_cluster(data, k=30, n_jobs=-1, clustering_algo='louvain',
                     seed=None, resolution_parameter=1.0):
        raise AssertionError('not called')
    for rs, want in ((42, 42), (2 ** 31 - 1, 2 ** 31 - 1),
                     (4294967289, 2147483641), (2505760063, 358276415)):
        kw, route = _phenograph_kwargs(fake_cluster, 30, 1, True, rs, 1.0)
        assert route == 'leiden' and kw['seed'] == want


def _run_cfg(method, seed):
    from openflo import workspace_run as W
    return dict(W.default_run_cfg(), method=method, seed=seed, umap=False,
                trimap=False, pacmap=False, phate=False, tsne=False,
                max_events=0)


@pytest.mark.parametrize('method', ['leiden', 'phenograph'])
def test_pipeline_run_with_a_phrase_or_negative_seed_succeeds(method,
                                                              tmp_path):
    from openflo import workspace_run as W
    df, _truth = _blobs()
    for i, seed in enumerate(('pilot run 3', '-7')):
        prep = {'data': df.copy(), 'channels': list(CH),
                'label': 'sample_a', 'n_events': len(df)}
        with contextlib.redirect_stdout(io.StringIO()):
            res = W.compute_run(prep, _run_cfg(method, seed),
                                str(tmp_path / f'{method}{i}'))
        assert res['ok'], res.get('error')


def test_flowsom_still_gets_the_unfolded_seed_and_the_same_labels(
        monkeypatch, tmp_path):
    """FlowSOM accepted every seed before; it must keep getting the seed
    resolve_seed gives (not the folded one), so its labels do not change."""
    from openflo import workspace_run as W
    df, _ = _blobs()
    seen = []
    orig = FlowSample.run_flowsom

    def spy(self, *a, **kw):
        out = orig(self, *a, **kw)
        seen.append((kw['seed'], self.data['flowsom_meta'].to_numpy().copy()))
        return out
    monkeypatch.setattr(FlowSample, 'run_flowsom', spy)
    prep = {'data': df.copy(), 'channels': list(CH), 'label': 'sample_a',
            'n_events': len(df)}
    with contextlib.redirect_stdout(io.StringIO()):
        res = W.compute_run(prep, _run_cfg('flowsom', 'pilot run 3'),
                            str(tmp_path / 'f'))
    assert res['ok'], res.get('error')
    ((seed, labels),) = seen
    assert seed == 2505760063
    monkeypatch.setattr(FlowSample, 'run_flowsom', orig)
    s = _sample(df)
    from openflo.som_kernel import FLOWSOM_DEFAULT_METACLUSTERS
    s.run_flowsom(channels=CH, max_events=len(df), seed=2505760063,
                  n_metaclusters=W.default_run_cfg().get(
                      'n_metaclusters', FLOWSOM_DEFAULT_METACLUSTERS))
    assert np.array_equal(labels, s.data['flowsom_meta'].to_numpy())


def test_cli_shared_clustering_runs_leiden_with_a_large_seed():
    """The CLI's --seed is an int handed straight through (here 3000000000,
    above 2**31) to cluster_by_method, the CLI's shared-clustering path."""
    from openflo.shared_clustering import cluster_by_method
    df, truth = _blobs()
    s = _sample(df)
    cluster_by_method(s, {'method': 'leiden', 'k': 15, 'max_events': None,
                          'random_state': 3_000_000_000})
    lab = s.data['cluster'].to_numpy()
    assert (lab >= 0).all() and _pure(lab, truth)
