"""How `to_anndata` assembles X, obs, var and layers, checked without the
optional ``anndata`` package.

The other to_anndata tests skip unless anndata (and its h5py stack) is
installed, which none of the test runs do, so a stand-in ``anndata`` module
records what to_anndata hands its constructor. Every check pins one piece of
the assembly: which markers are kept, how samples are subsampled, the obs
frame, the cluster labels, the transform columns and the linear layer.
All data are synthetic (channels FL1-A..FL3-A, samples s1, s2, ...).
"""
from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pandas as pd
import pytest

from openflo.channel_transforms import transform_spec, transform_values
from openflo.cluster_labels import cluster_label
from openflo.interop import to_anndata


class _FakeAnnData:
    def __init__(self, X, obs, var, layers):
        self.X, self.obs, self.var, self.layers = X, obs, var, layers
        self.uns = {}


@pytest.fixture
def fake_anndata(monkeypatch):
    mod = types.ModuleType('anndata')
    mod.AnnData = _FakeAnnData           # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'anndata', mod)
    return mod


def _frame(n, seed=0, cols=('FL1-A', 'FL2-A', 'FL3-A')):
    r = np.random.default_rng(seed)
    return pd.DataFrame({c: r.normal(0.5, 0.2, n) for c in cols})


def test_missing_package_error_names_the_install_and_keeps_the_cause(
        monkeypatch):
    monkeypatch.setitem(sys.modules, 'anndata', None)
    with pytest.raises(ImportError, match=r"pip install anndata") as info:
        to_anndata({'s1': _frame(3)}, ['FL1-A'])
    assert isinstance(info.value.__cause__, ImportError)


def test_no_samples_keeps_every_marker_with_empty_blocks(fake_anndata):
    a = to_anndata({}, ['FL1-A', 'FL2-A'], transforms={})
    assert list(a.var.index) == ['FL1-A', 'FL2-A']
    assert a.X.shape == (0, 2)
    assert a.layers['linear'].shape == (0, 2)
    assert a.layers['linear'].dtype == np.float32


def test_dropped_markers_are_logged_with_their_count(fake_anndata, caplog):
    samples = {'s1': _frame(4), 's2': _frame(4, cols=('FL1-A',))}
    with caplog.at_level(logging.WARNING, logger='openflo.interop'):
        a = to_anndata(samples, ['FL1-A', 'FL2-A', 'FL3-A', 'FL2-A'])
    assert list(a.var.index) == ['FL1-A']
    assert a.uns['dropped_markers'] == ['FL2-A', 'FL3-A', 'FL2-A']
    msgs = [r.getMessage() for r in caplog.records]
    assert msgs == ['to_anndata: 3 marker(s) omitted (missing from some '
                    'samples): FL2-A, FL3-A, FL2-A']


def test_subsample_is_the_seeded_draw_and_skips_samples_at_the_cap(
        fake_anndata):
    s1, s2 = _frame(5, seed=1), _frame(40, seed=2)
    a = to_anndata({'s1': s1, 's2': s2}, ['FL1-A'], max_events=5, seed=7)
    want_rows = np.random.default_rng(7).choice(40, 5, replace=False)
    np.testing.assert_array_equal(a.X[:5, 0], s1['FL1-A'].to_numpy())
    np.testing.assert_array_equal(a.X[5:, 0],
                                  s2['FL1-A'].to_numpy()[want_rows])


def test_linear_layer_follows_the_subsample_in_float32(fake_anndata):
    lin = np.random.default_rng(3).lognormal(np.log(3000), .4, 30)
    df = pd.DataFrame({'FL1-A': transform_values(lin)})
    a = to_anndata({'s1': df}, ['FL1-A'], max_events=8, seed=2,
                   transforms={'s1': {'FL1-A': transform_spec('logicle')}})
    rows = np.random.default_rng(2).choice(30, 8, replace=False)
    got = a.layers['linear']
    assert got.dtype == np.float32 and got.shape == (8, 1)
    np.testing.assert_allclose(got[:, 0], lin[rows], rtol=1e-4)


def test_empty_transforms_still_give_a_linear_layer(fake_anndata):
    df = _frame(6)
    a = to_anndata({'s1': df}, ['FL1-A', 'FL2-A'], transforms={})
    np.testing.assert_allclose(a.layers['linear'],
                               df[['FL1-A', 'FL2-A']].to_numpy(), rtol=1e-6)
    assert list(a.var['transform']) == ['linear', 'linear']


def test_no_shared_marker_layer_has_no_columns(fake_anndata):
    samples = {'s1': _frame(3, cols=('FL1-A',)), 's2': _frame(2, cols=('FL2-A',))}
    a = to_anndata(samples, ['FL1-A', 'FL2-A'], transforms={})
    assert a.layers['linear'].shape == (5, 0)


def test_obs_rows_carry_the_sample_key_and_a_running_string_index(
        fake_anndata):
    samples = {3: _frame(2), 's2': _frame(3)}
    a = to_anndata(samples, ['FL1-A'])
    assert list(a.obs['sample']) == [3, 3, 's2', 's2', 's2']
    assert list(a.obs.index) == ['0', '1', '2', '3', '4']


def test_an_empty_sample_alone_still_has_a_sample_column(fake_anndata):
    a = to_anndata({'s1': _frame(0)}, ['FL1-A'])
    assert list(a.obs.columns) == ['sample']
    assert len(a.obs) == 0


def test_only_cluster_columns_get_labels(fake_anndata):
    df = _frame(3)
    df['cluster'] = [2.0, -1.0, 0.0]
    df['batch'] = [1, 1, 2]
    a = to_anndata({'s1': df}, ['FL1-A'], obs_cols=['cluster', 'batch'])
    assert list(a.obs.columns) == ['sample', 'cluster', 'batch',
                                   'cluster_label']
    assert list(a.obs['cluster_label']) == [
        'Cluster 2', cluster_label(-1), 'Cluster 0']


def test_var_names_are_the_marker_names_as_text(fake_anndata):
    a = to_anndata({'s1': _frame(2)}, ['FL2-A', 'FL1-A'])
    assert list(a.var.index) == ['FL2-A', 'FL1-A']
