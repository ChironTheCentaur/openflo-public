"""Tests for FlowSample.run_leiden — Leiden community detection on an
SNN/Jaccard graph. Uses a lightweight stub (no FCS) bound to the method."""
from __future__ import annotations

import types

import numpy as np
import pandas as pd

from openflo.pipeline import FlowSample


def _stub(df, channels):
    return types.SimpleNamespace(data=df, fluor_channels=channels)


def _three_blobs(n=800, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.vstack([rng.multivariate_normal(m, np.eye(2), n)
                     for m in ([0, 0], [10, 10], [0, 10])])
    truth = np.repeat([0, 1, 2], n)
    df = pd.DataFrame({'M1': pts[:, 0], 'M2': pts[:, 1]})
    return _stub(df, ['M1', 'M2']), truth


def test_leiden_writes_pure_clusters():
    s, truth = _three_blobs()
    FlowSample.run_leiden(s, resolution=1.0)
    lab = s.data['leiden'].to_numpy()
    assert (lab >= 0).all()
    assert s.data['leiden'].nunique() >= 3        # separates the blobs
    # Every Leiden cluster maps to a single ground-truth blob (no bleed).
    for c in np.unique(lab):
        blobs = truth[lab == c]
        purity = np.bincount(blobs, minlength=3).max() / len(blobs)
        assert purity > 0.98, (c, purity)


def test_leiden_resolution_is_monotone():
    s, _ = _three_blobs()
    counts = []
    for r in (0.1, 0.3, 1.0):
        FlowSample.run_leiden(s, resolution=r)
        counts.append(s.data['leiden'].nunique())
    # Higher resolution → at least as many clusters (coarse→fine).
    assert counts[0] <= counts[1] <= counts[2]
    assert counts[0] < counts[2]                  # the knob actually moves


def test_leiden_marks_nonfinite_as_noise():
    s, _ = _three_blobs(n=300)
    s.data.loc[0, 'M1'] = np.nan
    s.data.loc[1, 'M2'] = np.inf
    FlowSample.run_leiden(s)
    assert s.data['leiden'].iloc[0] == -1
    assert s.data['leiden'].iloc[1] == -1
    assert (s.data['leiden'].iloc[2:] >= 0).all()


def test_leiden_no_channels_is_noop():
    df = pd.DataFrame({'M1': [1.0, 2.0, 3.0]})
    s = _stub(df, ['DoesNotExist'])
    FlowSample.run_leiden(s)
    assert 'leiden' not in s.data.columns          # skipped, no column written


def test_leiden_subsample_assigns_all():
    s, truth = _three_blobs(n=1000)
    # Force the subsample+KD-tree-assign branch.
    FlowSample.run_leiden(s, resolution=1.0, max_events=600)
    lab = s.data['leiden'].to_numpy()
    assert len(lab) == 3000
    assert (lab >= 0).all()                        # every event labelled
    for c in np.unique(lab):
        blobs = truth[lab == c]
        assert np.bincount(blobs, minlength=3).max() / len(blobs) > 0.95


def test_leiden_too_few_events():
    s = _stub(pd.DataFrame({'M1': [1.0, 2.0], 'M2': [1.0, 2.0]}), ['M1', 'M2'])
    FlowSample.run_leiden(s)
    # <3 finite events → skipped (no column), not a crash.
    assert 'leiden' not in s.data.columns
