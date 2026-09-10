"""`raw` must survive everything done to `data`.

`_load()` builds the events once and then does `self.data = self.raw.copy(
deep=False)` — a shallow copy, for speed and memory (a deep one cost 3x the
time and twice the RAM on load). The consequence is that `data` and `raw`
SHARE numpy buffers, and `raw` is the pristine untransformed events: it is
what recompensation, re-transformation, raw-value export and resetting a
transform all re-derive from. A write to `data` that reached `raw` would not
raise, would not look wrong on screen, and would quietly poison every later
derivation.

The protection is pandas' copy-on-write, which is a property of pandas, not of
this code — so it is pinned here rather than assumed. An aliasing bug was
caught once already before shipping (`data = raw`, with twelve in-place writes
downstream), which is why this file exists.

The test runs through the FCS reader on purpose. `from_dataframe` takes a
different path that does a real copy, so a test built on it shares no buffers
and proves nothing about the change it is meant to cover.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openflo.pipeline import FlowSample, write_fcs

CH = [f'FL{i}-A' for i in range(6)]


@pytest.fixture
def sample(tmp_path):
    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.uniform(10, 60_000, (3000, len(CH))), columns=CH)
    path = tmp_path / 'x.fcs'
    write_fcs(str(path), df, channels=CH)
    s = FlowSample(str(path))
    s.fluor_channels = [c for c in CH if c in s.data.columns]
    return s


def _snapshot(s):
    return s.raw[s.fluor_channels].to_numpy(copy=True)


def _assert_raw_intact(s, before, step):
    after = s.raw[s.fluor_channels].to_numpy()
    assert np.allclose(before, after, equal_nan=True), (
        f'{step} wrote through to `raw`. `data` shares buffers with `raw`, so '
        f'anything re-derived from the untransformed events is now wrong.')


def test_the_buffers_really_are_shared(sample):
    """If this fails the rest of the file proves nothing — it would be testing
    an independent copy, which is trivially safe. Guards against the tests
    below quietly becoming vacuous if `_load` changes."""
    shared = any(
        np.shares_memory(sample.data[c].to_numpy(), sample.raw[c].to_numpy())
        for c in sample.fluor_channels)
    assert shared, (
        'data no longer shares buffers with raw — either _load went back to a '
        'deep copy (then this file is obsolete) or the fixture is wrong')


def test_transform_does_not_reach_raw(sample):
    before = _snapshot(sample)
    sample.apply_transform()
    _assert_raw_intact(sample, before, 'apply_transform')


def test_compensation_does_not_reach_raw(sample):
    before = _snapshot(sample)
    n = len(sample.fluor_channels)
    sample.manual_compensate(np.eye(n), sample.fluor_channels)
    _assert_raw_intact(sample, before, 'manual_compensate')


def test_column_writes_do_not_reach_raw(sample):
    before = _snapshot(sample)
    col = sample.fluor_channels[0]
    sample.data['cluster'] = 0
    sample.data[col] = sample.data[col] * 2.0
    sample.data.loc[sample.data.index[:10], col] = -1.0
    _assert_raw_intact(sample, before, 'column and .loc writes')


def test_clustering_does_not_reach_raw(sample):
    before = _snapshot(sample)
    pytest.importorskip('igraph')
    pytest.importorskip('leidenalg')
    sample.run_leiden(channels=sample.fluor_channels, n_neighbors=10,
                      max_events=3000, random_state=42)
    _assert_raw_intact(sample, before, 'run_leiden')


def test_the_check_can_actually_fail(sample):
    """A corruption detector that never fires is not a detector. Write to raw
    deliberately and confirm the assertion catches it."""
    before = _snapshot(sample)
    col = sample.fluor_channels[0]
    sample.raw[col] = sample.raw[col] * 1.5
    with pytest.raises(AssertionError, match='wrote through to `raw`'):
        _assert_raw_intact(sample, before, 'deliberate corruption')
