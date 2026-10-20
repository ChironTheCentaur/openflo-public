"""The data a tool reads and the transform record it inverts with agree.

Two places took them from different moments or different sources: the
background statistics run copied the record but kept reading the live frame
(a transform change mid-run then inverted a new column with the old record),
and the Transform editor's worker undid each channel with the recorded METHOD
but default parameters (an asinh cofactor-5 channel was undone with 150).
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.test_transform_records import (
    capture_warnings,
    editor_with_samples,
    isolate_home,
)


@pytest.fixture
def ed_env(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)       # never a real (blocking) modal
    yield from editor_with_samples(tmp_path)


def test_stats_snapshot_is_not_torn_by_a_transform_change(ed_env):
    """The background stats run must see the data and the record of ONE
    moment: a column replaced by a transform change mid-run must not be
    inverted with the record taken before it."""
    from openflo.pipeline import linear_values, transform_spec, transform_values
    root, ed, _ = ed_env
    name = 'expt_s1'
    ed._sample_gates[name] = {'g1': {'kind': 'threshold', 'channel': 'FSC-A',
                                     'value': 0.0, 'parent_id': None,
                                     'name': 'All', 'enabled': True}}
    ed._sample_gate_order[name] = ['g1']
    want = {'Median', 'Mean'}
    before, _ = ed._stats_rows_from_snapshot(
        ed._stats_snapshot(want, samples=[name]), want)
    snap = ed._stats_snapshot(want, samples=[name])
    s = ed._samples[name]
    # What _apply_channel_transforms' on_done does while the stats thread runs.
    lin = linear_values(s, 'FL1-A')
    s.data['FL1-A'] = transform_values(lin, method='asinh')
    s.data_transforms = {**s.data_transforms,
                         'FL1-A': transform_spec('asinh')}
    after, _ = ed._stats_rows_from_snapshot(snap, want)
    assert after[0]['Mean CD901b'] == pytest.approx(before[0]['Mean CD901b'],
                                                   rel=1e-9)
    assert after[0]['Median CD901b'] == pytest.approx(
        before[0]['Median CD901b'], rel=1e-9)


def test_retransform_inverts_with_the_recorded_parameters(ed_env):
    """A channel recorded as asinh with cofactor 5 must be undone with
    cofactor 5, not the default 150."""
    from openflo.pipeline import linear_values, transform_spec, transform_values
    root, ed, _ = ed_env
    s = ed._samples['expt_s1']
    lin = linear_values(s, 'FL1-A')
    s.data['FL1-A'] = transform_values(lin, method='asinh', cofactor=5.0)
    s.data_transforms = {**s.data_transforms,
                         'FL1-A': transform_spec('asinh', cofactor=5.0)}
    ed._channel_transform['FL1-A'] = 'asinh'
    ed._apply_channel_transforms({'FL1-A': 'logicle'})
    np.testing.assert_allclose(s.data['FL1-A'].to_numpy(float),
                               transform_values(lin, method='logicle'),
                               rtol=1e-6, atol=1e-9)
    assert s.data_transforms['FL1-A'] == transform_spec('logicle')
