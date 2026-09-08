"""Batch-correction QC must not score the unmeasurable as perfect.

`CytoNorm.qc` reports a per-channel Wasserstein distance from each batch to
the pooled goal, where LOWER IS BETTER. It returned 0.0 for a channel with no
usable events — the best possible score on that scale, indistinguishable from
a genuinely perfect alignment, for a channel nobody could evaluate.

The same function also raised out of a QC call. The goal distribution is
pooled over events that are finite in EVERY channel, so one channel that was
never acquired empties it, and scipy then raised "Distribution can't be empty"
against a channel that did have data.

Both are now reported: a channel that cannot be evaluated is NaN, and an
absent goal distribution returns NaN for every channel with a warning rather
than an exception. The callers average with `nanmean` so one unmeasurable
marker no longer hides the real scores of the others.

SCOPE, stated honestly: neither case is reachable from the shipped call paths.
Both callers hand `qc()` exactly the `events_by_batch` they just fitted, and
`fit()` raises "no finite events to fit on" for any input that would empty the
metric lists; `from_dict` is only ever used with `apply()`, never `qc()`. These
tests drive `qc()` directly with data the model was not fitted on. That is
defensive hardening of a public method, not the repair of a live defect — the
first version of this file claimed otherwise, and it was wrong.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from openflo.pipeline import CytoNorm

CHANNELS = ['CD3', 'CD19']


def _batch(shift, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'CD3': rng.normal(1000.0 + shift, 120.0, n),
                         'CD19': rng.normal(600.0 + shift, 90.0, n)})


@pytest.fixture
def model():
    return CytoNorm(CHANNELS).fit({'b1': _batch(0.0, seed=1),
                                   'b2': _batch(220.0, seed=2)})


def test_real_batches_still_get_real_scores(model):
    """The guard must not turn QC into a wall of NaN for ordinary data."""
    qc = model.qc({'b1': _batch(0.0, seed=1), 'b2': _batch(220.0, seed=2)})
    assert set(qc) == set(CHANNELS)
    for channel, scores in qc.items():
        assert np.isfinite(scores['before']), channel
        assert np.isfinite(scores['after']), channel
        assert scores['before'] > 0


def test_an_absent_goal_distribution_is_reported_not_raised(model, caplog):
    """No event is finite in every channel, so there is no goal to compare
    against. This used to raise ValueError("Distribution can't be empty")."""
    n = 3000
    rng = np.random.default_rng(5)
    unusable = {b: pd.DataFrame({'CD3': np.full(n, np.nan),
                                 'CD19': rng.normal(600.0, 90.0, n)})
                for b in ('b1', 'b2')}
    with caplog.at_level(logging.WARNING):
        qc = model.qc(unusable)

    assert set(qc) == set(CHANNELS)
    for channel, scores in qc.items():
        assert np.isnan(scores['before']), (
            f'{channel} claimed a batch→goal distance of '
            f'{scores["before"]} with no goal distribution')
        assert np.isnan(scores['after'])
    assert any('goal distribution' in r.getMessage() for r in caplog.records)


def test_zero_is_never_used_for_undefined(model):
    """0.0 is the BEST score here. Whatever an unmeasurable channel reports,
    it must not be the value that means 'perfect'."""
    n = 2000
    rng = np.random.default_rng(6)
    unusable = {b: pd.DataFrame({'CD3': np.full(n, np.nan),
                                 'CD19': rng.normal(600.0, 90.0, n)})
                for b in ('b1', 'b2')}
    values = [v for scores in model.qc(unusable).values()
              for v in scores.values()]
    assert not any(v == 0.0 for v in values), (
        'an unevaluable channel reported 0.0, the best possible alignment')


def test_nanmean_keeps_a_measurable_channel_visible():
    """How the callers consume it: one unmeasurable marker must not erase the
    scores of the markers that were measured."""
    qc = {'CD3': {'before': float('nan'), 'after': float('nan')},
          'CD19': {'before': 110.0, 'after': 40.0}}
    before = float(np.nanmean([d['before'] for d in qc.values()]))
    after = float(np.nanmean([d['after'] for d in qc.values()]))
    assert before == pytest.approx(110.0)
    assert after == pytest.approx(40.0)
    assert np.isfinite(before) and before > 0
