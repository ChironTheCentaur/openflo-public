"""The CLI's FMO cut drops infinite values, as the app's FMO dialog does.

FMOGater.compute (the CLI and library path) took the percentile over
``dropna()``, which keeps +/-inf: one +inf event made the cut NaN, and -inf
events shifted it. The app's dialog (gating_helpers.fmo_threshold) uses finite
values only. Best-practices finding B6; both now share fmo_threshold.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import openflo.pipeline as fp
from openflo.gating_helpers import fmo_threshold


def _gater(values):
    rng = np.random.default_rng(3)
    s = fp.FlowSample.from_dataframe(
        pd.DataFrame({'FSC-A': rng.normal(5e4, 5e3, values.size),
                      'FL1-A': values}), name='fmo')
    g = fp.FMOGater()
    g.fmos['FL1-A'] = s
    return g, s


def test_infinite_fmo_values_do_not_reach_the_cut():
    rng = np.random.default_rng(0)
    finite = rng.normal(0.4, 0.1, 5000)
    want = float(np.percentile(finite, 99))
    for bad in ([np.inf], [np.inf] * 30, [-np.inf] * 200,
                [np.nan, np.inf, -np.inf]):
        vals = np.r_[finite, bad]
        g, s = _gater(vals)
        assert not np.isfinite(s.data['FL1-A']).all()   # the infs are there
        cut = g.compute()['FL1-A']
        assert cut == want, (bad, cut)
        # The same number the app's FMO dialog shows for these events.
        assert cut == fmo_threshold(vals, 99)


def test_finite_fmo_values_give_the_same_cut_as_before():
    rng = np.random.default_rng(1)
    vals = rng.lognormal(0, 1, 3000)
    g, s = _gater(vals)
    for p in (95.0, 99.0, 99.5):
        assert g.compute(p)['FL1-A'] == float(np.percentile(vals, p))


def test_a_channel_with_no_finite_values_is_skipped_with_a_note(caplog):
    g, _s = _gater(np.full(100, np.nan))
    with caplog.at_level(logging.INFO, logger=fp.log.name):
        assert g.compute() == {}
    assert any('no finite values' in r.getMessage() for r in caplog.records)
