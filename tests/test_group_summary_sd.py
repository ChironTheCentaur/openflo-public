"""The SD of a single replicate is undefined, not zero.

`compare_groups` reported `sd: 0.0` for a group with one value — the sample SD
divides by n-1, so it is undefined there. Zero is a confident claim of "this
group had no variability", and it is rendered straight into the group summary
that `ui_frequency` shows as `sd=0`.

Same shape as the calibration r2 and the pseudotime fill: a plausible number
standing in for "unknown".
"""
import math

import numpy as np
import pytest

from openflo.stats import compare_groups


def test_a_single_replicate_has_no_sd():
    out = compare_groups({'ctrl': [1.0], 'treat': [2.0, 3.0, 4.0]})
    ctrl = out['groups']['ctrl']
    assert ctrl['n'] == 1
    assert math.isnan(ctrl['sd']), (
        f"sd={ctrl['sd']} for a single replicate — it reads as 'this group "
        "had zero variability'")


def test_a_real_group_still_reports_its_sd():
    out = compare_groups({'ctrl': [1.0], 'treat': [2.0, 3.0, 4.0]})
    treat = out['groups']['treat']
    assert treat['n'] == 3
    assert treat['sd'] == pytest.approx(np.std([2.0, 3.0, 4.0], ddof=1))


def test_a_genuinely_constant_group_still_reports_zero():
    """Zero is the RIGHT answer when it is measured rather than assumed —
    several identical replicates really do have no spread."""
    out = compare_groups({'a': [5.0, 5.0, 5.0], 'b': [1.0, 2.0, 3.0]})
    assert out['groups']['a']['sd'] == pytest.approx(0.0)
    assert out['groups']['a']['n'] == 3


def test_the_other_summary_fields_are_unaffected():
    out = compare_groups({'a': [2.0], 'b': [1.0, 3.0]})
    a = out['groups']['a']
    assert a['mean'] == pytest.approx(2.0)
    assert a['median'] == pytest.approx(2.0)


def test_non_finite_values_are_dropped_before_summarising():
    out = compare_groups({'a': [1.0, np.nan, 3.0, np.inf],
                          'b': [1.0, 2.0, 3.0]})
    a = out['groups']['a']
    assert a['n'] == 2, 'non-finite values leaked into the summary'
    assert a['mean'] == pytest.approx(2.0)
    assert math.isfinite(a['sd'])
