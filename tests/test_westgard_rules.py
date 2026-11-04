"""Each Westgard rule in bead tracking is judged at its exact threshold, with
the other runs it looks back on, and listed in the documented order.

The baseline is mean 100, SD 10 throughout, so a value v sits at
z = (v - 100) / 10 and every expected list below is worked out by hand from
the rule definitions in openflo.bead_tracking. Values are synthetic. A
mutation check planted bugs in each rule (thresholds, look-back windows,
side checks, rule order) and these tests catch them (docs/test-assurance/).
"""
from __future__ import annotations

import numpy as np
import pytest

from openflo import bead_tracking as bt

NAN = float('nan')


def last_rules(values):
    """The rules the last run breaks against mean 100, SD 10."""
    return bt.westgard(values, 100.0, 10.0)[-1]


@pytest.mark.parametrize('value, rules', [
    (130.0, ['1-2s']),                  # z = 3 is not beyond 3 SD
    (130.5, ['1-3s', '1-2s']),
    (69.5, ['1-3s', '1-2s']),           # below the mean counts too
    (120.0, []),                        # z = 2 is not beyond 2 SD
    (120.5, ['1-2s']),
    (79.5, ['1-2s']),
])
def test_one_run_rules_are_strictly_beyond_their_limit(value, rules):
    assert last_rules([value]) == rules


@pytest.mark.parametrize('values, rules', [
    ([125.0], ['1-2s']),                 # no earlier run: no 2-2s
    ([125.0, 125.0], ['2-2s', '1-2s']),
    ([75.0, 75.0], ['2-2s', '1-2s']),
    ([120.0, 125.0], ['1-2s']),          # the earlier run is at 2 SD exactly
    ([125.0, 120.0], []),
    ([125.0, 100.0, 125.0], ['1-2s']),   # the run before must be beyond too
    ([75.0, 125.0, 125.0], ['2-2s', '1-2s']),  # older runs do not matter
])
def test_two_two_s_needs_two_runs_in_a_row_on_one_side(values, rules):
    assert last_rules(values) == rules


@pytest.mark.parametrize('values, rules', [
    ([125.0, 75.0], ['R-4s', '1-2s']),
    ([79.0, 121.0], ['R-4s', '1-2s']),
    ([120.0, 79.0], ['1-2s']),           # earlier run at 2 SD exactly
    ([121.0, 80.0], []),                 # this run at 2 SD exactly
    ([135.0, 75.0], ['R-4s', '1-2s']),
    ([125.0, 66.0], ['1-3s', 'R-4s', '1-2s']),
    ([125.0, 100.0, 75.0], ['1-2s']),    # only the run just before counts
])
def test_r_four_s_needs_two_runs_in_a_row_beyond_2sd_on_opposite_sides(
        values, rules):
    assert last_rules(values) == rules


@pytest.mark.parametrize('values, rules', [
    ([111.0] * 4, ['4-1s']),
    ([89.0] * 4, ['4-1s']),
    ([111.0] * 3, []),                   # three runs are not enough
    ([111.0, 110.0, 111.0, 111.0], []),  # one at 1 SD exactly
    ([111.0, 89.0, 111.0, 111.0], []),   # one on the other side
    ([89.0, 111.0, 111.0, 111.0, 111.0], ['4-1s']),  # only the last four
    ([111.0, 111.0, 111.0, 100.0, 111.0], []),
])
def test_four_one_s_needs_four_runs_in_a_row_beyond_1sd_on_one_side(
        values, rules):
    assert last_rules(values) == rules


@pytest.mark.parametrize('values, rules', [
    ([101.0] * 10, ['10x']),
    ([99.0] * 10, ['10x']),
    ([101.0] * 9, []),                                   # nine are not enough
    ([99.0] + [101.0] * 10, ['10x']),                    # only the last ten
    ([101.0] * 4 + [100.0] + [101.0] * 5, []),           # a run on the mean
    ([101.0] * 9 + [100.0], []),
    ([101.0] * 9 + [99.0], []),
    ([99.0] + [101.0] * 9, []),
    ([100.0] * 10, []),                                  # on the mean: no side
])
def test_ten_x_needs_ten_runs_in_a_row_on_one_side(values, rules):
    assert last_rules(values) == rules


def test_rules_are_listed_reject_first_then_warning_in_documented_order():
    assert last_rules([135.0] * 4) == ['1-3s', '2-2s', '4-1s', '1-2s']
    assert last_rules([135.0] * 10) == \
        ['1-3s', '2-2s', '4-1s', '10x', '1-2s']
    assert last_rules([125.0, 66.0]) == ['1-3s', 'R-4s', '1-2s']


def test_every_run_is_judged_as_the_last_run_of_the_pattern():
    assert bt.westgard([125.0, 125.0, 75.0, 100.0], 100.0, 10.0) == [
        ['1-2s'], ['2-2s', '1-2s'], ['R-4s', '1-2s'], []]


def test_missing_runs_are_skipped_and_do_not_break_a_pattern():
    v = [101.0] * 5 + [NAN] + [101.0] * 5
    out = bt.westgard(v, 100.0, 10.0)
    assert out[5] == [] and out[-1] == ['10x']
    assert out[-2] == []                 # only nine finite runs so far
    out = bt.westgard([125.0, NAN, np.inf, -np.inf, 125.0], 100.0, 10.0)
    assert out == [['1-2s'], [], [], [], ['2-2s', '1-2s']]
    assert last_rules([111.0, 111.0, NAN, 111.0, 111.0]) == ['4-1s']


def test_the_look_back_reaches_exactly_nine_earlier_runs():
    # Ten runs on one side after an eleventh run on the other: 10x on the
    # last only, since run 1 lies outside its ten-run window.
    out = bt.westgard([99.0] + [101.0] * 10, 100.0, 10.0)
    assert [r for r in out[:-1] if '10x' in r] == []
    assert out[-1] == ['10x']


@pytest.mark.parametrize('mean, sd', [
    (100.0, 0.0), (100.0, -10.0), (NAN, 10.0), (100.0, NAN),
    (np.inf, 10.0), (100.0, np.inf),
])
def test_without_a_usable_baseline_nothing_is_judged(mean, sd):
    assert bt.westgard([100.0, 500.0, 500.0], mean, sd) == [[], [], []]


def test_each_run_gets_its_own_list_and_empty_input_gives_none():
    assert bt.westgard([], 100.0, 10.0) == []
    out = bt.westgard([100.0, 100.0], NAN, NAN)
    out[0].append('x')
    assert out[1] == []
    out = bt.westgard([100.0, 100.0], 100.0, 10.0)
    out[0].append('x')
    assert out[1] == []


def test_integer_input_and_a_numpy_array_are_judged_alike():
    assert bt.westgard([100, 131], 100, 10) == [[], ['1-3s', '1-2s']]
    assert bt.westgard(np.array([100.0, 131.0]), 100.0, 10.0) == \
        [[], ['1-3s', '1-2s']]
