"""Boundaries a mutation campaign proved nothing was checking.

Two mutants survived the full suite — the source was changed and 1257 tests
stayed green. Both are minimum-sample guards, and in both the mutation moves
the boundary by exactly one sample, which is the case real data lands on and
synthetic fixtures usually miss.

    calibration.py:48   `if mfi.size < 2` -> `<= 2`
        Rejects a two-point calibration. Two points define a line, and the
        docstring promises ValueError only for FEWER than 2, so this breaks a
        documented, legitimate case.

    pipeline.py:972     `if ss_lo.size >= 50` -> `>`
        At exactly 50 low-FSC events the granularity valley is not computed,
        so `min_ssc_granular` is silently absent and that cleaning step stops
        happening. Nothing raises; the gate just quietly does less.

Each test asserts BOTH sides of its boundary. Asserting only the passing side
would leave the guard free to move outward, which is the same defect in the
other direction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openflo.calibration import fit_mesf_calibration
from openflo.pipeline import autoclean_method_diagnostic


def test_two_peak_pairs_are_enough_to_calibrate():
    """Two points define a line. The docstring promises ValueError for FEWER
    than two, so exactly two must fit — and a surviving mutant showed nothing
    held the boundary there."""
    out = fit_mesf_calibration([100.0, 200.0], [300.0, 500.0])

    assert out['n'] == 2
    assert out['slope'] == pytest.approx(2.0)
    assert out['intercept'] == pytest.approx(100.0)


def test_one_peak_pair_is_refused():
    """The other side of the same boundary: a single point cannot define a
    slope, and silently returning one would be worse than refusing."""
    with pytest.raises(ValueError, match='at least 2'):
        fit_mesf_calibration([100.0], [300.0])


# NOT TESTED, deliberately: pipeline.py:972 `ss_lo.size >= 50` -> `>`.
#
# The mutant survives, but it is unobservable in practice. The guard duplicates
# a floor _bimodal_valley already enforces on itself (`if v.size < 50: return
# None`), so the only input that could tell the two apart is EXACTLY 50 events
# — and there the detector barely works: measured over 8 seeds with 25+25
# bimodal points, a valley was found in 2/8 at sd=20 and 0/8 at sd=5, 10 and 2,
# because tight clusters collapse into one histogram bin and fail the 5%
# prominence test after smoothing.
#
# A test that killed this mutant would have to be pinned to one of the lucky
# seeds. It would then fail on a scipy or numpy bump rather than on a
# regression, which is worse than no test: it spends attention without
# protecting anything. The finding worth keeping is the redundancy itself.


def test_a_stale_viability_channel_falls_back_to_detection():
    """A configured channel that is no longer in the data must fall back.

    `if not ch or ch not in df.columns: ch = find_viability_channel(...)`.
    A mutant swapping that `or` for `and` survived the whole suite. The
    difference only shows for a channel name that IS set but is ABSENT from
    the data — exactly what a session saved against a different panel gives
    you. With `and` the stale name is kept, detection never runs, and the user
    is told "no viability dye detected" while a perfectly good dye sits in the
    frame.
    """
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        'FSC-A': rng.normal(60_000, 3_000, 500),
        'Zombie-Aqua-A': rng.normal(500, 100, 500),
    })

    msg = autoclean_method_diagnostic(
        'viability', df, {'channel': 'Comp-APC-Cy7-A-from-another-panel'})

    assert msg is None or 'no viability dye detected' not in msg, (
        'a stale channel name suppressed auto-detection: the frame contains '
        'Zombie-Aqua-A, so a viability dye was available and should have been '
        f'found. Got: {msg!r}')


def test_detection_still_reports_when_there_really_is_no_dye():
    """The other side: when no dye exists, saying so is correct."""
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        'FSC-A': rng.normal(60_000, 3_000, 500),
        'CD3-A': rng.normal(500, 100, 500),
    })

    msg = autoclean_method_diagnostic('viability', df, {'channel': 'Nope-A'})

    assert msg is not None and 'no viability dye detected' in msg, (
        f'no viability dye is present, so the diagnostic should say so; '
        f'got {msg!r}')
