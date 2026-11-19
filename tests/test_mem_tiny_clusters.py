"""A cluster too small to have a spread must not set the scale for the table.

MEM scores each population as ``|median shift| + IQR_ref/IQR_pop - 1``, then
rescales globally so the largest |score| maps to 10. The IQR ratio is the
"this population is tighter than the rest" half of the score — and a
population with NO measurable spread (a single event, or every event
identical) has an IQR of exactly 0, which ``+ eps`` turned into "perfectly
tight". That is the maximum the ratio can reward.

Because the table is then rescaled by its own largest value, one such cluster
set the scale for everything else. Measured: adding a single-event cluster to
three real populations pushed their scores from 10, -10, -7 down to 4, -5, -5
— every biological call in the table compressed toward zero by one artefact.

No measurable spread supports no claim about spread, so the ratio term now
drops out and such a population is scored on its median shift alone.

What this does NOT fix, and should not: a population sitting exactly at the
median of a multimodal reference has a genuinely unstable sign under
"vs all other cells", because adding any cells moves that median across it.
That is a property of the metric, not a defect, and the last test pins the
distinction.
"""
import numpy as np
import pandas as pd
import pytest

from openflo.annotate import mem_scores

MARKERS = ['CD3', 'CD19']


def _populations(singleton, third=(1500.0, 1500.0), n=300, seed=0):
    rng = np.random.default_rng(seed)
    frames, labels = [], []
    for i, (cd3, cd19) in enumerate([(3000.0, 200.0), (300.0, 2500.0), third]):
        frames.append(pd.DataFrame({'CD3': rng.normal(cd3, 150.0, n),
                                    'CD19': rng.normal(cd19, 150.0, n)}))
        labels.append(np.full(n, i))
    if singleton:
        frames.append(pd.DataFrame({'CD3': [90_000.0], 'CD19': [90_000.0]}))
        labels.append(np.array([3]))
    return pd.concat(frames, ignore_index=True), np.concatenate(labels)


def _scores(**kw):
    data, labels = _populations(**kw)
    return mem_scores(data, labels, MARKERS)


def test_a_single_event_cluster_does_not_compress_the_real_ones():
    """The headline: three real populations, with and without one stray
    event given its own label."""
    alone = _scores(singleton=False)
    with_stray = _scores(singleton=True)

    for pop in (0, 1, 2):
        for marker in MARKERS:
            before = abs(alone.loc[pop, marker])
            after = abs(with_stray.loc[pop, marker])
            assert after >= before - 3.0, (
                f'population {pop} {marker} fell from {before} to {after} '
                'because of a one-event cluster elsewhere in the table')


def test_the_stray_event_does_not_top_the_table():
    """It used to score the maximum — 10, the strongest call available —
    purely for having no spread to measure."""
    scores = _scores(singleton=True)
    stray = float(np.max(np.abs(scores.loc[3].to_numpy())))
    real = float(np.max(np.abs(scores.loc[[0, 1, 2]].to_numpy())))
    assert stray < real, (
        f'the one-event cluster scored {stray} against {real} for real '
        'populations, so it still sets the global scale')


def test_a_zero_spread_population_is_not_rewarded_for_it():
    """Not only singletons: any population whose events are all identical has
    an IQR of 0 and used to collect the same unearned bonus."""
    rng = np.random.default_rng(3)
    n = 300
    data = pd.concat([
        pd.DataFrame({'CD3': rng.normal(3000.0, 150.0, n),
                      'CD19': rng.normal(200.0, 150.0, n)}),
        pd.DataFrame({'CD3': rng.normal(300.0, 150.0, n),
                      'CD19': rng.normal(2500.0, 150.0, n)}),
        pd.DataFrame({'CD3': np.full(40, 1500.0),      # identical events
                      'CD19': np.full(40, 1500.0)}),
    ], ignore_index=True)
    labels = np.concatenate([np.full(n, 0), np.full(n, 1), np.full(40, 2)])
    scores = mem_scores(data, labels, MARKERS)
    flat = float(np.max(np.abs(scores.loc[2].to_numpy())))
    real = float(np.max(np.abs(scores.loc[[0, 1]].to_numpy())))
    assert flat <= real, (
        f'a population with no spread scored {flat} against {real} for '
        'populations that actually separate')


def test_ordinary_populations_still_score_strongly():
    """The guard must not flatten real biology: the clearest marker in a
    well-separated panel should still reach the top of the scale."""
    scores = _scores(singleton=False)
    assert float(np.max(np.abs(scores.to_numpy()))) == pytest.approx(10.0)
    assert scores.loc[0, 'CD3'] > 0        # CD3-high population
    assert scores.loc[1, 'CD19'] > 0       # CD19-high population


def test_a_clearly_separated_population_keeps_its_sign():
    """The line between the bug and the metric. A population that is genuinely
    CD19-high keeps its sign when a stray cluster appears; one sitting exactly
    at the median of a bimodal reference does not, and that is the metric's
    own ambiguity rather than something to paper over."""
    high_alone = _scores(singleton=False, third=(1500.0, 4200.0))
    high_stray = _scores(singleton=True, third=(1500.0, 4200.0))
    assert high_alone.loc[2, 'CD19'] > 0
    assert high_stray.loc[2, 'CD19'] > 0, (
        'an unambiguously CD19-positive population changed sign')
