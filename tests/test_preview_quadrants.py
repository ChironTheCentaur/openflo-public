"""A missing FMO threshold must not become a threshold at -999.

`preview.py` substituted -999 for an absent per-axis FMO threshold before
handing the value to the quadrant annotation. That sentinel is not internal:
the annotation renders four percentages onto the figure, so a plot with only
one real threshold still displayed a full quadrant breakdown, two of whose
numbers were defined by a boundary nobody measured.

The sentinel is also not safely out of range. Compensated fluorescence is
routinely negative — a dim population scattering around -2000 is ordinary —
and there -999 falls INSIDE the data. Measured on such a channel, the four
quadrant percentages came back 67.9 / 7.0 / 23.2 / 2.0: arbitrary numbers with
every appearance of a real gating result.

`None` now means "this axis has no threshold", and only the axis that has one
is annotated.
"""
import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from openflo.preview import annotate_quadrants  # noqa: E402


@pytest.fixture
def dim_negative():
    """A compensated channel whose population sits well below -999."""
    rng = np.random.default_rng(0)
    n = 20_000
    return pd.DataFrame({'CD11b': rng.normal(-2000.0, 1500.0, n),
                         'CD34': rng.normal(-2000.0, 1500.0, n)})


@pytest.fixture
def bimodal():
    rng = np.random.default_rng(1)
    n = 20_000
    neg, pos = n // 2, n // 2
    return pd.DataFrame({
        'CD11b': np.concatenate([rng.normal(-200.0, 300.0, neg),
                                 rng.normal(4000.0, 800.0, pos)]),
        'CD34': np.concatenate([rng.normal(-150.0, 250.0, neg),
                                rng.normal(3000.0, 600.0, pos)])})


def _labels(df, xth, yth):
    fig, ax = plt.subplots()
    try:
        ax.set_xlim(-6000, 6000)
        ax.set_ylim(-6000, 6000)
        annotate_quadrants(ax, df, 'CD11b', 'CD34', xth, yth)
        return [t.get_text() for t in ax.texts]
    finally:
        plt.close(fig)


def test_one_missing_threshold_does_not_draw_four_quadrants(bimodal):
    """Two of those four numbers would be defined by a boundary nobody set."""
    assert len(_labels(bimodal, 1200.0, None)) == 2
    assert len(_labels(bimodal, None, 900.0)) == 2


def test_both_thresholds_still_draw_four_quadrants(bimodal):
    """The guard must not disable the annotation when it IS meaningful."""
    labels = _labels(bimodal, 1200.0, 900.0)
    assert len(labels) == 4
    total = sum(float(t.rstrip('%')) for t in labels)
    assert total == pytest.approx(100.0, abs=0.2)


def test_no_threshold_at_all_draws_nothing(bimodal):
    assert _labels(bimodal, None, None) == []


def test_the_single_axis_split_sums_to_one_hundred(bimodal):
    for xth, yth in ((1200.0, None), (None, 900.0)):
        labels = _labels(bimodal, xth, yth)
        total = sum(float(t.rstrip('%')) for t in labels)
        assert total == pytest.approx(100.0, abs=0.2), (
            f'a one-axis split reported {total}%, not a whole sample')


def test_data_below_the_old_sentinel_is_not_split_by_it(dim_negative):
    """The case that makes the sentinel indefensible: -999 sits inside this
    channel's range, so it silently acted as a real threshold."""
    labels = _labels(dim_negative, 0.0, None)
    assert len(labels) == 2

    # What the sentinel produced: a four-way split of a single population,
    # driven entirely by the fabricated -999 boundary.
    below = float((dim_negative['CD34'] > -999).mean())
    assert 0.05 < below < 0.95, (
        'this fixture no longer straddles -999, so it cannot demonstrate the '
        'bug — pick a distribution that does')


def test_an_empty_sample_annotates_nothing(bimodal):
    assert _labels(bimodal.iloc[:0], 1200.0, 900.0) == []
