"""The doublet window must not depend on which scale the scatter is on.

`_autoclean_doublets_mask` kept events whose FSC-A/FSC-H ratio lay within
±`tol` of the median — a fixed FRACTION OF THE MEDIAN. How far that ratio
actually scatters, though, depends on the scale the channels are on. After an
arcsinh or logicle bake the median barely moves while the spread collapses:
measured, a CV of 14.5% became 1.7%.

A ±25% window then swallowed the doublet population whole. The same 4,000
planted doublets that were all removed on linear scatter were ALL RETAINED on
transformed scatter — the filter became a silent no-op, failing open, on data
that had merely been transformed first.

This is the same shape as the median-ratio bug already fixed in
`filter_doublets`: a window derived from a scale-dependent quantity. The band
is now also bounded by the ratio's own robust spread, which is what
`auto_singlet_gate` already used. `tol` keeps its meaning as the widest the
window may be, and the dispersion can only narrow it.

The bound is generous on purpose, and the residual is stated rather than
hidden: at 5 robust SDs every doublet went, but 0.2% of genuine singlets went
with them on the golden sample — a real accuracy loss on ordinary linear data,
which the golden baseline caught. At 10 the linear result is exactly what it
always was, and the transformed case goes from all 4,000 doublets retained to
roughly 3%.

Separately, a non-positive median ratio inverted the window (lo > hi), which
matches nothing — the whole sample would have been deleted.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from openflo.pipeline import _autoclean_doublets_mask, transform_values

N_SINGLET = 46_000
N_DOUBLET = 4_000


@pytest.fixture
def scatter():
    """Singlets at FSC-H ≈ FSC-A/2, doublets at FSC-A/3.1 — as the shipped
    generator builds them."""
    rng = np.random.default_rng(0)
    area = np.concatenate([rng.normal(60_000.0, 8_000.0, N_SINGLET),
                           rng.normal(118_000.0, 12_000.0, N_DOUBLET)])
    height = np.concatenate([
        area[:N_SINGLET] / 2.0 * rng.normal(1.0, 0.02, N_SINGLET),
        area[N_SINGLET:] / 3.1 * rng.normal(1.0, 0.04, N_DOUBLET)])
    return area, height


def _kept(area, height, params=None):
    df = pd.DataFrame({'FSC-A': area, 'FSC-H': height})
    return np.asarray(_autoclean_doublets_mask(df, params or {}), dtype=bool)


@pytest.mark.parametrize('method, kwargs', [
    pytest.param(None, {}, id='linear'),
    pytest.param('asinh', {'cofactor': 150.0}, id='arcsinh'),
    pytest.param('logicle', {}, id='logicle'),
])
def test_the_doublets_are_removed_on_every_scale(scatter, method, kwargs):
    area, height = scatter
    if method is not None:
        area = transform_values(area, method=method, **kwargs)
        height = transform_values(height, method=method, **kwargs)

    mask = _kept(area, height)
    doublets_kept = int(mask[N_SINGLET:].sum())
    # Not zero on every scale, and this says so rather than pretending. The
    # bound is deliberately generous — at 5 robust SDs it removed every
    # doublet but also trimmed 0.2% of genuine singlets from the golden
    # sample, a real accuracy loss on ordinary linear data. At 10 it leaves
    # linear results exactly as they were and still turns "all 4,000
    # doublets retained" into a residual of ~3%.
    assert doublets_kept <= N_DOUBLET * 0.05, (
        f'{doublets_kept} of {N_DOUBLET} planted doublets survived — the '
        'filter is close to a no-op on this scale')


@pytest.mark.parametrize('method, kwargs', [
    pytest.param(None, {}, id='linear'),
    pytest.param('asinh', {'cofactor': 150.0}, id='arcsinh'),
    pytest.param('logicle', {}, id='logicle'),
])
def test_the_singlets_are_kept_on_every_scale(scatter, method, kwargs):
    """Narrowing the band must not start cutting real cells."""
    area, height = scatter
    if method is not None:
        area = transform_values(area, method=method, **kwargs)
        height = transform_values(height, method=method, **kwargs)

    mask = _kept(area, height)
    singlets_kept = int(mask[:N_SINGLET].sum())
    assert singlets_kept >= N_SINGLET * 0.99, (
        f'only {singlets_kept} of {N_SINGLET} singlets survived')


def test_the_verdict_is_the_same_before_and_after_a_transform(scatter):
    """The invariant, stated directly: transforming the scatter first must not
    change which events are called doublets."""
    area, height = scatter
    linear = _kept(area, height)
    baked = _kept(transform_values(area, method='asinh', cofactor=150.0),
                  transform_values(height, method='asinh', cofactor=150.0))
    disagree = int((linear != baked).sum())
    assert disagree <= len(linear) * 0.01, (
        f'{disagree} events changed verdict purely because the scatter was '
        'transformed')


def test_a_non_positive_median_ratio_skips_rather_than_deleting(scatter,
                                                                caplog):
    """The window is multiplicative, so a non-positive median inverts it and
    matches nothing — the whole sample would go."""
    area, height = scatter
    with caplog.at_level(logging.WARNING):
        mask = _kept(-np.abs(area), np.abs(height))
    assert int(mask.sum()) == len(mask), (
        f'{len(mask) - int(mask.sum())} events were deleted by an inverted '
        'acceptance window')
    assert any('doublet filter skipped' in r.getMessage().lower()
               for r in caplog.records)


def test_tol_still_bounds_the_window(scatter):
    """`tol` keeps its meaning: the dispersion may narrow the band, never
    widen it, so a tiny tol is still respected."""
    area, height = scatter
    wide = _kept(area, height, {'tol': 0.25})
    narrow = _kept(area, height, {'tol': 0.001})
    assert int(narrow.sum()) < int(wide.sum())


def test_a_zero_tolerance_is_still_a_no_op(scatter):
    area, height = scatter
    assert int(_kept(area, height, {'tol': 0.0}).sum()) == len(area)
