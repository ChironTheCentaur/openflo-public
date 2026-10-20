"""One bad event must not destroy a whole sample's unmixing.

`unmix` solves every event in a single least-squares factorisation, so a single
non-finite detector reading contaminated the entire solution: one `inf` event
turned ALL events' abundances into NaN. (A `NaN` event happened to stay
contained — an accident of how the SVD propagates, not a guarantee.) Losing a
sample's unmixing to one saturated reading is silent data corruption.

v2.2.3 hardened the REFERENCE spectra against non-finite values; the event data
had no matching guard. These tests pin the containment and the exactness of the
events that were fine.
"""
import numpy as np
import pytest

from openflo.spectral import unmix

N_DET, N_FLUOR, N_EVENTS = 8, 4, 300


@pytest.fixture
def fixture():
    rng = np.random.default_rng(0)
    S = rng.random((N_FLUOR, N_DET)) + 0.1
    true = rng.random((N_EVENTS, N_FLUOR)) * 1000.0
    return S, true, true @ S


def test_clean_data_unmixes_exactly(fixture):
    S, true, Y = fixture
    np.testing.assert_allclose(unmix(Y, S), true, rtol=0, atol=1e-8)


@pytest.mark.parametrize('bad', [np.inf, -np.inf, np.nan])
def test_one_bad_event_is_contained(fixture, bad):
    """The whole point: the blast radius is one event, not the sample."""
    S, true, Y = fixture
    Yb = Y.copy()
    Yb[3, 2] = bad
    A = unmix(Yb, S)

    non_finite = (~np.isfinite(A)).any(axis=1)
    assert non_finite.sum() == 1, (
        f'a single {bad} reading left {int(non_finite.sum())} of {len(A)} '
        'events unmixable — one bad event destroyed the sample')
    assert non_finite[3], 'the wrong event was invalidated'


@pytest.mark.parametrize('bad', [np.inf, np.nan])
def test_the_good_events_are_still_exact(fixture, bad):
    """Holding bad rows out must not perturb the rest of the solve."""
    S, true, Y = fixture
    Yb = Y.copy()
    Yb[10, 0] = bad
    A = unmix(Yb, S)
    ok = np.isfinite(A).all(axis=1)
    np.testing.assert_allclose(A[ok], true[ok], rtol=0, atol=1e-8)


def test_many_bad_events_still_leave_the_rest_usable(fixture):
    S, true, Y = fixture
    Yb = Y.copy()
    victims = [1, 5, 50, 120, 299]
    for i in victims:
        Yb[i, i % N_DET] = np.inf
    A = unmix(Yb, S)
    bad_rows = set(np.flatnonzero((~np.isfinite(A)).any(axis=1)))
    assert bad_rows == set(victims)
    ok = np.isfinite(A).all(axis=1)
    np.testing.assert_allclose(A[ok], true[ok], rtol=0, atol=1e-8)


def test_an_all_bad_sample_returns_all_nan_without_raising(fixture):
    S, _true, Y = fixture
    A = unmix(np.full_like(Y, np.inf), S)
    assert A.shape == (N_EVENTS, N_FLUOR)
    assert np.isnan(A).all(), 'expected NaN, not a crash or fabricated values'


def test_non_finite_reference_spectra_raise_a_clear_error(fixture):
    """numpy's own failure here is 'SVD did not converge', which says nothing
    about single-stain controls."""
    S, _true, Y = fixture
    Sb = S.copy()
    Sb[0, 0] = np.nan
    with pytest.raises(ValueError, match='reference spectra'):
        unmix(Y, Sb)


def test_nonneg_clipping_still_applies_with_bad_events(fixture):
    S, _true, Y = fixture
    Yb = Y.copy()
    Yb[0, 0] = np.inf
    A = unmix(Yb - 5_000.0, S, nonneg=True)      # force negatives
    finite = np.isfinite(A)
    assert (A[finite] >= 0).all(), 'nonneg clipping was skipped'
