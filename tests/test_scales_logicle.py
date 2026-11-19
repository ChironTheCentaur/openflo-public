"""`view_funcs` against the PRODUCTION transform.

Every existing scales test used `asinh`, but `logicle` is what OpenFlo actually
bakes into loaded fluorescence channels — so the axis maths that positions
every gate and every plotted event was only ever verified on a transform the
app does not use by default. This closes that gap for `logicle` and `hyperlog`.

Auditing it found no defect: the log view round-trips exactly for positive
values, and its only error is on raw values <= 0, which a log axis cannot
represent (they are clamped at 1e-6 on purpose). These tests pin that
distinction so a future change cannot quietly widen it.
"""
import numpy as np
import pytest

from openflo.pipeline import transform_values
from openflo.scales import view_funcs

BAKED = ('logicle', 'hyperlog', 'asinh')
POSITIVE = np.array([1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 250_000.0])
WITH_NEGATIVES = np.array([-500.0, 0.0, 1.0, 100.0, 10_000.0])


@pytest.mark.parametrize('transform', BAKED)
@pytest.mark.parametrize('scale', ['log', 'symlog', 'linear'])
def test_round_trip_is_exact_for_positive_data(transform, scale):
    """forward → inverse must return the stored coordinate unchanged."""
    stored = transform_values(POSITIVE, method=transform)
    fns = view_funcs(transform, scale, data_sample=stored)
    assert fns is not None, f'{transform}/{scale} returned no view functions'
    fwd, inv = fns
    back = inv(fwd(stored))
    assert np.allclose(back, stored, rtol=0, atol=1e-9), (
        f'{transform}/{scale} round-trip drifted: '
        f'max {np.abs(back - stored).max():.3e}')


@pytest.mark.parametrize('transform', BAKED)
@pytest.mark.parametrize('scale', ['log', 'symlog', 'linear'])
def test_forward_is_strictly_increasing_on_positive_data(transform, scale):
    """Screen position must preserve data order — a non-monotonic axis would
    place gate boundaries on the wrong side of events."""
    stored = transform_values(POSITIVE, method=transform)
    fwd, _inv = view_funcs(transform, scale, data_sample=stored)
    p = np.asarray(fwd(stored), dtype=float)
    assert np.all(np.isfinite(p)), f'{transform}/{scale} produced non-finite'
    assert np.all(np.diff(p) > 0), f'{transform}/{scale} is not monotonic'


@pytest.mark.parametrize('transform', BAKED)
def test_symlog_and_linear_survive_negatives_exactly(transform):
    """Compensated data contains negatives; only the LOG view may lose them."""
    stored = transform_values(WITH_NEGATIVES, method=transform)
    for scale in ('symlog', 'linear'):
        fwd, inv = view_funcs(transform, scale, data_sample=stored)
        back = inv(fwd(stored))
        assert np.allclose(back, stored, rtol=0, atol=1e-9), (
            f'{transform}/{scale} lost precision on negative values')


@pytest.mark.parametrize('transform', BAKED)
def test_log_view_loses_only_the_non_positive_values(transform):
    """The documented, deliberate limit: a log axis cannot show <= 0, so those
    clamp. Everything positive must still be exact — if that ever stops being
    true the clamp has grown into a real bug."""
    stored = transform_values(WITH_NEGATIVES, method=transform)
    fwd, inv = view_funcs(transform, 'log', data_sample=stored)
    err = np.abs(inv(fwd(stored)) - stored)
    positive = WITH_NEGATIVES > 0
    assert np.all(err[positive] < 1e-9), (
        f'{transform}: log view is lossy for POSITIVE data — {err[positive]}')
    assert err[WITH_NEGATIVES < 0].max() > 0, (
        'expected the <=0 clamp to be visible; if it vanished, this test is '
        'stale rather than passing')


def test_linear_baked_channels_defer_to_matplotlib():
    """A linearly-stored channel returns None so the caller uses matplotlib's
    native scale (better tick locators)."""
    assert view_funcs('linear', 'log') is None
    assert view_funcs('linear', 'symlog',
                      data_sample=np.arange(10.0)) is None


@pytest.mark.parametrize('transform', BAKED)
def test_symlog_cofactor_adapts_to_the_data(transform):
    """symlog anchors its cofactor on the data's 5th percentile of |nonzero|.
    Two very different distributions must therefore not share a mapping."""
    dim = transform_values(np.linspace(1.0, 500.0, 400), method=transform)
    bright = transform_values(np.linspace(1e4, 2.5e5, 400), method=transform)
    f_dim, _ = view_funcs(transform, 'symlog', data_sample=dim)
    f_bright, _ = view_funcs(transform, 'symlog', data_sample=bright)
    probe = transform_values(np.array([1_000.0]), method=transform)
    assert not np.isclose(f_dim(probe)[0], f_bright(probe)[0]), (
        'the symlog cofactor ignored data_sample — the axis would be '
        'identical for a dim and a bright channel')
