"""Binned Gaussian-KDE density (Preferences > Performance, off by default).

openflo.density.binned_kde_eval approximates scipy's exact gaussian_kde (same
kernel, same Scott bandwidth) on a grid. These tests check it against scipy on
generated events of several shapes, check that nothing changes while the
preference is off, and check that the two places that compute the exact KDE
(the editor's 'True Gaussian KDE' option and Quick preview) follow the
preference. Every expected density is scipy's own exact value.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import gaussian_kde

from openflo.density import binned_kde_eval, kde_density
from tests.conftest import editor_or_skip

# The binned form promises to stay within 0.5% of the exact density's peak.
PEAK_TOL = 5e-3
# Plots colour density on a gamma-0.4 power scale (plotmath.density_norm),
# which stretches low densities; half a step of a 256-colour map is ~2e-3.
COLOUR_TOL = 2e-3


def _colour(z):
    """Position on the colour map, as plotmath.density_norm maps it."""
    from openflo.plotmath import density_norm
    return np.asarray(density_norm(z)(z))


def _exact(xs, ys, xq, yq):
    return gaussian_kde(np.vstack([xs, ys]))(np.vstack([xq, yq]))


def _shapes():
    rng = np.random.default_rng(0)
    yield 'one normal blob', rng.normal(0, 1, (6000, 2))
    yield 'blobs, a tilted one and sparse noise', np.vstack([
        rng.normal([0, 0], 0.3, (2500, 2)),
        rng.multivariate_normal([3, 2], [[1, 0.8], [0.8, 1]], 2500),
        rng.normal([-2, 4], 0.1, (800, 2)),
        rng.uniform(-6, 8, (400, 2))])
    yield 'skewed, on a linear scale', np.column_stack([
        rng.lognormal(6, 1.2, 6000), rng.lognormal(5, 1.5, 6000)])
    yield 'almost collinear', rng.multivariate_normal(
        [0, 0], [[1, 0.99], [0.99, 1]], 6000)


@pytest.mark.parametrize('name,pts', list(_shapes()),
                         ids=[n for n, _ in _shapes()])
def test_binned_matches_exact_kde(name, pts):
    xs, ys = pts[:, 0], pts[:, 1]
    q = np.random.default_rng(1).choice(xs.size, 1500, replace=False)
    ref = _exact(xs, ys, xs[q], ys[q])
    z = binned_kde_eval(xs, ys, xs[q], ys[q])
    assert z is not None and z.shape == ref.shape
    assert np.abs(z - ref).max() <= PEAK_TOL * ref.max(), name
    assert np.abs(_colour(z) - _colour(ref)).max() <= COLOUR_TOL, name
    # Typical points are far closer than the worst one.
    assert np.median(np.abs(z - ref) / ref) < 2e-3, name


def test_binned_handles_a_far_outlier_and_points_off_the_grid():
    """One event 1000 SD out would stretch the grid past its cell limit, so the
    grid covers the central events and the rest are summed exactly. The query
    set includes the outlier and a point beyond every event."""
    rng = np.random.default_rng(2)
    pts = np.vstack([rng.normal(0, 1, (5999, 2)), [[1e3, 1e3]]])
    xs, ys = pts[:, 0], pts[:, 1]
    xq = np.concatenate([xs[:1000], [1e3, -60.0]])
    yq = np.concatenate([ys[:1000], [1e3, 1100.0]])
    ref = _exact(xs, ys, xq, yq)
    z = binned_kde_eval(xs, ys, xq, yq)
    assert z is not None
    assert np.abs(z - ref).max() <= PEAK_TOL * ref.max()
    # Off the grid the sum is exact, so the outlier's own density matches.
    assert z[1000] == pytest.approx(ref[1000], rel=1e-9)
    assert z[1001] == pytest.approx(ref[1001], rel=1e-9, abs=1e-300)


def test_binned_sees_events_beyond_a_trimmed_grid():
    """A sparse tail plus one far event: the grid is trimmed to the central
    events, and tail events just past its edge must still add their density
    to queries inside the edge (and to queries past it). The queries run along
    the tail out to the far event."""
    rng = np.random.default_rng(3)
    tail = [20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 400.0]
    pts = np.vstack([rng.normal(0, 1, (6000 - len(tail), 2)),
                     np.column_stack([tail, np.zeros(len(tail))])])
    xs, ys = pts[:, 0], pts[:, 1]
    xq = np.concatenate([xs, np.linspace(0, 410, 3000)])
    yq = np.concatenate([ys, np.zeros(3000)])
    ref = _exact(xs, ys, xq, yq)
    z = binned_kde_eval(xs, ys, xq, yq)
    assert z is not None
    assert np.abs(z - ref).max() <= PEAK_TOL * ref.max()
    assert np.abs(_colour(z) - _colour(ref)).max() <= COLOUR_TOL


@pytest.mark.parametrize('xs,ys', [
    (np.ones(50), np.arange(50.0)),               # constant channel
    (np.arange(50.0), 2 * np.arange(50.0) + 1),   # exactly collinear
    (np.array([1.0, 2.0]), np.array([3.0, 1.0])),  # too few events
    (np.array([]), np.array([])),                  # no events
], ids=['constant', 'collinear', 'two events', 'empty'])
def test_binned_returns_none_where_scipy_cannot_fit(xs, ys):
    assert binned_kde_eval(xs, ys, xs, ys) is None


def test_kde_density_is_exact_unless_binned_is_asked_for():
    rng = np.random.default_rng(4)
    xs = rng.normal(0, 1, 3000)
    ys = 0.5 * xs + rng.normal(0, 1, 3000)
    xs_d, ys_d, z, n = kde_density(xs, ys, max_src=1000, max_display=2000)
    xb, yb, zb, nb = kde_density(xs, ys, max_src=1000, max_display=2000,
                                 binned=True)
    # Same subsample, same points drawn: only the evaluation differs.
    assert n == nb == 1000
    assert np.array_equal(xs_d, xb) and np.array_equal(ys_d, yb)
    src = np.random.default_rng(42).choice(3000, 1000, replace=False)
    ref = _exact(xs[src], ys[src], xs_d, ys_d)
    np.testing.assert_allclose(z, ref, rtol=1e-12)          # default: exact
    assert not np.array_equal(zb, ref)                      # binned: not exact
    assert np.abs(zb - ref).max() <= PEAK_TOL * ref.max()   # ...but close


def test_kde_density_binned_falls_back_to_flat_on_a_constant_channel():
    xs = np.full(200, 3.0)
    ys = np.random.default_rng(5).normal(0, 1, 200)
    _xd, _yd, z, _n = kde_density(xs, ys, binned=True)
    assert np.array_equal(z, np.ones(200))


def _density_colours(binned, monkeypatch, exact_raises=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from openflo import preview
    if exact_raises:
        def _boom(*_a, **_k):
            raise AssertionError('exact KDE should not run')
        monkeypatch.setattr('scipy.stats.gaussian_kde', _boom)
    rng = np.random.default_rng(6)
    x = np.concatenate([rng.normal(0, 1, 2000), rng.normal(4, 0.5, 1000)])
    y = np.concatenate([rng.normal(0, 1, 2000), rng.normal(3, 0.5, 1000)])
    fig, ax = plt.subplots()
    try:
        if binned is None:                     # the library's own default
            preview.density_scatter(ax, x, y)
        else:
            preview.density_scatter(ax, x, y, binned=binned)
        coll = ax.collections[0]
        arr = coll.get_array()
        return x, y, (None if arr is None else np.asarray(arr))
    finally:
        plt.close(fig)


def test_preview_density_scatter_default_is_exact(monkeypatch):
    x, y, colours = _density_colours(None, monkeypatch)
    ref = np.sort(_exact(x, y, x, y))
    np.testing.assert_allclose(colours, ref, rtol=1e-12)


def test_preview_density_scatter_binned_is_close_and_skips_the_exact_kde(
        monkeypatch):
    x, y, _ = _density_colours(False, monkeypatch)
    ref = np.sort(_exact(x, y, x, y))
    _x, _y, colours = _density_colours(True, monkeypatch, exact_raises=True)
    assert colours is not None                     # coloured, not the fallback
    assert np.abs(colours - ref).max() <= PEAK_TOL * ref.max()


def test_preview_density_scatter_binned_falls_back_to_the_exact_kde(
        monkeypatch):
    """Where the binned form cannot fit (it returns None) but scipy can, the
    scatter keeps scipy's exact colours rather than drawing flat."""
    import openflo.density
    x, y, _ = _density_colours(False, monkeypatch)
    ref = np.sort(_exact(x, y, x, y))
    monkeypatch.setattr(openflo.density, 'binned_kde_eval',
                        lambda *_a, **_k: None)
    _x, _y, colours = _density_colours(True, monkeypatch)
    assert colours is not None
    np.testing.assert_allclose(colours, ref, rtol=1e-12)


# ── GUI: the preference, its hover tip, and the two places that read it ──────

def test_preferences_checkbox_is_off_by_default_and_saves(monkeypatch):
    from openflo.prefs import load_density_binned, read_prefs
    from openflo.ui_preferences import PreferencesDialog
    root, ed = editor_or_skip()
    try:
        tips = {}
        monkeypatch.setattr(ed, '_tip', lambda w, t: tips.__setitem__(w, t),
                            raising=False)
        replots = []
        monkeypatch.setattr(ed, '_replot', lambda: replots.append(1))
        assert 'density_binned' not in read_prefs()
        dlg = PreferencesDialog(ed)
        root.update()
        cb = dlg.density_binned_cb
        assert dlg._density_binned_var.get() is False
        text = tips[cb]
        assert 'accuracy' in text and 'default' in text and '0.5%' in text
        cb.invoke()
        assert load_density_binned() is True and replots == [1]
        cb.invoke()
        assert load_density_binned() is False and replots == [1, 1]
        dlg.destroy()
        from openflo.prefs import write_pref
        write_pref('density_binned', True)
        dlg = PreferencesDialog(ed)                 # reopened: remembers it
        assert dlg._density_binned_var.get() is True
        dlg.destroy()
    finally:
        root.destroy()


@pytest.mark.parametrize('on', [False, True])
def test_true_kde_plot_option_follows_the_preference(on, monkeypatch):
    from openflo import density
    from openflo.prefs import write_pref
    root, ed = editor_or_skip()
    try:
        write_pref('density_binned', on)
        seen = []
        real = density.kde_density

        def _spy(xs, ys, **kw):
            seen.append(kw.get('binned', False))
            # Small caps keep the exact run quick; still below the 20k events
            # drawn, so the status line reports the subsample.
            return real(xs, ys, max_src=300, max_display=600, **kw)
        monkeypatch.setattr(density, 'kde_density', _spy)
        rng = np.random.default_rng(7)
        xs, ys = rng.normal(0, 1, 20000), rng.normal(0, 1, 20000)
        ed._density_scatter_truekde(xs, ys)
        assert seen == [on]
        status = ed.status_var.get()
        assert status.startswith('Binned KDE' if on else 'True KDE')
    finally:
        root.destroy()


def _fcs(path, seed=0):
    import pandas as pd

    from openflo.fcs_export import write_fcs
    rng = np.random.default_rng(seed)
    x = rng.normal(500, 100, 1500)
    write_fcs(pd.DataFrame({'FL1-A': x, 'FL2-A': x + rng.normal(0, 80, 1500)}),
              str(path))


@pytest.mark.parametrize('on', [False, True])
def test_quick_preview_follows_the_preference(on, tmp_path, monkeypatch):
    """With the exact KDE made to fail, the binned preference still colours the
    scatter by density; without it the dialog falls back to a flat scatter."""
    from openflo.prefs import write_pref
    from openflo.ui_preview import QuickPreviewDialog
    root, ed = editor_or_skip()
    try:
        write_pref('density_binned', on)

        def _boom(*_a, **_k):
            raise RuntimeError('exact KDE disabled for this test')
        monkeypatch.setattr('scipy.stats.gaussian_kde', _boom)
        path = tmp_path / 'sample_1.fcs'
        _fcs(path)
        dlg = QuickPreviewDialog(ed)
        dlg._load_sample(str(path))
        dlg._x_combo.set('FL1-A')
        dlg._y_combo.set('FL2-A')
        dlg._plot()
        arr = dlg._ax.collections[0].get_array()
        assert (arr is not None) is on
        dlg.destroy()
    finally:
        root.destroy()
