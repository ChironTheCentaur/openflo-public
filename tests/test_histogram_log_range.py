"""A histogram on a log x axis spans the data's decades, not a thousand.

The histogram's x range is the union of the samples' robust ranges padded by
2% of the span. On matplotlib's own log axis (a linear-stored channel, such
as scatter, shown on Log) that pad pushed the lower limit to zero or below
whenever the data reached near zero, and the limit was set as it was: a log
axis reads a non-positive limit as 10^-1000, so the curve was a sliver at
the right edge of an axis about a thousand decades wide. The bins already
started at a floor six decades below the top; the axis now starts where the
bins do, and a range that is positive throughout is padded in decades so it
stays positive.

Driven through the editor's real _replot. Generated data only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.test_gui_smoke import _editor_or_skip

N = 20_000
CH = 'SSC-A'


def _load(ed, name, values, transform='linear'):
    from types import SimpleNamespace

    from openflo.channel_transforms import transform_spec
    df = pd.DataFrame({CH: values})
    ed._samples[name] = SimpleNamespace(
        name=name, path=f'C:/x/{name}.fcs', data=df, fluor_channels=[],
        channel_labels={}, data_transforms={CH: transform_spec(transform)})
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    ed._sample_plot_enabled[name] = True
    ed._sample_gates[name] = {}
    ed._sample_gate_order[name] = []
    ed._sample_gate_seq[name] = 0


def _draw(values, scale='log', transform='linear'):
    """The x limits and x scale of a histogram of ``values`` drawn by
    ``_replot`` with the channel on ``scale``."""
    root, ed, _gui = _editor_or_skip()
    try:
        ed._plotcache_warm_enabled = False
        _load(ed, 'a', values, transform)
        ed._channels = [CH]
        ed._channel_labels = {CH: CH}
        ed._channel_transform[CH] = transform
        ed._populate_channel_combos()
        ed._set_active_sample('a')
        ed.x_combo.set(ed._fmt_channel(CH))
        ed.mode_var.set('histogram')
        ed.gate_display_var.set('all')
        ed._channel_scale[CH] = scale
        ed._replot()
        return ed.ax.get_xlim(), ed.ax.get_xscale()
    finally:
        try:
            root.destroy()
        except Exception:  # noqa: BLE001 - teardown; the root may already be destroyed, and a teardown error must not hide the test's own result
            pass


@pytest.fixture
def near_zero():
    """A third of the events at or below zero (a compensated channel)."""
    rng = np.random.default_rng(0)
    return np.concatenate([rng.normal(-300.0, 600.0, N // 3),
                           rng.normal(500.0, 900.0, N // 3),
                           rng.normal(9_000.0, 3_000.0, N - 2 * (N // 3))])


@pytest.fixture
def positive():
    """Every event positive, from about 10 to about 50,000."""
    rng = np.random.default_rng(1)
    return rng.lognormal(np.log(700.0), 1.3, N)


def test_a_range_reaching_zero_starts_at_the_bins_floor(near_zero):
    (lo, hi), scale = _draw(near_zero)
    assert scale == 'log'
    assert lo > 0, f'a log axis cannot start at {lo}'
    # The bins' floor: six decades below the top (plotmath.hist_bin_edges).
    assert np.log10(hi / lo) == pytest.approx(6.0, abs=0.05)


def test_a_positive_range_is_padded_in_decades(positive):
    (lo, hi), _scale = _draw(positive)
    p_lo, p_hi = np.percentile(positive, (0.1, 99.9))
    span = np.log10(p_hi / p_lo)
    # 2% of the span in decades each side: the axis starts just below the
    # data, not six decades below its top.
    assert np.log10(p_lo / lo) == pytest.approx(0.02 * span, rel=1e-6)
    assert np.log10(hi / p_hi) == pytest.approx(0.02 * span, rel=1e-6)


@pytest.mark.parametrize('data', ['near_zero', 'positive'])
def test_a_linear_axis_keeps_its_linear_pad(data, request):
    values = request.getfixturevalue(data)
    (lo, hi), scale = _draw(values, scale='linear')
    assert scale == 'linear'
    p_lo, p_hi = np.percentile(values, (0.1, 99.9))
    pad = (p_hi - p_lo) * 0.02
    assert (lo, hi) == pytest.approx((p_lo - pad, p_hi + pad))


def test_a_transformed_channel_on_log_keeps_its_stored_range(positive):
    """A channel stored on logicle and shown on Log is drawn through a
    function scale on the stored values: its range is padded as stored
    (linearly), not in decades of the stored number."""
    from openflo.channel_transforms import transform_values
    stored = transform_values(positive, method='logicle')
    (lo, hi), scale = _draw(stored, transform='logicle')
    assert scale == 'function'
    p_lo, p_hi = np.percentile(stored, (0.1, 99.9))
    pad = (p_hi - p_lo) * 0.02
    assert (lo, hi) == pytest.approx((p_lo - pad, p_hi + pad))


def test_the_warm_up_counts_serve_a_log_histogram(positive, monkeypatch):
    """The background warm-up bins on the same padded range as the plot, so
    a log histogram it warmed is drawn without binning again."""
    from tests._tk_wait import pump_until
    root, ed, _gui = _editor_or_skip()
    try:
        ed._plotcache_warm_enabled = False
        _load(ed, 'a', positive)
        _load(ed, 'b', positive[::-1].copy())
        ed._sample_plot_enabled['b'] = False
        ed._channels = [CH]
        ed._channel_labels = {CH: CH}
        ed._populate_channel_combos()
        ed._set_active_sample('a')
        ed.x_combo.set(ed._fmt_channel(CH))
        ed.mode_var.set('histogram')
        ed.gate_display_var.set('all')
        ed._channel_scale[CH] = 'log'
        ed._replot()
        ed._plotcache_warm_enabled = True
        ed._plotcache_closed = False
        ed._warm_schedule(0)
        ed.update()
        assert pump_until(ed, lambda: (
            getattr(ed, '_plotcache_inflight', None) is None
            and not getattr(ed, '_plotcache_warm_names', None)
            and getattr(ed, '_plotcache_sched_after', None) is None),
            20.0, pause=0.005), 'warm-up did not finish'
        seen = []
        real = np.histogram
        monkeypatch.setattr(np, 'histogram',
                            lambda *a, **k: seen.append(1) or real(*a, **k))
        ed._sample_plot_enabled['a'] = False
        ed._sample_plot_enabled['b'] = True
        ed._replot()
        assert seen == [], 'the warmed counts were not used'
    finally:
        try:
            root.destroy()
        except Exception:  # noqa: BLE001 - teardown; the root may already be destroyed, and a teardown error must not hide the test's own result
            pass
