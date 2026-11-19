"""'Highlight gated' on a histogram draws each gate's curve on the Y axis the
histogram uses, as FlowJo draws a gated population over its parent: Count is
the gate's events per bin, % of Max puts the gate's own tallest bin at 100,
Fraction is the gate's events per bin over the gate's events.

The curve was always a fraction, so on Count or % of Max it lay flat on the
bottom of an axis in the hundreds. And it was binned from the scatter
modes' capped subsample ('Max points'), which a histogram does not use, so
on Count it would have counted only the subsample.

Generated data: s1 with FSC-A, FL1-A, FL2-A; FL1-A ~ N(0.5, 0.2) with 20
events at +inf (no place on any axis). Gates are thresholds on FL1-A.
"""
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from tests.test_fmo_gating_dialog import _editor_or_skip, _register_fl

MODES = ('Fraction', 'Count', '% of Max')


def _frame(n, seed=7):
    rng = np.random.default_rng(seed)
    fl1 = rng.normal(0.5, 0.2, n)
    fl1[:20] = np.inf
    return pd.DataFrame({'FSC-A': rng.normal(0.6, 0.1, n),
                         'FL1-A': fl1,
                         'FL2-A': rng.normal(0.3, 0.05, n)})


def _setup(ed, n, cut):
    df = _frame(n)
    _register_fl(ed, 's1', df)
    ed._set_active_sample('s1')
    ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A', 'value': cut,
                  'name': 'FL1+', 'parent_id': None, 'enabled': True,
                  'color': '#d62728'})
    ed._channel_scale['FL1-A'] = 'linear'
    ed.gate_display_var.set('highlight')
    return df


def _curves(ed, ymode):
    """``(sample curve, gate curve)`` y data drawn for ``ymode``."""
    ed.hist_y_mode.set(ymode)
    ax = Figure().add_subplot()
    ed._render_into(ax, ['s1'], 'FL1-A', None, 'histogram', None,
                    population=None)
    lines = {ln.get_label(): np.asarray(ln.get_ydata(), dtype=float)
             for ln in ax.get_lines()}
    gate = [v for k, v in lines.items() if k.startswith('gate ')]
    assert len(gate) == 1, sorted(lines)
    sample = [v for k, v in lines.items()
              if not k.startswith(('gate ', '_'))]
    assert len(sample) == 1, sorted(lines)
    return sample[0], gate[0]


@pytest.mark.parametrize('n', [3000, 300])
@pytest.mark.parametrize('ymode', MODES)
def test_a_gate_holding_every_event_draws_over_the_sample(ymode, n):
    """Same events, same bins, same smoothing: the same curve, whatever the
    Y axis is; 300 events is sparse enough for a kernel wider than one bin."""
    root, ed = _editor_or_skip()
    try:
        _setup(ed, n, cut=-10.0)
        sample, gate = _curves(ed, ymode)
        np.testing.assert_allclose(gate, sample, rtol=1e-9, atol=1e-12)
    finally:
        root.destroy()


def test_count_counts_the_gates_events():
    """About a third of the events pass FL1-A >= 0.6; the Count curve holds
    that many events, under the sample's curve."""
    root, ed = _editor_or_skip()
    try:
        df = _setup(ed, 3000, cut=0.6)
        inside = int(np.isfinite(df['FL1-A']).sum()
                     - (df['FL1-A'] < 0.6).sum())
        assert 500 < inside < 2500
        sample, gate = _curves(ed, 'Count')
        # 2%: tail events beyond the plotted range are not binned
        assert gate.sum() == pytest.approx(inside, rel=0.02)
        assert gate.max() > 20                 # not flat on the bottom
        assert np.all(gate <= sample + 1e-9)
    finally:
        root.destroy()


def test_percent_of_max_puts_the_gates_peak_at_100():
    root, ed = _editor_or_skip()
    try:
        _setup(ed, 3000, cut=0.6)
        sample, gate = _curves(ed, '% of Max')
        assert sample.max() == pytest.approx(100.0)
        assert gate.max() == pytest.approx(100.0)
    finally:
        root.destroy()


def test_fraction_is_of_the_gates_own_events():
    root, ed = _editor_or_skip()
    try:
        _setup(ed, 3000, cut=0.6)
        sample, gate = _curves(ed, 'Fraction')
        # 2%: tail events beyond the plotted range are not binned
        assert sample.sum() == pytest.approx(1.0, rel=0.02)
        # 2%: tail events beyond the plotted range are not binned
        assert gate.sum() == pytest.approx(1.0, rel=0.02)
    finally:
        root.destroy()


def test_the_gate_curve_ignores_the_scatter_point_cap():
    """Max points caps the scatter modes only. With 5,000 events and a cap
    of 1,000, the gate's Count curve still counts every gated event, as the
    sample's curve counts every event."""
    root, ed = _editor_or_skip()
    try:
        df = _setup(ed, 5000, cut=0.6)
        ed.ds_display_var.set(True)
        ed.max_points_var.set('1000')
        ed._mp_last_good = '1000'
        assert ed._display_point_cap() == 1000
        inside = int(np.isfinite(df['FL1-A']).sum()
                     - (df['FL1-A'] < 0.6).sum())
        sample, gate = _curves(ed, 'Count')
        # 2%: tail events beyond the plotted range are not binned
        assert sample.sum() == pytest.approx(len(df) - 20, rel=0.02)
        # 2%: tail events beyond the plotted range are not binned
        assert gate.sum() == pytest.approx(inside, rel=0.02)
    finally:
        root.destroy()


def test_a_full_density_gate_counts_the_histograms_events():
    """Full density (the density column) is for the scatter modes. On a
    histogram downsampled to the smallest sample (1,500 events), a gate
    holding every event is the sample's curve, not all 5,000 events."""
    root, ed = _editor_or_skip()
    try:
        _setup(ed, 5000, cut=-10.0)
        _register_fl(ed, 's2', _frame(1500, seed=8))
        ed.ds_display_var.set(True)
        ed._gate_density_full.add(('s1', next(iter(ed._sample_gates['s1']))))
        sample, gate = _curves(ed, 'Count')
        # 2%: tail events beyond the plotted range are not binned
        assert sample.sum() == pytest.approx(1500, rel=0.02)
        np.testing.assert_allclose(gate, sample, rtol=1e-9, atol=1e-12)
    finally:
        root.destroy()


@pytest.mark.parametrize('scale', ['linear', 'log', 'symlog'])
def test_the_gate_curve_sits_on_the_histograms_bins(scale):
    """The gate's curve is drawn at the histogram's bin centres on every
    display scale (symlog's bins depend on the data they are built from)."""
    root, ed = _editor_or_skip()
    try:
        _setup(ed, 3000, cut=0.6)
        ed._channel_scale['FL1-A'] = scale
        ed.hist_y_mode.set('Count')
        ax = Figure().add_subplot()
        ed._render_into(ax, ['s1'], 'FL1-A', None, 'histogram', None,
                        population=None)
        xs = {ln.get_label(): np.asarray(ln.get_xdata(), dtype=float)
              for ln in ax.get_lines()}
        gate = [v for k, v in xs.items() if k.startswith('gate ')]
        sample = [v for k, v in xs.items()
                  if not k.startswith(('gate ', '_'))]
        assert len(gate) == len(sample) == 1, sorted(xs)
        np.testing.assert_allclose(gate[0], sample[0], rtol=1e-12)
    finally:
        root.destroy()


def test_the_gate_curve_is_drawn_filled_in_the_gates_colour():
    root, ed = _editor_or_skip()
    try:
        _setup(ed, 3000, cut=0.6)
        ed.hist_y_mode.set('Count')
        ax = Figure().add_subplot()
        ed._render_into(ax, ['s1'], 'FL1-A', None, 'histogram', None,
                        population=None)
        (line,) = [ln for ln in ax.get_lines()
                   if ln.get_label().startswith('gate ')]
        assert line.get_color() == '#d62728'
        assert line.get_linewidth() == 1.3
        alphas = sorted(round(float(c.get_alpha()), 2) for c in ax.collections)
        assert alphas == [0.30, 0.40]        # the sample's fill, the gate's
        fill = [c for c in ax.collections
                if round(float(c.get_alpha()), 2) == 0.40][0]
        top = fill.get_paths()[0].vertices[:, 1].max()
        assert top == pytest.approx(line.get_ydata().max())
    finally:
        root.destroy()


def test_the_y_mode_is_fraction_when_the_editor_has_no_control():
    from types import SimpleNamespace

    from openflo.editor_plot import PlotMixin
    assert PlotMixin._hist_y_mode(SimpleNamespace()) == 'Fraction'
    var = SimpleNamespace(get=lambda: 'Count')
    assert PlotMixin._hist_y_mode(SimpleNamespace(hist_y_mode=var)) == 'Count'


def test_a_gate_with_no_place_on_the_axis_draws_no_curve():
    """The gate (FL2-A >= 4) holds only the 20 events at FL1-A = +inf: no
    curve for it, rather than a flat one in the legend."""
    root, ed = _editor_or_skip()
    try:
        df = _frame(3000)
        df.loc[:19, 'FL2-A'] = 5.0
        _register_fl(ed, 's1', df)
        ed._set_active_sample('s1')
        ed._add_gate({'kind': 'threshold', 'channel': 'FL2-A', 'value': 4.0,
                      'name': 'FL2 hi', 'parent_id': None, 'enabled': True,
                      'color': '#d62728'})
        ed.gate_display_var.set('highlight')
        ed.hist_y_mode.set('Count')
        ax = Figure().add_subplot()
        ed._render_into(ax, ['s1'], 'FL1-A', None, 'histogram', None,
                        population=None)
        labels = [ln.get_label() for ln in ax.get_lines()]
        assert labels == ['s1']
        assert len(ax.collections) == 1        # the sample's fill only
    finally:
        root.destroy()


def test_a_sparse_gate_stays_under_its_sample_on_count():
    """A gate of 100 of 50,000 events: a tight cluster at FL1-A ~ 1.2 that
    is the only thing in its part of the axis (picked out on FL2-A; 200
    ungated events spread over 1.5-2.0 keep it inside the plotted range).
    Smoothed with its own wider kernel the gate's curve rose above the
    sample's beside the cluster; smoothed with the sample's kernel it stays
    under it, bin by bin."""
    root, ed = _editor_or_skip()
    try:
        rng = np.random.default_rng(11)
        df = _frame(50_000)
        df.loc[:99, 'FL1-A'] = rng.normal(1.2, 0.01, 100)
        df.loc[:99, 'FL2-A'] = 5.0
        df.loc[100:299, 'FL1-A'] = rng.uniform(1.5, 2.0, 200)
        _register_fl(ed, 's1', df)
        ed._set_active_sample('s1')
        ed._add_gate({'kind': 'threshold', 'channel': 'FL2-A', 'value': 4.0,
                      'name': 'FL2 hi', 'parent_id': None, 'enabled': True,
                      'color': '#d62728'})
        ed._channel_scale['FL1-A'] = 'linear'
        ed.gate_display_var.set('highlight')
        sample, gate = _curves(ed, 'Count')
        assert gate.sum() == pytest.approx(100, abs=1e-6)
        assert np.all(gate <= sample + 1e-9)
    finally:
        root.destroy()


def test_the_kernel_can_come_from_another_size():
    """``kernel_size`` sets the smoothing; ``size`` still divides Fraction."""
    from openflo.editor_plot import PlotMixin
    seen = []

    def sm(counts, sigma, mode):
        seen.append(sigma)
        return counts

    y = PlotMixin._hist_curve([2, 6], 8, 256, 'Fraction', sm,
                              kernel_size=256 * 9)
    assert seen == [1.0]                 # dense kernel: sqrt(1/9)*1.5 -> 1
    assert list(y) == [0.25, 0.75]
    PlotMixin._hist_curve([2, 6], 8, 256, 'Fraction', sm)
    assert seen[-1] == 4.0               # 8 events in 256 bins: widest
