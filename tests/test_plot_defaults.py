"""Plots drawn as flow papers draw them (UI audit batch 1).

- Scatter (FSC/SSC) and Time default to a linear axis; fluorescence to logicle,
  so events at or below zero stay on screen instead of piling on the axis.
- Logicle's width W and arcsinh's cofactor can be set per channel. They are
  views of the channel's linear intensity: gates (kept in stored coordinates)
  select the same events whatever the view.
- Each gate carries its name and frequency on the plot.
- Every power of ten on an axis reads 10^k; frequencies have one decimal.
- Plots default to pseudocolor with a near-uniform colour map (Turbo).

Every test builds its own synthetic events (generic channels FSC-A, SSC-A,
FL1-A, FL2-A) with a known answer, and each was run against master before the
change to confirm it fails there.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')

from openflo import pipeline as fp  # noqa: E402

# The module, not its names: a name a test needs is looked up when that test
# runs, so each test passes or fails on its own.
from openflo import scales as sc  # noqa: E402
from tests.test_gui_smoke import _editor_or_skip  # noqa: E402


def _events(n=4000, seed=7):
    """Synthetic sample: scatter in the tens of thousands (linear-stored),
    fluorescence with a negative-centred population and a bright one, stored
    as default logicle the way the loader stores it."""
    rng = np.random.default_rng(seed)
    half = n // 2
    fl_lin = np.concatenate([rng.normal(0.0, 150.0, half),       # straddles 0
                             rng.lognormal(np.log(5000.0), 0.3, n - half)])
    fl2_lin = np.concatenate([rng.lognormal(np.log(3000.0), 0.3, half),
                              rng.normal(-50.0, 120.0, n - half)])
    return pd.DataFrame({
        'FSC-A': rng.normal(60000.0, 12000.0, n),
        'SSC-A': rng.normal(30000.0, 8000.0, n),
        'FL1-A': fp.transform_values(fl_lin, 'logicle'),
        'FL2-A': fp.transform_values(fl2_lin, 'logicle'),
        'Time': np.linspace(0.0, 120.0, n),
    }), fl_lin


def _sample(name, df):
    return SimpleNamespace(
        name=name, path=f'{name}.fcs', data=df,
        channel_labels={c: c for c in df.columns},
        fluor_channels=['FL1-A', 'FL2-A'],
        scatter_channels=['FSC-A', 'SSC-A'],
        data_transforms={'FL1-A': fp.transform_spec('logicle'),
                         'FL2-A': fp.transform_spec('logicle')})


@pytest.fixture
def editor(monkeypatch):
    import openflo.prefs as prefs
    monkeypatch.setattr(prefs, 'write_pref', lambda *a, **k: None)
    root, ed, _gui = _editor_or_skip()
    df, _lin = _events()
    ed._on_loaded('synth_a', _sample('synth_a', df))
    ed._on_load_settled()
    ed.update()
    yield ed
    try:
        root.destroy()
    except Exception:
        pass


def _show(ed, x, y):
    ed.x_combo.set(ed._fmt_channel(x))
    ed.y_combo.set(ed._fmt_channel(y))
    ed._replot()
    ed.update()


# ── 1. default scales ────────────────────────────────────────────────────────

def test_scatter_and_time_default_to_linear_fluorescence_to_logicle(editor):
    """Positive: FSC/SSC/Time draw on a linear axis, fluorescence on logicle.
    Negative: master drew every one of them on a log axis."""
    ed = editor
    for ch in ('FSC-A', 'SSC-A', 'Time'):
        assert ed._scale_of(ch) == 'linear', ch
    for ch in ('FL1-A', 'FL2-A'):
        assert ed._scale_of(ch) == 'logicle', ch
    _show(ed, 'FSC-A', 'SSC-A')
    assert ed.ax.get_xscale() == 'linear' and ed.ax.get_yscale() == 'linear'


def test_negative_events_are_not_piled_on_the_axis(editor):
    """Positive: on the default fluorescence view, events at or below zero
    keep distinct screen positions (spread through the linear-like region).
    Negative: master's log view put every one of them at one position, the
    view floor."""
    ed = editor
    df, lin = _events()
    fwd, _inv = ed._axis_view_funcs('FL1-A')
    neg = lin <= 0
    assert neg.sum() > 500                   # the generated dataset has them
    scr = np.asarray(fwd(df['FL1-A'].to_numpy()[neg]))
    # Distinct positions, spread over a real range of the axis.
    assert np.unique(np.round(scr, 6)).size > 0.9 * neg.sum()
    assert np.ptp(scr) > 0.05


def test_explicit_user_scale_wins_over_the_new_default(editor):
    """A saved choice (an older session's 'log') is kept, not replaced."""
    ed = editor
    ed._channel_scale['FL1-A'] = 'log'
    ed._channel_scale['FSC-A'] = 'log'
    assert ed._scale_of('FL1-A') == 'log'
    assert ed._scale_of('FSC-A') == 'log'


# ── 2. adjustable logicle width / arcsinh cofactor ───────────────────────────

def test_scale_strings_round_trip_and_drop_defaults():
    """Positive: parameters round-trip and a default one is left out (so the
    default stays the plain name). Negative: bad input is dropped, clamped,
    never raised."""
    assert sc.make_scale('logicle', w=0.5) == 'logicle'
    assert sc.make_scale('logicle', w=1.25) == 'logicle:w=1.25'
    assert sc.parse_scale('logicle:w=1.25') == ('logicle', {'w': 1.25})
    assert sc.make_scale('asinh', c=150.0) == 'asinh'
    assert sc.parse_scale('asinh:c=5') == ('asinh', {'c': 5.0})
    assert sc.parse_scale('logicle:w=abc') == ('logicle', {})
    assert sc.parse_scale('logicle:w=99') == ('logicle', {'w': 2.0})   # clamped
    assert sc.parse_scale('logicle:c=5') == ('logicle', {})            # not its
    assert sc.parse_scale('log') == ('log', {})


def test_default_logicle_view_is_the_stored_coordinate():
    """Default logicle on default-logicle data costs nothing and is exact."""
    fwd, inv = sc.view_funcs('logicle', 'logicle')
    d = np.array([-0.1, 0.0, 0.3, 0.9])
    np.testing.assert_array_equal(fwd(d), d)
    np.testing.assert_array_equal(inv(d), d)


@pytest.mark.parametrize('scale', ['logicle:w=1.5', 'logicle:w=0.1',
                                   'asinh', 'asinh:c=20'])
def test_tuned_views_change_the_axis_but_not_the_gate(scale):
    """Positive: a tuned view moves events on screen, and a gate's bounds,
    mapped through the view and back, select exactly the events they select
    in stored coordinates. Negative (master): 'logicle:w=..' and 'asinh' did
    not exist; view_funcs drew them as a plain linear stretch."""
    df, lin = _events()
    stored = df['FL1-A'].to_numpy()
    fwd, inv = sc.view_funcs('logicle', scale)
    default_fwd, _ = sc.view_funcs('logicle', 'logicle')
    scr = np.asarray(fwd(stored))
    assert not np.allclose(scr, default_fwd(stored))   # the view did change
    # It is a logicle / arcsinh view of the intensity, not a linear stretch.
    kind, params = sc.parse_scale(scale)
    if kind == 'asinh':
        want = np.arcsinh(lin / params.get('c', 150.0))
    else:
        want = fp.transform_values(lin, 'logicle', w=params['w'])
    np.testing.assert_allclose(scr, want, atol=1e-6)
    # A gate from intensity 1,000 up, in stored units.
    lo = float(fp.transform_values(np.array([1000.0]), 'logicle')[0])
    in_stored = stored >= lo
    s_lo = float(fwd(np.array([lo]))[0])
    in_view = scr >= s_lo
    np.testing.assert_array_equal(in_stored, in_view)
    np.testing.assert_allclose(inv(np.array([s_lo])), [lo], atol=1e-9)
    assert 1000 < in_stored.sum() < len(stored)


def test_changing_the_width_keeps_gate_counts(editor):
    """Positive: setting a new width redraws the axis and every gate still
    counts the same events. Negative: a re-transform of the stored data
    (what a transform change does) would move the events under the gate."""
    ed = editor
    name = ed._active_sample
    lo = float(fp.transform_values(np.array([1000.0]), 'logicle')[0])
    ed._add_gate_multi({'kind': 'rect', 'x_channel': 'FL1-A',
                        'y_channel': 'FL2-A', 'x0': lo, 'x1': 1e12,
                        'y0': -1e12, 'y1': 1e12, 'name': 'Bright'})
    gid = list(ed._sample_gates[name])[-1]
    _show(ed, 'FL1-A', 'FL2-A')
    before = ed._pc_counts(name, [gid])[gid]
    data_before = ed._samples[name].data['FL1-A'].to_numpy().copy()
    ed._set_axis_config('FL1-A', 'logicle:w=1.5', None)
    _show(ed, 'FL1-A', 'FL2-A')
    assert ed._scale_of('FL1-A') == 'logicle:w=1.5'
    np.testing.assert_array_equal(ed._samples[name].data['FL1-A'].to_numpy(),
                                  data_before)
    assert ed._pc_counts(name, [gid])[gid] == before
    # The axis is now the tuned view: an intensity of 1,000 sits elsewhere.
    fwd = ed.ax.xaxis.get_transform().transform
    assert not np.isclose(fwd(np.array([lo]))[0], lo)


def test_axis_dialog_offers_and_validates_the_parameters(editor):
    """Positive: the dialog hands back 'logicle:w=1.5' / 'asinh:c=5'.
    Negative: out-of-range values are refused with a message."""
    from openflo.ui_axis_config import AxisConfigDialog
    got = []
    dlg = AxisConfigDialog(editor, 'FL1-A', 'logicle', None,
                           lambda s, r, link: got.append(s))
    try:
        dlg.w_var.set('1.5')
        dlg._on_ok()
        assert got == ['logicle:w=1.5']
    finally:
        if dlg.winfo_exists():
            dlg.destroy()
    dlg = AxisConfigDialog(editor, 'FL1-A', 'asinh:c=5', None,
                           lambda s, r, link: got.append(s))
    try:
        assert dlg.scale_var.get() == 'asinh' and dlg.c_var.get() == '5'
        dlg.c_var.set('-3')
        dlg._on_ok()
        assert 'Cofactor must be between' in dlg.err_var.get()
        assert got == ['logicle:w=1.5']                    # nothing applied
        dlg.scale_var.set('logicle')
        dlg.w_var.set('7')
        dlg._on_ok()
        assert 'Width must be between' in dlg.err_var.get()
    finally:
        if dlg.winfo_exists():
            dlg.destroy()


# ── 3. gate labels ───────────────────────────────────────────────────────────

def _label_texts(ed):
    return [t.get_text() for t in getattr(ed, '_gate_label_artists', [])]


def test_gates_are_labelled_with_name_and_frequency(editor):
    """Positive: each gate on the plot carries its name and its frequency of
    the parent, one decimal, computed from the generated events. Negative:
    master drew the outline only."""
    ed = editor
    name = ed._active_sample
    df = ed._samples[name].data
    ed._add_gate_multi({'kind': 'rect', 'x_channel': 'FSC-A',
                        'y_channel': 'SSC-A', 'x0': 50000.0, 'x1': 90000.0,
                        'y0': 0.0, 'y1': 40000.0, 'name': 'Cells'})
    _show(ed, 'FSC-A', 'SSC-A')
    inside = ((df['FSC-A'] >= 50000) & (df['FSC-A'] <= 90000)
              & (df['SSC-A'] >= 0) & (df['SSC-A'] <= 40000)).sum()
    want = f"Cells\n{100.0 * inside / len(df):.1f}%"
    assert want in _label_texts(ed)


def test_quadrant_labels_sum_to_the_parent(editor):
    """Positive: the four quadrant labels are percentages of the parent that
    add up to 100 (within rounding)."""
    ed = editor
    lo_x = float(fp.transform_values(np.array([1000.0]), 'logicle')[0])
    lo_y = float(fp.transform_values(np.array([800.0]), 'logicle')[0])
    for x0, x1, y0, y1 in ((lo_x, 1e12, lo_y, 1e12), (lo_x, 1e12, -1e12, lo_y),
                           (-1e12, lo_x, lo_y, 1e12), (-1e12, lo_x, -1e12, lo_y)):
        ed._add_gate_multi({'kind': 'rect', 'x_channel': 'FL1-A',
                            'y_channel': 'FL2-A', 'x0': x0, 'x1': x1,
                            'y0': y0, 'y1': y1, 'quad_set': 'qs1',
                            'quad_origin_x': lo_x, 'quad_origin_y': lo_y})
    _show(ed, 'FL1-A', 'FL2-A')
    texts = _label_texts(ed)
    assert len(texts) == 4, texts
    total = sum(float(t.rstrip('%')) for t in texts)
    assert abs(total - 100.0) <= 0.2


def test_gate_labels_can_be_turned_off(editor):
    ed = editor
    ed._add_gate_multi({'kind': 'rect', 'x_channel': 'FSC-A',
                        'y_channel': 'SSC-A', 'x0': 50000.0, 'x1': 90000.0,
                        'y0': 0.0, 'y1': 40000.0})
    _show(ed, 'FSC-A', 'SSC-A')
    assert len(_label_texts(ed)) == 1
    ed._gate_labels_var.set(False)
    _show(ed, 'FSC-A', 'SSC-A')
    assert _label_texts(ed) == []
    assert not [t for t in ed.ax.texts if t.get_text().endswith('%')]


def test_label_text_reads_na_for_an_uncomputable_population():
    from openflo.gating import format_percent, gate_label_text
    assert gate_label_text('Cells', 30, 40, None) == 'Cells\n75.0%'
    assert gate_label_text(None, 1, 3, None) == '33.3%'
    assert gate_label_text('Cells', 0, 0, 'needs an unknown scale') == \
        'Cells\nNA'
    assert format_percent(5, 0) == 'NA'


# ── 4. tick labels and percentages ───────────────────────────────────────────

def test_every_decade_reads_as_a_power_of_ten():
    """Positive: one notation for every decade. Negative: master wrote
    '1, 10, 100' and then '10^3'."""
    labels = [sc.format_intensity(v) for v in (1.0, 10.0, 100.0, 1000.0, 1e4)]
    assert labels == ['$10^{0}$', '$10^{1}$', '$10^{2}$', '$10^{3}$',
                      '$10^{4}$']


def test_logicle_axis_ticks_are_zero_and_decades():
    """The default fluorescence axis is ticked at 0 and whole decades, each
    tick at the stored position of the intensity its label names."""
    lo, hi = (float(v) for v in fp.transform_values(
        np.array([-200.0, 2e5]), 'logicle'))
    pos, vals = sc.intensity_ticks('logicle', 'logicle', lo, hi)
    assert 0.0 in vals and 1e5 in vals
    for v in vals:
        assert v == 0 or np.isclose(np.log10(abs(v)), round(np.log10(abs(v))))
    np.testing.assert_allclose(
        fp.inverse_transform_values(pos, 'logicle'), vals, rtol=1e-6,
        atol=1e-6)


def test_linear_axis_reads_in_thousands():
    """Positive: '50K', not '50000' or an offset '1e5'."""
    fmt = sc.linear_tick_formatter()
    assert [fmt(v) for v in (0.0, 500.0, 50000.0, 262144.0, -2500.0)] == \
        ['0', '500', '50K', '262.1K', '-2.5K']


def test_statistics_percentages_have_one_decimal():
    """Positive: '93.0', '0.5'. Negative: master printed '93' and '0.5'
    alike ('.3g'), so 92.96 and 93.0 read the same."""
    from openflo.ui_statistics import StatisticsWindow
    assert StatisticsWindow._fmt('%Parent', 92.96) == '93.0'
    assert StatisticsWindow._fmt('%Total', 0.4987) == '0.5'
    assert StatisticsWindow._fmt('%Parent', 100.0) == '100.0'
    assert StatisticsWindow._fmt('Count', 4648) == '4,648'


def test_status_bar_frequency_has_one_decimal():
    from openflo.gating import format_gate_count
    assert format_gate_count('Cells', 4648, 5000, 'all') == \
        'Cells:  n = 4,648   (93.0% of all)'


# ── 5. pseudocolor default ───────────────────────────────────────────────────

def test_plots_open_in_pseudocolor_with_turbo(editor):
    """Positive: a new editor plots density (pseudocolor) with Turbo.
    Negative: master opened on single-colour dots, and its density map was
    the 'jet' rainbow."""
    ed = editor
    assert ed.mode_var.get() == 'pseudocolor'
    _show(ed, 'FSC-A', 'SSC-A')
    cmaps = [c.get_cmap().name for c in ed.ax.collections
             if c.get_array() is not None]
    assert cmaps and set(cmaps) == {'turbo'}


def test_density_colour_map_follows_the_preference(monkeypatch):
    import openflo.theme as th
    monkeypatch.setattr(th, 'read_prefs', lambda: {'density_cmap': 'jet'})
    assert th.density_cmap() == 'jet'
    monkeypatch.setattr(th, 'read_prefs', lambda: {'density_cmap': 'nope'})
    assert th.density_cmap() == 'turbo'


# ── 6. a log view zoomed inside a decade ─────────────────────────────────────

@pytest.mark.parametrize('lim,want', [
    ((700.0, 950.0), ['700', '800', '900']),
    ((300.0, 2000.0), ['500', '1K', '1.5K', '2K']),
    ((1.0, 1e5), ['$10^{0}$', '$10^{1}$', '$10^{2}$', '$10^{3}$',
                  '$10^{4}$', '$10^{5}$']),
])
def test_native_log_axis_zoomed_inside_a_decade_reads_round_values(lim, want):
    """Positive: zoomed inside about a decade, a log axis is ticked at a few
    round values written one way; wider, at decades. Negative: master left
    matplotlib to label every minor tick there ('7x10^2', '8x10^2', ...),
    which overprint (the layout exports, PR #50, tick it the same way)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    try:
        ax.set_xscale('log')
        loc, fmt = sc.native_log_ticker()
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(fmt)
        ax.set_xlim(*lim)
        ticks = [v for v in loc() if lim[0] <= v <= lim[1]]
        assert [fmt(v) for v in ticks] == want
    finally:
        plt.close(fig)


def test_fluorescence_log_view_zoomed_inside_a_decade_keeps_ticks():
    """Positive: a logicle-stored channel on a log view zoomed to 700..950
    still gets round ticks at their stored positions. Negative: master
    ticked decades only, so the zoomed axis had none."""
    lo, hi = fp.transform_values(np.array([700.0, 950.0]), 'logicle')
    pos, vals = sc.intensity_ticks('logicle', 'log', lo, hi)
    assert list(vals) == [700.0, 800.0, 900.0]
    np.testing.assert_allclose(fp.inverse_transform_values(pos, 'logicle'),
                               vals, rtol=1e-6)
