"""Gate bounds and values are shown as the intensities the axes show.

Fluorescence is stored logicle-transformed and gates live in those stored
coordinates. The axes were relabelled in intensity, but every place that
SAID a gate bound or a value still printed the stored coordinate: a CD3 cut
at an intensity of ~1,000 read 'T  CD3-A >= 0.453' in the gate list, in the
Statistics / Frequencies population names (and their CSV exports), in the
boolean dialog and the status line; the histogram slider read '0.453'; the
colour-by bar was ticked 0.2 .. 1.0; and the CLI's scatter PNGs ticked their
fluorescence axes 0.2 .. 1.0 as well. Each now says the intensity, formatted
as the axis ticks are (scales.format_intensity / format_intensity_text).
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import openflo.pipeline as fp
from openflo.gating import population_keys, population_path, population_stats
from openflo.scales import format_intensity, format_intensity_text, intensity_ticks
from tests.conftest import gui_unavailable

LOGICLE = fp.transform_spec('logicle')
TF = {'CD3-A': LOGICLE, 'CD4-A': LOGICLE, 'OLD-A': fp.unknown_spec()}


def _stored(v):
    """The logicle coordinate of intensity `v`."""
    return float(fp.transform_values(np.array([float(v)]), 'logicle')[0])


def _inv(v):
    return fp.inverse_transform_values(np.asarray(v, dtype=float), 'logicle')


# ── Text formatting ──────────────────────────────────────────────────────────

@pytest.mark.parametrize('v,want', [
    (0.0, '0'), (0.453, '0.453'), (45.24, '45.2'), (-736.2, '-736'),
    (987.19, '987'), (999.6, '1K'), (1234.5, '1.23K'), (2500.0, '2.5K'),
    (52345.6, '52.3K'), (262144.0, '262K'), (float('nan'), 'nan'),
])
def test_format_intensity_text(v, want):
    """Plain ASCII, the tick labels' rules without their mathtext."""
    assert format_intensity_text(v) == want


def test_tick_labels_unchanged_by_the_shared_formatter():
    """format_intensity now delegates its non-power-of-ten cases to
    format_intensity_text; the axis labels must read exactly as before."""
    for v, want in [(50000.0, '50K'), (2500.0, '2.5K'), (-736.0, '-736'),
                    (1000.0, '$10^{3}$'), (0.0, '0'), (1.23456789e4, '12.3457K')]:
        assert format_intensity(v) == want


# ── describe_gate ────────────────────────────────────────────────────────────

def test_describe_gate_without_transforms_is_unchanged():
    """Library callers and logs keep the stored-coordinate text."""
    assert fp.describe_gate({'kind': 'threshold', 'channel': 'CD3-A',
                             'value': 0.453}) == 'T  CD3-A >= 0.453'
    assert fp.describe_gate({'kind': 'interval', 'channel': 'CD3-A',
                             'lo': 0.2, 'hi': 0.6}) == 'I  CD3-A in [0.2, 0.6]'
    assert fp.describe_gate({
        'kind': 'rect', 'name': 'Q1', 'x_channel': 'FSC-A', 'y_channel': 'CD3-A',
        'x0': -1e12, 'x1': 52345.6, 'y0': 0.454, 'y1': 1e12}) == (
        'R  Q1  FSC-A x CD3-A  [-1e+12,5.23e+04] x [0.454,1e+12]')


def test_describe_threshold_in_intensity():
    """Before: 'T  CD3-A >= 0.453' for a cut at an intensity of 987."""
    g = {'kind': 'threshold', 'channel': 'CD3-A', 'value': 0.453}
    assert float(_inv([0.453])[0]) == pytest.approx(987.19, abs=0.01)
    assert fp.describe_gate(g, transforms=TF) == 'T  CD3-A >= 987'
    g['value'] = _stored(1000.0)
    assert fp.describe_gate(g, transforms=TF) == 'T  CD3-A >= 1K'


def test_describe_interval_in_intensity_and_open_sides():
    """Closed: half open as gate_to_mask counts it. A side at the +-1e12
    open sentinel is said as open ('< 2.5K'), not as '-1e+12'."""
    def iv(lo, hi):
        return fp.describe_gate({'kind': 'interval', 'channel': 'CD3-A',
                                 'lo': lo, 'hi': hi}, transforms=TF)
    assert iv(0.2, 0.6) == 'I  CD3-A in [85.1, 4.25K)'
    assert iv(-1e12, _stored(2500.0)) == 'I  CD3-A < 2.5K'
    assert iv(_stored(1000.0), 1e12) == 'I  CD3-A >= 1K'
    assert iv(-1e12, 1e12) == 'I  CD3-A (all events)'


def test_describe_rect_in_intensity_with_open_sides():
    """A quadrant's outer sides are open; a linear channel (FSC-A, no
    transform record) is its stored value, written alike."""
    q1 = {'kind': 'rect', 'name': 'Q1', 'x_channel': 'FSC-A', 'y_channel': 'CD3-A',
          'x0': -1e12, 'x1': 52345.6, 'y0': _stored(1000.0), 'y1': 1e12}
    assert fp.describe_gate(q1, transforms=TF) == (
        'R  Q1  FSC-A x CD3-A  FSC-A < 52.3K, CD3-A >= 1K')
    box = {'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'CD4-A',
           'x0': 5e4, 'x1': 1e5, 'y0': _stored(1000.0), 'y1': _stored(2500.0)}
    assert fp.describe_gate(box, transforms=TF) == (
        'R  FSC-A x CD4-A  FSC-A in [50K, 100K), CD4-A in [1K, 2.5K)')
    # An axis open at both ends does not constrain and is not listed.
    band = dict(box, x0=-1e12, x1=1e12)
    assert fp.describe_gate(band, transforms=TF) == (
        'R  FSC-A x CD4-A  CD4-A in [1K, 2.5K)')


def test_describe_unknown_scale_keeps_stored_value_marked():
    """A channel whose transform is unknown has no intensity: its stored
    value, said to be one -- never a made-up intensity."""
    g = {'kind': 'threshold', 'channel': 'OLD-A', 'value': 0.453}
    assert fp.describe_gate(g, transforms=TF) == (
        'T  OLD-A >= 0.453 (stored, scale unknown)')


def test_describe_polygon_and_others_unaffected():
    poly = {'kind': 'polygon', 'x_channel': 'CD3-A', 'y_channel': 'CD4-A',
            'vertices': [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]}
    assert fp.describe_gate(poly, transforms=TF) == fp.describe_gate(poly) == (
        'P  CD3-A x CD4-A  (3 verts)')
    b = {'kind': 'boolean', 'op': 'and', 'operands': ['a', 'b']}
    assert fp.describe_gate(b, transforms=TF) == 'B  AND(2)'


def test_describe_gate_with_transforms_is_ascii():
    """It says so: the text must survive a cp1252 stdout."""
    for g in ({'kind': 'threshold', 'channel': 'CD3-A', 'value': 0.453},
              {'kind': 'interval', 'channel': 'CD3-A', 'lo': -1e12, 'hi': 0.6},
              {'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'CD3-A',
               'x0': 1.0, 'x1': 1e12, 'y0': -1e12, 'y1': 0.6}):
        fp.describe_gate(g, transforms=TF).encode('ascii')


# ── Population names ─────────────────────────────────────────────────────────

def _tree():
    return {
        'g1': {'kind': 'rect', 'name': 'Cells', 'x_channel': 'FSC-A',
               'y_channel': 'SSC-A', 'x0': 1e4, 'x1': 2e5, 'y0': 1e3, 'y1': 2e5,
               'parent_id': None},
        'g2': {'kind': 'threshold', 'channel': 'CD3-A', 'value': 0.453,
               'parent_id': 'g1'},
    }


def test_population_path_shows_intensity():
    """Before: 'Cells/T  CD3-A >= 0.453' in Statistics, Frequencies and
    their CSVs. Named gates are unchanged; without transforms, as before."""
    gates = _tree()
    assert population_path(gates, 'g2', transforms=TF) == 'Cells/T  CD3-A >= 987'
    assert population_path(gates, 'g2') == 'Cells/T  CD3-A >= 0.453'


def test_population_stats_names_in_intensity_keys_unchanged():
    """The displayed (and exported) name says intensity; the cross-sample
    identity (population_keys, what Frequencies matches by) carries no
    coordinates and does not change."""
    rng = np.random.default_rng(0)
    lin = rng.lognormal(7, 1, 4000)
    df = pd.DataFrame({'FSC-A': rng.uniform(2e4, 1e5, 4000),
                       'SSC-A': rng.uniform(2e3, 1e5, 4000),
                       'CD3-A': fp.transform_values(lin, 'logicle')})
    gates = _tree()
    keys_before = population_keys(gates, {})
    rows = population_stats('s1', df, gates, ['g1', 'g2'], {}, ['CD3-A'],
                            {'Count'}, ('Median',),
                            transforms={'CD3-A': LOGICLE})
    by = {r['__gid__']: r for r in rows}
    assert by['g2']['Population'] == 'Cells/T  CD3-A >= 987'
    assert by['g2']['Count'] == int((lin >= _inv([0.453])[0] - 1e-6).sum())
    assert by['g2']['__key__'] == keys_before['g2'] == 'n:cells/threshold:cd3-a'
    assert population_keys(gates, {}) == keys_before


def test_workspace_gate_path_in_intensity():
    from openflo.workspace import gate_path
    gates = _tree()
    assert gate_path(gates, 'g2', transforms=TF) == (
        'R  Cells  FSC-A x SSC-A  FSC-A in [10K, 200K), SSC-A in [1K, 200K)'
        ' / T  CD3-A >= 987')


# ── Ticks on an axis in stored units (colour bar, CLI PNG) ──────────────────

def test_stored_axis_ticks_sit_at_round_intensities():
    """An axis drawn in the stored coordinate (a colour bar, a CLI scatter)
    is ticked at round intensities, each at its stored position, spaced on
    that axis (100 would sit on 0 at this range)."""
    lo, hi = -0.1, 1.05
    pos, vals = intensity_ticks('logicle', 'stored', lo, hi)
    assert list(vals) == [0.0, 1e3, 1e4, 1e5]
    np.testing.assert_allclose(_inv(pos), vals, rtol=1e-6, atol=1e-6)
    assert np.all((pos >= lo) & (pos <= hi))
    # A recorded spec (parameters included) ticks the same.
    pos2, vals2 = intensity_ticks(LOGICLE, 'stored', lo, hi)
    np.testing.assert_allclose(pos2, pos)
    # Linear-stored or unknown: no intensity ticks.
    assert len(intensity_ticks('linear', 'stored', 0, 1e5)[0]) == 0
    assert len(intensity_ticks(fp.unknown_spec(), 'stored', 0, 1)[0]) == 0


def test_cli_scatter_axes_and_colorbar_in_intensity():
    """FlowSample.plot (the CLI's per-sample and per-group scatter PNGs)
    drew stored logicle values with matplotlib's ticks: 0.2 .. 1.0. Its
    transformed axes and colour bar now say intensities; a linear channel
    keeps matplotlib's ticks."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(1)
    n = 3000
    s = fp.FlowSample.from_dataframe(pd.DataFrame({
        'FSC-A': rng.uniform(2e4, 1e5, n),
        'CD3-A': rng.lognormal(7, 1.2, n), 'CD4-A': rng.lognormal(8, 1, n)}))
    s.fluor_channels = ['CD3-A', 'CD4-A']
    s.apply_transform()
    fig, ax = plt.subplots()
    try:
        s.plot('CD3-A', 'CD4-A', color_by='CD4-A', ax=ax)
        fig.canvas.draw()
        for axis in (ax.xaxis, ax.yaxis):
            locs = axis.get_majorticklocs()
            labels = [t.get_text() for t in axis.get_ticklabels()]
            assert len(locs) >= 3
            assert labels == [format_intensity(v) for v in _inv(locs)]
            assert not {'0.2', '0.4', '0.6', '0.8'} & set(labels)
        cax = fig.axes[-1]
        locs = cax.yaxis.get_majorticklocs()
        labels = [t.get_text() for t in cax.yaxis.get_ticklabels()]
        assert len(locs) >= 2
        assert labels == [format_intensity(v) for v in _inv(locs)]
        ax.clear()
        s.plot('FSC-A', 'CD3-A', color_by='density', ax=ax)
        fig.canvas.draw()
        # FSC-A is linear: matplotlib's locator, untouched.
        assert type(ax.xaxis.get_major_locator()).__name__ != 'IntensityLocator'
        assert type(ax.yaxis.get_major_locator()).__name__ == 'IntensityLocator'
    finally:
        plt.close(fig)


def test_cli_group_overlay_axes_in_intensity():
    """The group pair-scatter overlay draws every sample's stored values on
    one axes: ticked in intensity when the samples share the transform, and
    labelled as stored units when they do not (no one intensity per
    position)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from openflo.cli import _intensity_axes
    rng = np.random.default_rng(2)

    def sample(method):
        s = fp.FlowSample.from_dataframe(pd.DataFrame({
            'CD3-A': rng.lognormal(7, 1, 500), 'CD4-A': rng.lognormal(8, 1, 500)}))
        s.fluor_channels = ['CD3-A', 'CD4-A']
        s.apply_transform(method=method)
        return s
    a, b, c = sample('logicle'), sample('logicle'), sample('asinh')
    fig, ax = plt.subplots()
    try:
        ax.scatter(a.data['CD3-A'], a.data['CD4-A'])
        ax.set_xlabel('CD3')
        ax.set_ylabel('CD4')
        _intensity_axes(ax, [(a, 'CD3-A', 'CD4-A'), (b, 'CD3-A', 'CD4-A')])
        fig.canvas.draw()
        labels = [t.get_text() for t in ax.xaxis.get_ticklabels()]
        assert labels == [format_intensity(v)
                          for v in _inv(ax.xaxis.get_majorticklocs())]
        ax.clear()
        ax.scatter(a.data['CD3-A'], a.data['CD4-A'])
        ax.set_xlabel('CD3')
        _intensity_axes(ax, [(a, 'CD3-A', 'CD4-A'), (c, 'CD3-A', 'CD4-A')])
        assert ax.get_xlabel() == 'CD3 (stored display units)'
    finally:
        plt.close(fig)


# ── In the editor ────────────────────────────────────────────────────────────

def _editor_or_skip():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
    except ImportError:
        gui_unavailable("tkinter not available — headless environment")
    try:
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        gui_unavailable(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    import importlib
    gui = importlib.import_module('openflo.gui')
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    return root, ed


def _setup(ed):
    """One FlowSample as the loader stores it: FL1/FL2 logicle (recorded in
    its data_transforms), FSC-A linear."""
    rng = np.random.default_rng(0)
    n = 20_000
    df = pd.DataFrame({
        'FSC-A': rng.uniform(2e4, 1e5, n),
        'FL1-A': np.concatenate([rng.normal(0, 150, n // 4),
                                 rng.lognormal(np.log(5000), 0.6, n - n // 4)]),
        'FL2-A': rng.lognormal(np.log(3000), 0.8, n)})
    s = fp.FlowSample.from_dataframe(df, name='s1')
    s.fluor_channels = ['FL1-A', 'FL2-A']
    s.apply_transform()
    cols = list(s.data.columns)
    ed._samples['s1'] = s
    ed._sample_order.append('s1')
    ed._sample_colors['s1'] = '#1f77b4'
    ed._sample_trial['s1'] = 'T'
    ed._trial_order.append('T')
    ed._sample_plot_enabled['s1'] = True
    ed._channels = cols
    ed._channel_labels = {c: c for c in cols}
    ed._channel_transform = {'FSC-A': 'linear', 'FL1-A': 'logicle',
                             'FL2-A': 'logicle'}
    ed._populate_channel_combos()
    ed._set_active_sample('s1')
    return s


def test_gate_tree_row_says_intensity():
    """The gate list row of a logicle cut at 1,000 read
    'T  FL1-A >= 0.454'; a quadrant's '[-1e+12,...]'."""
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        gid = ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A',
                            'value': _stored(1000.0), 'parent_id': None})
        qid = ed._add_gate({'kind': 'rect', 'name': 'Q2', 'x_channel': 'FSC-A',
                            'y_channel': 'FL2-A', 'x0': 5e4, 'x1': 1e12,
                            'y0': _stored(2500.0), 'y1': 1e12,
                            'parent_id': None})
        ed._refresh_gate_list()
        assert ed.gate_tv.item(ed._gate_iid('s1', gid), 'text') == (
            'T  FL1-A >= 1K')
        assert ed.gate_tv.item(ed._gate_iid('s1', qid), 'text') == (
            'R  Q2  FSC-A x FL2-A  FSC-A >= 50K, FL2-A >= 2.5K')
    finally:
        root.destroy()


def test_slider_labels_say_intensity():
    """The histogram slider sits on the stored coordinate (it is what the
    gate gets); its label read '0.454' beside an axis ticked 10^3."""
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        ed.x_combo.set('FL1-A')
        ed.mode_var.set('histogram')
        ed._replot()
        assert ed._slider_channel == 'FL1-A'
        ed._slider_updating = True
        ed.slider_lo.set(_stored(1000.0))
        ed.slider_hi.set(_stored(2500.0))
        ed._slider_updating = False
        ed.slider_kind_var.set('threshold')
        ed._update_slider_ui()
        assert ed.slider_lo_lbl.cget('text') == '1K'
        ed.slider_kind_var.set('interval')
        ed._update_slider_ui()
        assert ed.slider_lo_lbl.cget('text') == 'lo 1K'
        assert ed.slider_hi_lbl.cget('text') == 'hi 2.5K'
        # A linear channel: the value is the intensity, written alike.
        ed.slider_kind_var.set('threshold')
        ed.x_combo.set('FSC-A')
        ed._replot()
        assert ed._slider_channel == 'FSC-A'
        ed._slider_updating = True
        ed.slider_lo.set(52345.6)
        ed._slider_updating = False
        ed._update_slider_ui()
        assert ed.slider_lo_lbl.cget('text') == '52.3K'
    finally:
        root.destroy()


def test_colour_bar_ticks_say_intensity():
    """Colour by a logicle channel: the bar was ticked in its stored units
    (0.2 .. 1.0). Each tick now names the intensity at its position."""
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        ed.x_combo.set('FSC-A')
        ed.y_combo.set('FL1-A')
        ed.mode_var.set('dot')
        ed.color_combo.set(ed._fmt_channel('FL2-A'))
        ed._replot()
        ed.canvas.draw()
        assert ed._cbar is not None
        axis = ed._cbar.ax.yaxis
        locs = axis.get_majorticklocs()
        labels = [t.get_text() for t in axis.get_ticklabels()]
        lo, hi = sorted(axis.get_view_interval())
        assert len(locs) >= 2 and np.all((locs >= lo) & (locs <= hi))
        assert labels == [format_intensity(v) for v in _inv(locs)]
        assert not {'0.2', '0.4', '0.6', '0.8'} & set(labels)
    finally:
        root.destroy()


def test_statistics_population_name_says_intensity():
    """The Statistics window's (and its CSV's) Population column."""
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        gid = ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A',
                            'value': 0.453, 'parent_id': None})
        rows, _cols = ed._collect_stats_rows({'Count'})
        (row,) = [r for r in rows if r['__gid__'] == gid]
        assert row['Population'] == 'T  FL1-A >= 987'
        snap_rows, _ = ed._stats_rows_from_snapshot(
            ed._stats_snapshot({'Count'}), {'Count'})
        (row,) = [r for r in snap_rows if r['__gid__'] == gid]
        assert row['Population'] == 'T  FL1-A >= 987'
    finally:
        root.destroy()


def test_boolean_dialog_lists_gates_in_intensity():
    """The boolean dialog's operand list names gates as the gate list does."""
    import tkinter as tk
    from tkinter import ttk
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A',
                      'value': _stored(1000.0), 'parent_id': None})
        ed._open_boolean_dialog()

        def walk(w):
            yield w
            for c in w.winfo_children():
                yield from walk(c)
        (dlg,) = [w for w in walk(ed.master) if isinstance(w, tk.Toplevel)
                  and w.title().startswith('Boolean gate')]
        listed = [w.cget('text') for w in walk(dlg)
                  if isinstance(w, ttk.Checkbutton)]
        assert listed == ['T  FL1-A >= 1K']
        dlg.destroy()
    finally:
        root.destroy()


def test_display_transforms_fall_back_to_the_axes_for_a_sample_without_record():
    """A sample object with no data_transforms record is drawn on the
    editor's _channel_transform; its gate text follows the axes."""
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        ed._samples['s2'] = SimpleNamespace(name='s2', data=ed._samples['s1'].data)
        g = {'kind': 'threshold', 'channel': 'FL1-A', 'value': _stored(1000.0)}
        assert ed._gate_text('s2', g) == 'T  FL1-A >= 1K'
    finally:
        root.destroy()
