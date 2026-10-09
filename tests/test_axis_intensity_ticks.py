"""Axes of a transformed channel are labelled, and set, in intensity.

Fluorescence is stored logicle-transformed and the display scale is a view of
the underlying linear intensity (scales.view_funcs). The tick labels were
matplotlib's defaults on the STORED coordinate: 0.0 .. 1.2, so a tick reading
'0.6' was an intensity of ~4,250 and '1.0' one of ~262,000; and the axis
dialog's Min / Max took and showed stored units too. Ticks now sit at round
intensities and say them; the dialog converts at its boundary.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import openflo.pipeline as fp
from openflo.scales import format_intensity, intensity_ticks
from tests.conftest import gui_unavailable


def _inv(v):
    return fp.inverse_transform_values(np.asarray(v, dtype=float), 'logicle')


@pytest.mark.parametrize('v,want', [
    (0.0, '0'), (1.0, '1'), (10.0, '10'), (100.0, '100'),
    (1000.0, '$10^{3}$'), (1e5, '$10^{5}$'), (-1000.0, '$-10^{3}$'),
    (50000.0, '50K'), (2500.0, '2.5K'), (-736.0, '-736'),
    (999.9999996, '$10^{3}$'), (1e-6, '$10^{-6}$'),
])
def test_format_intensity(v, want):
    assert format_intensity(v) == want


@pytest.mark.parametrize('scale', ['log', 'linear', 'symlog'])
def test_ticks_sit_at_round_intensities_inside_the_view(scale):
    """Over the default view of a logicle channel (stored 0.11 .. 1.01 on
    the log view), each tick's stored position inverts to the round value
    its label names, and every tick is inside the view."""
    lo, hi = {'log': (0.1111, 1.0134), 'linear': (-0.3935, 0.8966),
              'symlog': (-0.2618, 0.9578)}[scale]
    pos, vals = intensity_ticks('logicle', scale, lo, hi)
    assert 3 <= len(pos) <= 10, vals
    assert np.all((pos >= lo) & (pos <= hi))
    np.testing.assert_allclose(_inv(pos), vals, rtol=1e-6, atol=1e-6)
    for v in vals:
        if v != 0:
            m = abs(v) / 10 ** np.floor(np.log10(abs(v)))
            assert np.isclose(m, np.round(m * 2) / 2), v   # 1, 1.5, 2, 2.5, ...
    if scale == 'log':                              # decades only
        assert all(np.isclose(np.log10(v), np.round(np.log10(v))) for v in vals)


def test_linear_channel_has_no_intensity_ticks():
    """A linear-stored channel keeps matplotlib's own scale and ticks."""
    pos, vals = intensity_ticks('linear', 'log', 1.0, 1e5)
    assert len(pos) == 0 and len(vals) == 0


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
    rng = np.random.default_rng(0)
    lin = np.concatenate([rng.normal(0, 150, (15_000, 2)),
                          rng.lognormal(np.log(5000), 0.6, (35_000, 2))])
    df = pd.DataFrame(fp.transform_values(lin.ravel(), 'logicle')
                      .reshape(lin.shape), columns=['FL1-A', 'FL2-A'])
    cols = list(df.columns)
    ed._samples['s1'] = SimpleNamespace(
        name='s1', path='s1.fcs', data=df, fluor_channels=cols,
        channel_labels={c: c for c in cols})
    ed._sample_order.append('s1')
    ed._sample_colors['s1'] = '#1f77b4'
    ed._sample_trial['s1'] = 'T'
    ed._trial_order.append('T')
    ed._sample_plot_enabled['s1'] = True
    ed._channels = cols
    ed._channel_labels = {c: c for c in cols}
    ed._channel_transform = {c: 'logicle' for c in cols}
    ed._populate_channel_combos()
    ed._set_active_sample('s1')
    ed.x_combo.set('FL1-A')
    ed.y_combo.set('FL2-A')
    ed.mode_var.set('dot')
    ed._replot()
    ed.canvas.draw()


@pytest.mark.parametrize('scale', ['log', 'linear'])
def test_editor_axis_labels_are_intensities(scale):
    """Each drawn tick label names the intensity at its position. Before:
    '0.0' .. '1.2' (stored logicle)."""
    root, ed = _editor_or_skip()
    try:
        ed._channel_scale.update({'FL1-A': scale, 'FL2-A': scale})
        _setup(ed)
        for axis in (ed.ax.xaxis, ed.ax.yaxis):
            locs = axis.get_majorticklocs()
            labels = [t.get_text() for t in axis.get_ticklabels()]
            assert len(locs) >= 3
            assert all(labels), labels
            fmt = axis.get_major_formatter()
            for loc, text in zip(locs, labels, strict=True):
                assert text == fmt(loc) == format_intensity(_inv([loc])[0])
            assert '0.2' not in labels and '0.6' not in labels
        # A scroll zoom re-ticks without a replot.
        ed._on_scroll(SimpleNamespace(inaxes=ed.ax, xdata=float(np.mean(
            ed.ax.get_xlim())), ydata=float(np.mean(ed.ax.get_ylim())),
            button='up'))
        ed.canvas.draw()
        lo, hi = sorted(ed.ax.get_xlim())
        locs = ed.ax.xaxis.get_majorticklocs()
        assert len(locs) >= 2 and np.all((locs >= lo) & (locs <= hi))
    finally:
        root.destroy()


def test_axis_dialog_shows_and_takes_intensity():
    """The range dialog of a logicle channel shows the stored range as
    intensity and stores what is typed as the logicle of it. Before, the
    typed 100 .. 100000 became stored limits 100 .. 100000 (an empty plot)."""
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        stored = tuple(float(v) for v in fp.transform_values(
            np.array([1000.0, 50000.0]), 'logicle'))
        ed._channel_range['FL1-A'] = stored
        before = set(ed.winfo_children())
        ed._open_axis_dialog('x')
        (dlg,) = [w for w in ed.winfo_children() if w not in before]
        assert float(dlg.min_var.get()) == pytest.approx(1000.0, rel=1e-4)
        assert float(dlg.max_var.get()) == pytest.approx(50000.0, rel=1e-4)
        dlg.auto_var.set(False)
        dlg.min_var.set('100')
        dlg.max_var.set('100000')
        dlg._on_ok()
        lo, hi = ed._channel_range['FL1-A']
        np.testing.assert_allclose(_inv([lo, hi]), [100.0, 1e5], rtol=1e-6)
    finally:
        root.destroy()


def test_axis_dialog_link_converts_per_channel():
    """Link X & Y applies one INTENSITY range to two channels; each stores
    it in its own units (FSC linear, FL1 logicle)."""
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        ed._channel_transform['FSC-A'] = 'linear'
        ed._set_axis_config('FL1-A', 'log', (100.0, 1e5), other_channel='FSC-A')
        assert ed._channel_range['FSC-A'] == (100.0, 1e5)
        np.testing.assert_allclose(_inv(ed._channel_range['FL1-A']),
                                   [100.0, 1e5], rtol=1e-6)
    finally:
        root.destroy()
