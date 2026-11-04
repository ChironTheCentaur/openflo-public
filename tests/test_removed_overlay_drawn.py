"""View -> Show cleaned-out events draws the removed events on the plot.

The repo's other tests only ever reach the removed-events overlay with its
toggle off, so its drawing code ran in no test. Here the toggle is on and a
real auto-clean gate (the doublets method only) removes a known set of
events, and the plot is drawn through ``_render_into``, the same pipeline
the live plot and the figure exporter use.

Generated data: 12,000 singlets with FSC-H == FSC-A (ratio 1) and FL1-A
near 1,000, plus 1,200 doublets with FSC-A == 2 * FSC-H (ratio 2, far
outside the +-25% singlet window) and FL1-A near 50,000, and 5 more doublets
whose FL1-A is +inf (removed, but not drawable). So the overlay must report
exactly 1,200 drawable events, centred near 50,000, in the doublets colour.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.conftest import editor_or_skip

N_SINGLE, N_DOUBLE, N_INF = 12_000, 1_200, 5
DOUBLET_FL1 = 50_000.0
RED = '#e8000b'


def _data():
    rng = np.random.default_rng(11)
    nd = N_DOUBLE + N_INF
    h1 = rng.uniform(40_000, 80_000, N_SINGLE)
    h2 = rng.uniform(40_000, 80_000, nd)
    fl1_d = rng.normal(DOUBLET_FL1, 1_000, nd)
    fl1_d[-N_INF:] = np.inf
    return pd.DataFrame({
        'FSC-A': np.r_[h1, 2.0 * h2],
        'FSC-H': np.r_[h1, h2],
        'FL1-A': np.r_[rng.normal(1_000, 50, N_SINGLE), fl1_d],
        'FL2-A': rng.normal(2_000, 100, N_SINGLE + nd)})


def _editor():
    root, ed = editor_or_skip()
    df = _data()
    ed._samples['s1'] = SimpleNamespace(
        name='s1', path=None, data=df, fluor_channels=['FL1-A', 'FL2-A'],
        channel_labels={c: c for c in df.columns})
    ed._sample_order.append('s1')
    ed._sample_colors['s1'] = '#1f77b4'
    ed._sample_trial['s1'] = 'T'
    ed._trial_order.append('T')
    ed._sample_plot_enabled['s1'] = True
    ed._channels = list(df.columns)
    ed._channel_labels = {c: c for c in df.columns}
    ed._populate_channel_combos()
    ed._set_active_sample('s1')
    ed._sample_gates['s1'] = {'ac': {
        'kind': 'autoclean', 'parent_id': None, 'name': 'clean',
        'enabled': True,
        'methods': [{'key': 'doublets', 'label': 'Doublets',
                     'enabled': True, 'params': {'tol': 0.25}}]}}
    ed._sample_gate_order['s1'] = ['ac']
    return root, ed, df


def _render(ed, mode, y=None):
    from matplotlib.figure import Figure
    ax = Figure().add_subplot()
    ed._render_into(ax, ['s1'], 'FL1-A', y, mode,
                    'By sample', population=None)
    return ax


def _doublets(df):
    """The drawable doublets, found from the data directly (ratio outside
    +-25%, finite FL1-A)."""
    ratio = df['FSC-A'].to_numpy() / df['FSC-H'].to_numpy()
    return (np.abs(ratio - 1.0) > 0.25) & np.isfinite(df['FL1-A'].to_numpy())


def test_histogram_overlay_shows_the_removed_events():
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_hex
    root, ed, df = _editor()
    try:
        assert int(_doublets(df).sum()) == N_DOUBLE       # the data's design
        ed.show_removed_var.set(True)
        ax = _render(ed, 'histogram')
        (curve,) = [ln for ln in ax.get_lines()
                    if ln.get_label().startswith('cleaned-out')]
        assert curve.get_label() == 'cleaned-out (n=1,200, location)'
        assert to_hex(curve.get_color()) == RED
        assert curve.get_linewidth() == 1.5 and curve.get_zorder() == 6
        xs, ys = (np.asarray(v, dtype=float) for v in curve.get_data())
        assert xs.size == 256                              # one per bin
        peak_at = xs[int(np.argmax(ys))]
        assert abs(np.log10(peak_at) - np.log10(DOUBLET_FL1)) < 0.15
        # Scaled to peak at 85% of the axis height (location, not size).
        assert ys.max() == pytest.approx(0.85 * ax.get_ylim()[1], rel=1e-6)
        (fill,) = [c for c in ax.collections if isinstance(c, PolyCollection)
                   and c.get_zorder() == 5]
        assert to_hex(fill.get_facecolor()[0]) == RED
        assert fill.get_alpha() == pytest.approx(0.22)
        legend = ax.get_legend()
        assert legend is not None
        texts = {t.get_text(): t for t in legend.get_texts()}
        assert texts[curve.get_label()].get_fontsize() == 8

        ed.show_removed_var.set(False)                    # toggle off: gone
        ax = _render(ed, 'histogram')
        assert not [ln for ln in ax.get_lines()
                    if ln.get_label().startswith('cleaned-out')]
    finally:
        root.destroy()


def _overlay_dots(ed, ax):
    from matplotlib.colors import to_hex
    colour = ed._METHOD_COLORS['doublets']
    return [c for c in ax.collections
            if c.get_facecolors().shape[0]
            and to_hex(c.get_facecolors()[0]) == colour
            and c.get_zorder() == 5]


def _rows(a):
    return sorted(map(tuple, np.asarray(a, dtype=float)))


def test_scatter_overlay_colours_removed_events_by_method():
    root, ed, df = _editor()
    try:
        ed.show_removed_var.set(True)
        ax = _render(ed, 'dot', 'FL2-A')
        (dots,) = _overlay_dots(ed, ax)
        assert dots.get_sizes()[0] == 7 and dots.get_rasterized()
        assert dots.get_alpha() == pytest.approx(0.85)
        want = df.loc[_doublets(df), ['FL1-A', 'FL2-A']].to_numpy()
        assert _rows(dots.get_offsets()) == _rows(want)    # all, 1:1
        legend = ax.get_legend()
        assert legend is not None
        assert legend.get_title().get_text() == 'cleaned-out'
        assert [t.get_text() for t in legend.get_texts()] == [
            'doublets (n=1,200)']
        assert legend.get_frame().get_alpha() == pytest.approx(0.85)

        ed.show_removed_var.set(False)
        ax = _render(ed, 'dot', 'FL2-A')
        assert not [c for c in ax.collections if c.get_zorder() == 5]
    finally:
        root.destroy()


def test_scatter_overlay_thins_with_the_plot_but_counts_everything():
    """With the plot capped, the overlay shows the same fraction of the
    removed events (distinct ones, never repeats) and the legend still
    counts all of them."""
    root, ed, df = _editor()
    try:
        ed.ds_display_var.set(True)
        ed.max_points_var.set(3000)
        shown, total = ed._sample_display_count('s1')
        assert 0 < shown < total                           # the cap bites
        ed.show_removed_var.set(True)
        ax = _render(ed, 'dot', 'FL2-A')
        (dots,) = _overlay_dots(ed, ax)
        pts = _rows(dots.get_offsets())
        assert len(pts) == int(round(shown / total * N_DOUBLE))
        assert len(set(pts)) == len(pts)                   # no repeats
        want = set(_rows(df.loc[_doublets(df), ['FL1-A', 'FL2-A']]))
        assert set(pts) <= want
        assert [t.get_text() for t in ax.get_legend().get_texts()] == [
            'doublets (n=1,200)']
    finally:
        root.destroy()
