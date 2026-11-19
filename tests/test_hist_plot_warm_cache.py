"""A histogram drawn from a warm plot cache reads the same as from a cold
one: the FMO curve still carries its % above the gate (the values are
fetched for it when the cache held the counts), and a change of display
scale recounts the events that scale cannot place. Also the fill of the
FMO curve and of a sample's curve.

Generated data: s1 and an FMO control 'fmo' with FL1-A ~ N(0.3, 0.4)
(about a fifth non-positive), a threshold gate FL1-A >= 0.45 on each.
"""
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

from tests.test_fmo_gating_dialog import _editor_or_skip, _register_fl

CUT = 0.45


def _frame(seed, n=3000):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'FSC-A': rng.normal(0.6, 0.1, n),
                         'FL1-A': rng.normal(0.3, 0.4, n),
                         'FL2-A': rng.normal(0.3, 0.05, n)})


def _setup(ed):
    s1, fmo = _frame(1), _frame(2)
    _register_fl(ed, 's1', s1)
    _register_fl(ed, 'fmo', fmo)
    for name in ('fmo', 's1'):
        ed._set_active_sample(name)
        ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A', 'value': CUT,
                      'name': 'FL1+', 'parent_id': None, 'enabled': True,
                      'color': '#d62728'})
    ed._set_fmo_controls({'FL1-A': 'fmo'})
    return s1, fmo


def _render(ed):
    ax = Figure().add_subplot()
    ed._render_into(ax, ['s1'], 'FL1-A', None, 'histogram', None,
                    population=None)
    return ax


def _labels(ax):
    return sorted(ln.get_label() for ln in ax.get_lines()
                  if not ln.get_label().startswith('_'))


def test_a_warm_cache_keeps_the_fmo_percent():
    root, ed = _editor_or_skip()
    try:
        s1, fmo = _setup(ed)
        cold = _labels(_render(ed))
        warm = _labels(_render(ed))           # counts now come from the cache
        pct_f = 100.0 * np.mean(fmo['FL1-A'].to_numpy() >= CUT)
        pct_s = 100.0 * np.mean(s1['FL1-A'].to_numpy() >= CUT)
        assert cold == warm == sorted([f'FMO: fmo; ≥ FL1+: {pct_f:.1f}%',
                                       f's1; ≥ FL1+: {pct_s:.1f}%'])
    finally:
        root.destroy()


def test_fills_grey_behind_for_the_fmo_and_coloured_for_the_sample():
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        ax = _render(ed)
        fills = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert [round(float(c.get_alpha()), 2) for c in fills] == [0.30, 0.22]
        lines = {ln.get_label().split(';')[0]: ln for ln in ax.get_lines()}
        assert lines['FMO: fmo'].get_linestyle() == '--'
        assert lines['FMO: fmo'].get_linewidth() == 1.2
        assert lines['s1'].get_linewidth() == 1.4
    finally:
        root.destroy()


def _note(ax):
    return [t.get_text() for t in ax.texts if 'log axis' in t.get_text()]


def test_a_new_scale_recounts_what_it_cannot_show():
    root, ed = _editor_or_skip()
    try:
        s1, _fmo = _setup(ed)
        ed._fmo_hist_var.set(False)            # s1 alone
        ed._channel_scale['FL1-A'] = 'linear'
        assert _note(_render(ed)) == []        # linear shows every event
        ed._channel_scale['FL1-A'] = 'log'     # cache still holds linear's
        hidden = int(np.sum(s1['FL1-A'].to_numpy() <= 0))
        assert hidden > 0
        (note,) = _note(_render(ed))
        assert note.startswith(f'{hidden:,} of {len(s1):,} events')
    finally:
        root.destroy()
