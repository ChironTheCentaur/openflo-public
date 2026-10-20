"""Gates that re-fit to each sample: a threshold moves to the density valley
nearest the reference cut in each sample's own data, and auto-gate
proposals re-fit by the rule they carry; nothing changes until accepted.

Generated data: each sample is two populations on FL2-A, N(m, sd) and
N(m + 3, sd), with m drifting by sample, so the true valley is at m + 1.5.
A cut copied from the first sample misplaces the later samples' split; the
re-fit cut must land within 0.15 of each sample's own valley.
"""
import os
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openflo import gate_refit as gr
from tests.conftest import gui_unavailable


def _bimodal(seed, m, n=6000, sd=0.3):
    rng = np.random.default_rng(seed)
    lo = rng.normal(m, sd, n // 2)
    hi = rng.normal(m + 3.0, sd, n - n // 2)
    return np.concatenate([lo, hi])


def test_nearest_valley_follows_the_drift():
    for i, m in enumerate((0.0, 0.4, 0.8, -0.5)):
        cut, why = gr.nearest_valley(_bimodal(i, m), 1.5)
        assert why is None
        assert cut == pytest.approx(m + 1.5, abs=0.15)


def test_nearest_of_several_valleys_is_chosen():
    rng = np.random.default_rng(5)
    v = np.concatenate([rng.normal(0, 0.2, 3000), rng.normal(1.2, 0.2, 3000),
                        rng.normal(2.4, 0.2, 3000)])
    # Both valleys (0.6 and 1.8) are within reach of each reference;
    # the nearer one is taken.
    assert gr.nearest_valley(v, 1.3)[0] == pytest.approx(1.8, abs=0.1)
    assert gr.nearest_valley(v, 1.1)[0] == pytest.approx(0.6, abs=0.1)


@pytest.mark.parametrize('bright_share, bright_sd', [
    (0.2, 0.005), (0.3, 0.01), (0.4, 0.02), (0.5, 0.015)])
def test_a_valley_before_a_population_at_the_top_of_the_range_is_found(
        bright_share, bright_sd):
    """A tight bright population at the top of the data: the histogram's
    99.5th-percentile edge cut through its peak, and smoothing that mirrored
    the edge kept it at full height there, so the peak had no prominence and
    the empty valley before it (found on Windows, 10-19) was missed."""
    rng = np.random.default_rng(3)
    n_hi = int(4000 * bright_share)
    v = np.r_[rng.normal(0.25, 0.04, 4000 - n_hi),
              rng.normal(0.62, bright_sd, n_hi)]
    cut, why = gr.nearest_valley(v, 0.40)
    assert why is None, why
    # The empty gap between the populations: 0.25 + 4 sd to 0.62 - 4 sd.
    assert 0.40 < cut < 0.62 - 4 * bright_sd


def test_one_population_of_any_size_has_no_valley():
    # Noise in a few hundred events from one population is not a second
    # population: no sample of 200 to 20,000 normal events gets a cut.
    for n in (200, 1000, 20000):
        for seed in range(10):
            v = np.random.default_rng(seed).normal(0, 1, n)
            assert gr.nearest_valley(v, 0.3) == (None, 'one population, no valley')


def test_no_landmark_keeps_the_gate_and_says_why():
    rng = np.random.default_rng(6)
    g = {'kind': 'threshold', 'channel': 'FL2-A', 'value': 0.7}
    new, why = gr.refit(g, {'FL2-A': rng.normal(0, 1, 5000)})
    assert why == 'one population, no valley' and new == g
    new, why = gr.refit(g, {'FL2-A': np.arange(10.0)})
    assert why == 'too few events' and new == g
    far = dict(g, value=-1.0)                       # valley at 1.5 is 2.5 away
    new, why = gr.refit(far, {'FL2-A': _bimodal(7, 0.0)})
    assert why == 'no valley near the reference cut' and new == far


def test_interval_ends_refit_and_open_ends_stay():
    rng = np.random.default_rng(8)
    v = np.concatenate([rng.normal(0, 0.25, 3000), rng.normal(2.4, 0.25, 3000),
                        rng.normal(4.8, 0.25, 3000)])
    g = {'kind': 'interval', 'channel': 'FL2-A', 'lo': 1.0, 'hi': 3.8}
    new, why = gr.refit(g, {'FL2-A': v})
    assert why is None
    assert new['lo'] == pytest.approx(1.2, abs=0.15)
    assert new['hi'] == pytest.approx(3.6, abs=0.15)
    g = {'kind': 'interval', 'channel': 'FL2-A', 'lo': 1.0, 'hi': 1e12}
    new, why = gr.refit(g, {'FL2-A': v})
    assert why is None                       # the open end is not looked for
    assert new['hi'] == 1e12 and new['lo'] == pytest.approx(1.2, abs=0.15)


def test_refittable_kinds():
    assert gr.refittable({'kind': 'threshold', 'value': 0.4})
    assert not gr.refittable({'kind': 'threshold', 'value': 1e12})
    assert not gr.refittable({'kind': 'threshold', 'value': 0.4,
                              'scale_unknown': True})
    assert not gr.refittable({'kind': 'rect'})
    assert not gr.refittable({'kind': 'polygon'})
    assert gr.refittable({'kind': 'polygon', 'rule': {'method': 'singlet'}})
    assert gr.refittable({'kind': 'ellipsoid', 'rule': {'method': 'gmm'}})
    new, why = gr.refit({'kind': 'rect'}, {})
    assert why and new == {'kind': 'rect'}


def test_gmm_ellipse_follows_its_population():
    from openflo.pipeline import gmm_ellipse_gates
    rng = np.random.default_rng(9)

    def two(shift):
        a = rng.normal([0 + shift, 0], 0.3, (3000, 2))
        b = rng.normal([3 + shift, 3], 0.3, (3000, 2))
        return np.vstack([a, b])
    ref = two(0.0)
    props = gmm_ellipse_gates(ref[:, 0], ref[:, 1])
    (g,) = [dict(p, x_channel='FL1-A', y_channel='FL2-A',
                 rule={'method': 'gmm'}) for p, _i in props
            if p['mean'][0] > 1.5]
    moved = two(0.5)
    new, why = gr.refit(g, {'FL1-A': moved[:, 0], 'FL2-A': moved[:, 1]})
    assert why is None
    assert new['mean'] == [pytest.approx(3.5, abs=0.1), pytest.approx(3.0, abs=0.1)]
    gone = rng.normal([-5, -5], 0.3, (3000, 2))
    _new, why = gr.refit(g, {'FL1-A': gone[:, 0], 'FL2-A': gone[:, 1]})
    assert why == 'no population near the reference centre'


def test_singlet_rule_refits_the_band():
    rng = np.random.default_rng(10)
    h = rng.uniform(1e4, 1e5, 4000)
    a = h * rng.normal(1.3, 0.03, 4000)
    g = {'kind': 'polygon', 'x_channel': 'FSC-A', 'y_channel': 'FSC-H',
         'vertices': [[0, 0], [1, 0], [1, 1]], 'rule': {'method': 'singlet'}}
    new, why = gr.refit(g, {'FSC-A': a, 'FSC-H': h})
    assert why is None and len(new['vertices']) >= 3
    df = pd.DataFrame({'FSC-A': a, 'FSC-H': h})
    assert gr.fraction_in(new, df) > 0.95
    assert gr.fraction_in(g, df) == 0.0
    assert np.isnan(gr.fraction_in(new, df.iloc[:0]))


def test_review_figure_draws_a_panel_per_sample():
    from matplotlib.figure import Figure

    from openflo.ui_refit import draw_review
    v = _bimodal(11, 0.0)
    old = {'kind': 'threshold', 'channel': 'FL2-A', 'value': 1.0}
    new = dict(old, value=1.5)
    props = [{'sample': f's{i}', 'old': old, 'new': new,
              'why': None if i else 'kept', 'data': {'FL2-A': v},
              'before': 0.5, 'after': float('nan')} for i in range(3)]
    axes = draw_review(Figure(), props)
    assert len(axes) == 3
    assert axes[1].get_title().startswith('s1\n50.0% -> NA')
    assert 'kept' in axes[0].get_title()
    assert len(axes[1].lines) == 2


# ── In the editor ────────────────────────────────────────────────────────────

def _editor_or_skip():
    os.environ.setdefault('MPLBACKEND', 'Agg')
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


def _load(ed, name, df):
    ed._samples[name] = SimpleNamespace(
        name=name, path=f'{name}.fcs', data=df, data_transforms={},
        fluor_channels=['FL2-A'], channel_labels={c: c for c in df.columns})
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    if 'T' not in ed._trial_order:
        ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = True
    ed._channels = list(df.columns)
    ed._channel_labels = {c: c for c in df.columns}
    ed._populate_channel_combos()


def _drifting(ed):
    shifts = {'s1': 0.0, 's2': 0.9, 's3': -0.9}
    ids = {}
    for i, (n, m) in enumerate(shifts.items()):
        rng = np.random.default_rng(20 + i)
        v = _bimodal(30 + i, m, sd=0.5)
        _load(ed, n, pd.DataFrame({'FL1-A': rng.uniform(1, 2, v.size),
                                   'FL2-A': v}))
    for n in ('s3', 's2', 's1'):
        ed._set_active_sample(n)
        p = ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A',
                          'value': 1.0, 'name': 'All', 'parent_id': None})
        ids[n] = ed._add_gate({'kind': 'threshold', 'channel': 'FL2-A',
                               'value': 1.5, 'name': 'Pos', 'parent_id': p,
                               'color': '#123456'})
    ed._clear_undo_pending()
    return shifts, ids


def test_editor_replaces_the_gate_on_each_sample_with_one_undo():
    root, ed = _editor_or_skip()
    try:
        shifts, ids = _drifting(ed)
        n = ed._refit_gate_per_sample('s1', ids['s1'], review=False)
        assert n == 3
        for s, m in shifts.items():
            g = ed._sample_gates[s][ids[s]]
            assert g['value'] == pytest.approx(m + 1.5, abs=0.15)
            assert g['name'] == 'Pos' and g['color'] == '#123456'
            v = ed._samples[s].data['FL2-A'].to_numpy()
            true_pos = v.size // 2
            assert abs(int((v >= g['value']).sum()) - true_pos) < 0.01 * v.size
            # The copied cut at 1.5 misplaces the drifted samples' split.
            if m:
                assert abs(int((v >= 1.5).sum()) - true_pos) > 0.03 * v.size
        ed._undo()
        assert all(ed._sample_gates[s][ids[s]]['value'] == 1.5 for s in shifts)
    finally:
        root.destroy()


def test_review_window_installs_only_the_ticked_samples():
    root, ed = _editor_or_skip()
    try:
        _shifts, ids = _drifting(ed)
        win = ed._refit_gate_per_sample('s1', ids['s1'])
        assert win.title().startswith('Re-place on each sample')
        assert ed._sample_gates['s2'][ids['s2']]['value'] == 1.5  # not yet
        assert sorted(win.vars) == ['s1', 's2', 's3']
        win.vars['s3'].set(False)
        (acc,) = [w for w in win.winfo_children()[0].winfo_children()
                  if isinstance(w, ttk.Button) and w.cget('text') == 'Accept']
        acc.invoke()
        assert ed._sample_gates['s2'][ids['s2']]['value'] != 1.5
        assert ed._sample_gates['s3'][ids['s3']]['value'] == 1.5
        assert 're-placed on 2 sample' in ed.status_var.get()
        # A rectangle has no rule: refused with a reason.
        rid = ed._add_gate({'kind': 'rect', 'x_channel': 'FL1-A',
                            'y_channel': 'FL2-A', 'x0': 0, 'x1': 1, 'y0': 0,
                            'y1': 1, 'parent_id': None})
        assert ed._refit_gate_per_sample('s1', rid) is None
        assert 'Only threshold' in ed.status_var.get()
    finally:
        root.destroy()


def test_auto_gate_proposals_carry_their_rule():
    root, ed = _editor_or_skip()
    try:
        rng = np.random.default_rng(12)
        pts = np.vstack([rng.normal([0, 0], 0.3, (2000, 2)),
                         rng.normal([3, 3], 0.3, (2000, 2))])
        _load(ed, 's1', pd.DataFrame({'FL1-A': pts[:, 0], 'FL2-A': pts[:, 1]}))
        ed._set_active_sample('s1')
        ed.x_combo.set('FL1-A')
        ed.y_combo.set('FL2-A')
        ed._auto_gate_gmm('s1', {'max_components': 3, 'coverage': 0.9,
                                 'min_weight': 0.02})
        rules = [g.get('rule') for g in ed._gates.values()]
        assert rules and all(r == {'method': 'gmm', 'max_components': 3,
                                   'coverage': 0.9, 'min_weight': 0.02}
                             for r in rules)
    finally:
        root.destroy()
