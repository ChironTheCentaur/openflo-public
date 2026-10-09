"""Analyze -> Group comparison... reports the right medians and p-values.

ui_group_stats.GroupStatsWindow._run was only constructed by the suite (58 %
of _run ran, on an editor with no samples). Six samples with per-sample
medians fixed by construction give a hand-checkable answer:

    trial D1: medians 10, 12, 14      trial D2: medians 20, 22, 24

Each sample's values are skewed (51 at the median, 50 spread above it), so the
per-sample MEDIAN (what the window claims to compare) differs from the mean.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(__file__))

from test_gui_smoke import _editor_or_skip  # noqa: E402

SAMPLES = [('A1', 'D1', 10), ('A2', 'D1', 12), ('A3', 'D1', 14),
           ('B1', 'D2', 20), ('B2', 'D2', 22), ('B3', 'D2', 24)]


def _run(ed, ch):
    from openflo.ui_group_stats import GroupStatsWindow
    w = GroupStatsWindow(ed)
    w.withdraw()
    w._ch.set(ch)
    w._run()
    return w._txt.get('1.0', 'end')


def _setup():
    from openflo.pipeline import FlowSample
    root, ed, _gui = _editor_or_skip()
    for name, trial, med in SAMPLES:
        cd4 = med + np.concatenate([np.zeros(51), np.linspace(1, 100, 50)])
        cd8 = np.full(101, np.nan) if name == 'A3' else np.arange(101.0)
        s = FlowSample.from_dataframe(pd.DataFrame({'CD4-A': cd4, 'CD8-A': cd8}),
                                      name=name)
        ed._samples[name] = s
        ed._sample_order.append(name)
        ed._sample_trial[name] = trial
    return root, ed


def test_group_comparison_text_matches_hand_computation():
    from scipy.stats import kruskal
    root, ed = _setup()
    try:
        txt = _run(ed, 'CD4-A')
        assert 'D1: n=3  median=12' in txt
        assert 'D2: n=3  median=22' in txt
        kw = kruskal([10, 12, 14], [20, 22, 24])
        assert f"p={kw.pvalue:.4g}" in txt and 'k=2 groups' in txt
        # 3 v 3 fully separated: exact Mann-Whitney p = 0.1; one pair -> BH = p.
        assert "D1 vs D2:  p=0.1  p_adj=0.1  Cliff's δ=-1.00" in txt
    finally:
        root.destroy()


def test_a_compensation_control_is_not_a_replicate():
    from openflo.pipeline import FlowSample
    root, ed = _setup()
    try:
        comp = 'Compensation Controls_FL4 Stained Control'
        cd4 = np.full(101, 1000.0)
        ed._samples[comp] = FlowSample.from_dataframe(
            pd.DataFrame({'CD4-A': cd4, 'CD8-A': cd4}), name=comp)
        ed._sample_order.append(comp)
        ed._sample_trial[comp] = 'D1'
        txt = _run(ed, 'CD4-A')
        assert 'D1: n=3  median=12' in txt
        assert f'Left out (compensation controls): {comp}' in txt
    finally:
        root.destroy()


def test_only_comp_controls_or_flagged_samples_are_left_out():
    # The old substring rule also took 'Control_1' and 'Compound_A'; those
    # are real samples and stay in their trial.
    from openflo.pipeline import FlowSample
    root, ed = _setup()
    try:
        comp = 'Compensation Controls_FL4 Stained Control'
        extra = [('Control_1', 'D1', 11), ('Vehicle control', 'D1', 13),
                 ('Compound_A_10uM', 'D2', 21), (comp, 'D1', 1000)]
        for name, trial, med in extra:
            cd4 = med + np.concatenate([np.zeros(51), np.linspace(1, 100, 50)])
            ed._samples[name] = FlowSample.from_dataframe(
                pd.DataFrame({'CD4-A': cd4, 'CD8-A': cd4}), name=name)
            ed._sample_order.append(name)
            ed._sample_trial[name] = trial
        txt = _run(ed, 'CD4-A')
        assert 'D1: n=5  median=12' in txt          # 10, 11, 12, 13, 14
        assert 'D2: n=4  median=21.5' in txt        # 20, 21, 22, 24
        assert f'Left out (compensation controls): {comp}' in txt
        # A Comps/Samples flag the user set overrides the name either way.
        ed._sample_is_comp['Vehicle control'] = True
        ed._sample_is_comp[comp] = False
        txt = _run(ed, 'CD4-A')
        assert 'D1: n=5  median=12' in txt          # 10, 11, 12, 14, 1000
        assert 'Left out (compensation controls): Vehicle control\n' in txt
    finally:
        root.destroy()


def test_an_unticked_sample_still_counts():
    # The tree's tick only controls the plot; every loaded sample counts.
    root, ed = _setup()
    try:
        ed._sample_plot_enabled['A1'] = False
        txt = _run(ed, 'CD4-A')
        assert 'D1: n=3  median=12' in txt
    finally:
        root.destroy()


def test_an_all_nan_channel_is_not_a_replicate():
    root, ed = _setup()
    try:
        txt = _run(ed, 'CD8-A')
        assert 'D1: n=2  median=50' in txt     # A3 has no finite CD8 value
        assert 'D2: n=3  median=50' in txt
        assert 'nan' in txt.split('Omnibus', 1)[1].split('\n', 1)[0]  # all equal
    finally:
        root.destroy()


def test_group_medians_are_linear_on_transformed_data():
    """Logicle-stored samples (as the loader leaves them) must report the
    linear per-sample medians, the numbers the Statistics window gives."""
    root, ed = _setup()
    try:
        for s in ed._samples.values():
            s.apply_transform(channels=['CD4-A'])
        txt = _run(ed, 'CD4-A')
        assert 'linear' in txt.split('\n', 1)[0]
        assert 'D1: n=3  median=12' in txt
        assert 'D2: n=3  median=22' in txt
    finally:
        root.destroy()
