"""Analyze -> Marker expression: per-sample medians, grouping and exports.

ui_expression.MarkerExpressionWindow's ridgeline (6 %) and export paths were
barely run. Each sample here has skewed CD4 values with a known median (the
mean differs), and the tokens split them into Stim / Ctrl. Ticking a sample
in the tree only controls the plot: an unticked sample still counts.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(__file__))

from test_gui_smoke import _editor_or_skip  # noqa: E402

DESIGN = [('Stim_1', 300, True), ('Stim_2', 320, True), ('Ctrl_1', 100, True),
          ('Ctrl_2', 110, True)]


def _window(monkeypatch, design=DESIGN):
    from openflo.pipeline import FlowSample
    root, ed, gui = _editor_or_skip()
    monkeypatch.setattr(gui.messagebox, 'showinfo', lambda *a, **k: None)
    for name, med, enabled in design:
        # 51 values at the median, 50 far above it: median = med, mean > med.
        cd4 = np.r_[np.full(51, float(med)), np.linspace(med + 1, med + 1000, 50)]
        s = FlowSample.from_dataframe(pd.DataFrame({'CD4-A': cd4}), name=name,
                                      labels={'CD4-A': 'CD4'})
        ed._samples[name] = s
        ed._sample_order.append(name)
        ed._sample_trial[name] = 'T'
        ed._sample_plot_enabled[name] = enabled
        ed._sample_gates[name] = {}
        ed._sample_gate_order[name] = []
    ed._trial_order.append('T')
    ed._channels = ['CD4-A']
    ed._channel_labels = {'CD4-A': 'CD4'}
    ed._active_sample = 'Stim_1'
    from openflo.ui_expression import MarkerExpressionWindow
    w = MarkerExpressionWindow(ed)
    w.withdraw()
    w.factor_var.set('Name token')
    w.tokens_var.set('Stim, Ctrl')
    w._rebuild()
    return root, w


def test_medians_grouping_and_test(monkeypatch):
    root, w = _window(monkeypatch)
    try:
        # Values are per-sample MEDIANS.
        assert w._medians == {'Stim': [300.0, 320.0], 'Ctrl': [100.0, 110.0]}
        txt = w._summary.get('1.0', 'end')
        assert 'Mann-Whitney U' in txt
        assert 'Stim: n=2  median-of-medians=310' in txt
        assert 'Ctrl: n=2  median-of-medians=105' in txt
        # 2 v 2 fully separated: exact two-sided Mann-Whitney p = 2/6.
        assert 'p = 0.3333' in txt
    finally:
        root.destroy()


def test_an_unticked_sample_still_counts(monkeypatch, tmp_path):
    """The tree's tick only controls the plot. The window used to take the
    ticked samples only, while Frequencies and Group comparison take every
    loaded sample, so one experiment gave two different n."""
    import openflo.ui_expression as ue
    root, w = _window(monkeypatch, DESIGN + [('Ctrl_3', 120, False)])
    try:
        assert w._medians == {'Stim': [300.0, 320.0],
                              'Ctrl': [100.0, 110.0, 120.0]}
        assert 'Ctrl: n=3' in w._summary.get('1.0', 'end')
        out = tmp_path / 'prism.csv'
        monkeypatch.setattr(ue.filedialog, 'asksaveasfilename',
                            lambda **k: str(out))
        w._export_prism()
        assert pd.read_csv(out)['Ctrl'].tolist() == pytest.approx(
            [100.0, 110.0, 120.0])
    finally:
        root.destroy()


COMP = 'Compensation Controls_FL4 Stained Control'
GATED = [('Stim_1', 300, 'T'), ('Stim_2', 320, 'T'), ('Ctrl_1', 100, 'T'),
         ('Ctrl_2', 110, 'T'), ('Bead_X', 5000, 'T'), (COMP, 1000, 'T')]


def _gated_window(monkeypatch, samples=GATED):
    """Every sample has 100 debris events (FSC 0, CD4 0) next to 100 cells
    (FSC 1, CD4 = the value above) and a 'Cells' gate (FSC > 0.5). Median
    over the gated cells = the value; over all events = value / 2."""
    from openflo.pipeline import FlowSample
    root, ed, gui = _editor_or_skip()
    monkeypatch.setattr(gui.messagebox, 'showinfo', lambda *a, **k: None)
    for name, v, trial in samples:
        df = pd.DataFrame({'FSC-A': np.r_[np.zeros(100), np.ones(100)],
                           'CD4-A': np.r_[np.zeros(100), np.full(100, float(v))]})
        s = FlowSample.from_dataframe(df, name=name, labels={'CD4-A': 'CD4'})
        ed._samples[name] = s
        ed._sample_order.append(name)
        ed._sample_trial[name] = trial
        ed._sample_plot_enabled[name] = True
        ed._sample_gates[name] = {'g1': {
            'kind': 'threshold', 'channel': 'FSC-A', 'value': 0.5,
            'parent_id': None, 'name': 'Cells', 'enabled': True}}
        ed._sample_gate_order[name] = ['g1']
    ed._trial_order.append('T')
    ed._channels = ['FSC-A', 'CD4-A']
    ed._channel_labels = {'FSC-A': 'FSC-A', 'CD4-A': 'CD4'}
    ed._active_sample = 'Stim_1'
    from openflo.ui_expression import MarkerExpressionWindow
    w = MarkerExpressionWindow(ed)
    w.withdraw()
    w.marker_var.set(ed._fmt_channel('CD4-A'))
    w.factor_var.set('Name token')
    w.tokens_var.set('Stim, Ctrl')
    return root, w


def test_a_gated_population_gives_gated_medians(monkeypatch, tmp_path):
    # The Update tooltip promises the current gates; the window used to take
    # every event, so the medians and the Prism export were half the gated ones.
    import openflo.ui_expression as ue
    root, w = _gated_window(monkeypatch)
    try:
        assert tuple(w._pop_combo['values']) == ('All events', 'Cells')
        w.pop_var.set('Cells')
        w._rebuild()
        assert w._medians == {'Stim': [300.0, 320.0], 'Ctrl': [100.0, 110.0]}
        txt = w._summary.get('1.0', 'end')
        assert 'Population: Cells' in txt and '(ungated)' not in txt
        out = tmp_path / 'prism.csv'
        monkeypatch.setattr(ue.filedialog, 'asksaveasfilename', lambda **k: str(out))
        w._export_prism()
        got = pd.read_csv(out)
        assert got['Stim'].tolist() == pytest.approx([300.0, 320.0])
        # 'All events' is still there, and the summary says it is ungated.
        w.pop_var.set('All events')
        w._rebuild()
        assert w._medians == {'Stim': [150.0, 160.0], 'Ctrl': [50.0, 55.0]}
        assert 'Population: All events (ungated)' in w._summary.get('1.0', 'end')
    finally:
        root.destroy()


def test_no_token_match_and_comp_controls_are_left_out(monkeypatch):
    root, w = _gated_window(monkeypatch)
    try:
        w.pop_var.set('Cells')
        w._rebuild()
        # Bead_X matches no token and the comp control is not a replicate:
        # neither forms an 'Other' group.
        assert set(w._medians) == {'Stim', 'Ctrl'}
        txt = w._summary.get('1.0', 'end')
        assert 'Bead_X' in txt and COMP in txt
        w.factor_var.set('Trial / day')
        w._rebuild()
        assert w._medians == {'T': [300.0, 320.0, 100.0, 110.0, 5000.0]}
    finally:
        root.destroy()


def test_real_samples_named_like_controls_are_kept(monkeypatch):
    # The old substring rule took 'Control_1', 'Vehicle control' and
    # 'Compound_A' for comp controls; only the FlowJo form is left out here.
    named = [('Control_1', 100, 'T'), ('Control_2', 110, 'T'),
             ('Control_3', 120, 'T'), ('Treated_1', 300, 'T'),
             ('Treated_2', 310, 'T'), ('Treated_3', 320, 'T'),
             ('Vehicle control', 50, 'T'), ('Compound_A_10uM', 60, 'T'),
             (COMP, 1000, 'T')]
    root, w = _gated_window(monkeypatch, named)
    try:
        w.pop_var.set('Cells')
        w.tokens_var.set('Control, Treated')
        w._rebuild()
        # 'Vehicle control' matches the typed token 'Control'; the FlowJo
        # comp control stays out even though its name holds 'Control' too.
        assert w._medians == {'Control': [100.0, 110.0, 120.0, 50.0],
                              'Treated': [300.0, 310.0, 320.0]}
        assert w._left_out == ['Compound_A_10uM', COMP]
        assert 'Mann-Whitney U' in w._summary.get('1.0', 'end')
        w.factor_var.set('Trial / day')
        w._rebuild()
        assert w._medians == {'T': [100.0, 110.0, 120.0, 300.0, 310.0,
                                    320.0, 50.0, 60.0]}
        assert w._left_out == [COMP]
    finally:
        root.destroy()


def test_ridgeline_and_prism_export(monkeypatch, tmp_path):
    import openflo.ui_expression as ue
    root, w = _window(monkeypatch)
    try:
        w.plot_var.set('Ridgeline')
        w._rebuild()
        ax = w._fig.axes[0]
        assert [t.get_text() for t in ax.get_yticklabels()] == ['Stim', 'Ctrl']
        assert len(ax.collections) == 2          # one filled ridge per group
        out = tmp_path / 'prism.csv'
        monkeypatch.setattr(ue.filedialog, 'asksaveasfilename', lambda **k: str(out))
        w._export_prism()
        got = pd.read_csv(out)
        assert list(got.columns) == ['Stim', 'Ctrl']
        assert got['Stim'].tolist() == pytest.approx([300.0, 320.0])
        assert got['Ctrl'].tolist() == pytest.approx([100.0, 110.0])
    finally:
        root.destroy()


def test_medians_are_linear_on_transformed_data(monkeypatch):
    """The loader hands the window logicle values. Its medians -- tested,
    summarised and exported to Prism -- must be the compensated linear ones
    the Statistics window reports for the same events, and say so."""
    import openflo.ui_expression as ue
    root, w = _window(monkeypatch)
    try:
        for name, _med, _on in DESIGN:
            w.editor._samples[name].apply_transform(channels=['CD4-A'])
        w.editor._channel_transform['CD4-A'] = 'logicle'
        w._rebuild()
        assert w._medians['Stim'] == pytest.approx([300.0, 320.0], rel=1e-6)
        assert w._medians['Ctrl'] == pytest.approx([100.0, 110.0], rel=1e-6)
        txt = w._summary.get('1.0', 'end')
        assert 'linear' in txt and 'median-of-medians=310' in txt
        seen = {}
        monkeypatch.setattr(ue.filedialog, 'asksaveasfilename',
                            lambda **k: seen.update(k) or '')
        w._export_prism()
        assert 'linear' in seen['initialfile']
        # The violin is drawn on the display scale; its axis says so.
        assert 'display' in w._fig.axes[0].get_ylabel()
    finally:
        root.destroy()


def test_a_gated_population_gives_linear_medians_of_its_events(monkeypatch):
    """Both rules at once: on logicle-stored data, a gated population's
    medians are the linear medians of the gated events alone, not of the
    debris beside them and not on the display scale."""
    root, w = _gated_window(monkeypatch)
    try:
        for name, _v, _trial in GATED:
            w.editor._samples[name].apply_transform(channels=['CD4-A'])
        w.editor._channel_transform['CD4-A'] = 'logicle'
        w.pop_var.set('Cells')
        w._rebuild()
        assert w._medians['Stim'] == pytest.approx([300.0, 320.0], rel=1e-6)
        assert w._medians['Ctrl'] == pytest.approx([100.0, 110.0], rel=1e-6)
        txt = w._summary.get('1.0', 'end')
        assert 'Population: Cells' in txt and 'linear' in txt
    finally:
        root.destroy()
