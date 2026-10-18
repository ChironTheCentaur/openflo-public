"""Combination gates: every Boolean combination of N chosen gates as
populations in one step, and the polyfunctionality bars, pies and table.

Generated data: FL1-A ~ N(0, 1) decides the parent (FL1-A >= 0); FL2-A,
FL3-A and FL4-A ~ N(0, 1) are three independent "functions", each positive
at >= 0. Every combination's count is checked against the same rule written
directly in numpy, and against the count the gate tree gives each boolean
gate. Synthetic sample names only.
"""
import os
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openflo import gate_combos as gc
from openflo.gating import gate_counts
from openflo.pipeline import wsp_unexportable
from tests.conftest import gui_unavailable

CH = ['FL1-A', 'FL2-A', 'FL3-A', 'FL4-A']


def _frame(seed, n=4000, shift=0.0, nan_fl2=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(shift, 1, n) for c in CH})
    if nan_fl2:
        df.loc[:nan_fl2 - 1, 'FL2-A'] = np.nan
    return df


def _tree():
    """Parent P (FL1-A >= 0) with three function gates under it."""
    return {
        'p': {'kind': 'threshold', 'channel': 'FL1-A', 'value': 0.0,
              'name': 'P', 'parent_id': None},
        'a': {'kind': 'threshold', 'channel': 'FL2-A', 'value': 0.0,
              'name': 'A', 'parent_id': 'p'},
        'b': {'kind': 'threshold', 'channel': 'FL3-A', 'value': 0.0,
              'name': 'B', 'parent_id': 'p'},
        'c': {'kind': 'threshold', 'channel': 'FL4-A', 'value': 0.0,
              'name': 'C', 'parent_id': 'p'},
    }


def _adder(gates):
    seq = [0]

    def add(g):
        seq[0] += 1
        gid = f'n{seq[0]}'
        gates[gid] = g
        return gid
    return add


def test_every_combination_counts_what_its_rule_says():
    gates, df = _tree(), _frame(1, nan_fl2=50)
    res = gc.build_combinations(gates, ['a', 'b', 'c'], _adder(gates))
    assert res['parent'] == 'p' and res['labels'] == ['A', 'B', 'C']
    assert len(res['combos']) == 8 and res['added'] == 3 + 8
    p = df['FL1-A'].to_numpy() >= 0
    v = {k: df[c].to_numpy() for k, c in zip('abc', CH[1:], strict=True)}
    measured = np.isfinite(v['a'])
    n_par, counts = gc.combination_counts(gates, df, res['combos'], 'p')
    assert n_par == int(p.sum())
    for (signs, gid), n in zip(res['combos'], counts, strict=True):
        want = p.copy()
        for k, s in zip('abc', signs, strict=True):
            want &= (v[k] >= 0) if s else (~(v[k] >= 0) & np.isfinite(v[k]))
        assert n == int(want.sum()), gates[gid]['name']
        # ... and what the gate tree shows for that boolean gate.
        assert gate_counts(gates, gid, df)[0] == n
        assert gates[gid]['parent_id'] == 'p'
    # The combinations partition the measured part of the parent: an event
    # with no FL2-A value is in none of them, not counted negative.
    assert sum(counts) == int((p & measured).sum())
    assert int((p & ~measured).sum()) > 0


def test_order_names_and_reuse():
    gates = _tree()
    res = gc.build_combinations(gates, ['a', 'b'], _adder(gates))
    names = [gates[g]['name'] for _s, g in res['combos']]
    assert names == ['A+ B+', 'A+ B-', 'A- B+', 'A- B-']
    nots = [g for g in gates.values() if g.get('op') == 'not']
    assert sorted(g['name'] for g in nots) == ['A-', 'B-']
    n = len(gates)
    again = gc.build_combinations(gates, ['a', 'b'], _adder(gates))
    assert again['added'] == 0 and len(gates) == n
    assert [g for _s, g in again['combos']] == [g for _s, g in res['combos']]


def test_check_combinable_and_common_parent():
    gates = _tree()
    assert gc.check_combinable(gates, ['a']) == "Choose at least two gates."
    many = {f'g{i}': {'kind': 'threshold', 'channel': 'FL2-A', 'value': i,
                      'parent_id': None} for i in range(7)}
    assert 'at most 6' in gc.check_combinable(many, list(many))
    assert 'under another' in gc.check_combinable(gates, ['p', 'a'])
    assert gc.check_combinable(gates, ['a', 'b']) is None
    assert gc.common_parent(gates, ['a', 'b']) == 'p'
    gates['d'] = {'kind': 'threshold', 'channel': 'FL4-A', 'value': 1,
                  'parent_id': 'a'}
    assert gc.common_parent(gates, ['d', 'b']) == 'p'
    gates['r'] = {'kind': 'threshold', 'channel': 'FL4-A', 'value': 1,
                  'parent_id': None}
    assert gc.common_parent(gates, ['r', 'b']) is None
    with pytest.raises(ValueError):
        gc.build_combinations(gates, ['a'], _adder(gates))


def test_negative_is_within_the_parent_not_everything_else():
    gates, df = _tree(), _frame(2)
    res = gc.build_combinations(gates, ['a', 'b'], _adder(gates))
    _n, counts = gc.combination_counts(gates, df, res['combos'], 'p')
    p = df['FL1-A'].to_numpy() >= 0
    both_neg = p & (df['FL2-A'].to_numpy() < 0) & (df['FL3-A'].to_numpy() < 0)
    assert counts[-1] == int(both_neg.sum())
    assert counts[-1] < int(((df['FL2-A'] < 0) & (df['FL3-A'] < 0)).sum())


def test_unknown_operand_reads_na_not_zero():
    gates, df = _tree(), _frame(3).drop(columns=['FL4-A'])
    res = gc.build_combinations(gates, ['a', 'c'], _adder(gates))
    _n, counts = gc.combination_counts(gates, df, res['combos'], 'p')
    assert all(np.isnan(c) for c in counts)
    gates2 = _tree()
    gates2['p']['channel'] = 'FL9-A'
    res2 = gc.build_combinations(gates2, ['a', 'b'], _adder(gates2))
    n_par, counts2 = gc.combination_counts(gates2, _frame(3), res2['combos'],
                                           'p')
    assert np.isnan(n_par) and all(np.isnan(c) for c in counts2)


def test_rows_degrees_and_means():
    gates, df = _tree(), _frame(4)
    res = gc.build_combinations(gates, ['a', 'b'], _adder(gates))
    n_par, counts = gc.combination_counts(gates, df, res['combos'], 'p')
    rows = gc.frequency_rows({'s1': (n_par, res['combos'], counts),
                              's2': (0, res['combos'], [0, 0, 0, 0])},
                             ['A', 'B'], {'s1': 'G1', 's2': 'G1'})
    s1 = [r for r in rows if r['Sample'] == 's1']
    assert [r['Positives'] for r in s1] == [2, 1, 1, 0]
    assert sum(r['% of parent'] for r in s1) == pytest.approx(100.0)
    assert s1[0]['% of parent'] == pytest.approx(100.0 * counts[0] / n_par)
    s2 = [r for r in rows if r['Sample'] == 's2']
    assert all(np.isnan(r['% of parent']) for r in s2)       # empty parent
    deg = gc.degree_fractions(rows, 2)
    assert sum(deg['s1']) == pytest.approx(1.0)
    assert deg['s1'][2] == pytest.approx(counts[0] / n_par)
    means = gc.group_means(rows)
    assert means['G1']['A+ B+'] == pytest.approx(s1[0]['% of parent'])


def test_combinations_export_to_flowjo():
    gates = _tree()
    gc.build_combinations(gates, ['a', 'b', 'c'], _adder(gates))
    assert not wsp_unexportable(gates)


def test_polyfunctionality_figure_and_table(tmp_path):
    from matplotlib.figure import Figure

    from openflo.ui_polyfunction import bar_order, draw_polyfunctionality, write_table
    gates, df = _tree(), _frame(5)
    res = gc.build_combinations(gates, ['a', 'b', 'c'], _adder(gates))
    n_par, counts = gc.combination_counts(gates, df, res['combos'], 'p')
    rows = gc.frequency_rows({'s1': (n_par, res['combos'], counts),
                              's2': (0, res['combos'], [0] * 8)},
                             ['A', 'B', 'C'], {'s1': 'G1', 's2': 'G2'})
    assert len(bar_order(['A', 'B', 'C'])) == 7
    assert len(bar_order(['A', 'B', 'C'], show_negative=True)) == 8
    fig = Figure()
    ax, grid, pies = draw_polyfunctionality(fig, rows, ['A', 'B', 'C'])
    assert len(ax.patches) == 2 * 7              # two groups, 7 bars each
    assert len(pies) == 2
    assert [t.get_text() for t in pies[0].texts[:3]] == ['1', '2', '3']
    assert len(grid.lines) == 3 * 7
    path = tmp_path / 'combos.csv'
    write_table(str(path), rows)
    out = pd.read_csv(path, keep_default_na=False)
    assert len(out) == 16 and (out[out.Sample == 's2']['% of parent'] == 'NA').all()
    assert float(out.iloc[0]['% of parent']) == pytest.approx(
        100.0 * counts[0] / n_par, abs=1e-4)


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


def _load(ed, name, df, trial='T'):
    cols = list(df.columns)
    ed._samples[name] = SimpleNamespace(
        name=name, path=f'{name}.fcs', data=df, data_transforms={},
        fluor_channels=CH[1:], channel_labels={c: c for c in cols})
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = trial
    if trial not in ed._trial_order:
        ed._trial_order.append(trial)
    ed._sample_plot_enabled[name] = True
    ed._channels = cols
    ed._channel_labels = {c: c for c in cols}
    ed._populate_channel_combos()


def _install_tree(ed, name):
    ed._set_active_sample(name)
    ids = {}
    for k, g in _tree().items():
        g = dict(g, parent_id=ids.get(g['parent_id']))
        ids[k] = ed._add_gate(g)
    return ids


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def test_editor_makes_combinations_on_every_sample_with_one_undo(tmp_path):
    root, ed = _editor_or_skip()
    try:
        _load(ed, 's1', _frame(11), 'G1')
        _load(ed, 's2', _frame(12, shift=0.3), 'G2')
        _load(ed, 's3', _frame(13))
        _install_tree(ed, 's3')
        ed._sample_gates['s3'] = {k: g for k, g in ed._sample_gates['s3'].items()
                                  if g.get('name') != 'C'}
        ids2 = _install_tree(ed, 's2')
        ids1 = _install_tree(ed, 's1')
        before = {n: len(ed._sample_gates[n]) for n in ('s1', 's2', 's3')}
        ed._clear_undo_pending()          # the tree above is one earlier step
        built = ed._make_combinations([ids1['a'], ids1['b'], ids1['c']],
                                      every_sample=True, show=False)
        assert sorted(built) == ['s1', 's2']                  # s3 lacks C
        assert len(ed._sample_gates['s1']) == before['s1'] + 11
        assert len(ed._sample_gates['s2']) == before['s2'] + 11
        assert len(ed._sample_gates['s3']) == before['s3']
        assert 'lack these gates' in ed.status_var.get()
        # s2's combinations use s2's own gates, under s2's own parent.
        for _s, gid in built['s2']['combos']:
            g = ed._sample_gates['s2'][gid]
            assert g['parent_id'] == ids2['p']
            assert all(o in ed._sample_gates['s2'] for o in g['operands'])
        df2 = ed._samples['s2'].data
        n_par, counts = gc.combination_counts(
            ed._sample_gates['s2'], df2, built['s2']['combos'], ids2['p'])
        allpos = ((df2['FL1-A'] >= 0) & (df2['FL2-A'] >= 0)
                  & (df2['FL3-A'] >= 0) & (df2['FL4-A'] >= 0))
        assert counts[0] == int(allpos.sum())
        rows = ed._combination_rows(built, ['A', 'B', 'C'])
        assert {r['Group'] for r in rows} == {'G1', 'G2'}
        win = ed._show_polyfunctionality(built, ['A', 'B', 'C'])
        assert win.title().startswith('Polyfunctionality')
        assert win.export_table(str(tmp_path / 't.csv'))
        win.neg_var.set(True)
        win.redraw()
        win.destroy()
        ed._clear_undo_pending()
        ed._undo()
        assert {n: len(ed._sample_gates[n]) for n in ('s1', 's2', 's3')} == before
        # The active sample already has them: only s2 changes, and that
        # change is still one undo step of its own.
        ed._make_combinations([ids1['a'], ids1['b']], every_sample=False,
                              show=False)
        ed._clear_undo_pending()
        n1, n2 = len(ed._sample_gates['s1']), len(ed._sample_gates['s2'])
        ed._make_combinations([ids1['a'], ids1['b']], every_sample=True,
                              show=False)
        assert len(ed._sample_gates['s1']) == n1
        assert len(ed._sample_gates['s2']) == n2 + 6
        ed._clear_undo_pending()
        ed._undo()
        assert len(ed._sample_gates['s2']) == n2
        assert len(ed._sample_gates['s1']) == n1
    finally:
        root.destroy()


def test_dialog_creates_the_chosen_combinations():
    root, ed = _editor_or_skip()
    try:
        _load(ed, 's1', _frame(21))
        ids = _install_tree(ed, 's1')
        dlg = ed._open_combination_dialog()
        boxes = {w.cget('text'): w for w in _walk(dlg)
                 if isinstance(w, ttk.Checkbutton)}
        assert 'P/A' in boxes and 'P/B' in boxes
        boxes['P/A'].invoke()
        boxes['P/B'].invoke()
        labels = [w.cget('text') for w in _walk(dlg) if isinstance(w, ttk.Label)]
        assert any(t.startswith('4 combinations of A, B, as % of P')
                   for t in labels)
        (create,) = [w for w in _walk(dlg) if isinstance(w, ttk.Button)
                     and w.cget('text') == 'Create']
        create.invoke()
        names = {g.get('name') for g in ed._sample_gates['s1'].values()}
        assert {'A+ B+', 'A+ B-', 'A- B+', 'A- B-', 'A-', 'B-'} <= names
        (win,) = [w for w in _walk(ed.master) if isinstance(w, tk.Toplevel)
                  and w.title().startswith('Polyfunctionality')]
        win.destroy()
        assert ids
    finally:
        root.destroy()
