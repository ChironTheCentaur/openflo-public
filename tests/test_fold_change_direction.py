"""One fold-change direction everywhere, and no number made up for a zero.

Before: the same comparison (A mean 1, B mean 4) came out log2FC -2 from
differential_test (A over B) and +2 from compare_all_features and
differential_abundance (B over A), while the volcano's axis says B / A. And a
population at 0 % in one group got a 1e-9 floor: 0 % vs 0.6 % read +29.2 in
compare_all_features (-29.2 in differential_test) and set the volcano's x
range by itself.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from openflo.diffexp import differential_abundance, differential_test
from openflo.stats import compare_all_features, log2_fold_change, volcano_data


def test_the_three_apis_agree_on_b_over_a():
    a, b = [1.0, 1.0, 1.0], [4.0, 4.0, 4.0]
    dt = differential_test({'f': a}, {'f': b})[0]['log2fc']
    caf = compare_all_features({'f': {'A': a, 'B': b}})[0]['effect']
    counts = np.array([[10, 10, 10, 40, 40, 40], [90, 90, 90, 60, 60, 60]])
    da = {r['cluster']: r for r in
          differential_abundance(counts, ['A'] * 3 + ['B'] * 3)}[0]['log2fc']
    assert dt == pytest.approx(2.0)
    assert caf == pytest.approx(2.0)
    assert da == pytest.approx(2.0)


def test_a_zero_group_mean_is_infinite_not_29():
    """0 % in A, 0.6 % in B: the old floor gave +29.2 / -29.2."""
    a, b = [0.0, 0.0, 0.0], [0.5, 0.7, 0.6]
    res = compare_all_features({'pop': {'A': a, 'B': b}})
    assert res[0]['effect'] == math.inf
    assert differential_test({'pop': a}, {'pop': b})[0]['log2fc'] == math.inf
    assert differential_test({'pop': b}, {'pop': a})[0]['log2fc'] == -math.inf
    assert math.isnan(log2_fold_change(0.0, 0.0))
    assert math.isnan(log2_fold_change(-5.0, 10.0))      # negative mean
    # The volcano leaves it out instead of stretching its axis to 29.
    assert volcano_data(res) == []


def test_a_pseudocount_is_only_used_when_asked_for():
    assert log2_fold_change(0.0, 0.6, pseudocount=0.2) == \
        pytest.approx(math.log2(0.8 / 0.2))
    res = compare_all_features({'p': {'A': [0.0], 'B': [0.6]}}, eps=0.2)
    assert res[0]['effect'] == pytest.approx(2.0)


def test_effect_needs_exactly_two_groups_in_the_input():
    """Three groups, one with no data for this feature: the two left are
    not 'the' A and B the window labels, so no effect is reported."""
    res = compare_all_features({'p': {'A': [1.0, 2.0], 'B': [], 'C': [4.0, 8.0]}})
    assert res[0]['n_groups'] == 2
    assert math.isnan(res[0]['effect'])


def test_compare_all_window_states_the_direction_and_the_absent(monkeypatch):
    monkeypatch.setenv('MPLBACKEND', os.environ.get('MPLBACKEND', 'Agg'))
    tk = pytest.importorskip('tkinter')
    try:
        root = tk.Tk()
    except Exception:
        pytest.skip('no Tk display')
    root.withdraw()
    try:
        from openflo.ui_diff import CompareAllWindow
        res = compare_all_features({
            'gone': {'Ctrl': [0.4, 0.5, 0.6], 'Stim': [0.0, 0.0, 0.0]},
            'up': {'Ctrl': [1.0, 1.1, 0.9], 'Stim': [4.0, 4.2, 3.8]}})
        w = CompareAllWindow(root, res, ['Ctrl', 'Stim'], '%Parent')
        texts = []

        def walk(widget):
            for child in widget.winfo_children():
                try:
                    texts.append(str(child.cget('text')))
                except Exception:
                    pass
                walk(child)
        walk(w)
        joined = ' '.join(texts)
        assert 'log2(mean Stim / mean Ctrl)' in joined
        assert 'not on the volcano: gone' in joined
        assert [p['feature'] for p in w._volcano] == ['up']
        w.destroy()
    finally:
        root.destroy()
