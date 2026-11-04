"""Significance brackets over a group plot come from one helper,
``plotmath.draw_significance`` (best-practices survey R8).

The marker expression and frequency comparison windows each carried a
copy of the same bracket drawer; the copies had drifted (where the first
bracket sits, whether the y limit grows to show them, whether non-finite
values are skipped). The helper keeps each window's look through its
``gap`` and ``fit_top`` arguments. These tests pin which pairs get a
bracket, where each one is drawn, and that no window grows its own copy
again.
"""
from __future__ import annotations

import ast
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import numpy as np
import pytest
from matplotlib.figure import Figure

from openflo.plotmath import (
    MAX_SIGNIFICANCE_BRACKETS,
    draw_significance,
    significance_pairs,
)

SRC = Path(__file__).resolve().parent.parent / 'src' / 'openflo'
INK = '#123456'
GROUPS3 = ['GroupA', 'GroupB', 'GroupC']


def _ax():
    return Figure().add_subplot(1, 1, 1)


def _pair(a, b, p):
    return {'a': a, 'b': b, 'p_adj': p}


# ── which pairs ─────────────────────────────────────────────────────────

@pytest.mark.parametrize('p, stars', [
    (0.00001, '****'), (0.0005, '***'), (0.005, '**'), (0.03, '*'),
    (0.4, 'ns')])
def test_two_groups_bracket_the_overall_p_ns_included(p, stars):
    assert significance_pairs(['GroupA', 'GroupB'], {'p': p}) == [
        (0, 1, stars)]


@pytest.mark.parametrize('p', [None, float('nan')])
def test_two_groups_without_a_p_get_no_bracket(p):
    assert significance_pairs(['GroupA', 'GroupB'], {'p': p}) == []


def test_two_groups_ignore_posthoc():
    res = {'p': 0.5, 'posthoc': [_pair('GroupA', 'GroupB', 0.001)]}
    assert significance_pairs(['GroupA', 'GroupB'], res) == [(0, 1, 'ns')]


def test_more_groups_bracket_only_significant_drawn_pairs():
    res = {'p': 0.001, 'posthoc': [
        _pair('GroupA', 'GroupB', 0.2),         # ns: left out
        _pair('GroupA', 'GroupC', 0.004),
        _pair('GroupB', 'Stranger', 0.0001),    # not drawn: left out
        _pair('GroupB', 'GroupC', None),        # no p: left out
        _pair('GroupC', 'GroupB', 0.04),
    ]}
    assert significance_pairs(GROUPS3, res) == [
        (2, 1, '*'), (0, 2, '**')]


def test_more_groups_put_the_closest_pairs_first_and_keep_order_on_ties():
    groups = [f"Group{k}" for k in range(5)]
    res = {'posthoc': [
        _pair('Group0', 'Group4', 0.01),
        _pair('Group1', 'Group2', 0.02),
        _pair('Group0', 'Group2', 0.03),
        _pair('Group3', 'Group4', 0.0001),
    ]}
    assert significance_pairs(groups, res) == [
        (1, 2, '*'), (3, 4, '****'), (0, 2, '*'), (0, 4, '**')]


def test_more_groups_cap_the_bracket_count():
    groups = [f"Group{k}" for k in range(6)]
    res = {'posthoc': [_pair(groups[i], groups[j], 0.001)
                       for i in range(6) for j in range(i + 1, 6)]}
    pairs = significance_pairs(groups, res)
    assert len(pairs) == MAX_SIGNIFICANCE_BRACKETS == 6
    assert [abs(j - i) for i, j, _ in pairs] == [1] * 5 + [2]
    assert len(significance_pairs(groups, res, max_pairs=2)) == 2


def test_more_groups_without_posthoc_get_no_bracket():
    assert significance_pairs(GROUPS3, {'p': 0.0001}) == []


# ── where they are drawn ────────────────────────────────────────────────

def _lines(ax):
    return [(list(ln.get_xdata()), list(ln.get_ydata())) for ln in ax.lines]


def test_brackets_stack_above_the_highest_value():
    # Values 0..10: span 10, bracket height 0.5, first bracket 1.0 (gap
    # 0.10) above 10, the next 2.4 heights higher.
    vals = [np.array([0.0, 4.0]), np.array([10.0]), np.array([6.0])]
    res = {'posthoc': [_pair('GroupA', 'GroupB', 0.001),
                       _pair('GroupA', 'GroupC', 0.02)]}
    ax = _ax()
    ax.set_ylim(0, 10)
    draw_significance(ax, vals, GROUPS3, res, INK)
    assert _lines(ax) == [
        ([1, 1, 2, 2], [11.0, 11.5, 11.5, 11.0]),
        ([1, 1, 3, 3], [12.2, 12.7, 12.7, 12.2]),
    ]
    texts = [(t.get_position(), t.get_text()) for t in ax.texts]
    assert texts == [((1.5, 11.5), '***'), ((2.0, 12.7), '*')]
    assert all(ln.get_color() == INK for ln in ax.lines)
    assert all(t.get_color() == INK for t in ax.texts)
    assert ax.get_ylim() == (0, 10)        # fit_top off: limit untouched


def test_gap_and_fit_top():
    vals = [np.array([0.0]), np.array([10.0])]
    ax = _ax()
    ax.set_ylim(0, 10)
    draw_significance(ax, vals, ['GroupA', 'GroupB'], {'p': 0.01}, INK,
                      gap=0.08, fit_top=True)
    xs, ys = _lines(ax)[0]
    assert xs == [1, 1, 2, 2]
    assert ys == pytest.approx([10.8, 11.3, 11.3, 10.8])
    # Top = first bracket + one stacked step per bracket + 12% of the span.
    assert ax.get_ylim()[1] == pytest.approx(10.8 + 1.2 + 1.2)


def test_fit_top_leaves_the_limit_when_nothing_is_drawn():
    ax = _ax()
    ax.set_ylim(0, 10)
    draw_significance(ax, [np.array([0.0]), np.array([10.0])],
                      ['GroupA', 'GroupB'], {'p': None}, INK, fit_top=True)
    assert len(ax.lines) == 0 and ax.get_ylim() == (0, 10)


def test_non_finite_values_are_ignored():
    vals = [np.array([0.0, np.nan]), np.array([10.0, np.inf, -np.inf])]
    ax = _ax()
    draw_significance(ax, vals, ['GroupA', 'GroupB'], {'p': 0.01}, INK)
    assert _lines(ax)[0][1] == pytest.approx([11.0, 11.5, 11.5, 11.0])


def test_equal_values_space_by_their_size():
    ax = _ax()
    draw_significance(ax, [np.array([20.0]), np.array([20.0])],
                      ['GroupA', 'GroupB'], {'p': 0.01}, INK)
    assert _lines(ax)[0][1] == pytest.approx([22.0, 23.0, 23.0, 22.0])
    ax = _ax()
    draw_significance(ax, [np.array([0.0]), np.array([0.0])],
                      ['GroupA', 'GroupB'], {'p': 0.01}, INK)
    assert _lines(ax)[0][1] == pytest.approx([0.1, 0.15, 0.15, 0.1])


@pytest.mark.parametrize('vals', [
    [], [np.array([]), np.array([])], [np.array([np.nan]), np.array([])]])
def test_no_values_draw_nothing(vals):
    ax = _ax()
    draw_significance(ax, vals, ['GroupA', 'GroupB'], {'p': 0.0001}, INK,
                      fit_top=True)
    assert len(ax.lines) == 0 and len(ax.texts) == 0


def test_plain_lists_are_accepted():
    ax = _ax()
    draw_significance(ax, [[0.0, 1.0], [10.0]], ['GroupA', 'GroupB'],
                      {'p': 0.01}, INK)
    assert len(ax.lines) == 1


# ── one copy ────────────────────────────────────────────────────────────

def test_no_window_carries_its_own_bracket_drawer():
    """Only plotmath turns p-values into bracket stars on a plot: a module
    that both calls p_to_stars and draws ax.plot / ax.text brackets in the
    same function is a private copy."""
    copies = []
    for path in sorted(SRC.glob('*.py')):
        if path.name in ('plotmath.py', 'stats.py'):
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = {n.func.attr if isinstance(n.func, ast.Attribute)
                     else getattr(n.func, 'id', None)
                     for n in ast.walk(fn) if isinstance(n, ast.Call)}
            if 'p_to_stars' in calls and calls & {'plot', 'text'}:
                copies.append(f"{path.name}:{fn.name}")
    assert copies == []
