"""Every path through openflo-compare's per-sample comparison.

`_compare_one_sample` reads one sample, compensates it from the workspace,
links each population's gate to its parent, evaluates the gates and reports
one row per population. These tests drive each branch of it, and each of its
helpers directly, on a small synthetic sample: `FlowSample` is replaced by a
stand-in holding a seeded pandas table on FL1-A / FL2-A / FSC-A / SSC-A, so
no FCS or workspace file is read and no window is opened.

Every expected count is worked out here with numpy from the same arrays, not
by asking the code under test.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

import openflo.compare as cmp
from openflo.compare import (
    _alias_uncompensated,
    _compare_one_sample,
    _compensate_for_compare,
    _count_deltas,
    _count_population,
    _join_errors,
    _linked_gates,
    _unconverted_reason,
    _unread_reason,
)

N = 40
SPILL = 0.3          # the stand-in's compensation: FL2-A -= 0.3 * FL1-A
UNREAD = '_unread'   # compare's marker for a gate it could not read

_rng = np.random.default_rng(7)
FL1 = _rng.uniform(0, 1000, N).round(1)
FL2 = _rng.uniform(0, 1000, N).round(1)
FSC = _rng.uniform(0, 1000, N).round(1)
SSC = _rng.uniform(0, 1000, N).round(1)
FL2_COMP = FL2 - SPILL * FL1


def _raw():
    return pd.DataFrame({'FL1-A': FL1, 'FL2-A': FL2,
                         'FSC-A': FSC, 'SSC-A': SSC})


def _compensated():
    df = _raw()
    df['FL2-A'] = FL2_COMP
    return df


class _StandIn:
    """What compare uses of FlowSample: `data`, `raw`, `comp_channels`,
    `is_compensated()` and `compensate_from_wsp()`."""

    fail = None          # exception compensate_from_wsp raises, or None
    keep = None          # rows `data` keeps after compensation (a filter)
    made: list = []

    def __init__(self, path):
        self.path = path
        self.raw = _raw()
        self.data = self.raw.copy()
        self.comp_matrix = None
        self.comp_channels = []
        self.wsp_calls = []
        type(self).made.append(self)

    def compensate_from_wsp(self, wsp_path):
        self.wsp_calls.append(wsp_path)
        if self.fail is not None:
            raise self.fail
        self.data = _compensated()
        if self.keep is not None:
            self.data = self.data[self.keep].reset_index(drop=True)
        self.comp_matrix = np.array([[1.0, SPILL], [0.0, 1.0]])
        self.comp_channels = ['FL1-A', 'FL2-A']

    def is_compensated(self):
        return self.comp_matrix is not None and bool(self.comp_channels)


@pytest.fixture
def sample(monkeypatch):
    """The stand-in class, installed as compare's FlowSample; set `fail`
    or `keep` on it before comparing."""
    cls = type('Sample', (_StandIn,), {'made': []})
    monkeypatch.setattr(cmp, 'FlowSample', cls)
    return cls


def thr(ch, value, **extra):
    return {'kind': 'threshold', 'channel': ch, 'value': value, **extra}


def by_name(rows):
    out = {}
    for r in rows:
        assert r['population'] not in out
        out[r['population']] = r
    return out


# ── compensation ─────────────────────────────────────────────────────────────

def test_no_workspace_compares_raw_data_without_compensating(sample):
    pops = [(1, 'Pop1', None, thr('FL2-A', 400.0), None)]
    row, = _compare_one_sample('S1', 'S1.fcs', pops)
    assert sample.made[0].wsp_calls == []
    assert row['error'] == ''
    assert row['openflo_count'] == int(np.sum(FL2 >= 400.0))
    assert row['total_events'] == N


def test_compensation_success_gates_compensated_values(sample):
    assert np.sum(FL2 >= 400.0) != np.sum(FL2_COMP >= 400.0)
    pops = [(1, 'Pop1', None, thr('FL2-A', 400.0), None)]
    row, = _compare_one_sample('S1', 'S1.fcs', pops, wsp_path='w.wsp')
    assert sample.made[0].wsp_calls == ['w.wsp']
    assert row['error'] == ''
    assert row['openflo_count'] == int(np.sum(FL2_COMP >= 400.0))


def test_compensation_failure_flags_every_row_and_counts_raw(sample):
    sample.fail = RuntimeError('no spillover here')
    pops = [(1, 'Pop1', 5, thr('FL1-A', 500.0), None),
            (2, 'Pop2', 3, thr('FSC-A', 250.0), 1)]
    rows = by_name(_compare_one_sample('S1', 'S1.fcs', pops, wsp_path='w.wsp'))
    text = 'uncompensated (compensation failed: RuntimeError)'
    assert rows['Pop1']['error'] == text
    assert rows['Pop2']['error'] == text
    assert rows['Pop1']['openflo_count'] == int(np.sum(FL1 >= 500.0))
    assert rows['Pop2']['openflo_count'] == int(np.sum((FL1 >= 500.0)
                                                       & (FSC >= 250.0)))


def test_event_count_mismatch_is_on_every_row(sample):
    pops = [(1, 'Pop1', None, thr('FL1-A', 500.0), None),
            (2, 'Pop2', None, thr('SSC-A', 500.0), None)]
    rows = _compare_one_sample('S1', 'S1.fcs', pops, expected_events=N + 10)
    for r in rows:
        assert f'records {N + 10} events' in r['error']
        assert f'has {N}:' in r['error']
        assert r['openflo_count'] is not None
    agree = _compare_one_sample('S1', 'S1.fcs', pops, expected_events=N)
    assert [r['error'] for r in agree] == ['', '']


def test_compensation_failure_and_event_mismatch_join(sample):
    sample.fail = ValueError('bad matrix')
    pops = [(1, 'Pop1', None, thr('FL1-A', 500.0), None)]
    row, = _compare_one_sample('S1', 'S1.fcs', pops, wsp_path='w.wsp',
                               expected_events=N + 1)
    first, second = row['error'].split('; ')
    assert first == 'uncompensated (compensation failed: ValueError)'
    assert second.startswith(f'the workspace records {N + 1} events')


def test_total_events_is_the_gated_data_not_raw(sample):
    # `data` filtered after compensation: raw and data differ in length, so
    # no uncompensated alias is made (the rows would not line up).
    keep = FSC >= 300.0
    sample.keep = keep
    n = int(keep.sum())
    assert n != N
    pops = [(1, 'Pop1', None, thr('FL2-A', 400.0, wsp_comp_channels=[]), None)]
    row, = _compare_one_sample('S1', 'S1.fcs', pops, wsp_path='w.wsp')
    assert row['total_events'] == n
    assert row['openflo_count'] == int(np.sum(FL2_COMP[keep] >= 400.0))
    assert row['error'] == ''


# ── bare compensated channels read the uncompensated values ──────────────────

def test_bare_compensated_channel_counts_raw_values(sample):
    assert np.sum(FL2 >= 400.0) != np.sum(FL2_COMP >= 400.0)
    pops = [
        # named bare in the workspace: FlowJo's uncompensated parameter
        (1, 'Bare', None, thr('FL2-A', 400.0, wsp_comp_channels=[]), None),
        # named 'Comp-FL2-A' in the workspace: the compensated one
        (2, 'Comp', None, thr('FL2-A', 400.0, wsp_comp_channels=['FL2-A']),
         None),
        # not read from a workspace: the data's own column
        (3, 'Drawn', None, thr('FL2-A', 400.0), None),
    ]
    rows = by_name(_compare_one_sample('S1', 'S1.fcs', pops, wsp_path='w.wsp'))
    assert rows['Bare']['openflo_count'] == int(np.sum(FL2 >= 400.0))
    assert rows['Comp']['openflo_count'] == int(np.sum(FL2_COMP >= 400.0))
    assert rows['Drawn']['openflo_count'] == int(np.sum(FL2_COMP >= 400.0))
    assert all(r['error'] == '' for r in rows.values())


@pytest.mark.parametrize('axis', ['x', 'y'])
def test_bare_compensated_axis_of_a_rect_counts_raw_values(sample, axis):
    other = ('FSC-A', 'x') if axis == 'y' else ('FSC-A', 'y')
    g = {'kind': 'rect', 'wsp_comp_channels': [],
         f'{axis}_channel': 'FL2-A', f'{other[1]}_channel': other[0],
         f'{axis}0': 300.0, f'{axis}1': 900.0,
         f'{other[1]}0': 100.0, f'{other[1]}1': 800.0}
    pops = [(1, 'Pop1', None, g, None)]
    row, = _compare_one_sample('S1', 'S1.fcs', pops, wsp_path='w.wsp')
    fsc = (FSC >= 100.0) & (FSC < 800.0)
    want = int(np.sum((FL2 >= 300.0) & (FL2 < 900.0) & fsc))
    comp = int(np.sum((FL2_COMP >= 300.0) & (FL2_COMP < 900.0) & fsc))
    assert want != comp
    assert row['openflo_count'] == want


def test_caller_pops_are_not_changed(sample):
    pops = [(1, 'Pop1', None, thr('FL2-A', 400.0, wsp_comp_channels=[]), None),
            (2, 'Pop2', None, thr('FL1-A', 100.0), 1)]
    before = copy.deepcopy(pops)
    _compare_one_sample('S1', 'S1.fcs', pops, wsp_path='w.wsp')
    assert pops == before


# ── gates compare cannot read, or cannot convert ─────────────────────────────

def test_unread_gate_on_self_parent_and_boolean_operand(sample):
    sample.fail = RuntimeError('x')
    quad = {'kind': UNREAD, 'why': 'QuadrantGate'}
    pops = [
        (1, 'Quad', 10, dict(quad), None),
        (2, 'Pop1', 4, thr('FL1-A', 100.0), 1),         # child of unread
        (3, 'Pop2', 7, thr('FL2-A', 500.0), None),      # fine
        (4, 'Both', 2, {'kind': 'boolean', 'op': 'and',
                        'operands': [3, 1]}, None),      # operand unread
        (5, 'Pop3', 9, thr('SSC-A', 200.0), 4),         # under the boolean
        (6, 'NoWhy', 1, {'kind': UNREAD}, None),
    ]
    rows = by_name(_compare_one_sample('S1', 'S1.fcs', pops, wsp_path='w.wsp'))
    comp = 'uncompensated (compensation failed: RuntimeError)'
    dep = "depends on 'Quad', whose gate not read (QuadrantGate)"
    assert rows['Quad']['error'] == f'{comp}; gate not read (QuadrantGate)'
    assert rows['Pop1']['error'] == f'{comp}; {dep}'
    assert rows['Both']['error'] == f'{comp}; {dep}'
    assert rows['Pop3']['error'] == f'{comp}; {dep}'
    assert rows['NoWhy']['error'] == f'{comp}; gate not read (?)'
    for name in ('Quad', 'Pop1', 'Both', 'Pop3', 'NoWhy'):
        assert rows[name]['openflo_count'] is None
        assert rows[name]['delta'] is None
        assert rows[name]['rel_delta'] is None
    assert rows['Pop2']['error'] == comp
    assert rows['Pop2']['openflo_count'] == int(np.sum(FL2 >= 500.0))


def test_boolean_with_readable_operands_is_counted(sample):
    pops = [(1, 'Pop1', None, thr('FL1-A', 300.0), None),
            (2, 'Pop2', None, thr('FL2-A', 300.0), None),
            (3, 'Both', None, {'kind': 'boolean', 'op': 'and',
                               'operands': [1, 2]}, None)]
    rows = by_name(_compare_one_sample('S1', 'S1.fcs', pops))
    assert rows['Both']['error'] == ''
    assert rows['Both']['openflo_count'] == int(np.sum((FL1 >= 300.0)
                                                       & (FL2 >= 300.0)))


def test_flowjo_ellipse_reason_with_and_without_why(sample):
    pops = [
        (1, 'Ell', 12, {'kind': 'flowjo_ellipse',
                        'flowjo': {'why': 'Time axis'}}, None),
        (2, 'Pop1', 6, thr('FL1-A', 0.0), 1),
        (3, 'Bare', 3, {'kind': 'flowjo_ellipse'}, None),
        (4, 'Pop2', 8, thr('FL1-A', 0.0), None),
    ]
    rows = by_name(_compare_one_sample('S1', 'S1.fcs', pops))
    with_why = ('FlowJo ellipse not converted (Time axis); '
                'OpenFlo counts it as empty')
    assert rows['Ell']['error'] == with_why
    assert rows['Pop1']['error'] == with_why
    assert rows['Bare']['error'] == ('FlowJo ellipse not converted (its axis); '
                                     'OpenFlo counts it as empty')
    assert rows['Pop2']['error'] == ''
    assert rows['Ell']['openflo_count'] == 0
    assert rows['Pop1']['openflo_count'] == 0
    assert rows['Ell']['delta'] == -12
    assert rows['Pop2']['openflo_count'] == N


def test_flowjo_ellipse_reason_follows_compensation_error(sample):
    sample.fail = OSError('gone')
    pops = [(1, 'Ell', None, {'kind': 'flowjo_ellipse',
                              'flowjo': {'why': 'log axis'}}, None)]
    row, = _compare_one_sample('S1', 'S1.fcs', pops, wsp_path='w.wsp')
    assert row['error'] == ('uncompensated (compensation failed: OSError); '
                            'FlowJo ellipse not converted (log axis); '
                            'OpenFlo counts it as empty')


def test_gate_raising_in_evaluation_is_that_rows_error(sample):
    try:
        float('abc')
    except ValueError as exc:
        msg = f'ValueError: {exc}'
    pops = [(1, 'Bad', 5, thr('FL1-A', 'abc'), None),
            (2, 'Pop1', 5, thr('FL1-A', 500.0), None)]
    rows = by_name(_compare_one_sample('S1', 'S1.fcs', pops))
    assert rows['Bad']['error'] == msg
    assert rows['Bad']['openflo_count'] is None
    assert rows['Bad']['delta'] is None and rows['Bad']['rel_delta'] is None
    assert rows['Pop1']['openflo_count'] == int(np.sum(FL1 >= 500.0))
    sample.fail = RuntimeError('x')
    rows = by_name(_compare_one_sample('S1', 'S1.fcs', pops, wsp_path='w.wsp'))
    assert rows['Bad']['error'] == (
        f'uncompensated (compensation failed: RuntimeError); {msg}')


# ── linkage, names and row fields ────────────────────────────────────────────

def test_parent_missing_from_pops_makes_a_root(sample):
    pops = [(2, 'Pop1', None, thr('FL1-A', 600.0), 99)]
    row, = _compare_one_sample('S1', 'S1.fcs', pops)
    assert row['openflo_count'] == int(np.sum(FL1 >= 600.0))


def test_child_is_counted_inside_its_parent(sample):
    pops = [(1, 'Pop1', None, thr('FL1-A', 500.0), None),
            (2, 'Pop2', None, thr('FL2-A', 500.0), 1)]
    rows = by_name(_compare_one_sample('S1', 'S1.fcs', pops))
    inside = int(np.sum((FL1 >= 500.0) & (FL2 >= 500.0)))
    assert inside != int(np.sum(FL2 >= 500.0))
    assert rows['Pop2']['openflo_count'] == inside


def test_duplicate_population_names_are_counted_apart(sample):
    pops = [
        (1, 'A', None, thr('FL1-A', 500.0, _wsp_path=('A',)), None),
        (2, 'B', None, thr('FL2-A', 500.0, _wsp_path=('B',)), None),
        (3, 'Pop1', 11, thr('FSC-A', 500.0, _wsp_path=('A', 'Pop1')), 1),
        (4, 'Pop1', 22, thr('FSC-A', 500.0, _wsp_path=('B', 'Pop1')), 2),
    ]
    rows = _compare_one_sample('S1', 'S1.fcs', pops)
    under_a, under_b = rows[2], rows[3]
    assert under_a['population'] == under_b['population'] == 'Pop1'
    assert under_a['path'] == 'A/Pop1' and under_b['path'] == 'B/Pop1'
    assert under_a['flowjo_count'] == 11 and under_b['flowjo_count'] == 22
    want_a = int(np.sum((FL1 >= 500.0) & (FSC >= 500.0)))
    want_b = int(np.sum((FL2 >= 500.0) & (FSC >= 500.0)))
    assert want_a != want_b
    assert under_a['openflo_count'] == want_a
    assert under_b['openflo_count'] == want_b


def test_row_fields(sample):
    pops = [(1, 'Pop1', 7, thr('FL1-A', 500.0, _wsp_path=('Lymph', 'Pop1')),
             None),
            (2, 'Pop2', 7, thr('FL1-A', 500.0), None)]
    rows = _compare_one_sample('S1', 'S1.fcs', pops)
    assert list(rows[0]) == ['sample', 'population', 'path', 'flowjo_count',
                             'openflo_count', 'delta', 'rel_delta',
                             'total_events', 'error']
    assert rows[0]['sample'] == 'S1'
    assert rows[0]['population'] == 'Pop1'
    assert rows[0]['path'] == 'Lymph/Pop1'
    assert rows[1]['path'] == 'Pop2'          # no workspace path: the name
    assert [r['total_events'] for r in rows] == [N, N]


@pytest.mark.parametrize('flowjo', [None, 0, 13])
def test_deltas_per_flowjo_count(sample, flowjo):
    pops = [(1, 'Pop1', flowjo, thr('FL1-A', 500.0), None)]
    row, = _compare_one_sample('S1', 'S1.fcs', pops)
    got = int(np.sum(FL1 >= 500.0))
    assert got != 13
    assert row['openflo_count'] == got
    assert row['flowjo_count'] == flowjo
    if flowjo is None:
        assert row['delta'] is None and row['rel_delta'] is None
    elif flowjo == 0:
        assert row['delta'] == got
        assert row['rel_delta'] is None
    else:
        assert row['delta'] == got - 13
        assert row['rel_delta'] == pytest.approx((got - 13) / 13)


def test_one_mask_cache_for_the_whole_sample(sample, monkeypatch):
    seen = []
    real = cmp.cumulative_gate_mask

    def spy(*args, **kwargs):
        seen.append(kwargs.get('cache'))
        return real(*args, **kwargs)
    monkeypatch.setattr(cmp, 'cumulative_gate_mask', spy)
    pops = [(1, 'Pop1', None, thr('FL1-A', 500.0), None),
            (2, 'Pop2', None, thr('FL2-A', 500.0), 1),
            (3, 'Pop3', None, thr('SSC-A', 500.0), 2)]
    _compare_one_sample('S1', 'S1.fcs', pops)
    assert len(seen) == 3
    assert isinstance(seen[0], dict)
    assert all(c is seen[0] for c in seen)
    assert set(seen[0]) == {1, 2, 3}


# ── helpers, directly ────────────────────────────────────────────────────────

def test_join_errors():
    assert _join_errors() == ''
    assert _join_errors('', None, '') == ''
    assert _join_errors('', 'b', '') == 'b'
    assert _join_errors('a', '', 'b') == 'a; b'


def test_compensate_for_compare(sample):
    s = sample('S1.fcs')
    assert _compensate_for_compare(s, None) == ''
    assert _compensate_for_compare(s, '') == ''
    assert s.wsp_calls == []
    assert _compensate_for_compare(s, 'w.wsp') == ''
    assert s.wsp_calls == ['w.wsp']
    sample.fail = KeyError('Comp-FL2-A')
    s = sample('S1.fcs')
    assert _compensate_for_compare(s, 'w.wsp') == (
        'uncompensated (compensation failed: KeyError)')


def test_linked_gates():
    g1, g2 = thr('FL1-A', 1.0), thr('FL2-A', 2.0)
    pops = [(1, 'Pop1', None, g1, None), (2, 'Pop2', None, g2, 1),
            (3, 'Pop3', None, thr('SSC-A', 3.0), 42)]
    linked = _linked_gates(pops)
    assert {k: v['parent_id'] for k, v in linked.items()} == {
        1: None, 2: 1, 3: None}
    assert linked[2]['channel'] == 'FL2-A'
    assert 'parent_id' not in g1 and 'parent_id' not in g2
    assert linked[1] is not g1


def test_alias_uncompensated(sample):
    s = sample('S1.fcs')
    gates = {1: thr('FL2-A', 400.0, wsp_comp_channels=[]),
             2: thr('FL1-A', 400.0, wsp_comp_channels=['FL1-A']),
             3: {'kind': UNREAD, 'why': 'QuadrantGate'}}
    assert _alias_uncompensated(s, gates) is s.data   # not compensated
    assert gates[1]['channel'] == 'FL2-A'
    s.compensate_from_wsp('w.wsp')
    data = _alias_uncompensated(s, gates)
    assert gates[1]['channel'] == 'FL2-A (uncompensated)'
    assert gates[2]['channel'] == 'FL1-A'
    assert list(data.columns) == ['FL1-A', 'FL2-A', 'FSC-A', 'SSC-A',
                                  'FL2-A (uncompensated)']
    assert np.array_equal(data['FL2-A (uncompensated)'].to_numpy(), FL2)
    assert np.array_equal(data['FL2-A'].to_numpy(), FL2_COMP)


def test_unconverted_reason():
    gates = {1: {'kind': 'flowjo_ellipse', 'flowjo': {'why': 'Time axis'},
                 'parent_id': None},
             2: dict(thr('FL1-A', 0.0), parent_id=1),
             3: dict(thr('FL1-A', 0.0), parent_id=None),
             4: {'kind': 'flowjo_ellipse', 'flowjo': {}, 'parent_id': None}}
    text = 'FlowJo ellipse not converted ({}); OpenFlo counts it as empty'
    assert _unconverted_reason(gates, 1) == text.format('Time axis')
    assert _unconverted_reason(gates, 2) == text.format('Time axis')
    assert _unconverted_reason(gates, 3) == ''
    assert _unconverted_reason(gates, 4) == text.format('its axis')
    assert _unconverted_reason(gates, 99) == ''


def test_unread_reason():
    gates = {1: {'kind': UNREAD, 'why': 'QuadrantGate', 'parent_id': None},
             2: dict(thr('FL1-A', 0.0), parent_id=1),
             3: dict(thr('FL2-A', 0.0), parent_id=None),
             4: {'kind': 'boolean', 'operands': [3, 1], 'parent_id': None},
             5: dict(thr('SSC-A', 0.0), parent_id=4),
             6: {'kind': UNREAD, 'parent_id': None}}
    names = {1: 'Quad', 2: 'Pop1', 3: 'Pop2', 4: 'Both', 5: 'Pop3', 6: 'Pop4'}
    dep = "depends on 'Quad', whose gate not read (QuadrantGate)"
    assert _unread_reason(gates, names, 1) == 'gate not read (QuadrantGate)'
    assert _unread_reason(gates, names, 2) == dep
    assert _unread_reason(gates, names, 3) == ''
    assert _unread_reason(gates, names, 4) == dep
    assert _unread_reason(gates, names, 5) == dep
    assert _unread_reason(gates, names, 6) == 'gate not read (?)'
    assert _unread_reason(gates, {}, 2) == (
        "depends on '?', whose gate not read (QuadrantGate)")


def test_count_population():
    df = _raw()
    gates = {1: dict(thr('FL1-A', 500.0), parent_id=None),
             2: dict(thr('FL1-A', 'abc'), parent_id=None),
             3: {'kind': 'flowjo_ellipse', 'flowjo': {'why': 'log axis'},
                 'parent_id': None}}
    cache = {}
    assert _count_population(gates, 1, df, cache, '') == (
        int(np.sum(FL1 >= 500.0)), '')
    assert 1 in cache
    assert _count_population(gates, 1, df, {}, 'comp') == (
        int(np.sum(FL1 >= 500.0)), 'comp')
    count, err = _count_population(gates, 2, df, {}, 'comp')
    assert count is None
    assert err.startswith('comp; ValueError: ')
    assert _count_population(gates, 3, df, {}, '') == (
        0, 'FlowJo ellipse not converted (log axis); '
           'OpenFlo counts it as empty')


def test_count_deltas():
    assert _count_deltas(None, 5) == (None, None)
    assert _count_deltas(5, None) == (None, None)
    assert _count_deltas(7, 0) == (7, None)
    assert _count_deltas(0, 0) == (0, None)
    assert _count_deltas(6, 8) == (-2, pytest.approx(-0.25))
    assert _count_deltas(12, 8) == (4, pytest.approx(0.5))
    assert _count_deltas(0, 4) == (-4, pytest.approx(-1.0))
