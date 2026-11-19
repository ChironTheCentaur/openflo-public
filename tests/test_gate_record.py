"""A gate OpenFlo cannot evaluate selects nothing and reads as unknown.

A gate of a kind this version does not know (a session from a newer
OpenFlo, a damaged file) or with no kind at all used to select EVERY
event: under a parent of 2,500 events it reported 2,500 events, 100 % of
parent and the parent's median, under its own name. A gate missing a field
its kind needs crashed statistics for the whole sample. Both now fail
closed: no events, and the population (and every one under it) is unknown
(NA) with the reason, on the desktop and through the document the phone
uses.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from openflo.gate_eval import (
    _MASK_BY_KIND,
    cumulative_gate_mask,
    gate_to_mask,
    unknown_population_reason,
)
from openflo.gate_record import KNOWN_KINDS, gate_record_problem
from openflo.gating import population_stats

PARENT = {'kind': 'threshold', 'channel': 'FL1-A', 'value': 0.5,
          'parent_id': None, 'name': 'Parent'}
UNDER = {'kind': 'threshold', 'channel': 'FL2-A', 'value': 0.2,
         'parent_id': 'odd', 'name': 'Under'}


def _df(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'FL1-A': rng.random(n), 'FL2-A': rng.random(n)})


def _tree(odd):
    return {'p': dict(PARENT), 'odd': dict(odd, parent_id='p', name='Odd'),
            'under': dict(UNDER)}


ODD_GATES = {
    'unknown kind': {'kind': 'spline', 'x_channel': 'FL1-A',
                     'y_channel': 'FL2-A'},
    'no kind': {'x_channel': 'FL1-A', 'y_channel': 'FL2-A'},
    'kind not text': {'kind': ['rect']},
    'rect without x0': {'kind': 'rect', 'x_channel': 'FL1-A',
                        'y_channel': 'FL2-A', 'x1': 1.0, 'y0': 0.0,
                        'y1': 1.0},
    'threshold value is text': {'kind': 'threshold', 'channel': 'FL2-A',
                                'value': 'x'},
    'interval with no channel': {'kind': 'interval', 'lo': 0.0, 'hi': 1.0},
    'polygon of two points': {'kind': 'polygon', 'x_channel': 'FL1-A',
                              'y_channel': 'FL2-A',
                              'vertices': [[0, 0], [1, 1]]},
    'polygon point damaged': {'kind': 'polygon', 'x_channel': 'FL1-A',
                              'y_channel': 'FL2-A',
                              'vertices': [[0, 0], [1, 0], 3]},
    'ellipse without cov': {'kind': 'ellipsoid', 'x_channel': 'FL1-A',
                            'y_channel': 'FL2-A', 'mean': [0.5, 0.5]},
    'ellipse distance is text': {'kind': 'ellipsoid', 'x_channel': 'FL1-A',
                                 'y_channel': 'FL2-A', 'mean': [0.5, 0.5],
                                 'cov': [[0.1, 0], [0, 0.1]],
                                 'distance_sq': 'x'},
    'category value is a list': {'kind': 'category', 'channel': 'FL1-A',
                                 'value': [1]},
    'cluster column is a number': {'kind': 'cluster', 'channel': 3,
                                   'cluster_id': 1},
    'boolean operands not a list': {'kind': 'boolean', 'operands': 'p'},
    'auto-clean methods damaged': {'kind': 'autoclean', 'methods': [3]},
    'auto-clean params damaged': {'kind': 'autoclean',
                                  'methods': [{'key': 'margin',
                                               'params': 3}]},
    'threshold channel empty': {'kind': 'threshold', 'channel': '',
                                'value': 0.5},
    'rect bound is true/false': {'kind': 'rect', 'x_channel': 'FL1-A',
                                 'y_channel': 'FL2-A', 'x0': True,
                                 'x1': 1.0, 'y0': 0.0, 'y1': 1.0},
    'threshold value is NaN': {'kind': 'threshold', 'channel': 'FL2-A',
                               'value': float('nan')},
    'polygon vertex is NaN': {'kind': 'polygon', 'x_channel': 'FL1-A',
                              'y_channel': 'FL2-A',
                              'vertices': [[0, 0], [1, float('nan')],
                                           [1, 1]]},
    'polygon of 3-number points': {'kind': 'polygon', 'x_channel': 'FL1-A',
                                   'y_channel': 'FL2-A',
                                   'vertices': [[0, 0, 0], [1, 0, 0],
                                                [1, 1, 0]]},
    'polygon vertices flat': {'kind': 'polygon', 'x_channel': 'FL1-A',
                              'y_channel': 'FL2-A', 'vertices': [0, 1, 2]},
    'polygon vertices are text': {'kind': 'polygon', 'x_channel': 'FL1-A',
                                  'y_channel': 'FL2-A', 'vertices': 'abc'},
    'ellipse centre of three numbers': {'kind': 'ellipsoid',
                                        'x_channel': 'FL1-A',
                                        'y_channel': 'FL2-A',
                                        'mean': [0.5, 0.5, 0.5],
                                        'cov': [[0.1, 0], [0, 0.1]]},
    'cluster without id': {'kind': 'cluster', 'channel': 'cluster'},
    'boolean operand true/false': {'kind': 'boolean', 'operands': [True]},
}


def test_every_kind_gate_eval_evaluates_is_known():
    assert set(_MASK_BY_KIND) == KNOWN_KINDS


@pytest.mark.parametrize('label', sorted(ODD_GATES))
def test_an_odd_gate_selects_nothing(label):
    """Before: an unknown kind, a missing kind, a polygon of two points and
    an ellipse without its shape admitted every event; the others raised."""
    g, df = _tree(ODD_GATES[label]), _df()
    assert gate_record_problem(g['odd'])
    assert not gate_to_mask(g['odd'], df, g).any()
    assert not cumulative_gate_mask(g, 'under', df).any()


@pytest.mark.parametrize('label', sorted(ODD_GATES))
def test_an_odd_gate_and_those_under_it_are_unknown(label):
    g, df = _tree(ODD_GATES[label]), _df()
    why = gate_record_problem(g['odd'])
    assert unknown_population_reason(g, 'odd') == why
    assert unknown_population_reason(g, 'under') == why
    assert unknown_population_reason(g, 'p') is None
    assert unknown_population_reason(g, 'odd', columns=df.columns) == why
    rows = population_stats('s', df, g, list(g), {}, ['FL1-A'],
                            {'Count', '%Parent', '%Total', 'Median'},
                            ('Median',))
    by = {r['__gid__']: r for r in rows}
    for gid in ('odd', 'under'):
        assert by[gid]['__unknown__'] == why, gid
        assert all(math.isnan(by[gid][k]) for k in
                   ('Count', '%Parent', '%Total', 'Median FL1-A')), gid
    # The parent keeps its measured numbers.
    assert '__unknown__' not in by['p'] and by['p']['Count'] > 1000


def test_the_reason_says_what_is_wrong():
    assert 'not known to this version' in gate_record_problem(
        ODD_GATES['unknown kind'])
    assert 'spline' in gate_record_problem(ODD_GATES['unknown kind'])
    assert gate_record_problem(ODD_GATES['no kind']) == 'gate has no kind'
    assert "'x0'" in gate_record_problem(ODD_GATES['rect without x0'])
    assert "'channel'" in gate_record_problem(
        ODD_GATES['interval with no channel'])
    assert gate_record_problem('rect') == 'gate record is damaged'


WHOLE_GATES = [
    {'kind': 'threshold', 'channel': 'FL1-A', 'value': 0.5},
    {'kind': 'threshold', 'channel': 'FL1-A', 'value': np.float64(0.5)},
    {'kind': 'interval', 'channel': 'FL1-A', 'lo': -1e12, 'hi': 1},
    {'kind': 'rect', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
     'x0': 0, 'x1': 1, 'y0': 0.0, 'y1': 1.0},
    {'kind': 'polygon', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
     'vertices': [(0, 0), (1, 0), (1, 1)]},
    {'kind': 'polygon', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
     'vertices': np.array([[0, 0], [1, 0], [1, 1]])},
    {'kind': 'ellipsoid', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
     'mean': [0.5, 0.5], 'cov': [[0.1, 0], [0, 0.1]]},
    {'kind': 'ellipsoid', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
     'mean': (0.5, 0.5), 'cov': np.eye(2), 'distance_sq': 1.0},
    {'kind': 'cluster', 'channel': 'cluster', 'cluster_id': np.int64(2)},
    {'kind': 'cluster', 'cluster_id': 0},
    {'kind': 'category', 'channel': 'cell_cycle', 'value': 'G1'},
    {'kind': 'flowjo_ellipse'},
    {'kind': 'boolean', 'op': 'not', 'operands': ['p']},
    {'kind': 'boolean', 'operands': []},
    {'kind': 'autoclean', 'methods': [{'key': 'margin', 'params': None}]},
    {'kind': 'autoclean'},
    {'kind': 'group'},
]


@pytest.mark.parametrize('g', WHOLE_GATES, ids=lambda g: g['kind'])
def test_a_whole_gate_has_no_problem(g):
    """Every shape the app writes (lists, tuples, numpy arrays and
    numbers) passes, so no real session loses a population."""
    assert gate_record_problem(g) is None


@pytest.mark.parametrize('label', sorted(ODD_GATES))
def test_an_odd_gate_is_described_without_raising(label):
    """The run log and the gate list name a gate by its description; a
    known kind missing a field raised KeyError / ValueError there."""
    from openflo.gate_text import describe_gate
    g = dict(ODD_GATES[label], name='Odd')
    for tf in (None, {}):
        text = describe_gate(g, transforms=tf)
        assert isinstance(text, str) and text


def test_a_damaged_known_kind_says_it_cannot_be_read():
    from openflo.gate_text import describe_gate
    g = dict(ODD_GATES['rect without x0'], name='Odd')
    assert describe_gate(g) == 'Odd  ?  rect (cannot be read)'
    assert describe_gate(g, transforms={}) == 'Odd  ?  rect (cannot be read)'


def test_the_fail_closed_gate_is_logged_as_a_warning(caplog):
    """An empty population is loud: the log says which gate and why."""
    g = dict(ODD_GATES['unknown kind'], name='Odd')
    with caplog.at_level('WARNING', logger='openflo.pipeline'):
        gate_to_mask(g, _df())
    (rec,) = [r for r in caplog.records if 'Odd' in r.getMessage()]
    assert rec.levelname == 'WARNING'
    assert 'not known to this version' in rec.getMessage()
