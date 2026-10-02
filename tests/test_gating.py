"""Tests for openflo.gating — gate count / %-of-parent, extracted from gui.py.

Headless: builds a DataFrame + gate dicts directly, no editor window, so the
correctness-critical readout is verified without Tk.
"""
from __future__ import annotations

import pytest

from openflo.gating import (
    format_gate_count,
    gate_channels,
    population_path,
    population_stats,
)

STAT_CHAN = ('Median', 'Mean', 'CV')


def test_format_gate_count():
    assert format_gate_count('CD3+', 120, 200, 'all') == \
        'CD3+:  n = 120   (60.00% of all)'
    assert format_gate_count('CD4+', 20, 120, 'parent') == \
        'CD4+:  n = 20   (16.67% of parent)'
    # zero parent never divides by zero
    assert '0.00% of all' in format_gate_count('x', 0, 0, 'all')


def test_gate_channels():
    assert gate_channels({'channel': 'CD3'}) == {'CD3'}
    assert gate_channels({'x_channel': 'FSC-A', 'y_channel': 'SSC-A'}) == \
        {'FSC-A', 'SSC-A'}
    assert gate_channels({}) == set()
    assert gate_channels({'channel': None}) == set()       # falsy ignored


def test_population_path():
    gates = {
        'g1': {'name': 'Cells', 'parent_id': None},
        'g2': {'name': 'Singlets', 'parent_id': 'g1'},
        'g3': {'label': 'CD11b+', 'parent_id': 'g2'},
    }
    assert population_path(gates, 'g3') == 'Cells/Singlets/CD11b+'
    assert population_path(gates, 'g1') == 'Cells'
    assert population_path(gates, 'nope') == 'nope'         # unknown gid
    # cycle-safe: a parent loop terminates rather than hanging
    cyc = {'a': {'name': 'A', 'parent_id': 'b'},
           'b': {'name': 'B', 'parent_id': 'a'}}
    assert population_path(cyc, 'a')                        # returns, no hang


def test_population_stats():
    pytest.importorskip('flowio')
    import numpy as np
    import pandas as pd
    rng = np.random.RandomState(0)
    # 200 events; g1 = CD3>=0.4 (120), g2 = CD4>=0.5 under g1 (CD4 descends → 20)
    df = pd.DataFrame({'CD3': np.linspace(0, 1, 200),
                       'CD4': np.linspace(1, 0, 200),
                       'M': rng.normal(100, 10, 200)})
    gates = {
        'g1': {'kind': 'threshold', 'channel': 'CD3', 'value': 0.4, 'op': '>=',
               'parent_id': None, 'id': 'g1', 'name': 'CD3+'},
        'g2': {'kind': 'threshold', 'channel': 'CD4', 'value': 0.5, 'op': '>=',
               'parent_id': 'g1', 'id': 'g2', 'name': 'CD4+'},
    }
    want = {'Count', '%Parent', '%Total', 'Median'}
    rows = population_stats('s1', df, gates, ['g1', 'g2'], {'M': 'M'}, ['M'],
                            want, STAT_CHAN)
    by_pop = {r['Population']: r for r in rows}
    assert by_pop['CD3+']['Count'] == 120
    assert abs(by_pop['CD3+']['%Total'] - 60.0) < 1e-9
    assert by_pop['CD3+/CD4+']['Count'] == 20
    assert abs(by_pop['CD3+/CD4+']['%Parent'] - (20 / 120 * 100)) < 1e-9
    assert 'Median M' in by_pop['CD3+']        # per-channel stat emitted
    assert rows[0]['__gid__'] == 'g1'          # hidden gid carried

    # `select` restricts emitted rows but %Parent still correct
    only = population_stats('s1', df, gates, ['g1', 'g2'], {'M': 'M'}, ['M'],
                            want, STAT_CHAN, select=['g2'])
    assert len(only) == 1 and only[0]['Population'] == 'CD3+/CD4+'
    assert abs(only[0]['%Parent'] - (20 / 120 * 100)) < 1e-9


def test_gate_counts_root_and_child():
    pytest.importorskip('flowio')              # gate masks live in pipeline
    import numpy as np
    import pandas as pd

    from openflo.gating import gate_counts
    # CD3 ascends, CD4 descends → independent. 200 events.
    df = pd.DataFrame({'CD3': np.linspace(0, 1, 200),
                       'CD4': np.linspace(1, 0, 200)})
    gates = {
        'g1': {'kind': 'threshold', 'channel': 'CD3', 'value': 0.4, 'op': '>=',
               'parent_id': None, 'id': 'g1'},
        'g2': {'kind': 'threshold', 'channel': 'CD4', 'value': 0.5, 'op': '>=',
               'parent_id': 'g1', 'id': 'g2'},
    }
    n, parent, of = gate_counts(gates, 'g1', df)
    assert (n, parent, of) == (120, 200, 'all')        # 60% of all events
    n, parent, of = gate_counts(gates, 'g2', df)
    assert (n, parent, of) == (20, 120, 'parent')      # 16.67% of parent


def test_gate_to_mask_interval_is_half_open():
    """An interval gate is HALF-OPEN [lo, hi): the lower bound is inside, the
    upper bound is not.

    This test previously pinned "strictly between", i.e. both bounds excluded.
    That was changed deliberately, not re-baselined to hide a regression: with
    both bounds open, two adjacent intervals [0,10] and [10,20] put a value of
    exactly 10 in NEITHER, so those events vanished from both populations and
    the percentages did not sum. On integer $DATATYPE I data that was 14.3% of
    events. Half-open also matches Gating-ML's RectangleGate convention and
    the polygon rule, so the same region gates identically however it is
    expressed. See test_interval_and_rect_partition below.

    The original purpose is preserved: this still pins the actual mask, so a
    complement (`<lo | >hi`) or a dropped upper bound is still caught.
    """
    import pandas as pd

    from openflo.pipeline import gate_to_mask
    df = pd.DataFrame({'X': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    m = gate_to_mask({'kind': 'interval', 'channel': 'X', 'lo': 2.0, 'hi': 5.0}, df)
    assert list(m) == [False, True, True, True, False, False]


def test_interval_and_rect_partition_adjacent_ranges():
    """Adjacent gates must claim each boundary event exactly once.

    The property the half-open change exists for. Fully-open bounds lost every
    event on a shared edge; fully-closed bounds would double-count them. Both
    are wrong, and both are silent.
    """
    import numpy as np
    import pandas as pd

    from openflo.pipeline import gate_to_mask
    rng = np.random.RandomState(0)
    # Integer-valued events land exactly on bounds, which is where this bites.
    df = pd.DataFrame({'X': rng.randint(0, 21, 5000).astype(float),
                       'Y': rng.randint(0, 21, 5000).astype(float)})

    lower = gate_to_mask({'kind': 'interval', 'channel': 'X',
                          'lo': 0.0, 'hi': 10.0}, df)
    upper = gate_to_mask({'kind': 'interval', 'channel': 'X',
                          'lo': 10.0, 'hi': 20.0}, df)
    on_edge = df['X'].values == 10.0
    assert on_edge.sum() > 0, 'fixture must contain events on the shared edge'
    assert int((lower & upper).sum()) == 0, 'boundary events double-counted'
    assert int((on_edge & ~lower & ~upper).sum()) == 0, (
        'boundary events fell into NEITHER interval — they vanish from both '
        'populations and the frequencies no longer sum')

    left = gate_to_mask({'kind': 'rect', 'x_channel': 'X', 'y_channel': 'Y',
                         'x0': 0.0, 'x1': 10.0, 'y0': 0.0, 'y1': 20.0}, df)
    right = gate_to_mask({'kind': 'rect', 'x_channel': 'X', 'y_channel': 'Y',
                          'x0': 10.0, 'x1': 20.0, 'y0': 0.0, 'y1': 20.0}, df)
    assert int((left & right).sum()) == 0
    assert int((on_edge & (df['Y'].values < 20) & ~left & ~right).sum()) == 0


def test_a_rect_equals_the_same_region_drawn_as_a_polygon():
    """The same square must gate identically whether drawn as a rect or a
    polygon. They used different boundary rules, so on integer data 823 of
    20 000 events landed in one and not the other."""
    import numpy as np
    import pandas as pd

    from openflo.pipeline import gate_to_mask
    rng = np.random.RandomState(0)
    df = pd.DataFrame({'X': rng.randint(0, 21, 5000).astype(float),
                       'Y': rng.randint(0, 21, 5000).astype(float)})
    rect = gate_to_mask({'kind': 'rect', 'x_channel': 'X', 'y_channel': 'Y',
                         'x0': 0.0, 'x1': 10.0, 'y0': 0.0, 'y1': 10.0}, df)
    poly = gate_to_mask({'kind': 'polygon', 'x_channel': 'X',
                         'y_channel': 'Y',
                         'vertices': [[0, 0], [10, 0], [10, 10], [0, 10]]}, df)
    assert int((rect != poly).sum()) == 0, (
        f'{int((rect != poly).sum())} event(s) gate differently as a rect vs '
        f'an identical polygon')


def test_cumulative_gate_mask_ands_ancestor_regardless_of_enabled():
    """The cumulative mask ANDs every ancestor up the chain regardless of its
    'enabled' flag (enabled = highlight visibility, not participation)."""
    import pandas as pd

    from openflo.pipeline import cumulative_gate_mask
    df = pd.DataFrame({'X': [1., 6., 6.], 'Y': [6., 1., 6.]})
    gates = {'P': {'kind': 'threshold', 'channel': 'X', 'value': 5.0,
                   'parent_id': None, 'enabled': False},
             'C': {'kind': 'threshold', 'channel': 'Y', 'value': 5.0,
                   'parent_id': 'P'}}
    # C (Y>5)=[T,F,T]; P (X>5, marked disabled)=[F,T,T]; cumulative = P AND C.
    assert list(cumulative_gate_mask(gates, 'C', df)) == [False, False, True]


def test_cumulative_gate_mask_uses_overrides():
    """overrides[gid] replaces evaluating that gate (injects a cached mask)."""
    import numpy as np
    import pandas as pd

    from openflo.pipeline import cumulative_gate_mask
    df = pd.DataFrame({'Y': [6., 6., 6.]})
    gates = {'P': {'kind': 'threshold', 'channel': 'MISSING', 'value': 0.0,
                   'parent_id': None},                     # would no-op to all-True
             'C': {'kind': 'threshold', 'channel': 'Y', 'value': 5.0,
                   'parent_id': 'P'}}
    ov = {'P': np.array([True, False, True])}
    # C (Y>5)=[T,T,T] AND the injected P mask [T,F,T] = [T,F,T].
    assert list(cumulative_gate_mask(gates, 'C', df, overrides=ov)) == \
        [True, False, True]


def test_quadrant_gates_partition_without_the_nextafter_hack():
    """Four quadrant rects must tile the plane exactly.

    The Gating-ML quadrant importer used to nudge each divider down by one ULP
    (`np.nextafter`) so that a strict `>` would still keep an event sitting
    exactly on it — otherwise such events fell into NO quadrant. Half-open
    bounds make that unnecessary, and the nudge was removed; this pins the
    property it was protecting, so it cannot regress silently.

    The fixture deliberately includes a large pile-up at exactly zero, which is
    what `arcsinh(0)` produces and what the removed comment called out.
    """
    import numpy as np
    import pandas as pd

    from openflo.pipeline import gate_to_mask
    rng = np.random.RandomState(0)
    x = np.concatenate([rng.randint(-5, 6, 8000).astype(float),
                        np.zeros(2000)])
    y = np.concatenate([rng.randint(-5, 6, 8000).astype(float),
                        np.zeros(2000)])
    df = pd.DataFrame({'X': x, 'Y': y})

    div, big = 0.0, 1e12
    quads = [(div, big, div, big), (div, big, -big, div),
             (-big, div, div, big), (-big, div, -big, div)]
    counts = np.vstack([
        gate_to_mask({'kind': 'rect', 'x_channel': 'X', 'y_channel': 'Y',
                      'x0': a, 'x1': b, 'y0': c, 'y1': d}, df)
        for a, b, c, d in quads]).sum(axis=0)

    assert int(((x == 0) | (y == 0)).sum()) > 0, 'fixture must sit on a divider'
    assert int((counts >= 2).sum()) == 0, 'events counted in two quadrants'
    assert int((counts == 0).sum()) == 0, (
        'events fell into NO quadrant — the four quadrants no longer tile the '
        'plane, so the percentages do not sum to 100')
