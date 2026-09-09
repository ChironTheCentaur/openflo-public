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


def test_gate_to_mask_interval_is_strictly_between():
    """An interval gate selects events STRICTLY between lo and hi (both bounds
    exclusive) — pins the actual mask, not just parse/round-trip. A complement
    (`<lo | >hi`) or drop-upper-bound bug would pass every other test."""
    import pandas as pd

    from openflo.pipeline import gate_to_mask
    df = pd.DataFrame({'X': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    m = gate_to_mask({'kind': 'interval', 'channel': 'X', 'lo': 2.0, 'hi': 5.0}, df)
    assert list(m) == [False, False, True, True, False, False]


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
