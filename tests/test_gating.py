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
