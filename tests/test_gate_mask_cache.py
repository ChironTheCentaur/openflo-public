"""Regression tests for ``cumulative_gate_mask``'s optional memoisation.

The cache exists because a polygon gate costs ~250x a threshold gate
(``Path.contains_points`` over every event) and the ancestor chain used to be
re-evaluated once per descendant. Correctness is the whole point: a cached walk
MUST return exactly what an uncached walk returns, including for cycles,
missing parents, and injected ``overrides``.
"""
import numpy as np
import pandas as pd
import pytest

from openflo.gating import population_stats
from openflo.pipeline import cumulative_gate_mask

CH = ['FSC-A', 'SSC-A', 'CD3', 'CD4']


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({c: rng.random(2000) for c in CH})


def _poly(gid, parent, xc='FSC-A', yc='SSC-A'):
    return {'id': gid, 'kind': 'polygon', 'parent_id': parent, 'name': gid,
            'x_channel': xc, 'y_channel': yc, 'enabled': True,
            'vertices': [[0.1, 0.1], [0.9, 0.15], [0.85, 0.9], [0.15, 0.8]]}


def _tree(depth=4, fanout=2):
    """A spine with leaves hanging off each node."""
    gates, parent = {}, None
    for i in range(depth):
        sid = f's{i}'
        gates[sid] = _poly(sid, parent, CH[i % len(CH)], CH[(i + 1) % len(CH)])
        for j in range(fanout):
            lid = f'l{i}_{j}'
            gates[lid] = _poly(lid, sid, CH[j % len(CH)], CH[(j + 2) % len(CH)])
        parent = sid
    return gates


def test_cached_matches_uncached_for_every_gate(df):
    gates = _tree()
    cache = {}
    for gid in gates:
        assert np.array_equal(
            cumulative_gate_mask(gates, gid, df),
            cumulative_gate_mask(gates, gid, df, cache=cache)), gid


def test_cache_is_populated_and_reused(df):
    """A full sweep caches every gate, and a second sweep hits the cache."""
    gates = _tree(depth=3, fanout=1)
    cache = {}
    for gid in gates:
        cumulative_gate_mask(gates, gid, df, cache=cache)
    assert set(cache) == set(gates)
    # Second call returns the SAME object (a cache hit, not a recompute).
    for gid in gates:
        assert cumulative_gate_mask(gates, gid, df, cache=cache) is cache[gid]


def test_no_cache_returns_a_fresh_array_each_call(df):
    """Default behaviour is unchanged: callers may treat the result as owned."""
    gates = _tree(depth=2, fanout=0)
    a = cumulative_gate_mask(gates, 's1', df)
    b = cumulative_gate_mask(gates, 's1', df)
    assert a is not b
    assert np.array_equal(a, b)


def test_cycle_is_still_safe(df):
    gates = {'a': _poly('a', 'b'), 'b': _poly('b', 'a')}
    assert np.array_equal(cumulative_gate_mask(gates, 'a', df),
                          cumulative_gate_mask(gates, 'a', df, cache={}))


def test_missing_parent_is_still_safe(df):
    gates = {'x': _poly('x', 'ghost')}
    assert np.array_equal(cumulative_gate_mask(gates, 'x', df),
                          cumulative_gate_mask(gates, 'x', df, cache={}))


def test_unknown_gate_id_yields_all_true(df):
    assert cumulative_gate_mask({}, 'nope', df).all()
    assert cumulative_gate_mask({}, 'nope', df, cache={}).all()


def test_overrides_are_honoured_through_the_cache(df):
    """An injected mask must still replace that gate's own evaluation."""
    gates = _tree(depth=3, fanout=1)
    ov = {'s0': np.zeros(len(df), dtype=bool)}     # root excludes everything
    for gid in gates:
        cached = cumulative_gate_mask(gates, gid, df, overrides=ov, cache={})
        assert np.array_equal(
            cumulative_gate_mask(gates, gid, df, overrides=ov), cached)
        assert not cached.any()                     # root override wins


def test_population_stats_counts_match_an_uncached_walk(df):
    """population_stats now shares one cache — its counts must not move."""
    gates = _tree()
    order = list(gates)
    rows = population_stats('s', df, gates, order, {}, CH,
                            {'Count', '%Parent', '%Total'}, {})
    by_gid = {r['__gid__']: r for r in rows}
    for gid in order:
        expected = int(cumulative_gate_mask(gates, gid, df).sum())
        assert by_gid[gid]['Count'] == expected, gid
