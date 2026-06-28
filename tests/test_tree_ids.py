"""Tests for openflo.tree_ids — the editor tree-row id codec (extracted)."""
from __future__ import annotations

from openflo.tree_ids import (
    gate_iid,
    method_iid,
    parse_iid,
    sample_iid,
    subgroup_iid,
    trial_iid,
)


def test_encode_decode_roundtrip():
    assert parse_iid(sample_iid('s1')) == ('sample', 's1')
    assert parse_iid(gate_iid('s1', 'g3')) == ('gate', 's1', 'g3')
    assert parse_iid(trial_iid('Day 1')) == ('trial', 'Day 1')
    assert parse_iid(method_iid('s1', 'g3', 'debris')) == \
        ('method', 's1', 'g3', 'debris')
    assert parse_iid(subgroup_iid('comp', 'Day 1')) == \
        ('subgroup', 'comp', 'Day 1')


def test_gate_id_with_slashes_in_name():
    # sample names can contain '/'; gid is split off the RIGHT, so the name
    # survives intact.
    iid = gate_iid('a/b/c', 'g9')
    assert parse_iid(iid) == ('gate', 'a/b/c', 'g9')


def test_parse_rejects_garbage():
    assert parse_iid('') is None
    assert parse_iid('Z:nope') is None
    assert parse_iid('G:no-slash') is None       # gate id needs name/gid
    assert parse_iid('M:s/g') is None            # method needs 3 parts
    assert parse_iid('SG:onlyone') is None       # subgroup needs kind:trial
