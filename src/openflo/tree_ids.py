"""Editor tree-row id encoding — pure, Tk-free.

The sample/gate tree identifies each row with a prefixed string id
(``S:`` sample, ``G:`` gate, ``T:`` trial, ``M:`` auto-clean method,
``SG:`` Comps/Samples sub-header). Encoding and decoding is pure string work,
lifted out of the editor so it's unit-testable without a window.

(The Pipeline Workspace tree has its own, separate id scheme in
:mod:`openflo.workspace`; these are the gate-editor's.)
"""
from __future__ import annotations


def sample_iid(name: str) -> str:
    return f'S:{name}'


def gate_iid(sample_name: str, gid: str) -> str:
    return f'G:{sample_name}/{gid}'


def trial_iid(trial: str) -> str:
    return f'T:{trial}'


def method_iid(sample_name: str, gid: str, key: str) -> str:
    """Synthetic row for one auto-clean method under its 'autoclean' gate."""
    return f'M:{sample_name}/{gid}/{key}'


def subgroup_iid(kind: str, trial: str) -> str:
    """Comps/Samples sub-header under a trial. ``kind`` ∈ {'comp', 'samp'}."""
    return f'SG:{kind}:{trial}'


def parse_iid(iid: str):
    """Decode a row id:

    ``('sample', name)`` | ``('gate', sample_name, gid)`` |
    ``('method', sample_name, gid, key)`` | ``('subgroup', kind, trial)`` |
    ``('trial', trial)`` | ``None``.
    """
    if iid.startswith('S:'):
        return ('sample', iid[2:])
    if iid.startswith('SG:'):
        parts = iid.split(':', 2)
        if len(parts) == 3:
            return ('subgroup', parts[1], parts[2])
        return None
    if iid.startswith('T:'):
        return ('trial', iid[2:])
    if iid.startswith('M:'):
        parts = iid[2:].rsplit('/', 2)
        if len(parts) == 3:
            return ('method', parts[0], parts[1], parts[2])
        return None
    if iid.startswith('G:'):
        rest = iid[2:]
        if '/' not in rest:
            return None
        name, gid = rest.rsplit('/', 1)
        return ('gate', name, gid)
    return None
