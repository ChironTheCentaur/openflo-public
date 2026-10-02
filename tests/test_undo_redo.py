"""Editor undo/redo snapshot/restore logic.

Tk isn't constructed; we drive the pure snapshot/restore/checkpoint
methods off a stub carrying the gate-state attributes they touch, the
same approach as the other editor-helper tests.
"""
# pyright: reportArgumentType=false, reportCallIssue=false
from __future__ import annotations

import os
import types

import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')

try:
    import tkinter as _tk  # noqa: F401

    from openflo.gui import ViewGateEditorWindow as V
except (ImportError, RuntimeError) as e:
    pytest.skip(f"openflo.gui not importable: {e}", allow_module_level=True)


def _editor():
    """Stub with the undo machinery bound as real methods."""
    stub = types.SimpleNamespace(
        _sample_gates={'A': {'g1': {'kind': 'threshold', 'channel': 'CD11b',
                                    'value': 1.0, 'parent_id': None}}},
        _sample_gate_order={'A': ['g1']},
        _sample_gate_seq={'A': 1},
        _cluster_labels={},
        _quad_set_seq=0,
        _active_sample='A',
        _gate_id_seq=1,
        _gates={}, _gate_id_order=[],
        _undo_stack=[], _redo_stack=[],
        _undo_pending=False, _suspend_undo=False, _UNDO_MAX=100,
        _refresh_gate_list=lambda *a, **k: None,
        _schedule_replot=lambda *a, **k: None,
        after_idle=lambda *a, **k: None,
        status_var=types.SimpleNamespace(set=lambda *a, **k: None),
    )
    # Bind active-sample shortcut to the real dict.
    stub._gates = stub._sample_gates['A']
    stub._gate_id_order = stub._sample_gate_order['A']
    for fn in ('_gate_state_snapshot', '_restore_gate_state', '_checkpoint',
               '_clear_undo_pending', '_undo', '_redo'):
        setattr(stub, fn, types.MethodType(getattr(V, fn), stub))
    return stub


def test_snapshot_is_deep_copy():
    ed = _editor()
    snap = ed._gate_state_snapshot()
    ed._sample_gates['A']['g1']['value'] = 999.0   # mutate after snapshot
    assert snap['gates']['A']['g1']['value'] == 1.0  # snapshot unaffected


def test_checkpoint_then_undo_restores_previous_state():
    ed = _editor()
    ed._checkpoint()                       # capture state with 1 gate
    # Simulate adding a gate.
    ed._sample_gates['A']['g2'] = {'kind': 'threshold', 'channel': 'CD45',
                                   'value': 2.0, 'parent_id': None}
    ed._sample_gate_order['A'].append('g2')
    assert len(ed._sample_gates['A']) == 2

    ed._undo()
    assert set(ed._sample_gates['A']) == {'g1'}        # g2 gone
    assert ed._gates is ed._sample_gates['A']          # shortcut rebound


def test_redo_reapplies():
    ed = _editor()
    ed._checkpoint()
    ed._sample_gates['A']['g2'] = {'kind': 'threshold', 'parent_id': None}
    ed._sample_gate_order['A'].append('g2')
    ed._undo()
    assert set(ed._sample_gates['A']) == {'g1'}
    ed._redo()
    assert set(ed._sample_gates['A']) == {'g1', 'g2'}


def test_new_action_clears_redo():
    ed = _editor()
    ed._checkpoint()
    ed._sample_gates['A']['g2'] = {'parent_id': None}
    ed._undo()
    assert ed._redo_stack                  # something to redo
    ed._undo_pending = False               # simulate a fresh event tick
    ed._checkpoint()                       # a new edit
    assert ed._redo_stack == []            # redo history discarded


def test_checkpoint_coalesces_within_event():
    ed = _editor()
    ed._checkpoint()
    ed._checkpoint()      # same "event" (pending flag still set) → no 2nd push
    assert len(ed._undo_stack) == 1


def test_checkpoint_suspended_is_noop():
    ed = _editor()
    ed._suspend_undo = True
    ed._checkpoint()
    assert ed._undo_stack == []


def test_undo_empty_is_safe():
    ed = _editor()
    ed._undo()           # nothing to undo
    assert ed._undo_stack == [] and ed._redo_stack == []


def test_undo_caps_at_max():
    ed = _editor()
    ed._UNDO_MAX = 3
    for i in range(6):
        ed._undo_pending = False          # simulate distinct events
        ed._checkpoint()
        ed._sample_gates['A'][f'x{i}'] = {'parent_id': None}
    assert len(ed._undo_stack) == 3       # oldest dropped
