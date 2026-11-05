"""Shared pieces for the tests that check the main window keeps answering
while OpenFlo works: an editor with no blocking dialogs (``env``), a bare
hidden Tk root (``tk_root``), a gated sample, waits and spies.

Generated data, synthetic sample names and generic channels only.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from tests._tk_wait import pump_until, settle
from tests.conftest import gui_unavailable
from tests.test_beta_look import _load, _write_fcs
from tests.test_transform_records import _new_editor, capture_warnings, isolate_home


@pytest.fixture
def env(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    notes = capture_warnings(monkeypatch)       # never a real (blocking) modal
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001 - any Tk start failure means no GUI
        gui_unavailable(f"Tk cannot initialise: {e}")
    root.withdraw()
    ed = _new_editor(root)
    yield root, ed, tmp_path, notes
    root.destroy()


@pytest.fixture
def tk_root():
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001 - any Tk start failure means no GUI
        gui_unavailable(f"Tk cannot initialise: {e}")
    root.withdraw()
    yield root
    root.destroy()


def _on_main():
    return threading.current_thread() is threading.main_thread()


def _gated(root, ed, tmp):
    """sample_a loaded, a rect gate on FSC-A x SSC-A plotted there."""
    _load(root, ed, [('sample_a', _write_fcs(tmp / 'a.fcs'))])
    ed._set_active_sample('sample_a')
    ed.x_combo.set('FSC-A')
    ed.y_combo.set('SSC-A')
    gid = ed._add_gate({
        'kind': 'rect', 'name': 'Cells', 'x_channel': 'FSC-A',
        'y_channel': 'SSC-A', 'x0': 60_000.0, 'x1': 180_000.0,
        'y0': 20_000.0, 'y1': 100_000.0, 'parent_id': None})
    ed._replot()
    settle(root, 0.6)
    return gid


def _quiet(root, ed, timeout=20.0):
    """Wait until the background warm-up has nothing left to compute, and
    let the tree take in the counts it brought (a count arriving is a real
    change, which rebuilds the tree)."""
    pump_until(root, lambda: not (
        getattr(ed, '_plotcache_inflight', None) is not None
        or getattr(ed, '_plotcache_warm_names', None)
        or getattr(ed, '_plotcache_sched_after', None)), timeout)
    settle(root, 0.2)
    ed._refresh_gate_list()


def _count(ed, name, gid):
    from openflo.gate_eval import cumulative_gate_mask
    return int(np.asarray(cumulative_gate_mask(
        ed._sample_gates[name], gid, ed._samples[name].data),
        dtype=bool).sum())


def _spy(monkeypatch, obj, attr):
    """Record each call of ``obj.attr`` (still calling through)."""
    calls = []
    real = getattr(obj, attr)

    def spy(*a, **k):
        calls.append(a)
        return real(*a, **k)
    monkeypatch.setattr(obj, attr, spy)
    return calls


def _let_the_load_settle(root, ed):
    """Run the load's debounced settle pass now and drain what it queues.

    Loading arms a 120 ms timer whose pass writes "N sample(s) loaded" and
    re-selects a sample row, whose queued readout writes "name:  N events".
    Left pending, a slow machine (the Windows CI legs) ran both after a
    dialog's job had answered, so the status line no longer held its answer.
    """
    pending = getattr(ed, '_load_settle_after', None)
    if pending:
        ed.after_cancel(pending)
        ed._load_settle_after = None
        ed._on_load_settled()
    root.update()
