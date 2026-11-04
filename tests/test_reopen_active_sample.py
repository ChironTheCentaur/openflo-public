"""Reopening a session restores the highlighted sample and the plot ticks
it was saved with, whatever order the samples land in.

A session saved with sample_a highlighted and sample_b unticked sometimes
came back with sample_b highlighted and ticked. The load settle highlights
the sample active WHILE loading (the first to land); when that pass ran
before the deferred restore, the restore made sample_a active but left the
tree's selection on the other row. The restore's own tree rebuild then
re-selected that row, the selection handler made it active again, and
highlighting it ticked it for plotting. Gates, compensation and clustering
act on the highlighted sample, so the next gate could land on the wrong one.

Each load order is driven explicitly, load settle before the restore (the
order that failed) and after it, so the race is reproduced every run.
Synthetic files and generic channels only.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from tests._fcs_writer import write_float_fcs
from tests._tk_wait import pump_until, settle
from tests.conftest import gui_unavailable
from tests.test_transform_records import _new_editor, capture_warnings, isolate_home

CH = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A']
NAMES = ('sample_a', 'sample_b', 'sample_c')
SAVED_TICKS = {'sample_a': True, 'sample_b': False, 'sample_c': True}


def _fcs(path, seed):
    rng = np.random.default_rng(seed)
    ev = np.column_stack([rng.lognormal(10, .2, 400), rng.lognormal(9, .2, 400),
                          rng.lognormal(6, .5, 400), rng.lognormal(6, .5, 400)])
    write_float_fcs(path, ev, CH)
    return str(path)


@pytest.fixture
def saved(tmp_path, monkeypatch):
    """A session of three samples saved with sample_a highlighted, sample_b
    unticked and sample_c ticked; yields (root, session path)."""
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001 - any Tk start failure means no GUI
        gui_unavailable(f"Tk cannot initialise: {e}")
    root.withdraw()
    try:
        from openflo.pipeline import FlowSample
        ed = _new_editor(root)
        for i, n in enumerate(NAMES):
            p = _fcs(tmp_path / f'{n}.fcs', i)
            s = FlowSample(p)
            s.apply_transform()
            ed._on_loaded(ed._sample_name_for(p), s)
        settle(root, 0.3)
        assert sorted(ed._samples) == list(NAMES)
        ed._sample_plot_enabled.update(SAVED_TICKS)
        ed._set_active_sample('sample_a')
        ed._refresh_gate_list()
        path = tmp_path / 'three.flowsession'
        data = ed._write_session(str(path))
        assert data['active_sample'] == 'sample_a'
        assert {e['name']: bool(e.get('plot_enabled'))
                for e in data['samples']} == SAVED_TICKS
        ed.destroy()
        yield root, path
    finally:
        root.destroy()


def _reopen(root, path, order, settle_first):
    """Reopen ``path`` in a fresh editor, landing the samples in ``order``,
    then run the load settle and the deferred restore in the given order
    (both are timers in the app; here they are called directly)."""
    ed = _new_editor(root)
    jobs = []
    ed._queue_fcs_loads = lambda paths, front_names=(): jobs.extend(paths)
    ed._queue_processed_loads = lambda items, front_names=(): None
    ed._load_session_path(str(path))
    by_name = {ed._sample_name_for(p): p for p in jobs}
    assert sorted(by_name) == list(NAMES)
    for n in order:
        ed._load_worker(n, by_name[n])
    assert pump_until(root, lambda: len(ed._samples) == 3,
                      timeout=10, pause=0)
    if settle_first:
        ed._on_load_settled()
        ed._apply_session_restore()
    else:
        ed._apply_session_restore()
        ed._on_load_settled()
    settle(root, 1.5)       # queued selection events and the app's timers
    return ed


def _selected_owner(ed):
    sel = ed.gate_tv.selection()
    p = ed._parse_iid(sel[0]) if sel else None
    return p[1] if p else None


@pytest.mark.parametrize('settle_first', [True, False],
                         ids=['settle-then-restore', 'restore-then-settle'])
def test_reopen_highlights_the_saved_sample_and_keeps_its_ticks(
        saved, settle_first):
    root, path = saved
    for order in itertools.permutations(NAMES):
        ed = _reopen(root, path, order, settle_first)
        try:
            got = (ed._active_sample, _selected_owner(ed),
                   {n: bool(ed._sample_plot_enabled.get(n)) for n in NAMES})
            assert got == ('sample_a', 'sample_a', SAVED_TICKS), order
        finally:
            ed.destroy()
