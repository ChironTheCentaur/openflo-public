"""The real gate editor opens a session and imports a workspace through
session_open / wsp_open: what those plan is what the editor ends up with.

A session is saved by an editor (one sample with a processed sidecar), then
edited as a moved project would be: a sample found by its relative path
with its own matrix, QC choice, group, colour, FMO role and gates whose
saved ids have gaps, a sample saved under a name that is not its file's,
and a sample whose file is gone. Reopened in a fresh editor, each loads
from where the plan says under its saved name, prepared as staged, with its
gates, flags and booleans; the missing one is kept and saved back. A tray
tab's own gates move with a sample reopened on other units. A
workspace import puts each sample in its group with its own matrix, its
gates on the stored units with booleans remapped, and names the gate on an
uncompensated parameter.

GUI test: hidden editors, loads run on this thread; skipped only where Tk
cannot start. Generated data and generic channel names only.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from openflo.editor_session import _offline_parts
from tests._open_fixtures import make_fcs, session_gates, write_wsp
from tests._tk_wait import pump_until, settle
from tests.conftest import gui_unavailable
from tests.test_transform_records import _new_editor, capture_warnings, isolate_home

CUSTOM = [[1.0, 0.25], [0.05, 1.0]]


@pytest.fixture
def env(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)
    from tkinter import messagebox
    monkeypatch.setattr(messagebox, 'askyesno', lambda *a, **k: False)
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001 - any Tk start failure means no GUI
        gui_unavailable(f"Tk cannot initialise: {e}")
    root.withdraw()
    yield root, tmp_path
    root.destroy()


def _run_loads(root, ed, start):
    """Call `start()` (which queues loads) with the editor's load queues
    replaced by ones that load on this thread; returns the jobs."""
    jobs = {'fcs': [], 'csv': []}
    ed._queue_fcs_loads = lambda paths, front_names=(): jobs['fcs'].extend(
        paths)
    ed._queue_processed_loads = lambda items, front_names=(): jobs[
        'csv'].extend(items)
    start()
    for p in jobs['fcs']:
        ed._load_worker(ed._sample_name_for(p), p)
    for name, p in jobs['csv']:
        ed._load_csv_worker(name, p, {})
    n = len(jobs['fcs']) + len(jobs['csv'])
    assert pump_until(root, lambda: len(ed._samples) >= n, timeout=20)
    settle(root, 0.5)
    return jobs


def _saved_session(root, tmp):
    """A session saved by an editor, edited as a moved project: returns
    (session path, A's FCS path)."""
    proj = tmp / 'proj'
    a = make_fcs(proj / 'raw' / 'A.fcs', seed=1)
    b = make_fcs(proj / 'raw' / 'B.fcs', seed=2)
    d = make_fcs(proj / 'raw' / 'D.fcs', seed=4)
    ed = _new_editor(root)
    # D was saved under a name of its own, not its file's.
    for nm, p in (('A', a), ('B', b), ('renamed D', d)):
        ed._on_loaded(nm, ed._prepared_fcs_sample(p))
    ed._samples['B'].data['cluster'] = np.arange(
        len(ed._samples['B'].data)) % 3
    path = proj / 'two.flowsession'
    ed._write_session(str(path))
    ed.destroy()
    data = json.loads(path.read_text(encoding='utf-8'))
    by = {e['name']: e for e in data['samples']}
    assert 'processed_data' in by['B'] and 'processed_data' not in by['A']
    by['A'].update(path=str(tmp / 'old_computer' / 'A.fcs'),
                   comp={'channels': ['FL1-A', 'FL2-A'], 'matrix': CUSTOM,
                         'source': 'custom'},
                   qc_applied=False, trial='Day 1', is_comp=True,
                   color='#123456', fmo_of=['FL1-A'])
    gates = session_gates('A') + [
        {'id': 'g9', 'parent_id': None, 'kind': 'threshold',
         'channel': 'FL2-A', 'value': 0.2, 'name': 'A no colour'}]
    # C: its file is gone; its entry names it relative to the session, with
    # a sidecar pointer that is gone too.
    data['samples'].append({'name': 'C', 'path': 'raw/C.fcs',
                            'processed_data': 'two_data/C.npz',
                            'trial': 'Day 2'})
    data['sample_gates'] = {'A': gates, 'C': session_gates('C')}
    path.write_text(json.dumps(data), encoding='utf-8')
    return path, a


def test_a_reopened_session_is_what_the_plan_says(env):
    from openflo.session_open import plan_session
    root, tmp = env
    path, a_fcs = _saved_session(root, tmp)
    plan = plan_session(json.loads(path.read_text(encoding='utf-8')),
                        str(path.parent),
                        data_dir=str(path.with_suffix('')) + '_data')
    pa, pb, _pd, pc = plan.samples
    ed = _new_editor(root)
    _run_loads(root, ed, lambda: ed._load_session_path(str(path)))

    # D comes back under its saved name.
    assert set(ed._samples) == {'A', 'B', 'renamed D'}
    a, b = ed._samples['A'], ed._samples['B']
    # Each loads from where the plan says: A relinked by its relative path,
    # B from its sidecar (its cluster column is there).
    assert os.path.samefile(a.path, pa.load_path)
    assert os.path.samefile(a.path, a_fcs)
    assert b.path == pb.processed_path and 'cluster' in b.data.columns
    # A is prepared as staged: its own matrix, its QC choice.
    np.testing.assert_allclose(a.comp_matrix, CUSTOM)
    assert a.compensation_source == 'custom'
    assert a.qc_applied is False
    # Its group, flags and look.
    assert ed._sample_trial['A'] == 'Day 1'
    assert ed._sample_is_comp['A'] is True
    assert ed._sample_colors['A'] == '#123456'
    assert ed._fmo_controls == {'FL1-A': 'A'}
    # Its gates, with fresh ids, parents and operands remapped and saved
    # flags kept; a gate saved without a colour gets one.
    gates = ed._sample_gates['A']
    by_name = {g['name']: gid for gid, g in gates.items()}
    order = ed._sample_gate_order['A']
    assert [gates[g]['name'] for g in order] == [
        'A pos', 'A mid', 'A either', 'A no colour']
    assert gates[by_name['A mid']]['parent_id'] == by_name['A pos']
    assert gates[by_name['A mid']]['enabled'] is False
    assert gates[by_name['A either']]['operands'] == [
        by_name['A mid'], by_name['A pos']]
    assert gates[by_name['A pos']]['color'] == '#aa0000'
    assert gates[by_name['A no colour']].get('color')
    assert ed._sample_gate_seq['A'] == len(order)
    # C's file is gone: kept offline with its gates, and saved back.
    assert not pc.raw_found
    entries, off_gates = _offline_parts(ed)
    assert [e['name'] for e in entries] == ['C']
    assert entries[0]['path'] == str(path.parent / 'raw' / 'C.fcs')
    assert 'processed_data' not in entries[0]
    assert [g['name'] for g in off_gates['C']] == [
        g['name'] for g in session_gates('C')]
    data = ed._write_session(str(path))
    assert {e['name'] for e in data['samples']} == {'A', 'B', 'renamed D',
                                                    'C'}
    assert [g['name'] for g in data['sample_gates']['C']] == [
        g['name'] for g in session_gates('C')]
    ed.destroy()


def test_an_imported_workspace_is_what_the_plan_says(env):
    from openflo.wsp_open import plan_wsp
    root, tmp = env
    a = make_fcs(tmp / 'run' / 'day1' / 'a.fcs', seed=1)
    b = make_fcs(tmp / 'run' / 'day2' / 'b.fcs', seed=2)
    wsp = write_wsp(tmp / 'run' / 'x.wsp',
                    [('a', a, 'Stim', True), ('b', b, None, False)],
                    uncompensated=True)
    plan = plan_wsp(wsp)
    ed = _new_editor(root)
    jobs = _run_loads(root, ed, lambda: ed._ingest_wsp(wsp))
    assert sorted(jobs['fcs']) == sorted(s.fcs_path for s in plan.samples)
    sa, sb = ed._samples['a'], ed._samples['b']
    # Group and matrix as the workspace has them.
    assert ed._sample_trial['a'] == 'Stim'
    from openflo.workspace_run import derive_trial_name
    assert ed._sample_trial['b'] == derive_trial_name(b)
    assert sa.compensation_source == 'wsp:Acquisition-defined'
    np.testing.assert_allclose(sa.comp_matrix, plan.samples[0].comp['matrix'])
    assert sb.compensation_source != sa.compensation_source
    # Gates: fresh ids, the tree and the boolean remapped, on stored units.
    for name in ('a', 'b'):
        gates = ed._sample_gates[name]
        order = ed._sample_gate_order[name]
        lab = {gates[g]['label']: g for g in order}
        assert list(lab) == ['Cells', 'FL1+', 'FL2 mid', 'both']
        assert order == ['g1', 'g2', 'g3', 'g4']
        assert gates[lab['FL1+']]['parent_id'] == lab['Cells']
        assert gates[lab['both']]['operands'] == [lab['FL1+'],
                                                  lab['FL2 mid']]
        assert gates[lab['FL1+']]['value'] < 10       # not FlowJo's 2000
        assert all(g['enabled'] and g.get('color') for g in gates.values())
    # The gate FlowJo counts on uncompensated FL2-A is named for sample a.
    notes = [e['details'].get('note') for e in ed._audit_log.entries()
             if e['action'] == 'load.note']
    unc = [n for n in notes if 'uncompensated' in (n or '')]
    assert len(unc) == 1 and "'FL2 mid'" in unc[0] and 'FL2-A' in unc[0]
    assert not ed._pending_sample_gates and not ed._pending_wsp_trial
    ed.destroy()


def test_a_tabs_own_gates_move_with_a_reopened_sample(env):
    """A sample saved on asinh cofactor 5 reopens on the editor's 150: the
    gates another tray tab keeps for it move with the data, as the live
    tab's do."""
    from openflo.tree_ids import sample_iid
    from tests.test_session_restore_transforms import (
        _fcs_without_spill,
        _loaded,
        _restore,
    )
    root, tmp = env
    ed = _new_editor(root)
    ed._on_loaded('A', _loaded(_fcs_without_spill(tmp / 'A.fcs', 1)))
    ed._apply_channel_transforms({'FL1-A': 'asinh'})
    ed._samples['A'].retransform(['FL1-A'], method='asinh', cofactor=5.0)
    cut = float(np.arcsinh(2000 / 5))
    ed._set_active_sample('A')
    ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A', 'value': cut,
                  'parent_id': None, 'name': 'FL1 bright', 'enabled': True})
    ed._refresh_gate_list()
    ed.gate_tv.selection_set(sample_iid('A'))
    ed._new_tab_from_selection()
    path = tmp / 'tabs.flowsession'
    ed._write_session(str(path))
    ed.destroy()
    ed2 = _restore(root, path, raw_first=True)
    settle(root, 0.3)
    live = [g['value'] for g in ed2._sample_gates['A'].values()]
    ts = ed2._view_tabs
    other = [g['value'] for t in ts.tabs if t is not ts.active
             for g in ((t.gates or {}).get('A') or {}).values()]
    assert len(live) == 1 and live[0] != pytest.approx(cut)
    assert other == pytest.approx(live)
    ed2.destroy()
