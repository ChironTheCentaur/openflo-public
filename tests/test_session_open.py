"""session_open: planning the reopen of a saved session without a window.

Where each sample's data is found (absolute path, then relative path, then
file name in the session folder; a sidecar pointer only inside the session
folder), what is staged for its load (compensation, QC) and restore (gates,
group, records, look), which samples are missing and kept offline, and how
saved gates move onto new units. Generated files and generic channel names
only; no Tk.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from openflo import session_open as so
from openflo.session_format import SessionVersionError
from tests._open_fixtures import make_fcs, session_dict, session_gates


def _touch(path):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('x')
    return str(path)


# ── where the raw file is found ─────────────────────────────────────────────

def test_an_existing_absolute_path_wins_over_rel_path_and_file_name(tmp_path):
    sess = tmp_path / 'proj'
    absolute = _touch(tmp_path / 'raw' / 'a.fcs')
    _touch(sess / 'data' / 'a.fcs')
    _touch(sess / 'a.fcs')
    got = so.resolve_sample_path(
        {'path': absolute, 'rel_path': 'data/a.fcs', 'basename': 'a.fcs'},
        str(sess))
    assert got == absolute


def test_rel_path_wins_over_the_file_name_when_the_path_is_gone(tmp_path):
    sess = tmp_path / 'proj'
    rel = _touch(sess / 'data' / 'a.fcs')
    _touch(sess / 'a.fcs')
    got = so.resolve_sample_path(
        {'path': str(tmp_path / 'gone' / 'a.fcs'), 'rel_path': 'data/a.fcs',
         'basename': 'a.fcs'}, str(sess))
    assert os.path.samefile(got, rel)


def test_a_windows_rel_path_is_read(tmp_path):
    sess = tmp_path / 'proj'
    rel = _touch(sess / 'data' / 'a.fcs')
    got = so.resolve_sample_path({'path': 'C:\\old\\a.fcs',
                                  'rel_path': 'data\\a.fcs'}, str(sess))
    assert os.path.samefile(got, rel)


def test_the_file_name_in_the_session_folder_is_the_last_resort(tmp_path):
    sess = tmp_path / 'proj'
    here = _touch(sess / 'b.fcs')
    assert so.resolve_sample_path(
        {'path': str(tmp_path / 'gone' / 'b.fcs')}, str(sess)) == here
    assert so.resolve_sample_path(
        {'path': '', 'basename': 'b.fcs'}, str(sess)) == here


def test_nothing_found_gives_back_the_saved_path(tmp_path):
    gone = str(tmp_path / 'gone' / 'c.fcs')
    assert so.resolve_sample_path({'path': gone}, str(tmp_path)) == gone
    assert so.resolve_sample_path({}, str(tmp_path)) == ''


# ── where the processed sidecar is found ────────────────────────────────────

def test_a_sidecar_pointer_resolves_inside_the_session_folder(tmp_path):
    side = _touch(tmp_path / 's_data' / 'A.npz')
    got = so.resolve_processed_path({'processed_data': 's_data/A.npz'}, 'A',
                                    str(tmp_path))
    assert os.path.samefile(got, side)


@pytest.mark.parametrize('pointer', ['../outside/A.npz', 'ABS'])
def test_a_sidecar_pointer_leaving_the_session_folder_is_refused(
        tmp_path, pointer, caplog):
    """A crafted session cannot make the reader open a file outside its
    folder, even one that exists."""
    outside = _touch(tmp_path / 'outside' / 'A.npz')
    sess = tmp_path / 'proj'
    sess.mkdir()
    ptr = outside if pointer == 'ABS' else pointer
    assert so.resolve_processed_path({'processed_data': ptr}, 'A',
                                     str(sess)) == ''
    assert 'out-of-tree' in caplog.text


def test_the_conventional_sidecar_is_found_when_the_pointer_is_stale(
        tmp_path):
    from openflo.paths import safe_sidecar_name
    data_dir = tmp_path / 'proj_data'
    side = _touch(data_dir / (safe_sidecar_name('A 1') + '.npz'))
    got = so.resolve_processed_path({'processed_data': 'proj_data/gone.npz'},
                                    'A 1', str(tmp_path), str(data_dir))
    assert got == side


# ── the plan ────────────────────────────────────────────────────────────────

@pytest.fixture
def project(tmp_path):
    """A session folder with three samples: A (raw FCS, beside the
    session), B (a processed sidecar and its raw FCS), C (file missing),
    plus an entry naming no sample."""
    sess = tmp_path / 'proj'
    a = make_fcs(sess / 'raw' / 'A.fcs', seed=1)
    b = make_fcs(sess / 'raw' / 'B.fcs', seed=2)
    side = _touch(sess / 'proj_data' / 'B.npz')
    unnamed = make_fcs(sess / 'raw' / 'U.fcs', seed=3)
    comp = {'channels': ['FL1-A', 'FL2-A'],
            'matrix': [[1.0, 0.2], [0.1, 1.0]], 'source': 'manual'}
    entries = [
        {'name': 'A', 'path': '/elsewhere/raw/A.fcs', 'rel_path': 'raw/A.fcs',
         'trial': 'Day 1', 'is_comp': False, 'comp': comp,
         'qc_applied': False, 'qc_removed': 7, 'color': '#123456',
         'plot_enabled': False, 'fmo_of': ['FL1-A'],
         'downsample_trim': {'from': 600, 'steps': [400], 'seed': 42}},
        {'name': 'B', 'path': b, 'processed_data': 'proj_data/B.npz',
         'color': '#1f77b4', 'plot_enabled': False,
         'transforms': 'not a record'},
        {'name': 'C', 'path': 'raw/C.fcs', 'basename': 'C.fcs',
         'processed_data': 'proj_data/C.npz', 'trial': 'Day 2'},
        {'path': unnamed},
    ]
    gates = {'A': session_gates('A'), 'C': session_gates('C')}
    return {'dir': str(sess), 'data_dir': str(sess / 'proj_data'), 'a': a,
            'b': b, 'side': side, 'unnamed': unnamed,
            'data': session_dict(entries, gates), 'comp': comp}


def _plan(project):
    return so.plan_session(project['data'], project['dir'],
                           data_dir=project['data_dir'])


def test_each_sample_loads_from_its_sidecar_else_its_raw_file(project):
    a, b, c, u = _plan(project).samples
    assert os.path.samefile(a.load_path, project['a'])
    assert not a.processed_path and a.raw_found
    assert b.load_path == project['side'] and b.processed_path
    assert b.raw_path == project['b']
    assert not c.raw_found and not c.processed_path
    assert u.name == '' and not u.staged and u.raw_path == project['unnamed']
    assert a.key == os.path.normcase(os.path.abspath(a.load_path))


def test_compensation_and_qc_are_staged_for_the_load(project):
    a, b, _c, _u = _plan(project).samples
    assert a.prep['qc'] is False and a.prep['qc_removed'] == 7
    assert a.prep['source_fcs'] is None
    np.testing.assert_array_equal(a.prep['comp']['matrix'],
                                  project['comp']['matrix'])
    assert a.prep['comp']['channels'] == ['FL1-A', 'FL2-A']
    # Saved before qc_applied existed: QC ran, so it runs again. A sidecar
    # is told its raw FCS.
    assert b.prep == {'comp': None, 'source_fcs': project['b'], 'qc': True,
                      'qc_removed': None, 'lmd_data_set': None,
                      'instrument_compensated': False, 'set_number': None}


def test_a_bad_matrix_is_not_staged(project):
    project['data']['samples'][0]['comp'] = {
        'channels': ['FL1-A', 'FL2-A'], 'matrix': [[1.0, float('nan')],
                                                   [0.0, 1.0]]}
    assert _plan(project).samples[0].prep['comp'] is None


def test_the_restore_record_holds_gates_group_and_records(project, caplog):
    a, b, c, _u = _plan(project).samples
    assert a.meta['gates'] == session_gates('A')
    assert a.meta['gates'] is not a.gates
    assert a.meta['trial'] == 'Day 1' and a.meta['is_comp'] is False
    assert a.meta['downsample_trim'] == {'from': 600, 'steps': [400],
                                         'seed': 42}
    assert a.fmo_of == ['FL1-A'] and a.color == '#123456'
    # The placeholder colour of a sample never shown is not restored.
    assert b.color is None and b.meta['gates'] == []
    assert 'transforms' not in b.meta
    assert 'transform record ignored' in caplog.text
    # A missing raw file is still staged under its saved path, so a relink
    # finds its record.
    assert c.staged and c.meta['trial'] == 'Day 2'


def test_missing_samples_are_kept_offline_not_dropped(project):
    plan = _plan(project)
    paths, gone = so.raw_loads(plan.samples, {'B'})
    assert [os.path.basename(p) for p, _n in paths] == ['A.fcs', 'U.fcs']
    assert [p.label for p in gone] == ['C']
    (c,) = gone
    assert c.gates == session_gates('C')
    kept = so.offline_entry(c.entry, project['dir'])
    assert 'processed_data' not in kept and 'processed_csv' not in kept
    assert kept['path'] == os.path.abspath(
        os.path.join(project['dir'], 'raw', 'C.fcs'))
    assert c.entry['processed_data'] == 'proj_data/C.npz'   # not changed


def test_without_its_sidecar_loading_a_processed_sample_loads_raw(project):
    """A caller that cannot read sidecars loads those samples raw."""
    paths, gone = so.raw_loads(_plan(project).samples)
    assert [os.path.basename(p) for p, _n in paths] == ['A.fcs', 'B.fcs',
                                                    'U.fcs']
    assert [p.label for p in gone] == ['C']


def test_an_unnamed_missing_entry_is_reported_by_its_file_name(tmp_path):
    (p,) = so.plan_samples([{'path': str(tmp_path / 'gone' / 'x.fcs')}], {},
                           str(tmp_path))
    assert p.label == 'x.fcs' and not p.staged


def test_an_old_session_is_upgraded_and_a_newer_one_refused(project):
    project['data']['version'] = 1
    plan = _plan(project)
    assert plan.notes and plan.data['version'] > 1
    assert plan.active == 'A'
    project['data']['version'] = 999
    with pytest.raises(SessionVersionError):
        _plan(project)


# ── gates onto new units ────────────────────────────────────────────────────

def test_gates_move_onto_the_units_the_sample_reopened_on():
    gates = session_gates('A')
    double = {'FL1-A': lambda v: np.asarray(v, dtype=float) * 2}
    moved = so.rescaled_gates(gates, double)
    assert moved[0]['value'] == 1.0
    assert moved[1] == gates[1]                       # FL2-A: not moved
    assert gates[0]['value'] == 0.5                   # input kept
    assert so.rescaled_gates(gates, {}) is gates
    assert so.rescaled_gates([], double) is None
    assert so.rescaled_gates(None, double) is None


def test_a_stamped_gate_keeps_its_stamp_when_its_units_change():
    from openflo.result_stamp import live_gate_stamp, stamp_gate
    g = stamp_gate(dict(session_gates('A')[0]), 'auto_gate')
    (moved,) = so.rescaled_gates(
        [g], {'FL1-A': lambda v: np.asarray(v, dtype=float) * 2})
    assert moved['value'] == 1.0
    assert live_gate_stamp(moved) is not None


# ── records restored with the sample ────────────────────────────────────────

def test_derived_channels_float_qc_counts_and_an_upper_case_fcs(tmp_path):
    """A derived-channel record is restored; a QC count written as a float
    is read; a sidecar whose raw file is named '.FCS' is told it."""
    sess = tmp_path / 'proj'
    raw = make_fcs(sess / 'raw' / 'B.FCS', seed=2)
    _touch(sess / 'proj_data' / 'B.npz')
    derived = {'D:ratio': {'expr': 'FL1-A / FL2-A'}}
    (p,) = so.plan_samples(
        [{'name': 'B', 'path': raw, 'processed_data': 'proj_data/B.npz',
          'derived': derived, 'qc_removed': 7.0}], {}, str(sess))
    assert p.meta['derived'] == derived
    assert p.meta['derived'] is not derived
    assert p.prep['qc_removed'] == 7
    assert p.prep['source_fcs'] == raw


# ── the down-sample trim record ─────────────────────────────────────────────

@pytest.mark.parametrize('rec, ok', [
    ({'from': 600, 'steps': [400, 250]}, True),
    ({'from': 600, 'steps': [400, 400]}, False),     # a step that cuts nothing
    ({'from': 600, 'steps': [600]}, False),
    ({'from': 600, 'steps': [400, 0]}, False),
    ({'from': 600, 'steps': []}, False),
    ({'from': 0, 'steps': [1]}, False),
    ({'steps': [1]}, False),
])
def test_a_trim_record_needs_each_step_smaller_than_the_last(rec, ok):
    from openflo.downsample_trim import TRIM_SEED, trim_record_from_entry
    got = trim_record_from_entry(rec)
    if ok:
        assert got == dict(rec, seed=TRIM_SEED)
    else:
        assert got is None
