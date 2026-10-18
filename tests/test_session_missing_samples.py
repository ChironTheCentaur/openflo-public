"""A session sample whose FCS file is missing survives the next save.

Opening a session whose FCS for one sample had moved skipped that sample,
and the save (or autosave) after it wrote the session without it: the
sample and every gate on it were gone from the file for good. The only
sign was a status-bar line the next loaded sample overwrote.

Generated data: two synthetic FCS files with a gate on each. The negative
cases: a sample whose file IS there is saved as a normal sample (not an
offline copy), and relinking from a folder without the file finds nothing
and keeps the sample offline.
"""
from __future__ import annotations

import json
import os
import shutil

import pytest

from tests.test_session_restore_transforms import (
    _fcs_without_spill,
    _loaded,
)
from tests.test_transform_records import (
    _new_editor,
    _pump,
    capture_warnings,
    isolate_home,
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"Tk cannot initialise: {e}")
    root.withdraw()
    yield root, tmp_path
    root.destroy()


def _session_with_two_samples(root, tmp):
    data_dir = tmp / 'data'
    data_dir.mkdir()
    ed = _new_editor(root)
    for nm, seed in (('synthetic_A', 1), ('synthetic_B', 2)):
        ed._on_loaded(nm, _loaded(_fcs_without_spill(data_dir / f'{nm}.fcs',
                                                     seed)))
        ed._set_active_sample(nm)
        ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A', 'value': 0.5,
                      'parent_id': None, 'name': f'{nm} pos',
                      'enabled': True})
    path = tmp / 'proj' / 'two.flowsession'
    path.parent.mkdir()
    ed._write_session(str(path))
    return path, data_dir


def _reopen(root, path, monkeypatch):
    """A fresh editor on `path`, running its loads here. The missing-file
    prompt is answered No (the test drives relinking itself)."""
    from tkinter import messagebox
    asked = []
    monkeypatch.setattr(messagebox, 'askyesno',
                        lambda *a, **k: asked.append(a) or False)
    ed = _new_editor(root)
    jobs = []
    ed._queue_fcs_loads = lambda paths, front_names=(): jobs.extend(paths)
    ed._queue_processed_loads = lambda items, front_names=(): None
    ed._load_session_path(str(path))
    for job in list(jobs):
        ed._load_worker(ed._sample_name_for(job), job)
        root.update()
    _pump(root, lambda: len(ed._samples) == len(jobs), timeout=10)
    _pump(root, lambda: bool(asked), timeout=3)
    ed._queue_fcs_loads = lambda paths, front_names=(): [
        (ed._load_worker(ed._sample_name_for(p), p), root.update())
        for p in paths]
    return ed, asked


def _saved(path):
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    return ({s['name'] for s in d['samples']},
            {n: [g.get('name') for g in gl]
             for n, gl in d['sample_gates'].items()})


def test_missing_sample_and_gates_survive_a_resave(env, monkeypatch):
    root, tmp = env
    path, data_dir = _session_with_two_samples(root, tmp)
    away = tmp / 'elsewhere'
    away.mkdir()
    shutil.move(str(data_dir / 'synthetic_B.fcs'), away / 'synthetic_B.fcs')
    ed, asked = _reopen(root, path, monkeypatch)
    assert set(ed._samples) == {'synthetic_A'}
    assert asked, 'the user was not told which files are missing'
    ed._write_session(str(path))
    names, gates = _saved(path)
    assert names == {'synthetic_A', 'synthetic_B'}
    assert gates['synthetic_B'] == ['synthetic_B pos']
    assert gates['synthetic_A'] == ['synthetic_A pos']


def test_locate_folder_relinks_and_restores_gates(env, monkeypatch):
    root, tmp = env
    path, data_dir = _session_with_two_samples(root, tmp)
    away = tmp / 'elsewhere' / 'deeper'
    away.mkdir(parents=True)
    shutil.move(str(data_dir / 'synthetic_B.fcs'), away / 'synthetic_B.fcs')
    ed, _ = _reopen(root, path, monkeypatch)
    found, still = ed._relink_missing(str(tmp / 'elsewhere'))
    assert (found, still) == (['synthetic_B'], [])
    assert _pump(root, lambda: 'synthetic_B' in ed._samples, timeout=10)
    assert _pump(root, lambda: any(
        g.get('name') == 'synthetic_B pos'
        for g in ed._sample_gates.get('synthetic_B', {}).values()), timeout=5)
    data = ed._write_session(str(path))
    entry = next(s for s in data['samples'] if s['name'] == 'synthetic_B')
    assert os.path.samefile(entry['path'], away / 'synthetic_B.fcs')


# ── negative cases ────────────────────────────────────────────────────────


def test_nothing_offline_when_every_file_is_there(env, monkeypatch):
    root, tmp = env
    path, _ = _session_with_two_samples(root, tmp)
    ed, asked = _reopen(root, path, monkeypatch)
    assert set(ed._samples) == {'synthetic_A', 'synthetic_B'}
    assert not asked
    assert not getattr(ed, '_offline_samples', None)
    assert ed._offline_session_parts() == ([], {})


def test_relink_from_a_folder_without_the_file_keeps_it_offline(
        env, monkeypatch):
    root, tmp = env
    path, data_dir = _session_with_two_samples(root, tmp)
    os.remove(data_dir / 'synthetic_B.fcs')
    other = tmp / 'other'
    other.mkdir()
    _fcs_without_spill(other / 'synthetic_C.fcs', 3)    # not the one wanted
    ed, _ = _reopen(root, path, monkeypatch)
    assert ed._relink_missing(str(other)) == ([], ['synthetic_B'])
    assert 'synthetic_B' not in ed._samples
    ed._write_session(str(path))
    assert 'synthetic_B' in _saved(path)[0]
