"""A saved-data file cut short does not load as the whole sample.

A session keeps each sample with computed columns (clusters, ...) in a
data file beside it, and reopening prefers that file to the FCS. A file cut
short by a crash mid-write loaded as the sample, silently: 401 of 1,000
events. Now a file that can't be read, or that holds fewer rows than its
record says, is set aside: the sample is rebuilt from its raw FCS, keeps its
gates, and the user is told.

Generated data: one synthetic FCS with a cluster column (so it gets a data
file) and a gate. Negative cases: an intact file still loads from the file
(cluster column kept, no warning), and a cut-short file whose FCS is gone
fails to load rather than loading part of the sample.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from tests.test_session_restore_transforms import (
    _fcs_without_spill,
    _loaded,
    env,  # noqa: F401  (fixture)
)
from tests.test_transform_records import _new_editor, _pump


def _saved_session(root, tmp):
    ed = _new_editor(root)
    s = _loaded(_fcs_without_spill(tmp / 'synthetic_A.fcs', 1))
    s.data['cluster'] = np.arange(len(s.data)) % 3
    ed._on_loaded('synthetic_A', s)
    ed._set_active_sample('synthetic_A')
    ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A', 'value': 0.5,
                  'parent_id': None, 'name': 'pos', 'enabled': True})
    path = tmp / 'cut.flowsession'
    data = ed._write_session(str(path))
    side = tmp / data['samples'][0]['processed_data']
    assert side.is_file()
    return path, side, len(s.data)


def _reopen(root, path):
    ed = _new_editor(root)
    jobs = {'fcs': [], 'csv': []}
    ed._queue_fcs_loads = lambda paths, front_names=(): jobs['fcs'].extend(
        paths)
    ed._queue_processed_loads = lambda items, front_names=(): jobs[
        'csv'].extend(items)
    ed._load_session_path(str(path))
    for nm, p in jobs['csv']:
        ed._load_csv_worker(nm, p, {})
        root.update()
    return ed


def _cut_rows(side, keep):
    from openflo.session_sidecar import read_frame, write_frame
    df = read_frame(str(side))
    write_frame(df.iloc[:keep], str(side))


def _gate_names(ed):
    return [g.get('name')
            for g in ed._sample_gates.get('synthetic_A', {}).values()]


@pytest.mark.parametrize('how', ['fewer_rows', 'truncated_bytes'])
def test_cut_short_file_reloads_from_the_fcs(env, how):  # noqa: F811
    root, _, tmp = env
    path, side, n = _saved_session(root, tmp)
    if how == 'fewer_rows':
        _cut_rows(side, int(n * 0.4))
    else:
        blob = side.read_bytes()
        side.write_bytes(blob[:int(len(blob) * 0.4)])
    from tkinter import messagebox
    shown = []
    messagebox.showwarning = lambda *a, **k: shown.append(a)
    ed = _reopen(root, path)
    assert _pump(root, lambda: 'synthetic_A' in ed._samples, timeout=20)
    s = ed._samples['synthetic_A']
    assert len(s.data) == n                       # every event, not 40 %
    assert s.path.endswith('synthetic_A.fcs')
    assert _pump(root, lambda: bool(shown), timeout=5)
    assert 'synthetic_A' in shown[0][1]
    assert _pump(root, lambda: _gate_names(ed) == ['pos'], timeout=5)


# ── negative cases ────────────────────────────────────────────────────────


def test_intact_file_still_loads_from_the_file(env):  # noqa: F811
    root, _, tmp = env
    path, side, n = _saved_session(root, tmp)
    from tkinter import messagebox
    shown = []
    messagebox.showwarning = lambda *a, **k: shown.append(a)
    ed = _reopen(root, path)
    assert _pump(root, lambda: 'synthetic_A' in ed._samples, timeout=20)
    s = ed._samples['synthetic_A']
    assert 'cluster' in s.data.columns and len(s.data) == n
    assert os.path.samefile(s.path, side)
    root.update()
    assert not [a for a in shown if 'incomplete' in str(a[0])]


def test_cut_short_file_without_its_fcs_does_not_load_part(env):  # noqa: F811
    root, _, tmp = env
    path, side, n = _saved_session(root, tmp)
    _cut_rows(side, int(n * 0.4))
    os.remove(tmp / 'synthetic_A.fcs')
    ed = _reopen(root, path)
    _pump(root, lambda: not ed._loading, timeout=5)
    assert 'synthetic_A' not in ed._samples
    assert 'FCS file was not found' in ed.status_var.get()
