"""Normalize batches (CytoNorm) applied to the loaded samples, then the
session saved and reopened.

CytoNorm replaces a sample's values in place and adds no column, so a
sample with no other reason for a processed-data sidecar (uncompensated,
not clustered) used to be saved without one: the reopen read the raw FCS
again and the normalization was gone, with gates and statistics silently
on other values. Generated data: two days, two samples each, day 2 reading
40 % brighter on FL1-A and FL2-A. The reopened values must equal the
normalized ones (and differ from the raw ones), compensated or not, through
Save and Save As alike; after a reopen the in-memory originals are gone, so
Undo batch normalization must say there is nothing to undo and change
nothing.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from tests.test_derived_kinetics import _loaded, _restore_session
from tests.test_transform_records import (
    _new_editor,
    capture_warnings,
    isolate_home,
)

CH = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A']
SPILL = {'SPILL': '2,FL1-A,FL2-A,1,0.1,0.05,1'}


def _fcs(path, seed, scale, meta=None, n=1500):
    import flowio
    rng = np.random.default_rng(seed)
    fl = np.r_[rng.lognormal(6, .25, (n // 2, 2)),
               rng.lognormal(8, .25, (n - n // 2, 2))] * scale
    ev = np.column_stack([rng.lognormal(10, .2, n),
                          rng.lognormal(9, .2, n), fl])
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.astype(np.float32).flatten().tolist(), CH,
                          metadata_dict=meta)
    return str(path)


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


def _dim(vals):
    """Where the dim population of FL1-A sits (its median)."""
    v = vals[:, 0]
    return float(np.median(v[v < np.median(v)]))


def _normalized_editor(root, tmp, comp):
    """Four loaded samples over two days, CytoNorm applied (pooled goal,
    one metacluster). Returns the editor, the raw values and the
    normalized ones."""
    ed = _new_editor(root)
    meta = SPILL if comp else None
    for i, (name, day, scale) in enumerate([
            ('d1_a', 'Run1', 1.0), ('d1_b', 'Run1', 1.0),
            ('d2_a', 'Run2', 1.4), ('d2_b', 'Run2', 1.4)]):
        s = _loaded(_fcs(tmp / f'{name}.fcs', i + 1, scale, meta))
        assert bool(getattr(s, 'comp_channels', None)) == comp
        ed._on_loaded(name, s)
        ed._sample_trial[name] = day
    raw = {n: ed._samples[n].data[['FL1-A', 'FL2-A']].to_numpy(copy=True)
           for n in ed._sample_order}
    from openflo.ui_batch_normalize import BatchNormalizeWindow
    w = BatchNormalizeWindow(ed)
    w.mode_var.set('goal')
    w.meta_var.set(1)
    w._run(sync=True)
    assert w._result is not None, w.status_var.get()
    w.destroy()
    norm = {n: ed._samples[n].data[['FL1-A', 'FL2-A']].to_numpy(copy=True)
            for n in ed._sample_order}
    # The normalization moved day 2 a long way (else the test proves
    # nothing): its dim population now sits by day 1's.
    assert not np.allclose(norm['d2_a'], raw['d2_a'], atol=1e-3)
    gap = _dim(raw['d2_a']) - _dim(raw['d1_a'])
    assert gap > 0
    assert abs(_dim(norm['d2_a']) - _dim(norm['d1_a'])) < 0.3 * gap
    return ed, raw, norm


def _reopened_values(ed2):
    return {n: ed2._samples[n].data[['FL1-A', 'FL2-A']].to_numpy()
            for n in ed2._sample_order}


@pytest.mark.parametrize('comp', [False, True],
                         ids=['uncompensated', 'compensated'])
def test_cytonorm_survives_save_and_reopen(env, comp):
    root, tmp = env
    ed, raw, norm = _normalized_editor(root, tmp, comp)
    path = tmp / 'norm.flowsession'
    data = ed._write_session(str(path))
    for entry in data['samples']:
        assert entry.get('processed_data'), entry['name']
        assert entry.get('batch_norm'), entry['name']
    ed2, jobs = _restore_session(root, path)
    assert len(jobs['csv']) == 4 and not jobs['fcs'], jobs
    got = _reopened_values(ed2)
    for n in norm:
        np.testing.assert_allclose(got[n], norm[n], rtol=1e-5, atol=1e-5,
                                   err_msg=n)
        # Both days move onto the goal (their mean), so none is raw.
        assert not np.allclose(got[n], raw[n], atol=1e-3), n
    # The record goes on with the sample, so a second save keeps it too.
    again = ed2._write_session(str(tmp / 'again.flowsession'))
    assert all(e.get('batch_norm') and e.get('processed_data')
               for e in again['samples'])


def test_cytonorm_survives_save_as_and_autosave(env, monkeypatch):
    root, tmp = env
    ed, _raw, norm = _normalized_editor(root, tmp, comp=False)
    # Save As: a new session file in another folder.
    (tmp / 'elsewhere').mkdir()
    target = tmp / 'elsewhere' / 'copy.flowsession'
    from tkinter import filedialog
    monkeypatch.setattr(filedialog, 'asksaveasfilename',
                        lambda **k: str(target))
    ed._save_session_as()
    assert target.is_file()
    ed2, jobs = _restore_session(root, target)
    assert len(jobs['csv']) == 4, jobs
    got = _reopened_values(ed2)
    for n in norm:
        np.testing.assert_allclose(got[n], norm[n], rtol=1e-5, atol=1e-5)
    # The autosave plans its sidecars with the same rule.
    plan = ed._plan_session_write(str(tmp / 'auto.flowsession'))
    assert sorted(j['name'] for j in plan['jobs']) == sorted(norm)


def test_undo_after_reopen_has_nothing_to_undo(env):
    root, tmp = env
    ed, _raw, norm = _normalized_editor(root, tmp, comp=False)
    path = tmp / 'norm.flowsession'
    ed._write_session(str(path))
    ed2, _jobs = _restore_session(root, path)
    # The values from before the normalization lived only in memory: after
    # a reopen Undo changes nothing and says so (reload the FCS files to go
    # back to the raw values).
    assert ed2._restore_batch_norm() == 0
    assert 'reload' in ed2.status_var.get().lower()
    got = _reopened_values(ed2)
    for n in norm:
        np.testing.assert_allclose(got[n], norm[n], rtol=1e-5, atol=1e-5)
    # Undo in the session that normalized still works, and clears the
    # record: the next save no longer keeps a sidecar for that reason.
    assert ed._restore_batch_norm() == 4
    assert not any(getattr(ed._samples[n], 'batch_norm', None)
                   for n in ed._sample_order)
    data = ed._write_session(str(tmp / 'undone.flowsession'))
    assert not any(e.get('processed_data') or e.get('batch_norm')
                   for e in data['samples'])


def test_missing_sidecar_reopens_raw_and_says_so(env):
    """The saved values gone (the session's _data folder deleted): the raw
    FCS is all there is. The sample opens on it, without a record claiming
    a normalization, and the load notes say it was not restored."""
    import shutil
    root, tmp = env
    ed, raw, _norm = _normalized_editor(root, tmp, comp=False)
    path = tmp / 'norm.flowsession'
    ed._write_session(str(path))
    shutil.rmtree(tmp / 'norm_data')
    notes = []
    from tests.test_transform_records import _new_editor as real_new

    def new_editor(r):
        e = real_new(r)
        e._add_load_note = notes.append
        return e
    import tests.test_derived_kinetics as tdk
    old = tdk._new_editor
    tdk._new_editor = new_editor
    try:
        ed2, jobs = _restore_session(root, path)
    finally:
        tdk._new_editor = old
    assert len(jobs['fcs']) == 4 and not jobs['csv'], jobs
    got = _reopened_values(ed2)
    for n in raw:
        np.testing.assert_allclose(got[n], raw[n], rtol=1e-5, atol=1e-5)
        assert not getattr(ed2._samples[n], 'batch_norm', None)
    hit = [t for t in notes if 'batch normalization not restored' in t]
    assert sorted(t.split(':')[0] for t in hit) == sorted(raw)
