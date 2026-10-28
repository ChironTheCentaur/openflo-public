"""A 'Display + data' (propagate) downsample survives a session save/reopen.

Two losses, both measured on two synthetic samples of 1,500 and 600 events
trimmed together, so the larger one held 600:

* The trim was not recorded. The reopen took the floor again, over the
  samples present and ticked THEN, so once the smallest sample had been
  removed or unticked after the trim, a sample reloaded from its raw FCS came
  back with all 1,500 events where 600 were saved -- and every gate count and
  frequency on it changed. The session now records each trim (the length it
  was cut from, each floor and the seed) inside the sample's entry, and the
  reopen replays it: the same 600 events, not just 600 of them.
* A compensated sample is saved as a processed sidecar, which holds the
  trimmed frame; the full frame lived only in memory. Turning the trim off
  after a reopen left it at 600 events without a word while raw samples came
  back whole. It is now rebuilt from its FCS file; a sample carrying a column
  computed on the kept events only (clusters) cannot be, and the status line
  says which and why.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from tests.test_session_restore_transforms import _restore
from tests.test_transform_records import (
    _new_editor,
    capture_warnings,
    isolate_home,
    settle,
)

CH = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A', 'Time']
SPILL = '3,FL1-A,FL2-A,FL3-A,1,0.1,0.02,0.05,1,0.1,0.01,0.03,1'


def _write_fcs(path, seed, n, spill=False):
    import flowio
    rng = np.random.default_rng(seed)
    ev = np.column_stack([
        rng.lognormal(10, .2, n), rng.lognormal(9, .2, n),
        rng.choice([150., 20000.], n) + rng.normal(0, 40, n),
        rng.choice([150., 20000.], n) + rng.normal(0, 40, n),
        rng.normal(800, 100, n), np.sort(rng.uniform(0, 1000, n))])
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.astype(np.float32).flatten().tolist(), CH,
                          metadata_dict={'SPILL': SPILL} if spill else {})
    return str(path)


def _loaded(path):
    from openflo.pipeline import FlowSample
    s = FlowSample(path)            # what the loader does, QC off
    s.auto_compensate()
    s.apply_transform()
    return s


@pytest.fixture
def env(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)       # never a real (blocking) modal
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"Tk cannot initialise: {e}")
    root.withdraw()
    yield root, _new_editor(root), tmp_path
    root.destroy()


def _load(ed, tmp, sizes, spill=False):
    """Load one synthetic sample per (name, events), all ticked; returns
    each one's full frame."""
    full = {}
    for seed, (nm, n) in enumerate(sizes, start=1):
        s = _loaded(_write_fcs(tmp / f'{nm}.fcs', seed, n, spill=spill))
        ed._on_loaded(nm, s)
        ed._sample_plot_enabled[nm] = True
        full[nm] = s.data
    return full


def _trim_on(ed):
    ed.ds_propagate_var.set(True)
    ed._on_downsample_propagate_toggled()


def _trim_off(ed):
    ed.ds_propagate_var.set(False)
    ed._on_downsample_propagate_toggled()


def _reopen(root, path):
    ed2 = _restore(root, path, raw_first=True)
    settle(root, 1.5)       # the load settle and _apply_session_restore re-trim
    return ed2


def _same_rows(got, want):
    assert list(got.columns) == list(want.columns)
    assert len(got) == len(want)
    for c in want.columns:
        assert np.array_equal(got[c].to_numpy(), want[c].to_numpy(),
                              equal_nan=True), c


@pytest.mark.parametrize('how', ['removed', 'unticked'])
def test_a_trimmed_raw_sample_reopens_with_the_events_it_was_saved_with(
        env, how):
    """The trim's floor came from a sample that was then removed or
    unticked; the reopen took the floor again without it and gave the other
    sample all its 1,500 events back."""
    root, ed, tmp = env
    full = _load(ed, tmp, [('A', 1500), ('B', 600)])
    _trim_on(ed)
    if how == 'removed':
        ed._remove_samples(['B'])
    else:
        ed._sample_plot_enabled['B'] = False
    saved = ed._samples['A'].data.copy()
    assert len(saved) == 600
    path = tmp / 'trim.flowsession'
    data = ed._write_session(str(path))
    entry = next(e for e in data['samples'] if e['name'] == 'A')
    assert not entry.get('processed_data')        # reloads from its raw FCS
    assert entry['downsample_trim'] == {'from': 1500, 'steps': [600],
                                        'seed': 42}

    ed2 = _reopen(root, path)
    _same_rows(ed2._samples['A'].data, saved)

    # The full frame is kept as before the save: turning the trim off gives
    # back every event.
    _trim_off(ed2)
    _same_rows(ed2._samples['A'].data, full['A'])


def test_a_sample_trimmed_twice_reopens_with_the_same_events(env):
    """A later, smaller floor trims the trimmed frame again; one seeded cut
    of the full frame to the last floor picks other events. Both cuts are
    recorded and replayed in order."""
    root, ed, tmp = env
    _load(ed, tmp, [('A', 1500), ('B', 900)])
    _trim_on(ed)
    assert len(ed._samples['A'].data) == 900
    s = _loaded(_write_fcs(tmp / 'C.fcs', 3, 600))
    ed._on_loaded('C', s)
    ed._sample_plot_enabled['C'] = True
    ed._apply_propagate_downsample()
    saved = {nm: ed._samples[nm].data.copy() for nm in ('A', 'B')}
    assert len(saved['A']) == len(saved['B']) == 600
    ed._remove_samples(['C'])
    path = tmp / 'twice.flowsession'
    data = ed._write_session(str(path))
    trims = {e['name']: e.get('downsample_trim') for e in data['samples']}
    assert trims == {'A': {'from': 1500, 'steps': [900, 600], 'seed': 42},
                     'B': {'from': 900, 'steps': [600], 'seed': 42}}

    ed2 = _reopen(root, path)
    for nm in ('A', 'B'):
        _same_rows(ed2._samples[nm].data, saved[nm])


def test_a_sidecar_sample_gets_all_its_events_back_after_a_reopen(env):
    """A compensated sample is saved as a sidecar holding its trimmed
    frame; its full frame lived only in memory, so turning the trim off
    after a reopen left it at 600 events, silently. It is rebuilt from its
    FCS file: all 1,500 events, the kept 600 where the trim took them."""
    root, ed, tmp = env
    full = _load(ed, tmp, [('A', 1500), ('B', 600)], spill=True)
    _trim_on(ed)
    saved = ed._samples['A'].data.copy()
    path = tmp / 'side.flowsession'
    data = ed._write_session(str(path))
    entry = next(e for e in data['samples'] if e['name'] == 'A')
    assert entry.get('processed_data')            # reloads from its sidecar
    ed2 = _reopen(root, path)
    assert len(ed2._samples['A'].data) == 600

    _trim_off(ed2)
    got = ed2._samples['A'].data
    assert len(got) == 1500, ed2.status_var.get()
    assert list(got.columns) == list(full['A'].columns)
    for c in full['A'].columns:
        assert np.allclose(got[c].to_numpy(), full['A'][c].to_numpy(),
                           rtol=1e-6, atol=1e-9, equal_nan=True), c
    # The kept events sit where the trim took them from.
    pos = full['A'].sample(600, random_state=42).index.to_numpy()
    for c in saved.columns:
        assert np.allclose(got[c].to_numpy()[pos], saved[c].to_numpy(),
                           rtol=1e-6, atol=1e-9, equal_nan=True), c
    assert 'Restored 1 sample(s)' in ed2.status_var.get()   # B was the floor


def test_a_sidecar_sample_with_clusters_says_why_it_stays_trimmed(env):
    """Clusters computed on the kept events have no value for the others;
    such a sample cannot be rebuilt whole, and the status line now names it
    and the column instead of leaving it trimmed without a word."""
    root, ed, tmp = env
    _load(ed, tmp, [('A', 1500), ('B', 600)], spill=True)
    _trim_on(ed)
    a = ed._samples['A']
    a.data = a.data.assign(cluster=np.arange(len(a.data)) % 3)
    ed._data_changed('A')
    path = tmp / 'clus.flowsession'
    ed._write_session(str(path))
    ed2 = _reopen(root, path)

    _trim_off(ed2)
    assert len(ed2._samples['A'].data) == 600
    assert len(ed2._samples['B'].data) == 600
    msg = ed2.status_var.get()
    assert ('A stays trimmed to 600 events: cluster was computed on the '
            'kept events only') in msg, msg
