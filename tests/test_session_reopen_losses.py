"""What a saved session lost on reopen, and must now keep.

* A compensated sample reopens from its processed-data sidecar, a CSV of
  events. It came back with no FCS keywords (metadata {}), no QC count and no
  linear $PnR ranges, so "Apply to: this batch" (instrument / date /
  voltages), CytoNorm's keyword batches, kinetics' $TIMESTEP, the .wsp QC
  note and the FCS export's ranges all worked from nothing.
* The channel labels the session saved were replaced by the file's own $PnS
  whenever a sample reloaded from its raw FCS was the first to land.
* A raw-FCS sample saved under a disambiguated name ('tube [Day 9]') came
  back as 'tube', so everything keyed by its saved name (FMO mapping,
  colour, plot toggle, active sample) missed it. Its colour was not read
  back at all.

Synthetic data and names only.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from tests._fcs_writer import write_fcs as write_int_fcs
from tests.test_transform_records import (
    _new_editor,
    _pump,
    capture_warnings,
    isolate_home,
    settle,
)

CH = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A', 'Time']
SPILL = '3,FL1-A,FL2-A,FL3-A,1,0.1,0.02,0.05,1,0.1,0.01,0.03,1'


def _events(seed, n=800):
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.lognormal(10, .2, n), rng.lognormal(9, .2, n),
        rng.choice([150., 20000.], n) + rng.normal(0, 40, n),
        rng.choice([150., 20000.], n) + rng.normal(0, 40, n),
        rng.normal(800, 100, n), np.sort(rng.uniform(0, 1000, n))])


def _fcs(path, seed, meta=None, labels=None, events=None):
    import flowio
    ev = _events(seed) if events is None else events
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.astype(np.float32).flatten().tolist(), CH,
                          opt_channel_names=labels, metadata_dict=meta or {})
    return str(path)


def _loaded(path, qc=False):
    from openflo.pipeline import FlowSample
    s = FlowSample(path)            # what editor_loadpool._load_worker does
    if qc:
        s.run_qc()
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


def _reopen(root, session_path, raw_first=True):
    """Reopen `session_path` in a fresh editor, running the restore's own
    loaders on this thread (raw-FCS samples first or last, each landing
    before the next starts), then let the deferred restore run."""
    ed2 = _new_editor(root)
    jobs = {'fcs': [], 'csv': []}
    ed2._queue_fcs_loads = lambda paths, front_names=(): jobs['fcs'].extend(
        paths)
    ed2._queue_processed_loads = lambda items, front_names=(): jobs[
        'csv'].extend(items)
    ed2._load_session_path(str(session_path))
    for kind in (['fcs', 'csv'] if raw_first else ['csv', 'fcs']):
        for job in jobs[kind]:
            if kind == 'fcs':
                ed2._load_worker(ed2._sample_name_for(job), job)
            else:
                ed2._load_csv_worker(job[0], job[1],
                                     dict(ed2._channel_labels))
            root.update()
    want = len(jobs['fcs']) + len(jobs['csv'])
    assert _pump(root, lambda: len(ed2._samples) == want), \
        ed2.status_var.get()
    settle(root, 1.5)               # _apply_session_restore, load settle
    return ed2, jobs


# ── 1. FCS keywords, QC count and linear ranges of a sidecar sample ────────

KW = {'SPILL': SPILL, 'CYT': 'SynthCyt', 'CYTSN': 'X1',
      'DATE': '01-JAN-2026', 'TIMESTEP': '0.01', 'P3V': '500', 'P4V': '510',
      'P5V': '520', 'BATCHKEY': 'b1'}


def test_fcs_keywords_survive_reopen_from_the_sidecar(env):
    """A compensated sample reopens from its sidecar CSV, which has no FCS
    keywords: metadata came back {}, so "Apply to: this batch" grouped
    nothing, CytoNorm's keyword batch read None and $TIMESTEP was lost. They
    are read again from the sample's FCS (TEXT segment only)."""
    from openflo.acquisition import batch_of
    from openflo.batch_normalize import keyword_value
    from openflo.kinetics import seconds_per_tick
    from openflo.pipeline import FlowSample
    root, ed, tmp = env
    paths = {nm: _fcs(tmp / f'{nm}.fcs', sd, meta=KW)
             for nm, sd in (('A', 1), ('B', 2))}
    for nm, p in paths.items():
        ed._on_loaded(nm, _loaded(p))
    path = tmp / 'kw.flowsession'
    data = ed._write_session(str(path))
    assert all(e.get('processed_data') for e in data['samples'])

    ed2, jobs = _reopen(root, path)
    assert [nm for nm, _ in jobs['csv']] == ['A', 'B'] and not jobs['fcs']
    for nm, p in paths.items():
        assert ed2._samples[nm].metadata == FlowSample(p).metadata
    a = ed2._samples['A']
    assert a.metadata['cyt'] == 'SynthCyt'
    chans = ['FL1-A', 'FL2-A', 'FL3-A']
    meta = {n: s.metadata for n, s in ed2._samples.items()}
    assert batch_of('A', meta, chans)[0] == ['A', 'B']
    assert keyword_value(a.metadata, 'BATCHKEY') == 'b1'
    assert seconds_per_tick(a) == 0.01


def test_qc_removed_count_survives_reopen_from_the_sidecar(env):
    """The sidecar holds only the events QC kept, so the count QC removed
    cannot be recomputed from it; it came back 0. It is saved in the
    sample's entry and put back."""
    root, ed, tmp = env
    ev = _events(5, n=2000)
    ev[:100, 2] = 262143.0                      # off-scale pile-up on FL1-A
    s = _loaded(_fcs(tmp / 'qc.fcs', 5, meta={'SPILL': SPILL}, events=ev),
                qc=True)
    assert s.qc_removed > 0
    ed._on_loaded('qc', s)
    path = tmp / 'qc.flowsession'
    data = ed._write_session(str(path))
    assert data['samples'][0]['qc_removed'] == s.qc_removed

    ed2, jobs = _reopen(root, path)
    assert jobs['csv'] and not jobs['fcs']
    s2 = ed2._samples['qc']
    assert len(s2.data) == len(s.data)
    assert s2.qc_removed == s.qc_removed
    assert s2.qc_applied is True
    # ... and the next save writes it again.
    again = ed2._write_session(str(tmp / 'qc2.flowsession'))
    assert again['samples'][0]['qc_removed'] == s.qc_removed


def test_linear_ranges_survive_reopen_from_the_sidecar(env):
    """An integer FCS's log-amplified ($PnE) and gained ($PnG) channels
    record their linear range, which the FCS export writes as $PnR. A
    sidecar sample came back with none."""
    root, ed, tmp = env
    rng = np.random.default_rng(3)
    n = 600
    raw = np.column_stack([rng.integers(100, 900, n) for _ in CH])
    p = str(write_int_fcs(
        tmp / 'int.fcs', raw, CH,
        pne=['0,0', '0,0', '4,1', '0,0', '0,0', '0,0'],
        png=[1, 1, 1, 2, 1, 1], extra={'$SPILL': SPILL}))
    s = _loaded(p)
    assert s.linear_ranges == {'FL1-A': 10000.0, 'FL2-A': 512.0}
    ed._on_loaded('int', s)
    path = tmp / 'int.flowsession'
    assert ed._write_session(str(path))['samples'][0].get('processed_data')

    ed2, jobs = _reopen(root, path)
    assert jobs['csv'] and not jobs['fcs']
    s2 = ed2._samples['int']
    assert s2.linear_ranges == {'FL1-A': 10000.0, 'FL2-A': 512.0}
    assert s2.fcs_decoding == {'FL1-A': 'log, 4 decades', 'FL2-A': 'gain 2'}


# ── 2. Saved channel labels vs the first raw-FCS sample to land ────────────

def test_saved_channel_labels_win_over_the_first_raw_sample(env):
    """A label the session saved (a template's, an edit) was replaced by the
    file's $PnS when a sample reloaded from its raw FCS landed first. Saved
    labels win; the file only fills in channels the session left
    unlabelled."""
    root, ed, tmp = env
    p = _fcs(tmp / 'A.fcs', 1,
             labels=['', '', 'MarkA', 'MarkB', '', ''])
    ed._on_loaded('A', _loaded(p))
    ed._channel_labels.update({'FL1-A': 'LabelX'})
    path = tmp / 'lbl.flowsession'
    data = ed._write_session(str(path))
    assert not data['samples'][0].get('processed_data')   # raw FCS reload
    assert data['channel_labels']['FL1-A'] == 'LabelX'
    # A channel the session has no label for comes from the file.
    data['channel_labels'].pop('FL2-A')
    import json
    path.write_text(json.dumps(data), encoding='utf-8')

    ed2, jobs = _reopen(root, path)
    assert jobs['fcs'] and not jobs['csv']
    assert ed2._channel_labels['FL1-A'] == 'LabelX'
    assert ed2._channel_labels['FL2-A'] == 'MarkB'
    assert ed2._channel_labels['FSC-A'] == 'FSC-A'
    data2 = ed2._write_session(str(tmp / 'lbl2.flowsession'))
    assert data2['channel_labels']['FL1-A'] == 'LabelX'


def test_an_ordinary_first_load_still_takes_the_files_labels(env):
    """Outside a session restore the first sample to land still sets the
    labels from its file, replacing what an earlier, removed sample left."""
    root, ed, tmp = env
    ed._on_loaded('old', _loaded(_fcs(tmp / 'old.fcs', 1,
                                      labels=['', '', 'OldA', '', '', ''])))
    ed._channel_labels['FL2-A'] = 'Stale'
    ed._remove_samples(['old'])
    ed._on_loaded('new', _loaded(_fcs(tmp / 'new.fcs', 2,
                                      labels=['', '', 'NewA', '', '', ''])))
    assert ed._channel_labels['FL1-A'] == 'NewA'
    assert ed._channel_labels['FL2-A'] == 'FL2-A'


# ── 3. A disambiguated name and the state keyed by it ──────────────────────

def _two_tubes(ed, tmp):
    (tmp / 'Day 6').mkdir()
    (tmp / 'Day 9').mkdir()
    p6 = _fcs(tmp / 'Day 6' / 'tube.fcs', 1)
    p9 = _fcs(tmp / 'Day 9' / 'tube.fcs', 2)
    n6 = ed._sample_name_for(p6)
    ed._on_loaded(n6, _loaded(p6))
    n9 = ed._sample_name_for(p9)
    ed._on_loaded(n9, _loaded(p9))
    return n6, n9


def test_disambiguated_raw_sample_name_survives_reopen(env):
    """Two 'tube.fcs' from different days load as 'tube' and 'tube [Day 9]'.
    With only the Day 9 one saved, the reopen named it 'tube' again, and its
    FMO mapping, plot toggle and active-sample slot, saved under
    'tube [Day 9]', pointed at nothing."""
    root, ed, tmp = env
    n6, n9 = _two_tubes(ed, tmp)
    assert (n6, n9) == ('tube', 'tube [Day 9]')
    ed._remove_samples([n6])
    ed._sample_plot_enabled[n9] = True
    ed._set_fmo_controls({'FL1-A': n9})
    ed._set_active_sample(n9)
    path = tmp / 'nm.flowsession'
    ed._write_session(str(path))

    ed2, jobs = _reopen(root, path)
    assert jobs['fcs'] and not jobs['csv']
    assert list(ed2._samples) == ['tube [Day 9]']
    assert ed2._fmo_controls == {'FL1-A': 'tube [Day 9]'}
    assert ed2._sample_plot_enabled['tube [Day 9]'] is True
    assert ed2._active_sample == 'tube [Day 9]'
    assert ed2._sample_name_for(jobs['fcs'][0]) == 'tube [Day 9]'


def test_sample_colour_survives_reopen(env):
    """A sample's overlay colour was saved but never read back. A colour
    set (or assigned on display) comes back; a sample never displayed,
    saved with the placeholder colour, still gets its colour lazily."""
    root, ed, tmp = env
    n6, n9 = _two_tubes(ed, tmp)
    ed._sample_plot_enabled[n9] = True
    ed._sample_colors[n9] = '#123456'
    ed._sample_plot_enabled[n6] = False
    ed._sample_colors.pop(n6, None)
    path = tmp / 'col.flowsession'
    data = ed._write_session(str(path))
    assert {e['name']: e['color'] for e in data['samples']} == {
        'tube': '#1f77b4', 'tube [Day 9]': '#123456'}

    ed2, _jobs = _reopen(root, path)
    assert sorted(ed2._samples) == ['tube', 'tube [Day 9]']
    assert ed2._sample_colors['tube [Day 9]'] == '#123456'
    assert ed2._sample_colors.get('tube') != '#1f77b4'
