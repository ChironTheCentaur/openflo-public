"""A restored session gives every sample back its own data and transforms.

Two holes, both measured on samples without $SPILL (uncompensated samples get
no processed sidecar unless they carry a computed column):

* U: abundances were not a "computed column", so a GUI-unmixed uncompensated
  sample reloaded from its raw FCS and its U: columns were gone.
* The editor's per-channel transforms were not saved. A sample reloaded from
  its raw FCS comes back on the loader's logicle; when it landed first it
  seeded the editor, every sidecar sample was conformed to logicle, and gates
  drawn on an asinh channel (stored in asinh coordinates) then selected 0
  events where 380 and 390 were right.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from tests.test_transform_records import (
    _new_editor,
    _pump,
    capture_warnings,
    isolate_home,
)

CH = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A']
MK = ['', '', 'CD901b', 'CD902', 'CD903']


def _fcs_without_spill(path, seed):
    import flowio
    rng = np.random.default_rng(seed)
    n = 800
    ev = np.column_stack([rng.lognormal(10, .2, n), rng.lognormal(9, .2, n),
                          rng.choice([150., 20000.], n) + rng.normal(0, 40, n),
                          rng.choice([150., 20000.], n) + rng.normal(0, 40, n),
                          rng.normal(800, 100, n)])
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.astype(np.float32).flatten().tolist(), CH,
                          opt_channel_names=MK)
    return str(path)


def _loaded(path):
    from openflo.pipeline import FlowSample
    s = FlowSample(path)            # what editor_loadpool._load_worker does
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


def _restore(root, session_path, raw_first):
    """Reopen `session_path` in a fresh editor, running the restore's own
    loaders on this thread, raw-FCS samples first or last."""
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
                ed2._load_csv_worker(job[0], job[1], {})
            root.update()               # land it before the next one starts
    want = len(jobs['fcs']) + len(jobs['csv'])
    assert _pump(root, lambda: len(ed2._samples) == want), \
        ed2.status_var.get()
    return ed2


def test_gui_unmixed_columns_survive_a_session_round_trip(env):
    """An uncompensated sample's U: abundances are analysis results; the
    session must keep them, record and all."""
    from openflo.pipeline import linear_values, transform_spec, transform_values
    root, ed, tmp = env
    s = _loaded(_fcs_without_spill(tmp / 'spectral.fcs', 1))
    assert not s.comp_channels
    ed._on_loaded('spectral', s)
    abund = np.linspace(-50.0, 5000.0, len(s.data))
    s.data['U:F1'] = transform_values(abund, method='logicle')
    s.data_transforms = {**s.data_transforms,
                         'U:F1': transform_spec('logicle')}
    path = tmp / 'unmixed.flowsession'
    data = ed._write_session(str(path))
    assert data['samples'][0].get('processed_csv'), data['samples'][0]
    ed2 = _restore(root, path, raw_first=True)
    s2 = ed2._samples['spectral']
    assert 'U:F1' in s2.data.columns
    np.testing.assert_allclose(linear_values(s2, 'U:F1'), abund,
                               rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('raw_first', [False, True])
def test_restored_gates_select_the_same_events_in_either_load_order(
        env, raw_first):
    from openflo.pipeline import linear_values
    root, ed, tmp = env
    for nm, seed in (('A', 1), ('B', 2)):
        s = _loaded(_fcs_without_spill(tmp / f'{nm}.fcs', seed))
        if nm == 'A':
            s.data['cluster'] = np.arange(len(s.data)) % 3   # gets a sidecar
        ed._on_loaded(nm, s)
    ed._apply_channel_transforms({'FL1-A': 'asinh'})
    cut = float(np.arcsinh(2000 / 150))
    for nm in ('A', 'B'):
        ed._set_active_sample(nm)
        ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A', 'value': cut,
                      'op': '>=', 'parent_id': None, 'name': 'CD901b+',
                      'enabled': True})
    before, _ = ed._collect_stats_rows({'Count'})
    counts = {r['Sample']: r['Count'] for r in before}
    assert min(counts.values()) > 300                  # the gate selects
    path = tmp / 'asinh.flowsession'
    data = ed._write_session(str(path))
    assert [bool(e.get('processed_csv')) for e in data['samples']] == [
        True, False]                                   # A sidecar, B raw FCS
    ed2 = _restore(root, path, raw_first)
    after, _ = ed2._collect_stats_rows({'Count'})
    assert {r['Sample']: r['Count'] for r in after} == counts
    assert ed2._channel_transform['FL1-A'] == 'asinh'
    for nm in ('A', 'B'):
        s2 = ed2._samples[nm]
        assert s2.data_transforms['FL1-A']['method'] == 'asinh', nm
        np.testing.assert_allclose(
            linear_values(s2, 'FL1-A'),
            linear_values(ed._samples[nm], 'FL1-A'), rtol=1e-5, atol=1e-3)
    assert os.path.isfile(path)


@pytest.mark.parametrize('how', ['save_as', 'autosave'])
def test_a_sample_restored_from_its_sidecar_survives_the_next_save(
        env, how, monkeypatch):
    """A compensated sample's only reason for a sidecar was its
    compensation. Restored, it is built FROM that CSV (no comp_channels, no
    computed column), so the next save -- Save As under a new name, or the
    autosave -- wrote it no sidecar, and an entry whose path is the OLD CSV,
    which the next restore read as an FCS: the sample was lost (2.6.1 too).
    """
    from pathlib import Path

    from openflo import editor_session
    from openflo.pipeline import linear_values
    from tests.test_editor_tools import _fcs as _fcs_with_spill
    root, ed, tmp = env
    monkeypatch.setattr(editor_session.messagebox, 'showerror',
                        lambda *a, **k: pytest.fail(f'save failed: {a}'))
    ed._on_loaded('A', _loaded(_fcs_with_spill(tmp / 'A.fcs', 1)))
    assert ed._samples['A'].comp_channels                 # compensated
    first = tmp / 'one' / 's1.flowsession'
    first.parent.mkdir()
    data = ed._write_session(str(first))
    assert data['samples'][0].get('processed_csv')
    ed2 = _restore(root, first, raw_first=True)
    assert ed2._samples['A'].path.endswith('.csv')        # built from the CSV
    if how == 'save_as':
        again = tmp / 'two' / 's2.flowsession'
        again.parent.mkdir()
        monkeypatch.setattr(editor_session.filedialog, 'asksaveasfilename',
                            lambda **k: str(again))
        ed2._save_session()
    else:
        ed2._primary = True
        ed2._periodic_autosave()
        again = Path(ed2._session_autosave_path())
    assert again.is_file()
    ed3 = _restore(root, again, raw_first=True)
    np.testing.assert_allclose(linear_values(ed3._samples['A'], 'FL1-A'),
                               linear_values(ed._samples['A'], 'FL1-A'),
                               rtol=1e-6, atol=1e-6)


def test_a_restored_sample_lists_the_markers_the_live_one_did(env):
    """A sample rebuilt from its sidecar CSV (FlowSample.from_dataframe)
    called every column that is not scatter or an analysis name a
    fluorescence marker: a MESF: calibration column and a '_pos' flag became
    markers on restore, which the live sample never had, and the clustering
    dialog selects every marker by default. MESF:CD901b (linear, up to
    ~60,000) beside logicle detectors (0-1) is all a kNN distance sees."""
    from openflo.pipeline import linear_values
    root, ed, tmp = env
    s = _loaded(_fcs_without_spill(tmp / 'm.fcs', 4))
    ed._on_loaded('m', s)
    live = list(s.fluor_channels)
    mesf = 3.0 * linear_values(s, 'FL1-A')
    s.data['MESF:CD901b'] = mesf
    s.data['FL1-A_pos'] = s.data['FL1-A'] > 0.5
    data = ed._write_session(str(tmp / 'mesf.flowsession'))
    assert data['samples'][0].get('processed_csv')
    ed2 = _restore(root, tmp / 'mesf.flowsession', raw_first=True)
    s2 = ed2._samples['m']
    assert s2.fluor_channels == live == ['FL1-A', 'FL2-A', 'FL3-A']
    np.testing.assert_allclose(s2.data['MESF:CD901b'], mesf, rtol=1e-6)
    assert 'MESF:CD901b' not in s2.data_transforms        # linear, as written
