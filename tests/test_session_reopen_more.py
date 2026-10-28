"""Four suspected session save -> close -> reopen losses, each proven or
ruled out on a real save and reopen.

* A. Spectral unmixing kept its reference spectra, detectors and QC report
  only in memory. The U: abundance columns survived a reopen (in the
  processed sidecar), but the spectra did not: Tools > Panel check's "Use
  last unmix" was greyed out ("No unmix in this session yet") and the
  Spectral QC report and its unmixed-pairs plot had nothing to show. They
  now go into the session entry of the first unmixed sample, checked on
  restore against its U: columns. (Real; fixed.)
* B. The MESF calibration fit lives on the Calibration dialog, not the
  editor: closing the dialog drops it in a live session too, so a reopen
  loses nothing a live session still had. The MESF: column, its stamp and
  the fit's report (slope, model, R² in the audit trail and the Methods
  paragraph) all come back. (Ruled out.)
* C. A sample's ``cluster_settings`` is read only by the run that set it,
  straight after clustering, to put the resolution into the audit entry;
  that entry is saved. After a reopen the attribute is None, but the
  labels, their stamps, the audit and the Methods paragraph are the same.
  (Ruled out.)
* D. A sample whose file was missing at reopen and later relinked from a
  folder. A single relinked sample keeps a disambiguated name ('tube
  [Day 9]'), as the normal restore does; but two missing samples with the
  same file name ('tube.fcs' from Day 6 and Day 9) were both relinked to
  the FIRST 'tube.fcs' found in the folder, so one sample's data loaded
  under the other's name and gates, and the other's file was never used.
  A relink now prefers the file whose folders match the saved path's.
  (Real; fixed.)

Synthetic data and names only.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest

from tests.test_session_restore_transforms import _restore
from tests.test_transform_records import (
    _new_editor,
    _pump,
    capture_warnings,
    isolate_home,
    settle,
)

FL = [f'FL{i}-A' for i in range(1, 7)]
CH = ['FSC-A', 'SSC-A', *FL]


def _write(path, det, seed=0):
    """An FCS of FSC-A, SSC-A and the six detectors `det` (n x 6)."""
    import flowio
    rng = np.random.default_rng(seed)
    n = len(det)
    ev = np.column_stack([rng.lognormal(10, .2, n), rng.lognormal(9, .2, n),
                          det])
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.astype(np.float32).flatten().tolist(), CH)
    return str(path)


def _loaded(path):
    from openflo.pipeline import FlowSample
    s = FlowSample(path)            # what editor_loadpool._load_worker does
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


def _reopen(root, path):
    ed2 = _restore(root, path, raw_first=True)
    settle(root, 1.0)               # let the load settle / restore run
    return ed2


def _entry(data, name):
    return next(e for e in data['samples'] if e['name'] == name)


# ── A. Spectral unmixing: reference spectra, detectors and QC ──────────────

SPECTRA = np.array([
    [1.0, 0.6, 0.2, 0.05, 0.0, 0.0],
    [0.0, 0.2, 1.0, 0.6, 0.1, 0.0],
    [0.0, 0.0, 0.05, 0.3, 1.0, 0.5],
])
FLUORS = ['F1', 'F2', 'F3']
AF = 50.0


def _unmixed(ed, tmp, n=1500):
    """Three single stains, an unstained control and a mixed sample, loaded
    and unmixed in the editor (Tools > Spectral unmix, Apply)."""
    rng = np.random.default_rng(7)
    noise = 5.0
    det = {}
    for i, f in enumerate(FLUORS):
        amp = rng.uniform(500, 20000, (n, 1))
        det[f'ss{f}'] = amp * SPECTRA[i][None, :] + AF
    det['unstained'] = np.full((n, 6), AF)
    det['mix'] = rng.uniform(0, 20000, (n, 3)) @ SPECTRA + AF
    for k, (nm, d) in enumerate(det.items()):
        d = d + rng.normal(0, noise, d.shape)
        ed._on_loaded(nm, _loaded(_write(tmp / f'{nm}.fcs', d, seed=k)))
    ed._apply_spectral_unmix({f'ss{f}': f for f in FLUORS}, 'unstained',
                             list(FL), False)
    return ed._samples['mix']


def _same_qc(a, b):
    assert list(a) == list(b)
    for k in a:
        if isinstance(a[k], np.ndarray):
            assert b[k].dtype == a[k].dtype and b[k].shape == a[k].shape, k
            assert np.array_equal(a[k], b[k], equal_nan=True), k
        else:
            assert a[k] == b[k], k


def test_unmix_spectra_and_qc_come_back_on_reopen(env):
    """The reference spectra, their detectors and the QC report were memory
    only: after a reopen the U: columns were there but Panel check's "Use
    last unmix" was greyed out and Spectral QC said "Run Unmix first". They
    come back equal to the ones saved."""
    from openflo.ui_panel_check import PanelCheckWindow
    root, ed, tmp = env
    _unmixed(ed, tmp)
    spectra, fluors = ed._last_unmix_spectra
    qc, dets = ed._last_unmix_qc, ed._last_unmix_detectors
    pname, A, pfluors = ed._last_unmix_pairs
    assert fluors == [*FLUORS, 'Autofluorescence'] and dets == FL
    assert qc is not None
    live = PanelCheckWindow(ed)
    live.unmix_btn.invoke()
    path = tmp / 'unmix.flowsession'
    data = ed._write_session(str(path))
    assert _entry(data, pname).get('processed_data')

    ed2 = _reopen(root, path)
    assert all(f'U:{f}' in ed2._samples['mix'].data.columns for f in fluors)
    assert ed2._last_unmix_spectra is not None, "spectra lost on reopen"
    S2, f2 = ed2._last_unmix_spectra
    assert S2.dtype == spectra.dtype and np.array_equal(S2, spectra)
    assert f2 == fluors
    assert ed2._last_unmix_detectors == dets
    _same_qc(qc, ed2._last_unmix_qc)
    # Panel check offers the last unmix again and reports the same spectra.
    win = PanelCheckWindow(ed2)
    assert win.unmix_btn.instate(['!disabled'])
    win.unmix_btn.invoke()
    assert win.source_var.get() == 'Last unmix (this session)'
    assert win.report['detectors'] == live.report['detectors']
    assert np.array_equal(win.report['similarity'],
                          live.report['similarity'])
    # Spectral QC opens on the saved report (it said "Run Unmix first"),
    # from its own entry on the Tools menu (it had none: only an unmix or
    # Panel check opened it).
    from openflo.ui_spectral_qc import SpectralQCWindow
    from tests.test_gui_toolings import invoke_menu_entry
    invoke_menu_entry(ed2, 'Tools', 'Spectral QC…')
    assert any(isinstance(w, SpectralQCWindow) for w in ed2.winfo_children())
    # The unmixed-pairs plot gets the same sample's abundances back, from
    # its U: columns (stored transformed, so equal to rounding).
    n2, A2, pf2 = ed2._last_unmix_pairs
    assert (n2, pf2) == (pname, pfluors)
    assert A2.shape == A.shape
    np.testing.assert_allclose(A2, A, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('damage', [
    'other_version', 'not_a_record', 'wrong_shape', 'qc_for_other_fluors',
    'missing_column'])
def test_a_malformed_unmix_record_is_ignored_with_a_log_line(
        env, capsys, damage):
    """An old or damaged unmix record is left out with a log line: the
    sample and its U: columns restore as before, with no unmix to offer."""
    root, ed, tmp = env
    _unmixed(ed, tmp)
    pname = ed._last_unmix_pairs[0]
    path = tmp / 'bad.flowsession'
    ed._write_session(str(path))
    data = json.loads(path.read_text(encoding='utf-8'))
    e = _entry(data, pname)
    rec = e['unmix_model']
    if damage == 'other_version':
        rec['version'] = 99
    elif damage == 'not_a_record':
        e['unmix_model'] = 'F1 F2 F3'
    elif damage == 'wrong_shape':
        rec['model']['spectra']['$ndarray'] = [[1.0, 2.0]]
    elif damage == 'qc_for_other_fluors':
        rec['model']['qc']['fluors'] = ['F9']
    elif damage == 'missing_column':
        rec['model']['fluors'][0] = 'F9'
    path.write_text(json.dumps(data), encoding='utf-8')
    capsys.readouterr()
    ed2 = _reopen(root, path)
    assert all(f'U:{f}' in ed2._samples[pname].data.columns
               for f in FLUORS)
    assert ed2._last_unmix_spectra is None and ed2._last_unmix_qc is None
    assert getattr(ed2, '_last_unmix_pairs', None) is None
    assert 'unmix_model' in capsys.readouterr().out


def test_no_unmix_record_without_its_columns(env):
    """A save writes the unmix only while a sample still has its U:
    columns, so the spectra never outlive the abundances they made."""
    root, ed, tmp = env
    _unmixed(ed, tmp)
    fluors = ed._last_unmix_spectra[1]
    for s in ed._samples.values():
        s.data = s.data.drop(columns=[f'U:{f}' for f in fluors
                                      if f'U:{f}' in s.data.columns])
    data = ed._write_session(str(tmp / 'none.flowsession'))
    assert not any('unmix_model' in e for e in data['samples'])


# ── B. MESF calibration (ruled out) ────────────────────────────────────────

PEAKS = np.array([50, 300, 1500, 6000, 25000, 100000], float)


def _beads_and_cells(tmp):
    rng = np.random.default_rng(3)
    beads = np.concatenate([rng.normal(p, 0.03 * p, 800) for p in PEAKS])
    det_b = np.column_stack([beads] + [rng.normal(800, 50, beads.size)
                                       for _ in FL[1:]])
    det_c = np.column_stack([np.full(500, 2000.0)] + [
        rng.normal(800, 50, 500) for _ in FL[1:]])
    return (_write(tmp / 'beads.fcs', det_b, 1),
            _write(tmp / 'cells.fcs', det_c, 2))


def _audit_of(ed, action):
    return [e['details'] for e in ed._audit_log.to_list()
            if e['action'] == action]


def test_mesf_calibration_and_its_report_survive_reopen(env):
    """Suspected: the fit (the dialog's `_cal`) is not saved. It is not
    editor state: it lives on the Calibration dialog, so closing the dialog
    drops it in a live session too, and a dialog opened again there says
    "Fit a calibration first" just as one opened after a reopen does. What
    the fit made does come back exactly: the MESF: column, its stamp, and
    the fit's report (channel, slope, model, R²) in the audit trail and the
    Methods paragraph."""
    from openflo.provenance import methods_paragraph
    from openflo.ui_calibration import CalibrationDialog
    root, ed, tmp = env
    pb, pc = _beads_and_cells(tmp)
    ed._on_loaded('beads', _loaded(pb))
    ed._on_loaded('cells', _loaded(pc))
    dlg = CalibrationDialog(ed)
    dlg.model_var.set('linear')
    dlg.bead_var.set('beads')
    dlg._sync_channels()
    dlg.chan_var.set(ed._fmt_channel('FL1-A'))
    dlg._detect()
    shown = [ln.strip() for ln in
             dlg.txt.get('1.0', 'end').strip().splitlines()]
    assert len(shown) == 6
    dlg.txt.delete('1.0', 'end')
    dlg.txt.insert('1.0', '\n'.join(f"{s}\t{3 * p:g}"
                                    for s, p in zip(shown, PEAKS,
                                                    strict=True)))
    dlg._fit()
    dlg.apply_var.set(dlg.chan_var.get())
    dlg._apply()
    col = 'MESF:FL1-A'
    mesf = ed._samples['cells'].data[col].to_numpy(float)
    np.testing.assert_allclose(mesf, 6000.0, rtol=0.01)
    (report,) = _audit_of(ed, 'calibration')
    stamp = ed._samples['cells'].result_stamps[col]
    para = methods_paragraph(ed._audit_log,
                             samples=list(ed._samples.values()))
    dlg.destroy()
    # The live session has no fit once the dialog is closed.
    assert CalibrationDialog(ed)._cal is None
    path = tmp / 'cal.flowsession'
    data = ed._write_session(str(path))
    assert _entry(data, 'cells').get('processed_data')

    ed2 = _reopen(root, path)
    s2 = ed2._samples['cells']
    assert np.array_equal(s2.data[col].to_numpy(float), mesf)
    assert s2.result_stamps[col] == stamp
    assert _audit_of(ed2, 'calibration') == [report]
    assert report['channel'] == 'FL1-A' and report['model'] == 'linear'
    assert report['slope'] == pytest.approx(3.0, rel=0.01)
    assert methods_paragraph(ed2._audit_log,
                             samples=list(ed2._samples.values())) == para
    assert CalibrationDialog(ed2)._cal is None


def test_methods_paragraph_counts_each_sample_once_after_reopen(env):
    """Found while ruling out B and C: every load is audited, and a reopen
    loads each sample again into the same trail, so the Methods paragraph's
    event count doubled at every reopen (900 events said 1,800, then
    2,700). Each sample now counts once, at its last load."""
    from openflo.provenance import methods_paragraph
    root, ed, tmp = env
    ed._on_loaded('tube', _loaded(_write(
        tmp / 'tube.fcs', np.full((900, 6), 700.0), 5)))
    path = tmp / 'm.flowsession'
    ed._write_session(str(path))
    for _ in range(2):
        ed = _reopen(root, path)
        ed._write_session(str(path))
    loads = [e for e in ed._audit_log.to_list()
             if e['action'] == 'sample.load']
    assert len(loads) == 3
    para = methods_paragraph(ed._audit_log,
                             samples=list(ed._samples.values()))
    assert para.startswith(
        "Flow-cytometry data from 1 sample(s) (900 total events)"), para


# ── C. cluster_settings (ruled out) ────────────────────────────────────────

def test_cluster_settings_loss_changes_no_output(env):
    """Suspected: a sample's `cluster_settings` (method, k, resolution,
    route) is not saved. It is read only by the clustering run that set it,
    straight after the clustering, to put the applied resolution into the
    run's audit entry; that entry is saved. After a reopen the attribute is
    None, and the labels, their stamps, the audit entry and the Methods
    paragraph are all the same."""
    import contextlib
    import io

    from openflo.provenance import methods_paragraph
    root, ed, tmp = env
    rng = np.random.default_rng(11)
    n = 900
    centres = np.array([[300., 300.], [300., 20000.], [20000., 300.]])
    lab = rng.integers(0, 3, n)
    det = np.column_stack([centres[lab] + rng.normal(0, 60, (n, 2)),
                           rng.normal(800, 50, (n, 4))])
    ed._on_loaded('tube', _loaded(_write(tmp / 'tube.fcs', det, 4)))
    ed._set_active_sample('tube')
    with contextlib.redirect_stdout(io.StringIO()):
        ed._run_clustering(method='phenograph', all_samples=False, k=30,
                           grid=10, n_meta=10, embedding='none',
                           resolution=0.5, channels=['FL1-A', 'FL2-A'],
                           reproducible=True)
    s = ed._samples['tube']
    assert s.cluster_settings and s.cluster_settings['resolution'] == 0.5
    labels = s.data['cluster'].to_numpy()
    (entry,) = _audit_of(ed, 'cluster')
    assert (entry['method'], entry['k'], entry['resolution']) == (
        'phenograph', 30, 0.5)
    stamps = dict(s.result_stamps)
    para = methods_paragraph(ed._audit_log, samples=[s])
    path = tmp / 'cl.flowsession'
    ed._write_session(str(path))

    ed2 = _reopen(root, path)
    s2 = ed2._samples['tube']
    assert s2.cluster_settings is None
    assert np.array_equal(s2.data['cluster'].to_numpy(), labels)
    assert dict(s2.result_stamps) == stamps
    assert _audit_of(ed2, 'cluster') == [entry]
    assert methods_paragraph(ed2._audit_log, samples=[s2]) == para


# ── D. A missing sample relinked later ─────────────────────────────────────

def _reopen_missing(root, path, monkeypatch):
    """A fresh editor on `path`, its loads run here; the missing-file prompt
    is answered No and the test relinks itself."""
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


def _tubes(ed, tmp):
    """'tube.fcs' from Day 6 and Day 9 (different events), loaded as 'tube'
    and 'tube [Day 9]', each with a gate of its own, and another sample."""
    base = tmp / 'data'
    out = {}
    for day, seed in (('Day 6', 1), ('Day 9', 2)):
        (base / day).mkdir(parents=True)
        rng = np.random.default_rng(seed)
        det = rng.normal(500 * seed, 40, (400 + 100 * seed, 6))
        p = _write(base / day / 'tube.fcs', det, seed)
        nm = ed._sample_name_for(p)
        ed._on_loaded(nm, _loaded(p))
        ed._set_active_sample(nm)
        ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A', 'value': 0.5,
                      'parent_id': None, 'name': f'{day} pos',
                      'enabled': True})
        out[nm] = (day, p, len(ed._samples[nm].data))
    other = _write(base / 'other.fcs', np.full((300, 6), 700.0), 9)
    ed._on_loaded('other', _loaded(other))
    return out


def _gate_names(ed, name):
    return sorted(g.get('name') for g in ed._sample_gates.get(name,
                                                              {}).values())


def test_a_relinked_disambiguated_sample_keeps_its_name(env, monkeypatch):
    """A sample saved as 'tube [Day 9]' whose file had moved was relinked
    from a folder: it must come back under that name, with its gates,
    colour, plot toggle and FMO mapping (all keyed by the name)."""
    root, ed, tmp = env
    tubes = _tubes(ed, tmp)
    assert sorted(tubes) == ['tube', 'tube [Day 9]']
    ed._remove_samples(['tube'])
    n9 = 'tube [Day 9]'
    ed._sample_plot_enabled[n9] = True
    ed._sample_colors[n9] = '#123456'
    ed._set_fmo_controls({'FL1-A': n9})
    path = tmp / 'proj' / 'one.flowsession'
    path.parent.mkdir()
    ed._write_session(str(path))
    away = tmp / 'moved' / 'Day 9'
    away.mkdir(parents=True)
    shutil.move(tubes[n9][1], away / 'tube.fcs')

    ed2, asked = _reopen_missing(root, path, monkeypatch)
    assert asked and n9 not in ed2._samples
    assert ed2._relink_missing(str(tmp / 'moved')) == ([n9], [])
    assert _pump(root, lambda: n9 in ed2._samples, timeout=10), \
        sorted(ed2._samples)
    settle(root, 0.5)
    assert sorted(ed2._samples) == ['other', n9]
    assert len(ed2._samples[n9].data) == tubes[n9][2]
    assert _gate_names(ed2, n9) == ['Day 9 pos']
    assert ed2._sample_colors[n9] == '#123456'
    assert ed2._fmo_controls == {'FL1-A': n9}
    data = ed2._write_session(str(tmp / 'again.flowsession'))
    assert sorted(e['name'] for e in data['samples']) == ['other', n9]


def test_relinking_two_samples_of_one_file_name_keeps_each_its_file(
        env, monkeypatch):
    """Both 'tube.fcs' (Day 6 and Day 9) moved, and were relinked from the
    folder they moved to. Each name was matched by file name only, so both
    took the first 'tube.fcs' found: one sample's events loaded under the
    other's name and gates, and the other file was never used. Each now
    takes the file whose folder matches its saved path's."""
    root, ed, tmp = env
    tubes = _tubes(ed, tmp)
    path = tmp / 'proj' / 'two.flowsession'
    path.parent.mkdir()
    ed._write_session(str(path))
    moved = tmp / 'moved'
    for day, p, _n in tubes.values():
        (moved / day).mkdir(parents=True)
        shutil.move(p, moved / day / 'tube.fcs')

    ed2, asked = _reopen_missing(root, path, monkeypatch)
    assert asked and sorted(ed2._samples) == ['other']
    found, still = ed2._relink_missing(str(moved))
    assert (sorted(found), still) == (sorted(tubes), [])
    assert _pump(root, lambda: all(nm in ed2._samples for nm in tubes),
                 timeout=10), sorted(ed2._samples)
    settle(root, 0.5)
    for nm, (day, _p, n_ev) in tubes.items():
        s = ed2._samples[nm]
        assert os.path.samefile(s.path, moved / day / 'tube.fcs'), nm
        assert len(s.data) == n_ev, nm
        assert _gate_names(ed2, nm) == [f'{day} pos'], nm
    data = ed2._write_session(str(path))
    for nm, (day, _p, _n) in tubes.items():
        assert os.path.samefile(_entry(data, nm)['path'],
                                moved / day / 'tube.fcs')
