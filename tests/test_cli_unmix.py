"""End-to-end test for the spectral batch-unmixing CLI mode
(`openflo --unmix`). Writes synthetic single-stain + mixed FCS via FlowIO,
runs run_batch_unmix, and checks the unmixed CSVs and QC report."""
from __future__ import annotations

import json
import os
import types

import numpy as np

from openflo.cli import run_batch_unmix

DETECTORS = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']
CHANNELS = ['FSC-A', 'SSC-A', *DETECTORS]
# 3 fluor spectra over 6 detectors (distinct shapes).
SPECTRA = np.array([
    [1.0, 0.7, 0.3, 0.1, 0.0, 0.0],
    [0.0, 0.2, 0.8, 1.0, 0.4, 0.1],
    [0.0, 0.0, 0.1, 0.3, 0.8, 1.0],
])


def _write_fcs(path, events):
    import flowio
    flat = np.asarray(events, dtype=np.float32).flatten().tolist()
    with open(path, 'wb') as f:
        flowio.create_fcs(f, flat, CHANNELS)


def _scatter(rng, n):
    return np.column_stack([rng.lognormal(10, 0.3, n),
                            rng.lognormal(9, 0.4, n)])


def _single_stain(rng, fluor_idx, n=4000):
    amp = rng.uniform(50, 800, (n, 1))
    det = amp * SPECTRA[fluor_idx][None, :] + rng.normal(0, 1.0, (n, 6))
    return np.hstack([_scatter(rng, n), det])


def _mixed(rng, n=3000):
    A = rng.uniform(0, 500, (n, 3))
    det = A @ SPECTRA + rng.normal(0, 1.0, (n, 6))
    return np.hstack([_scatter(rng, n), det]), A


def _make_dataset(tmp):
    rng = np.random.default_rng(0)
    paths = {}
    for i, fl in enumerate(['F1', 'F2', 'F3']):
        p = os.path.join(tmp, f'ss_{fl}.fcs')
        _write_fcs(p, _single_stain(rng, i))
        paths[fl] = p
    # Faint unstained.
    p_un = os.path.join(tmp, 'unstained.fcs')
    _write_fcs(p_un, np.hstack([_scatter(rng, 2000),
                                rng.normal(0, 1.0, (2000, 6)) + 2.0]))
    paths['unstained'] = p_un
    # One mixed sample to unmix.
    mixed, A_true = _mixed(rng)
    p_mix = os.path.join(tmp, 'mixed.fcs')
    _write_fcs(p_mix, mixed)
    return paths, p_mix, A_true


def _args(controls, in_dir, out_dir, **kw):
    base = dict(unmix=True, out=out_dir,
                unmix_controls=json.dumps(controls),
                unmix_input=in_dir, unmix_detectors='auto',
                unmix_nonneg=False, fcs='', trials='')
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_batch_unmix_writes_csv_and_qc(tmp_path):
    tmp = str(tmp_path)
    paths, p_mix, A_true = _make_dataset(tmp)
    out_dir = os.path.join(tmp, 'out')
    # Only the mixed file should be unmixed (controls live in the same dir but
    # are excluded by path).
    rc = run_batch_unmix(_args(paths, tmp, out_dir))
    assert rc == 0

    # QC artifacts.
    assert os.path.isfile(os.path.join(out_dir, 'spectral_qc.md'))
    assert os.path.isfile(os.path.join(out_dir, 'reference_spectra.png'))
    qc_json = os.path.join(out_dir, 'spectral_qc.json')
    assert os.path.isfile(qc_json)
    with open(qc_json) as f:
        qc = json.load(f)
    assert qc['format'] == 'openflo-spectral-qc'
    # F1..F3 (+ Autofluorescence from the unstained control).
    assert qc['fluors'][:3] == ['F1', 'F2', 'F3']
    assert len(qc['detectors']) == 6

    # Unmixed CSV for the mixed sample.
    csv = os.path.join(out_dir, 'mixed_unmixed.csv')
    assert os.path.isfile(csv)
    import pandas as pd
    df = pd.read_csv(csv)
    for f in ('U:F1', 'U:F2', 'U:F3'):
        assert f in df.columns
    assert len(df) == len(A_true)
    # Recovered abundances correlate strongly with the truth (per fluor).
    for j, f in enumerate(['U:F1', 'U:F2', 'U:F3']):
        r = np.corrcoef(df[f].to_numpy(), A_true[:, j])[0, 1]
        assert r > 0.9, (f, r)


def test_batch_unmix_missing_controls_errors(tmp_path):
    out_dir = os.path.join(str(tmp_path), 'out')
    rc = run_batch_unmix(_args({}, str(tmp_path), out_dir,
                               unmix_controls=''))
    assert rc == 2


def test_batch_unmix_spectra_only_without_input(tmp_path):
    tmp = str(tmp_path)
    paths, _p_mix, _A = _make_dataset(tmp)
    out_dir = os.path.join(tmp, 'out')
    # No --unmix-input dir with samples: point at an empty subdir → spectra +
    # QC are still produced, return 0.
    empty = os.path.join(tmp, 'empty')
    os.makedirs(empty, exist_ok=True)
    rc = run_batch_unmix(_args(paths, empty, out_dir))
    assert rc == 0
    assert os.path.isfile(os.path.join(out_dir, 'spectral_qc.json'))
