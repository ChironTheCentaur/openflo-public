"""Double-check the synthetic example dataset (openflo.synthetic): the
differentiation model has the intended biology and loads through FlowSample,
and the spectral controls unmix correctly. This both validates the generator
and serves as a worked example of the data the rest of the suite models."""
from __future__ import annotations

import json
import os
import types

import numpy as np
import pytest

from openflo.pipeline import FlowSample
from openflo.spectral import (
    apply_unmixing,
    build_reference_spectra,
    unmixing_qc,
)
from openflo.synthetic import (
    DIFF_LABELS,
    PBMC_MARKERS,
    SIZE_BEAD_FSC_PER_UM,
    SIZE_BEAD_UM,
    SPECTRAL_DETECTORS,
    cell_cycle_sample,
    differentiation_sample,
    immunophenotyping_sample,
    make_compensation_controls,
    make_dataset,
    make_size_beads,
    make_spectral_dataset,
    size_bead_sample,
)


def _cells(df):
    """The singlet-cell core (exclude debris low-FSC and high-FSC doublets)."""
    return df[(df['FSC-A'] > 2e4) & (df['FSC-A'] < 9e4)]


# ── differentiation biology ───────────────────────────────────────────────────

def test_diff_cd34_falls_cd11b_rises_over_time():
    early = _cells(differentiation_sample(3, 'Ctrl', n=8000, seed=1))
    late = _cells(differentiation_sample(15, 'Ctrl', n=8000, seed=2))
    # CD34 (APC-A) progenitor marker falls; CD11b (BV421-A) myeloid marker rises.
    assert early['APC-A'].median() > late['APC-A'].median()
    assert early['BV421-A'].median() < late['BV421-A'].median()


def test_diff_stim_boosts_cd11b():
    stim = _cells(differentiation_sample(9, 'Stim', n=8000, seed=3))
    ctrl = _cells(differentiation_sample(9, 'Ctrl', n=8000, seed=4))
    # Stim accelerates maturation → more CD11b at the same day.
    assert stim['BV421-A'].median() > ctrl['BV421-A'].median()


def test_diff_has_debris_and_doublets():
    df = differentiation_sample(9, 'Stim', n=10000, seed=5)
    debris = df[df['FSC-A'] < 2e4]
    doublets = df[(df['FSC-A'] > 9e4) & (df['FSC-A'] / df['FSC-H'] > 2.6)]
    assert 0.05 < len(debris) / len(df) < 0.20        # ~10 % debris
    assert 0.03 < len(doublets) / len(df) < 0.12       # ~7 % doublets


# ── loads through FlowSample ──────────────────────────────────────────────────

def test_make_dataset_loads_with_labels(tmp_path):
    info = make_dataset(out_dir=str(tmp_path), n=3000, spectral_n=1500,
                        days=(3, 9), conditions=('Stim', 'Ctrl'), reps=1)
    assert info['differentiation_files'] == 4          # 2 days × 2 conds × 1
    # Folder structure groups by day.
    d3 = tmp_path / 'diff' / 'Day 3' / 'Stim_m1.fcs'
    assert d3.is_file()

    s = FlowSample(str(d3))
    s.run_qc()
    assert {'FSC-A', 'BV421-A', 'PE-Cy7-A', 'APC-A'} <= set(s.channel_names)
    # The $PnS antibody labels ride along.
    assert s.channel_labels.get('BV421-A') == 'CD11b'
    assert s.channel_labels.get('APC-A') == 'CD34'
    # Panel spreadsheet + README + spectral controls were written too.
    assert (tmp_path / 'staining panel.xlsx').is_file()
    assert (tmp_path / 'README.txt').is_file()
    assert (tmp_path / 'spectral' / 'controls.json').is_file()


# ── spectral controls unmix correctly ─────────────────────────────────────────

def test_spectral_controls_unmix(tmp_path):
    paths, controls_json = make_spectral_dataset(str(tmp_path), n=4000, seed=7)
    with open(controls_json) as f:
        controls = json.load(f)
    fluors = [k for k in controls if k != 'unstained']

    stains = {}
    for fl in fluors:
        s = FlowSample(controls[fl]); s.run_qc()
        stains[fl] = s.raw[SPECTRAL_DETECTORS].to_numpy(dtype=float)
    us = FlowSample(controls['unstained']); us.run_qc()
    un = us.raw[SPECTRAL_DETECTORS].to_numpy(dtype=float)

    spectra, fl_names = build_reference_spectra(stains, unstained=un)
    # QC: the three signatures are well separated (low pairwise similarity).
    qc = unmixing_qc(dict(stains, Autofluorescence=un), spectra, fl_names,
                     sim_threshold=0.98)
    assert qc['similar_pairs'] == []                  # distinct spectra
    assert np.isfinite(qc['condition_number'])

    # Unmixing the mixed sample recovers per-fluor abundances.
    mixed_path = os.path.join(str(tmp_path), 'mixed_sample.fcs')
    m = FlowSample(mixed_path); m.run_qc()
    view = type('V', (), {'data': m.raw.copy()})()
    apply_unmixing(view, spectra, fl_names, SPECTRAL_DETECTORS)
    for fl in fluors:
        assert f'U:{fl}' in view.data.columns
        assert view.data[f'U:{fl}'].std() > 0          # real, varying signal


def test_diff_labels_constant():
    # The panel the generator embeds matches the documented CD↔fluor map.
    assert DIFF_LABELS == {'BV421-A': 'CD11b', 'PE-Cy7-A': 'CD45',
                           'APC-A': 'CD34'}


# ── PBMC immunophenotyping ────────────────────────────────────────────────────

def test_pbmc_has_separable_lineages():
    df = immunophenotyping_sample(n=10000, seed=1)
    live = df[df['LiveDead-A'] < 1000]
    # CD3+CD4+ (T helper) and CD19+ (B) subsets are present and distinct.
    cd4t = live[(live['BV510-A'] > 2000) & (live['FITC-A'] > 2000)]
    bcell = live[(live['BV510-A'] < 1000) & (live['PE-A'] > 2000)]
    assert len(cd4t) > 0.15 * len(live)              # CD4 T is a major subset
    assert len(bcell) > 0.03 * len(live)             # B cells present
    # B cells are CD3-negative (lineage exclusivity).
    assert bcell['BV510-A'].median() < cd4t['BV510-A'].median()


def test_pbmc_group_effect_and_batch_gain():
    ctrl = immunophenotyping_sample(n=8000, seed=2, group='ctrl')
    treat = immunophenotyping_sample(n=8000, seed=3, group='treat')
    # 'treat' shifts toward NK (CD56+, BV605-A).
    nk_ctrl = (ctrl['BV605-A'] > 2000).mean()
    nk_treat = (treat['BV605-A'] > 2000).mean()
    assert nk_treat > nk_ctrl
    # Batch gain scales the fluorescence channels but not scatter.
    base = immunophenotyping_sample(n=4000, seed=4, batch_gain=1.0)
    hi = immunophenotyping_sample(n=4000, seed=4, batch_gain=1.5)
    assert hi['BV510-A'].median() > base['BV510-A'].median() * 1.3
    assert abs(hi['FSC-A'].median() - base['FSC-A'].median()) < 5000


def test_pbmc_clusters_into_lineages():
    df = immunophenotyping_sample(n=6000, seed=5)
    s = types.SimpleNamespace(
        data=df.copy(),
        fluor_channels=['BV510-A', 'FITC-A', 'APC-A', 'PE-A', 'BV605-A',
                        'APC-Fire-A'])
    FlowSample.run_leiden(s, resolution=0.5)
    # The major lineages give several well-populated clusters.
    assert s.data['leiden'].nunique() >= 4


# ── cell cycle ────────────────────────────────────────────────────────────────

def test_cell_cycle_has_g1_and_g2m_peaks():
    df = cell_cycle_sample(n=12000, seed=6)
    cells = df[(df['FSC-A'] > 2e4) & (df['FSC-A'] < 9e4)]
    dna = cells['DAPI-A']
    # Two DNA peaks at ~2N and ~4N.
    g1 = ((dna > 44000) & (dna < 56000)).mean()
    g2m = ((dna > 94000) & (dna < 106000)).mean()
    assert g1 > 0.3 and g2m > 0.05
    # 4N is ~twice 2N.
    assert 1.8 < dna[dna > 94000].median() / dna[(dna > 44000) &
                                                 (dna < 56000)].median() < 2.2


# ── compensation controls ─────────────────────────────────────────────────────

def test_compensation_controls_and_matrix(tmp_path):
    paths, csv = make_compensation_controls(str(tmp_path), n=4000, seed=8)
    assert len(paths) == len(PBMC_MARKERS)
    assert os.path.isfile(csv)
    from openflo.pipeline import read_compensation_matrix
    chans, mat = read_compensation_matrix(csv)
    assert mat.shape == (len(PBMC_MARKERS), len(PBMC_MARKERS))
    # Known APC-A → APC-Fire-A leak is in the matrix.
    i, j = chans.index('APC-A'), chans.index('APC-Fire-A')
    assert mat[i, j] == pytest.approx(0.18, abs=1e-6)
    # The APC single stain shows real spill into APC-Fire (raw, uncompensated).
    s = FlowSample(os.path.join(str(tmp_path), 'APC-A_stain.fcs'))
    s.run_qc()
    pos = s.raw[s.raw['APC-A'] > 3000]
    assert pos['APC-Fire-A'].median() > s.raw['APC-Fire-A'].median()


def test_make_dataset_writes_all_subdatasets(tmp_path):
    info = make_dataset(out_dir=str(tmp_path), n=2000, spectral_n=1200,
                        days=(3, 9), conditions=('Stim',), reps=1, donors=1)
    for sub in ('pbmc', 'pbmc_batches', 'fmo', 'cellcycle', 'compensation',
                'diff', 'spectral', 'beads'):
        assert (tmp_path / sub).is_dir(), sub
    assert info['pbmc_files'] >= 1
    assert info['fmo_files'] == len(PBMC_MARKERS)
    assert (tmp_path / 'compensation' / 'compensation.csv').is_file()
    assert (tmp_path / 'pbmc_batches' / 'Batch 1').is_dir()
    # size beads written + listed in the summary
    assert (tmp_path / 'beads' / 'size_beads.fcs').is_file()
    assert (tmp_path / 'beads' / 'size_beads.csv').is_file()
    assert info['size_bead_fcs'].endswith('size_beads.fcs')


# ── size-calibration beads (FSC → µm anchor) ────────────────────────────────────

def test_size_beads_scale_is_recoverable():
    """The bead population is a tight single mode whose median FSC-A encodes a
    known FSC-per-µm scale — the anchor the auto-clean debris 'bead' mode uses.
    The name contains 'bead' so the GUI auto-detects it."""
    df = size_bead_sample(diameter_um=SIZE_BEAD_UM, n=8000, seed=950)
    fsc = df['FSC-A'].to_numpy(float)
    med = float(np.median(fsc))
    assert med / SIZE_BEAD_UM == pytest.approx(SIZE_BEAD_FSC_PER_UM, rel=0.03)
    assert fsc.std() / fsc.mean() < 0.06                # tight (~4 % CV)


def test_make_size_beads_csv_and_load(tmp_path):
    fcs, csv = make_size_beads(str(tmp_path), seed=950)
    assert 'bead' in os.path.basename(fcs).lower()      # auto-detectable name
    s = FlowSample(fcs)
    assert {'FSC-A', 'FSC-H', 'SSC-A'} <= set(s.data.columns)
    import pandas as pd
    ref = pd.read_csv(csv)
    assert float(ref['diameter_um'].iloc[0]) == SIZE_BEAD_UM
    assert float(ref['fsc_median'].iloc[0]) == pytest.approx(
        float(np.median(s.data['FSC-A'])), rel=1e-3)
