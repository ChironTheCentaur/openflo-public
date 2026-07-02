"""Synthetic flow-cytometry datasets — a realistic, self-contained example
that exercises every OpenFlo feature, and a double-check you can regenerate.

Two generators, both pure (numpy / pandas; FlowIO only to write ``.fcs``):

  * **Differentiation time-course** — a myeloid model (CD34 progenitors →
    CD11b myeloids) across days, with Stim vs Ctrl conditions and
    replicates. Each sample is a mixture of debris, doublets and three real
    populations whose proportions and intensities shift with maturation, so it
    drives cleaning, (auto-)gating, clustering/Leiden, trajectory, frequency &
    expression comparison, and the report.
  * **Spectral controls** — single-stain + unstained + mixed samples over a
    detector array with known reference spectra, for spectral unmixing, the
    QC diagnostics, and the ``--unmix`` CLI.

``make_dataset(out_dir)`` writes both plus a staining-panel ``.xlsx`` and a
README. The ``scripts/make_synthetic_dataset.py`` CLI is a thin wrapper.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

# ── Differentiation model ─────────────────────────────────────────────────────

DIFF_CHANNELS = ['FSC-A', 'FSC-H', 'SSC-A', 'SSC-H',
                 'BV421-A', 'PE-Cy7-A', 'APC-A', 'Time']
# Detector → antibody (the staining panel). A generic example panel.
DIFF_LABELS = {'BV421-A': 'CD11b', 'PE-Cy7-A': 'CD45', 'APC-A': 'CD34'}


def _maturation(day, condition):
    """Maturation fraction t∈[0,1] for a day (Stim accelerates differentiation)."""
    base = (day - 3) / 12.0
    boost = 0.18 if str(condition).lower().startswith('stim') else 0.0
    return float(np.clip(base + boost, 0.0, 1.0))


def differentiation_sample(day, condition, n=20_000, seed=0):
    """One synthetic cell-differentiation sample as a DataFrame over
    :data:`DIFF_CHANNELS`.

    Composition: ~10 % debris (low FSC), ~7 % doublets (off the FSC-A/FSC-H
    diagonal), and singlet cells split among **progenitors** (CD34⁺CD11b⁻),
    **intermediate**, and **mature myeloid** (CD34⁻CD11b⁺). The mature fraction and
    CD11b intensity rise with day and are higher under Stim; CD34 falls. A
    monotone Time channel with mild rate drift gives the flow-rate / drift
    cleaning something to find."""
    rng = np.random.default_rng(seed)
    t = _maturation(day, condition)
    stim = str(condition).lower().startswith('stim')

    n_debris = int(n * 0.10)
    n_doublet = int(n * 0.07)
    n_cells = n - n_debris - n_doublet

    # Cell-population proportions shift with maturation.
    f_mature = 0.08 + 0.80 * t
    f_prog = 0.80 * (1.0 - t) + 0.05
    f_inter = max(0.05, 1.0 - f_mature - f_prog)
    probs = np.array([f_prog, f_inter, f_mature])
    probs = probs / probs.sum()
    pop = rng.choice(3, size=n_cells, p=probs)        # 0 prog, 1 inter, 2 mature

    def _pos(mean, sd, size):
        return np.clip(rng.normal(mean, sd, size), 1.0, None)

    cd34 = np.empty(n_cells)
    cd11b = np.empty(n_cells)
    cd45 = _pos(2500, 600, n_cells)                    # pan-leukocyte, ~flat
    for code, (m34, m41) in enumerate([(8000, 300), (3000, 3000),
                                       (450, 9000)]):
        m = pop == code
        cnt = int(m.sum())
        if cnt:
            cd34[m] = _pos(m34, max(m34 * 0.2, 120), cnt)
            boost = 1.25 if (stim and code == 2) else 1.0
            cd11b[m] = _pos(m41 * boost, max(m41 * 0.22, 120), cnt)

    # Scatter: singlets sit on FSC-A ≈ 2·FSC-H; SSC similar.
    fsc_a = _pos(60000, 8000, n_cells)
    fsc_h = fsc_a / 2.0 * rng.normal(1.0, 0.02, n_cells)
    ssc_a = _pos(40000, 9000, n_cells)
    ssc_h = ssc_a / 2.0 * rng.normal(1.0, 0.03, n_cells)
    cells = np.column_stack([fsc_a, fsc_h, ssc_a, ssc_h, cd11b, cd45, cd34])

    # Debris: low FSC/SSC, dim, near-uniform markers.
    d_fa = _pos(8000, 3000, n_debris)
    debris = np.column_stack([
        d_fa, d_fa / 2.0 * rng.normal(1.0, 0.05, n_debris),
        _pos(6000, 2500, n_debris),
        _pos(3000, 1500, n_debris),
        _pos(200, 150, n_debris), _pos(200, 150, n_debris),
        _pos(200, 150, n_debris)])

    # Doublets: high FSC-A, off the singlet diagonal (more area per height).
    db_fa = _pos(115000, 12000, n_doublet)
    doublet = np.column_stack([
        db_fa, db_fa / 3.1 * rng.normal(1.0, 0.04, n_doublet),
        _pos(75000, 12000, n_doublet),
        _pos(38000, 9000, n_doublet),
        _pos(6000, 2500, n_doublet), _pos(4000, 1500, n_doublet),
        _pos(4000, 2000, n_doublet)])

    body = np.vstack([cells, debris, doublet])
    rng.shuffle(body)
    # Time: monotone acquisition order with a slight, drifting rate.
    time = np.cumsum(rng.gamma(2.0, 1.0, len(body))
                     * np.linspace(1.0, 1.4, len(body)))
    time = time / time[-1] * 300.0                    # ~5 min run, arbitrary units
    df = pd.DataFrame(np.column_stack([body, time]), columns=pd.Index(DIFF_CHANNELS))
    return df


def make_differentiation_dataset(out_dir, days=(3, 6, 9, 12, 15),
                                 conditions=('Stim', 'Ctrl'), reps=2,
                                 n=20_000, seed=0):
    """Write the differentiation time-course as ``Day N/<cond>_mX.fcs`` under
    ``out_dir`` (grouped so a folder-drop yields the day structure). Returns the
    list of written paths."""
    paths = []
    s = seed
    for day in days:
        day_dir = os.path.join(out_dir, f'Day {day}')
        os.makedirs(day_dir, exist_ok=True)
        for cond in conditions:
            for rep in range(1, reps + 1):
                df = differentiation_sample(day, cond, n=n, seed=s)
                s += 1
                path = os.path.join(day_dir, f'{cond}_m{rep}.fcs')
                _write_fcs(path, df, DIFF_LABELS)
                paths.append(path)
    return paths


# ── Spectral model ────────────────────────────────────────────────────────────

SPECTRAL_DETECTORS = [f'D{i + 1}-A' for i in range(8)]
# Distinct emission signatures over the 8 detectors (peak shifts across fluors).
SPECTRAL_SPECTRA = {
    'CD11b-BV421':  np.array([1.0, 0.7, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0]),
    'CD45-PECy7':  np.array([0.0, 0.1, 0.4, 1.0, 0.8, 0.3, 0.1, 0.0]),
    'CD34-APC':    np.array([0.0, 0.0, 0.0, 0.1, 0.3, 0.7, 1.0, 0.6]),
}


def _spectral_single_stain(fluor, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    spec = SPECTRAL_SPECTRA[fluor]
    amp = rng.uniform(50, 800, (n, 1))
    sig = amp * spec[None, :]
    noise = rng.normal(0, 1.0, (n, len(spec))) * np.sqrt(np.clip(sig, 0, None))
    return np.clip(sig + noise + rng.normal(0, 2.0, (n, len(spec))), 0, None)


def make_spectral_dataset(out_dir, n=6000, seed=100):
    """Write single-stain + unstained + mixed spectral samples (over
    :data:`SPECTRAL_DETECTORS`) plus a ``controls.json`` for the ``--unmix``
    CLI. Returns ``(paths, controls_json_path)``."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths, controls = [], {}
    s = seed
    for fluor in SPECTRAL_SPECTRA:
        arr = _spectral_single_stain(fluor, n=n, seed=s)
        s += 1
        p = os.path.join(out_dir, f'ss_{fluor}.fcs')
        _write_fcs(p, pd.DataFrame(arr, columns=pd.Index(SPECTRAL_DETECTORS)))
        paths.append(p)
        controls[fluor] = os.path.abspath(p)
    # Unstained (faint autofluorescence).
    un = np.clip(rng.normal(0, 1.0, (n, len(SPECTRAL_DETECTORS))) + 3.0, 0, None)
    p_un = os.path.join(out_dir, 'unstained.fcs')
    _write_fcs(p_un, pd.DataFrame(un, columns=pd.Index(SPECTRAL_DETECTORS)))
    paths.append(p_un)
    controls['unstained'] = os.path.abspath(p_un)
    # A mixed sample = random combination of all three fluors.
    A = rng.uniform(0, 500, (n, len(SPECTRAL_SPECTRA)))
    S = np.array([SPECTRAL_SPECTRA[f] for f in SPECTRAL_SPECTRA])
    mixed = np.clip(A @ S + rng.normal(0, 1.0, (n, len(SPECTRAL_DETECTORS))),
                    0, None)
    p_mix = os.path.join(out_dir, 'mixed_sample.fcs')
    _write_fcs(p_mix, pd.DataFrame(mixed, columns=pd.Index(SPECTRAL_DETECTORS)))
    paths.append(p_mix)
    controls_path = os.path.join(out_dir, 'controls.json')
    with open(controls_path, 'w', encoding='utf-8') as f:
        json.dump(controls, f, indent=2)
    return paths, controls_path


# ── Immunophenotyping model (generic PBMC) ────────────────────────────────────
#
# The canonical, panel-agnostic flow example: human PBMCs stained for the major
# lineages. Rich multi-population structure for gating hierarchies, clustering /
# Leiden / UMAP, frequency & expression comparison, heatmaps and the report —
# not tied to any one study.

PBMC_CHANNELS = ['FSC-A', 'FSC-H', 'SSC-A', 'LiveDead-A', 'BV510-A', 'FITC-A',
                 'APC-A', 'PE-A', 'BV605-A', 'APC-Fire-A', 'Time']
PBMC_MARKERS = ['CD3', 'CD4', 'CD8', 'CD19', 'CD56', 'CD14']
# Marker → detector (the staining panel for this dataset).
PBMC_MARKER_DET = {'CD3': 'BV510-A', 'CD4': 'FITC-A', 'CD8': 'APC-A',
                   'CD19': 'PE-A', 'CD56': 'BV605-A', 'CD14': 'APC-Fire-A'}
PBMC_LABELS = dict({d: m for m, d in PBMC_MARKER_DET.items()},
                   **{'LiveDead-A': 'L/D'})
_HI, _LO = 5000.0, 150.0
# population → (frequency, {marker: positive mean}, SSC mean)
PBMC_POPS = {
    'CD4 T':    (0.34, {'CD3': _HI, 'CD4': _HI}, 30000),
    'CD8 T':    (0.20, {'CD3': _HI, 'CD8': _HI}, 30000),
    'B cell':   (0.11, {'CD19': _HI}, 32000),
    'NK cell':  (0.10, {'CD56': _HI}, 35000),
    'Monocyte': (0.18, {'CD14': _HI}, 62000),
    'DN':       (0.07, {}, 30000),
}


def immunophenotyping_sample(n=20_000, seed=0, group='ctrl', batch_gain=1.0,
                             fmo=None):
    """One synthetic PBMC immunophenotyping sample as a DataFrame over
    :data:`PBMC_CHANNELS`.

    Live singlets are split among CD4 T, CD8 T, B, NK, monocyte and
    double-negative populations (plus ~8 % dead, ~7 % debris, ~5 % doublets).
    ``group='treat'`` shifts the composition (more NK/CD8, fewer CD4) so a
    two-group comparison has an effect. ``batch_gain`` multiplies the
    fluorescence channels (a technical batch shift for CytoNorm). ``fmo`` (a
    marker name) collapses that marker to background everywhere — an FMO
    control."""
    rng = np.random.default_rng(seed)
    pops = {k: list(v) for k, v in PBMC_POPS.items()}
    if str(group).lower().startswith('treat'):
        for nm, df_ in (('NK cell', 0.06), ('CD8 T', 0.04), ('CD4 T', -0.10)):
            pops[nm][0] = max(0.01, pops[nm][0] + df_)

    n_dead = int(n * 0.08)
    n_debris = int(n * 0.07)
    n_doublet = int(n * 0.05)
    n_live = n - n_dead - n_debris - n_doublet

    names = list(pops)
    freqs = np.array([pops[k][0] for k in names], dtype=float)
    freqs /= freqs.sum()
    assign = rng.choice(len(names), n_live, p=freqs)
    nm = len(PBMC_MARKERS)

    markers = np.clip(rng.normal(_LO, 40, (n_live, nm)), 1.0, None)
    ld = np.clip(rng.normal(220, 90, n_live), 1.0, None)        # live → low
    ssc = np.empty(n_live)
    for ci, name in enumerate(names):
        m = assign == ci
        cnt = int(m.sum())
        if not cnt:
            continue
        prof, sscm = pops[name][1], pops[name][2]
        for mi, mk in enumerate(PBMC_MARKERS):
            if mk in prof:
                markers[m, mi] = np.clip(
                    rng.normal(prof[mk], prof[mk] * 0.20, cnt), 1.0, None)
        ssc[m] = np.clip(rng.normal(sscm, sscm * 0.18, cnt), 1.0, None)
    fsc_a = np.clip(rng.normal(60000, 8000, n_live), 1.0, None)
    fsc_h = fsc_a / 2.0 * rng.normal(1.0, 0.02, n_live)
    live = np.column_stack([fsc_a, fsc_h, ssc, ld, markers])

    # Dead cells: LiveDead high, markers dim, normal scatter.
    d_fsc = np.clip(rng.normal(55000, 9000, n_dead), 1.0, None)
    dead = np.column_stack([
        d_fsc, d_fsc / 2.0 * rng.normal(1.0, 0.03, n_dead),
        np.clip(rng.normal(32000, 8000, n_dead), 1.0, None),
        np.clip(rng.normal(5000, 1200, n_dead), 1.0, None),       # L/D high
        np.clip(rng.normal(_LO, 120, (n_dead, nm)), 1.0, None)])

    db_fsc = np.clip(rng.normal(7000, 2500, n_debris), 1.0, None)
    debris = np.column_stack([
        db_fsc, db_fsc / 2.0 * rng.normal(1.0, 0.05, n_debris),
        np.clip(rng.normal(5000, 2000, n_debris), 1.0, None),
        np.clip(rng.normal(800, 400, n_debris), 1.0, None),
        np.clip(rng.normal(_LO, 100, (n_debris, nm)), 1.0, None)])

    dbl_fsc = np.clip(rng.normal(118000, 12000, n_doublet), 1.0, None)
    doublet = np.column_stack([
        dbl_fsc, dbl_fsc / 3.1 * rng.normal(1.0, 0.04, n_doublet),
        np.clip(rng.normal(70000, 12000, n_doublet), 1.0, None),
        np.clip(rng.normal(400, 200, n_doublet), 1.0, None),
        np.clip(rng.normal(3000, 1500, (n_doublet, nm)), 1.0, None)])

    body = np.vstack([live, dead, debris, doublet])
    rng.shuffle(body)
    # Columns 3.. are fluorescence (LiveDead + markers) → apply the batch gain.
    if batch_gain != 1.0:
        body[:, 3:] *= float(batch_gain)
    # FMO: collapse one marker to background everywhere.
    if fmo and fmo in PBMC_MARKERS:
        col = 4 + PBMC_MARKERS.index(fmo)
        body[:, col] = np.clip(rng.normal(_LO, 40, len(body)), 1.0, None)
    time = np.linspace(0.0, 300.0, len(body)) + rng.normal(0, 0.5, len(body))
    df = pd.DataFrame(np.column_stack([body, time]),
                      columns=pd.Index(PBMC_CHANNELS))
    return df


def make_immunophenotyping_dataset(out_dir, groups=('ctrl', 'treat'), donors=3,
                                   n=20_000, seed=0, batches=1):
    """Write PBMC samples. With ``batches`` > 1 they are organised as
    ``Batch {b}/<group>_{donor}.fcs`` with a per-batch fluorescence gain (a
    technical shift for CytoNorm batch correction); otherwise flat
    ``<group>_d{donor}.fcs``. Returns the written paths."""
    paths = []
    s = seed
    gains = np.linspace(0.8, 1.3, batches) if batches > 1 else [1.0]
    for b in range(batches):
        bdir = (os.path.join(out_dir, f'Batch {b + 1}') if batches > 1
                else out_dir)
        os.makedirs(bdir, exist_ok=True)
        for group in groups:
            for d in range(1, donors + 1):
                df = immunophenotyping_sample(n=n, seed=s, group=group,
                                              batch_gain=float(gains[b]))
                s += 1
                path = os.path.join(bdir, f'{group}_d{d}.fcs')
                _write_fcs(path, df, PBMC_LABELS)
                paths.append(path)
    return paths


def make_fmo_controls(out_dir, n=10_000, seed=500):
    """Write one FMO control per marker (that marker collapsed to background)
    into ``out_dir`` as ``FMO_<marker>.fcs`` — for FMO-based gate placement."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, mk in enumerate(PBMC_MARKERS):
        df = immunophenotyping_sample(n=n, seed=seed + i, fmo=mk)
        path = os.path.join(out_dir, f'FMO_{mk}.fcs')
        _write_fcs(path, df, PBMC_LABELS)
        paths.append(path)
    return paths


# ── Cell-cycle model (DNA content) ────────────────────────────────────────────

CELLCYCLE_CHANNELS = ['FSC-A', 'FSC-H', 'SSC-A', 'DAPI-A', 'Time']


def cell_cycle_sample(n=20_000, seed=0, s_frac=0.20, g2m_frac=0.14):
    """One synthetic DNA-content sample over :data:`CELLCYCLE_CHANNELS`:
    a 2N G1 peak, a 4N G2/M peak, an S-phase plateau between them, plus debris
    and G1 doublets (4N DNA but high FSC) — drives the cell-cycle tool and
    doublet discrimination."""
    rng = np.random.default_rng(seed)
    n_debris = int(n * 0.04)
    n_doublet = int(n * 0.04)
    n_cells = n - n_debris - n_doublet
    g1_frac = max(0.05, 1.0 - s_frac - g2m_frac)
    code = rng.choice(3, n_cells, p=np.array([g1_frac, s_frac, g2m_frac])
                      / (g1_frac + s_frac + g2m_frac))
    dna = np.empty(n_cells)
    g1 = code == 0
    s_ph = code == 1
    g2m = code == 2
    dna[g1] = rng.normal(50000, 2200, int(g1.sum()))
    dna[g2m] = rng.normal(100000, 3200, int(g2m.sum()))
    dna[s_ph] = rng.uniform(56000, 94000, int(s_ph.sum()))
    dna = np.clip(dna, 1.0, None)
    fsc = np.clip(rng.normal(60000, 7000, n_cells), 1.0, None)
    fsc_h = fsc / 2.0 * rng.normal(1.0, 0.02, n_cells)
    ssc = np.clip(rng.normal(35000, 7000, n_cells), 1.0, None)
    cells = np.column_stack([fsc, fsc_h, ssc, dna])
    # Debris: low FSC, sub-G1 DNA.
    d_fsc = np.clip(rng.normal(9000, 3000, n_debris), 1.0, None)
    debris = np.column_stack([
        d_fsc, d_fsc / 2.0 * rng.normal(1.0, 0.05, n_debris),
        np.clip(rng.normal(7000, 2500, n_debris), 1.0, None),
        np.clip(rng.normal(15000, 6000, n_debris), 1.0, None)])
    # Doublets: 4N DNA (two G1) but high FSC area off the diagonal.
    db_fsc = np.clip(rng.normal(115000, 12000, n_doublet), 1.0, None)
    doublet = np.column_stack([
        db_fsc, db_fsc / 3.1 * rng.normal(1.0, 0.04, n_doublet),
        np.clip(rng.normal(60000, 10000, n_doublet), 1.0, None),
        np.clip(rng.normal(100000, 3500, n_doublet), 1.0, None)])
    body = np.vstack([cells, debris, doublet])
    rng.shuffle(body)
    time = np.linspace(0.0, 300.0, len(body)) + rng.normal(0, 0.5, len(body))
    return pd.DataFrame(np.column_stack([body, time]),
                        columns=pd.Index(CELLCYCLE_CHANNELS))


def make_cell_cycle_dataset(out_dir, samples=2, n=20_000, seed=300):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i in range(1, samples + 1):
        df = cell_cycle_sample(n=n, seed=seed + i,
                               s_frac=0.15 + 0.06 * i, g2m_frac=0.12)
        path = os.path.join(out_dir, f'cellcycle_{i}.fcs')
        _write_fcs(path, df)
        paths.append(path)
    return paths


# ── Conventional compensation (spillover) controls ────────────────────────────

def make_compensation_controls(out_dir, n=8000, seed=700):
    """Single-stain controls with a KNOWN spillover matrix baked into the data,
    plus the matching ``compensation.csv`` the compensation editor auto-imports.
    One ``<detector>_stain.fcs`` per marker (a negative + a bright positive
    population), spilled into neighbouring detectors. Returns
    ``(paths, compensation_csv)``."""
    from .pipeline import write_compensation_matrix
    os.makedirs(out_dir, exist_ok=True)
    dets = [PBMC_MARKER_DET[m] for m in PBMC_MARKERS]
    k = len(dets)
    # Known spillover: identity + a few realistic off-diagonal leaks.
    spill = np.eye(k)
    leaks = {('APC-A', 'APC-Fire-A'): 0.18, ('PE-A', 'BV605-A'): 0.10,
             ('FITC-A', 'BV510-A'): 0.08, ('BV510-A', 'FITC-A'): 0.05,
             ('BV605-A', 'PE-A'): 0.06}
    for (src, dst), v in leaks.items():
        spill[dets.index(src), dets.index(dst)] = v

    rng = np.random.default_rng(seed)
    paths = []
    for i, det in enumerate(dets):
        n_pos = n // 2
        true = np.clip(rng.normal(_LO, 60, (n, k)), 1.0, None)   # all-negative
        true[:n_pos, i] = np.clip(rng.normal(6000, 1200, n_pos), 1.0, None)
        observed = true @ spill                                  # apply spillover
        df = pd.DataFrame(observed, columns=pd.Index(dets))
        path = os.path.join(out_dir, f'{det}_stain.fcs')
        _write_fcs(path, df, {d: PBMC_LABELS[d] for d in dets})
        paths.append(path)
    csv = os.path.join(out_dir, 'compensation.csv')
    write_compensation_matrix(csv, spill, dets)
    return paths, csv


# ── MESF / ABC calibration beads ──────────────────────────────────────────────

# Rainbow-bead-like peaks: known MESF values and the (linear) MFI the instrument
# reads for each — a clean MESF = 2·MFI + 100 relationship to recover.
CAL_CHANNEL = 'FITC-A'
CAL_PEAK_MESF = [400, 1600, 6000, 22000, 80000, 300000]


def make_calibration_beads(out_dir, n=12_000, seed=900):
    """Write a MESF calibration-bead ``.fcs`` (a fluorescence channel with six
    discrete bead peaks) plus a ``mesf_peaks.csv`` of each peak's assigned MESF
    value — for the fluorescence-intensity calibration tool. Returns
    ``(fcs_path, peaks_csv)``."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    # MFI = (MESF - 100) / 2 for each peak; tight log-normal-ish populations.
    peak_mfi = [(v - 100) / 2.0 for v in CAL_PEAK_MESF]
    per = n // len(peak_mfi)
    chunks = [np.clip(rng.normal(mfi, max(mfi * 0.06, 20), per), 1.0, None)
              for mfi in peak_mfi]
    fitc = np.concatenate(chunks)
    m = len(fitc)
    order = rng.permutation(m)                 # interleave the bead peaks
    fitc = fitc[order]
    fsc = np.clip(rng.normal(55000, 4000, m), 1.0, None)
    df = pd.DataFrame({
        'FSC-A': fsc, 'FSC-H': fsc / 2.0 * rng.normal(1.0, 0.02, m),
        'SSC-A': np.clip(rng.normal(30000, 4000, m), 1.0, None),
        CAL_CHANNEL: fitc,
        'Time': np.linspace(0.0, 200.0, m)})
    fcs = os.path.join(out_dir, 'rainbow_beads.fcs')
    _write_fcs(fcs, df)
    peaks_csv = os.path.join(out_dir, 'mesf_peaks.csv')
    pd.DataFrame({'peak': range(1, len(CAL_PEAK_MESF) + 1),
                  'MFI': peak_mfi, 'MESF': CAL_PEAK_MESF}).to_csv(
        peaks_csv, index=False)
    return fcs, peaks_csv


# ── Size-calibration beads (FSC → µm) ──────────────────────────────────────────

# Instrument scatter scale: a real bead/cell of diameter D µm reads
# D * SIZE_BEAD_FSC_PER_UM on FSC-A. Chosen so an 8 µm bead lands at ~65 000
# FSC-A, matching the real BM instrument (comp-bead median ≈ 65 790).
SIZE_BEAD_FSC_PER_UM = 8125.0
SIZE_BEAD_UM = 8.0


def size_bead_sample(diameter_um=SIZE_BEAD_UM, n=10_000, seed=950):
    """One tight size-calibration bead population as a DataFrame (FSC-A/H,
    SSC-A, Time).

    FSC-A is centred at ``diameter_um * SIZE_BEAD_FSC_PER_UM`` with a tight
    ~4 % CV like real calibration beads (and low, non-granular SSC). Its
    **median FSC-A is the absolute-size anchor** the auto-clean debris 'bead'
    mode consumes: ``median / diameter`` recovers FSC-per-µm, so a ``min_um``
    cut maps to ``FSC-A >= min_um * median / diameter``. Deterministic for a
    given ``seed`` so downstream results stay reproducible."""
    rng = np.random.default_rng(seed)
    fsc_mean = float(diameter_um) * SIZE_BEAD_FSC_PER_UM
    fsc = np.clip(rng.normal(fsc_mean, fsc_mean * 0.04, n), 1.0, None)
    return pd.DataFrame({
        'FSC-A': fsc,
        'FSC-H': fsc / 1.04 * rng.normal(1.0, 0.015, n),    # tight singlets
        'SSC-A': np.clip(rng.normal(8000, 1500, n), 1.0, None),
        'Time': np.linspace(0.0, 120.0, n)})


def make_size_beads(out_dir, diameter_um=SIZE_BEAD_UM, n=10_000, seed=950):
    """Write a size-calibration bead ``.fcs`` (a single tight population at a
    known diameter) plus ``size_beads.csv`` (``diameter_um, fsc_median``). The
    filename contains 'bead' so the GUI auto-detects it as the debris size
    anchor. Returns ``(fcs_path, csv_path)``."""
    os.makedirs(out_dir, exist_ok=True)
    df = size_bead_sample(diameter_um=diameter_um, n=n, seed=seed)
    fcs = os.path.join(out_dir, 'size_beads.fcs')
    _write_fcs(fcs, df)
    csv = os.path.join(out_dir, 'size_beads.csv')
    pd.DataFrame({'diameter_um': [float(diameter_um)],
                  'fsc_median': [float(np.median(df['FSC-A']))]}).to_csv(
        csv, index=False)
    return fcs, csv


# ── Shared helpers ────────────────────────────────────────────────────────────

def _write_fcs(path, df, labels=None):
    import flowio
    channels = [str(c) for c in df.columns]
    mat = np.nan_to_num(df.to_numpy(dtype=float), nan=0.0, posinf=0.0,
                        neginf=0.0)
    opt = ([str((labels or {}).get(c, '') or '') for c in channels]
           if labels else None)
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, mat.flatten().tolist(), channels,
                          opt_channel_names=opt)
    return path


def write_panel_xlsx(path):
    """Write a staining-panel spreadsheet (CD ↔ fluorophore rows) matching the
    differentiation channels — readable by ``cli.read_staining_panel``."""
    rows = [['CD11b', 'BV421'], ['CD45', 'PE/Cy7'], ['CD34', 'APC']]
    pd.DataFrame(rows, columns=pd.Index(["marker", "fluorophore"])).to_excel(
        path, index=False)
    return path


_README = """\
OpenFlo synthetic example dataset
=================================
Generated by `openflo.synthetic` — regenerate any time with
`python scripts/make_synthetic_dataset.py`. Generic, not tied to any one study;
between the sub-datasets it exercises every feature.

pbmc/            generic PBMC immunophenotyping (the marquee example)
  {ctrl,treat}_d{1..3}.fcs
  Channels: FSC-A/H, SSC-A, LiveDead, CD3, CD4, CD8, CD19, CD56, CD14, Time
  Populations: CD4 T, CD8 T, B, NK, monocyte, DN (+ dead/debris/doublets).
  → gating hierarchy, auto-gate, Cluster/Leiden/UMAP, Frequencies & Expression
    (group token ctrl,treat), cluster heatmap, Analysis report.

pbmc_batches/    same panel across Batch {1..3} with a technical gain shift
  → Batch-norm (CytoNorm): load 2+ batches, normalize, compare before/after.

fmo/             FMO_<marker>.fcs (each marker collapsed to background)
  → FMO-based gate placement.

cellcycle/       DNA-content samples (G1 / S / G2-M peaks + G1 doublets)
  → Cell cycle tool; doublet discrimination on DAPI-A vs FSC.

compensation/    single-stain controls with a known spillover matrix +
  compensation.csv → the Compensation editor auto-imports the sibling CSV.

calibration/     MESF calibration beads (6 peaks on FITC-A) + mesf_peaks.csv
  → Calibration… : detect peaks, paste the MESF values, fit, apply (MESF=2*MFI+100).

beads/           size-calibration beads (single tight 8 µm population) + size_beads.csv
  → auto-clean Debris 'bead' mode: load it alongside samples (name has 'bead',
    so it's auto-detected) → FSC-A is calibrated to µm; events below the min
    size (default 4 µm) are dropped. Median FSC-A / 8 µm = FSC per µm.

diff/            differentiation time-course (CD34 -> CD11b over days, +/-Stim)
  → Trajectory / pseudotime (root CD34, High) -> CD34 down / CD11b up.
staining panel.xlsx   CD<->fluorophore map for --panel / the GUI.

spectral/        spectral-unmixing controls (8 detectors, 3 fluors)
  controls.json  ready for: openflo-run --unmix --unmix-controls spectral/controls.json
                            --unmix-input spectral/mixed_sample.fcs --out spectral_out

Tip: frequency/expression significance needs replicates — generate with
--reps 4 (or use the Parametric toggle) for p < 0.05 on real effects.
"""


def make_dataset(out_dir='synthetic_data', n=20_000, spectral_n=6000, seed=0,
                 days=(3, 6, 9, 12, 15), conditions=('Stim', 'Ctrl'), reps=2,
                 donors=3):
    """Write the full example dataset under ``out_dir``: a generic PBMC
    immunophenotyping set (flat + a 3-batch variant), FMO controls, a
    cell-cycle set, conventional-compensation controls, the differentiation
    time-course, spectral controls, a panel and a README. Returns a summary
    dict of how many files each sub-dataset produced."""
    os.makedirs(out_dir, exist_ok=True)
    pbmc = make_immunophenotyping_dataset(
        os.path.join(out_dir, 'pbmc'), donors=donors, n=n, seed=seed)
    batches = make_immunophenotyping_dataset(
        os.path.join(out_dir, 'pbmc_batches'), donors=2, n=n, seed=seed + 200,
        batches=3)
    fmo = make_fmo_controls(os.path.join(out_dir, 'fmo'),
                            n=max(4000, n // 2), seed=seed + 500)
    cyc = make_cell_cycle_dataset(os.path.join(out_dir, 'cellcycle'),
                                  n=n, seed=seed + 300)
    comp, comp_csv = make_compensation_controls(
        os.path.join(out_dir, 'compensation'), n=max(4000, n // 2),
        seed=seed + 700)
    cal_fcs, cal_csv = make_calibration_beads(
        os.path.join(out_dir, 'calibration'), seed=seed + 900)
    size_fcs, size_csv = make_size_beads(
        os.path.join(out_dir, 'beads'), seed=seed + 950)
    diff_paths = make_differentiation_dataset(
        os.path.join(out_dir, 'diff'), days=days, conditions=conditions,
        reps=reps, n=n, seed=seed)
    write_panel_xlsx(os.path.join(out_dir, 'staining panel.xlsx'))
    spec_paths, controls = make_spectral_dataset(
        os.path.join(out_dir, 'spectral'), n=spectral_n, seed=seed + 1000)
    with open(os.path.join(out_dir, 'README.txt'), 'w', encoding='utf-8') as f:
        f.write(_README)
    return {'out_dir': os.path.abspath(out_dir),
            'pbmc_files': len(pbmc),
            'pbmc_batch_files': len(batches),
            'fmo_files': len(fmo),
            'cellcycle_files': len(cyc),
            'compensation_files': len(comp),
            'compensation_csv': comp_csv,
            'calibration_fcs': cal_fcs,
            'calibration_csv': cal_csv,
            'size_bead_fcs': size_fcs,
            'size_bead_csv': size_csv,
            'differentiation_files': len(diff_paths),
            'spectral_files': len(spec_paths),
            'controls_json': controls}


def main(argv=None):
    """``openflo-synth`` — write the full synthetic example dataset (the same
    data the test suite and ``openflo-selftest`` use) so end users can try
    features and regression-check changes on data they don't have to provide."""
    import argparse
    ap = argparse.ArgumentParser(
        prog='openflo-synth',
        description='Generate the OpenFlo synthetic example dataset.')
    ap.add_argument('--out', default='synthetic_data',
                    help='Output directory (default: synthetic_data)')
    ap.add_argument('--events', type=int, default=20_000,
                    help='Events per sample (default 20000)')
    ap.add_argument('--spectral-events', type=int, default=6000,
                    dest='spectral_events',
                    help='Events per spectral control (default 6000)')
    ap.add_argument('--reps', type=int, default=2,
                    help='Replicates per day×condition (default 2)')
    ap.add_argument('--seed', type=int, default=0, help='Random seed')
    args = ap.parse_args(argv)
    info = make_dataset(out_dir=args.out, n=args.events,
                        spectral_n=args.spectral_events, reps=args.reps,
                        seed=args.seed)
    print(f"Wrote synthetic dataset → {info['out_dir']}")
    print(f"  PBMC: {info['pbmc_files']}  batches: {info['pbmc_batch_files']}  "
          f"FMO: {info['fmo_files']}  cell-cycle: {info['cellcycle_files']}  "
          f"comp: {info['compensation_files']}  diff: "
          f"{info['differentiation_files']}  spectral: {info['spectral_files']}")
    print("  beads/size_beads.fcs (debris anchor) + calibration/rainbow_beads.fcs")
    print("Then run `openflo-selftest` to check behavior against the baseline.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
