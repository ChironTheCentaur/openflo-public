"""Spectral unmixing (reference spectra + least-squares unmix)."""
import types

import numpy as np
import pandas as pd
import pytest

from openflo.spectral import (
    apply_unmixing,
    build_reference_spectra,
    spectral_condition_number,
    spectral_similarity_matrix,
    spillover_spread_matrix,
    unmix,
    unmixing_qc,
)


def _known_spectra():
    # 3 fluors over 5 detectors, distinct emission shapes.
    return np.array([
        [1.0, 0.6, 0.2, 0.0, 0.0],
        [0.0, 0.3, 1.0, 0.4, 0.1],
        [0.0, 0.0, 0.1, 0.5, 1.0],
    ])


# ── unmix ────────────────────────────────────────────────────────────────────

def test_unmix_recovers_abundances():
    S = _known_spectra()
    rng = np.random.default_rng(0)
    A_true = rng.uniform(0, 100, (2000, 3))
    Y = A_true @ S                     # noise-free mixture
    A = unmix(Y, S)
    assert np.allclose(A, A_true, atol=1e-6)


def test_unmix_tolerates_noise_and_nonneg():
    S = _known_spectra()
    rng = np.random.default_rng(1)
    A_true = rng.uniform(0, 100, (3000, 3))
    Y = A_true @ S + rng.normal(0, 0.5, (3000, 5))
    A = unmix(Y, S, nonneg=True)
    assert (A >= 0).all()
    assert np.corrcoef(A[:, 0], A_true[:, 0])[0, 1] > 0.99


# ── build_reference_spectra ──────────────────────────────────────────────────

def test_build_reference_spectra_recovers_shape():
    S = _known_spectra()
    rng = np.random.default_rng(2)
    stains = {}
    for i, name in enumerate(['F1', 'F2', 'F3']):
        # Single stain i: bright events along spectrum i + a dim negative pop.
        bright = rng.uniform(50, 100, (1500, 1))[:, [0]] * S[i][None, :]
        dim = rng.uniform(0, 2, (500, 5))
        stains[name] = np.vstack([bright, dim])
    Sref, fluors = build_reference_spectra(stains, bright_pct=80)
    assert fluors == ['F1', 'F2', 'F3']
    # Each recovered (max-normalised) spectrum matches the true shape.
    for i in range(3):
        true = S[i] / S[i].max()
        assert np.allclose(Sref[i], true, atol=0.1)


def test_build_reference_spectra_adds_autofluorescence():
    stains = {'F1': np.array([[1.0, 0.5, 0.0]] * 10)}
    unstained = np.array([[0.2, 0.2, 0.2]] * 50)
    Sref, fluors = build_reference_spectra(stains, unstained=unstained)
    assert fluors[-1] == 'Autofluorescence'
    assert Sref.shape == (2, 3)


# ── apply_unmixing ───────────────────────────────────────────────────────────

def test_apply_unmixing_adds_columns():
    S = _known_spectra()
    rng = np.random.default_rng(3)
    A_true = rng.uniform(0, 100, (500, 3))
    Y = A_true @ S
    dets = ['D1', 'D2', 'D3', 'D4', 'D5']
    df = pd.DataFrame({d: Y[:, j] for j, d in enumerate(dets)})
    sample = types.SimpleNamespace(data=df)
    cols = apply_unmixing(sample, S, ['F1', 'F2', 'F3'], dets)
    assert cols == ['U:F1', 'U:F2', 'U:F3']
    assert np.allclose(sample.data['U:F1'].to_numpy(), A_true[:, 0], atol=1e-6)


def test_apply_unmixing_detector_mismatch_raises():
    S = _known_spectra()                      # expects 5 detectors
    df = pd.DataFrame({'D1': [1.0], 'D2': [2.0]})
    sample = types.SimpleNamespace(data=df)
    with pytest.raises(ValueError):
        apply_unmixing(sample, S, ['F1', 'F2', 'F3'], ['D1', 'D2'])


# ── spectral_similarity_matrix ────────────────────────────────────────────────

def test_similarity_matrix_diag_and_symmetry():
    S = _known_spectra()
    M = spectral_similarity_matrix(S)
    assert M.shape == (3, 3)
    assert np.allclose(np.diag(M), 1.0)
    assert np.allclose(M, M.T)
    assert (M >= 0).all() and (M <= 1).all()


def test_similarity_identical_spectra_is_one():
    S = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])   # duplicate rows
    M = spectral_similarity_matrix(S)
    assert M[0, 1] == pytest.approx(1.0)


def test_similarity_orthogonal_spectra_is_zero():
    S = np.array([[1.0, 0.0], [0.0, 1.0]])
    M = spectral_similarity_matrix(S)
    assert M[0, 1] == pytest.approx(0.0)


# ── spectral_condition_number ─────────────────────────────────────────────────

def test_condition_number_orthonormal_is_one():
    S = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert spectral_condition_number(S) == pytest.approx(1.0)


def test_condition_number_large_for_near_collinear():
    S = np.array([[1.0, 0.0], [1.0, 1e-3]])            # nearly collinear rows
    assert spectral_condition_number(S) > 100


def test_condition_number_rank_deficient_is_inf():
    S = np.array([[1.0, 2.0], [2.0, 4.0]])             # rank 1
    assert spectral_condition_number(S) == float('inf')


# ── spillover_spread_matrix ───────────────────────────────────────────────────

def _ssm_controls(seed=0, n=6000):
    """Single-stain controls for 3 fluors over 5 detectors where stain F1
    bleeds a sqrt-scaling spread into F2 but not F3."""
    S = _known_spectra()
    rng = np.random.default_rng(seed)
    stains = {}
    # F1: bright primary; its measured spectrum carries Poisson-like noise that
    # makes the F2 unmix channel spread with sqrt(primary).
    prim = rng.uniform(5, 500, (n, 1))
    base = prim * S[0][None, :]
    noise = rng.normal(0, 1, (n, 5)) * np.sqrt(np.clip(base, 0, None))
    stains['F1'] = base + noise
    # F2 and F3: plain bright single stains (no special spillover structure).
    for i, name in [(1, 'F2'), (2, 'F3')]:
        p = rng.uniform(5, 500, (n, 1))
        b = p * S[i][None, :]
        stains[name] = b + rng.normal(0, 1, (n, 5)) * np.sqrt(np.clip(b, 0, None))
    return stains, S, ['F1', 'F2', 'F3']


def test_ssm_shape_and_zero_diagonal():
    stains, S, fluors = _ssm_controls()
    SSM, used = spillover_spread_matrix(stains, S, fluors)
    assert SSM.shape == (3, 3)
    assert used == fluors
    # Stains we provided get a 0 self-spread on the diagonal.
    assert SSM[0, 0] == 0.0 and SSM[1, 1] == 0.0 and SSM[2, 2] == 0.0


def test_ssm_detects_spread_into_neighbour():
    stains, S, fluors = _ssm_controls()
    SSM, _ = spillover_spread_matrix(stains, S, fluors)
    # Column F1 (index 0): spread into F2 (index 1) should be finite & positive.
    assert np.isfinite(SSM[1, 0]) and SSM[1, 0] > 0


def test_ssm_missing_stain_is_nan_column():
    stains, S, fluors = _ssm_controls()
    del stains['F3']                              # no control for F3
    SSM, _ = spillover_spread_matrix(stains, S, fluors)
    assert np.isnan(SSM[:, 2]).all()              # F3 column undefined


# ── unmixing_qc ───────────────────────────────────────────────────────────────

def test_unmixing_qc_bundle():
    stains, S, fluors = _ssm_controls()
    qc = unmixing_qc(stains, S, fluors, sim_threshold=0.9)
    assert qc['fluors'] == fluors
    assert qc['similarity'].shape == (3, 3)
    assert qc['ssm'].shape == (3, 3)
    assert np.isfinite(qc['condition_number'])
    assert isinstance(qc['similar_pairs'], list)
    assert isinstance(qc['worst_spread'], list)
    # worst_spread entries are sorted descending and well-formed.
    vals = [d['spread'] for d in qc['worst_spread']]
    assert vals == sorted(vals, reverse=True)
    for d in qc['worst_spread']:
        assert {'into', 'from', 'spread'} <= set(d)


def test_unmixing_qc_flags_similar_pair():
    # Two near-identical spectra → flagged as a similar pair.
    S = np.array([[1.0, 0.5, 0.1],
                  [1.0, 0.5, 0.11],
                  [0.0, 0.1, 1.0]])
    qc = unmixing_qc({}, S, ['A', 'B', 'C'], sim_threshold=0.98)
    assert qc['similar_pairs']
    top = qc['similar_pairs'][0]
    assert {top['fluor_a'], top['fluor_b']} == {'A', 'B'}
    assert top['similarity'] >= 0.98


def test_condition_number_inf_when_underdetermined():
    """More fluors than detectors → the unmix is underdetermined; the condition
    number must be inf, not a finite value from the min(shape) singular values."""
    from openflo.spectral import spectral_condition_number
    S = np.eye(6)[:, :4]          # 6 fluors x 4 detectors (rows > cols)
    assert spectral_condition_number(S) == float('inf')
    assert np.isfinite(spectral_condition_number(np.eye(4)))   # well-posed stays finite
