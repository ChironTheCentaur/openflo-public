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


def test_similarity_matrix_marks_a_zero_norm_spectrum_as_undefined():
    """A degenerate fluor must not be hidden behind a confident number.

    This used to assert a diagonal of 1.0, because the concern was that a 0
    there would hide the degenerate fluor. NaN answers that concern and is
    what is actually true: the cosine similarity of a zero vector — with
    another spectrum or with itself — is 0/0. The old normalise-by-1.0 also
    reported 0.0 against every OTHER fluor, which on this scale reads as
    "maximally distinct, trivially unmixable" for a fluor that cannot be
    unmixed at all. See tests/test_spectral_degenerate_reference.py.
    """
    from openflo.spectral import spectral_similarity_matrix
    M = spectral_similarity_matrix(np.array([[1.0, 0.0], [0.0, 0.0]]))
    assert M[0, 0] == 1.0                  # the usable fluor is unchanged
    assert np.isnan(M[1, 1])               # not 0.0, which would hide it
    assert np.isnan(M[0, 1]) and np.isnan(M[1, 0])


def test_build_reference_spectra_ignores_non_finite_events():
    """NaN/inf events in a single-stain control must not poison its spectrum."""
    from openflo.spectral import build_reference_spectra
    good = np.array([[10.0, 1.0], [12.0, 1.2], [11.0, 0.9]])
    bad = np.vstack([good, [np.nan, np.nan], [np.inf, 0.0]])
    S_good, _ = build_reference_spectra({'F': good})
    S_bad, _ = build_reference_spectra({'F': bad})
    assert np.all(np.isfinite(S_bad))
    np.testing.assert_allclose(S_bad, S_good)      # non-finite events dropped


def test_build_reference_spectra_subtracts_autofluorescence_values():
    """Autofluorescence is SUBTRACTED from each single-stain spectrum (and added
    as its own endmember) — pins the VALUES, not just shape/label. A '+auto' or
    missing-subtraction bug would pass the shape-only test."""
    from openflo.spectral import build_reference_spectra
    single = {'F1': np.array([[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]])}
    unstained = np.array([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]])
    S, fluors = build_reference_spectra(single, unstained=unstained)
    # F1 mean [1,0.5,0] - auto [0.2,0.2,0.2] = [0.8,0.3,-0.2] -clip-> [0.8,0.3,0]
    #   -max-normalize-> [1.0, 0.375, 0.0]; the auto endmember -> [1,1,1].
    np.testing.assert_allclose(S[0], [1.0, 0.375, 0.0], rtol=1e-6)
    np.testing.assert_allclose(S[-1], [1.0, 1.0, 1.0], rtol=1e-6)
    assert fluors[-1] == 'Autofluorescence'


def test_build_reference_spectra_uses_only_bright_events():
    """bright_pct keeps only the brightest events so a dim/near-flat population
    doesn't drag the spectrum. Bright F1 = [10,5,0] (shape [1,0.5,0]); adding a
    900-event flat dim pop must NOT change the recovered max-normalized spectrum."""
    bright = np.tile([10.0, 5.0, 0.0], (100, 1))
    dim = np.tile([0.3, 0.3, 0.3], (900, 1))
    S, _ = build_reference_spectra({'F1': np.vstack([bright, dim])}, bright_pct=90.0)
    np.testing.assert_allclose(S[0], [1.0, 0.5, 0.0], rtol=1e-6)
    # sanity: without filtering (all events) the flat dim pop drags it off-shape
    S_all, _ = build_reference_spectra({'F1': np.vstack([bright, dim])},
                                       bright_pct=0.0)
    assert not np.allclose(S_all[0], [1.0, 0.5, 0.0], atol=0.05)


def test_normalize_l2_branch():
    """_normalize(mode='l2') divides by the L2 norm (reachable via
    build_reference_spectra(normalize='l2')); only 'max' was exercised."""
    from openflo.spectral import _normalize
    np.testing.assert_allclose(_normalize([3.0, 4.0], mode='l2'), [0.6, 0.8])
    np.testing.assert_allclose(_normalize([0.0, 0.0], mode='l2'), [0.0, 0.0])


def test_apply_unmixing_uses_detector_order_not_dataframe_order():
    """cols follow the `detectors` argument order (matching S's columns), not the
    DataFrame's column order — the existing test supplies them already aligned."""
    import pandas as pd

    import openflo.pipeline as fp
    from openflo.spectral import apply_unmixing
    s = fp.FlowSample.from_dataframe(
        pd.DataFrame({'d0': [1.0], 'd1': [10.0]}), name='x')
    S = np.eye(2)                                    # 2 fluors × 2 detectors (d1,d0)
    apply_unmixing(s, S, fluors=['F_d1', 'F_d0'], detectors=['d1', 'd0'])
    # Y read in detector order [d1,d0]=[10,1]; S=I → abundances = [10,1].
    assert s.data['U:F_d1'].iloc[0] == 10.0         # would be 1.0 if DF-ordered
    assert s.data['U:F_d0'].iloc[0] == 1.0
