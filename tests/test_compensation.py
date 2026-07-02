"""Compensation matrix IO — CSV/TSV round-trip and error paths.

Covers the surface most likely to be hit by users (CSV in/out from an
external tool) plus the new CompensationError raises so we don't
regress back to silent ``return None, None`` on malformed input.
"""

import numpy as np
import pytest

import openflo.pipeline as fp

# ── Round-trip ────────────────────────────────────────────────────────────────

def test_csv_round_trip(tmp_path):
    channels = ['BV421-A', 'APC-A', 'PE-Cy7-A']
    matrix = np.array([
        [1.0,  0.05, 0.00],
        [0.00, 1.0,  0.01],
        [0.00, 0.02, 1.0],
    ])
    out = tmp_path / 'comp.csv'
    fp.write_compensation_matrix(str(out), matrix, channels)
    assert out.is_file()

    chans_back, mat_back = fp.read_compensation_matrix(str(out))
    assert chans_back == channels
    assert mat_back is not None
    np.testing.assert_allclose(mat_back, matrix)


def test_tsv_round_trip(tmp_path):
    channels = ['Ch1', 'Ch2']
    matrix = np.array([[1.0, 0.1], [0.0, 1.0]])
    out = tmp_path / 'comp.tsv'
    fp.write_compensation_matrix(str(out), matrix, channels)
    chans_back, mat_back = fp.read_compensation_matrix(str(out))
    assert chans_back == channels
    assert mat_back is not None
    np.testing.assert_allclose(mat_back, matrix)


# ── Error paths (these are exactly what we converted from soft None) ─────────

def test_unsupported_extension_raises(tmp_path):
    p = tmp_path / 'mystery.xyz'
    p.write_text('nothing')
    with pytest.raises(fp.CompensationError, match='unsupported'):
        fp.read_compensation_matrix(str(p))


def test_non_square_csv_raises(tmp_path):
    p = tmp_path / 'rect.csv'
    p.write_text(',Ch1,Ch2,Ch3\nCh1,1,0,0\nCh2,0,1,0\n')   # 2x3, not square
    with pytest.raises(fp.CompensationError, match='not square'):
        fp.read_compensation_matrix(str(p))


def test_missing_file_raises_via_fcs_path(tmp_path):
    p = tmp_path / 'does-not-exist.fcs'
    with pytest.raises(fp.FcsParseError):
        fp.read_compensation_matrix(str(p))


# ── Synthetic FCS spillover keyword (uses the synthetic_fcs fixture) ─────────

def test_synthetic_fcs_has_no_spill(synthetic_fcs):
    """The fixture builds an FCS without a $SPILL keyword, so the read
    should soft-succeed with (None, None) rather than raising."""
    chans, mat = fp.read_compensation_matrix(synthetic_fcs)
    assert chans is None
    assert mat is None


def test_compensation_recovers_true_signal():
    """Compensating observed data must RECOVER the true signal. The spillover
    convention is source->dest (measured = true @ M), so un-mixing is
    `observed @ inv(M)`. Regression guard for the transpose bug where `inv(M).T`
    left asymmetric spillover uncorrected AND corrupted clean channels — the
    direction of the matmul had zero coverage (only parse/IO round-trips did)."""
    import pandas as pd
    chans = ['A', 'B', 'C', 'D']
    S = np.eye(4)
    S[0, 1] = 0.18      # A -> B
    S[2, 3] = 0.10      # C -> D
    S[1, 2] = 0.05      # B -> C  (asymmetric: exactly what the bug corrupted)
    rng = np.random.default_rng(0)
    true = rng.uniform(100, 6000, size=(3000, 4))
    observed = true @ S
    s = fp.FlowSample.from_dataframe(pd.DataFrame(observed, columns=chans),
                                     name='x')
    s.manual_compensate(S.copy(), chans)
    np.testing.assert_allclose(s.data[chans].to_numpy(), true,
                               rtol=1e-6, atol=1e-6)
