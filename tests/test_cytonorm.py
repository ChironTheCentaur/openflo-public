"""Tests for CytoNorm batch normalization (pipeline.CytoNorm)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from openflo.pipeline import CytoNorm

CHS = ['M1', 'M2']


def _bimodal(rng, n, shift=0.0, scale=1.0):
    cols = [np.concatenate([rng.normal(0, 0.3, n // 2),
                            rng.normal(3, 0.4, n - n // 2)])
            for _ in CHS]
    return pd.DataFrame(np.column_stack(cols) * scale + shift, columns=CHS)


def _wass(a, b):
    from scipy.stats import wasserstein_distance
    return float(np.mean([wasserstein_distance(a[c], b[c]) for c in CHS]))


def test_global_quantile_aligns_batches():
    """1 metacluster = global quantile norm — should nearly eliminate a
    uniform shift/scale batch effect (the core transform is correct)."""
    rng = np.random.default_rng(0)
    a = _bimodal(rng, 6000, 0.0, 1.0)
    b = _bimodal(rng, 6000, 0.8, 1.15)
    cn = CytoNorm(CHS, n_metaclusters=1, grid=(6, 6), seed=1).fit({'A': a, 'B': b})
    d0 = _wass(a, b)
    d1 = _wass(cn.apply(a, 'A'), cn.apply(b, 'B'))
    assert d1 < d0 * 0.2          # >80% reduction


def test_metacluster_mode_runs_and_helps():
    """Default per-metacluster mode runs end-to-end and reduces (not
    increases) the inter-batch distance on a moderate effect."""
    rng = np.random.default_rng(1)
    a = _bimodal(rng, 6000, 0.0, 1.0)
    b = _bimodal(rng, 6000, 0.3, 1.05)
    cn = CytoNorm(CHS, n_metaclusters=6, grid=(6, 6), seed=1).fit({'A': a, 'B': b})
    d0 = _wass(a, b)
    d1 = _wass(cn.apply(a, 'A'), cn.apply(b, 'B'))
    assert d1 <= d0               # never worse on a moderate, co-clustering effect


def test_single_batch_is_near_identity():
    rng = np.random.default_rng(2)
    a = _bimodal(rng, 4000)
    cn = CytoNorm(CHS, n_metaclusters=4, grid=(5, 5), seed=1).fit({'A': a})
    out = cn.apply(a, 'A')
    # goal == the only batch, so correction barely moves anything.
    assert np.allclose(out[CHS].to_numpy(), a[CHS].to_numpy(), atol=0.05)


def test_apply_unknown_batch_passes_through():
    rng = np.random.default_rng(3)
    a = _bimodal(rng, 3000)
    cn = CytoNorm(CHS, n_metaclusters=4, grid=(5, 5), seed=1).fit({'A': a})
    b = _bimodal(rng, 3000, 0.5)
    out = cn.apply(b, 'NOT_FIT')          # batch never fit → identity
    assert np.allclose(out[CHS].to_numpy(), b[CHS].to_numpy(), equal_nan=True)


def test_apply_preserves_other_columns():
    rng = np.random.default_rng(4)
    a = _bimodal(rng, 3000)
    a = a.assign(FSC=rng.normal(5, 1, len(a)), cluster=0)
    cn = CytoNorm(CHS, n_metaclusters=3, grid=(5, 5), seed=1).fit(
        {'A': a[CHS], 'B': _bimodal(rng, 3000, 0.4)})
    out = cn.apply(a, 'A')
    assert 'FSC' in out.columns and 'cluster' in out.columns
    assert np.allclose(out['FSC'].to_numpy(), a['FSC'].to_numpy())


def test_save_load_roundtrip():
    rng = np.random.default_rng(5)
    a = _bimodal(rng, 4000, 0.0)
    b = _bimodal(rng, 4000, 0.6, 1.1)
    cn = CytoNorm(CHS, n_metaclusters=6, grid=(6, 6), seed=1).fit({'A': a, 'B': b})
    cn2 = CytoNorm.from_dict(json.loads(json.dumps(cn.to_dict())))
    out1 = cn.apply(b, 'B')[CHS].to_numpy()
    out2 = cn2.apply(b, 'B')[CHS].to_numpy()
    assert np.allclose(out1, out2, equal_nan=True)


def test_qc_structure_and_improvement():
    rng = np.random.default_rng(6)
    a = _bimodal(rng, 5000, 0.0)
    b = _bimodal(rng, 5000, 0.8, 1.15)
    cn = CytoNorm(CHS, n_metaclusters=1, grid=(6, 6), seed=1).fit({'A': a, 'B': b})
    qc = cn.qc({'A': a, 'B': b})
    assert set(qc) == set(CHS)
    for ch in CHS:
        assert 'before' in qc[ch] and 'after' in qc[ch]
        assert qc[ch]['after'] <= qc[ch]['before'] + 1e-9


def test_fit_requires_events():
    with pytest.raises(ValueError):
        CytoNorm(CHS).fit({'A': pd.DataFrame({'M1': [], 'M2': []})})


def test_from_dict_handles_pipe_in_batch_label():
    """Batch ids can be POSIX paths containing '|'; the to_dict/from_dict
    round-trip must not corrupt or crash on them (the delimiter collision)."""
    cn = CytoNorm(['CD3'], n_metaclusters=2)
    cn._batch_q = {(0, 'trial|A', 1): np.array([1.0, 2.0, 3.0])}
    cn._goal_q = {(0, 1): np.array([4.0, 5.0, 6.0])}
    back = CytoNorm.from_dict(cn.to_dict())
    assert (0, 'trial|A', 1) in back._batch_q
    np.testing.assert_allclose(back._batch_q[(0, 'trial|A', 1)], [1.0, 2.0, 3.0])
    assert (0, 1) in back._goal_q
