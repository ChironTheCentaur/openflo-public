"""gmm_ellipse_gates: the per-component info the auto-gate dialog reports, and
its coverage promise at more than one level.

tests/test_autogate.py covers the fit itself. Two of the returned fields were
unchecked: removing the separation computation (every separation None, so the
editor's "N overlap heavily (separation < 2); review those" warning can never
fire) and counting n_events for the wrong component both left it green.
The max_events subsampling branch (pipeline.py:2722) was never run.

Ground truth: two bivariate normals with known sizes and a known mean
separation, on very different raw scales (x ~ 1e4, y ~ 1e2).
"""
import numpy as np
import pandas as pd
import pytest

from openflo.pipeline import gate_to_mask, gmm_ellipse_gates


def _two(sep_sd, n=(6000, 4000), seed=0):
    rng = np.random.default_rng(seed)
    a = rng.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], n[0])
    b = rng.multivariate_normal([sep_sd, 0], [[1, -0.2], [-0.2, 1]], n[1])
    pts = np.vstack([a, b]) * [5e3, 40.0] + [5e4, 300.0]
    lab = np.r_[np.zeros(n[0], int), np.ones(n[1], int)]
    return pts[:, 0], pts[:, 1], lab


def test_well_separated_components_report_size_and_separation():
    x, y, lab = _two(8.0)
    res = gmm_ellipse_gates(x, y, max_components=4)
    assert [info['n_components'] for _, info in res] == [2, 2]
    df = pd.DataFrame({'X': x, 'Y': y})
    for (gate, info), true_n in zip(res, (6000, 4000), strict=True):   # by weight
        assert info['n_events'] == pytest.approx(true_n, rel=0.01)
        assert info['weight'] == pytest.approx(true_n / 10000, abs=0.01)
        assert info['separation'] is not None and info['separation'] > 5.0
        m = gate_to_mask(dict(gate, x_channel='X', y_channel='Y'), df)
        own = 0 if true_n == 6000 else 1
        assert m[lab == own].mean() == pytest.approx(0.90, abs=0.015)
        assert m[lab != own].mean() < 0.001


def test_overlapping_components_are_flagged_below_two():
    # The editor marks a proposal for review when separation < 2.
    x, y, _lab = _two(1.5)
    res = gmm_ellipse_gates(x, y, max_components=4)
    assert len(res) == 2
    assert all(info['separation'] is not None and info['separation'] < 2.0
               for _, info in res)


@pytest.mark.parametrize('coverage', [0.5, 0.9, 0.99])
def test_ellipse_encloses_the_requested_coverage_after_subsampling(coverage):
    rng = np.random.default_rng(4)
    p = rng.multivariate_normal([1e4, 50], [[4e6, 3e3], [3e3, 25]], 30000)
    ((gate, info),) = gmm_ellipse_gates(p[:, 0], p[:, 1], coverage=coverage,
                                        max_components=3, max_events=20_000)
    assert info['n_events'] == 20_000                   # fitted on the subsample
    m = gate_to_mask(dict(gate, x_channel='X', y_channel='Y'),
                     pd.DataFrame({'X': p[:, 0], 'Y': p[:, 1]}))
    assert m.mean() == pytest.approx(coverage, abs=0.01)
