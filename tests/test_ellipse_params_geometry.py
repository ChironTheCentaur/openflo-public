"""plotmath.ellipse_params / ellipse_geom against the gate's own equation.

The existing tests (tests/test_plotmath.py, tests/test_gate_editor_helpers.py)
use only AXIS-ALIGNED covariances and accept any multiple of 90 degrees for the
angle, so the eigenvector/angle choice is never checked: drawing the width
along the wrong axis, or rotating the wrong way, still passes them.

Here the ground truth is the gate itself: every point on the drawn ellipse
must satisfy (p-mu)^T Sigma^-1 (p-mu) == distance_sq, for a correlated
covariance. Pure numpy + matplotlib, no Tk.
"""
import numpy as np
import pytest
from matplotlib.patches import Ellipse

from openflo.plotmath import ellipse_geom, ellipse_params

_CASES = [
    # mean, cov, distance_sq
    ([2.0, -1.0], [[9.0, 2.4], [2.4, 4.0]], 1.0),        # rho = 0.4, tilted
    ([100.0, 50.0], [[4e4, -3e4], [-3e4, 4e4]], 4.605),   # rho = -0.75
    ([0.0, 0.0], [[1.0, 0.0], [0.0, 16.0]], 2.0),         # axis-aligned, tall
]


def _d2(points, mean, cov):
    d = np.asarray(points) - np.asarray(mean)
    return np.einsum('ij,jk,ik->i', d, np.linalg.inv(cov), d)


@pytest.mark.parametrize('mean, cov, dsq', _CASES)
def test_drawn_ellipse_boundary_is_the_gate_rim(mean, cov, dsq):
    cx, cy, w, h, ang = ellipse_params({'mean': mean, 'cov': cov, 'distance_sq': dsq})
    patch = Ellipse((cx, cy), w, h, angle=ang)
    t = np.linspace(0, 2 * np.pi, 73)
    unit = np.column_stack([np.cos(t), np.sin(t)])
    rim = patch.get_patch_transform().transform(unit)    # data coordinates
    assert np.allclose(_d2(rim, mean, cov), dsq, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize('mean, cov, dsq', _CASES)
def test_drawn_ellipse_area_matches_gate(mean, cov, dsq):
    _cx, _cy, w, h, _ang = ellipse_params({'mean': mean, 'cov': cov, 'distance_sq': dsq})
    area = np.pi * (w / 2) * (h / 2)
    assert area == pytest.approx(np.pi * np.sqrt(np.linalg.det(cov)) * dsq, rel=1e-12)


@pytest.mark.parametrize('mean, cov, dsq', _CASES)
def test_rotation_handle_sits_beyond_the_rim_on_the_major_axis(mean, cov, dsq):
    (mx, my), inv, r0, (hx, hy) = ellipse_geom(
        {'mean': mean, 'cov': cov, 'distance_sq': dsq})
    assert (mx, my) == tuple(mean)
    assert np.allclose(inv, np.linalg.inv(cov))
    assert r0 == pytest.approx(np.sqrt(dsq))
    # The handle is 1.18 x the rim distance out, along the +height axis
    # (the larger eigenvector), so it is grabbable and not on the rim.
    d2 = _d2([[hx, hy]], mean, cov)[0]
    assert d2 == pytest.approx(1.18 ** 2 * dsq, rel=1e-9)
    w, v = np.linalg.eigh(cov)
    major = v[:, 1]
    off = np.array([hx - mx, hy - my])
    assert abs(off[0] * major[1] - off[1] * major[0]) <= 1e-9 * np.linalg.norm(off)
