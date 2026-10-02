"""Tests for openflo.plotmath — pure plot/maths helpers extracted from gui.py."""
from __future__ import annotations

import numpy as np

from openflo.plotmath import (
    drop_suffix,
    ellipse_geom,
    ellipse_params,
    gid_from_hit,
    hist_bin_edges,
    in_box,
    point_segment_dist,
    symlog_linthresh,
)


def test_drop_suffix():
    assert drop_suffix(0, 0) == ''
    assert drop_suffix(None, 1000) == ''
    s = drop_suffix(150, 1000)
    assert 'drops 150' in s and '15.0%' in s


def test_in_box():
    assert in_box((5, 5), (0, 0, 10, 10)) is True
    assert in_box((5, 5), (6, 6, 10, 10)) is False
    assert in_box((0, 0), (0, 0, 10, 10)) is True      # inclusive edges
    assert in_box(None, (0, 0, 1, 1)) is False
    assert in_box((1, 1), None) is False


def test_symlog_linthresh():
    assert symlog_linthresh(None) == 1.0               # default
    assert symlog_linthresh([1.0, 2.0]) == 1.0         # too few points
    data = np.concatenate([np.zeros(100), np.linspace(10, 1000, 100)])
    lt = symlog_linthresh(data)
    assert lt >= 1e-6 and lt < 1000                    # 5th pct of |nonzero|


def test_hist_bin_edges_linear_and_log():
    lin = hist_bin_edges(0.0, 10.0, 'linear', n_bins=10)
    assert len(lin) == 11 and lin[0] == 0.0 and lin[-1] == 10.0
    log = hist_bin_edges(1.0, 1000.0, 'log', n_bins=3)
    assert len(log) == 4
    # log spacing → roughly geometric (ratios ~constant), not arithmetic
    ratios = [log[i + 1] / log[i] for i in range(3)]
    assert max(ratios) - min(ratios) < 0.5
    # non-positive lo on a log axis still yields a usable, sorted grid
    safe = hist_bin_edges(-5.0, 100.0, 'log', n_bins=10)
    assert len(safe) == 11 and all(b > 0 for b in safe)


def test_ellipse_params():
    # axis-aligned covariance diag(4, 1), dist_sq=1 → full axes 2*sqrt(4)=4 and
    # 2*sqrt(1)=2 (which maps to width vs height depends on eigh's ascending
    # order, so assert the axis *set*); axis-aligned → angle a multiple of 90.
    gate = {'mean': [0.0, 0.0], 'cov': [[4.0, 0.0], [0.0, 1.0]],
            'distance_sq': 1.0}
    cx, cy, w, h, ang = ellipse_params(gate)
    assert (cx, cy) == (0.0, 0.0)
    assert {round(w, 6), round(h, 6)} == {4.0, 2.0}
    assert abs(ang) % 90.0 < 1e-6 or abs(abs(ang) % 90.0 - 90.0) < 1e-6
    # malformed → None
    assert ellipse_params({'mean': [0.0], 'cov': [[1.0]]}) is None
    assert ellipse_params({}) is None


def test_ellipse_geom():
    gate = {'mean': [1.0, 2.0], 'cov': [[4.0, 0.0], [0.0, 1.0]],
            'distance_sq': 4.0}
    res = ellipse_geom(gate)
    assert res is not None
    (cx, cy), inv, r0, (hx, hy) = res
    assert (cx, cy) == (1.0, 2.0)
    assert abs(r0 - 2.0) < 1e-9                      # sqrt(distance_sq)
    assert inv.shape == (2, 2)
    # handle sits off the centre (the rotation grip), not at it
    assert (hx, hy) != (cx, cy)
    assert ellipse_geom({}) is None
    assert ellipse_geom({'mean': [0, 0], 'cov': [[0, 0], [0, 0]],
                         'distance_sq': 1.0}) is None   # singular cov


def test_point_segment_dist():
    # point on the segment → ~0
    assert point_segment_dist(5, 0, 0, 0, 10, 0, 10, 10) < 1e-9
    # perpendicular offset, normalised by span: point (5,2) to x-axis seg,
    # span_y=10 → 0.2
    assert abs(point_segment_dist(5, 2, 0, 0, 10, 0, 10, 10) - 0.2) < 1e-9
    # degenerate segment (a == b) → distance to the point
    d = point_segment_dist(3, 4, 0, 0, 0, 0, 1, 1)
    assert abs(d - 5.0) < 1e-9                       # 3-4-5


def test_gid_from_hit():
    assert gid_from_hit(('line', 'g1')) == 'g1'
    assert gid_from_hit(('line', 'g1:lo')) == 'g1'   # threshold packs gid:edge
    assert gid_from_hit(('line', 'g1:hi')) == 'g1'
    assert gid_from_hit(None) is None
    assert gid_from_hit(('only-one',)) is None
    assert gid_from_hit(('x', 123)) is None          # non-str second
