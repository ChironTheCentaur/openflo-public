"""Unit tests for gate-editor helper methods added in fd932a7 (Edit
tool, downsample floor, axis-scale apply path).

These don't instantiate ViewGateEditorWindow — that's a tk.Toplevel
that requires a Tk root and ~5000 lines of widget setup. Instead we
call the methods as unbound functions against a SimpleNamespace stub
that holds only the attributes each method reaches for. Fast (no Tk)
and the tests pinpoint regressions in the logic rather than the
widget plumbing around it.

Skipped wholesale on systems without a working tkinter install (e.g.
some headless Linux CI runners) since the module-level imports in
openflo.gui pull in tk.
"""
# pyright: reportArgumentType=false, reportCallIssue=false
#
# We invoke instance methods of ViewGateEditorWindow with SimpleNamespace
# stubs as `self`. Runtime is fine (Python doesn't enforce method binding
# types), but pyright correctly flags every call. Suppress at module
# scope — the alternative is `# type: ignore` on every call which adds
# 30+ lines of noise to no benefit.
from __future__ import annotations

import os
import types

import numpy as np
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')

try:
    import tkinter as _tk  # noqa: F401  (probe only)

    from openflo.gui import ViewGateEditorWindow as V
except (ImportError, RuntimeError) as e:
    pytest.skip(
        f"openflo.gui not importable in this environment: {e}",
        allow_module_level=True)


# ── _gid_from_hit (@staticmethod) ────────────────────────────────────────────

@pytest.mark.parametrize('hit,expected', [
    (('poly_vertex',    'g1', 3),      'g1'),
    (('poly_edge',      'g2', 0),      'g2'),
    (('poly_translate', 'g3'),         'g3'),
    (('rect_translate', 'g4'),         'g4'),
    (('rect_edge',      'g5', 'top'),  'g5'),
    (('rect_corner',    'g6', 'tl'),   'g6'),
    (('v',              'g7:lo'),      'g7'),
    (('v',              'g7:hi'),      'g7'),
    (('h',              'g8'),         'g8'),
    (('quad_origin',    'g9'),         'g9'),
    (None,                             None),
    ((),                               None),
    (('lonely',),                      None),
    (('weird', 12345),                 None),   # non-string second elem
])
def test_gid_from_hit(hit, expected):
    assert V._gid_from_hit(hit) == expected


# ── _delete_polygon_vertex ───────────────────────────────────────────────────

def _stub_with_gates(gates):
    s = types.SimpleNamespace()
    s._gates = gates
    s._redraw_only_gates = lambda: None
    s._refresh_gate_list = lambda: None
    return s


def test_delete_polygon_vertex_removes_at_index():
    g = {'kind': 'polygon',
         'vertices': [[0, 0], [1, 0], [1, 1], [0, 1]]}
    stub = _stub_with_gates({'g1': g})
    V._delete_polygon_vertex(stub, 'g1', 1)
    assert g['vertices'] == [[0, 0], [1, 1], [0, 1]]


def test_delete_polygon_vertex_refuses_below_three():
    """Polygons with 3 vertices cannot lose another — the gate would
    degenerate to a line. The method should no-op in that case."""
    g = {'kind': 'polygon', 'vertices': [[0, 0], [1, 0], [1, 1]]}
    stub = _stub_with_gates({'g1': g})
    V._delete_polygon_vertex(stub, 'g1', 0)
    assert len(g['vertices']) == 3


def test_delete_polygon_vertex_out_of_range_noop():
    g = {'kind': 'polygon',
         'vertices': [[0, 0], [1, 0], [1, 1], [0, 1]]}
    stub = _stub_with_gates({'g1': g})
    V._delete_polygon_vertex(stub, 'g1', 99)
    assert g['vertices'] == [[0, 0], [1, 0], [1, 1], [0, 1]]


def test_delete_polygon_vertex_wrong_kind_noop():
    g = {'kind': 'rect', 'x0': 0, 'x1': 1, 'y0': 0, 'y1': 1}
    stub = _stub_with_gates({'g1': g})
    V._delete_polygon_vertex(stub, 'g1', 0)
    # 'vertices' shouldn't have been added; nothing should have crashed.
    assert 'vertices' not in g


def test_delete_polygon_vertex_unknown_gid_noop():
    stub = _stub_with_gates({})
    V._delete_polygon_vertex(stub, 'nonexistent', 0)   # must not raise


# ── _insert_polygon_vertex ───────────────────────────────────────────────────

def test_insert_polygon_vertex_basic():
    g = {'kind': 'polygon', 'vertices': [[0, 0], [1, 0], [1, 1]]}
    stub = _stub_with_gates({'g1': g})
    V._insert_polygon_vertex(stub, 'g1', 1, 0.5, -0.5)
    assert g['vertices'] == [[0, 0], [0.5, -0.5], [1, 0], [1, 1]]


def test_insert_polygon_vertex_clamps_idx():
    g = {'kind': 'polygon', 'vertices': [[0, 0], [1, 0], [1, 1]]}
    stub = _stub_with_gates({'g1': g})
    V._insert_polygon_vertex(stub, 'g1', 999, 2.0, 2.0)
    assert g['vertices'][-1] == [2.0, 2.0]
    V._insert_polygon_vertex(stub, 'g1', -5, -1.0, -1.0)
    assert g['vertices'][0] == [-1.0, -1.0]


def test_insert_polygon_vertex_none_coords_noop():
    g = {'kind': 'polygon', 'vertices': [[0, 0], [1, 0], [1, 1]]}
    stub = _stub_with_gates({'g1': g})
    V._insert_polygon_vertex(stub, 'g1', 1, None, 0.0)
    assert g['vertices'] == [[0, 0], [1, 0], [1, 1]]
    V._insert_polygon_vertex(stub, 'g1', 1, 0.0, None)
    assert g['vertices'] == [[0, 0], [1, 0], [1, 1]]


def test_insert_polygon_vertex_wrong_kind_noop():
    g = {'kind': 'rect', 'x0': 0, 'x1': 1, 'y0': 0, 'y1': 1}
    stub = _stub_with_gates({'g1': g})
    V._insert_polygon_vertex(stub, 'g1', 0, 5.0, 5.0)
    assert 'vertices' not in g


# ── _polygon_under_point ─────────────────────────────────────────────────────

class _ComboStub:
    def __init__(self, value): self.value = value
    def get(self): return self.value


def _stub_with_polys(polys, x_ch='FSC-A', y_ch='SSC-A'):
    s = types.SimpleNamespace()
    s._gates = polys
    s.x_combo = _ComboStub(x_ch)
    s.y_combo = _ComboStub(y_ch)
    s._resolve_channel = lambda ch: ch
    return s


def test_polygon_under_point_containment_wins():
    square = {'kind': 'polygon',
              'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
              'vertices': [[0, 0], [10, 0], [10, 10], [0, 10]]}
    far = {'kind': 'polygon',
           'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
           'vertices': [[100, 100], [110, 100], [105, 110]]}
    stub = _stub_with_polys({'g1': square, 'g2': far})
    assert V._polygon_under_point(stub, 5, 5) == 'g1'


def test_polygon_under_point_nearest_fallback():
    """When no polygon contains the point, fall back to the nearest
    polygon by vertex distance."""
    p1 = {'kind': 'polygon',
          'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
          'vertices': [[0, 0], [10, 0], [10, 10], [0, 10]]}
    p2 = {'kind': 'polygon',
          'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
          'vertices': [[100, 100], [110, 100], [110, 110], [100, 110]]}
    stub = _stub_with_polys({'g1': p1, 'g2': p2})
    # (50, 50) is between both; p1's [10,10] is much closer than p2's [100,100]
    assert V._polygon_under_point(stub, 50, 50) == 'g1'


def test_polygon_under_point_skips_wrong_channels():
    """A polygon on different channels than the current view must not match."""
    p_other = {'kind': 'polygon',
               'x_channel': 'OTHER-A', 'y_channel': 'SSC-A',
               'vertices': [[0, 0], [10, 0], [10, 10], [0, 10]]}
    stub = _stub_with_polys({'g1': p_other})
    assert V._polygon_under_point(stub, 5, 5) is None


def test_polygon_under_point_no_polygons_among_gates():
    rect = {'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
            'x0': 0, 'x1': 10, 'y0': 0, 'y1': 10}
    stub = _stub_with_polys({'g1': rect})
    assert V._polygon_under_point(stub, 5, 5) is None


def test_polygon_under_point_none_coords():
    stub = _stub_with_polys({})
    assert V._polygon_under_point(stub, None, 0.5) is None
    assert V._polygon_under_point(stub, 0.5, None) is None


def test_polygon_under_point_degenerate_polygon_ignored():
    """A polygon with fewer than 3 vertices is undisplayable; skip it
    rather than letting it match an inappropriate click."""
    bad = {'kind': 'polygon',
           'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
           'vertices': [[0, 0], [1, 0]]}
    stub = _stub_with_polys({'g1': bad})
    assert V._polygon_under_point(stub, 0.5, 0.5) is None


# ── _smallest_loaded_sample_size ─────────────────────────────────────────────

class _SampleStub:
    def __init__(self, n): self.data = list(range(n))


def test_smallest_loaded_sample_size_basic():
    stub = types.SimpleNamespace()
    stub._sample_order = ['a', 'b', 'c']
    stub._samples = {'a': _SampleStub(100), 'b': _SampleStub(50),
                     'c': _SampleStub(200)}
    stub._sample_plot_enabled = {'a': True, 'b': True, 'c': True}
    assert V._smallest_loaded_sample_size(stub) == 50


def test_smallest_loaded_sample_size_skips_disabled():
    """Samples whose plot toggle is off must not pin the floor —
    otherwise turning a sample off would silently re-downsample
    every other plot."""
    stub = types.SimpleNamespace()
    stub._sample_order = ['a', 'b']
    stub._samples = {'a': _SampleStub(100), 'b': _SampleStub(50)}
    stub._sample_plot_enabled = {'a': True, 'b': False}
    assert V._smallest_loaded_sample_size(stub) == 100


def test_smallest_loaded_sample_size_none_when_empty():
    stub = types.SimpleNamespace()
    stub._sample_order = []
    stub._samples = {}
    stub._sample_plot_enabled = {}
    assert V._smallest_loaded_sample_size(stub) is None


def test_smallest_loaded_sample_size_none_when_all_disabled():
    stub = types.SimpleNamespace()
    stub._sample_order = ['a']
    stub._samples = {'a': _SampleStub(100)}
    stub._sample_plot_enabled = {'a': False}
    assert V._smallest_loaded_sample_size(stub) is None


# ── _apply_axis_to_ax ────────────────────────────────────────────────────────
# These need a real matplotlib axes — cheap under the Agg backend.

def _axis_stub(scales=None, ranges=None, transforms=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    s = types.SimpleNamespace()
    s.ax = ax
    s._channel_scale = dict(scales or {})
    s._channel_range = dict(ranges or {})
    s._channel_transform = dict(transforms or {})
    s._default_channel_scale = 'symlog'
    # _apply_axis_to_ax delegates to _axis_view_funcs; bind it to the stub
    # so the unbound-method call resolves. With an empty _channel_transform
    # every channel is 'linear' → returns None → native mpl scales.
    s._axis_view_funcs = lambda ch, ds=None: V._axis_view_funcs(s, ch, ds)
    s._symlog_linthresh = V._symlog_linthresh   # staticmethod, used by symlog
    # _apply_axis_to_ax now resolves an embedding-aware default scale.
    s._EMBED_AXIS_PREFIXES = V._EMBED_AXIS_PREFIXES
    s._default_scale_for = lambda ch: V._default_scale_for(s, ch)
    return s, fig


def test_apply_axis_to_ax_linear():
    import matplotlib.pyplot as plt
    stub, fig = _axis_stub(scales={'X': 'linear'})
    V._apply_axis_to_ax(stub, 'X', 'x', None)
    assert stub.ax.get_xscale() == 'linear'
    plt.close(fig)


def test_apply_axis_to_ax_log():
    import matplotlib.pyplot as plt
    stub, fig = _axis_stub(scales={'X': 'log'})
    V._apply_axis_to_ax(stub, 'X', 'x', None)
    assert stub.ax.get_xscale() == 'log'
    plt.close(fig)


def test_apply_axis_to_ax_symlog_with_data():
    """Symlog with a data sample picks linthresh from the 5th
    percentile of |nonzero|. Just verify the scale was set and a
    finite linthresh resulted."""
    import matplotlib.pyplot as plt
    stub, fig = _axis_stub(scales={'X': 'symlog'})
    arr = np.concatenate([
        np.linspace(-1000, -1, 100),
        np.linspace(1, 1000, 100)])
    V._apply_axis_to_ax(stub, 'X', 'x', arr)
    assert stub.ax.get_xscale() == 'symlog'
    plt.close(fig)


def test_apply_axis_to_ax_symlog_no_data_uses_fallback():
    """Symlog with no data sample falls back to linthresh=1.0; should
    not raise."""
    import matplotlib.pyplot as plt
    stub, fig = _axis_stub(scales={'X': 'symlog'})
    V._apply_axis_to_ax(stub, 'X', 'x', None)
    assert stub.ax.get_xscale() == 'symlog'
    plt.close(fig)


def test_apply_axis_to_ax_default_when_channel_unset():
    """Channels not explicitly configured use _default_channel_scale."""
    import matplotlib.pyplot as plt
    stub, fig = _axis_stub(scales={})
    V._apply_axis_to_ax(stub, 'UNKNOWN', 'y', None)
    assert stub.ax.get_yscale() == 'symlog'   # default
    plt.close(fig)


def test_apply_axis_to_ax_custom_range_applied():
    import matplotlib.pyplot as plt
    stub, fig = _axis_stub(scales={'X': 'linear'},
                           ranges={'X': (10.0, 50.0)})
    V._apply_axis_to_ax(stub, 'X', 'x', None)
    lo, hi = stub.ax.get_xlim()
    assert lo == 10.0 and hi == 50.0
    plt.close(fig)


# ── composite FuncScale views (nonlinear-baked channels) ────────────────────
# A channel whose data is baked into logicle space renders linear/symlog/log
# as proper VIEWS of the underlying linear intensity via matplotlib's
# 'function' scale — no double-transform, and each scale differs.

@pytest.mark.parametrize('scale', ['linear', 'symlog', 'log'])
def test_apply_axis_to_ax_logicle_channel_uses_funcscale(scale):
    import matplotlib.pyplot as plt

    from openflo.pipeline import transform_values
    stub, fig = _axis_stub(scales={'FL1-A': scale},
                           transforms={'FL1-A': 'logicle'})
    # Baked logicle data (~0..1) spanning a realistic intensity range.
    lin = np.array([-50.0, 0, 10, 100, 1000, 10000, 100000, 250000])
    data = transform_values(lin, method='logicle')
    V._apply_axis_to_ax(stub, 'FL1-A', 'x', data)
    # Nonlinear-baked channels render via the composite FuncScale, which
    # matplotlib reports as the 'function' scale (not log/symlog/linear).
    assert stub.ax.get_xscale() == 'function'
    stub.ax.scatter(data, np.arange(len(data)))
    fig.canvas.draw()          # must not raise (shape-preserving funcs)
    plt.close(fig)


def test_apply_axis_to_ax_logicle_views_differ():
    """linear vs symlog vs log of the same logicle channel must map data
    to genuinely different screen positions (the regression: they were
    all collapsing to an identical linear axis)."""
    import matplotlib.pyplot as plt

    from openflo.pipeline import transform_values
    lin = np.array([1.0, 100.0, 10000.0])
    data = transform_values(lin, method='logicle')
    screens = {}
    for scale in ('linear', 'symlog', 'log'):
        stub, fig = _axis_stub(scales={'FL1-A': scale},
                               transforms={'FL1-A': 'logicle'})
        fwd, _inv = stub._axis_view_funcs('FL1-A', data)
        screens[scale] = np.asarray(fwd(data), dtype=float).ravel()
        plt.close(fig)
    # No two views should produce the same screen mapping.
    assert not np.allclose(screens['linear'], screens['symlog'])
    assert not np.allclose(screens['linear'], screens['log'])
    assert not np.allclose(screens['symlog'], screens['log'])


# ── _hist_bin_edges (@staticmethod) ──────────────────────────────────────────

def test_hist_bin_edges_linear_count():
    edges = V._hist_bin_edges(0.0, 10.0, 'linear', n_bins=200)
    assert len(edges) == 201
    assert edges[0] == 0.0
    assert edges[-1] == 10.0
    # Linear: equal spacing
    diffs = np.diff(edges)
    assert np.allclose(diffs, diffs[0])


def test_hist_bin_edges_symlog_uses_linear_spacing():
    """Symlog's display transform is linear-near-zero; using log-spaced
    bins would over-compress the centre. We deliberately stay linear."""
    edges_sym = V._hist_bin_edges(-100.0, 100.0, 'symlog', n_bins=200)
    edges_lin = V._hist_bin_edges(-100.0, 100.0, 'linear', n_bins=200)
    np.testing.assert_allclose(edges_sym, edges_lin)


def test_hist_bin_edges_log_positive_range():
    edges = V._hist_bin_edges(1.0, 10_000.0, 'log', n_bins=200)
    assert len(edges) == 201
    assert edges[0] == pytest.approx(1.0)
    assert edges[-1] == pytest.approx(10_000.0)
    # Log spacing: the RATIOS between consecutive edges are constant.
    ratios = np.array(edges[1:]) / np.array(edges[:-1])
    assert np.allclose(ratios, ratios[0], rtol=1e-9)


def test_hist_bin_edges_log_clamps_nonpositive_lo():
    """A channel with negative or zero values still needs to produce
    a usable histogram on a log axis — the low edge is clamped to a
    small positive floor rather than crashing."""
    edges = V._hist_bin_edges(-5.0, 100.0, 'log', n_bins=200)
    assert len(edges) == 201
    assert edges[0] > 0, "log scale requires positive low edge"
    assert edges[-1] == pytest.approx(100.0)


def test_hist_bin_edges_log_falls_back_when_degenerate():
    """If clamping the low edge eats the whole range (hi <= lo_pos),
    fall back to linear spacing rather than emit a 0-bin histogram."""
    edges = V._hist_bin_edges(-1.0, -0.5, 'log', n_bins=200)
    assert len(edges) == 201
    # Linear fallback — equal spacing.
    diffs = np.diff(edges)
    assert np.allclose(diffs, diffs[0])


def test_hist_bin_edges_returns_python_list():
    """Matplotlib's bins= stub declares Sequence[float], not ndarray."""
    edges = V._hist_bin_edges(0.0, 1.0, 'linear', n_bins=10)
    assert isinstance(edges, list)
    assert all(isinstance(e, float) for e in edges)


def test_apply_axis_to_ax_auto_range_does_not_pin():
    """No entry in _channel_range means matplotlib's autoscale should
    own the limits — we must NOT call set_xlim."""
    import matplotlib.pyplot as plt
    stub, fig = _axis_stub(scales={'X': 'linear'}, ranges={})
    stub.ax.set_xlim(7, 13)                # pre-existing limits
    V._apply_axis_to_ax(stub, 'X', 'x', None)
    lo, hi = stub.ax.get_xlim()
    assert (lo, hi) == (7, 13)
    plt.close(fig)


# ── _ellipse_params / _ellipse_geom (ellipsoid render + edit geometry) ───────

def test_ellipse_params_axis_aligned():
    """Axis-aligned cov diag(4,1) at distance_sq=1 → semi-axes 2 and 1,
    so the full width/height are {4, 2}; angle is a multiple of 90°."""
    gate = {'kind': 'ellipsoid', 'x_channel': 'X', 'y_channel': 'Y',
            'mean': [0.0, 0.0], 'cov': [[4.0, 0.0], [0.0, 1.0]],
            'distance_sq': 1.0}
    params = V._ellipse_params(gate)
    assert params is not None
    cx, cy, w, h, angle = params
    assert (cx, cy) == (0.0, 0.0)
    assert {round(w, 6), round(h, 6)} == {4.0, 2.0}
    assert abs(angle % 90.0) < 1e-6 or abs(angle % 90.0 - 90.0) < 1e-6


def test_ellipse_params_distance_scales_size():
    """distance_sq=4 doubles the linear size vs distance_sq=1."""
    base = {'kind': 'ellipsoid', 'x_channel': 'X', 'y_channel': 'Y',
            'mean': [0.0, 0.0], 'cov': [[1.0, 0.0], [0.0, 1.0]]}
    p1 = V._ellipse_params({**base, 'distance_sq': 1.0})
    p4 = V._ellipse_params({**base, 'distance_sq': 4.0})
    assert p1 is not None and p4 is not None
    # width at dist=4 is 2× width at dist=1 (sqrt(4)=2).
    assert p4[2] == pytest.approx(2.0 * p1[2])
    assert p4[3] == pytest.approx(2.0 * p1[3])


def test_ellipse_params_degenerate_returns_none():
    for bad in (
        {'mean': [0, 0], 'cov': [[0, 0], [0, 0]], 'distance_sq': 1.0},   # singular
        {'mean': [0, 0, 0], 'cov': [[1, 0], [0, 1]], 'distance_sq': 1.0},  # 3-vec mean
        {'mean': [0, 0], 'cov': [[1, 0], [0, 1]], 'distance_sq': 0.0},   # zero radius
    ):
        gate = {'kind': 'ellipsoid', 'x_channel': 'X', 'y_channel': 'Y', **bad}
        assert V._ellipse_params(gate) is None


def test_ellipse_geom_inverse_and_radius():
    gate = {'kind': 'ellipsoid', 'x_channel': 'X', 'y_channel': 'Y',
            'mean': [10.0, 20.0], 'cov': [[4.0, 0.0], [0.0, 9.0]],
            'distance_sq': 4.0}
    geom = V._ellipse_geom(gate)
    assert geom is not None
    (mx, my), inv, r0, (hx, hy) = geom
    assert (mx, my) == (10.0, 20.0)
    assert r0 == pytest.approx(2.0)              # sqrt(distance_sq)
    # inv must invert cov
    np.testing.assert_allclose(inv, np.linalg.inv([[4.0, 0.0], [0.0, 9.0]]))
    # The rotation handle sits OUTSIDE the rim (Mahalanobis radius > r0).
    d = np.array([hx - mx, hy - my])
    md = float(np.sqrt(d @ inv @ d))
    assert md > r0


def test_ellipse_geom_degenerate_returns_none():
    gate = {'kind': 'ellipsoid', 'x_channel': 'X', 'y_channel': 'Y',
            'mean': [0.0, 0.0], 'cov': [[0.0, 0.0], [0.0, 0.0]],
            'distance_sq': 1.0}
    assert V._ellipse_geom(gate) is None


# ── Batch template application helpers ───────────────────────────────────────

def test_gate_channels_extraction():
    assert V._gate_channels({'kind': 'threshold', 'channel': 'BV421-A'}) == {'BV421-A'}
    assert V._gate_channels(
        {'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A'}) == {'FSC-A', 'SSC-A'}
    assert V._gate_channels(
        {'kind': 'ellipsoid', 'x_channel': 'APC-A', 'y_channel': 'PE-Cy7-A'}) == {'APC-A', 'PE-Cy7-A'}
    assert V._gate_channels({'kind': 'polygon'}) == set()   # no channels set


class _SampleCols:
    def __init__(self, cols):
        self.data = types.SimpleNamespace(columns=list(cols))


def _mismatch_stub(samples):
    # _count_channel_mismatches calls self._gate_channels (a staticmethod);
    # give the stub that bound reference.
    return types.SimpleNamespace(_samples=samples,
                                 _gate_channels=V._gate_channels)


def test_count_channel_mismatches_all_present():
    stub = _mismatch_stub({'A': _SampleCols(['FSC-A', 'SSC-A', 'BV421-A'])})
    gates = [
        {'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A'},
        {'kind': 'threshold', 'channel': 'BV421-A'},
    ]
    assert V._count_channel_mismatches(stub, 'A', gates) == 0


def test_count_channel_mismatches_flags_absent_channels():
    stub = _mismatch_stub({'A': _SampleCols(['FSC-A', 'SSC-A'])})
    gates = [
        {'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A'},   # ok
        {'kind': 'threshold', 'channel': 'BV421-A'},                    # missing
        {'kind': 'ellipsoid', 'x_channel': 'APC-A', 'y_channel': 'SSC-A'},  # APC-A missing
    ]
    assert V._count_channel_mismatches(stub, 'A', gates) == 2


def test_count_channel_mismatches_unknown_sample_is_zero():
    stub = _mismatch_stub({})
    gates = [{'kind': 'threshold', 'channel': 'X'}]
    assert V._count_channel_mismatches(stub, 'nope', gates) == 0
