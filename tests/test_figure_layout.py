"""Headless tests for the multi-panel figure-layout helpers
(ViewGateEditorWindow._parse_pairs_str / _resolve_token_to_channel /
_short_sample / _build_layout_figure). Stub `self` so no Tk display is
needed; matplotlib runs on the Agg backend."""
from __future__ import annotations

import types

import matplotlib

matplotlib.use('Agg')

from openflo.gui import ViewGateEditorWindow as W


def _chan_stub():
    """A minimal stub carrying the channel/label state the resolver needs,
    with the resolver methods bound to it."""
    stub = types.SimpleNamespace(
        _channels=['FSC-A', 'SSC-A', 'BV421-A', 'PE-Cy7-A', 'APC-A'],
        _channel_labels={'BV421-A': 'CD11b', 'PE-Cy7-A': 'CD45',
                         'APC-A': 'CD34'},
    )
    stub._resolve_channel = types.MethodType(W._resolve_channel, stub)
    stub._resolve_token_to_channel = types.MethodType(
        W._resolve_token_to_channel, stub)
    stub._parse_pairs_str = types.MethodType(W._parse_pairs_str, stub)
    return stub


def test_short_sample():
    assert W._short_sample('day3') == 'day3'
    long = 'a' * 40
    out = W._short_sample(long, width=10)
    assert len(out) == 10
    assert out.endswith('…')


def test_resolve_token_exact_channel():
    s = _chan_stub()
    assert s._resolve_token_to_channel('FSC-A') == 'FSC-A'


def test_resolve_token_by_marker_label():
    s = _chan_stub()
    assert s._resolve_token_to_channel('CD34') == 'APC-A'
    assert s._resolve_token_to_channel('cd45') == 'PE-Cy7-A'  # case-insensitive


def test_resolve_token_label_paren_form():
    s = _chan_stub()
    assert s._resolve_token_to_channel('CD11b (BV421-A)') == 'BV421-A'


def test_resolve_token_unknown():
    s = _chan_stub()
    assert s._resolve_token_to_channel('nonsense') is None
    assert s._resolve_token_to_channel('') is None


def test_parse_pairs_markers_and_channels():
    s = _chan_stub()
    pairs = s._parse_pairs_str('CD34/CD11b, CD11b/CD45')
    assert pairs == [('APC-A', 'BV421-A'), ('BV421-A', 'PE-Cy7-A')]


def test_parse_pairs_separators():
    s = _chan_stub()
    # Accept '/', 'x', '×', 'vs' and comma/semicolon/newline delimiters.
    pairs = s._parse_pairs_str('FSC-A x SSC-A; CD34 vs CD45')
    assert pairs == [('FSC-A', 'SSC-A'), ('APC-A', 'PE-Cy7-A')]


def test_parse_pairs_skips_unresolvable():
    s = _chan_stub()
    pairs = s._parse_pairs_str('CD34/CD11b, junk/CD45, CD45')
    assert pairs == [('APC-A', 'BV421-A')]


def _build_stub():
    """Stub for _build_layout_figure: record _render_into calls, real Figure."""
    calls = []

    stub = types.SimpleNamespace()

    def fake_render(ax, samples, x, y, mode, color, draw_gates=True):
        calls.append(dict(samples=samples, x=x, y=y, mode=mode,
                          color=color, draw_gates=draw_gates))
        ax.plot([0, 1], [0, 1])

    stub._render_into = fake_render
    stub._build_layout_figure = types.MethodType(W._build_layout_figure, stub)
    return stub, calls


def test_build_layout_figure_grid_shape():
    stub, calls = _build_stub()
    panels = [
        dict(samples=['s1'], x='FSC-A', y='SSC-A', mode='dot',
             color='By density', title='s1'),
        dict(samples=['s2'], x='FSC-A', y='SSC-A', mode='dot',
             color='By density', title='s2'),
        dict(samples=['s3'], x='FSC-A', y='SSC-A', mode='dot',
             color='By density', title='s3'),
    ]
    fig = stub._build_layout_figure(panels, ncols=2, draw_gates=False)
    assert fig is not None
    # 3 panels, 2 columns -> 2 rows -> 3 axes drawn.
    assert len(fig.axes) == 3
    assert len(calls) == 3
    assert all(c['draw_gates'] is False for c in calls)
    # Titles propagate.
    assert fig.axes[0].get_title() == 's1'


def test_build_layout_figure_empty():
    stub, _ = _build_stub()
    assert stub._build_layout_figure([], ncols=3) is None


def test_build_layout_figure_ncols_clamped():
    stub, _ = _build_stub()
    panels = [dict(samples=['s'], x='FSC-A', y='SSC-A', mode='dot',
                   color='By density', title=None)]
    # ncols larger than panel count is clamped down; single panel -> 1 axis.
    fig = stub._build_layout_figure(panels, ncols=8)
    assert len(fig.axes) == 1
