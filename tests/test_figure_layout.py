"""Headless tests for the multi-panel figure-layout helpers
(ViewGateEditorWindow._parse_pairs_report / _resolve_token_to_channel /
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
        _channels=['FSC-A', 'SSC-A', 'FL1-A', 'FL3-A', 'FL2-A'],
        _channel_labels={'FL1-A': 'CD901b', 'FL3-A': 'CD903',
                         'FL2-A': 'CD902'},
    )
    stub._resolve_token_to_channel = types.MethodType(
        W._resolve_token_to_channel, stub)
    stub._parse_pairs_report = types.MethodType(W._parse_pairs_report, stub)
    stub._pair_splits = W._pair_splits
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
    assert s._resolve_token_to_channel('CD902') == 'FL2-A'
    assert s._resolve_token_to_channel('cd903') == 'FL3-A'  # case-insensitive


def test_resolve_token_label_paren_form():
    s = _chan_stub()
    assert s._resolve_token_to_channel('CD901b (FL1-A)') == 'FL1-A'


def test_resolve_token_unknown():
    s = _chan_stub()
    assert s._resolve_token_to_channel('nonsense') is None
    assert s._resolve_token_to_channel('') is None


def test_parse_pairs_markers_and_channels():
    s = _chan_stub()
    pairs = s._parse_pairs_report('CD902/CD901b, CD901b/CD903')[0]
    assert pairs == [('FL2-A', 'FL1-A'), ('FL1-A', 'FL3-A')]


def test_parse_pairs_separators():
    s = _chan_stub()
    # Accept '/', 'x', '×', 'vs' and comma/semicolon/newline delimiters.
    pairs = s._parse_pairs_report('FSC-A x SSC-A; CD902 vs CD903')[0]
    assert pairs == [('FSC-A', 'SSC-A'), ('FL2-A', 'FL3-A')]


def test_parse_pairs_skips_unresolvable():
    s = _chan_stub()
    pairs, bad = s._parse_pairs_report('CD902/CD901b, junk/CD903, CD903')
    assert pairs == [('FL2-A', 'FL1-A')]
    assert bad == ["'junk/CD903': no channel 'junk'",
                   "'CD903': not a pair (write X/Y)"]


def test_parse_pairs_x_without_spaces_and_names_holding_an_x():
    s = _chan_stub()
    s._channels = s._channels + ['FLX-A']
    s._channel_labels = dict(s._channel_labels, **{'FLX-A': 'Low X High'})
    # A bare x splits where both sides are channels, not inside FLX-A.
    assert s._parse_pairs_report('FSC-AxSSC-A')[0] == [('FSC-A', 'SSC-A')]
    assert s._parse_pairs_report('FLX-AxSSC-A')[0] == [('FLX-A', 'SSC-A')]
    assert s._parse_pairs_report('SSC-AXFLX-A')[0] == [('SSC-A', 'FLX-A')]
    # A label with ' X ' in it stays whole; the split is at the next x.
    assert s._parse_pairs_report('Low X High x FSC-A')[0] == [
        ('FLX-A', 'FSC-A')]
    # One channel with an x in its name is not a pair, and part of a name
    # ('FL' of FLX-A) does not count as a channel on a bare x.
    assert s._parse_pairs_report('FLX-A') == (
        [], ["'FLX-A': not a pair (write X/Y)"])
    assert s._parse_pairs_report('FSC-AxJunk') == (
        [], ["'FSC-AxJunk': no channel 'Junk'"])


def _build_stub():
    """Stub for _build_layout_figure: record _render_into calls, real Figure."""
    calls = []

    stub = types.SimpleNamespace()

    def fake_render(ax, samples, x, y, mode, color, draw_gates=True):
        calls.append({'samples': samples, 'x': x, 'y': y, 'mode': mode,
                      'color': color, 'draw_gates': draw_gates})
        ax.plot([0, 1], [0, 1])

    stub._render_into = fake_render
    stub._build_layout_figure = types.MethodType(W._build_layout_figure, stub)
    return stub, calls


def test_build_layout_figure_grid_shape():
    stub, calls = _build_stub()
    panels = [
        {'samples': ['s1'], 'x': 'FSC-A', 'y': 'SSC-A', 'mode': 'dot',
         'color': 'By density', 'title': 's1'},
        {'samples': ['s2'], 'x': 'FSC-A', 'y': 'SSC-A', 'mode': 'dot',
         'color': 'By density', 'title': 's2'},
        {'samples': ['s3'], 'x': 'FSC-A', 'y': 'SSC-A', 'mode': 'dot',
         'color': 'By density', 'title': 's3'},
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
    panels = [{'samples': ['s'], 'x': 'FSC-A', 'y': 'SSC-A', 'mode': 'dot',
               'color': 'By density', 'title': None}]
    # ncols larger than panel count is clamped down; single panel -> 1 axis.
    fig = stub._build_layout_figure(panels, ncols=8)
    assert len(fig.axes) == 1
