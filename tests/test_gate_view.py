"""Go to a gate's plot (gate list double-click / right-click entry).

Selecting a gate row keeps the plot as it is; the explicit action switches
the plot to the gate's channels (and a 2-D mode for a 2-D gate) so the gate
is drawn, and drops a zoom that leaves part of it out. Generated data and
generic channels (FSC-A, SSC-A, FL1-A, FL2-A) only; the editor stays
withdrawn.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from openflo.gate_view import (
    DEFAULT_2D_MODE,
    gate_bounds,
    gate_plot_channels,
    outside_view,
    view_for_gate,
    view_limits,
)
from tests.conftest import (
    editor_or_skip,
    invoke_menu_path,
    load_menu_sample,
    menu_labels,
    right_click_row,
)

RECT = {'kind': 'rect', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
        'x0': 100.0, 'x1': 400.0, 'y0': 200.0, 'y1': 600.0}
THRESH = {'kind': 'threshold', 'channel': 'FL1-A', 'op': '>', 'value': 500.0}
INTERVAL = {'kind': 'interval', 'channel': 'FL2-A', 'lo': 10.0, 'hi': 1e12}
POLY = {'kind': 'polygon', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
        'vertices': [[1.0, 2.0], [5.0, 9.0], [3.0, -4.0]]}
ELLIPSE = {'kind': 'ellipsoid', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
           'mean': [10.0, 20.0], 'cov': [[4.0, 0.0], [0.0, 9.0]],
           'distance_sq': 4.0}
BOOL = {'kind': 'boolean', 'op': 'and', 'operands': ['g1', 'g2']}


# ── the pure rules ──────────────────────────────────────────────────────

def test_gate_channels_by_kind():
    assert gate_plot_channels(RECT) == ('FL1-A', 'FL2-A')
    assert gate_plot_channels(THRESH) == ('FL1-A', None)
    assert gate_plot_channels(ELLIPSE) == ('FL1-A', 'FL2-A')
    assert gate_plot_channels(BOOL) is None
    assert gate_plot_channels({'kind': 'rect', 'x_channel': 'FL1-A'}) is None


def test_two_d_gate_takes_its_own_axes_in_order():
    # Swapped axes are not the gate's view: the shape is drawn only with
    # its x on X.
    assert view_for_gate(RECT, 'FL2-A', 'FL1-A', 'dot') == (
        'FL1-A', 'FL2-A', 'dot')
    assert view_for_gate(RECT, 'FSC-A', 'SSC-A', 'contour') == (
        'FL1-A', 'FL2-A', 'contour')


def test_two_d_gate_leaves_a_histogram():
    assert view_for_gate(RECT, 'FL1-A', 'FL2-A', 'histogram') == (
        'FL1-A', 'FL2-A', DEFAULT_2D_MODE)
    assert DEFAULT_2D_MODE != 'histogram'


def test_one_d_gate_keeps_a_view_that_already_draws_it():
    assert view_for_gate(THRESH, 'FL1-A', 'SSC-A', 'histogram') == (
        'FL1-A', 'SSC-A', 'histogram')
    # On Y of a 2-D plot it is drawn as a horizontal line.
    assert view_for_gate(THRESH, 'FSC-A', 'FL1-A', 'dot') == (
        'FSC-A', 'FL1-A', 'dot')


def test_one_d_gate_goes_on_x_otherwise():
    assert view_for_gate(THRESH, 'FSC-A', 'SSC-A', 'pseudocolor') == (
        'FL1-A', 'SSC-A', 'pseudocolor')
    # A histogram never draws Y: the channel goes on X, and the old X
    # takes Y so the two never match.
    assert view_for_gate(THRESH, 'FSC-A', 'FL1-A', 'histogram') == (
        'FL1-A', 'FSC-A', 'histogram')


def test_no_view_for_a_gate_the_plot_never_draws():
    assert view_for_gate(BOOL, 'FSC-A', 'SSC-A', 'dot') is None


def test_bounds_per_kind_leave_open_sides_out():
    assert gate_bounds(RECT) == {'FL1-A': (100.0, 400.0),
                                 'FL2-A': (200.0, 600.0)}
    assert gate_bounds(THRESH) == {'FL1-A': (500.0, 500.0)}
    assert gate_bounds(INTERVAL) == {'FL2-A': (10.0, 10.0)}
    assert gate_bounds(dict(INTERVAL, hi=70.0)) == {'FL2-A': (10.0, 70.0)}
    assert gate_bounds(POLY) == {'FSC-A': (1.0, 5.0), 'SSC-A': (-4.0, 9.0)}
    # Semi-axes sqrt(4 * 4) = 4 and sqrt(4 * 9) = 6.
    assert gate_bounds(ELLIPSE) == {'FL1-A': (6.0, 14.0),
                                    'FL2-A': (14.0, 26.0)}
    assert gate_bounds(BOOL) == {}
    open_rect = dict(RECT, x1=1e12)
    assert gate_bounds(open_rect)['FL1-A'] == (100.0, 100.0)


def test_outside_view():
    b = {'FL1-A': (100.0, 400.0), 'FL2-A': (200.0, 600.0)}
    assert not outside_view(b, {'FL1-A': (0, 500), 'FL2-A': (650, 100)})
    assert outside_view(b, {'FL1-A': (150, 500)})       # low side cut
    assert outside_view(b, {'FL2-A': (0, 599)})         # high side cut
    assert not outside_view(b, {'FSC-A': (0, 1)})       # other channel


# ── the editor ──────────────────────────────────────────────────────────

@pytest.fixture
def editor():
    """s1 and s2 loaded (a threshold gate on FL1-A each); s1 also has a
    rect gate on FL1-A x FL2-A and a boolean. The plot shows FSC-A x SSC-A
    as a histogram; s2 is active."""
    root, ed = editor_or_skip(accept_confirms=False)
    try:
        t1 = load_menu_sample(ed, 's1', 1)
        ed._set_active_sample('s1')
        rect = ed._add_gate(dict(RECT, name='Box', parent_id=None),
                            audit=False)
        both = ed._add_gate(dict(BOOL, name='Both', operands=[t1, rect],
                                 parent_id=None), audit=False)
        t2 = load_menu_sample(ed, 's2', 2)
        ed._refresh_gate_list()
        ed.x_combo.set('FSC-A')
        ed.y_combo.set('SSC-A')
        ed.mode_var.set('histogram')
        ed._update_mode_options()
        ed._sample_plot_enabled['s1'] = False
        replots = []
        ed._schedule_replot = lambda delay_ms=100: replots.append(delay_ms)
        yield ed, {'t1': t1, 'rect': rect, 'both': both, 't2': t2}, replots
    finally:
        root.destroy()


def _view(ed):
    return ed.x_combo.get(), ed.y_combo.get(), ed.mode_var.get()


def _double_click(ed, monkeypatch, iid, column='#0'):
    monkeypatch.setattr(ed.gate_tv, 'identify_row', lambda _y: iid)
    monkeypatch.setattr(ed.gate_tv, 'identify_column', lambda _x: column)
    return ed._on_tv_double_click(SimpleNamespace(x=5, y=5))


def test_selecting_a_gate_keeps_the_plot_view(editor):
    from openflo.tree_ids import gate_iid
    ed, gids, _ = editor
    ed.gate_tv.selection_set(gate_iid('s1', gids['rect']))
    ed._on_tree_select()
    assert ed._active_sample == 's1'
    assert _view(ed) == ('FSC-A', 'SSC-A', 'histogram')


def test_double_click_goes_to_the_gates_plot(editor, monkeypatch):
    from openflo.tree_ids import gate_iid
    ed, gids, replots = editor
    assert _double_click(ed, monkeypatch,
                         gate_iid('s1', gids['rect'])) == 'break'
    assert _view(ed) == ('FL1-A', 'FL2-A', DEFAULT_2D_MODE)
    # The gate belongs to s1: it is the sample the gates come from, and it
    # is shown, so its gate is drawn.
    assert ed._active_sample == 's1'
    assert gids['rect'] in ed._gates
    assert ed._sample_plot_enabled['s1'] is True
    assert replots == [0]
    assert 'Box' in ed.status_var.get()


def test_double_click_on_the_on_off_column_keeps_the_view(editor,
                                                          monkeypatch):
    from openflo.tree_ids import gate_iid
    ed, gids, _ = editor
    assert _double_click(ed, monkeypatch, gate_iid('s1', gids['rect']),
                         column='#1') is None
    assert _view(ed) == ('FSC-A', 'SSC-A', 'histogram')


def test_one_d_gate_goes_on_x_and_keeps_the_mode(editor, monkeypatch):
    from openflo.tree_ids import gate_iid
    ed, gids, _ = editor
    _double_click(ed, monkeypatch, gate_iid('s2', gids['t2']))
    assert _view(ed) == ('FL1-A', 'SSC-A', 'histogram')


def test_boolean_gate_has_no_plot_to_go_to(editor, monkeypatch, menu_popups):
    from openflo.tree_ids import gate_iid
    ed, gids, replots = editor
    iid = gate_iid('s1', gids['both'])
    _double_click(ed, monkeypatch, iid)
    assert _view(ed) == ('FSC-A', 'SSC-A', 'histogram')
    assert replots == []
    assert 'no plot axes' in ed.status_var.get()
    menu = right_click_row(ed, monkeypatch, menu_popups, iid)
    i = menu_labels(menu).index("Go to gate's plot (double-click)")
    entries = [k for k in range(menu.index('end') + 1)
               if menu.type(k) != 'separator']
    assert menu.entrycget(entries[i], 'state') == 'disabled'


def test_menu_entry_goes_to_the_gates_plot(editor, monkeypatch, menu_popups):
    from openflo.tree_ids import gate_iid
    ed, gids, _ = editor
    menu = right_click_row(ed, monkeypatch, menu_popups,
                           gate_iid('s1', gids['rect']))
    invoke_menu_path(menu, "Go to gate's plot (double-click)")
    assert _view(ed) == ('FL1-A', 'FL2-A', DEFAULT_2D_MODE)


def test_channel_not_loaded_keeps_the_view(editor, monkeypatch):
    from openflo.tree_ids import gate_iid
    ed, gids, replots = editor
    ed._sample_gates['s1'][gids['rect']]['y_channel'] = 'FL9-A'
    _double_click(ed, monkeypatch, gate_iid('s1', gids['rect']))
    assert _view(ed) == ('FSC-A', 'SSC-A', 'histogram')
    assert replots == []
    assert 'FL9-A' in ed.status_var.get()


def _on_gate_view(ed, xlim, ylim):
    ed.x_combo.set('FL1-A')
    ed.y_combo.set('FL2-A')
    ed.mode_var.set('dot')
    ed.ax.set_xlim(*xlim)
    ed.ax.set_ylim(*ylim)
    ed._kept_zoom = {'key': 'kept', 'xlim': list(xlim), 'ylim': list(ylim)}


@pytest.mark.parametrize('xlim, ylim', [
    ((0, 1000), (300, 1000)),                        # y0 = 200 is cut off
    ((150, 1000), (0, 1000)),                        # x0 = 100 is cut off
])
def test_zoom_that_cuts_the_gate_goes(editor, monkeypatch, xlim, ylim):
    from openflo.tree_ids import gate_iid
    ed, gids, replots = editor
    _on_gate_view(ed, xlim, ylim)
    _double_click(ed, monkeypatch, gate_iid('s1', gids['rect']))
    assert ed._kept_zoom is None
    assert _view(ed) == ('FL1-A', 'FL2-A', 'dot')
    assert replots == [0]


def test_zoom_that_shows_the_gate_stays(editor, monkeypatch):
    from openflo.tree_ids import gate_iid
    ed, gids, _ = editor
    _on_gate_view(ed, (0, 1000), (0, 1000))
    _double_click(ed, monkeypatch, gate_iid('s1', gids['rect']))
    assert ed._kept_zoom is not None


def test_the_gate_is_drawn_after_the_jump(editor, monkeypatch):
    """The real replot draws the gate's outline on the new view (and did
    not on the old one)."""
    from openflo.tree_ids import gate_iid
    ed, gids, _ = editor
    ed.gate_tv.selection_set(gate_iid('s1', gids['rect']))
    ed._on_tree_select()
    ed._replot()
    assert gids['rect'] not in ed._shape_artists
    _double_click(ed, monkeypatch, gate_iid('s1', gids['rect']))
    ed._replot()
    assert gids['rect'] in ed._shape_artists


def test_view_limits():
    assert view_limits('FL1-A', 'FL2-A', (0, 5), [3, 9]) == {
        'FL1-A': (0.0, 5.0), 'FL2-A': (3.0, 9.0)}
    assert view_limits('FL1-A', None, (0, 5), (3, 9)) == {'FL1-A': (0.0, 5.0)}
    assert view_limits('FL1-A', 'FL2-A', None, (3, 9)) == {
        'FL2-A': (3.0, 9.0)}


def _set_view(ed, x, y, mode):
    ed.x_combo.set(ed._fmt_channel(x))
    ed.y_combo.set(ed._fmt_channel(y))
    ed.mode_var.set(mode)


def test_zoom_kept_from_an_earlier_visit_goes(editor, monkeypatch):
    """Zoomed on the gate's view (cutting x0 = 100 off), then moved to
    another view: the jump back must not bring that zoom back with it."""
    from openflo.tree_ids import gate_iid
    ed, gids, _ = editor
    _set_view(ed, 'FL1-A', 'FL2-A', 'dot')
    ed._replot()
    ed.ax.set_xlim(150, 1000)
    ed._remember_zoom()
    _set_view(ed, 'FSC-A', 'SSC-A', 'dot')
    ed._replot()
    _double_click(ed, monkeypatch, gate_iid('s1', gids['rect']))
    ed._replot()
    lo, hi = sorted(ed.ax.get_xlim())
    assert lo <= 100.0 and hi >= 400.0, (lo, hi)
    assert gids['rect'] in ed._shape_artists


def test_zoom_kept_from_an_earlier_visit_that_shows_the_gate_stays(
        editor, monkeypatch):
    from openflo.tree_ids import gate_iid
    ed, gids, _ = editor
    _set_view(ed, 'FL1-A', 'FL2-A', 'dot')
    ed._replot()
    ed.ax.set_xlim(50, 450)
    ed._remember_zoom()
    _set_view(ed, 'FSC-A', 'SSC-A', 'dot')
    ed._replot()
    _double_click(ed, monkeypatch, gate_iid('s1', gids['rect']))
    ed._replot()
    assert sorted(ed.ax.get_xlim()) == pytest.approx([50, 450])


def test_closed_interval_cut_at_its_high_end_drops_the_zoom(editor,
                                                           monkeypatch):
    from openflo.tree_ids import gate_iid
    ed, _, _ = editor
    ed._set_active_sample('s2')
    gid = ed._add_gate({'kind': 'interval', 'name': 'Mid',
                        'channel': 'FL2-A', 'lo': 100.0, 'hi': 700.0,
                        'parent_id': None}, audit=False)
    ed._refresh_gate_list()
    _set_view(ed, 'FL2-A', 'SSC-A', 'histogram')
    ed.ax.set_xlim(50, 600)                          # hi = 700 is cut off
    ed._kept_zoom = {'key': ed._zoom_key(), 'xlim': [50, 600],
                     'ylim': [0, 1]}
    _double_click(ed, monkeypatch, gate_iid('s2', gid))
    assert ed._kept_zoom is None


def test_labelled_channels_are_set_by_their_shown_names(editor, monkeypatch):
    """With labels the combo shows 'Label (FL1-A)': the jump must set and
    look for that text, not the bare detector."""
    from openflo.tree_ids import gate_iid
    ed, gids, _ = editor
    ed._channel_labels = {'FL1-A': 'Marker one', 'FL2-A': 'Marker two'}
    ed._populate_channel_combos()
    assert ed._fmt_channel('FL1-A') != 'FL1-A'
    _double_click(ed, monkeypatch, gate_iid('s1', gids['rect']))
    assert ed.x_combo.get() == ed._fmt_channel('FL1-A')
    assert ed.y_combo.get() == ed._fmt_channel('FL2-A')
    assert 'not loaded' not in ed.status_var.get()


def test_leaving_a_histogram_hides_its_options(editor, monkeypatch):
    from openflo.tree_ids import gate_iid
    ed, gids, _ = editor
    assert ed.hist_y_combo.winfo_manager() == 'pack'
    _double_click(ed, monkeypatch, gate_iid('s1', gids['rect']))
    assert ed.hist_y_combo.winfo_manager() == ''


def test_double_click_where_tk_cannot_name_the_column_keeps_the_view(
        editor, monkeypatch):
    import tkinter as tk

    from openflo.tree_ids import gate_iid
    ed, gids, _ = editor

    def _gone(_x):
        raise tk.TclError('no column')
    monkeypatch.setattr(ed.gate_tv, 'identify_row',
                        lambda _y: gate_iid('s1', gids['rect']))
    monkeypatch.setattr(ed.gate_tv, 'identify_column', _gone)
    assert ed._on_tv_double_click(SimpleNamespace(x=5, y=5)) is None
    assert _view(ed) == ('FSC-A', 'SSC-A', 'histogram')


def test_zoom_kept_for_another_view_is_left_alone(editor, monkeypatch):
    """Only the zoom kept for the gate's own view can come back with the
    jump: one kept for another view stays for when that view returns."""
    from openflo.tree_ids import gate_iid
    ed, gids, _ = editor
    _set_view(ed, 'FSC-A', 'SSC-A', 'dot')
    ed._replot()
    ed.ax.set_xlim(500, 1000)
    ed._remember_zoom()
    kept = ed._kept_zoom
    _double_click(ed, monkeypatch, gate_iid('s1', gids['rect']))
    assert ed._kept_zoom is kept
