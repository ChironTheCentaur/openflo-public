"""Editor controls that do what their tips and rows say.

* Max points caps each sample in pseudocolor too, not the pooled events.
* The Color combo is greyed out (with a tip saying why) in the modes that
  don't read it.
* Zoom +/- and the mouse wheel zoom in the axis's own scale (log here),
  and zooming out stops at 4x the drawn view, a log-x histogram included.
* Delete and Ctrl+C / X / V work after a click on the plot, not only in the
  gate list.
* On macOS Command+C / X / V and Command+Z / Shift+Z / Y work too.
* The Find box says when nothing matches.
* A right-click ticks its row's sample (the active sample is always on the
  plot), and "On/off state, to other tabs" pushes the ticks from before it.
* Copy gates with nothing ticked stays open and says why; its tips and the
  menu help say a same-name gate on a target is updated to match.
* Cut gate says it cut (and Ctrl+Z), and its help says the gate goes now.
* The side Remove button with nothing selected says what to select.

Generated data, synthetic sample names and generic channels only. No menu
or window is shown: tk_popup is replaced by a recorder and the editor stays
withdrawn, except in the key test, which shows a small one.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests._editor_ui import press_key
from tests.conftest import (
    editor_or_skip,
    invoke_menu_path,
    load_menu_sample,
    menu_popups,  # noqa: F401 - pytest fixture
    right_click_row,
)

COLS = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A']


@pytest.fixture
def ed():
    root, editor = editor_or_skip(accept_confirms=False)
    try:
        editor._finish_plot_canvas()
        yield editor
    finally:
        root.destroy()


def _three(ed):
    """s1, s2, s3 (300 events, one 'Pos' gate each); s1 active."""
    gids = {n: load_menu_sample(ed, n, i)
            for i, n in enumerate(('s1', 's2', 's3'))}
    ed._set_active_sample('s1')
    ed._refresh_gate_list()
    return gids


# ── Max points is per sample in pseudocolor ───────────────────────────────

def _load_blocks(ed, n_events):
    """Three samples whose FL1-A values sit in separate blocks (sample k in
    [k*10000, k*10000 + 1000)), so each drawn point tells its sample."""
    from openflo.pipeline import FlowSample
    rng = np.random.default_rng(3)
    for k, name in enumerate(('s1', 's2', 's3')):
        df = pd.DataFrame(rng.uniform(1, 1000, (n_events, len(COLS))),
                          columns=COLS)
        df['FL1-A'] += k * 10000
        ed._samples[name] = FlowSample.from_dataframe(df, name=name,
                                                      path=f'{name}.fcs')
        ed._sample_order.append(name)
        ed._sample_colors[name] = '#1f77b4'
        ed._sample_trial[name] = 'T'
        ed._sample_plot_enabled[name] = True
        ed._sample_gates.setdefault(name, {})
        ed._sample_gate_order.setdefault(name, [])
        ed._sample_gate_seq.setdefault(name, 0)
    ed._trial_order.append('T')
    ed._channels = list(COLS)
    ed._channel_labels = {c: c for c in COLS}
    ed._populate_channel_combos()
    ed._set_active_sample('s1')
    ed.x_combo.set('FL1-A')
    ed.y_combo.set('FL2-A')


def _drawn_x(ed):
    return np.concatenate([np.asarray(c.get_offsets())[:, 0]
                           for c in ed.ax.collections
                           if len(c.get_offsets())])


@pytest.mark.parametrize('mode', ['pseudocolor', 'dot'])
def test_max_points_caps_each_sample_in_every_mode(ed, mode):
    _load_blocks(ed, 1500)
    ed.ds_display_var.set(True)
    ed.max_points_var.set('1000')
    ed.mode_var.set(mode)
    ed._replot()
    xs = _drawn_x(ed)
    per_sample = [int(((xs >= k * 10000) & (xs < k * 10000 + 1000)).sum())
                  for k in range(3)]
    # Each sample: min(cap, its size) = min(1000, 1500), as its row says.
    assert per_sample == [1000, 1000, 1000]
    assert ed._sample_display_count('s2') == (1000, 1500)


@pytest.mark.parametrize('text', ['1e400', 'abc', 'nan', '-5', '0'])
def test_max_points_refuses_bad_text_with_a_message_and_restores_the_box(
        ed, text):
    """MAIN-2/3: 1e400 raised OverflowError in the Tk callback; abc / nan
    quietly meant 60,000, -5 meant 1,000 and 0 meant no cap, with the bad
    text left in the box. Now: no error, the status line says what Max
    points accepts and quotes the text, the box goes back to the last valid
    value, and that is the cap in use."""
    _load_blocks(ed, 1500)
    ed.ds_display_var.set(True)
    ed.max_points_var.set('20000')
    ed._on_max_points_changed()
    ed.status_var.set('')
    ed.max_points_var.set(text)
    ed._on_max_points_changed()             # Enter
    assert ed.status_var.get() == (
        "Max points must be a whole number of at least 1,000 (250k or 1.5m "
        f"also work), or All for no cap (it says “{text}”); kept 20000.")
    assert ed.max_points_var.get() == '20000'
    assert ed._display_point_cap() == 20_000


def test_max_points_applies_a_typed_value_on_focus_out(ed):
    """MAIN-3: a value typed and left (click elsewhere) was shown but not
    used until Enter. Leaving the box commits it."""
    _load_blocks(ed, 1500)
    ed.ds_display_var.set(True)
    ed.max_points_var.set('1.2k')
    ed._mp_combo.event_generate('<FocusOut>')
    ed.update()
    assert ed._mp_last_good == '1.2k'
    assert ed._sample_display_count('s2') == (1200, 1500)
    # A bad value left behind is refused the same way as on Enter.
    ed.max_points_var.set('banana')
    ed._mp_combo.event_generate('<FocusOut>')
    ed.update()
    assert ed.max_points_var.get() == '1.2k'
    assert 'it says “banana”' in ed.status_var.get()


# ── Color is greyed out where it isn't used ───────────────────────────────

def test_color_combo_is_greyed_out_where_the_mode_ignores_it(ed):
    _load_blocks(ed, 300)
    ed.color_combo.set('FL1-A')
    for mode, used in (('pseudocolor', False), ('contour', False),
                       ('histogram', False), ('dot', True)):
        ed.mode_var.set(mode)
        ed._on_mode_changed()
        ed._replot()
        assert ed.color_combo.instate(['disabled']) is (not used), mode
        tip = ed._tip_texts[str(ed.color_combo)]
        text = tip() if callable(tip) else tip
        if used:
            assert text.startswith('Color the dots'), text
        else:
            assert text.startswith('Not used in this mode'), text
            assert 'dot' in text
    # A mode set from code (a tab, a clustering run) is caught at replot.
    ed.mode_var.set('pseudocolor')
    ed._replot()
    assert ed.color_combo.instate(['disabled'])
    assert ed.color_combo.get() == 'FL1-A'      # kept for dot mode


# ── Zoom in the axis's own scale ──────────────────────────────────────────

def test_zoom_steps_and_wheel_zoom_in_log_space(ed):
    _load_blocks(ed, 300)
    ed.mode_var.set('dot')
    ed._channel_scale['FL1-A'] = 'log'
    ed._replot()
    assert ed.ax.get_xscale() == 'log'
    ed.ax.set_xlim(826.0, 60783.0)
    lo, hi = np.log10(826.0), np.log10(60783.0)
    c, h = (lo + hi) / 2, (hi - lo) / 2
    ed._zoom_step(1.25)                                   # zoom out
    got = np.log10(ed.ax.get_xlim())
    assert got == pytest.approx([c - 1.25 * h, c + 1.25 * h], rel=1e-9)
    assert ed.ax.get_xlim()[0] < 826.0                    # the bottom moved too
    ed._zoom_step(1 / 1.25)                               # back in
    assert np.log10(ed.ax.get_xlim()) == pytest.approx([lo, hi], rel=1e-9)
    # The wheel keeps the point under the cursor fixed, in log space.
    cur = 5000.0
    yl = ed.ax.get_ylim()
    ed._on_scroll(SimpleNamespace(inaxes=ed.ax, xdata=cur,
                                  ydata=sum(yl) / 2, button='up'))
    lc = np.log10(cur)
    want = [lc + (lo - lc) / 1.2, lc + (hi - lc) / 1.2]
    assert np.log10(ed.ax.get_xlim()) == pytest.approx(want, rel=1e-9)


@pytest.mark.parametrize('scale', ['linear', 'log', 'symlog', 'logicle',
                                   'asinh'])
def test_zoom_out_stops_at_four_times_the_drawn_view(ed, scale):
    import warnings
    _load_blocks(ed, 300)
    ed.mode_var.set('dot')
    ed._channel_scale['FL1-A'] = scale
    ed._channel_scale['FL2-A'] = scale
    ed._replot()
    tr = ed.ax.xaxis.get_transform()

    def span(lims):                       # in the axis's own scale
        a, b = tr.transform(np.asarray(lims, float))
        return abs(b - a)
    home = span(ed.ax.get_xlim())
    if scale == 'log':
        assert home == pytest.approx(
            abs(np.diff(np.log10(ed.ax.get_xlim())))[0], rel=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter('error')    # no overflow on the way out
        for _ in range(60):
            xl, yl = ed.ax.get_xlim(), ed.ax.get_ylim()
            ed._on_scroll(SimpleNamespace(
                inaxes=ed.ax, xdata=float(np.sqrt(xl[0] * xl[1]))
                if scale == 'log' else sum(xl) / 2,
                ydata=sum(yl) / 2, button='down'))
        for _ in range(5):
            ed._zoom_step(1.25)
    assert np.all(np.isfinite(ed.ax.get_xlim() + ed.ax.get_ylim()))
    assert span(ed.ax.get_xlim()) == pytest.approx(4 * home, rel=1e-9)
    ed._zoom_step(1 / 1.25)                   # and back in from there
    assert span(ed.ax.get_xlim()) == pytest.approx(4 * home / 1.25,
                                                   rel=1e-9)


def test_zoom_out_cap_holds_on_a_log_x_histogram(ed):
    """A log axis reads a non-positive limit as 1e-1000: the 4x cap
    measured from it never bound, and 80 wheel steps reached ~1e275.
    Non-positive limits count as the smallest positive value now, as
    matplotlib's own log autoscale does. A histogram's own range no longer
    starts below zero on a log axis (test_histogram_log_range), nor does
    a middle-button pan (test_pan_scale). The drawn view here is given such
    a limit directly: set while the axis is linear, then the axis made log
    (matplotlib refuses a non-positive limit set on a log axis)."""
    import warnings
    _load_blocks(ed, 300)
    ed.mode_var.set('histogram')
    ed._channel_scale['FL1-A'] = 'log'
    ed._replot()
    _lo, hi = ed.ax.get_xlim()
    ed.ax.set_xscale('linear')
    ed.ax.set_xlim(-500.0, hi)
    ed.ax.set_xscale('log')
    ed._home_view = (ed.ax.get_xlim(), ed.ax.get_ylim())
    assert ed.ax.get_xscale() == 'log'
    home_lo, hi = ed.ax.get_xlim()
    assert home_lo <= 0                         # the case under test
    minpos = ed.ax.xaxis.get_minpos()
    assert 0 < minpos < hi
    home = np.log10(hi) - np.log10(minpos)      # decades, worked here
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # The first wheel step already zooms out 1.2x from the drawn span
        # (read raw, the limit made the cap clamp it to no zoom at all).
        xl, yl = ed.ax.get_xlim(), ed.ax.get_ylim()
        ed._on_scroll(SimpleNamespace(
            inaxes=ed.ax, xdata=float(np.sqrt(minpos * xl[1])),
            ydata=sum(yl) / 2, button='down'))
        lo1, hi1 = ed.ax.get_xlim()
        assert np.log10(hi1) - np.log10(lo1) == pytest.approx(1.2 * home,
                                                              rel=1e-9)
        for _ in range(79):
            xl, yl = ed.ax.get_xlim(), ed.ax.get_ylim()
            ed._on_scroll(SimpleNamespace(
                inaxes=ed.ax, xdata=float(np.sqrt(max(xl[0], minpos) * xl[1])),
                ydata=sum(yl) / 2, button='down'))
    lo, hi2 = ed.ax.get_xlim()
    assert lo > 0 and np.isfinite(hi2)
    assert np.log10(hi2) - np.log10(lo) == pytest.approx(4 * home, rel=1e-9)
    ed._zoom_step(1 / 1.25)                     # and back in from there
    lo, hi2 = ed.ax.get_xlim()
    assert np.log10(hi2) - np.log10(lo) == pytest.approx(4 * home / 1.25,
                                                         rel=1e-9)


# ── Delete / Ctrl+C after a click on the plot ─────────────────────────────

def _press(widget, key):
    """Send ``key`` to ``widget`` with the focus on it (see
    tests._editor_ui.press_key) and let the editor take it in."""
    assert press_key(widget, key) == str(widget), (
        f'{key} did not reach {widget}: the focus was elsewhere')
    widget.update()


def _show_with_gate_selected(ed, gids, name):
    ed._set_active_sample(name)
    ed._refresh_gate_list()
    ed.gate_tv.selection_set(ed._gate_iid(name, gids[name]))
    ed.update()


def test_delete_and_copy_work_with_the_plot_focused(ed):
    gids = _three(ed)
    ed.geometry('900x700+0+0')
    ed.deiconify()
    ed.update()
    _show_with_gate_selected(ed, gids, 's1')
    plot = ed.canvas.get_tk_widget()
    _press(plot, '<Control-c>')
    assert ed._clip_kind == 'gate_tree'
    assert [g['name'] for g in ed._clip_payload] == ['Pos']
    _press(plot, '<Delete>')
    assert gids['s1'] not in ed._sample_gates['s1']
    # A text box keeps its own Delete: the gates stay.
    _show_with_gate_selected(ed, gids, 's2')
    _press(ed._find_entry, '<Delete>')
    assert gids['s2'] in ed._sample_gates['s2']
    # So does any other widget: a button (reached with Tab, say) does not
    # delete or cut the selected gate.
    from tkinter import ttk
    btn = next(w for w in _walk(ed) if isinstance(w, ttk.Button)
               and w.winfo_ismapped())
    for key in ('<Delete>', '<Control-x>'):
        _press(btn, key)
        assert gids['s2'] in ed._sample_gates['s2'], key
    # A spin box keeps its own Ctrl+Z: the gate history is not undone.
    spin = ttk.Spinbox(ed, from_=0, to=9)
    spin.place(x=0, y=0)
    ed.update()
    n_undo = len(ed._undo_stack)
    assert n_undo                          # the Delete above can be undone
    _press(spin, '<Control-z>')
    assert len(ed._undo_stack) == n_undo
    assert gids['s1'] not in ed._sample_gates['s1']
    spin.destroy()


def test_a_macs_delete_key_deletes_the_gate():
    """A Mac keyboard's "delete" key sends BackSpace (Delete is Fn+delete
    there), so on macOS BackSpace on the plot or the gate list deletes the
    selected gate, as Delete does, while in a text box it still only edits
    the text. Run everywhere with the editor built as on aqua."""
    from unittest import mock

    import openflo.tool_window as tw
    with mock.patch.object(tw, 'is_aqua', lambda _w: True):
        root, ed = editor_or_skip(accept_confirms=False)
    try:
        ed._finish_plot_canvas()
        gids = _three(ed)
        ed.geometry('900x700+0+0')
        ed.deiconify()
        ed.update()
        _show_with_gate_selected(ed, gids, 's1')
        _press(ed._find_entry, '<BackSpace>')
        assert gids['s1'] in ed._sample_gates['s1']
        _press(ed.canvas.get_tk_widget(), '<BackSpace>')
        assert gids['s1'] not in ed._sample_gates['s1']
        _show_with_gate_selected(ed, gids, 's2')
        _press(ed.gate_tv, '<BackSpace>')
        assert gids['s2'] not in ed._sample_gates['s2']
    finally:
        root.destroy()


@pytest.mark.parametrize('platform', ['darwin', 'linux'])
def test_a_mac_copies_cuts_pastes_and_undoes_with_command(monkeypatch,
                                                          platform):
    """On macOS Command+C / X / V in the gate list or on the plot copy, cut
    and paste the selected gate, and Command+Z / Shift+Z / Y undo and redo,
    as Control does everywhere. Elsewhere Command is not bound: on X11 Tk
    reads it as Alt. Run everywhere, with the shortcut helper told it is on macOS or Linux
    (only it: a global sys.platform patch sends other imports down their
    macOS paths)."""
    import openflo.ui_logic as ul
    monkeypatch.setattr(ul, 'sys', SimpleNamespace(platform=platform))
    root, ed = editor_or_skip(accept_confirms=False)
    mac = platform == 'darwin'
    try:
        ed._finish_plot_canvas()
        gids = _three(ed)
        ed.geometry('900x700+0+0')
        ed.deiconify()
        ed.update()
        _show_with_gate_selected(ed, gids, 's1')
        _press(ed.gate_tv, '<Command-c>')
        assert (ed._clip_kind == 'gate_tree') is mac
        # Cut from the plot (after a click on it): the gate goes now.
        _show_with_gate_selected(ed, gids, 's2')
        _press(ed.canvas.get_tk_widget(), '<Command-x>')
        assert (gids['s2'] not in ed._sample_gates['s2']) is mac
        if not mac:
            return
        # Paste the cut gate back onto s2's sample row.
        ed.gate_tv.selection_set(ed._sample_iid('s2'))
        ed.update()
        _press(ed.gate_tv, '<Command-v>')
        assert [g['name'] for g in ed._sample_gates['s2'].values()] == ['Pos']
        # Undo the paste, redo it with Shift, undo again, redo with Y.
        for key, n in (('<Command-z>', 0), ('<Command-Z>', 1),
                       ('<Command-z>', 0), ('<Command-y>', 1)):
            _press(ed.gate_tv, key)
            assert len(ed._sample_gates['s2']) == n, key
        # Control still works on a Mac too.
        _press(ed.gate_tv, '<Control-z>')
        assert len(ed._sample_gates['s2']) == 0
        ed._clip_kind = None
        _show_with_gate_selected(ed, gids, 's1')
        _press(ed.gate_tv, '<Control-c>')
        assert ed._clip_kind == 'gate_tree'
    finally:
        root.destroy()


# ── Find says when nothing matches ────────────────────────────────────────

def test_find_says_no_match_and_keeps_the_selection(ed):
    gids = _three(ed)
    ed._find_var.set('s2')
    found = ed.gate_tv.selection()
    assert found == (ed._sample_iid('s2'),)
    ed._find_var.set('zzz-no-such')
    assert ed.status_var.get() == "No sample or gate matches 'zzz-no-such'."
    assert ed.gate_tv.selection() == found               # kept, consistently
    ed._find_var.set('s2')                               # same row again
    assert ed.status_var.get() == "Found 's2'."
    assert gids                                          # gates untouched


# ── A right-click ticks the row and pushes the earlier on/off ─────────────

def test_right_click_ticks_the_sample_and_pushes_its_earlier_off_state(
        ed, monkeypatch, menu_popups):  # noqa: F811 - pytest fixture
    from openflo.tree_ids import sample_iid
    gids = _three(ed)
    ts = ed._tabs()
    a = ts.new_tab('A', members={'s1': None, 's2': None})
    b = ts.new_tab('B', members={'s1': None, 's2': None})
    ed._rebuild_tab_strip()
    ed._switch_tab(a)
    ed._sample_plot_enabled['s2'] = False                # s2 unticked here
    ed._refresh_gate_list()
    ed.gate_tv.selection_set(sample_iid('s1'))
    ed.update()
    # Right-click s2's gate row (to Edit it, say) and dismiss the menu.
    right_click_row(ed, monkeypatch, menu_popups, ed._gate_iid('s2', gids['s2']),
                    select=False)
    ed.update()                       # the select event the right-click queued
    # The active sample is on the plot, so a new gate lands where it is seen.
    assert ed._active_sample == 's2'
    assert ed._sample_plot_enabled['s2'] is True
    assert ed.gate_tv.item(sample_iid('s2'), 'values')[0] == '☑'
    ed._gate_all_var.set(False)
    assert ed._gate_targets_for_new() == ['s2']
    ed._gate_all_var.set(True)
    assert ed._gate_targets_for_new() == ['s1', 's2']    # both shown on tab A
    ed._gate_all_var.set(False)
    # On/off state, to other tabs pushes s2's state from before the click.
    ed._sample_plot_enabled['s2'] = False
    ed._refresh_gate_list()
    ed.gate_tv.selection_set(sample_iid('s1'))
    ed.update()
    menu = right_click_row(ed, monkeypatch, menu_popups, sample_iid('s2'),
                           select=False)
    ed.update()
    assert ed.gate_tv.selection() == (sample_iid('s2'),)
    assert ed._sample_plot_enabled['s2'] is True         # ticked, as before
    invoke_menu_path(menu, "Copy for…", "On/off state, to other tabs",
                     "All tabs")
    others = [t for i, t in enumerate(ts.tabs) if i != ts.current]
    assert len(others) == 2
    for t in others:
        assert t.plot_enabled.get('s2') is False, t.name
    # A sample that was on before the click pushes "on".
    menu = right_click_row(ed, monkeypatch, menu_popups, sample_iid('s1'),
                           select=False)
    ed.update()
    invoke_menu_path(menu, "Copy for…", "On/off state, to other tabs",
                     "All tabs")
    for t in others:
        assert t.plot_enabled.get('s1') is True, t.name
        assert t.plot_enabled.get('s2') is False, t.name
    # Copy for… > Another tab keeps the earlier state too.
    ts.tabs[b].plot_enabled['s2'] = True
    ed._sample_plot_enabled['s2'] = False
    ed._refresh_gate_list()
    ed.gate_tv.selection_set(sample_iid('s1'))
    ed.update()
    menu = right_click_row(ed, monkeypatch, menu_popups, sample_iid('s2'),
                           select=False)
    ed.update()
    invoke_menu_path(menu, "Copy for…", "Another tab", "B")
    assert ts.tabs[b].plot_enabled['s2'] is False


# ── Copy with nothing ticked ──────────────────────────────────────────────

def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def test_copy_gates_with_nothing_ticked_stays_open_and_says_so(ed):
    import tkinter as tk
    from tkinter import ttk
    _three(ed)
    before = {n: dict(g) for n, g in ed._sample_gates.items()}
    ed._open_copy_gates_dialog()
    (dlg,) = [w for w in _walk(ed.master) if isinstance(w, tk.Toplevel)
              and w.title().startswith('Copy gates')]
    try:
        (copy,) = [w for w in _walk(dlg) if isinstance(w, ttk.Button)
                   and w.cget('text') == 'Copy']
        copy.invoke()
        assert dlg.winfo_exists()
        labels = [str(w.cget('text')) for w in _walk(dlg)
                  if isinstance(w, ttk.Label)]
        assert "Tick at least one sample first." in labels
        assert 'no sample ticked' in ed.status_var.get()
        assert {n: dict(g) for n, g in ed._sample_gates.items()} == before
    finally:
        dlg.destroy()


# ── The copy-gates wording matches what it does ───────────────────────────

def test_copy_gates_wording_says_same_name_gates_are_updated(ed):
    import tkinter as tk
    from tkinter import ttk
    gids = _three(ed)
    ed._open_copy_gates_dialog()
    (dlg,) = [w for w in _walk(ed.master) if isinstance(w, tk.Toplevel)
              and w.title().startswith('Copy gates')]
    try:
        tips = {str(w.cget('text')): ed._tip_texts.get(str(w))
                for w in _walk(dlg)
                if isinstance(w, (ttk.Button, ttk.Checkbutton))}
        for key in ('s2', 'Copy'):
            assert 'same name' in tips[key] or 'same-name' in tips[key], key
            assert 'updated to match' in tips[key], key
            assert 'other gates are kept' in tips[key], key
        # And it is what happens: s2's own 'Pos' takes s1's geometry.
        ed._sample_gates['s2'][gids['s2']]['value'] = 123.0
        (cb,) = [w for w in _walk(dlg) if isinstance(w, ttk.Checkbutton)
                 and w.cget('text') == 's2']
        cb.invoke()
        (copy,) = [w for w in _walk(dlg) if isinstance(w, ttk.Button)
                   and w.cget('text') == 'Copy']
        copy.invoke()
    finally:
        if dlg.winfo_exists():
            dlg.destroy()
    assert ed._sample_gates['s2'][gids['s2']]['value'] == 500.0
    assert '1 updated in place' in ed.status_var.get()
    help_text = ed._MENU_ITEM_HELP["Gates only, to other samples…"]
    assert 'updated to match' in help_text
    assert 'other gates are kept' in help_text


# ── Cut says it cut ───────────────────────────────────────────────────────

def test_cut_gate_status_and_help_say_it_is_removed_now(ed):
    gids = _three(ed)
    ed.update()                       # end the gesture that added the gates
    ed.gate_tv.selection_set(ed._gate_iid('s1', gids['s1']))
    ed._on_cut()
    assert gids['s1'] not in ed._sample_gates['s1']     # gone at once
    status = ed.status_var.get()
    assert status.startswith("Cut 1 gate(s) from 's1'"), status
    assert 'Ctrl+Z to undo' in status
    assert not status.startswith('Copied')
    help_text = ed._MENU_ITEM_HELP["Cut gate (Ctrl+X)"]
    assert 'once pasted' not in help_text
    assert 'now' in help_text and 'Ctrl+Z' in help_text
    ed._undo()
    assert gids['s1'] in ed._sample_gates['s1']


# ── Remove with nothing selected ──────────────────────────────────────────

def test_side_remove_with_nothing_selected_says_what_to_select(ed):
    from tkinter import ttk
    _three(ed)
    ed.gate_tv.selection_set(())
    ed.status_var.set('old text')
    (rm,) = [w for w in _walk(ed) if isinstance(w, ttk.Button)
             and w.cget('text') == 'Remove']
    rm.invoke()
    assert ed.status_var.get() == ("Select a sample or trial row to remove "
                                   "from the session.")
    assert set(ed._samples) == {'s1', 's2', 's3'}
