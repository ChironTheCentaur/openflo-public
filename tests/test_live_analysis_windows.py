"""Open analysis windows follow the editor (editor_watch), and the stale
controls and status lines of control-sweep batch 4 (part a).

Every sample is generated (uniform or lognormal values on FSC-A, SSC-A,
FL1-A, FL2-A and Time), with synthetic names; each expected number is
recomputed here with numpy from the sample's own values.
"""
from __future__ import annotations

import os
import sys
import tkinter as tk
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from tests._inline_editor import _inline  # noqa: E402
from tests._tk_wait import pump_until, settle  # noqa: E402
from tests.conftest import editor_or_skip  # noqa: E402
from tests.test_transform_records import isolate_home  # noqa: E402

COLS = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'Time']
# The editor waits this long after the last change, then a little more.
QUIET = 1.2
# MAD to SD for a normal population: the default rCV form.
K = 1.0 / NormalDist().inv_cdf(0.75)


@pytest.fixture
def ed(monkeypatch, tmp_path):
    import openflo.async_task as _at
    isolate_home(monkeypatch, tmp_path)
    monkeypatch.setattr(_at, 'run_async', _inline)
    root, editor = editor_or_skip()
    yield editor
    try:
        editor.destroy()
        root.destroy()
    except tk.TclError:
        pass


def _add(ed, name, seed, trial='T', n=400):
    """Register a generated sample with a gate ``Pos`` (FL1-A > 500);
    return the gate id."""
    from openflo.pipeline import FlowSample
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.uniform(1, 1000, (n, 4)), columns=COLS[:4])
    df['Time'] = np.arange(n, dtype=float)
    ed._samples[name] = FlowSample.from_dataframe(df, name=name)
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = trial
    if trial not in ed._trial_order:
        ed._trial_order.append(trial)
    ed._sample_plot_enabled[name] = True
    ed._channels = list(COLS)
    ed._channel_labels = {c: c for c in COLS}
    ed._populate_channel_combos()
    ed._set_active_sample(name)
    return ed._add_gate({'kind': 'threshold', 'name': 'Pos',
                         'channel': 'FL1-A', 'op': '>', 'value': 500.0,
                         'parent_id': None}, audit=False)


def _move(ed, name, gid, value):
    """Move a gate the way every gate edit ends: write it, rebuild the
    tree."""
    ed._sample_gates[name][gid]['value'] = float(value)
    ed._refresh_gate_list()


def _shown(ed, win):
    """Show ``win`` on screen. A dialog is transient to the editor, which
    tests keep withdrawn, so it is shown on its own."""
    settle(ed, 0.2)                      # its first map (standardise_dialog)
    win.wm_transient('')
    win.deiconify()
    assert pump_until(ed, lambda: win.winfo_viewable(), 10)
    settle(ed, QUIET)                    # the map's own catch-up, if any
    return win


def _above(ed, name, value, col='FL1-A'):
    x = ed._samples[name].data[col].to_numpy()
    return x > value


# ── the hook itself ──────────────────────────────────────────────────────


def test_a_burst_of_edits_refreshes_a_window_once(ed):
    gid = _add(ed, 's1', 1)
    win = _shown(ed, tk.Toplevel(ed))
    calls = []
    ed._watch_changes(win, lambda: calls.append(1), ('gates',))
    for v in (400, 450, 300):            # a drag: three edits in a row
        _move(ed, 's1', gid, v)
    settle(ed, QUIET)
    assert calls == [1]
    ed._refresh_gate_list()              # a zoom: nothing watched changed
    settle(ed, QUIET)
    assert calls == [1]
    for v in (350, 400, 450):            # a slow drag: the loop runs between
        _move(ed, 's1', gid, v)
        settle(ed, 0.1)                  # well inside WATCH_DELAY_MS
    settle(ed, QUIET)
    assert calls == [1, 1]


def test_a_closed_window_is_unhooked(ed):
    gid = _add(ed, 's1', 1)
    win = _shown(ed, tk.Toplevel(ed))
    calls = []
    ed._watch_changes(win, lambda: calls.append(1), ('gates',))
    assert len(ed._watchers) == 1
    win.destroy()
    assert ed._watchers == {}
    _move(ed, 's1', gid, 300)
    settle(ed, QUIET)
    assert calls == []


def test_a_minimised_window_catches_up_when_shown(ed):
    gid = _add(ed, 's1', 1)
    win = _shown(ed, tk.Toplevel(ed))
    calls = []
    ed._watch_changes(win, lambda: calls.append(1), ('gates',))
    win.withdraw()
    _move(ed, 's1', gid, 300)
    settle(ed, QUIET)
    assert calls == []
    win.deiconify()
    assert pump_until(ed, lambda: calls == [1], 10)



def test_a_clustering_run_defers_the_refresh_to_its_finish(ed, monkeypatch):
    """While a clustering worker rewrites sample frames off the Tk thread
    a flush fingerprints nothing and calls no window; the run's finish
    (_data_changed) catches the window up."""
    gid = _add(ed, 's1', 1)
    win = _shown(ed, tk.Toplevel(ed))
    calls, prints = [], []
    ed._watch_changes(win, lambda: calls.append(1), ('gates',))
    real = ed._watch_fingerprint
    monkeypatch.setattr(ed, '_watch_fingerprint',
                        lambda p: prints.append(p) or real(p))
    ed._clustering_busy = True
    _move(ed, 's1', gid, 300)
    settle(ed, QUIET)
    assert (calls, prints) == ([], [])
    ed._clustering_busy = False
    ed._data_changed('s1')               # what a run's finish does
    assert pump_until(ed, lambda: calls == [1], 10)


@pytest.mark.parametrize('change', ['trial', 'set_aside'])
def test_a_samples_window_follows_a_new_group_or_a_set_aside(ed, change):
    """Moving a sample to another group or setting it aside changes grouped
    statistics, so a window watching ``samples`` is called back."""
    for i, n in enumerate(('s1', 's2'), start=1):
        _add(ed, n, i)
    win = _shown(ed, tk.Toplevel(ed))
    calls = []
    ed._watch_changes(win, lambda: calls.append(1), ('samples',))
    if change == 'trial':
        ed._apply_sample_groups({'s2': 'T2'})
    else:
        assert ed._set_sample_aside(['s2'], 'clogged') == 1
    assert pump_until(ed, lambda: calls == [1], 10)


def test_one_failing_window_does_not_stop_the_others(ed):
    """A window whose refresh raises is noted; the windows after it are
    still refreshed."""
    gid = _add(ed, 's1', 1)
    bad = _shown(ed, tk.Toplevel(ed))
    good = _shown(ed, tk.Toplevel(ed))
    calls = []

    def _fails():
        calls.append('bad')
        raise RuntimeError('refresh failed')
    ed._watch_changes(bad, _fails, ('gates',))
    ed._watch_changes(good, lambda: calls.append('good'), ('gates',))
    _move(ed, 's1', gid, 300)
    assert pump_until(ed, lambda: 'good' in calls, 10), calls
    assert calls == ['bad', 'good']


def test_a_child_widget_closing_leaves_its_window_hooked(ed):
    """A table rebuilt inside a watched window destroys child widgets; only
    the window's own <Destroy> unhooks it."""
    gid = _add(ed, 's1', 1)
    win = _shown(ed, tk.Toplevel(ed))
    calls = []
    ed._watch_changes(win, lambda: calls.append(1), ('gates',))
    child = tk.Label(win, text='table')
    child.pack()
    settle(ed, 0.1)
    child.destroy()
    assert str(win) in ed._watchers
    _move(ed, 's1', gid, 300)
    assert pump_until(ed, lambda: calls == [1], 10)


def test_the_folder_watch_survives_a_live_refresh(ed, tmp_path, monkeypatch):
    """The folder watch's poll timer and the live-refresh timer are two
    timers: a gate edit with a watched window open must not cancel the
    poll (which would stop loading new files while the status line and
    menu still say Watching)."""
    gid = _add(ed, 's1', 1)
    win = _shown(ed, tk.Toplevel(ed))
    hits = []
    ed._watch_changes(win, lambda: hits.append(1), ('gates',))
    d = tmp_path / 'w'
    d.mkdir()
    monkeypatch.setattr('openflo.editor_load.filedialog.askdirectory',
                        lambda **k: str(d))
    ed._toggle_watch_folder()
    assert ed._watch_dir
    _move(ed, 's1', gid, 600)            # a gate edit with a window watching
    assert pump_until(ed, lambda: hits == [1], 10)
    loaded = []
    monkeypatch.setattr(ed, '_queue_fcs_loads', lambda p: loaded.extend(p))
    (d / 'new.fcs').write_bytes(b'x')    # never parsed: the load is stubbed
    assert pump_until(ed, lambda: loaded, 10), \
        "folder watch silently stopped"
    assert ed._watch_dir
    ed._toggle_watch_folder()            # stop: cancels the poll ...
    _move(ed, 's1', gid, 700)            # ... not a pending refresh
    assert pump_until(ed, lambda: hits == [1, 1], 10)

# ── STATS-3: each analysis window follows the editor ──────────────────────


def _stat_row(win, name, gid):
    return [r for r in win._rows
            if r['Sample'] == name and r.get('__gid__') == gid][0]


def test_statistics_table_follows_a_moved_gate(ed):
    gid = _add(ed, 's1', 1)
    win = _shown(ed, ed._open_stats_window())
    assert _stat_row(win, 's1', gid)['Count'] == _above(ed, 's1', 500).sum()
    _move(ed, 's1', gid, 300)
    want = int(_above(ed, 's1', 300).sum())
    assert pump_until(ed, lambda: _stat_row(win, 's1', gid)['Count'] == want,
                      10)


def test_statistics_follows_the_rcv_preference(ed):
    """GATE-6: picking FlowJo's rCV in Preferences recomputes the open
    table."""
    from openflo.ui_preferences import PreferencesDialog
    gid = _add(ed, 's1', 1)
    win = _shown(ed, ed._open_stats_window())
    win._stat_vars['rCV'].set(True)
    win._refresh()
    v = ed._samples['s1'].data['FL1-A'].to_numpy()
    v = v[v > 500]
    med = np.median(v)
    mad = 100 * K * np.median(np.abs(v - med)) / med
    assert _stat_row(win, 's1', gid)['rCV FL1-A'] == pytest.approx(mad,
                                                                   rel=1e-9)
    dlg = PreferencesDialog(ed)
    var = dlg._stat_rcv_var
    combo = next(w for w in _descendants(dlg)
                 if w.winfo_class() == 'TCombobox'
                 and str(w.cget('textvariable')) == str(var))
    var.set(combo.cget('values')[1])         # FlowJo's percentile form
    combo.event_generate('<<ComboboxSelected>>')
    pct = 100 * (np.percentile(v, 84.13) - np.percentile(v, 15.87)) / 2 / med
    assert abs(pct - mad) > 0.05             # the two forms differ here
    assert pump_until(ed, lambda: _stat_row(win, 's1', gid)['rCV FL1-A']
                      == pytest.approx(pct, rel=1e-6), 10)
    dlg.destroy()


def _descendants(w):
    yield w
    for c in w.winfo_children():
        yield from _descendants(c)


def test_frequencies_tab_follows_a_moved_gate_and_the_table_waits(ed):
    gids = {n: _add(ed, n, i, trial=t) for i, (n, t) in enumerate(
        [('a1', 'D1'), ('a2', 'D1'), ('b1', 'D2'), ('b2', 'D2')], start=1)}
    win = _shown(ed, ed._open_frequency_window())
    panel = win.compare
    assert pump_until(ed, lambda: panel._rows, 10)
    calls = []
    real = win._refresh
    win._refresh = lambda: (calls.append(1), real())[1]
    _move(ed, 'b1', gids['b1'], 300)
    want = 100 * _above(ed, 'b1', 300).mean()

    def shown():
        rows = [r for r in panel._rows if r['Sample'] == 'b1'
                and r.get('__gid__') == gids['b1']]
        return rows and rows[0]['%Parent'] == pytest.approx(want, rel=1e-9)
    assert pump_until(ed, shown, 10)
    assert calls == []                       # the hidden table waited
    win.show_table()
    assert pump_until(ed, lambda: calls == [1], 10)


def test_plate_view_follows_moved_and_added_gates(ed):
    from openflo.ui_plate import PlateWindow
    gid = _add(ed, 'Plate1_B01', 1)
    win = _shown(ed, PlateWindow(ed))
    win.stat_var.set('Pos: % of parent')
    win.draw()
    _move(ed, 'Plate1_B01', gid, 300)
    want = 100 * _above(ed, 'Plate1_B01', 300).mean()

    def drawn():                         # the plate on screen, well B1
        return float(win.fig.axes[0].images[0].get_array()[1, 0])
    assert drawn() != pytest.approx(want)
    assert pump_until(ed, lambda: drawn() == pytest.approx(want, rel=1e-9),
                      10)
    ed._set_active_sample('Plate1_B01')
    ed._add_gate({'kind': 'threshold', 'name': 'Hi2', 'channel': 'FL2-A',
                  'op': '>', 'value': 700.0, 'parent_id': None}, audit=False)
    assert pump_until(ed, lambda: 'Hi2: count' in win.stat_combo['values'],
                      10)
    assert win.stat_var.get() == 'Pos: % of parent'   # the pick stays


def test_kinetics_follows_a_moved_gate(ed):
    gid = _add(ed, 's1', 1)
    ed._open_kinetics_window()
    win = _shown(ed, ed._tool_windows['kinetics'])
    win.chan_var.set(ed._fmt_channel('FL2-A'))
    win.pop_var.set('Pos')
    win._refresh()
    assert win._results[0]['n_events'] == _above(ed, 's1', 500).sum()
    _move(ed, 's1', gid, 300)
    want = int(_above(ed, 's1', 300).sum())
    assert pump_until(ed, lambda: win._results[0]['n_events'] == want, 10)



@pytest.mark.parametrize('bins', ['100', '1'])
def test_kinetics_follows_a_vanished_population_and_a_new_channel(ed, bins):
    """A renamed gate takes the chosen population away: the window falls
    back to All events and says so, also when a bad Bins box keeps it from
    redrawing. A channel added to the editor joins the Channel list."""
    gid = _add(ed, 's1', 1)
    ed._open_kinetics_window()
    win = _shown(ed, ed._tool_windows['kinetics'])
    win.chan_var.set(ed._fmt_channel('FL2-A'))
    win.pop_var.set('Pos')
    win._refresh()
    assert win._results[0]['n_events'] == _above(ed, 's1', 500).sum()
    s = ed._samples['s1']
    s.data['FL3-A'] = s.data['FL1-A'] * 2.0
    s.raw['FL3-A'] = s.raw['FL1-A'] * 2.0
    ed._channels = [*COLS, 'FL3-A']
    before = win._results
    win.bins_var.set(bins)
    ed._sample_gates['s1'][gid]['name'] = 'Hi'
    ed._data_changed('s1')
    ed._refresh_gate_list()
    assert pump_until(ed, lambda: win.pop_var.get() == win.ALL_EVENTS, 10)
    said = win.status_var.get()
    assert said.startswith('Population Pos is gone; showing All events.'), \
        said
    if bins == '1':
        assert 'Not redrawn' in said and win._results is before
    else:
        assert win._results[0]['n_events'] == 400
    assert ed._fmt_channel('FL3-A') in win._chan_combo['values']

def test_saturation_follows_rewritten_data(ed):
    _add(ed, 's1', 1)
    ed._open_saturation_report()
    win = _shown(ed, ed._tool_windows['saturation'])
    assert win._rows[0]['Events'] == 400
    s = ed._samples['s1']
    s.data = s.data.iloc[:250].copy()    # a trim, as the downsample does
    s.raw = s.raw.iloc[:250].copy()
    ed._data_changed('s1')
    assert pump_until(ed, lambda: win._rows[0]['Events'] == 250, 10)


def test_sample_qc_follows_an_unticked_sample(ed):
    from openflo.async_task import wait_for_job
    for i, n in enumerate(('s1', 's2', 's3'), start=1):
        _add(ed, n, i)
    ed._open_sample_qc_window()
    win = _shown(ed, ed._tool_windows['sample_qc'])
    assert pump_until(ed, lambda: len(win._names) == 3, 30)
    ed._sample_plot_enabled['s3'] = False
    ed._refresh_gate_list()
    assert pump_until(ed, lambda: win._names == ['s1', 's2']
                      and not win._busy.running, 30)
    assert wait_for_job(win._busy)
    assert win._D.shape == (2, 2)


def test_methods_follows_a_recorded_step(ed):
    _add(ed, 's1', 1)
    ed._open_methods_report()
    win = _shown(ed, ed._tool_windows['methods'])
    before = win._para
    assert 'compensated' not in before
    ed._audit('compensate', sample='s1')
    assert pump_until(ed, lambda: 'spillover was compensated' in win._para,
                      10)
    assert win._texts[0].get('1.0', 'end-1c') == win._para


def _medians(ed, names, col, pop=None):
    out = {}
    for n in names:
        x = ed._samples[n].data[col].to_numpy()
        if pop is not None:
            x = x[_above(ed, n, pop)]
        out[n] = float(np.median(x))
    return out


def _expression(ed):
    from openflo.async_task import wait_for_job
    gids = {n: _add(ed, n, i) for i, n in enumerate(
        ('Stim_1', 'Stim_2', 'Ctrl_1', 'Ctrl_2'), start=1)}
    ed._open_expression_window()
    win = _shown(ed, ed._tool_windows['expression'])
    assert wait_for_job(win._busy)
    return win, gids


def _shown_medians(win):
    return sorted(m for ms in win._medians.values() for m in ms)


def test_expression_follows_a_moved_gate(ed):
    from openflo.async_task import wait_for_job
    win, gids = _expression(ed)
    win.pop_var.set('Pos')
    win._rebuild()
    assert wait_for_job(win._busy)
    _move(ed, 'Stim_1', gids['Stim_1'], 300)
    want = _medians(ed, ['Stim_1'], 'FL1-A', 300)['Stim_1']
    assert pump_until(ed, lambda: want in _shown_medians(win), 30)


# ── STATS-4: Expression's controls take effect without Update ─────────────


def _pick(combo, value):
    combo.set(value)
    combo.event_generate('<<ComboboxSelected>>')


def _boxes(win):
    return {str(w.cget('textvariable')): w for w in _descendants(win)
            if w.winfo_class() in ('TCombobox', 'TEntry')}


def test_expression_marker_box_takes_effect_at_once(ed):
    from openflo.async_task import wait_for_job
    win, _g = _expression(ed)
    _pick(_boxes(win)[str(win.marker_var)], ed._fmt_channel('FL2-A'))
    assert wait_for_job(win._busy)
    names = ['Stim_1', 'Stim_2', 'Ctrl_1', 'Ctrl_2']
    assert _shown_medians(win) == sorted(_medians(ed, names, 'FL2-A').values())


def test_expression_population_and_grouping_take_effect_at_once(ed):
    from openflo.async_task import wait_for_job
    win, _g = _expression(ed)
    boxes = _boxes(win)
    _pick(boxes[str(win.pop_var)], 'Pos')
    assert wait_for_job(win._busy)
    names = ['Stim_1', 'Stim_2', 'Ctrl_1', 'Ctrl_2']
    assert _shown_medians(win) == sorted(
        _medians(ed, names, 'FL1-A', 500).values())
    _pick(boxes[str(win.factor_var)], 'Name token')
    assert wait_for_job(win._busy)
    assert sorted(win._groups) == ['Ctrl', 'Stim']
    win.tokens_var.set('Stim')
    entry = boxes[str(win.tokens_var)]
    entry.focus_force()
    settle(ed, 0.1)
    entry.event_generate('<Return>')
    assert wait_for_job(win._busy)
    assert win._groups == ['Stim']


def test_expression_test_and_paired_take_effect_at_once(ed):
    from openflo.async_task import wait_for_job
    win, _g = _expression(ed)
    boxes = _boxes(win)
    _pick(boxes[str(win.factor_var)], 'Name token')
    assert wait_for_job(win._busy)
    test_box = boxes[str(win.test_var)]
    _pick(test_box, test_box.cget('values')[1])        # the rank test
    assert wait_for_job(win._busy)
    assert 'Mann' in win._summary.get('1.0', 'end')
    chk = next(w for w in _descendants(win)
               if w.winfo_class() == 'TCheckbutton'
               and str(w.cget('variable')) == str(win.paired_var))
    chk.invoke()
    assert wait_for_job(win._busy)
    assert 'aired' in win._summary.get('1.0', 'end')


def test_expression_plot_box_redraws_without_recomputing(ed):
    from openflo.async_task import wait_for_job
    win, _g = _expression(ed)
    runs = []
    real = win._busy.run
    win._busy.run = lambda *a, **k: (runs.append(1), real(*a, **k))[1]
    boxes = _boxes(win)
    _pick(boxes[str(win.factor_var)], 'Name token')
    assert wait_for_job(win._busy)
    runs.clear()
    _pick(boxes[str(win.plot_var)], 'Ridgeline')
    assert wait_for_job(win._busy)
    assert runs == []
    ax = win._fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == win._groups
    assert len(ax.collections) == 2          # one ridge per group


# ── STATS-6: Kinetics trace export names what was computed ────────────────


def test_kinetics_trace_export_names_the_statistic_computed(ed, monkeypatch):
    import openflo.ui_kinetics as uk
    _add(ed, 's1', 1)
    ed._open_kinetics_window()
    win = ed._tool_windows['kinetics']
    win.stat_var.set('median')
    win._refresh()
    medians = win._results[0]['trace'].copy()
    win.stat_var.set('mean')                  # changed, Plot not pressed
    got = {}
    monkeypatch.setattr(uk, 'ask_csv_path', lambda *a, **k: 'traces.csv')
    monkeypatch.setattr(win, '_write_csv',
                        lambda rows, *a: got.setdefault('rows', rows))
    win._export_traces()
    head, first = got['rows'][0], got['rows'][1]
    assert head[2] == 'median'
    assert first[2] == pytest.approx(float(medians[0]))
    t = ed._samples['s1'].data['Time'].to_numpy()
    v = ed._samples['s1'].data[win._channel()].to_numpy()
    edges = np.linspace(t.min(), t.max(), 101)
    assert first[2] == pytest.approx(float(np.median(v[t < edges[1]])))


# ── Batch 3 x batch 4: a live refresh never reads a half-typed box ────────


@pytest.mark.parametrize('box, text, says', [
    ('bins_var', 'abc', 'Bins must be a whole number from 2 to 10000'),
    ('bins_var', '1', 'Bins must be a whole number from 2 to 10000'),
    ('bins_var', '20000', 'Bins must be a whole number from 2 to 10000'),
    ('base_var', '1x', 'Baseline ends at must be a number'),
    ('stim_var', 'nan', 'Stimulus at must be a number'),
])
def test_kinetics_live_refresh_skips_a_box_plot_would_refuse(
        ed, monkeypatch, box, text, says):
    """Kinetics refreshes when a gate moves; Plot refuses bad Bins /
    Baseline / Stimulus text (batch 3). A live refresh with such text must
    not raise, open a dialog or swap in a default: the last result stays
    and the status line names the box. Once the box is fixed, the next
    change refreshes again."""
    from tkinter import messagebox
    gid = _add(ed, 's1', 1)
    ed._open_kinetics_window()
    win = _shown(ed, ed._tool_windows['kinetics'])
    win.chan_var.set(ed._fmt_channel('FL2-A'))
    win.pop_var.set('Pos')
    win._refresh()
    before = win._results
    assert before[0]['n_events'] == _above(ed, 's1', 500).sum()
    dialogs = []
    for kind in ('showerror', 'showwarning', 'showinfo'):
        monkeypatch.setattr(messagebox, kind,
                            lambda *a, _k=kind, **k: dialogs.append(_k))
    getattr(win, box).set(text)
    _move(ed, 's1', gid, 300)
    assert pump_until(ed, lambda: says in win.status_var.get(), 10), \
        win.status_var.get()
    assert win.status_var.get().startswith('Not redrawn')
    assert win._results is before                 # the last result stays
    assert dialogs == []
    default = {'bins_var': '100'}.get(box, '')
    getattr(win, box).set(default)
    _move(ed, 's1', gid, 200)
    want = int(_above(ed, 's1', 200).sum())
    assert pump_until(ed, lambda: win._results[0]['n_events'] == want, 10)
    assert dialogs == []


def test_quick_preview_channel_pick_with_a_bad_threshold_opens_no_dialog(
        ed, tmp_path, monkeypatch):
    """Picking X or Y redraws by itself (batch 4); a threshold box that is
    not a number (refused by Plot with a dialog, batch 3) leaves the last
    plot and says so on the status line, with no dialog."""
    from tkinter import messagebox

    from openflo.ui_preview import QuickPreviewDialog
    from tests._fcs_writer import write_float_fcs
    rng = np.random.default_rng(4)
    path = str(tmp_path / 'p2.fcs')
    write_float_fcs(path, rng.uniform(1, 1000, (300, 4)), COLS[:4])
    d = QuickPreviewDialog(ed)
    d._load_sample(path)
    d._plot()
    assert d._ax.get_ylabel().startswith('SSC-A')
    dialogs = []
    monkeypatch.setattr(messagebox, 'showerror',
                        lambda *a, **k: dialogs.append(a))
    d._ythr_var.set('5o0')
    _pick(d._y_combo, 'FL2-A')
    assert dialogs == []
    assert d._ax.get_ylabel().startswith('SSC-A')       # the last plot
    assert 'Y thresh must be a number' in ed.status_var.get()
    d._ythr_var.set('500')
    _pick(d._y_combo, 'FL2-A')
    assert d._ax.get_ylabel().startswith('FL2-A')
    d._ythr_var.set('x')
    d._plot()                                  # Plot itself still asks
    assert len(dialogs) == 1
    d.destroy()


# ── FILE-4: Quick preview redraws on a new channel ────────────────────────


def test_quick_preview_redraws_on_a_new_channel(ed, tmp_path):
    from openflo.ui_preview import QuickPreviewDialog
    from tests._fcs_writer import write_float_fcs
    rng = np.random.default_rng(3)
    path = str(tmp_path / 'p1.fcs')
    write_float_fcs(path, rng.uniform(1, 1000, (300, 4)), COLS[:4])
    d = QuickPreviewDialog(ed)
    d._load_sample(path)
    d._plot()
    assert d._ax.get_ylabel().startswith('SSC-A')
    _pick(d._y_combo, 'FL2-A')
    assert d._ax.get_ylabel().startswith('FL2-A')
    _pick(d._x_combo, 'FL1-A')
    assert d._ax.get_xlabel().startswith('FL1-A')
    d.destroy()


# ── MAIN-8: the status after Display + data -> Display only ───────────────


def test_display_only_after_display_and_data_says_the_plot_still_downsamples(
        ed):
    _add(ed, 's1', 1)
    _add(ed, 's2', 2, n=200)
    ed._ds_mode_var.set('Display + data')
    ed._on_ds_mode_changed()
    assert len(ed._samples['s1'].data) == 200
    ed._ds_mode_var.set('Display only')
    ed._on_ds_mode_changed()
    assert len(ed._samples['s1'].data) == 400
    assert ed.ds_display_var.get()
    msg = ed.status_var.get()
    assert not msg.startswith('Downsampling off')
    assert 'still shows a subsample' in msg
    ed._ds_mode_var.set('Off')
    ed._on_ds_mode_changed()
    ed._ds_mode_var.set('Display + data')
    ed._on_ds_mode_changed()
    ed._ds_mode_var.set('Off')
    ed._on_ds_mode_changed()
    assert ed.status_var.get().startswith('Downsampling off')


# ── MAIN-9: Highlight / Filter with no gate to act on ─────────────────────


def _display_menu(ed):
    from tests.conftest import submenu
    btn = next(b for b in ed._menubar_buttons if b.cget('text') == 'View')
    return submenu(btn.nametowidget(btn.cget('menu')), 'Display')


def test_highlight_and_filter_are_greyed_with_no_gate(ed):
    settle(ed, 0.3)
    radios = ed._display_radios
    assert radios['highlight'].instate(['disabled'])     # empty editor
    assert radios['filter'].instate(['disabled'])
    menu = _display_menu(ed)
    ed.tk.eval(str(menu.cget('postcommand')))             # as it opens
    assert menu.entrycget(1, 'state') == 'disabled'
    assert menu.entrycget(2, 'state') == 'disabled'
    assert menu.entrycget(0, 'state') == 'normal'
    _add(ed, 's1', 1)
    ed.tk.eval(str(menu.cget('postcommand')))
    assert menu.entrycget(2, 'state') == 'normal'


def test_filter_shortcut_with_no_gate_says_so(ed):
    _add(ed, 's1', 1)
    for gid in list(ed._sample_gates['s1']):
        ed._sample_gates['s1'].pop(gid)
        ed._sample_gate_order['s1'].remove(gid)
    ed._gates = ed._sample_gates['s1']
    ed._set_display_mode('filter')
    assert ed.gate_display_var.get() == 'all'
    assert 'Filter to gated' not in ed.status_var.get()
    assert 'No gates' in ed.status_var.get()
