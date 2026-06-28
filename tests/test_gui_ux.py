"""Tests for the recent GUI UX additions (gating-loop ergonomics).

Covers behaviour that ships in the editor but had no automated check:
swap-axes, the selected-gate % -of-parent readout, the type-to-filter
channel pickers, the first-run empty overlay, display-mode shortcuts, the
zoom-tool cancel, the resize freeze/thaw, and that the accelerators bind.

All guarded behind a Tk-availability skip, mirroring tests/test_gui_smoke.py.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _editor_or_skip():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
    except ImportError:
        pytest.skip("tkinter not available — headless environment")
    try:
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    import importlib
    gui = importlib.import_module('openflo.gui')
    gui.messagebox.askyesno = lambda *a, **k: True
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    return root, ed


def _load_sample(ed, name='s1', cols=('FSC-A', 'SSC-A', 'CD3', 'CD4'), n=200):
    df = pd.DataFrame({c: np.linspace(0, 1, n) for c in cols})
    s = SimpleNamespace(name=name, path=rf'C:\exp\{name}.fcs', data=df,
                        fluor_channels=[c for c in cols if c not in
                                        ('FSC-A', 'SSC-A')],
                        channel_labels={c: c for c in cols})
    ed._samples[name] = s
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    if 'T' not in ed._trial_order:
        ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = True
    ed._channels = list(cols)
    ed._channel_labels = {c: c for c in cols}
    ed._populate_channel_combos()
    return s


def test_swap_axes():
    root, ed = _editor_or_skip()
    try:
        _load_sample(ed)
        ed.x_combo.set('CD3')
        ed.y_combo.set('CD4')
        ed._swap_axes()
        assert ed.x_combo.get() == 'CD4' and ed.y_combo.get() == 'CD3'
    finally:
        root.destroy()


def test_gate_count_percent_of_parent():
    """g1 = CD3>=0.4 (120/200 = 60% of all); g2 = CD4>=0.5 under g1
    (20/120 = 16.67% of parent). The status bar must report both."""
    root, ed = _editor_or_skip()
    try:
        _load_sample(ed)
        # CD4 descends so it's independent of CD3 (which ascends): g1 = CD3>=0.4
        # is rows 80..199 (120); within those, CD4>=0.5 keeps rows 80..99 (20).
        ed._samples['s1'].data['CD4'] = np.linspace(1, 0, 200)
        ed._sample_gates['s1'] = {
            'g1': {'kind': 'threshold', 'channel': 'CD3', 'value': 0.4,
                   'op': '>=', 'parent_id': None, 'color': '#111',
                   'enabled': True, 'id': 'g1', 'name': 'CD3+'},
            'g2': {'kind': 'threshold', 'channel': 'CD4', 'value': 0.5,
                   'op': '>=', 'parent_id': 'g1', 'color': '#222',
                   'enabled': True, 'id': 'g2', 'name': 'CD4+'},
        }
        ed._sample_gate_order['s1'] = ['g1', 'g2']
        ed._set_active_sample('s1')

        ed._show_gate_count('s1', 'g1')
        msg = ed.status_var.get()
        assert 'n = 120' in msg and '60.00% of all' in msg

        ed._show_gate_count('s1', 'g2')
        msg = ed.status_var.get()
        assert 'n = 20' in msg and '16.67% of parent' in msg
    finally:
        root.destroy()


def test_sample_count_readout():
    root, ed = _editor_or_skip()
    try:
        _load_sample(ed, n=200)
        ed._show_sample_count('s1')
        assert '200 events' in ed.status_var.get()
    finally:
        root.destroy()


def test_filterable_combo_filters_and_commits():
    root, ed = _editor_or_skip()
    try:
        _load_sample(ed)
        ed.x_combo.delete(0, 'end')
        ed.x_combo.insert(0, 'cd4')
        ed.x_combo._filter_type()                 # narrows the dropdown
        assert tuple(ed.x_combo['values']) == ('CD4',)
        ed.x_combo._filter_commit()               # snaps to the match
        assert ed.x_combo.get() == 'CD4'
        # full list restored after commit
        assert 'FSC-A' in ed.x_combo['values']
    finally:
        root.destroy()


def test_filterable_combo_reverts_invalid():
    root, ed = _editor_or_skip()
    try:
        _load_sample(ed)
        ed.x_combo.set('CD3')
        ed.x_combo._filter_commit()               # establishes last-valid
        ed.x_combo.delete(0, 'end')
        ed.x_combo.insert(0, 'zzz-not-a-channel')
        ed.x_combo._filter_commit()
        assert ed.x_combo.get() == 'CD3'          # reverted to last valid
    finally:
        root.destroy()


def test_empty_overlay_shows_then_hides():
    root, ed = _editor_or_skip()
    try:
        ed._render_placeholder()                  # no samples loaded
        ov = getattr(ed, '_empty_overlay', None)
        assert ov is not None and ov.winfo_manager() == 'place'

        _load_sample(ed)
        ed._set_active_sample('s1')
        ed._replot()
        assert ed._empty_overlay.winfo_manager() == ''   # hidden once plotting
    finally:
        root.destroy()


def test_empty_overlay_not_shown_when_samples_exist():
    root, ed = _editor_or_skip()
    try:
        _load_sample(ed)
        ed._show_empty_overlay()                  # samples present → no-op
        ov = getattr(ed, '_empty_overlay', None)
        assert ov is None or ov.winfo_manager() == ''
    finally:
        root.destroy()


def test_display_mode_sets_apply_gates():
    root, ed = _editor_or_skip()
    try:
        ed._set_display_mode('filter')
        assert ed.gate_display_var.get() == 'filter'
        assert ed.apply_gates_var.get() is True
        ed._set_display_mode('all')
        assert ed.gate_display_var.get() == 'all'
        assert ed.apply_gates_var.get() is False
        ed._set_display_mode('highlight')
        assert ed.gate_display_var.get() == 'highlight'
        assert 'Highlight' in ed.status_var.get()
    finally:
        root.destroy()


def test_display_modes_greyed_without_real_gates():
    root, ed = _editor_or_skip()
    try:
        _load_sample(ed)                       # no gates
        ed._set_active_sample('s1')
        ed._sync_display_mode_availability()
        assert 'disabled' in ed._display_radios['highlight'].state()
        assert 'disabled' in ed._display_radios['filter'].state()

        # a stale highlight selection falls back to 'all'
        ed.gate_display_var.set('highlight')
        ed._sync_display_mode_availability()
        assert ed.gate_display_var.get() == 'all'

        # a real (positive) gate enables them
        ed._sample_gates['s1'] = {
            'g1': {'kind': 'rect', 'parent_id': None, 'id': 'g1'}}
        ed._sync_display_mode_availability()
        assert 'disabled' not in ed._display_radios['highlight'].state()

        # an autoclean-only gate does NOT count as a real gate
        ed._sample_gates['s1'] = {
            'a1': {'kind': 'autoclean', 'parent_id': None, 'id': 'a1'}}
        ed._sync_display_mode_availability()
        assert 'disabled' in ed._display_radios['filter'].state()
    finally:
        root.destroy()


def test_cancel_active_tool_disarms_zoom():
    root, ed = _editor_or_skip()
    try:
        ed._zoom_mode_var.set(True)
        ed._zoom_mode = True
        ed._cancel_active_tool()
        assert ed._zoom_mode_var.get() is False
        assert getattr(ed, '_zoom_mode', False) is False
    finally:
        root.destroy()


def test_freeze_and_thaw_plot_redraw():
    root, ed = _editor_or_skip()
    try:
        ed._freeze_plot_redraw()
        frozen = ed.canvas.draw_idle              # the no-op suppressor
        assert ed._plot_frozen is True
        assert frozen() is None                   # calling it does nothing
        ed._thaw_plot_redraw()
        assert ed._plot_frozen is False
        assert ed.canvas.draw_idle is not frozen  # restored on release
    finally:
        root.destroy()


def test_environment_dialog_opens():
    root, ed = _editor_or_skip()
    try:
        before = set(ed.winfo_children())
        ed._show_environment()
        ed.update_idletasks()
        new = [w for w in ed.winfo_children() if w not in before]
        titles = [w.title() for w in new if hasattr(w, 'title')]
        assert any('Environment' in t for t in titles)
    finally:
        root.destroy()


def test_provenance_footer_added_during_save_and_removed_after(tmp_path,
                                                               monkeypatch):
    root, ed = _editor_or_skip()           # ensures gui importable + Tk up
    try:
        import matplotlib.figure as mfig

        from openflo import gui
        monkeypatch.setattr('openflo.theme.read_prefs',
                            lambda: {'export_provenance': True})
        fig = mfig.Figure()
        fig.add_subplot(1, 1, 1).plot([0, 1], [0, 1])
        n0 = len(fig.texts)

        captured = {}
        real = fig.savefig

        def spy(*a, **k):
            captured['n'] = len(fig.texts)     # count while the footer is on
            return real(*a, **k)
        fig.savefig = spy

        out = tmp_path / 'p.png'
        gui.savefig_background(fig, str(out))
        assert out.exists()
        assert captured['n'] == n0 + 1         # footer present during save
        assert len(fig.texts) == n0            # and removed afterward
    finally:
        root.destroy()


def test_provenance_footer_respects_pref_off(tmp_path, monkeypatch):
    root, ed = _editor_or_skip()
    try:
        import matplotlib.figure as mfig

        from openflo import gui
        monkeypatch.setattr('openflo.theme.read_prefs',
                            lambda: {'export_provenance': False})
        fig = mfig.Figure()
        fig.add_subplot(1, 1, 1).plot([0, 1], [0, 1])
        n0 = len(fig.texts)

        captured = {}
        real = fig.savefig

        def spy(*a, **k):
            captured['n'] = len(fig.texts)
            return real(*a, **k)
        fig.savefig = spy

        gui.savefig_background(fig, str(tmp_path / 'q.png'))
        assert captured['n'] == n0             # no footer when disabled
    finally:
        root.destroy()


def test_shortcuts_are_bound():
    """The accelerators wired in _bind_shortcuts must actually be bound."""
    root, ed = _editor_or_skip()
    try:
        for seq in ('<Control-f>', '<Control-Key-0>', '<F5>',
                    '<Control-Key-1>', '<Control-Key-3>', '<Control-t>',
                    '<F9>', '<Escape>'):
            assert ed.bind(seq), f'{seq} is not bound'
    finally:
        root.destroy()


def test_loading_placeholders_persist_in_tree():
    """A still-loading sample shows as a muted '⏳' row alongside already-loaded
    samples, instead of vanishing on the first load's tree rebuild. Without this
    a multi-file (or big-file) load looks stalled — the tree should list every
    queued sample as 'loading' until each one lands."""
    root, ed = _editor_or_skip()
    try:
        _load_sample(ed, 's1')                      # one fully-loaded sample
        # Simulate a second file still in the load queue (same trial as s1).
        ed._loading.add('big2')
        ed._sample_trial['big2'] = 'T'
        ed._refresh_gate_list()

        def _all_rows():
            out = []

            def walk(parent=''):
                for iid in ed.gate_tv.get_children(parent):
                    out.append(iid)
                    walk(iid)
            walk()
            return out

        texts = [ed.gate_tv.item(i, 'text') for i in _all_rows()]
        assert any('s1' in t for t in texts), texts
        assert any('big2' in t and '⏳' in t for t in texts), texts

        # The placeholder carries the muted 'loading' tag and the right iid, so
        # _on_loaded later swaps it for the real subtree in place.
        iid = ed._sample_iid('big2')
        assert ed.gate_tv.exists(iid)
        assert 'loading' in ed.gate_tv.item(iid, 'tags')

        # Once it lands, the placeholder is replaced by a real row (no ⏳).
        _load_sample(ed, 'big2')                    # now in _samples
        ed._loading.discard('big2')
        ed._refresh_gate_list()
        texts2 = [ed.gate_tv.item(i, 'text') for i in _all_rows()]
        assert not any('⏳' in t for t in texts2), texts2
        assert any('big2' in t for t in texts2)
    finally:
        root.destroy()


def test_session_resume_shows_loading_placeholders(tmp_path):
    """Resuming a session paints a ⏳ row for every sample up front — before the
    background FCS loads finish — so a big session doesn't look frozen while it
    loads. The real parse is stubbed out so the intermediate state is stable."""
    root, ed = _editor_or_skip()
    try:
        # Stub the per-file load so the pool threads never post _on_loaded /
        # _on_load_error (which would swap/remove the rows mid-assert).
        ed._load_worker = lambda name, path: None
        f1 = tmp_path / 'sampleA.fcs'; f1.write_bytes(b'')
        f2 = tmp_path / 'sampleB.fcs'; f2.write_bytes(b'')
        data = {
            'samples': [
                {'name': 'sampleA', 'path': str(f1), 'trial': 'Day 1',
                 'plot_enabled': True},
                {'name': 'sampleB', 'path': str(f2), 'trial': 'Day 1',
                 'plot_enabled': False},
            ],
            'sample_gates': {},
        }
        ed._apply_session(data)
        # Every sample is registered as loading and shown as a ⏳ row right away.
        assert {'sampleA', 'sampleB'} <= ed._loading
        for nm in ('sampleA', 'sampleB'):
            iid = ed._sample_iid(nm)
            assert ed.gate_tv.exists(iid), f'{nm} row missing'
            assert '⏳' in ed.gate_tv.item(iid, 'text')
            assert 'loading' in ed.gate_tv.item(iid, 'tags')
    finally:
        try:
            ed._load_stop.set()
            for _ in range(4):
                ed._load_queue.put(None)
        except Exception:
            pass
        try:
            root.destroy()
        except Exception:
            pass


def test_autoclean_applies_to_all_selected_samples():
    """Auto-clean with several samples selected adds the recipe gate to EACH of
    them — not just the active/last-selected one (the reported bug)."""
    root, ed = _editor_or_skip()
    try:
        for nm in ('s1', 's2', 's3'):
            _load_sample(ed, nm)
        ed._set_active_sample('s3')                 # last selected = active
        ed._refresh_gate_list()
        ed.gate_tv.selection_set([ed._sample_iid(n) for n in ('s1', 's2', 's3')])
        ed._create_autoclean_gate()
        for nm in ('s1', 's2', 's3'):
            gates = ed._sample_gates.get(nm, {})
            assert any(g.get('kind') == 'autoclean' for g in gates.values()), \
                f'{nm} got no autoclean gate'
        # Originally-active sample is preserved.
        assert ed._active_sample == 's3'
    finally:
        root.destroy()


def test_autoclean_no_selection_falls_back_to_active_and_skips_dupes():
    root, ed = _editor_or_skip()
    try:
        _load_sample(ed, 's1')
        ed._set_active_sample('s1')
        ed._refresh_gate_list()
        ed.gate_tv.selection_remove(*ed.gate_tv.selection())
        ed._create_autoclean_gate()                 # falls back to active s1
        g1 = ed._sample_gates.get('s1', {})
        n_ac = sum(1 for g in g1.values() if g.get('kind') == 'autoclean')
        assert n_ac == 1
        # Running again must not stack a second autoclean gate.
        ed.gate_tv.selection_set(ed._sample_iid('s1'))
        ed._create_autoclean_gate()
        n_ac2 = sum(1 for g in ed._sample_gates['s1'].values()
                    if g.get('kind') == 'autoclean')
        assert n_ac2 == 1
    finally:
        root.destroy()


def test_autoclean_toggle_replots_when_showing_removed():
    """Toggling an auto-clean gate while 'show cleaned-out events' is on must
    trigger a full replot so the removed-events overlay updates — it previously
    only redrew gate lines in 'all' mode, leaving the overlay stale."""
    root, ed = _editor_or_skip()
    try:
        _load_sample(ed, 's1')
        ed._set_active_sample('s1')
        ed._sample_gates['s1'] = {
            'g1': {'kind': 'autoclean', 'parent_id': None, 'id': 'g1',
                   'enabled': True, 'methods': []}}
        ed._sample_gate_order['s1'] = ['g1']
        ed._refresh_gate_list()
        ed.gate_display_var.set('all')        # not filter/highlight
        ed.show_removed_var.set(True)         # cleaned-out overlay ON
        calls = {'replot': 0, 'gates_only': 0}
        ed._schedule_replot = lambda *a, **k: calls.__setitem__(
            'replot', calls['replot'] + 1)
        ed._redraw_only_gates = lambda *a, **k: calls.__setitem__(
            'gates_only', calls['gates_only'] + 1)
        ed._press_selection = ()
        ed._handle_checkbox_click(ed._gate_iid('s1', 'g1'))
        assert calls['replot'] >= 1, calls
    finally:
        root.destroy()


def test_theme_menu_sets_disabled_foreground():
    """Popup menus get palette colours incl. disabledforeground, so a greyed-out
    item (e.g. Paste with nothing to paste) renders legibly on a dark theme."""
    root, ed = _editor_or_skip()
    try:
        import tkinter as tk
        m = tk.Menu(ed.gate_tv, tearoff=0)
        ed._theme_menu(m)
        assert str(m.cget('disabledforeground')) not in ('', 'SystemDisabledText')
    finally:
        root.destroy()


def test_queue_processed_loads_enqueues_csv_jobs(tmp_path):
    """Processed-data sidecars are queued on the same bounded pool as raw FCS
    (4-tuple 'csv' jobs) and shown as ⏳ rows, instead of loading synchronously
    on the Tk thread."""
    root, ed = _editor_or_skip()
    try:
        import queue as _q
        ed._load_pool_started = True          # don't spawn real worker threads
        ed._load_queue = _q.Queue()
        ed._load_total = 0
        c1 = tmp_path / 'a_events.csv'; c1.write_text('x\n1\n')
        c2 = tmp_path / 'b_events.csv'; c2.write_text('x\n1\n')
        ed._queue_processed_loads([('a', str(c1)), ('b', str(c2))])
        assert {'a', 'b'} <= ed._loading
        assert ed._load_total == 2
        payloads = []
        while not ed._load_queue.empty():
            _prio, _seq, payload = ed._load_queue.get_nowait()
            payloads.append(payload)
        assert all(len(p) == 4 and p[2] == 'csv' for p in payloads), payloads
        assert {p[0] for p in payloads} == {'a', 'b'}
        for nm in ('a', 'b'):
            assert ed.gate_tv.exists(ed._sample_iid(nm))
    finally:
        root.destroy()


def test_load_csv_worker_builds_sample_off_thread(tmp_path):
    """_load_csv_worker reads the CSV + builds the FlowSample and posts
    _on_loaded (here run inline) — no FCS QC, columns preserved."""
    root, ed = _editor_or_skip()
    try:
        import pandas as pd
        csv = tmp_path / 's.csv'
        pd.DataFrame({'FSC-A': [1.0, 2.0], 'CD3': [0.1, 0.2]}).to_csv(
            csv, index=False)
        captured = {}
        ed._on_loaded = lambda name, s: captured.update(name=name, s=s)
        ed.after = lambda _ms, fn=None: (fn() if fn else None)   # run inline
        ed._load_csv_worker('s', str(csv), {})
        assert captured.get('name') == 's'
        assert list(captured['s'].data.columns) == ['FSC-A', 'CD3']
    finally:
        root.destroy()


def test_pool_size_honors_preference_override(monkeypatch):
    """_pool_size returns the Preferences override (clamped 1–8) when set, else
    a hardware-derived default."""
    root, ed = _editor_or_skip()
    try:
        import openflo.editor_loadpool as lp
        monkeypatch.setattr(lp, 'read_prefs', lambda: {'load_workers': 5},
                            raising=False)
        # read_prefs is imported inside _pool_size via `from .prefs import …`,
        # so patch the source module too.
        import openflo.prefs as prefs
        monkeypatch.setattr(prefs, 'read_prefs', lambda: {'load_workers': 5})
        assert ed._pool_size() == 5
        monkeypatch.setattr(prefs, 'read_prefs', lambda: {'load_workers': 99})
        assert ed._pool_size() == 8        # clamped to max
        monkeypatch.setattr(prefs, 'read_prefs', lambda: {'load_workers': 0})
        assert ed._pool_size() == 1        # clamped to min
        monkeypatch.setattr(prefs, 'read_prefs', lambda: {})
        assert 1 <= ed._pool_size() <= 8   # auto default in range
    finally:
        root.destroy()


def test_load_priority_orders_front_sample_first():
    """_enqueue_load puts priority-0 jobs ahead of priority-1, FIFO within a
    priority — so the active/first sample loads before the rest."""
    root, ed = _editor_or_skip()
    try:
        import queue as _q
        ed._load_queue = _q.PriorityQueue()
        ed._load_seq = 0
        ed._enqueue_load(('late', '/x/late.fcs'), priority=1)
        ed._enqueue_load(('front', '/x/front.fcs'), priority=0)
        ed._enqueue_load(('late2', '/x/late2.fcs'), priority=1)
        order = []
        while not ed._load_queue.empty():
            _p, _s, payload = ed._load_queue.get_nowait()
            order.append(payload[0])
        assert order == ['front', 'late', 'late2'], order
    finally:
        root.destroy()


def test_run_async_editor_wrapper_runs_and_clears_busy():
    """ComputeMixin.run_async runs work off-thread and delivers the result via
    the editor's `after` marshaling, with the busy bar shown then cleared.
    `after` is stubbed inline so the cross-thread hand-off is deterministic in
    the (mainloop-less) test harness."""
    root, ed = _editor_or_skip()
    try:
        ed.after = lambda _ms, fn=None: (fn() if fn else None)
        got = {}
        t = ed.run_async(lambda: 'ok',
                         on_done=lambda r: got.__setitem__('r', r),
                         busy_msg='working…')
        t.join(timeout=3)
        assert got.get('r') == 'ok'
    finally:
        root.destroy()


def test_export_report_builds_html(tmp_path):
    """_export_report assembles the HTML report and writes it. The heavy build
    is now backgrounded; here run_async is stubbed to run inline so the file is
    written synchronously for the assertion."""
    import tkinter.filedialog as fd
    import webbrowser
    root, ed = _editor_or_skip()
    saved = (fd.asksaveasfilename, webbrowser.open)
    try:
        out = tmp_path / 'r.html'
        fd.asksaveasfilename = lambda *a, **k: str(out)
        webbrowser.open = lambda *a, **k: None
        ed.run_async = lambda work, on_done=None, on_error=None, busy_msg=None: (
            on_done(work()) if on_done else work())
        _load_sample(ed, 's1')
        ed._set_active_sample('s1')
        ed._replot()
        ed._export_report()
        assert out.is_file()
        html = out.read_text(encoding='utf-8')
        assert 'OpenFlo analysis report' in html and 'Samples &' in html
    finally:
        fd.asksaveasfilename, webbrowser.open = saved
        root.destroy()


def test_target_samples_modes():
    """_target_samples is the one selection→samples resolver, by mode."""
    root, ed = _editor_or_skip()
    try:
        for nm in ('s1', 's2', 's3'):
            _load_sample(ed, nm)
        ed._sample_plot_enabled['s2'] = False
        ed._set_active_sample('s2')
        ed._refresh_gate_list()
        assert ed._target_samples('all') == ['s1', 's2', 's3']
        assert ed._target_samples('enabled') == ['s1', 's3']
        assert ed._target_samples('active') == ['s2']
        ed.gate_tv.selection_set([ed._sample_iid('s1'), ed._sample_iid('s3')])
        assert ed._target_samples('selected') == ['s1', 's3']
        ed.gate_tv.selection_remove(*ed.gate_tv.selection())
        assert ed._target_samples('selected') == ['s2']        # active fallback
    finally:
        root.destroy()


def test_export_flowjo_wsp_writes(tmp_path):
    """_export_flowjo_wsp builds + writes the workspace (now backgrounded;
    run_async stubbed inline so the file exists for the assertion)."""
    import tkinter.filedialog as fd
    root, ed = _editor_or_skip()
    saved = fd.asksaveasfilename
    try:
        out = tmp_path / 'x.wsp'
        fd.asksaveasfilename = lambda *a, **k: str(out)
        ed.run_async = lambda work, on_done=None, on_error=None, busy_msg=None: (
            on_done(work()) if on_done else work())
        _load_sample(ed, 's1')
        ed._set_active_sample('s1')
        ed._export_flowjo_wsp()
        assert out.is_file()
    finally:
        fd.asksaveasfilename = saved
        root.destroy()


def test_gate_applies_to_all_selected_samples():
    """Drawing/auto gating with multiple samples selected (or the 'all shown'
    toggle on) fans the new gate out to every target, not just the active."""
    root, ed = _editor_or_skip()
    try:
        for nm in ('s1', 's2', 's3'):
            _load_sample(ed, nm)
        ed._set_active_sample('s1')
        ed._refresh_gate_list()
        # Toggle off → the tree multi-selection drives the targets.
        ed._gate_all_var.set(False)
        # Multi-select s1 + s2 (not s3).
        ed.gate_tv.selection_set([ed._sample_iid('s1'), ed._sample_iid('s2')])
        ed._add_gate_multi({'kind': 'rect', 'x_channel': 'FSC-A',
                            'y_channel': 'SSC-A', 'x0': 0.1, 'x1': 0.5,
                            'y0': 0.1, 'y1': 0.5})
        for nm in ('s1', 's2'):
            assert any(g.get('kind') == 'rect'
                       for g in ed._sample_gates.get(nm, {}).values()), nm
        assert not any(g.get('kind') == 'rect'
                       for g in ed._sample_gates.get('s3', {}).values())

        # The 'apply to all displayed' toggle fans out to every enabled sample.
        ed.gate_tv.selection_remove(*ed.gate_tv.selection())
        ed._gate_all_var.set(True)
        ed._add_gate_multi({'kind': 'threshold', 'channel': 'CD3',
                            'value': 0.5})
        for nm in ('s1', 's2', 's3'):
            assert any(g.get('kind') == 'threshold'
                       for g in ed._sample_gates.get(nm, {}).values()), nm
    finally:
        root.destroy()


def test_gate_single_selection_only_active():
    """With one sample selected and the toggle off, a new gate lands only on
    the active sample."""
    root, ed = _editor_or_skip()
    try:
        for nm in ('s1', 's2'):
            _load_sample(ed, nm)
        ed._set_active_sample('s1')
        ed._refresh_gate_list()
        ed._gate_all_var.set(False)              # not 'all shown'
        ed.gate_tv.selection_set(ed._sample_iid('s1'))
        ed._add_gate_multi({'kind': 'rect', 'x_channel': 'FSC-A',
                            'y_channel': 'SSC-A', 'x0': 0.1, 'x1': 0.5,
                            'y0': 0.1, 'y1': 0.5})
        assert any(g.get('kind') == 'rect'
                   for g in ed._sample_gates.get('s1', {}).values())
        assert not ed._sample_gates.get('s2', {})
    finally:
        root.destroy()


def test_gate_all_toggle_defaults_on():
    """'→ all shown' defaults to on, so a new gate applies to every displayed
    sample unless the user turns it off."""
    root, ed = _editor_or_skip()
    try:
        assert ed._gate_all_var.get() is True
    finally:
        root.destroy()
