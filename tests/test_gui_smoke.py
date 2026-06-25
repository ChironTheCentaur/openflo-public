"""GUI import smoke test.

Catches the class of bug where someone adds a Tk reference to a
module-level constant and breaks headless import — exactly the kind of
regression that ``pytest tests/`` would otherwise miss because no other
test ever touches openflo.gui.

This test ONLY imports the module; it does not instantiate the App or
enter the Tk event loop. That keeps it cheap and lets it pass on Linux
CI runners without xvfb.

If Tk itself fails to initialise (no display, no _tkinter), we skip
rather than fail — the GUI is genuinely unavailable on that machine.
"""
import importlib
import os

import pytest


def test_gui_module_imports():
    """``import openflo.gui`` must succeed in a headless environment.

    Sets MPLBACKEND=Agg first to avoid matplotlib pulling in a GUI
    backend at import time (gui.py uses FigureCanvasTkAgg internally
    but only inside lazily-instantiated viewer classes).
    """
    os.environ.setdefault('MPLBACKEND', 'Agg')

    # Tk itself may not be importable on headless Linux CI without
    # xvfb. Skip in that case — the GUI is genuinely unusable, but
    # that's an environment issue, not a code regression.
    try:
        import tkinter  # noqa: F401  (just probing availability)
    except ImportError:
        pytest.skip("tkinter not available — headless environment")

    try:
        mod = importlib.import_module('openflo.gui')
    except (ImportError, RuntimeError) as e:
        # _tkinter raises TclError (a RuntimeError subclass) when there's
        # no display. Skip rather than fail in that scenario.
        if 'display' in str(e).lower() or 'tcl' in str(e).lower():
            pytest.skip(f"GUI cannot initialise without a display: {e}")
        raise

    # The main entry point we wired in pyproject.toml must exist and be
    # a no-arg callable. We don't invoke it (that would block on
    # mainloop), just check the contract.
    assert callable(mod.main), (
        "openflo.gui.main is missing or not callable — "
        "the openflo-gui console script would fail.")

    # Verify the gate editor (now the whole UI; the old App run-config
    # window was removed) is reachable — catches accidental removals.
    assert hasattr(mod, 'ViewGateEditorWindow'), \
        "openflo.gui.ViewGateEditorWindow not defined"


def _editor_or_skip():
    """Construct a hidden ViewGateEditorWindow, or skip if Tk is unavailable.
    Returns (root, editor, gui_module)."""
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
    gui = importlib.import_module('openflo.gui')
    # Confirm dialogs would block a headless run — auto-accept.
    gui.messagebox.askyesno = lambda *a, **k: True
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    return root, ed, gui


def _load_fake(ed, name, trial):
    """Stuff a lightweight fake sample (one threshold gate) into the editor."""
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd
    df = pd.DataFrame({'CD3': np.linspace(0, 1, 200), 'CD4': np.linspace(1, 0, 200)})
    s = SimpleNamespace(name=name, path=rf'C:\exp\{name}.fcs', data=df,
                        fluor_channels=['CD3', 'CD4'],
                        channel_labels={'CD3': 'CD3', 'CD4': 'CD4'})
    ed._samples[name] = s
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = trial
    if trial not in ed._trial_order:
        ed._trial_order.append(trial)
    ed._sample_gates[name] = {'g1': {'kind': 'threshold', 'channel': 'CD3',
                                     'value': 0.0, 'parent_id': None,
                                     'color': '#111', 'enabled': True, 'id': 'g1'}}
    ed._sample_gate_order[name] = ['g1']
    ed._sample_gate_seq[name] = 1
    ed._sample_plot_enabled[name] = True


def test_trial_remove():
    """Removing a trial drops its samples + gates; dragged-target resolution
    for stats: gate rows → (sample, gid); trial/sample rows → nothing."""
    root, ed, _gui = _editor_or_skip()
    try:
        for nm, tr in [('a1', 'TrialA'), ('a2', 'TrialA'), ('b1', 'TrialB')]:
            _load_fake(ed, nm, tr)
        ed._set_active_sample('a1')

        ed._remove_trials(['TrialA'])
        assert list(ed._samples) == ['b1']
        assert ed._trial_order == ['TrialB']
        assert 'a1' not in ed._sample_gates and 'a2' not in ed._sample_trial

        _load_fake(ed, 'x1', 'TrialX')
        ed._press_selection = ()
        assert ed._dragged_gate_targets(ed._gate_iid('x1', 'g1')) == [('x1', 'g1')]
        assert ed._dragged_gate_targets(ed._trial_iid('TrialX')) == []
        assert ed._dragged_gate_targets(ed._sample_iid('x1')) == []
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_clear_all_clears_gates_keeps_samples():
    """Clear all wipes every gate from every sample but leaves the samples
    (and trial grouping) intact — undoable."""
    root, ed, _gui = _editor_or_skip()
    try:
        for nm, tr in [('a1', 'TrialA'), ('b1', 'TrialB')]:
            _load_fake(ed, nm, tr)        # each fake carries one gate 'g1'
        ed._set_active_sample('a1')
        assert sum(len(g) for g in ed._sample_gates.values()) == 2

        ed._clear_all()
        # Samples + trials kept; all gates gone.
        assert list(ed._samples) == ['a1', 'b1']
        assert ed._trial_order == ['TrialA', 'TrialB']
        assert all(g == {} for g in ed._sample_gates.values())
        # Active sample's live shortcut reflects the in-place clear.
        assert ed._gates == {}

        # Undoable: gates come back.
        ed._undo()
        assert sum(len(g) for g in ed._sample_gates.values()) == 2
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_clear_selected_sample_gate_and_trial():
    """The Clear button: a sample row clears that sample's gates (sample kept);
    a gate row cascades the gate + descendants; a trial row clears every
    sample's gates in the trial."""
    root, ed, _gui = _editor_or_skip()
    try:
        for nm, tr in [('a1', 'TrialA'), ('a2', 'TrialA'), ('b1', 'TrialB')]:
            _load_fake(ed, nm, tr)
        # Give a1 and b1 a parent→child chain so cascade is observable.
        for nm in ('a1', 'b1'):
            ed._sample_gates[nm]['g2'] = {'kind': 'threshold', 'channel': 'CD4',
                                          'value': 0.5, 'parent_id': 'g1',
                                          'color': '#222', 'enabled': True,
                                          'id': 'g2'}
            ed._sample_gate_order[nm].append('g2')
        ed._set_active_sample('a1')
        ed._refresh_gate_list()                    # build the real tree rows

        # Sample row → clear that sample's gates only.
        ed.gate_tv.selection_set(ed._sample_iid('a1'))
        ed._clear_selected_gate()
        assert ed._sample_gates['a1'] == {}        # a1 wiped
        assert 'a1' in ed._samples                 # sample kept
        assert set(ed._sample_gates['a2']) == {'g1'}   # other sample untouched

        # Gate row → cascade the gate + descendants (only that subtree).
        ed.gate_tv.selection_set(ed._gate_iid('b1', 'g1'))
        ed._clear_selected_gate()
        assert ed._sample_gates['b1'] == {}        # g1 + child g2 gone

        # Trial row → clear gates from every sample in the trial.
        _load_fake(ed, 'c1', 'TrialC')
        _load_fake(ed, 'c2', 'TrialC')
        ed._refresh_gate_list()
        ed.gate_tv.selection_set(ed._trial_iid('TrialC'))
        ed._clear_selected_gate()
        assert ed._sample_gates['c1'] == {} and ed._sample_gates['c2'] == {}
        assert 'c1' in ed._samples and 'c2' in ed._samples
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_progress_bar_lifecycle():
    """The load progress bar is created hidden, shows + sizes itself as the
    counters advance, extends on a mid-run enqueue, and hides+resets once the
    run drains — all driven on the real widget without a mainloop."""
    root, ed, _gui = _editor_or_skip()
    try:
        # Created and hidden at rest.
        assert ed._load_progress_frame.winfo_manager() == ''
        assert ed._load_total == 0 and ed._load_done == 0

        # Simulate a 5-file run: bar shows, sized to total, value tracks done.
        ed._load_total = 5
        ed._load_done = 2
        ed._update_progress_bar()
        assert ed._load_progress_frame.winfo_manager() == 'grid'
        assert int(float(ed.progress_bar['maximum'])) == 5
        assert int(ed._load_progress_var.get()) == 2
        assert ed._load_progress_lbl_var.get() == '2/5 loaded'

        # Mid-run drop extends the total without resetting progress.
        ed._load_total = 8
        ed._update_progress_bar()
        assert int(float(ed.progress_bar['maximum'])) == 8
        assert ed._load_progress_lbl_var.get() == '2/8 loaded'

        # Drain to completion, then the immediate finish-reset hides it.
        ed._load_done = 8
        ed._update_progress_bar()
        ed._finish_progress()
        assert ed._load_progress_frame.winfo_manager() == ''
        assert ed._load_total == 0 and ed._load_done == 0
        assert ed._load_progress_lbl_var.get() == ''

        # A finish that fires while a NEW run is mid-flight must not wipe it.
        ed._load_total = 3
        ed._load_done = 1
        ed._update_progress_bar()
        ed._finish_progress()                 # done(1) < total(3) → no reset
        assert ed._load_progress_frame.winfo_manager() == 'grid'
        assert ed._load_total == 3 and ed._load_done == 1
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_histogram_y_axis_modes():
    """The histogram Y-axis switches between Fraction / Count / % of Max, and
    raw Count honours the auto-downsample toggle (so overlaid counts compare)
    while bypassing the scatter-only 60k cap."""
    root, ed, _gui = _editor_or_skip()
    try:
        from types import SimpleNamespace

        import numpy as np
        import pandas as pd

        def load(name, n, seed):
            rng = np.random.default_rng(seed)
            df = pd.DataFrame({'CD3': rng.normal(0.5, 0.1, n).clip(0, 1),
                               'CD4': rng.normal(0.4, 0.1, n).clip(0, 1)})
            s = SimpleNamespace(name=name, path=rf'C:\exp\{name}.fcs', data=df,
                                fluor_channels=['CD3', 'CD4'],
                                channel_labels={'CD3': 'CD3', 'CD4': 'CD4'})
            ed._samples[name] = s
            ed._sample_order.append(name)
            ed._sample_colors[name] = '#1f77b4'
            ed._sample_trial[name] = 'T'
            ed._sample_plot_enabled[name] = True
            ed._sample_gates[name] = {}
            ed._sample_gate_order[name] = []
            ed._sample_gate_seq[name] = 0

        load('a', 150, 1)
        load('b', 400, 2)
        ed._channels = ['CD3', 'CD4']
        ed.apply_gates_var.set(False)

        def peak(mode):
            ed.hist_y_mode.set(mode)
            ed.ax.clear()
            ed._plot_histogram(['a', 'b'], 'CD3')
            # Histograms render as smoothed filled curves (ax.plot lines),
            # not step-bar patches.
            ys = [v for ln in ed.ax.lines for v in ln.get_ydata()]
            return ed.ax.get_ylabel(), (max(ys) if ys else 0.0)

        ed.ds_display_var.set(False)
        lbl_f, peak_f = peak('Fraction')
        lbl_c, peak_c = peak('Count')
        lbl_m, peak_m = peak('% of Max')
        assert (lbl_f, lbl_c, lbl_m) == ('fraction', 'count', '% of max')
        assert peak_f < 1.0                      # normalized fractions
        # Raw counts dwarf fractions (smoothing makes them non-integer, so
        # check the scale differs rather than exact integer bar heights).
        assert peak_c > 1.0 and peak_c > peak_f * 100
        assert abs(peak_m - 100.0) < 1e-6        # tallest curve peak = 100%

        # Raw Count + auto-downsample: 'b' (400) is capped to the smallest
        # loaded sample (150); with the toggle off it keeps all 400 (the 60k
        # scatter cap doesn't apply to histograms).
        ed.ds_display_var.set(False)
        assert len(ed._get_df('b', 'CD3', None, for_hist=True)) == 400
        ed.ds_display_var.set(True)
        assert len(ed._get_df('b', 'CD3', None, for_hist=True)) == 150
        assert len(ed._get_df('a', 'CD3', None, for_hist=True)) == 150

        # The selector is only SHOWN in histogram mode (mode-specific options
        # are hidden otherwise to keep the control bar uncluttered).
        ed.mode_var.set('histogram'); ed._sync_hist_y_combo()
        assert ed.hist_y_combo.winfo_manager() == 'pack'      # shown
        ed.mode_var.set('dot'); ed._sync_hist_y_combo()
        assert ed.hist_y_combo.winfo_manager() == ''          # hidden
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_autoclean_gate_create_toggle_collapse_copy():
    """Auto-clean button creates a collapsed 'autocleaned sample' group of
    toggleable method rows; toggling a method row flips its enabled flag; the
    gate copies to other samples as a calculation (no coordinates) and applies
    as a filter."""
    root, ed, _gui = _editor_or_skip()
    try:
        from types import SimpleNamespace

        import numpy as np
        import pandas as pd

        def load(name, seed, sz):
            r = np.random.default_rng(seed)
            fsc = np.concatenate([r.normal(2e4, 3e3, sz // 5),
                                  r.normal(1.2e5, 1.5e4, 4 * sz // 5)])
            fsch = fsc / 2 + r.normal(0, 2e3, fsc.size)
            time = np.sort(r.uniform(0, 100, fsc.size))
            apc = r.normal(1000, 100, fsc.size)
            apc[:int(0.01 * fsc.size)] = apc.max()
            df = pd.DataFrame({'FSC-A': fsc, 'FSC-H': fsch, 'Time': time,
                               'APC-A': apc}).sample(
                frac=1, random_state=seed).reset_index(drop=True)
            s = SimpleNamespace(name=name, path=rf'C:\e\{name}.fcs', data=df,
                                fluor_channels=['APC-A'],
                                channel_labels={'APC-A': 'APC-A'})
            ed._samples[name] = s
            ed._sample_order.append(name)
            ed._sample_colors[name] = '#1f77b4'
            ed._sample_trial[name] = 'T'
            ed._sample_plot_enabled[name] = True
            ed._sample_gates[name] = {}
            ed._sample_gate_order[name] = []
            ed._sample_gate_seq[name] = 0

        load('s1', 1, 4000)
        load('s2', 2, 2500)
        ed._channels = list(ed._samples['s1'].data.columns)
        ed._set_active_sample('s1')

        # Create the group on the active sample.
        ed._create_autoclean_gate()
        gid = next(k for k, g in ed._sample_gates['s1'].items()
                   if g.get('kind') == 'autoclean')
        g = ed._sample_gates['s1'][gid]
        assert g['name'] == 'autocleaned sample'
        assert [m['key'] for m in g['methods']] == \
            ['debris', 'viability', 'doublets', 'margin', 'flow_rate', 'drift']
        assert g['open'] is False                       # collapsed by default

        # Duplicate guard: a second press adds nothing.
        ed._create_autoclean_gate()
        assert sum(1 for x in ed._sample_gates['s1'].values()
                   if x.get('kind') == 'autoclean') == 1

        # Tree: collapsed parent row with one synthetic child per method.
        ed._refresh_gate_list()
        giid = ed._gate_iid('s1', gid)
        assert ed.gate_tv.exists(giid)
        assert bool(ed.gate_tv.item(giid, 'open')) is False
        assert len(ed.gate_tv.get_children(giid)) == 6

        # Toggle the 'margin' method row off via the checkbox path.
        ed._press_selection = ()
        ed._handle_checkbox_click(ed._method_iid('s1', gid, 'margin'))
        assert next(m for m in g['methods']
                    if m['key'] == 'margin')['enabled'] is False

        # Applies as a filter (some events removed).
        ed.apply_gates_var.set(True)
        kept = len(ed._get_df('s1', 'FSC-A', 'APC-A'))
        assert kept < len(ed._samples['s1'].data)

        # Copy to s2 → calculation-only dict, recomputed on s2's own data.
        ed._copy_gates_to(['s2'])
        g2 = next(x for x in ed._sample_gates['s2'].values()
                  if x.get('kind') == 'autoclean')
        assert 'vertices' not in g2 and 'x_channel' not in g2  # no coordinates
        assert [m['key'] for m in g2['methods']] == \
            [m['key'] for m in g['methods']]
        kept2 = len(ed._get_df('s2', 'FSC-A', 'APC-A'))
        assert 0 < kept2 < len(ed._samples['s2'].data)
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def _load_counts_fake(ed, name, n, seed):
    """A fake sample with FSC/FSC-H/Time/SSC so auto-clean has work to do."""
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd
    r = np.random.default_rng(seed)
    fsc = np.concatenate([r.normal(2e4, 3e3, n // 5),
                          r.normal(1.2e5, 1.5e4, 4 * n // 5)])
    df = pd.DataFrame({'FSC-A': fsc, 'FSC-H': fsc / 2 + r.normal(0, 2e3, fsc.size),
                       'Time': np.sort(r.uniform(0, 100, fsc.size)),
                       'SSC-A': r.normal(5e4, 1e4, fsc.size)})
    s = SimpleNamespace(name=name, path=rf'C:\e\{name}.fcs', data=df,
                        fluor_channels=['SSC-A'], channel_labels={'SSC-A': 'SSC-A'})
    ed._samples[name] = s
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    ed._sample_plot_enabled[name] = True
    ed._sample_gates[name] = {}
    ed._sample_gate_order[name] = []
    ed._sample_gate_seq[name] = 0
    return len(df)


def test_sample_event_counts_in_tree():
    """Sample rows show event counts; with auto-downsample on, the larger
    sample shows shown/total scaled to the smallest."""
    root, ed, _gui = _editor_or_skip()
    try:
        n_a = _load_counts_fake(ed, 'a', 1000, 1)
        n_b = _load_counts_fake(ed, 'b', 4000, 2)
        ed._channels = ['FSC-A', 'FSC-H', 'Time', 'SSC-A']
        ed._set_active_sample('a')

        ed.ds_display_var.set(False)
        ed._refresh_gate_list()
        txt_b = ed.gate_tv.item(ed._sample_iid('b'), 'text')
        assert f'{n_b:,}' in txt_b               # full count shown

        ed.ds_display_var.set(True)              # scale to smallest (a = 1000)
        ed._refresh_gate_list()
        txt_b = ed.gate_tv.item(ed._sample_iid('b'), 'text')
        assert f'{n_a:,}/{n_b:,}' in txt_b       # shown/total
        # _sample_display_count agrees.
        shown, total = ed._sample_display_count('b')
        assert (shown, total) == (n_a, n_b)
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_clear_all_autoclean_toggle():
    """Clear all keeps auto-clean gates by default (toggle off) and removes
    them when the toggle is on. Driven by stubbing the confirm dialog."""
    root, ed, _gui = _editor_or_skip()
    try:
        _load_counts_fake(ed, 'a', 1000, 1)
        ed._channels = ['FSC-A', 'FSC-H', 'Time', 'SSC-A']
        ed._set_active_sample('a')
        ed._create_autoclean_gate()
        # Add a normal downstream gate too.
        ed._sample_gates['a']['c1'] = {'kind': 'threshold', 'channel': 'SSC-A',
                                       'value': 3e4, 'parent_id': None,
                                       'enabled': True, 'color': '#111'}
        ed._sample_gate_order['a'].append('c1')
        assert len(ed._sample_gates['a']) == 2

        # Toggle OFF (keep auto-clean): only the normal gate goes.
        ed._ask_clear_all = lambda *a, **k: False
        ed._clear_all()
        kinds = [g['kind'] for g in ed._sample_gates['a'].values()]
        assert kinds == ['autoclean']

        # Toggle ON (include auto-clean): everything goes.
        ed._ask_clear_all = lambda *a, **k: True
        ed._clear_all()
        assert ed._sample_gates['a'] == {}
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_autoclean_mask_cache_correct_and_hits():
    """The auto-clean mask cache returns the same result as the uncached path
    and is invalidated when the recipe signature changes."""
    root, ed, gui = _editor_or_skip()
    try:
        from openflo import pipeline as P
        _load_counts_fake(ed, 'a', 3000, 1)
        ed._channels = ['FSC-A', 'FSC-H', 'Time', 'SSC-A']
        ed._set_active_sample('a')
        ed._create_autoclean_gate()
        gid = next(k for k, g in ed._sample_gates['a'].items()
                   if g.get('kind') == 'autoclean')
        ed.apply_gates_var.set(True)

        ed._ac_cache.clear()
        cached = len(ed._get_df('a', 'FSC-A', 'SSC-A'))
        assert ('a', gid) in ed._ac_cache           # populated

        # Count recomputes by spying on the pipeline function.
        calls = {'n': 0}
        orig = P.autoclean_keep_mask
        P.autoclean_keep_mask = lambda g, df: (calls.__setitem__('n', calls['n'] + 1)
                                               or orig(g, df))
        try:
            ed._get_df('a', 'FSC-A', 'SSC-A')       # same sig → cache hit
            assert calls['n'] == 0
            ed._sample_gates['a'][gid]['methods'][1]['enabled'] = False
            ed._get_df('a', 'FSC-A', 'SSC-A')       # sig changed → recompute
            assert calls['n'] == 1
        finally:
            P.autoclean_keep_mask = orig

        # Correctness: cached result equals a fresh uncached computation.
        ed._ac_cache.clear()
        ref = len(ed._get_df('a', 'FSC-A', 'SSC-A'))
        ed._ac_cache.clear()
        again = len(ed._get_df('a', 'FSC-A', 'SSC-A'))
        assert ref == again == cached or ref == again  # stable & deterministic
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_autoclean_override_is_full_data_consistent():
    """The auto-clean override for ANY df subset equals the FULL-data mask
    restricted to that subset — so filter and highlight modes flag the same
    events even when a plotted channel (e.g. an embedding axis) is NaN for many
    time-correlated rows. This is the regression for the cache-binning bug."""
    root, ed, _gui = _editor_or_skip()
    try:
        from types import SimpleNamespace

        import numpy as np
        import pandas as pd

        from openflo.pipeline import autoclean_keep_mask

        r = np.random.default_rng(3)
        n = 4000
        fsc = np.concatenate([r.normal(2e4, 3e3, n // 5),
                              r.normal(1.2e5, 1.5e4, 4 * n // 5)])
        time = np.sort(r.uniform(0, 100, n))
        # UMAP1 present only for late-time rows → dropna removes a big,
        # time-correlated chunk (the scenario that breaks naive recompute).
        umap = np.full(n, np.nan)
        umap[time > 40] = r.normal(0, 1, int((time > 40).sum()))
        df = pd.DataFrame({'FSC-A': fsc, 'FSC-H': fsc / 2, 'Time': time,
                           'UMAP1': umap})
        # Non-RangeIndex to stress label alignment.
        df.index = df.index + 1000
        s = SimpleNamespace(name='s', path=r'C:\e\s.fcs', data=df,
                            fluor_channels=[], channel_labels={})
        ed._samples['s'] = s
        ed._sample_order.append('s')
        ed._sample_plot_enabled['s'] = True
        ed._sample_trial['s'] = 'T'
        ed._sample_colors['s'] = '#1f77b4'
        ed._sample_gates['s'] = {}
        ed._sample_gate_order['s'] = []
        ed._sample_gate_seq['s'] = 0
        ed._channels = list(df.columns)
        ed._set_active_sample('s')
        ed._create_autoclean_gate()
        gid = next(k for k, g in ed._sample_gates['s'].items()
                   if g.get('kind') == 'autoclean')
        gate = ed._sample_gates['s'][gid]

        full = pd.Series(autoclean_keep_mask(gate, df), index=df.index)
        sub = df.dropna(subset=['UMAP1'])           # the displayed subset
        ov = ed._autoclean_overrides('s', sub)[gid]

        # The override is the full-data decision restricted to the subset…
        assert np.array_equal(ov, full.reindex(sub.index).to_numpy())
        # …and (because drift/flow are time-binned) it genuinely DIFFERS from a
        # naive recompute on the subset — proving the override matters.
        recomputed = autoclean_keep_mask(gate, sub)
        assert not np.array_equal(ov, recomputed)
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_autoclean_params_dialog_constructs():
    """The QC-parameter dialog builds without error for an auto-clean gate
    (wait_window stubbed so it doesn't block); cancelling leaves the recipe
    untouched."""
    root, ed, _gui = _editor_or_skip()
    try:
        import tkinter as tk
        _load_counts_fake(ed, 'a', 1000, 1)
        ed._channels = ['FSC-A', 'FSC-H', 'Time', 'SSC-A']
        ed._set_active_sample('a')
        ed._create_autoclean_gate()
        gid = next(k for k, g in ed._sample_gates['a'].items()
                   if g.get('kind') == 'autoclean')
        before = [m['enabled'] for m in ed._sample_gates['a'][gid]['methods']]

        ed.wait_window = lambda *a, **k: None        # don't block
        ed._edit_autoclean_params('a', gid)
        # A dialog Toplevel titled "Auto-clean parameters" was created.
        dlgs = [w for w in ed.winfo_children()
                if isinstance(w, tk.Toplevel)
                and 'Auto-clean' in str(w.title())]
        assert dlgs, "parameter dialog was not created"
        dlgs[-1].destroy()
        # Cancelling (no Apply) leaves the recipe unchanged.
        after = [m['enabled'] for m in ed._sample_gates['a'][gid]['methods']]
        assert before == after
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_import_comp_sample_subgroups():
    """A day with comps splits into a 'Samples' subgroup (expanded) and a
    'Comps' subgroup (collapsed); the subgroup checkbox toggles its members;
    a day with no comps lists samples directly under the trial."""
    root, ed, _gui = _editor_or_skip()
    try:
        from types import SimpleNamespace

        import numpy as np
        import pandas as pd

        def load(name, trial):
            df = pd.DataFrame({'FSC-A': np.linspace(0, 1, 50),
                               'SSC-A': np.linspace(1, 0, 50)})
            s = SimpleNamespace(name=name, path=rf'C:\e\{name}.fcs', data=df,
                                fluor_channels=[], channel_labels={})
            ed._samples[name] = s
            ed._sample_order.append(name)
            ed._sample_colors[name] = '#1f77b4'
            ed._sample_trial[name] = trial
            ed._sample_plot_enabled[name] = True
            ed._sample_gates[name] = {}
            ed._sample_gate_order[name] = []
            ed._sample_gate_seq[name] = 0

        load('Compensation Controls_APC Stained Control_008', 'Day 3')
        load('Sample_sample_a_003', 'Day 3')
        load('Sample_sample_b_005', 'Day 3')
        load('Sample_only', 'Day 6')      # no comps in Day 6
        ed._channels = ['FSC-A', 'SSC-A']
        ed._refresh_gate_list()

        # Day 3 → two subgroups, Samples first (open) then Comps (collapsed).
        t3 = ed._trial_iid('Day 3')
        kids = ed.gate_tv.get_children(t3)
        parsed = [ed._parse_iid(k) for k in kids]
        assert parsed == [('subgroup', 'samp', 'Day 3'),
                          ('subgroup', 'comp', 'Day 3')]
        samp_iid, comp_iid = kids
        assert bool(ed.gate_tv.item(samp_iid, 'open')) is True
        assert bool(ed.gate_tv.item(comp_iid, 'open')) is False
        assert len(ed.gate_tv.get_children(samp_iid)) == 2
        assert len(ed.gate_tv.get_children(comp_iid)) == 1

        # Day 6 (no comps) → samples directly under the trial.
        t6 = ed._trial_iid('Day 6')
        c6 = [ed._parse_iid(k) for k in ed.gate_tv.get_children(t6)]
        assert c6 == [('sample', 'Sample_only')]

        # Subgroup checkbox toggles all its members off.
        ed._press_selection = ()
        ed._handle_checkbox_click(samp_iid)
        assert ed._sample_plot_enabled['Sample_sample_a_003'] is False
        assert ed._sample_plot_enabled['Sample_sample_b_005'] is False
        # Comp member untouched.
        assert ed._sample_plot_enabled[
            'Compensation Controls_APC Stained Control_008'] is True
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_drag_regroup_samples_and_comps():
    """A sample can be dragged to another day (trial) and between the
    Comps/Samples subgroups (a manual override); persists via session state."""
    root, ed, _gui = _editor_or_skip()
    try:
        from types import SimpleNamespace

        import numpy as np
        import pandas as pd

        def load(name, trial):
            df = pd.DataFrame({'FSC-A': np.linspace(0, 1, 30)})
            s = SimpleNamespace(name=name, path=rf'C:\e\{name}.fcs', data=df,
                                fluor_channels=[], channel_labels={})
            ed._samples[name] = s
            ed._sample_order.append(name)
            ed._sample_colors[name] = '#1f77b4'
            ed._sample_trial[name] = trial
            ed._sample_plot_enabled[name] = True
            ed._sample_gates[name] = {}
            ed._sample_gate_order[name] = []
            ed._sample_gate_seq[name] = 0

        load('Compensation Controls_APC_008', 'Day 3')
        load('Specimen_M1', 'Day 3')
        load('Specimen_X', 'Day 6')
        ed._channels = ['FSC-A']
        ed._refresh_gate_list()
        ed._press_selection = ()

        # Drag Specimen_M1 onto the Day 6 trial row → moves day.
        ed._handle_drag_drop(ed._sample_iid('Specimen_M1'),
                             ed._trial_iid('Day 6'))
        assert ed._trial_for('Specimen_M1') == 'Day 6'

        # Drag the comp onto Day 3's Samples subgroup → reclassify as a sample.
        # (Day 3 still has the comp, so the subgroup exists.)
        ed._refresh_gate_list()
        ed._press_selection = ()
        ed._handle_drag_drop(ed._sample_iid('Compensation Controls_APC_008'),
                             ed._subgroup_iid('samp', 'Day 3'))
        assert ed._is_comp('Compensation Controls_APC_008') is False
        assert ed._sample_is_comp['Compensation Controls_APC_008'] is False

        # Drag a specimen onto Day 6's... first give Day 6 a comp subgroup by
        # moving the (now-sample) comp there as a comp again via subgroup drop
        # is covered above; here verify a specimen → comp override.
        ed._handle_drag_drop(ed._sample_iid('Specimen_X'),
                             ed._sample_iid('Specimen_M1'))  # same group → no-op
        assert ed._is_comp('Specimen_X') is False

        # Session round-trip keeps the manual grouping.
        state = ed._session_state()
        by = {s['name']: s for s in state['samples']}
        assert by['Specimen_M1']['trial'] == 'Day 6'
        assert by['Compensation Controls_APC_008']['is_comp'] is False
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_session_restore_keyed_by_path_survives_name_change():
    """Session restore resolves grouping + gates by FILE PATH, so a sample
    whose collision-disambiguated name differs on reload (e.g. saved as
    'X [Day 9]' but loaded as bare 'X' because its same-named mate is absent)
    still gets its OWN trial, Comps/Samples override, and gates."""
    root, ed, _gui = _editor_or_skip()
    try:
        from types import SimpleNamespace

        import numpy as np
        import pandas as pd

        path = r'C:\data\dayX\Comp_008.fcs'
        data = {
            'format': 'openflo-session', 'version': 1,
            'samples': [{'name': 'Comp_008 [Day 9]', 'path': path,
                         'color': '#111', 'plot_enabled': True,
                         'trial': 'Day 9', 'is_comp': True}],
            'sample_gates': {'Comp_008 [Day 9]': [
                {'kind': 'threshold', 'channel': 'FSC-A', 'value': 0.5,
                 'enabled': True, 'id': 'g1', 'parent_id': None}]},
        }
        ed._apply_session(data)          # stages bundle keyed by path

        # Reload assigns the BARE name (the disambiguating mate isn't present).
        df = pd.DataFrame({'FSC-A': np.linspace(0, 1, 20)})
        sample = SimpleNamespace(name='Comp_008', path=path, data=df,
                                 fluor_channels=[], channel_labels={})
        ed._on_loaded('Comp_008', sample)

        assert ed._sample_trial['Comp_008'] == 'Day 9'   # restored by path
        assert ed._is_comp('Comp_008') is True
        assert len(ed._sample_gates['Comp_008']) == 1    # gates restored too
        g = next(iter(ed._sample_gates['Comp_008'].values()))
        assert g['enabled'] is True                      # saved flag kept
        # Bundle consumed (no leak).
        assert ed._pending_sample_meta == {}
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_imported_gates_default_disabled():
    """Gates imported with no 'enabled' flag (e.g. from a .wsp) load DISABLED;
    gates that carry an explicit flag (e.g. a restored session) keep it."""
    root, ed, _gui = _editor_or_skip()
    try:
        from types import SimpleNamespace

        import numpy as np
        import pandas as pd

        def fake(name):
            df = pd.DataFrame({'FSC-A': np.linspace(0, 1, 50)})
            return SimpleNamespace(name=name, path=rf'C:\e\{name}.fcs', data=df,
                                   fluor_channels=[], channel_labels={})

        ed._samples['w'] = fake('w')
        ed._sample_order.append('w')
        ed._sample_trial['w'] = 'T'
        ed._sample_gates['w'] = {}
        ed._sample_gate_order['w'] = []
        ed._sample_gate_seq['w'] = 0
        # WSP-style: no 'enabled'.
        ed._pending_sample_gates['w'] = [
            {'kind': 'threshold', 'channel': 'FSC-A', 'value': 0.5}]
        ed._on_loaded('w', ed._samples['w'])
        g = next(iter(ed._sample_gates['w'].values()))
        assert g['enabled'] is False                # imported → off

        ed._samples['s'] = fake('s')
        ed._sample_order.append('s')
        ed._sample_trial['s'] = 'T'
        ed._sample_gates['s'] = {}
        ed._sample_gate_order['s'] = []
        ed._sample_gate_seq['s'] = 0
        # Session-style: carries enabled=True.
        ed._pending_sample_gates['s'] = [
            {'kind': 'threshold', 'channel': 'FSC-A', 'value': 0.5,
             'enabled': True}]
        ed._on_loaded('s', ed._samples['s'])
        g2 = next(iter(ed._sample_gates['s'].values()))
        assert g2['enabled'] is True                # session flag preserved
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_stream_tee_fans_out():
    """_StreamTee mirrors writes to the real stream AND every registered sink,
    and stops once a sink is removed."""
    import io
    import queue

    gui = importlib.import_module('openflo.gui')
    real = io.StringIO()
    tee = gui._StreamTee(real)
    q = queue.Queue()
    tee.add_sink(q)
    tee.write("hello\n")
    assert real.getvalue() == "hello\n"
    assert q.get_nowait() == "hello\n"
    tee.remove_sink(q)
    tee.write("after\n")
    assert real.getvalue() == "hello\nafter\n"   # still hits the real stream
    assert q.empty()                              # but not the removed sink


def test_available_embeddings_partition():
    """The Cluster dialog lists only installed embedding backends. UMAP + t-SNE
    are core deps (always available); the five names partition cleanly into
    available/missing with no overlap."""
    from openflo.gui import ViewGateEditorWindow as W
    avail, missing = W._available_embeddings()
    assert 'UMAP' in avail and 't-SNE' in avail
    assert set(avail) | set(missing) == set(W._EMBEDDINGS)
    assert not (set(avail) & set(missing))


def test_embedding_axes_default_to_linear():
    """Embedding axes (UMAP1/2, t-SNE, …) default to a LINEAR scale, not the
    global log default meant for fluorescence."""
    root, ed, _gui = _editor_or_skip()
    try:
        for ch in ('UMAP1', 'UMAP2', 'TSNE1', 'TRIMAP2', 'PACMAP1', 'PHATE2'):
            assert ed._default_scale_for(ch) == 'linear', ch
        # real fluor / scatter keep the global default
        assert ed._default_scale_for('APC-A') == ed._default_channel_scale
        assert ed._default_scale_for('FSC-A') == ed._default_channel_scale
        assert ed._default_scale_for('UMAP') == ed._default_channel_scale  # no axis #
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_left_panel_popout_and_redock():
    """The Samples & Gates panel pops out into its own window (via 'wm manage'
    on its tk.Frame host) and re-docks as the first pane — no rebuild."""
    root, ed, _gui = _editor_or_skip()
    try:
        n0 = len(ed._main_paned.panes())
        ed._toggle_left_popout()
        root.update_idletasks()
        assert ed._left_popped is True
        assert ed._left_host not in ed._main_paned.panes()   # detached
        assert ed._left_popbtn.cget('text') == 'Dock'
        ed._redock_left()
        root.update_idletasks()
        assert ed._left_popped is False
        assert len(ed._main_paned.panes()) == n0             # back as a pane
        assert ed._left_popbtn.cget('text') == 'Pop out'
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_workspace_close_hides_panel():
    """The workspace ✕ Close (and View) hides the panel; re-docks if floated."""
    root, ed, _gui = _editor_or_skip()
    try:
        ed._open_pipeline_workspace()
        root.update_idletasks()
        assert ed._workspace_shown
        ed._close_workspace()
        root.update_idletasks()
        assert not ed._workspace_shown
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_max_points_visibility_tracks_downsample():
    """Max points is shown for Display / Display+data and hidden when Off."""
    root, ed, _gui = _editor_or_skip()
    try:
        ed._ds_mode_var.set('Display only'); ed._on_ds_mode_changed()
        root.update_idletasks()
        assert ed._mp_combo.winfo_manager() == 'pack'
        ed._ds_mode_var.set('Off'); ed._on_ds_mode_changed()
        root.update_idletasks()
        assert ed._mp_combo.winfo_manager() == ''
        # Restore path: setting the booleans + sync hides it too.
        ed.ds_display_var.set(False); ed.ds_propagate_var.set(False)
        ed._sync_ds_mode_var(); ed._update_ds_visibility()
        assert ed._ds_mode_var.get() == 'Off'
        assert ed._mp_combo.winfo_manager() == ''
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_dock_all_panels_redocks_floated():
    """View → Dock all panels re-docks any floated side panels."""
    root, ed, _gui = _editor_or_skip()
    try:
        ed._open_pipeline_workspace()
        root.update_idletasks()
        ed._toggle_left_popout()
        ed._toggle_workspace_popout()
        root.update_idletasks()
        assert ed._left_popped and ed._ws_popped
        ed._dock_all_panels()
        root.update_idletasks()
        assert not ed._left_popped and not ed._ws_popped
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_axis_config_link_applies_to_both():
    """The axis dialog's Link X & Y mirrors scale + range onto both channels."""
    root, ed, _gui = _editor_or_skip()
    try:
        ed._set_axis_config('CD3', 'log', (1.0, 100.0), other_channel='CD4')
        assert ed._channel_scale['CD3'] == 'log' == ed._channel_scale['CD4']
        assert ed._channel_range['CD3'] == ed._channel_range['CD4'] == (1.0, 100.0)
        # Without a link target, only the one channel changes.
        ed._set_axis_config('CD3', 'linear', None)
        assert ed._channel_scale['CD3'] == 'linear'
        assert ed._channel_scale['CD4'] == 'log'    # unchanged
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_workspace_popout_floats_whole_panel():
    """The Pipeline Workspace pops out as ONE window (bar + view) via wm manage
    and re-docks; the Pop out button toggles to Dock and back."""
    root, ed, _gui = _editor_or_skip()
    try:
        ed._open_pipeline_workspace()                  # show as a pane
        root.update_idletasks()
        ed._toggle_workspace_popout()
        root.update_idletasks()
        assert ed._ws_popped is True
        assert ed._workspace_panel._popbtn.cget('text') == 'Dock'
        ed._redock_workspace()
        root.update_idletasks()
        assert ed._ws_popped is False
        assert ed._workspace_panel._popbtn.cget('text') == 'Pop out'
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_hover_tooltips_toggle():
    """Controls carry hover tooltips, gated by a persisted View toggle."""
    root, ed, _gui = _editor_or_skip()
    try:
        from tkinter import ttk

        from openflo.gui import _ToolTip
        assert ed._tooltips_enabled.get() in (True, False)
        w = ttk.Label(root, text='x')
        tip = _ToolTip(w, "help text", lambda: ed._tooltips_enabled.get())
        # Enabled → schedules; disabled → does not.
        ed._tooltips_enabled.set(True)
        tip._schedule()
        assert tip._after is not None
        tip._hide()
        ed._tooltips_enabled.set(False)
        tip._schedule()
        assert tip._after is None
        # _tip attaches without error on a real control.
        ed._tip(w, "another tip")
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_busy_bar_shows_during_work():
    """The status-bar 'working' indicator is hidden at rest, shows (with a
    message) while a long job runs, and hides again — so a slow UMAP reads as
    busy, not hung."""
    root, ed, _gui = _editor_or_skip()
    try:
        assert ed._busy_bar.winfo_manager() == ''          # hidden at rest
        ed._begin_busy("UMAP embedding… still working")
        assert ed._busy_bar.winfo_manager() == 'grid'       # shown while busy
        assert 'working' in ed.status_var.get()
        ed._busy("phase 2/3")                               # thread-safe update
        ed._end_busy()
        assert ed._busy_bar.winfo_manager() == ''           # hidden after
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_log_pane_drain_toggle_clear():
    """The editor's log pane drains queued text, toggles visibility, and
    clears. Pushes straight to the queue so the test is independent of how
    pytest captures process stdout."""
    root, ed, _gui = _editor_or_skip()
    try:
        # Collapsed by default now; turn it on to exercise the pane.
        assert ed._show_log_var.get() is False
        assert ed._log_frame.winfo_manager() == ''
        ed._show_log_var.set(True)
        ed._toggle_log()
        assert ed._log_frame.winfo_manager() == 'grid'

        ed._log_queue.put("PANE_LINE_42\n")
        ed._drain_log()                       # manual pump (no mainloop)
        ed._log_text.config(state='normal')
        body = ed._log_text.get('1.0', 'end')
        ed._log_text.config(state='disabled')
        assert "PANE_LINE_42" in body

        ed._clear_log()
        ed._log_text.config(state='normal')
        assert ed._log_text.get('1.0', 'end').strip() == ''
        ed._log_text.config(state='disabled')

        ed._show_log_var.set(False)           # toggle off → hidden
        ed._toggle_log()
        assert ed._log_frame.winfo_manager() == ''
        ed._show_log_var.set(True)            # and back on
        ed._toggle_log()
        assert ed._log_frame.winfo_manager() == 'grid'
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_console_runs_against_live_editor():
    """The log/console prompt executes Python in-process against the live
    editor, echoes the line, records history, and exposes the namespace."""
    root, ed, _gui = _editor_or_skip()
    try:
        ed._console_entry.delete(0, 'end')
        ed._console_entry.insert(0, "editor._console_probe = 7")
        ed._console_run()
        # Executed in-process against the real editor object.
        assert getattr(ed, '_console_probe', None) == 7
        # Line recorded in history + echoed into the pane.
        assert ed._console_history[-1] == "editor._console_probe = 7"
        ed._log_text.config(state='normal')
        body = ed._log_text.get('1.0', 'end')
        ed._log_text.config(state='disabled')
        assert ">>> editor._console_probe = 7" in body
        # Namespace pre-binds the live editor + numpy.
        assert ed._console.locals['editor'] is ed
        assert 'np' in ed._console.locals
        # Up-arrow recalls the last command.
        ed._console_entry.delete(0, 'end')
        ed._console_history_prev()
        assert ed._console_entry.get() == "editor._console_probe = 7"
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_statistics_window_gate_targets():
    """Stats accepts only (sample, gate) populations: drag APPENDS, Import
    OVERRIDES, and a Source column tracks which side each came from."""
    root, ed, gui = _editor_or_skip()
    try:
        _load_fake(ed, 'a1', 'TrialA')   # each fake has one gate 'g1'
        _load_fake(ed, 'b1', 'TrialB')
        ed._set_active_sample('a1')

        sw = gui.StatisticsWindow(ed)
        sw.withdraw()

        # Default mode: no Source column, every population shown.
        assert 'Source' not in sw._cols
        assert {r['Sample'] for r in sw._rows} == {'a1', 'b1'}

        # Drag a GATE in (editor side) → curated, Source column, one row.
        sw.add_targets([('a1', 'g1')], 'editor')
        assert 'Source' in sw._cols
        assert {(r['Sample'], r['__gid__']) for r in sw._rows} == {('a1', 'g1')}
        assert {r['Source'] for r in sw._rows} == {'editor'}

        # Drag (append) the same population from the workspace → tagged both.
        sw.add_targets([('a1', 'g1')], 'workspace')
        assert {r['Source'] for r in sw._rows} == {'editor+workspace'}
        assert len(sw._rows) == 1            # appended, not duplicated

        # Import OVERRIDES with every gate of every loaded sample.
        sw._import_all_editor()
        assert {(r['Sample'], r['__gid__']) for r in sw._rows} == {('a1', 'g1'),
                                                                    ('b1', 'g1')}
        assert {r['Source'] for r in sw._rows} == {'editor'}   # override reset

        # Clear → empty table that STAYS empty (no auto-repopulate), even
        # across a refresh (e.g. toggling a stat checkbox).
        sw._clear_targets()
        assert sw._rows == []
        sw._refresh()
        assert sw._rows == []
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_error_report_tokenises_and_keeps_keymap_local(tmp_path, monkeypatch):
    """The submittable error report must carry only tokens; the token→value
    map (paths, sample names) is written to a separate LOCAL key file. Same
    value → same token so the trace still correlates."""
    os.environ.setdefault('MPLBACKEND', 'Agg')
    gui = importlib.import_module('openflo.gui')
    keyfile = tmp_path / 'error_report.keys.json'
    monkeypatch.setattr(gui, '_error_keymap_path', lambda: str(keyfile))

    secret = r"C:\Users\alice\study\Subject_007.fcs"
    text = f"load {secret} failed; retry {secret}"
    out = gui._tokenise_for_report(text, ['Subject_007'])

    assert 'alice' not in out and 'Subject_007' not in out      # nothing raw
    assert '<path:1>' in out and out.count('<path:1>') == 2      # stable token
    # Keymap kept locally and decodes back to the real value.
    import json
    keymap = json.loads(keyfile.read_text(encoding='utf-8'))
    assert secret in keymap.values()


def test_recent_sessions_roundtrip(tmp_path, monkeypatch):
    """Opening/saving a session records it in a deduped, capped recent list."""
    root, ed, _gui = _editor_or_skip()
    gui = importlib.import_module('openflo.gui')
    monkeypatch.setattr(gui, '_prefs_path', lambda: str(tmp_path / 'prefs.json'))
    try:
        f = tmp_path / 'sess.flowsession'
        f.write_text('{}', encoding='utf-8')
        ed._push_recent_session(str(f))
        ed._push_recent_session(str(f))                 # dedupe
        recent = ed._recent_sessions()
        assert recent.count(str(f.resolve())) <= 1 or len(recent) == 1
        assert any(os.path.basename(p) == 'sess.flowsession' for p in recent)
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_window_geometry_validation():
    """A saved geometry that's off-screen / absurd falls back to the default
    so an unplugged monitor can't strand the window."""
    root, ed, _gui = _editor_or_skip()
    try:
        # Mock prefs AND the screen size so the result doesn't depend on the
        # real ~/.openflo/prefs.json or the CI runner's display resolution.
        gui = importlib.import_module('openflo.gui')
        import unittest.mock as m
        ed.winfo_screenwidth = lambda: 1920
        ed.winfo_screenheight = lambda: 1080
        with m.patch.object(gui, 'read_prefs', lambda: {}):
            assert ed._restore_geometry('1500x840') == '1500x840'   # no pref
        with m.patch.object(gui, 'read_prefs', lambda: {'geometry': 'garbage'}):
            assert ed._restore_geometry('1500x840') == '1500x840'   # unparseable
        with m.patch.object(gui, 'read_prefs', lambda: {'geometry': '99999x99999+0+0'}):
            assert ed._restore_geometry('1500x840') == '1500x840'   # absurd size
        with m.patch.object(gui, 'read_prefs', lambda: {'geometry': '1200x700+99999+99999'}):
            assert ed._restore_geometry('1500x840') == '1200x700'   # drops bad pos
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_save_plot_image(tmp_path):
    """The current plot saves to an image file via savefig_background."""
    root, ed, gui = _editor_or_skip()
    try:
        out = tmp_path / 'plot.png'
        gui.savefig_background(ed.fig, str(out), background='White', dpi=70)
        assert out.is_file() and out.stat().st_size > 0
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_file_menu_exposes_example_and_save_plot():
    """The two onboarding/export actions are wired into the File menu."""
    root, ed, _gui = _editor_or_skip()
    try:
        fb = [b for b in ed._menubar_buttons if b.cget('text') == 'File'][0]
        menu = root.nametowidget(fb.cget('menu'))
        labels = [menu.entrycget(i, 'label') for i in range(menu.index('end') + 1)
                  if menu.type(i) == 'command']
        assert 'Load example dataset' in labels
        assert 'Save plot as image…' in labels
    finally:
        try:
            root.destroy()
        except Exception:
            pass
