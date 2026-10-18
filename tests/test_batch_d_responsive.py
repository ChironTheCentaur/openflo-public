"""Batch D: the window keeps answering while OpenFlo works.

- A replot (every zoom or axis change) rebuilt the whole Samples & Gates
  tree; the tree is now rebuilt only when what it shows changed.
- Dragging a gate redrew every plotted event on each mouse move and, with
  gates applied, recounted every event on the window thread; it now redraws
  the outline alone over a saved picture of the plot, and the counts are
  made after the release by the plot cache's background worker.
- Trajectory, Marker expression, FMO gating, cell cycle, population export,
  Sample QC and compensation optimisation computed on the window thread; they
  now run in the background with progress and a Cancel button, and layout
  batch export shows its progress and stops on Cancel.

Every test builds its own data (generated FCS files or frames with synthetic
sample names and generic channels) and checks both sides: what should
happen, and what must not.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np
import pandas as pd
import pytest

from openflo.async_task import BusyStrip, wait_for_job
from tests.test_beta_look import _load, _write_fcs
from tests.test_transform_records import (
    _new_editor,
    capture_warnings,
    isolate_home,
    settle,
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    notes = capture_warnings(monkeypatch)       # never a real (blocking) modal
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"Tk cannot initialise: {e}")
    root.withdraw()
    ed = _new_editor(root)
    yield root, ed, tmp_path, notes
    root.destroy()


def _on_main():
    return threading.current_thread() is threading.main_thread()


def _gated(root, ed, tmp):
    """sample_a loaded, a rect gate on FSC-A x SSC-A plotted there."""
    _load(root, ed, [('sample_a', _write_fcs(tmp / 'a.fcs'))])
    ed._set_active_sample('sample_a')
    ed.x_combo.set('FSC-A')
    ed.y_combo.set('SSC-A')
    gid = ed._add_gate({
        'kind': 'rect', 'name': 'Cells', 'x_channel': 'FSC-A',
        'y_channel': 'SSC-A', 'x0': 60_000.0, 'x1': 180_000.0,
        'y0': 20_000.0, 'y1': 100_000.0, 'parent_id': None})
    ed._replot()
    settle(root, 0.6)
    return gid


def _quiet(root, ed, timeout=20.0):
    """Wait until the background warm-up has nothing left to compute, and
    let the tree take in the counts it brought (a count arriving is a real
    change, which rebuilds the tree)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end and (
            getattr(ed, '_pc_inflight', None) is not None
            or getattr(ed, '_pc_warm_names', None)
            or getattr(ed, '_pc_sched_after', None)):
        root.update()
        time.sleep(0.01)
    settle(root, 0.2)
    ed._refresh_gate_list()


def _count(ed, name, gid):
    from openflo.pipeline import cumulative_gate_mask
    return int(np.asarray(cumulative_gate_mask(
        ed._sample_gates[name], gid, ed._samples[name].data),
        dtype=bool).sum())


def _spy(monkeypatch, obj, attr):
    """Record each call of ``obj.attr`` (still calling through)."""
    calls = []
    real = getattr(obj, attr)

    def spy(*a, **k):
        calls.append(a)
        return real(*a, **k)
    monkeypatch.setattr(obj, attr, spy)
    return calls


# ── the tree is rebuilt only when what it shows changed ─────────────────────

def test_a_zoom_replot_leaves_the_tree_alone(env, monkeypatch):
    root, ed, tmp, _ = env
    _gated(root, ed, tmp)
    _quiet(root, ed)
    inserts = _spy(monkeypatch, ed.gate_tv, 'insert')
    ed.ax.set_xlim(70_000, 160_000)
    ed._remember_zoom()
    ed._replot()
    ed.mode_var.set('contour')
    ed._replot()
    settle(root, 0.3)
    assert inserts == [], f"tree rebuilt by a replot: {len(inserts)} rows"


def test_a_new_gate_still_rebuilds_the_tree(env, monkeypatch):
    root, ed, tmp, _ = env
    _gated(root, ed, tmp)
    inserts = _spy(monkeypatch, ed.gate_tv, 'insert')
    gid = ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A',
                        'value': 5_000.0, 'name': 'FL1 pos',
                        'parent_id': None})
    ed._replot()
    assert inserts, "a new gate did not reach the tree"
    assert ed.gate_tv.exists(ed._gate_iid('sample_a', gid))
    assert 'FL1 pos' in ed.gate_tv.item(ed._gate_iid('sample_a', gid), 'text')


def test_a_row_put_in_by_hand_is_cleared_by_the_next_refresh(env):
    """The loader adds and removes ⏳ rows itself; the next refresh must not
    take the tree for unchanged and leave them standing."""
    root, ed, tmp, _ = env
    _gated(root, ed, tmp)
    ed._sample_lb_insert_loading('sample_late')      # a load that then failed
    assert ed.gate_tv.exists(ed._sample_iid('sample_late'))
    ed._refresh_gate_list()
    assert not ed.gate_tv.exists(ed._sample_iid('sample_late'))
    assert ed.gate_tv.exists(ed._sample_iid('sample_a'))


# ── a gate drag draws the outline only, and counts after the release ────────

def _ev(ed, x, y, button=1):
    from types import SimpleNamespace
    px, py = ed.ax.transData.transform((x, y))
    return SimpleNamespace(inaxes=ed.ax, xdata=x, ydata=y, x=px, y=py,
                           button=button, key=None, dblclick=False)


def _scan_spy(monkeypatch):
    """(on the window thread?, the gates counted) for each plot-cache pass."""
    import openflo.editor_plotcache as epc
    calls = []
    real = epc._pcm.scan

    def scan(*a, **k):
        calls.append((_on_main(), tuple(k.get('count_gids') or ())))
        return real(*a, **k)
    monkeypatch.setattr(epc._pcm, 'scan', scan)
    return calls


@pytest.mark.parametrize('gated_view', [False, True])
def test_dragging_a_corner_redraws_the_outline_alone(env, monkeypatch,
                                                     gated_view):
    root, ed, tmp, _ = env
    gid = _gated(root, ed, tmp)
    ed.apply_gates_var.set(gated_view)
    ed._replot()
    settle(root, 0.4)
    before = _count(ed, 'sample_a', gid)
    draws = _spy(monkeypatch, ed.canvas, 'draw_idle')
    blits = _spy(monkeypatch, ed.canvas, 'blit')
    replots = _spy(monkeypatch, ed, '_schedule_replot')
    scans = _scan_spy(monkeypatch)

    ed._on_press(_ev(ed, 180_000.0, 100_000.0))          # the top-right corner
    assert ed._drag_state is not None and ed._drag_state[0] == 'rect_corner'
    for step in range(1, 21):
        ed._on_motion(_ev(ed, 180_000.0 - step * 3_500.0,
                          100_000.0 - step * 2_250.0))
    assert draws == [], f"{len(draws)} full redraws during the drag"
    assert len(blits) >= 20, "the moving outline was not drawn"
    assert replots == [] and scans == [], "the drag recounted while moving"

    ed._on_release(_ev(ed, 110_000.0, 55_000.0))
    end = time.monotonic() + 20
    while time.monotonic() < end and getattr(ed, '_replot_when_warm', None):
        root.update()
        time.sleep(0.01)
    settle(root, 0.4)
    after = _count(ed, 'sample_a', gid)
    assert after < before                             # the gate really shrank
    counted = [main for main, gids in scans if gid in gids]
    assert counted and not any(counted), (
        "the moved gate was counted on the window thread", scans)
    row = ed.gate_tv.item(ed._gate_iid('sample_a', gid), 'text')
    assert f"{after:,}" in row and f"{before:,}" not in row, row
    assert ed._gate_label_artists, "labels did not come back after the drop"
    assert not any(a.get_animated() for a in ed.ax.get_children())


def test_without_the_background_worker_the_release_replots_itself(env,
                                                                  monkeypatch):
    root, ed, tmp, _ = env
    gid = _gated(root, ed, tmp)
    ed._pc_warm_enabled = False
    ed._on_press(_ev(ed, 180_000.0, 100_000.0))
    ed._on_motion(_ev(ed, 200_000.0, 120_000.0))
    replots = _spy(monkeypatch, ed, '_schedule_replot')
    ed._on_release(_ev(ed, 200_000.0, 120_000.0))
    assert replots == [(0,)]
    settle(root, 0.4)
    after = _count(ed, 'sample_a', gid)
    assert f"{after:,}" in ed.gate_tv.item(ed._gate_iid('sample_a', gid),
                                           'text')


# ── the strip: progress, Cancel, errors ─────────────────────────────────────

@pytest.fixture
def tk_root():
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"Tk cannot initialise: {e}")
    root.withdraw()
    yield root
    root.destroy()


def test_a_job_runs_off_the_window_thread_and_reports_progress(tk_root):
    strip = BusyStrip(tk_root)
    strip.pack()
    go = threading.Event()
    seen = {}

    def work(job):
        seen['main'] = _on_main()
        job.progress(3, 10, "step three")
        go.wait(10)
        return 42

    got = []
    strip.run(work, on_done=got.append)
    end = time.monotonic() + 5
    while 'main' not in seen or strip.text != "step three":
        assert time.monotonic() < end, "no progress shown"
        tk_root.update()
        time.sleep(0.005)
    assert seen['main'] is False
    assert str(strip._bar.cget('mode')) == 'determinate'
    assert float(strip._bar.cget('value')) == 3
    assert strip._cancel_btn.winfo_manager() == 'grid'     # Cancel is offered
    go.set()
    assert wait_for_job(strip)
    assert got == [42]
    assert strip._cancel_btn.winfo_manager() == ''         # and gone after


def test_cancel_stops_the_job_and_drops_its_result(tk_root):
    strip = BusyStrip(tk_root)
    btn = __import__('tkinter').ttk.Button(tk_root, text="Run")
    go = threading.Event()
    reached = []

    def work(job):
        go.wait(10)
        job.check()
        reached.append('after the cancel point')
        return 'result'

    got, cancelled = [], []
    strip.run(work, on_done=got.append, on_cancel=lambda: cancelled.append(1),
              disable=[btn])
    assert str(btn.cget('state')) == 'disabled'
    strip.cancel()
    go.set()
    assert wait_for_job(strip)
    assert cancelled == [1] and got == [] and reached == []
    assert str(btn.cget('state')) == 'normal'


def test_an_error_reaches_on_error(tk_root):
    strip = BusyStrip(tk_root)
    errors, got = [], []

    def work(job):
        raise ValueError("bad input")

    strip.run(work, on_done=got.append, on_error=errors.append)
    assert wait_for_job(strip)
    assert got == [] and [str(e) for e in errors] == ["bad input"]


def test_a_newer_run_replaces_the_running_one(tk_root):
    strip = BusyStrip(tk_root)
    hold = threading.Event()
    got = []
    older = strip.run(lambda job: (hold.wait(10), job.check(), 'old')[2],
                      on_done=got.append)
    strip.run(lambda job: 'new', on_done=got.append)
    assert older.cancelled            # the older job is told to stop
    hold.set()
    assert wait_for_job(strip)
    settle(tk_root, 0.2)
    assert got == ['new']


def test_steps_run_one_per_turn_and_cancel_closes_them(tk_root):
    strip = BusyStrip(tk_root)
    done_steps, closed = [], []

    def steps():
        try:
            for i in range(5):
                yield (i, 5, f"step {i}")
                done_steps.append(i)
        finally:
            closed.append(True)

    cancelled = []
    strip.run_steps(steps(), on_done=lambda r: pytest.fail("finished"),
                    on_cancel=lambda: cancelled.append(1))
    while len(done_steps) < 2:
        tk_root.update()
    strip.cancel()
    assert wait_for_job(strip)
    assert cancelled == [1] and closed == [True]
    assert len(done_steps) < 5


def test_steps_return_their_value(tk_root):
    strip = BusyStrip(tk_root)

    def steps():
        yield (0, 1, None)
        return 'all written'

    got = []
    strip.run_steps(steps(), on_done=got.append)
    assert wait_for_job(strip)
    assert got == ['all written']


# ── the windows that froze now work in the background ──────────────────────

def _cell_cycle_sample(ed, name='sample_cc'):
    from openflo.pipeline import FlowSample
    rng = np.random.default_rng(3)
    dna = np.r_[rng.normal(50_000, 2_500, 5000),
                rng.uniform(55_000, 95_000, 1500),
                rng.normal(100_000, 4_000, 1500)]
    s = FlowSample.from_dataframe(
        pd.DataFrame({'FSC-A': np.full(dna.size, 8e4), 'FL3-A': dna}),
        name=name)
    ed._on_loaded(name, s)
    ed._set_active_sample(name)
    # The load selects the sample's row, and that selection's queued
    # readout ("name:  N events") rewrites the status line. Wait for it,
    # however slow the machine, before a test reads what the job said there.
    end = time.monotonic() + 10.0
    while (time.monotonic() < end
           and not ed.status_var.get().startswith(f"{name}:")):
        ed.update()
        time.sleep(0.01)
    settle(ed, 0.3)
    return s


def test_cell_cycle_models_off_the_window_thread(env, monkeypatch):
    import openflo.pipeline as fp
    root, ed, _tmp, _ = env
    s = _cell_cycle_sample(ed)
    threads = []
    real = fp.analyze_dna
    monkeypatch.setattr(fp, 'analyze_dna', lambda *a, **k: (
        threads.append(_on_main()), real(*a, **k))[1])
    ed._run_cell_cycle('FL3-A', all_samples=False, k=2.0)
    assert 'cell_cycle' not in s.data.columns       # not written on the spot
    assert wait_for_job(ed._job_strip)
    assert threads == [False]
    assert 'cell_cycle' in s.data.columns
    assert s.cell_cycle_result and s.cell_cycle_result['ok']
    assert 'done for 1 sample' in ed.status_var.get()


def test_cell_cycle_model_is_what_cell_cycle_writes():
    from openflo.pipeline import FlowSample
    rng = np.random.default_rng(4)
    dna = np.r_[rng.normal(40_000, 2_000, 4000),
                rng.normal(80_000, 3_000, 1000)]
    frame = pd.DataFrame({'FSC-A': np.full(dna.size, 8e4), 'FL3-A': dna})
    a = FlowSample.from_dataframe(frame.copy(), name='sample_x')
    b = FlowSample.from_dataframe(frame.copy(), name='sample_x')
    phases, model = a.cell_cycle_model(dna_channel='FL3-A', k=2.0)
    assert 'cell_cycle' not in a.data.columns          # nothing written
    b.cell_cycle(dna_channel='FL3-A', k=2.0)
    assert list(b.data['cell_cycle']) == list(phases)
    assert b.cell_cycle_result == model
    none = FlowSample.from_dataframe(frame[['FSC-A']].copy(), name='sample_y')
    assert none.cell_cycle_model(dna_channel='FL3-A') is None
    assert none.cell_cycle(dna_channel='FL3-A').cell_cycle_result is None


def test_cell_cycle_cancel_changes_no_sample(env, monkeypatch):
    import openflo.pipeline as fp
    root, ed, _tmp, _ = env
    first = _cell_cycle_sample(ed, 'sample_cc1')
    second = _cell_cycle_sample(ed, 'sample_cc2')
    hold = threading.Event()
    real = fp.analyze_dna
    monkeypatch.setattr(fp, 'analyze_dna', lambda *a, **k: (
        hold.wait(10), real(*a, **k))[1])
    ed._run_cell_cycle('FL3-A', all_samples=True, k=2.0)
    end = time.monotonic() + 5
    while not ed._job_strip.text.startswith('Cell cycle: sample_cc1'):
        assert time.monotonic() < end, ed._job_strip.text
        root.update()                            # the window still answers
    ed._job_strip.cancel()
    hold.set()
    assert wait_for_job(ed._job_strip)
    for s in (first, second):
        assert 'cell_cycle' not in s.data.columns
        assert getattr(s, 'cell_cycle_result', None) is None
    assert ed.status_var.get() == ("Cell cycle cancelled; no sample was "
                                   "changed.")


def test_one_job_at_a_time_in_the_main_window(env, monkeypatch):
    import openflo.pipeline as fp
    root, ed, _tmp, _ = env
    _cell_cycle_sample(ed)
    hold = threading.Event()
    real = fp.analyze_dna
    monkeypatch.setattr(fp, 'analyze_dna', lambda *a, **k: (
        hold.wait(10), real(*a, **k))[1])
    assert ed.run_job(lambda job: hold.wait(10)) is not None
    assert ed.run_job(lambda job: 1) is None
    assert ed.status_var.get().startswith("Still working")
    hold.set()
    assert wait_for_job(ed._job_strip)


def test_population_export_writes_off_the_window_thread(env, monkeypatch):
    import openflo.fcs_export as fe
    from openflo import editor_populations
    root, ed, tmp, _ = env
    gid = _gated(root, ed, tmp)
    threads = []
    real = fe.population_export_frame
    monkeypatch.setattr(fe, 'population_export_frame', lambda *a, **k: (
        threads.append(_on_main()), real(*a, **k))[1])
    path = tmp / 'one.fcs'
    monkeypatch.setattr(editor_populations.filedialog, 'asksaveasfilename',
                        lambda **k: str(path))
    ed._export_population_fcs('sample_a', gid)
    assert wait_for_job(ed._job_strip)
    assert threads == [False] and path.exists()
    assert f"Exported {_count(ed, 'sample_a', gid):,} events" in (
        ed.status_var.get())
    out = tmp / 'bulk'
    out.mkdir()
    monkeypatch.setattr(editor_populations.filedialog, 'askdirectory',
                        lambda **k: str(out))
    ed._export_populations_fcs()
    assert wait_for_job(ed._job_strip)
    assert threads == [False, False]
    assert [p.name for p in out.glob('*.fcs')] == ['Cells.fcs']


def test_population_export_cancel_writes_nothing(env, monkeypatch):
    import openflo.fcs_export as fe
    from openflo import editor_populations
    root, ed, tmp, _ = env
    _gated(root, ed, tmp)
    ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A',
                  'value': 5_000.0, 'name': 'FL1 pos', 'parent_id': None})
    hold = threading.Event()
    real = fe.population_export_frame
    monkeypatch.setattr(fe, 'population_export_frame', lambda *a, **k: (
        hold.wait(10), real(*a, **k))[1])
    out = tmp / 'bulk'
    out.mkdir()
    monkeypatch.setattr(editor_populations.filedialog, 'askdirectory',
                        lambda **k: str(out))
    ed._export_populations_fcs()
    root.update()
    ed._job_strip.cancel()
    hold.set()
    assert wait_for_job(ed._job_strip)
    assert list(out.iterdir()) == []
    assert 'cancelled' in ed.status_var.get()


def _trajectory_window(ed):
    from openflo.pipeline import FlowSample
    from openflo.ui_trajectory import TrajectoryWindow
    rng = np.random.default_rng(0)
    t = rng.uniform(0, 1, 800)
    df = pd.DataFrame({'FL1-A': 10 * (1 - t) + rng.normal(0, .2, t.size),
                       'FL2-A': 10 * t + rng.normal(0, .2, t.size)})
    for name, sel in (('sample_d3', t >= .5), ('sample_d1', t < .5)):
        s = FlowSample.from_dataframe(df[sel].reset_index(drop=True),
                                      name=name)
        ed._samples[name] = s
        ed._sample_order.append(name)
        ed._sample_trial[name] = name
        ed._sample_plot_enabled[name] = True
        ed._sample_gates[name] = {}
        ed._sample_gate_order[name] = []
    ed._channels = ['FL1-A', 'FL2-A']
    ed._channel_labels = {'FL1-A': 'FL1-A', 'FL2-A': 'FL2-A'}
    ed._active_sample = 'sample_d1'
    w = TrajectoryWindow(ed)
    w.withdraw()
    w.root_var.set(ed._fmt_channel('FL1-A'))
    return w


def test_trajectory_computes_off_the_window_thread_and_cancels(env,
                                                               monkeypatch):
    import openflo.trajectory as tr
    root, ed, _tmp, _ = env
    w = _trajectory_window(ed)
    threads, hold = [], threading.Event()
    real = tr.compute_pseudotime

    def slow(*a, **k):
        threads.append(_on_main())
        hold.wait(10)
        return real(*a, **k)
    monkeypatch.setattr(tr, 'compute_pseudotime', slow)
    w._compute()
    root.update()                                    # the window still answers
    assert w._busy.running
    w._busy.cancel()
    hold.set()
    assert wait_for_job(w._busy)
    assert 'pseudotime' not in ed._samples['sample_d1'].data.columns
    assert 'Cancelled' in str(w.status.cget('text'))
    w._compute()
    assert wait_for_job(w._busy)
    assert threads == [False, False]
    assert 'pseudotime' in ed._samples['sample_d1'].data.columns
    assert w._means is not None


def test_marker_expression_reads_off_the_window_thread(env, monkeypatch):
    import openflo.stats as st
    from openflo.pipeline import FlowSample
    from openflo.ui_expression import MarkerExpressionWindow
    root, ed, _tmp, _ = env
    for name, med in (('Stim_1', 300), ('Stim_2', 320), ('Ctrl_1', 100),
                      ('Ctrl_2', 110)):
        v = np.r_[np.full(51, float(med)), np.linspace(med + 1, med + 900, 50)]
        s = FlowSample.from_dataframe(pd.DataFrame({'FL1-A': v}), name=name)
        ed._samples[name] = s
        ed._sample_order.append(name)
        ed._sample_trial[name] = 'T'
        ed._sample_plot_enabled[name] = True
        ed._sample_gates[name] = {}
        ed._sample_gate_order[name] = []
    ed._channels = ['FL1-A']
    ed._channel_labels = {'FL1-A': 'FL1-A'}
    threads = []
    real = st.compare_groups
    monkeypatch.setattr(st, 'compare_groups', lambda *a, **k: (
        threads.append(_on_main()), real(*a, **k))[1])
    w = MarkerExpressionWindow(ed)
    w.withdraw()
    w.factor_var.set('Name token')
    w.tokens_var.set('Stim, Ctrl')
    w._rebuild()
    assert wait_for_job(w._busy)
    assert threads and not any(threads)
    assert w._medians == {'Stim': [300.0, 320.0], 'Ctrl': [100.0, 110.0]}


def test_sample_qc_compares_off_the_window_thread(env, monkeypatch):
    import openflo.interop as io_
    from openflo.pipeline import FlowSample
    from openflo.ui_sample_qc import SampleQCWindow
    root, ed, _tmp, _ = env
    rng = np.random.default_rng(1)
    for name, shift in (('sample_1', 0.0), ('sample_2', 0.1),
                        ('sample_3', 5.0)):
        s = FlowSample.from_dataframe(pd.DataFrame(
            rng.normal(shift, 1.0, (500, 2)), columns=['FL1-A', 'FL2-A']),
            name=name)
        s.fluor_channels = ['FL1-A', 'FL2-A']
        ed._samples[name] = s
        ed._sample_order.append(name)
        ed._sample_plot_enabled[name] = True
    ed._selected_samples = lambda: ['sample_1', 'sample_2', 'sample_3']
    threads = []
    real = io_.sample_distance_matrix
    monkeypatch.setattr(io_, 'sample_distance_matrix', lambda *a, **k: (
        threads.append(_on_main()), real(*a, **k))[1])
    w = SampleQCWindow(ed)
    w.withdraw()
    w._compute()
    assert wait_for_job(w._busy)
    assert threads and not any(threads)
    D = np.asarray(w._D)
    assert D[0, 1] < D[0, 2] and w._xy is not None


def test_compensation_optimisation_reads_off_the_window_thread(tk_root,
                                                               monkeypatch):
    import openflo.pipeline as fp
    from openflo.ui_comp import OptimizeCompensationDialog
    threads, got = [], []
    monkeypatch.setattr(fp, 'optimize_compensation', lambda ch, paths: (
        threads.append(_on_main()), (list(ch), np.eye(len(ch))))[1])
    dlg = OptimizeCompensationDialog(tk_root, ['FL1-A', 'FL2-A'],
                                     on_complete=lambda c, m: got.append(c))
    dlg.path_vars['FL1-A'].set('control_fl1.fcs')
    dlg._run()
    assert got == []                       # nothing applied on the spot
    assert wait_for_job(dlg._busy)
    assert threads == [False] and got == [['FL1-A', 'FL2-A']]


def test_fmo_cutoffs_are_read_off_the_window_thread(env, monkeypatch):
    import openflo.gating_helpers as gh
    from openflo.pipeline import FlowSample
    from openflo.ui_fmo import FMOGatingDialog
    root, ed, _tmp, _ = env
    rng = np.random.default_rng(2)
    for name, pos in (('sample_stained', 0.5), ('sample_fmo', 0.0)):
        n = 4000
        fl1 = np.where(rng.random(n) < pos, rng.normal(0.7, 0.04, n),
                       rng.normal(0.2, 0.05, n))
        s = FlowSample.from_dataframe(pd.DataFrame(
            {'FSC-A': rng.normal(0.6, 0.05, n), 'FL1-A': fl1}), name=name)
        s.fluor_channels = ['FL1-A']
        ed._on_loaded(name, s)
    ed._set_active_sample('sample_stained')
    threads = []
    real = gh.fmo_threshold
    monkeypatch.setattr(gh, 'fmo_threshold', lambda *a, **k: (
        threads.append(_on_main()), real(*a, **k))[1])
    dlg = FMOGatingDialog(ed)
    dlg._map['FL1-A'].set('sample_fmo')
    dlg._apply()
    assert not [g for g in ed._gates.values() if g.get('kind') == 'threshold']
    assert wait_for_job(dlg._busy)
    assert threads == [False]
    gates = [g for g in ed._gates.values() if g.get('kind') == 'threshold']
    assert [g['channel'] for g in gates] == ['FL1-A']


def _let_the_load_settle(root, ed):
    """Run the load's debounced settle pass now and drain what it queues.

    Loading arms a 120 ms timer whose pass writes "N sample(s) loaded" and
    re-selects a sample row, whose queued readout writes "name:  N events".
    Left pending, a slow machine (the Windows CI legs) ran both after a
    dialog's job had answered, so the status line no longer held its answer.
    """
    pending = getattr(ed, '_load_settle_after', None)
    if pending:
        ed.after_cancel(pending)
        ed._load_settle_after = None
        ed._on_load_settled()
    root.update()


def test_fmo_gate_is_not_put_on_a_sample_chosen_meanwhile(env, monkeypatch):
    import openflo.gating_helpers as gh
    from openflo.pipeline import FlowSample
    from openflo.ui_fmo import FMOGatingDialog
    root, ed, _tmp, _ = env
    rng = np.random.default_rng(5)
    for name in ('sample_stained', 'sample_fmo', 'sample_other'):
        s = FlowSample.from_dataframe(pd.DataFrame(
            {'FSC-A': rng.normal(0.6, 0.05, 500),
             'FL1-A': rng.normal(0.2, 0.05, 500)}), name=name)
        s.fluor_channels = ['FL1-A']
        ed._on_loaded(name, s)
    _let_the_load_settle(root, ed)
    ed._set_active_sample('sample_stained')
    hold = threading.Event()
    real = gh.fmo_threshold
    monkeypatch.setattr(gh, 'fmo_threshold', lambda *a, **k: (
        hold.wait(10), real(*a, **k))[1])
    dlg = FMOGatingDialog(ed)
    dlg._map['FL1-A'].set('sample_fmo')
    dlg._apply()
    ed._set_active_sample('sample_other')
    hold.set()
    assert wait_for_job(dlg._busy)
    for name in ('sample_stained', 'sample_other'):
        assert not [g for g in ed._sample_gates[name].values()
                    if g.get('kind') == 'threshold'], name
    assert 'active sample changed' in ed.status_var.get()


def test_layout_batch_export_shows_progress_and_stops_on_cancel(monkeypatch,
                                                                tmp_path):
    import openflo.layout_render as lr
    import openflo.ui_layout_editor as ule
    from tests.test_layout_editor import _editor_or_skip
    from tests.test_layout_editor import _load as _load_layout
    root, ed = _editor_or_skip()
    try:
        for i, name in enumerate(('sample_1', 'sample_2', 'sample_3')):
            _load_layout(ed, name, i, 300, 300.0)
        win = ule.LayoutEditorWindow(ed)
        win.update()
        win._add_text()
        ed._selected_samples = lambda: ['sample_1', 'sample_2', 'sample_3']
        out = tmp_path / 'batch.pdf'
        monkeypatch.setattr(ule.filedialog, 'asksaveasfilename',
                            lambda *a, **k: str(out))
        rendered, texts = [], []
        real = lr.render_page

        def render(*a, **k):
            rendered.append(a[2])
            texts.append(win._busy.text)
            if len(rendered) == 1:
                win._busy.cancel()       # as if Cancel were pressed meanwhile
            return real(*a, **k)
        monkeypatch.setattr(lr, 'render_page', render)
        win._export_batch()
        assert wait_for_job(win._busy)
        assert rendered == ['sample_1']
        assert texts == ["Exporting sample_1 (1 of 3)…"]
        assert win.status.get().startswith(
            "Batch export cancelled after 1 of 3 sample(s)")
        data = out.read_bytes()
        assert data.startswith(b'%PDF') and b'/Count 1' in data
        win.destroy()
    finally:
        root.destroy()
