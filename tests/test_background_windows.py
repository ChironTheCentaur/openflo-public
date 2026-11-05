"""The windows that froze the main window now work in the background with
progress and a Cancel button: cell cycle, trajectory, marker expression,
Sample QC, compensation optimisation and FMO gating; one job runs at a
time, and layout batch export shows its progress and stops on Cancel.
Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by a
mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import threading

import numpy as np
import pandas as pd

import tests._background_work as shared
from openflo.async_task import wait_for_job
from tests._background_work import _let_the_load_settle, _on_main
from tests._tk_wait import pump_until, settle

env = shared.env
tk_root = shared.tk_root


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
    pump_until(ed, lambda: ed.status_var.get().startswith(f"{name}:"), 10.0)
    settle(ed, 0.3)
    return s


def test_cell_cycle_models_off_the_window_thread(env, monkeypatch):
    import openflo.flow_sample as fp
    _root, ed, _tmp, _ = env
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
    import openflo.flow_sample as fp
    root, ed, _tmp, _ = env
    first = _cell_cycle_sample(ed, 'sample_cc1')
    second = _cell_cycle_sample(ed, 'sample_cc2')
    hold = threading.Event()
    real = fp.analyze_dna
    monkeypatch.setattr(fp, 'analyze_dna', lambda *a, **k: (
        hold.wait(10), real(*a, **k))[1])
    ed._run_cell_cycle('FL3-A', all_samples=True, k=2.0)
    # the window still answers
    assert pump_until(
        root, lambda: ed._job_strip.text.startswith('Cell cycle: sample_cc1'),
        5, pause=0), ed._job_strip.text
    ed._job_strip.cancel()
    hold.set()
    assert wait_for_job(ed._job_strip)
    for s in (first, second):
        assert 'cell_cycle' not in s.data.columns
        assert getattr(s, 'cell_cycle_result', None) is None
    assert ed.status_var.get() == ("Cell cycle cancelled; no sample was "
                                   "changed.")


def test_one_job_at_a_time_in_the_main_window(env, monkeypatch):
    import openflo.flow_sample as fp
    _root, ed, _tmp, _ = env
    _cell_cycle_sample(ed)
    hold = threading.Event()
    real = fp.analyze_dna
    monkeypatch.setattr(fp, 'analyze_dna', lambda *a, **k: (
        hold.wait(10), real(*a, **k))[1])
    job = ed.run_job(lambda job: hold.wait(10))
    assert job is ed._job_strip._job                 # the job now running
    assert ed.run_job(lambda job: 1) is None
    assert ed.status_var.get().startswith("Still working")
    hold.set()
    assert wait_for_job(ed._job_strip)
    # The handle the caller got is the job that ran: its result is the
    # work's own return value (hold.wait returns True once set).
    assert job.finished.is_set() and job.result is True


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
    # 20 bins x the two shared channels, in channel order. Rooted at high
    # FL1-A, FL1-A (10 * (1 - t)) falls along pseudotime and FL2-A (10 * t)
    # rises, each across most of its 0..10 span.
    assert w._means.shape == (20, 2)
    assert w._channels == ['FL1-A', 'FL2-A']
    fl1, fl2 = (w._means[:, j][np.isfinite(w._means[:, j])] for j in (0, 1))
    assert fl1[0] - fl1[-1] > 6 and fl2[-1] - fl2[0] > 6


def test_marker_expression_reads_off_the_window_thread(env, monkeypatch):
    import openflo.stats as st
    from openflo.pipeline import FlowSample
    from openflo.ui_expression import MarkerExpressionWindow
    _root, ed, _tmp, _ = env
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
    _root, ed, _tmp, _ = env
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
    import openflo.compensation_optimizer as co
    from openflo.ui_comp import OptimizeCompensationDialog
    threads, got = [], []
    monkeypatch.setattr(co, 'optimize_compensation', lambda ch, paths: (
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
    _root, ed, _tmp, _ = env
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
