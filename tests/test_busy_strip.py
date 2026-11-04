"""The busy strip under the main window: a background job reports progress,
stops on Cancel and drops its result, sends its error to on_error, gives
way to a newer run, and runs steps one per turn of the event loop.
Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by a
mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import threading

import pytest

import tests._background_work as shared
from openflo.async_task import BusyStrip, wait_for_job
from tests._background_work import _on_main
from tests._tk_wait import pump_until, settle

tk_root = shared.tk_root


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
    assert pump_until(
        tk_root, lambda: 'main' in seen and strip.text == "step three",
        5, pause=0.005), "no progress shown"
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
    assert pump_until(tk_root, lambda: len(done_steps) >= 2, pause=0)
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
