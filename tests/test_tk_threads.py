"""The app's root hands Tk clean-up from background threads to its event loop.

The embedding warm-up (and autosave, loading, the plot cache) run on
background threads, and the garbage collector runs on whichever thread
allocates. A Tk variable freed there used to make that thread wait on the
main thread, inside the collection, for as long as the main thread was busy:
the warm-up was seen stalled for seconds. openflo.tk_threads queues that
clean-up instead, and the root made by gui._make_root runs the queue.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from tests.conftest import gui_unavailable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Run in a fresh interpreter: the suite's own conftest installs the same
# guard, which would hide a root that forgot to.
SCRIPT = textwrap.dedent('''
    import gc, sys, threading, time, tkinter as tk
    sys.path.insert(0, {src!r})
    from openflo import gui, tk_threads
    root = gui._make_root()
    root.withdraw()
    gc.disable()                 # only the background thread's collection runs
    names = []
    for i in range(3):
        var = tk.StringVar(root, str(i))
        var.me = var             # in a cycle: only the collector frees it
        names.append(str(var))
        del var
    worker = threading.Thread(target=gc.collect)
    t0 = time.perf_counter()
    worker.start()
    worker.join()                # the main thread is busy right here
    took = time.perf_counter() - t0
    gc.enable()
    end = time.perf_counter() + 3 * tk_threads.DRAIN_MS / 1000
    while time.perf_counter() < end:
        root.update()
        time.sleep(0.02)
    left = [n for n in names if root.getboolean(root.tk.call('info', 'exists', n))]
    print('took=%.3f left=%d' % (took, len(left)))
''')


def _run():
    try:
        import tkinter as tk
        r = tk.Tk()
        r.destroy()
    except Exception as exc:                       # noqa: BLE001
        gui_unavailable(f"Tk unavailable: {exc}")
    out = subprocess.run(
        [sys.executable, '-c', SCRIPT.format(src=os.path.join(ROOT, 'src'))],
        capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    line = out.stdout.strip().splitlines()[-1]
    took = float(line.split()[0].split('=')[1])
    left = int(line.split()[1].split('=')[1])
    return took, left, out.stderr


def test_a_background_collection_does_not_wait_on_a_busy_main_thread():
    took, _left, err = _run()
    assert took < 0.5, f"the collection waited {took:.2f}s on the main thread"
    assert 'main thread is not in main loop' not in err


def test_the_event_loop_runs_the_queued_clean_up():
    _took, left, _err = _run()
    assert left == 0, f"{left} Tk variables were never unset"


@pytest.mark.parametrize('n', [0, 2])
def test_drain_runs_everything_queued(n):
    from openflo import tk_threads
    ran = []
    for i in range(n):
        tk_threads._deferred.append(lambda i=i: ran.append(i))
    assert tk_threads.drain() == n
    assert ran == list(range(n))
    assert tk_threads.pending() == 0
