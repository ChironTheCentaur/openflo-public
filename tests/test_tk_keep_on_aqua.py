"""On macOS the suite keeps alive the interpreters Tk's application object may
point at (tests/tk_main_thread.py ``_keep_on_aqua``).

There is no Mac here, so the macOS branch is forced with
OPENFLO_TEST_AQUA_KEEP=1 in a fresh interpreter (and turned off with =0): roots are made, destroyed
and released the way the suite does it, and the first and the newest
interpreter must still answer while the ones in between are freed."""
import os
import subprocess
import sys
import textwrap

import pytest

_SCRIPT = textwrap.dedent('''
    import gc, json, sys, tkinter as tk
    from tests import tk_main_thread as t
    t.install()
    ids = []
    for _ in range(4):
        r = tk.Tk()
        ids.append(id(r.tk))
        r.destroy()
        del r
        t.release(force=True)
        gc.collect()
    kept = t._kept
    alive = {k: (v is not None and v.call('info', 'patchlevel') != '')
             for k, v in kept.items()}
    print(json.dumps({
        'first': kept['first'] is not None and id(kept['first']) == ids[0],
        'latest': kept['latest'] is not None and id(kept['latest']) == ids[-1],
        'alive': alive, 'held': t.held()}))
''')


def _run(force):
    pytest.importorskip('tkinter')
    env = dict(os.environ)
    env['OPENFLO_TEST_AQUA_KEEP'] = '1' if force else '0'
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, '-c', _SCRIPT], capture_output=True,
                       text=True, timeout=120, env=env, cwd=root)
    if r.returncode and 'TclError' in r.stderr and 'display' in r.stderr:
        pytest.skip('Tk cannot open a display here')
    assert r.returncode == 0, r.stderr[-2000:]
    import json
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_the_first_and_newest_interpreters_stay_alive_on_aqua():
    got = _run(force=True)
    assert got['first'] and got['latest'], got
    assert got['alive'] == {'first': True, 'latest': True}, got
    # The two in between were freed as usual: only the kept two are held.
    assert got['held'] == 2, got


def test_with_keeping_off_every_interpreter_is_freed():
    got = _run(force=False)
    assert not got['first'] and not got['latest'], got
    assert got['held'] == 0, got
