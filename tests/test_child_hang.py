"""tests/_child_hang.py: a test's child process that hangs says where.

The macOS branch (``sample``, ``lsappinfo``) is driven here with stand-in
programs, since there is no Mac on most machines that run the suite."""
from __future__ import annotations

import os
import sys
import textwrap

import pytest

from tests import _child_hang

_QUICK = [sys.executable, '-c', 'print("done")']
_SLOW = [sys.executable, '-c', 'import time; time.sleep(4); print("late")']
_STUCK = [sys.executable, '-X', 'faulthandler', '-c',
          'import time\ndef stuck_here():\n    time.sleep(600)\nstuck_here()']


def _fake(tmp_path, name, body):
    """A stand-in program: a Python script run by this interpreter."""
    script = tmp_path / f'{name}.py'
    script.write_text(textwrap.dedent(body), encoding='utf-8')
    return [sys.executable, str(script)]


# A sampler that writes a call graph to the file after '-file'.
_SAMPLES = '''
    import sys
    args = sys.argv[1:]
    with open(args[args.index('-file') + 1], 'w') as f:
        f.write('Analysis of sampling pid %s\\nCall graph:\\n  1 Tk_Init\\n    1 TkpInit\\n' % args[0])
'''
_REFUSED = '''
    import sys
    print('sample cannot examine process: not allowed', file=sys.stderr)
    sys.exit(1)
'''


def test_a_child_that_ends_in_time_has_no_report():
    out = _child_hang.run(_QUICK, timeout=60, report_after=30)
    assert out.returncode == 0 and out.stdout.strip() == 'done'
    assert out.hang_report is None


def test_a_slow_child_is_reported_and_still_answers():
    out = _child_hang.run(_SLOW, timeout=60, report_after=1)
    assert out.returncode == 0 and out.stdout.strip() == 'late'
    assert out.hang_report.startswith('child pid ')
    assert 'still running after 1s' in out.hang_report


@pytest.mark.skipif(os.name != 'posix', reason='SIGABRT prints a Python stack on POSIX')
def test_a_stuck_child_fails_the_test_with_its_stack():
    with pytest.raises(pytest.fail.Exception) as caught:
        _child_hang.run(_STUCK, timeout=4, report_after=1)
    msg = str(caught.value)
    assert 'no answer in 4s' in msg
    assert 'still running after 1s' in msg
    assert 'stuck_here' in msg, msg          # faulthandler's dump on SIGABRT


def test_off_macos_there_is_no_native_sampler(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'linux')
    assert _child_hang.native_stacks(os.getpid()) == 'no native sampler on this platform'
    assert _child_hang.front_app() == ''


def test_a_refused_sample_is_retried_with_the_next_sampler(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.setattr(_child_hang, 'SAMPLERS', (
        _fake(tmp_path, 'refused', _REFUSED), _fake(tmp_path, 'samples', _SAMPLES)))
    got = _child_hang.native_stacks(4321)
    assert got.startswith('Call graph:')
    assert 'TkpInit' in got


def test_a_sampler_that_writes_no_call_graph_is_not_taken(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'platform', 'darwin')
    silent = _fake(tmp_path, 'silent', 'print("sampling finished")')
    monkeypatch.setattr(_child_hang, 'SAMPLERS', (silent, _fake(tmp_path, 'samples', _SAMPLES)))
    assert _child_hang.native_stacks(4321).startswith('Call graph:')


def test_when_no_sampler_works_each_says_why(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'platform', 'darwin')
    refused = _fake(tmp_path, 'refused', _REFUSED)
    monkeypatch.setattr(_child_hang, 'SAMPLERS', (refused, refused))
    got = _child_hang.native_stacks(4321)
    assert got.startswith('no native stack:')
    assert got.count('not allowed') == 2


def test_the_frontmost_application_is_named_with_this_process(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.setattr(_child_hang, 'FRONT_APP',
                        _fake(tmp_path, 'front', 'print("ASN:0x0-0x1234:")'))
    monkeypatch.setattr(_child_hang, 'APP_INFO', _fake(
        tmp_path, 'info', 'import sys; print("name=Python pid=99 for", sys.argv[1])'))
    got = _child_hang.front_app()
    assert 'ASN:0x0-0x1234:' in got
    assert 'name=Python pid=99 for ASN:0x0-0x1234:' in got
    assert f'this test process: pid {os.getpid()}' in got


def test_the_report_holds_the_front_app_and_the_native_stack(monkeypatch):
    monkeypatch.setattr(_child_hang, 'front_app', lambda: 'FRONT')
    monkeypatch.setattr(_child_hang, 'native_stacks', lambda pid: f'STACK {pid}')
    assert _child_hang.hang_report(77, 40) == (
        'child pid 77 still running after 40s\nFRONT\nSTACK 77')
