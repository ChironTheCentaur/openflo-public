"""Run a child process for a test and, when it hangs, say where it hangs.

A child that starts Tk has hung inside Tk's start on the macOS CI runner
(Python's stack ends in ``_tkinter.create``, called from gui._make_root),
now and then and in bursts: every Tk child one worker started for some
minutes, while other workers' Tk children came up. Python's stack stops at
that C call, so when a child is still running after ``report_after``
seconds this records, while it still hangs:

- on macOS, the child's native stack, sampled with ``sample`` (with
  ``sudo -n`` when the plain call is refused, as it is for a signed
  python.org build), which names the system call Tk's start waits on;
- on macOS, the frontmost application and this process's id: one theory is
  that Tk's start waits on the active application, which may be the test
  worker itself, blocked in this very wait with its own Tk window open.

The record is kept on the result as ``hang_report`` (a child with its own
watchdog may still end, with an error, before ``timeout``). A child still
running at ``timeout`` is sent SIGABRT, so one with faulthandler on prints
its Python stack, then killed, and the test fails with all of it."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile

import pytest

# Each is tried in turn until one gives a call graph. A call is
# [program..., pid, seconds, '-file', path] once the pid is appended.
SAMPLERS = (['/usr/bin/sample'], ['sudo', '-n', '/usr/bin/sample'])
SAMPLE_S = 3
FRONT_APP = ['lsappinfo', 'front']
APP_INFO = ['lsappinfo', 'info', '-only', 'name,pid']
# Lines of the call graph kept: the main thread's comes first.
KEEP_LINES = 300


def _tool(cmd, timeout=60):
    """Run a diagnostic tool: (exit code, output) or (None, why it could not)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"{cmd[0]}: {exc}"
    return r.returncode, (r.stdout + r.stderr).strip()


def native_stacks(pid):
    """The child's native call graph on macOS, or why there is none."""
    if sys.platform != 'darwin':
        return 'no native sampler on this platform'
    notes = []
    for sampler in SAMPLERS:
        fd, path = tempfile.mkstemp(suffix='.sample.txt')
        os.close(fd)
        try:
            code, said = _tool([*sampler, str(pid), str(SAMPLE_S), '-file', path])
            try:
                with open(path, encoding='utf-8', errors='replace') as f:
                    text = f.read()
            except OSError:
                text = ''
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        if code == 0 and 'Call graph' in text:
            lines = text[text.index('Call graph'):].splitlines()
            return '\n'.join(lines[:KEEP_LINES])
        notes.append(f"{' '.join(sampler)}: exit {code}: {said[-300:]}")
    return 'no native stack:\n' + '\n'.join(notes)


def front_app():
    """Which application is frontmost (macOS), and who is asking."""
    if sys.platform != 'darwin':
        return ''
    code, front = _tool(FRONT_APP, timeout=20)
    if code != 0:
        return f"frontmost application unknown: {front}"
    _code, info = _tool([*APP_INFO, front], timeout=20)
    return f"frontmost application: {front} {info}; this test process: pid {os.getpid()}"


def hang_report(pid, after):
    """What a child still running after ``after`` seconds is doing."""
    parts = [f"child pid {pid} still running after {after:.0f}s"]
    front = front_app()
    if front:
        parts.append(front)
    parts.append(native_stacks(pid))
    return '\n'.join(parts)


def run(args, *, timeout, report_after, **popen_kw):
    """``subprocess.run(args, capture_output=True, text=True, timeout=...)``
    that records a hang report at ``report_after`` seconds (kept on the
    result as ``hang_report``, None when the child ended sooner), and fails
    the test with the child's output and that report at ``timeout``."""
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, **popen_kw)
    report = None
    try:
        out, err = proc.communicate(timeout=report_after)
    except subprocess.TimeoutExpired:
        report = hang_report(proc.pid, report_after)
        try:
            out, err = proc.communicate(timeout=max(1.0, timeout - report_after))
        except subprocess.TimeoutExpired:
            if os.name == 'posix':
                proc.send_signal(signal.SIGABRT)  # faulthandler prints the Python stack
                try:
                    out, err = proc.communicate(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    out, err = proc.communicate()
            else:
                proc.kill()
                out, err = proc.communicate()
            pytest.fail(f"no answer in {timeout}s from {args[:4]!r}...\n"
                        f"{report}\n--- its stdout ---\n{out[-4000:]}\n"
                        f"--- its stderr ---\n{err[-4000:]}", pytrace=False)
    except BaseException:
        proc.kill()
        proc.communicate()
        raise
    result = subprocess.CompletedProcess(args, proc.returncode, out, err)
    result.hang_report = report
    return result
