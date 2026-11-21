"""A test that hangs leaves every thread's stack in a file CI prints.

CI stops a test still running after ``--timeout`` seconds (pytest-timeout,
``--timeout-method=thread``): it prints every thread's stack and ends the
worker. On Windows a pytest-xdist worker's output does not reach the job's
log, so all that shows is "[gw1] node down: Not properly terminated" and
"worker 'gw1' crashed while running <test>", with no hint of where it hung
(the Windows py3.13 leg after the 3.1.1 release, the first time this was seen
outside the dialog sweep, which has its own watchdog).

With ``OPENFLO_TEST_REPORTS`` set (CI sets it on Windows, and prints the
``*.txt`` files there when a job fails), this plugin arms
``faulthandler.dump_traceback_later`` as each test's timeout is set, a little
before pytest-timeout would end the worker: a test still running then writes
its id and every thread's stack to ``hung-test-<pid>.txt``. faulthandler's
own C watchdog writes it, so a test that hangs holding the GIL, or inside Tk
or Windows, is caught too, and it adds no Python thread a test could see.
The file is emptied when each test's timeout is cancelled and deleted at the
end of the session when nothing hung. Writing the report does not end the
test; pytest-timeout still does that at the timeout.

A test that fails makes pytest-timeout stop its timer (it does so for pdb)
and pytest's faulthandler plugin cancel faulthandler's: a hang in the
clean-up after a failure then runs until the CI job's own time limit, which
cancels the job (CI prints the reports on cancel too). Unless pdb is on, the
report is set again for the time it had left, so that hang names itself.

pytest's own ``faulthandler_timeout`` uses the same single faulthandler
timer, so when that is set this plugin stays out of the way.
"""
from __future__ import annotations

import faulthandler
import os
import time

import pytest

FOLDER_ENV = 'OPENFLO_TEST_REPORTS'
NAME = 'hung-test-{pid}.txt'
# Seconds before the timeout to write the stacks: at most a minute, at most a
# tenth of the timeout, so a short timeout still leaves time to write them.
MARGIN_MAX_S = 60.0


def dump_after(timeout: float) -> float:
    """Seconds into a test with ``timeout`` at which its stacks are written."""
    return timeout - min(MARGIN_MAX_S, timeout / 10)


class HangReport:
    """One process's report file and faulthandler timer."""

    def __init__(self, folder: str):
        self.folder = folder
        self.path = os.path.join(folder, NAME.format(pid=os.getpid()))
        self.file = None
        self.armed = False
        self.holding = False
        self.nodeid = ''
        self.dump_at = 0.0          # time.monotonic() when the stacks are due

    def arm(self, nodeid: str, timeout: float | None) -> bool:
        """Write ``nodeid``'s stacks after ``dump_after(timeout)`` s unless
        ``disarm`` comes first. No timeout, no report."""
        if not timeout or timeout <= 0:
            return False
        if self.file is None:
            os.makedirs(self.folder, exist_ok=True)
            self.file = open(self.path, 'w', encoding='utf-8')  # noqa: SIM115 - held open: faulthandler writes to it later
        after = dump_after(timeout)
        self.nodeid, self.dump_at = nodeid, time.monotonic() + after
        self._start(after, f"Still running after {after:g} s; the timeout "
                           f"({timeout:g} s) ends this worker next.")
        self.armed, self.holding = True, False
        return True

    def hold(self) -> None:
        """The test raised: keep its report through the ``disarm`` that
        pytest-timeout's exception hook sends next, until ``rearm``."""
        self.holding = self.armed

    def rearm(self) -> bool:
        """After the test raised, set faulthandler's timer again (pytest's
        own exception hook cancelled it) for the time left until its stacks
        were due. pytest-timeout no longer guards the test then, so a later
        hang (its clean-up, say) is ended only by the job's time limit."""
        if not self.holding:
            return False
        self.holding = False
        left = max(self.dump_at - time.monotonic(), 0.001)
        self._start(left, "It failed first, so pytest-timeout stopped its "
                          "timer and nothing ends this worker but the job's "
                          "own time limit.")
        return True

    def disarm(self) -> None:
        """The test finished in time: forget its report."""
        if not self.armed or self.holding:
            return
        faulthandler.cancel_dump_traceback_later()
        self.armed = False
        self._empty()

    def close(self) -> None:
        """End of session: stop the timer, delete the file if nothing hung."""
        self.disarm()
        if self.file is None:
            return
        self.file.close()
        self.file = None
        try:
            if os.path.getsize(self.path) == 0:
                os.remove(self.path)
        except OSError:
            pass                    # a report left behind is only noise

    def _start(self, after, note):
        """Write the header and set faulthandler's timer for ``after`` s."""
        self._empty()
        self.file.write(f"Hung test: {self.nodeid}\n{note} Every thread's "
                        "stack, most recent call first:\n")
        self.file.flush()
        faulthandler.dump_traceback_later(after, repeat=False,
                                          file=self.file, exit=False)

    def _empty(self):
        self.file.seek(0)
        self.file.truncate()
        self.file.flush()


class Plugin:
    """The pytest hooks; registered by tests/conftest.py (or ``-p``)."""

    def __init__(self, report: HangReport):
        self.report = report

    @pytest.hookimpl(optionalhook=True, tryfirst=True)
    def pytest_timeout_set_timer(self, item, settings):
        self.report.arm(item.nodeid, settings.timeout)
        # None: pytest-timeout's own hook still sets its timer.

    @pytest.hookimpl(optionalhook=True, tryfirst=True)
    def pytest_timeout_cancel_timer(self, item):
        self.report.disarm()

    # A test that raised: pytest-timeout (for pdb's sake) cancels its timer,
    # and with it this report, and pytest's faulthandler plugin cancels
    # faulthandler's timer. Unless pdb is on, keep the report and set the
    # timer again after both, so a hang in the clean-up that follows a
    # failure (a dialog left open, say) still names itself.
    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(self, node, call, report):
        if not node.config.getoption('usepdb', False):
            self.report.hold()

    @pytest.hookimpl(trylast=True, specname='pytest_exception_interact')
    def pytest_exception_interact_rearm(self, node, call, report):
        self.report.rearm()

    def pytest_unconfigure(self, config):
        self.report.close()


def register(config) -> None:
    """Register the plugin when ``OPENFLO_TEST_REPORTS`` names a folder and
    pytest's own ``faulthandler_timeout`` is not using faulthandler's timer.
    Once per process: it is also loadable as ``-p tests._hang_report``."""
    folder = os.environ.get(FOLDER_ENV)
    if not folder or config.pluginmanager.has_plugin('openflo-hang-report'):
        return
    if float(config.getini('faulthandler_timeout') or 0) > 0:
        return
    config.pluginmanager.register(Plugin(HangReport(folder)),
                                  'openflo-hang-report')


def pytest_configure(config):
    register(config)
