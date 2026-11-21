"""A hung test leaves its id and every thread's stack in the CI report folder
(tests/_hang_report.py), which CI prints when a Windows job fails.

Each case runs pytest (or a short script) in a child process on a generated
test file, so the faulthandler timer and the worker that pytest-timeout ends
are the child's, never this run's.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytest.importorskip('pytest_timeout')

TESTS = textwrap.dedent('''
    import os
    import pathlib
    import time

    import pytest

    def _stuck_here():
        time.sleep(60)

    def test_quick():
        pass

    def test_hangs():
        _stuck_here()

    def test_slow_but_in_time():
        time.sleep(9.3)             # past the report (9 s), not the timeout

    def _stuck_in_cleanup():
        time.sleep(60)

    @pytest.fixture
    def cleanup_that_sticks():
        yield
        _stuck_in_cleanup()

    def test_fails_fast():
        assert False, 'failed and cleaned up at once'

    def test_fails_then_cleanup_sticks(cleanup_that_sticks):
        assert False, 'failed with a dialog still open'

    @pytest.mark.timeout(0)
    def test_slow_without_a_timeout():
        time.sleep(5)
        folder = pathlib.Path(os.environ['OPENFLO_TEST_REPORTS'])
        written = {p.name: p.read_text() for p in folder.glob('*.txt')}
        assert not any(written.values()), written
''')


def _env(reports):
    env = dict(os.environ)
    env['OPENFLO_TEST_REPORTS'] = str(reports)
    env['PYTHONPATH'] = os.pathsep.join(
        [str(ROOT)] + [p for p in [env.get('PYTHONPATH')] if p])
    return env


def _pytest(tmp_path, select, *extra, timeout=4):
    """Run the generated tests matching ``select`` in a child pytest with
    the plugin and ``--timeout``; returns the report folder and the run."""
    (tmp_path / 'pytest.ini').write_text('[pytest]\n', encoding='utf-8')
    (tmp_path / 'test_gen.py').write_text(TESTS, encoding='utf-8')
    reports = tmp_path / 'reports'
    done = subprocess.run(
        [sys.executable, '-m', 'pytest', '-p', 'tests._hang_report',
         '-p', 'no:cacheprovider', '-q', f'--timeout={timeout}',
         '--timeout-method=thread', '-k', select, *extra, 'test_gen.py'],
        cwd=tmp_path, env=_env(reports), capture_output=True, text=True,
        timeout=120)
    return reports, done


def _reports(folder):
    return sorted(folder.glob('*.txt')) if folder.is_dir() else []


@pytest.mark.parametrize('workers', [None, '1'],
                         ids=['one-process', 'xdist-worker'])
def test_a_hung_test_writes_its_id_and_stacks_before_the_timeout(
        tmp_path, workers):
    # CI runs under pytest-xdist, where the worker's own output is lost.
    if workers is not None:
        pytest.importorskip('xdist')
    extra = ('-n', workers) if workers else ('-p', 'no:xdist')
    reports, _done = _pytest(tmp_path, 'quick or hangs', *extra)
    found = _reports(reports)
    assert len(found) == 1, found
    text = found[0].read_text(encoding='utf-8', errors='replace')
    assert text.startswith('Hung test: test_gen.py::test_hangs\n')
    assert 'Still running after 3.6 s; the timeout (4 s)' in text
    # faulthandler's stack of the main thread, down to where it sleeps.
    assert 'most recent call first' in text
    assert '_stuck_here' in text and 'test_hangs' in text


def test_tests_that_finish_leave_no_report(tmp_path):
    reports, done = _pytest(tmp_path, 'quick')
    assert done.returncode == 0, done.stdout
    assert _reports(reports) == []


def test_a_finished_tests_timer_does_not_fire_in_the_next_test(tmp_path):
    # test_quick's timer (3.6 s) is cancelled when it passes; left running,
    # it would write a report naming test_quick while the next test, which
    # sets no timeout of its own, is still sleeping.
    reports, done = _pytest(tmp_path, 'quick or slow_without')
    assert done.returncode == 0, done.stdout     # the next test saw none
    assert _reports(reports) == []


def test_a_report_does_not_end_the_run(tmp_path):
    # The stacks are written and the test carries on: a slow test that ends
    # between the report (9 s) and the timeout (10 s) passes, and its report
    # is dropped.
    reports, done = _pytest(tmp_path, 'slow_but_in_time', timeout=10)
    assert done.returncode == 0, done.stdout
    assert _reports(reports) == []


def _hung_child(tmp_path, *extra, typed=None):
    """Start the generated test that fails and then sticks in its clean-up
    (for 60 s, past every timer here) in a child pytest with ``--timeout=4``.
    Returns the child and its report folder; the caller ends the child."""
    (tmp_path / 'pytest.ini').write_text('[pytest]\n', encoding='utf-8')
    (tmp_path / 'test_gen.py').write_text(TESTS, encoding='utf-8')
    reports = tmp_path / 'reports'
    child = subprocess.Popen(
        [sys.executable, '-m', 'pytest', '-p', 'tests._hang_report',
         '-p', 'no:cacheprovider', '-p', 'no:xdist', '-q', '--timeout=4',
         '--timeout-method=thread', '-k', 'cleanup_sticks', *extra,
         'test_gen.py'],
        cwd=tmp_path, env=_env(reports),
        stdin=subprocess.PIPE if typed else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if typed:
        child.stdin.write(typed)
        child.stdin.close()
    return child, reports


def _stacks_within(reports, seconds):
    """The report's text once its stacks are in, or '' after ``seconds``."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        for path in _reports(reports):
            text = path.read_text(encoding='utf-8', errors='replace')
            if 'most recent call first):' in text:
                return text
        time.sleep(0.1)
    return ''


def _end(child):
    child.kill()
    out = child.stdout.read()
    child.wait(timeout=30)
    return out


def test_a_hang_in_the_cleanup_after_a_failure_still_reports(tmp_path):
    # A failure makes pytest-timeout and pytest's faulthandler plugin stop
    # their timers (for pdb); the report is set again for the time left.
    child, reports = _hung_child(tmp_path)
    try:
        text = _stacks_within(reports, 30)
        still_running = child.poll() is None    # nothing ended the worker
    finally:
        out = _end(child)
    assert still_running, out
    assert text.startswith(
        'Hung test: test_gen.py::test_fails_then_cleanup_sticks\n'
        'It failed first, so pytest-timeout stopped its timer'), text or out
    assert '_stuck_in_cleanup' in text


def test_a_failed_tests_report_is_dropped_when_it_ends(tmp_path):
    # Set again after the failure, the report still goes when the test
    # ends: it must not fire later, in the next test.
    reports, done = _pytest(tmp_path, 'fails_fast or slow_without', '-rf')
    assert 'FAILED test_gen.py::test_fails_fast' in done.stdout, done.stdout
    assert 'test_slow_without_a_timeout' not in done.stdout, done.stdout
    assert _reports(reports) == []


def test_with_pdb_a_failure_stops_the_report_too(tmp_path):
    # Someone at pdb is not hung: as pytest-timeout does, stop the report.
    # Here they type 'c' (continue) at the prompt, and the clean-up sticks.
    child, reports = _hung_child(tmp_path, '--pdb', typed='c\n')
    try:
        text = _stacks_within(reports, 8)       # 3.6 s were the stacks due
    finally:
        out = _end(child)
    assert 'Pdb' in out or 'PDB' in out, out
    assert text == ''


def test_pytests_own_faulthandler_timer_is_left_alone(tmp_path):
    # faulthandler has one timer: with faulthandler_timeout set pytest uses
    # it, and this plugin must not take it over.
    reports, done = _pytest(tmp_path, 'hangs',
                            '-o', 'faulthandler_timeout=100')
    assert done.returncode != 0             # it did hang until the timeout
    assert _reports(reports) == []


def test_a_finished_tests_timer_is_cancelled(tmp_path):
    # Disarmed before its time: nothing is written later, by the timer of a
    # test that already finished, and the empty file is removed.
    reports = tmp_path / 'reports'
    script = textwrap.dedent('''
        import time
        from tests._hang_report import HangReport
        report = HangReport(REPORTS)
        assert not report.arm('test_gen.py::test_quick', 0)   # no timeout
        assert not report.arm('test_gen.py::test_quick', None)
        assert report.file is None                          # nothing made
        assert report.arm('test_gen.py::test_quick', 1.0)
        report.disarm()
        time.sleep(1.5)
        report.close()
    ''').replace('REPORTS', repr(str(reports)))
    done = subprocess.run([sys.executable, '-c', script], cwd=tmp_path,
                          env=_env(reports), capture_output=True, text=True,
                          timeout=60)
    assert done.returncode == 0, done.stderr
    assert reports.is_dir()                 # the file was made, then removed
    assert _reports(reports) == []


def test_the_suites_conftest_turns_it_on_where_ci_asks(tmp_path):
    # tests/conftest.py registers the plugin when OPENFLO_TEST_REPORTS is
    # set (CI's Windows step), and not otherwise.
    script = textwrap.dedent('''
        import sys
        import pytest
        seen = []

        class Probe:
            def pytest_collection_finish(self, session):
                seen.append(session.config.pluginmanager.has_plugin(
                    'openflo-hang-report'))

        pytest.main(['--co', '-q', '-p', 'no:cacheprovider', '-p',
                     'no:xdist', 'tests/test_hang_report.py'],
                    plugins=[Probe()])
        print('registered', seen)
    ''')
    found = {}
    for name, reports in (('set', tmp_path / 'reports'), ('unset', None)):
        env = _env(reports)
        if reports is None:
            del env['OPENFLO_TEST_REPORTS']
        done = subprocess.run([sys.executable, '-c', script], cwd=ROOT,
                              env=env, capture_output=True, text=True,
                              timeout=300)
        found[name] = [ln for ln in done.stdout.splitlines()
                       if ln.startswith('registered')]
    assert found == {'set': ['registered [True]'],
                     'unset': ['registered [False]']}
