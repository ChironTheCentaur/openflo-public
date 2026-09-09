"""A legacy Windows console must not kill a command-line entry point.

Every CLI here prints characters outside cp1252 — check and cross marks in the
self-test table, arrows in titration output, `delta` in the comparison report,
superscripts in synthetic marker names. A legacy Windows console encodes stdout
as cp1252, so printing any of them raises UnicodeEncodeError and the command
dies having reported nothing.

This is not hypothetical and it is not cosmetic. CI caught `openflo-selftest`
crashing mid-table on windows-latest: the command that exists to tell a new
user whether their install reproduces reference behaviour instead gave them a
traceback. Probing the rest turned up `openflo-doctor --help` failing the same
way *despite* having a guard, because the guard sat after `parse_args` and
argparse prints help and exits from inside it.

The fix is `openflo._console.force_utf8_streams()`, called first thing in
`main()`. These tests pin the behaviour, not the mechanism: they install a real
cp1252 stream and check the command survives.
"""
import contextlib
import io
import sys

import pytest

# (module, console-script name, does main() take argv?)
ENTRY_POINTS = [
    ('openflo.cli', 'openflo-run', False),
    ('openflo.compare', 'openflo-compare', False),
    ('openflo.voltage', 'openflo-voltage', True),
    ('openflo.synthetic', 'openflo-synth', True),
    ('openflo.selftest', 'openflo-selftest', True),
    ('openflo.diagnostics', 'openflo-doctor', True),
    ('openflo.selftest_controls', 'openflo-selftest --controls', True),
]


class _Cp1252Stream(io.TextIOWrapper):
    """Stdout as a legacy Windows console sees it."""

    def __init__(self):
        super().__init__(io.BytesIO(), encoding='cp1252', newline='')


@contextlib.contextmanager
def cp1252_console():
    """Run the body with stdout/stderr a legacy Windows console cannot encode.

    Deliberately a context manager rather than a fixture: pytest re-installs
    its own capture object into `sys.stdout` at the start of the CALL phase,
    after fixtures have run, so a stream swapped in during setup is silently
    discarded and the command under test never sees it. That produced a test
    which passed while exercising nothing.
    """
    out, err = _Cp1252Stream(), _Cp1252Stream()
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        yield out
    finally:
        sys.stdout, sys.stderr = real_out, real_err


@pytest.mark.parametrize('module, script, takes_argv', ENTRY_POINTS,
                         ids=[e[1] for e in ENTRY_POINTS])
def test_help_survives_a_cp1252_console(monkeypatch, module, script,
                                        takes_argv):
    """`--help` is the first thing anyone runs, and argparse prints it from
    inside parse_args — so a guard placed after that line does not cover it."""
    import importlib
    mod = importlib.import_module(module)

    monkeypatch.setattr(sys, 'argv', [script.split()[0], '--help'])
    with cp1252_console():
        with pytest.raises(SystemExit) as exc:
            mod.main(['--help']) if takes_argv else mod.main()
    assert exc.value.code in (0, None), f'{script} --help exited {exc.value.code}'


def test_the_selftest_results_table_survives_a_cp1252_console():
    """The actual CI failure: `openflo-selftest` printed its results table and
    raised UnicodeEncodeError on the check mark, so the command reported
    nothing at all on a legacy console."""
    from openflo import selftest

    with cp1252_console() as console:
        rc = selftest.main([])
        console.flush()
        text = console.buffer.getvalue().decode('utf-8', 'replace')

    assert rc == 0, 'the self-test itself failed, so this proves nothing'
    assert 'golden baseline' in text or 'self-test' in text.lower(), (
        'the table was not written to the console at all')


def test_the_guard_is_best_effort_on_a_stream_that_cannot_reconfigure():
    """Never let the guard itself break a command. A pytest capture buffer, a
    pipe wrapper or an embedded interpreter may not support `reconfigure`."""
    from openflo._console import force_utf8_streams

    class Stubborn:
        def reconfigure(self, **kwargs):
            raise AttributeError('not supported here')

    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = Stubborn()               # type: ignore[assignment]
    try:
        force_utf8_streams()                            # must not raise
    finally:
        sys.stdout, sys.stderr = real_out, real_err
