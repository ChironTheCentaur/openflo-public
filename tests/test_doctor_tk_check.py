"""A missing Tk must name its own fix.

`tkinter` is the one requirement that cannot be a dependency: it is a stdlib
module backed by a system Tcl/Tk, so pip can neither install it nor notice its
absence. On Windows and macOS it ships with the interpreter; on Debian/Ubuntu
the python3 package deliberately omits it.

So `pip install openflo` can succeed completely and then `openflo-gui` dies
with an ImportError that does not say what to do about it. `openflo-doctor` is
where a user without a checkout goes to find out, so it has to answer that —
including which command to run on their platform, and that the CLI and library
API work regardless.
"""
import builtins
import sys

import pytest

from openflo import diagnostics as D
from tests.conftest import gui_unavailable


@pytest.fixture
def no_tk(monkeypatch):
    """Make `import tkinter` fail the way a Tk-less Linux box does."""
    real_import = builtins.__import__

    def fake(name, *args, **kwargs):
        if name == 'tkinter':
            raise ImportError('libtk8.6.so: cannot open shared object file')
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, 'tkinter', raising=False)
    monkeypatch.setattr(builtins, '__import__', fake)


def test_a_working_tk_is_reported_with_its_version():
    tk = D.check_tk()
    if not tk['available']:                      # a genuinely Tk-less runner
        gui_unavailable('Tk is not available in this environment')
    assert tk['version']
    assert tk['remedy'] == ''


@pytest.mark.parametrize('platform, expected', [
    pytest.param('linux', 'apt install python3-tk', id='debian'),
    pytest.param('darwin', 'brew install python-tk', id='macos'),
    pytest.param('win32', 'python.org', id='windows'),
])
def test_a_missing_tk_names_the_platform_fix(no_tk, monkeypatch, platform,
                                             expected):
    monkeypatch.setattr(sys, 'platform', platform)
    tk = D.check_tk()
    assert tk['available'] is False
    assert expected in tk['remedy'], (
        f'the {platform} remedy does not mention {expected!r}: '
        f'{tk["remedy"]!r}')


def test_an_unknown_platform_still_says_something_useful(no_tk, monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'sunos5')
    assert 'Tk' in D.check_tk()['remedy']


def test_the_report_tells_the_user_the_gui_cannot_start(no_tk, monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'linux')
    report = D.run_diagnostics(include_selftest=False)
    text = D.format_report(report)

    assert 'openflo-gui cannot start' in text
    assert 'apt install python3-tk' in text
    assert 'do not need Tk' in text, (
        'the report does not say the CLI still works, so a CLI user would '
        'think the install is broken')


def test_missing_tk_does_not_by_itself_fail_the_install(no_tk, monkeypatch):
    """A headless server running `openflo-run` is a perfectly good install —
    reporting it as broken would be wrong."""
    monkeypatch.setattr(sys, 'platform', 'linux')
    report = D.run_diagnostics(include_selftest=False)
    assert report['tk']['available'] is False
    assert report['ok'] is True, (
        'a Tk-less but otherwise complete install was reported unhealthy'
    )


def test_the_report_includes_tk_when_it_is_present():
    report = D.run_diagnostics(include_selftest=False)
    if not report['tk']['available']:
        gui_unavailable('Tk is not available in this environment')
    assert 'Graphical interface' in D.format_report(report)
