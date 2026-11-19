"""Handled errors in the GUI go through errors.note (survey R9, slice 2).

The GUI used to print its handled errors as tagged console lines
(``[session] tabs not restored: ...``). Those reached the log pane but not
``errors.noted()``, which the diagnostics read, and each module spelled them
its own way. Now they go through ``errors.note``: one ``[handled] ...`` line
on stderr (the pane mirrors stderr) and an entry in ``noted()``. The
best-practices rule ``print-in-gui`` keeps new prints out of GUI modules.
"""
from __future__ import annotations

import logging
import types

from openflo import editor_session, errors


def test_note_with_none_names_no_exception_even_inside_a_handler(capsys):
    """``note(what, None)`` is a problem with no exception behind it (a
    record skipped as not understood). Called while some unrelated error is
    being handled further up, it must not name that error."""
    try:
        raise KeyError('FL1-A')
    except KeyError:
        errors.note('session: s1: stat_axes ignored (not understood)', None)
    assert errors.noted()[-1] == (
        '[handled] session: s1: stat_axes ignored (not understood)')
    assert capsys.readouterr().err.strip().endswith('(not understood)')


def test_note_left_out_still_names_the_exception_being_handled():
    try:
        raise ValueError('bad value')
    except ValueError:
        errors.note('reading a preference')
    assert errors.noted()[-1] == (
        '[handled] reading a preference: ValueError: bad value')


def test_a_value_json_cannot_hold_is_noted(capsys):
    """Saving a session replaces NaN/inf with a stand-in and says so."""
    editor_session._report_nonfinite('stats.FL1-A', float('nan'), None)
    line = ('[handled] session: stats.FL1-A was nan, which JSON cannot '
            'represent; stored as None')
    assert errors.noted()[-1] == line
    captured = capsys.readouterr()
    assert line in captured.err
    assert captured.out == ''


def _autosave_stub():
    statuses = []
    return types.SimpleNamespace(
        _AUTOSAVE_WARN_AFTER=99, _AUTOSAVE_MS=300_000, _closing=False,
        _set_autosave_status=statuses.append), statuses


def test_a_periodic_autosave_failure_is_noted_and_its_recovery_logged(
        capsys, caplog):
    """Each failed periodic autosave is a handled error; the autosave that
    works again after a warned streak says so on the log too."""
    ed, statuses = _autosave_stub()
    finished = editor_session.SessionMixin._autosave_finished
    finished(ed, 'disk full')
    assert errors.noted()[-1] == (
        '[handled] session: periodic autosave failed (1 in a row): disk full')
    assert capsys.readouterr().out == ''
    ed._autosave_warned = True
    with caplog.at_level(logging.WARNING, logger='openflo.editor_session'):
        finished(ed, None)
    assert [r.getMessage() for r in caplog.records] == [
        'session: periodic autosave is working again']
    assert statuses[-1] == 'Autosave is working again.'
