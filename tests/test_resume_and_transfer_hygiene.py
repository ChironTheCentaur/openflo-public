"""Startup-resume robustness and transfer-directory hygiene.

Three findings from the editor-mixin audit, all in code that runs at launch or
accumulates across an install's lifetime:

* a valid-JSON but non-dict autosave crashed the resume with an uncaught
  ``AttributeError`` — during startup, outside any handler;
* a resume refused by the version check still left ``_session_dir`` /
  ``_session_data_dir`` pointing at the rejected file;
* ``~/.openflo/transfer`` accumulated ``.done`` markers forever, with no
  equivalent of the autosave directory's ``_prune_autosaves``.
"""
import json
import os
import time

os.environ.setdefault('MPLBACKEND', 'Agg')

import pytest  # noqa: E402

from tests.conftest import gui_unavailable


def _resume(ed, path, monkeypatch):
    """`_maybe_resume_session()` finds its own file via
    `_find_resumable_session`; point that at the fixture instead of passing a
    path it does not accept."""
    monkeypatch.setattr(type(ed), '_find_resumable_session',
                        lambda self: str(path))
    monkeypatch.setattr(type(ed), '_prune_autosaves', lambda self: None)
    ed._maybe_resume_session()


def _editor_or_skip():
    try:
        import tkinter as tk
    except ImportError:
        gui_unavailable('tkinter not available')
    try:
        root = tk.Tk()
        root.withdraw()
    except Exception as e:                            # noqa: BLE001
        pytest.skip(str(e))
    import importlib
    gui = importlib.import_module('openflo.gui')
    gui.messagebox.askyesno = lambda *a, **k: True
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    return root, ed


# ── transfer-directory hygiene ───────────────────────────────────────────────

def test_stale_done_markers_are_pruned(tmp_path, monkeypatch):
    """A marker is normally consumed by the source instance, but not when that
    instance exited or cancelled — nothing else ever removed them."""
    root, ed = _editor_or_skip()
    try:
        monkeypatch.setattr(os.path, 'expanduser', lambda p: str(tmp_path))
        d = ed._transfer_dir()
        old = os.path.join(d, 'ancient.done')
        fresh = os.path.join(d, 'recent.done')
        for p in (old, fresh):
            open(p, 'w').close()
        os.utime(old, (time.time() - 3 * 86400,) * 2)

        ed._transfer_dir()                            # prunes on access

        assert not os.path.exists(old), 'a 3-day-old marker was not pruned'
        assert os.path.exists(fresh), 'a fresh marker must survive'
    finally:
        root.destroy()


def test_pruning_never_touches_other_files(tmp_path, monkeypatch):
    root, ed = _editor_or_skip()
    try:
        monkeypatch.setattr(os.path, 'expanduser', lambda p: str(tmp_path))
        d = ed._transfer_dir()
        keep = os.path.join(d, 'notes.txt')
        open(keep, 'w').close()
        os.utime(keep, (time.time() - 30 * 86400,) * 2)
        ed._transfer_dir()
        assert os.path.exists(keep), 'pruning removed a non-marker file'
    finally:
        root.destroy()


# ── resume robustness ────────────────────────────────────────────────────────

@pytest.mark.parametrize('payload', ['[]', '"a string"', '42', 'null'])
def test_a_non_dict_session_file_does_not_crash_startup(tmp_path, payload,
                                                       monkeypatch):
    """Valid JSON is not necessarily a session. `.get` on a list raises
    AttributeError — at launch, outside any handler."""
    root, ed = _editor_or_skip()
    try:
        p = tmp_path / 'bad.flowsession'
        p.write_text(payload, encoding='utf-8')
        _resume(ed, p, monkeypatch)                    # must simply return
    finally:
        root.destroy()


def test_a_refused_session_does_not_leave_the_session_dir_pointing_at_it(
        tmp_path, monkeypatch):
    """The version check refuses an autosave from a NEWER build; the paths
    must not be left addressing the rejected file."""
    root, ed = _editor_or_skip()
    try:
        import openflo.editor_lifecycle as L
        monkeypatch.setattr(L.messagebox, 'showwarning',
                            lambda *a, **k: None)
        from openflo.session_format import SESSION_FORMAT

        p = tmp_path / 'future.flowsession'
        p.write_text(json.dumps({
            'format': SESSION_FORMAT, 'version': 9_999,
            'samples': [{'name': 's1'}]}), encoding='utf-8')

        before_dir = getattr(ed, '_session_dir', None)
        before_data = getattr(ed, '_session_data_dir', None)
        _resume(ed, p, monkeypatch)

        assert getattr(ed, '_session_dir', None) == before_dir, (
            '_session_dir was left pointing at a session this build refused')
        assert getattr(ed, '_session_data_dir', None) == before_data
    finally:
        root.destroy()


def test_an_empty_session_is_ignored(tmp_path, monkeypatch):
    root, ed = _editor_or_skip()
    try:
        p = tmp_path / 'empty.flowsession'
        p.write_text(json.dumps({'samples': []}), encoding='utf-8')
        _resume(ed, p, monkeypatch)
        assert not ed._samples
    finally:
        root.destroy()


def test_a_missing_or_unreadable_file_is_ignored(tmp_path, monkeypatch):
    root, ed = _editor_or_skip()
    try:
        _resume(ed, tmp_path / 'nope.flowsession', monkeypatch)
        bad = tmp_path / 'garbage.flowsession'
        bad.write_text('{not json at all', encoding='utf-8')
        _resume(ed, bad, monkeypatch)
        assert not ed._samples
    finally:
        root.destroy()
