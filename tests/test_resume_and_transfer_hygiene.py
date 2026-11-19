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
import tkinter.messagebox

import pytest  # noqa: E402

from tests.conftest import editor_or_skip


def _resume(ed, path, monkeypatch):
    """`_maybe_resume_session()` finds its own file via
    `_find_resumable_session`; point that at the fixture instead of passing a
    path it does not accept."""
    monkeypatch.setattr(type(ed), '_find_resumable_session',
                        lambda self: str(path))
    monkeypatch.setattr(type(ed), '_prune_autosaves', lambda self: None)
    ed._maybe_resume_session()


def _editor_or_skip():
    return editor_or_skip()


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
        monkeypatch.setattr(tkinter.messagebox, 'showwarning',
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


def test_a_damaged_autosave_is_refused_in_plain_words(tmp_path, monkeypatch):
    """A damaged autosave was told apart from nothing: it raised a raw
    TypeError in the restore. Now the warning says what is damaged, and
    no sample is restored."""
    root, ed = _editor_or_skip()
    try:
        warned = []
        monkeypatch.setattr(tkinter.messagebox, 'showwarning',
                            lambda title, text, **k: warned.append(text))
        monkeypatch.setattr(tkinter.messagebox, 'askyesno', lambda *a, **k: True)
        from openflo.session_format import SESSION_FORMAT, SESSION_VERSION
        p = tmp_path / 'damaged.flowsession'
        p.write_text(json.dumps({
            'format': SESSION_FORMAT, 'version': SESSION_VERSION,
            'samples': [{'name': 's1', 'path': 7}]}), encoding='utf-8')
        before_dir = getattr(ed, '_session_dir', None)
        _resume(ed, p, monkeypatch)
        assert len(warned) == 1
        assert "damaged and can't be opened" in warned[0]
        assert "sample 's1': 'path' is a number" in warned[0]
        assert getattr(ed, '_session_dir', None) == before_dir
        assert not ed._samples
    finally:
        root.destroy()


@pytest.mark.parametrize('samples', [5, True, {'name': 's1'}, 'abc'])
def test_an_autosave_whose_sample_list_is_not_a_list_is_ignored(
        tmp_path, samples, monkeypatch):
    """`len()` of a number raised TypeError at launch; a record was offered
    as one sample."""
    root, ed = _editor_or_skip()
    try:
        asked = []
        monkeypatch.setattr(tkinter.messagebox, 'askyesno',
                            lambda *a, **k: asked.append(a) or False)
        from openflo.session_format import SESSION_FORMAT
        p = tmp_path / 'odd.flowsession'
        p.write_text(json.dumps({'format': SESSION_FORMAT,
                                 'samples': samples}), encoding='utf-8')
        _resume(ed, p, monkeypatch)
        assert asked == []
        assert not ed._samples
    finally:
        root.destroy()


def test_upgrade_saved_session_names_the_damage(tmp_path, monkeypatch):
    """File > Upgrade saved session... on a damaged file says what is
    damaged in its own words, not 'could not be upgraded' and a raw error
    under Show details."""
    root, ed = _editor_or_skip()
    try:
        import openflo.editor_lifecycle as L
        from openflo.session_format import SESSION_FORMAT
        p = tmp_path / 'old.flowsession'
        p.write_text(json.dumps({
            'format': SESSION_FORMAT, 'version': 1,
            'samples': [{'name': 's1', 'trial': ['x']}]}), encoding='utf-8')
        monkeypatch.setattr(L.filedialog, 'askopenfilename',
                            lambda **k: str(p))
        shown = []
        monkeypatch.setattr(tkinter.messagebox, 'showerror',
                            lambda title, text, **k: shown.append(text))
        monkeypatch.setattr(L, 'show_error',
                            lambda *a, **k: shown.append('generic'))
        ed._upgrade_session_file()
        assert len(shown) == 1
        assert "sample 's1': 'trial' is a list" in shown[0]
    finally:
        root.destroy()
