"""Broad ``except Exception`` handlers narrowed to the error they are for (R11).

Each narrowed handler gets two tests:

* the normal case: the failure the handler was written for (a Tk widget or
  window that is gone, a file that will not open, read or parse) is still
  handled exactly as before, so these pass on the old code and the new;
* the narrowing: any other error (a bug) now propagates instead of being
  swallowed, so these fail on the old code and pass on the new.

Generated data, synthetic sample names and generic channels only.
"""
from __future__ import annotations

import json
import os
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.conftest import gui_unavailable


class Bug(RuntimeError):
    """An error no narrowed handler was written for."""


def _raise(exc):
    def f(*_a, **_k):
        raise exc
    return f


@pytest.fixture
def root():
    try:
        r = tk.Tk()
    except tk.TclError as e:
        gui_unavailable(f"Tk cannot initialise: {e}")
    r.withdraw()
    yield r
    try:
        r.destroy()
    except tk.TclError:
        pass


def _gone(widget):
    widget.destroy()
    return widget


# ── Tk teardown → tk.TclError ────────────────────────────────────────────────

def test_busy_strip_finish_on_a_destroyed_strip_is_quiet(root):
    from openflo.async_task import BusyStrip
    strip = BusyStrip(root)
    strip._job = object()
    strip.destroy()
    strip._finish()
    assert strip._job is None and strip._disabled == []


def test_busy_strip_finish_lets_a_bug_through(root):
    from openflo.async_task import BusyStrip
    strip = BusyStrip(root)
    strip._bar = SimpleNamespace(stop=_raise(Bug('stop')))
    with pytest.raises(Bug):
        strip._finish()


def _polling_strip(root):
    from openflo.async_task import BusyStrip, Job
    strip = BusyStrip(root)
    strip._job = Job()                       # running, no progress yet
    return strip


def test_busy_strip_poll_stops_when_tk_refuses_the_next_poll(root, monkeypatch):
    strip = _polling_strip(root)
    monkeypatch.setattr(strip, 'after',
                        _raise(tk.TclError('application has been destroyed')))
    strip._poll()
    assert strip._poll_id is None


def test_busy_strip_poll_keeps_polling_while_alive(root):
    strip = _polling_strip(root)
    strip._poll()
    assert strip._poll_id is not None
    strip.after_cancel(strip._poll_id)


def test_busy_strip_poll_lets_a_bug_through(root, monkeypatch):
    strip = _polling_strip(root)
    monkeypatch.setattr(strip, 'after', _raise(Bug('after')))
    with pytest.raises(Bug):
        strip._poll()


def test_schedule_drain_stops_when_the_window_is_gone():
    from openflo.tk_threads import schedule_drain
    asked = []

    def after(ms, fn):
        asked.append(ms)
        raise tk.TclError('application has been destroyed')
    schedule_drain(SimpleNamespace(after=after))
    assert len(asked) == 1


def test_schedule_drain_lets_a_bug_through():
    from openflo.tk_threads import schedule_drain
    with pytest.raises(Bug):
        schedule_drain(SimpleNamespace(after=_raise(Bug('after'))))


def test_close_window_on_a_destroyed_window_is_quiet(root):
    from openflo.tool_window import close_window
    win = _gone(tk.Toplevel(root))
    close_window(win)                        # neither protocol nor destroy raise


def test_close_window_runs_the_close_box_handler(root):
    from openflo.tool_window import close_window
    win = tk.Toplevel(root)
    asked = []
    win.protocol('WM_DELETE_WINDOW', lambda: asked.append(1))
    close_window(win)
    assert asked == [1] and win.winfo_exists()
    win.destroy()


def test_close_window_lets_a_bug_through():
    from openflo.tool_window import close_window
    with pytest.raises(Bug):
        close_window(SimpleNamespace(protocol=_raise(Bug('protocol'))))


def test_selected_tab_rows_of_a_destroyed_tree_is_empty(root):
    from openflo.editor_tabs import TabsMixin
    ed = SimpleNamespace(gate_tv=_gone(ttk.Treeview(root)))
    assert TabsMixin._selected_tab_rows(ed) == []


def test_selected_tab_rows_lets_a_bug_through():
    from openflo.editor_tabs import TabsMixin
    ed = SimpleNamespace(gate_tv=SimpleNamespace(selection=_raise(Bug('sel'))))
    with pytest.raises(Bug):
        TabsMixin._selected_tab_rows(ed)


_TREE_TAG_KEYS = ('muted', 'highlight', 'highlight_fg', 'accent',
                  'console_bg', 'console_fg')


def test_every_theme_has_the_colours_the_tree_tags_use():
    """A missing palette key is a bug the narrowed handler no longer hides."""
    from openflo.theme import THEMES
    for name, pal in THEMES.items():
        assert set(_TREE_TAG_KEYS) <= set(pal), name


def test_theme_tree_tags_on_destroyed_widgets_is_quiet(root):
    from openflo.editor_chrome import ChromeMixin
    from openflo.theme import THEMES
    ed = SimpleNamespace(gate_tv=_gone(ttk.Treeview(root)),
                         _log_text=_gone(tk.Text(root)))
    ChromeMixin._theme_tree_tags(ed, THEMES['dark'])


def test_theme_tree_tags_colours_live_widgets(root):
    from openflo.editor_chrome import ChromeMixin
    from openflo.theme import THEMES
    pal = THEMES['dark']
    ed = SimpleNamespace(gate_tv=ttk.Treeview(root), _log_text=tk.Text(root))
    ChromeMixin._theme_tree_tags(ed, pal)
    assert str(ed._log_text.cget('background')) == pal['console_bg']
    assert str(ed.gate_tv.tag_configure('pending_move', 'foreground')) == pal['accent']


def test_theme_tree_tags_lets_a_bug_through(root):
    from openflo.editor_chrome import ChromeMixin
    from openflo.theme import THEMES
    ed = SimpleNamespace(gate_tv=ttk.Treeview(root),
                         _log_text=SimpleNamespace(configure=_raise(Bug('cfg'))))
    with pytest.raises(Bug):
        ChromeMixin._theme_tree_tags(ed, THEMES['dark'])


def test_retheme_open_tk_with_its_root_gone_is_quiet():
    from openflo.editor_chrome import ChromeMixin
    from openflo.theme import THEMES
    try:
        other = tk.Tk()
    except tk.TclError as e:
        gui_unavailable(f"Tk cannot initialise: {e}")
    other.destroy()                          # winfo_children now raises TclError
    ed = SimpleNamespace(_root=lambda: other)
    ChromeMixin._retheme_open_tk(ed, THEMES['dark'])


def test_retheme_open_tk_recolours_a_grey_frame(root):
    from openflo.editor_chrome import ChromeMixin
    from openflo.theme import THEMES
    pal = THEMES['dark']
    frame = tk.Frame(root, background='#d9d9d9')
    ed = SimpleNamespace(_root=lambda: root)
    ChromeMixin._retheme_open_tk(ed, pal)
    assert str(frame.cget('background')) == pal['bg']


def test_retheme_open_tk_lets_a_bug_through():
    from openflo.editor_chrome import ChromeMixin
    from openflo.theme import THEMES
    ed = SimpleNamespace(_root=lambda: SimpleNamespace(
        winfo_children=_raise(Bug('children'))))
    with pytest.raises(Bug):
        ChromeMixin._retheme_open_tk(ed, THEMES['dark'])


# ── one file read or write → OSError (plus ValueError for parsing) ───────────

def test_done_marker_is_written_with_the_paths_taken(tmp_path):
    from openflo.editor_clipboard import ClipboardMixin
    ed = SimpleNamespace(
        _inbound_move={'id': 'move1', 'pending': {'k'}, 'taken': []},
        _transfer_dir=lambda: str(tmp_path))
    ClipboardMixin._note_inbound_landed(ed, 'k')
    with open(tmp_path / 'move1.done', encoding='utf-8') as fh:
        assert json.load(fh) == {'taken': ['k']}


@pytest.mark.parametrize('where', ['missing folder', 'directory'])
def test_done_marker_that_cannot_be_written_is_noted(tmp_path, monkeypatch,
                                                     where):
    from openflo import editor_clipboard
    from openflo.editor_clipboard import ClipboardMixin
    notes = []
    monkeypatch.setattr(editor_clipboard, 'note',
                        lambda what, exc: notes.append((what, exc)))
    if where == 'directory':                 # IsADirectoryError / PermissionError
        (tmp_path / 'move1.done').mkdir()
        folder = tmp_path
    else:
        folder = tmp_path / 'missing'
    ed = SimpleNamespace(
        _inbound_move={'id': 'move1', 'pending': {'k'}, 'taken': []},
        _transfer_dir=lambda: str(folder))
    ClipboardMixin._note_inbound_landed(ed, 'k')
    assert [w for w, _e in notes] == ["write the transfer done-marker"]
    assert isinstance(notes[0][1], OSError)


def test_done_marker_lets_a_bug_through(tmp_path, monkeypatch):
    from openflo import editor_clipboard
    from openflo.editor_clipboard import ClipboardMixin
    monkeypatch.setattr(editor_clipboard.json, 'dump', _raise(Bug('dump')))
    ed = SimpleNamespace(
        _inbound_move={'id': 'move1', 'pending': {'k'}, 'taken': []},
        _transfer_dir=lambda: str(tmp_path))
    with pytest.raises(Bug):
        ClipboardMixin._note_inbound_landed(ed, 'k')


@pytest.fixture
def keymap(tmp_path, monkeypatch):
    from openflo import editor_lifecycle
    path = tmp_path / 'report.keys.json'
    monkeypatch.setattr(editor_lifecycle, '_error_keymap_path', lambda: str(path))
    return editor_lifecycle, path


SECRET = '/synthetic/place/sample_A1.fcs'


@pytest.mark.parametrize('content', [None, '{not json', b'\xff\xfe\x00'])
def test_report_keymap_missing_or_unreadable_starts_a_new_map(keymap, content):
    mod, path = keymap
    if isinstance(content, str):
        path.write_text(content, encoding='utf-8')
    elif isinstance(content, bytes):
        path.write_bytes(content)
    out = mod._tokenise_for_report(f"failed on {SECRET}", extra_values=[SECRET])
    assert SECRET not in out and '<' in out
    assert SECRET in json.loads(path.read_text(encoding='utf-8')).values()


def test_report_keymap_is_reused(keymap):
    mod, path = keymap
    path.write_text(json.dumps({'<value:7>': SECRET}), encoding='utf-8')
    out = mod._tokenise_for_report(f"failed on {SECRET}", extra_values=[SECRET])
    assert out == 'failed on <value:7>'


def test_report_keymap_read_lets_a_bug_through(keymap, monkeypatch):
    mod, path = keymap
    path.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(mod.json, 'load', _raise(Bug('load')))
    with pytest.raises(Bug):
        mod._tokenise_for_report('text')


def test_report_keymap_that_cannot_be_saved_is_noted(keymap, monkeypatch):
    mod, path = keymap
    path.mkdir()                             # a directory: open() fails
    notes = []
    monkeypatch.setattr(mod, 'note', lambda what, exc: notes.append(what))
    out = mod._tokenise_for_report(f"failed on {SECRET}", extra_values=[SECRET])
    assert SECRET not in out
    assert notes == ["write the error-report keymap"]


def test_report_keymap_write_lets_a_bug_through(keymap, monkeypatch):
    mod, _path = keymap
    monkeypatch.setattr(mod.json, 'dump', _raise(Bug('dump')))
    with pytest.raises(Bug):
        mod._tokenise_for_report('text')


@pytest.mark.parametrize('where', ['missing folder', 'directory'])
def test_log_sink_falls_back_to_the_null_device(tmp_path, where):
    from openflo.gui import _open_log_sink
    if where == 'directory':                 # IsADirectoryError / PermissionError
        log = tmp_path / 'openflo-gui.log'
        log.mkdir()
    else:
        log = tmp_path / 'missing' / 'openflo-gui.log'
    sink = _open_log_sink(str(log))
    try:
        assert sink.name == os.devnull
    finally:
        sink.close()


def test_log_sink_appends_to_the_log(tmp_path):
    from openflo.gui import _open_log_sink
    path = tmp_path / 'openflo-gui.log'
    path.write_text('before\n', encoding='utf-8')
    sink = _open_log_sink(str(path))
    sink.write('after\n')
    sink.close()
    assert path.read_text(encoding='utf-8') == 'before\nafter\n'


def test_log_sink_lets_a_bug_through(tmp_path, monkeypatch):
    from openflo import gui
    log = str(tmp_path / 'openflo-gui.log')

    def fake_open(path, *a, **k):            # only the log; the null device opens
        if path == log:
            raise Bug('open')
        return open(path, *a, **k)
    monkeypatch.setattr(gui, 'open', fake_open, raising=False)
    with pytest.raises(Bug):
        gui._open_log_sink(log)


@pytest.mark.parametrize('content', [None, '{not json', b'\xff\xfe\x00', 'dir'])
def test_prefs_missing_or_corrupt_read_as_defaults(tmp_path, monkeypatch, content):
    from openflo import prefs
    path = tmp_path / 'prefs.json'
    monkeypatch.setattr(prefs, '_prefs_path', lambda: str(path))
    if content == 'dir':                     # IsADirectoryError / PermissionError
        path.mkdir()
    elif isinstance(content, str):
        path.write_text(content, encoding='utf-8')
    elif isinstance(content, bytes):
        path.write_bytes(content)
    assert prefs.read_prefs() == {}


def test_prefs_are_read(tmp_path, monkeypatch):
    from openflo import prefs
    path = tmp_path / 'prefs.json'
    monkeypatch.setattr(prefs, '_prefs_path', lambda: str(path))
    path.write_text(json.dumps({'theme': 'dark'}), encoding='utf-8')
    assert prefs.read_prefs() == {'theme': 'dark'}


def test_prefs_read_lets_a_bug_through(tmp_path, monkeypatch):
    from openflo import prefs
    path = tmp_path / 'prefs.json'
    path.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(prefs, '_prefs_path', lambda: str(path))
    monkeypatch.setattr(prefs.json, 'load', _raise(Bug('load')))
    with pytest.raises(Bug):
        prefs.read_prefs()


@pytest.mark.parametrize('content', ['{not json', b'\xff\xfe\x00'])
def test_unreadable_lock_reads_as_a_holder_with_no_details(tmp_path, content):
    from openflo import session_lock as L
    session = tmp_path / 'synthetic.ofs'
    lp = L.lock_path(str(session))
    with open(lp, 'wb') as fh:
        fh.write(content.encode('utf-8') if isinstance(content, str) else content)
    assert L.read_lock(str(session)) == {}


def test_a_lock_that_is_a_directory_reads_as_held(tmp_path):
    from openflo import session_lock as L
    session = tmp_path / 'synthetic.ofs'
    os.mkdir(L.lock_path(str(session)))
    assert L.read_lock(str(session)) == {}


def test_lock_read_lets_a_bug_through(tmp_path, monkeypatch):
    from openflo import session_lock as L
    session = tmp_path / 'synthetic.ofs'
    with open(L.lock_path(str(session)), 'w', encoding='utf-8') as fh:
        fh.write('{}')
    monkeypatch.setattr(L.json, 'load', _raise(Bug('load')))
    with pytest.raises(Bug):
        L.read_lock(str(session))


def _verify(monkeypatch, path):
    from openflo import ui_audit
    shown = []
    monkeypatch.setattr(ui_audit.filedialog, 'askopenfilename',
                        lambda **_k: str(path))
    monkeypatch.setattr(ui_audit, 'show_error',
                        lambda _p, title, msg, exc, *_a: shown.append(
                            (title, msg, type(exc))))
    ui_audit.AuditWindow._verify_record(SimpleNamespace())
    return shown


@pytest.mark.parametrize('content,kind', [
    ('{not json', json.JSONDecodeError),
    (b'\xff\xfe\x00', UnicodeDecodeError),
])
def test_a_malformed_compliance_record_is_shown(tmp_path, monkeypatch,
                                                content, kind):
    path = tmp_path / 'record.json'
    if isinstance(content, str):
        path.write_text(content, encoding='utf-8')
    else:
        path.write_bytes(content)
    assert _verify(monkeypatch, path) == [
        ("Compliance", "The compliance record could not be read.", kind)]


def test_a_missing_compliance_record_is_shown(tmp_path, monkeypatch):
    assert _verify(monkeypatch, tmp_path / 'missing.json') == [
        ("Compliance", "The compliance record could not be read.",
         FileNotFoundError)]


def test_compliance_record_read_lets_a_bug_through(tmp_path, monkeypatch):
    from openflo import ui_audit
    path = tmp_path / 'record.json'
    path.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(ui_audit.json, 'load', _raise(Bug('load')))
    with pytest.raises(Bug):
        _verify(monkeypatch, path)


# ── labels sidecars and sessions, read by a real editor ──────────────────────

def _events_frame():
    rng = np.random.default_rng(11)
    return pd.DataFrame({
        'FSC-A': rng.lognormal(10, 0.3, 200),
        'SSC-A': rng.lognormal(9, 0.4, 200),
        'FL1-A': rng.normal(500, 80, 200),
    })


def _write_labels(folder, base, content):
    path = folder / f'{base}_labels.json'
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding='utf-8')


def _npz_events(folder):
    from openflo.session_sidecar import write_frame
    path = folder / 'synthA_events.npz'
    assert write_frame(_events_frame(), str(path))
    return path


def test_run_events_load_with_their_labels(editor):
    ed, folder = editor
    path = _npz_events(folder)
    _write_labels(folder, 'synthA', json.dumps({'FL1-A': 'Marker one'}))
    name = ed._import_processed_csv(str(path))
    assert name == 'synthA'
    assert ed._samples[name].channel_labels['FL1-A'] == 'Marker one'


@pytest.mark.parametrize('content', ['{not json', b'\xff\xfe\x00'])
def test_run_events_with_a_bad_labels_file_load_without_labels(editor, content):
    ed, folder = editor
    path = _npz_events(folder)
    _write_labels(folder, 'synthA', content)
    name = ed._import_processed_csv(str(path))
    assert name == 'synthA'
    assert ed._samples[name].channel_labels['FL1-A'] == 'FL1-A'


def test_run_events_labels_read_lets_a_bug_through(editor, monkeypatch):
    ed, folder = editor
    path = _npz_events(folder)
    _write_labels(folder, 'synthA', '{}')
    monkeypatch.setattr(json, 'load', _raise(Bug('load')))
    with pytest.raises(Bug):
        ed._import_processed_csv(str(path))


def _csv_events(folder, monkeypatch):
    from openflo import editor_load
    path = folder / 'synthB_processed.csv'
    _events_frame().to_csv(path, index=False)
    monkeypatch.setattr(editor_load.filedialog, 'askopenfilenames',
                        lambda **_k: [str(path)])
    return path


def test_processed_csv_loads_with_its_labels(editor, monkeypatch):
    ed, folder = editor
    _csv_events(folder, monkeypatch)
    _write_labels(folder, 'synthB', json.dumps({'FL1-A': 'Marker two'}))
    ed._load_processed_data()
    assert ed._samples['synthB'].channel_labels['FL1-A'] == 'Marker two'


@pytest.mark.parametrize('content', ['{not json', b'\xff\xfe\x00'])
def test_processed_csv_with_a_bad_labels_file_loads_without_labels(
        editor, monkeypatch, content):
    ed, folder = editor
    _csv_events(folder, monkeypatch)
    _write_labels(folder, 'synthB', content)
    ed._load_processed_data()
    assert 'synthB' in ed._samples
    assert ed._samples['synthB'].channel_labels['FL1-A'] == 'FL1-A'


def test_processed_csv_labels_read_lets_a_bug_through(editor, monkeypatch):
    ed, folder = editor
    _csv_events(folder, monkeypatch)
    _write_labels(folder, 'synthB', '{}')
    monkeypatch.setattr(json, 'load', _raise(Bug('load')))
    with pytest.raises(Bug):
        ed._load_processed_data()


def _session_file(folder, content):
    path = folder / 'synthetic.ofs'
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding='utf-8')
    return path


@pytest.mark.parametrize('content,kind', [
    ('{not json', json.JSONDecodeError),
    (b'\xff\xfe\x00', UnicodeDecodeError),
])
def test_a_malformed_session_is_reported_and_its_lock_released(
        editor, monkeypatch, content, kind):
    from openflo import session_lock as L
    ed, folder = editor
    path = _session_file(folder, content)
    reported = []
    monkeypatch.setattr(ed, '_report_unreadable_session',
                        lambda p, exc: reported.append((p, type(exc))))
    ed._load_session_path(str(path))
    assert reported == [(str(path), kind)]
    assert L.read_lock(str(path)) is None
    assert ed.status_var.get().startswith("Load session failed:")


def test_session_read_lets_a_bug_through(editor, monkeypatch):
    from openflo import editor_session
    ed, folder = editor
    path = _session_file(folder, '{}')
    real_load = json.load

    def load(fh, *a, **k):                   # only the session file; the
        if os.path.samefile(fh.name, path):  # error report's keymap reads fine
            raise Bug('load')
        return real_load(fh, *a, **k)
    monkeypatch.setattr(editor_session.json, 'load', load)
    with pytest.raises(Bug):
        ed._load_session_path(str(path))


@pytest.fixture
def editor(tmp_path, monkeypatch):
    from tests.test_transform_records import _new_editor, capture_warnings, isolate_home
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)
    try:
        r = tk.Tk()
    except tk.TclError as e:
        gui_unavailable(f"Tk cannot initialise: {e}")
    r.withdraw()
    ed = _new_editor(r)
    folder = tmp_path / 'data'
    folder.mkdir()
    yield ed, folder
    r.destroy()
