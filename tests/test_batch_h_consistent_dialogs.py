"""Batch H (UI 16, 17, 20, 21, 25 to 30): every dialog behaves the same way,
errors read as plain words, and the wording is consistent.

- Every window opened from the menus belongs to the editor (``transient``),
  closes on Escape, and has its main action, then Cancel or Close, at the
  right; Return runs the main action (tool_window.standardise_dialog).
- Errors say what failed and why in plain words, with the exception and its
  traceback under Show details (ui_errors); an uncaught error opens that
  window instead of the developer console.
- Keyboard shortcuts and About are proper windows, and every listed shortcut
  is really bound.
- Labels: no arrows used as words, sentence case, American spelling, the
  methods' published names, no "FlowJo-style", and display names instead
  of settings keys.
- The title is "session name — OpenFlo", with * while there are unsaved
  changes.

Each check runs on generated data (synthetic sample names, generic
channels) and is proved to fail on the old behaviour by
/mnt/project-files/batch-h/mutation_check.py.
"""
from __future__ import annotations

import ast
import os
import re
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import pytest

import tests.test_windows_nothing_hidden as sweep
from tests.conftest import gui_unavailable

os.environ.setdefault('MPLBACKEND', 'Agg')

SRC = Path(__file__).resolve().parents[1] / 'src' / 'openflo'


# ── The sweep: every window opened from the menus ──────────────────────────

def _open_windows(root, ed):
    """Yield (menu label, window) for each window a menu item opens; the
    window is destroyed after the caller is done with it."""
    for menu, i, label in list(sweep._menu_items(ed)):
        if label.startswith(sweep.NOT_A_WINDOW):
            continue
        before = set(sweep._toplevels(root))
        try:
            menu.invoke(i)
            root.update()
        except Exception:                              # noqa: BLE001
            continue
        for key, win in sweep._toplevels(root).items():
            if key in before or not win.winfo_exists():
                continue
            root.update()
            yield label, win
            if win.winfo_exists():
                win.destroy()
            root.update()


@pytest.fixture
def editor(virtual_screen, no_dialogs, engines_known):
    virtual_screen(1920, 1080)
    root = sweep._root(1.0)
    try:
        yield root, sweep._editor(root)
    finally:
        root.destroy()


virtual_screen = sweep.virtual_screen
no_dialogs = sweep.no_dialogs
engines_known = sweep.engines_known


def test_every_dialog_is_attached_closes_on_escape_and_orders_its_buttons(
        editor):
    from openflo.tool_window import DISMISS_LABELS, button_rows
    root, ed = editor
    opened, failures = [], []
    for label, win in _open_windows(root, ed):
        opened.append(label)
        name = f"{label} [{win.title()}]"
        if getattr(win, 'independent', False):
            if str(win.wm_transient()):
                failures.append(f"{name}: a work window made transient")
            continue
        if str(win.wm_transient()) != str(ed):
            failures.append(f"{name}: not attached to the editor "
                            f"({win.wm_transient()!r})")
        for _frame, buttons in button_rows(win):
            texts = [str(b.cget('text')) for b in buttons]
            if texts[-1] not in DISMISS_LABELS:
                failures.append(f"{name}: {texts} does not end with "
                                "Cancel or Close")
            if 'Cancel' in texts and len(texts) > 1:
                main = buttons[-2]
                if str(main.cget('default')) != 'active':
                    failures.append(f"{name}: {texts[-2]!r} beside Cancel "
                                    "is not the default (Return) action")
        win.focus_force()
        win.event_generate('<Escape>')
        root.update()
        if win.winfo_exists():
            failures.append(f"{name}: Escape did not close it")
    assert len(opened) >= 25, opened
    assert not failures, "\n".join(failures)


def _texts(win, out, where):
    """Every label, button, check box, radio button, menu entry, dropdown
    choice and column heading of ``win`` (not of windows opened from it)."""
    for w in sweep._walk(win):
        if w is not win and isinstance(w, tk.Toplevel):
            continue
        cls = w.winfo_class()
        if cls == 'Menu':
            end = w.index('end')
            for i in range(0 if end is None else end + 1):
                if w.type(i) in ('command', 'cascade', 'checkbutton',
                                 'radiobutton'):
                    out.append((where, 'menu', w.entrycget(i, 'label')))
            continue
        try:
            t = w.cget('text')
        except tk.TclError:
            t = ''
        if isinstance(t, str) and t:
            out.append((where, cls, t))
        if cls == 'TCombobox':
            out += [(where, 'choice', str(v)) for v in w.cget('values') or ()]
        if cls == 'Treeview':
            out += [(where, 'heading', w.heading(c, 'text'))
                    for c in w['columns']]


# Words that keep their capital inside a sentence-case label: names.
NAMES = {'OpenFlo', 'FlowJo', 'Gating', 'PhenoGraph', 'Leiden', 'FlowSOM',
         'CytoNorm', 'AnnData', 'Markdown', 'Gaussian', 'Prism', 'Column',
         'Grouped', 'TriMap', 'PaCMAP', 'Python', 'Louvain', 'Excel',
         'Okabe', 'Ito'}

# Spellings and names the labels must not use (UI 26, 27, 29).
BANNED = (
    (re.compile(r'[Cc]olour'), "British spelling 'colour'"),
    (re.compile(r'[Oo]ptimis'), "British spelling 'optimise'"),
    (re.compile(r'FlowJo-style'), "describes a feature by another product"),
    (re.compile(r'\bPhenograph\b'), "PhenoGraph's name misspelled"),
    (re.compile(r'\b(TSNE|TRIMAP|PACMAP)\b'), "a method's name in capitals"),
    (re.compile(r'^(phenograph|leiden|flowsom|light|dark|midnight)$'),
     "a settings key shown as a choice"),
    (re.compile(r'^(k|res|max ev|seed)$'), "a settings key as a label"),
    (re.compile(r'\.\.\.'), "three dots instead of an ellipsis"),
)


def problems_in_texts(texts):
    """Arrows used as words, banned spellings and names, and Title Case in
    menus, titles and buttons; one line each."""
    out = []
    for where, kind, t in texts:
        one = t.replace('\n', ' ')
        if '→' in t:
            out.append(f"{where} {kind} {one!r}: arrow used as a word")
        for pat, why in BANNED:
            if pat.search(t):
                out.append(f"{where} {kind} {one!r}: {why}")
        if kind in ('menu', 'title', 'TButton', 'TCheckbutton',
                    'TRadiobutton'):
            head = one.split(' — ')[0]
            words = [x for x in re.split(r"[\s/(),:&+-]+", head) if x]
            words = [x for x in words if any(ch.isalpha() for ch in x)]
            for x in words[1:]:
                core = x.strip('…."\'')
                if (core[:1].isupper() and not core.isupper()
                        and not any(ch.isdigit() for ch in core)
                        and core not in NAMES):
                    out.append(f"{where} {kind} {one!r}: {core!r} in Title "
                               "Case")
    return out


def test_labels_use_one_wording(editor):
    root, ed = editor
    texts = [('editor', 'title', ed.title())]
    _texts(ed, texts, 'editor')
    for label, win in _open_windows(root, ed):
        texts.append((label, 'title', win.title()))
        _texts(win, texts, label)
    assert len(texts) > 500
    found = problems_in_texts(texts)
    assert not found, "\n".join(sorted(set(found)))


def test_the_wording_check_finds_each_old_label():
    old = [('editor', 'menu', 'Export → FlowJo .wsp…'),
           ('editor', 'menu', 'Open Recent'),
           ('editor', 'title', 'Compensation Matrix'),
           ('x', 'TRadiobutton', 'Phenograph'),
           ('x', 'TCheckbutton', 'TSNE'),
           ('x', 'choice', 'phenograph'),
           ('x', 'TLabel', 'max ev'),
           ('x', 'menu', 'Colours…'),
           ('x', 'TButton', 'Run optimisation'),
           ('x', 'TButton', 'Browse...'),
           ('x', 'TLabel', 'FlowJo-style stats table')]
    found = problems_in_texts(old)
    for _w, _k, t in old:
        assert any(repr(t.replace('\n', ' ')) in f for f in found), t
    assert not problems_in_texts([
        ('editor', 'menu', 'Export FlowJo workspace…'),
        ('x', 'TButton', 'Prism Column…'),
        ('x', 'title', 'Compensation matrix'),
        ('x', 'choice', 'PhenoGraph')])


def test_side_panel_buttons_say_what_they_do(editor):
    _root, ed = editor
    labels = {str(w.cget('text')): w for w in sweep._walk(ed)
              if w.winfo_class() == 'TButton'}
    for want in ('Delete gate', 'Delete all', 'Copy gates…', 'Populations…',
                 'Undo', 'Redo', 'Fit', 'Pipeline', 'Layout…'):
        assert want in labels, want
    for old in ('Clear', 'Clear all', 'Pops', 'Copy', '↶', '↷', '⌂'):
        assert old not in labels, old
    tips = ed.__dict__.get('_tip_texts', {})
    for w in sweep._walk(ed):            # a symbol alone needs its tip
        if w.winfo_class() != 'TButton':
            continue
        text = str(w.cget('text'))
        if text and not any(ch.isalpha() for ch in text):
            assert str(w) in tips, f"{text!r} ({w}) has no tooltip"


# ── standardise_dialog on windows built for the test ───────────────────────

@pytest.fixture
def root():
    try:
        r = tk.Tk()
    except tk.TclError as exc:
        gui_unavailable(f"Tk cannot start: {exc}")
    r.withdraw()
    yield r
    r.destroy()


def _dialog(root, owner_cls=tk.Toplevel):
    owner = owner_cls(root)
    win = tk.Toplevel(owner)
    ran = []
    bar = ttk.Frame(win)
    bar.pack(side='bottom')
    ttk.Button(bar, text="Cancel", command=win.destroy).pack(side='right')
    ok = ttk.Button(bar, text="Apply", default='active',
                    command=lambda: ran.append('apply'))
    ok.pack(side='right')
    entry = ttk.Entry(win)
    entry.pack()
    text = tk.Text(win, height=2)
    text.pack()
    return owner, win, ran, ok, entry, text


def test_standardise_attaches_and_binds_the_keys(root):
    from openflo.tool_window import standardise_dialog
    owner, win, ran, ok, entry, text = _dialog(root)
    standardise_dialog(win)
    root.update()
    assert str(win.wm_transient()) == str(owner)
    entry.focus_force()
    root.update()
    entry.event_generate('<Return>')
    root.update()
    assert ran == ['apply']
    text.focus_force()
    root.update()
    text.event_generate('<Return>')          # a new line, not the action
    root.update()
    assert ran == ['apply']
    ok.state(['disabled'])
    entry.focus_force()
    entry.event_generate('<Return>')
    root.update()
    assert ran == ['apply']                  # a disabled action doesn't run
    entry.event_generate('<Escape>')
    root.update()
    assert not win.winfo_exists()


def test_named_choice_shows_names_and_keeps_keys(root):
    from openflo.tool_window import CLUSTER_METHOD_NAMES, named_choice
    var = tk.StringVar(master=root, value='phenograph')
    combo = named_choice(root, var, CLUSTER_METHOD_NAMES)
    assert list(combo.cget('values')) == ['PhenoGraph', 'Leiden', 'FlowSOM']
    assert combo.get() == 'PhenoGraph'
    combo.set('FlowSOM')                     # picked from the list
    assert var.get() == 'flowsom'
    var.set('leiden')                        # set by code (a loaded recipe)
    assert combo.get() == 'Leiden'
    var.set('custom')                        # unknown key shows as itself
    assert combo.get() == 'custom' and var.get() == 'custom'


def test_escape_asks_a_window_that_asks_before_closing(root):
    from openflo.tool_window import standardise_dialog
    _owner, win, *_ = _dialog(root)
    asked = []
    win.protocol('WM_DELETE_WINDOW', lambda: asked.append(1))
    standardise_dialog(win)
    win.focus_force()
    root.update()
    win.event_generate('<Escape>')
    root.update()
    assert asked == [1] and win.winfo_exists()


def test_work_windows_and_root_windows_are_left_alone(root):
    from openflo.tool_window import standardise_dialog
    work = tk.Toplevel(root)
    work.independent = True                  # type: ignore[attr-defined]
    child = tk.Toplevel(work)
    child.independent = True                 # type: ignore[attr-defined]
    standardise_dialog(child)
    assert not str(child.wm_transient())
    assert not child.bind('<Escape>')
    other_editor = tk.Toplevel(root)          # made on the hidden root
    standardise_dialog(other_editor)
    assert not str(other_editor.wm_transient())


def test_a_window_keeps_its_own_escape(root):
    from openflo.tool_window import standardise_dialog
    _owner, win, *_ = _dialog(root)
    own = []
    win.bind('<Escape>', lambda _e: own.append(1))
    standardise_dialog(win)
    win.focus_force()
    root.update()
    win.event_generate('<Escape>')
    root.update()
    assert own == [1] and win.winfo_exists()


# ── Errors in plain words ──────────────────────────────────────────────────

def _raised(exc):
    try:
        raise exc
    except BaseException as e:               # noqa: BLE001
        return e


def test_plain_error_names_no_exception_types():
    import errno

    from openflo.ui_errors import plain_error
    cases = {
        FileNotFoundError(2, 'No such file', 'synthetic_1.fcs'):
            'could not be found (synthetic_1.fcs)',
        PermissionError(13, 'Denied', 'out.csv'): 'open in another program',
        MemoryError(): 'not enough memory',
        OSError(errno.ENOSPC, 'No space left on device'): 'disk is full',
        KeyError('FL9-A'): 'was not found: FL9-A',
        ValueError('the percentile must be between 0 and 100'):
            'The percentile must be between 0 and 100.',
        RuntimeError(''): 'did not expect',
        UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'bad'): 'not be the kind',
    }
    for exc, want in cases.items():
        text = plain_error(exc)
        assert want in text, (exc, text)
        assert type(exc).__name__ not in text, text
        assert 'Errno' not in text, text


def test_error_dialog_hides_details_until_asked(root):
    from openflo.ui_errors import ErrorDialog, error_details, error_message
    exc = _raised(KeyError('FL9-A'))
    msg = error_message("The statistics could not be exported.", exc,
                        'synthetic.csv')
    assert msg.startswith("The statistics could not be exported.")
    assert 'File: synthetic.csv' in msg and 'KeyError' not in msg
    details = error_details(exc)
    assert details.startswith('Traceback') and "KeyError: 'FL9-A'" in details
    dlg = ErrorDialog(root, "Export failed", msg, details)
    root.update()
    assert str(dlg.message_label.cget('text')) == msg
    assert not dlg.details_shown
    dlg.details_button.invoke()
    root.update()
    assert dlg.details_shown
    assert str(dlg.details_button.cget('text')) == "Hide details"
    assert "KeyError: 'FL9-A'" in dlg._box.get('1.0', 'end')
    dlg.copy_button.invoke()
    assert root.clipboard_get() == details
    dlg.details_button.invoke()
    root.update()
    assert not dlg.details_shown
    dlg.focus_force()
    root.update()
    dlg.event_generate('<Escape>')
    root.update()
    assert not dlg.winfo_exists()


def test_show_error_reports_through_the_dialog(monkeypatch):
    from openflo import ui_errors
    seen = []
    monkeypatch.setattr(ui_errors, '_present',
                        lambda parent, title, message, details:
                        seen.append((title, message, details)))
    exc = _raised(PermissionError(13, 'Denied', 'synthetic.csv'))
    ui_errors.show_error(None, "Export failed",
                         "The table could not be exported.", exc,
                         'synthetic.csv')
    (title, message, details), = seen
    assert title == "Export failed"
    assert 'PermissionError' not in message
    assert 'open in another program' in message
    assert 'PermissionError' in details


def _error_calls():
    """(file:line, source) of every message box call in the package."""
    out = []
    for f in sorted(SRC.glob('*.py')):
        src = f.read_text(encoding='utf-8')
        for n in ast.walk(ast.parse(src)):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr in ('showerror', 'showwarning',
                                        'showinfo')):
                out.append((f"{f.name}:{n.lineno}",
                            ast.get_source_segment(src, n) or ''))
    return out


def raw_exception_boxes(calls):
    return [where for where, seg in calls
            if re.search(r'type\(\w+\)\.__name__', seg)]


def test_no_message_box_shows_a_raw_exception():
    calls = _error_calls()
    assert len(calls) > 50
    assert not raw_exception_boxes(calls)
    # and the check finds the old form
    assert raw_exception_boxes([('x.py:1', 'messagebox.showerror("Export '
                                 'failed", f"{type(exc).__name__}: {exc}")')])


def test_an_uncaught_error_opens_the_error_window_once(editor):
    root, ed = editor
    from openflo.ui_errors import ErrorDialog
    shown_log = ed._show_log_var.get()
    for _ in range(2):
        exc = _raised(ZeroDivisionError('division by zero'))
        ed._report_callback_exception(type(exc), exc, exc.__traceback__)
    root.update()
    dialogs = [w for w in sweep._walk(root) if isinstance(w, ErrorDialog)]
    assert len(dialogs) == 1
    assert 'did not expect' in str(dialogs[0].message_label.cget('text'))
    assert 'ZeroDivisionError' in dialogs[0].details
    assert ed._show_log_var.get() == shown_log    # the console stays shut
    assert 'Something went wrong' in ed.status_var.get()


# ── Shortcuts and About ────────────────────────────────────────────────────

_KEYSYMS = {'Ctrl': 'Control', ',': 'comma', '`': 'grave', '+': 'plus',
            '-': 'minus', 'Esc': 'Escape'}


def _sequence(keys):
    """'Ctrl+Shift+S' -> '<Control-Shift-Key-S>', the form Tk reports."""
    parts = keys.split('+') if keys != 'Ctrl++' else ['Ctrl', '+']
    mods, key = parts[:-1], _KEYSYMS.get(parts[-1], parts[-1])
    if len(key) == 1 and key.isalpha() and 'Shift' not in mods:
        key = key.lower()
    return '<' + '-'.join([_KEYSYMS.get(m, m) for m in mods]
                          + [f'Key-{key}']) + '>'


def test_every_listed_shortcut_is_bound(editor):
    from openflo.editor_help import SHORTCUTS
    _root, ed = editor
    bound = set(ed.bind()) | set(ed.bind_all()) | set(ed.gate_tv.bind())
    listed = [keys for _s, items in SHORTCUTS for keys, _a in items]
    assert len(listed) >= 25
    missing = [k for k in listed if _sequence(k) not in bound]
    assert not missing, (missing, sorted(bound))


def test_the_shortcut_window_lists_them_in_two_columns(editor):
    from openflo.editor_help import SHORTCUTS, ShortcutsWindow
    root, ed = editor
    win = ShortcutsWindow(ed)
    root.update()
    rows = [win.tree.item(i, 'values') for i in win.tree.get_children()]
    assert all(len(r) == 2 for r in rows)
    assert len(rows) == sum(len(items) + 1 for _s, items in SHORTCUTS)
    assert ('   Ctrl+S', 'Save session') in [tuple(r) for r in rows]
    assert str(win.wm_transient()) == str(ed)
    win.focus_force()
    root.update()
    win.event_generate('<Escape>')
    root.update()
    assert not win.winfo_exists()


def test_about_has_version_licence_citation_and_links(editor, monkeypatch):
    from openflo import __version__
    from openflo.editor_help import CITATION, AboutWindow
    from openflo.update import DOCS_URL
    root, ed = editor
    opened = []
    monkeypatch.setattr('webbrowser.open', opened.append)
    win = AboutWindow(ed)
    root.update()
    texts = []
    _texts(win, texts, 'about')
    flat = ' '.join(t for _w, _k, t in texts)
    assert f"Version {__version__}" in flat and 'MIT' in flat
    assert 'please cite' in flat
    buttons = {t: w for w in sweep._walk(win) if w.winfo_class() == 'TButton'
               for t in [str(w.cget('text'))]}
    buttons['Copy citation'].invoke()
    assert root.clipboard_get() == CITATION
    buttons['Documentation'].invoke()
    assert opened == [DOCS_URL]


# ── The title ───────────────────────────────────────────────────────────────

def test_title_names_the_session_and_marks_unsaved_changes(
        editor, tmp_path, monkeypatch):
    root, ed = editor
    import openflo.editor_session as es
    # The sweep's editor starts with gates added: open from a clean state.
    ed._note_saved()
    assert ed.title() == "Untitled session — OpenFlo"
    ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A', 'op': '>',
                  'value': 100.0, 'name': 'FL1 pos'})
    root.update()
    assert ed.title() == "*Untitled session — OpenFlo"
    path = tmp_path / 'synthetic_run.flowsession'
    monkeypatch.setattr(ed, '_write_session', lambda p: {'samples': []})
    monkeypatch.setattr(es.filedialog, 'asksaveasfilename',
                        lambda **k: str(path))
    ed._save_session(save_as=True)
    assert ed.title() == "synthetic_run.flowsession — OpenFlo"
    ed._undo()
    assert ed.title() == "*synthetic_run.flowsession — OpenFlo"
    ed._note_saved()
    ed._remove_samples(['beta_1'])
    assert ed.title() == "*synthetic_run.flowsession — OpenFlo"
