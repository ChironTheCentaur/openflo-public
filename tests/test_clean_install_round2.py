"""Two things a clean install on Windows turned up.

* Analyze > Compare embeddings opened at a fixed 420x360 and packed its Run
  and Cancel row last, so with larger text (a 150 % Windows display) the row
  was squeezed to half height and the cell-count box cut short. The window
  now takes the size its content asks for, with the button row packed first.
* The menu sweep clicked Help > Online documentation, which opened the real
  browser; the browser then wrote a profile into the test home. Every test
  now gets a stand-in ``webbrowser.open`` (conftest ``_no_real_browser``).
* On a busy Windows PC a window could be fitted and then put back where
  Windows first placed it, off the bottom of a laptop screen. The editor
  now checks the place again shortly after (``keep_on_screen``).
"""
from __future__ import annotations

import os
import tkinter as tk
import tkinter.font as tkfont
from types import SimpleNamespace

import pytest

from tests.conftest import BROWSER_OPENS, gui_unavailable

os.environ.setdefault('MPLBACKEND', 'Agg')


def test_tests_cannot_open_the_real_browser():
    import webbrowser
    n = len(BROWSER_OPENS)
    webbrowser.open('https://example.invalid/docs')
    webbrowser.open_new_tab('https://example.invalid/tab')
    assert BROWSER_OPENS[n:] == ['https://example.invalid/docs',
                                 'https://example.invalid/tab']


def test_online_documentation_goes_to_the_stand_in():
    """The menu command the sweep clicks reaches the stand-in, not a browser."""
    from openflo.editor_help import HelpMixin
    from openflo.update import DOCS_URL
    ed = SimpleNamespace(status_var=SimpleNamespace(set=lambda _t: None))
    n = len(BROWSER_OPENS)
    HelpMixin._open_documentation(ed)
    assert BROWSER_OPENS[n:] == [DOCS_URL]


@pytest.fixture
def big_text_root():
    """A root whose text is about 1.6x the default size: what a 150 % Windows
    display with Windows' wider fonts gives the dialog."""
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        gui_unavailable(f"Tk cannot start: {exc}")
    root.withdraw()
    base = float(root.tk.call('tk', 'scaling'))
    root.tk.call('tk', 'scaling', base * 1.5)
    for name in ('TkDefaultFont', 'TkTextFont', 'TkHeadingFont'):
        f = tkfont.nametofont(name, root=root)
        size = int(f.cget('size')) or 9      # negative = pixels; keeps sign
        f.configure(size=int(round(size * 1.6)))
    tkfont.Font(root=root, name='OpenFloHeading', exists=False,
                size=14, weight='bold')
    yield root
    # Tk keeps the scaling per display, not per interpreter: left at 1.5x it
    # widens every later window in this process.
    root.tk.call('tk', 'scaling', base)
    root.destroy()


def _shown_controls_problems(top):
    from tests.test_windows_nothing_hidden import problems
    return problems(top)


def test_compare_embeddings_shows_run_and_cancel_with_big_text(big_text_root,
                                                              monkeypatch):
    from openflo import ui_embedding, warmup
    monkeypatch.setattr(warmup, 'start', lambda *a, **k: None)
    editor = tk.Toplevel(big_text_root)
    editor.status_var = tk.StringVar(master=editor)
    dlg = ui_embedding.EmbeddingDialog(editor, 'alpha_1', 50_000,
                                       ('umap', 'tsne'), None, [])
    dlg.update_idletasks()
    dlg.update()
    found = _shown_controls_problems(dlg)
    assert not found, "\n".join(found)
    names = {w.cget('text') for w in dlg.winfo_children()[0].winfo_children()
             for w in ([w] + list(w.winfo_children()))
             if w.winfo_class() == 'TButton'}
    assert {'Run', 'Cancel'} <= names


def test_window_the_system_moves_after_the_fit_comes_back_on_screen(
        monkeypatch):
    """Under a busy 4-worker run on Windows, Plate view was fitted and then
    put back where Windows first placed it, running past the bottom of a
    1366x768 screen. The editor now checks the place again shortly after."""
    from openflo import tool_window
    from tests import test_windows_nothing_hidden as sweep
    from tests.conftest import skip_unless_the_screen_holds
    monkeypatch.setattr(tk.Misc, 'winfo_screenwidth', lambda self: 1366)
    monkeypatch.setattr(tk.Misc, 'winfo_screenheight', lambda self: 768)
    real_fit = tool_window.fit_window

    def fit_then_system_moves(win, *a, **k):
        out = real_fit(win, *a, **k)
        win.geometry('750x591+286+286')       # the place Windows chose
        return out
    monkeypatch.setattr(tool_window, 'fit_window', fit_then_system_moves)
    root = sweep._root()
    try:
        skip_unless_the_screen_holds(root, 1366, 768)
        ed = sweep._editor(root)
        win = tk.Toplevel(ed)
        tk.Label(win, text='a window').pack()
        sweep._settle(root, ms=600)
        assert win.winfo_y() + win.winfo_height() <= 768, win.winfo_geometry()
        assert win.winfo_x() + win.winfo_width() <= 1366
    finally:
        root.destroy()
