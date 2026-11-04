"""Copy to the clipboard (complaint 18): tables as tab-separated text with
safe cells, figures as PNG (a bitmap on Windows), the right clipboard tool
on Linux, a plain message when nothing can copy, and Copy plot in the menu
and on a figure window's right-click menu.

Generated data, synthetic sample names and generic channels only.
"""
from __future__ import annotations

import io
import struct
import subprocess
import sys

import numpy as np
import pytest

import tests._inline_editor as shared
from openflo import clipboard_export, export_stamp
from tests._inline_editor import _figure, _register, _sample

editor = shared.editor
prefs = shared.prefs


def test_table_text_is_tab_separated_with_safe_cells():
    text = clipboard_export.table_text(
        [['Sample', 'Population', 'Median FL1-A (linear)'],
         ['synthetic_1', 'A\tB', 12.5],
         ['synthetic_2', 'line\nbreak', float('nan')],
         ['synthetic_3', None, 'NA']])
    lines = text.splitlines()
    assert lines[0] == 'Sample\tPopulation\tMedian FL1-A (linear)'
    assert lines[1] == 'synthetic_1\tA B\t12.5'
    assert lines[2] == 'synthetic_2\tline break\t'
    assert lines[3] == 'synthetic_3\t\tNA'
    assert len(lines) == 4


def test_png_becomes_a_windows_bitmap():
    from PIL import Image
    img = Image.new('RGB', (5, 3), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    dib = clipboard_export.png_to_dib(buf.getvalue())
    size, w, h, planes, bits = struct.unpack('<IiiHH', dib[:16])
    assert (size, w, h, planes, bits) == (40, 5, 3, 1, 24)
    stride = (5 * 3 + 3) // 4 * 4
    assert len(dib) == 40 + stride * 3
    # Bottom-up BGR: the first pixel is blue 0, green 0, red 255.
    assert dib[40:43] == bytes([0, 0, 255])


def test_linux_picks_the_clipboard_tool_that_is_there(monkeypatch):
    have = set()
    monkeypatch.setattr(clipboard_export.shutil, 'which',
                        lambda t: f'/usr/bin/{t}' if t in have else None)
    monkeypatch.delenv('WAYLAND_DISPLAY', raising=False)
    assert clipboard_export.linux_command() is None
    have.add('xclip')
    assert clipboard_export.linux_command()[0] == 'xclip'
    have.add('wl-copy')
    assert clipboard_export.linux_command()[0] == 'xclip'   # X11 session
    monkeypatch.setenv('WAYLAND_DISPLAY', 'wayland-0')
    assert clipboard_export.linux_command()[0] == 'wl-copy'


def test_copy_figure_sends_a_png_and_says_when_it_cannot(monkeypatch, prefs):
    calls = []

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0)

    # clipboard_export.subprocess is the shared module, so the fake also sees
    # the git call the PNG's version stamp makes the first time a process
    # asks for it. Read the (cached) version first, so the test does not
    # depend on whether an earlier test in this worker already did.
    export_stamp.version()
    monkeypatch.setattr(clipboard_export.sys, 'platform', 'linux')
    monkeypatch.setattr(clipboard_export.subprocess, 'run', fake_run)
    monkeypatch.delenv('WAYLAND_DISPLAY', raising=False)
    monkeypatch.setattr(clipboard_export.shutil, 'which', lambda t: None)
    ok, msg = clipboard_export.copy_figure(_figure())
    assert not ok and 'xclip' in msg and not calls
    monkeypatch.setattr(clipboard_export.shutil, 'which',
                        lambda t: '/usr/bin/xclip' if t == 'xclip' else None)
    ok, msg = clipboard_export.copy_figure(_figure())
    assert ok, msg
    cmd, kw = calls[-1]
    assert cmd[0] == 'xclip' and 'image/png' in cmd
    assert kw['input'][:8] == b'\x89PNG\r\n\x1a\n'
    # Not captured: xclip keeps serving the clipboard in the background,
    # and waiting on its pipes would hang.
    assert kw.get('stdout') == subprocess.DEVNULL
    assert not kw.get('capture_output')


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows clipboard')
def test_copy_figure_reaches_the_windows_clipboard(prefs):  # pragma: no cover
    import ctypes
    user32 = ctypes.windll.user32
    ok, msg = clipboard_export.copy_figure(_figure())
    assert ok, msg
    assert user32.IsClipboardFormatAvailable(8)              # CF_DIB
    assert user32.IsClipboardFormatAvailable(
        user32.RegisterClipboardFormatW('PNG'))


def test_copy_plot_menu_item_copies_the_main_figure(editor, monkeypatch):
    seen = []
    monkeypatch.setattr(clipboard_export, 'copy_figure',
                        lambda fig, dpi=None: (seen.append(fig)
                                               or (True, 'Copied.')))
    editor._copy_plot_image()
    assert seen == []                       # nothing loaded: nothing to copy
    _register(editor, _sample('synthetic_1', np.arange(10.0)), {})
    editor._copy_plot_image()
    assert seen == [editor.fig]
    assert editor.status_var.get() == 'Copied.'


def test_figure_windows_offer_copy_on_right_click(editor, monkeypatch):
    import tkinter as tk

    from openflo.tool_window import add_copy_menu, figure_panel
    top = tk.Toplevel(editor)
    _frame, fig, canvas = figure_panel(top, dark=False)
    seen = []
    monkeypatch.setattr(clipboard_export, 'copy_figure',
                        lambda f, dpi=None: (seen.append(f) or (True, '')))
    canvas.get_tk_widget().copy_figure()       # the panel wired it up
    assert seen == [fig]
    add_copy_menu(top, fig)()
    assert seen == [fig, fig]
    top.destroy()
