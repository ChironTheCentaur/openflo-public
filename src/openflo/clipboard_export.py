"""Copy a table or a figure to the system clipboard, to paste into Excel,
Prism, PowerPoint or a document.

A table goes as tab-separated text through Tk's own clipboard, which every
platform supports. Tk carries text only, so a figure goes through the
platform: on Windows as a device-independent bitmap (what Office and Paint
paste) plus PNG, on macOS through ``osascript``, and on Linux through
``wl-copy`` or ``xclip``. The figure is rendered as the export would be (white
background, editable-text settings, export DPI), not as the screen shows it.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tempfile


def table_text(lines, decimal=None):
    """``lines`` (rows of cells, the first the headings) as tab-separated
    text. NaN and None become empty cells; a tab or line break inside a cell
    becomes a space, so it cannot split the row. Real numbers use
    ``decimal`` as their decimal mark; None follows the CSV style
    preference, so a European Excel reads a pasted 0,5 as a number."""
    if decimal is None:
        from .table_export import csv_format
        decimal = csv_format()[1]

    def cell(v):
        if v is None or (isinstance(v, float) and v != v):
            return ''
        if isinstance(v, float) and decimal != '.':
            return repr(float(v)).replace('.', decimal)
        return (str(v).replace('\t', ' ').replace('\r\n', ' ')
                .replace('\n', ' ').replace('\r', ' '))
    return '\n'.join('\t'.join(cell(v) for v in row) for row in lines) + '\n'


def copy_table(widget, lines):
    """Put ``lines`` on the clipboard as tab-separated text."""
    widget.clipboard_clear()
    widget.clipboard_append(table_text(lines))


def figure_png(fig, dpi=None):
    """``fig`` rendered as PNG bytes, as an export would save it."""
    from .theme import savefig_background
    buf = io.BytesIO()
    savefig_background(fig, buf, background='White', dpi=dpi, fmt='png')
    return buf.getvalue()


def png_to_dib(png):
    """PNG bytes as a Windows CF_DIB payload: the BMP file without its
    14-byte file header (BITMAPINFOHEADER and pixels, 24-bit, bottom-up)."""
    from PIL import Image
    img = Image.open(io.BytesIO(png)).convert('RGB')
    out = io.BytesIO()
    img.save(out, format='BMP')
    return out.getvalue()[14:]


def copy_figure(fig, dpi=None):
    """Copy ``fig`` to the clipboard as an image. Returns ``(ok, message)``;
    the message says what was copied or why it could not be."""
    png = figure_png(fig, dpi=dpi)
    try:
        if sys.platform == 'win32':
            _copy_windows(png)
        elif sys.platform == 'darwin':
            _copy_macos(png)
        else:
            _copy_linux(png)
    except Exception as exc:                    # noqa: BLE001
        return False, f"Could not copy the figure: {exc}"
    return True, "Copied the figure to the clipboard."


def _copy_windows(png):                         # pragma: no cover - Windows
    if sys.platform != 'win32':          # also tells the type checker
        raise OSError("the Windows clipboard needs Windows")
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL('user32', use_last_error=True)
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    # Handles are pointer-sized: without these, 64-bit Python truncates
    # them to 32 bits and the clipboard receives garbage.
    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE
    user32.RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]
    user32.RegisterClipboardFormatW.restype = wintypes.UINT
    gmem_moveable, cf_dib = 0x0002, 8

    def _put(fmt, data):
        h = kernel32.GlobalAlloc(gmem_moveable, len(data))
        if not h:
            raise OSError(ctypes.get_last_error(), "GlobalAlloc failed")
        p = kernel32.GlobalLock(h)
        ctypes.memmove(p, data, len(data))
        kernel32.GlobalUnlock(h)
        if not user32.SetClipboardData(fmt, h):
            kernel32.GlobalFree(h)
            raise OSError(ctypes.get_last_error(), "SetClipboardData failed")

    import time
    for _ in range(10):                 # another program may hold it briefly
        if user32.OpenClipboard(None):
            break
        time.sleep(0.05)
    else:
        raise OSError(ctypes.get_last_error(), "the clipboard is in use")
    try:
        user32.EmptyClipboard()
        _put(cf_dib, png_to_dib(png))
        # Office and browsers prefer PNG, which keeps the exact pixels.
        _put(user32.RegisterClipboardFormatW('PNG'), png)
    finally:
        user32.CloseClipboard()


def _copy_macos(png):                           # pragma: no cover - macOS
    path = _temp_png(png)
    try:
        subprocess.run(
            ['osascript', '-e',
             f'set the clipboard to (read (POSIX file "{path}") '
             'as «class PNGf»)'],
            check=True, capture_output=True, timeout=20)
    finally:
        _remove(path)


def linux_command():
    """The command that puts a PNG read from stdin on the clipboard, or None
    when neither ``wl-copy`` (Wayland) nor ``xclip`` (X11) is installed."""
    if os.environ.get('WAYLAND_DISPLAY') and shutil.which('wl-copy'):
        return ['wl-copy', '--type', 'image/png']
    if shutil.which('xclip'):
        return ['xclip', '-selection', 'clipboard', '-t', 'image/png', '-i']
    return None


def _copy_linux(png):
    cmd = linux_command()
    if cmd is None:
        raise OSError("install xclip (X11) or wl-clipboard (Wayland), or "
                      "save the figure instead")
    # Both tools fork into the background to serve the clipboard, so this
    # returns once they have read the image.
    # Their output is not captured: the background copy keeps any pipe
    # open, and reading one would wait for it to exit.
    subprocess.run(cmd, input=png, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, timeout=20)


def _temp_png(png):
    fd, path = tempfile.mkstemp(suffix='.png', prefix='openflo_copy_')
    with os.fdopen(fd, 'wb') as f:
        f.write(png)
    return path


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass
