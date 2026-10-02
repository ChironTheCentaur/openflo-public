"""apply_theme() applies a coherent dark/light chrome without error, and no
near-white surface survives in the midnight theme (the recurring-bug guard)."""
from __future__ import annotations

import os
import sys

import pytest

_SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'scripts')


def test_apply_theme_sets_a_theme_and_density(monkeypatch):
    monkeypatch.setenv('MPLBACKEND', 'Agg')
    tk = pytest.importorskip('tkinter')
    try:
        root = tk.Tk()
    except Exception:
        pytest.skip('no Tk display')
    root.withdraw()
    try:
        from tkinter import ttk

        from openflo.gui import apply_theme
        apply_theme(root)
        style = ttk.Style(root)
        # 'sleek' is the built-in clam theme hand-styled to the palette.
        assert style.theme_use() == 'clam'
        assert str(style.lookup('Treeview', 'rowheight')) == '26'
    finally:
        root.destroy()


def test_apply_theme_switches_palette(monkeypatch):
    monkeypatch.setenv('MPLBACKEND', 'Agg')
    tk = pytest.importorskip('tkinter')
    try:
        root = tk.Tk()
    except Exception:
        pytest.skip('no Tk display')
    root.withdraw()
    try:
        from openflo.gui import apply_theme, current_palette
        pal = apply_theme(root, 'dark')
        assert current_palette() is pal
        dark_bg = pal['bg']
        light_bg = apply_theme(root, 'light')['bg']
        assert dark_bg != light_bg                 # the two chromes differ
        # Unknown mode falls back to light, never raises.
        assert apply_theme(root, 'nope')['bg'] == light_bg
        # Midnight shares dark chrome but darkens the plot canvas.
        assert apply_theme(root, 'midnight')['bg'] == dark_bg
    finally:
        root.destroy()


def test_no_white_chrome_in_midnight(monkeypatch):
    """The check that was missed ~6 times: build every widget under midnight and
    confirm nothing near-white survives. Uses scripts/theme_audit.audit_offenders."""
    monkeypatch.setenv('MPLBACKEND', 'Agg')
    tk = pytest.importorskip('tkinter')
    try:
        tk.Tk().destroy()
    except Exception:
        pytest.skip('no Tk display')
    if _SCRIPTS not in sys.path:
        sys.path.insert(0, _SCRIPTS)
    import theme_audit
    offenders = theme_audit.audit_offenders('midnight')
    assert offenders == [], (
        'near-white chrome in midnight:\n'
        + '\n'.join(f'  [{c}] {p} bg={bg} lum={lm}' for p, c, bg, lm in offenders))


def test_theme_pref_round_trips(tmp_path, monkeypatch):
    """The theme choice persists via the prefs file (no Tk needed)."""
    import openflo.prefs as prefs
    monkeypatch.setattr(prefs, '_prefs_path', lambda: str(tmp_path / 'prefs.json'))
    prefs.write_pref('theme', 'dark')
    assert prefs.read_prefs().get('theme') == 'dark'
    prefs.write_pref('theme', 'light')
    assert prefs.read_prefs().get('theme') == 'light'
