"""apply_theme() applies the clean ttk style without error (Tk-guarded)."""
from __future__ import annotations

import pytest


def test_apply_theme_sets_clam_and_density(monkeypatch):
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
        assert style.theme_use() == 'clam'
        # roomier Treeview rows + padded headings were configured
        assert str(style.lookup('Treeview', 'rowheight')) == '24'
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
        from openflo.gui import THEMES, apply_theme, current_palette
        pal = apply_theme(root, 'dark')
        assert pal is THEMES['dark']
        assert current_palette()['bg'] == THEMES['dark']['bg']
        apply_theme(root, 'light')
        assert current_palette()['bg'] == THEMES['light']['bg']
        # Unknown mode falls back to light, never raises.
        assert apply_theme(root, 'nope') is THEMES['light']
    finally:
        root.destroy()


def test_theme_pref_round_trips(tmp_path, monkeypatch):
    """The theme choice persists via the prefs file (no Tk needed)."""
    import openflo.gui as g
    monkeypatch.setattr(g, '_prefs_path', lambda: str(tmp_path / 'prefs.json'))
    g.write_pref('theme', 'dark')
    assert g.read_prefs().get('theme') == 'dark'
    g.write_pref('theme', 'light')
    assert g.read_prefs().get('theme') == 'light'
