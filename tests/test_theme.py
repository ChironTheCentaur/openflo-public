"""apply_theme() applies a coherent dark/light chrome without error, and no
near-white surface survives in the midnight theme (the recurring-bug guard)."""
from __future__ import annotations

import os
import sys

import pytest

from tests.conftest import gui_unavailable

_SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'scripts')


def test_apply_theme_sets_a_theme_and_density(monkeypatch):
    monkeypatch.setenv('MPLBACKEND', 'Agg')
    tk = pytest.importorskip('tkinter')
    try:
        root = tk.Tk()
    except Exception:  # noqa: BLE001 - any Tk start failure means no GUI
        gui_unavailable('no Tk display')
    root.withdraw()
    try:
        from tkinter import ttk

        from openflo.ui_style import apply_theme
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
    except Exception:  # noqa: BLE001 - any Tk start failure means no GUI
        gui_unavailable('no Tk display')
    root.withdraw()
    try:
        from openflo.gui import current_palette
        from openflo.ui_style import apply_theme
        pal = apply_theme(root, 'dark')
        assert current_palette() is pal
        dark_bg = pal['bg']
        light_bg = apply_theme(root, 'light')['bg']
        assert dark_bg != light_bg                 # the two chromes differ
        # Unknown mode falls back to light, never raises.
        assert apply_theme(root, 'nope')['bg'] == light_bg
        # Midnight shares dark chrome but darkens the plot canvas.
        assert apply_theme(root, 'midnight')['bg'] == dark_bg

        # The palette must reach the widgets, not just come back as a dict:
        # what ttk will paint a frame and a label with follows each switch.
        from tkinter import ttk
        style = ttk.Style(root)
        for mode in ('dark', 'light', 'dark'):
            pal = apply_theme(root, mode)
            assert style.lookup('TFrame', 'background') == pal['bg'], mode
            assert style.lookup('TLabel', 'foreground') == pal['fg'], mode
    finally:
        root.destroy()


def test_no_white_chrome_in_midnight(monkeypatch):
    """The check that was missed ~6 times: build every widget under midnight and
    confirm nothing near-white survives. Uses scripts/theme_audit.audit_offenders."""
    monkeypatch.setenv('MPLBACKEND', 'Agg')
    tk = pytest.importorskip('tkinter')
    try:
        tk.Tk().destroy()
    except Exception:  # noqa: BLE001 - any Tk start failure means no GUI
        gui_unavailable('no Tk display')
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


def test_ui_style_apply_theme_themes_a_root(monkeypatch):
    """The ttk theming lives in openflo.ui_style, and its apply_theme styles a
    root there: clam, the palette's frame colour, the flat indicators."""
    monkeypatch.setenv('MPLBACKEND', 'Agg')
    tk = pytest.importorskip('tkinter')
    import openflo.ui_style as ui_style
    try:
        root = tk.Tk()
    except Exception:  # noqa: BLE001 - any Tk start failure means no GUI
        gui_unavailable('no Tk display')
    root.withdraw()
    # The indicator images are made once per process; another test in this
    # worker may already have made them on a root since destroyed.
    monkeypatch.setattr(ui_style, '_FLAT_IND', {'made': set(), 'imgs': {}})
    try:
        from tkinter import ttk
        pal = ui_style.apply_theme(root, 'dark')
        style = ttk.Style(root)
        assert style.theme_use() == 'clam'
        assert style.lookup('TFrame', 'background') == pal['bg']
        assert 'flatcheck_dark.indicator' in style.element_names()
    finally:
        root.destroy()


# Old name on gui -> the name it has in ui_style.
_THEME_NAMES_MOVED = {
    'apply_theme': 'apply_theme',
    '_apply_sleek': 'apply_sleek',
    '_install_flat_indicators': 'install_flat_indicators',
    '_FLAT_IND': '_FLAT_IND',
}


@pytest.mark.parametrize('old, new', sorted(_THEME_NAMES_MOVED.items()))
def test_old_gui_theme_names_still_work_with_a_warning(old, new):
    """gui's old theming names still resolve, to the very ui_style objects,
    with a DeprecationWarning that names the new home."""
    import openflo.gui as gui
    import openflo.ui_style as ui_style
    with pytest.warns(DeprecationWarning, match=f'ui_style.{new}'):
        assert getattr(gui, old) is getattr(ui_style, new)


@pytest.mark.parametrize('old', sorted(_THEME_NAMES_MOVED))
def test_gui_no_longer_defines_the_theming(old):
    """The theming is defined once, in ui_style, not in gui."""
    import openflo.gui as gui
    assert old not in vars(gui)


def _theme_audit_or_skip(monkeypatch):
    monkeypatch.setenv('MPLBACKEND', 'Agg')
    tk = pytest.importorskip('tkinter')
    try:
        tk.Tk().destroy()
    except Exception:  # noqa: BLE001 - any Tk start failure means no GUI
        gui_unavailable('no Tk display')
    if _SCRIPTS not in sys.path:
        sys.path.insert(0, _SCRIPTS)
    import theme_audit
    return theme_audit


def _boom(*_a, **_k):
    raise RuntimeError('planted failure')


@pytest.mark.parametrize('method, step', [
    ('_set_theme', "live switch to the 'midnight' theme failed"),
    ('_set_active_sample', 'loading the test sample failed'),
])
def test_theme_audit_reports_a_failed_setup_step(monkeypatch, capfd,
                                                method, step):
    """A live theme switch (or the test sample) that raises used to be
    swallowed, and the walk then passed on a window that was not in the theme
    it claimed to check. Now audit_offenders raises AuditSetupError naming the
    step, and the command line exits 2 with that message instead of 'OK'."""
    theme_audit = _theme_audit_or_skip(monkeypatch)
    import openflo.gui as gui
    monkeypatch.setattr(gui.ViewGateEditorWindow, method, _boom)
    rc = theme_audit.main(['--mode', 'midnight'])
    out = capfd.readouterr().out
    assert 'OK' not in out, out
    assert rc == 2 and 'audit could not run' in out and step in out, out
    with pytest.raises(theme_audit.AuditSetupError, match=step) as info:
        theme_audit.audit_offenders('midnight')
    assert 'RuntimeError: planted failure' in str(info.value)
