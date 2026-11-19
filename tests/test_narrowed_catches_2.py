"""Broad ``except Exception`` handlers narrowed to the error they are for
(R11 option B, slice 2). Slice 1 is tests/test_narrowed_catches.py.

Each narrowed handler gets:

* normal-case tests: the failure the handler was written for (a Tk widget or
  font that is gone or missing, tkdnd refusing, a file that will not open or
  write, a marker that is not JSON) is still handled as before;
* a "lets a bug through" test: any other error now propagates instead of
  being swallowed. These fail on the old broad handler and pass on the new.

``Bug`` is a LookupError so that none of the narrowed handlers (Tk, OS,
value, attribute, runtime and encoding errors) can catch it.

Generated data, synthetic names and generic channels only.
"""
from __future__ import annotations

import json
import os
import tkinter as tk
from types import SimpleNamespace

import pytest

from tests.conftest import gui_unavailable


class Bug(LookupError):
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


@pytest.fixture
def noted(monkeypatch):
    """Messages passed to ``note`` by the modules under test."""
    from openflo import editor_clipboard, gui
    seen = []

    def note(msg, exc=None):
        seen.append((msg, type(exc) if exc is not None else None))
    for mod in (gui, editor_clipboard):
        monkeypatch.setattr(mod, 'note', note)
    return seen


# ── gui: Tk callback error handler ───────────────────────────────────────────

def _install(root_obj):
    from openflo.gui import ViewGateEditorWindow
    handler = object()
    me = SimpleNamespace(_root=lambda: root_obj,
                         _report_callback_exception=handler)
    ViewGateEditorWindow._install_callback_error_handler(me)
    return handler


def test_callback_error_handler_is_installed_on_the_root(root):
    handler = _install(root)
    assert root.report_callback_exception is handler


def test_callback_error_handler_that_cannot_be_set_is_noted(noted):
    _install(object())           # takes no new attributes: AttributeError
    assert noted == [("install the error handler", AttributeError)]


def test_callback_error_handler_lets_a_bug_through(noted):
    from openflo.gui import ViewGateEditorWindow
    me = SimpleNamespace(_root=_raise(Bug('root')),
                         _report_callback_exception=None)
    with pytest.raises(Bug):
        ViewGateEditorWindow._install_callback_error_handler(me)


# ── gui: OS drag-and-drop registration ───────────────────────────────────────

def _dnd_editor(drop=None, drag=None):
    calls = []

    def ok(*a):
        calls.append(a[0] if a else None)
    tree = SimpleNamespace()
    if drag is not None:
        tree.drag_source_register = drag
        tree.dnd_bind = ok
    return SimpleNamespace(
        drop_target_register=drop or ok, dnd_bind=ok, _on_dnd_drop=None,
        gate_tv=tree, _on_tree_os_drag_init=None), calls


def _register(monkeypatch, me):
    from openflo import gui
    monkeypatch.setattr(gui, 'HAS_DND', True)
    gui.ViewGateEditorWindow._register_os_dnd(me)


def test_dnd_registration_refused_by_tkdnd_is_noted(monkeypatch, noted):
    me, _ = _dnd_editor(drop=_raise(tk.TclError('no tkdnd')),
                        drag=_raise(tk.TclError('no tkdnd')))
    _register(monkeypatch, me)
    assert noted == [
        ("drag-and-drop: file drop not registered", tk.TclError),
        ("drag-and-drop: drag-source not registered", tk.TclError)]


def test_dnd_registration_without_dnd_methods_is_noted(monkeypatch, noted):
    me, _ = _dnd_editor()        # the tree has no drag_source_register
    del me.drop_target_register
    _register(monkeypatch, me)
    assert noted == [
        ("drag-and-drop: file drop not registered", AttributeError),
        ("drag-and-drop: drag-source not registered", AttributeError)]


def test_dnd_registration_registers_both(monkeypatch, noted):
    me, calls = _dnd_editor(drag=lambda *a: None)
    _register(monkeypatch, me)
    assert noted == [] and '<<Drop>>' in calls and '<<DragInitCmd>>' in calls


def test_dnd_file_drop_registration_lets_a_bug_through(monkeypatch, noted):
    me, _ = _dnd_editor(drop=_raise(Bug('drop')), drag=lambda *a: None)
    with pytest.raises(Bug):
        _register(monkeypatch, me)


def test_dnd_drag_source_registration_lets_a_bug_through(monkeypatch, noted):
    me, _ = _dnd_editor(drag=_raise(Bug('drag')))
    with pytest.raises(Bug):
        _register(monkeypatch, me)


# ── theme.ensure_fonts: a named font not made yet ────────────────────────────

_OPENFLO_FONTS = ('OpenFloHeading', 'OpenFloSmall', 'OpenFloNote',
                  'OpenFloLarge', 'OpenFloTitle', 'OpenFloMono')


def test_ensure_fonts_creates_then_resizes_the_named_fonts(root):
    import tkinter.font as tkfont

    from openflo.theme import ensure_fonts
    for name in _OPENFLO_FONTS:
        assert name not in tkfont.names(root)
    ensure_fonts(root, size=11)            # not there yet: created
    assert all(n in tkfont.names(root) for n in _OPENFLO_FONTS)
    ensure_fonts(root, size=14)            # there: reconfigured
    assert tkfont.nametofont('OpenFloTitle', root=root).cget('size') == 17


def test_ensure_fonts_lets_a_bug_through(root, monkeypatch):
    import tkinter.font as tkfont

    from openflo.theme import ensure_fonts
    real = tkfont.nametofont

    def nametofont(name, root=None):
        if name.startswith('OpenFlo'):
            raise Bug(name)
        return real(name, root=root)
    monkeypatch.setattr(tkfont, 'nametofont', nametofont)
    with pytest.raises(Bug):
        ensure_fonts(root, size=11)


# ── ui_statistics: column width from the table font ──────────────────────────

def test_text_width_measures_the_longest_text(root):
    from openflo.ui_statistics import StatisticsWindow
    narrow = StatisticsWindow._text_width(['a'], 10, 900)
    wide = StatisticsWindow._text_width(['a' * 60], 10, 900)
    assert 10 <= narrow < wide <= 900


@pytest.mark.parametrize('exc', [tk.TclError('no font'),
                                 RuntimeError('no default root')])
def test_text_width_without_a_measurable_font_is_the_minimum(monkeypatch, exc):
    import tkinter.font as tkfont

    from openflo.ui_statistics import StatisticsWindow
    monkeypatch.setattr(tkfont, 'nametofont', _raise(exc))
    assert StatisticsWindow._text_width(['abc'], 40, 300) == 40


def test_text_width_lets_a_bug_through(monkeypatch):
    import tkinter.font as tkfont

    from openflo.ui_statistics import StatisticsWindow
    monkeypatch.setattr(tkfont, 'nametofont', _raise(Bug('font')))
    with pytest.raises(Bug):
        StatisticsWindow._text_width(['abc'], 40, 300)


# ── editor_plot: the kept-zoom key ───────────────────────────────────────────

def _zoom_editor(x_get=lambda: 'CD-A (FL1-A)', mode='dot'):
    return SimpleNamespace(
        x_combo=SimpleNamespace(get=x_get),
        y_combo=SimpleNamespace(get=lambda: 'FL2-A'),
        mode_var=SimpleNamespace(get=lambda: mode),
        _channel_scale={'FL1-A': 'logicle', 'FL2-A': 'linear'})


def test_zoom_key_names_the_view():
    from openflo.editor_plot import PlotMixin
    assert PlotMixin._zoom_key(_zoom_editor()) == (
        'FL1-A', 'FL2-A', 'dot', 'logicle', 'linear')


def test_zoom_key_while_widgets_are_gone_or_not_built_is_none():
    from openflo.editor_plot import PlotMixin
    gone = _zoom_editor(x_get=_raise(tk.TclError('gone')))
    assert PlotMixin._zoom_key(gone) is None
    unbuilt = _zoom_editor()
    del unbuilt.x_combo
    assert PlotMixin._zoom_key(unbuilt) is None


def test_zoom_key_lets_a_bug_through():
    from openflo.editor_plot import PlotMixin
    with pytest.raises(Bug):
        PlotMixin._zoom_key(_zoom_editor(x_get=_raise(Bug('get'))))


# ── ui_notice.report_saved: the widget's window ──────────────────────────────

def test_report_saved_from_a_destroyed_widget_still_says_so(root, tmp_path):
    from openflo.ui_notice import report_saved
    var = tk.StringVar(root)
    w = tk.Frame(root)
    w.status_var = var             # type: ignore[attr-defined]
    w.destroy()                    # winfo_toplevel now raises TclError
    text = report_saved(w, str(tmp_path / 'out.csv'))
    assert 'out.csv' in text and var.get() == text


def test_report_saved_from_a_non_widget_still_says_so(tmp_path):
    from openflo.ui_notice import report_saved
    owner = SimpleNamespace(status_var=SimpleNamespace(set=lambda t: None))
    text = report_saved(owner, str(tmp_path / 'out.csv'))
    assert 'out.csv' in text


def test_report_saved_lets_a_bug_through(tmp_path):
    from openflo.ui_notice import report_saved
    w = SimpleNamespace(winfo_toplevel=_raise(Bug('toplevel')))
    with pytest.raises(Bug):
        report_saved(w, str(tmp_path / 'out.csv'))


# ── tool_window.standardise_dialog: the marker attribute ─────────────────────

class _Sealed:
    """Takes no new attributes, as a slotted window object would."""
    __slots__ = ('independent',)

    def __init__(self):
        self.independent = True


def test_standardise_dialog_on_an_object_without_room_for_the_marker():
    from openflo.tool_window import standardise_dialog
    win = _Sealed()
    standardise_dialog(win)            # AttributeError swallowed, then left
    assert not hasattr(win, '_openflo_standard')


def test_standardise_dialog_marks_the_window():
    from openflo.tool_window import standardise_dialog
    win = SimpleNamespace(independent=True)
    standardise_dialog(win)
    assert win._openflo_standard is True


def test_standardise_dialog_marker_lets_a_bug_through():
    from openflo.tool_window import standardise_dialog

    class Faulty(_Sealed):
        __slots__ = ()

        def __setattr__(self, name, value):
            if name == '_openflo_standard':
                raise Bug(name)
            object.__setattr__(self, name, value)
    with pytest.raises(Bug):
        standardise_dialog(Faulty())


# ── tool_window.figure_panel: dark dialog colours ────────────────────────────

@pytest.fixture
def dark_dialog(monkeypatch):
    from openflo import tool_window
    monkeypatch.setattr(tool_window, 'dialog_dark_on', lambda _p: True)


def _panel(root):
    from openflo.tool_window import figure_panel
    return figure_panel(root, figsize=(2, 2), dpi=50, dark=False)


def test_figure_panel_paints_a_dark_dialog_dark(root, dark_dialog):
    from openflo.theme import THEMES, current_palette
    _frame, fig, canvas = _panel(root)
    from matplotlib.colors import to_hex
    assert to_hex(fig.get_facecolor()) == to_hex(THEMES['midnight']['plot_bg'])
    assert canvas.get_tk_widget().cget('bg') == current_palette()['bg']


def test_figure_panel_with_a_colour_matplotlib_refuses(root, dark_dialog,
                                                       monkeypatch):
    from openflo.theme import THEMES
    monkeypatch.setitem(THEMES['midnight'], 'plot_bg', 'not-a-colour')
    _frame, fig, _canvas = _panel(root)       # ValueError swallowed
    assert fig is not None


def test_figure_panel_with_a_colour_tk_refuses(root, dark_dialog, monkeypatch):
    from openflo import theme
    monkeypatch.setattr(theme, 'current_palette',
                        lambda: {'bg': 'not-a-colour'})
    _frame, _fig, canvas = _panel(root)       # TclError swallowed
    assert canvas.get_tk_widget().cget('bg') != 'not-a-colour'


def test_figure_panel_facecolour_lets_a_bug_through(root, dark_dialog,
                                                    monkeypatch):
    import matplotlib.figure as mfig

    class Faulty(mfig.Figure):
        def set_facecolor(self, color):
            raise Bug(color)
    monkeypatch.setattr(mfig, 'Figure', Faulty)
    with pytest.raises(Bug):
        _panel(root)


def test_figure_panel_background_lets_a_bug_through(root, dark_dialog,
                                                    monkeypatch):
    from openflo import theme

    class Colour:
        def __str__(self):
            raise Bug('bg')
    monkeypatch.setattr(theme, 'current_palette', lambda: {'bg': Colour()})
    with pytest.raises(Bug):
        _panel(root)


def test_figure_panel_palette_lets_a_bug_through(root, dark_dialog,
                                                 monkeypatch):
    from openflo import theme
    monkeypatch.setattr(theme, 'current_palette', _raise(Bug('palette')))
    with pytest.raises(Bug):
        _panel(root)


# ── ui_audit: signed compliance record ───────────────────────────────────────

@pytest.fixture
def signer(monkeypatch):
    """Answers the signature prompts; records dialogs and saved notices."""
    import tkinter.simpledialog as sd

    from openflo import ui_audit
    answers = iter(['Signer A', 'Reviewed'])
    monkeypatch.setattr(sd, 'askstring', lambda *_a, **_k: next(answers))
    seen = SimpleNamespace(shown=[], saved=[])
    monkeypatch.setattr(ui_audit, 'show_error',
                        lambda _p, title, _m, exc, *_a: seen.shown.append(
                            (title, type(exc))))
    monkeypatch.setattr(ui_audit, 'report_saved',
                        lambda _w, path, *_a: seen.saved.append(path))
    return seen


def _sign(monkeypatch, path):
    from openflo import ui_audit
    monkeypatch.setattr(ui_audit.filedialog, 'asksaveasfilename',
                        lambda **_k: str(path))
    me = SimpleNamespace(
        editor=SimpleNamespace(_samples={}, _sample_order=[],
                               _audit=lambda *_a, **_k: None),
        _log=SimpleNamespace(to_list=list))
    ui_audit.AuditWindow._sign_record(me)


def test_signed_record_is_written_with_its_markdown(tmp_path, monkeypatch,
                                                    signer):
    path = tmp_path / 'record.json'
    _sign(monkeypatch, path)
    assert json.loads(path.read_text(encoding='utf-8'))
    assert (tmp_path / 'record.md').read_text(encoding='utf-8')
    assert signer.shown == [] and signer.saved == [str(path)]


def test_signed_record_that_cannot_be_written_is_shown(tmp_path, monkeypatch,
                                                       signer):
    _sign(monkeypatch, tmp_path / 'missing' / 'record.json')
    assert signer.shown == [("Compliance", FileNotFoundError)]
    assert signer.saved == []


def test_signed_record_lets_a_bug_through(tmp_path, monkeypatch, signer):
    from openflo import compliance
    monkeypatch.setattr(compliance, 'record_to_markdown', _raise(Bug('md')))
    with pytest.raises(Bug):
        _sign(monkeypatch, tmp_path / 'record.json')


# ── ui_methods: run manifest save ────────────────────────────────────────────

@pytest.fixture
def methods_save(monkeypatch):
    from openflo import ui_methods
    seen = SimpleNamespace(shown=[], saved=[])
    monkeypatch.setattr(ui_methods, 'show_error',
                        lambda _p, title, _m, exc, *_a: seen.shown.append(
                            (title, type(exc))))
    monkeypatch.setattr(ui_methods, 'report_saved',
                        lambda _w, path, *_a: seen.saved.append(path))

    def save(path):
        monkeypatch.setattr(ui_methods.filedialog, 'asksaveasfilename',
                            lambda **_k: str(path))
        ui_methods.MethodsWindow._save(SimpleNamespace(_manifest='{"n": 1}'))
    seen.save = save
    return seen


def test_manifest_is_saved_and_said(tmp_path, methods_save):
    path = tmp_path / 'manifest.json'
    methods_save.save(path)
    assert path.read_text(encoding='utf-8') == '{"n": 1}'
    assert methods_save.saved == [str(path)] and methods_save.shown == []


def test_manifest_that_cannot_be_saved_is_shown(tmp_path, methods_save):
    methods_save.save(tmp_path / 'missing' / 'manifest.json')
    assert methods_save.shown == [("Save failed", FileNotFoundError)]
    assert methods_save.saved == []


def test_manifest_save_lets_a_bug_through(tmp_path, methods_save, monkeypatch):
    from openflo import ui_methods
    monkeypatch.setattr(ui_methods, 'report_saved', _raise(Bug('notice')))
    with pytest.raises(Bug):
        methods_save.save(tmp_path / 'manifest.json')


# ── ui_spectral_qc: Markdown export ──────────────────────────────────────────

@pytest.fixture
def qc_export(monkeypatch):
    from openflo import ui_spectral_qc
    seen = SimpleNamespace(shown=[], saved=[])
    monkeypatch.setattr(ui_spectral_qc, 'show_error',
                        lambda _p, title, _m, exc, *_a: seen.shown.append(
                            (title, type(exc))))
    monkeypatch.setattr(ui_spectral_qc, 'report_saved',
                        lambda _w, path, *_a: seen.saved.append(path))
    monkeypatch.setattr(ui_spectral_qc, 'record_export', lambda *_a, **_k: None)

    def export(path, markdown):
        monkeypatch.setattr(ui_spectral_qc.filedialog, 'asksaveasfilename',
                            lambda **_k: str(path))
        ui_spectral_qc.SpectralQCWindow._export_md(
            SimpleNamespace(_markdown=markdown))
    seen.export = export
    return seen


def test_spectral_qc_report_is_exported(tmp_path, qc_export):
    path = tmp_path / 'qc.md'
    qc_export.export(path, lambda: '# QC\nFL1-A')
    assert path.read_text(encoding='utf-8') == '# QC\nFL1-A'
    assert qc_export.saved == [str(path)] and qc_export.shown == []


@pytest.mark.parametrize('where,text,kind', [
    ('missing/qc.md', '# QC', FileNotFoundError),
    ('qc.md', '# QC \ud800', UnicodeEncodeError),
])
def test_spectral_qc_report_that_cannot_be_written_is_shown(
        tmp_path, qc_export, where, text, kind):
    qc_export.export(tmp_path / where, lambda: text)
    assert qc_export.shown == [("Spectral QC", kind)]
    assert qc_export.saved == []


def test_spectral_qc_export_lets_a_bug_through(tmp_path, qc_export):
    with pytest.raises(Bug):
        qc_export.export(tmp_path / 'qc.md', _raise(Bug('markdown')))


# ── editor_clipboard: the transfer done-marker ───────────────────────────────

def _marker(tmp_path, content):
    path = tmp_path / 'move.done'
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding='utf-8')
    return str(path)


def test_done_marker_lists_the_paths_taken(tmp_path, noted):
    from openflo.editor_clipboard import _read_done_marker
    path = _marker(tmp_path, json.dumps({'taken': ['a.fcs', 'b.fcs']}))
    assert _read_done_marker(path) == {'a.fcs', 'b.fcs'}
    assert noted == []


@pytest.mark.parametrize('content', ['null', '{}', '{"taken": null}'])
def test_an_empty_done_marker_took_nothing(tmp_path, noted, content):
    from openflo.editor_clipboard import _read_done_marker
    assert _read_done_marker(_marker(tmp_path, content)) == set()
    assert noted == []


@pytest.mark.parametrize('content', [
    '{not json', b'\xff\xfe\x00', '[1, 2]', '{"taken": 5}',
    '{"taken": [["a.fcs"]]}', '{"taken": "a.fcs"}'])
def test_an_unreadable_or_malformed_done_marker_is_unknown(tmp_path, noted,
                                                           content):
    from openflo.editor_clipboard import _read_done_marker
    assert _read_done_marker(_marker(tmp_path, content)) is None
    assert noted == [("could not read the sample transfer marker", None)]


def test_a_missing_done_marker_is_unknown(tmp_path, noted):
    from openflo.editor_clipboard import _read_done_marker
    assert _read_done_marker(str(tmp_path / 'gone.done')) is None
    assert len(noted) == 1


def test_done_marker_read_lets_a_bug_through(tmp_path, monkeypatch, noted):
    from openflo import editor_clipboard
    path = _marker(tmp_path, '{"taken": []}')
    monkeypatch.setattr(editor_clipboard.json, 'load', _raise(Bug('load')))
    with pytest.raises(Bug):
        editor_clipboard._read_done_marker(path)
    assert os.path.exists(path)
