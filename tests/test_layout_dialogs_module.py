"""The layout editor's dialogs live in ui_layout_dialogs under public names;
ui_layout_editor no longer defines them, and its old private names still
work with a DeprecationWarning. Synthetic samples and generic channels only.
"""
from __future__ import annotations

import importlib

import pytest  # noqa: E402

from openflo import layout_page as lp  # noqa: E402
from tests.test_layout_editor import _editor_or_skip, _load  # noqa: E402

MOVED = {'_ItemDialog': 'ItemDialog', '_ColorsDialog': 'ColorsDialog',
         '_LayerColorsDialog': 'LayerColorsDialog'}


def _modules():
    pytest.importorskip('tkinter')
    return (importlib.import_module('openflo.ui_layout_editor'),
            importlib.import_module('openflo.ui_layout_dialogs'))


@pytest.mark.parametrize('old, new', sorted(MOVED.items()))
def test_the_old_editor_name_is_a_deprecated_alias(old, new):
    ule, uld = _modules()
    with pytest.warns(DeprecationWarning, match=f'ui_layout_dialogs.{new}'):
        alias = getattr(ule, old)
    assert alias is getattr(uld, new)


@pytest.mark.parametrize('old, new', sorted(MOVED.items()))
def test_the_editor_module_no_longer_defines_the_dialogs(old, new):
    ule, uld = _modules()
    assert old not in vars(ule)
    # The window imports what it opens; the class is ui_layout_dialogs' own.
    assert vars(ule).get(new, vars(uld)[new]) is vars(uld)[new]
    assert [n for n, v in vars(ule).items() if isinstance(v, type)
            and v.__module__ == ule.__name__] == ['LayoutEditorWindow']
    cls = vars(uld)[new]
    assert cls.__module__ == 'openflo.ui_layout_dialogs'
    assert cls.__name__ == new


def test_an_unknown_editor_name_is_still_an_attribute_error():
    ule, _uld = _modules()
    with pytest.raises(AttributeError):
        ule._NoSuchDialog                                 # noqa: B018


def test_the_dialogs_open_and_ok_from_their_own_module(monkeypatch):
    root, ed = _editor_or_skip()
    try:
        import openflo.prefs as prefs
        from openflo import ui_layout_dialogs as uld
        from openflo.ui_layout_editor import LayoutEditorWindow
        store = {}
        monkeypatch.setattr(prefs, 'read_prefs', lambda: dict(store))
        monkeypatch.setattr(prefs, 'write_pref',
                            lambda k, v: store.__setitem__(k, v))
        _load(ed, 'sample_01', 1, 300, 300.0)
        win = LayoutEditorWindow(ed)
        win.update()
        # An item's settings: edited, then OK hands a new item to on_ok.
        got = []
        dlg = uld.ItemDialog(win, lp.text_item('a', size=12), win._choices(),
                             on_ok=got.append)
        assert dlg.title() == 'Text'
        dlg.text_var.set('Run {sample}')
        dlg.size_var.set('14')
        dlg._ok()
        assert [(g['kind'], g['text'], g['size']) for g in got] == [
            ('text', 'Run {sample}', 14)]
        assert not dlg.winfo_exists()
        # The default colours: Save stores the list and calls on_ok.
        saved = []
        dlg = uld.ColorsDialog(win, on_ok=lambda: saved.append(1))
        assert dlg.title() == 'Layout colors'
        dlg.colors.append('#123456')
        dlg._save()
        assert store['layout_colors'][-1] == '#123456' and saved == [1]
        # One item's layer colours: OK hands {layer id: colour} to on_ok.
        picked = []
        dlg = uld.LayerColorsDialog(
            win, [('L1', 'sample_01 · All events', '#1f77b4')],
            {'L1': '#00aa00'}, picked.append)
        assert dlg.title() == 'Layer colors'
        dlg._unset('L1')
        dlg._ok()
        assert picked == [{}]
        win.destroy()
    finally:
        root.destroy()
