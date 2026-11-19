"""Windows use named fonts (Heading, Small, Note, Mono, ...) of 9 pt or
more that follow the window scaling, never a ("TkDefaultFont", 8) tuple.

Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by a
mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import ast

import tests._sample_editor as shared
from tests._editor_ui import SRC

editor = shared.editor
virtual_screen = shared.virtual_screen
no_dialogs = shared.no_dialogs
engines_known = shared.engines_known


def test_named_fonts_exist_follow_the_chrome_and_stay_readable(editor):
    import tkinter.font as tkfont

    from openflo.theme import FONT_MIN, NAMED_FONTS, ensure_fonts
    root, ed = editor
    for size in (7, 10, 12):
        ensure_fonts(root, size)
        sizes = {n: abs(int(tkfont.nametofont(n, root=root).actual('size')))
                 for n in NAMED_FONTS}
        assert min(sizes.values()) >= FONT_MIN, (size, sizes)
        assert sizes['OpenFloHeading'] == max(FONT_MIN, size)
        assert tkfont.nametofont('OpenFloHeading',
                                 root=root).actual('weight') == 'bold'
    base = tkfont.nametofont('TkDefaultFont', root=root).actual('family')
    assert tkfont.nametofont('OpenFloSmall',
                             root=root).actual('family') == base
    # The window-width scaling resizes them with the rest of the chrome.
    ed._apply_chrome_scale(force=True)
    assert abs(int(tkfont.nametofont('OpenFloLarge', root=root)
                   .actual('size'))) == ed._chrome_font_size + 1


_FAMILIES = {'TkDefaultFont', 'TkFixedFont', 'TkHeadingFont', 'TkTextFont',
             'Consolas', 'Segoe UI', 'Arial', 'Courier'}


def _font_tuples(tree):
    for node in ast.walk(tree):
        if (isinstance(node, ast.Tuple) and len(node.elts) >= 2
                and isinstance(node.elts[0], ast.Constant)
                and node.elts[0].value in _FAMILIES
                and isinstance(node.elts[1], ast.Constant)
                and isinstance(node.elts[1].value, int)):
            yield node


def test_no_window_asks_for_a_font_tuple():
    found = []
    for path in sorted(SRC.glob('*.py')):
        if path.name == 'theme.py':
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        found += [f"{path.name}:{n.lineno}" for n in _font_tuples(tree)]
    assert not found, found


def test_the_check_finds_a_font_tuple():
    tree = ast.parse("ttk.Label(f, font=('TkDefaultFont', 8, 'bold'))")
    assert len(list(_font_tuples(tree))) == 1
