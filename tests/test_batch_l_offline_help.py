"""Batch L (Complaints 27): the user guide ships with OpenFlo.

Help > User guide (F1) opens the bundled guide and tutorial with no
connection; Help > Online documentation still opens the web page. Every
menu path the guide names is a real menu item, so the guide cannot drift
from the menus unnoticed.

Proved to fail on the old behaviour by
a mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import os
import re

import pytest

import tests.test_windows_nothing_hidden as sweep
from openflo import user_guide as ug

os.environ.setdefault('MPLBACKEND', 'Agg')


def test_every_topic_ships_with_a_title():
    items = ug.topics()
    assert len(items) >= 10
    slugs = [s for s, _t, _x in items]
    assert slugs[0] == '01-getting-started' and slugs == sorted(slugs)
    for slug, title, text in items:
        assert text.startswith('# ') and title and title != slug + '.md'
        assert len(text) > 400, slug


def test_the_guide_is_package_data():
    text = open('pyproject.toml', encoding='utf-8').read()
    line = next(ln for ln in text.splitlines() if ln.startswith('openflo = ['))
    assert '"guide/*.md"' in line


def test_markdown_blocks_and_inline_runs():
    text = ("# Title\n\nFirst line\nsame paragraph with **bold** and `code`."
            "\n\n## Part\n\n- a **Menu > Item** bullet\n1. a step\n")
    got = ug.blocks(text)
    assert [k for k, _r in got] == ['h1', 'para', 'h2', 'bullet', 'step']
    assert got[1][1] == [('text', 'First line same paragraph with '),
                         ('bold', 'bold'), ('text', ' and '),
                         ('code', 'code'), ('text', '.')]
    assert got[3][1][1] == ('bold', 'Menu > Item')
    assert ''.join(t for _k, t in got[4][1]) == '1. a step'
    assert ug.plain(text).startswith('Title First line same paragraph')
    assert ug.inline('a ** b') == [('text', 'a ** b')]
    assert ug.menu_paths("see **Tools > Bead tracking…** and **bold**") == \
        [['Tools', 'Bead tracking…']]


def test_search_finds_topics_and_ranks_titles_first():
    items = [('a', 'Gating', 'Draw a gate here. Then more.'),
             ('b', 'Other', 'Nothing about it. A gate is mentioned. End.'),
             ('c', 'Third', 'No match at all.')]
    hits = ug.search('gate', items)
    assert [h[0] for h in hits] == ['a', 'b']
    assert hits[1][2] == 'A gate is mentioned.'
    assert ug.search('gate mentioned', items)[0][0] == 'b'
    assert ug.search('   ', items) == [] and ug.search('zzz', items) == []
    real = ug.search('westgard')
    assert real and real[0][0] == '08-quality-control'


# ── In the editor ──────────────────────────────────────────────────────────

virtual_screen = sweep.virtual_screen
no_dialogs = sweep.no_dialogs
engines_known = sweep.engines_known


@pytest.fixture
def editor(virtual_screen, no_dialogs, engines_known):
    virtual_screen(1920, 1080)
    root = sweep._root(1.0)
    try:
        yield root, sweep._editor(root)
    finally:
        root.destroy()


def _menu_tree(ed):
    """``{(top, ..., label)}`` of every menu entry, cascades followed."""
    import tkinter as tk
    out = set()

    def walk(menu, path):
        end = menu.index('end')
        for i in range(0 if end is None else end + 1):
            kind = menu.type(i)
            if kind in ('separator', 'tearoff'):
                continue
            label = menu.entrycget(i, 'label')
            out.add(path + (label,))
            if kind == 'cascade':
                walk(menu.nametowidget(menu.entrycget(i, 'menu')),
                     path + (label,))
    for b in ed._menubar_buttons:
        m = b.cget('menu')
        if m:
            walk(b.nametowidget(m), (str(b.cget('text')).strip(),))
    assert out and all(isinstance(p, tuple) for p in out)
    del tk
    return out


def test_every_menu_path_in_the_guide_is_a_real_menu_item(editor):
    _root, ed = editor
    tree = _menu_tree(ed)
    named = [(slug, tuple(p)) for slug, _t, text in ug.topics()
             for p in ug.menu_paths(text)]
    assert len(named) > 60
    missing = [(slug, ' > '.join(p)) for slug, p in named if p not in tree]
    assert not missing, missing


def test_the_guide_mentions_every_analyze_and_tools_window(editor):
    _root, ed = editor
    named = {tuple(p) for _s, _t, text in ug.topics()
             for p in ug.menu_paths(text)}
    for path in _menu_tree(ed):
        if path[0] in ('Analyze', 'Tools') and len(path) == 2 and \
                path[1].endswith('…'):
            assert path in named, path


def test_help_menu_offers_the_guide_and_the_web_page(editor, monkeypatch):
    import webbrowser
    root, ed = editor
    tree = _menu_tree(ed)
    assert ('Help', 'User guide') in tree
    assert ('Help', 'Online documentation') in tree
    opened = []
    monkeypatch.setattr(webbrowser, 'open', lambda url: opened.append(url))
    ed._open_documentation()
    assert opened and opened[0].startswith('https://')


def test_guide_window_shows_searches_and_offers_the_example(editor,
                                                           monkeypatch):
    from openflo.ui_guide import GuideWindow
    root, ed = editor
    loaded = []
    monkeypatch.setattr(type(ed), '_load_example_data',
                        lambda self: loaded.append(1))
    win = GuideWindow(ed)
    root.update()
    assert win.current == '01-getting-started'
    shown = win.rendered()
    assert shown.startswith('Getting started')
    assert '**' not in shown and '`' not in shown
    assert 'File > Load example dataset' in shown
    assert win.example_btn.winfo_ismapped()
    win.example_btn.invoke()
    assert loaded == [1]
    assert win.topic_list.size() == len(ug.topics())
    win.query_var.set('levey jennings')
    root.update()
    assert win.shown == ['08-quality-control']
    assert 'Levey-Jennings' in win.match_label.cget('text') or \
        'Levey' in win.match_label.cget('text')
    # Key events go to the focused widget (Windows redirects them there),
    # so give the box focus as typing in it would.
    win.search_entry.focus_force()
    root.update()
    win.search_entry.event_generate('<Return>')      # Enter opens it
    root.update()
    assert win.current == '08-quality-control'
    assert not win.example_btn.winfo_ismapped()
    assert re.search(r'•  \S', win.rendered())
    win.query_var.set('qqqqzz')
    assert win.topic_list.size() == 0
    assert win.match_label.cget('text') == "No topic matches."
    win.query_var.set('')
    assert win.topic_list.size() == len(ug.topics())
    win.topic_list.selection_clear(0, 'end')
    win.topic_list.selection_set(2)
    win._picked()
    assert win.current == ug.topics()[2][0]
    win.destroy()
    win = GuideWindow(ed, topic='07-spectral')
    assert win.rendered().startswith('Spectral cytometry')
    win.destroy()
