"""Layout editor → ＋ Strategy: a page laying out the gating strategy, one
plot per gating step (the parent population on the gate's axes, its gates
drawn), which batch export then draws for every sample.

Generated gate trees on generic channels; the GUI test uses the layout
editor tests' synthetic samples."""
from __future__ import annotations

import pytest

from openflo import layout_page as lp
from openflo.gating import population_keys

TREE = {
    'a': {'kind': 'rect', 'name': 'Cells', 'parent_id': None,
          'x_channel': 'FSC-A', 'y_channel': 'SSC-A'},
    'b': {'kind': 'polygon', 'name': 'Singlets', 'parent_id': 'a',
          'x_channel': 'FSC-A', 'y_channel': 'FSC-H'},
    'c': {'kind': 'threshold', 'name': 'Pos', 'parent_id': 'b',
          'channel': 'FL1-A'},
    'q1': {'kind': 'rect', 'name': 'Q1', 'parent_id': 'b',
           'x_channel': 'FL2-A', 'y_channel': 'FL3-A'},
    'q2': {'kind': 'rect', 'name': 'Q2', 'parent_id': 'b',
           'x_channel': 'FL2-A', 'y_channel': 'FL3-A'},
    'bool': {'kind': 'boolean', 'name': 'Both', 'parent_id': 'b',
             'op': 'and', 'operands': ['c', 'q1']},
}
ORDER = ['a', 'b', 'c', 'q1', 'q2', 'bool']


def test_steps_follow_the_tree_and_share_axes():
    assert lp.gating_strategy_steps(TREE, ORDER) == [
        (None, 'FSC-A', 'SSC-A', ['a']),
        ('a', 'FSC-A', 'FSC-H', ['b']),
        ('b', 'FL1-A', None, ['c']),
        ('b', 'FL2-A', 'FL3-A', ['q1', 'q2']),   # quadrants share a plot
    ]


def test_steps_down_to_a_selected_gate_only():
    assert [s[3] for s in lp.gating_strategy_steps(TREE, ORDER, 'c')] == [
        ['a'], ['b'], ['c']]


def test_page_has_a_title_and_a_plot_per_step_on_the_page():
    keys = population_keys(TREE)
    page = lp.gating_strategy_page(TREE, keys, ORDER)
    items = list(page.items.values())
    assert items[0]['kind'] == 'text' and '{sample}' in items[0]['text']
    plots = [it for it in items if it['kind'] == 'plot']
    assert [(p['populations'], p['x'], p['y'], p['mode'], p['title'])
            for p in plots] == [
        ([lp.ALL_EVENTS], 'FSC-A', 'SSC-A', 'pseudocolor', 'Cells'),
        ([keys['a']], 'FSC-A', 'FSC-H', 'pseudocolor', 'Singlets'),
        ([keys['b']], 'FL1-A', None, 'histogram', 'Pos'),
        ([keys['b']], 'FL2-A', 'FL3-A', 'pseudocolor', 'Q1, Q2'),
    ]
    assert all(p['draw_gates'] for p in plots)
    # Every box on the page, none overlapping.
    boxes = [p['box'] for p in plots]
    for x, y, w, h in boxes:
        assert 0 <= x and 0 <= y and x + w <= 1 + 1e-9 and y + h <= 1 + 1e-9
    for i, (x, y, w, h) in enumerate(boxes):
        for X, Y, W, H in boxes[i + 1:]:
            assert x + w <= X + 1e-9 or X + W <= x + 1e-9 \
                or y + h <= Y + 1e-9 or Y + H <= y + 1e-9


def test_no_gates_gives_just_the_title():
    page = lp.gating_strategy_page({}, {})
    assert [it['kind'] for it in page.items.values()] == ['text']


def test_window_adds_the_page_and_it_resolves_on_every_sample():
    from tests.test_layout_editor import _editor_or_skip, _load
    root, ed = _editor_or_skip()
    try:
        import openflo.ui_layout_editor as ule
        from openflo.layout_render import render_page
        _load(ed, 's1', 1, 800, 300.0)
        _load(ed, 's2', 2, 600, 500.0, extra_gate=True)
        ed._set_active_sample('s1')
        win = ule.LayoutEditorWindow(ed)
        win.update()
        n_pages = len(win.doc.pages)
        win._add_strategy_page()
        assert len(win.doc.pages) == n_pages + 1
        page = win.doc.pages[win.page_idx]
        assert page.name == 'Gating strategy'
        plots = [it for it in page.items.values() if it['kind'] == 'plot']
        assert [(p['x'], p['y']) for p in plots] == [
            ('FSC-A', 'SSC-A'), ('FL1-A', None)]
        for name in ('s1', 's2'):        # gate ids differ; keys match
            fig = render_page(ed, page, name)
            texts = [t.get_text() for ax in fig.axes for t in ax.texts]
            assert not any('no such population' in t for t in texts), texts
        win.destroy()
    finally:
        root.destroy()


@pytest.mark.parametrize('mode', ['dot', 'contour'])
def test_mode_is_used_for_two_axis_steps(mode):
    page = lp.gating_strategy_page(TREE, population_keys(TREE), ORDER,
                                   mode=mode)
    modes = [it['mode'] for it in page.items.values() if it['kind'] == 'plot']
    assert modes == [mode, mode, 'histogram', mode]
