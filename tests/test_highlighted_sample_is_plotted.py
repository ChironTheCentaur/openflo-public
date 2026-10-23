"""The highlighted sample is always one on the plot.

Gates, compensation and clustering act on the highlighted (active) sample,
while the plot shows the ticked ones. After a load, the first sample was
plotted but the last one to arrive was highlighted; a gate drawn with "all
shown" then went to both, and compensation went to a sample not on screen.
Clicking a checkbox also moved the highlight onto the row being unticked.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')

from tests.test_gui_smoke import _editor_or_skip  # noqa: E402

NAMES = ('s1', 's2', 's3')


def _sample(name, n=2_000):
    rng = np.random.default_rng(int(name[1:]))
    return SimpleNamespace(
        name=name, path=f'C:/d/{name}.fcs',
        data=pd.DataFrame({'FSC-A': rng.normal(1000.0, 200.0, n),
                           'FL1-A': rng.normal(500.0, 100.0, n)}),
        channel_labels={'FSC-A': 'FSC-A', 'FL1-A': 'FL1-A'},
        fluor_channels=['FL1-A'], scatter_channels=['FSC-A'])


@pytest.fixture
def editor():
    root, ed, _gui = _editor_or_skip()
    for name in NAMES:
        ed._on_loaded(name, _sample(name))
    ed._on_load_settled()
    ed.update()
    yield ed
    try:
        root.destroy()
    except Exception:
        pass


def _plotted(ed):
    return [n for n in NAMES if ed._sample_plot_enabled.get(n, True)]


def _row_glyph(ed, name):
    return ed.gate_tv.item(ed._sample_iid(name), 'values')[0]


def _click_checkbox(ed, name):
    """A press and release on ``name``'s ☑ column, through the handlers the
    tree binds (the row is located by name: the hidden test window has no
    geometry to hit-test)."""
    tv = ed.gate_tv
    tv.identify_row = lambda _y: ed._sample_iid(name)
    tv.identify_column = lambda _x: '#1'
    try:
        ev = SimpleNamespace(x=0, y=0, state=0)
        out = ed._on_tv_press(ev)
        ed._on_tv_release(ev)
        ed.update()
    finally:
        del tv.identify_row, tv.identify_column
    return out


def test_after_load_the_highlight_is_the_plotted_sample(editor):
    assert _plotted(editor) == ['s1']
    assert editor._active_sample == 's1'
    assert editor.gate_tv.selection() == (editor._sample_iid('s1'),)


def test_highlighting_an_unticked_sample_ticks_it(editor):
    editor.gate_tv.selection_set(editor._sample_iid('s3'))
    editor.update()
    assert editor._active_sample == 's3'
    assert 's3' in _plotted(editor)
    assert _row_glyph(editor, 's3') == '☑'


def test_a_new_gate_goes_only_to_what_is_on_screen(editor):
    editor.gate_tv.selection_set(editor._sample_iid('s3'))
    editor.update()
    assert set(editor._target_samples('enabled')) == set(_plotted(editor))
    assert editor._active_sample in editor._target_samples('enabled')


def test_ticking_a_checkbox_leaves_the_highlight(editor):
    assert _click_checkbox(editor, 's2') == 'break'
    assert _plotted(editor) == ['s1', 's2']
    assert editor._active_sample == 's1'
    assert editor.gate_tv.selection() == (editor._sample_iid('s1'),)


def test_unticking_the_highlighted_sample_moves_the_highlight(editor):
    _click_checkbox(editor, 's2')
    _click_checkbox(editor, 's1')
    assert _plotted(editor) == ['s2']
    assert editor._active_sample == 's2'
    assert editor.gate_tv.selection() == (editor._sample_iid('s2'),)
