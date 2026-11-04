"""Index-sort keywords give each sorted cell its plate well, shown in the
plate view.
Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by a
mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import tests._sample_editor as shared
from openflo import index_sort as ix
from tests._sample_editor import _set

editor = shared.editor
virtual_screen = shared.virtual_screen
no_dialogs = shared.no_dialogs
engines_known = shared.engines_known


def test_parse_locations_forms():
    assert ix.parse_locations('0,0;0,1;7,11') == [(0, 0), (0, 1), (7, 11)]
    assert ix.parse_locations('1,1;2,3') == [(0, 0), (1, 2)]
    assert ix.parse_locations('A1;B02;junk;H12') == [(0, 0), (1, 1), None,
                                                    (7, 11)]
    assert ix.parse_locations('nothing here') == []
    assert ix.parse_locations('') == []


def test_event_wells_take_each_events_location():
    import pandas as pd
    data = pd.DataFrame({'FL1-A': [10.0, 20.0, 30.0, 40.0]})
    s = SimpleNamespace(name='sort_1', data=data, metadata={
        'INDEX SORTING LOCATIONS': '0,0;0,1;2,5'})
    assert ix.has_index_sort(s)
    rows, cols = ix.event_wells(s)
    assert list(rows) == [0, 0, 2, -1] and list(cols) == [0, 1, 5, -1]
    assert ix.well_values(s, 'FL1-A') == {(0, 0): 10.0, (0, 1): 20.0,
                                          (2, 5): 30.0}
    s2 = SimpleNamespace(name='plain', data=data, metadata={'$INDEX': 'C3'})
    assert ix.well_values(s2, 'FL1-A') == {(2, 2): 10.0}
    assert not ix.has_index_sort(SimpleNamespace(name='x', data=data,
                                                 metadata={}))


def test_plate_view_shows_index_sorted_cells(editor):
    from openflo.ui_plate import INDEX_PREFIX, PlateWindow
    root, ed = editor
    vals = np.arange(400, dtype=float)
    _set(ed, 'alpha_1', **{'FL1-A': vals})
    ed._samples['alpha_1'].metadata = {
        'INDEX SORTING LOCATIONS': '0,0;0,1;1,0;7,11'}
    win = PlateWindow(ed)
    root.update()
    assert win.index_samples == ['alpha_1']
    win.source_var.set(INDEX_PREFIX + 'alpha_1')
    win._source_changed()
    win.stat_var.set('FL1-A')
    win.draw()
    mat = win.matrix()
    assert mat[0, 0] == 0 and mat[0, 1] == 1 and mat[1, 0] == 2 \
        and mat[7, 11] == 3
    assert np.isfinite(mat).sum() == 4
    assert win.rows()[0] == ('A01', 'alpha_1 event 1', 0.0)
    win.source_var.set('Samples')
    win._source_changed()
    assert win.stat_var.get() == 'Events'
    win.destroy()
