"""Statistics: a parent and its child no longer read alike.

The Population column showed each gate's full path ('FL4 pos/FL2 pos') in a
220-pixel column, so a parent and child began with the same text and were
cut off before they differed (GUI smoke test, item 7). It now shows the
gate's own name, indented one step per ancestor, in a column as wide as its
longest entry; the CSV export keeps the full path. Other columns are as wide
as their headings ('Median FL2-A (linear)' was cut to 'Median FL').

Generated data: three made-up samples on generic channels, a gate and a
child gate drawn on them.
"""
from __future__ import annotations

import csv
import os
import tkinter as tk
import tkinter.font as tkfont
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.conftest import gui_unavailable

os.environ.setdefault('MPLBACKEND', 'Agg')

COLS = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL4-A']
LONG = 'a gate name longer than the old column'


@pytest.fixture
def window():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        gui_unavailable(f"Tk cannot start: {exc}")
    root.withdraw()
    import importlib
    gui = importlib.import_module('openflo.gui')
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    rng = np.random.default_rng(11)
    for nm in ('sample_a', 'sample_b'):
        df = pd.DataFrame({c: rng.random(500) * 50000 for c in COLS})
        ed._samples[nm] = SimpleNamespace(
            name=nm, path=f'{nm}.fcs', data=df, fluor_channels=COLS[2:],
            raw=df.copy(), channel_labels={c: c for c in COLS})
        ed._sample_order.append(nm)
        ed._sample_trial[nm] = 'Day1'
        ed._sample_plot_enabled[nm] = True
        ed._sample_gates[nm] = {}
        ed._sample_gate_order[nm] = []
        ed._sample_gate_seq[nm] = 0
        ed._active_sample = nm
        ed._gates = ed._sample_gates[nm]
        p = ed._add_gate({'kind': 'threshold', 'channel': 'FL4-A', 'op': '>',
                          'value': 20000.0, 'name': 'FL4 pos'}, audit=False)
        c = ed._add_gate({'kind': 'threshold', 'channel': 'FL2-A', 'op': '>',
                          'value': 20000.0, 'name': 'FL2 pos'}, parent_id=p,
                         audit=False)
        ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A', 'op': '>',
                      'value': 10000.0, 'name': LONG}, parent_id=c,
                     audit=False)
    ed._channels = list(COLS)
    ed._channel_labels = {c: c for c in COLS}
    from openflo.ui_statistics import StatisticsWindow
    win = StatisticsWindow(ed)
    win.withdraw()
    # The table fills in off the Tk thread; fill it here instead.
    want = win._selected_stats()
    rows, cols = ed._stats_rows_from_snapshot(ed._stats_snapshot(want), want)
    win._populate(rows, cols, curated=False)
    yield win, ed
    root.destroy()


def _cells(win, col):
    i = list(win.tv['columns']).index(col)
    return [win.tv.item(r, 'values')[i] for r in win.tv.get_children()]


def test_a_child_row_does_not_start_like_its_parent(window):
    win, _ed = window
    pops = _cells(win, 'Population')
    parent, child, grandchild = pops[:3]
    assert parent == 'FL4 pos'
    assert child.strip('  ').endswith('FL2 pos')
    assert not child.startswith(parent)          # the path did: 'FL4 pos/…'
    assert grandchild.strip('  ').endswith(LONG)
    # One indent step deeper per generation.
    indent = [len(p) - len(p.lstrip(' ')) for p in pops[:3]]
    assert indent[0] == 0 < indent[1] < indent[2]


def test_the_population_column_is_wide_enough_for_its_names(window):
    win, _ed = window
    font = tkfont.nametofont('TkDefaultFont')
    need = max(font.measure(p) for p in _cells(win, 'Population'))
    width = int(win.tv.column('Population', 'width'))
    assert width >= need
    assert width > 220                              # the old fixed width


def test_statistic_columns_fit_their_headings(window):
    win, _ed = window
    font = tkfont.nametofont('TkDefaultFont')
    for c in win.tv['columns']:
        if c.startswith('Median'):
            heading = win.tv.heading(c, 'text')
            assert heading.endswith('(linear)')
            assert int(win.tv.column(c, 'width')) >= font.measure(heading)


def test_the_csv_keeps_the_full_path(window, tmp_path, monkeypatch):
    win, _ed = window
    out = tmp_path / 'stats.csv'
    from openflo import ui_statistics
    monkeypatch.setattr(ui_statistics.filedialog, 'asksaveasfilename',
                        lambda **k: str(out))
    win._export_csv()
    with open(out, encoding='utf-8') as f:
        pops = [r['Population'] for r in csv.DictReader(f)]
    assert pops[:3] == ['FL4 pos', 'FL4 pos/FL2 pos',
                        f'FL4 pos/FL2 pos/{LONG}']


def test_a_population_not_in_the_editor_shows_its_path(window):
    win, _ed = window
    row = {'Sample': 'sample_a', 'Population': 'from/elsewhere',
           '__gid__': 'no-such-gate'}
    assert win._population_label(row) == 'from/elsewhere'
