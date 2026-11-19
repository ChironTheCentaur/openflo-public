"""Excel export (item 12): one sheet per statistic in table order, numbers
stay numbers, sheet names are valid and unique, the CLI writes a workbook,
and the Statistics window saves one with the set-aside column.

All data are generated here (synthetic sample names, generic FL channels).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import tests._four_sample_editor as shared
from openflo import table_export

ed = shared.ed


_HEADER = ['Sample', 'Population', 'Count', '%Parent']


# Populations and samples deliberately out of alphabetical order.
_ROWS = [['synthetic_2', 'Singlets', 900, 90.0],
         ['synthetic_2', 'Lymphs', 810, 90.0],
         ['synthetic_1', 'Singlets', 500, 50.0],
         ['synthetic_1', 'Lymphs', 'NA', 'NA'],
         ['synthetic_3', 'Singlets', 300, '']]


def test_statistics_sheets_one_per_statistic_in_table_order():
    sheets = table_export.statistics_sheets(_HEADER, _ROWS, 2)
    assert list(sheets) == ['All statistics', 'Count', '%Parent']
    count = sheets['Count']
    # Populations down the side, samples across, both in table order.
    assert count.columns.tolist() == ['Population', 'synthetic_2',
                                      'synthetic_1', 'synthetic_3']
    assert count['Population'].tolist() == ['Singlets', 'Lymphs']
    assert count.loc[0, 'synthetic_1'] == 500
    assert count.loc[1, 'synthetic_2'] == 810
    # A population that cannot be computed keeps reading NA...
    assert count.loc[1, 'synthetic_1'] == 'NA'
    # ...a blank stays blank, and a missing combination is empty.
    pct = sheets['%Parent']
    assert pd.isna(pct.loc[0, 'synthetic_3'])
    assert pd.isna(pct.loc[1, 'synthetic_3'])


def test_workbook_round_trips_with_numbers_as_numbers(tmp_path):
    path = tmp_path / 'stats.xlsx'
    sheets = table_export.statistics_sheets(_HEADER, _ROWS, 2)
    names = table_export.write_workbook(
        str(path), sheets, na_rep='', about={'Saved by': 'OpenFlo test'})
    assert names == ['All statistics', 'Count', '%Parent', 'About']
    back = pd.read_excel(path, sheet_name=None, keep_default_na=False)
    assert back['Count'].loc[0, 'synthetic_1'] == 500
    assert isinstance(back['Count'].loc[0, 'synthetic_1'], (int, np.integer))
    assert back['Count'].loc[1, 'synthetic_1'] == 'NA'
    assert back['About'].loc[0, 'Value'] == 'OpenFlo test'


def test_sheet_names_are_valid_and_unique():
    assert table_export.sheet_name('Median FL1-A [linear]') == \
        'Median FL1-A _linear_'
    long = 'x' * 40
    a = table_export.sheet_name(long)
    b = table_export.sheet_name(long, [a])
    assert len(a) == 31 and len(b) == 31 and a != b
    assert table_export.sheet_name('Count', ['count']) == 'Count (2)'
    assert table_export.sheet_name('  ') == 'Sheet'


def test_csvs_to_workbook_collects_new_tables_only(tmp_path):
    import time
    out = tmp_path / 'run'
    (out / 'trial_a').mkdir(parents=True)
    (out / 'trial_b').mkdir()
    old = out / 'old.csv'
    old.write_text('a\n1\n')
    os.utime(old, (1_000_000, 1_000_000))
    since = time.time() - 1
    (out / 'trial_a' / 'stats.csv').write_text('Sample,Count\ns1,5\n')
    (out / 'trial_b' / 'stats.csv').write_text('Sample,Count\ns2,7\n')
    big = out / 'events.csv'
    big.write_text('x\n' + '1\n' * 50)
    book = out / 'tables.xlsx'
    inc, skipped = table_export.csvs_to_workbook(
        str(out), str(book), since=since, max_bytes=60)
    assert sorted(inc) == [os.path.join('trial_a', 'stats.csv'),
                           os.path.join('trial_b', 'stats.csv')]
    assert skipped == ['events.csv: too large for a sheet']
    sheets = pd.read_excel(book, sheet_name=None)
    # Two files share a name: each sheet says which folder it came from.
    assert set(sheets) == {'trial_a stats', 'trial_b stats', 'About'}
    assert sheets['trial_b stats'].loc[0, 'Count'] == 7


def test_cli_accepts_xlsx_and_writes_the_workbook(tmp_path, capsys):
    from openflo import cli_main
    from openflo.cli_args import build_parser
    assert build_parser().parse_args(['--xlsx']).xlsx is True
    assert build_parser().parse_args([]).xlsx is False
    (tmp_path / 't.csv').write_text('a,b\n1,2.5\n')
    cli_main.write_tables_workbook(str(tmp_path), 0)
    book = tmp_path / 'openflo_tables.xlsx'
    assert pd.read_excel(book, sheet_name='t').loc[0, 'b'] == 2.5
    assert '1 table(s)' in capsys.readouterr().out


def test_statistics_window_saves_a_workbook_with_set_aside(ed, tmp_path,
                                                           monkeypatch):
    from openflo import ui_statistics
    gid = ed._next_gate_id()
    ed._gates[gid] = {'kind': 'threshold', 'channel': 'FL1-A', 'op': '>',
                      'value': -1e12, 'parent_id': None, 'name': 'All',
                      'enabled': True}
    ed._sample_gate_order['synthetic_a1'].append(gid)
    try:
        ed._set_sample_aside(['synthetic_a1'], 'clogged')
        w = ui_statistics.StatisticsWindow(ed)
        rows = [{'Sample': 'synthetic_a1', 'Population': 'All',
                 'Count': 2000, '%Parent': 100.0},
                {'Sample': 'synthetic_b1', 'Population': 'All',
                 'Count': 1500, '%Parent': 100.0}]
        w._populate(rows, ['Sample', 'Population', 'Count', '%Parent'],
                    False)
        assert w._cols[:3] == ['Sample', 'Population', 'Set aside']
        path = tmp_path / 'stats.xlsx'
        monkeypatch.setattr(ui_statistics.filedialog, 'asksaveasfilename',
                            lambda **k: str(path))
        w._export_xlsx()
        book = pd.read_excel(path, sheet_name=None, keep_default_na=False)
        assert list(book)[:3] == ['All statistics', 'Count', '%Parent']
        assert book['Count'].columns.tolist() == ['Population',
                                                  'synthetic_a1',
                                                  'synthetic_b1']
        assert book['Count'].loc[0, 'synthetic_b1'] == 1500
        assert book['All statistics'].loc[0, 'Set aside'] == 'clogged'
        assert book['All statistics'].loc[1, 'Set aside'] == ''
        # The CSV follows the system style when asked.
        csv_path = tmp_path / 'stats.csv'
        monkeypatch.setattr(ui_statistics.filedialog, 'asksaveasfilename',
                            lambda **k: str(csv_path))
        monkeypatch.setattr(table_export, 'csv_style', lambda: 'system')
        monkeypatch.setattr(table_export, 'system_csv_format',
                            lambda: (';', ','))
        w._export_csv()
        first = csv_path.read_text(encoding='utf-8-sig').splitlines()[1]
        assert first == 'synthetic_a1;All;clogged;2000;100,0'
        w.destroy()
    finally:
        ed._return_sample(['synthetic_a1'])
        ed._gates.pop(gid, None)
        ed._sample_gate_order['synthetic_a1'].remove(gid)
