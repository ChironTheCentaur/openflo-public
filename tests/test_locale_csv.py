"""Locale-aware CSV export (item 12): with the system style, numbers use the
computer's list separator and decimal mark so the file opens as columns in
Excel; the standard CSV is unchanged whatever the locale.

All data are generated here (synthetic sample names, generic FL channels).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openflo import clipboard_export, table_export


@pytest.fixture
def europe(monkeypatch):
    """This computer uses ';' and ',' (as Excel does across most of
    Europe)."""
    monkeypatch.setattr(table_export, 'system_csv_format',
                        lambda: (';', ','))


def test_format_cell_uses_the_decimal_mark_on_real_numbers_only():
    assert table_export.format_cell(0.25, ',') == '0,25'
    assert table_export.format_cell(np.float64(1234.5), ',') == '1234,5'
    # Integers and text are not numbers with a decimal mark.
    assert table_export.format_cell(1000, ',') == '1000'
    assert table_export.format_cell('FL1-A', ',') == 'FL1-A'
    # Missing reads NA, never 0 or a blank.
    assert table_export.format_cell(float('nan'), ',') == 'NA'
    assert table_export.format_cell(None) == 'NA'
    # Full precision: nothing rounded away on the way to Excel.
    assert table_export.format_cell(1 / 3) == repr(1 / 3)


def test_system_csv_opens_as_columns_in_a_semicolon_locale(tmp_path, europe):
    df = pd.DataFrame({'Sample': ['synthetic_1', 'synthetic_2'],
                       'Count': [120, 80], '%Parent': [12.5, float('nan')]})
    path = tmp_path / 'stats.csv'
    table_export.write_frame_csv(str(path), df, style='system')
    raw = path.read_bytes()
    assert raw.startswith(b'\xef\xbb\xbf')            # BOM, so Excel reads UTF-8
    text = raw.decode('utf-8-sig')
    assert text.splitlines()[0] == 'Sample;Count;%Parent'
    assert text.splitlines()[1] == 'synthetic_1;120;12,5'
    assert text.splitlines()[2] == 'synthetic_2;80;NA'
    back = pd.read_csv(path, sep=';', decimal=',', encoding='utf-8-sig')
    assert back['%Parent'].iloc[0] == 12.5
    assert back['Count'].tolist() == [120, 80]


def test_standard_csv_is_unchanged_whatever_the_locale(tmp_path, europe):
    df = pd.DataFrame({'Sample': ['synthetic_1'], '%Parent': [12.5]})
    path = tmp_path / 'stats.csv'
    table_export.write_frame_csv(str(path), df, style='standard')
    raw = path.read_bytes()
    assert not raw.startswith(b'\xef\xbb\xbf')
    assert raw.decode() == 'Sample,%Parent\r\nsynthetic_1,12.5\r\n'


def test_csv_style_follows_the_preference(monkeypatch, europe):
    monkeypatch.setattr(table_export, 'read_prefs',
                        lambda: {'csv_style': 'system'})
    assert table_export.csv_format() == (';', ',')
    monkeypatch.setattr(table_export, 'read_prefs', dict)
    assert table_export.csv_format() == (',', '.')
    # An unknown value falls back to the standard style.
    monkeypatch.setattr(table_export, 'read_prefs',
                        lambda: {'csv_style': 'bogus'})
    assert table_export.csv_style() == 'standard'
    assert table_export.csv_format() == (',', '.')


def test_system_format_reads_windows_then_the_locale(monkeypatch):
    monkeypatch.setattr(table_export, '_windows_number_format',
                        lambda: ('\t', ','))
    assert table_export.system_csv_format() == ('\t', ',')
    monkeypatch.setattr(table_export, '_windows_number_format', lambda: None)
    monkeypatch.setattr(table_export, '_locale_decimal_mark', lambda: ',')
    assert table_export.system_csv_format() == (';', ',')
    monkeypatch.setattr(table_export, '_locale_decimal_mark', lambda: '.')
    assert table_export.system_csv_format() == (',', '.')


def test_index_column_is_headed_as_pandas_heads_it(tmp_path):
    df = pd.DataFrame({'a': [1.5]}, index=pd.Index(['d1'], name='Day'))
    path = tmp_path / 'g.csv'
    table_export.write_frame_csv(str(path), df, style='standard', index=True)
    assert path.read_text().splitlines() == ['Day,a', 'd1,1.5']


def test_copied_table_uses_the_decimal_mark():
    lines = [['Sample', 'Mean'], ['synthetic_1', 0.5], ['synthetic_2', 7]]
    assert clipboard_export.table_text(lines, decimal=',') == (
        'Sample\tMean\nsynthetic_1\t0,5\nsynthetic_2\t7\n')
    assert clipboard_export.table_text(lines, decimal='.') == (
        'Sample\tMean\nsynthetic_1\t0.5\nsynthetic_2\t7\n')
