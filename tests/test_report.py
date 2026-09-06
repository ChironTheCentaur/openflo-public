"""Tests for openflo.report — the self-contained HTML report builder."""
from __future__ import annotations

import base64

import matplotlib
import pandas as pd

matplotlib.use('Agg')
from matplotlib.figure import Figure

from openflo.report import (
    build_html_report,
    df_to_html_table,
    figure_html,
    figure_to_data_uri,
)


def test_figure_to_data_uri_is_embeddable_png():
    fig = Figure()
    fig.add_subplot(111).plot([0, 1], [0, 1])
    uri = figure_to_data_uri(fig)
    assert uri.startswith('data:image/png;base64,')
    raw = base64.b64decode(uri.split(',', 1)[1])
    assert raw[:8] == b'\x89PNG\r\n\x1a\n'          # PNG magic


def test_figure_html_wraps_img():
    fig = Figure()
    fig.add_subplot(111).plot([0, 1], [1, 0])
    tag = figure_html(fig, alt='my plot')
    assert tag.startswith('<img ')
    assert 'alt="my plot"' in tag
    assert 'data:image/png;base64,' in tag


def test_df_to_html_table_escapes_and_blanks_nan():
    df = pd.DataFrame({'Population': ['CD11b+ <gate>'], 'Count': [1234],
                       '%Parent': [float('nan')]})
    htm = df_to_html_table(df)
    assert '<table>' in htm and '<th>Population</th>' in htm
    assert 'CD11b+ &lt;gate&gt;' in htm                # HTML-escaped
    assert '1234' in htm
    # NaN renders as an empty cell, not 'nan'.
    assert 'nan' not in htm.lower()


def test_df_to_html_table_truncates():
    df = pd.DataFrame({'x': list(range(100))})
    htm = df_to_html_table(df, max_rows=10)
    assert 'Showing 10 of 100 rows' in htm
    assert htm.count('<tr>') == 1 + 10               # header + 10 rows


def test_build_html_report_structure_and_escaping():
    fig = Figure()
    fig.add_subplot(111).plot([0, 1], [0, 1])
    sections = [
        {'heading': 'Current plot', 'html': figure_html(fig)},
        {'heading': 'Stats', 'html': '<table><tr><td>x</td></tr></table>'},
    ]
    doc = build_html_report('My <Report>', meta={'version': '1.2', 'n': 3},
                            sections=sections)
    assert doc.startswith('<!DOCTYPE html>')
    assert '<title>My &lt;Report&gt;</title>' in doc   # title escaped
    assert '<h1>My &lt;Report&gt;</h1>' in doc
    assert '<b>version:</b> 1.2' in doc
    assert '<h2>Current plot</h2>' in doc
    assert '<h2>Stats</h2>' in doc
    assert 'data:image/png;base64,' in doc
    assert doc.rstrip().endswith('</html>')


def test_build_html_report_empty():
    doc = build_html_report('Empty')
    assert '<h1>Empty</h1>' in doc
    assert '</html>' in doc
