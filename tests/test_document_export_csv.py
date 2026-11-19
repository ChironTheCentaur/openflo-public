"""Document.export_stats_csv: the phone's "Export stats" writes the CSV the
editor's Statistics window exports."""
from __future__ import annotations

import csv

import pytest

from openflo.document import Document, DocumentError
from tests.test_document import gated_doc, spill_fcs


def _read(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def test_the_csv_holds_each_populations_counts_and_linear_statistics(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    out = tmp_path / 'out' / 'stats.csv'
    out.parent.mkdir()
    r = doc.export_stats_csv(str(out), channels=['FL1-A'], stats=['Median', 'CV'])
    head, rows = _read(out)
    assert head == ['Sample', 'Population', 'Count', '%Parent', '%Total',
                    'Median FL1-A (linear)', 'CV FL1-A (linear)']
    assert r == {'path': str(out), 'rows': 4, 'columns': 7}
    want = doc.stats(name, channels=['FL1-A'], stats=['Median', 'CV'])
    assert [row[1] for row in rows] == [w['population'] for w in want]
    for row, w in zip(rows, want, strict=True):
        assert row[0] == name
        assert int(row[2]) == w['count']
        assert float(row[5]) == pytest.approx(w['channels']['FL1-A']['median'], rel=1e-12)
        assert float(row[6]) == pytest.approx(w['channels']['FL1-A']['cv'], rel=1e-12)
        if w['pct_parent'] is None:
            assert row[3] == ''          # All events has no parent: blank, not NA
        else:
            assert float(row[3]) == pytest.approx(w['pct_parent'], rel=1e-12)
    assert rows[0][1] == 'All events'
    assert any(row[1].endswith('FL1+') for row in rows)


def test_every_loaded_sample_by_default_and_fluorescence_channels(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    other = doc.open_fcs(spill_fcs(tmp_path / 'b' / 's2.fcs', seed=2))
    out = tmp_path / 'all.csv'
    doc.export_stats_csv(str(out))
    head, rows = _read(out)
    assert {row[0] for row in rows} == {name, other}
    assert [row[0] for row in rows].index(other) > max(
        i for i, row in enumerate(rows) if row[0] == name)
    # The default statistics on each fluorescence channel.
    for st in ('Median', 'Mean', 'GeoMean', 'CV'):
        for ch in ('FL1-A', 'FL2-A', 'FL3-A'):
            assert f'{st} {ch} (linear)' in head
    assert not any(h.startswith(('Median FSC', 'Median SSC')) for h in head)
    # One sample only, when asked.
    doc.export_stats_csv(str(tmp_path / 'one.csv'), [other])
    assert {row[0] for row in _read(tmp_path / 'one.csv')[1]} == {other}


def test_a_population_that_cannot_be_computed_reads_na(tmp_path):
    doc, name, _cells, pos, _mid = gated_doc(tmp_path)
    # A boolean that lost its operand: its region is unknown.
    doc._samples[name].gates[pos] = {**doc._samples[name].gates[pos],
                                     'kind': 'boolean', 'op': 'and',
                                     'operands': ['gone']}
    out = tmp_path / 'na.csv'
    doc.export_stats_csv(str(out), channels=['FL1-A'], stats=['Median'])
    _head, rows = _read(out)
    (row,) = [r for r in rows if r[1].endswith('FL1+')]
    assert row[2:] == ['NA'] * 4
    assert row[0] == name


def test_refusals(tmp_path):
    doc, *_ = gated_doc(tmp_path)
    src = doc.samples()[0]['path']
    with pytest.raises(DocumentError, match='a file this document read'):
        doc.export_stats_csv(src)
    with open(src, 'rb') as f:
        assert f.read(6) == b'FCS3.1'
    with pytest.raises(DocumentError, match='unknown statistic'):
        doc.export_stats_csv(str(tmp_path / 'x.csv'), stats=['Mode'])
    with pytest.raises(DocumentError, match='no loaded sample'):
        doc.export_stats_csv(str(tmp_path / 'x.csv'), ['nope'])
    with pytest.raises(DocumentError, match='no loaded sample to export'):
        Document().export_stats_csv(str(tmp_path / 'x.csv'))


def test_export_stats_csv_through_rpc(tmp_path):
    from openflo import document_rpc as rpc
    r = rpc.DocumentRpc()
    s = r.call('open_fcs', {'path': spill_fcs(tmp_path / 'a' / 's.fcs')})
    out = str(tmp_path / 'r.csv')
    got = r.call('export_stats_csv', {'path': out, 'stats': ['Median']})
    assert got['rows'] == 1 and _read(out)[1][0][0] == s


def test_a_geomean_on_a_workspace_axis_names_that_axis_as_the_editor_does(tmp_path):
    from tests.test_geomean_flowjo_axis import _axis_xml, _fcs, _wsp
    fcs, _ev = _fcs(tmp_path / 'wsp.fcs')
    wsp = _wsp(tmp_path / 'w.wsp', fcs, [_axis_xml('Comp-FL1-A')])
    doc = Document()
    doc.import_wsp(wsp)
    out = tmp_path / 'g.csv'
    doc.export_stats_csv(str(out), channels=['FL1-A', 'FL2-A'], stats=['GeoMean', 'Median'])
    head, _rows = _read(out)
    assert sorted(head[5:]) == ['GeoMean FL1-A (linear, FlowJo axis: logicle W=1 M=4.42)',
                                'GeoMean FL2-A (linear)', 'Median FL1-A (linear)',
                                'Median FL2-A (linear)']


def test_one_heading_rule_for_the_window_and_the_export():
    from openflo.gating import CHANNEL_STAT_ORDER, GUESSED_TAG, stat_column_heading
    rows = [{'__geomean_axis__': {'FL1-A': 'FlowJo axis: log 5 decades'}},
            {'__geomean_axis__': {'FL1-A': 'FlowJo axis: log 4 decades'}}]
    h = lambda c, r=None: stat_column_heading(c, r, CHANNEL_STAT_ORDER)  # noqa: E731
    assert h(f'Median FL1-A {GUESSED_TAG}') == 'Median FL1-A (linear, guessed logicle)'
    assert h('GeoMean FL1-A', rows[:1]) == 'GeoMean FL1-A (linear, FlowJo axis: log 5 decades)'
    assert h('GeoMean FL1-A', rows) == 'GeoMean FL1-A (linear, axis differs by sample)'
    assert h('CV FL1-A', rows[:1]) == 'CV FL1-A (linear)'
    for plain in ('Count', '%Parent', 'Population'):
        assert h(plain) == plain


def test_semicolon_separator_writes_decimal_commas_for_excel(tmp_path):
    doc, *_ = gated_doc(tmp_path)
    comma = tmp_path / 'comma.csv'
    semi = tmp_path / 'semi.csv'
    doc.export_stats_csv(str(comma), channels=['FL1-A'], stats=['Median'])
    r = doc.export_stats_csv(str(semi), channels=['FL1-A'], stats=['Median'],
                             separator=';')
    raw = semi.read_bytes()
    assert raw.startswith(b'\xef\xbb\xbf')       # a byte-order mark: Excel reads UTF-8
    assert not comma.read_bytes().startswith(b'\xef\xbb\xbf')
    with open(semi, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.reader(f, delimiter=';'))
    head, body = rows[0], rows[1:]
    want_head, want = _read(comma)
    assert head == want_head and r['rows'] == len(want) == len(body)
    for got, plain in zip(body, want, strict=True):
        assert got[:3] == plain[:3]               # names and integer counts unchanged
        for g, p in zip(got[3:], plain[3:], strict=True):
            assert g == p.replace('.', ',')       # the same digits, decimal comma
    assert any(',' in c for row in body for c in row[3:])
    assert all('.' not in c for row in body for c in row[3:])


def test_an_unknown_separator_is_refused(tmp_path):
    doc, *_ = gated_doc(tmp_path)
    out = tmp_path / 'x.csv'
    with pytest.raises(DocumentError, match="separator '\\\\t'"):
        doc.export_stats_csv(str(out), separator='\t')
    assert not out.exists()


def test_separator_through_rpc(tmp_path):
    from openflo import document_rpc as rpc
    r = rpc.DocumentRpc()
    r.call('open_fcs', {'path': spill_fcs(tmp_path / 'a' / 's.fcs')})
    out = str(tmp_path / 'r.csv')
    r.call('export_stats_csv', {'path': out, 'stats': ['Median'], 'separator': ';'})
    with open(out, encoding='utf-8-sig') as f:
        assert f.readline().startswith('Sample;Population;Count;')
