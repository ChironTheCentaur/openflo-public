"""Send to Statistics app: population statistics as a Pocket Lab Table
(stats_plt, Document.export_stats_plt), read back with the format's own
reader, and the vendored copy of the format module kept byte-identical."""
from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime

import numpy as np
import pytest

from openflo import stats_plt
from openflo.document import Document, DocumentError
from openflo.document_rpc import METHODS
from openflo.gating import CHANNEL_STAT_ORDER
from openflo.pocketlab_table import read_plt, sniff_plt
from tests.test_document import gated_doc, spill_fcs

# pocketsuite packages/plt/python/plt.py as of pocketsuite master f45b071
# (last changed in 9009d7f, the version pocketstat's reader holds). Change it there, copy it here unchanged and
# update this hash; never edit the copy.
VENDOR_SHA256 = 'b1257a7e61a1d6c3ea8f27173564c46fcaad3bc2a638086f9f4a06d626e69004'


def _read(path):
    with open(path, 'rb') as f:
        data = f.read()
    table, notices, _extra = read_plt(data)
    return data, table, notices


def _col(table, name):
    return next(c for c in table['columns'] if c['name'] == name)


def _values(table, name):
    k = [c['name'] for c in table['columns']].index(name)
    return [row[k] for row in table['data']]


def test_the_vendored_format_module_is_unchanged():
    here = os.path.join(os.path.dirname(stats_plt.__file__), 'pocketlab_table.py')
    with open(here, 'rb') as f:
        assert hashlib.sha256(f.read()).hexdigest() == VENDOR_SHA256


def test_off_unless_released_or_asked_for(monkeypatch):
    assert stats_plt.RELEASED is False
    assert not stats_plt.send_to_statistics_enabled({})
    assert not stats_plt.send_to_statistics_enabled({stats_plt.ENV_FLAG: '0'})
    assert stats_plt.send_to_statistics_enabled({stats_plt.ENV_FLAG: '1'})
    monkeypatch.setattr(stats_plt, 'RELEASED', True)
    assert stats_plt.send_to_statistics_enabled({})


def test_the_document_writes_a_table_statistics_reads_with_its_roles(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    out = tmp_path / 'stats.plt.csv'
    r = doc.export_stats_plt(str(out), channels=['FL1-A'], stats=['Median', 'CV'])
    data, table, notices = _read(out)
    assert notices == []
    assert sniff_plt(data) and data.startswith(b'\xef\xbb\xbf#plt,1.0\r\n')
    assert r == {'path': str(out), 'rows': 4, 'columns': 7, 'name': 'population_stats.plt.csv'}
    assert table['source']['toolbox'] == 'openflo'
    assert table['source']['analysis'] == 'population-stats'
    assert (table['assay'], table['rows'], table['design']) == ('generic', 'population', 'independent')
    by_name = {c['name']: c for c in table['columns']}
    assert by_name['sample']['role'] == 'id'
    assert by_name['population']['role'] == 'analyte'
    assert 'group' not in by_name          # no workspace groups
    assert (by_name['Count']['unit'], by_name['Count']['scale']) == ('counts', 'count')
    assert (by_name['Parent']['unit'], by_name['Parent']['scale']) == ('%', 'percent')
    assert (by_name['Total']['unit'], by_name['Total']['scale']) == ('%', 'percent')
    assert (by_name['Median_FL1-A']['unit'], by_name['Median_FL1-A']['scale']) == ('AU', 'linear')
    assert (by_name['CV_FL1-A']['unit'], by_name['CV_FL1-A']['scale']) == ('%', 'percent')
    assert by_name['Median_FL1-A']['label'] == 'Median FL1-A (linear)'
    assert by_name['Parent']['label'] == '%Parent'
    # The same numbers as the engine's statistics, in the same order.
    want = doc.stats(name, channels=['FL1-A'], stats=['Median', 'CV'])
    assert _values(table, 'population') == [w['population'] for w in want]
    assert _values(table, 'sample') == [name] * 4
    assert _values(table, 'Count') == [w['count'] for w in want]
    for got, w in zip(_values(table, 'Median_FL1-A'), want, strict=True):
        assert got == pytest.approx(w['channels']['FL1-A']['median'], rel=1e-12)
    for got, w in zip(_values(table, 'CV_FL1-A'), want, strict=True):
        assert got == pytest.approx(w['channels']['FL1-A']['cv'], rel=1e-12)
    assert _values(table, 'Parent')[0] is None      # All events has no parent


def test_workspace_groups_become_the_group_column(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    other = doc.open_fcs(spill_fcs(tmp_path / 'b' / 's2.fcs', seed=2))
    third = doc.open_fcs(spill_fcs(tmp_path / 'c' / 's3.fcs', seed=3))
    doc._samples[name].trial = 'Control'
    doc._samples[other].trial = 'Treated'
    out = tmp_path / 'g.plt.csv'
    doc.export_stats_plt(str(out), channels=['FL1-A'], stats=['Median'])
    _data, table, _notices = _read(out)
    assert _col(table, 'group')['role'] == 'group'
    by_sample = dict(zip(_values(table, 'sample'), _values(table, 'group'), strict=True))
    assert by_sample == {name: 'Control', other: 'Treated', third: 'No group'}
    # One group only: no group column (nothing to compare).
    doc._samples[other].trial = 'Control'
    doc._samples[third].trial = 'Control'
    out2 = tmp_path / 'g2.plt.csv'
    doc.export_stats_plt(str(out2), channels=['FL1-A'], stats=['Median'])
    assert 'group' not in [c['name'] for c in _read(out2)[1]['columns']]


def test_a_population_that_cannot_be_computed_is_missing(tmp_path):
    doc, name, _cells, pos, _mid = gated_doc(tmp_path)
    doc._samples[name].gates[pos] = {**doc._samples[name].gates[pos],
                                     'kind': 'boolean', 'op': 'and',
                                     'operands': ['gone']}
    out = tmp_path / 'na.plt.csv'
    doc.export_stats_plt(str(out), channels=['FL1-A'], stats=['Median'])
    _data, table, _notices = _read(out)
    k = _values(table, 'population').index(
        next(p for p in _values(table, 'population') if p.endswith('FL1+')))
    row = table['data'][k]
    assert row[0] == name
    assert row[2:] == [None] * 4


def test_refusals_and_the_phone_may_call_it(tmp_path):
    doc, *_ = gated_doc(tmp_path)
    assert 'export_stats_plt' in METHODS
    with pytest.raises(DocumentError, match='a file this document read'):
        doc.export_stats_plt(doc.samples()[0]['path'])
    with pytest.raises(DocumentError, match='no loaded sample'):
        Document().export_stats_plt(str(tmp_path / 'none.plt.csv'))


def test_stats_table_names_units_and_missing_values():
    rows = [
        {'Sample': 's1', 'Population': 'All events', 'Count': np.int64(10), '%Parent': float('nan'),
         'Median FL1-A': np.float64(1.5), 'CV FL 2/A': 3.0, 'MFI ratio FL1-A': 2.0, 'Source': 'tab 1'},
        {'Sample': 's1', 'Population': 'A', 'Count': 5, '%Parent': 50.0,
         'Median FL1-A': float('inf'), 'CV FL 2/A': None, 'MFI ratio FL1-A': '', 'Source': 'tab 1',
         '__unknown__': None},
        {'Sample': 's1', 'Population': '=cmd', 'Count': 5, '%Parent': 50.0,
         'Median FL1-A': 2.0, 'CV FL 2/A': 1.0, 'MFI ratio FL1-A': 1.0, 'Source': 'tab 2',
         '__unknown__': 'gone'},
    ]
    cols = ['Sample', 'Population', 'Count', '%Parent', 'Median FL1-A', 'CV FL 2/A',
            'MFI ratio FL1-A', 'Source']
    t = stats_plt.stats_table(rows, cols, {}, CHANNEL_STAT_ORDER, version='3.0.0',
                              created=datetime(2026, 11, 18, 4, 5, 59, tzinfo=UTC))
    assert t['created'] == '2026-11-18T04:05Z'
    assert t['source'] == {'toolbox': 'openflo', 'version': '3.0.0', 'analysis': 'population-stats'}
    assert t['settings'] == [['Channel statistics', 'Computed on linear values']]
    cs = {c['name']: c for c in t['columns']}
    assert list(cs) == ['sample', 'population', 'Count', 'Parent', 'Median_FL1-A', 'CV_FL_2_A',
                        'MFI_ratio_FL1-A', 'Source']
    assert (cs['CV_FL_2_A']['unit'], cs['CV_FL_2_A']['scale']) == ('%', 'percent')
    assert (cs['MFI_ratio_FL1-A']['role'], cs['MFI_ratio_FL1-A']['unit']) == ('measure', '1')
    assert (cs['Source']['role'], cs['Source']['type']) == ('text', 'text')
    assert t['data'][0] == ['s1', 'All events', 10.0, None, 1.5, 3.0, 2.0, 'tab 1']
    assert t['data'][1] == ['s1', 'A', 5.0, 50.0, None, None, None, 'tab 1']
    assert t['data'][2] == ['s1', '=cmd', None, None, None, None, None, 'tab 2']
    # A text cell that would run as a spreadsheet formula travels quoted
    # and comes back as written.
    from openflo.pocketlab_table import write_plt
    text = write_plt(t)
    assert "'=cmd" in text
    back, _n, _e = read_plt(text)
    assert back['data'][2][1] == '=cmd'
    # A version the format cannot hold falls back to '0'.
    assert stats_plt.stats_table(rows, cols, {}, CHANNEL_STAT_ORDER,
                                 version='bad version')['source']['version'] == '0'


def test_identity_text_columns_stay_text_even_when_empty():
    rows = [{'Sample': 's', 'Population': 'All events', 'Count': 3, 'Source': '', 'Set aside': None},
            {'Sample': 's', 'Population': 'A', 'Count': 2, 'Source': '', 'Set aside': None}]
    t = stats_plt.stats_table(rows, ['Sample', 'Population', 'Count', 'Source', 'Set aside'], {},
                              CHANNEL_STAT_ORDER)
    cs = {c['name']: c for c in t['columns']}
    for name in ('Source', 'Set_aside'):
        assert (cs[name]['role'], cs[name]['type'], cs[name]['unit']) == ('text', 'text', '')
    assert [r[3:] for r in t['data']] == [['', ''], ['', '']]


def test_column_names_stay_unique_and_valid():
    taken: set[str] = set()
    names = [stats_plt._column_name(t, taken) for t in
             ('Count', 'count', '%Total', '9 lives', '', 'x' * 80, 'x' * 80)]
    assert names[:5] == ['Count', 'count_2', 'Total', 'c_9_lives', 'col']
    assert names[5] == 'x' * 60 and names[6] == 'x' * 59 + '_2'
    assert all(len(n) <= 64 for n in names)


def test_the_file_is_written_whole_and_named_for_sharing(tmp_path):
    rows = [{'Sample': 's', 'Population': 'All events', 'Count': 3}]
    t = stats_plt.stats_table(rows, ['Sample', 'Population', 'Count'], {}, CHANNEL_STAT_ORDER)
    p = tmp_path / 'x.plt.csv'
    p.write_text('old')
    stats_plt.write_stats_plt(str(p), t)
    assert read_plt(p.read_bytes())[0]['data'] == [['s', 'All events', 3.0]]
    assert stats_plt.plt_file_name() == 'population_stats.plt.csv'
    assert stats_plt.STATISTICS_APP_URL == 'https://pocketlabessentials.com/stats'


def test_only_the_hash_checked_copy_is_left_out_of_the_code_rules():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]

    def vendored(script):
        spec = importlib.util.spec_from_file_location(f'_v_{script}', root / 'scripts' / f'{script}.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.VENDORED
    want = frozenset({'src/openflo/pocketlab_table.py'})
    assert vendored('best_practices') == want
    assert vendored('function_size') == want
