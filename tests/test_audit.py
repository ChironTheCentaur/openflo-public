"""Tests for the provenance / audit-trail log (openflo.audit.AuditLog).
Pure stdlib; no Tk."""
from __future__ import annotations

import csv
import io

from openflo.audit import AuditLog, _short


def test_record_assigns_monotonic_seq():
    log = AuditLog()
    e1 = log.record('sample.load', name='s1')
    e2 = log.record('gate.add', kind='polygon')
    assert e1['seq'] == 1 and e2['seq'] == 2
    assert len(log) == 2
    assert bool(log) is True


def test_record_merges_kwargs_into_details():
    log = AuditLog()
    e = log.record('cytonorm', time='2026-06-16T10:00:00',
                   details={'mode': 'goal'}, n_metaclusters=10)
    assert e['details'] == {'mode': 'goal', 'n_metaclusters': 10}
    assert e['time'] == '2026-06-16T10:00:00'
    assert e['action'] == 'cytonorm'


def test_entries_are_copies():
    log = AuditLog()
    log.record('a', x=1)
    es = log.entries()
    es[0]['details']['x'] = 999
    assert log.entries()[0]['details']['x'] == 1     # internal state untouched


def test_clear():
    log = AuditLog()
    log.record('a')
    log.record('b')
    log.clear()
    assert len(log) == 0 and not log
    assert log.record('c')['seq'] == 1               # seq resets


def test_roundtrip_to_from_list():
    log = AuditLog()
    log.record('sample.load', time='t1', name='s1', n_events=1000)
    log.record('gate.add', time='t2', kind='ellipsoid')
    data = log.to_list()
    log2 = AuditLog.from_list(data)
    assert log2.entries() == log.entries()
    # next seq continues past the restored max
    assert log2.record('x')['seq'] == 3


def test_from_list_repairs_missing_seq():
    log = AuditLog.from_list([
        {'action': 'a', 'details': {}},
        {'action': 'b', 'details': {}},
    ])
    seqs = [e['seq'] for e in log.entries()]
    assert seqs == [1, 2]


def test_to_markdown_has_header_and_table():
    log = AuditLog()
    log.record('sample.load', time='2026-06-16T10:00:00', name='s1',
               n_events=12345)
    md = log.to_markdown(meta={'openflo_version': '0.1.0'})
    assert '# OpenFlo analysis audit trail' in md
    assert '**openflo_version**: 0.1.0' in md
    assert '| # | Time | Action | Details |' in md
    assert '`sample.load`' in md
    assert 'name=s1' in md
    assert 'n_events=12345' in md


def test_to_markdown_escapes_pipes():
    log = AuditLog()
    log.record('note', text='a|b|c')
    md = log.to_markdown()
    # The literal pipes in the value must be escaped so the table stays valid.
    row = [ln for ln in md.splitlines() if 'note' in ln][0]
    assert 'a\\|b\\|c' in row


def test_to_csv_parses_back():
    log = AuditLog()
    log.record('sample.load', time='t1', name='s1', n_events=100)
    log.record('cytonorm', time='t2', mode='goal')
    rows = list(csv.reader(io.StringIO(log.to_csv())))
    assert rows[0] == ['seq', 'time', 'action', 'details']
    assert rows[1][0] == '1' and rows[1][2] == 'sample.load'
    assert 'name=s1' in rows[1][3]
    assert rows[2][2] == 'cytonorm'


def test_to_text_one_line_per_entry():
    log = AuditLog()
    log.record('a', time='t1', x=1)
    log.record('b')                       # no time, no details
    lines = log.to_text().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith('[  1]')
    assert 'a' in lines[0] and 'x=1' in lines[0]


def test_short_truncates_and_compacts():
    assert _short(3.14159265) == '3.142'
    assert _short(True) == 'yes'
    assert _short([1, 2, 3]) == '[1, 2, 3]'
    long = 'x' * 200
    assert len(_short(long)) <= 80
    assert _short(list(range(20))).endswith('more]')


def test_short_dict_and_newlines():
    assert _short({'a': 1, 'b': 2}) == 'a=1, b=2'
    assert '\n' not in _short('line1\nline2')
