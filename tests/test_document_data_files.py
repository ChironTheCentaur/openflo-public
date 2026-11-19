"""Document.data_files: the data files a session or workspace names, and
whether opening it now would find each one. The phone asks first, copies the
files the person picks into the session's or workspace's folder, then opens
it."""
from __future__ import annotations

import json
import shutil

import pytest

from openflo.document import Document, DocumentError
from tests._open_fixtures import make_fcs, write_wsp
from tests.test_document import gated_doc


def _rows(got):
    return [(r['name'], r['found']) for r in got['files']]


def test_a_workspace_copied_alone_finds_each_file_once_put_beside_it(tmp_path):
    a = make_fcs(tmp_path / 'run' / 'a.fcs', seed=1)
    b = make_fcs(tmp_path / 'run' / 'b.fcs', seed=2)
    wsp = write_wsp(tmp_path / 'run' / 'x.wsp',
                    [('a', a, None, False), ('b', b, None, False)])
    # The phone holds a copy of the workspace in a folder of its own; the
    # files it names are not on this device.
    bundle = tmp_path / 'in' / '1'
    bundle.mkdir(parents=True)
    copy = str(bundle / 'x.wsp')
    shutil.copy(wsp, copy)
    shutil.move(str(tmp_path / 'run'), str(tmp_path / 'gone'))
    doc = Document()
    got = doc.data_files(copy)
    assert got['kind'] == 'workspace'
    assert _rows(got) == [('a.fcs', False), ('b.fcs', False)]
    shutil.copy(tmp_path / 'gone' / 'a.fcs', bundle / 'a.fcs')
    assert _rows(doc.data_files(copy)) == [('a.fcs', True), ('b.fcs', False)]
    assert doc.samples() == []          # asking opens nothing
    opened = doc.import_wsp(copy)
    assert opened['samples'] == ['a'] and len(opened['unresolved']) == 1


def test_a_session_copied_alone_lists_its_raw_files(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    out = tmp_path / 'proj' / 'run.flowsession'
    out.parent.mkdir()
    doc.save_session(str(out))
    bundle = tmp_path / 'in' / '2'
    bundle.mkdir(parents=True)
    copy = bundle / 'run.flowsession'
    shutil.copy(out, copy)
    raw = doc.samples()[0]['path']
    assert _rows(Document().data_files(str(copy))) == [('s1.fcs', True)]
    shutil.move(raw, str(tmp_path / 'kept.fcs'))
    got = Document().data_files(str(copy))
    assert got == {'kind': 'session', 'files': [{'name': 's1.fcs',
                                                'found': False}]}
    shutil.copy(tmp_path / 'kept.fcs', bundle / 's1.fcs')
    assert _rows(Document().data_files(str(copy))) == [('s1.fcs', True)]
    doc2 = Document()
    assert doc2.open_session(str(copy))['samples'] == [name]


def test_names_saved_on_another_os_and_repeats(tmp_path):
    p = tmp_path / 'w.flowsession'
    p.write_text(json.dumps({'format': 'openflo-session', 'samples': [
        {'name': 'one', 'basename': 'C:\\data\\day 1\\one.fcs'},
        {'name': 'two', 'path': 'D:\\data\\two.fcs'},
        {'name': 'again', 'path': '/data/one.fcs'},
        {'name': 'nothing'}, 'not an entry',
        # Found where it was saved; a later entry of that name is not.
        {'name': 'three', 'path': str(tmp_path / 'kept' / 'three.fcs')},
        {'name': 'three again', 'basename': 'three.fcs'}]}), 'utf-8')
    (tmp_path / 'two.fcs').write_bytes(b'')
    (tmp_path / 'kept').mkdir()
    (tmp_path / 'kept' / 'three.fcs').write_bytes(b'')
    assert _rows(Document().data_files(str(p))) == [
        ('one.fcs', False), ('two.fcs', True), ('three.fcs', True)]


def test_refusals(tmp_path):
    p = tmp_path / 'x.flowsession'
    p.write_text('{"samples": []}', 'utf-8')
    with pytest.raises(DocumentError, match='format marker'):
        Document().data_files(str(p))
    p.write_text('not json', 'utf-8')
    with pytest.raises(DocumentError, match='not a readable session'):
        Document().data_files(str(p))
    w = tmp_path / 'x.wsp'
    w.write_text('<not closed', 'utf-8')
    with pytest.raises(DocumentError, match='not a readable FlowJo'):
        Document().data_files(str(w))


def test_data_files_through_rpc(tmp_path):
    from openflo import document_rpc as rpc
    a = make_fcs(tmp_path / 'a.fcs')
    wsp = write_wsp(tmp_path / 'x.wsp', [('a', a, None, False)])
    got = rpc.DocumentRpc().call('data_files', {'path': wsp})
    assert got == {'kind': 'workspace', 'files': [{'name': 'a.fcs',
                                                  'found': True}]}
