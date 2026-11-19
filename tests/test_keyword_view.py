"""openflo.keyword_view and Document.keywords: what an FCS file says about
itself, for the phone's File info card. Generated data only."""
import os

import numpy as np

from openflo import document_rpc as rpc
from openflo.document import Document
from openflo.keyword_view import keyword_view
from tests._fcs_writer import write_fcs


def test_summary_lists_what_the_file_says_in_order():
    v = keyword_view([('$TOT', '500'), ('$CYT', ' Bench '), ('$DATE', '01-JAN-2026'),
                      ('$CREATOR', ''), ('$PAR', '2')])
    assert v['summary'] == [['Events', '500'], ['Parameters', '2'],
                            ['Date', '01-JAN-2026'], ['Cytometer', 'Bench']]


def test_keywords_sorted_by_name_channel_numbers_as_numbers_last_value_kept():
    # Given twice in any case: the last, as FlowIO keeps it.
    v = keyword_view([('$P10N', 'FL10-A'), ('$P2N', 'FL2-A'), ('ZKEY', 'z'),
                      ('$DATE', 'd1'), ('$date', 'd2'), ('$P2S', 'FL2 marker'),
                      ('akey', 'a')])
    assert v['keywords'] == [['akey', 'a'], ['$date', 'd2'], ['$P2N', 'FL2-A'],
                             ['$P2S', 'FL2 marker'], ['$P10N', 'FL10-A'],
                             ['ZKEY', 'z']]


def test_channels_up_to_par_with_blanks_for_what_is_not_said():
    v = keyword_view([('$PAR', '3'), ('$P1N', 'FSC-A'), ('$P3N', 'FL1-A'),
                      ('$P3S', 'FL1 marker'), ('$P3V', '450'), ('$P3G', '2'),
                      ('$P3R', '262144'), ('$P3E', '0,0')])
    assert [c['number'] for c in v['channels']] == [1, 2, 3]
    assert v['channels'][1] == {'number': 2, 'name': '', 'label': '', 'voltage': '',
                                'gain': '', 'range': '', 'amplification': ''}
    assert v['channels'][2] == {'number': 3, 'name': 'FL1-A', 'label': 'FL1 marker',
                                'voltage': '450', 'gain': '2', 'range': '262144',
                                'amplification': '0,0'}


def test_channels_reach_the_highest_pnn_when_par_is_short_or_bad():
    assert len(keyword_view([('$PAR', '1'), ('$P4N', 'x')])['channels']) == 4
    assert len(keyword_view([('$PAR', 'two'), ('$P2N', 'x')])['channels']) == 2
    assert keyword_view([('$PAR', 'two')])['channels'] == []
    # $PAR above the highest $PnN: up to $PAR; only $PnN adds channels.
    assert len(keyword_view([('$PAR', '5'), ('$P2N', 'x')])['channels']) == 5
    assert len(keyword_view([('$PAR', '3'), ('$P3N', 'x'), ('P4DISPLAY', 'LOG')])['channels']) == 3


def _doc(tmp_path):
    rng = np.random.default_rng(3)
    path = write_fcs(str(tmp_path / 'raw' / 'k.fcs'), rng.uniform(0, 1e3, (250, 2)),
                     ['FSC-A', 'FL1-A'], datatype='F',
                     extra={'$CYT': 'Bench', '$P2S': 'FL1 marker', '$P2V': '450',
                            'LabKey': 'x'})
    doc = Document()
    return doc, doc.open_fcs(path), path


def test_document_reads_the_keywords_as_written(tmp_path):
    os.makedirs(tmp_path / 'raw')
    doc, name, _ = _doc(tmp_path)
    info = doc.keywords(name)
    assert info['file'] == 'k.fcs'
    keys = dict(map(tuple, info['keywords']))
    assert keys['$CYT'] == 'Bench' and keys['LabKey'] == 'x' and keys['$TOT'] == '250'
    assert ['Events', '250'] in info['summary']
    assert [c['label'] for c in info['channels']] == ['', 'FL1 marker']
    assert info['channels'][1]['voltage'] == '450'


def test_document_falls_back_on_the_kept_keywords_when_the_file_is_gone(tmp_path):
    os.makedirs(tmp_path / 'raw')
    doc, name, path = _doc(tmp_path)
    os.remove(path)
    info = doc.keywords(name)
    keys = dict(map(tuple, info['keywords']))
    # FlowIO's lower-case, $-less keys; the rows still found.
    assert keys['cyt'] == 'Bench' and keys['labkey'] == 'x'
    assert ['Cytometer', 'Bench'] in info['summary']
    assert info['channels'][1]['label'] == 'FL1 marker'


def test_the_same_keyword_twice_reads_as_the_loader_read_it(tmp_path):
    os.makedirs(tmp_path / 'raw')
    path = write_fcs(str(tmp_path / 'raw' / 'k.fcs'), np.ones((10, 2)), ['FSC-A', 'FL1-A'],
                     datatype='F', extra={'$CYT': 'first', 'XCYT': 'x', '$cyt': 'second'})
    doc = Document()
    name = doc.open_fcs(path)
    loaded = doc._samples[name].data.metadata['cyt']
    assert loaded == 'second'
    assert ['Cytometer', loaded] in doc.keywords(name)['summary']


def test_a_file_cut_short_since_falls_back_on_the_kept_keywords(tmp_path):
    os.makedirs(tmp_path / 'raw')
    doc, name, path = _doc(tmp_path)
    # Its header's TEXT offsets now point past the end of the file.
    b = bytearray(open(path, 'rb').read())
    b[10:26] = b'99999990' + b'99999999'
    open(path, 'wb').write(bytes(b))
    keys = dict(map(tuple, doc.keywords(name)['keywords']))
    assert keys['cyt'] == 'Bench' and keys['labkey'] == 'x'


def test_the_rpc_answers_keywords(tmp_path):
    os.makedirs(tmp_path / 'raw')
    _, name, path = _doc(tmp_path)
    r = rpc.DocumentRpc()
    name = r.handle({'id': 0, 'method': 'open_fcs', 'args': {'path': path}})['result']
    out = r.handle({'id': 1, 'method': 'keywords', 'args': {'sample': name}})
    assert out['ok'] and out['result']['file'] == 'k.fcs'
