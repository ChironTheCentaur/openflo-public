"""openflo.document_rpc: a Document driven by named calls, with plain-data
answers, and FCS files above an event cap refused before they load.
Generated data only."""
import numpy as np
import pytest

from openflo import document_rpc as rpc
from openflo.document import Document
from tests._fcs_writer import write_fcs, write_float_fcs

NAMES = ['FSC-A', 'SSC-A', 'FL1-A']


def _fcs(path, n=500, seed=0):
    rng = np.random.default_rng(seed)
    return write_float_fcs(path, rng.uniform(0, 1e4, (n, 3)), NAMES)


# ── event count from the header ──────────────────────────────────────────

def test_event_count_reads_tot(tmp_path):
    assert rpc.fcs_event_count(_fcs(tmp_path / 'a.fcs', n=321)) == 321


@pytest.mark.parametrize('datatype,bits', [('I', 16), ('F', 32), ('D', 64)])
def test_event_count_without_tot_comes_from_the_data_size(tmp_path, datatype,
                                                          bits):
    data = np.arange(3 * 77).reshape(77, 3)
    p = tmp_path / 'a.fcs'
    write_fcs(p, data, NAMES, datatype=datatype, bits=bits, drop=('$TOT',))
    assert rpc.fcs_event_count(str(p)) == 77


def test_event_count_prefers_header_offsets_over_text_zero(tmp_path):
    # A file whose TEXT says $BEGINDATA 0 (as when the HEADER holds the
    # offsets) still sizes its DATA from the HEADER.
    p = tmp_path / 'a.fcs'
    write_fcs(p, np.zeros((40, 3)), NAMES, drop=('$TOT',))
    raw = p.read_bytes()
    begin = int(raw[26:34])
    end = int(raw[34:42])
    text = raw.replace(f'$BEGINDATA/{begin}/'.encode(),
                       b'$BEGINDATA/0/'.ljust(len(f'$BEGINDATA/{begin}/')))
    assert text != raw
    p.write_bytes(text)
    assert int(p.read_bytes()[26:34]) == begin and end > begin
    assert rpc.fcs_event_count(str(p)) == 40


@pytest.mark.parametrize('datatype', ['F', 'D'])
def test_event_count_without_pnb_uses_the_float_width(tmp_path, datatype):
    p = tmp_path / 'a.fcs'
    write_fcs(p, np.zeros((25, 3)), NAMES, datatype=datatype,
              drop=('$TOT', '$P2B'))
    assert rpc.fcs_event_count(str(p)) == 25


def test_event_count_reads_text_offsets_when_the_header_holds_zeros(tmp_path):
    # A file past 99,999,999 bytes writes its DATA offsets in TEXT only and
    # zeros in the HEADER.
    p = tmp_path / 'a.fcs'
    write_fcs(p, np.zeros((60, 3)), NAMES, drop=('$TOT',))
    raw = bytearray(p.read_bytes())
    raw[26:42] = b'       0       0'
    p.write_bytes(bytes(raw))
    assert rpc.fcs_event_count(str(p)) == 60


def test_event_count_keywords_are_case_insensitive(tmp_path):
    # No $P2B, so the DATA size cannot stand in: only $tot gives the count.
    p = tmp_path / 'a.fcs'
    write_fcs(p, np.zeros((33, 3)), NAMES, drop=('$P2B',))
    raw = p.read_bytes()
    assert raw.count(b'$TOT') == 1
    p.write_bytes(raw.replace(b'$TOT', b'$tot'))
    assert rpc.fcs_event_count(str(p)) == 33


def test_a_tot_that_is_not_a_whole_number_falls_back_to_the_data_size(
        tmp_path):
    p = tmp_path / 'a.fcs'
    write_fcs(p, np.zeros((12, 3)), NAMES, extra={'$TOT': '1e5'})
    assert rpc.fcs_event_count(str(p)) == 12


def test_event_count_is_none_when_bits_are_unknown(tmp_path):
    p = tmp_path / 'a.fcs'
    write_fcs(p, np.zeros((10, 3)), NAMES, drop=('$TOT', '$P2B'),
              extra={'$DATATYPE': 'I'})
    assert rpc.fcs_event_count(str(p)) is None


def test_event_count_refuses_a_file_that_is_not_fcs(tmp_path):
    p = tmp_path / 'a.fcs'
    p.write_bytes(b'not an fcs file at all' * 5)
    with pytest.raises(rpc.RpcError, match='not an FCS file'):
        rpc.fcs_event_count(str(p))


# ── the cap ──────────────────────────────────────────────────────────────

def test_cap_allows_up_to_and_including_the_limit(tmp_path):
    p = _fcs(tmp_path / 'a.fcs', n=100)
    rpc.check_event_cap(p, None)
    rpc.check_event_cap(p, 100)
    with pytest.raises(rpc.TooManyEvents, match='has 100 events.*up to 99 '):
        rpc.check_event_cap(p, 99)


def test_cap_refuses_a_file_whose_count_is_unknown(tmp_path):
    p = tmp_path / 'a.fcs'
    write_fcs(p, np.zeros((10, 3)), NAMES, drop=('$TOT', '$P2B'),
              extra={'$DATATYPE': 'I'})
    rpc.check_event_cap(str(p), None)
    with pytest.raises(rpc.TooManyEvents, match='cannot be read'):
        rpc.check_event_cap(str(p), 10**9)


def test_open_fcs_above_the_cap_is_refused_before_loading(tmp_path,
                                                         monkeypatch):
    p = _fcs(tmp_path / 'a.fcs', n=200)
    calls = []
    monkeypatch.setattr(Document, 'open_fcs',
                        lambda self, path: calls.append(path) or 'a')
    r = rpc.DocumentRpc(max_events=199)
    reply = r.handle({'id': 7, 'method': 'open_fcs', 'args': {'path': p}})
    assert reply['ok'] is False and reply['id'] == 7
    assert reply['error']['type'] == 'TooManyEvents'
    assert calls == []
    assert rpc.DocumentRpc(max_events=200).handle(
        {'id': 8, 'method': 'open_fcs', 'args': {'path': p}})['ok']
    assert calls == [p]


# ── encoding ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize('dtype', ['<f8', '>f8', '<f4', '>i4', '<i8', '?'])
def test_encode_packs_arrays_little_endian(dtype):
    a = np.arange(6).reshape(2, 3).astype(dtype)
    packed = rpc.encode(a)
    assert packed['shape'] == [2, 3]
    back = np.frombuffer(packed['data'],
                         dtype=np.dtype(packed['__ndarray__']).newbyteorder('<'))
    assert np.array_equal(back.reshape(2, 3), a)
    assert packed['__ndarray__'] == np.dtype(dtype).newbyteorder('=').name


def test_encode_turns_numpy_values_and_containers_into_plain_data():
    out = rpc.encode({1: (np.float64(1.5), np.int64(2)), 'b': {np.bool_(True)},
                      'o': np.array(['x', None], dtype=object)})
    assert out == {'1': [1.5, 2], 'b': [True], 'o': ['x', None]}
    assert type(out['1'][0]) is float and type(out['1'][1]) is int


def test_encode_keeps_a_non_contiguous_array_in_order():
    a = np.arange(12, dtype=float).reshape(3, 4)[:, ::2]
    packed = rpc.encode(a)
    assert np.array_equal(
        np.frombuffer(packed['data'], '<f8').reshape(packed['shape']), a)


# ── requests and replies ─────────────────────────────────────────────────

@pytest.mark.parametrize('method', ['__init__', '_get', 'labels', 'nope', ''])
def test_only_listed_methods_can_be_named(method):
    reply = rpc.DocumentRpc().handle({'id': 1, 'method': method})
    assert reply == {'id': 1, 'ok': False, 'error': {
        'type': 'RpcError', 'message': f'unknown method {method!r}'}}


@pytest.mark.parametrize('request_', [None, [], 'x',
                                      {'id': 2, 'method': 'samples',
                                       'args': ['x']}])
def test_a_malformed_request_gets_an_error_reply(request_):
    reply = rpc.DocumentRpc().handle(request_)
    assert reply['ok'] is False and reply['error']['type'] == 'RpcError'


def test_bad_arguments_are_an_error_reply_not_a_raise():
    reply = rpc.DocumentRpc().handle({'id': 3, 'method': 'samples',
                                      'args': {'bogus': 1}})
    assert reply['ok'] is False and reply['error']['type'] == 'TypeError'


def test_argument_names_must_be_strings():
    with pytest.raises(rpc.RpcError, match='must be strings'):
        rpc.DocumentRpc().call('samples', {1: 2})


def test_reset_keeps_the_document_options():
    r = rpc.DocumentRpc(qc=True, labels='FL1-A=L1')
    first = r.doc
    r.reset()
    assert r.doc is not first
    assert (r.doc.qc, r.doc.labels) == (True, 'FL1-A=L1')


def test_open_count_density_and_reset(tmp_path):
    p = _fcs(tmp_path / 'a.fcs')
    r = rpc.DocumentRpc(max_events=10_000)
    assert r.handle({'id': 1, 'method': 'open_fcs',
                     'args': {'path': p}}) == {'id': 1, 'ok': True,
                                               'result': 'a'}
    assert r.call('active') == 'a'
    with pytest.raises(rpc.RpcError):
        r.call('active', {'x': 1})
    g = r.call('add_gate', {'sample': 'a', 'gate': {
        'kind': 'interval', 'channel': 'FSC-A', 'lo': 0.0, 'hi': 5e3,
        'name': 'low'}})
    rows = r.call('stats', {'sample': 'a', 'stats': []})
    direct = r.doc.stats('a', stats=[])
    assert rows == direct and [x['id'] for x in rows] == [None, g['id']]
    dens = r.call('density', {'sample': 'a', 'x': 'FSC-A', 'y': 'FL1-A',
                              'bins': 8})
    want = r.doc.density('a', 'FSC-A', 'FL1-A', bins=8)
    got = np.frombuffer(dens['counts']['data'], '<f4').reshape(
        dens['counts']['shape'])
    assert dens['counts']['__ndarray__'] == 'float32'
    assert np.array_equal(got, want['counts'])
    r.call('reset')
    assert r.call('samples') == [] and r.call('active') is None


def test_histogram_overlays_and_quadrants_are_callable(tmp_path):
    p = _fcs(tmp_path / 'a.fcs')
    r = rpc.DocumentRpc()
    r.call('open_fcs', {'path': p})
    g = r.call('add_gate', {'sample': 'a', 'gate': {
        'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
        'x0': 0.0, 'x1': 5e3, 'y0': 0.0, 'y1': 5e3, 'name': 'box'}})
    h = r.call('histogram', {'sample': 'a', 'channel': 'FSC-A', 'bins': 10})
    assert h['counts']['__ndarray__'] == 'float32'
    assert np.frombuffer(h['counts']['data'], '<f4').sum() == 500
    (o,) = r.call('overlays', {'sample': 'a', 'x': 'FSC-A', 'y': 'SSC-A'})
    assert o['id'] == g['id'] and o['points']['__ndarray__'] == 'float64'
    pts = np.frombuffer(o['points']['data'], '<f8').reshape(o['points']['shape'])
    assert pts.shape[1] == 2 and (pts[0] == pts[-1]).all()
    q = r.call('add_quadrant', {'sample': 'a', 'x': 'FSC-A', 'y': 'SSC-A',
                                'x0': 5e3, 'y0': 5e3, 'parent_id': g['id']})
    assert len(q) == 4
    assert [r.call('gate', {'sample': 'a', 'gid': i})['parent_id']
            for i in q] == [g['id']] * 4


def test_a_marker_label_can_be_set_by_name(tmp_path):
    """The phone's axis sheet names a marker through set_channel_label."""
    r = rpc.DocumentRpc()
    name = r.call('open_fcs', {'path': _fcs(tmp_path / 'a.fcs')})
    assert r.handle({'id': 1, 'method': 'set_channel_label',
                     'args': {'channel': 'FL1-A', 'label': 'M1'}}) == {
        'id': 1, 'ok': True, 'result': None}
    labels = {c['name']: c['label']
              for c in r.call('channels', {'sample': name})}
    assert labels['FL1-A'] == 'M1'


def test_the_same_population_is_callable(tmp_path):
    r = rpc.DocumentRpc()
    for n in ('a', 'b'):
        r.call('open_fcs', {'path': _fcs(tmp_path / f'{n}.fcs')})
    box = {'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
           'x0': 0.0, 'x1': 5e3, 'y0': 0.0, 'y1': 5e3, 'name': 'box'}
    g = r.call('add_gate', {'sample': 'a', 'gate': box})
    assert r.call('same_population',
                  {'sample': 'a', 'gate': g['id'], 'other': 'b'}) is None
    h = r.call('add_gate', {'sample': 'b', 'gate': box})
    assert r.call('same_population',
                  {'sample': 'a', 'gate': g['id'], 'other': 'b'}) == h['id']
