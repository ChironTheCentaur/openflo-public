"""Nearby send protocol core (src/openflo/nearby/protocol.py): handshake,
records, messages, the checks a receiver makes, and the shared test vectors
the Kotlin and Swift twins read."""
from __future__ import annotations

import json

import pytest

from openflo.nearby import protocol as p
from openflo.nearby import vectors

V = json.load(open(vectors.PATH, encoding='utf-8'))
X = bytes.fromhex


def _pair(app: str = 'openflo') -> tuple[p.SessionKeys, p.SessionKeys]:
    ini, res = p.Initiator(app), p.Responder(app)
    m1 = ini.first()
    m2 = res.on_hello1(m1[4:])
    m3 = ini.on_hello2(m2[4:])
    kr = res.on_hello3(m3[4:])
    assert ini.keys is not None
    return ini.keys, kr


# ── handshake ────────────────────────────────────────────────────────────────

def test_handshake_both_sides_agree_and_keys_cross():
    ki, kr = _pair()
    assert ki.sas == kr.sas and len(ki.sas) == 6 and ki.sas.isdigit()
    assert ki.h == kr.h
    assert ki.send_key == kr.recv_key and ki.recv_key == kr.send_key
    assert ki.send_key != ki.recv_key
    msg = p.encode_message(p.Offer(b'\1' * 16, 5, b'\2' * 32, 'fcs', 'a.fcs'))
    assert kr.opener().feed(ki.sealer().seal(msg)) == [msg]
    assert ki.opener().feed(kr.sealer().seal(msg)) == [msg]


def test_every_connection_has_fresh_ephemerals_and_code():
    # A responder that reused eR/nR would let an attacker grind the code.
    a, b = p.Responder('openflo'), p.Responder('openflo')
    m1 = p.Initiator('openflo').first()[4:]
    assert a.on_hello1(m1) != b.on_hello1(m1)
    codes = {_pair()[0].h for _ in range(3)}
    assert len(codes) == 3


def test_bad_commitment_is_refused():
    ei = p.Ephemeral.fixed(vectors.E_I, vectors.N_I)
    res = p.Responder('openflo', p.Ephemeral.fixed(vectors.E_R, vectors.N_R))
    res.on_hello1(X(V['handshake']['m1']))
    with pytest.raises(p.ProtocolError) as e:
        res.on_hello3(X(V['negative']['m3_bad_commit']))
    assert e.value.reason == 'bad-commit'
    assert res.keys is None
    # The right reveal still works on a fresh responder.
    res2 = p.Responder('openflo', p.Ephemeral.fixed(vectors.E_R, vectors.N_R))
    res2.on_hello1(X(V['handshake']['m1']))
    assert res2.on_hello3(p.hello3(ei)).sas == V['handshake']['sas']


def test_off_curve_point_is_refused():
    ini = p.Initiator('openflo', p.Ephemeral.fixed(vectors.E_I, vectors.N_I))
    with pytest.raises(p.ProtocolError) as e:
        ini.on_hello2(X(V['negative']['m2_off_curve']))
    assert e.value.reason == 'bad-point'
    with pytest.raises(p.ProtocolError, match='not an uncompressed'):
        p.decode_point(b'\x02' + bytes(64))
    with pytest.raises(p.ProtocolError, match='not an uncompressed'):
        p.decode_point(X(V['handshake']['e_r_pub'])[:33])


def test_hello_refusals():
    with pytest.raises(p.ProtocolError) as e:
        p.Responder('openflo').on_hello1(X(V['refusals']['m1_version_2']))
    assert e.value.reason == 'unsupported-version'
    assert e.value.reply == p.frame(X(V['refusals']['unsupported_m2']))
    with pytest.raises(p.ProtocolError) as e:
        p.Initiator('openflo').on_hello2(X(V['refusals']['unsupported_m2']))
    assert e.value.reason == 'unsupported-version' and e.value.max_version == 1
    with pytest.raises(p.ProtocolError) as e:
        p.Responder('pocketlab-gel').on_hello1(X(V['handshake']['m1']))
    assert e.value.reason == 'wrong-app'
    assert e.value.reply == p.frame(X(V['refusals']['wrong_app_m2']))
    with pytest.raises(p.ProtocolError, match='wrong-app'):
        p.Initiator('openflo').on_hello2(X(V['refusals']['wrong_app_m2']))
    with pytest.raises(p.ProtocolError, match='not-pl-send'):
        p.Responder('openflo').on_hello1(X(V['negative']['m1_bad_magic']))


def test_handshake_messages_out_of_order_are_refused():
    res = p.Responder('openflo')
    with pytest.raises(p.ProtocolError, match='out-of-state'):
        res.on_hello3(bytes(97))
    m1 = p.Initiator('openflo').first()[4:]
    res.on_hello1(m1)
    with pytest.raises(p.ProtocolError, match='out-of-state'):
        res.on_hello1(m1)
    ini = p.Initiator('openflo')
    m2 = p.Responder('openflo').on_hello1(ini.first()[4:])[4:]
    ini.on_hello2(m2)
    with pytest.raises(p.ProtocolError, match='out-of-state'):
        ini.on_hello2(m2)


def test_frame_buffer_splits_and_refuses_long_lengths():
    fb = p.FrameBuffer()
    whole = p.frame(b'abc') + p.frame(b'') + p.frame(b'xyz')
    assert fb.feed(whole[:5]) == []
    assert fb.feed(whole[5:13]) == [b'abc', b'']
    assert fb.feed(whole[13:] + b'\0\0') == [b'xyz']
    assert fb.rest() == b'\0\0'
    with pytest.raises(p.ProtocolError, match='too-long'):
        p.FrameBuffer().feed((p.MAX_HANDSHAKE_BODY + 1).to_bytes(4, 'big'))


# ── records ──────────────────────────────────────────────────────────────────

def _keys_from_vectors() -> tuple[bytes, bytes]:
    return X(V['handshake']['k_ir']), X(V['handshake']['k_ri'])


def test_records_decrypt_in_order_byte_by_byte():
    k_ir, _ = _keys_from_vectors()
    o = p.RecordOpener(k_ir)
    stream = b''.join(X(r['record']) for r in V['records_i_to_r'])
    got: list[bytes] = []
    for i in range(len(stream)):
        got += o.feed(stream[i:i + 1])
    assert got == [X(r['plaintext']) for r in V['records_i_to_r']]


def test_tampered_record_is_refused():
    k_ir, _ = _keys_from_vectors()
    with pytest.raises(p.ProtocolError, match='bad-record'):
        p.RecordOpener(k_ir).feed(X(V['negative']['record_bad_tag']))


def test_replayed_record_is_refused():
    k_ir, _ = _keys_from_vectors()
    o = p.RecordOpener(k_ir)
    first = X(V['records_i_to_r'][0]['record'])
    assert len(o.feed(first)) == 1
    with pytest.raises(p.ProtocolError, match='bad-record'):
        o.feed(first)


def test_record_from_wrong_direction_is_refused():
    k_ir, _ = _keys_from_vectors()
    with pytest.raises(p.ProtocolError, match='bad-record'):
        p.RecordOpener(k_ir).feed(X(V['records_r_to_i'][0]['record']))


def test_oversize_record_length_refused_from_header_alone():
    k_ir, _ = _keys_from_vectors()
    with pytest.raises(p.ProtocolError, match='too-long'):
        p.RecordOpener(k_ir).feed(X(V['negative']['record_length_too_big']))
    with pytest.raises(p.ProtocolError, match='too-long'):
        p.RecordOpener(k_ir).feed((p.TAG_LEN - 1).to_bytes(4, 'big'))
    # The largest legal record is accepted.
    big = p.RecordSealer(k_ir).seal(bytes(p.MAX_PLAINTEXT))
    assert int.from_bytes(big[:4], 'big') == p.MAX_RECORD
    assert p.RecordOpener(k_ir).feed(big) == [bytes(p.MAX_PLAINTEXT)]
    with pytest.raises(ValueError):
        p.RecordSealer(k_ir).seal(bytes(p.MAX_PLAINTEXT + 1))


def test_counter_limit_ends_the_link():
    k_ir, _ = _keys_from_vectors()
    s = p.RecordSealer(k_ir)
    s._n = p.MAX_COUNTER - 1
    rec = s.seal(b'\x14')
    with pytest.raises(p.ProtocolError, match='counter-exhausted'):
        s.seal(b'\x14')
    o = p.RecordOpener(k_ir)
    o._n = p.MAX_COUNTER - 1
    assert o.feed(rec) == [b'\x14']
    with pytest.raises(p.ProtocolError, match='counter-exhausted'):
        o.feed(rec)


# ── messages ─────────────────────────────────────────────────────────────────

ALL = [p.Identity(b'\4' + bytes(64), 'Phone ✓', b'\x30' * 70), p.PairState(True),
       p.PairState(False), p.Confirm(), p.Abort(p.R_TIMEOUT),
       p.Offer(bytes(16), 2 ** 40, bytes(range(32)), 'project', 'x.flowproject'),
       p.Accept(7), p.Decline(p.R_KIND), p.Data(b'x'), p.Data(bytes(p.DATA_CHUNK)),
       p.Done(), p.Result(False, p.R_HASH), p.Close()]


@pytest.mark.parametrize('m', ALL, ids=lambda m: type(m).__name__)
def test_message_round_trip(m):
    assert p.decode_message(p.encode_message(m)) == m


@pytest.mark.parametrize('m', [m for m in ALL if not isinstance(m, p.Data)
                               and len(p.encode_message(m)) > 1],
                         ids=lambda m: type(m).__name__)
def test_truncated_or_padded_message_is_refused(m):
    b = p.encode_message(m)
    with pytest.raises(p.ProtocolError, match='malformed'):
        p.decode_message(b[:-1])
    with pytest.raises(p.ProtocolError, match='malformed'):
        p.decode_message(b + b'\0')


def test_bad_messages_are_refused():
    with pytest.raises(p.ProtocolError, match='malformed'):
        p.decode_message(X(V['negative']['pair_state_flag_2']))
    with pytest.raises(p.ProtocolError, match='malformed'):
        p.decode_message(X(V['negative']['unknown_type']))
    with pytest.raises(p.ProtocolError, match='malformed'):
        p.decode_message(b'')
    name = 'n' * (p.MAX_DEVICE_NAME + 1)
    long_name = (bytes([p.T_IDENTITY]) + bytes(65) + len(name).to_bytes(2, 'big')
                 + name.encode() + b'\0\0')
    with pytest.raises(p.ProtocolError, match='text too long'):
        p.decode_message(long_name)
    long_sig = bytes([p.T_IDENTITY]) + bytes(65) + b'\0\0' + (p.MAX_SIGNATURE + 1).to_bytes(2, 'big')
    with pytest.raises(p.ProtocolError, match='signature too long'):
        p.decode_message(long_sig)
    bad_utf8 = bytes([p.T_IDENTITY]) + bytes(65) + b'\0\1\xff'
    with pytest.raises(p.ProtocolError, match='UTF-8'):
        p.decode_message(bad_utf8)
    with pytest.raises(ValueError):
        p.encode_message(p.Data(bytes(p.DATA_CHUNK + 1)))


def test_vector_records_decode_to_the_listed_messages():
    k_ir, k_ri = _keys_from_vectors()
    for key, rows in ((k_ir, V['records_i_to_r']), (k_ri, V['records_r_to_i'])):
        o = p.RecordOpener(key)
        for r in rows:
            (pt,) = o.feed(X(r['record']))
            assert pt == X(r['plaintext'])
            assert p.encode_message(p.decode_message(pt)) == pt


# ── identities and pairing ───────────────────────────────────────────────────

def test_identity_signs_this_connection_and_role():
    ki, kr = _pair()
    me = p.new_static_key()
    ident = p.decode_message(p.encode_message(p.make_identity(me, 'PC', p.ROLE_I, ki.h)))
    assert isinstance(ident, p.Identity)
    p.verify_identity(ident, p.ROLE_I, kr.h)
    with pytest.raises(p.ProtocolError, match='bad-identity'):
        p.verify_identity(ident, p.ROLE_R, kr.h)
    other, _ = _pair()
    with pytest.raises(p.ProtocolError, match='bad-identity'):
        p.verify_identity(ident, p.ROLE_I, other.h)
    # The receiver signs as R, and only R verifies.
    theirs = p.make_identity(me, 'Phone', p.ROLE_R, kr.h)
    p.verify_identity(theirs, p.ROLE_R, ki.h)
    with pytest.raises(p.ProtocolError, match='bad-identity'):
        p.verify_identity(theirs, p.ROLE_I, ki.h)


def test_vector_identities_verify_only_for_their_role():
    h = X(V['handshake']['h'])
    for role, key in ((p.ROLE_I, 'i'), (p.ROLE_R, 'r')):
        d = V['identities'][key]
        ident = p.Identity(X(d['static_pub']), d['name'], X(d['signature']))
        p.verify_identity(ident, role, h)
        assert p.fingerprint(ident.static_pub).hex() == d['fingerprint']
        assert p.txt_id(ident.static_pub) == d['txt_id']
    d = V['identities']['r']
    with pytest.raises(p.ProtocolError, match='bad-identity'):
        p.verify_identity(p.Identity(X(d['static_pub']), d['name'], X(d['signature'])),
                          p.ROLE_I, h)


def test_code_shown_unless_both_sides_know_each_other():
    assert p.needs_pairing(True, True) is False
    assert p.needs_pairing(True, False) is True
    assert p.needs_pairing(False, True) is True
    assert p.needs_pairing(False, False) is True


def test_paired_device_with_another_key_is_refused():
    p.check_paired_key(None, b'any')            # not a paired pick: no check
    p.check_paired_key(b'k1', b'k1')
    with pytest.raises(p.ProtocolError) as e:
        p.check_paired_key(b'k1', b'k2')
    assert e.value.reason == 'not-paired-device'


# ── receiver checks ──────────────────────────────────────────────────────────

def test_resume_offset_must_be_inside_the_file():
    p.check_offset(0, 10)
    p.check_offset(10, 10)
    for bad in (-1, 11):
        with pytest.raises(p.ProtocolError, match='bad-offset'):
            p.check_offset(bad, 10)


@pytest.mark.parametrize('row', V['file_names'], ids=lambda r: r['sent'][:20] or 'empty')
def test_file_names_match_vectors(row):
    assert p.safe_file_name(row['sent'], row['ext']) == row['saved']


def test_file_name_rules():
    assert p.safe_file_name('../../x.fcs', '.fcs') == 'x.fcs'
    assert p.safe_file_name('a\\b\\c.fcs', '.fcs') == 'c.fcs'
    assert p.safe_file_name('nul.fcs', '.fcs') == 'received nul.fcs'
    assert p.safe_file_name('LPT1', '.wsp') == 'received LPT1.wsp'
    assert '\u202e' not in p.safe_file_name('a\u202eb.fcs', '.fcs')
    assert p.safe_file_name('a<b>"c|d', '.fcs') == 'a_b__c_d.fcs'
    assert p.safe_file_name('end. .', '.fcs') == 'end.fcs'
    assert p.safe_file_name('. .', '.fcs') == 'received.fcs'
    assert p.safe_file_name('.hidden', '.fcs') == 'hidden.fcs'
    assert p.safe_file_name('tool.exe', '.fcs').endswith('.fcs')
    assert len(p.safe_file_name('y' * 500, '.flowproject')) == p.SAFE_NAME_BYTES
    for long in ('😀' * 300, 'é' * 250, 'a' + 'é' * 250):
        saved = p.safe_file_name(long, '.fcs')
        assert len(saved.encode('utf-8')) <= p.SAFE_NAME_BYTES and saved.endswith('.fcs')
        assert len(saved.encode('utf-8')) > p.SAFE_NAME_BYTES - 4    # cut, not emptied
    for reserved in ('CONIN$', 'conout$', 'CON ', 'com1', 'CON .x', 'nul.'):
        assert p.safe_file_name(reserved, '.fcs').startswith('received ')
    for kept in ('console', 'contour', 'auxiliary', 'nullable', 'com10', 'lpt'):
        assert p.safe_file_name(kept, '.fcs') == kept + '.fcs'
    assert p.safe_file_name('. x.fcs', '.fcs') == 'x.fcs'
    assert p.safe_file_name('a\U000F0000b', '.fcs') == 'ab.fcs'       # private use
    assert p.safe_file_name('a\u0378b', '.fcs') == 'a\u0378b.fcs'      # unassigned is kept


@pytest.mark.parametrize('row', V['addresses'], ids=lambda r: r['addr'])
def test_lan_addresses_match_vectors(row):
    assert p.is_lan_address(row['addr']) is row['lan']


# ── key derivation and vectors file ──────────────────────────────────────────

def test_hkdf_matches_rfc5869():
    c = V['hkdf_rfc5869_case1']
    assert p._hkdf(X(c['ikm']), X(c['salt']), X(c['info']), c['length']).hex() == c['okm']


def test_vector_handshake_replays_from_its_inputs():
    hs = V['handshake']
    ini = p.Initiator(hs['app'], p.Ephemeral.fixed(int(hs['e_i_scalar'], 16), X(hs['n_i'])))
    res = p.Responder(hs['app'], p.Ephemeral.fixed(int(hs['e_r_scalar'], 16), X(hs['n_r'])))
    assert ini.first()[4:].hex() == hs['m1']
    assert res.on_hello1(X(hs['m1']))[4:].hex() == hs['m2']
    assert ini.on_hello2(X(hs['m2']))[4:].hex() == hs['m3']
    kr = res.on_hello3(X(hs['m3']))
    assert kr.h.hex() == hs['h'] and kr.sas == hs['sas']
    assert kr.recv_key.hex() == hs['k_ir'] and kr.send_key.hex() == hs['k_ri']
    k_ir, k_ri, sas = p.derive(X(hs['ss']), X(hs['h']))
    assert (k_ir.hex(), k_ri.hex(), sas) == (hs['k_ir'], hs['k_ri'], hs['sas'])


def test_vectors_file_is_current():
    assert vectors.main([]) == 0


# ── review round 1 additions ─────────────────────────────────────────────────

NEG = V['negative']


@pytest.mark.parametrize('key', ['m2_trailing_byte', 'm2_unsupported_trailing_byte',
                                 'm2_wrong_app_trailing_byte'])
def test_hello2_with_trailing_bytes_is_refused(key):
    with pytest.raises(p.ProtocolError, match='malformed'):
        p.parse_hello2(X(NEG[key]))


def test_hello1_app_longer_than_32_is_refused():
    with pytest.raises(p.ProtocolError, match='app id too long'):
        p.parse_hello1(X(NEG['m1_app_too_long']))
    ok = p.MAGIC + bytes([1, 32]) + b'a' * 32 + bytes(32)
    assert p.parse_hello1(ok)[1] == 'a' * 32


def test_bad_app_id_is_refused_when_made():
    for bad in ('Open Flo', '', 'a' * 33, '-x'):
        with pytest.raises(ValueError):
            p.Responder(bad)
        with pytest.raises(ValueError):
            p.Initiator(bad)


@pytest.mark.parametrize('key', ['point_infinity', 'point_x_ge_p', 'point_compressed_02',
                                 'point_prefix_03_65_bytes'])
def test_invalid_points_are_refused(key):
    with pytest.raises(p.ProtocolError, match='bad-point'):
        p.decode_point(X(NEG[key]))


def test_short_record_and_long_handshake_lengths_are_refused():
    k_ir, _ = _keys_from_vectors()
    with pytest.raises(p.ProtocolError, match='too-long'):
        p.RecordOpener(k_ir).feed(X(NEG['record_length_15']))
    with pytest.raises(p.ProtocolError, match='too-long'):
        p.FrameBuffer().feed(X(NEG['handshake_length_257']))
    assert p.FrameBuffer().feed(p.frame(bytes(p.MAX_HANDSHAKE_BODY))) == [bytes(p.MAX_HANDSHAKE_BODY)]


@pytest.mark.parametrize('key', ['identity_name_not_utf8', 'data_empty', 'offer_size_2_63',
                                 'accept_offset_2_63'])
def test_bad_record_messages_are_refused(key):
    with pytest.raises(p.ProtocolError, match='malformed'):
        p.decode_message(X(NEG[key]))


def test_sizes_up_to_2_63_minus_1_round_trip_and_beyond_are_refused():
    top = 2 ** 63 - 1
    assert p.decode_message(p.encode_message(p.Accept(top))) == p.Accept(top)
    o = p.Offer(bytes(16), top, bytes(32), 'fcs', 'a.fcs')
    assert p.decode_message(p.encode_message(o)) == o
    with pytest.raises(ValueError):
        p.encode_message(p.Accept(top + 1))
    with pytest.raises(ValueError):
        p.encode_message(p.Offer(bytes(16), top + 1, bytes(32), 'fcs', 'a.fcs'))
    with pytest.raises(ValueError):
        p.encode_message(p.Data(b''))


def test_a_refused_responder_takes_nothing_more():
    res = p.Responder('openflo')
    with pytest.raises(p.ProtocolError, match='unsupported-version'):
        res.on_hello1(X(V['refusals']['m1_version_2']))
    with pytest.raises(p.ProtocolError, match='out-of-state'):
        res.on_hello1(X(V['handshake']['m1']))
    res = p.Responder('pocketlab-gel')
    with pytest.raises(p.ProtocolError, match='wrong-app'):
        res.on_hello1(X(V['handshake']['m1']))
    with pytest.raises(p.ProtocolError, match='out-of-state'):
        res.on_hello1(X(V['handshake']['m1']))
    res = p.Responder('openflo')
    with pytest.raises(p.ProtocolError):
        res.on_hello1(X(NEG['m1_bad_magic']))
    with pytest.raises(p.ProtocolError, match='out-of-state'):
        res.on_hello1(X(V['handshake']['m1']))


@pytest.mark.parametrize('row', V['device_names'], ids=lambda r: repr(r['sent'])[:20])
def test_device_names_match_vectors(row):
    assert p.safe_device_name(row['sent']) == row['shown']


def test_device_name_rules():
    assert p.safe_device_name('a\u202eb') == 'ab'
    assert p.safe_device_name('a\tb\nc') == 'a b c'
    assert p.safe_device_name(' \u200b ') == 'Unnamed device'
    assert len(p.safe_device_name('😀' * 40).encode()) <= p.MAX_DEVICE_NAME


@pytest.mark.parametrize('sc', V['sessions'], ids=lambda sc: sc['name'])
def test_session_order_matches_vectors(sc):
    sess = p.Session(sc['role'].encode())
    failed = None
    for i, step in enumerate(sc['steps']):
        m = p.decode_message(X(step['msg']))
        if step['way'] == 'send':
            sess.sent(m)
            continue
        try:
            sess.received(m)
        except p.ProtocolError as e:
            assert e.reason == 'out-of-state'
            failed = i
            break
    assert failed == sc['fails_at']


def test_session_pairing_state():
    s = p.Session(p.ROLE_R)
    assert s.pairing is None and not s.paired
    s.received(p.Identity(b'\4' + bytes(64), 'x', b''))
    s.sent(p.PairState(True))
    s.received(p.PairState(True))
    assert s.pairing is False and s.paired
    s = p.Session(p.ROLE_I)
    s.sent(p.PairState(True))
    s.received(p.Identity(b'\4' + bytes(64), 'x', b''))
    s.received(p.PairState(False))
    assert s.pairing is True and not s.paired
    s.sent(p.Confirm())
    assert not s.paired
    s.received(p.Confirm())
    assert s.paired
    with pytest.raises(ValueError):
        p.Session(b'X')


def test_session_ends_after_abort_and_own_close():
    s = p.Session(p.ROLE_R)
    s.received(p.Abort(p.R_USER))
    assert s.ended
    with pytest.raises(p.ProtocolError):
        s.received(p.Abort(p.R_USER))
    s = p.Session(p.ROLE_I)
    s.sent(p.Close())
    with pytest.raises(p.ProtocolError):
        s.received(p.Identity(b'\4' + bytes(64), 'x', b''))


def test_session_refuses_empty_data_built_directly():
    s = p.Session(p.ROLE_R)
    for m in (p.Identity(b'\4' + bytes(64), 'x', b''), ):
        s.received(m)
    s.sent(p.PairState(True))
    s.received(p.PairState(True))
    s.received(p.Offer(bytes(16), 10, bytes(32), 'fcs', 'a.fcs'))
    s.sent(p.Accept(0))
    with pytest.raises(p.ProtocolError, match='out-of-state'):
        s.received(p.Data(b''))


def _paired(role):
    s = p.Session(role)
    s.received(p.Identity(b'\4' + bytes(64), 'x', b''))
    s.sent(p.PairState(True))
    s.received(p.PairState(True))
    return s


def test_session_refuses_a_second_done_at_the_receiver():
    s = _paired(p.ROLE_R)
    s.received(p.Offer(bytes(16), 10, bytes(32), 'fcs', 'a.fcs'))
    s.sent(p.Accept(0))
    s.received(p.Data(b'x' * 10))
    s.received(p.Done())
    with pytest.raises(p.ProtocolError, match='out-of-state'):
        s.received(p.Done())
    assert s.ended


def test_session_refuses_a_second_result_at_the_sender():
    s = _paired(p.ROLE_I)
    s.sent(p.Offer(bytes(16), 10, bytes(32), 'fcs', 'a.fcs'))
    s.received(p.Accept(0))
    with pytest.raises(p.ProtocolError, match='out-of-state'):
        _paired(p.ROLE_I).received(p.Result(True, 0))      # nothing offered yet
    s.sent(p.Data(b'x' * 10))
    s.sent(p.Done())
    s.received(p.Result(True, 0))
    with pytest.raises(p.ProtocolError, match='out-of-state'):
        s.received(p.Result(True, 0))
    assert s.ended
