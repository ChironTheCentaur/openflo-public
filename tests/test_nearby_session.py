"""Nearby send end to end over a local socket pair (src/openflo/nearby/
session.py, store.py): pairing, the file arriving intact, refusals, resume
and the checks that end a transfer. Generated files only."""
from __future__ import annotations

import hashlib
import os
import socket
import threading

import pytest

from openflo.nearby import protocol as p
from openflo.nearby import session as s
from openflo.nearby.store import NearbyStore

KINDS = {'project': '.flowproject', 'fcs': '.fcs'}


class SendUI(s.SendUI):
    def __init__(self, code_from=None, typed=None, cancel_after=None):
        self.code_from = code_from      # a RecvUI whose shown code is typed
        self.typed = typed
        self.asked = 0
        self.cancel_after = cancel_after
        self.sent = []

    def enter_code(self, peer_name):
        self.asked += 1
        if self.typed is not None:
            return self.typed
        assert self.code_from is not None
        assert self.code_from.shown.wait(5)
        return self.code_from.code

    def progress(self, done, total):
        self.sent.append(done)

    def cancelled(self):
        return self.cancel_after is not None and self.sent and self.sent[-1] >= self.cancel_after


class RecvUI(s.ReceiveUI):
    def __init__(self, confirm=True, accept=True):
        self.shown = threading.Event()
        self.code = None
        self._confirm = confirm
        self._accept = accept
        self.got = []

    def show_code(self, code, peer_name):
        self.code = code
        self.shown.set()

    def confirm_pairing(self, code, peer_name):
        self.confirm_code = code
        return self._confirm

    def accept(self, name, size, kind, peer_name):
        self.offered = (name, size, kind)
        return self._accept

    def progress(self, done, total):
        self.got.append(done)


def _file(tmp_path, name='run.flowproject', size=300_000, seed=1):
    data = bytes((i * seed * 31 + i // 7) % 256 for i in range(size))
    path = tmp_path / name
    path.write_bytes(data)
    return str(path), data


@pytest.fixture
def stores(tmp_path):
    return (NearbyStore(str(tmp_path / 'a'), 'Computer'),
            NearbyStore(str(tmp_path / 'b'), 'Phone'))


def run(stores, out, sui, rui, folder, *, app_r='openflo', expected=None,
        max_bytes=None, free=None):
    """Sender in this thread, receiver in another; each side's outcome."""
    a, b = socket.socketpair()
    got = {}

    def recv():
        try:
            got['r'] = s.receive_one(b, app_r, stores[1], KINDS, folder, rui,
                                     max_bytes=max_bytes,
                                     free_bytes=(lambda _f: free) if free else None)
        except Exception as e:  # noqa: BLE001 - recorded for the test
            got['r'] = e
        finally:
            b.close()

    t = threading.Thread(target=recv)
    t.start()
    try:
        got['s'] = s.send_file(a, 'openflo', stores[0], out, sui, expected_pub=expected)
    except Exception as e:  # noqa: BLE001 - recorded for the test
        got['s'] = e
    finally:
        a.close()
    t.join(20)
    assert not t.is_alive()
    return got['s'], got['r']


def test_first_send_pairs_then_the_file_arrives_intact(stores, tmp_path):
    path, data = _file(tmp_path)
    rui = RecvUI()
    sui = SendUI(code_from=rui)
    sent, got = run(stores, s.Outgoing(path, 'project'), sui, rui, str(tmp_path / 'in'))
    assert isinstance(got, s.Received), got
    assert open(got.path, 'rb').read() == data
    assert os.path.basename(got.path) == 'run.flowproject'
    assert sent.newly_paired and got.peer.newly_paired
    assert stores[0].paired_key(stores[1].public_key()) is not None
    assert stores[1].paired_key(stores[0].public_key()) is not None
    assert rui.got[-1] == len(data) and sui.sent[-1] == len(data)
    # No partial file left behind.
    assert not [f for _d, _s, fs in os.walk(stores[1].root) for f in fs if f.endswith('.part')]


def test_paired_devices_send_without_a_code_and_never_overwrite(stores, tmp_path):
    path, data = _file(tmp_path)
    rui = RecvUI()
    run(stores, s.Outgoing(path, 'project'), SendUI(code_from=rui), rui, str(tmp_path / 'in'))
    rui2 = RecvUI()
    sui2 = SendUI(typed='000000')
    sent, got = run(stores, s.Outgoing(path, 'project'), sui2, rui2, str(tmp_path / 'in'),
                    expected=stores[1].public_key())
    assert isinstance(got, s.Received), got
    assert sui2.asked == 0 and not rui2.shown.is_set()
    assert not sent.newly_paired
    assert os.path.basename(got.path) == 'run (2).flowproject'
    assert open(got.path, 'rb').read() == data


def test_wrong_code_ends_it_and_pairs_nobody(stores, tmp_path):
    path, _ = _file(tmp_path)
    rui = RecvUI()
    sent, got = run(stores, s.Outgoing(path, 'project'), SendUI(typed='999999x'), rui,
                    str(tmp_path / 'in'))
    assert isinstance(sent, s.TransferError) and sent.reason == 'wrong-code'
    assert isinstance(got, s.TransferError) and got.peer_reason == p.R_WRONG_CODE
    assert stores[0].paired() == [] and stores[1].paired() == []


def test_receiver_saying_no_to_pairing_pairs_nobody(stores, tmp_path):
    path, _ = _file(tmp_path)
    rui = RecvUI(confirm=False)
    sent, got = run(stores, s.Outgoing(path, 'project'), SendUI(code_from=rui), rui,
                    str(tmp_path / 'in'))
    assert isinstance(sent, s.TransferError) and sent.peer_reason == p.R_USER
    assert isinstance(got, s.TransferError) and got.reason == 'cancelled'
    assert stores[0].paired() == [] and stores[1].paired() == []


def test_one_side_forgot_the_pairing_so_the_code_is_asked_again(stores, tmp_path):
    path, _ = _file(tmp_path)
    rui = RecvUI()
    run(stores, s.Outgoing(path, 'project'), SendUI(code_from=rui), rui, str(tmp_path / 'in'))
    stores[1].forget(stores[0].public_key())
    rui2 = RecvUI()
    sui2 = SendUI(code_from=rui2)
    _sent, got = run(stores, s.Outgoing(path, 'project'), sui2, rui2, str(tmp_path / 'in'))
    assert isinstance(got, s.Received), got
    assert sui2.asked == 1


def test_paired_pick_answered_by_another_key_is_refused(stores, tmp_path):
    path, _ = _file(tmp_path)
    impostor = NearbyStore(str(tmp_path / 'c'), 'Phone')
    sent, got = run((stores[0], impostor), s.Outgoing(path, 'project'), SendUI(typed='1'),
                    RecvUI(), str(tmp_path / 'in'), expected=stores[1].public_key())
    assert isinstance(sent, p.ProtocolError) and sent.reason == 'not-paired-device'
    assert not isinstance(got, s.Received)


@pytest.mark.parametrize('case', ['kind', 'too-big', 'no-space', 'user'])
def test_receiver_refusals(stores, tmp_path, case):
    path, data = _file(tmp_path, size=5000)
    rui = RecvUI(accept=case != 'user')
    kind = 'workspace' if case == 'kind' else 'project'
    sent, got = run(stores, s.Outgoing(path, kind), SendUI(code_from=rui), rui,
                    str(tmp_path / 'in'),
                    max_bytes=4999 if case == 'too-big' else None,
                    free=len(data) + s.DISK_MARGIN - 1 if case == 'no-space' else None)
    want = {'kind': p.R_KIND, 'too-big': p.R_TOO_BIG, 'no-space': p.R_NO_SPACE,
            'user': p.R_USER}[case]
    assert isinstance(sent, s.TransferError) and sent.reason == 'declined'
    assert sent.peer_reason == want
    assert isinstance(got, s.TransferError)
    assert not os.path.exists(tmp_path / 'in')


def test_interrupted_transfer_resumes_where_it_stopped(stores, tmp_path):
    path, data = _file(tmp_path, size=400_000)
    rui = RecvUI()
    sui = SendUI(code_from=rui, cancel_after=150_000)
    sent, got = run(stores, s.Outgoing(path, 'project'), sui, rui, str(tmp_path / 'in'))
    assert isinstance(sent, s.TransferError) and sent.reason == 'cancelled'
    assert isinstance(got, s.TransferError)
    part = stores[1].partial_path(stores[0].public_key(), hashlib.sha256(data).digest())
    kept = os.path.getsize(part)
    assert 150_000 <= kept < len(data)
    rui2 = RecvUI()
    sui2 = SendUI()
    _sent2, got2 = run(stores, s.Outgoing(path, 'project'), sui2, rui2, str(tmp_path / 'in'))
    assert isinstance(got2, s.Received), got2
    assert open(got2.path, 'rb').read() == data
    assert sui2.sent[0] > kept                   # nothing before the kept part resent
    assert not os.path.exists(part)


def test_a_bad_kept_part_fails_the_hash_and_is_deleted(stores, tmp_path):
    path, data = _file(tmp_path, size=200_000)
    rui = RecvUI()
    run(stores, s.Outgoing(path, 'fcs'), SendUI(code_from=rui), rui, str(tmp_path / 'x'))
    part = stores[1].partial_path(stores[0].public_key(), hashlib.sha256(data).digest())
    with open(part, 'wb') as fh:
        fh.write(b'\0' * 1000)                   # not the file's first bytes
    sent, got = run(stores, s.Outgoing(path, 'fcs'), SendUI(), RecvUI(), str(tmp_path / 'in'))
    assert isinstance(sent, s.TransferError) and sent.reason == 'peer-rejected'
    assert sent.peer_reason == p.R_HASH
    assert isinstance(got, s.TransferError) and got.reason == 'hash-mismatch'
    assert not os.path.exists(part)
    assert not os.path.exists(tmp_path / 'in' / 'run.flowproject.fcs')


def test_part_longer_than_the_file_starts_over(stores, tmp_path):
    path, data = _file(tmp_path, size=50_000)
    rui = RecvUI()
    run(stores, s.Outgoing(path, 'fcs'), SendUI(code_from=rui), rui, str(tmp_path / 'x'))
    part = stores[1].partial_path(stores[0].public_key(), hashlib.sha256(data).digest())
    with open(part, 'wb') as fh:
        fh.write(b'\0' * 60_000)
    sui = SendUI()
    _sent, got = run(stores, s.Outgoing(path, 'fcs'), sui, RecvUI(), str(tmp_path / 'in'))
    assert isinstance(got, s.Received), got
    assert open(got.path, 'rb').read() == data
    assert sui.sent[0] <= p.DATA_CHUNK


def test_file_changed_while_sending_is_not_finished(stores, tmp_path, monkeypatch):
    path, _ = _file(tmp_path, size=200_000)
    rui = RecvUI()
    calls = []
    real = s.Outgoing.snapshot

    def snap(self):
        calls.append(1)
        size, mtime = real(self)
        return size, mtime + (len(calls) > 1)      # "changed" after the first look
    monkeypatch.setattr(s.Outgoing, 'snapshot', snap)
    sent, got = run(stores, s.Outgoing(path, 'project'), SendUI(code_from=rui), rui,
                    str(tmp_path / 'in'))
    assert isinstance(sent, s.TransferError) and sent.reason == 'file-changed'
    assert isinstance(got, s.TransferError) and got.peer_reason == p.R_HASH
    assert not os.path.exists(tmp_path / 'in')


def test_receiver_of_another_app_refuses_at_hello(stores, tmp_path):
    path, _ = _file(tmp_path)
    sent, got = run(stores, s.Outgoing(path, 'project'), SendUI(typed='1'), RecvUI(),
                    str(tmp_path / 'in'), app_r='pocketlab-gel')
    assert isinstance(sent, p.ProtocolError) and sent.reason == 'wrong-app'
    assert isinstance(got, p.ProtocolError) and got.reason == 'wrong-app'


def test_store_identity_is_kept_and_private(tmp_path):
    a = NearbyStore(str(tmp_path / 'a'), 'Computer')
    pub = a.public_key()
    assert NearbyStore(str(tmp_path / 'a')).public_key() == pub
    if os.name == 'posix':
        assert os.stat(os.path.join(a.root, 'identity.pem')).st_mode & 0o077 == 0
    name = a.device_name()
    assert name.startswith('Computer ') and len(name) == len('Computer ') + 4
    assert NearbyStore(str(tmp_path / 'a'), 'Computer').device_name() == name
    a.set_device_name('  Lab bench  ')
    assert a.device_name() == 'Lab bench'
    a.set_device_name('')
    assert a.device_name() == name


def test_paired_list_drops_damaged_rows(tmp_path):
    a = NearbyStore(str(tmp_path / 'a'))
    other = NearbyStore(str(tmp_path / 'b')).public_key()
    a.add_paired(other, 'Phone')
    with open(os.path.join(a.root, 'paired.json'), 'a', encoding='utf-8') as fh:
        fh.write('')
    data = open(os.path.join(a.root, 'paired.json'), encoding='utf-8').read()
    damaged = data.replace('"devices": [', '"devices": [{"pub": "zz", "name": "x", "at": 1}, ')
    open(os.path.join(a.root, 'paired.json'), 'w', encoding='utf-8').write(damaged)
    assert [d.name for d in a.paired()] == ['Phone']
    a.forget(other)
    assert a.paired() == []


def test_old_and_excess_partials_are_tidied(tmp_path, monkeypatch):
    from openflo.nearby import store as st
    a = NearbyStore(str(tmp_path / 'a'))
    peer = NearbyStore(str(tmp_path / 'b')).public_key()
    old = a.partial_path(peer, b'\1' * 32)
    new1 = a.partial_path(peer, b'\2' * 32)
    new2 = a.partial_path(peer, b'\3' * 32)
    for path, age in ((old, 2 * 86400), (new1, 100), (new2, 10)):
        open(path, 'wb').write(b'x' * 10)
        t = 1_000_000 - age
        os.utime(path, (t, t))
    monkeypatch.setattr(st, 'PARTIAL_TOTAL_CAP', 15)
    a.tidy_partials(now=1_000_000)
    assert not os.path.exists(old)            # over a day old
    assert not os.path.exists(new1)           # oldest of the rest, over the cap
    assert os.path.exists(new2)
    a.tidy_partials(now=1_000_000 + 86400)      # a day on, the newest goes too
    assert not os.path.exists(new2)


@pytest.mark.parametrize('extra_records', [1, 2])
def test_records_arriving_with_the_last_handshake_message_are_kept(stores, extra_records):
    """A sender may write M3 and its first encrypted records before the
    receiver reads: they arrive in one read and must reach the Link, not be
    taken as handshake messages (which refused them, or called them too long)."""
    a, b = socket.socketpair()
    got = {}

    def recv():
        try:
            link = s.handshake_as_receiver(b, 'openflo')
            got['m'] = [link.recv(5) for _ in range(extra_records)]
        except Exception as e:  # noqa: BLE001 - recorded for the test
            got['m'] = e

    th = threading.Thread(target=recv)
    th.start()
    try:
        ini = p.Initiator('openflo')
        a.sendall(ini.first())
        fb = p.FrameBuffer()
        m2 = []
        while not m2:
            m2 = fb.feed(a.recv(4096))
        m3 = ini.on_hello2(m2[0])
        seal = ini.keys.sealer()
        ident = p.make_identity(stores[0].identity(), 'Computer', p.ROLE_I, ini.keys.h)
        records = [p.encode_message(ident)] + [p.encode_message(p.PairState(True))] * (extra_records - 1)
        a.sendall(m3 + b''.join(seal.seal(r) for r in records))   # one write
        th.join(10)
    finally:
        a.close()
        b.close()
    assert isinstance(got['m'], list), got['m']
    assert isinstance(got['m'][0], p.Identity)


def test_an_oversized_handshake_message_is_refused_before_it_is_read():
    a, b = socket.socketpair()
    try:
        a.sendall((10 ** 6).to_bytes(4, 'big'))     # announces 1 MB, sends none of it
        with pytest.raises(p.ProtocolError, match='too-long'):
            s.handshake_as_receiver(b, 'openflo')
    finally:
        a.close()
        b.close()


def test_paired_key_answers_only_for_that_exact_key(tmp_path):
    a = NearbyStore(str(tmp_path / 'a'))
    known = NearbyStore(str(tmp_path / 'b')).public_key()
    stranger = NearbyStore(str(tmp_path / 'c')).public_key()
    a.add_paired(known, 'Phone')
    assert a.paired_key(known).name == 'Phone'
    assert a.paired_key(stranger) is None


def test_a_hand_edited_name_is_trimmed(tmp_path):
    a = NearbyStore(str(tmp_path / 'a'), 'Computer')
    a.device_name()
    import json
    with open(os.path.join(a.root, 'settings.json'), 'w', encoding='utf-8') as fh:
        json.dump({'name': '  Bench  '}, fh)
    assert NearbyStore(str(tmp_path / 'a'), 'Computer').device_name() == 'Bench'


def test_a_forged_identity_signature_ends_it(stores, tmp_path, monkeypatch):
    real = p.make_identity

    def forged(static, name, role, h):
        if role != p.ROLE_I:                       # only the sender forges
            return real(static, name, role, h)
        m = real(static, name, role, h)
        return p.Identity(m.static_pub, m.name, real(static, name, role, bytes(32)).signature)
    monkeypatch.setattr(p, 'make_identity', forged)
    path, _data = _file(tmp_path)
    rui = RecvUI()
    _sent, got = run(stores, s.Outgoing(path, 'project'), SendUI(code_from=rui), rui,
                     str(tmp_path / 'in'))
    assert isinstance(got, p.ProtocolError) and got.reason == 'bad-identity', got
    assert stores[1].paired() == []


def test_a_peer_claiming_this_devices_own_key_is_refused(tmp_path):
    same = NearbyStore(str(tmp_path / 'a'), 'Computer')
    path, _data = _file(tmp_path)
    rui = RecvUI()
    _sent, got = run((same, same), s.Outgoing(path, 'project'), SendUI(code_from=rui), rui,
                     str(tmp_path / 'in'))
    assert isinstance(got, p.ProtocolError) and got.reason == 'bad-identity', got


def test_first_identity_is_made_once_when_asked_from_many_places(tmp_path, monkeypatch):
    """Sending and receiving can start together on first use, and two
    windows (two stores) can share one folder: all must get one key."""
    import time
    real_new = p.new_static_key

    def slow_new():                       # widen the window every caller races in
        time.sleep(0.05)
        return real_new()
    monkeypatch.setattr(p, 'new_static_key', slow_new)
    root = str(tmp_path / 'a')
    shared = NearbyStore(root)
    stores_ = [shared] * 4 + [NearbyStore(root) for _ in range(4)]
    out = []
    start = threading.Barrier(len(stores_))

    def ask(st):
        start.wait()
        try:
            out.append(st.public_key())
        except Exception as e:  # noqa: BLE001 - recorded for the test
            out.append(e)
    ths = [threading.Thread(target=ask, args=(st,)) for st in stores_]
    for th in ths:
        th.start()
    for th in ths:
        th.join(10)
    assert len(set(out)) == 1 and isinstance(out[0], bytes), out
    assert NearbyStore(root).public_key() == out[0]
    assert [f for f in os.listdir(root) if f.endswith('.tmp')] == []


@pytest.mark.parametrize('waiting_for', ['accept', 'confirm'])
def test_cancel_is_noticed_within_a_poll_not_after_the_answer(stores, tmp_path, waiting_for):
    import time
    asking = threading.Event()
    answered = threading.Event()

    class Slow(RecvUI):
        def accept(self, *a):
            if waiting_for == 'accept':
                asking.set()
                time.sleep(4)
                answered.set()
            return True

        def confirm_pairing(self, code, peer_name):
            if waiting_for == 'confirm':
                asking.set()
                time.sleep(4)
                answered.set()
            return True

    class CancelWhileAsked(SendUI):
        def cancelled(self):
            return asking.is_set()
    if waiting_for == 'accept':
        stores[0].add_paired(stores[1].public_key(), 'Phone')
        stores[1].add_paired(stores[0].public_key(), 'Computer')
    path, _data = _file(tmp_path)
    rui = Slow()
    a, b = socket.socketpair()
    th = threading.Thread(target=lambda: _swallow(lambda: s.receive_one(
        b, 'openflo', stores[1], KINDS, str(tmp_path / 'in'), rui)), daemon=True)
    th.start()
    try:
        with pytest.raises(s.TransferError, match='cancelled'):
            s.send_file(a, 'openflo', stores[0], s.Outgoing(path, 'project'),
                        CancelWhileAsked(code_from=rui))
        assert asking.is_set() and not answered.is_set()   # out before the answer came
    finally:
        a.close()
        th.join(10)
        b.close()


def _swallow(fn):
    try:
        fn()
    except Exception:  # noqa: BLE001 - only the sender's side is under test
        pass


def test_room_is_checked_on_the_disk_the_part_grows_on_too(stores, tmp_path, monkeypatch):
    path, _data = _file(tmp_path)
    store_root = stores[1].root
    monkeypatch.setattr(s, '_free_bytes',
                        lambda f: 10 if f == store_root else 10 ** 15)
    rui = RecvUI()
    sent, _got = run(stores, s.Outgoing(path, 'project'), SendUI(code_from=rui), rui,
                    str(tmp_path / 'in'))
    assert isinstance(sent, s.TransferError) and sent.peer_reason == p.R_NO_SPACE, sent


def test_without_hard_links_the_first_saved_key_is_kept(tmp_path, monkeypatch):
    def no_link(*_a):
        raise PermissionError('no hard links here')
    monkeypatch.setattr(os, 'link', no_link)
    root = str(tmp_path / 'a')
    first = NearbyStore(root).public_key()
    assert os.path.exists(os.path.join(root, 'identity.pem'))
    second = NearbyStore(root)
    second._key = None
    pem_before = open(os.path.join(root, 'identity.pem'), 'rb').read()
    second._create_key_file(b'another key')        # a second writer loses
    assert open(os.path.join(root, 'identity.pem'), 'rb').read() == pem_before
    assert second.public_key() == first


def test_a_receiver_that_stalls_for_a_second_does_not_fail_the_send(stores, tmp_path):
    """Waiting for a reply polls Cancel with a short socket timeout; sending
    must not inherit it, or a Wi-Fi hiccup or a slow disk fails the send."""
    import time
    path, data = _file(tmp_path, size=6_000_000)
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16384)
    srv.bind(('127.0.0.1', 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    out = {}

    class StallOnce(RecvUI):
        def progress(self, done, total):
            super().progress(done, total)
            if len(self.got) == 3:
                time.sleep(1.0)
    rui = StallOnce()

    def recv():
        conn, _ = srv.accept()
        try:
            out['r'] = s.receive_one(conn, 'openflo', stores[1], KINDS, str(tmp_path / 'in'), rui)
        except Exception as e:  # noqa: BLE001 - recorded for the test
            out['r'] = e
        finally:
            conn.close()
    th = threading.Thread(target=recv)
    th.start()
    c = socket.socket()
    c.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16384)
    c.connect(('127.0.0.1', port))
    try:
        out['s'] = s.send_file(c, 'openflo', stores[0], s.Outgoing(path, 'project'),
                               SendUI(code_from=rui))
    except Exception as e:  # noqa: BLE001 - recorded for the test
        out['s'] = e
    finally:
        c.close()
        th.join(20)
        srv.close()
    assert isinstance(out['s'], s.Peer), (out['s'], out.get('r'))
    assert open(out['r'].path, 'rb').read() == data
