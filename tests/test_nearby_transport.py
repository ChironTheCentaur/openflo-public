"""Nearby send sockets and discovery (src/openflo/nearby/transport.py,
discovery.py): the listener's address checks, one-at-a-time serving,
backoff, lockout after wrong codes, the pairing abort when a second device
connects, connecting, and the DNS-SD record. Loopback and generated files."""
from __future__ import annotations

import socket
import threading
import time

import pytest

from openflo.nearby import discovery as d
from openflo.nearby import protocol as p
from openflo.nearby import session as s
from openflo.nearby import transport as t
from openflo.nearby.store import NearbyStore
from tests._tk_wait import wait_until

KINDS = {'fcs': '.fcs'}


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _pair():
    a, b = socket.socketpair()
    return a, b


def test_listener_refuses_addresses_off_the_local_network():
    seen = []
    lst = t.Listener(lambda c, h, pr: seen.append(h))
    for host in ('8.8.8.8', '2001:db8::1', '::ffff:8.8.8.8'):
        a, b = _pair()
        assert lst.take(b, host) is None
        assert b.fileno() == -1                 # closed
        a.close()
    assert seen == []
    a, b = _pair()
    th = lst.take(b, '192.168.1.9')
    th.join(5)
    a.close()
    assert seen == ['192.168.1.9']


def test_failing_address_backs_off_and_success_clears_it():
    clock = Clock()
    errors = []
    lst = t.Listener(lambda c, h, pr: None, on_error=errors.append, clock=clock)
    lst._failed('10.0.0.2', s.TransferError('timeout'), False)
    assert not lst.admits('10.0.0.2') and lst.admits('10.0.0.3')
    clock.now += 2.0
    assert lst.admits('10.0.0.2')
    lst._failed('10.0.0.2', s.TransferError('timeout'), False)
    clock.now += 3.9
    assert not lst.admits('10.0.0.2')              # second failure waits 4 s
    clock.now += 0.2
    assert lst.admits('10.0.0.2')
    for _ in range(20):
        lst._failed('10.0.0.2', s.TransferError('timeout'), False)
    clock.now += t.BACKOFF_MAX - 1
    assert not lst.admits('10.0.0.2')
    clock.now += 1.1
    assert lst.admits('10.0.0.2')                  # capped
    a, b = _pair()
    lst.take(b, '10.0.0.2').join(5)
    a.close()
    assert lst.admits('10.0.0.2') and '10.0.0.2' not in lst._fails
    assert len(errors) == 22


def test_three_wrong_codes_within_five_minutes_stop_listening():
    clock = Clock()
    locked = []
    lst = t.Listener(lambda c, h, pr: None, on_lockout=lambda: locked.append(1), clock=clock)
    wrong = s.TransferError('peer-stopped', p.R_WRONG_CODE)
    lst._failed('10.0.0.2', wrong, True)
    lst._failed('10.0.0.3', s.TransferError('timeout'), False)   # not a wrong code
    clock.now += t.LOCKOUT_WINDOW + 1
    lst._failed('10.0.0.4', wrong, True)                         # the first has aged out
    lst._failed('10.0.0.5', wrong, True)
    assert locked == [] and lst.running
    lst._failed('10.0.0.6', wrong, True)
    assert locked == [1] and not lst.running


def test_second_connection_while_busy_is_closed_and_the_first_goes_on():
    release = threading.Event()
    served = []

    def handle(c, h, pairing):
        release.wait(5)
        served.append(h)
    lst = t.Listener(handle)
    a1, b1 = _pair()
    th = lst.take(b1, '10.0.0.1')
    assert lst.busy
    a2, b2 = _pair()
    assert lst.take(b2, '10.0.0.2') is None and b2.fileno() == -1
    release.set()
    th.join(5)
    assert served == ['10.0.0.1'] and not lst.busy
    a1.close()
    a2.close()


def test_second_connection_while_pairing_cuts_the_pairing():
    errors = []

    def handle(c, h, pairing):
        pairing.set()
        c.settimeout(5)
        if not c.recv(10):                      # cut: the read ends
            raise s.TransferError('disconnected')
    lst = t.Listener(handle, on_error=errors.append)
    a1, b1 = _pair()
    th = lst.take(b1, '10.0.0.1')
    assert wait_until(lambda: lst._pairing.is_set(), 5)
    a2, b2 = _pair()
    assert lst.take(b2, '10.0.0.2') is None
    th.join(5)
    assert not th.is_alive()
    assert [e.reason for e in errors] == ['interrupted']
    assert not lst._pairing.is_set() and not lst.busy
    a1.close()
    a2.close()


class _WindowsSocket:
    """A connection as Winsock treats it: shutdown() from another thread
    leaves a blocked recv() waiting; only close() ends it (with OSError)."""

    def __init__(self):
        self.closed = threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def shutdown(self, _how):
        pass

    def close(self):
        self.closed.set()

    def settimeout(self, _t):
        pass

    def recv(self, _n):
        if self.closed.wait(5):
            raise OSError(10038, 'An operation was attempted on something '
                                 'that is not a socket')
        return b''


def test_a_cut_ends_a_read_on_windows_too(monkeypatch):
    """On Windows a second connection while pairing (or Stop) left the read
    of the one being served blocked: shutdown() does not end it there."""
    monkeypatch.setattr(t, '_CUT_CLOSES', True)
    errors = []

    def handle(c, h, pairing):
        pairing.set()
        if not c.recv(10):
            raise s.TransferError('disconnected')
    lst = t.Listener(handle, on_error=errors.append)
    first = _WindowsSocket()
    th = lst.take(first, '10.0.0.1')
    assert wait_until(lambda: lst._pairing.is_set(), 5)
    a2, b2 = _pair()
    assert lst.take(b2, '10.0.0.2') is None
    th.join(2)                                  # well short of the 5 s wait
    assert not th.is_alive()
    assert [e.reason for e in errors] == ['interrupted']
    assert not lst.busy
    a2.close()


def test_send_and_receive_over_a_real_listener(tmp_path):
    sender = NearbyStore(str(tmp_path / 's'))
    receiver = NearbyStore(str(tmp_path / 'r'))
    data = bytes(range(256)) * 400
    src = tmp_path / 'a.fcs'
    src.write_bytes(data)
    got, errors, shown = [], [], threading.Event()
    code = {}

    class RUI(s.ReceiveUI):
        def show_code(self, c, name):
            code['c'] = c
            shown.set()

        def confirm_pairing(self, c, name):
            return True

        def accept(self, *a):
            return True

    class SUI(s.SendUI):
        def enter_code(self, name):
            assert shown.wait(5)
            return code['c']

    def handle(c, h, pairing):
        got.append(s.receive_one(c, 'openflo', receiver, KINDS, str(tmp_path / 'in'), RUI(),
                                 pairing=pairing))
    lst = t.Listener(handle, on_error=errors.append)
    port = lst.start()
    try:
        with t.connect(['127.0.0.1'], port) as sock:
            peer = s.send_file(sock, 'openflo', sender, s.Outgoing(str(src), 'fcs'), SUI())
        assert wait_until(lambda: got or errors, 10)
    finally:
        lst.stop()
    assert errors == []
    assert open(got[0].path, 'rb').read() == data
    assert peer.newly_paired and got[0].peer.newly_paired


def test_a_third_device_during_pairing_stops_it(tmp_path):
    sender = NearbyStore(str(tmp_path / 's'))
    receiver = NearbyStore(str(tmp_path / 'r'))
    src = tmp_path / 'a.fcs'
    src.write_bytes(b'x' * 1000)
    errors, shown = [], threading.Event()

    class RUI(s.ReceiveUI):
        def show_code(self, c, name):
            shown.set()

        def confirm_pairing(self, c, name):
            return True

    class SUI(s.SendUI):
        def enter_code(self, name):
            assert shown.wait(5)
            time.sleep(0.5)                      # the user is still typing
            return

    lst = t.Listener(lambda c, h, pr: s.receive_one(c, 'openflo', receiver, KINDS,
                                                   str(tmp_path / 'in'), RUI(), pairing=pr),
                     on_error=errors.append)
    port = lst.start()
    result = {}

    def send():
        try:
            with t.connect(['127.0.0.1'], port) as sock:
                s.send_file(sock, 'openflo', sender, s.Outgoing(str(src), 'fcs'), SUI())
        except Exception as e:  # noqa: BLE001 - recorded for the test
            result['e'] = e
    th = threading.Thread(target=send)
    th.start()
    try:
        assert shown.wait(5)
        with socket.create_connection(('127.0.0.1', port), timeout=5) as intruder:
            intruder.settimeout(5)
            assert intruder.recv(10) == b''         # turned away
        th.join(10)
        assert wait_until(lambda: errors, 5)
    finally:
        lst.stop()
    assert [e.reason for e in errors] == ['interrupted']
    assert receiver.paired() == [] and sender.paired() == []
    assert isinstance(result.get('e'), (s.TransferError, OSError))


def test_connect_skips_off_network_addresses_and_says_unreachable():
    srv = socket.socket()
    srv.bind(('127.0.0.1', 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        with t.connect(['8.8.8.8', '127.0.0.1'], port, timeout=2) as c:
            assert c.getpeername()[1] == port
    finally:
        srv.close()
    with pytest.raises(s.TransferError) as e:
        t.connect(['8.8.8.8', '127.0.0.1'], port, timeout=2)
    assert e.value.reason == 'unreachable'


# ── discovery ────────────────────────────────────────────────────────────────

def test_found_entries_are_filtered_and_names_made_safe():
    props = {b'v': b'1', b'app': b'openflo', b'id': b'abcdefgh12345678xyz', b'n': b'evil\xe2\x80\xaePC'}
    f = d.found_from('x._pl-send._tcp.local.', props, ['192.168.1.5', '2001:db8::1'], 5000,
                     {'openflo'})
    assert f is not None
    assert f.name == 'evilPC' and f.addresses == ('192.168.1.5',) and f.txt_id == 'abcdefgh12345678'
    assert d.found_from('x', {**props, b'v': b'2'}, ['192.168.1.5'], 5000, {'openflo'}) is None
    assert d.found_from('x', props, ['192.168.1.5'], 5000, {'pocketlab-gel'}) is None
    assert d.found_from('x', props, ['8.8.8.8'], 5000, {'openflo'}) is None
    assert d.found_from('x', props, ['192.168.1.5'], 0, {'openflo'}) is None
    assert d.found_from('x', props, ['192.168.1.5'], 65536, {'openflo'}) is None
    assert d.found_from('x', {'v': '1', 'app': 'openflo'}, ['10.0.0.1'], 1, {'openflo'}).name \
        == 'Unnamed device'


def test_announced_record_and_instance_name():
    key = p.encode_point(p.new_static_key().public_key())
    rec = d.txt_record('openflo', key, 'PC 1')
    assert rec == {'v': '1', 'app': 'openflo', 'id': p.txt_id(key), 'n': 'PC 1'}
    name = d.instance_name('My.PC\x07', key)
    assert name.endswith('._pl-send._tcp.local.')
    assert name.startswith('MyPC ' + p.txt_id(key)[:6])


def _multicast_comes_back(timeout=2.0):
    """Whether a datagram this machine sends to the mDNS group (on a port of
    its own, not 5353) reaches a socket of its own: what two Discovery
    objects on one machine need. Some virtual machines (CI's macOS runners
    among them) never deliver it, though Zeroconf() starts there."""
    import struct
    group = '224.0.0.251'
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        rx.bind(('', 0))
        rx.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                      struct.pack('4s4s', socket.inet_aton(group),
                                  socket.inet_aton('0.0.0.0')))
        rx.settimeout(timeout)
        tx.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        tx.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
        tx.sendto(b'openflo-probe', (group, rx.getsockname()[1]))
        return rx.recv(64) == b'openflo-probe'
    except OSError:
        return False
    finally:
        rx.close()
        tx.close()


def test_announce_and_browse_on_this_machine(monkeypatch):
    from zeroconf import Zeroconf
    try:
        Zeroconf().close()
    except OSError as e:
        pytest.skip(f'no multicast here: {e}')
    if not _multicast_comes_back():
        # Discovery is multicast; with none delivered on this machine there
        # is nothing to see. The parts above it (found_from, txt_record,
        # instance_name) are checked without a network.
        pytest.skip('multicast sent here does not come back to this machine')
    monkeypatch.setattr(d, 'local_addresses', lambda: ['192.168.77.5'])
    key = p.encode_point(p.new_static_key().public_key())
    a, b = d.Discovery(), d.Discovery()
    seen = []
    try:
        a.announce('openflo', key, 'Bench PC', 5555)
        b.browse({'openflo'}, seen.append)
        assert wait_until(lambda: seen and seen[-1], 10)
        f = seen[-1][0]
        assert (f.name, f.app, f.port, f.addresses, f.txt_id) == \
            ('Bench PC', 'openflo', 5555, ('192.168.77.5',), p.txt_id(key))
        a.stop_announcing()
        assert wait_until(lambda: seen[-1] == [], 10)
    finally:
        a.close()
        b.close()


def test_connect_never_tries_an_internet_address(monkeypatch):
    tried = []

    def fake(addr, timeout=None):
        tried.append(addr[0])
        raise OSError('refused')
    monkeypatch.setattr(t.socket, 'create_connection', fake)
    with pytest.raises(s.TransferError):
        t.connect(['8.8.8.8', '2001:4860:4860::8888', '192.168.1.5'], 5000, timeout=1)
    assert tried == ['192.168.1.5']


def test_stop_ends_a_connection_in_progress_and_reports_it_interrupted():
    started, ended = threading.Event(), threading.Event()
    errors, locked = [], []

    def handle(sock, host, pairing):
        pairing.set()
        started.set()
        sock.settimeout(10)
        try:
            data = sock.recv(10)     # cut: ends, or raises OSError on Windows
        finally:
            ended.set()
        if not data:
            raise s.TransferError('disconnected')
    lst = t.Listener(handle, on_error=errors.append, on_lockout=lambda: locked.append(1))
    lst._failed_pairings = [time.monotonic()] * (t.LOCKOUT_FAILS - 1)   # one short of a lockout
    port = lst.start()
    c = socket.create_connection(('127.0.0.1', port))
    try:
        assert started.wait(5)
        lst.stop()
        assert ended.wait(2)
        assert wait_until(lambda: not lst.busy, 5)
    finally:
        c.close()
    assert [e.reason for e in errors] == ['interrupted']
    assert locked == []                          # stopping is not a failed pairing


def _serve_three(lst, raise_exc, set_pairing, clock):
    def handle(sock, host, pairing):
        if set_pairing:
            pairing.set()
        raise raise_exc
    lst._handle = handle
    for i in range(3):
        a, b = _pair()
        lst.take(b, f'192.168.1.{10 + i}').join(5)
        a.close()
        clock.now += 10


@pytest.mark.parametrize('exc, set_pairing, stops', [
    (s.TransferError('cancelled'), True, True),        # refused here while pairing
    (s.TransferError('timeout'), True, True),          # silence while pairing
    (s.TransferError('peer-stopped', p.R_WRONG_CODE), False, True),   # owned-up wrong code
    (s.TransferError('timeout'), False, False),        # a transfer that stalled: not pairing
])
def test_any_failed_pairing_counts_toward_the_lockout(exc, set_pairing, stops):
    clock = Clock()
    lst = t.Listener(lambda c, h, pr: None, clock=clock)
    _serve_three(lst, exc, set_pairing, clock)
    assert lst.running is not stops


def test_pairings_refused_here_through_receive_one_lock_out(tmp_path):
    """The real path: the receiving user says No to three pairings."""
    from tests.test_nearby_session import KINDS, RecvUI, SendUI, _file
    sender = NearbyStore(str(tmp_path / 'a'), 'Computer')
    receiver = NearbyStore(str(tmp_path / 'b'), 'Phone')
    path, _data = _file(tmp_path)
    clock = Clock()
    ruis, sent = [], []

    def handle(sock, host, pairing):
        s.receive_one(sock, 'openflo', receiver, KINDS, str(tmp_path / 'in'),
                      ruis[-1], pairing=pairing)
    lst = t.Listener(handle, clock=clock)
    for i in range(3):
        ruis.append(RecvUI(confirm=False))          # the user here says No
        a, b = _pair()
        th = lst.take(b, f'192.168.1.{10 + i}')
        try:
            s.send_file(a, 'openflo', sender, s.Outgoing(path, 'project'),
                        SendUI(code_from=ruis[-1]))
        except s.TransferError as e:
            sent.append(e.peer_reason)
        a.close()
        th.join(10)
        clock.now += 10
    assert sent == [p.R_USER] * 3
    assert not lst.running and not lst._pairing.is_set()
