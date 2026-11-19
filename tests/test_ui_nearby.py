"""File > Nearby send… (src/openflo/ui_nearby.py): the device list, sending
from the window, receiving into it with the code shown and the questions
asked, failures told in words, and nothing left listening once it closes.
A stand-in discovery; real sockets on loopback; generated files only."""
from __future__ import annotations

import os
import threading

import pytest

from openflo import ui_nearby as un
from openflo.nearby import discovery as d
from openflo.nearby import protocol as p
from openflo.nearby import session as s
from openflo.nearby import transport as t
from openflo.nearby.store import NearbyStore
from tests._tk_wait import pump_until
from tests.test_windows_nothing_hidden import _root


class FakeDiscovery:
    """Records what the window announces; lets the test say what is found."""

    def __init__(self):
        self.announced = []
        self.on_change = None
        self.closed = False
        self.stopped = 0

    def announce(self, app, pub, name, port):
        self.announced.append((app, pub, name, port))

    def stop_announcing(self):
        self.stopped += 1

    def browse(self, apps, on_change):
        self.apps = apps
        self.on_change = on_change

    def close(self):
        self.closed = True


@pytest.fixture
def root():
    r = _root()
    try:
        yield r
    finally:
        r.destroy()


@pytest.fixture
def win(root, tmp_path):
    disc = FakeDiscovery()
    w = un.NearbyWindow(root, store=NearbyStore(str(tmp_path / 'me'), 'Computer'),
                        discovery_factory=lambda: disc, receive_folder=str(tmp_path / 'in'))
    w.disc = disc
    yield w
    if w.winfo_exists():
        w.destroy()


def _found(name, txt_id, port=5000):
    return d.Found(f'{name}._pl-send._tcp.local.', name, txt_id, 'openflo', ('127.0.0.1',), port)


def test_devices_found_are_listed_with_their_paired_mark(win, tmp_path):
    other = NearbyStore(str(tmp_path / 'o')).public_key()
    win.store.add_paired(other, 'Phone A')
    assert win.disc.apps == {'openflo'}
    win.disc.on_change([_found('Phone A', p.txt_id(other)), _found('Laptop', 'zzzz')])
    assert pump_until(win, lambda: len(win.devices.get_children()) == 2, 5)
    rows = [(win.devices.item(i, 'text'), win.devices.item(i, 'values')[0])
            for i in win.devices.get_children()]
    assert rows == [('Phone A', 'paired'), ('Laptop', '')]
    assert win.send_status.get() == '2 device(s) found.'
    win.disc.on_change([])
    assert pump_until(win, lambda: not win.devices.get_children(), 5)
    assert 'No devices found' in win.send_status.get()


def test_receive_takes_a_file_after_the_code_and_the_questions(win, tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(win, 'ask_yes_no', lambda title, text: asked.append(text) or True)
    win.toggle_receiving()
    assert win.receiving
    (_app, pub, name, port), = win.disc.announced
    assert pub == win.store.public_key() and name == win.store.device_name()
    data = bytes(range(256)) * 300
    src = tmp_path / 'run.flowproject'
    src.write_bytes(data)
    sender = NearbyStore(str(tmp_path / 'sender'), 'Phone')

    # The sender's thread must not touch Tk, so take the code the window
    # shows from the receive UI as it is handed over.
    shown = []
    real_show = un._TkReceiveUI.show_code

    def show_code(self, code, peer_name):
        shown.append(code)
        real_show(self, code, peer_name)
    monkeypatch.setattr(un._TkReceiveUI, 'show_code', show_code)

    class SUI(s.SendUI):
        def enter_code(self, peer):
            import time
            end = time.monotonic() + 5
            while not shown and time.monotonic() < end:
                time.sleep(0.02)
            return shown[0] if shown else None

    out = {}

    def send():
        try:
            with t.connect(['127.0.0.1'], port) as sock:
                out['peer'] = s.send_file(sock, 'openflo', sender, s.Outgoing(str(src), 'project'),
                                          SUI())
        except Exception as e:  # noqa: BLE001 - recorded for the test
            out['e'] = e
    th = threading.Thread(target=send)
    th.start()
    assert pump_until(win, lambda: win.received or 'e' in out, 15)
    th.join(5)
    assert 'e' not in out, out
    assert win.received
    got = win.received[0]
    assert open(got.path, 'rb').read() == data
    assert os.path.dirname(got.path) == str(tmp_path / 'in')
    code = shown[0]
    assert f'{code[:3]} {code[3:]}' in asked[0]
    assert 'typed on' in asked[0] and 'Code correct' in asked[0]
    assert asked[1].startswith('Accept run.flowproject (77 KB) from ')
    assert 'Received run.flowproject' in win.receive_status.get()
    assert str(win.open_btn.cget('state')) == 'normal'
    assert win.paired_list.get_children()          # the sender is now listed
    win.toggle_receiving()
    assert not win.receiving and win.disc.stopped == 1


def test_send_from_the_window_with_the_code_typed(win, tmp_path, monkeypatch):
    receiver = NearbyStore(str(tmp_path / 'r'), 'Phone')
    shown = {}
    got = []

    class RUI(s.ReceiveUI):
        def show_code(self, code, name):
            shown['code'] = code

        def confirm_pairing(self, code, name):
            return True

        def accept(self, *a):
            return True

    lst = t.Listener(lambda c, h, pr: got.append(
        s.receive_one(c, 'openflo', receiver, {'fcs': '.fcs'}, str(tmp_path / 'r_in'), RUI(),
                      pairing=pr)))
    port = lst.start()
    monkeypatch.setattr(win, 'ask_code', lambda name: shown.get('code'))
    src = tmp_path / 'a.fcs'
    src.write_bytes(b'z' * 70_000)
    win.file_var.set(str(src))
    win.disc.on_change([_found('Phone', p.txt_id(receiver.public_key()), port)])
    assert pump_until(win, lambda: win.devices.get_children(), 5)
    win.devices.selection_set(win.devices.get_children()[0])

    # The code only exists once the receiver shows it: answer the window's
    # question when it does.
    def ask_code(name):
        import time
        end = time.monotonic() + 5
        while 'code' not in shown and time.monotonic() < end:
            time.sleep(0.02)
        return shown.get('code')
    monkeypatch.setattr(win, 'ask_code', ask_code)
    try:
        win._send_clicked()
        assert str(win.send_btn.cget('state')) == 'disabled'
        assert pump_until(win, lambda: not win._sending, 15)
    finally:
        lst.stop()
    assert win.send_status.get().startswith('Sent to Phone')
    assert 'now paired' in win.send_status.get()
    assert open(got[0].path, 'rb').read() == b'z' * 70_000
    assert win.send_progress['value'] == 1000
    assert str(win.send_btn.cget('state')) == 'normal'


def test_send_checks_the_file_before_connecting(win, tmp_path):
    win.disc.on_change([_found('Phone', 'x')])
    assert pump_until(win, lambda: win.devices.get_children(), 5)
    win._send_clicked()
    assert win.send_status.get() == 'Pick a device in the list first.'
    win.devices.selection_set(win.devices.get_children()[0])
    win._send_clicked()
    assert win.send_status.get() == 'Choose a file to send first.'
    other = tmp_path / 'notes.txt'
    other.write_text('x')
    win.file_var.set(str(other))
    win._send_clicked()
    assert 'OpenFlo sends' in win.send_status.get()
    assert not win._sending


def test_send_by_address_refuses_internet_addresses(win, tmp_path, monkeypatch):
    from tkinter import simpledialog
    src = tmp_path / 'a.fcs'
    src.write_bytes(b'1')
    win.file_var.set(str(src))
    monkeypatch.setattr(simpledialog, 'askstring', lambda *a, **k: '8.8.8.8:5000')
    win._send_by_address()
    assert win.send_status.get() == 'That is not a local-network address and port.'
    monkeypatch.setattr(simpledialog, 'askstring', lambda *a, **k: '192.168.1.5:notaport')
    win._send_by_address()
    assert win.send_status.get() == 'That is not a local-network address and port.'
    assert not win._sending


def test_paired_devices_tab_renames_and_forgets(win, tmp_path):
    other = NearbyStore(str(tmp_path / 'o')).public_key()
    win.store.add_paired(other, 'Phone A')
    win._refresh_paired()
    assert [win.paired_list.item(i, 'text') for i in win.paired_list.get_children()] == ['Phone A']
    win.paired_list.selection_set(other.hex())
    win._forget()
    assert win.paired_list.get_children() == () and win.store.paired() == []
    win.name_var.set('  Bench PC ')
    win._save_name()
    assert win.store.device_name() == 'Bench PC' == win.name_var.get()


def test_closing_stops_listening_and_discovery(win):
    win.toggle_receiving()
    lst = win._listener
    assert lst.running
    win.destroy()
    assert not lst.running and win.disc.closed and win.bridge.closed.is_set()


def test_a_question_left_unanswered_gets_the_safe_answer(win):
    import time
    b = un._Bridge()
    start = time.monotonic()
    assert b.ask(lambda: True, default=False, timeout=0.3) is False   # nobody drains
    assert time.monotonic() - start < 2
    b.closed.set()
    assert b.ask(lambda: True, default='no', timeout=5) == 'no'


@pytest.mark.parametrize('exc, words', [
    (s.TransferError('wrong-code'), 'did not match'),
    (s.TransferError('peer-stopped', p.R_WRONG_CODE), 'typed wrong on the other device'),
    (s.TransferError('declined', p.R_NO_SPACE), 'does not have room'),
    (s.TransferError('unreachable'), 'same Wi-Fi'),
    (s.TransferError('interrupted'), 'Another device tried to connect'),
    (p.ProtocolError('not-paired-device'), 'not the device you paired with'),
    (p.ProtocolError('out-of-state'), 'broke the transfer protocol'),
    (ConnectionResetError(104, 'Connection reset'), 'Network error: Connection reset'),
])
def test_failures_are_told_in_words(exc, words):
    assert words in un.describe_failure(exc)


def test_kinds_and_sizes():
    assert un.kind_for_path('/x/Run.FlowProject') == 'project'
    assert un.kind_for_path('a.fcs') == 'fcs' and un.kind_for_path('b.wsp') == 'workspace'
    assert un.kind_for_path('.fcs') is None and un.kind_for_path('a.txt') is None
    assert [un.size_text(n) for n in (5, 5_000, 284_000_000, 2_500_000_000)] == \
        ['5 bytes', '5 KB', '284 MB', '2.5 GB']


def test_the_picked_device_stays_picked_when_another_appears_above_it(win):
    win.disc.on_change([_found('Laptop', 'llll', port=5001)])
    assert pump_until(win, lambda: len(win.devices.get_children()) == 1, 5)
    win.devices.selection_set(win.devices.get_children()[0])
    win.disc.on_change([_found('Android phone', 'aaaa', port=5002),
                        _found('Laptop', 'llll', port=5001)])
    assert pump_until(win, lambda: len(win.devices.get_children()) == 2, 5)
    sel = win.devices.selection()
    assert [win.devices.item(i, 'text') for i in sel] == ['Laptop']
    win.disc.on_change([_found('Android phone', 'aaaa', port=5002)])   # Laptop left
    assert pump_until(win, lambda: len(win.devices.get_children()) == 1, 5)
    win._send_clicked()
    assert win.send_status.get() == 'Pick a device in the list first.' and not win._sending


def test_receiving_does_not_start_again_while_the_last_connection_winds_down(win):
    class Busy:
        running = False
        busy = True

        def stop(self):
            pass
    win._listener = Busy()
    win.toggle_receiving()
    assert win._listener.__class__ is Busy and win.disc.announced == []
    assert 'Still ending the last connection' in win.receive_status.get()
