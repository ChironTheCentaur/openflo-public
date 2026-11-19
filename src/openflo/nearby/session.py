"""One Nearby send connection, end to end, over a connected socket:
handshake, identities, pairing when needed, then one file.

Blocking; run it on a worker thread. The user's part (typing or showing the
code, accepting the file, progress, Cancel) comes through a ``SendUI`` or
``ReceiveUI`` object, whose methods are called on that worker thread.

Pairing (docs/nearby-send.md): the receiver shows the code; the sender's
user types it and the sender compares; on a match the sender confirms, then
the receiver's user confirms. Each side saves the other only after sending
AND receiving Confirm.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import shutil
import socket
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from . import protocol as p
from .store import NearbyStore, move_into_place, running_sha256

HANDSHAKE_TIMEOUT = 10.0
CODE_TIMEOUT = 60.0
IDLE_TIMEOUT = 30.0
DISK_MARGIN = 64 * 1024 ** 2
CANCEL_POLL = 0.25         # seconds between Cancel checks while waiting on the peer


class TransferError(Exception):
    """The connection or transfer ended without a file. ``reason`` is a
    fixed word the UI turns into a message; ``peer_reason`` is the reason
    code the other side sent, if it stopped the transfer."""

    def __init__(self, reason: str, peer_reason: int | None = None):
        super().__init__(reason)
        self.reason = reason
        self.peer_reason = peer_reason


@dataclass
class Peer:
    """The device at the other end, after its identity checked out.
    ``name`` is what the user is shown: the label stored when it paired, or
    else its sent name made safe (protocol.safe_device_name)."""
    static_pub: bytes
    name: str
    newly_paired: bool = False


class SendUI:
    """The sending user's side. Override what the window shows."""

    def enter_code(self, peer_name: str) -> str | None:
        """The 6 digits the user read off the other device; None = cancel."""
        return None

    def waiting_for_peer_confirm(self, peer_name: str) -> None:
        """Code matched: tell the user to confirm on the other device."""

    def progress(self, done: int, total: int) -> None:
        """Bytes sent so far."""

    def cancelled(self) -> bool:
        """True once the user pressed Cancel."""
        return False


class ReceiveUI:
    """The receiving user's side. Override what the window shows."""

    def show_code(self, code: str, peer_name: str) -> None:
        """Show the 6-digit code for the sender's user to type."""

    def confirm_pairing(self, code: str, peer_name: str) -> bool:
        """The sender typed the right code: does this user confirm? (Ask with
        the code repeated, and only if the device in their hand says so.)"""
        return False

    def accept(self, name: str, size: int, kind: str, peer_name: str) -> bool:
        """Ask whether to take this file (always asked, paired or not)."""
        return False

    def progress(self, done: int, total: int) -> None:
        """Bytes received so far (including a resumed part)."""

    def cancelled(self) -> bool:
        """True once the user pressed Cancel."""
        return False


class Link:
    """Encrypted messages over the socket once the handshake is done."""

    def __init__(self, sock: socket.socket, keys: p.SessionKeys, role: bytes, rest: bytes = b'',
                 cancelled: Callable[[], bool] = lambda: False):
        self.sock = sock
        self.cancelled = cancelled   # asked while waiting, so Cancel needs no reply first
        self.order = p.Session(role)   # refuses messages out of order
        self.h = keys.h          # transcript hash, signed by both identities
        self.sas = keys.sas      # the pairing code
        self._seal = keys.sealer()
        self._open = keys.opener()
        self._queue: list[bytes] = self._open.feed(rest) if rest else []

    def send(self, m: p.Message) -> None:
        """Encrypt and send one message. A receiver that stops reading for a
        while (a Wi-Fi hiccup, a slow disk) is waited for up to IDLE_TIMEOUT,
        whatever timeout a wait for a reply left on the socket."""
        self.sock.settimeout(IDLE_TIMEOUT)
        self.sock.sendall(self._seal.seal(p.encode_message(m)))
        self.order.sent(m)

    def recv(self, timeout: float = IDLE_TIMEOUT) -> p.Message:
        """The next message; a closed socket or silence past ``timeout`` raises."""
        deadline = time.monotonic() + timeout
        while not self._queue:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TransferError('timeout')
            if self.cancelled():
                self.abort(p.R_USER)
                raise TransferError('cancelled')
            self.sock.settimeout(min(left, CANCEL_POLL))
            try:
                data = self.sock.recv(1 << 16)
            except TimeoutError:
                continue                 # asks about Cancel again, then waits on
            except OSError:
                raise TransferError('disconnected') from None
            if not data:
                raise TransferError('disconnected')
            self._queue = self._open.feed(data)
        return self.order.received(p.decode_message(self._queue.pop(0)))

    def expect(self, *types: type, timeout: float = IDLE_TIMEOUT) -> p.Message:
        """The next message, which must be one of ``types``. An Abort or
        Decline from the peer raises with its reason; anything else is a
        protocol error."""
        m = self.recv(timeout)
        if isinstance(m, p.Abort) and p.Abort not in types:
            raise TransferError('peer-stopped', m.reason)
        if isinstance(m, p.Decline) and p.Decline not in types:
            raise TransferError('declined', m.reason)
        if not isinstance(m, types):
            raise p.ProtocolError('out-of-state', type(m).__name__)
        return m

    def abort(self, reason: int) -> None:
        """Tell the peer we stop (best effort; the link may be gone)."""
        try:
            self.send(p.Abort(reason))
        except OSError:
            pass


class _HandshakeReader:
    """Reads one length-prefixed handshake message at a time and keeps the
    bytes after it unparsed: the peer's first encrypted records can arrive
    in the same read as its last handshake message, and they belong to the
    Link, not to the handshake."""

    def __init__(self, sock: socket.socket, deadline: float):
        self.sock = sock
        self.deadline = deadline
        self.buf = bytearray()

    def message(self) -> bytes:
        """The next handshake message body; refuses one over the limit
        before reading it."""
        while True:
            if len(self.buf) >= 4:
                (n,) = struct.unpack_from('>I', self.buf)
                if n > p.MAX_HANDSHAKE_BODY:
                    raise p.ProtocolError('too-long', f'handshake message of {n} bytes')
                if len(self.buf) >= 4 + n:
                    body = bytes(self.buf[4:4 + n])
                    del self.buf[:4 + n]
                    return body
            self.buf += _recv_some(self.sock, self.deadline)

    def rest(self) -> bytes:
        """Bytes after the last message taken (records, handed to the Link)."""
        return bytes(self.buf)


def _recv_some(sock: socket.socket, deadline: float) -> bytes:
    left = deadline - time.monotonic()
    if left <= 0:
        raise TransferError('timeout')
    sock.settimeout(left)
    try:
        data = sock.recv(4096)
    except TimeoutError:
        raise TransferError('timeout') from None
    except OSError:
        raise TransferError('disconnected') from None
    if not data:
        raise TransferError('disconnected')
    return data


def handshake_as_sender(sock: socket.socket, app: str) -> Link:
    """Run M1-M3 as the connecting side."""
    deadline = time.monotonic() + HANDSHAKE_TIMEOUT
    ini = p.Initiator(app)
    sock.sendall(ini.first())
    rd = _HandshakeReader(sock, deadline)
    sock.sendall(ini.on_hello2(rd.message()))
    assert ini.keys is not None
    return Link(sock, ini.keys, p.ROLE_I, rd.rest())


def handshake_as_receiver(sock: socket.socket, app: str) -> Link:
    """Run M1-M3 as the listening side (fresh keys for this connection)."""
    deadline = time.monotonic() + HANDSHAKE_TIMEOUT
    res = p.Responder(app)
    rd = _HandshakeReader(sock, deadline)
    try:
        sock.sendall(res.on_hello1(rd.message()))
    except p.ProtocolError as e:
        if e.reply:
            sock.sendall(e.reply)
        raise
    keys = res.on_hello3(rd.message())
    return Link(sock, keys, p.ROLE_R, rd.rest())


def _identities(link: Link, store: NearbyStore, role: bytes) -> tuple[Peer, bool, bool]:
    """Exchange Identity and PairState; (peer, I know it, it knows me)."""
    h = link.h
    link.send(p.make_identity(store.identity(), store.device_name(), role, h))
    peer = link.expect(p.Identity)
    assert isinstance(peer, p.Identity)
    p.verify_identity(peer, p.ROLE_R if role == p.ROLE_I else p.ROLE_I, h)
    if peer.static_pub == store.public_key():
        raise p.ProtocolError('bad-identity', 'the peer claims this device\'s own key')
    stored = store.paired_key(peer.static_pub)
    link.send(p.PairState(stored is not None))
    state = link.expect(p.PairState)
    assert isinstance(state, p.PairState)
    shown = stored.name if stored is not None else p.safe_device_name(peer.name)
    return Peer(peer.static_pub, shown), stored is not None, state.known


# ── sending ──────────────────────────────────────────────────────────────────

@dataclass
class Outgoing:
    """A file to send: path, kind, and the name the receiver is shown."""
    path: str
    kind: str
    name: str = ''
    _stat: tuple[int, int] = field(default=(0, 0), repr=False)

    def snapshot(self) -> tuple[int, int]:
        """(size, mtime_ns) now: compared before and after, so a file that
        changes mid-send is caught, not sent half old, half new."""
        st = os.stat(self.path)
        return st.st_size, st.st_mtime_ns


def send_file(sock: socket.socket, app: str, store: NearbyStore, out: Outgoing,
              ui: SendUI, expected_pub: bytes | None = None) -> Peer:
    """Send one file to the receiver on ``sock``. ``expected_pub``: the
    stored key of the paired device the user picked (a different key is
    refused). Returns the peer; raises TransferError or ProtocolError."""
    link = handshake_as_sender(sock, app)
    link.cancelled = ui.cancelled
    peer, i_know, they_know = _identities(link, store, p.ROLE_I)
    p.check_paired_key(expected_pub, peer.static_pub)
    if p.needs_pairing(i_know, they_know):
        _pair_as_sender(link, store, peer, ui)
        peer.newly_paired = True
    _offer_and_stream(link, out, ui)
    return peer


def _pair_as_sender(link: Link, store: NearbyStore, peer: Peer, ui: SendUI) -> None:
    typed = ui.enter_code(peer.name)
    if typed is None:
        link.abort(p.R_USER)
        raise TransferError('cancelled')
    if not hmac.compare_digest(typed.strip().encode(), link.sas.encode()):
        link.abort(p.R_WRONG_CODE)
        raise TransferError('wrong-code')
    link.send(p.Confirm())
    ui.waiting_for_peer_confirm(peer.name)
    link.expect(p.Confirm, timeout=CODE_TIMEOUT)
    store.add_paired(peer.static_pub, peer.name)


def _offer_and_stream(link: Link, out: Outgoing, ui: SendUI) -> None:
    before = out.snapshot()
    digest = running_sha256(out.path).digest()
    size = before[0]
    name = out.name or os.path.basename(out.path)
    link.send(p.Offer(os.urandom(16), size, digest, out.kind, name[:p.MAX_FILE_NAME]))
    reply = link.expect(p.Accept, timeout=CODE_TIMEOUT + IDLE_TIMEOUT)
    assert isinstance(reply, p.Accept)
    p.check_offset(reply.offset, size)
    sent = reply.offset
    with open(out.path, 'rb') as fh:
        fh.seek(sent)
        while sent < size:
            if ui.cancelled():
                link.abort(p.R_USER)
                raise TransferError('cancelled')
            chunk = fh.read(min(p.DATA_CHUNK, size - sent))
            if not chunk:
                break
            link.send(p.Data(chunk))
            sent += len(chunk)
            ui.progress(sent, size)
    if sent != size or out.snapshot() != before:
        link.abort(p.R_HASH)
        raise TransferError('file-changed')
    link.send(p.Done())
    res = link.expect(p.Result, timeout=IDLE_TIMEOUT + size / 50e6)
    assert isinstance(res, p.Result)
    try:
        link.send(p.Close())
    except OSError:
        pass
    if not res.ok:
        raise TransferError('peer-rejected', res.reason)


# ── receiving ────────────────────────────────────────────────────────────────

@dataclass
class Received:
    """A file that arrived and checked out."""
    path: str
    kind: str
    peer: Peer


def receive_one(sock: socket.socket, app: str, store: NearbyStore, kinds: dict[str, str],
                folder: str, ui: ReceiveUI, max_bytes: int | None = None,
                free_bytes: Callable[[str], int] | None = None,
                pairing: threading.Event | None = None) -> Received:
    """Take one file from the sender on ``sock`` into ``folder``. ``kinds``
    maps each kind this app takes to its extension. ``pairing`` is set
    while a code is being checked: the listener aborts the pairing when
    another device connects meanwhile. It is left set when the pairing
    fails, so the listener counts that toward its lockout; the caller
    clears it. Raises TransferError or
    ProtocolError; a part already received is kept for a resume."""
    link = handshake_as_receiver(sock, app)
    link.cancelled = ui.cancelled
    peer, i_know, they_know = _identities(link, store, p.ROLE_R)
    if p.needs_pairing(i_know, they_know):
        if pairing is not None:
            pairing.set()
        _pair_as_receiver(link, store, peer, ui)
        # Cleared on success only: a pairing that fails leaves it set, so the
        # caller (the listener) sees the failure was a pairing and counts it
        # toward the lockout, then clears it itself.
        if pairing is not None:
            pairing.clear()
        peer.newly_paired = True
    offer = link.expect(p.Offer, timeout=CODE_TIMEOUT)
    assert isinstance(offer, p.Offer)
    ext = kinds.get(offer.kind)
    # The part grows under the store and is then moved into the folder: both
    # disks need the room (they are often the same one).
    free = (free_bytes(folder) if free_bytes is not None
            else min(_free_bytes(folder), _free_bytes(store.root)))
    refusal = (p.R_KIND if ext is None
               else p.R_TOO_BIG if max_bytes is not None and offer.size > max_bytes
               else p.R_NO_SPACE if offer.size + DISK_MARGIN > free else None)
    if refusal is not None:
        link.send(p.Decline(refusal))
        raise TransferError('refused', refusal)
    assert ext is not None
    if not ui.accept(p.safe_file_name(offer.name, ext), offer.size, offer.kind, peer.name):
        link.send(p.Decline(p.R_USER))
        raise TransferError('declined-here')
    store.tidy_partials()
    part = store.partial_path(peer.static_pub, offer.sha256)
    digest, have = _resume_point(part, offer.size)
    link.send(p.Accept(have))
    _receive_bytes(link, part, offer, digest, have, ui)
    saved = move_into_place(part, folder, p.safe_file_name(offer.name, ext))
    link.send(p.Result(True, 0))
    try:
        link.expect(p.Close, timeout=5)
    except (TransferError, p.ProtocolError, OSError):
        pass
    return Received(saved, offer.kind, peer)


def _pair_as_receiver(link: Link, store: NearbyStore, peer: Peer, ui: ReceiveUI) -> None:
    ui.show_code(link.sas, peer.name)
    link.expect(p.Confirm, timeout=CODE_TIMEOUT)
    if not ui.confirm_pairing(link.sas, peer.name):
        link.abort(p.R_USER)
        raise TransferError('cancelled')
    link.send(p.Confirm())
    store.add_paired(peer.static_pub, peer.name)


def _resume_point(part: str, size: int):
    """(running hash, bytes already here) for a kept partial file; a part
    longer than the file is stale and starts over."""
    try:
        have = os.path.getsize(part)
    except OSError:
        have = 0
    if have > size:
        os.remove(part)
        have = 0
    return (running_sha256(part, have) if have else hashlib.sha256()), have


def _receive_bytes(link: Link, part: str, offer: p.Offer, digest, have: int,
                   ui: ReceiveUI) -> None:
    with open(part, 'ab') as fh:
        fh.truncate(have)
        while True:
            if ui.cancelled():
                link.abort(p.R_USER)
                raise TransferError('cancelled')
            m = link.expect(p.Data, p.Done)
            if isinstance(m, p.Done):
                break
            assert isinstance(m, p.Data)
            if have + len(m.payload) > offer.size:
                link.abort(p.R_HASH)
                _discard(part)
                raise TransferError('too-much-data')
            fh.write(m.payload)
            digest.update(m.payload)
            have += len(m.payload)
            ui.progress(have, offer.size)
    if have != offer.size or not hmac.compare_digest(digest.digest(), offer.sha256):
        _discard(part)
        link.send(p.Result(False, p.R_HASH))
        raise TransferError('hash-mismatch')


def _discard(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _free_bytes(folder: str) -> int:
    probe = folder
    while probe and not os.path.isdir(probe):
        probe = os.path.dirname(probe)
    return shutil.disk_usage(probe or '.').free
