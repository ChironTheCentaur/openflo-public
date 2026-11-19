"""Sockets for Nearby send: the receiver's listener and the sender's connect.

The listener takes one connection at a time from local-network addresses
only, slows down an address that keeps failing, and stops after repeated
wrong pairing codes (docs/nearby-send.md, "Limits and abuse").
"""
from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Callable

from . import protocol as p
from .session import TransferError

CONNECT_TIMEOUT = 5.0
BACKOFF_MAX = 300.0
LOCKOUT_FAILS = 3
LOCKOUT_WINDOW = 300.0
# Windows does not end a read blocked in another thread on shutdown(), only
# on close(); elsewhere shutdown() does, and closing there could hand the
# number to a new socket while the reader still holds it.
_CUT_CLOSES = os.name == 'nt'


def is_wrong_code(exc: BaseException) -> bool:
    """A pairing that failed on the code (counts toward the lockout)."""
    return isinstance(exc, TransferError) and exc.peer_reason == p.R_WRONG_CODE


class Listener:
    """Accepts sender connections on a free port, both IPv4 and IPv6 where
    the system allows, and serves one at a time.

    ``handle(sock, host, pairing)`` runs each connection to its end on a
    worker thread; it sets the ``pairing`` event while a code is being
    checked. A connection that arrives while one is served is closed; if
    the served one is pairing, it is cut too (whoever connects first while a
    code is shown also learns the code, so a second device during pairing
    means the pairing cannot be trusted) and its failure is reported as
    TransferError('interrupted').

    ``on_error(exc)`` hears each failed connection; ``on_lockout()`` is
    called (and listening stops) after ``LOCKOUT_FAILS`` failed pairings within
    ``LOCKOUT_WINDOW`` seconds. An address that fails waits 2, 4, 8 ...
    seconds (at most ``BACKOFF_MAX``) before it is taken again."""

    def __init__(self, handle: Callable[[socket.socket, str, threading.Event], None],
                 on_error: Callable[[BaseException], None] = lambda _e: None,
                 on_lockout: Callable[[], None] = lambda: None,
                 clock: Callable[[], float] = time.monotonic):
        self._handle = handle
        self._on_error = on_error
        self._on_lockout = on_lockout
        self._clock = clock
        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._fails: dict[str, tuple[int, float]] = {}
        self._failed_pairings: list[float] = []
        self._lock = threading.Lock()
        self._active: socket.socket | None = None
        self._pairing = threading.Event()
        self._interrupted = False
        self.port = 0

    def start(self) -> int:
        """Bind and start accepting; returns the port."""
        self._sock = _bind_any()
        self._sock.listen(4)
        self._sock.settimeout(0.3)
        self.port = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._loop, name='nearby-listener', daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        """Stop accepting, and end a connection in progress (pairing or
        transfer): nothing is taken once receiving is off. The cut one is
        reported as interrupted; a part already received is kept."""
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        with self._lock:
            if self._active is not None:
                self._interrupted = True
                _cut(self._active)

    @property
    def running(self) -> bool:
        """Whether the listener is still accepting."""
        return not self._stop.is_set()

    @property
    def busy(self) -> bool:
        """Whether a connection is being served now."""
        return self._active is not None

    def _loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                conn, addr = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            self.take(conn, addr[0])

    def take(self, conn: socket.socket, host: str) -> threading.Thread | None:
        """An accepted connection: refused (closed) when the address is not
        admitted or another connection is being served, else served on a
        worker thread, which is returned. The accept loop calls this; tests
        call it directly."""
        if not self.admits(host):
            conn.close()
            return None
        with self._lock:
            if self._active is not None:
                if self._pairing.is_set():
                    self._interrupted = True
                    _cut(self._active)
                conn.close()
                return None
            self._active = conn
            self._interrupted = False
        t = threading.Thread(target=self._serve, args=(conn, host), name='nearby-serve',
                             daemon=True)
        t.start()
        return t

    def _serve(self, conn: socket.socket, host: str) -> None:
        try:
            with conn:
                self._handle(conn, host, self._pairing)
        except Exception as e:  # noqa: BLE001 - every failure is reported and counted; the listener keeps running
            # A pairing that fails for any reason (a wrong code the sender
            # owns up to, a refusal here, silence, a cut) counts toward the
            # lockout: a hostile sender never reports its own wrong code.
            pairing_failed = self._pairing.is_set() or is_wrong_code(e)
            self._failed(host, TransferError('interrupted') if self._interrupted else e,
                         pairing_failed)
        else:
            self._fails.pop(host, None)
        finally:
            self._pairing.clear()
            with self._lock:
                self._active = None

    def admits(self, host: str) -> bool:
        """Whether a connection from ``host`` is taken now: a local-network
        address that is not waiting out a backoff."""
        if not p.is_lan_address(host):
            return False
        _n, until = self._fails.get(host, (0, 0.0))
        return self._clock() >= until

    def _failed(self, host: str, exc: BaseException, pairing_failed: bool) -> None:
        now = self._clock()
        n, _until = self._fails.get(host, (0, 0.0))
        self._fails[host] = (n + 1, now + min(2.0 ** (n + 1), BACKOFF_MAX))
        self._on_error(exc)
        if pairing_failed and not self._stop.is_set():
            self._failed_pairings = [t for t in self._failed_pairings if now - t < LOCKOUT_WINDOW]
            self._failed_pairings.append(now)
            if len(self._failed_pairings) >= LOCKOUT_FAILS:
                self.stop()
                self._on_lockout()


def _cut(sock: socket.socket) -> None:
    """End a connection from another thread (its blocked read returns, or on
    Windows raises OSError)."""
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    if _CUT_CLOSES:
        try:
            sock.close()
        except OSError:
            pass


def _bind_any() -> socket.socket:
    """A TCP socket on a free port: dual-stack IPv6 where possible, else IPv4."""
    if socket.has_ipv6:
        try:
            s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            try:
                s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
                s.bind(('::', 0))
                return s
            except OSError:
                s.close()
        except OSError:
            pass
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('0.0.0.0', 0))
    return s


def connect(addresses: list[str], port: int, timeout: float = CONNECT_TIMEOUT) -> socket.socket:
    """A connection to the first of ``addresses`` that answers (each is a
    local-network address; others are skipped). Raises TransferError
    ('unreachable') when none does: a network with client isolation looks
    like this."""
    for host in addresses:
        if not p.is_lan_address(host):
            continue
        try:
            return socket.create_connection((host, port), timeout=timeout)
        except OSError:
            continue
    raise TransferError('unreachable')
