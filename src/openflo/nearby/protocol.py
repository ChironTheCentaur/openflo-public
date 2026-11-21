"""Nearby send wire format and cryptography, version 1 (docs/nearby-send.md).

Sans-IO: every function here takes and returns bytes; sockets, threads,
discovery and files live elsewhere. Nothing here imports OpenFlo, so the
other Pocket Lab apps (and the Kotlin and Swift twins, checked against
``vectors/v1.json``) share one definition.

Handshake (I connects, R listens), each message length-prefixed:

    M1 I->R  "OFNS" | ver | app | C = SHA-256("pl-send v1 commit" | eI | nI)
    M2 R->I  status 0 | eR | nR          (fresh for every connection)
    M3 I->R  eI | nI                     (R checks C)

then ``h = SHA-256(framed M1 | framed M2 | framed M3)``, ``ss`` = the P-256
ECDH x-coordinate, two 32-byte record keys and the 6-digit code from HKDF
over (ss, h). The code depends on M1-M3 and ss only: nothing sent later
(static keys, names) may ever be mixed into it, or a man in the middle could
pick those until the codes match.

Records after the handshake: ``len (4 bytes BE) | AES-256-GCM(plaintext)``,
the 4 length bytes as associated data, nonce = 4 zero bytes | 8-byte BE
counter per direction. The first plaintext byte is the message type.
"""
from __future__ import annotations

import base64
import hashlib
import ipaddress
import os
import re
import struct
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

VERSION = 1
MAGIC = b'OFNS'
# DNS-SD type shared by every Pocket Lab app; the TXT record's `app` says
# which app is listening.
SERVICE_TYPE = '_pl-send._tcp.local.'

POINT_LEN = 65              # uncompressed P-256: 0x04 | x | y
NONCE_LEN = 32
HASH_LEN = 32
TAG_LEN = 16
MAX_PLAINTEXT = 64 * 1024
MAX_RECORD = MAX_PLAINTEXT + TAG_LEN          # 65,552: the largest len field
MAX_HANDSHAKE_BODY = 256
MAX_COUNTER = 2 ** 32       # records per direction before the link must end

MAX_APP = 32
MAX_DEVICE_NAME = 64        # UTF-8 bytes
MAX_KIND = 32
MAX_FILE_NAME = 1024        # as sent; the receiver cuts it to SAFE_NAME_BYTES
SAFE_NAME_BYTES = 200       # UTF-8 bytes of a saved name, extension included
# Sizes and offsets are u64 on the wire but must fit a signed 64-bit integer
# (Kotlin Long, Swift Int64), or a size check there sees a negative number.
MAX_SIZE = 2 ** 63 - 1
MAX_SIGNATURE = 80          # DER ECDSA P-256 is at most 72 bytes

COMMIT_LABEL = b'pl-send v1 commit'
KEYS_INFO = b'pl-send v1 keys'
SAS_INFO = b'pl-send v1 sas'
ID_LABEL = b'pl-send v1 id'
ROLE_I = b'I'
ROLE_R = b'R'

# M2 status byte.
HELLO_OK = 0
HELLO_UNSUPPORTED = 1       # followed by the highest version R speaks
HELLO_WRONG_APP = 2

_APP_RE = re.compile(r'[a-z0-9][a-z0-9-]{0,31}')


class ProtocolError(ValueError):
    """The peer broke the protocol, or a check failed: end the connection.

    ``reason`` is a short fixed word the UI maps to a message (never the
    peer's own text)."""

    def __init__(self, reason: str, detail: str = '', max_version: int | None = None,
                 reply: bytes | None = None):
        super().__init__(f'{reason}: {detail}' if detail else reason)
        self.reason = reason
        self.max_version = max_version
        # Bytes to send the peer before closing (a refused hello), if any.
        self.reply = reply


# ── encoding helpers ─────────────────────────────────────────────────────────

def frame(body: bytes) -> bytes:
    """A handshake message as it goes on the wire: 4-byte BE length, body."""
    return struct.pack('>I', len(body)) + body


class FrameBuffer:
    """Splits handshake bytes from the socket into message bodies; refuses a
    length above ``limit`` before reading it."""

    def __init__(self, limit: int = MAX_HANDSHAKE_BODY):
        self._buf = bytearray()
        self._limit = limit

    def feed(self, data: bytes) -> list[bytes]:
        """Add bytes from the socket; return the whole message bodies now complete."""
        self._buf += data
        out = []
        while len(self._buf) >= 4:
            (n,) = struct.unpack_from('>I', self._buf)
            if n > self._limit:
                raise ProtocolError('too-long', f'handshake message of {n} bytes')
            if len(self._buf) < 4 + n:
                break
            out.append(bytes(self._buf[4:4 + n]))
            del self._buf[:4 + n]
        return out

    def rest(self) -> bytes:
        """Bytes after the last whole message (records that followed the
        handshake in the same read)."""
        out = bytes(self._buf)
        self._buf.clear()
        return out


class _Reader:
    def __init__(self, data: bytes):
        self._d = data
        self._i = 0

    def take(self, n: int) -> bytes:
        if n < 0 or self._i + n > len(self._d):
            raise ProtocolError('malformed', 'message ends early')
        out = self._d[self._i:self._i + n]
        self._i += n
        return out

    def u8(self) -> int:
        return self.take(1)[0]

    def u16(self) -> int:
        return struct.unpack('>H', self.take(2))[0]

    def size(self) -> int:
        n = struct.unpack('>Q', self.take(8))[0]
        if n > MAX_SIZE:
            raise ProtocolError('malformed', 'size or offset of 2^63 or more')
        return n

    def text(self, limit: int) -> str:
        n = self.u16()
        if n > limit:
            raise ProtocolError('malformed', 'text too long')
        try:
            return self.take(n).decode('utf-8')
        except UnicodeDecodeError:
            raise ProtocolError('malformed', 'text is not UTF-8') from None

    def end(self) -> None:
        if self._i != len(self._d):
            raise ProtocolError('malformed', 'extra bytes after message')


def _text(s: str, limit: int) -> bytes:
    b = s.encode('utf-8')
    if len(b) > limit:
        raise ValueError(f'text longer than {limit} bytes')
    return struct.pack('>H', len(b)) + b


# ── keys ─────────────────────────────────────────────────────────────────────

def encode_point(pub: ec.EllipticCurvePublicKey) -> bytes:
    """A public key as its 65-byte uncompressed point."""
    return pub.public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)


def decode_point(data: bytes) -> ec.EllipticCurvePublicKey:
    """A peer's P-256 point: exactly 65 uncompressed bytes, on the curve
    (the library refuses off-curve points; infinity has no 65-byte form)."""
    if len(data) != POINT_LEN or data[0] != 4:
        raise ProtocolError('bad-point', 'not an uncompressed P-256 point')
    try:
        return ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), data)
    except ValueError:
        raise ProtocolError('bad-point', 'point is not on the curve') from None


def fingerprint(static_pub: bytes) -> bytes:
    """A device's identity: SHA-256 of its 65-byte public key."""
    return hashlib.sha256(static_pub).digest()


def txt_id(static_pub: bytes) -> str:
    """The short id announced in the DNS-SD TXT record (first 8 bytes of the
    fingerprint, base32, lower case, no padding)."""
    return base64.b32encode(fingerprint(static_pub)[:8]).decode().rstrip('=').lower()


def new_static_key() -> ec.EllipticCurvePrivateKey:
    """A new long-term identity key pair for this install."""
    return ec.generate_private_key(ec.SECP256R1())


@dataclass(frozen=True)
class Ephemeral:
    """One connection's key pair and random nonce. Made fresh for every
    connection (reuse lets an attacker grind the code offline)."""
    key: ec.EllipticCurvePrivateKey
    nonce: bytes

    @property
    def pub(self) -> bytes:
        """This connection's public key, 65 bytes."""
        return encode_point(self.key.public_key())

    @classmethod
    def fresh(cls, rand: Callable[[int], bytes] = os.urandom) -> Ephemeral:
        """A new random key pair and nonce (every connection)."""
        return cls(ec.generate_private_key(ec.SECP256R1()), rand(NONCE_LEN))

    @classmethod
    def fixed(cls, scalar: int, nonce: bytes) -> Ephemeral:
        """For test vectors only."""
        return cls(ec.derive_private_key(scalar, ec.SECP256R1()), nonce)


def commitment(e_pub: bytes, nonce: bytes) -> bytes:
    """C in M1: binds the initiator to its key and nonce before it sees R's."""
    return hashlib.sha256(COMMIT_LABEL + e_pub + nonce).digest()


def _hkdf(ikm: bytes, salt: bytes, info: bytes, n: int) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=n, salt=salt, info=info).derive(ikm)


@dataclass(frozen=True)
class SessionKeys:
    """What a finished handshake gives one side."""
    h: bytes            # transcript hash, signed by both identities
    send_key: bytes
    recv_key: bytes
    sas: str            # the 6-digit code shown while pairing

    def sealer(self) -> RecordSealer:
        """Encrypts the records this side sends."""
        return RecordSealer(self.send_key)

    def opener(self) -> RecordOpener:
        """Decrypts the records this side receives."""
        return RecordOpener(self.recv_key)


def derive(ss: bytes, h: bytes) -> tuple[bytes, bytes, str]:
    """(k_IR, k_RI, code) from the ECDH secret and the transcript hash."""
    okm = _hkdf(ss, h, KEYS_INFO, 64)
    sas_bytes = _hkdf(ss, h, SAS_INFO, 4)
    code = struct.unpack('>I', sas_bytes)[0] % 1_000_000
    return okm[:32], okm[32:], f'{code:06d}'


def _ecdh(e: Ephemeral, peer: ec.EllipticCurvePublicKey) -> bytes:
    return e.key.exchange(ec.ECDH(), peer)


def transcript(m1: bytes, m2: bytes, m3: bytes) -> bytes:
    """h: SHA-256 of the three handshake messages, length prefixes included."""
    return hashlib.sha256(frame(m1) + frame(m2) + frame(m3)).digest()


# ── handshake ────────────────────────────────────────────────────────────────

def _check_app(app: str) -> None:
    if not _APP_RE.fullmatch(app):
        raise ValueError(f'app id {app!r}: lower-case letters, digits and -')


def hello1(app: str, commit: bytes, version: int = VERSION) -> bytes:
    """M1 body: magic, version, app id, commitment."""
    _check_app(app)
    a = app.encode()
    return MAGIC + bytes([version, len(a)]) + a + commit


def parse_hello1(body: bytes) -> tuple[int, str, bytes]:
    """(version, app, commitment) from an M1 body."""
    r = _Reader(body)
    if r.take(4) != MAGIC:
        raise ProtocolError('not-pl-send', 'wrong magic')
    version = r.u8()
    n = r.u8()
    if n > MAX_APP:
        raise ProtocolError('malformed', 'app id too long')
    app = r.take(n).decode('ascii', 'replace')
    commit = r.take(HASH_LEN)
    r.end()
    return version, app, commit


def hello2(e: Ephemeral) -> bytes:
    """M2 body: status OK, R's ephemeral key and nonce."""
    return bytes([HELLO_OK]) + e.pub + e.nonce


def hello2_refusal(status: int, max_version: int = VERSION) -> bytes:
    """M2 body that refuses: unsupported version (with the highest R speaks) or wrong app."""
    if status == HELLO_UNSUPPORTED:
        return bytes([status, max_version])
    return bytes([status])


def parse_hello2(body: bytes) -> tuple[bytes, bytes]:
    """(eR, nR) from an M2 body; a refusal raises with its reason."""
    r = _Reader(body)
    status = r.u8()
    if status == HELLO_UNSUPPORTED:
        highest = r.u8()
        r.end()
        raise ProtocolError('unsupported-version', max_version=highest)
    if status == HELLO_WRONG_APP:
        r.end()
        raise ProtocolError('wrong-app')
    if status != HELLO_OK:
        raise ProtocolError('malformed', 'unknown hello status')
    pub, nonce = r.take(POINT_LEN), r.take(NONCE_LEN)
    r.end()
    return pub, nonce


def hello3(e: Ephemeral) -> bytes:
    """M3 body: I's ephemeral key and nonce (the reveal)."""
    return e.pub + e.nonce


def parse_hello3(body: bytes) -> tuple[bytes, bytes]:
    """(eI, nI) from an M3 body."""
    r = _Reader(body)
    pub, nonce = r.take(POINT_LEN), r.take(NONCE_LEN)
    r.end()
    return pub, nonce


class Initiator:
    """The sending side's handshake: ``first()``, then ``on_hello2(M2)`` gives
    M3, after which ``keys`` is set."""

    def __init__(self, app: str, eph: Ephemeral | None = None):
        self._e = eph or Ephemeral.fresh()
        self._m1 = hello1(app, commitment(self._e.pub, self._e.nonce))
        self.keys: SessionKeys | None = None

    def first(self) -> bytes:
        """M1, framed, to send first."""
        return frame(self._m1)

    def on_hello2(self, m2: bytes) -> bytes:
        """Take M2; return M3 (framed) and set ``keys``."""
        if self.keys is not None:
            raise ProtocolError('out-of-state', 'second hello')
        pub, _nonce = parse_hello2(m2)
        peer = decode_point(pub)
        m3 = hello3(self._e)
        h = transcript(self._m1, m2, m3)
        k_ir, k_ri, sas = derive(_ecdh(self._e, peer), h)
        self.keys = SessionKeys(h, k_ir, k_ri, sas)
        return frame(m3)


class Responder:
    """The receiving side's handshake: ``on_hello1(M1)`` gives M2 (or a
    refusal, then raises), ``on_hello3(M3)`` checks the commitment and sets
    ``keys``. Make one per connection: its ephemeral key is never reused."""

    def __init__(self, app: str, eph: Ephemeral | None = None):
        _check_app(app)
        self._app = app
        self._e = eph or Ephemeral.fresh()
        self._m1: bytes | None = None
        self._commit = b''
        self._m2 = b''
        self._spent = False
        self.keys: SessionKeys | None = None

    def on_hello1(self, m1: bytes) -> bytes:
        """M2 to send. On a version or app this side does not take, the
        refusal is in ``ProtocolError.reply`` (send it, then close)."""
        if self._spent:
            raise ProtocolError('out-of-state', 'second hello')
        self._spent = True          # one M1 per Responder, refused or not
        version, app, commit = parse_hello1(m1)
        if version != VERSION:
            raise ProtocolError('unsupported-version', max_version=VERSION,
                                reply=frame(hello2_refusal(HELLO_UNSUPPORTED)))
        if app != self._app:
            raise ProtocolError('wrong-app', reply=frame(hello2_refusal(HELLO_WRONG_APP)))
        self._m1, self._commit = m1, commit
        self._m2 = hello2(self._e)
        return frame(self._m2)

    def on_hello3(self, m3: bytes) -> SessionKeys:
        """Take M3: check the commitment and the point; return the keys."""
        if self._m1 is None or self.keys is not None:
            raise ProtocolError('out-of-state', 'hello 3 out of order')
        pub, nonce = parse_hello3(m3)
        if commitment(pub, nonce) != self._commit:
            raise ProtocolError('bad-commit', 'key does not match its commitment')
        peer = decode_point(pub)
        h = transcript(self._m1, self._m2, m3)
        k_ir, k_ri, sas = derive(_ecdh(self._e, peer), h)
        self.keys = SessionKeys(h, k_ri, k_ir, sas)
        return self.keys


# ── records ──────────────────────────────────────────────────────────────────

def _nonce(counter: int) -> bytes:
    return b'\0\0\0\0' + struct.pack('>Q', counter)


class RecordSealer:
    """Encrypts outgoing records with one direction key and counter."""
    def __init__(self, key: bytes):
        self._aead = AESGCM(key)
        self._n = 0

    def seal(self, plaintext: bytes) -> bytes:
        """One record (length header + ciphertext) for ``plaintext``."""
        if len(plaintext) > MAX_PLAINTEXT:
            raise ValueError('record too long')
        if self._n >= MAX_COUNTER:
            raise ProtocolError('counter-exhausted')
        head = struct.pack('>I', len(plaintext) + TAG_LEN)
        ct = self._aead.encrypt(_nonce(self._n), plaintext, head)
        self._n += 1
        return head + ct


class RecordOpener:
    """Splits and decrypts records. A length field above ``MAX_RECORD`` (or
    below a tag) is refused before anything is allocated for it; a record
    that fails to decrypt (tampered, reordered, replayed) ends the link."""

    def __init__(self, key: bytes):
        self._aead = AESGCM(key)
        self._n = 0
        self._buf = bytearray()

    def feed(self, data: bytes) -> list[bytes]:
        """Add bytes from the socket; return the plaintexts of the records now complete."""
        self._buf += data
        out = []
        while len(self._buf) >= 4:
            (n,) = struct.unpack_from('>I', self._buf)
            if n > MAX_RECORD or n < TAG_LEN:
                raise ProtocolError('too-long', f'record length {n}')
            if len(self._buf) < 4 + n:
                break
            if self._n >= MAX_COUNTER:
                raise ProtocolError('counter-exhausted')
            head, ct = bytes(self._buf[:4]), bytes(self._buf[4:4 + n])
            try:
                out.append(self._aead.decrypt(_nonce(self._n), ct, head))
            except InvalidTag:
                raise ProtocolError('bad-record', 'record failed to decrypt') from None
            self._n += 1
            del self._buf[:4 + n]
        return out


# ── messages inside records ──────────────────────────────────────────────────

T_IDENTITY, T_PAIR_STATE, T_CONFIRM, T_ABORT = 0x01, 0x02, 0x03, 0x04
T_OFFER, T_ACCEPT, T_DECLINE, T_DATA, T_DONE, T_RESULT = 0x10, 0x11, 0x12, 0x13, 0x14, 0x15
T_CLOSE = 0x7F

# Abort / Decline / Result reasons (the UI words them; never peer text).
R_USER = 1          # the user said no / cancelled
R_TIMEOUT = 2
R_WRONG_CODE = 3
R_KIND = 4          # this app does not take that kind of file
R_TOO_BIG = 5
R_NO_SPACE = 6
R_HASH = 7          # bytes did not match the offered SHA-256
R_BUSY = 8
R_NOT_PAIRED = 9    # key is not the one stored for the chosen device


@dataclass(frozen=True)
class Identity:
    """A device introduces itself: static key, display name, signature over h."""
    static_pub: bytes
    name: str
    signature: bytes        # DER ECDSA-SHA256 over ID_LABEL | role | h


@dataclass(frozen=True)
class PairState:
    """Whether the sender of this already has the peer stored as paired."""
    known: bool             # "I have your key stored as paired"


@dataclass(frozen=True)
class Confirm:
    """The user on this side accepted the pairing code."""


@dataclass(frozen=True)
class Abort:
    """This side stops the connection, with a reason code."""
    reason: int


@dataclass(frozen=True)
class Offer:
    """The sender offers one file."""
    transfer_id: bytes      # 16 random bytes
    size: int
    sha256: bytes
    kind: str
    name: str


@dataclass(frozen=True)
class Accept:
    """The receiver takes the file, from ``offset`` (resume)."""
    offset: int             # resume point; 0 for a new file


@dataclass(frozen=True)
class Decline:
    """The receiver refuses the file, with a reason code."""
    reason: int


@dataclass(frozen=True)
class Data:
    """The next bytes of the file."""
    payload: bytes


@dataclass(frozen=True)
class Done:
    """The sender has sent every byte."""


@dataclass(frozen=True)
class Result:
    """The receiver's verdict after checking SHA-256."""
    ok: bool
    reason: int


@dataclass(frozen=True)
class Close:
    """The session ends cleanly (a bare TCP close is an interruption)."""


Message = (Identity | PairState | Confirm | Abort | Offer | Accept | Decline
           | Data | Done | Result | Close)

DATA_CHUNK = MAX_PLAINTEXT - 1      # Data payload that fills one record


def encode_message(m: Message) -> bytes:
    """One message's record plaintext (type byte + body)."""
    if isinstance(m, Identity):
        if len(m.static_pub) != POINT_LEN or len(m.signature) > MAX_SIGNATURE:
            raise ValueError('bad identity')
        return (bytes([T_IDENTITY]) + m.static_pub + _text(m.name, MAX_DEVICE_NAME)
                + struct.pack('>H', len(m.signature)) + m.signature)
    if isinstance(m, PairState):
        return bytes([T_PAIR_STATE, int(m.known)])
    if isinstance(m, Confirm):
        return bytes([T_CONFIRM])
    if isinstance(m, Abort):
        return bytes([T_ABORT, m.reason])
    if isinstance(m, Offer):
        if len(m.transfer_id) != 16 or len(m.sha256) != HASH_LEN or not 0 <= m.size <= MAX_SIZE:
            raise ValueError('bad offer')
        return (bytes([T_OFFER]) + m.transfer_id + struct.pack('>Q', m.size) + m.sha256
                + _text(m.kind, MAX_KIND) + _text(m.name, MAX_FILE_NAME))
    if isinstance(m, Accept):
        if not 0 <= m.offset <= MAX_SIZE:
            raise ValueError('bad offset')
        return bytes([T_ACCEPT]) + struct.pack('>Q', m.offset)
    if isinstance(m, Decline):
        return bytes([T_DECLINE, m.reason])
    if isinstance(m, Data):
        if not 0 < len(m.payload) <= DATA_CHUNK:
            raise ValueError('data chunk empty or too long')
        return bytes([T_DATA]) + m.payload
    if isinstance(m, Done):
        return bytes([T_DONE])
    if isinstance(m, Result):
        return bytes([T_RESULT, int(m.ok), m.reason])
    if isinstance(m, Close):
        return bytes([T_CLOSE])
    raise TypeError(type(m).__name__)


def _flag(b: int) -> bool:
    if b not in (0, 1):
        raise ProtocolError('malformed', 'flag is not 0 or 1')
    return bool(b)


def decode_message(plaintext: bytes) -> Message:
    """One record's message. Unknown types and malformed bodies raise."""
    if not plaintext:
        raise ProtocolError('malformed', 'empty record')
    t, r = plaintext[0], _Reader(plaintext[1:])
    m: Message
    if t == T_DATA:
        if len(plaintext) == 1:
            raise ProtocolError('malformed', 'empty data record')
        return Data(plaintext[1:])
    if t == T_IDENTITY:
        pub = r.take(POINT_LEN)
        name = r.text(MAX_DEVICE_NAME)
        n = r.u16()
        if n > MAX_SIGNATURE:
            raise ProtocolError('malformed', 'signature too long')
        m = Identity(pub, name, r.take(n))
    elif t == T_PAIR_STATE:
        m = PairState(_flag(r.u8()))
    elif t == T_CONFIRM:
        m = Confirm()
    elif t == T_ABORT:
        m = Abort(r.u8())
    elif t == T_OFFER:
        m = Offer(r.take(16), r.size(), r.take(HASH_LEN), r.text(MAX_KIND),
                  r.text(MAX_FILE_NAME))
    elif t == T_ACCEPT:
        m = Accept(r.size())
    elif t == T_DECLINE:
        m = Decline(r.u8())
    elif t == T_DONE:
        m = Done()
    elif t == T_RESULT:
        m = Result(_flag(r.u8()), r.u8())
    elif t == T_CLOSE:
        m = Close()
    else:
        raise ProtocolError('malformed', f'unknown message type {t}')
    r.end()
    return m


# ── identities ───────────────────────────────────────────────────────────────

def _id_message(role: bytes, h: bytes) -> bytes:
    return ID_LABEL + role + h


def make_identity(static: ec.EllipticCurvePrivateKey, name: str, role: bytes,
                  h: bytes) -> Identity:
    """This device's Identity for the connection with transcript ``h``."""
    sig = static.sign(_id_message(role, h), ec.ECDSA(hashes.SHA256()))
    return Identity(encode_point(static.public_key()), name, sig)


def verify_identity(m: Identity, role: bytes, h: bytes) -> None:
    """The peer holds the private key of the static key it sent, for this
    connection (``role`` is the PEER's role)."""
    pub = decode_point(m.static_pub)
    try:
        pub.verify(m.signature, _id_message(role, h), ec.ECDSA(hashes.SHA256()))
    except InvalidSignature:
        raise ProtocolError('bad-identity', 'signature does not verify') from None


def needs_pairing(i_know_peer: bool, peer_knows_me: bool) -> bool:
    """The code is shown unless BOTH sides have the other stored."""
    return not (i_know_peer and peer_knows_me)


def check_paired_key(expected_pub: bytes | None, got_pub: bytes) -> None:
    """The user picked a paired device: its key must be the stored one. A
    different key is refused outright, never turned into a new pairing."""
    if expected_pub is not None and expected_pub != got_pub:
        raise ProtocolError('not-paired-device', 'key differs from the paired one')


# ── message order after the handshake ───────────────────────────────────────

class Session:
    """The order of messages after the handshake, for one side (``role`` is
    THIS side's: ROLE_I sends the file, ROLE_R receives it).

    Every message this side sends goes through ``sent`` and every message
    from the peer through ``received``; ``received`` raises ProtocolError
    ('out-of-state') for anything the peer may not send now:

    - Identity, then PairState, once each, in that order.
    - When either side does not know the other (``needs_pairing``): I sends
      Confirm first, R answers with Confirm; nothing else until both have.
      Confirm is refused when no pairing is needed.
    - Then I offers one file (Offer); R answers Accept or Decline.
    - After Accept: Data records of at least one byte, never more than
      size - offset in total, then Done exactly at size - offset.
    - After Done: R's Result; then I's Close. Close before that is not a
      clean end (it is refused, so the transfer counts as interrupted).
    - Abort ends the session at any point. Nothing is taken after Abort or
      Close.
    """

    def __init__(self, role: bytes):
        if role not in (ROLE_I, ROLE_R):
            raise ValueError('role is ROLE_I or ROLE_R')
        self.role = role
        self._got_identity = False
        self._my_known: bool | None = None
        self._peer_known: bool | None = None
        self._sent_confirm = False
        self._got_confirm = False
        self._offer: Offer | None = None
        self._answer: Accept | Decline | None = None
        self._data = 0
        self._done = False
        self._result: Result | None = None
        self.ended = False

    @property
    def pairing(self) -> bool | None:
        """Whether this session pairs (None until both PairStates are known)."""
        if self._my_known is None or self._peer_known is None:
            return None
        return needs_pairing(self._my_known, self._peer_known)

    @property
    def paired(self) -> bool:
        """Pairing is not needed, or both Confirms have gone both ways."""
        return self.pairing is False or (self._sent_confirm and self._got_confirm)

    def _bad(self, m: Message) -> ProtocolError:
        self.ended = True
        return ProtocolError('out-of-state', f'{type(m).__name__} not expected now')

    def sent(self, m: Message) -> None:
        """Record a message this side sent (its own code is trusted; the
        checks that matter are on ``received``)."""
        if isinstance(m, PairState):
            self._my_known = m.known
        elif isinstance(m, Confirm):
            self._sent_confirm = True
        elif isinstance(m, Offer):
            self._offer = m
        elif isinstance(m, (Accept, Decline)):
            self._answer = m
        elif isinstance(m, Done):
            self._done = True
        elif isinstance(m, Result):
            self._result = m
        elif isinstance(m, (Abort, Close)):
            self.ended = True

    def received(self, m: Message) -> Message:
        """Check a message from the peer against the order; return it."""
        if self.ended:
            raise self._bad(m)
        if isinstance(m, Abort):
            self.ended = True
            return m
        ok = (self._expect_r(m) if self.role == ROLE_R else self._expect_i(m))
        if not ok:
            raise self._bad(m)
        return m

    def _common(self, m: Message) -> bool | None:
        """Identity / PairState / Confirm, the same for both roles; None when
        ``m`` is none of those."""
        if isinstance(m, Identity):
            if self._got_identity:
                return False
            self._got_identity = True
            return True
        if isinstance(m, PairState):
            if not self._got_identity or self._peer_known is not None:
                return False
            self._peer_known = m.known
            return True
        if isinstance(m, Confirm):
            if self.pairing is not True or self._got_confirm:
                return False
            if self.role == ROLE_I and not self._sent_confirm:
                return False            # R confirms only after I did
            self._got_confirm = True
            return True
        return None

    def _expect_r(self, m: Message) -> bool:
        common = self._common(m)
        if common is not None:
            return common
        if isinstance(m, Offer):
            if not self.paired or self._offer is not None:
                return False
            self._offer = m
            return True
        offer, accepted = self._offer, isinstance(self._answer, Accept)
        if isinstance(m, Data):
            # (Data after Done always runs past the size, so it fails below.)
            if offer is None or not accepted:
                return False
            assert isinstance(self._answer, Accept)
            self._data += len(m.payload)
            return len(m.payload) > 0 and self._data <= offer.size - self._answer.offset
        if isinstance(m, Done):
            if offer is None or not accepted or self._done:
                return False
            assert isinstance(self._answer, Accept)
            self._done = True
            return self._data == offer.size - self._answer.offset
        if isinstance(m, Close):
            self.ended = True
            return self._result is not None or isinstance(self._answer, Decline)
        return False

    def _expect_i(self, m: Message) -> bool:
        common = self._common(m)
        if common is not None:
            return common
        if isinstance(m, (Accept, Decline)):
            if self._offer is None or self._answer is not None:
                return False
            if isinstance(m, Accept) and m.offset > self._offer.size:
                return False
            self._answer = m
            return True
        if isinstance(m, Result):
            if not self._done or self._result is not None:
                return False
            self._result = m
            return True
        if isinstance(m, Close):
            self.ended = True
            return False                # R never sends Close in version 1
        return False


# ── transfer checks ──────────────────────────────────────────────────────────

def check_offset(offset: int, size: int) -> None:
    """The sender's check on Accept: a resume point inside the file."""
    if not 0 <= offset <= size:
        raise ProtocolError('bad-offset', f'offset {offset} outside 0..{size}')


_WIN_RESERVED = re.compile(r'(con|prn|aux|nul|conin\$|conout\$|com[0-9¹²³]|lpt[0-9¹²³])', re.I)
# Removed from names: control, format (bidi overrides, zero-width), surrogate
# and private-use characters. Unassigned code points are KEPT, so platforms
# on different Unicode versions agree.
_DROP_CATEGORIES = ('Cc', 'Cf', 'Cs', 'Co')


def _clean_text(s: str) -> str:
    return ''.join(c for c in s if unicodedata.category(c) not in _DROP_CATEGORIES)


def _cut_utf8(s: str, limit: int) -> str:
    """``s`` cut to at most ``limit`` UTF-8 bytes, on a code-point boundary."""
    b = s.encode('utf-8')
    if len(b) <= limit:
        return s
    return b[:limit].decode('utf-8', 'ignore')


def safe_device_name(sent: str) -> str:
    """A peer's display name as shown (sent Identity ``name`` or TXT ``n``):
    control, format, surrogate and private-use characters removed, spaces
    trimmed, at most ``MAX_DEVICE_NAME`` UTF-8 bytes; ``Unnamed device`` when
    nothing is left. A paired device is shown by its stored label instead."""
    name = ' '.join(_clean_text(re.sub(r'\s', ' ', sent)).split())
    return _cut_utf8(name, MAX_DEVICE_NAME).strip() or 'Unnamed device'
_BAD_CHARS = re.compile(r'[\\/:*?"<>|]')


def safe_file_name(sent: str, ext: str) -> str:
    """The name a received file is saved under: the sender's base name with
    control, format (bidi override), surrogate, private-use and reserved
    characters removed, Windows reserved names and leading/trailing dots and
    spaces avoided, at most ``SAFE_NAME_BYTES`` UTF-8 bytes with the
    extension ``ext`` (from the validated kind, never from the sender)."""
    base = re.split(r'[\\/]', sent)[-1]
    base = _clean_text(base)
    base = _BAD_CHARS.sub('_', base)
    if base.lower().endswith(ext.lower()):
        base = base[:-len(ext)]
    base = base.strip().lstrip('. ')
    stem = base.split('.')[0].rstrip(' ')
    if not base or _WIN_RESERVED.fullmatch(stem):
        base = f'received {base}'.strip() if base else 'received'
    base = _cut_utf8(base, SAFE_NAME_BYTES - len(ext.encode('utf-8'))).rstrip('. ')
    return base + ext


def is_lan_address(addr: str) -> bool:
    """Whether a peer address is on a local network (private, link-local,
    unique-local or loopback); everything else is refused. IPv4-mapped IPv6
    is unwrapped first; an IPv6 zone (``%eth0``) is allowed (Python's
    ipaddress reads it; other platforms strip it before parsing)."""
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if ip.is_loopback or ip.is_link_local:
        return True
    if isinstance(ip, ipaddress.IPv4Address):
        return any(ip in n for n in _V4_LAN)
    return ip in _V6_ULA


_V4_LAN = tuple(ipaddress.ip_network(n) for n in
                ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16'))
_V6_ULA = ipaddress.ip_network('fc00::/7')
