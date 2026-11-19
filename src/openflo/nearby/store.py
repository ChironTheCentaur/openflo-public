"""What Nearby send keeps between runs: this device's identity key and name,
the devices it is paired with, and part-received files kept for resuming.

All of it lives in one folder (the desktop app passes ``~/.openflo/nearby``);
nothing here is shared with any other machine. The identity key file is
written owner-only where the system has POSIX permissions; on Windows the
user profile's own access rules protect it.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import threading
import time
from dataclasses import dataclass
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from ..atomic_write import atomic_write_json
from . import protocol as p

PARTIAL_KEEP_SECONDS = 24 * 3600
PARTIAL_TOTAL_CAP = 2 * 1024 ** 3       # all kept partial files together
_SUFFIX_ALPHABET = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789'


@dataclass(frozen=True)
class PairedDevice:
    """A device this one has paired with, known by its key (never its name)."""
    static_pub: bytes
    name: str
    paired_at: float

    @property
    def fingerprint(self) -> str:
        """Hex SHA-256 of the key: the id the paired list is kept under."""
        return p.fingerprint(self.static_pub).hex()


class NearbyStore:
    """Identity, paired devices and partial files under ``root``."""

    def __init__(self, root: str, default_name: str = 'Computer'):
        self.root = root
        self._default_name = default_name
        os.makedirs(root, exist_ok=True)
        self._key: ec.EllipticCurvePrivateKey | None = None

    # ── identity ──────────────────────────────────────────────────────────
    @property
    def _key_path(self) -> str:
        return os.path.join(self.root, 'identity.pem')

    def identity(self) -> ec.EllipticCurvePrivateKey:
        """This install's identity key, made and saved on first use. Safe to
        call from several threads (sending and receiving can start together)
        and from two OpenFlo windows at once: whoever saves first wins, and
        the other uses that key."""
        if self._key is None:
            # Threads racing here all end up with the one key the link saved.
            self._key = self._load_key() or self._save_new_key()
        return self._key

    def _load_key(self) -> ec.EllipticCurvePrivateKey | None:
        try:
            with open(self._key_path, 'rb') as fh:
                data = fh.read()
        except FileNotFoundError:
            return None
        key = serialization.load_pem_private_key(data, password=None)
        if not isinstance(key, ec.EllipticCurvePrivateKey) \
                or not isinstance(key.curve, ec.SECP256R1):
            raise ValueError('the Nearby send identity file holds another kind of key')
        return key

    def _save_new_key(self) -> ec.EllipticCurvePrivateKey:
        key = p.new_static_key()
        pem = key.private_bytes(serialization.Encoding.PEM,
                                serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
        # Written whole under a private name, then linked into place: a
        # reader never sees half a key, and an existing key is never replaced.
        tmp = f'{self._key_path}.{os.getpid()}.{threading.get_ident()}.tmp'
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, 'wb') as fh:
                fh.write(pem)
            try:
                os.link(tmp, self._key_path)
            except FileExistsError:
                pass                      # another process saved first: use its key
            except OSError:               # no hard links on this file system
                self._create_key_file(pem)
        finally:
            _remove(tmp)
        loaded = self._load_key()
        assert loaded is not None
        return loaded

    def _create_key_file(self, pem: bytes) -> None:
        """Without hard links: create the key file exclusively, so a key
        another process saved first is kept, never replaced."""
        try:
            fd = os.open(self._key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return
        with os.fdopen(fd, 'wb') as fh:
            fh.write(pem)

    def public_key(self) -> bytes:
        """This device's 65-byte public key."""
        return p.encode_point(self.identity().public_key())

    def device_name(self) -> str:
        """The name other devices see: the saved one, else the default with a
        4-character suffix from this device's key (no account or owner name)."""
        saved = self._read_json('settings.json').get('name')
        if isinstance(saved, str) and saved.strip():
            return saved.strip()[:40]
        fp = p.fingerprint(self.public_key())
        rnd = random.Random(fp)
        return f'{self._default_name} ' + ''.join(rnd.choice(_SUFFIX_ALPHABET) for _ in range(4))

    def set_device_name(self, name: str) -> None:
        """Save the name other devices see; blank goes back to the default."""
        name = name.strip()[:40]
        data = self._read_json('settings.json')
        if name:
            data['name'] = name
        else:
            data.pop('name', None)
        atomic_write_json(os.path.join(self.root, 'settings.json'), data)

    # ── paired devices ────────────────────────────────────────────────────
    def paired(self) -> list[PairedDevice]:
        """Every paired device, oldest first."""
        out = []
        for row in self._read_json('paired.json').get('devices', []):
            try:
                pub = bytes.fromhex(row['pub'])
                p.decode_point(pub)
                out.append(PairedDevice(pub, str(row['name'])[:p.MAX_DEVICE_NAME],
                                        float(row['at'])))
            except (KeyError, TypeError, ValueError):
                continue                         # a damaged row is dropped
        return sorted(out, key=lambda d: d.paired_at)

    def paired_key(self, pub: bytes) -> PairedDevice | None:
        """The paired entry with exactly this key, if any."""
        return next((d for d in self.paired() if d.static_pub == pub), None)

    def add_paired(self, pub: bytes, name: str) -> None:
        """Remember a device after both sides confirmed the code."""
        devices = [d for d in self.paired() if d.static_pub != pub]
        devices.append(PairedDevice(pub, name[:p.MAX_DEVICE_NAME], time.time()))
        self._write_paired(devices)

    def forget(self, pub: bytes) -> None:
        """Remove a paired device (it will need the code again)."""
        self._write_paired([d for d in self.paired() if d.static_pub != pub])

    def _write_paired(self, devices: list[PairedDevice]) -> None:
        atomic_write_json(os.path.join(self.root, 'paired.json'), {'devices': [
            {'pub': d.static_pub.hex(), 'name': d.name, 'at': d.paired_at} for d in devices]})

    # ── partial files (resume) ────────────────────────────────────────────
    def partial_path(self, peer_pub: bytes, sha256: bytes) -> str:
        """Where a file from ``peer_pub`` with this hash is kept while it arrives."""
        d = os.path.join(self.root, 'partial', p.fingerprint(peer_pub).hex()[:32])
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, sha256.hex() + '.part')

    def tidy_partials(self, now: float | None = None) -> None:
        """Delete partial files older than a day, then the oldest until the
        rest fit under ``PARTIAL_TOTAL_CAP``."""
        now = time.time() if now is None else now
        base = os.path.join(self.root, 'partial')
        found = []
        for dirpath, _dirs, files in os.walk(base):
            for f in files:
                path = os.path.join(dirpath, f)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if now - st.st_mtime > PARTIAL_KEEP_SECONDS:
                    _remove(path)
                else:
                    found.append((st.st_mtime, st.st_size, path))
        total = sum(s for _m, s, _p in found)
        for _m, size, path in sorted(found):
            if total <= PARTIAL_TOTAL_CAP:
                break
            _remove(path)
            total -= size

    # ── helpers ───────────────────────────────────────────────────────────
    def _read_json(self, name: str) -> dict:
        try:
            with open(os.path.join(self.root, name), encoding='utf-8') as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def running_sha256(path: str, upto: int | None = None, chunk: int = 1 << 20) -> Any:
    """A running SHA-256 of the file's first ``upto`` bytes (all if None)."""
    h = hashlib.sha256()
    left = upto
    with open(path, 'rb') as fh:
        while left is None or left > 0:
            b = fh.read(chunk if left is None else min(chunk, left))
            if not b:
                break
            h.update(b)
            if left is not None:
                left -= len(b)
    return h


def unique_destination(folder: str, name: str) -> str:
    """``folder/name``, or ``name (2)``, ``(3)``... so nothing is overwritten."""
    stem, ext = os.path.splitext(name)
    candidate = os.path.join(folder, name)
    n = 2
    while os.path.exists(candidate):
        candidate = os.path.join(folder, f'{stem} ({n}){ext}')
        n += 1
    return candidate


def move_into_place(part: str, folder: str, name: str) -> str:
    """Move a checked partial file to its final, never-overwriting name."""
    os.makedirs(folder, exist_ok=True)
    while True:
        dest = unique_destination(folder, name)
        try:
            # O_EXCL claims the name so a file appearing meanwhile is not replaced.
            fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            continue
        os.close(fd)
        try:
            os.replace(part, dest)
        except OSError:
            shutil.move(part, dest)
        return dest
