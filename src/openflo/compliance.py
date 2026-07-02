"""Tamper-evident sign-off records on top of the audit trail.

A lightweight integrity + electronic-signature layer in the spirit of 21 CFR
Part 11: hash the input data files and the audit trail into a **manifest**, then
attach one or more **electronic signatures** (signer, meaning, time) each bound
to the hash of that manifest. ``verify_record`` recomputes the hashes, so any
change to the data or the recorded analysis after signing is detectable.

Scope, stated honestly: this provides **tamper-evidence and attributable
sign-off**, not access control — it cannot by itself authenticate the signer's
identity (a desktop app has no trusted login), so it complements, but does not
replace, an organization's controlled-access environment. Pure stdlib.
"""
from __future__ import annotations

import hashlib
import json
import os


def sha256_file(path, chunk=1 << 20):
    """Streaming SHA-256 hex digest of a file (or None if unreadable)."""
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(chunk), b''):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def sha256_obj(obj):
    """SHA-256 hex digest of a JSON-able object, canonicalized (sorted keys)
    so the hash is stable regardless of dict ordering."""
    data = json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      default=str).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def build_manifest(files, audit_entries, software_version, created=None):
    """Build the integrity manifest.

    ``files`` : ``{name: path}`` of the input data files (each hashed).
    ``audit_entries`` : the audit-trail list (hashed as a whole).
    Returns a dict with per-file SHA-256s, the audit hash, the software version
    and a creation timestamp — everything a signature will be bound to."""
    file_hashes = {}
    for name, path in (files or {}).items():
        if path and os.path.isfile(path):
            file_hashes[name] = {'path': path, 'sha256': sha256_file(path),
                                 'bytes': os.path.getsize(path)}
        else:
            file_hashes[name] = {'path': path or '', 'sha256': None}
    return {
        'format': 'openflo-compliance', 'version': 1,
        'created': created, 'software_version': str(software_version),
        'files': file_hashes,
        'audit_sha256': sha256_obj(audit_entries or []),
        'n_audit_entries': len(audit_entries or []),
    }


def _manifest_only(record):
    """The manifest content of a (possibly signed) record — i.e. everything a
    signature is computed over, which is the record WITHOUT its signatures."""
    return {k: v for k, v in record.items() if k != 'signatures'}


def sign_manifest(record, signer, meaning, time):
    """Append an electronic signature to a manifest/record. The signature
    stores the SHA-256 of the manifest content (excluding signatures), so it is
    invalidated by any later change to the data/audit/version. Returns a new
    record dict (the input is not mutated)."""
    manifest = _manifest_only(record)
    sig = {'signer': str(signer), 'meaning': str(meaning), 'time': time,
           'manifest_sha256': sha256_obj(manifest)}
    return {**manifest, 'signatures': [*record.get('signatures', []), sig]}


def verify_record(record):
    """Recompute the manifest hash and re-hash any still-present files, and
    check every signature against it. Returns::

        {manifest_sha256, signatures:[{signer, meaning, time, valid}],
         files_ok:{name: bool}, all_valid: bool}

    ``valid`` is False for a signature whose recorded hash no longer matches —
    i.e. the signed analysis was altered after signing (tamper detected)."""
    current = sha256_obj(_manifest_only(record))
    sigs = []
    for s in record.get('signatures', []):
        sigs.append({'signer': s.get('signer'), 'meaning': s.get('meaning'),
                     'time': s.get('time'),
                     'valid': s.get('manifest_sha256') == current})
    files_ok = {}
    for name, info in (record.get('files') or {}).items():
        path, want = info.get('path'), info.get('sha256')
        if path and want and os.path.isfile(path):
            files_ok[name] = (sha256_file(path) == want)
    all_valid = (bool(sigs) and all(s['valid'] for s in sigs)
                 and all(files_ok.values()))
    return {'manifest_sha256': current, 'signatures': sigs,
            'files_ok': files_ok, 'all_valid': all_valid}


def record_to_markdown(record):
    """Human-readable compliance record (header, file hashes, signatures)."""
    out = ['# OpenFlo compliance / sign-off record', '',
           f"- **software_version**: {record.get('software_version')}",
           f"- **created**: {record.get('created')}",
           f"- **audit_sha256**: `{record.get('audit_sha256')}`",
           f"- **audit_entries**: {record.get('n_audit_entries')}", '',
           '## Data files', '',
           '| File | SHA-256 | Bytes |', '|---|---|---|']
    for name, info in (record.get('files') or {}).items():
        out.append(f"| {name} | `{info.get('sha256') or '(missing)'}` | "
                   f"{info.get('bytes', '')} |")
    out += ['', '## Electronic signatures', '']
    sigs = record.get('signatures', [])
    if sigs:
        out += ['| Signer | Meaning | Time | Manifest SHA-256 |',
                '|---|---|---|---|']
        for s in sigs:
            out.append(f"| {s.get('signer')} | {s.get('meaning')} | "
                       f"{s.get('time')} | `{s.get('manifest_sha256')}` |")
    else:
        out.append('_(unsigned)_')
    out.append('')
    return '\n'.join(out)
