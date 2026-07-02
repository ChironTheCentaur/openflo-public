"""Tests for openflo.compliance — integrity manifest + e-signatures."""
from __future__ import annotations

import json

from openflo.compliance import (
    build_manifest,
    record_to_markdown,
    sha256_file,
    sha256_obj,
    sign_manifest,
    verify_record,
)


def test_sha256_obj_is_order_independent():
    a = sha256_obj({'x': 1, 'y': 2})
    b = sha256_obj({'y': 2, 'x': 1})
    assert a == b
    assert a != sha256_obj({'x': 1, 'y': 3})


def test_sha256_file(tmp_path):
    p = tmp_path / 'd.bin'
    p.write_bytes(b'hello world')
    import hashlib
    assert sha256_file(str(p)) == hashlib.sha256(b'hello world').hexdigest()
    assert sha256_file(str(tmp_path / 'missing')) is None


def _manifest(tmp_path):
    f = tmp_path / 'sample.fcs'
    f.write_bytes(b'FCS3.1 fake events')
    audit = [{'seq': 1, 'action': 'sample.load', 'details': {'n': 100}},
             {'seq': 2, 'action': 'gate.add', 'details': {}}]
    return build_manifest({'sample': str(f)}, audit, '0.1.0',
                          created='2026-06-17T08:00:00'), f


def test_build_manifest_hashes_files_and_audit(tmp_path):
    man, f = _manifest(tmp_path)
    assert man['files']['sample']['sha256'] == sha256_file(str(f))
    assert man['n_audit_entries'] == 2
    assert len(man['audit_sha256']) == 64


def test_sign_and_verify_valid(tmp_path):
    man, _ = _manifest(tmp_path)
    rec = sign_manifest(man, 'Dr. A', 'Reviewed and approved',
                        '2026-06-17T08:05:00')
    assert len(rec['signatures']) == 1
    v = verify_record(rec)
    assert v['all_valid'] is True
    assert v['signatures'][0]['valid'] is True
    assert v['signatures'][0]['signer'] == 'Dr. A'


def test_tamper_invalidates_signature(tmp_path):
    man, _ = _manifest(tmp_path)
    rec = sign_manifest(man, 'Dr. A', 'Approved', 't1')
    # Alter the signed content (e.g. someone edits the audit hash) → invalid.
    rec['audit_sha256'] = '0' * 64
    v = verify_record(rec)
    assert v['signatures'][0]['valid'] is False
    assert v['all_valid'] is False


def test_changed_file_detected(tmp_path):
    man, f = _manifest(tmp_path)
    rec = sign_manifest(man, 'Dr. A', 'Approved', 't1')
    f.write_bytes(b'tampered data')          # change the data after signing
    v = verify_record(rec)
    assert v['files_ok']['sample'] is False
    assert v['all_valid'] is False


def test_multiple_signatures(tmp_path):
    man, _ = _manifest(tmp_path)
    rec = sign_manifest(man, 'Analyst', 'Performed', 't1')
    rec = sign_manifest(rec, 'Reviewer', 'Reviewed', 't2')
    assert len(rec['signatures']) == 2
    v = verify_record(rec)
    assert v['all_valid'] is True
    assert {s['signer'] for s in v['signatures']} == {'Analyst', 'Reviewer'}


def test_record_serializable_and_markdown(tmp_path):
    man, _ = _manifest(tmp_path)
    rec = sign_manifest(man, 'Dr. A', 'Approved', 't1')
    json.loads(json.dumps(rec))              # round-trips as JSON
    md = record_to_markdown(rec)
    assert '# OpenFlo compliance' in md
    assert 'Dr. A' in md and 'Approved' in md
    assert 'SHA-256' in md
