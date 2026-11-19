"""openflo.document.Document.save_session never writes a file the document
read, and leaves every file already there as it was when a save fails.

Generated FCS files and sessions only (FSC-A, SSC-A, FL1-A ...; samples
s1, a, b). No Tk.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np
import pytest

from openflo import document
from openflo.document import Document, DocumentError
from tests.test_document import gated_doc


def _digest(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _tree(root):
    """{relative path: digest or link target} of everything under `root`."""
    out = {}
    for d, dirs, files in os.walk(root):
        for n in dirs + files:
            p = os.path.join(d, n)
            rel = os.path.relpath(p, root)
            if os.path.islink(p):
                out[rel] = 'link:' + os.readlink(p)
            elif os.path.isfile(p):
                out[rel] = _digest(p)
            else:
                out[rel] = 'dir'
    return out


def sidecar_doc(tmp_path):
    """A document holding session ``run.flowsession`` whose sample s1
    loads from its sidecar ``run_data/s1.npz`` (with its transform record
    ``run_data/s1_transforms.json``); its raw FCS is ``raw/s1.fcs``."""
    from openflo.session_sidecar import write_frame
    from openflo.transform_sidecar import write_transforms_sidecar
    doc, name, *_ = gated_doc(tmp_path)
    s = doc._samples[name].data
    sess = tmp_path / 'run.flowsession'
    # Saved first: a save of s1 from its raw FCS removes an s1 sidecar
    # already in run_data (it would be read for it by name).
    doc.save_session(str(sess))
    side = tmp_path / 'run_data' / 's1.npz'
    side.parent.mkdir()
    assert write_frame(s.data, str(side))
    write_transforms_sidecar(str(side), s.data_transforms, s.data.columns,
                             len(s.data))
    data = json.loads(sess.read_text('utf-8'))
    data['samples'][0]['processed_data'] = 'run_data/s1.npz'
    sess.write_text(json.dumps(data), 'utf-8')
    doc2 = Document()
    doc2.open_session(str(sess))
    assert doc2._samples[name].sidecar == str(side)
    assert doc2._samples[name].data.source_fcs == str(tmp_path / 'raw'
                                                      / 's1.fcs')
    return doc2, name


# ── H1: every file read is a source ─────────────────────────────────────────

@pytest.mark.parametrize('target', ['raw/s1.fcs',
                                    'run_data/s1_transforms.json',
                                    'run_data/s1.npz', 'run.flowsession'])
def test_a_sidecar_samples_files_are_never_written(tmp_path, target):
    """A sidecar sample's raw FCS (its keywords are read) and the sidecar's
    transform record are read as much as the sidecar is: a save onto any
    of them is refused before anything is written, naming that file."""
    doc, _name = sidecar_doc(tmp_path)
    before = _tree(tmp_path)
    with pytest.raises(DocumentError, match=re.escape(
            os.path.basename(target) + ' is a file this document was')):
        doc.save_session(str(tmp_path / target))
    assert _tree(tmp_path) == before


def test_a_sidecar_that_could_not_be_used_is_still_a_source(tmp_path):
    """A sidecar shorter than its transform record says was read (both
    were) before it gave way to the raw FCS: neither may be written."""
    from openflo.transform_sidecar import write_transforms_sidecar
    doc, name = sidecar_doc(tmp_path)
    s = doc._samples[name].data
    side = tmp_path / 'run_data' / 's1.npz'
    write_transforms_sidecar(str(side), s.data_transforms, s.data.columns,
                             len(s.data) + 5)
    doc2 = Document()
    got = doc2.open_session(str(tmp_path / 'run.flowsession'))
    assert doc2._samples[name].sidecar == ''
    assert any('was not used' in n for n in got['notes'])
    before = _tree(tmp_path)
    for t in ('run_data/s1.npz', 'run_data/s1_transforms.json'):
        with pytest.raises(DocumentError, match='opened from'):
            doc2.save_session(str(tmp_path / t))
    assert _tree(tmp_path) == before


def test_a_workspace_file_whose_events_do_not_match_is_a_source(tmp_path):
    """A file whose event count is not the workspace's is imported anyway
    (and said): it is a source like the others, as is the workspace."""
    from tests._open_fixtures import make_fcs, write_wsp
    a = make_fcs(tmp_path / 'run' / 'a.fcs', seed=1)
    b = make_fcs(tmp_path / 'run' / 'b.fcs', seed=2)
    wsp = write_wsp(tmp_path / 'run' / 'x.wsp',
                    [('a', a, None, False), ('b', b, None, False)])
    make_fcs(b, seed=3, n=500)
    doc = Document()
    got = doc.import_wsp(wsp)
    assert got['samples'] == ['a', 'b'] and got['mismatched']
    before = _tree(tmp_path)
    for t in (b, a, wsp):
        with pytest.raises(DocumentError, match='opened from'):
            doc.save_session(t)
    assert _tree(tmp_path) == before


# ── M1: identity by inode, temporary files made new ─────────────────────────

def test_a_hard_link_to_a_file_read_is_refused(tmp_path):
    doc, _name = sidecar_doc(tmp_path)
    os.link(tmp_path / 'run.flowsession', tmp_path / 'y.flowsession')
    os.link(tmp_path / 'raw' / 's1.fcs', tmp_path / 'z.flowsession')
    before = _tree(tmp_path)
    for t in ('y.flowsession', 'z.flowsession'):
        with pytest.raises(DocumentError, match='opened from'):
            doc.save_session(str(tmp_path / t))
    assert _tree(tmp_path) == before


@pytest.mark.parametrize('out, planted, points_at, kind', [
    ('x', 'x_data/s1.npz.tmp', 'run.flowsession', 'symlink'),
    ('x', 'x_data/s1_transforms.json.tmp', 'run_data/s1.npz', 'symlink'),
    ('y', 'y.flowsession.tmp', 'run.flowsession', 'hardlink'),
    ('y', 'y.flowsession.tmp', 'raw/s1.fcs', 'symlink'),
])
def test_a_planted_tmp_file_cannot_redirect_a_write(tmp_path, out, planted,
                                                    points_at, kind):
    """The reviewer's probes: a link at ``<file>.tmp`` beside a file the
    save writes is never written through; the save writes its own new
    temporary file and leaves the planted one, and every file read, as it
    was."""
    doc, name = sidecar_doc(tmp_path)
    p = tmp_path / planted
    p.parent.mkdir(exist_ok=True)
    if kind == 'symlink':
        try:
            os.symlink(tmp_path / points_at, p)
        except (OSError, NotImplementedError):
            pytest.skip('no symbolic links here')
    else:
        os.link(tmp_path / points_at, p)
    before = _tree(tmp_path)
    out += '.flowsession'
    got = doc.save_session(str(tmp_path / out))
    after = _tree(tmp_path)
    assert {k: v for k, v in after.items() if k in before} == before
    new = sorted(set(after) - set(before))
    assert not any(n.endswith('.tmp') for n in new), new
    assert got['path'] == str(tmp_path / out)
    doc2 = Document()
    assert doc2.open_session(got['path'])['samples'] == [name]


@pytest.mark.skipif(os.name == 'nt', reason='POSIX file modes')
def test_a_saved_session_is_readable_by_others_as_a_plain_write(tmp_path):
    """The temporary file is made with mode 0o666 less the umask, as
    ``open(path, 'w')`` makes a file."""
    doc, *_ = gated_doc(tmp_path)
    old = os.umask(0o022)
    try:
        doc.save_session(str(tmp_path / 'm.flowsession'))
    finally:
        os.umask(old)
    assert (tmp_path / 'm.flowsession').stat().st_mode & 0o777 == 0o644


# ── L6: a failed save leaves no trace ───────────────────────────────────────

def _fail(*_a, **_k):
    raise OSError('disk full (planted)')


@pytest.mark.parametrize('broken', ['_write_json', '_copy_file'])
def test_a_failed_save_leaves_no_copies_and_no_temporary_files(
        tmp_path, monkeypatch, broken):
    """The session or a sidecar copy cannot be written: nothing is moved
    into place, the temporary files and the new data folder are gone."""
    doc, _name = sidecar_doc(tmp_path)
    before = _tree(tmp_path)
    monkeypatch.setattr(document, broken, _fail)
    with pytest.raises(OSError, match='planted'):
        doc.save_session(str(tmp_path / 'new.flowsession'))
    assert _tree(tmp_path) == before


def test_a_failed_save_over_an_earlier_one_keeps_it(tmp_path, monkeypatch):
    """Saving again over a session and its copies: when the session cannot
    be written, the earlier session and its copies stay as they were."""
    from openflo.session_sidecar import write_frame
    doc, name = sidecar_doc(tmp_path)
    doc.save_session(str(tmp_path / 'new.flowsession'))
    s = doc._samples[name].data
    frame = s.data.copy()
    frame['FL3-A'] = np.zeros(len(frame))
    # The earlier save's copy (not a file the document read) now differs
    # from the source, so a second save that moved its copy in would show.
    assert write_frame(frame, str(tmp_path / 'new_data' / 's1.npz'))
    before = _tree(tmp_path)
    assert 'new_data/s1.npz' in before
    monkeypatch.setattr(document, '_write_json', _fail)
    with pytest.raises(OSError, match='planted'):
        doc.save_session(str(tmp_path / 'new.flowsession'))
    assert _tree(tmp_path) == before
