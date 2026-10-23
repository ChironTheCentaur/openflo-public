"""Two checks on opening a FlowJo workspace.

1. FlowJo 9 writes a bare ``<!DOCTYPE Workspace>`` (no internal subset, so no
   entity can be declared). The entity-expansion guard refused it as a
   "possible entity-expansion attack"; it is now told apart from a DTD that
   declares something, and a workspace older than FlowJo 10 gets a plain
   "not supported yet" error instead.
2. A workspace records each sample's event count (on its <SampleNode>). A
   workspace can hold several files of one name, so a file found by name may
   be another sample's: when its event count differs, openflo-compare says
   so in every row of that sample, and the editor's import in its notes.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from openflo import pipeline as fp
from openflo.compare import _compare_one_sample, _per_sample_inventory

N = 500

FJ9_HEAD = ('<?xml version="1.0" encoding="ISO-8859-1"?><!DOCTYPE Workspace>'
            '<Workspace version="2.0" flowJoVersion="9.9.6">')


def _fj9(path, head=FJ9_HEAD):
    path.write_text(head + '<SampleList/></Workspace>', encoding='latin-1')
    return str(path)


def test_flowjo9_workspace_gets_a_not_supported_error(tmp_path):
    with pytest.raises(fp.WspParseError) as exc:
        fp.WspReader(_fj9(tmp_path / 'old.wsp'))
    msg = str(exc.value)
    assert 'FlowJo 9' in msg and 'not supported' in msg and '9.9.6' in msg
    assert 'entity' not in msg


def test_old_version_without_flowjo_version_is_refused_too(tmp_path):
    head = '<?xml version="1.0"?><!DOCTYPE Workspace><Workspace version="2.0">'
    with pytest.raises(fp.WspParseError, match='not supported'):
        fp.WspReader(_fj9(tmp_path / 'old.wsp', head))


def test_an_internal_dtd_subset_is_still_refused(tmp_path):
    head = ('<?xml version="1.0"?><!DOCTYPE Workspace [<!ATTLIST Workspace a CDATA "b">]>'
            '<Workspace version="2.0" flowJoVersion="9.9.6">')
    with pytest.raises(fp.WspParseError, match='entity-expansion'):
        fp.WspReader(_fj9(tmp_path / 'subset.wsp', head))


def test_flowjo10_workspace_with_a_bare_doctype_opens(tmp_path):
    head = ('<?xml version="1.0"?><!DOCTYPE Workspace>'
            '<Workspace version="20.0" flowJoVersion="10.8.1">')
    assert fp.WspReader(_fj9(tmp_path / 'new.wsp', head)).root is not None


def test_openflo_export_is_not_judged_by_its_version_string(tmp_path):
    w = fp.WspWriter()
    out = tmp_path / 'ours.wsp'
    w.write(str(out))
    assert fp.WspReader(str(out)).root is not None


def test_compare_cli_prints_the_message_not_a_traceback(tmp_path, monkeypatch, capsys):
    from openflo import compare
    wsp = _fj9(tmp_path / 'old.wsp')
    monkeypatch.setattr(sys, 'argv', ['openflo-compare', wsp])
    with pytest.raises(SystemExit) as exc:
        compare.main()
    assert 'not supported' in str(exc.value.code)


def test_gatingml_with_a_bare_doctype_is_read(tmp_path):
    from openflo.gatingml import read_gatingml
    path = tmp_path / 'g.xml'
    path.write_text(
        '<?xml version="1.0"?><!DOCTYPE Gating-ML>'
        '<gating:Gating-ML xmlns:gating="http://www.isac-net.org/std/Gating-ML/v2.0/gating"'
        ' xmlns:data-type="http://www.isac-net.org/std/Gating-ML/v2.0/datatypes">'
        '<gating:RectangleGate gating:id="R"><gating:dimension gating:min="1">'
        '<data-type:fcs-dimension data-type:name="FL1-A"/></gating:dimension>'
        '</gating:RectangleGate></gating:Gating-ML>', encoding='utf-8')
    gates, _w = read_gatingml(str(path))
    assert [g['kind'] for g in gates] == ['threshold']


# ── Event count of the matched file ─────────────────────────────────────────

def _fcs(path, n=N):
    import flowio
    rng = np.random.default_rng(1)
    ev = np.column_stack([rng.uniform(0, 1000, n), rng.uniform(0, 1000, n)])
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.astype(np.float32).ravel().tolist(),
                          ['FL1-A', 'FL2-A'])
    return str(path)


def _wsp(path, fcs, count):
    path.write_text(f'''<?xml version="1.0" encoding="UTF-8"?>
<Workspace version="20.0" flowJoVersion="10.8.1"
  xmlns:gating="http://www.isac-net.org/std/Gating-ML/v2.0/gating"
  xmlns:data-type="http://www.isac-net.org/std/Gating-ML/v2.0/datatypes">
 <SampleList><Sample>
  <DataSet uri="{fp._wsp_dataset_uri(fcs)}" sampleID="1"/>
  <SampleNode name="s1" sampleID="1" count="{count}"><Subpopulations>
   <Population name="Pos" count="10">
    <Gate gating:id="ID1"><gating:RectangleGate>
     <gating:dimension gating:min="500"><data-type:fcs-dimension data-type:name="FL1-A"/></gating:dimension>
    </gating:RectangleGate></Gate>
   </Population>
  </Subpopulations></SampleNode>
 </Sample></SampleList>
</Workspace>''', encoding='utf-8')
    return str(path)


def _rows(wsp, fcs):
    entry, = _per_sample_inventory(fp.WspReader(wsp))
    name, _uri, pops = entry
    return _compare_one_sample(name, fcs, pops,
                               expected_events=getattr(entry, 'events', None))


def test_compare_flags_a_file_with_another_event_count(tmp_path):
    fcs = _fcs(tmp_path / 's1.fcs')
    row, = _rows(_wsp(tmp_path / 'w.wsp', fcs, N + 37), fcs)
    assert str(N + 37) in row['error'] and str(N) in row['error']
    assert 'different file' in row['error']
    assert row['openflo_count'] is not None          # still counted


@pytest.mark.parametrize('count', [N, 0, -1])
def test_compare_is_quiet_when_the_count_agrees_or_is_not_recorded(tmp_path, count):
    fcs = _fcs(tmp_path / 's1.fcs')
    row, = _rows(_wsp(tmp_path / 'w.wsp', fcs, count), fcs)
    assert not row['error']


def test_compare_cli_entry_passes_the_count(tmp_path):
    from openflo.compare import FcsLocator, _compare_entry
    fcs = _fcs(tmp_path / 's1.fcs')
    wsp = _wsp(tmp_path / 'w.wsp', fcs, N + 1)
    entry, = _per_sample_inventory(fp.WspReader(wsp))
    _line, rows = _compare_entry(wsp, None, entry, FcsLocator(wsp))
    assert all('different file' in r['error'] for r in rows)


def test_editor_import_notes_a_file_with_another_event_count(tmp_path):
    sys.path.insert(0, os.path.dirname(__file__))
    from test_gui_smoke import _editor_or_skip
    root, ed, _gui = _editor_or_skip()
    try:
        fcs = _fcs(tmp_path / 's1.fcs')
        ed._queue_fcs_loads = lambda paths, *a, **k: None
        ed._load_notes = []
        ed._ingest_wsp(_wsp(tmp_path / 'w.wsp', fcs, N + 37))
        notes = [n for n in ed._load_notes if 'different file' in n]
        assert len(notes) == 1 and str(N + 37) in notes[0]
        assert 'different number of events' in ed.status_var.get()
        assert ed._pending_sample_gates            # imported, not refused
        ed._load_notes = []
        ed._pending_sample_gates.clear()
        ed._ingest_wsp(_wsp(tmp_path / 'ok.wsp', fcs, N))
        assert not [n for n in ed._load_notes if 'different file' in n]
    finally:
        root.destroy()
