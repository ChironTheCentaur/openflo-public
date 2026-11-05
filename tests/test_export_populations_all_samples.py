"""File > Export populations, all samples writes every sample's populations,
a folder per sample; the single-sample export writes only the active one.

Generated FCS files with synthetic sample names; proved to fail on the old
behaviour by a mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import tests._sample_editor as shared
from tests._sample_editor import _mask_count

loaded = shared.loaded


def _tot(path):
    raw = open(path, 'rb').read()
    start, end = int(raw[10:18]), int(raw[18:26])
    text = raw[start:end + 1].decode('latin-1')
    parts = text[1:].split(text[0])
    kv = dict(zip(parts[::2], parts[1::2], strict=False))
    return int(kv['$TOT'])



def test_export_populations_for_every_sample(loaded, monkeypatch):
    from openflo import editor_populations
    from openflo.async_task import wait_for_job
    root, ed, tmp = loaded
    for name in ('sample_a', 'sample_b'):
        ed._set_active_sample(name)
        ed._add_gate({'kind': 'threshold', 'channel': 'FSC-A', 'op': '>',
                      'value': 60_000.0, 'name': 'Big'}, audit=False)
    ed._set_active_sample('sample_a')
    out = tmp / 'pops'
    out.mkdir()
    monkeypatch.setattr(editor_populations.filedialog, 'askdirectory',
                        lambda **k: str(out))
    ed._export_populations_fcs_all()
    assert wait_for_job(ed._job_strip)
    root.update()
    assert sorted(p.name for p in out.iterdir()) == ['sample_a', 'sample_b']
    for name in ('sample_a', 'sample_b'):
        (f,) = list((out / name).glob('*.fcs'))
        assert _tot(f) == _mask_count(ed, name, 'Big')
    assert '2 sample folders' in ed.status_var.get()


def test_export_populations_of_the_active_sample_only(loaded, monkeypatch):
    from openflo import editor_populations
    from openflo.async_task import wait_for_job
    _root, ed, tmp = loaded
    for name in ('sample_a', 'sample_c', 'sample_b'):
        ed._set_active_sample(name)
        ed._add_gate({'kind': 'threshold', 'channel': 'FSC-A', 'op': '>',
                      'value': 60_000.0, 'name': 'Big'}, audit=False)
    out = tmp / 'one'
    out.mkdir()
    monkeypatch.setattr(editor_populations.filedialog, 'askdirectory',
                        lambda **k: str(out))
    ed._export_populations_fcs()
    assert wait_for_job(ed._job_strip)
    assert [p.name for p in out.iterdir()] == ['Big.fcs']
    (f,) = list(out.glob('*.fcs'))
    assert _tot(f) == _mask_count(ed, 'sample_b', 'Big')
    assert _tot(f) != _mask_count(ed, 'sample_c', 'Big')
