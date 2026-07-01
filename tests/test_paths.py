"""Tests for openflo.paths — drop expansion + sidecar name (extracted)."""
from __future__ import annotations

from openflo.paths import expand_dropped_paths, safe_sidecar_name


def test_safe_sidecar_name():
    assert safe_sidecar_name('CD3+ / Singlets') == 'CD3____Singlets'  # +, sp, /, sp
    assert safe_sidecar_name('a-b_c') == 'a-b_c'          # kept chars
    assert safe_sidecar_name('') == 'sample'              # empty → fallback
    assert safe_sidecar_name('***') == '___'


def test_expand_dropped_paths(tmp_path):
    # a folder tree with mixed files; only .fcs / .wsp surface, recursively
    (tmp_path / 'a.fcs').write_bytes(b'x')
    (tmp_path / 'note.txt').write_text('skip me')
    sub = tmp_path / 'day1'
    sub.mkdir()
    (sub / 'b.FCS').write_bytes(b'x')                     # case-insensitive
    (sub / 'gates.wsp').write_text('<w/>')

    fcs, wsp = expand_dropped_paths([str(tmp_path)])
    assert [p.lower().endswith('.fcs') for p in fcs] == [True, True]
    assert len(fcs) == 2 and len(wsp) == 1
    assert fcs == sorted(fcs)                             # deterministic order

    # a single file, with stray quotes/whitespace (as drop payloads arrive)
    fcs2, wsp2 = expand_dropped_paths([f'  "{tmp_path / "a.fcs"}" '])
    assert len(fcs2) == 1 and wsp2 == []

    # empties / blanks ignored
    assert expand_dropped_paths(['', None, '   ']) == ([], [])
