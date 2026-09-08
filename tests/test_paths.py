"""Tests for openflo.paths — drop expansion + sidecar name (extracted)."""
from __future__ import annotations

import pytest

from openflo.paths import expand_dropped_paths, safe_sidecar_name


def test_safe_sidecar_name():
    # A name that is already filesystem-safe keeps its plain stem, so sidecars
    # written by earlier versions are still found.
    assert safe_sidecar_name('a-b_c') == 'a-b_c'
    assert safe_sidecar_name('Sample01') == 'Sample01'
    assert safe_sidecar_name('') == 'sample'              # empty → fallback

    # A name that had to be sanitised carries a digest of the ORIGINAL, so the
    # mapping is injective. It used to be a plain substitution, which is not:
    # see test_sidecar_names_do_not_collide below.
    assert safe_sidecar_name('CD3+ / Singlets').startswith('CD3____Singlets_')
    assert safe_sidecar_name('***').startswith('___')
    assert len(safe_sidecar_name('***')) == len('___') + 9


def test_the_stem_is_stable_for_the_same_name():
    """The loader guesses this path from the sample name when the recorded
    pointer is missing, so the function must be deterministic."""
    assert safe_sidecar_name('Day 1') == safe_sidecar_name('Day 1')


@pytest.mark.parametrize('first, second', [
    ('Tube_01_Rep_A', 'Tube 01 Rep A'),
    ('Day 1', 'Day.1'),
    ('Patient 01', 'Patient#01'),
    ('Live/Dead', 'Live Dead'),
])
def test_sidecar_names_do_not_collide(first, second):
    """Two distinct samples must never be handed the same sidecar file.

    They were: the session writer wrote `<stem>.csv` with no uniqueness check
    and recorded it on BOTH entries, and restore prefers the sidecar over the
    raw FCS — so one sample came back holding the other's events, clusters and
    compensated values, silently.
    """
    assert safe_sidecar_name(first) != safe_sidecar_name(second)


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
