"""Index-sort data: the plate well each sorted cell went to (Complaints 28).

An index-sorted FCS file holds the sorted events in sort order and names
each one's destination in its text section. The forms read here:

- ``INDEX SORTING LOCATIONS`` (BD sorters): ``row,col;row,col;...``, one
  pair per sorted event, counted from 0. A long list is split across
  numbered keywords (``INDEX SORTING LOCATIONS_1`` .. ``_16``), joined here
  in numeric order.
- ``$INDEX`` / ``INDEX SORT POSITIONS`` / ``SORT LOCATIONS`` and the like:
  the same pairs, or well names (``A1;A2;B01``).

Pairs are taken as counted from 0 when any pair holds a 0 and from 1
otherwise; a well name is unambiguous. Each event takes the location at its
position in the list; with fewer locations than events the last events have
none (-1), never a guessed well.

Pure Python / numpy.
"""
from __future__ import annotations

import re

import numpy as np

from .sample_meta import PLATE_ROWS, keywords, parse_well

INDEX_KEYWORDS = ('INDEX SORTING LOCATIONS', 'INDEX SORT LOCATIONS',
                  'INDEX SORTING POSITIONS', 'INDEX SORT POSITIONS',
                  'INDEX', 'SORT LOCATIONS', 'INDEXSORT')
_PAIR = re.compile(r'^\s*(\d+)\s*[,:xX ]\s*(\d+)\s*$')


def parse_locations(text):
    """``[(row, col)]`` from 0, one per sorted event, from a locations
    keyword value; ``[]`` when it names none. A token that is neither a
    pair nor a well reads ``None`` so the events after it keep their
    places."""
    tokens = [t for t in re.split(r'[;|\n]+', str(text or '')) if t.strip()]
    pairs = [_PAIR.match(t) for t in tokens]
    numeric = [(int(m.group(1)), int(m.group(2))) for m in pairs if m]
    base = 0 if any(r == 0 or c == 0 for r, c in numeric) else 1
    out = []
    for t, m in zip(tokens, pairs, strict=True):
        if m:
            out.append((int(m.group(1)) - base, int(m.group(2)) - base))
        else:
            out.append(parse_well(t))
    if not any(p is not None for p in out):
        return []
    return out


def _keyword_text(kw, key):
    """The value of `key` in `kw`, joined with its numbered parts
    (``KEY_1``, ``KEY_2``, ... in numeric order, so ``_10`` follows ``_9``):
    BD sorters split a long locations list across them, each part ending
    where a pair ends. ``''`` when none is there."""
    parts = []
    for k, v in kw.items():
        if k == key:
            parts.append((0, v))
        elif k.startswith(key + '_') and k[len(key) + 1:].isdigit():
            parts.append((int(k[len(key) + 1:]), v))
    return ';'.join(v for _n, v in sorted(parts, key=lambda p: p[0]))


def locations_of(sample):
    """The sample's per-event locations list (see parse_locations) from the
    first index keyword it carries, or ``[]``."""
    kw = keywords(sample)
    for k in INDEX_KEYWORDS:
        text = _keyword_text(kw, k)
        if text:
            locs = parse_locations(text)
            if locs:
                return locs
    return []


def has_index_sort(sample) -> bool:
    return bool(locations_of(sample))


def event_wells(sample):
    """``(rows, cols)``: int arrays, one entry per event of ``sample.data``,
    -1 for an event with no location."""
    n = len(getattr(sample, 'data', []))
    rows = np.full(n, -1, dtype=int)
    cols = np.full(n, -1, dtype=int)
    for i, pos in enumerate(locations_of(sample)[:n]):
        if pos is not None:
            rows[i], cols[i] = pos
    return rows, cols


def well_values(sample, channel):
    """``{(row, col): value}`` of ``channel`` for each index-sorted event on
    a 96-well plate (a well sorted twice keeps the later cell, as the plate
    then holds both)."""
    rows, cols = event_wells(sample)
    v = np.asarray(sample.data[channel], dtype=float)
    out = {}
    for i in range(len(v)):
        if 0 <= rows[i] < len(PLATE_ROWS) and 0 <= cols[i] < 12:
            out[(int(rows[i]), int(cols[i]))] = float(v[i])
    return out
