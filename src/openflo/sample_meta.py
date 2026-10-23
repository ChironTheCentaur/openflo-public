"""Sample metadata: groups from FCS keywords or a condition table, and plate
wells (Complaints 17).

Pure functions on FlowSample-like objects (``.name``, ``.path``,
``.metadata``) and plain data, so the editor's menus, the plate window and
the tests share one reading of the keywords.
"""
from __future__ import annotations

import csv
import io
import os
import re

import numpy as np

# Keywords that describe the file's layout or every event, never a sample's
# condition: offering them as groupings would only add noise.
_LAYOUT = re.compile(
    r'(p\d+[a-z]+|begin\w*|end\w*|nextdata|byteord|datatype|mode|par|tot|'
    r'spillover|spill|comp|timestep|btim|etim|date|fil|cells|abrt|lost|'
    r'csmode|csvbits|cstot|unicode|flowjo\w*|openflo\w*|guid|'
    r'creator|cyt|cytsn|sys|inst|op|fcsversion|last_modified|origin|'
    r'laser\w*|window extension|threshold|apply compensation|'
    r'fsc asf|autobs|bd\$.*|cst .*|spill\w*)', re.I)

# Plate wells: the keyword each acquisition program writes, in the order
# they are tried; else a well in the sample's name ("Plate1_B07", "C3").
WELL_KEYWORDS = ('WELL ID', 'WELL_ID', 'WELLID', 'WELL', 'WELL NAME',
                 'SAMPLE WELL', 'PLATE WELL', 'LOCATION', 'POSITION')
PLATE_ROWS = 'ABCDEFGH'
PLATE_COLS = 12
_WELL = re.compile(r'(?<![A-Za-z0-9])([A-Ha-h])[ _-]?0?(1[0-2]|[1-9])(?![0-9])')


def keywords(sample) -> dict:
    """``{KEYWORD: value}`` of a sample's FCS text section, names without
    the leading ``$`` and upper-cased, values as stripped text."""
    out = {}
    for k, v in (getattr(sample, 'metadata', None) or {}).items():
        key = str(k).lstrip('$').strip().upper()
        val = str(v).strip() if v is not None else ''
        if key and val:
            out[key] = val
    return out


def grouping_keywords(samples) -> list[tuple[str, int]]:
    """Keywords that could group ``samples``: ``[(keyword, n_values)]`` for
    each keyword that is not about the file's layout and that splits the
    samples into 2 or more groups, fewer than one per sample when there are
    more than two. Sorted by how many samples carry it, then by name."""
    samples = list(samples)
    seen: dict[str, dict] = {}
    for s in samples:
        for k, v in keywords(s).items():
            if _LAYOUT.fullmatch(k.lower()):
                continue
            seen.setdefault(k, {})[s.name] = v
    out = []
    for k, vals in seen.items():
        n = len(set(vals.values()))
        if n < 2 or (len(samples) > 2 and n >= len(samples)
                     and len(vals) == len(samples)):
            continue
        out.append((k, n, len(vals)))
    out.sort(key=lambda r: (-r[2], r[0]))
    return [(k, n) for k, n, _c in out]


def groups_by_keyword(samples, key: str) -> dict[str, str]:
    """``{sample name: group}``: the sample's value of ``key``, or
    "No <key>" for a sample without it."""
    key = key.lstrip('$').upper()
    return {s.name: keywords(s).get(key) or f"No {key}" for s in samples}


def _norm(name: str) -> str:
    base = os.path.basename(str(name)).strip().lower()
    return base[:-4] if base.endswith('.fcs') else base


def read_condition_table(text: str, sample_names) -> dict:
    """Read a sample-to-condition table (CSV, or tab separated) with a
    header. The first column names the sample (its name, or its file name
    with or without ``.fcs``; case is ignored); the column headed
    condition, group or trial, else the second column, gives its group; any
    other columns are kept as the sample's annotations.

    Returns ``{'groups': {sample: group}, 'annotations': {sample: {col:
    value}}, 'unmatched_rows': [first-column text], 'missing': [loaded
    samples the table does not name]}``."""
    sample_names = list(sample_names)
    lookup = {_norm(n): n for n in sample_names}
    sample_text = text.lstrip('﻿')
    try:
        dialect = csv.Sniffer().sniff(sample_text[:2048], delimiters=',\t;')
    except csv.Error:
        dialect = csv.excel
    rows = [r for r in csv.reader(io.StringIO(sample_text), dialect)
            if any(c.strip() for c in r)]
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError("The table needs a header row and at least two "
                         "columns: the sample, then its condition.")
    header = [h.strip() for h in rows[0]]
    lower = [h.lower() for h in header]
    gcol = next((i for i, h in enumerate(lower)
                 if h in ('condition', 'group', 'trial')), 1)
    groups, notes, unmatched = {}, {}, []
    for r in rows[1:]:
        r = [c.strip() for c in r] + [''] * (len(header) - len(r))
        name = lookup.get(_norm(r[0]))
        if name is None:
            unmatched.append(r[0])
            continue
        if r[gcol]:
            groups[name] = r[gcol]
        notes[name] = {header[i]: r[i] for i in range(1, len(header))
                       if i != gcol and r[i]}
    missing = [n for n in sample_names if n not in groups]
    return {'groups': groups, 'annotations': notes,
            'unmatched_rows': unmatched, 'missing': missing}


def parse_well(text) -> tuple[int, int] | None:
    """``(row, column)``, both from 0, of a 96-well plate position written
    as A1, A01, a-1 or B_07; None when ``text`` names no well."""
    m = _WELL.search(str(text or ''))
    if not m:
        return None
    return PLATE_ROWS.index(m.group(1).upper()), int(m.group(2)) - 1


def well_label(pos) -> str:
    r, c = pos
    return f"{PLATE_ROWS[r]}{c + 1:02d}"


def well_of(sample) -> tuple[int, int] | None:
    """The sample's plate well: from its well keyword, else from its name,
    else None."""
    kw = keywords(sample)
    for k in WELL_KEYWORDS:
        pos = parse_well(kw.get(k))
        if pos is not None:
            return pos
    return parse_well(getattr(sample, 'name', ''))


def plate_matrix(values: dict) -> np.ndarray:
    """An 8 x 12 array of ``{(row, col): value}``, NaN where no well has a
    value."""
    out = np.full((len(PLATE_ROWS), PLATE_COLS), np.nan)
    for (r, c), v in values.items():
        if 0 <= r < len(PLATE_ROWS) and 0 <= c < PLATE_COLS and v is not None:
            out[r, c] = float(v)
    return out
