"""Headless FCS export — write pandas DataFrames of events back to FCS 3.1
files (via FlowIO), and export a set of named populations to a directory.

No Tk, no gate logic: callers pass already-subset DataFrames (the GUI computes
the masks). Columns are channels; an optional ``channel_labels`` map populates
the per-parameter ``$PnS`` antibody names, so the files re-open in FlowJo /
FCS Express with labels intact. Non-finite cells are zeroed (FCS stores finite
floats). numpy / pandas / flowio only.
"""
from __future__ import annotations

import logging
import os
import re

import numpy as np
import pandas as pd

__all__ = ['write_fcs', 'export_populations', 'population_export_frame',
           'source_keywords', 'safe_filename']

log = logging.getLogger(__name__)

# Characters that are unsafe in a filename on Windows / POSIX.
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_filename(name: str) -> str:
    """Turn an arbitrary population name into a filesystem-safe stem:
    spaces → ``_``, path-unsafe characters stripped, trailing dots/spaces
    removed. Falls back to ``population`` if nothing usable remains."""
    # Treat BOTH separators as separators regardless of OS (so a name with a
    # backslash behaves the same on Linux/macOS as on Windows) → deterministic.
    stem = str(name).strip()
    for sep in ('/', '\\', os.sep, os.altsep or ''):
        if sep:
            stem = stem.replace(sep, '_')
    stem = stem.replace(' ', '_')
    stem = _UNSAFE.sub('', stem)
    stem = stem.strip('. ')
    return stem or 'population'


_DEFAULT_RANGE = 262144          # 2^18: flowio's and most cytometers' $PnR


def population_export_frame(sample, mask):
    """``(frame, ranges, dropped)``: the events of `sample` selected by the
    boolean `mask`, as they belong in an FCS file.

    An FCS file holds measurements, which FlowJo, FCS Express and this app's
    loader transform themselves, so `frame` holds each column's compensated
    LINEAR values (the sample's recorded display transform undone). Writing
    the editor's display coordinates made a re-import transform them twice
    (measured: a FL1 population exported at median 0.97 where 200,003 is
    right, and re-opened at 0.11). `ranges` maps each column to its $PnR:
    the source file's when it has one, else 262144 or the next power of two
    above the data. `dropped` names the columns an FCS cannot carry: text /
    categorical ones, and those whose display scale is unknown (no linear
    values exist)."""
    from .pipeline import UnknownScaleError, linear_values
    mask = np.asarray(mask, dtype=bool)
    sub = sample.data.loc[mask]
    cols, dropped = {}, []
    for c in sub.columns:
        col = sub[c]
        if pd.api.types.is_bool_dtype(col):
            cols[c] = col.to_numpy(dtype=float)
        elif not pd.api.types.is_numeric_dtype(col):
            dropped.append(str(c))
        else:
            try:
                cols[c] = linear_values(sample, c, col.to_numpy(dtype=float))
            except UnknownScaleError:
                dropped.append(str(c))
    frame = pd.DataFrame(cols, index=sub.index)
    md = {str(k).lower().lstrip('$'): v
          for k, v in (getattr(sample, 'metadata', None) or {}).items()}
    source = {}
    for i in range(1, int(md.get('par', 0) or 0) + 1):
        if md.get(f'p{i}n') and md.get(f'p{i}r'):
            source[str(md[f'p{i}n'])] = str(md[f'p{i}r'])
    ranges = {}
    for c in frame.columns:
        if str(c) in source:
            ranges[str(c)] = source[str(c)]
            continue
        v = frame[c].to_numpy(dtype=float)
        v = np.abs(v[np.isfinite(v)])
        if v.size and v.max() > _DEFAULT_RANGE:
            ranges[str(c)] = str(int(2 ** np.ceil(np.log2(v.max()))))
        else:
            ranges[str(c)] = str(_DEFAULT_RANGE)
    return frame, ranges, dropped


# Acquisition keywords an exported population keeps from its source: without
# $TIMESTEP its Time channel has no unit; the rest say what was measured,
# when and on what. $SPILL is NOT among them -- the values are compensated.
_CARRIED_KEYWORDS = ('TIMESTEP', 'DATE', 'BTIM', 'ETIM', 'CYT')


def source_keywords(sample):
    """``(keywords, voltages)`` from `sample`'s source FCS: the
    `_CARRIED_KEYWORDS` it has, and ``{PnN: $PnV}`` for each parameter with a
    voltage. Both empty for a sample with no FCS metadata."""
    md = {str(k).lower().lstrip('$'): v
          for k, v in (getattr(sample, 'metadata', None) or {}).items()}
    keywords = {k: str(md[k.lower()]) for k in _CARRIED_KEYWORDS
                if md.get(k.lower()) not in (None, '')}
    voltages = {}
    for i in range(1, int(md.get('par', 0) or 0) + 1):
        if md.get(f'p{i}n') and md.get(f'p{i}v') not in (None, ''):
            voltages[str(md[f'p{i}n'])] = str(md[f'p{i}v'])
    return keywords, voltages


def write_fcs(df: pd.DataFrame, path: str, channel_labels: dict | None = None,
              ranges: dict | None = None, keywords: dict | None = None,
              voltages: dict | None = None) -> int:
    """Write a DataFrame of events (rows) × channels (columns) to an FCS 3.1
    file at ``path``. ``channel_labels`` (``{column: antibody}``) populates the
    per-parameter ``$PnS`` marker names; ``ranges`` (``{column: $PnR}``) the
    ranges, which otherwise default to 262144; ``voltages``
    (``{column: $PnV}``) the detector voltages; ``keywords`` adds TEXT
    keywords such as ``{'TIMESTEP': '0.01'}`` (see `source_keywords`).
    Non-finite cells are zeroed. Returns the number of events written."""
    import flowio
    channels = [str(c) for c in df.columns]
    arr = df.to_numpy(dtype=float)
    n_bad = int(np.count_nonzero(~np.isfinite(arr)))
    mat = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if n_bad:
        # Zeroing is documented, but count it at runtime so a lossy export isn't
        # completely silent (0.0 is a real value in flow space).
        log.warning("write_fcs: zeroed %d non-finite cell(s) writing %s",
                    n_bad, path)
    opt = ([str((channel_labels or {}).get(c, '') or '') for c in channels]
           if channel_labels else None)
    ranges, voltages = ranges or {}, voltages or {}
    md = {str(k): str(v) for k, v in (keywords or {}).items()}
    md.update({f'P{i}R': str(ranges[c]) for i, c in enumerate(channels, 1)
               if c in ranges})
    md.update({f'P{i}V': str(voltages[c]) for i, c in enumerate(channels, 1)
               if c in voltages})
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, mat.flatten().tolist(), channels,
                          opt_channel_names=opt, metadata_dict=md or None)
    return len(mat)


def export_populations(populations: dict[str, pd.DataFrame], out_dir: str,
                       channel_labels: dict | None = None,
                       ranges: dict | None = None,
                       population_ranges: dict | None = None,
                       keywords: dict | None = None,
                       voltages: dict | None = None) -> list[str]:
    """Write each ``name -> DataFrame`` in ``populations`` to
    ``<out_dir>/<safe_name>.fcs`` (names sanitised via :func:`safe_filename`).
    Colliding sanitised names are disambiguated with a numeric suffix so no
    file silently overwrites another. ``ranges`` sets each column's $PnR for
    every file; ``population_ranges`` (``{name: {column: $PnR}}``) sets one
    population's own, which a range taken from the data must be.
    ``keywords`` / ``voltages`` go to every file (see `write_fcs`).
    Returns the list of written paths in insertion order."""
    os.makedirs(out_dir, exist_ok=True)
    paths: list[str] = []
    used: set[str] = set()
    for name, df in populations.items():
        stem = safe_filename(name)
        candidate = stem
        i = 1
        while candidate.lower() in used:
            candidate = f'{stem}_{i}'
            i += 1
        used.add(candidate.lower())
        path = os.path.join(out_dir, f'{candidate}.fcs')
        write_fcs(df, path, channel_labels=channel_labels,
                  ranges=(population_ranges or {}).get(name, ranges),
                  keywords=keywords, voltages=voltages)
        paths.append(path)
    return paths
