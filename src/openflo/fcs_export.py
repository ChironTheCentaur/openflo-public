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
from array import array

import numpy as np
import pandas as pd

from .export_stamp import FCS_CORRECTIONS_KEY, fcs_keywords

__all__ = ['write_fcs', 'fcs_event_block', 'export_populations', 'population_export_frame',
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
    # A channel the loader decoded from log / gain storage holds values up
    # to its linear equivalent, not up to the stored $PnR.
    for c, top in (getattr(sample, 'linear_ranges', None) or {}).items():
        source[str(c)] = str(int(np.ceil(top)))
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


# TEXT keywords that describe the file's own layout or the values as
# stored, which the exported file sets for itself (or must not carry): its
# segments, data type, event and parameter counts, each parameter's $Pn*
# keywords, and anything about compensation -- the exported values are
# compensated, so a spillover or an "apply compensation" flag carried over
# would have them compensated twice.
_STRUCTURAL = re.compile(
    r'(begin|end)(analysis|data|stext)|byteord|datatype|mode|nextdata|par|'
    r'tot|p\d+[a-z]+|unicode|.*spill.*|.*comp.*', re.I)


def source_keywords(sample):
    """``(keywords, voltages)`` from `sample`'s source FCS: every keyword
    that is not about the file's own layout (see `_STRUCTURAL`) -- $TIMESTEP
    gives Time its unit, $DATE / $CYT / $SRC / $FIL / $OP and the
    acquisition software's own (tube, well, plate) say what was measured,
    when, where and on what -- and ``{PnN: $PnV}`` for each parameter with
    a voltage. Only six were carried before, so an exported population lost
    its tube, well and source names.

    The values an export writes are compensated, so the file also records
    that, with the matrix, in OpenFlo's own keywords (OPENFLO_MATRIX_SOURCE,
    and OPENFLO_APPLIED_MATRIX in $SPILLOVER's layout). Their names say
    neither 'spill' nor 'comp': software that looks for a spillover keyword
    loosely must not apply it a second time, and a reader can still tell
    what was done. Both empty for a sample with no FCS metadata."""
    md = {str(k).lstrip('$'): v
          for k, v in (getattr(sample, 'metadata', None) or {}).items()}
    keywords = {}
    for k, v in md.items():
        if v in (None, '') or _STRUCTURAL.fullmatch(k.lower()):
            continue
        if k.lower() == 'fil':
            # $FIL names THIS file; the source's name is kept beside it.
            keywords['OPENFLO_SOURCE_FIL'] = str(v)
            continue
        keywords[k.upper()] = str(v)
    low = {k.lower(): v for k, v in md.items()}
    voltages = {}
    for i in range(1, int(low.get('par', 0) or 0) + 1):
        if low.get(f'p{i}n') and low.get(f'p{i}v') not in (None, ''):
            voltages[str(low[f'p{i}n'])] = str(low[f'p{i}v'])
    m = getattr(sample, 'comp_matrix', None)
    chans = list(getattr(sample, 'comp_channels', None) or [])
    if m is not None and chans:
        m = np.asarray(m, dtype=float)
        keywords['OPENFLO_MATRIX_SOURCE'] = str(
            getattr(sample, 'compensation_source', None) or 'yes')
        keywords['OPENFLO_APPLIED_MATRIX'] = ','.join(
            [str(len(chans))] + [str(c) for c in chans]
            + [repr(float(x)) for x in m.ravel()])
    return keywords, voltages


def fcs_event_block(mat) -> array:
    """The event matrix as the float32 block ``flowio.create_fcs`` writes,
    ready to pass in its place.

    ``create_fcs`` takes a flat list and converts it with
    ``array('f', values)``. Handing it ``mat.flatten().tolist()`` built one
    Python float per value first: 32 million objects for 2M events x 16
    channels, about 9 s and 1 GB. This converts in numpy instead. Both round
    each double to the nearest float32 (the C cast), so the file is
    byte-for-byte the same; ``create_fcs`` writes an ``array('f')`` it is
    given as is."""
    block = array('f')
    # Doubles beyond float32's range become +/-inf in both; the old path did
    # that silently, so this does too.
    with np.errstate(over='ignore'):
        block.frombytes(np.ascontiguousarray(mat, dtype=np.float32).tobytes())
    return block


def write_fcs(df: pd.DataFrame, path: str, channel_labels: dict | None = None,
              ranges: dict | None = None, keywords: dict | None = None,
              voltages: dict | None = None, corrections=None) -> int:
    """Write a DataFrame of events (rows) × channels (columns) to an FCS 3.1
    file at ``path``. ``channel_labels`` (``{column: antibody}``) populates the
    per-parameter ``$PnS`` marker names; ``ranges`` (``{column: $PnR}``) the
    ranges, which otherwise default to 262144; ``voltages``
    (``{column: $PnV}``) the detector voltages; ``keywords`` adds TEXT
    keywords such as ``{'TIMESTEP': '0.01'}`` (see `source_keywords`).
    OPENFLO_VERSION (and OPENFLO_CORRECTIONS, from ``corrections``) say
    which OpenFlo wrote the file. Non-finite cells are zeroed. Returns the
    number of events written."""
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
    # A channel with no marker is labelled with its own name in the editor;
    # $PnS is the marker, so it stays empty rather than repeating $PnN.
    opt = None
    if channel_labels:
        opt = [str(channel_labels.get(c, '') or '') for c in channels]
        opt = ['' if lab == c else lab for lab, c in zip(opt, channels, strict=True)]
    ranges, voltages = ranges or {}, voltages or {}
    md = {str(k): str(v) for k, v in (keywords or {}).items()}
    # This file's stamp, never one carried over from an OpenFlo-written
    # source (source_keywords copies its keywords).
    md.pop(FCS_CORRECTIONS_KEY, None)
    md.update(fcs_keywords(corrections))
    md.update({f'P{i}R': str(ranges[c]) for i, c in enumerate(channels, 1)
               if c in ranges})
    md.update({f'P{i}V': str(voltages[c]) for i, c in enumerate(channels, 1)
               if c in voltages})
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, fcs_event_block(mat), channels,
                          opt_channel_names=opt, metadata_dict=md or None)
    return len(mat)


def export_populations(populations: dict[str, pd.DataFrame], out_dir: str,
                       channel_labels: dict | None = None,
                       ranges: dict | None = None,
                       population_ranges: dict | None = None,
                       keywords: dict | None = None,
                       voltages: dict | None = None,
                       corrections=None) -> list[str]:
    """Write each ``name -> DataFrame`` in ``populations`` to
    ``<out_dir>/<safe_name>.fcs`` (names sanitised via :func:`safe_filename`).
    Colliding sanitised names are disambiguated with a numeric suffix so no
    file silently overwrites another. ``ranges`` sets each column's $PnR for
    every file; ``population_ranges`` (``{name: {column: $PnR}}``) sets one
    population's own, which a range taken from the data must be.
    ``keywords`` / ``voltages`` / ``corrections`` go to every file (see
    `write_fcs`).
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
                  keywords=keywords, voltages=voltages,
                  corrections=corrections)
        paths.append(path)
    return paths
