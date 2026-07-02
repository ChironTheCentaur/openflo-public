"""Headless FCS export — write pandas DataFrames of events back to FCS 3.1
files (via FlowIO), and export a set of named populations to a directory.

No Tk, no gate logic: callers pass already-subset DataFrames (the GUI computes
the masks). Columns are channels; an optional ``channel_labels`` map populates
the per-parameter ``$PnS`` antibody names, so the files re-open in FlowJo /
FCS Express with labels intact. Non-finite cells are zeroed (FCS stores finite
floats). numpy / pandas / flowio only.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

__all__ = ['write_fcs', 'export_populations', 'safe_filename']

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


def write_fcs(df: pd.DataFrame, path: str, channel_labels: dict | None = None) -> int:
    """Write a DataFrame of events (rows) × channels (columns) to an FCS 3.1
    file at ``path``. ``channel_labels`` (``{column: antibody}``) populates the
    per-parameter ``$PnS`` marker names. Non-finite cells are zeroed. Returns
    the number of events written."""
    import flowio
    channels = [str(c) for c in df.columns]
    mat = np.nan_to_num(df.to_numpy(dtype=float), nan=0.0, posinf=0.0,
                        neginf=0.0)
    opt = ([str((channel_labels or {}).get(c, '') or '') for c in channels]
           if channel_labels else None)
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, mat.flatten().tolist(), channels,
                          opt_channel_names=opt)
    return len(mat)


def export_populations(populations: dict[str, pd.DataFrame], out_dir: str,
                       channel_labels: dict | None = None) -> list[str]:
    """Write each ``name -> DataFrame`` in ``populations`` to
    ``<out_dir>/<safe_name>.fcs`` (names sanitised via :func:`safe_filename`).
    Colliding sanitised names are disambiguated with a numeric suffix so no
    file silently overwrites another. Returns the list of written paths in
    insertion order."""
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
        write_fcs(df, path, channel_labels=channel_labels)
        paths.append(path)
    return paths
