"""Transform records beside a written CSV: the small JSON file
(``<stem>_transforms.json``) that says which transform each column of an
event-table CSV carries, so a reader inverts exactly what was applied instead
of guessing; and the logicle display range the no-record rule tests against.

Core module: numpy + flowutils, no Tk, and it does not import
``openflo.pipeline`` (pipeline imports it).
"""

import contextlib
import os

import numpy as np
from flowutils import transforms

from .channel_transforms import (
    TRANSFORM_METHODS,
    guessed_spec,
    is_guessed_spec,
    is_unknown_spec,
    transform_params,
    transform_spec,
    unknown_spec,
)
from .scales import LOGICLE_A, LOGICLE_M, LOGICLE_T, LOGICLE_W

# ── Transform records beside a written CSV ────────────────────────────────────
#
# A CSV carries values, not the transform baked into them. Every event-table
# CSV OpenFlo writes (session sidecar, workspace _events.csv, pipeline
# _processed.csv, CLI _unmixed.csv) gets a small JSON record beside it so a
# reader inverts exactly what was applied instead of guessing. A workspace
# run's binary _events.npz has one too: the record is named from the stem,
# so it and an _events.csv of the same run share it.

TRANSFORMS_FORMAT = 'openflo-data-transforms'
TRANSFORMS_VERSION = 1

# Headroom past the logicle's top of scale for the no-record rule, in decades.
# 16 x T is a 22-bit spectral cytometer's range, and compensation can push a
# value a few-fold past range; two decades (100 x T) covers both.
_LOGICLE_HEADROOM_DECADES = 2


def transforms_sidecar_path(csv_path):
    """``<stem>_transforms.json`` beside `csv_path`.

    Named from the CSV's FULL stem (``X_events.csv`` ->
    ``X_events_transforms.json``). Stripping ``_events`` / ``_processed``, as
    the ``_labels.json`` lookup does, would hand two CSVs of one folder the
    same record: a session's samples 'a' and 'a_events' both map to
    ``a_transforms.json``."""
    return os.path.splitext(csv_path)[0] + '_transforms.json'


def write_transforms_sidecar(csv_path, transforms, columns, n_rows,
                             corrections=None):
    """Record, beside the CSV at `csv_path`, the transform each of its
    `columns` carries. `transforms` is a ``data_transforms`` dict (partial
    specs are completed with `transform_spec`'s defaults); a column it does
    not name is linear. The column list and the row count `n_rows` are
    stored too, so a reader can tell a record that belongs to another CSV --
    another sample's export with the same panel matches the columns alone.
    The record also names the OpenFlo version that wrote the CSV (and
    ``corrections``): the CSV itself stays a plain table, as every reader
    of it, OpenFlo's own included, expects. Returns the record's path."""
    import json

    from .export_stamp import export_metadata
    stamp = export_metadata(corrections)
    cols = [str(c) for c in columns]
    keep = {}
    for ch, spec in (transforms or {}).items():
        # A linear entry says nothing -- unless it is a guess, which the
        # next reader must still be told.
        if (str(ch) in cols and spec
                and (spec.get('method', 'linear') != 'linear'
                     or is_guessed_spec(spec))):
            if is_unknown_spec(spec):
                keep[str(ch)] = unknown_spec(spec.get('origin'))
                continue
            full = transform_spec(**transform_params(spec))
            keep[str(ch)] = guessed_spec(full) if is_guessed_spec(spec) else full
    path = transforms_sidecar_path(csv_path)
    tmp = path + '.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump({'format': TRANSFORMS_FORMAT,
                       'version': TRANSFORMS_VERSION,
                       'columns': cols, 'rows': int(n_rows),
                       'transforms': keep,
                       'openflo_version': stamp['OpenFlo version'],
                       **({'corrections': stamp['Corrections']}
                          if 'Corrections' in stamp else {})},
                      fh, indent=1, default=float)
        os.replace(tmp, path)          # atomic: the record is whole or absent
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise
    return path


def discard_transforms_sidecar(csv_path):
    """Remove the record beside `csv_path`. Every writer calls this BEFORE
    it rewrites the CSV: if the new record then fails to write, no record is
    better than the previous one, which describes other data (measured: an
    asinh column re-saved and then inverted with its old logicle record,
    median 123.5 where 19,922 is right)."""
    with contextlib.suppress(FileNotFoundError):
        os.remove(transforms_sidecar_path(csv_path))


def recorded_sidecar_rows(csv_path):
    """The row count the transform record beside `csv_path` was written
    for, or None when there is no readable record. A sidecar holding fewer
    rows than that was cut short (a crash mid-write)."""
    import json
    try:
        with open(transforms_sidecar_path(csv_path), encoding='utf-8') as fh:
            rec = json.load(fh)
        return int(rec['rows'])
    except (OSError, ValueError, TypeError, KeyError):
        return None


def read_transforms_sidecar(csv_path, columns, n_rows):
    """The ``data_transforms`` dict recorded beside `csv_path` by
    `write_transforms_sidecar`, or ``None`` when there is no record.

    Raises ValueError for a record that cannot be trusted: unreadable, not
    this format, from a newer OpenFlo, naming a transform this build does not
    know, or written for a column list other than `columns` or a row count
    other than `n_rows` (a stale record beside an edited or replaced CSV)."""
    import json
    path = transforms_sidecar_path(csv_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding='utf-8') as fh:
            rec = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ValueError(f"unreadable ({exc})") from exc
    if not isinstance(rec, dict) or rec.get('format') != TRANSFORMS_FORMAT:
        raise ValueError("not an OpenFlo transform record")
    if int(rec.get('version') or 0) > TRANSFORMS_VERSION:
        raise ValueError("written by a newer OpenFlo")
    if list(rec.get('columns') or []) != [str(c) for c in columns]:
        raise ValueError("written for different columns")
    if rec.get('rows') != int(n_rows):
        raise ValueError(f"written for {rec.get('rows')} rows; the CSV has "
                         f"{int(n_rows)}")
    return transform_record_from_json(rec.get('transforms') or {})


def transform_record_from_json(obj):
    """A ``data_transforms`` dict rebuilt from its JSON form
    (``{column: spec}``): each spec checked -- a known method, numeric
    parameters, 'guessed' a boolean -- and completed with `transform_spec`'s
    defaults; 'linear' entries dropped, as a FlowSample records none, unless
    guessed. Raises ValueError for anything else (a crafted or corrupt
    file)."""
    if not isinstance(obj, dict):
        raise ValueError("not a column -> transform mapping")
    out = {}
    for ch, spec in obj.items():
        if isinstance(spec, dict) and is_unknown_spec(spec):
            out[str(ch)] = unknown_spec(spec.get('origin'))
            continue
        if (not isinstance(spec, dict)
                or spec.get('method') not in TRANSFORM_METHODS):
            raise ValueError(f"unknown transform for {ch!r}")
        guessed = spec.get('guessed', False)
        if not isinstance(guessed, bool):
            raise ValueError(f"bad 'guessed' flag for {ch!r}")
        try:
            full = transform_spec(**{k: v for k, v in spec.items()
                                     if k != 'guessed'})
        except TypeError as exc:
            raise ValueError(f"bad transform for {ch!r} ({exc})") from exc
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               for k, v in full.items() if k != 'method'):
            raise ValueError(f"non-numeric parameter for {ch!r}")
        if guessed:
            out[str(ch)] = guessed_spec(full)
        elif full['method'] != 'linear':
            out[str(ch)] = full
    return out


def logicle_display_bounds(t=LOGICLE_T, m=LOGICLE_M,
                           w=LOGICLE_W, a=LOGICLE_A):
    """The range the loader's logicle maps cytometer data into: its image of
    raw values within +-100 x `t` (the top of scale plus two decades of
    headroom). [-1.22, 1.44] at the defaults.

    The rule for a CSV column with no transform record: it is logicle only
    if every finite value lies inside. Linear intensities run from the
    hundreds up and essentially never do. Computed with flowutils directly,
    so the GPU LUT's clamping cannot narrow it."""
    r = float(t) * 10.0 ** _LOGICLE_HEADROOM_DECADES
    lo, hi = transforms.logicle(np.array([[-r], [r]]), channel_indices=[0],
                                t=t, m=m, w=w, a=a).flatten()
    return float(lo), float(hi)
