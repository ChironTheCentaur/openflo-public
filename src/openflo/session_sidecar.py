"""Processed-data sidecars: the per-sample event tables a saved session keeps
beside its ``.flowsession`` file (``<stem>_data/<name>.npz``).

They used to be CSV. Formatting every value as text made a save about 45x
slower than writing the numbers (measured: 10.4 s against 0.23 s for 300,000
events x 24 channels, at 2.3x the size on disk), and it holds the GIL, so
writing several at once on threads was slower still. A session with a few
large samples took about a minute to close.

The binary sidecar is an uncompressed ``.npz``: one array per column plus
the column names, read back with ``allow_pickle=False`` so a crafted session
can't run code. Every dtype is kept exactly (the CSV turned float32 into
float64 and '' into NaN). A frame holding something this format can't carry
without pickle -- a nullable or categorical column, mixed objects -- is
still written as CSV, and CSV sidecars from older sessions still load.
"""
from __future__ import annotations

import contextlib
import os
from typing import Any

SIDECAR_EXT = '.npz'
LEGACY_EXT = '.csv'
_COLUMNS = '__columns__'
_TEXT = '__text__'


def _column_array(col) -> Any:
    """The numpy array to store for one column, or None when it can't be
    stored without pickle."""
    import numpy as np
    import pandas as pd
    dt = col.dtype
    if isinstance(dt, np.dtype) and dt.kind in 'biuf':
        return col.to_numpy()
    # Text: an object column of str, or pandas' own string dtype (the
    # default for text from pandas 3). Only when no value is missing, since
    # a numpy str array has no NA.
    if (pd.api.types.is_string_dtype(dt) and len(col)
            and pd.api.types.infer_dtype(col, skipna=False) == 'string'):
        return np.asarray(col.to_numpy(dtype=object), dtype=str)
    return None


def _text_kind(dtype) -> str:
    """How a text column comes back: 'o' for a numpy object column, else
    the name of the pandas string dtype it had ('str', 'string')."""
    import numpy as np
    return 'o' if isinstance(dtype, np.dtype) else str(dtype)


def frame_arrays(df) -> list | None:
    """Each column of `df` as a numpy array, or None when one of them can't
    go in a binary sidecar (the caller then writes CSV)."""
    out = []
    for i in range(df.shape[1]):
        arr = _column_array(df.iloc[:, i])
        if arr is None:
            return None
        out.append(arr)
    return out


def write_frame(df, path) -> bool:
    """Write `df` to `path` as a binary sidecar. Returns False, writing
    nothing, when the frame holds a column the format can't carry.

    Atomic: written to a temporary file, then moved over `path`, so a
    failed or interrupted save leaves the previous sidecar intact. That also
    makes every write a new file, never a change to the old one in place."""
    import numpy as np
    arrays = frame_arrays(df)
    if arrays is None:
        return False
    payload = {f'c{i}': a for i, a in enumerate(arrays)}
    payload[_COLUMNS] = np.asarray([str(c) for c in df.columns], dtype=str)
    payload[_TEXT] = np.asarray([_text_kind(df.iloc[:, i].dtype)
                                 if a.dtype.kind == 'U' else ''
                                 for i, a in enumerate(arrays)], dtype=str)
    tmp = path + '.tmp'
    try:
        with open(tmp, 'wb') as fh:
            np.savez(fh, allow_pickle=False, **payload)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise
    return True


def read_frame(path):
    """The DataFrame a `write_frame` sidecar at `path` holds."""
    import numpy as np
    import pandas as pd
    with np.load(path, allow_pickle=False) as z:
        if _COLUMNS not in z.files:
            raise ValueError(f"{os.path.basename(path)}: not an OpenFlo "
                             f"sidecar (no column list)")
        names = [str(c) for c in z[_COLUMNS].tolist()]
        text = (z[_TEXT].tolist() if _TEXT in z.files else [])
        text += [''] * (len(names) - len(text))
        arrays = []
        for i in range(len(names)):
            key = f'c{i}'
            if key not in z.files:
                raise ValueError(f"{os.path.basename(path)}: column "
                                 f"{names[i]!r} is missing")
            a = z[key]
            if a.dtype.kind == 'U':
                a = (pd.Series(a, dtype=object, copy=False)
                     if text[i] in ('', 'o') else pd.array(a, dtype=text[i]))
            arrays.append(a)
    if len({len(a) for a in arrays}) > 1:
        raise ValueError(f"{os.path.basename(path)}: columns differ in "
                         f"length")
    df = pd.DataFrame(dict(enumerate(arrays)), copy=False)
    df.columns = pd.Index(names)
    return df


def read_sidecar(path):
    """Read a processed-data sidecar of either format: binary (this
    module's) or the CSV older sessions and the CSV fallback write."""
    if path.lower().endswith(SIDECAR_EXT):
        return read_frame(path)
    import pandas as pd
    return pd.read_csv(path)
