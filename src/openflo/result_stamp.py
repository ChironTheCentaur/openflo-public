"""Which OpenFlo version produced each result (Tk-free).

A *stamp* records the step that made a result and the OpenFlo version that
ran it, so a fix registry can later find the results made before a fix and
offer to redo them. It is a small JSON-able dict::

    {'step': 'gate.refit', 'version': '2.7.0b1', 'inputs': ['g3', 'FL1-A']}

``step`` names the producing operation (``gate.refit``, ``gate.auto.gmm``,
``cell_cycle``, ``calibration``, ``cluster.leiden``, ``embed.umap``, ...);
``inputs`` lists what it ran on (a parent gate id, channels), as strings.
A gate's stamp also carries ``geom``, a digest of the geometry it was made
with (:func:`seal_gate_stamp`): a gate since moved by hand no longer matches
it, and :func:`settle_gate_stamps` takes the stamp off, so the gate is the
user's and a fix never overwrites the edit.

Where a stamp lives:

* a gate's on the gate dict, under :data:`STAMP_KEY` (it travels with the
  gate through copies, undo and the session; the plot cache ignores it, as it
  ignores the gate's name and colour);
* a computed column's in the sample's ``result_stamps`` attribute,
  ``{column: stamp}``, written to the session's sample entry.

A stamp is written once, when the result is made. Saving a session writes it
back as it was read, so a result made by an older version keeps saying so. A
result with no stamp (made before stamps existed) or a stamp with no version
is *unknown*, which sorts below every version: :func:`made_before` is True
for it, so a fix registry errs on the side of flagging it.
"""
from __future__ import annotations

import functools

STAMP_KEY = 'stamp'          # a gate dict's stamp
SAMPLE_ATTR = 'result_stamps'  # a sample's {column: stamp}
GEOM_KEY = 'geom'            # in a gate's stamp: the geometry it was made with
ORIGINAL_KEY = 'original'    # a corrected gate's geometry and stamp before
ORIGINALS_ATTR = 'result_originals'  # a sample's {column: original}


@functools.lru_cache(maxsize=1)
def app_version() -> str:
    """The running OpenFlo version (:func:`openflo.update.current_version`),
    read once: for a source checkout it asks git, and the code running does
    not change version mid-process."""
    from .update import current_version
    return str(current_version())


def make_stamp(step, inputs=None) -> dict:
    """A new stamp for a result made now by ``step`` from ``inputs``."""
    return {'step': str(step), 'version': app_version(),
            'inputs': [str(v) for v in (inputs or ()) if v is not None]}


def normalize_stamp(d) -> dict | None:
    """``d`` as a well-formed stamp, or None when it is not one. A missing or
    empty version is kept as None (unknown); it is never filled in with the
    running version, which would claim a result this build did not make."""
    if not isinstance(d, dict) or not d.get('step'):
        return None
    v = d.get('version')
    inputs = d.get('inputs')
    out = {'step': str(d['step']),
           'version': str(v) if v not in (None, '') else None,
           'inputs': ([str(x) for x in inputs]
                      if isinstance(inputs, (list, tuple)) else [])}
    if isinstance(d.get(GEOM_KEY), str):
        out[GEOM_KEY] = d[GEOM_KEY]
    return out


def stamp_of(obj) -> dict | None:
    """The stamp of a gate dict (its :data:`STAMP_KEY`), of a stamp dict
    itself, or of an object with a ``stamp`` attribute; None when it has
    none."""
    if isinstance(obj, dict):
        return normalize_stamp(obj.get(STAMP_KEY) if STAMP_KEY in obj
                               else obj)
    return normalize_stamp(getattr(obj, STAMP_KEY, None))


def stamp_gate(gate: dict, step, inputs=None) -> dict:
    """Put a new stamp on ``gate`` (in place), sealed with its geometry, and
    return the gate."""
    gate[STAMP_KEY] = make_stamp(step, inputs)
    return seal_gate_stamp(gate)


# ── A gate edited by hand is the user's ─────────────────────────────────────

# Gate keys that are not where the gate is: how it is shown (the plot cache's
# display-only keys), its place in the tree, and the provenance kept on it.
_NOT_GEOMETRY = frozenset({'name', 'label', 'color', 'open', 'enabled',
                           STAMP_KEY, ORIGINAL_KEY, 'parent_id', 'id'})


def _as_saved(v):
    """``v`` as a session save writes it (editor_session._json_safe): an
    open bound becomes the +-1e12 sentinel, NaN None, a numpy value its
    Python one. A gate reopened from a session then has the digest it was
    saved with."""
    import math
    if hasattr(v, 'tolist') and not isinstance(v, (str, bytes)):
        v = v.tolist()
    if isinstance(v, float):
        if math.isnan(v):
            return None
        if math.isinf(v):
            return -1e12 if v < 0 else 1e12
        return v
    if isinstance(v, dict):
        return {str(k): _as_saved(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_as_saved(x) for x in v]
    return v


def geometry_sig(gate: dict) -> str:
    """A short digest of what selects ``gate``'s events: every key but how it
    is shown, its parent and its provenance (keys starting ``_`` are the
    editor's scratch and left out too)."""
    import hashlib
    import json
    body = {str(k): _as_saved(v) for k, v in gate.items()
            if k not in _NOT_GEOMETRY and not str(k).startswith('_')}
    text = json.dumps(body, sort_keys=True, default=repr,
                      separators=(',', ':'))
    return hashlib.blake2b(text.encode('utf-8', 'surrogatepass'),
                           digest_size=8).hexdigest()


def seal_gate_stamp(gate: dict) -> dict:
    """Record ``gate``'s current geometry in its stamp when the stamp has
    none yet (a stamp made inline, as auto-gate's are, or read from a session
    saved before stamps were sealed). Returns the gate."""
    st = gate.get(STAMP_KEY)
    if isinstance(st, dict) and st.get('step') and GEOM_KEY not in st:
        gate[STAMP_KEY] = dict(st, **{GEOM_KEY: geometry_sig(gate)})
    return gate


def live_gate_stamp(gate) -> dict | None:
    """``gate``'s stamp while the gate is still where the stamped step put
    it; None when it has none or it has been moved since."""
    st = stamp_of(gate) if isinstance(gate, dict) else None
    if st is None:
        return None
    geom = st.get(GEOM_KEY)
    if geom is not None and geom != geometry_sig(gate):
        return None
    return st


def settle_gate_stamps(gates: dict) -> list:
    """Seal each stamped gate of ``{gid: gate}`` and take the stamp off any
    gate moved by hand since it was stamped (a drag, typed coordinates, the
    arrow keys, a slider). Returns the ids whose stamp was dropped. The one
    place hand edits are told from computed results: every edit path ends in
    a gate-list refresh, which calls this, and a save calls it too."""
    dropped = []
    for gid, g in gates.items():
        if not isinstance(g, dict) or STAMP_KEY not in g:
            continue
        seal_gate_stamp(g)
        if live_gate_stamp(g) is None:
            g.pop(STAMP_KEY, None)
            dropped.append(gid)
    return dropped


def as_users_gate(gate: dict) -> dict:
    """``gate`` (in place) without its stamp and kept original: a gate
    pasted or copied onto a sample was not computed on that sample's data."""
    gate.pop(STAMP_KEY, None)
    gate.pop(ORIGINAL_KEY, None)
    return gate


def carry_gate_stamp(before: dict, after: dict) -> dict:
    """``after`` (``before`` with its coordinates converted, as a restore
    onto other transforms does) keeps ``before``'s stamp, resealed, when that
    stamp was live: a change of units is not a hand edit."""
    if live_gate_stamp(before) is not None and STAMP_KEY in after:
        st = dict(after[STAMP_KEY])
        st.pop(GEOM_KEY, None)
        after[STAMP_KEY] = st
        seal_gate_stamp(after)
    return after


# ── A corrected column's original ───────────────────────────────────────────

def original_column(column, version) -> str:
    """The name a corrected column's original values are kept under in the
    sample's data (and so its sidecar): ``'<col> [original 2.6.0]'``."""
    return f"{column} [original {version or 'unknown version'}]"


def is_original_column(name) -> bool:
    """Whether ``name`` is a kept original (:func:`original_column`): not a
    channel to plot, cluster on or put in a table."""
    n = str(name)
    return n.endswith(']') and ' [original ' in n


def originals_from_entry(d) -> dict:
    """A sample entry's ``result_originals`` (``{column: {'stamp',
    'column'}}``), with anything malformed left out."""
    if not isinstance(d, dict):
        return {}
    out = {}
    for c, rec in d.items():
        if not isinstance(rec, dict) or not isinstance(rec.get('column'), str):
            continue
        out[str(c)] = {'stamp': normalize_stamp(rec.get('stamp')),
                       'column': rec['column']}
    return out


def column_originals(sample) -> dict:
    """``sample``'s kept originals whose column is still in its data,
    JSON-ready (for the session's sample entry)."""
    rec = getattr(sample, ORIGINALS_ATTR, None) or {}
    data = getattr(sample, 'data', None)
    cols = set(map(str, data.columns)) if data is not None else set()
    return {c: o for c, o in originals_from_entry(rec).items()
            if o['column'] in cols}


def stamp_columns(sample, columns, step, inputs=None) -> None:
    """Record that ``step`` just wrote ``columns`` of ``sample``."""
    rec = dict(getattr(sample, SAMPLE_ATTR, None) or {})
    for c in columns:
        rec[str(c)] = make_stamp(step, inputs)
    setattr(sample, SAMPLE_ATTR, rec)


def column_stamps(sample) -> dict:
    """``{column: stamp}`` of ``sample``'s columns that are still in its
    data, JSON-ready (for the session's sample entry). A column since dropped
    takes its stamp with it."""
    rec = getattr(sample, SAMPLE_ATTR, None) or {}
    data = getattr(sample, 'data', None)
    cols = set(map(str, data.columns)) if data is not None else set()
    out = {}
    for c, st in rec.items():
        st = normalize_stamp(st)
        if st is not None and str(c) in cols:
            out[str(c)] = st
    return out


def stamps_from_entry(d) -> dict:
    """Inverse of :func:`column_stamps`: a sample entry's ``result_stamps``,
    with anything malformed left out (never an exception that aborts a
    restore)."""
    if not isinstance(d, dict):
        return {}
    out = {}
    for c, st in d.items():
        st = normalize_stamp(st)
        if st is not None:
            out[str(c)] = st
    return out


# ── version order ────────────────────────────────────────────────────────────

def _beta_order(a: str, b: str) -> int:
    """-1, 0 or 1 for two versions at least one of which is a pre-release,
    as openflo.update orders versions (:mod:`packaging` when available, else
    ``update._fallback_key``); 0 when they cannot be read."""
    from .update import _fallback_key
    try:
        from packaging.version import Version
        va, vb = Version(a), Version(b)
        return (va > vb) - (va < vb)
    except (ImportError, ValueError, TypeError):
        pass
    try:
        ka, kb = _fallback_key(a), _fallback_key(b)
    except (ValueError, TypeError):
        return 0
    return (ka > kb) - (ka < kb)


def compare_versions(a, b) -> int:
    """-1, 0 or 1 as version ``a`` is older than, the same as or newer than
    ``b``. None or '' is unknown, older than every version. The order is
    :func:`openflo.update.is_newer`'s, without its update-nag rule (a final
    release never sees a beta as newer): here a fix in 2.7.0b2 must still
    count as newer than a result from 2.6.2. Text that cannot be read as a
    version gives 0 (no order is claimed)."""
    from .update import _is_prerelease, is_newer
    a = str(a) if a not in (None, '') else None
    b = str(b) if b not in (None, '') else None
    if a is None or b is None:
        return (a is not None) - (b is not None)
    if a == b:
        return 0
    if not (_is_prerelease(a) or _is_prerelease(b)):
        return 1 if is_newer(a, b) else (-1 if is_newer(b, a) else 0)
    return _beta_order(a, b)


def made_before(stamp_or_version, version) -> bool:
    """True when a result (its stamp, or its version string) was made by a
    version older than ``version`` -- or by an unknown one."""
    v = stamp_or_version
    if v is not None and not isinstance(v, str):
        st = stamp_of(v)
        v = st['version'] if st else None
    return compare_versions(v, version) < 0
