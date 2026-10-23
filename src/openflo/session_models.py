"""Analysis models a session keeps with their sample (Tk-free).

Cell cycle and FlowSOM each write label columns (``cell_cycle``;
``flowsom`` / ``flowsom_meta``) AND keep a model on the sample
(``cell_cycle_result``, ``flowsom_result``). The session kept the columns, in
the processed sidecar, but not the models, so a reopened sample had its
phases and metaclusters while the Cell cycle window said "No cell-cycle
result" and the SOM tree "Run FlowSOM first".

A model goes into its sample's session entry as a JSON record (additive keys,
like 'bead_norm', so SESSION_KEYS / SESSION_VERSION are unchanged and an older
OpenFlo ignores them). Neither model holds per-event arrays: FlowSOM's are its
``(n_nodes, n_markers)`` prototypes and its node -> metacluster map; the
per-event node and metacluster ids live in the data columns, which the
sidecar already carries, so they are not written twice.

A record is written only while its columns are in the data, and taken back
only when the reopened sample has them, so a model never outlives the labels
it explains (a recompensation that rebuilds a sample from its FCS drops
both). A record that is malformed or from another format version is ignored
with a reason; it never fails the restore.

A spectral unmix's reference spectra and QC report, which the editor keeps,
go into the entry of one unmixed sample the same way (see below).
"""
from __future__ import annotations

import math

CELL_CYCLE_KEY = 'cell_cycle_model'
FLOWSOM_KEY = 'flowsom_model'
RECORD_VERSION = 1

# Model values JSON cannot carry exactly are written as tagged objects, so
# the model reads back equal to the one saved: an array keeps its dtype and
# shape, a tuple stays a tuple (FlowSOM's grid), and a NaN / inf stays one
# (an unfitted cell-cycle model's means are NaN, which the session writer
# would otherwise turn into null).
_ND, _TUPLE, _FLOAT = '$ndarray', '$tuple', '$float'


def _encode(v):
    import numpy as np
    if isinstance(v, np.ndarray):
        if v.dtype.kind not in 'biuf':
            raise TypeError(f"array of {v.dtype} is not a model value")
        return {_ND: _encode(v.tolist()), 'dtype': v.dtype.str}
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return {_FLOAT: repr(v)}            # 'nan' / 'inf' / '-inf'
    if isinstance(v, tuple):
        return {_TUPLE: [_encode(x) for x in v]}
    if isinstance(v, list):
        return [_encode(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _encode(x) for k, x in v.items()}
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    raise TypeError(f"{type(v).__name__} is not a model value")


def _decode(v):
    import numpy as np
    if isinstance(v, list):
        return [_decode(x) for x in v]
    if not isinstance(v, dict):
        return v
    if _ND in v:
        dt = np.dtype(str(v['dtype']))
        if dt.kind not in 'biuf':
            raise ValueError(f"array dtype {dt} is not numeric")
        return np.asarray(_decode(v[_ND]), dtype=dt)
    if _TUPLE in v:
        return tuple(_decode(x) for x in v[_TUPLE])
    if _FLOAT in v:
        return float(v[_FLOAT])
    return {k: _decode(x) for k, x in v.items()}


def _record(model):
    return {'version': RECORD_VERSION, 'model': _encode(model)}


def _model_of(rec):
    """The model a record holds; ValueError when it is not one we wrote."""
    if not isinstance(rec, dict) or rec.get('version') != RECORD_VERSION:
        raise ValueError("unknown record version")
    model = _decode(rec.get('model'))
    if not isinstance(model, dict):
        raise ValueError("no model in the record")
    return model


def _number(model, key):
    v = model.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"'{key}' is not a number")


# ── cell cycle ──────────────────────────────────────────────────────────────

def cell_cycle_record(sample):
    """The session record of ``sample.cell_cycle_result``, or None when it has
    none or its ``cell_cycle`` column is gone."""
    model = getattr(sample, 'cell_cycle_result', None)
    if not isinstance(model, dict) or 'cell_cycle' not in sample.data.columns:
        return None
    return _record(model)


def restore_cell_cycle(sample, rec):
    """Put a saved cell-cycle model back on ``sample``. Returns '' or why it
    was not restored (the sample is then left without one)."""
    try:
        model = _model_of(rec)
        if 'cell_cycle' not in sample.data.columns:
            return "the sample has no cell_cycle column"
        if model.get('channel') not in sample.data.columns:
            return f"its DNA channel {model.get('channel')!r} is not loaded"
        if model.get('ok'):
            # What the Cell cycle window draws.
            for key in ('g1_mean', 'g2_mean', 'g1_hi', 'g2_lo', 'pct_g1',
                        'pct_s', 'pct_g2m'):
                _number(model, key)
    except Exception as exc:     # noqa: BLE001 — a bad record never fails a restore
        return f"unreadable record: {exc}"
    sample.cell_cycle_result = model
    return ''


# ── FlowSOM ─────────────────────────────────────────────────────────────────

_FLOWSOM_COLUMNS = ('flowsom', 'flowsom_meta')


def flowsom_record(sample):
    """The session record of ``sample.flowsom_result``, or None when it has
    none or its node / metacluster columns are gone."""
    model = getattr(sample, 'flowsom_result', None)
    if not isinstance(model, dict) or any(
            c not in sample.data.columns for c in _FLOWSOM_COLUMNS):
        return None
    return _record(model)


def restore_flowsom(sample, rec):
    """Put a saved FlowSOM model back on ``sample``, checked against the
    node column it labelled. Returns '' or why it was not restored."""
    import numpy as np
    try:
        model = _model_of(rec)
        missing = [c for c in _FLOWSOM_COLUMNS
                   if c not in sample.data.columns]
        if missing:
            return f"the sample has no {', '.join(missing)} column"
        grid = tuple(int(g) for g in model['grid'])
        n_nodes = int(np.prod(grid))
        W = model['weights']
        chans = model['channels']
        mon = model['meta_of_node']
        _number(model, 'n_metaclusters')
        if (not isinstance(W, np.ndarray) or W.ndim != 2
                or W.shape != (n_nodes, len(chans))
                or int(model.get('n_nodes', n_nodes)) != n_nodes):
            return "its prototypes do not match its grid and channels"
        if not isinstance(mon, np.ndarray) or mon.shape != (n_nodes,):
            return "its node-to-metacluster map does not match its grid"
        node = np.asarray(sample.data['flowsom'], dtype=float)
        node = node[np.isfinite(node)]
        if node.size and node.max() >= n_nodes:
            return "the flowsom column names nodes the model does not have"
    except Exception as exc:     # noqa: BLE001 — a bad record never fails a restore
        return f"unreadable record: {exc}"
    sample.flowsom_result = model
    return ''


# ── spectral unmixing ───────────────────────────────────────────────────────
#
# An unmix writes U:<fluor> abundance columns into every sample it unmixed
# and keeps, on the EDITOR, the reference spectra, their detectors and the QC
# report (similarity, spillover spread, condition number). The columns were
# saved; the rest was not, so after a reopen Panel check's "Use last unmix"
# was greyed out and Spectral QC had nothing to show. The record goes into
# the entry of one unmixed sample (the one the unmixed-pairs plot shows), not
# a new top-level key: it describes that sample's U: columns, it is written
# only while they are in the data, and an older OpenFlo ignores it. The
# spectra are fluors x detectors and the QC report fluors x fluors, so it is
# small; the pairs plot's per-event abundances are read back from the
# sample's U: columns rather than written twice.

UNMIX_KEY = 'unmix_model'


def _unmix_columns(fluors):
    return [f'U:{f}' for f in fluors]


def unmix_record(editor):
    """``(sample name, record)`` of the editor's last unmix, or None when it
    has none or no loaded sample still has all of its U: columns. The record
    goes to the sample the unmixed-pairs plot shows, else the first sample
    in order that has the columns."""
    last = getattr(editor, '_last_unmix_spectra', None)
    if not last:
        return None
    spectra, fluors = last
    fluors = [str(f) for f in fluors]
    cols = _unmix_columns(fluors)
    samples = getattr(editor, '_samples', {}) or {}
    pairs = getattr(editor, '_last_unmix_pairs', None)
    order = ([pairs[0]] if pairs else []) + list(
        getattr(editor, '_sample_order', []) or [])
    for name in order:
        s = samples.get(name)
        if s is not None and all(c in s.data.columns for c in cols):
            dets = getattr(editor, '_last_unmix_detectors', None)
            return name, _record({
                'spectra': spectra, 'fluors': fluors,
                'detectors': None if dets is None else [str(d) for d in dets],
                'qc': getattr(editor, '_last_unmix_qc', None)})
    return None


def _check_unmix_qc(qc, fluors):
    import numpy as np
    if qc is None:
        return
    nf = len(fluors)
    if not isinstance(qc, dict) or qc.get('fluors') != fluors:
        raise ValueError("its QC report is for other fluors")
    for key in ('similarity', 'ssm'):
        m = qc.get(key)
        if not isinstance(m, np.ndarray) or m.shape != (nf, nf):
            raise ValueError(f"its QC '{key}' is not {nf} x {nf}")
    _number(qc, 'condition_number')
    for key in ('similar_pairs', 'worst_spread', 'degenerate_fluors',
                'reference_spectra'):
        if not isinstance(qc.get(key), list):
            raise ValueError(f"its QC '{key}' is not a list")


def restore_unmix(editor, name, sample, rec):
    """Put a saved unmix back on `editor` (its spectra, detectors, QC report,
    and `name`'s abundances for the pairs plot), checked against the U:
    columns of `sample`. Returns '' or why it was not restored (the editor
    is then left as it was)."""
    import numpy as np

    from .pipeline import linear_values
    try:
        model = _model_of(rec)
        fluors = model.get('fluors')
        if (not isinstance(fluors, list) or not fluors
                or not all(isinstance(f, str) for f in fluors)):
            return "its fluors are not a list of names"
        missing = [c for c in _unmix_columns(fluors)
                   if c not in sample.data.columns]
        if missing:
            return f"the sample has no {', '.join(missing)} column"
        S = model.get('spectra')
        if (not isinstance(S, np.ndarray) or S.ndim != 2
                or S.shape[0] != len(fluors) or S.dtype.kind != 'f'):
            return "its spectra do not match its fluors"
        dets = model.get('detectors')
        if dets is not None and (
                not isinstance(dets, list) or len(dets) != S.shape[1]
                or not all(isinstance(d, str) for d in dets)):
            return "its detectors do not match its spectra"
        qc = model.get('qc')
        _check_unmix_qc(qc, fluors)
        A = np.column_stack([np.asarray(linear_values(sample, c),
                                        dtype=float)
                             for c in _unmix_columns(fluors)])
    except Exception as exc:     # noqa: BLE001 — a bad record never fails a restore
        return f"unreadable record: {exc}"
    editor._last_unmix_spectra = (S, list(fluors))
    editor._last_unmix_detectors = dets
    editor._last_unmix_qc = qc
    editor._last_unmix_pairs = (name, A, list(fluors))
    return ''
