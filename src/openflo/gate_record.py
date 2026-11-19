"""Is a gate record whole? ``gate_record_problem`` says why a gate dict
cannot be evaluated: a kind this version of OpenFlo does not know, or a
field its kind needs that is missing or of the wrong type.

``gate_eval.gate_to_mask`` makes such a gate select no events, and
``gate_eval.unknown_population_reason`` reports its population (and every
population under it) as unknown with this reason, so statistics show NA,
never the parent's numbers under the gate's name. A session written by a
newer OpenFlo with a new gate kind, or a damaged session, then reads as
"cannot be computed" instead of a real-looking population.

Core module: numpy and the standard library. No Tk.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

# Every kind gate_eval evaluates. A kind added there must be added here; a
# test (tests/test_gate_record.py) checks the two lists agree.
KNOWN_KINDS = frozenset({
    'threshold', 'interval', 'rect', 'polygon', 'ellipsoid', 'cluster',
    'category', 'flowjo_ellipse', 'boolean', 'autoclean', 'group'})

# The channel fields and numeric fields each geometric kind must carry.
_CHANNELS: dict[str, tuple[str, ...]] = {
    'threshold': ('channel',), 'interval': ('channel',),
    'rect': ('x_channel', 'y_channel'),
    'polygon': ('x_channel', 'y_channel'),
    'ellipsoid': ('x_channel', 'y_channel')}
_NUMBERS: dict[str, tuple[str, ...]] = {
    'threshold': ('value',), 'interval': ('lo', 'hi'),
    'rect': ('x0', 'x1', 'y0', 'y1')}


def _is_number(v: Any) -> bool:
    """Whether `v` reads as a number the evaluators can compare with: an
    int, a float or numeric text, not NaN and not a bool."""
    if isinstance(v, bool):
        return False
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return not math.isnan(f)


def _float_shape(v: Any) -> tuple[int, ...] | None:
    """The shape of `v` read as an array of numbers (a list, a tuple or an
    array, as the evaluators read it), or None when it does not read as
    one or holds NaN (None itself reads as NaN)."""
    try:
        arr = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(arr).any() else arr.shape


def _is_label(v: Any) -> bool:
    """Whether `v` can be matched against a label column's values: one
    value, not a list, a tuple, a set, a mapping or None."""
    return v is not None and not isinstance(
        v, (list, tuple, set, frozenset, Mapping, np.ndarray))


def _shape_problem(kind: str, g: Mapping[str, Any]) -> str | None:
    """The polygon / ellipse / label / boolean / auto-clean field that is
    missing or malformed, or None."""
    if kind == 'polygon':
        shape = _float_shape(g.get('vertices'))
        if shape is None or len(shape) != 2 or shape[1] != 2 or shape[0] < 3:
            return 'polygon vertices are missing or damaged'
    if kind == 'ellipsoid':
        if (_float_shape(g.get('mean')) != (2,)
                or _float_shape(g.get('cov')) != (2, 2)):
            return 'ellipse center or shape is missing or damaged'
        if 'distance_sq' in g and not _is_number(g['distance_sq']):
            return "ellipse field 'distance_sq' is not a number"
    if kind in ('cluster', 'category'):
        col = g.get('channel')
        if col is not None and not isinstance(col, str):
            return 'label column name is damaged'
        key = 'cluster_id' if kind == 'cluster' else 'value'
        if not _is_label(g.get(key)):
            return f"field '{key}' is missing or damaged"
    if kind == 'boolean':
        ops = g.get('operands')
        if ops is not None and not (isinstance(ops, (list, tuple)) and all(
                isinstance(o, (str, int)) and not isinstance(o, bool)
                for o in ops)):
            return 'boolean operands are damaged'
    if kind == 'autoclean':
        methods = g.get('methods')
        if methods is not None and not (isinstance(methods, (list, tuple)) and all(
                isinstance(m, Mapping) and isinstance(
                    m.get('params') or {}, Mapping) for m in methods)):
            return 'auto-clean methods are damaged'
    return None


def gate_record_problem(g: Any) -> str | None:
    """Why gate record `g` cannot be evaluated, in words for the
    statistics table, or None when it can.

    Checked: `g` is a mapping; its ``kind`` is one this version knows
    (``KNOWN_KINDS``); a geometric kind's channels are text and its bounds
    numbers; a polygon has at least three [x, y] vertices, an ellipse a
    2-number centre and a 2 x 2 shape; a cluster / category gate a label to
    match; a boolean's operands are gate ids; an auto-clean gate's methods
    are records. Whether the values make sense (a channel the sample has,
    an ellipse that is an ellipse) is gate_eval's business."""
    if not isinstance(g, Mapping):
        return 'gate record is damaged'
    kind = g.get('kind')
    if kind is None:
        return 'gate has no kind'
    if not isinstance(kind, str) or kind not in KNOWN_KINDS:
        return (f'gate kind {str(kind)[:40]!r} is not known to this '
                'version of OpenFlo')
    for key in _CHANNELS.get(kind, ()):
        if not isinstance(g.get(key), str) or not g.get(key):
            return f"gate field '{key}' is missing or damaged"
    for key in _NUMBERS.get(kind, ()):
        if not _is_number(g.get(key)):
            return f"gate field '{key}' is missing or not a number"
    return _shape_problem(kind, g)
