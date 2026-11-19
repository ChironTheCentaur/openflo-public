"""Drive a ``Document`` by named calls with plain-data answers -- Tk-free.

A front end that cannot hold the Python object itself (a JavaScript UI
talking to a Pyodide worker, a test driving the same calls) sends
requests ``{'id', 'method', 'args'}`` and gets back replies ``{'id', 'ok',
'result'}`` or ``{'id', 'ok': False, 'error': {'type', 'message'}}``
(``DocumentRpc.handle``). Only the ``Document`` calls in ``METHODS`` can be
named; ``args`` are keyword arguments.

Answers are made of dicts, lists, strings, numbers, booleans and None,
plus numpy arrays packed as ``{'__ndarray__': dtype, 'shape': [...],
'data': little-endian bytes}`` (``encode``), which a bridge can hand to
JavaScript as a typed array without copying each number.

``open_fcs`` first reads the file's event count from its HEADER and TEXT
segments (``fcs_event_count``, which never reads the DATA segment) and
refuses a file above ``max_events`` with a message saying so, before any
event is loaded: a phone's WebView is killed outright when it runs out of
memory.

Core module: no Tk.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import numpy as np

from .document import Document

# The Document calls a request may name. ``active`` is a property: its value
# is the answer.
METHODS = frozenset({
    'open_fcs', 'open_session', 'import_wsp', 'remove_sample',
    'samples', 'active', 'channels', 'keywords', 'set_channel_label', 'gate', 'gate_tree',
    'add_gate', 'add_quadrant', 'copy_gates', 'restore_gates',
    'apply_gatingml', 'update_gate',
    'delete_gate',
    'lost_operands', 'same_population', 'stats', 'to_view',
    'density', 'density_events', 'histogram', 'overlays', 'compensation',
    'set_compensation', 'save_session', 'export_stats_csv', 'export_stats_plt',
    'export_population_fcs', 'export_wsp', 'data_files',
    'unpack_project',
})

# Byte width of each FCS $DATATYPE when $PnB does not say (it always should).
_DATATYPE_BITS = {'F': 32, 'D': 64}


class RpcError(ValueError):
    """A request that cannot be carried out as asked."""


class TooManyEvents(RpcError):
    """An FCS file holds more events than the front end can load."""


def _text_keywords(path: str) -> tuple[dict[str, str], bytes]:
    """The primary TEXT keywords of the FCS file at `path` (upper-case keys,
    ``$`` kept) and its 58-byte HEADER."""
    from .fcs_text import fcs_text_pairs
    with open(path, 'rb') as fh:
        header = fh.read(58)
    if len(header) < 58 or not header.startswith(b'FCS'):
        raise RpcError(f"{os.path.basename(path)} is not an FCS file")
    try:
        pairs = fcs_text_pairs(path)
    except (ValueError, IndexError) as exc:
        raise RpcError(
            f"{os.path.basename(path)}: unreadable FCS TEXT ({exc})") from exc
    return {str(k).strip().upper(): str(v).strip() for k, v in pairs}, header


def _offset(in_header: bytes, in_text: str | None) -> int:
    """A DATA offset: the HEADER's, unless it is 0 (an offset past
    99,999,999 is written in TEXT only)."""
    value = int(in_header.strip() or b'0')
    return value if value > 0 else int(in_text or '')


def _count_from_data_size(kw: Mapping[str, str], header: bytes) -> int | None:
    """Events implied by the DATA segment's size and the bits per event, or
    None when either is unknown."""
    try:
        start = _offset(header[26:34], kw.get('$BEGINDATA'))
        end = _offset(header[34:42], kw.get('$ENDDATA'))
        n_par = int(kw['$PAR'])
    except (KeyError, ValueError):
        return None
    bits = 0
    for i in range(1, n_par + 1):
        b = kw.get(f'$P{i}B', '')
        if b.isdigit():
            bits += int(b)
        elif kw.get('$DATATYPE', '').upper() in _DATATYPE_BITS:
            bits += _DATATYPE_BITS[kw['$DATATYPE'].upper()]
        else:
            return None
    if bits <= 0 or end < start:
        return None
    return ((end - start + 1) * 8) // bits


def fcs_event_count(path: str) -> int | None:
    """The number of events in the FCS file at `path`, read from its HEADER
    and TEXT only: ``$TOT``, else the DATA segment's size over the bits of
    one event. None when neither can be worked out. Raises ``RpcError`` for
    a file that is not FCS."""
    kw, header = _text_keywords(path)
    tot = kw.get('$TOT', '')
    if tot.isdigit():
        return int(tot)
    return _count_from_data_size(kw, header)


def check_event_cap(path: str, max_events: int | None) -> None:
    """Raise ``TooManyEvents`` when the FCS file at `path` holds more than
    `max_events` events (None: no cap). A file whose count cannot be read
    from its header is refused too, since its size is unknown."""
    if max_events is None:
        return
    n = fcs_event_count(path)
    name = os.path.basename(path)
    if n is None:
        raise TooManyEvents(
            f"{name}: its event count cannot be read from the file header, "
            "so it is not opened here.")
    if n > max_events:
        raise TooManyEvents(
            f"{name} has {n:,} events; this device opens files of up to "
            f"{max_events:,} events. Open it in OpenFlo on a computer, or "
            "export a smaller subset there.")


def encode(value: Any) -> Any:
    """`value` as plain data: numpy arrays packed (see the module
    docstring), numpy scalars as Python numbers, tuples and sets as lists,
    mapping keys as strings."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'biuf':
            return encode(value.tolist())
        arr = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder('<'))
        return {'__ndarray__': arr.dtype.name, 'shape': list(arr.shape),
                'data': arr.tobytes()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [encode(v) for v in value]
    return value


class DocumentRpc:
    """One ``Document`` and the calls a front end makes on it.

    `max_events` caps the events of a file ``open_fcs`` loads (None: no
    cap); the remaining keyword arguments go to ``Document``."""

    def __init__(self, *, max_events: int | None = None,
                 **document_options: Any) -> None:
        self.max_events = max_events
        self._options = document_options
        self.doc = Document(**document_options)

    def reset(self) -> None:
        """Drop every sample: a new, empty Document."""
        self.doc = Document(**self._options)

    def call(self, method: str, args: Mapping[str, Any] | None = None) -> Any:
        """Carry out `method` with keyword `args`; the encoded answer."""
        if method == 'reset':
            self.reset()
            return None
        if method not in METHODS:
            raise RpcError(f"unknown method {method!r}")
        kwargs = dict(args or {})
        if any(not isinstance(k, str) for k in kwargs):
            raise RpcError("argument names must be strings")
        if method == 'active':
            if kwargs:
                raise RpcError("'active' takes no arguments")
            return self.doc.active
        if method == 'open_fcs':
            check_event_cap(str(kwargs.get('path', '')), self.max_events)
        return encode(getattr(self.doc, method)(**kwargs))

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """The reply to `request` (see the module docstring). Every failure
        is a reply; none is raised."""
        rid = request.get('id') if isinstance(request, Mapping) else None
        try:
            if not isinstance(request, Mapping):
                raise RpcError("a request must be a mapping")
            args = request.get('args')
            if args is not None and not isinstance(args, Mapping):
                raise RpcError("'args' must be a mapping")
            result = self.call(str(request.get('method', '')), args)
        except Exception as exc:  # noqa: BLE001 - every failure becomes a reply the UI shows
            return {'id': rid, 'ok': False,
                    'error': {'type': type(exc).__name__,
                              'message': str(exc)}}
        return {'id': rid, 'ok': True, 'result': result}
