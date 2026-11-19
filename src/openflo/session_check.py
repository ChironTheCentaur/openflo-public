"""Is a parsed session's structure whole? ``session_structure_problem``
says, in plain words, what in a ``.flowsession`` dict has the wrong shape
for OpenFlo to open it: the sample list, a sample entry's text and record
fields, the per-sample gate lists and each gate's id, parent, name and
operands.

``session_format.migrate_session`` calls it after upgrading, so the desktop
editor and the headless document (the phone) refuse a damaged session the
same way: with ``SessionDamagedError`` and a message that names the sample
or gate, never a raw ``TypeError`` from deep inside the restore.

What a gate SELECTS is not checked here: a gate of an unknown kind, or one
missing a bound, opens and reads as an unknown population
(``gate_record.gate_record_problem``), so one bad gate does not lose the
whole session.

Core module: the standard library only. No Tk.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# A sample entry's fields that hold text (a name, a colour, a path) and
# those that hold a record. Absent or None is always allowed: older
# sessions lack most of them.
# Only fields whose wrong type stops the open or a later step are listed:
# a wrong informational field (``app_version``, say) is not a reason to
# refuse a session that opened before.
_ENTRY_TEXT = ('name', 'path', 'rel_path', 'basename', 'processed_data',
               'processed_csv', 'color', 'trial', 'container_member')
_ENTRY_RECORDS = ('derived',)
# A gate's fields that name it, and those that point at a gate (an id is
# text, or a number in a hand-written file).
_GATE_TEXT = ('name', 'label')
_GATE_IDS = ('id', '_import_id', 'parent_id', 'quad_set')
# The session's own records and lists. Empty values (None, {}, [], 0) read
# as absent wherever they are used.
_TOP_RECORDS = ('channel_scale', 'channel_range', 'channel_labels',
                'cluster_labels')
_TOP_LISTS = ('audit', 'fixes_applied', 'fixes_declined')


def _what(v: Any) -> str:
    """`v`'s type, said for a person ('a list', 'a number', ...)."""
    if isinstance(v, bool):
        return 'true/false'
    if isinstance(v, (int, float)):
        return 'a number'
    if isinstance(v, str):
        return 'text'
    if isinstance(v, (list, tuple)):
        return 'a list'
    if isinstance(v, Mapping):
        return 'a record'
    return 'empty' if v is None else type(v).__name__


def _sample_problem(i: int, entry: Any) -> str | None:
    """What is wrong with sample entry number `i` (from 0), or None."""
    if not isinstance(entry, Mapping):
        return f"sample {i + 1} is {_what(entry)}, not a sample record"
    label = entry.get('name')
    who = (f"sample '{label}'" if isinstance(label, str) and label
           else f"sample {i + 1}")
    for key in _ENTRY_TEXT:
        v = entry.get(key)
        if v is not None and not isinstance(v, str):
            return f"{who}: '{key}' is {_what(v)}, not text"
    for key in _ENTRY_RECORDS:
        v = entry.get(key)
        if v is not None and not isinstance(v, Mapping):
            return f"{who}: '{key}' is {_what(v)}, not a record"
    return None


def _gate_problem(sample: str, j: int, g: Any) -> str | None:
    """What is wrong with gate number `j` (from 0) of `sample`, or None."""
    if not isinstance(g, Mapping):
        return (f"gate {j + 1} of sample '{sample}' is {_what(g)}, "
                "not a gate record")
    for key in _GATE_TEXT:
        v = g.get(key)
        if v is not None and not isinstance(v, str):
            return (f"gate {j + 1} of sample '{sample}': '{key}' is "
                    f"{_what(v)}, not text")
    for key in _GATE_IDS:
        v = g.get(key)
        if v is not None and not _is_id(v):
            return (f"gate {j + 1} of sample '{sample}': '{key}' is "
                    f"{_what(v)}, not a gate id")
    ops = g.get('operands')
    if ops is not None and not (isinstance(ops, list) and all(
            _is_id(o) for o in ops)):
        return (f"gate {j + 1} of sample '{sample}': its boolean operands "
                "are damaged")
    return None


def _is_id(v: Any) -> bool:
    """Whether `v` can be a gate id: text or a whole number."""
    return isinstance(v, str) or (isinstance(v, int)
                                  and not isinstance(v, bool))


def _gates_problem(sample_gates: Any) -> str | None:
    """What is wrong with the session's ``sample_gates``, or None."""
    if not sample_gates:
        return None
    if not isinstance(sample_gates, Mapping):
        return f"the gate list is {_what(sample_gates)}, not a record"
    for sample, gates in sample_gates.items():
        if not isinstance(gates, list):
            return (f"the gates of sample '{sample}' are {_what(gates)}, "
                    "not a list")
        for j, g in enumerate(gates):
            why = _gate_problem(sample, j, g)
            if why:
                return why
    return None


def session_structure_problem(data: Mapping[str, Any]) -> str | None:
    """What in session `data` (parsed, already upgraded) has the wrong
    shape to open, in words for an error message, or None when nothing
    does.

    Checked: the session's records (``_TOP_RECORDS``) and lists
    (``_TOP_LISTS``), when not empty, are records and lists; ``samples``,
    when present, is a list of records whose text fields
    (``_ENTRY_TEXT``) are text and record fields (``_ENTRY_RECORDS``)
    records; ``sample_gates`` maps each sample to a list of gate records
    whose names (``_GATE_TEXT``) are text, whose ids and parent
    (``_GATE_IDS``) are ids and whose boolean operands are a list of ids;
    so are the gates a tab keeps of its own (``view_tabs``). The first
    problem found is returned."""
    for key in _TOP_RECORDS:
        v = data.get(key)
        if v and not isinstance(v, Mapping):
            return f"'{key}' is {_what(v)}, not a record"
    for key in _TOP_LISTS:
        v = data.get(key)
        if v and not isinstance(v, list):
            return f"'{key}' is {_what(v)}, not a list"
    samples = data.get('samples', [])
    if not isinstance(samples, list):
        return f"the sample list is {_what(samples)}, not a list"
    for i, entry in enumerate(samples):
        why = _sample_problem(i, entry)
        if why:
            return why
    return (_gates_problem(data.get('sample_gates'))
            or _tab_gates_problem(data.get('view_tabs')))


def _tab_gates_problem(view_tabs: Any) -> str | None:
    """What is wrong with a gate a tab keeps of its own
    (``view_tabs['tabs'][i]['sample_gates']``), or None. The tab restore
    (``view_tabs.ViewTab.from_dict``) already drops a tab that is not a
    record, a gate list that is not a list and a gate that is not a record;
    the gates it keeps are checked as the session's are."""
    tabs = view_tabs.get('tabs') if isinstance(view_tabs, Mapping) else None
    if not isinstance(tabs, (list, tuple)):
        return None
    for tab in tabs:
        sg = tab.get('sample_gates') if isinstance(tab, Mapping) else None
        if not isinstance(sg, Mapping):
            continue
        for sample, gates in sg.items():
            if not isinstance(gates, (list, tuple)):
                continue
            for j, g in enumerate(gates):
                why = (_gate_problem(sample, j, g)
                       if isinstance(g, Mapping) else None)
                if why:
                    return f"tab '{tab.get('name') or 'Tab'}': {why}"
    return None
