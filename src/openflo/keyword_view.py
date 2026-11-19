"""What an FCS file says about itself, for a viewer (the phone's File info
card; the desktop's Inspect FCS shows the same keywords).

``keyword_view(pairs)`` turns a file's TEXT keywords, in file order, into a
short summary (events, date, cytometer, software...), one row per channel
($PnN, $PnS, $PnV, $PnG, $PnR, $PnE) and every keyword, as written.

Core module: no Tk.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

# The summary rows, in order: label and keyword ($-less, upper case). A
# keyword the file lacks is left out.
SUMMARY = (
    ('Events', 'TOT'), ('Parameters', 'PAR'), ('Date', 'DATE'),
    ('Started', 'BTIM'), ('Ended', 'ETIM'), ('Cytometer', 'CYT'),
    ('Serial number', 'CYTSN'), ('Software', 'CREATOR'), ('Operator', 'OP'),
    ('Institution', 'INST'), ('Specimen source', 'SRC'),
    ('Original file', 'FIL'),
)

# A channel row's fields and the $Pn keyword each comes from.
CHANNEL_FIELDS = (('name', 'N'), ('label', 'S'), ('voltage', 'V'),
                  ('gain', 'G'), ('range', 'R'), ('amplification', 'E'))

_PN = re.compile(r'^P(\d+)([A-Z]+)$')


def _norm(key: str) -> str:
    return str(key).strip().lstrip('$').upper()


def _order(key: str) -> tuple[Any, ...]:
    """Sort key: by name, with the numbers in $PnX keywords taken as
    numbers ($P2N before $P10N)."""
    m = _PN.match(_norm(key))
    if m:
        return ('P', int(m.group(1)), m.group(2))
    return (_norm(key), 0, '')


def keyword_view(pairs: Iterable[tuple[str, str]]) -> dict[str, Any]:
    """``{'summary': [[label, value]], 'channels': [{'number', 'name',
    'label', 'voltage', 'gain', 'range', 'amplification'}], 'keywords':
    [[key, value]]}`` from a file's TEXT keywords `pairs`.

    Keys keep their spelling (``$CYT``, or ``cyt`` when only FlowIO's
    lower-cased, $-less copy is at hand). Values are stripped; a keyword
    given twice (in any case) keeps its last value, as the FCS loader
    (FlowIO) does, so the card shows what the analysis read. The keyword
    list is sorted by name, $PnX by channel number; a channel is listed for
    every n up to $PAR (or the highest $PnN) with '' for what the file
    does not say."""
    kept: dict[str, tuple[str, str]] = {}
    for key, text in pairs:
        norm = _norm(key)
        if norm:
            kept[norm] = (str(key).strip(), str(text).strip())

    def value(norm: str) -> str:
        return kept.get(norm, ('', ''))[1]

    summary = [[label, value(k)] for label, k in SUMMARY if value(k)]
    numbers = [int(m.group(1)) for k in kept if (m := _PN.match(k)) and m.group(2) == 'N']
    try:
        count = int(value('PAR'))
    except ValueError:
        count = 0
    count = max([count, *numbers]) if numbers or count > 0 else 0
    channels = [{'number': n, **{field: value(f'P{n}{suffix}') for field, suffix in CHANNEL_FIELDS}}
                for n in range(1, count + 1)]
    keywords = [list(kv) for kv in sorted(kept.values(), key=lambda kv: _order(kv[0]))]
    return {'summary': summary, 'channels': channels, 'keywords': keywords}
