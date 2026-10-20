"""Which samples were acquired together, from the FCS keywords.

A spillover matrix holds for the samples run on the same instrument with
the same detector voltages, usually in one session. Folder names and export
times don't say that: files exported together can come from different days
and settings. The cytometer writes the facts into each file's TEXT segment
when it acquires the sample, and exporting it later doesn't change them:

  * ``$CYTSN`` (serial number), else ``$CYT`` (model): the instrument;
  * ``$PnV``: each detector's voltage, compared on the matrix's channels;
  * ``$DATE``: the acquisition date;
  * ``$SPILLOVER`` / ``SPILL``: the matrix the instrument stored, which
    changes when its compensation setup does.

Two samples are in one batch when every one of these that both files carry
agrees, and they share the instrument, the date and at least one voltage on
the matrix's channels. A file missing those (a CSV, a file written by a
converter that drops the keywords) can't be placed in a batch, so it is
left out rather than guessed into one.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _text(metadata) -> dict:
    """TEXT keywords with '$' stripped and lower-cased (FlowIO's spelling;
    other readers keep the '$')."""
    return {str(k).lstrip('$').strip().lower(): str(v).strip()
            for k, v in (metadata or {}).items() if v is not None}


def _number(value):
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Acquisition:
    """What the FCS file says about how it was acquired; None when absent."""
    instrument: str | None = None
    date: str | None = None
    voltages: dict = field(default_factory=dict)    # $PnN -> volts
    spillover: str | None = None

    def known(self, channels) -> bool:
        """Enough to place the file in a batch for a matrix on ``channels``."""
        return bool(self.instrument and self.date
                    and any(c in self.voltages for c in channels))


def acquisition(metadata) -> Acquisition:
    """Read ``metadata`` (an FCS TEXT dict) into an :class:`Acquisition`."""
    kw = _text(metadata)
    serial, model = kw.get('cytsn'), kw.get('cyt')
    instrument = (f'SN {serial}' if serial else model) or None
    volts = {}
    for key, name in kw.items():
        # pNn -> its voltage pNv
        if (len(key) > 2 and key[0] == 'p' and key[-1] == 'n'
                and key[1:-1].isdigit()):
            v = _number(kw.get(f'p{key[1:-1]}v'))
            if v is not None and name:
                volts[name] = v
    spill = kw.get('spill') or kw.get('spillover') or None
    if spill:
        spill = ','.join(p.strip() for p in spill.split(','))
    return Acquisition(instrument=instrument,
                       date=(kw.get('date') or '').upper() or None,
                       voltages=volts, spillover=spill)


def mismatch(a: Acquisition, b: Acquisition, channels) -> str | None:
    """Why ``b`` was not acquired like ``a`` for a matrix on ``channels``,
    or None when it was."""
    if not b.known(channels):
        return 'no acquisition keywords'
    if a.instrument != b.instrument:
        return f'another instrument ({b.instrument})'
    if a.date != b.date:
        return f'acquired on {b.date}'
    shared = [c for c in channels if c in a.voltages and c in b.voltages]
    if not shared:
        return 'no voltages on these channels'
    differ = [c for c in shared if a.voltages[c] != b.voltages[c]]
    if differ:
        return f'other voltages ({", ".join(differ[:3])})'
    if a.spillover and b.spillover and a.spillover != b.spillover:
        return 'another stored spillover matrix'
    return None


def batch_of(name, metadata_by_sample: dict, channels) -> tuple[list, dict]:
    """The samples acquired like ``name`` for a matrix on ``channels``.

    ``metadata_by_sample`` maps each loaded sample to its FCS TEXT dict.
    Returns ``(members, left_out)``: members in the given order, ``name``
    first; ``left_out`` maps every other sample to why it isn't one. When
    ``name`` itself carries too little to compare, the batch is just it.
    """
    channels = list(channels)
    me = acquisition(metadata_by_sample.get(name))
    if not me.known(channels):
        return [name], {n: f"'{name}' has no acquisition keywords"
                        for n in metadata_by_sample if n != name}
    members, left_out = [name], {}
    for other, meta in metadata_by_sample.items():
        if other == name:
            continue
        why = mismatch(me, acquisition(meta), channels)
        if why is None:
            members.append(other)
        else:
            left_out[other] = why
    return members, left_out
