"""Reading FCS files through FlowIO, without its slow integer bit-masking.

Integer FCS files often store each value in more bits than the instrument
uses: 10-bit data in 16-bit words, with ``$PnR`` = 1024. The standard says the
unused high bits must be ignored, so FlowIO masks every value. It does that
one byte at a time in a Python ``map``/``lambda``, which made such a file load
about 60x slower than one that needs no mask (2M events x 20 channels: 9.5 s
against 0.16 s).

:class:`FlowData` is FlowIO's own reader with only that step replaced. FlowIO
reads the words exactly as before, with the mask switched off, and the same
masks (``next power of two of $PnR, minus 1``) are then applied with numpy, in
place on the same ``array.array``. Everything else, offsets, TEXT parsing,
byte order and every error, is FlowIO's.

FlowIO is pinned (``FlowIO==1.4.0`` in pyproject.toml); the hook overrides one
of its private methods, so ``tests/test_fcs_read.py`` checks that the hook is
still called and that every value matches plain ``flowio.FlowData``.

BD Accuri C6 data ($CYT naming Accuri, which every well of a .c6 file and
the C6 software's FCS export carry) is the one exception. Its area channels
record values above $PnR (16,777,216): in the public C6 files checked
(Zenodo records 4972042 and 15351644) up to 4.5 times the range, in up to
6 % of a well's FSC-A events. Masked as the standard says, such a value
wraps round to a small one (18,042,191 reads as 1,264,975) and lands among
the debris. For these files a value above ``$PnR - 1`` is pinned at
``$PnR - 1``, the top of the scale, before the mask, and the count per
channel is kept in ``over_range_pinned`` (the loader's note says it).
Pinning at the range is what flowCore's ``read.FCS(truncate_max_range =
TRUE)`` does with a value above $PnR, but flowCore applies the bit mask
first (R/IO.R, readFCSdata), so it, FlowIO and FlowCal all wrap these
values; FlowJo's Accuri page documents the 16.7M range and not how it reads
values past it. Every other file keeps the standard's mask.
"""
from __future__ import annotations

from typing import Any

import flowio
import numpy as np

__all__ = ['FlowData']

_UINT = {8: np.uint8, 16: np.uint16, 32: np.uint32}


def is_accuri(text) -> bool:
    """Whether FCS TEXT `text` (FlowIO's keyword dict) names a BD Accuri
    instrument in $CYT."""
    return 'accuri' in str(text.get('cyt', '')).lower()


class FlowData(flowio.FlowData):
    """``flowio.FlowData`` with numpy bit-masking of integer data, and BD
    Accuri values above the range pinned at its top (see the module)."""

    # {channel name: values pinned at the top of its range}; empty unless a
    # BD Accuri file held values above $PnR.
    over_range_pinned: dict[str, int] = {}

    def _FlowData__parse_int_data(self, offset, start, stop, bit_width_lut,
                                  max_range_lut, order):
        parse: Any = super()._FlowData__parse_int_data  # type: ignore[attr-defined]
        widths = set(bit_width_lut.values())
        if len(widths) != 1:
            return parse(offset, start, stop, bit_width_lut, max_range_lut,
                         order)
        width = widths.pop()
        top = 2 ** width
        needs_mask = any(top > max_range_lut[c] for c in bit_width_lut)
        # A $PnR beyond the word size makes FlowIO raise when it builds its
        # mask; leave that case (and the no-mask case) entirely to FlowIO.
        if (width not in _UINT or not needs_mask
                or any(mr > top for mr in max_range_lut.values())):
            return parse(offset, start, stop, bit_width_lut, max_range_lut,
                         order)

        # Read with masking off: FlowIO masks only when some 2**width exceeds
        # its range, so give every channel the full word as its range.
        data = parse(offset, start, stop, bit_width_lut,
                     dict.fromkeys(max_range_lut, top), order)

        masks = np.array([mr - 1 for mr in max_range_lut.values()],
                         dtype=_UINT[width])
        n_par = len(masks)
        n_events = len(data) // n_par
        # FlowIO's mask covers whole events only and its byte-wise AND stops
        # at the shorter input, dropping any trailing partial event.
        if len(data) != n_events * n_par:
            del data[n_events * n_par:]
        values = np.frombuffer(data, dtype=_UINT[width])
        if is_accuri(self.text):
            self.over_range_pinned = _pin_over_range(
                values.reshape(n_events, n_par), self.text,
                list(max_range_lut))
        values.reshape(n_events, n_par)[...] &= masks
        del values  # release the buffer so the array can be resized later
        return data


def _pin_over_range(values, text, channels) -> dict[str, int]:
    """Set each value of `values` (events x channels, unsigned, written in
    place) above its channel's ``$PnR - 1`` to ``$PnR - 1``; returns
    ``{channel name: values pinned}`` for the channels that had any.
    `channels` are the 1-based channel numbers of the columns."""
    pinned = {}
    for col, ch in enumerate(channels):
        try:
            top = int(float(str(text.get(f'p{ch}r', '')).strip())) - 1
        except ValueError:
            continue
        if top < 0:
            continue
        over = values[:, col] > top
        n = int(np.count_nonzero(over))
        if n:
            values[over, col] = top
            pinned[str(text.get(f'p{ch}n') or f'Ch{ch}')] = n
    return pinned
