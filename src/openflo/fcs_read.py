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
"""
from __future__ import annotations

from typing import Any

import flowio
import numpy as np

__all__ = ['FlowData']

_UINT = {8: np.uint8, 16: np.uint16, 32: np.uint32}


class FlowData(flowio.FlowData):
    """``flowio.FlowData`` with numpy bit-masking of integer data."""

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
                     {c: top for c in max_range_lut}, order)

        masks = np.array([mr - 1 for mr in max_range_lut.values()],
                         dtype=_UINT[width])
        n_par = len(masks)
        n_events = len(data) // n_par
        # FlowIO's mask covers whole events only and its byte-wise AND stops
        # at the shorter input, dropping any trailing partial event.
        if len(data) != n_events * n_par:
            del data[n_events * n_par:]
        values = np.frombuffer(data, dtype=_UINT[width])
        values.reshape(n_events, n_par)[...] &= masks
        del values  # release the buffer so the array can be resized later
        return data
