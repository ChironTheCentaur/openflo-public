"""Reading an FCS file whose TEXT lacks a required $PnR, as FlowJo does:
``RangeFillingFlowData``.

Core module: the package's FCS reader and the standard library. No Tk, and it
does not import ``openflo.pipeline``.
"""

from . import fcs_read as _fcs_read


class RangeFillingFlowData(_fcs_read.FlowData):
    """openflo.fcs_read.FlowData for a file whose TEXT lacks a required $PnR: each
    missing range is taken as 2^$PnB (the most an n-bit value can hold), as
    FlowJo does, and a sentence saying so is appended to `filled`."""

    def __init__(self, path, filled, **kw):
        self._openflo_filled = filled
        super().__init__(path, **kw)

    def _FlowData__parse_text(self, *args, **kwargs):   # flowio's private
        text = super()._FlowData__parse_text(*args, **kwargs)  # type: ignore[attr-defined]
        try:
            npar = int(str(text.get('par', '')).strip())
        except ValueError:
            return text
        for i in range(1, npar + 1):
            if f'p{i}r' in text:
                continue
            try:
                bits = int(str(text.get(f'p{i}b', '')).strip())
            except ValueError:
                continue
            if not 0 < bits <= 64:
                continue
            text[f'p{i}r'] = str(2 ** bits)
            name = text.get(f'p{i}n') or f'Ch{i}'
            self._openflo_filled.append(
                f"$P{i}R (range of {name}) is missing: taken as 2^{bits} "
                f"= {2 ** bits:,} from its $P{i}B")
        return text
