"""Compensation arithmetic that keeps an unknown value to the outputs it
feeds: ``matmul_known``.

Core module: numpy only. No Tk, and it does not import ``openflo.pipeline``.
"""

import numpy as np


def matmul_known(values, matrix):
    """``values @ matrix`` where a non-finite input spoils only the outputs
    it feeds. A plain product spreads it through zero coefficients too, as
    NaN * 0 is NaN: one unknown value on a channel with no spillover would
    blank every channel of its event."""
    values = np.asarray(values, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    bad = ~np.isfinite(values)
    out = np.where(bad, 0.0, values) @ matrix
    if bad.any():
        out[(bad.astype(float) @ (matrix != 0).astype(float)) > 0] = np.nan
    return out
