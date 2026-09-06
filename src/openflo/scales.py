"""Axis display-scale view transforms — Tk-free, testable.

Fluor data is stored baked into a nonlinear transform (e.g. logicle). The
underlying *linear intensity* is the canonical master; the chosen display scale
(linear / log / symlog) is a pure VIEW of that intensity, composed as::

    forward(d) = view_forward(inverse_baked(d))
    inverse(p) = forward_baked(view_inverse(p))

so every scale is an independent, equation-derived view of the same intensity
(no double-transform) and gates kept in stored-data coords auto-follow the axis.
This is the maths behind the matplotlib ``FuncScale`` the editor installs.
"""
from __future__ import annotations

import numpy as np


def view_funcs(transform: str, scale: str, data_sample=None):
    """``(forward, inverse)`` callables mapping a channel's STORED data
    coordinate to screen position for ``scale`` — or ``None`` when the channel
    is stored linearly (``transform == 'linear'``), in which case the caller
    uses matplotlib's native scale (nicer tick locators).

    ``symlog`` uses an arcsinh view whose cofactor is anchored on the data's 5th
    percentile of ``|nonzero intensity|`` when ``data_sample`` is given.
    """
    from .pipeline import inverse_transform_values, transform_values
    if transform == 'linear':
        return None

    def inv_baked(a):
        return inverse_transform_values(np.asarray(a, dtype=float),
                                        method=transform)

    def fwd_baked(a):
        return transform_values(np.asarray(a, dtype=float), method=transform)

    if scale == 'log':
        def forward(d):  # pyright: ignore[reportRedeclaration]
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.log10(np.clip(inv_baked(d), 1e-6, None))

        def inverse(p):
            return fwd_baked(np.power(10.0, np.asarray(p, dtype=float)))
    elif scale == 'symlog':
        cof = 150.0
        if data_sample is not None:
            lin = inv_baked(np.asarray(data_sample, dtype=float))
            lin = lin[np.isfinite(lin)]
            nz = np.abs(lin[lin != 0])
            if nz.size > 50:
                cof = max(float(np.percentile(nz, 5)), 1e-3)

        def forward(d):  # pyright: ignore[reportRedeclaration]
            return np.arcsinh(inv_baked(d) / cof)

        def inverse(p):
            return fwd_baked(np.sinh(np.asarray(p, dtype=float)) * cof)
    else:  # 'linear' view of nonlinear-baked data → stretch to intensity
        def forward(d):
            return inv_baked(d)

        def inverse(p):
            return fwd_baked(p)

    def _finite(fn):
        # matplotlib's FuncScale needs shape-preserving callables, but FlowKit's
        # (inverse_)logicle flattens to 1-D — reshape back and scrub NaNs.
        def wrapped(a):
            arr = np.asarray(a, dtype=float)
            out = np.nan_to_num(np.asarray(fn(arr), dtype=float), nan=0.0)
            return out.reshape(arr.shape)
        return wrapped

    return _finite(forward), _finite(inverse)
