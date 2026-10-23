"""Clean-up on a mass cytometer's Gaussian discrimination parameters.

Each event of a mass cytometer is a cloud of ions that arrives over several
pushes; the instrument fits a Gaussian to it and records the fit as
parameters beside the event length: Center, Offset, Width and Residual
(Standard BioTools / Fluidigm Helios and CyTOF XT; Bagwell et al. 2020,
Cytometry A 97:184). Ion clouds that are not one cell -- two cells fused
(doublets), debris, a cloud split across pushes -- fit badly or come out too
long or too short, so they sit in the tails of these parameters.

The clean-up keeps the events within robust bounds on each chosen
parameter: the median plus or minus `n_mad` scaled MADs (1 / Phi^-1(3/4) =
1.4826... x the median absolute deviation, the SD of a normal spread), worked out per
sample from its own events. By default on ``arcsinh(x / 5)``, since these
parameters are skewed to the right like intensities; 'linear' takes them
as they are. A parameter with no spread (MAD 0) cuts nothing.

The editor runs it as a method of an auto-clean recipe gate (key
``'gaussian'``), so it is recomputed for every sample and saved with the
session like the other auto-clean methods. Pure numpy; no Tk.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .robust import MAD_TO_SD

__all__ = ['PARAMETERS', 'DEFAULT_N_MAD', 'find_gaussian_channels',
           'parse_channels', 'robust_bounds', 'gaussian_keep_mask',
           'cleanup_summary', 'default_method']

# The parameter names, as instruments write them, and the normalised form
# (upper case, letters and digits only) they are matched by.
PARAMETERS = ('Event_length', 'Center', 'Offset', 'Width', 'Residual')
_KEYS = {'EVENTLENGTH': 'Event_length', 'CENTER': 'Center',
         'CENTRE': 'Center', 'OFFSET': 'Offset', 'WIDTH': 'Width',
         'RESIDUAL': 'Residual'}
# Scaled MADs either side of the median. 3 is the usual outlier rule (Leys
# et al. 2013); per parameter about 0.3 % of a normal spread lies beyond it.
DEFAULT_N_MAD = 3.0
SCALES = ('asinh', 'linear')
_MAD_SD = MAD_TO_SD
_COFACTOR = 5.0
_MIN_EVENTS = 20


def _norm(name: object) -> str:
    return re.sub(r'[^A-Z0-9]', '', str(name).upper())


def find_gaussian_channels(columns: Iterable[object]) -> list[str]:
    """The Gaussian parameters and event length among `columns`, in the
    instrument's order (Event_length, Center, Offset, Width, Residual)."""
    found: dict[str, str] = {}
    for c in columns:
        key = _KEYS.get(_norm(c))
        if key and key not in found:
            found[key] = str(c)
    return [found[k] for k in PARAMETERS if k in found]


def parse_channels(text: object, columns: Iterable[object]) -> list[str]:
    """The channels a comma-separated `text` names (blank: every Gaussian
    parameter found), kept only when in `columns`."""
    cols = [str(c) for c in columns]
    raw = str(text or '').strip()
    if not raw:
        return find_gaussian_channels(cols)
    want = [p.strip() for p in raw.split(',') if p.strip()]
    return [c for c in want if c in cols]


def _scaled(values: np.ndarray, scale: str) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if scale == 'linear':
        return v
    if scale != 'asinh':
        raise ValueError(f"scale must be one of {SCALES}, got {scale!r}")
    return np.arcsinh(v / _COFACTOR)


def robust_bounds(values: np.ndarray, n_mad: float = DEFAULT_N_MAD,
                  scale: str = 'asinh') -> tuple[float, float]:
    """``(low, high)`` kept on one parameter, in its own (untransformed)
    units: median +/- `n_mad` scaled MADs on `scale`. ``(-inf, inf)`` when
    there are too few finite values or no spread."""
    z = _scaled(values, scale)
    z = z[np.isfinite(z)]
    if z.size < _MIN_EVENTS or not n_mad > 0:
        return -np.inf, np.inf
    med = float(np.median(z))
    sd = _MAD_SD * float(np.median(np.abs(z - med)))
    if not sd > 0:
        return -np.inf, np.inf
    lo, hi = med - n_mad * sd, med + n_mad * sd
    if scale == 'asinh':
        return float(_COFACTOR * np.sinh(lo)), float(_COFACTOR * np.sinh(hi))
    return float(lo), float(hi)


def _params(params: Mapping[str, Any] | None) -> tuple[str, float, str]:
    p = params or {}
    try:
        n = float(p.get('n_mad', DEFAULT_N_MAD))
    except (TypeError, ValueError):
        n = DEFAULT_N_MAD
    # A scale typed into the auto-clean dialog is free text: anything but
    # 'linear' falls back to the default rather than failing the gate.
    scale = str(p.get('scale') or 'asinh').strip().lower()
    return (str(p.get('channels') or ''), n,
            scale if scale in SCALES else 'asinh')


def cleanup_summary(df, params: Mapping[str, Any] | None = None
                    ) -> list[dict[str, Any]]:
    """One row per parameter used: ``{'channel', 'low', 'high', 'removed'}``
    (events outside that parameter's bounds, on its own)."""
    text, n_mad, scale = _params(params)
    rows = []
    for c in parse_channels(text, df.columns):
        v = np.asarray(df[c].values, dtype=float)
        lo, hi = robust_bounds(v, n_mad, scale)
        out = np.isfinite(v) & ((v < lo) | (v > hi))
        rows.append({'channel': c, 'low': lo, 'high': hi,
                     'removed': int(out.sum())})
    return rows


def gaussian_keep_mask(df, params: Mapping[str, Any] | None = None
                       ) -> np.ndarray:
    """Keep-mask: the events within every chosen parameter's bounds. An
    event whose value is not finite is kept (nothing can be said about it);
    without any Gaussian parameter nothing is removed."""
    text, n_mad, scale = _params(params)
    keep = np.ones(len(df), dtype=bool)
    for c in parse_channels(text, df.columns):
        v = np.asarray(df[c].values, dtype=float)
        lo, hi = robust_bounds(v, n_mad, scale)
        keep &= ~(np.isfinite(v) & ((v < lo) | (v > hi)))
    return keep


def default_method(channels: Iterable[str] = (), n_mad: float = DEFAULT_N_MAD,
                   scale: str = 'asinh') -> dict[str, Any]:
    """The auto-clean method dict for this clean-up (blank channels: every
    Gaussian parameter the sample has)."""
    return {'key': 'gaussian', 'label': 'Gaussian parameters (mass cytometry)',
            'enabled': True,
            'params': {'channels': ','.join(channels), 'n_mad': float(n_mad),
                       'scale': scale}}
