"""Display/analysis transforms for fluorescence channels.

The forward and inverse channel transforms (logicle, hyperlog, asinh, log,
linear), the per-channel transform record (``transform_spec``) a FlowSample
keeps in ``data_transforms``, and the helpers that undo it to recover linear
values. Core module: numpy + flowutils, no Tk, and it does not import
``openflo.pipeline``.
"""

import functools
import logging

import numpy as np
from flowutils import transforms

from .scales import (
    ASINH_DEFAULT_COFACTOR,
    LOGICLE_A,
    LOGICLE_M,
    LOGICLE_T,
    LOGICLE_W,
)

# Name kept as 'openflo.pipeline' so existing log filters and caplog tests see this warning.
log = logging.getLogger('openflo.pipeline')


# ── Transforms ─────────────────────────────────────────────────────────────
#
# Display/analysis transforms for fluorescence channels. logicle + log were
# the originals; asinh and hyperlog round out FlowJo parity, plus a 'linear'
# pass-through. asinh is parametrised by the intuitive `cofactor`
# (arcsinh(x / cofactor)); the others share FlowJo's t/m/w/a knobs.
TRANSFORM_METHODS = ('logicle', 'hyperlog', 'asinh', 'log', 'linear')


@functools.lru_cache(maxsize=8)
def _biexp_lut(method, t, m, w, a, n=16384):
    """Monotone (linear-value -> transformed-scale) lookup table for the GPU
    forward logicle / hyperlog transform.

    Built from flowutils' EXACT inverse sampled uniformly in SCALE, so it's dense
    exactly where the transform is steep (the linear region near 0). Interpolating
    against it reproduces flowutils' forward transform to ~1e-8 of scale (see the
    Phase-1 spike), which is far below any gating tolerance. Cached per parameter
    set. The scale span [-1.0, 1.1] maps to data ~[-2.6e6 .. 7.4e5]; a value
    outside it is transformed exactly by transform_values, not clamped."""
    inv = (transforms.logicle_inverse if method == 'logicle'
           else transforms.hyperlog_inverse)
    s = np.linspace(-1.0, 1.1, n)
    d = inv(s.reshape(-1, 1), channel_indices=[0],
            t=t, m=m, w=w, a=a).flatten()
    return d, s


def transform_values(values, method='logicle', t=LOGICLE_T,
                     m=LOGICLE_M, w=LOGICLE_W,
                     a=LOGICLE_A, cofactor=ASINH_DEFAULT_COFACTOR):
    """Transform a 1-D array by `method`. Pure; returns a new array.

    logicle / hyperlog : FlowJo biexponential family (t/m/w/a).
    asinh              : arcsinh(x / cofactor) — cofactor ~150 (fluor),
                         ~5 (mass cytometry).
    log                : log10, clamped at >0 (0 elsewhere).
    linear             : pass-through.

    When GPU acceleration is enabled (Preferences), logicle / hyperlog use a
    flowutils-derived LUT + GPU interp (~1e-8 match); otherwise the exact
    flowutils path runs — so the default (GPU off) is bitwise unchanged.
    """
    v = np.asarray(values, dtype=float)
    if method in ('logicle', 'hyperlog'):
        from . import gpu_accel
        # The biexponential backends map every NON-FINITE input to the BOTTOM
        # of the scale (-1.0) — NaN, +inf and absurd magnitudes alike. That is
        # silent corruption twice over: a maximally bright reading is rendered
        # and gated as maximally DIM, and, worse, a NaN becomes a finite
        # coordinate, so the `dropna` that protects the plot and the gate
        # masks no longer removes it. The event then counts as a real, very
        # negative measurement in every population, median and frequency.
        # asinh already propagates NaN correctly; this makes the family
        # consistent. Finite in-range values are untouched.
        finite = np.isfinite(v)
        vv = np.where(finite, v, 0.0)                 # placeholder, masked out
        fn = (transforms.logicle if method == 'logicle'
              else transforms.hyperlog)
        if gpu_accel.enabled():
            d, s = _biexp_lut(method, t, m, w, a)
            out = np.array(gpu_accel.interp(vv, d, s), dtype=float)
            # The table spans scale [-1, 1.1], data ~[-2.6e6, 7.4e5] at the
            # defaults; interp clamps outside it, so every value above 7.4e5
            # read 1.1 (2e6 is 1.196 exactly, 4e6 1.263) and its inverse came
            # back as 7.4e5. A 22-bit spectral cytometer reaches 4.2e6 and
            # compensation can push past range, so those few events take the
            # exact path instead of a clamped coordinate.
            off = (vv < d[0]) | (vv > d[-1])
            if off.any():
                out[off] = fn(vv[off].reshape(-1, 1), channel_indices=[0],
                              t=t, m=m, w=w, a=a).flatten()
        else:
            out = fn(vv.reshape(-1, 1), channel_indices=[0],
                     t=t, m=m, w=w, a=a).flatten()
        if not finite.all():
            out = np.where(finite, out, np.nan)
        return out
    if method == 'asinh':
        from . import gpu_accel
        return gpu_accel.arcsinh(v, cofactor)
    if method == 'log':
        # A non-positive value has no logarithm. Folding it to 0.0 put it at
        # raw intensity 1.0 — ABOVE every genuinely positive event dimmer than
        # 1.0 — and piled a third of a compensated channel onto that single
        # coordinate. Measured: a gate at log > -1.0 selected 99,995 of
        # 100,000 events where 67,375 was correct, so 32,620 negative events
        # were called positive; the artificial spike is also what
        # auto_threshold and _bimodal_valley lock onto. NaN instead, matching
        # what the logicle/hyperlog branch above already does for input it
        # cannot place, so `dropna` keeps these events out of gates and
        # medians rather than inventing a coordinate for them.
        pos = v > 0
        if not pos.all():
            log.warning(
                "log transform: %d of %d value(s) are non-positive and have "
                "no logarithm (NaN) — use logicle or hyperlog to represent "
                "negative compensated values.", int((~pos).sum()), v.size)
        safe = np.log10(np.clip(np.where(pos, v, 1.0), 1e-6, None))
        return np.where(pos, safe, np.nan)
    if method == 'linear':
        return v
    raise ValueError(f"Unknown transform method '{method}'.")


def inverse_transform_values(values, method='logicle',
                             t=LOGICLE_T, m=LOGICLE_M,
                             w=LOGICLE_W, a=LOGICLE_A,
                             cofactor=ASINH_DEFAULT_COFACTOR):
    """Invert `transform_values` — map a transformed channel back to its
    (compensated) linear scale. Used to re-transform a channel from one
    method to another without re-running compensation. 'log' is not
    perfectly invertible where it clamped to 0; that region maps to ~0."""
    v = np.asarray(values, dtype=float)
    if method == 'logicle':
        return transforms.logicle_inverse(v.reshape(-1, 1), channel_indices=[0],
                                          t=t, m=m, w=w, a=a).flatten()
    if method == 'hyperlog':
        return transforms.hyperlog_inverse(v.reshape(-1, 1),
                                           channel_indices=[0],
                                           t=t, m=m, w=w, a=a).flatten()
    if method == 'asinh':
        return np.sinh(v) * float(cofactor)
    if method == 'log':
        # `v` here is ALREADY log-scaled, and a log-scaled value is negative
        # for every raw intensity below 1.0 — log10(0.5) is -0.301. The old
        # `v > 0` guard confused the transformed value with the raw one and
        # sent all of those to 0.0, so a round trip destroyed every event
        # dimmer than raw 1.0 (measured: 0.01, 0.1, 0.5 and 1.0 all came back
        # as 0.0). That runs whenever a channel's scale is changed, which does
        # inverse-then-forward. 10**v is the inverse on the whole real line;
        # NaN stays NaN.
        return np.power(10.0, v)
    if method == 'linear':
        return v
    raise ValueError(f"Unknown transform method '{method}'.")


def transform_spec(method: str = 'logicle', t: float = LOGICLE_T,
                   m: float = LOGICLE_M, w: float = LOGICLE_W,
                   a: float = LOGICLE_A,
                   cofactor: float = ASINH_DEFAULT_COFACTOR) -> dict:
    """The arguments of one `transform_values` call, as the dict both it and
    `inverse_transform_values` accept. A FlowSample records one per transformed
    channel (``data_transforms``) so the transform baked into its data can be
    undone exactly, parameters included."""
    return {'method': method, 't': t, 'm': m, 'w': w, 'a': a,
            'cofactor': cofactor}


# A column known to hold display-TRANSFORMED values whose transform is not
# known: an OpenFlo CSV written before transform records existed, outside
# the logicle range. Its values are kept exactly as stored (so gates select
# what they did), never re-transformed, and no linear statistic is reported.
UNKNOWN_SPEC = {'method': 'unknown'}


class UnknownScaleError(ValueError):
    """A channel's linear values were asked for, but its display transform is
    unknown (see UNKNOWN_SPEC), so they cannot be recovered."""


def is_unknown_spec(spec):
    """True for a record entry saying the transform is unknown."""
    return bool(spec) and spec.get('method') == 'unknown'


def unknown_spec(origin=None):
    """An UNKNOWN_SPEC naming where the unknown scale was found. Two columns
    of unknown scale are only known to share it when their origins match:
    ``'session:<data folder>'`` for one older session's sidecars (one 2.6.1
    editor, one transform per channel), ``'file:<csv>'`` for anything else.
    The origin rides along through re-saves, so it keeps meaning that."""
    return dict(UNKNOWN_SPEC, **({'origin': str(origin)} if origin else {}))


# The keys of a record entry that ARE transform parameters. An entry may also
# carry 'guessed' (below) or an unknown one's 'origin'; neither may reach a
# transform call.
_SPEC_PARAMS = ('method', 't', 'm', 'w', 'a', 'cofactor')


def transform_params(spec):
    """`spec`'s transform parameters only -- what transform_values /
    inverse_transform_values / FlowSample.retransform take."""
    return {k: spec[k] for k in _SPEC_PARAMS if k in (spec or {})}


def is_guessed_spec(spec):
    """True for a record entry whose transform was GUESSED: a column of an
    OpenFlo CSV with no record whose values all lie in the logicle range.
    Its linear values are a guess too -- right for the default-logicle
    case, ~20x wrong for a dim channel the old editor had on asinh -- so
    nothing may present them as certain."""
    return bool(spec) and spec.get('guessed') is True


def guessed_spec(spec):
    """`spec`, marked as guessed (see is_guessed_spec)."""
    return dict(spec, guessed=True)


def to_linear(values, spec):
    """Undo the transform `spec` (a `transform_spec` dict) on `values`.
    ``None`` or a 'linear' spec means the values are already linear; an
    unknown transform gives NaN (nothing linear can be said about it)."""
    v = np.asarray(values, dtype=float)
    if not spec or spec.get('method', 'linear') == 'linear':
        return v
    if is_unknown_spec(spec):
        return np.full(v.shape, np.nan)
    return inverse_transform_values(v, **transform_params(spec))


def linear_values(sample, channel, values=None):
    """`channel`'s compensated LINEAR values in `sample`.

    The editor's loader transforms each fluor channel IN PLACE (QC ->
    compensate -> logicle), so ``sample.data`` holds display coordinates. A
    tool that models intensity -- a G2 = 2 x G1 DNA fit, a least-squares
    unmix, a straight MESF line, a mean or a CV -- needs the linear values
    those coordinates encode. This undoes the transform the sample recorded
    for the channel (``data_transforms``); a channel with no record, or a
    sample without one, is already linear. `values` replaces the column (e.g.
    a row subset of it).

    Inverting on demand rather than keeping a linear copy: logicle_inverse
    measured 0.04 s per 2M events and round-trips to 2e-9 absolute, while a
    kept copy would double the resident data the loader works to halve."""
    if values is None:
        values = sample.data[channel].values
    spec = (getattr(sample, 'data_transforms', None) or {}).get(channel)
    if is_unknown_spec(spec):
        raise UnknownScaleError(
            f"'{channel}' in '{getattr(sample, 'name', '?')}' holds values on "
            "an unknown display scale (an OpenFlo CSV written before transform "
            "records), so its linear values cannot be recovered.")
    return to_linear(values, spec)
