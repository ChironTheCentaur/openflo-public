"""Pure-numpy logicle and hyperlog: a drop-in for the four
``flowutils.transforms`` functions OpenFlo calls, for platforms where
FlowUtils' compiled extension cannot be built (a browser runtime such as
Pyodide). Desktop installs keep using FlowUtils itself; the importers fall
back to this module only when the ``flowutils`` package is not installed at
all (a broken install still fails; see ``openflo.flowutils_import``).

``logicle``, ``logicle_inverse``, ``hyperlog`` and ``hyperlog_inverse`` take
the same arguments, defaults and array semantics as FlowUtils 1.2.2. The
arithmetic is a vectorised port of FlowUtils' ``logicle.c`` (the Parks /
Moore / Moore logicle reference implementation, as used for Gating-ML 2.0
logicle and hyperlog), kept in the same evaluation order: the scale
parameters and Taylor coefficients are computed with the same scalar libm
calls, the inverses are the closed forms, and the forward transforms run the
same Halley iteration from the same initial guess with the same tolerance and
iteration cap, including the reference's quirks (a non-converging input,
which is any non-finite one, maps to -1.0). The only difference is numpy's
vector exp/log against the C library's, an ulp at most per call. On realistic
scales that leaves results within 1-2 ulp of FlowUtils' (300 random scales
with M 2-8, T 1e3-1e7 and 90,000 values each over [-10T, 100T]: largest
difference 4.4e-16 of scale, no input converging differently). At
pathological settings (M <= 1, W near M/2, values far above T) the
reference's iteration is on a knife edge and that 1-ulp difference can
change which inputs fail to converge (-1.0): for example 62 of 26,000
hyperlog inputs at T=262144, M=0.5, W=0.1. There, too, converged values
can differ by an ulp of a larger number (1.4e-14 of scale for logicle at
M=1 in the extended range; 6.5e-15 relative for logicle_inverse at
M=0.1).

Core module: numpy and the standard library. No Tk.

Ported from FlowUtils (https://github.com/whitews/FlowUtils), BSD-licensed;
this is the notice from FlowUtils' LICENSE file:

    Copyright (c) 2013-14, Duke University
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:
        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in
          the documentation and/or other materials provided with the
          distribution.
        * Neither the name of the copyright holder nor the names of its
          contributors may be used to endorse or promote products derived
          from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL DUKE UNIVERSITY BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
    OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
    ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import annotations

import functools
import math
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .scales import LOGICLE_A, LOGICLE_M, LOGICLE_T, LOGICLE_W

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

_EPS = sys.float_info.epsilon
# 16 terms are enough for full double precision on typical scales.
_TAYLOR_LENGTH = 16
# The reference's iteration caps: the bracketed root solve for d, the
# logicle Halley iteration and the hyperlog Halley iteration.
_SOLVE_ITERATIONS = 40
_LOGICLE_ITERATIONS = 40
_HYPERLOG_ITERATIONS = 10
# What the reference returns for an input whose iteration does not converge
# (every non-finite input among them).
_NOT_CONVERGED = -1.0


@dataclass(frozen=True)
class _Params:
    """The derived constants of one (T, W, M, A) scale."""

    a: float
    b: float
    c: float
    d: float
    f: float
    w: float
    x1: float
    x_taylor: float
    taylor: tuple[float, ...]
    hyperlog_x0_value: float = 0.0  # hyperlog's initial-guess break point


def _exp(x: float) -> np.float64:
    """C's exp on a scalar: the C library's value, inf on overflow."""
    try:
        return np.float64(math.exp(x))
    except OverflowError:
        return np.float64(math.inf)


def _log(x: float) -> np.float64:
    """C's log on a scalar: -inf at zero, NaN below it, instead of raising."""
    if x > 0:
        return np.float64(math.log(x))
    return np.float64(-math.inf if x == 0 else math.nan)


def _solve(b: float, w: float) -> float:
    """d such that 2 (ln d - ln b) + w (b + d) = 0: RTSAFE (Newton with a
    bisection fallback) on the bracket [0, b], as the reference does."""
    if w == 0:                      # w == 0 means it is really arcsinh
        return b
    tolerance = 2 * b * _EPS
    d_lo = 0.0
    d_hi = b
    d = (d_lo + d_hi) / 2
    last_delta = d_hi - d_lo
    f_b = -2 * _log(b) + w * b
    f = 2 * _log(d) + w * d + f_b
    last_f = math.nan
    for _ in range(1, _SOLVE_ITERATIONS):
        df = 2 / d + w
        if (((d - d_hi) * df - f) * ((d - d_lo) * df - f) >= 0
                or abs(1.9 * f) > abs(last_delta * df)):
            delta = (d_hi - d_lo) / 2
            d = d_lo + delta
            if d == d_lo:
                return d
        else:
            delta = f / df
            t = d
            d -= delta
            if d == t:
                return d
        if abs(delta) < tolerance:
            return d
        last_delta = delta
        f = 2 * _log(d) + w * d + f_b
        if f == 0 or f == last_f:
            return d
        last_f = f
        if f < 0:
            d_lo = d
        else:
            d_hi = d
    return _NOT_CONVERGED


def _ieee(fn: Callable[..., _Params]) -> Callable[..., _Params]:
    """Run a parameter setup in C double arithmetic: the scale parameters
    as numpy float64 scalars with floating-point errors ignored, so a
    degenerate scale (W = 0 for hyperlog, say) gives the inf / NaN the
    reference computes instead of raising."""
    @functools.wraps(fn)
    def wrapper(t: float, w: float, m: float, a: float) -> _Params:
        with np.errstate(all='ignore'):
            return fn(np.float64(t), np.float64(w), np.float64(m),
                      np.float64(a))
    return wrapper


@_ieee
def _logicle_params(t: float, w: float, m: float, a: float) -> _Params:
    """The logicle constants, formulas from the biexponential paper."""
    w_ = w / (m + a)
    x2 = a / (m + a)
    x1 = x2 + w_
    x0 = x2 + 2 * w_
    b = (m + a) * math.log(10.)
    d = _solve(b, w_)
    c_a = _exp(x0 * (b + d))
    mf_a = _exp(b * x1) - c_a / _exp(d * x1)
    a_ = t / ((_exp(b) - mf_a) - c_a / _exp(d))
    c = c_a * a_
    f = -mf_a * a_
    pos_coef = a_ * _exp(b * x1)
    neg_coef = -c / _exp(d * x1)
    taylor = []
    for i in range(_TAYLOR_LENGTH):
        pos_coef *= b / (i + 1)
        neg_coef *= -d / (i + 1)
        taylor.append(pos_coef + neg_coef)
    taylor[1] = 0.0                 # exact result of the logicle condition
    return _Params(a=a_, b=b, c=c, d=d, f=f, w=w_, x1=x1,
                   x_taylor=x1 + w_ / 4, taylor=tuple(taylor))


@_ieee
def _hyperlog_params(t: float, w: float, m: float, a: float) -> _Params:
    """The hyperlog constants (the reference's ``hyperlog_scale`` setup)."""
    w_ = w / (m + a)
    x2 = a / (m + a)
    x1 = x2 + w_
    x0 = x2 + 2 * w_
    b = (m + a) * math.log(10)
    e0 = _exp(b * x0)
    c_a = e0 / w_
    f_a = _exp(b * x1) + c_a * x1
    a_ = t / (_exp(b) + c_a - f_a)
    c = c_a * a_
    f = f_a * a_
    coef = a_ * _exp(b * x1)
    taylor = []
    for i in range(_TAYLOR_LENGTH):
        coef *= b / (i + 1)
        taylor.append(coef)
    taylor[0] += c
    p = _Params(a=a_, b=b, c=c, d=0.0, f=f, w=w_, x1=x1,
                x_taylor=x1 + w_ / 4, taylor=tuple(taylor))
    # The initial-guess break point. The reference evaluates it with the
    # logicle series (which skips taylor[1]) and without subtracting f;
    # both are kept, since the result only seeds the iteration.
    negative = x0 < x1
    tmp_x0 = 2 * x1 - x0 if negative else x0
    if tmp_x0 < p.x_taylor:
        inverse = float(_series_biexponential(p, np.array([tmp_x0]))[0])
    else:
        inverse = a_ * _exp(b * tmp_x0) + c * tmp_x0
    if negative:
        inverse = -inverse
    return _Params(a=a_, b=b, c=c, d=0.0, f=f, w=w_, x1=x1,
                   x_taylor=p.x_taylor, taylor=p.taylor,
                   hyperlog_x0_value=inverse)


def _series_biexponential(p: _Params, scale: np.ndarray) -> np.ndarray:
    """The logicle Taylor series about x1, skipping taylor[1] (zero by the
    logicle condition)."""
    x = scale - p.x1
    s = p.taylor[_TAYLOR_LENGTH - 1] * x
    for i in range(_TAYLOR_LENGTH - 2, 1, -1):
        s = (s + p.taylor[i]) * x
    return (s * x + p.taylor[0]) * x


def _taylor_series(p: _Params, scale: np.ndarray) -> np.ndarray:
    """The full Taylor series about x1 (hyperlog)."""
    x = scale - p.x1
    s = p.taylor[_TAYLOR_LENGTH - 1] * x
    for i in range(_TAYLOR_LENGTH - 2, -1, -1):
        s = (s + p.taylor[i]) * x
    return s


def _halley(p: _Params, value: np.ndarray, x: np.ndarray,
            tolerance: np.ndarray, iterations: int,
            step: Callable[[_Params, np.ndarray, np.ndarray], np.ndarray],
            ) -> tuple[np.ndarray, np.ndarray]:
    """Run `step` (one Halley step per element, returning delta) until each
    element's |delta| < its tolerance, at most `iterations` times. Each
    element stops at the iteration it converges, as the scalar reference
    does. Returns the roots and which elements converged."""
    out = np.array(x)
    converged = np.zeros(value.shape, dtype=bool)
    idx = np.arange(value.size)
    for _ in range(iterations):
        if idx.size == 0:
            break
        xs = x[idx]
        delta = step(p, value[idx], xs)
        xs = xs - delta
        x[idx] = xs
        done = np.abs(delta) < tolerance[idx]
        out[idx[done]] = xs[done]
        converged[idx[done]] = True
        idx = idx[~done]
    return out, converged


def _logicle_step(p: _Params, value: np.ndarray, x: np.ndarray) -> np.ndarray:
    ae2bx = p.a * np.exp(p.b * x)
    ce2mdx = p.c / np.exp(p.d * x)
    # Near zero use the Taylor series; elsewhere this formulation has the
    # better round-off behaviour.
    y = np.where(x < p.x_taylor, _series_biexponential(p, x) - value,
                 (ae2bx + p.f) - (ce2mdx + value))
    abe2bx = p.b * ae2bx
    cde2mdx = p.d * ce2mdx
    dy = abe2bx + cde2mdx
    ddy = p.b * abe2bx - p.d * cde2mdx
    return y / (dy * (1 - y * ddy / (2 * dy * dy)))


def _hyperlog_step(p: _Params, value: np.ndarray, x: np.ndarray) -> np.ndarray:
    ae2bx = p.a * np.exp(p.b * x)
    y = np.where(x < p.x_taylor, _taylor_series(p, x) - value,
                 (ae2bx + p.c * x) - (p.f + value))
    abe2bx = p.b * ae2bx
    dy = abe2bx + p.c
    ddy = p.b * abe2bx
    return y / (dy * (1 - y * ddy / (2 * dy * dy)))


def _reflect_and_solve(
        p: _Params, values: np.ndarray,
        guess_fn: Callable[[_Params, np.ndarray], np.ndarray],
        tolerance_fn: Callable[[np.ndarray], np.ndarray], iterations: int,
        step: Callable[[_Params, np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """The shared forward driver: zero maps to x1, negative values are
    solved as their reflection about x1, the rest by Halley iteration."""
    v = np.asarray(values, dtype=float)
    negative = v < 0
    mag = np.where(negative, -v, v)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        x = guess_fn(p, mag)
        tolerance = tolerance_fn(x)
        x, converged = _halley(p, mag, x, tolerance, iterations, step)
    out = np.where(negative, 2 * p.x1 - x, x)
    out = np.where(converged, out, _NOT_CONVERGED)
    return np.where(v == 0, p.x1, out)


def _logicle_guess(p: _Params, value: np.ndarray) -> np.ndarray:
    # Linear approximation in the quasi-linear region, else ordinary log.
    return np.where(value < p.f, p.x1 + value / p.taylor[0],
                    np.log(value / p.a) / p.b)


def _logicle_tolerance(x: np.ndarray) -> np.ndarray:
    # Double precision, unless in the extended range.
    return np.where(x > 1, 3 * x * _EPS, 3 * _EPS)


def _hyperlog_guess(p: _Params, value: np.ndarray) -> np.ndarray:
    return np.where(value < p.hyperlog_x0_value,
                    p.x1 + value * p.w / p.hyperlog_x0_value,
                    np.log(value / p.a) / p.b)


def _hyperlog_tolerance(x: np.ndarray) -> np.ndarray:
    return np.full(x.shape, 3 * _EPS)


def _logicle_1d(y: Iterable[float], t: float, m: float, w: float,
                a: float) -> np.ndarray:
    p = _logicle_params(t, w, m, a)
    return _reflect_and_solve(p, np.array(y, dtype='double'), _logicle_guess,
                              _logicle_tolerance, _LOGICLE_ITERATIONS,
                              _logicle_step)


def _logicle_inverse_1d(y: Iterable[float], t: float, m: float, w: float,
                        a: float) -> np.ndarray:
    p = _logicle_params(t, w, m, a)
    v = np.array(y, dtype='double')
    negative = v < p.x1                 # reflect negative scale regions
    s = np.where(negative, 2 * p.x1 - v, v)
    with np.errstate(over='ignore', invalid='ignore'):
        inverse = np.where(s < p.x_taylor, _series_biexponential(p, s),
                           (p.a * np.exp(p.b * s) + p.f) - p.c / np.exp(p.d * s))
    return np.where(negative, -inverse, inverse)


def _hyperlog_1d(y: Iterable[float], t: float, m: float, w: float,
                 a: float) -> np.ndarray:
    p = _hyperlog_params(t, w, m, a)
    return _reflect_and_solve(p, np.array(y, dtype='double'), _hyperlog_guess,
                              _hyperlog_tolerance, _HYPERLOG_ITERATIONS,
                              _hyperlog_step)


def _hyperlog_inverse_1d(y: Iterable[float], t: float, m: float, w: float,
                         a: float) -> np.ndarray:
    p = _hyperlog_params(t, w, m, a)
    v = np.array(y, dtype='double')
    negative = v < p.x1
    s = np.where(negative, 2 * p.x1 - v, v)
    with np.errstate(over='ignore', invalid='ignore'):
        inverse = np.where(s < p.x_taylor, _taylor_series(p, s),
                           (p.a * np.exp(p.b * s) + p.c * s) - p.f)
    return np.where(negative, -inverse, inverse)


def _per_channel(fn: Callable[..., np.ndarray], data: np.ndarray,
                 channel_indices: Iterable[int] | None, t: float, m: float,
                 w: float, a: float) -> np.ndarray:
    """FlowUtils' wrapper: a 1-D array is transformed whole (as float64);
    a 2-D one channel by channel into a copy of its own dtype, every
    channel when `channel_indices` is None."""
    data_copy = data.copy()
    if len(data.shape) == 1:
        return fn(data_copy, t, m, w, a)
    if channel_indices is None:
        channel_indices = range(data.shape[1])
    for i in channel_indices:
        data_copy.T[i] = fn(data_copy[:, i].T, t, m, w, a)
    return data_copy


def logicle(data: np.ndarray, channel_indices: Iterable[int] | None,
            t: float = LOGICLE_T, m: float = LOGICLE_M,
            w: float = LOGICLE_W, a: float = LOGICLE_A) -> np.ndarray:
    """Logicle transform (Gating-ML 2.0): ``root(B(y, T, W, M, A) - x)``
    with ``B = a e^(by) - c e^(-dy) - f``. Same arguments and result as
    ``flowutils.transforms.logicle``."""
    return _per_channel(_logicle_1d, data, channel_indices, t, m, w, a)


def logicle_inverse(data: np.ndarray, channel_indices: Iterable[int] | None,
                    t: float = LOGICLE_T, m: float = LOGICLE_M,
                    w: float = LOGICLE_W, a: float = LOGICLE_A) -> np.ndarray:
    """Inverse logicle, the closed-form biexponential. Same arguments and
    result as ``flowutils.transforms.logicle_inverse``."""
    return _per_channel(_logicle_inverse_1d, data, channel_indices, t, m, w, a)


def hyperlog(data: np.ndarray, channel_indices: Iterable[int] | None,
             t: float = LOGICLE_T, m: float = LOGICLE_M,
             w: float = LOGICLE_W, a: float = LOGICLE_A) -> np.ndarray:
    """Hyperlog transform (Gating-ML 2.0): ``root(EH(y, T, W, M, A) - x)``
    with ``EH = a e^(by) + cy - f``. Same arguments and result as
    ``flowutils.transforms.hyperlog``."""
    return _per_channel(_hyperlog_1d, data, channel_indices, t, m, w, a)


def hyperlog_inverse(data: np.ndarray, channel_indices: Iterable[int] | None,
                     t: float = LOGICLE_T, m: float = LOGICLE_M,
                     w: float = LOGICLE_W, a: float = LOGICLE_A) -> np.ndarray:
    """Inverse hyperlog, the closed form. Same arguments and result as
    ``flowutils.transforms.hyperlog_inverse``."""
    return _per_channel(_hyperlog_inverse_1d, data, channel_indices,
                        t, m, w, a)
