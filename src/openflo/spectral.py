"""Spectral unmixing for full-spectrum cytometers (Cytek Aurora, BD S8…).

Conventional compensation subtracts spillover between a few detectors;
spectral cytometers instead record each event's full emission spectrum
across many detectors and *unmix* it into per-fluorophore abundances using
reference spectra measured from single-stain controls (plus an
autofluorescence spectrum from an unstained control).

Two layers, both pure (numpy; scipy's KD-tree for matched autofluorescence):

  * ``build_reference_spectra`` — turn single-stain (and unstained) control
    arrays into a reference spectra matrix S (n_fluors × n_detectors), with
    per-control diagnostics (how each spectrum was estimated, which controls
    are too dim to trust).
  * ``unmix`` — solve, per event, the least-squares abundances A such that
    A · S ≈ Y (the raw detector matrix). OLS by default (what Cytek/SpectroFlo
    use); optional non-negativity.

``apply_unmixing`` wires it onto a FlowSample, adding one abundance column
per fluor.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np

log = logging.getLogger(__name__)


def _normalize(spec, mode='max'):
    spec = np.asarray(spec, dtype=float)
    if mode == 'l2':
        n = np.linalg.norm(spec)
        return spec / n if n > 0 else spec
    if mode == 'max':
        mx = spec.max()
        return spec / mx if mx > 0 else spec
    return spec


def _cosine(a, b):
    """Cosine between two spectra; NaN when either is all zero (undefined,
    not 0 -- see :func:`spectral_similarity_matrix`)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if not (na > 0 and nb > 0):
        return float('nan')
    return float(np.clip(a @ b / (na * nb), -1.0, 1.0))


# ── Reference spectrum of one single-stain control ──────────────────────────
#
# 'total' (the original estimate, unchanged): the mean of the brightest
# `bright_pct` events by TOTAL signal, minus the unstained control's MEAN
# autofluorescence (AF). On cells whose AF varies (lymphocytes beside
# myeloid cells with ~10x their AF), a dim dye loses that selection to the
# high-AF cells, and subtracting the AVERAGE AF leaves most of theirs in the
# dye's spectrum. Measured on the 16-detector validation model (dye 400 at
# its peak, 30% of cells at 10x AF): cosine to the true spectrum 0.55-0.58
# for a dye far from the AF band, 0.84 for one 0.6-similar to it, and an
# unmixing error 12x the floor (0.35 against 0.046); on the 8-detector
# audit case 0.856.
#
# 'matched': select by the dye's OWN signal and subtract each event's OWN AF.
#   1. Primary detector: where the stained control's mean stands furthest
#      above the unstained one's, in units of the unstained's spread.
#   2. Matching detectors: where the dye is negligible against AF. Each
#      stained event is matched, by k nearest neighbours in those detectors
#      (standardised; at most 4 principal components), to the unstained
#      events with the same AF there; their mean is that event's AF in
#      every detector. A detector qualifies when the dye's excess over the
#      unstained per unit of AF is at most _LEAK_TAU of the primary's AND
#      the dye moves a positive event there by at most _LEAK_SI of the AF
#      spread, each allowed _LEAK_Z standard errors (a detector is shut out
#      only when the dye is shown to be there). Both limits are measured:
#      - by separation alone, a detector whose wide AF band hid a dye at 25%
#        of its peak (separation < 1) was matched on; each positive moved to
#        a brighter-AF neighbour and AF was over-subtracted everywhere
#        (cosine 0.9926 against the total estimate's 0.9949 on a dye 0.9-
#        similar to AF; 0.9997 with the ratio limit);
#      - by the ratio alone, a dye of 5,000 at 5% of its peak in a detector
#        still moved its positives past every unstained event there, and
#        the dye's own tail came out 20% low (a panel built that way
#        unmixed with error 0.0137 against the total estimate's 0.0087);
#      - without the error allowance, sampling noise on 3,000 events (10%
#        of cells at 20x AF) shut out every detector, and the total
#        estimate's 0.939 stood.
#   3. Positives: the brightest events by EXCESS over that predicted AF in
#      the primary, above what the unstained shows against its own
#      neighbours (99.5th percentile), up to the same `bright_pct` share.
#   4. Spectrum: their mean excess over their matched AF, in every detector.
#
# 'auto' computes both and keeps 'total' -- bit for bit -- unless they
# disagree (cosine < _AGREE_COS) or the dye barely clears the AF band
# (separation < _LOW_SEPARATION); see _choose_spectrum for the guard. Where
# AF cannot bias the total selection by 2% of the spectrum (bright dyes,
# homogeneous AF, beads) the matched estimate is not even computed.

SPECTRA_METHODS = ('auto', 'total', 'matched')

_AGREE_COS = 0.99          # 'auto' keeps the 'total' spectrum at or above
# ... unless the primary separation index is below this. On the validation
# grid, every control that agreed at >= 0.99 with separation < 20 unmixed
# better from the matched spectrum -- error 0.12-0.77x the total-signal
# one's (2,000 at 20x AF: 0.130 -> 0.025, floor 0.022) -- while above 40
# the two were within +-3% either way.
_LOW_SEPARATION = 20.0
_LEAK_TAU = 0.05           # dye-to-AF ratio of a matching detector vs primary
_LEAK_SI = 1.0             # and the dye's shift there / AF spread
_LEAK_Z = 2.0              # standard errors either may exceed its limit by
_MATCH_K = 16              # unstained neighbours averaged per event
_MATCH_MAX_EVENTS = 20_000  # per control, seeded subsample (speed)
_NULL_EVENTS = 5_000       # unstained events matched to each other
_MATCH_MAX_DIMS = 4        # principal components of the matching detectors
_MIN_EVENTS = 200          # fewer and the matched estimate is not attempted
_MIN_POSITIVES = 20
_NULL_Q = 0.995            # positives exceed this share of the unstained
_HOMOGENEOUS_AF = 0.02     # AF bias bound / spectrum below which 'auto' skips
_SI_QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.98)
_DIM_SEPARATION = 3.0      # primary separation below this -> 'too dim'
_DIM_SPLIT_COS = 0.98      # split-half reproducibility below this -> 'too dim'


def _total_spectrum(arr, auto, bright_pct):
    """The original estimate: mean of the brightest `bright_pct` events by
    total signal, minus the mean AF `auto`, clipped at 0 (not normalised)."""
    total = arr.sum(axis=1)
    thr = np.percentile(total, bright_pct)
    bright = arr[total >= thr] if np.any(total >= thr) else arr
    spec = bright.mean(axis=0)
    if auto is not None:
        spec = spec - auto
    return np.clip(spec, 0.0, None)


def _subsample(arr, n_max, rng):
    if len(arr) <= n_max:
        return arr
    return arr[np.sort(rng.choice(len(arr), n_max, replace=False))]


def _separation_index(stain, unstained):
    """``(index, spread)`` per detector: how far the stained control's upper
    quantiles (median to 98th percentile; the largest shift wins, so a
    control with few positives still registers) sit above the unstained
    control's, in units of the unstained's spread ((p98 - p2) / 4.1, an SD
    for a normal population, and wide where AF varies -- so a dye under a
    variable AF band scores low however bright the band is)."""
    qs = np.asarray(_SI_QUANTILES)
    qx = np.quantile(stain, qs, axis=0)
    qu = np.quantile(unstained, qs, axis=0)
    lo, hi = np.quantile(unstained, [0.02, 0.98], axis=0)
    spread = (hi - lo) / 4.1
    floor = max(float(np.max(spread)) * 1e-3, 1e-12)
    spread = np.maximum(spread, floor)
    return ((qx - qu) / spread).max(axis=0), spread


def _primary_detector(stain, unstained):
    """``(p, separation_index, spread, dye)``: the primary detector `p` is
    where the dye's MEAN excess over the unstained (`dye`, per detector) is
    largest in units of the unstained's spread. Means, not the quantiles of
    the separation index: where AF is bimodal (10% of cells at 20x) a
    quantile near the boundary between the two populations jumps with their
    sampling ratio -- a dye-free detector scored 1.0 on 3,000 events."""
    si, spread = _separation_index(stain, unstained)
    dye = stain.mean(axis=0) - unstained.mean(axis=0)
    return int(np.argmax(dye / spread)), si, spread, dye


def _matched_spectrum(X, U, bright_pct, rng):
    """The 'matched' estimate for one control (see the block comment above).
    Returns ``(spectrum or None, info)``; ``info['reason']`` says why when
    there is no spectrum. `X`, `U`: finite ``(n, d)`` stained / unstained."""
    from scipy.spatial import KDTree

    X = _subsample(X, _MATCH_MAX_EVENTS, rng)
    U = _subsample(U, _MATCH_MAX_EVENTS, rng)
    p, si, spread, dye = _primary_detector(X, U)
    mu_u = U.mean(axis=0)
    se = np.sqrt(X.var(axis=0) / len(X) + U.var(axis=0) / len(U))
    shift = dye / spread
    info = {'primary_index': p, 'separation_index': float(si[p]),
            'n_positives': None, 'matching_detectors': []}
    if len(X) < _MIN_EVENTS or len(U) < _MIN_EVENTS:
        info['reason'] = (f'fewer than {_MIN_EVENTS} events in the stained '
                          'or unstained control')
        return None, info
    if not dye[p] > 0:
        info['reason'] = ('the stained control is no brighter than the '
                          'unstained one in any detector')
        return None, info

    q02, q50 = np.quantile(U, [0.02, 0.5], axis=0)
    # The unstained's spread below its median (noise plus the low-AF
    # population's own width) floors the ratio's denominator: a detector
    # with no AF would otherwise divide by ~0.
    low = np.maximum((q50 - q02) / 2.05, max(float(np.max(spread)) * 1e-3,
                                             1e-12))
    af_level = np.maximum(mu_u, low)
    rho_p = dye[p] / af_level[p]
    # Dye-to-AF ratio vs the primary's, and how far the dye moves a positive
    # event in that detector in units of the AF spread (the primary's
    # separation index, scaled by the detector's share of the mean shift).
    # Each may exceed its limit by its own sampling error: a detector is
    # shut out only when the dye is shown to be there.
    rel = dye / af_level / rho_p
    rel_se = se / af_level / rho_p
    disp = shift / shift[p] * max(si[p], 0.0)
    disp_se = se / spread / shift[p] * max(si[p], 0.0)
    use = ((rel - _LEAK_Z * rel_se <= _LEAK_TAU)
           & (disp - _LEAK_Z * disp_se <= _LEAK_SI))
    use[p] = False
    M = np.flatnonzero(use)
    info['matching_detectors'] = [int(d) for d in M]
    if not M.size:
        info['reason'] = ('no detector where autofluorescence dominates the '
                          'dye to match on')
        return None, info

    med = np.median(U[:, M], axis=0)
    zU = (U[:, M] - med) / spread[M]
    zX = (X[:, M] - med) / spread[M]
    if M.size > _MATCH_MAX_DIMS:
        # AF varies along a few directions; the leading components carry it
        # and keep the tree query fast on 30-60 detector instruments.
        c = zU.mean(axis=0)
        _, _, vt = np.linalg.svd(zU - c, full_matrices=False)
        V = vt[:_MATCH_MAX_DIMS].T
        zU, zX = (zU - c) @ V, (zX - c) @ V
    k = min(_MATCH_K, len(U) - 1)
    tree = KDTree(zU)
    idx = np.asarray(tree.query(zX, k=k, workers=-1)[1], dtype=np.intp)
    excess = X[:, p] - U[idx, p].mean(axis=1)
    # The same excess for unstained events against their OTHER neighbours:
    # what AF alone produces, so what a positive must clear.
    nul = np.sort(rng.choice(len(U), min(len(U), _NULL_EVENTS),
                             replace=False))
    idn = np.asarray(tree.query(zU[nul], k=k + 1, workers=-1)[1],
                     dtype=np.intp)
    null = U[nul, p] - U[idn[:, 1:], p].mean(axis=1)
    thr = float(np.quantile(null, _NULL_Q))
    n_top = max(int(np.ceil(len(X) * (1.0 - bright_pct / 100.0))), 1)
    order = np.argsort(excess)[::-1][:n_top]
    pos = order[excess[order] > thr]
    info['n_positives'] = int(pos.size)
    if pos.size < _MIN_POSITIVES:
        info['reason'] = (f'only {pos.size} event(s) stand above their '
                          'matched autofluorescence')
        return None, info
    resid = X[pos] - U[idx[pos]].mean(axis=1)
    half = rng.permutation(pos.size)
    info['split_half_cosine'] = _cosine(
        resid[half[::2]].mean(axis=0), resid[half[1::2]].mean(axis=0))
    return np.clip(resid.mean(axis=0), 0.0, None), info


def _af_bias_bound(un, auto, bright_pct):
    """Norm of the most AF the 'total' selection can leave in a spectrum:
    the unstained control's own brightest `bright_pct` events by total,
    minus its mean. Selecting on dye + AF picks less extreme AF than
    selecting on AF alone. (`un` has finite rows, so the percentile is at
    most the largest total and `top` is never empty.)"""
    total = un.sum(axis=1)
    top = un[total >= np.percentile(total, bright_pct)]
    return float(np.linalg.norm(top.mean(axis=0) - auto))


def _choose_spectrum(arr, un, auto, bright_pct, method, bead, rng):
    """``(spectrum, diagnostics)`` for one control under `method`."""
    old = _total_spectrum(arr, auto, bright_pct)
    diag = {'method': 'total', 'reason': '', 'agreement': None,
            'primary_index': int(np.argmax(old)) if old.size else None,
            'separation_index': None, 'n_positives': None,
            'warning': None}
    usable_un = (un is not None and un.ndim == 2 and arr.ndim == 2
                 and un.shape[1] == arr.shape[1] and len(un) > 1)
    if method == 'total':
        diag['reason'] = "method='total' requested"
    elif bead:
        diag['reason'] = 'bead control: no cellular autofluorescence to match'
    elif not usable_un:
        diag['reason'] = 'no unstained control to match autofluorescence on'
    if usable_un and (method == 'total' or bead):
        p, si, _, _ = _primary_detector(_subsample(arr, _MATCH_MAX_EVENTS, rng),
                                        _subsample(un, _MATCH_MAX_EVENTS, rng))
        diag.update(primary_index=p, separation_index=float(si[p]))
    elif usable_un:
        bound = _af_bias_bound(un, auto, bright_pct)
        norm_old = float(np.linalg.norm(old))
        if (method == 'auto' and norm_old > 0
                and bound <= _HOMOGENEOUS_AF * norm_old):
            # The total selection can leave at most 2% of this spectrum's
            # size in AF (cosine >= 0.9998 whatever it picked), so there is
            # nothing for matching to fix: skip it (and its cost).
            p, si, _, _ = _primary_detector(
                _subsample(arr, _MATCH_MAX_EVENTS, rng),
                _subsample(un, _MATCH_MAX_EVENTS, rng))
            diag.update(primary_index=p, separation_index=float(si[p]),
                        reason=('autofluorescence varies too little to bias '
                                'this control (at most '
                                f'{100 * bound / norm_old:.1f}% of its '
                                'spectrum)'))
        else:
            new, info = _matched_spectrum(arr, un, bright_pct, rng)
            diag.update({k: v for k, v in info.items() if k != 'reason'})
            if new is None:
                diag['reason'] = ('matched estimate unavailable ('
                                  + info['reason'] + '); kept the total-'
                                  'signal estimate')
                if method == 'matched':
                    diag['warning'] = diag['reason']
            else:
                agree = _cosine(old, new)
                diag['agreement'] = agree
                sep = info['separation_index']
                if method == 'matched':
                    diag.update(method='matched',
                                reason="method='matched' requested")
                    old = new
                elif not (np.isfinite(agree) and agree >= _AGREE_COS):
                    diag.update(method='matched', reason=(
                        'total-signal estimate disagrees with the matched-'
                        f'autofluorescence one (cosine {agree:.4f}): its '
                        'brightest events are AF-rich, not dye-rich'))
                    old = new
                elif sep < _LOW_SEPARATION:
                    # Agreeing to 0.99 is not agreeing for the unmix when
                    # the dye sits inside the AF band: see _LOW_SEPARATION.
                    diag.update(method='matched', reason=(
                        f'estimates agree (cosine {agree:.4f}) but the dye '
                        f'stands only {sep:.1f} AF spreads clear '
                        f'(< {_LOW_SEPARATION:g}), where the AF the total-'
                        'signal estimate keeps still biases the unmix'))
                    old = new
                else:
                    diag['reason'] = (f'total and matched estimates agree '
                                      f'(cosine {agree:.4f}); kept total')
    return old, diag


def _dim_warning(diag, spec):
    """Why this control is too dim to define a spectrum reliably, or None."""
    if not np.any(spec > 0):
        return ('no signal above the unstained control in any detector: '
                'the spectrum is empty')
    split = diag.get('split_half_cosine')
    if split is not None:
        # The matched estimate ran: judge by whether its positives agree
        # with each other, not by the raw separation, which a variable AF
        # band drags below 1 even where matching recovers the spectrum
        # (cosine 0.999 at separation 0.36).
        if np.isfinite(split) and split < _DIM_SPLIT_COS:
            return (f'the spectrum is not reproducible (two halves of its '
                    f'positives agree to cosine {split:.3f})')
        return None
    si = diag.get('separation_index')
    if si is not None and np.isfinite(si) and si < _DIM_SEPARATION:
        return (f'primary-detector separation index {si:.2f} < '
                f'{_DIM_SEPARATION:g}: the dye barely stands out from the '
                'unstained control')
    return None


def build_reference_spectra(single_stains, unstained=None, bright_pct=90.0,
                            normalize='max', method='auto',
                            beads: bool | str | Iterable[str] = False,
                            detectors=None, diagnostics=None,
                            seed=0) -> tuple[np.ndarray, list[str]]:
    """Build the reference spectra matrix from control arrays.

    `single_stains` : ``{fluor_name: ndarray (n_events, n_detectors)}`` —
                      each a single-stain control over the SAME detectors.
    `unstained`     : optional ndarray (m, n_detectors); its mean (over its
                      finite events) is the autofluorescence spectrum,
                      subtracted from every single-stain spectrum and added
                      as its own endmember.
    `bright_pct`    : use only the brightest events of each single stain to
                      estimate its spectrum (avoids the negative population
                      dragging the mean) -- by total signal ('total'), or by
                      excess over each event's matched AF ('matched').
    `method`        : ``'total'`` -- brightest events by total signal minus
                      the MEAN autofluorescence (the original estimate);
                      ``'matched'`` -- brightest by the dye's own excess,
                      each minus the AF of the unstained events matched to
                      it (see the block comment above); ``'auto'``
                      (default) -- 'total', bit for bit, unless the two
                      disagree (cosine < 0.99: the total selection picked
                      AF) or the dye's separation index is below 20 (it
                      sits inside the AF band), when 'matched' is used.
    `beads`         : True, or the names of the controls that are beads:
                      those always use 'total' (no cellular AF to match).
    `detectors`     : optional detector names, to label diagnostics.
    `diagnostics`   : optional list; one dict per single stain is appended
                      (``fluor, method, reason, primary_detector,
                      separation_index, n_positives, agreement, af_cosine,
                      warning`` ...). Changes no numbers.
    `seed`          : seeds the matched estimate's subsampling.

    Returns ``(S, fluors)`` where ``S`` is ``(n_fluors, n_detectors)`` (rows
    L-normalised per `normalize`) and `fluors` is the row label list (with
    ``'Autofluorescence'`` last when `unstained` is given)."""
    if method not in SPECTRA_METHODS:
        raise ValueError(f"method must be one of {SPECTRA_METHODS}, "
                         f"not {method!r}")
    if beads is True:
        bead_names = set(single_stains)
    elif not beads:
        bead_names = set()
    elif isinstance(beads, str):
        bead_names = {beads}
    else:
        bead_names = set(beads)
    auto = None
    un = None
    if unstained is not None and len(unstained):
        un = np.asarray(unstained, dtype=float)
        if un.ndim == 2:
            # As for the single stains below: one inf or NaN event made the
            # mean -- the autofluorescence row and every spectrum it is
            # subtracted from -- non-finite, and unmix then refused the
            # whole reference set.
            un = un[np.all(np.isfinite(un), axis=1)]
        if un.size:
            auto = np.mean(un, axis=0)
        else:
            un = None
    names = list(detectors) if detectors is not None else None

    fluors, rows = [], []
    for name, arr in single_stains.items():
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 2:
            arr = arr[np.all(np.isfinite(arr), axis=1)]   # drop non-finite events
        if arr.size == 0:
            continue
        # One generator per control: its result does not depend on which
        # other controls came before it.
        spec, diag = _choose_spectrum(arr, un, auto, bright_pct, method,
                                      name in bead_names,
                                      np.random.default_rng(seed))
        fluors.append(name)
        rows.append(_normalize(spec, normalize))
        diag['warning'] = diag.get('warning') or _dim_warning(diag, spec)
        if diag['warning']:
            log.warning("reference spectrum %r: %s -- consider beads or a "
                        "brighter fluor.", name, diag['warning'])
        if diagnostics is not None:
            pi = diag.get('primary_index')
            diag.update(
                fluor=name, n_events=int(len(arr)), requested_method=method,
                primary_detector=(names[pi] if names is not None and pi
                                  is not None and pi < len(names) else pi),
                af_cosine=(_cosine(spec, auto) if auto is not None
                           and np.shape(auto) == np.shape(spec) else None))
            diagnostics.append(diag)
    if auto is not None:
        fluors.append('Autofluorescence')
        rows.append(_normalize(np.clip(auto, 0.0, None), normalize))
    return np.asarray(rows, dtype=float), fluors


def unmix(Y, S, nonneg=False):
    """Unmix raw detector signals into fluorophore abundances.

    `Y` : ``(n_events, n_detectors)`` raw signal.
    `S` : ``(n_fluors, n_detectors)`` reference spectra.
    Solves ``A · S ≈ Y`` for ``A`` (n_events, n_fluors) by ordinary least
    squares (vectorised). With `nonneg`, negatives are clipped to 0 (a fast
    approximation to NNLS, adequate once spectra are clean)."""
    Y = np.asarray(Y, dtype=float)
    S = np.asarray(S, dtype=float)
    if not np.isfinite(S).all():
        raise ValueError(
            "unmix: reference spectra contain non-finite values — "
            "check the single-stain controls (numpy would otherwise fail with "
            "an opaque 'SVD did not converge').")

    # Solve per EVENT-BLOCK, excluding non-finite rows.
    #
    # lstsq solves every event in one factorisation, so a single non-finite
    # reading contaminates the whole solution: one `inf` event turned all
    # events' abundances into NaN (a NaN event happened to stay contained, an
    # inf did not). Losing an entire sample's unmixing to one saturated
    # reading is silent data corruption, so bad events are held out and
    # returned as NaN — the same containment a NaN already got by accident.
    # 2.2.3 hardened the REFERENCE spectra against non-finite values; this is
    # the matching guard on the event data.
    good = np.isfinite(Y).all(axis=1)
    if good.all():
        # Solve S.T (det × fluor) @ A.T (fluor × events) = Y.T (det × events).
        A = np.linalg.lstsq(S.T, Y.T, rcond=None)[0].T
    else:
        A = np.full((Y.shape[0], S.shape[0]), np.nan, dtype=float)
        if good.any():
            A[good] = np.linalg.lstsq(S.T, Y[good].T, rcond=None)[0].T
        n_bad = int((~good).sum())
        log.warning("unmix: %d of %d event(s) carry non-finite detector "
                    "values and were left unmixed (NaN); the remaining "
                    "events are unaffected.", n_bad, Y.shape[0])
    if nonneg:
        A = np.clip(A, 0.0, None)
    return A


def measured_signal(sample, detectors) -> np.ndarray:
    """``(n_events, len(detectors))``: each detector's signal in `sample` as
    the instrument MEASURED it -- linear and uncompensated, the scale
    reference spectra are built on and unmixing solves on.

    A sample loaded in the editor holds compensated, logicle-transformed
    values (QC -> compensate -> logicle). ``linear_values`` undoes the
    transform; this also undoes the compensation the sample records
    (``compensated @ comp_matrix`` is what was measured). The editor
    unmixed the compensated values. Where compensation keeps every control's
    spectrum non-negative that changes little (the linear map commutes with
    least squares), but where it pushes a spectrum below zero,
    build_reference_spectra clips it: with a $SPILL of 0.5 from D1 into D6
    on the synthetic 8-detector panel, the median abundance error was
    36-54% against 7-9% on the measured values. Raises
    ``pipeline.UnknownScaleError`` when a channel it needs has no
    recoverable linear values."""
    from .pipeline import linear_values
    detectors = list(detectors)
    if not detectors:
        return np.empty((len(sample.data), 0))
    cols = {d: np.asarray(linear_values(sample, d), dtype=float)
            for d in detectors}
    cm = getattr(sample, 'comp_matrix', None)
    cc = list(getattr(sample, 'comp_channels', None) or [])
    hit = [d for d in detectors if d in cc]
    if cm is not None and hit:
        comp = np.column_stack([
            cols[c] if c in cols else np.asarray(linear_values(sample, c),
                                                 dtype=float)
            for c in cc])
        measured = comp @ np.asarray(cm, dtype=float)
        for d in hit:
            cols[d] = measured[:, cc.index(d)]
    return np.column_stack([cols[d] for d in detectors])


def apply_unmixing(sample, S, fluors, detectors, nonneg=False, prefix='U:'):
    """Unmix `sample` in place: read the `detectors` columns as measured
    (:func:`measured_signal`: linear, uncompensated), unmix against spectra
    `S` (labelled `fluors`), and add one ``f'{prefix}{fluor}'`` abundance
    column per fluor, on the linear scale. Returns the list of new column
    names."""
    cols = [d for d in detectors if d in sample.data.columns]
    if len(cols) != S.shape[1]:
        raise ValueError(
            f"spectra have {S.shape[1]} detectors but {len(cols)} of the "
            f"requested detectors are present in the sample")
    Y = measured_signal(sample, cols)
    A = unmix(Y, S, nonneg=nonneg)
    new_cols = []
    for j, f in enumerate(fluors):
        name = f'{prefix}{f}'
        sample.data[name] = A[:, j]
        new_cols.append(name)
    # The columns are linear now; a transform recorded for an earlier unmix
    # under the same names no longer describes them (rebound, not mutated).
    rec = getattr(sample, 'data_transforms', None)
    if rec and any(c in rec for c in new_cols):
        sample.data_transforms = {k: v for k, v in rec.items()
                                  if k not in new_cols}
    return new_cols


# ── Unmixing quality control ──────────────────────────────────────────────────
#
# Two complementary diagnostics for how trustworthy an unmix will be:
#   • the spectral SIMILARITY matrix — purely a function of the reference
#     spectra; flags fluorophore pairs whose signatures are nearly collinear
#     (hard to resolve, the unmix amplifies their noise);
#   • the Spillover Spread Matrix (SSM, Nguyen 2013 / Cytek) — measured from
#     the single-stain controls; quantifies the spreading error each stain
#     introduces into every other fluor's unmixed channel.

def spectral_similarity_matrix(S):
    """Cosine-similarity matrix between reference-spectra rows.

    ``M[i, j] = <Sᵢ, Sⱼ> / (|Sᵢ|·|Sⱼ|)`` — in ``[0, 1]`` for the non-negative
    spectra ``build_reference_spectra`` produces. Diagonal is 1. A high
    off-diagonal value (≳ 0.98) means the two fluorophores are spectrally
    almost indistinguishable, so unmixing them is ill-conditioned and their
    abundances will be noisy/anti-correlated. Returns an ``(n, n)`` ndarray."""
    S = np.asarray(S, dtype=float)
    norm = np.linalg.norm(S, axis=1)
    # Cosine similarity against a zero vector is UNDEFINED, not zero. Dividing
    # by 1.0 instead made a degenerate spectrum come back as 0.0 against every
    # other fluor — which on this scale reads as "maximally distinct, trivially
    # unmixable", the most reassuring answer available, for a fluor that cannot
    # be unmixed at all. A single stain dimmer than the unstained control in
    # every detector clips to all-zero in build_reference_spectra, so this is
    # reachable from ordinary data. NaN says what is true.
    usable = norm > 0
    safe = np.where(usable, norm, 1.0)
    U = S / safe[:, None]
    M = np.clip(U @ U.T, -1.0, 1.0)
    M[~usable, :] = np.nan
    M[:, ~usable] = np.nan
    diag = np.arange(len(norm))
    M[diag[usable], diag[usable]] = 1.0
    return M


def spectral_condition_number(S):
    """2-norm condition number of the spectra matrix ``S`` (n_fluors ×
    n_detectors): ``σ_max / σ_min`` of its singular values. A large value
    (≳ 100) means the unmixing system is ill-posed — small detector noise is
    amplified into large abundance errors. ``inf`` for a rank-deficient S."""
    S = np.asarray(S, dtype=float)
    # More fluors than detectors → the unmix is underdetermined (abundances not
    # uniquely identifiable). svd returns only min(shape) singular values, which
    # can all be nonzero, so cond would look finite/small; report inf instead.
    if S.ndim != 2 or S.shape[0] > S.shape[1]:
        return float('inf')
    sv = np.linalg.svd(S, compute_uv=False)
    if sv.size == 0 or sv.size < min(S.shape):
        return float('inf')
    smax = float(sv.max())
    # Treat singular values below the standard numerical-rank tolerance as
    # zero, so a (near-)rank-deficient matrix reports inf rather than a
    # meaningless 1e16 from a residual float.
    tol = smax * max(S.shape) * np.finfo(float).eps
    smin = float(sv.min())
    if smax <= 0 or smin <= tol:
        return float('inf')
    return smax / smin


def _split_negative_positive(values, cofactor=150.0, bins=128):
    """``(negative, positive, separated)`` for the primary-channel `values`
    of a single-stain control (finite, linear scale).

    The split is Otsu's on ``asinh(values / cofactor)``, which is log-like
    above the cofactor so a bright population a few % of the events still
    separates. `separated` says the two sides are distinct populations: the
    smoothed histogram dips between the highest point on each side to at most
    half the lower of the two. Otsu always returns a split, so without this a
    control with no negative events (or a dim stain inside its negatives) was
    cut through the middle of its one population.

    Shared by the compensation optimizer (pipeline.optimize_compensation)
    and the spillover spread matrix below."""
    v = np.asarray(values, dtype=float)
    none = np.zeros(v.shape, dtype=bool)
    if v.size < 4:
        return none, none, False
    z = np.arcsinh(v / float(cofactor))
    lo, hi = np.percentile(z, [0.5, 99.5])
    if not hi > lo:
        return none, none, False
    hist, edges = np.histogram(z, bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    p = hist / max(hist.sum(), 1)
    omega, mu = np.cumsum(p), np.cumsum(p * centers)
    with np.errstate(divide='ignore', invalid='ignore'):
        between = (mu[-1] * omega - mu) ** 2 / (omega * (1.0 - omega))
    between[~np.isfinite(between)] = -1.0
    # Across an empty gap the between-class variance is flat, and argmax took
    # its first bin, at the edge of the lower population: the smoothed foot of
    # an 80%-negative peak then stood taller than the positive peak and the
    # control read as "not separated". Cut in the middle of the plateau.
    lo_k = int(np.argmax(between))
    hi_k = lo_k
    top = between[lo_k] * (1.0 - 1e-9)
    while hi_k + 1 < between.size and between[hi_k + 1] >= top:
        hi_k += 1
    k = (lo_k + hi_k) // 2 + 1               # first bin on the positive side
    cut = edges[k]
    neg, pos = z < cut, z >= cut
    if not (neg.sum() >= 2 and pos.sum() >= 2 and 0 < k < bins):
        return neg, pos, False
    kernel = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    sm = np.convolve(hist.astype(float), kernel / kernel.sum(), mode='same')
    a = int(np.argmax(sm[:k]))
    b = k + int(np.argmax(sm[k:]))
    valley = float(sm[a:b + 1].min())
    return neg, pos, valley <= 0.5 * float(min(sm[a], sm[b]))


def _robust_sd(v):
    """The robust SD of Nguyen et al. 2013: 84.13th minus 50th percentile."""
    p84, p50 = np.percentile(v, [84.13, 50.0])
    return float(p84 - p50)


def spillover_spread_matrix(single_stains, S, fluors, nonneg=False,
                            n_bins=8, min_bin=30, unstained=None):
    """Spillover Spread Matrix (Nguyen et al. 2013; Cytek SpectroFlo).

    For each single-stain control ``j`` (an ``(events, detectors)`` array in
    ``single_stains[fluor_j]``), unmix it against ``S`` and measure the
    spreading error it puts into every other fluor ``i``:

        SSᵢⱼ = sqrt(σ²pos(Aᵢ) − σ²neg(Aᵢ)) / sqrt(Fpos(Aⱼ) − Fneg(Aⱼ))

    over the control's positive and negative populations on its primary
    abundance ``Aⱼ`` (σ the robust SD, 84th − 50th percentile; F the median).
    The negative population's spread is the instrument's baseline, not the
    stain's, and is subtracted. A control with no separate negative uses the
    `unstained` control (``(events, detectors)``) as its negative; with
    neither, the spread is fitted as σ² = a + SS²·F over ``n_bins`` quantile
    bins of the primary (bins of fewer than ``min_bin`` events skipped).

    The abundances are always UNCONSTRAINED: clipping them at 0 removes half
    the spread being measured (measured: 0.58x the unconstrained SSM).
    `nonneg` is accepted for compatibility and ignored.

    Previously the spread was the median over quantile bins of the primary of
    SD(Aᵢ)/sqrt(median Aⱼ) over events with Aⱼ > 0 -- no baseline subtracted,
    so the noise of the control's negative events, divided by the root of
    their near-zero primary, grew the SSM with the negative fraction: 0.39,
    0.39, 1.65, 2.26 at 0%, 50%, 80%, 90% negative for the same stain.

    Returns ``(SSM, fluors)`` where ``SSM`` is ``(n_fluors, n_fluors)`` in the
    row-order of ``fluors`` (``SSM[i, j]`` = spread INTO ``i`` FROM stain
    ``j``; diagonal 0) -- the transpose of the usual display, which puts the
    stain on the rows. A stain absent from ``single_stains`` (or with too few
    events) leaves its column ``NaN``."""
    S = np.asarray(S, dtype=float)
    nf = len(fluors)
    SSM = np.full((nf, nf), np.nan, dtype=float)
    idx = {f: i for i, f in enumerate(fluors)}
    A_un = None
    if unstained is not None:
        U = np.asarray(unstained, dtype=float)
        if U.ndim == 2 and U.shape[1] == S.shape[1] and len(U) >= min_bin:
            A_un = unmix(U, S)
            A_un = A_un[np.isfinite(A_un).all(axis=1)]
            if len(A_un) < min_bin:
                A_un = None
    for jname, arr in single_stains.items():
        if jname not in idx:
            continue
        j = idx[jname]
        Y = np.asarray(arr, dtype=float)
        if Y.ndim != 2 or Y.shape[1] != S.shape[1] or len(Y) < min_bin * 2:
            continue
        A = unmix(Y, S)
        A = A[np.isfinite(A).all(axis=1)]
        if len(A) < min_bin * 2:
            continue
        prim = A[:, j]
        neg, pos, separated = _split_negative_positive(prim)
        if separated and min(neg.sum(), pos.sum()) >= min_bin:
            col = _ssm_two_populations(A[pos], A[neg], j)
        elif A_un is not None and int((prim > 0).sum()) >= min_bin:
            col = _ssm_two_populations(A, A_un, j)
        else:
            col = _ssm_binned(A, j, n_bins, min_bin)
        if col is not None:
            SSM[:, j] = col
    return SSM, list(fluors)


def _ssm_two_populations(A_pos, A_neg, j):
    """SSM column `j` from the unmixed positive and negative events of stain
    `j`: sqrt((σ²pos − σ²neg) / (Fpos − Fneg)) per fluor, 0 on the diagonal
    (a stain's spread into its own channel is not part of the matrix). None
    when the populations do not differ on fluor `j`."""
    dF = float(np.median(A_pos[:, j]) - np.median(A_neg[:, j]))
    if not dF > 0:
        return None
    var = np.array([_robust_sd(A_pos[:, i]) ** 2 - _robust_sd(A_neg[:, i]) ** 2
                    for i in range(A_pos.shape[1])])
    col = np.sqrt(np.maximum(var, 0.0) / dF)
    col[j] = 0.0
    return col


def _ssm_binned(A, j, n_bins, min_bin):
    """SSM column `j` for a stain with no negative population: the slope of
    σ² against the primary F over its quantile bins (σ² = a + SS²·F), 0 on the
    diagonal. None with fewer than 3 usable bins or no spread in F."""
    prim = A[:, j]
    edges = np.unique(np.quantile(prim, np.linspace(0.0, 1.0, n_bins + 1)))
    binidx = np.clip(np.searchsorted(edges, prim, side='right') - 1,
                     0, max(edges.size - 2, 0))
    use = [b for b in range(edges.size - 1)
           if int((binidx == b).sum()) >= min_bin]
    F = np.array([np.median(prim[binidx == b]) for b in use])
    if len(use) < 3 or not np.ptp(F) > 0:
        return None
    col = np.array([
        np.sqrt(max(float(np.polyfit(
            F, [_robust_sd(A[binidx == b, i]) ** 2 for b in use], 1)[0]), 0.0))
        for i in range(A.shape[1])])
    col[j] = 0.0
    return col


def unmixing_qc(single_stains, S, fluors, nonneg=False, sim_threshold=0.98,
                top_spread=5, unstained=None, reference_spectra=None):
    """Bundle the spectral-unmixing diagnostics into one report dict.

    Returns ``{fluors, similarity, ssm, condition_number, similar_pairs,
    worst_spread, degenerate_fluors, reference_spectra}`` where
    ``similar_pairs`` lists fluor pairs whose spectral
    cosine similarity is ≥ ``sim_threshold`` (descending), and
    ``worst_spread`` lists the ``top_spread`` largest finite SSM entries
    (``into``/``from`` fluor + value) — the pairs most worth scrutinizing.
    `unstained` is the negative for a single stain with none of its own (see
    :func:`spillover_spread_matrix`); `nonneg` does not affect the SSM, which
    is measured on unconstrained abundances. `reference_spectra` is the
    ``diagnostics`` list :func:`build_reference_spectra` filled (how each
    control's spectrum was estimated, and which are too dim), carried
    through JSON-safe as ``reference_spectra`` (``[]`` without it)."""
    sim = spectral_similarity_matrix(S)
    cond = spectral_condition_number(S)
    ssm, _ = spillover_spread_matrix(single_stains, S, fluors,
                                     unstained=unstained)
    nf = len(fluors)

    similar_pairs = []
    for i in range(nf):
        for j in range(i + 1, nf):
            if sim[i, j] >= sim_threshold:
                similar_pairs.append({'fluor_a': fluors[i], 'fluor_b': fluors[j],
                                      'similarity': float(sim[i, j])})
    similar_pairs.sort(key=lambda d: d['similarity'], reverse=True)

    spread = []
    for i in range(nf):
        for j in range(nf):
            v = ssm[i, j]
            if i != j and np.isfinite(v):
                spread.append({'into': fluors[i], 'from': fluors[j],
                               'spread': float(v)})
    spread.sort(key=lambda d: d['spread'], reverse=True)

    # A fluor with no usable reference spectrum can never appear in
    # similar_pairs (its similarities are undefined), so without this the
    # report would say "no problematic pairs" about a fluor that cannot be
    # unmixed at all. Name it instead.
    degenerate = [fluors[i] for i in range(nf)
                  if not np.isfinite(sim[i, i])]

    return {'fluors': list(fluors), 'similarity': sim, 'ssm': ssm,
            'condition_number': cond, 'similar_pairs': similar_pairs,
            'worst_spread': spread[:top_spread],
            'degenerate_fluors': degenerate,
            'reference_spectra': reference_spectra_report(reference_spectra)}


# ── Reference-spectrum diagnostics, for the QC window and the CLI report ────

_REPORT_KEYS = ('fluor', 'method', 'requested_method', 'reason',
                'primary_detector', 'separation_index', 'n_events',
                'n_positives', 'agreement', 'af_cosine', 'warning',
                'split_half_cosine')


def reference_spectra_report(diagnostics):
    """JSON-safe copy of :func:`build_reference_spectra` ``diagnostics``:
    the reported keys only, numpy scalars as Python ones, NaN as None."""
    out = []
    for d in diagnostics or []:
        row = {}
        for k in _REPORT_KEYS:
            v = d.get(k)
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (float, np.floating)):
                v = float(v) if np.isfinite(v) else None
            row[k] = v
        out.append(row)
    return out


def _fmt(v, spec):
    return '–' if v is None else format(v, spec)


def reference_spectra_lines(report):
    """Plain-text lines describing how each reference spectrum was built
    (one per control) and which controls are too dim to trust."""
    lines = []
    for d in report or []:
        how = ('matched autofluorescence' if d['method'] == 'matched'
               else 'total signal')
        bits = [f"primary {d['primary_detector']}"
                if d['primary_detector'] is not None else None,
                f"separation {_fmt(d['separation_index'], '.1f')}"
                if d['separation_index'] is not None else None,
                f"{d['n_positives']} positives"
                if d['n_positives'] is not None else None]
        lines.append(f"  • {d['fluor']}: {how} ("
                     + ', '.join(b for b in bits if b) + f") — {d['reason']}")
        if d['warning']:
            lines.append(f"    [!] too dim: {d['warning']}. Use beads or a "
                         "brighter fluor for this control.")
    return lines


def reference_spectra_markdown(report):
    """Markdown section (list of lines) for the reference-spectra report."""
    if not report:
        return []
    md = ["## Reference spectra",
          "How each single-stain spectrum was estimated: 'total' = brightest "
          "events by total signal minus the mean autofluorescence; 'matched' "
          "= brightest by the dye's own excess, each minus the "
          "autofluorescence of matched unstained events. Agreement is the "
          "cosine between the two estimates.", "",
          "| Fluor | Method | Primary detector | Separation index | "
          "Positives | Agreement | Cosine to AF | Reason |",
          "|---|---|---|---|---|---|---|---|"]
    for d in report:
        md.append(
            f"| {d['fluor']} | {d['method']} | "
            f"{_fmt(d['primary_detector'], '')} | "
            f"{_fmt(d['separation_index'], '.2f')} | "
            f"{_fmt(d['n_positives'], 'd')} | {_fmt(d['agreement'], '.4f')} | "
            f"{_fmt(d['af_cosine'], '.3f')} | {d['reason']} |")
    dim = [d for d in report if d['warning']]
    if dim:
        md += ["", "### Controls too dim to define a spectrum reliably"]
        md += [f"- **{d['fluor']}**: {d['warning']}. Use beads or a brighter "
               "fluor for this control." for d in dim]
    md.append("")
    return md
