"""DNA-content cell-cycle modelling: the best-guess DNA-stain channel
(``find_dna_channel``), the two-peak model of its histogram with phase
boundaries and percentages (``analyze_dna``), and per-event phase labels
(``assign_phase``, one of ``CELL_CYCLE_PHASES`` or 'NA').

Core module: numpy, with scipy imported where it is used. No Tk, and it does
not import ``openflo.pipeline`` (pipeline imports it).
"""

import re

import numpy as np

from .auto_gating import resolution_bins as _resolution_bins
from .robust import FWHM_TO_SD, robust_sd

# DNA-content cell-cycle modelling. Given a DNA-stain intensity (PI, DAPI,
# FxCycle, 7-AAD, Hoechst, DRAQ5, …), the histogram is bimodal: a G0/G1
# peak and a G2/M peak at ~2× the DNA content, with S phase spread between.
# We locate the two peaks, estimate each peak's spread robustly, and assign
# every event to a phase by intensity boundaries — a pragmatic, explainable
# alternative to the Dean-Jett-Fox / Watson deconvolution that doesn't need
# a curve-fitter and degrades gracefully on non-cycling samples.

# Dye name tokens we recognise as DNA-content stains (lowercased substrings,
# matched against antibody label first, then detector name).
DNA_DYES = (
    'fxcycle', 'propidium iodide', 'propidium', 'hoechst', 'draq5', 'draq7',
    'vybrant dyecycle', 'dyecycle', 'sytox', 'to-pro', 'topro', 'dapi',
    '7-aad', '7aad', 'pi',
)

# Ordered phase labels. 'cycling' = G1+S+G2M; sub-G1 (apoptotic/debris) and
# >G2M (aggregates/polyploid) are reported but excluded from the cycle %.
CELL_CYCLE_PHASES = ('sub-G1', 'G1', 'S', 'G2M', '>G2M')

# The widest G1 peak (CV, from its half-maximum width) still taken as a DNA
# peak. Stained nuclei give 2-8 %, a poor stain 10-12 %; a scatter channel's
# single bell is 20 % and more, and is no G1 peak.
MAX_G1_CV = 0.15

# A peak this many times the G1 peak's position (low, high) is taken as G2/M.
G2_WINDOW = (1.7, 2.3)


def find_dna_channel(sample):
    """Best-guess DNA-content detector for `sample`, or None.

    Matches known DNA-dye tokens against each channel's antibody label
    first, then its detector name. Prefers an Area (`-A`) channel. The
    short token 'pi' only matches as a whole word so it doesn't fire on
    'PI3K' etc."""
    labels = getattr(sample, 'channel_labels', {}) or {}
    cols = list(sample.data.columns)

    def matches(text):
        t = str(text).lower()
        for dye in DNA_DYES:
            if dye == 'pi':
                # Exclude digits from the boundaries too, else 'PI3K' matches
                # ('3' satisfies a bare (?![a-z]) lookahead) and gets mis-tagged
                # as the PI DNA stain — matching find_viability_channel's guard.
                if re.search(r'(?<![a-z0-9])pi(?![a-z0-9])', t):
                    return True
            elif dye in t:
                return True
        return False

    candidates = [det for det in cols
                  if matches(labels.get(det, det)) or matches(det)]
    if not candidates:
        return None
    for c in candidates:
        if c.upper().endswith('-A'):
            return c
    return candidates[0]


def analyze_dna(values, k=1.5, bins=256):
    """Model the DNA-content histogram → cell-cycle phase boundaries + %.

    Pure. `values` is the DNA-stain intensity (linear scale). `k` sets how
    many robust SDs around each peak count as G1 / G2M (the rest, between,
    is S). Returns a model dict:
        g1_mean, g1_sd, g2_mean, g2_sd, g1_hi, g2_lo  (phase boundaries)
        pct_g1, pct_s, pct_g2m   (% of cycling events)
        counts  {phase: n}, n_cycling, n  (totals)
        g2_found (bool — a G2/M peak was found near 2× G1; else G2/M is
                  placed at 2× G1)
        ok      (bool — False when no usable peak was found)
        reason  (why ``ok`` is False, '' when it is True)
    Use assign_phase(values, model) to label arbitrary arrays with the
    same boundaries."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    nan = float('nan')
    model = {'g1_mean': nan, 'g1_sd': nan, 'g2_mean': nan, 'g2_sd': nan,
             'g1_hi': nan, 'g2_lo': nan, 'pct_g1': nan, 'pct_s': nan,
             'pct_g2m': nan, 'counts': {}, 'n_cycling': 0, 'n': 0, 'ok': False,
             'g2_found': False, 'reason': 'too few events'}

    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    model['n'] = int(v.size)
    if v.size < 50:
        return model

    lo, hi = np.percentile(v, [0.5, 99.5])
    if hi <= lo:
        model['reason'] = 'no spread in the values'
        return model
    bins = _resolution_bins(v, lo, hi, bins)
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    sm = gaussian_filter1d(hist.astype(float), sigma=2.0)
    if sm.max() <= 0:
        model['reason'] = 'no spread in the values'
        return model

    peaks, _ = find_peaks(sm, prominence=sm.max() * 0.05)
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(sm))])
    # G1 = the most prominent (tallest) peak.
    order = peaks[np.argsort(sm[peaks])[::-1]]
    g1_mean = float(centers[order[0]])
    if g1_mean <= 0:
        g1_mean = float(np.median(v))

    # G2/M ≈ 2× G1: nearest remaining peak in G2_WINDOW; else nominal 2×.
    g2_mean = None
    for pk in order[1:]:
        c = float(centers[pk])
        if G2_WINDOW[0] * g1_mean <= c <= G2_WINDOW[1] * g1_mean:
            g2_mean = c
            break
    g2_found = g2_mean is not None
    model['reason'] = _no_g1_peak_reason(sm, centers, order[0], g1_mean)
    if model['reason']:
        return model
    if g2_mean is None:
        g2_mean = 2.0 * g1_mean

    g1_win = v[(v >= 0.85 * g1_mean) & (v <= 1.15 * g1_mean)]
    g1_sd = robust_sd(g1_win)
    if not np.isfinite(g1_sd) or g1_sd <= 0:
        g1_sd = 0.05 * g1_mean
    g2_win = v[(v >= 0.9 * g2_mean) & (v <= 1.1 * g2_mean)]
    g2_sd = robust_sd(g2_win)
    if not np.isfinite(g2_sd) or g2_sd <= 0:
        g2_sd = g1_sd * 1.4   # G2 CV ~ G1 CV; width scales with mean

    g1_hi = g1_mean + k * g1_sd
    g2_lo = g2_mean - k * g2_sd
    if g1_hi >= g2_lo:                      # peaks overlap → split at midpoint
        mid = 0.5 * (g1_mean + g2_mean)
        g1_hi = min(g1_hi, mid)
        g2_lo = max(g2_lo, mid)

    model.update(g1_mean=g1_mean, g1_sd=g1_sd, g2_mean=g2_mean, g2_sd=g2_sd,
                 g1_hi=g1_hi, g2_lo=g2_lo, ok=True, g2_found=g2_found,
                 reason='')

    phases = assign_phase(v, model)
    counts = {p: int(np.count_nonzero(phases == p)) for p in CELL_CYCLE_PHASES}
    cyc = counts['G1'] + counts['S'] + counts['G2M']
    model['counts'] = counts
    model['n_cycling'] = cyc
    if cyc > 0:
        model['pct_g1'] = 100.0 * counts['G1'] / cyc
        model['pct_s'] = 100.0 * counts['S'] / cyc
        model['pct_g2m'] = 100.0 * counts['G2M'] / cyc
    return model


def _no_g1_peak_reason(sm, centers, i, mean):
    """Why the smoothed histogram ``sm``'s tallest peak (bin ``i``, at
    ``mean``) cannot be a G1 DNA peak, or '' when it can. A peak wider than
    MAX_G1_CV (a scatter channel's single bell) would put nearly every event
    in G1. Further peaks are not checked: an aneuploid clone's G1 and G2/M
    sit beside and beyond the diploid ones."""
    g1_cv = _peak_cv(sm, centers, i, mean)
    if not np.isfinite(g1_cv):
        return "no clear G1 peak"
    if g1_cv > MAX_G1_CV:
        return (f"the tallest peak is too wide for a G1 DNA peak "
                f"(CV {100 * g1_cv:.0f} %, at most {100 * MAX_G1_CV:.0f} %)")
    return ''


def _peak_cv(sm, centers, i, mean):
    """CV of the smoothed histogram ``sm``'s peak at bin ``i``, from its full
    width at half maximum (FWHM * FWHM_TO_SD / ``mean``); inf when the peak
    does not fall to half its height on both sides inside the histogram."""
    half = sm[i] / 2.0
    left = np.flatnonzero(sm[:i] < half)
    right = np.flatnonzero(sm[i:] < half)
    if not left.size or not right.size or mean <= 0:
        return float('inf')
    a, b = left[-1], i + right[0]
    # Interpolate where the curve crosses half height on each side.
    xl = np.interp(half, [sm[a], sm[a + 1]], [centers[a], centers[a + 1]])
    xr = np.interp(half, [sm[b], sm[b - 1]], [centers[b], centers[b - 1]])
    return float((xr - xl) * FWHM_TO_SD / mean)


def assign_phase(values, model):
    """Label each value with its cell-cycle phase using `model`'s
    boundaries. Non-finite values and any input when the model is invalid
    become 'NA'. Returns an object ndarray of CELL_CYCLE_PHASES (+ 'NA')."""
    v = np.asarray(values, dtype=float)
    out = np.full(v.size, 'NA', dtype=object)
    if not model.get('ok'):
        return out
    g1_lo = model['g1_mean'] - (model['g1_hi'] - model['g1_mean'])
    g1_hi = model['g1_hi']
    g2_lo = model['g2_lo']
    g2_hi = model['g2_mean'] + (model['g2_mean'] - model['g2_lo'])
    finite = np.isfinite(v)
    out[finite] = '>G2M'                         # default for finite > g2_hi
    out[finite & (v < g1_lo)] = 'sub-G1'
    out[finite & (v >= g1_lo) & (v <= g1_hi)] = 'G1'
    out[finite & (v > g1_hi) & (v < g2_lo)] = 'S'
    out[finite & (v >= g2_lo) & (v <= g2_hi)] = 'G2M'
    return out
