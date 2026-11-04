"""Auto-clean (acquisition-cleaning) gates: the cleaning recipe
(AUTOCLEAN_METHODS), each method's keep-mask and diagnostics, and
``autoclean_keep_mask``, the mask of an 'autoclean' gate.

Core module: numpy, pandas and the standard library. No Tk, and it does not
import ``openflo.pipeline``.
"""

import copy
import logging
import re

import numpy as np

from .acquisition_qc import AcquisitionQC
from .auto_gating import resolution_bins as _resolution_bins
from .robust import MAD_TO_SD

# Name kept as 'openflo.pipeline' so existing log filters and caplog tests see
# these messages.
log = logging.getLogger('openflo.pipeline')


# ── Auto-clean (acquisition-cleaning) gate ─────────────────────────────────
#
# An 'autoclean' gate stores a RECIPE — a list of cleaning METHODS — not
# coordinates. Each method recomputes its keep-mask from whatever sample the
# gate is evaluated against, so copying the gate to other samples re-runs the
# calculations rather than reusing one sample's geometry (bubbles/debris/clogs
# sit in different places per sample). The gate's mask is the AND of every
# ENABLED method's keep-mask: events clean of ALL selected anomaly types.

AUTOCLEAN_METHODS = [
    {'key': 'debris',    'label': 'Debris (size: beads; valley opt-in)', 'params': {'mode': 'bead', 'bead_um': 8.0, 'min_um': 4.0}},
    {'key': 'viability', 'label': 'Dead cells (viability dye)',    'params': {}},
    {'key': 'doublets',  'label': 'Doublets (FSC-A/FSC-H)',        'params': {'tol': 0.25}},
    {'key': 'margin',    'label': 'Margin (saturation)',           'params': {'margin_frac': 0.01}},
    {'key': 'flow_rate', 'label': 'Flow rate (bubbles/clogs)',     'params': {'n_bins': 200, 'flow_rate_threshold': 5.0}},
    {'key': 'drift',     'label': 'Signal drift',                  'params': {'n_bins': 200, 'threshold': 5}},
]


def default_autoclean_methods():
    """A fresh, all-enabled copy of the standard cleaning recipe."""
    return [{'key': m['key'], 'label': m['label'], 'enabled': True,
             'params': copy.deepcopy(m['params'])} for m in AUTOCLEAN_METHODS]


def autoclean_methods_signature(gate):
    """A hashable signature of an autoclean gate's recipe (each method's key,
    enabled flag, and sorted params). Two gates with the same signature produce
    the same mask on the same data — used as a mask-cache key by the GUI."""
    out = []
    for m in gate.get('methods') or []:
        params = m.get('params') or {}
        out.append((m.get('key'), bool(m.get('enabled', True)),
                    tuple(sorted(params.items()))))
    return tuple(out)


def find_scatter_channel(columns, prefix, suffix='-A', fallback=True):
    """The scatter channel named ``prefix``...``suffix`` (e.g. FSC-A, SSC-H),
    case-insensitive; with ``fallback``, the first ``prefix`` channel when
    none ends in ``suffix``. None if none fits. ``columns`` is any iterable
    of names, or a DataFrame (its columns). The one lookup the auto-clean
    steps, FlowSample and the CLI share, so that they always agree."""
    names = [str(c) for c in columns]
    pu, su = prefix.upper(), suffix.upper()
    for c in names:
        cu = c.upper()
        if cu.startswith(pu) and cu.endswith(su):
            return c
    if fallback:
        for c in names:
            if c.upper().startswith(pu):
                return c
    return None


# Dye name tokens we recognise as viability / live-dead stains (lowercased
# substrings, matched against antibody label first, then detector name).
# Dead cells take up the dye and read HIGH; live cells exclude it and read low.
# Overlaps with cell_cycle.DNA_DYES (7-AAD, PI, DAPI, SYTOX, TO-PRO double as
# viability).
VIABILITY_DYES = (
    'live/dead', 'livedead', 'live-dead', 'l/d', 'viability', 'viadye',
    'viable', 'zombie', 'ghost dye', 'ghost', 'fixable viability',
    'fixable viable', 'fixable live', 'fvs', 'fvd', 'efluor 506',
    'efluor 780', 'ef506', 'ef780', 'aqua', 'near-ir', 'sytox', 'to-pro',
    'topro', '7-aad', '7aad', 'propidium', 'dapi', 'pi',
)


def find_viability_channel(columns, channel_labels=None):
    """Best-guess viability / live-dead detector among ``columns``, or None.

    Matches known viability-dye tokens against each channel's antibody label
    first (from ``channel_labels``, a ``{detector: label}`` dict), then its
    detector name. Prefers an Area (``-A``) channel. The short tokens 'pi' /
    'l/d' only match as whole words so they don't fire inside longer names."""
    labels = channel_labels or {}
    cols = list(columns)

    def matches(text):
        t = str(text).lower()
        for dye in VIABILITY_DYES:
            if dye in ('pi', 'l/d'):
                if re.search(r'(?<![a-z0-9/])' + re.escape(dye) + r'(?![a-z0-9])', t):
                    return True
            elif dye in t:
                return True
        return False

    candidates = [det for det in cols
                  if matches(labels.get(det, det)) or matches(det)]
    if not candidates:
        return None
    for c in candidates:
        if str(c).upper().endswith('-A'):
            return c
    return candidates[0]


def debris_mode(params):
    """The debris method's mode as the user meant it: trimmed and lower case,
    'bead' when unset. The parameters dialog stored it as typed, so 'Valley'
    must still mean the valley cut. Anything else is not a mode and cuts
    nothing (see :func:`autoclean_method_diagnostic`)."""
    return str((params or {}).get('mode') or 'bead').strip().lower()


def _autoclean_debris_mask(df, params):
    """Drop debris, following the standard manual gating hierarchy as closely
    as the mode allows.

    Resolution order (first applicable wins):
      1. **manual** — an explicit ``min_fsc`` FSC-A floor is used verbatim.
      2. **bead-calibrated absolute size** (the default, ``mode='bead'``) —
         when a bead anchor ``bead_fsc`` (the median FSC-A of size-calibration
         beads of diameter ``bead_um`` µm) is present and a target ``min_um``
         is set, keep events whose implied size is ≥ ``min_um`` µm, i.e.
         ``FSC-A >= min_um * bead_fsc / bead_um``. A pure 1-D size ruler — the
         most reproducible cut and, with a sub-cell ``min_um`` (≈4 µm), the most
         conservative (it removes only genuine sub-cellular fragments, never
         small-but-real cells such as lymphocytes).
      3. **2-D scatter gate** (``mode='valley'`` only — opt-in) — emulates the
         manual **FSC-A × SSC-A** debris polygon: an event is debris only when
         it is low on BOTH FSC-A AND SSC-A (the bottom-left cloud), each
         boundary being that channel's density valley below its median. This
         keeps low-FSC / high-SSC granular cells (granulocytes, etc.) that a
         1-D FSC cut would wrongly discard. Degrades to a 1-D FSC valley when
         there's no usable SSC-A. Its limits are in :func:`_debris_fsc_valley`.
    No-op when there's no FSC-A column or no usable cutoff. In particular
    ``mode='bead'`` (the default) with no bead anchor cuts NOTHING: it must
    not fall back to the valley cut, which cannot tell debris from small real
    cells by scatter alone (it takes all the platelets of a platelet assay,
    all the bacteria of a bacterial-load assay, and the lymphocytes under
    large tumour cells), so the valley cut runs only when asked for."""
    n = len(df)
    fsc = find_scatter_channel(df, 'FSC', '-A')
    if fsc is None:
        return np.ones(n, dtype=bool)
    vals = np.asarray(df[fsc].values, dtype=float)
    fin  = np.isfinite(vals)
    params = params or {}
    # (1) explicit FSC-A floor — deterministic, applies regardless of N.
    manual = params.get('min_fsc')
    if manual is not None:
        keep = fin & (vals >= float(manual))
        # A FROZEN valley gate pins the SSC-granular threshold too: add back the
        # low-FSC / high-SSC granulocytes the 2-D valley gate rescued, so a
        # frozen gate replays the full 2-D cut instead of a lossy 1-D floor.
        gthr = params.get('min_ssc_granular')
        if gthr is not None:
            ssc = params.get('ssc_channel') or find_scatter_channel(
                df, 'SSC', '-A')
            if ssc is not None and ssc in df.columns:
                svals = np.asarray(df[ssc].values, dtype=float)
                low_fsc = fin & (vals < float(manual))
                granular = (low_fsc & np.isfinite(svals)
                            & (svals >= float(gthr)))
                keep = keep | granular
        return keep
    # (2) bead-calibrated absolute size — deterministic, applies regardless of N.
    mode     = debris_mode(params)
    bead_fsc = params.get('bead_fsc')
    min_um   = params.get('min_um')
    bead_um  = params.get('bead_um', 8.0)
    if (mode == 'bead' and bead_fsc and min_um
            and float(bead_fsc) > 0 and float(bead_um) > 0):
        thr = float(min_um) * float(bead_fsc) / float(bead_um)
        return fin & (vals >= thr)
    # (3) the 2-D scatter gate, only when asked for (needs enough events to
    # estimate). Uses the strict bimodal valley (not Otsu) so a UNIMODAL
    # FSC-A is never bisected — only a separate low-FSC mode below the median
    # is cut.
    finite = vals[fin]
    if mode != 'valley' or finite.size < 50:
        return np.ones(n, dtype=bool)
    fthr = _debris_fsc_cut(finite)
    if fthr is None:
        return np.ones(n, dtype=bool)
    low_fsc = fin & (vals < float(fthr))
    ssc = find_scatter_channel(df, 'SSC', '-A')
    if ssc is not None and ssc != fsc and params.get('use_ssc', True):
        # Rescue granular cells: among the small (low-FSC) events, is there a
        # genuinely separate HIGH-SSC subpopulation (granulocytes)? Look for a
        # bimodal SSC split WITHIN those events only — a global SSC valley
        # tends to split off the high-SSC tail, not the debris floor. Only when
        # such a split exists do we keep the high-SSC side (= the manual
        # polygon's upper-left lobe); otherwise the small events are all debris.
        svals = np.asarray(df[ssc].values, dtype=float)
        ss_lo = svals[low_fsc & np.isfinite(svals)]
        sthr  = _bimodal_valley(ss_lo) if ss_lo.size >= 50 else None
        if sthr is not None:
            granular = low_fsc & np.isfinite(svals) & (svals >= float(sthr))
            return ~(low_fsc & ~granular)      # debris = small AND non-granular
    return ~low_fsc                            # 1-D FSC fallback (no granular lobe)


# The histogram bin count capped at the number of distinct values
# (resolution_bins) lives in auto_gating; it is imported at the top as
# _resolution_bins.


# Fewest events a histogram bin may average in _bimodal_valley.
_VALLEY_MIN_PER_BIN = 10


def _bimodal_valley(values, bins=256, smooth=2.0, lowest=False):
    """The valley between the two most prominent peaks of a smoothed
    histogram, but ONLY when the data is genuinely bimodal. Returns None for
    unimodal data — unlike :func:`auto_threshold`, it does NOT fall back to
    Otsu, so a caller can treat 'no valley' as 'do not split'.

    With ``lowest`` it is the first genuine valley scanning up from the low
    end instead: the valley above the LOWEST mode, whatever else is taller.
    The debris cut needs that (see :func:`_debris_fsc_cut`).

    Three things kept this from returning None for unimodal data, or from
    finding a real second mode:

    * A bin must average ``_VALLEY_MIN_PER_BIN`` events. Capped only by the
      distinct values, a small continuous sample gets ~n bins of ~1 event
      each, and 2-bin smoothing leaves noise peaks above the 5% prominence
      bar: single normal samples of 200 events were then split in 43 of 100
      seeds, uniform ones of 1000 in 71; with the cap, 0 and 2.
      The granular rescue calls this on the low-FSC subset (>= 50 events),
      where a false split kept debris as "granulocytes". From 2560 events
      the cap does not bind.
    * The pair of peaks is the two most PROMINENT, not the two tallest.
      Below ~50k events, noise bumps on the major mode pass the 5% bar
      and are taller than a minor mode, and the valley between two of
      them is shallow. A 70/30 mixture of modes 4 SD apart is split in 100
      of 100 seeds at 5000 events (39 with the two tallest).
    * The smoothed histogram has a zero on each side, so a mode at the edge
      of the 0.5-99.5 percentile window (debris piled at the origin, a
      flat debris plateau) is a peak. ``find_peaks`` cannot return an end
      point, and an edge plateau had almost no prominence.

    Measured over 100 seeds each: unimodal normal, uniform, log-normal and
    exponential samples of 60 to 5000 events are split in at most 3 (up to
    92 without these three rules); 2-mode mixtures (minor mode 10-50%,
    3.5-5 SD apart) are split between the modes in 57-100 of 100 from 500
    events (0-80 without them). Modes 3 SD apart are not split. At 200
    events and fewer a real minor mode is found in 0-17 of 100: at this
    smoothing the data cannot tell it from noise, and 'do not split' is the
    safe answer. (Without the rules it is found in 1-83 of 100 there, but
    11-90 of 100 unimodal samples of those sizes are split too.)"""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 50:
        return None
    lo, hi = np.percentile(v, [0.5, 99.5])
    if hi <= lo:
        return None
    bins = max(8, min(_resolution_bins(v, lo, hi, bins),
                      v.size // _VALLEY_MIN_PER_BIN))
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    sm = gaussian_filter1d(hist.astype(float), smooth)
    if sm.max() <= 0:
        return None
    sm = np.r_[0.0, sm, 0.0]              # sm[i] is the bin at centers[i - 1]
    peaks, props = find_peaks(sm, prominence=sm.max() * 0.05)
    if peaks.size < 2:
        return None
    if lowest:
        pairs = list(zip(peaks[:-1], peaks[1:], strict=True))
    else:
        top = peaks[np.argsort(props['prominences'])[::-1][:2]]
        pairs = [(min(top), max(top))]
    for a, b in pairs:
        valley = a + int(np.argmin(sm[a:b + 1]))
        # Demand a genuine separation: the valley must dip to ≤ half the
        # SHORTER of the two peaks. Sampling-noise bumps on a single mode
        # leave a shallow 'valley' near the peak height — not a split.
        if sm[valley] <= 0.5 * float(min(sm[a], sm[b])):
            return float(centers[valley - 1])
    return None


# The lowest FSC-A mode is debris only when its events' median FSC-A is at
# most this fraction of the larger events' median. See _debris_fsc_valley.
_DEBRIS_MAX_FSC_RATIO = 0.25


def _debris_fsc_valley(finite):
    """``(cut, why_not)`` for the 2-D scatter gate's FSC-A cut. ``cut`` is
    the valley above the LOWEST FSC-A mode, or None (do not cut) with
    ``why_not`` 'unimodal' (no such valley below the median) or
    'cell-sized' (the lowest mode is too large to be debris). ``finite``
    is the finite FSC-A values, at least 50.

    The lowest mode, not the two most prominent: doublets and large cells
    are modes too, and when the second most prominent sat ABOVE the cells
    the valley landed above the median and nothing was cut. With 1000
    debris spread over U(2k, 15k), 8000 cells and 1000 doublets, the cut
    fell between cells and doublets, 0 of 1000 debris were removed and the
    diagnostic called FSC-A unimodal.

    Being the lowest mode below the median does not make a population
    debris. In lysed whole blood with no debris (lymphocytes 30%, monocytes
    8%, granulocytes 62%) the lowest mode is the lymphocytes, and their SSC
    is unimodal, so the granular rescue cannot save them: 99.9% of them
    were cut (a two-tallest rule cuts 65%). Debris is small:
    the cut is made only when the lowest mode's median FSC-A is at most
    ``_DEBRIS_MAX_FSC_RATIO`` of the median of the events above it.
    Measured over 20 seeds a case, on linear FSC-A: a debris mode (piled at
    the origin, uniform, or a peak at 8% of the cells' FSC-A; 1-45% of
    events; under PBMC or whole blood) gives 0.036-0.12; the lymphocytes
    under granulocytes give 0.48. On transformed scatter the ratio is not
    a size ratio and errs toward 0.5 and above, toward not cutting. When
    the evidence is not there, not cutting is the safe answer: a debris
    mode too weak to form the lowest valley (50 events in 10000 whole
    blood cells) is left in, and so is a debris majority (the valley is
    then above the median).

    KNOWN LIMITS — why this runs only when ``mode='valley'`` is chosen.
    Scatter alone cannot tell debris from small real cells; this rule was
    kept over the two-tallest rule (28641c3) and the lowest-valley rule
    without the size test (d883512) because, over two independent attack
    batteries, it removed the fewest real cells (mean 12.7% against 13.9%
    and 16.7% over 72 class-scenarios; lymphocytes in whole blood with 0
    to 100 debris events: none, against 65-95% and 90-99.9%), at the price
    of leaving more debris in (53% against 88% and 25%). Measured, 20
    seeds a case:

    * It removes small real cells under a much larger population:
      lymphocytes (45k FSC-A) under tumour cells, macrophages or other
      large cells whose median FSC-A is 190k or more are all cut (50-65% at
      180k, none at 150k).
    * It removes small particles that are the point of the assay: all the
      platelets of an unlysed platelet sample, all the free bacteria of a
      bacterial-load sample.
    * It keeps debris that piles up at a high instrument FSC threshold, or
      overlaps the cells: of debris piled at a 12k threshold under PBMC
      (60k) 0-9% is cut, of overlapping debris 0-27%.
    * Linear FSC-A only: on log-amplified or arcsinh-scaled scatter it cuts
      nothing.

    A bead anchor (``mode='bead'``) or a manual ``min_fsc`` has none of
    these limits."""
    thr = _bimodal_valley(finite, lowest=True)
    if thr is None or thr >= float(np.median(finite)):
        return None, 'unimodal'
    small = float(np.median(finite[finite < thr]))
    rest = float(np.median(finite[finite >= thr]))
    if not (rest > 0 and small <= _DEBRIS_MAX_FSC_RATIO * rest):
        return None, 'cell-sized'
    return thr, ''


def _debris_fsc_cut(finite):
    """The 2-D scatter gate's FSC-A cut, or None: see
    :func:`_debris_fsc_valley`."""
    return _debris_fsc_valley(finite)[0]


def _autoclean_viability_mask(df, params):
    """Keep LIVE cells — drop the high-signal dead population on a viability
    dye. The detector is ``params['channel']`` when set (and present), else
    auto-detected by dye-name tokens among the columns. A manual
    ``max_signal`` ceiling is used verbatim; otherwise the live/dead split is
    the density valley, applied only when a dead mode sits ABOVE the median
    (so an all-live, unimodal sample is never bisected). No-op when no
    viability channel is found or there's too little data."""
    n = len(df)
    params = params or {}
    ch = params.get('channel')
    if not ch or ch not in df.columns:
        ch = find_viability_channel(list(df.columns))   # labels unavailable here
    if not ch or ch not in df.columns:
        return np.ones(n, dtype=bool)
    vals   = np.asarray(df[ch].values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size < 50:
        return np.ones(n, dtype=bool)
    manual = params.get('max_signal')
    if manual is not None:
        return np.isfinite(vals) & (vals <= float(manual))
    # Require a genuine bimodal live/dead split (no Otsu fallback) and the
    # dead mode must sit ABOVE the median — so an all-live, unimodal sample
    # is never bisected.
    thr = _bimodal_valley(finite)
    if thr is None or thr <= float(np.median(finite)):
        return np.ones(n, dtype=bool)
    return np.isfinite(vals) & (vals <= float(thr))


# How many robust SDs of the FSC-A/FSC-H ratio the singlet window may span.
# See _autoclean_doublets_mask: generous on purpose.
_DOUBLET_SIGMA_K = 10.0


def _autoclean_doublets_mask(df, params, ref_mask=None):
    """Keep singlets via the FSC-A/FSC-H ratio (within ±tol of the median
    ratio). No-op if FSC-A or FSC-H is missing.

    ``ref_mask`` (optional) restricts which events the *median* ratio is taken
    over — the standard gating hierarchy removes debris BEFORE the singlet gate,
    so the window is centred on cell-sized events, not debris. The keep
    predicate is still applied to every event (AND-of-methods semantics); only
    the reference population for the median changes."""
    n   = len(df)
    tol = float(params.get('tol', 0.25))
    if tol <= 0:
        return np.ones(n, dtype=bool)
    fa = find_scatter_channel(df, 'FSC', '-A')
    fh = find_scatter_channel(df, 'FSC', '-H')
    if fa is None or fh is None or fa == fh:
        return np.ones(n, dtype=bool)
    a = np.asarray(df[fa].values, dtype=float)
    h = np.asarray(df[fh].values, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(h > 0, a / h, np.nan)
    valid = np.isfinite(ratio)
    if not valid.any():
        return np.ones(n, dtype=bool)
    ref = valid
    if ref_mask is not None:
        ref = valid & np.asarray(ref_mask, dtype=bool)
        if not ref.any():                    # debris removed everything → fall back
            ref = valid
    med = float(np.nanmedian(ratio[ref]))
    if not med > 0:
        # A non-positive median ratio inverts the multiplicative window
        # (lo > hi), which matches nothing and would delete the entire
        # sample. Doublets cannot be identified from such a ratio at all.
        log.warning("doublet filter skipped: the median FSC-A/FSC-H ratio is "
                    "%.4g, so no acceptance window can be formed.", med)
        return np.ones(n, dtype=bool)

    # `tol` is a FRACTION OF THE MEDIAN, but how far the singlet ratio
    # actually scatters depends on the scale the scatter channels are on.
    # After an arcsinh or logicle bake the ratio's median barely moves while
    # its spread collapses — measured, a CV of 14.5% became 1.7% — so a fixed
    # ±25% window alone swallows the doublet population whole: 4,000 of 4,000
    # doublets retained, where the same events on linear scatter lose all
    # 4,000. The filter would silently be a no-op on transformed data.
    #
    # A window derived from a scale-dependent quantity breaks this way (see
    # the median-ratio handling in filter_doublets). The band is therefore
    # also bounded by the ratio's OWN robust spread, which is what
    # auto_singlet_gate already uses. `tol` keeps its meaning as the widest
    # the window may be; the dispersion can only narrow it. With no
    # measurable spread there is nothing to narrow by, so `tol` stands alone.
    #
    # k is deliberately generous. At 5 robust SDs the bound would also trim
    # 0.2% of genuine singlets from the golden sample (the golden baseline
    # catches it), so it is set where a well-behaved linear panel is left
    # exactly as it was while a collapsed spread (8x tighter after arcsinh)
    # is still caught with room to spare.
    dev = np.abs(ratio[ref] - med)
    sigma = MAD_TO_SD * float(np.nanmedian(dev))
    half = tol * med
    if sigma > 0:
        half = min(half, _DOUBLET_SIGMA_K * sigma)
    lo, hi = med - half, med + half
    return valid & (ratio >= lo) & (ratio <= hi)


def autoclean_debris_threshold(df, params):
    """The scalar FSC-A keep-threshold the debris method resolves to on ``df``
    (manual ``min_fsc`` → bead absolute size → density valley when
    ``mode='valley'``), or None when it wouldn't cut — as for bead mode with
    no bead anchor. Used to FREEZE an auto cut to a fixed value when
    copying."""
    params = params or {}
    fsc = find_scatter_channel(df, 'FSC', '-A')
    if fsc is None:
        return None
    manual = params.get('min_fsc')
    if manual is not None:
        return float(manual)
    mode = debris_mode(params)
    bead_fsc, min_um = params.get('bead_fsc'), params.get('min_um')
    bead_um = params.get('bead_um', 8.0)
    if (mode == 'bead' and bead_fsc and min_um
            and float(bead_fsc) > 0 and float(bead_um) > 0):
        return float(min_um) * float(bead_fsc) / float(bead_um)
    if mode != 'valley':
        return None
    vals = np.asarray(df[fsc].values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size < 50:
        return None
    thr = _debris_fsc_cut(finite)
    return None if thr is None else float(thr)


def autoclean_debris_freeze(df, params):
    """Frozen params that reproduce this sample's debris cut on any target:
    ``{'min_fsc': ...}`` for a manual / bead / 1-D-valley cut, plus
    ``{'min_ssc_granular', 'ssc_channel'}`` when the valley path found a
    high-SSC granulocyte lobe — so a frozen valley gate replays the full 2-D
    rescue instead of collapsing to a lossy 1-D floor. ``{}`` if it wouldn't cut."""
    params = params or {}
    thr = autoclean_debris_threshold(df, params)
    if thr is None:
        return {}
    out: dict[str, float | str] = {'min_fsc': float(thr)}
    # A manual floor or a bead absolute-size cut is a genuine 1-D threshold —
    # only the 2-D valley cut carries an SSC granulocyte rescue to pin.
    if params.get('min_fsc') is not None:
        return out
    bead_fsc = params.get('bead_fsc')
    if (debris_mode(params) == 'bead' and bead_fsc
            and params.get('min_um') and float(bead_fsc) > 0):
        return out
    fsc = find_scatter_channel(df, 'FSC', '-A')
    ssc = find_scatter_channel(df, 'SSC', '-A')
    if (fsc is None or ssc is None or ssc == fsc
            or not params.get('use_ssc', True)):
        return out
    vals = np.asarray(df[fsc].values, dtype=float)
    low_fsc = np.isfinite(vals) & (vals < float(thr))
    svals = np.asarray(df[ssc].values, dtype=float)
    ss_lo = svals[low_fsc & np.isfinite(svals)]
    sthr = _bimodal_valley(ss_lo) if ss_lo.size >= 50 else None
    if sthr is not None:
        out['min_ssc_granular'] = float(sthr)
        out['ssc_channel'] = ssc
    return out


def autoclean_viability_threshold(df, params):
    """The scalar dye-signal ceiling the viability method resolves to (events
    above it are dead), or None when it wouldn't cut. For freezing on copy."""
    params = params or {}
    ch = params.get('channel')
    if not ch or ch not in df.columns:
        ch = find_viability_channel(list(df.columns))
    if not ch or ch not in df.columns:
        return None
    manual = params.get('max_signal')
    if manual is not None:
        return float(manual)
    vals = np.asarray(df[ch].values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size < 50:
        return None
    thr = _bimodal_valley(finite)
    if thr is None or thr <= float(np.median(finite)):
        return None
    return float(thr)


def freeze_autoclean_gate(gate, df, channel_labels=None):
    """Return a deep copy of an ``autoclean`` ``gate`` with its auto-derived
    cuts PINNED to the values computed from ``df`` (one specific sample), so the
    copy applies identical thresholds everywhere instead of recomputing per
    sample. Debris → fixed ``min_fsc`` (plus ``min_ssc_granular`` + ``ssc_channel``
    when the valley path has a 2-D granulocyte rescue, so freezing keeps those
    cells instead of collapsing to a lossy 1-D floor); viability → resolved
    ``channel`` + fixed ``max_signal``. Methods without a single threshold
    (doublets, margin, flow-rate, drift) are left untouched (still per-sample).
    Non-autoclean gates are returned unchanged."""
    g = copy.deepcopy(gate)
    if g.get('kind') != 'autoclean':
        return g
    for m in g.get('methods') or []:
        key = m.get('key')
        mp = m.setdefault('params', {})
        if key == 'debris':
            mp.update(autoclean_debris_freeze(df, mp))
        elif key == 'viability':
            ch = mp.get('channel') or find_viability_channel(
                list(df.columns), channel_labels)
            if ch:
                mp['channel'] = ch
            thr = autoclean_viability_threshold(df, mp)
            if thr is not None:
                mp['max_signal'] = float(thr)
    return g


def _finite_values(df, ch):
    vals = np.asarray(df[ch].values, dtype=float)
    return vals[np.isfinite(vals)]


def _debris_diagnostic(df, params, channel_labels):
    fsc = find_scatter_channel(df, 'FSC', '-A')
    if fsc is None:
        return "no FSC-A channel"
    if params.get('min_fsc') is not None:
        return "nothing below the manual min_fsc"
    mode = debris_mode(params)
    has_bead = bool(params.get('bead_fsc')) and bool(params.get('min_um'))
    if mode == 'bead':
        if has_bead:
            return None        # deterministic bead cut — if it dropped 0, that's real
        return ("no bead size anchor — nothing cut. Load beads, set a "
                "manual FSC floor, or switch to 'valley' to cut by "
                "scatter")
    if mode != 'valley':
        return (f"unknown debris mode {params.get('mode')!r} — nothing "
                "cut. Use 'bead' or 'valley'")
    finite = _finite_values(df, fsc)
    if finite.size < 50:
        return "too few events"
    cut, why_not = _debris_fsc_valley(finite)
    if cut is not None:
        return None
    if why_not == 'cell-sized':
        return "the lowest FSC-A mode is cell-sized, not debris — nothing cut"
    return "FSC-A is unimodal — no low-debris mode to cut"


def _viability_diagnostic(df, params, channel_labels):
    ch = params.get('channel')
    if not ch or ch not in df.columns:
        ch = find_viability_channel(list(df.columns), channel_labels)
    if not ch or ch not in df.columns:
        return ("no viability dye detected — set the channel via right-click "
                "(panel has none?)")
    finite = _finite_values(df, ch)
    if finite.size < 50:
        return "too few events"
    if params.get('max_signal') is not None:
        return f"nothing above the manual ceiling on {ch}"
    thr = _bimodal_valley(finite)
    if thr is None:
        return f"no bimodal live/dead split on {ch}"
    if thr <= float(np.median(finite)):
        return (f"the high-signal population is the majority on {ch} — "
                "not treated as dead")
    return None


def _doublets_diagnostic(df, params, channel_labels):
    fa = find_scatter_channel(df, 'FSC', '-A')
    fh = find_scatter_channel(df, 'FSC', '-H')
    if fa is None or fh is None or fa == fh:
        return "needs both FSC-A and FSC-H"
    if float(params.get('tol', 0.25)) <= 0:
        return "tolerance is 0"
    return None


def _gaussian_diagnostic(df, params, channel_labels):
    from .gaussian_cleanup import parse_channels
    if not parse_channels(params.get('channels'), df.columns):
        return "no Gaussian parameter (Event_length, Center, Offset, Width, Residual)"
    return None


def _beads_diagnostic(df, params, channel_labels):
    want = [c.strip() for c in str(params.get('channels') or '').split(',')
            if c.strip()]
    if not want or any(c not in df.columns for c in want):
        return "a bead channel is missing from this sample"
    return None


_METHOD_DIAGNOSTICS = {
    'debris': _debris_diagnostic,
    'viability': _viability_diagnostic,
    'doublets': _doublets_diagnostic,
    'gaussian': _gaussian_diagnostic,
    'beads': _beads_diagnostic,
}


def autoclean_method_diagnostic(key, df, params, channel_labels=None):
    """A short, human reason a cleaning method removed **nothing** — or None
    when it's operating normally. Lets the GUI explain a silent 0-drop ("no
    viability dye detected", "FSC-A is unimodal — no debris mode") instead of
    leaving the user guessing whether the method is broken."""
    params = params or {}
    if len(df) == 0:
        return "no events"
    diagnose = _METHOD_DIAGNOSTICS.get(key)
    return diagnose(df, params, channel_labels) if diagnose else None


def autoclean_keep_mask(gate, df):
    """Boolean keep-mask for an 'autoclean' gate: the AND of every ENABLED
    method's per-sample keep-mask, recomputed from ``df``. A group with no
    enabled methods (or an empty df) is a no-op (all-True)."""
    n    = len(df)
    keep = np.ones(n, dtype=bool)
    if n == 0:
        return keep
    enabled = [m for m in (gate.get('methods') or []) if m.get('enabled', True)]
    if not enabled:
        return keep
    want = {m.get('key') for m in enabled}

    def _p(key):
        for m in enabled:
            if m.get('key') == key:
                return m.get('params', {}) or {}
        return {}

    # Time-binned (drift, flow-rate) + margin detectors share AcquisitionQC's
    # pd.cut binning and MAD logic — call it once with per-method flags.
    if want & {'drift', 'flow_rate', 'margin'}:
        dp, fp, mp = _p('drift'), _p('flow_rate'), _p('margin')
        qc  = AcquisitionQC(df.reset_index(drop=True))
        idx = qc.run(
            n_bins=int(dp.get('n_bins', fp.get('n_bins', 200))),
            threshold=float(dp.get('threshold', 5)),
            drift=('drift' in want),
            flow_rate=('flow_rate' in want),
            margins=('margin' in want),
            flow_rate_threshold=float(fp.get('flow_rate_threshold', 5.0)),
            margin_frac=float(mp.get('margin_frac', 0.01)))
        m = np.zeros(n, dtype=bool)
        m[np.asarray(idx, dtype=int)] = True   # idx are positions (reset index)
        keep &= m
    debris_keep = None
    if 'debris' in want:
        debris_keep = _autoclean_debris_mask(df, _p('debris'))
        keep &= debris_keep
    if 'viability' in want:
        keep &= _autoclean_viability_mask(df, _p('viability'))
    if 'doublets' in want:
        # Standard gating order removes debris first, so the singlet-ratio median
        # is centred on cell-sized events, not debris — pass the debris keep-mask
        # as the reference population when debris cleaning is also enabled.
        keep &= _autoclean_doublets_mask(df, _p('doublets'), ref_mask=debris_keep)
    # Mass cytometry: Gaussian discrimination parameters and bead events
    # (added by Tools > Normalize & clean > Mass cytometry; not in the default recipe).
    if 'gaussian' in want:
        from .gaussian_cleanup import gaussian_keep_mask
        keep &= gaussian_keep_mask(df, _p('gaussian'))
    if 'beads' in want:
        from .bead_normalize import bead_keep_mask
        keep &= bead_keep_mask(df, _p('beads'))
    return keep
