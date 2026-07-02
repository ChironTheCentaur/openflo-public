"""
flow_pipeline.py
----------------
Generalized flow cytometry analysis pipeline.

Features
--------
- Auto-reads FCS metadata (channels, labels, spillover)
- FlowJo .wsp compensation matrix reader
- Time-based QC (acquisition anomaly detection)
- FMO-based gate threshold calculation
- Logicle / log transform
- Phenograph clustering
- UMAP dimensionality reduction
- Sample concatenation with origin labeling
- Condition-level frequency comparison
- Statistics export (FlowJo Table-style)

Requirements
------------
    pip install flowio flowutils numpy pandas matplotlib seaborn \
                phenograph scikit-learn umap-learn scipy
"""

import contextlib
import copy
import functools
import logging
import os
import re
import uuid
import warnings
import xml.etree.ElementTree as ET
from typing import cast

import flowio
import numpy as np
import pandas as pd
from flowutils import transforms

# Heavy / optional deps are loaded on demand via the PEP-562 ``__getattr__``
# hook at the bottom of this module:
#   - ``phenograph``       — only needed by FlowSample.cluster()
#   - ``seaborn``          — only by heatmap-style plots
#   - ``gaussian_kde``     — only by density plot paths
#   - ``matplotlib.pyplot`` — only by the plot methods (~660 ms saved)
# Importing the module no longer pulls in igraph + scikit-learn's community
# detection OR matplotlib's Tk backend probing. Matters for the gate
# editor / compare tool / WSP-only callers — `import openflo.pipeline`
# drops from ~1050 ms to ~400 ms.
#
# Plot methods inside this module add a local `import matplotlib.pyplot
# as plt` at the top because PEP-562 __getattr__ only fires for OTHER
# modules accessing `pipeline.plt`; bare-name lookups inside this module
# follow normal scoping rules and would NameError without the local.

warnings.filterwarnings('ignore', category=FutureWarning)


# Module logger. Configure the root logger via the CLI (see openflo.cli)
# or programmatically with ``logging.basicConfig(level=logging.INFO)`` to
# see pipeline progress. Levels in use here:
#   DEBUG    — fine-grained per-event diagnostics (none currently)
#   INFO     — normal progress: load, QC, comp, transform, cluster, UMAP
#   WARNING  — recoverable issues that the pipeline routed around
#              (missing channels, GPU fallback, malformed gates, …)
#   ERROR    — only used by exception-raising code paths (rare; we mostly
#              raise OpenFloError subclasses instead)
log = logging.getLogger(__name__)


# ── Exception hierarchy ───────────────────────────────────────────────────────
# All recoverable errors raised by the pipeline are subclasses of
# OpenFloError. Top-level callers can ``except OpenFloError`` to surface a
# user-friendly message, then fall through to ``except Exception`` for
# unexpected bugs (which deserve a full traceback).

class OpenFloError(Exception):
    """Base class for every error raised intentionally by OpenFlo."""


class FcsParseError(OpenFloError):
    """Raised when an FCS file can't be read or its metadata is malformed."""


class CompensationError(OpenFloError):
    """Raised when a compensation matrix can't be parsed or applied."""


class WspParseError(OpenFloError):
    """Raised when a FlowJo .wsp can't be parsed."""


class GateError(OpenFloError):
    """Raised when a gate definition is invalid or can't be applied."""


class ClusteringError(OpenFloError):
    """Raised when clustering (Phenograph CPU or RAPIDS GPU) fails."""


# ── Constants ─────────────────────────────────────────────────────────────────

SCATTER_KEYWORDS = ['FSC', 'SSC', 'Time', 'time', 'Width', 'width']
EXCLUDE_CLUSTER  = ['Time', 'time',
                    # Derived / analysis columns — never treated as markers
                    # for clustering, stats, or channel classification.
                    'cluster', 'flowsom', 'flowsom_meta', 'cell_cycle',
                    'UMAP1', 'UMAP2', 'TRIMAP1', 'TRIMAP2',
                    'PACMAP1', 'PACMAP2']

# Phenograph (native Louvain backend) writes scratch files — kNN graph
# `*.bin`, dendrogram `*.tree`, and `*_graph.weights` — into the current
# working directory. Anchor them to a hidden cache folder next to this
# module so they don't litter the project root and so ProcessPoolExecutor
# workers (which inherit an arbitrary CWD) write to the same place.
_PHENOGRAPH_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '.phenograph_cache',
)

# Default categorical palette for per-sample / per-condition colouring on
# plots that group by `sample_origin` etc. `'auto'` picks tab10 for ≤10
# groups (well-separated hues) and falls back to tab20 / gist_ncar as the
# group count grows. Override via `set_default_palette('Set2')` or by
# passing `--palette` / the GUI palette dropdown to the pipeline.
_DEFAULT_CATEGORICAL_PALETTE = 'auto'


def set_default_palette(name):
    """Process-wide default for the categorical colour palette used by
    FlowSample.plot when colouring by a string column (sample_origin,
    condition, etc.). Common picks: 'auto', 'tab10', 'Set1', 'Set2',
    'Dark2', 'Paired'. Anything matplotlib's get_cmap accepts works."""
    global _DEFAULT_CATEGORICAL_PALETTE
    _DEFAULT_CATEGORICAL_PALETTE = str(name) or 'auto'


# ── Gate model & evaluator ────────────────────────────────────────────────────
#
# A gate is a JSON-friendly dict describing a region in event space. The same
# schema is consumed by the pipeline (this module) and authored by the GUI
# editor. Five kinds today:
#
#   {"kind": "threshold", "channel": str, "value": float}
#       1D one-sided:  x > value
#
#   {"kind": "interval",  "channel": str, "lo": float, "hi": float}
#       1D two-sided:  lo < x < hi
#
#   {"kind": "rect",      "x_channel": str, "y_channel": str,
#                          "x0": float, "x1": float, "y0": float, "y1": float}
#       2D axis-aligned rectangle.
#
#   {"kind": "polygon",   "x_channel": str, "y_channel": str,
#                          "vertices": [[x, y], ...]}
#       2D polygon (>=3 vertices). Membership via matplotlib.path.Path.
#
# Coordinates are in the SAME space as the data being gated (typically
# post-transform — logicle / log). FlowJo .wsp gates round-trip in this space
# already, matching apply_threshold_gates() semantics.

def gate_to_mask(gate, df, gates_by_id=None, _depth=0):
    """Evaluate one gate dict against a DataFrame. Returns a 1-D bool ndarray
    aligned to df. Missing/unknown channels yield an all-True (no-op) mask
    with a warning; missing/unknown kinds also no-op.

    `gates_by_id` is only needed for 'boolean' gates, whose operands are
    OTHER gates in the same sample (resolved via their cumulative masks);
    `_depth` guards against operand cycles."""
    kind = gate.get('kind')
    n    = len(df)
    if kind == 'threshold':
        ch = gate['channel']
        if ch not in df.columns:
            log.info(f"  [gate] threshold: channel '{ch}' not in data — skipped")
            return np.ones(n, dtype=bool)
        return np.asarray(df[ch].values > float(gate['value']))
    if kind == 'interval':
        ch = gate['channel']
        if ch not in df.columns:
            log.info(f"  [gate] interval: channel '{ch}' not in data — skipped")
            return np.ones(n, dtype=bool)
        vals = np.asarray(df[ch].values, dtype=float)
        return (vals > float(gate['lo'])) & (vals < float(gate['hi']))
    if kind == 'rect':
        xc, yc = gate['x_channel'], gate['y_channel']
        if xc not in df.columns or yc not in df.columns:
            log.info(f"  [gate] rect: channel(s) {xc!r}/{yc!r} missing — skipped")
            return np.ones(n, dtype=bool)
        xs = np.asarray(df[xc].values, dtype=float)
        ys = np.asarray(df[yc].values, dtype=float)
        return ((xs > float(gate['x0'])) & (xs < float(gate['x1'])) &
                (ys > float(gate['y0'])) & (ys < float(gate['y1'])))
    if kind == 'polygon':
        from matplotlib.path import Path as _MplPath
        xc, yc = gate['x_channel'], gate['y_channel']
        if xc not in df.columns or yc not in df.columns:
            log.info(f"  [gate] polygon: channel(s) {xc!r}/{yc!r} missing — skipped")
            return np.ones(n, dtype=bool)
        verts = np.asarray(gate['vertices'], dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 2 or len(verts) < 3:
            log.info(f"  [gate] polygon: malformed vertices (shape={verts.shape}) — skipped")
            return np.ones(n, dtype=bool)
        pts = np.column_stack([
            np.asarray(df[xc].values, dtype=float),
            np.asarray(df[yc].values, dtype=float),
        ])
        return _MplPath(verts).contains_points(pts)
    if kind == 'ellipsoid':
        # Gating-ML 2.0 EllipsoidGate: an event is inside when its
        # squared Mahalanobis distance from the mean is within
        # `distance_sq`:  (p-µ)ᵀ Σ⁻¹ (p-µ) ≤ distance_sq.
        xc, yc = gate['x_channel'], gate['y_channel']
        if xc not in df.columns or yc not in df.columns:
            log.info(f"  [gate] ellipsoid: channel(s) {xc!r}/{yc!r} missing — skipped")
            return np.ones(n, dtype=bool)
        mean = np.asarray(gate['mean'], dtype=float)
        cov  = np.asarray(gate['cov'], dtype=float)
        dist_sq = float(gate.get('distance_sq', 4.0))
        if mean.shape != (2,) or cov.shape != (2, 2):
            log.info("  [gate] ellipsoid: malformed mean/cov — skipped")
            return np.ones(n, dtype=bool)
        try:
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            log.info("  [gate] ellipsoid: singular covariance — skipped")
            return np.ones(n, dtype=bool)
        pts = np.column_stack([
            np.asarray(df[xc].values, dtype=float),
            np.asarray(df[yc].values, dtype=float),
        ])
        d = pts - mean
        # row-wise quadratic form (d @ inv * d).sum(axis=1)
        md_sq = np.einsum('ij,jk,ik->i', d, inv, d)
        return md_sq <= dist_sq
    if kind == 'cluster':
        # Membership in one clustering label. Unlike the geometric gates,
        # a missing column means the population is undefined for this
        # sample, so it selects NOTHING (empty) rather than no-op all-True
        # — an unclustered sample shouldn't masquerade as "all events".
        ch = gate.get('channel', 'cluster')
        if ch not in df.columns:
            log.info(f"  [gate] cluster: column '{ch}' not in data — empty")
            return np.zeros(n, dtype=bool)
        return np.asarray(df[ch].values == gate.get('cluster_id'))
    if kind == 'category':
        # Membership in a categorical label column (e.g. cell-cycle phase
        # in a 'cell_cycle' column). Like 'cluster', a missing column means
        # the population is undefined → selects nothing.
        ch = gate.get('channel')
        if not ch or ch not in df.columns:
            log.info(f"  [gate] category: column '{ch}' not in data — empty")
            return np.zeros(n, dtype=bool)
        return np.asarray(df[ch].values == gate.get('value'))
    if kind == 'boolean':
        # Combine OTHER gates' cumulative masks. op ∈ {and, or, not}; 'not'
        # negates the OR of its operands (so a single operand → plain NOT).
        op = gate.get('op', 'and')
        operands = gate.get('operands', []) or []
        if gates_by_id is None or not operands or _depth > 20:
            log.info("  [gate] boolean: unresolved operands — skipped")
            return np.ones(n, dtype=bool)
        masks = [cumulative_gate_mask(gates_by_id, gid, df, _depth + 1)
                 for gid in operands if gid in gates_by_id]
        if not masks:
            return np.ones(n, dtype=bool)
        out = masks[0].copy()
        if op == 'or':
            for m in masks[1:]:
                out |= m
            return out
        if op == 'not':
            for m in masks[1:]:
                out |= m
            return ~out
        for m in masks[1:]:            # default: and
            out &= m
        return out
    if kind == 'autoclean':
        # Recipe gate (no coordinates): the AND of every enabled cleaning
        # method, recomputed from THIS df. See autoclean_keep_mask.
        return autoclean_keep_mask(gate, df)
    if kind == 'group':
        # Pure organisational container (e.g. a 'Phenograph (N)' folder over
        # cluster populations): no geometry, never filters — children carry
        # the actual masks.
        return np.ones(n, dtype=bool)
    log.info(f"  [gate] unknown kind {kind!r} — skipped")
    return np.ones(n, dtype=bool)


# ── Auto-clean (acquisition-cleaning) gate ─────────────────────────────────
#
# An 'autoclean' gate stores a RECIPE — a list of cleaning METHODS — not
# coordinates. Each method recomputes its keep-mask from whatever sample the
# gate is evaluated against, so copying the gate to other samples re-runs the
# calculations rather than reusing one sample's geometry (bubbles/debris/clogs
# sit in different places per sample). The gate's mask is the AND of every
# ENABLED method's keep-mask: events clean of ALL selected anomaly types.

AUTOCLEAN_METHODS = [
    {'key': 'debris',    'label': 'Debris (size: beads → valley)', 'params': {'mode': 'bead', 'bead_um': 8.0, 'min_um': 4.0}},
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


def _autoclean_find_scatter(df, prefix, suffix='-A'):
    pu, su = prefix.upper(), suffix.upper()
    for c in df.columns:
        cu = c.upper()
        if cu.startswith(pu) and cu.endswith(su):
            return c
    for c in df.columns:
        if c.upper().startswith(pu):
            return c
    return None


# Dye name tokens we recognise as viability / live-dead stains (lowercased
# substrings, matched against antibody label first, then detector name).
# Dead cells take up the dye and read HIGH; live cells exclude it and read low.
# Overlaps with DNA_DYES (7-AAD, PI, DAPI, SYTOX, TO-PRO double as viability).
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
    'l/d' only match as whole words so they don't fire on 'PE' / 'APC' etc."""
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
      3. **2-D scatter gate** (fallback, ``mode='valley'`` or no bead anchor) —
         emulates the manual **FSC-A × SSC-A** debris polygon: an event is
         debris only when it is low on BOTH FSC-A AND SSC-A (the bottom-left
         cloud), each boundary being that channel's density valley below its
         median. This keeps low-FSC / high-SSC granular cells (granulocytes,
         etc.) that a 1-D FSC cut would wrongly discard. Degrades to a 1-D FSC
         valley when there's no usable SSC-A.
    No-op when there's no FSC-A column or no usable cutoff (so a missing bead
    reference degrades to the scatter gate, never to a wild cut)."""
    n = len(df)
    fsc = _autoclean_find_scatter(df, 'FSC', '-A')
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
            ssc = params.get('ssc_channel') or _autoclean_find_scatter(
                df, 'SSC', '-A')
            if ssc is not None and ssc in df.columns:
                svals = np.asarray(df[ssc].values, dtype=float)
                low_fsc = fin & (vals < float(manual))
                granular = (low_fsc & np.isfinite(svals)
                            & (svals >= float(gthr)))
                keep = keep | granular
        return keep
    # (2) bead-calibrated absolute size — deterministic, applies regardless of N.
    mode     = params.get('mode', 'bead')
    bead_fsc = params.get('bead_fsc')
    min_um   = params.get('min_um')
    bead_um  = params.get('bead_um', 8.0)
    if (mode == 'bead' and bead_fsc and min_um
            and float(bead_fsc) > 0 and float(bead_um) > 0):
        thr = float(min_um) * float(bead_fsc) / float(bead_um)
        return fin & (vals >= thr)
    # (3) 2-D scatter-gate fallback (needs enough events to estimate). Uses the
    # strict bimodal valley (not Otsu) so a UNIMODAL FSC-A is never bisected —
    # only a genuinely separate low-FSC debris mode below the median is cut.
    finite = vals[fin]
    if finite.size < 50:
        return np.ones(n, dtype=bool)
    fthr = _bimodal_valley(finite)
    if fthr is None or fthr >= float(np.median(finite)):
        return np.ones(n, dtype=bool)
    low_fsc = fin & (vals < float(fthr))
    ssc = _autoclean_find_scatter(df, 'SSC', '-A')
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


def _bimodal_valley(values, bins=256, smooth=2.0):
    """The valley between the two tallest peaks of a smoothed histogram, but
    ONLY when the data is genuinely bimodal (≥2 prominent peaks). Returns None
    for unimodal data — unlike :func:`auto_threshold`, it does NOT fall back to
    Otsu, so a caller can treat 'no valley' as 'do not split'."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 50:
        return None
    lo, hi = np.percentile(v, [0.5, 99.5])
    if hi <= lo:
        return None
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    sm = gaussian_filter1d(hist.astype(float), smooth)
    if sm.max() <= 0:
        return None
    peaks, _ = find_peaks(sm, prominence=sm.max() * 0.05)
    if peaks.size < 2:
        return None
    a, b = sorted(peaks[np.argsort(sm[peaks])[::-1]][:2])
    valley = a + int(np.argmin(sm[a:b + 1]))
    # Demand a genuine separation: the valley must dip to ≤ half the SHORTER
    # of the two peaks. Sampling-noise bumps on a single mode leave a shallow
    # 'valley' near the peak height — reject those as unimodal.
    if sm[valley] > 0.5 * float(min(sm[a], sm[b])):
        return None
    return float(centers[valley])


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


def _autoclean_doublets_mask(df, params):
    """Keep singlets via the FSC-A/FSC-H ratio (within ±tol of the median
    ratio). No-op if FSC-A or FSC-H is missing."""
    n   = len(df)
    tol = float(params.get('tol', 0.25))
    if tol <= 0:
        return np.ones(n, dtype=bool)
    fa = _autoclean_find_scatter(df, 'FSC', '-A')
    fh = _autoclean_find_scatter(df, 'FSC', '-H')
    if fa is None or fh is None or fa == fh:
        return np.ones(n, dtype=bool)
    a = np.asarray(df[fa].values, dtype=float)
    h = np.asarray(df[fh].values, dtype=float)
    ratio = np.where(h > 0, a / h, np.nan)
    valid = np.isfinite(ratio)
    if not valid.any():
        return np.ones(n, dtype=bool)
    med    = float(np.nanmedian(ratio[valid]))
    lo, hi = med * (1.0 - tol), med * (1.0 + tol)
    return valid & (ratio >= lo) & (ratio <= hi)


def autoclean_debris_threshold(df, params):
    """The scalar FSC-A keep-threshold the debris method resolves to on ``df``
    (manual ``min_fsc`` → bead absolute size → density valley), or None when it
    wouldn't cut. Used to FREEZE an auto cut to a fixed value when copying."""
    params = params or {}
    fsc = _autoclean_find_scatter(df, 'FSC', '-A')
    if fsc is None:
        return None
    manual = params.get('min_fsc')
    if manual is not None:
        return float(manual)
    mode = params.get('mode', 'bead')
    bead_fsc, min_um = params.get('bead_fsc'), params.get('min_um')
    bead_um = params.get('bead_um', 8.0)
    if (mode == 'bead' and bead_fsc and min_um
            and float(bead_fsc) > 0 and float(bead_um) > 0):
        return float(min_um) * float(bead_fsc) / float(bead_um)
    vals = np.asarray(df[fsc].values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size < 50:
        return None
    thr = _bimodal_valley(finite)
    if thr is None or thr >= float(np.median(finite)):
        return None
    return float(thr)


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
    out = {'min_fsc': float(thr)}
    # A manual floor or a bead absolute-size cut is a genuine 1-D threshold —
    # only the 2-D valley fallback carries an SSC granulocyte rescue to pin.
    if params.get('min_fsc') is not None:
        return out
    bead_fsc = params.get('bead_fsc')
    if (params.get('mode', 'bead') == 'bead' and bead_fsc
            and params.get('min_um') and float(bead_fsc) > 0):
        return out
    fsc = _autoclean_find_scatter(df, 'FSC', '-A')
    ssc = _autoclean_find_scatter(df, 'SSC', '-A')
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


def autoclean_method_diagnostic(key, df, params, channel_labels=None):
    """A short, human reason a cleaning method removed **nothing** — or None
    when it's operating normally. Lets the GUI explain a silent 0-drop ("no
    viability dye detected", "FSC-A is unimodal — no debris mode") instead of
    leaving the user guessing whether the method is broken."""
    params = params or {}
    if len(df) == 0:
        return "no events"

    if key == 'debris':
        fsc = _autoclean_find_scatter(df, 'FSC', '-A')
        if fsc is None:
            return "no FSC-A channel"
        if params.get('min_fsc') is not None:
            return "nothing below the manual min_fsc"
        mode = params.get('mode', 'bead')
        has_bead = bool(params.get('bead_fsc')) and bool(params.get('min_um'))
        if mode == 'bead' and has_bead:
            return None        # deterministic bead cut — if it dropped 0, that's real
        vals = np.asarray(df[fsc].values, dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size < 50:
            return "too few events"
        thr = _bimodal_valley(finite)
        if thr is None or thr >= float(np.median(finite)):
            msg = "FSC-A is unimodal — no low-debris mode to cut"
            if mode == 'bead' and not has_bead:
                msg += "; load size beads for an absolute-size cut"
            return msg
        return None

    if key == 'viability':
        ch = params.get('channel')
        if not ch or ch not in df.columns:
            ch = find_viability_channel(list(df.columns), channel_labels)
        if not ch or ch not in df.columns:
            return ("no viability dye detected — set the channel via right-click "
                    "(panel has none?)")
        vals = np.asarray(df[ch].values, dtype=float)
        finite = vals[np.isfinite(vals)]
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

    if key == 'doublets':
        fa = _autoclean_find_scatter(df, 'FSC', '-A')
        fh = _autoclean_find_scatter(df, 'FSC', '-H')
        if fa is None or fh is None or fa == fh:
            return "needs both FSC-A and FSC-H"
        if float(params.get('tol', 0.25)) <= 0:
            return "tolerance is 0"
        return None

    return None


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
    if 'debris' in want:
        keep &= _autoclean_debris_mask(df, _p('debris'))
    if 'viability' in want:
        keep &= _autoclean_viability_mask(df, _p('viability'))
    if 'doublets' in want:
        keep &= _autoclean_doublets_mask(df, _p('doublets'))
    return keep


# Categorical palette used by the GUI editor (and any other gate authors)
# to auto-assign a colour when a gate is created without one. Same order
# as the well-known "20 distinct colours" list (Sasha Trubetskoy 2017).
GATE_PALETTE = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
    '#008080', '#e6beff', '#9a6324', '#800000', '#aaffc3',
    '#808000', '#ffd8b1', '#000075', '#808080', '#000000',
]


def cumulative_gate_mask(gates_by_id, gid, df, _depth=0, overrides=None):
    """AND every gate's mask from `gid` up the parent chain to the root.
    Cycle-safe.

    The chain ALWAYS includes every ancestor regardless of each one's
    `enabled` flag: the toggle controls visibility ("draw this gate's
    highlight overlay") and pipeline inclusion separately, not whether a
    gate participates in defining the population at this node. So the
    cumulative meaning of leaf C inside parent P stays `P AND C` even
    when P's highlight is hidden.

    `gates_by_id` is a dict of gate_id -> gate_dict where each gate_dict
    may carry a 'parent_id' field naming another key (None = root).

    `overrides` (optional) maps gate_id -> a precomputed df-aligned bool mask
    used INSTEAD of evaluating that gate. The GUI uses it to inject cached
    auto-clean masks (whose recompute is expensive) so a chain that nests
    populations under an auto-clean root doesn't re-run the cleaning per node.
    """
    mask = np.ones(len(df), dtype=bool)
    seen = set()
    cur = gid
    while cur is not None and cur not in seen:
        seen.add(cur)
        g = gates_by_id.get(cur)
        if g is None:
            break
        if overrides is not None and cur in overrides:
            mask &= overrides[cur]
        else:
            mask &= gate_to_mask(g, df, gates_by_id, _depth)
        cur = g.get('parent_id')
    return mask


# ── FlowJo .wsp writer ────────────────────────────────────────────────────────
#
# Emits Gating-ML v2 XML that FlowJo v10 reads. Mirrors the structure WspReader
# expects (and that the real workspaces in our test set use), so a round-trip
# `extract_gates → write → extract_gates` preserves every gate.
#
# Usage:
#     w = WspWriter(cytometer='LSRFortessa')
#     w.set_compensation(['BV421-A', 'APC-A', 'PE-Cy7-A'], spillover_matrix)
#     w.add_sample('sample_1', '/path/to/sample_1.fcs', channels=[...],
#                  gates=[{'kind': 'polygon', 'id': 'g1', ...}, ...])
#     w.write('out.wsp')

_WSP_NS = {
    'gating':     'http://www.isac-net.org/std/Gating-ML/v2.0/gating',
    'transforms': 'http://www.isac-net.org/std/Gating-ML/v2.0/transformations',
    'data-type':  'http://www.isac-net.org/std/Gating-ML/v2.0/datatypes',
    'xsi':        'http://www.w3.org/2001/XMLSchema-instance',
}


def _q(prefix, local):
    """Build a Clark-notation tag/attr name (`{namespace}localname`) for ET."""
    return f'{{{_WSP_NS[prefix]}}}{local}'


class WspWriter:
    """Build a FlowJo-compatible .wsp workspace from in-memory gate dicts.

    Round-trips through WspReader: every gate kind we author (threshold,
    interval, rect, polygon) survives extract → write → extract.

    Out of scope: ellipsoid / quadrant-as-single-gate / boolean gates.
    Cells / event counts are emitted as 0 — FlowJo recomputes them.
    """

    def __init__(self, *, cytometer='Generic',
                 flowjo_version='OpenFlo-export-1.0'):
        self.cytometer       = cytometer
        self.flowjo_version  = flowjo_version
        self.samples         = []   # list of dicts (see add_sample)
        self.matrix          = None # (name, channels, np.ndarray)

    def set_compensation(self, channels, matrix, name='Acquisition-defined'):
        """Register a spillover matrix. `matrix` is an NxN numpy array
        whose rows are source channels in `channels` order, columns
        destination channels in the same order. Diagonals are usually 1.0."""
        m = np.asarray(matrix, dtype=float)
        if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] != len(channels):
            raise ValueError(
                f"matrix shape {m.shape} doesn't match {len(channels)} channels")
        self.matrix = (name, list(channels), m)

    def add_sample(self, name, fcs_path, channels, gates):
        """Register one sample's gating tree. `gates` is a list of gate
        dicts in the shared schema (see `gate_to_mask`) with `id` and
        `parent_id` fields resolved within the list (use
        `read_template_gates` if you have a .wsp / template to convert)."""
        self.samples.append({
            'name':     name,
            'fcs_path': fcs_path or '',
            'channels': list(channels),
            'gates':    list(gates),
        })

    def write(self, out_path):
        """Render the workspace and write it to `out_path`."""
        xml_str = self._build_xml()
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(xml_str)

    # ── XML construction (private) ────────────────────────────────────────

    def _build_xml(self):
        for prefix, uri in _WSP_NS.items():
            ET.register_namespace(prefix, uri)

        root = ET.Element('Workspace', {
            'version':       '20.0',
            'modDate':       _now_for_wsp(),
            'flowJoVersion': self.flowjo_version,
            _q('xsi', 'schemaLocation'): (
                f"{_WSP_NS['gating']} {_WSP_NS['gating']}/Gating-ML.v2.0.xsd "
                f"{_WSP_NS['transforms']} "
                f"{_WSP_NS['transforms']}/Transformations.v2.0.xsd "
                f"{_WSP_NS['data-type']} "
                f"{_WSP_NS['data-type']}/DataTypes.v2.0.xsd"),
        })

        # ── Matrices ──────────────────────────────────────────────────────
        matrices = ET.SubElement(root, 'Matrices')
        if self.matrix is not None:
            self._emit_matrix(matrices, *self.matrix)

        # ── Cytometers (minimal — FlowJo wants the element present) ───────
        cyts = ET.SubElement(root, 'Cytometers')
        ET.SubElement(cyts, 'Cytometer', {
            'name':          self.cytometer,
            'cyt':           self.cytometer,
            'transformType': 'BIEX',
            'manufacturer':  '',
            'serialnumber':  '',
        })

        # ── Groups (one default group, sample refs only) ──────────────────
        groups = ET.SubElement(root, 'Groups')
        grp_node = ET.SubElement(groups, 'GroupNode', {
            'name':         'All Samples',
            'owningGroup':  'All Samples',
            'expanded':     '1',
            'sortPriority': '10',
        })
        ET.SubElement(grp_node, 'Group', {'name': 'All Samples'})

        # ── SampleList ────────────────────────────────────────────────────
        sample_list = ET.SubElement(root, 'SampleList')
        for i, s in enumerate(self.samples, 1):
            sample = ET.SubElement(sample_list, 'Sample')
            if s['fcs_path']:
                uri = 'file:' + s['fcs_path'].replace('\\', '/')
                ET.SubElement(sample, 'DataSet', {
                    'uri':      uri,
                    'sampleID': str(i),
                })
            sn_attrs = {
                'name':         s['name'],
                'sampleID':     str(i),
                'owningGroup':  '',
                'expanded':     '1',
                'sortPriority': '10',
                'count':        '0',
            }
            sample_node = ET.SubElement(sample, 'SampleNode', sn_attrs)
            self._emit_gate_tree(sample_node, s['gates'])

        # Pretty-print (Python 3.9+).
        if hasattr(ET, 'indent'):
            ET.indent(root, space='  ')
        body = ET.tostring(root, encoding='unicode')
        return '<?xml version="1.0" encoding="UTF-8"?>\n' + body

    def _emit_matrix(self, parent, name, channels, matrix):
        sm = ET.SubElement(parent, _q('transforms', 'spilloverMatrix'), {
            'prefix':                'Comp-',
            'name':                  name,
            'editable':              '0',
            'color':                 '#c0c0c0',
            'version':               self.flowjo_version,
            'status':                'FINALIZED',
            _q('transforms', 'id'):  str(uuid.uuid4()),
            'suffix':                '',
        })
        params = ET.SubElement(sm, _q('data-type', 'parameters'))
        for ch in channels:
            ET.SubElement(params, _q('data-type', 'parameter'), {
                _q('data-type', 'name'):  ch,
                'userProvidedCompInfix':  f'Comp-{ch}',
            })
        for i, src in enumerate(channels):
            sp = ET.SubElement(sm, _q('transforms', 'spillover'), {
                _q('data-type', 'parameter'): src,
                'userProvidedCompInfix':      f'Comp-{src}',
            })
            for j, dst in enumerate(channels):
                ET.SubElement(sp, _q('transforms', 'coefficient'), {
                    _q('data-type', 'parameter'): dst,
                    _q('transforms', 'value'):    repr(float(matrix[i, j])),
                })

    @staticmethod
    def _collapse_quad_sets(gates):
        """Collapse each group of rect gates sharing a `quad_set` id into
        a single synthetic 'quadrant' gate (two dividers at the shared
        quad_origin). Re-parents any child of a collapsed rect onto the
        new quadrant gate. Gates without a quad_set pass through
        unchanged.

        This mirrors WspReader.parse_quadrant in reverse — the editor and
        the reader both represent a quadrant as 4 linked rects; FlowJo
        wants one QuadrantGate, so we fold at the write boundary.
        """
        groups = {}
        for g in gates:
            qs = g.get('quad_set')
            if qs and g.get('kind') == 'rect':
                groups.setdefault(qs, []).append(g)
        if not groups:
            return gates

        # Map every collapsed rect id → its quadrant's synthetic id so
        # children re-parent correctly.
        rect_to_quad = {}
        quad_gates = {}
        for qs, members in groups.items():
            first = members[0]
            quad_id = f'quad_{qs}'
            quad_gates[qs] = {
                'kind': 'quadrant',
                'id': quad_id,
                'parent_id': first.get('parent_id'),
                'x_channel': first.get('x_channel'),
                'y_channel': first.get('y_channel'),
                'quad_origin_x': first.get('quad_origin_x'),
                'quad_origin_y': first.get('quad_origin_y'),
                'name': 'Quadrants',
            }
            for m in members:
                if 'id' in m:
                    rect_to_quad[m['id']] = quad_id

        out = []
        emitted_quads = set()
        for g in gates:
            qs = g.get('quad_set')
            if qs and g.get('kind') == 'rect':
                if qs not in emitted_quads:
                    out.append(quad_gates[qs])
                    emitted_quads.add(qs)
                continue  # drop the individual rect
            # Re-parent any gate whose parent was a collapsed rect.
            g2 = dict(g)
            pid = g2.get('parent_id')
            if pid in rect_to_quad:
                g2['parent_id'] = rect_to_quad[pid]
            out.append(g2)
        return out

    def _emit_gate_tree(self, sample_node, gates):
        """Build <Subpopulations><Population><Gate>...</Gate><Subpopulations>...
        recursively from a flat list of gate dicts linked by parent_id."""
        if not gates:
            return
        gates = self._collapse_quad_sets(gates)
        # Tolerate gates that carry only the reader's `_import_id` (a direct
        # WspReader→WspWriter round-trip, with no GUI id-assignment in between):
        # synthesise an `id` from it so the hierarchy still builds. parent_id in
        # reader output references `_import_id`, so this stays consistent.
        for _i, g in enumerate(gates):
            if 'id' not in g:
                g['id'] = g.get('_import_id') or f'_g{_i}'
        by_id    = {g['id']: g for g in gates}
        children = {}
        for g in gates:
            children.setdefault(g.get('parent_id'), []).append(g['id'])
        roots = children.get(None, [])
        if not roots:
            return
        sub = ET.SubElement(sample_node, 'Subpopulations')
        for gid in roots:
            self._emit_population(sub, gid, by_id, children)

    def _emit_population(self, parent, gid, by_id, children):
        g = by_id[gid]
        pop_name = g.get('label') or g.get('name') or gid
        pop = ET.SubElement(parent, 'Population', {
            'name':         str(pop_name),
            'count':        '0',
            'owningGroup':  'All Samples',
            'expanded':     '1',
            'sortPriority': '10',
        })
        gate_wrap = ET.SubElement(pop, 'Gate', {
            _q('gating', 'id'): gid,
        })
        self._emit_gate_xml(gate_wrap, g)
        for child in children.get(gid, []):
            sub = pop.find('Subpopulations')
            if sub is None:
                sub = ET.SubElement(pop, 'Subpopulations')
            self._emit_population(sub, child, by_id, children)

    def _emit_gate_xml(self, gate_wrap, g):
        kind = g.get('kind')
        if kind == 'threshold':
            rect = ET.SubElement(gate_wrap, _q('gating', 'RectangleGate'))
            dim  = ET.SubElement(rect, _q('gating', 'dimension'), {
                _q('gating', 'min'): repr(float(g['value'])),
            })
            ET.SubElement(dim, _q('data-type', 'fcs-dimension'), {
                _q('data-type', 'name'): g['channel'],
            })
        elif kind == 'interval':
            rect = ET.SubElement(gate_wrap, _q('gating', 'RectangleGate'))
            dim  = ET.SubElement(rect, _q('gating', 'dimension'), {
                _q('gating', 'min'): repr(float(g['lo'])),
                _q('gating', 'max'): repr(float(g['hi'])),
            })
            ET.SubElement(dim, _q('data-type', 'fcs-dimension'), {
                _q('data-type', 'name'): g['channel'],
            })
        elif kind == 'rect':
            rect = ET.SubElement(gate_wrap, _q('gating', 'RectangleGate'))
            x0, x1 = sorted([float(g['x0']), float(g['x1'])])
            y0, y1 = sorted([float(g['y0']), float(g['y1'])])
            for ch, lo, hi in (
                    (g['x_channel'], x0, x1),
                    (g['y_channel'], y0, y1)):
                dim = ET.SubElement(rect, _q('gating', 'dimension'), {
                    _q('gating', 'min'): repr(lo),
                    _q('gating', 'max'): repr(hi),
                })
                ET.SubElement(dim, _q('data-type', 'fcs-dimension'), {
                    _q('data-type', 'name'): ch,
                })
        elif kind == 'polygon':
            poly = ET.SubElement(gate_wrap, _q('gating', 'PolygonGate'))
            for ch in (g['x_channel'], g['y_channel']):
                dim = ET.SubElement(poly, _q('gating', 'dimension'))
                ET.SubElement(dim, _q('data-type', 'fcs-dimension'), {
                    _q('data-type', 'name'): ch,
                })
            for vx, vy in g.get('vertices', []):
                vert = ET.SubElement(poly, _q('gating', 'vertex'))
                ET.SubElement(vert, _q('gating', 'coordinate'), {
                    _q('data-type', 'value'): repr(float(vx)),
                })
                ET.SubElement(vert, _q('gating', 'coordinate'), {
                    _q('data-type', 'value'): repr(float(vy)),
                })
        elif kind == 'ellipsoid':
            ell = ET.SubElement(gate_wrap, _q('gating', 'EllipsoidGate'))
            for ch in (g['x_channel'], g['y_channel']):
                dim = ET.SubElement(ell, _q('gating', 'dimension'))
                ET.SubElement(dim, _q('data-type', 'fcs-dimension'), {
                    _q('data-type', 'name'): ch,
                })
            mean = ET.SubElement(ell, _q('gating', 'mean'))
            for mv in g['mean']:
                ET.SubElement(mean, _q('gating', 'coordinate'), {
                    _q('data-type', 'value'): repr(float(mv)),
                })
            cov = ET.SubElement(ell, _q('gating', 'covarianceMatrix'))
            for row_vals in g['cov']:
                row = ET.SubElement(cov, _q('gating', 'row'))
                for entry_val in row_vals:
                    ET.SubElement(row, _q('gating', 'entry'), {
                        _q('data-type', 'value'): repr(float(entry_val)),
                    })
            ET.SubElement(ell, _q('gating', 'distanceSquare'), {
                _q('data-type', 'value'): repr(float(g.get('distance_sq', 4.0))),
            })
        elif kind == 'quadrant':
            # Emitted by _emit_gate_tree when it collapses a quad_set of
            # 4 rects. `g` here is a synthetic dict carrying the two
            # divider channels + values.
            quad = ET.SubElement(gate_wrap, _q('gating', 'QuadrantGate'))
            for ch, val in ((g['x_channel'], g['quad_origin_x']),
                            (g['y_channel'], g['quad_origin_y'])):
                div = ET.SubElement(quad, _q('gating', 'divider'))
                dim = ET.SubElement(div, _q('gating', 'dimension'))
                ET.SubElement(dim, _q('data-type', 'fcs-dimension'), {
                    _q('data-type', 'name'): ch,
                })
                v = ET.SubElement(div, _q('gating', 'value'))
                v.text = repr(float(val))
        else:
            log.info(f"[WspWriter] unknown gate kind {kind!r} — skipped")


def _now_for_wsp():
    """FlowJo's modDate format, eg 'Fri Mar 20 14:22:19 EDT 2026'."""
    import time
    return time.strftime('%a %b %d %H:%M:%S %Z %Y')


# ── Compensation matrix IO ────────────────────────────────────────────────────
#
# Polymorphic read / write so the GUI's matrix editor and the pipeline can
# treat all of these as a single concept:
#   .wsp   FlowJo workspace (any number of matrices; we return the first)
#   .csv / .tsv  Header row = destination channels, first column of each
#                data row = source channel. Optionally header-less (all cells
#                numeric); then `channels` is returned as None and the caller
#                supplies the channel names.
#   .fcs   Reads the $SPILL / $SPILLOVER FCS keyword (BD's standard).

def read_compensation_matrix(path):
    """Returns `(channels, matrix)`. `channels` is a list[str] or None
    (only None for header-less CSVs); `matrix` is an NxN numpy array, or
    ``(None, None)`` when the file is a legitimate FCS/WSP that simply
    has no spillover defined.

    Raises:
        CompensationError: malformed input (non-square matrix, channel /
            shape mismatch, unparseable SPILL keyword, unsupported
            extension).
    """
    p = path.lower()

    if p.endswith('.wsp'):
        reader = WspReader(path)
        m = reader.get_matrix()
        if m is None:
            return None, None
        return list(m['channels']), np.asarray(m['matrix'], dtype=float)

    if p.endswith(('.csv', '.tsv')):
        sep = '\t' if p.endswith('.tsv') else ','
        with open(path, encoding='utf-8') as f:
            # Don't .strip() the whole file: TSVs start with a leading
            # tab to align the header row above the row-label column,
            # and that tab is significant (an empty first cell ↔ "no
            # row label here"). Strip only newlines per-line via
            # splitlines(), and skip blank lines via rstrip().
            rows = [ln.split(sep) for ln in f.read().splitlines()
                    if ln.rstrip()]
        if not rows:
            return None, None
        # First cell is either a header label (string) or a number.
        first = rows[0][0].strip()
        try:
            float(first)
            has_header = False
        except ValueError:
            has_header = True
        if has_header:
            # Header row is dst channels; row labels are src channels.
            channels = [c.strip() for c in rows[0][1:] if c.strip()]
            n = len(channels)
            mat = []
            for r in rows[1:]:
                if len(r) < n + 1:
                    continue
                mat.append([float(c) for c in r[1:n + 1]])
            matrix = np.asarray(mat, dtype=float)
        else:
            matrix = np.asarray(
                [[float(c) for c in r] for r in rows], dtype=float)
            channels = None
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise CompensationError(
                f"matrix in {path} is not square (shape {matrix.shape})")
        if channels is not None and len(channels) != matrix.shape[0]:
            raise CompensationError(
                f"{path}: {len(channels)} channels vs {matrix.shape} matrix")
        return channels, matrix

    if p.endswith('.fcs'):
        # FCS $SPILL/$SPILLOVER: "N,ch1,ch2,...,chN,v11,v12,...,vNN"
        try:
            fcs = flowio.FlowData(path)
        except Exception as e:
            raise FcsParseError(f"could not read FCS {path}: {e}") from e
        spill = (fcs.text.get('SPILL')
                 or fcs.text.get('SPILLOVER')
                 or fcs.text.get('$SPILL')
                 or fcs.text.get('$SPILLOVER'))
        if not spill:
            return None, None
        parts = [p.strip() for p in spill.split(',')]
        try:
            n = int(parts[0])
        except ValueError as e:
            raise CompensationError(
                f"{path}: SPILL keyword header is not an integer: "
                f"{parts[0]!r}") from e
        if len(parts) < 1 + n + n * n:
            raise CompensationError(
                f"{path}: SPILL keyword promises {n} channels + "
                f"{n*n} values but only {len(parts)-1} entries provided")
        channels = parts[1:1 + n]
        try:
            vals = [float(x) for x in parts[1 + n: 1 + n + n * n]]
        except ValueError as e:
            raise CompensationError(
                f"{path}: non-numeric value in SPILL matrix") from e
        matrix = np.asarray(vals, dtype=float).reshape(n, n)
        return channels, matrix

    raise CompensationError(
        f"unsupported compensation file format: {path} "
        "(expected .wsp / .csv / .tsv / .fcs)")


def write_compensation_matrix(path, matrix, channels):
    """Write `matrix` (NxN numpy) labelled by `channels` (list[str]) to
    `path`. Format dispatched on extension:
      .wsp        -> WspWriter (matrix only; no samples / gates)
      .csv / .tsv -> header row + row-labelled rows
      .fcs        -> not supported (FCS spillover lives inside an FCS,
                     no use-case for writing one as a standalone file)
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square, got {matrix.shape}")
    if len(channels) != matrix.shape[0]:
        raise ValueError(
            f"{len(channels)} channels vs {matrix.shape} matrix")
    p = path.lower()
    if p.endswith('.wsp'):
        w = WspWriter()
        w.set_compensation(list(channels), matrix)
        w.write(path)
        return
    if p.endswith(('.csv', '.tsv')):
        sep = '\t' if p.endswith('.tsv') else ','
        with open(path, 'w', encoding='utf-8') as f:
            f.write(sep + sep.join(channels) + '\n')
            for i, ch in enumerate(channels):
                row = [ch] + [repr(float(matrix[i, j]))
                              for j in range(matrix.shape[1])]
                f.write(sep.join(row) + '\n')
        return
    raise ValueError(f"unsupported compensation file format for writing: {path}")


# ── Compensation matrix optimizer ─────────────────────────────────────────────
#
# Standard single-stain regression: for each source channel that has a
# dedicated bright single-stain control, fit a least-squares line through
# the brightest events relating destination signal to source signal. The
# slope IS the spillover coefficient. Diagonals stay 1.0.

def optimize_compensation(channels, single_stain_paths,
                          positive_percentile=95.0, min_events=100):
    """Estimate an NxN spillover matrix from per-channel single-stain controls.

    Parameters
    ----------
    channels : list[str]
        Fluor channel names in the order the matrix rows / columns will use.
    single_stain_paths : dict[str, str]
        Maps each source channel name to an FCS file in which only that
        fluor is bright. Channels with no entry contribute their identity
        row only.
    positive_percentile : float
        Cut-off for "bright positive" events on the source channel. Events
        above this percentile of the source signal are used in the regression.
    min_events : int
        Minimum number of positive events required to estimate a row.

    Returns (channels, matrix). Diagonal entries are always 1.0.
    """
    n = len(channels)
    matrix = np.eye(n, dtype=float)
    diag_report = {}
    for i, src_ch in enumerate(channels):
        path = single_stain_paths.get(src_ch)
        if not path:
            continue
        try:
            fcs = flowio.FlowData(path)
        except Exception as exc:
            log.info(f"[optimize] {src_ch}: load failed ({exc})")
            continue
        ch_names = [fcs.channels[k].get('pnn') or fcs.channels[k].get('PnN')
                    or f'Ch{k}' for k in range(1, fcs.channel_count + 1)]
        events = np.reshape(np.asarray(fcs.events),
                            (-1, fcs.channel_count)).astype(float)
        if src_ch not in ch_names:
            log.info(f"[optimize] {src_ch}: not present in {os.path.basename(path)}")
            continue
        src_idx = ch_names.index(src_ch)
        src_vals = events[:, src_idx]
        threshold = np.percentile(src_vals, positive_percentile)
        pos = src_vals > threshold
        n_pos = int(pos.sum())
        if n_pos < min_events:
            log.info(f"[optimize] {src_ch}: only {n_pos} positive events; skipped")
            continue
        diag_report[src_ch] = n_pos
        x = src_vals[pos]
        x_mean = x.mean()
        x_var = float(((x - x_mean) ** 2).sum())
        if x_var < 1e-9:
            continue
        for j, dst_ch in enumerate(channels):
            if j == i or dst_ch not in ch_names:
                continue
            y = events[pos, ch_names.index(dst_ch)]
            slope = float(((x - x_mean) * (y - y.mean())).sum() / x_var)
            matrix[i, j] = max(0.0, slope)   # clamp negatives to 0
    log.info(f"[optimize] used positive-event counts per source: {diag_report}")
    return list(channels), matrix


def read_template_gates(path):
    """Read gates from a v2 JSON template OR a FlowJo `.wsp` workspace.

    Returns `(gates, labels)` where:
      * `gates` is a list[dict] of gate definitions. Each entry has an
        `id` field (allocated `g1`, `g2`, … as needed) and a `parent_id`
        that references another id within the same list (or None for a
        root). Schema otherwise matches `gate_to_mask`.
      * `labels` is a {detector: antibody_label} dict from the template,
        or None if the source doesn't carry labels.

    Callers that want their own id namespace (the editor allocates ids
    per sample) can remap by walking the list in order and substituting
    parent_id pointers via an old->new map.
    """
    import json as _json
    p = path.lower()
    if p.endswith('.wsp'):
        reader = WspReader(path)
        raw = reader.extract_gates()
        # extract_gates assigns `_import_id` keys; rewrite to gN form so
        # the returned shape is consistent regardless of source.
        imp_to_id = {g.get('_import_id'): f'g{i}'
                     for i, g in enumerate(raw, 1)}
        gates = []
        for g in raw:
            d = dict(g)
            d.pop('_import_id', None)
            d['id'] = imp_to_id[g.get('_import_id')]
            pid = g.get('parent_id')
            d['parent_id'] = imp_to_id.get(pid) if pid else None
            gates.append(d)
        return gates, None

    if p.endswith('.json'):
        with open(path, encoding='utf-8') as f:
            data = _json.load(f)
        if not isinstance(data, dict):
            raise ValueError("template JSON must be an object")
        gates_field = data.get('gates')
        if not isinstance(gates_field, list):
            raise ValueError(
                "template JSON must have a 'gates' field of type list")
        labels = (data.get('labels')
                  if isinstance(data.get('labels'), dict) else None)
        out = [dict(g) for g in gates_field
               if isinstance(g, dict) and g.get('kind')]
        return out, labels

    raise ValueError(f"unsupported template format: {path}")


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
    set. The scale span [-1.0, 1.1] maps to data ~[-2.6e6 .. 7.4e5] — beyond any
    real flow value; the rare out-of-range event clamps to the nearest end."""
    inv = (transforms.logicle_inverse if method == 'logicle'
           else transforms.hyperlog_inverse)
    s = np.linspace(-1.0, 1.1, n)
    d = inv(s.reshape(-1, 1), channel_indices=[0],
            t=t, m=m, w=w, a=a).flatten()
    return d, s


def transform_values(values, method='logicle', t=262144, m=4.5, w=0.5, a=0,
                     cofactor=150.0):
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
    if method == 'logicle':
        from . import gpu_accel
        if gpu_accel.enabled():
            d, s = _biexp_lut('logicle', t, m, w, a)
            return gpu_accel.interp(v, d, s)
        return transforms.logicle(v.reshape(-1, 1), channel_indices=[0],
                                  t=t, m=m, w=w, a=a).flatten()
    if method == 'hyperlog':
        from . import gpu_accel
        if gpu_accel.enabled():
            d, s = _biexp_lut('hyperlog', t, m, w, a)
            return gpu_accel.interp(v, d, s)
        return transforms.hyperlog(v.reshape(-1, 1), channel_indices=[0],
                                   t=t, m=m, w=w, a=a).flatten()
    if method == 'asinh':
        from . import gpu_accel
        return gpu_accel.arcsinh(v, cofactor)
    if method == 'log':
        return np.where(v > 0, np.log10(np.clip(v, 1e-6, None)), 0.0)
    if method == 'linear':
        return v
    raise ValueError(f"Unknown transform method '{method}'.")


def inverse_transform_values(values, method='logicle', t=262144, m=4.5,
                             w=0.5, a=0, cofactor=150.0):
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
        return np.where(v > 0, np.power(10.0, v), 0.0)
    if method == 'linear':
        return v
    raise ValueError(f"Unknown transform method '{method}'.")


def describe_gate(gate):
    """Short human-readable label, for logs and the GUI gate list.
    ASCII-only so it survives cp1252 stdout when callers haven't
    reconfigured (e.g. ad-hoc scripts importing flow_pipeline)."""
    k = gate.get('kind')
    if k == 'threshold':
        return f"T  {gate['channel']} > {float(gate['value']):.3g}"
    if k == 'interval':
        return (f"I  {gate['channel']} in "
                f"[{float(gate['lo']):.3g}, {float(gate['hi']):.3g}]")
    if k == 'rect':
        nm = gate.get('name') or gate.get('label')
        pre = f"{nm}  " if nm else ""
        return (f"R  {pre}{gate['x_channel']} x {gate['y_channel']}  "
                f"[{float(gate['x0']):.3g},{float(gate['x1']):.3g}] x "
                f"[{float(gate['y0']):.3g},{float(gate['y1']):.3g}]")
    if k == 'polygon':
        nm = gate.get('name') or gate.get('label')
        pre = f"{nm}  " if nm else ""
        return (f"P  {pre}{gate['x_channel']} x {gate['y_channel']}  "
                f"({len(gate.get('vertices', []))} verts)")
    if k == 'ellipsoid':
        nm = gate.get('name') or gate.get('label')
        pre = f"{nm}  " if nm else ""
        return f"E  {pre}{gate.get('x_channel')} x {gate.get('y_channel')}"
    if k == 'cluster':
        nm = gate.get('name') or gate.get('label')
        return f"C  {nm}" if nm else f"C  cluster {gate.get('cluster_id')}"
    if k == 'category':
        nm = gate.get('name') or gate.get('label') or gate.get('value')
        return f"=  {nm}"
    if k == 'boolean':
        nm = gate.get('name')
        if nm:
            return f"B  {nm}"
        op = (gate.get('op') or 'and').upper()
        return f"B  {op}({len(gate.get('operands', []))})"
    if k == 'autoclean':
        nm = gate.get('name') or 'autocleaned sample'
        on = sum(1 for m in (gate.get('methods') or [])
                 if m.get('enabled', True))
        return f"AC  {nm}  ({on} on)"
    if k == 'group':
        return gate.get('name') or 'group'
    return f"?  {k}"


# ── Automated density-based gating ─────────────────────────────────────────────
#
# Lightweight "suggest a gate" helpers. They don't replace expert gating —
# they propose a sensible threshold or polygon from the data's density so the
# user can accept/adjust. Pure (numpy/scipy/contourpy), no Tk.

def _otsu_threshold(hist, centers):
    """Otsu's between-class-variance threshold on a 1-D histogram."""
    hist = np.asarray(hist, dtype=float)
    total = hist.sum()
    if total <= 0:
        return float(centers[len(centers) // 2])
    p = hist / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    k = int(np.nanargmax(sigma_b))
    return float(centers[k])


def auto_threshold(values, bins=256, smooth=2.0):
    """Suggest a 1-D split point for `values`.

    Bimodal data → the deepest valley between the two tallest peaks of the
    smoothed histogram. Unimodal data → Otsu's threshold. None when there's
    too little data. Returns a value in the data's own units."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 50:
        return None
    lo, hi = np.percentile(v, [0.5, 99.5])
    if hi <= lo:
        return None
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    sm = gaussian_filter1d(hist.astype(float), smooth)
    if sm.max() <= 0:
        return None
    peaks, _ = find_peaks(sm, prominence=sm.max() * 0.05)
    if peaks.size >= 2:
        a, b = sorted(peaks[np.argsort(sm[peaks])[::-1]][:2])
        valley = a + int(np.argmin(sm[a:b + 1]))
        return float(centers[valley])
    return _otsu_threshold(hist, centers)


def _polygon_area(verts):
    """Absolute shoelace area of an Nx2 polygon."""
    v = np.asarray(verts, dtype=float)
    if len(v) < 3:
        return 0.0
    x, y = v[:, 0], v[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) -
                          np.dot(y, np.roll(x, -1))))


def auto_polygon_gate(x, y, bins=128, level_frac=0.2, smooth=2.0,
                      max_verts=40):
    """Suggest a polygon around the dominant 2-D density mode of (x, y).

    Builds a smoothed 2-D histogram, then takes the density contour at
    `level_frac` of the peak density that encloses the global-max bin (the
    main population), simplified to at most `max_verts` vertices. Returns a
    list of ``[x, y]`` vertices in data coords, or None when it can't."""
    import contourpy
    from scipy.ndimage import gaussian_filter
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 50:
        return None
    xlo, xhi = np.percentile(x, [0.5, 99.5])
    ylo, yhi = np.percentile(y, [0.5, 99.5])
    if xhi <= xlo or yhi <= ylo:
        return None
    H, xe, ye = np.histogram2d(x, y, bins=bins, range=[[xlo, xhi], [ylo, yhi]])
    H = gaussian_filter(H, smooth)
    if H.max() <= 0:
        return None
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    # contourpy wants Z indexed [row=y, col=x] over a meshgrid.
    Z = H.T
    X, Y = np.meshgrid(xc, yc)
    peak_ix, peak_iy = np.unravel_index(int(np.argmax(H)), H.shape)
    peak_xy = (xc[peak_ix], yc[peak_iy])
    cg = contourpy.contour_generator(X, Y, Z)
    lines = cg.lines(level_frac * float(H.max()))
    polys = [np.asarray(p, dtype=float) for p in lines
             if p is not None and len(p) >= 3]
    if not polys:
        return None

    # Prefer the polygon that contains the density peak; among those (or all
    # if none contain it) take the largest by area.
    from matplotlib.path import Path as _MplPath
    containing = [p for p in polys if _MplPath(p).contains_point(peak_xy)]
    pool = containing or polys
    best = max(pool, key=_polygon_area)
    if len(best) > max_verts:
        idx = np.linspace(0, len(best) - 1, max_verts).astype(int)
        best = best[idx]
    return [[float(a), float(b)] for a, b in best]


def auto_singlet_gate(area, height, k=3.0, h_pct=(0.1, 99.9)):
    """Singlet-discrimination gate from an area channel (e.g. FSC-A, ``x``)
    vs its height channel (FSC-H, ``y``).

    Singlets satisfy ``area ≈ slope·height`` — a tight diagonal — while
    doublets / aggregates carry more area per unit height and sit *above* it.
    Keeps events whose ``area/height`` ratio lies within a robust band around
    the population median (``median ± k·1.4826·MAD``), returned as a polygon
    in ``(area, height)`` data coordinates (x = area, y = height) so it drops
    straight into a 'polygon' gate.

    Returns ``(vertices, quality)`` or ``(None, None)`` when undefined.
    ``quality = {'frac_kept', 'ratio_cv', 'slope'}``: a clean singlet gate
    keeps ~85–99 % of events with a low ratio CV; a high CV or low keep
    fraction is the signal NOT to trust it blindly.
    """
    area = np.asarray(area, dtype=float)
    height = np.asarray(height, dtype=float)
    m = np.isfinite(area) & np.isfinite(height) & (height > 0)
    a, h = area[m], height[m]
    if a.size < 50:
        return None, None
    ratio = a / h
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size < 50:
        return None, None
    med = float(np.median(ratio))
    if not np.isfinite(med) or med <= 0:
        return None, None
    mad = float(np.median(np.abs(ratio - med)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(ratio))
    if sigma <= 0:
        return None, None
    r_lo = max(med - k * sigma, 1e-9)
    r_hi = med + k * sigma
    h_lo, h_hi = np.percentile(h, list(h_pct))
    if h_hi <= h_lo:
        return None, None
    # Band between the two rays area = r_lo·height and area = r_hi·height,
    # clipped to the populated height range — a quadrilateral (x = area).
    verts = [[r_lo * h_lo, h_lo],
             [r_lo * h_hi, h_hi],
             [r_hi * h_hi, h_hi],
             [r_hi * h_lo, h_lo]]
    a_ratio = a / h
    inside = ((a_ratio >= r_lo) & (a_ratio <= r_hi)
              & (h >= h_lo) & (h <= h_hi))
    quality = {'frac_kept': float(inside.mean()),
               'ratio_cv': float(sigma / med),
               'slope': med}
    return verts, quality


def gmm_ellipse_gates(x, y, max_components=6, min_weight=0.02,
                      coverage=0.90, max_events=20_000, seed=42):
    """Decompose a 2-D ``(x, y)`` distribution into Gaussian populations and
    return one ellipsoid gate per component — a principled replacement for a
    single arbitrary density contour.

    Fits ``sklearn.mixture.GaussianMixture`` (full covariance) for
    ``k = 1..max_components`` on STANDARDIZED coordinates and selects ``k`` by
    minimum BIC. Each retained component (mixing weight ≥ ``min_weight``)
    becomes an ellipsoid gate ``{mean, cov, distance_sq}`` in the ORIGINAL
    data coordinates, with ``distance_sq`` set to the chi-square quantile
    (df = 2) so the ellipse encloses ``coverage`` of a bivariate-normal
    component (e.g. coverage 0.90 → distance_sq ≈ 4.605).

    Returns a list of ``(gate, info)`` tuples sorted by descending weight,
    where ``gate`` is a partial ellipsoid-gate dict WITHOUT
    ``x_channel`` / ``y_channel`` (the caller fills those in), and
    ``info = {'weight', 'n_events', 'separation', 'n_components'}``.
    ``separation`` is the Mahalanobis distance (component metric) to the
    nearest other component mean — larger means a cleaner, more trustworthy
    split (< ~2 means the components overlap heavily). Empty list when the
    fit is undefined.
    """
    from scipy.stats import chi2
    from sklearn.mixture import GaussianMixture
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 100:
        return []
    P = np.column_stack([x, y])
    rng = np.random.default_rng(seed)
    if len(P) > max_events:
        P = P[rng.choice(len(P), max_events, replace=False)]
    # Standardize so the GMM fit is scale-invariant and numerically stable
    # (FSC counts and a logicle marker differ by orders of magnitude).
    mu = P.mean(0)
    sd = P.std(0)
    sd[sd == 0] = 1.0
    Z = (P - mu) / sd

    best = None
    kmax = min(int(max_components), len(Z))
    for k in range(1, max(1, kmax) + 1):
        gm = GaussianMixture(n_components=k, covariance_type='full',
                             random_state=seed, reg_covar=1e-6, n_init=1)
        try:
            gm.fit(Z)
            bic = float(gm.bic(Z))
        except Exception:
            continue
        if best is None or bic < best[0]:
            best = (bic, gm)
    if best is None:
        return []
    gm = best[1]
    dist_sq = float(chi2.ppf(coverage, df=2))
    labels = gm.predict(Z)
    # Local ndarray copies of the fitted parameters (sklearn types them
    # Optional; they're always populated post-fit).
    weights = np.asarray(gm.weights_, dtype=float)
    means = np.asarray(gm.means_, dtype=float)
    covs = np.asarray(gm.covariances_, dtype=float)
    n_comp = int(gm.n_components)
    S = np.diag(sd)   # cov back-transform: Cov(P) = S · Cov(Z) · S
    out = []
    for i in range(n_comp):
        w = float(weights[i])
        if w < min_weight:
            continue
        mean_z = means[i]
        cov_z = covs[i]
        mean = mean_z * sd + mu
        cov = S @ cov_z @ S
        # Separation: nearest other component mean in THIS component's metric.
        sep = np.inf
        try:
            inv_z = np.linalg.inv(cov_z)
            for j in range(n_comp):
                if j == i:
                    continue
                d = mean_z - means[j]
                sep = min(sep, float(np.sqrt(max(d @ inv_z @ d, 0.0))))
        except np.linalg.LinAlgError:
            sep = np.inf
        gate = {'kind': 'ellipsoid',
                'mean': [float(mean[0]), float(mean[1])],
                'cov': [[float(cov[0, 0]), float(cov[0, 1])],
                        [float(cov[1, 0]), float(cov[1, 1])]],
                'distance_sq': dist_sq}
        info = {'weight': w,
                'n_events': int(np.sum(labels == i)),
                'separation': (None if not np.isfinite(sep) else float(sep)),
                'n_components': n_comp}
        out.append((gate, info))
    out.sort(key=lambda t: t[1]['weight'], reverse=True)
    return out


# ── FlowSOM (self-organizing map + metaclustering) ────────────────────────────
#
# A compact, dependency-free FlowSOM: train a rectangular SOM over the marker
# space, assign each event to its best-matching unit (node), then agglomerate
# the node prototypes into a handful of metaclusters. Pure numpy + sklearn
# (already a dependency). Not as tuned as the R FlowSOM, but the same shape:
# nodes capture fine structure, metaclusters give interpretable populations.

def _som_train(X, grid=(10, 10), iters=10, max_events=50_000, seed=42):
    """Train a rectangular SOM. Returns (weights[n_nodes, d],
    coords[n_nodes, 2]). Online updates over a (sub-sampled) event stream
    with linearly decaying neighbourhood + learning rate."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    if len(X) > max_events:
        X = X[rng.choice(len(X), max_events, replace=False)]
    gx, gy = grid
    n_nodes = gx * gy
    init = rng.choice(len(X), n_nodes, replace=(len(X) < n_nodes))
    W = X[init].astype(float).copy()
    coords = np.array([(i // gy, i % gy) for i in range(n_nodes)], dtype=float)
    sigma0 = max(gx, gy) / 2.0
    lr0 = 0.5
    for t in range(iters):
        frac = t / max(1, iters)
        sigma = sigma0 * (1.0 - frac) + 0.5
        lr = lr0 * (1.0 - frac)
        two_sig2 = 2.0 * sigma * sigma
        for i in rng.permutation(len(X)):
            x = X[i]
            bmu = int(np.argmin(((W - x) ** 2).sum(1)))
            gd = ((coords - coords[bmu]) ** 2).sum(1)
            h = np.exp(-gd / two_sig2)
            W += (lr * h)[:, None] * (x - W)
    return W, coords


def _som_assign(X, W):
    """Best-matching-unit node index for each row of X (chunked)."""
    X = np.asarray(X, dtype=float)
    out = np.empty(len(X), dtype=int)
    step = 10_000
    for s in range(0, len(X), step):
        chunk = X[s:s + step]
        # squared distances to every node: (chunk·node) expansion
        d = (np.sum(chunk ** 2, 1)[:, None]
             - 2.0 * chunk @ W.T
             + np.sum(W ** 2, 1)[None, :])
        out[s:s + step] = np.argmin(d, axis=1)
    return out


def _som_metacluster(W, n_metaclusters):
    """Agglomerate node prototypes into `n_metaclusters` labels (one per
    node). Falls back to one cluster when there are too few nodes."""
    from sklearn.cluster import AgglomerativeClustering
    k = max(1, min(int(n_metaclusters), len(W)))
    if k == 1:
        return np.zeros(len(W), dtype=int)
    return AgglomerativeClustering(n_clusters=k).fit_predict(W)


def flowsom_mst(weights):
    """Minimal-spanning-tree edges over FlowSOM node prototypes — the backbone
    of the classic FlowSOM star-tree plot.

    ``weights`` : ``(n_nodes, n_markers)`` SOM prototypes (``flowsom_result
    ['weights']``). Returns ``(edges, dist)`` where ``edges`` is a list of
    ``(i, j)`` node-index pairs (``n_nodes − 1`` of them, forming a tree) and
    ``dist`` is the full pairwise Euclidean distance matrix (so the caller can
    weight or lay out the tree)."""
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import pdist, squareform
    W = np.asarray(weights, dtype=float)
    n = len(W)
    if n < 2:
        return [], np.zeros((n, n))
    dist = squareform(pdist(W))
    mst = minimum_spanning_tree(dist).toarray()
    i, j = np.nonzero(mst)
    edges = list(zip(i.tolist(), j.tolist(), strict=True))
    return edges, dist


def flowsom_layout(n_nodes, edges, seed=42):
    """2-D layout for the FlowSOM MST. Uses igraph's Fruchterman-Reingold on
    the tree; falls back to a circle if igraph isn't available. Returns an
    ``(n_nodes, 2)`` float array of positions."""
    if n_nodes == 0:
        return np.zeros((0, 2))
    try:
        import igraph as ig
        g = ig.Graph(n=int(n_nodes), edges=[(int(a), int(b)) for a, b in edges])
        lay = g.layout_fruchterman_reingold(niter=500, seed=None)
        return np.asarray(lay.coords, dtype=float)
    except Exception:
        ang = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        return np.column_stack([np.cos(ang), np.sin(ang)])


# ── CytoNorm batch normalization ──────────────────────────────────────────────
#
# CytoNorm (Van Gassen 2020; CytoNorm 2.0, Quintelier 2025): the standard for
# removing technical batch/acquisition variation in cytometry. FlowSOM-cluster
# the pooled data into metaclusters, then for each metacluster and channel,
# quantile-normalize every batch's intensity distribution onto a shared GOAL
# distribution via a monotone (PCHIP) spline. Population-aware, so it doesn't
# smear distinct populations together the way a global quantile norm can.
#
# Two modes (same engine — only the events you fit on differ):
#   • 'goal'     (CytoNorm 2.0, default): fit on ALL samples; goal = pooled
#                aggregate. No dedicated control samples required.
#   • 'controls' (classic CytoNorm): fit on per-batch CONTROL samples; the
#                fitted per-batch transform is then applied to every sample in
#                that batch. Most rigorous when you ran a shared control aliquot
#                in each batch.
# The fitted model serializes (to_dict/from_dict) so it can be applied to new
# samples later — which cyCombine-style methods can't do.

def _strict_increasing(x):
    """Nudge ties so `x` is strictly increasing (PCHIP needs strictly
    increasing knots). Tiny relative epsilon, preserves order/scale."""
    x = np.asarray(x, dtype=float).copy()
    for i in range(1, len(x)):
        if x[i] <= x[i - 1]:
            x[i] = x[i - 1] + 1e-9 * (abs(x[i - 1]) + 1.0)
    return x


class CytoNorm:
    """FlowSOM + per-metacluster per-channel quantile normalization.

    Usage::

        cn = CytoNorm(channels=fluor_markers, mode='goal')
        cn.fit({batch_id: events_df, ...})       # events_df = a sample (or pooled)
        corrected = cn.apply(sample.data, batch_id)
        report = cn.qc({batch_id: events_df, ...})

    ``events_by_batch`` maps a batch label to that batch's events (a DataFrame
    carrying ``channels``, or an ndarray in channel order). In 'goal' mode pass
    all samples per batch (concatenate per batch); in 'controls' mode pass the
    control sample per batch, then ``apply`` to every sample of that batch.
    """

    def __init__(self, channels, n_metaclusters=10, grid=(10, 10),
                 n_quantiles=101, min_cell_events=50, mode='goal', seed=42):
        self.channels = [str(c) for c in channels]
        self.n_metaclusters = int(n_metaclusters)
        self.grid = (int(grid[0]), int(grid[1]))
        self.n_quantiles = int(n_quantiles)
        self.min_cell_events = int(min_cell_events)
        self.mode = str(mode)
        self.seed = int(seed)
        self.batches = []
        self._mean = None
        self._std = None
        self._W = None
        self._node_meta = None
        self._qs = None
        self._goal_q = {}        # (meta, ch_idx) -> goal quantile array | None
        self._batch_q = {}       # (meta, batch, ch_idx) -> quantile array | None

    # -- helpers --
    def _events(self, ev):
        if hasattr(ev, 'columns'):
            return ev[self.channels].to_numpy(dtype=float)
        return np.asarray(ev, dtype=float)

    def _z(self, X):
        return (X - self._mean) / self._std

    def _meta_of(self, X):
        assert self._node_meta is not None and self._W is not None
        return self._node_meta[_som_assign(self._z(X), self._W)]

    # -- fit / apply / qc --
    def fit(self, events_by_batch):
        self.batches = [str(b) for b in events_by_batch]
        pools = []
        for ev in events_by_batch.values():
            X = self._events(ev)
            X = X[np.isfinite(X).all(1)]
            if len(X):
                pools.append(X)
        if not pools:
            raise ValueError("CytoNorm.fit: no finite events to fit on.")
        pooled = np.vstack(pools)
        self._mean = pooled.mean(0)
        self._std = pooled.std(0)
        self._std[self._std == 0] = 1.0
        self._W, _coords = _som_train(self._z(pooled), self.grid, seed=self.seed)
        self._node_meta = _som_metacluster(self._W, self.n_metaclusters)
        self._qs = np.linspace(0.0, 1.0, self.n_quantiles)

        # Goal quantiles per (metacluster, channel) — the pooled aggregate.
        pooled_meta = self._meta_of(pooled)
        for m in np.unique(self._node_meta):
            sub = pooled[pooled_meta == m]
            for j in range(len(self.channels)):
                col = sub[:, j][np.isfinite(sub[:, j])]
                self._goal_q[(int(m), j)] = (
                    np.quantile(col, self._qs)
                    if col.size >= self.min_cell_events else None)

        # Per-batch quantiles per (metacluster, channel).
        for b, ev in events_by_batch.items():
            X = self._events(ev)
            X = X[np.isfinite(X).all(1)]
            bmeta = self._meta_of(X) if len(X) else np.array([], dtype=int)
            for m in np.unique(self._node_meta):
                sub = X[bmeta == m] if len(X) else X
                for j in range(len(self.channels)):
                    col = sub[:, j][np.isfinite(sub[:, j])] if len(sub) else sub
                    self._batch_q[(int(m), str(b), j)] = (
                        np.quantile(col, self._qs)
                        if col.size >= self.min_cell_events else None)
        return self

    def apply(self, df, batch_id):
        """Return a corrected copy of ``df`` for ``batch_id``. Rows/channels
        without a usable transform pass through unchanged."""
        from scipy.interpolate import PchipInterpolator
        out = df.copy()
        if self._W is None or not all(c in out.columns for c in self.channels):
            return out
        X = out[self.channels].to_numpy(dtype=float)
        finite = np.isfinite(X).all(1)
        meta = np.full(len(X), -1, dtype=int)
        if finite.any():
            meta[finite] = self._meta_of(X[finite])
        bkey = str(batch_id)
        for j, ch in enumerate(self.channels):
            vals = out[ch].to_numpy(dtype=float).copy()
            for m in np.unique(meta[meta >= 0]):
                bq = self._batch_q.get((int(m), bkey, j))
                gq = self._goal_q.get((int(m), j))
                if bq is None or gq is None:
                    continue
                sel = (meta == m) & np.isfinite(vals)
                if not sel.any():
                    continue
                xq = _strict_increasing(bq)
                v = np.clip(vals[sel], xq[0], xq[-1])
                nv = PchipInterpolator(xq, gq, extrapolate=False)(v)
                vals[sel] = np.where(np.isfinite(nv), nv, vals[sel])
            out[ch] = vals
        return out

    def qc(self, events_by_batch):
        """Per-channel mean Wasserstein distance batch→goal, before vs after.
        ``{channel: {'before': x, 'after': y}}`` — lower 'after' = better."""
        import pandas as pd
        from scipy.stats import wasserstein_distance
        pooled = np.vstack([self._events(ev) for ev in events_by_batch.values()])
        pooled = pooled[np.isfinite(pooled).all(1)]
        res = {}
        for j, ch in enumerate(self.channels):
            goal = pooled[:, j]
            before, after = [], []
            for b, ev in events_by_batch.items():
                X = self._events(ev)
                col = X[:, j][np.isfinite(X[:, j])]
                if col.size:
                    before.append(wasserstein_distance(col, goal))
                cor = self.apply(pd.DataFrame(X, columns=pd.Index(self.channels)), b)
                cc = cor[ch].to_numpy(dtype=float)
                cc = cc[np.isfinite(cc)]
                if cc.size:
                    after.append(wasserstein_distance(cc, goal))
            res[ch] = {'before': float(np.mean(before)) if before else 0.0,
                       'after': float(np.mean(after)) if after else 0.0}
        return res

    # -- serialization --
    def to_dict(self):
        def qmap(d):
            return {f'{k[0]}|{k[1]}' if len(k) == 2 else f'{k[0]}|{k[1]}|{k[2]}':
                    (None if v is None else [float(x) for x in v])
                    for k, v in d.items()}
        return {
            'format': 'openflo-cytonorm', 'version': 1,
            'channels': self.channels, 'n_metaclusters': self.n_metaclusters,
            'grid': list(self.grid), 'n_quantiles': self.n_quantiles,
            'min_cell_events': self.min_cell_events, 'mode': self.mode,
            'seed': self.seed, 'batches': self.batches,
            'mean': None if self._mean is None else self._mean.tolist(),
            'std': None if self._std is None else self._std.tolist(),
            'W': None if self._W is None else self._W.tolist(),
            'node_meta': None if self._node_meta is None
            else [int(x) for x in self._node_meta],
            'qs': None if self._qs is None else self._qs.tolist(),
            'goal_q': qmap(self._goal_q),
            'batch_q': qmap(self._batch_q),
        }

    @classmethod
    def from_dict(cls, d):
        cn = cls(d['channels'], d.get('n_metaclusters', 10),
                 tuple(d.get('grid', (10, 10))), d.get('n_quantiles', 101),
                 d.get('min_cell_events', 50), d.get('mode', 'goal'),
                 d.get('seed', 42))
        cn.batches = list(d.get('batches', []))
        cn._mean = None if d.get('mean') is None else np.asarray(d['mean'])
        cn._std = None if d.get('std') is None else np.asarray(d['std'])
        cn._W = None if d.get('W') is None else np.asarray(d['W'])
        cn._node_meta = (None if d.get('node_meta') is None
                         else np.asarray(d['node_meta'], dtype=int))
        cn._qs = None if d.get('qs') is None else np.asarray(d['qs'])

        def unqmap(m):
            out = {}
            for k, v in (m or {}).items():
                parts = k.split('|')
                key = ((int(parts[0]), int(parts[1])) if len(parts) == 2
                       else (int(parts[0]), parts[1], int(parts[2])))
                out[key] = None if v is None else np.asarray(v, dtype=float)
            return out
        cn._goal_q = unqmap(d.get('goal_q'))
        cn._batch_q = unqmap(d.get('batch_q'))
        return cn


# ── GPU probe ─────────────────────────────────────────────────────────────────

def _probe_gpu():
    """Return (available, display_name, CumlUMAP_class, cluster_kit).

    `cluster_kit` is a dict {cunn, cugraph, cupy, cudf} if the RAPIDS pieces
    needed for GPU clustering are present, else None. cuML UMAP can work
    standalone, but clustering needs the full RAPIDS stack.
    """
    try:
        from cuml.manifold import UMAP as _CU  # type: ignore[import-not-found]
        name = 'GPU'
        try:
            import pynvml  # type: ignore[import-not-found]
            pynvml.nvmlInit()
            h   = pynvml.nvmlDeviceGetHandleByIndex(0)
            raw = pynvml.nvmlDeviceGetName(h)
            name = raw.decode() if isinstance(raw, bytes) else raw
        except Exception:
            pass

        cluster_kit = None
        try:
            import cudf as _cudf  # type: ignore[import-not-found]
            import cugraph as _cugraph  # type: ignore[import-not-found]
            import cupy as _cupy  # type: ignore[import-not-found]
            from cuml.neighbors import NearestNeighbors as _CuNN  # type: ignore[import-not-found]
            cluster_kit = {
                'cunn':    _CuNN,
                'cugraph': _cugraph,
                'cupy':    _cupy,
                'cudf':    _cudf,
            }
        except ImportError:
            pass

        return True, name, _CU, cluster_kit
    except ImportError:
        return False, '', None, None

GPU_AVAILABLE, GPU_NAME, _CumlUMAP, _GPU_CLUSTER_KIT = _probe_gpu()
GPU_CLUSTERING_AVAILABLE = _GPU_CLUSTER_KIT is not None


def _vram_free_gb():
    """Free VRAM in GB via pynvml → nvidia-smi fallback. None if no GPU."""
    try:
        import pynvml  # type: ignore[import-not-found]
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(h).free / (1024 ** 3)
    except Exception:
        pass
    try:
        import subprocess
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=4,
            creationflags=flags)
        if out.returncode == 0:
            return float(out.stdout.strip().splitlines()[0]) / 1024.0
    except Exception:
        pass
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_scatter(name):
    return any(k in name for k in SCATTER_KEYWORDS)

def _is_excluded(name):
    return any(k in name for k in EXCLUDE_CLUSTER)


# ══════════════════════════════════════════════════════════════════════════════
# WSP COMPENSATION READER
# ══════════════════════════════════════════════════════════════════════════════

class WspReader:
    """
    Parse compensation matrices from a FlowJo v10 .wsp workspace file.

    Usage
    -----
        reader = WspReader('experiment.wsp')
        reader.print_matrices()
        m = reader.get_matrix()           # first matrix
        m = reader.get_matrix('My Comp') # by name
        # m keys: matrix (ndarray), channels (list), prefix, suffix
    """

    def __init__(self, wsp_path):
        self.path     = wsp_path
        self.matrices = {}
        self.root     = None        # populated by _parse; needed for gate extraction
        self._parse()

    def _parse(self):
        try:
            tree = ET.parse(self.path)
        except FileNotFoundError as e:
            raise WspParseError(f"WSP file not found: {self.path}") from e
        except ET.ParseError as e:
            raise WspParseError(
                f"WSP file is not valid XML ({self.path}): {e}") from e
        root = tree.getroot()
        # Strip XML namespace from BOTH element tags AND attribute keys.
        # FlowJo v10's Gating-ML v2 .wsp files namespace attributes too
        # (gating:min, data-type:name, data-type:value, ...), and
        # ElementTree expands those to {namespace_uri}localname. Without
        # this attribute-side strip, our find/get calls all miss and the
        # reader silently extracts zero matrices and zero gates.
        ns_re = re.compile(r'\{.*?\}')
        for elem in root.iter():
            elem.tag = ns_re.sub('', elem.tag)
            if elem.attrib:
                stripped = {ns_re.sub('', k): v for k, v in elem.attrib.items()}
                elem.attrib.clear()
                elem.attrib.update(stripped)
        self.root = root
        for tag in ('CompensationMatrix', 'spilloverMatrix', 'Compensation'):
            for node in root.iter(tag):
                self._extract_matrix(node)
        if not self.matrices:
            log.info("[WspReader] No compensation matrices found.")

    @staticmethod
    def _channel_name(dim_elem):
        """Pull the FCS channel name from a gating <dimension>, stripping
        FlowJo's 'Comp-' prefix so it matches FlowSample.data columns."""
        ch_elem = dim_elem.find('fcs-dimension')
        if ch_elem is None:
            ch_elem = next(dim_elem.iter('fcs-dimension'), None)
        if ch_elem is None:
            return None
        ch_name = ch_elem.get('name') or ch_elem.get('PnN')
        if not ch_name:
            return None
        if ch_name.startswith('Comp-'):
            ch_name = ch_name[len('Comp-'):]
        return ch_name

    def extract_gates(self, *, sample_node=None):
        """Extract supported gates from the .wsp as a list of gate dicts in
        topological order (parents before children).

        Parameters
        ----------
        sample_node : ET.Element | None
            When None (default), walk every ``<SampleNode>`` in the
            document and return a single flattened list — preserves the
            historical behaviour used by :func:`read_template_gates` and
            the GUI's per-template loader.
            When given a specific ``<SampleNode>`` element, walk only
            that sample's gate tree. Used by the gate editor's
            ``Add FCS / Workspace`` to attach the right gate tree to
            the right sample when opening a multi-sample .wsp.

        Each returned gate carries an `_import_id` (a temporary string, stable
        within this call) and a `parent_id` referencing another gate's
        `_import_id` (or None for roots). The GUI editor remaps these to its
        own gate_ids on load.

        Hierarchy comes from FlowJo's nested <Population>/<Subpopulations>
        structure. If no such hierarchy is found, falls back to a flat scan
        of every RectangleGate / PolygonGate in the document.

        Supports:
          • 1-D RectangleGate (min only)   → 'threshold'
          • 1-D RectangleGate (min + max)  → 'interval'
          • 2-D RectangleGate              → 'rect'
          • PolygonGate                    → 'polygon'
          • EllipsoidGate                  → 'ellipsoid'
          • QuadrantGate                   → 4 linked 'rect' (quad_set)
        Skipped (with a warning): CurlyQuad, BooleanGate.
        """
        if self.root is None:
            return []

        gates = []
        next_id = [0]

        def make_id():
            i = next_id[0]
            next_id[0] += 1
            return f'imp_{i}'

        def parse_rect(rect_elem, parent_imp_id):
            parsed = []
            for dim in rect_elem.iter('dimension'):
                ch = self._channel_name(dim)
                if not ch:
                    continue
                min_attr = dim.get('min')
                max_attr = dim.get('max')
                try:
                    lo = float(min_attr) if min_attr is not None else None
                    hi = float(max_attr) if max_attr is not None else None
                except (TypeError, ValueError):
                    continue
                parsed.append((ch, lo, hi))
            if len(parsed) == 1:
                ch, lo, hi = parsed[0]
                if lo is not None and hi is not None:
                    return {'kind': 'interval', 'channel': ch,
                            'lo': lo, 'hi': hi,
                            'parent_id': parent_imp_id,
                            '_import_id': make_id()}
                if lo is not None:
                    return {'kind': 'threshold', 'channel': ch, 'value': lo,
                            'parent_id': parent_imp_id,
                            '_import_id': make_id()}
                if hi is not None:
                    # Max-only 1-D rect (x < hi): represent as an interval with
                    # a large negative sentinel lo (JSON-safe, unlike -inf) so
                    # the constraint — and this population's children — stay in
                    # the tree instead of being silently re-parented up to the
                    # grandparent (which drops `x < hi` from every descendant).
                    return {'kind': 'interval', 'channel': ch,
                            'lo': -1e12, 'hi': hi,
                            'parent_id': parent_imp_id,
                            '_import_id': make_id()}
            elif len(parsed) == 2:
                (xc, x0, x1), (yc, y0, y1) = parsed
                if None not in (x0, x1, y0, y1):
                    return {'kind': 'rect',
                            'x_channel': xc, 'y_channel': yc,
                            'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1,
                            'parent_id': parent_imp_id,
                            '_import_id': make_id()}
                log.warning(
                    "[WspReader] Skipped 2-D RectangleGate "
                    "(%s x %s) with missing bounds", xc, yc)
            return None

        def parse_polygon(poly_elem, parent_imp_id):
            dims = list(poly_elem.iter('dimension'))
            if len(dims) != 2:
                return None
            xc = self._channel_name(dims[0])
            yc = self._channel_name(dims[1])
            if not xc or not yc:
                return None
            verts = []
            for v in poly_elem.iter('vertex'):
                coords = list(v.iter('coordinate'))
                if len(coords) < 2:
                    continue
                vx_attr = coords[0].get('value')
                vy_attr = coords[1].get('value')
                if vx_attr is None or vy_attr is None:
                    continue
                try:
                    vx = float(vx_attr)
                    vy = float(vy_attr)
                except (TypeError, ValueError):
                    continue
                verts.append([vx, vy])
            if len(verts) >= 3:
                return {'kind': 'polygon',
                        'x_channel': xc, 'y_channel': yc,
                        'vertices': verts,
                        'parent_id': parent_imp_id,
                        '_import_id': make_id()}
            return None

        def parse_ellipsoid(ell_elem, parent_imp_id):
            """Gating-ML 2.0 EllipsoidGate: 2 dimensions + <mean> +
            <covarianceMatrix> + <distanceSquare> (squared Mahalanobis
            radius). An event is inside when
            (x-µ)ᵀ Σ⁻¹ (x-µ) ≤ distanceSquare.

            NOTE: validated against the Gating-ML 2.0 spec + our own
            writer. Real FlowJo v10 files may additionally carry
            `foci` / `edge` hint elements — those are ignored here
            (the mean+cov+distance form is authoritative). Confirm
            against a genuine FlowJo ellipse before relying on it."""
            dims = list(ell_elem.findall('dimension'))
            if len(dims) != 2:
                return None
            xc = self._channel_name(dims[0])
            yc = self._channel_name(dims[1])
            if not xc or not yc:
                return None
            mean_elem = ell_elem.find('mean')
            if mean_elem is None:
                return None
            mean_vals = []
            for c in mean_elem.findall('coordinate'):
                v = c.get('value')
                if v is None:
                    return None
                try:
                    mean_vals.append(float(v))
                except ValueError:
                    return None
            if len(mean_vals) != 2:
                return None
            cov_elem = ell_elem.find('covarianceMatrix')
            if cov_elem is None:
                return None
            cov = []
            for row in cov_elem.findall('row'):
                entries = []
                for e in row.findall('entry'):
                    v = e.get('value')
                    if v is None:
                        return None
                    try:
                        entries.append(float(v))
                    except ValueError:
                        return None
                if len(entries) != 2:
                    return None
                cov.append(entries)
            if len(cov) != 2:
                return None
            dsq_elem = ell_elem.find('distanceSquare')
            try:
                dist_sq = (float(dsq_elem.get('value'))
                           if dsq_elem is not None else 4.0)
            except (TypeError, ValueError):
                dist_sq = 4.0
            return {'kind': 'ellipsoid', 'x_channel': xc, 'y_channel': yc,
                    'mean': mean_vals, 'cov': cov, 'distance_sq': dist_sq,
                    'parent_id': parent_imp_id, '_import_id': make_id()}

        def parse_quadrant(quad_elem, parent_imp_id):
            """Gating-ML 2.0 QuadrantGate: two <divider> elements, each
            naming a dimension + a threshold value. Expanded into FOUR
            rect gates sharing a `quad_set` id + `quad_origin`, spanning
            ±1e12 on their open sides — matching the editor's internal
            quadrant representation so the round-trip stays in one model.

            Returns a LIST of 4 gate dicts (not a single dict)."""
            divs = quad_elem.findall('divider')
            if len(divs) != 2:
                return None
            parsed = []
            for d in divs:
                ch = self._channel_name(d)
                # Divider threshold is a <value> child element (Gating-ML)
                # or a value= attribute (defensive fallback).
                val = None
                v_elem = d.find('value')
                if v_elem is not None and (v_elem.text or '').strip():
                    val = v_elem.text.strip()
                if val is None:
                    val = d.get('value')
                try:
                    val = float(val) if val is not None else None
                except (TypeError, ValueError):
                    val = None
                if not ch or val is None:
                    return None
                parsed.append((ch, val))
            (xc, xdiv), (yc, ydiv) = parsed
            big = 1e12
            # Rect masks use strict `>`/`<`, so make each divider inclusive on
            # the "greater" side only: nudge the lower bound just below the
            # divider (nextafter) so `x > xlo` also keeps x == xdiv. Otherwise an
            # event exactly on a divider (e.g. arcsinh(0)=0, integer pile-ups)
            # is excluded from BOTH sides and lands in no quadrant. The four
            # rects then tile the plane as a true partition.
            xlo, ylo = np.nextafter(xdiv, -np.inf), np.nextafter(ydiv, -np.inf)
            qs_id = make_id()
            out = []
            for label, x0, x1, y0, y1 in [
                    ('Q++ (x>, y>)',  xlo,   big,  ylo,   big),
                    ('Q+- (x>, y<)',  xlo,   big, -big,  ydiv),
                    ('Q-+ (x<, y>)', -big,  xdiv,  ylo,   big),
                    ('Q-- (x<, y<)', -big,  xdiv, -big,  ydiv)]:
                out.append({'kind': 'rect',
                            'x_channel': xc, 'y_channel': yc,
                            'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1,
                            'label': label,
                            'quad_set': qs_id,
                            'quad_origin_x': xdiv, 'quad_origin_y': ydiv,
                            'parent_id': parent_imp_id,
                            '_import_id': make_id()})
            return out

        def walk_population(pop_elem, parent_imp_id):
            """Process a <Population>: parse its <Gate> (if any), then recurse
            into its <Subpopulations>. Descendants attach to the nearest
            ancestor that did produce a gate (so unparseable nodes don't
            break the chain)."""
            this_id = None
            for gate_wrapper in pop_elem.findall('Gate'):
                for child in gate_wrapper:
                    g = None
                    if child.tag == 'RectangleGate':
                        g = parse_rect(child, parent_imp_id)
                    elif child.tag == 'PolygonGate':
                        g = parse_polygon(child, parent_imp_id)
                    elif child.tag == 'EllipsoidGate':
                        g = parse_ellipsoid(child, parent_imp_id)
                    elif child.tag == 'QuadrantGate':
                        quad = parse_quadrant(child, parent_imp_id)
                        if quad:
                            # parse_quadrant returns 4 linked rects.
                            gates.extend(quad)
                            this_id = quad[0]['_import_id']
                            break
                    if g is not None:
                        gates.append(g)
                        this_id = g['_import_id']
                        break
                if this_id is not None:
                    break
            attach_to = this_id if this_id is not None else parent_imp_id
            for sub in pop_elem.findall('Subpopulations'):
                for child_pop in sub.findall('Population'):
                    walk_population(child_pop, attach_to)

        if sample_node is not None:
            # Per-sample walk: just this <SampleNode>'s subpopulations.
            # Skip the fallback flat scan + the cross-document unsupported-
            # kinds warning (extract_gates() with no sample_node still runs
            # both for the full-document case).
            for sub in sample_node.findall('Subpopulations'):
                for pop in sub.findall('Population'):
                    walk_population(pop, None)
        else:
            # Walk hierarchy starting at every SampleNode/Subpopulations/
            # Population.
            walked = False
            for sn in self.root.iter('SampleNode'):
                for sub in sn.findall('Subpopulations'):
                    for pop in sub.findall('Population'):
                        walk_population(pop, None)
                        walked = True

            # Fallback flat scan when the .wsp has gates but no Population
            # wrappers.
            if not walked:
                for rect in self.root.iter('RectangleGate'):
                    g = parse_rect(rect, None)
                    if g is not None:
                        gates.append(g)
                for poly in self.root.iter('PolygonGate'):
                    g = parse_polygon(poly, None)
                    if g is not None:
                        gates.append(g)

            # Unsupported kinds: report once.
            skipped = set()
            for gt in ('EllipsoidGate', 'QuadrantGate', 'CurlyQuad', 'BooleanGate'):
                if any(True for _ in self.root.iter(gt)):
                    skipped.add(gt)
            if skipped:
                log.warning(
                    "[WspReader] Skipped unsupported gate types: %s",
                    sorted(skipped))

        if gates:
            # Indented log so the hierarchy is visible.
            id_to_depth = {}
            for g in gates:
                pid = g.get('parent_id')
                id_to_depth[g['_import_id']] = (
                    id_to_depth.get(pid, -1) + 1 if pid else 0)
            log.info(f"[WspReader] Extracted {len(gates)} gate(s):")
            for g in gates:
                d = id_to_depth.get(g['_import_id'], 0)
                log.info(f"   {'  ' * d}{describe_gate(g)}")
        else:
            log.info("[WspReader] No supported gates found.")

        return gates

    def _extract_matrix(self, node):
        name   = node.get('name') or node.get('matrixName') or 'unnamed'
        prefix = node.get('prefix', 'Comp-')
        suffix = node.get('suffix', '')
        channels = [p.get('name') or p.get('PnN')
                    for p in node.iter('parameter')
                    if p.get('name') or p.get('PnN')]
        if not channels:
            return
        n      = len(channels)
        values = []
        for vn in node.iter('spilloverValues'):
            try:
                values = [float(v) for v in vn.get('values', '').split(',')]
            except ValueError:
                pass
        matrix = None
        if len(values) == n * n:
            matrix = np.array(values).reshape(n, n)
        else:
            rows = []
            for row in node.iter('spilloverRow'):
                try:
                    rows.append([float(v) for v in row.get('values','').split(',')])
                except ValueError:
                    pass
            if len(rows) == n:
                matrix = np.array(rows)
        # Gating-ML v2 layout (FlowJo v10 .wsp):
        #   <spillover parameter="src"><coefficient parameter="dst" value="x"/></spillover>
        # Build the matrix row-by-row from the source/destination channel
        # pairs against our channel index. Identity diagonal as the default.
        if matrix is None:
            ch_idx = {c: i for i, c in enumerate(channels)}
            m = np.eye(n)
            seen = 0
            for sp in node.iter('spillover'):
                src = sp.get('parameter')
                if src not in ch_idx:
                    continue
                i = ch_idx[src]
                for coef in sp.iter('coefficient'):
                    dst = coef.get('parameter')
                    val = coef.get('value')
                    if dst in ch_idx and val is not None:
                        try:
                            m[i, ch_idx[dst]] = float(val)
                            seen += 1
                        except ValueError:
                            pass
            if seen >= n:    # at minimum the diagonal entries should show up
                matrix = m
        if matrix is None:
            log.info(f"[WspReader] Could not parse values for '{name}'.")
            return
        self.matrices[name] = dict(matrix=matrix, channels=channels,
                                   prefix=prefix, suffix=suffix)

    def get_matrix(self, name=None):
        if not self.matrices:
            raise RuntimeError("No matrices available.")
        if name is None:
            return next(iter(self.matrices.values()))
        if name not in self.matrices:
            raise KeyError(f"'{name}' not found. Available: {list(self.matrices)}")
        return self.matrices[name]

    def print_matrices(self):
        for name, m in self.matrices.items():
            log.info(f"  '{name}' — {len(m['channels'])} ch: {m['channels']}")


# ══════════════════════════════════════════════════════════════════════════════
# QC MODULE
# ══════════════════════════════════════════════════════════════════════════════

class AcquisitionQC:
    """
    Acquisition QC. Three independent anomaly detectors, combined into one
    clean-event index:

      1. **Signal drift** — time bins where any channel's median deviates
         more than ``threshold`` MADs from its global median (sensor drift,
         settling, sustained instability).
      2. **Flow-rate anomalies** — time bins whose event count is a robust
         (MAD) outlier, or interior bins that are empty: clogs (rate
         collapses) and bubbles (a gap or a burst).
      3. **Margin / saturation events** — events piled up at a channel's
         ceiling (off-scale), the classic signature of a bubble/clog or
         electronic saturation. These are dropped per-event, not per-bin.

    Detectors 1–2 need a Time channel and no-op without one; detector 3
    does not. A clean acquisition trips none of them.

    Usage
    -----
        qc = AcquisitionQC(sample.data)
        clean_idx = qc.run(n_bins=200, threshold=5)
        sample.data = sample.data.loc[clean_idx].reset_index(drop=True)
        qc.plot()
        qc.report   # {'drift': n, 'flow_rate': n, 'margin': n, 'total': n}
    """

    def __init__(self, data, time_channel='Time'):
        self.data         = data
        self.time_channel = self._find_time(data, time_channel)
        self.flag         = None
        self.bin_stats    = None
        self.bin_counts   = None
        self.report       = {}

    @staticmethod
    def _find_time(data, hint):
        if hint in data.columns:
            return hint
        for col in data.columns:
            if 'time' in col.lower():
                return col
        return None

    @staticmethod
    def _mad_outliers(vals, threshold):
        """Boolean mask of robust (median ± threshold·MAD) outliers."""
        vals = np.asarray(vals, dtype=float)
        med  = np.median(vals)
        mad  = np.median(np.abs(vals - med)) + 1e-10
        return np.abs(vals - med) > threshold * mad

    def _drift_bad_bins(self, bins, channels, n_bins, threshold):
        """Time bins whose per-channel median drifts > threshold·MAD from
        that channel's across-bin median. Also populates self.bin_stats.

        Vectorised: one ``groupby`` over the bin labels replaces the former
        per-bin (boolean-mask + per-channel ``.median()``) Python loop — the
        SAME medians, but O(N) instead of O(N·n_bins). Bins with < 10 events
        contribute no median (NaN), exactly as before; bin_stats keeps a row
        per bin (len == n_bins) for the QC plot."""
        b = np.asarray(bins)
        g = self.data[channels].groupby(b)
        counts = g.size()
        med = g.median(numeric_only=True)
        med = med[counts.values >= 10]              # drop sparse bins (as before)
        med.index = med.index.astype(int)
        self.bin_stats = med.reindex(range(n_bins))  # all bins; NaN where sparse

        bad = set()
        for ch in channels:
            if ch not in self.bin_stats.columns:
                continue
            series  = self.bin_stats[ch].dropna()
            out     = self._mad_outliers(series.values, threshold)
            idx_arr = np.asarray(series.index)
            bad.update(idx_arr[out].tolist())
        return bad

    def _flowrate_bad_bins(self, bins, n_bins, threshold):
        """Time bins whose event count is a MAD outlier, plus empty
        interior bins (a gap = bubble; a collapse = clog). Edge bins are
        exempt from the 'empty' rule — acquisitions routinely start/stop
        mid-bin. Also populates self.bin_counts."""
        # Vectorised per-bin counts (bincount) — identical to the former
        # per-bin count_nonzero loop, O(N) instead of O(N·n_bins).
        b = np.asarray(bins)
        valid = ~pd.isna(b)
        counts = np.bincount(b[valid].astype(int),
                             minlength=n_bins)[:n_bins].astype(int)
        self.bin_counts = counts
        nonempty = counts[counts > 0]
        if nonempty.size < 3:
            return set()
        bad = set()
        # Count outliers among bins that actually have events.
        nz_idx = np.where(counts > 0)[0]
        out    = self._mad_outliers(counts[nz_idx], threshold)
        bad.update(nz_idx[out].tolist())
        # Empty interior bins (gaps), ignoring leading/trailing empties.
        # Only meaningful when bins are densely populated — on a sparse
        # file an empty interior bin is expected, not an anomaly.
        first, last = int(nz_idx[0]), int(nz_idx[-1])
        if np.median(nonempty) >= 20:
            for b in range(first + 1, last):
                if counts[b] == 0:
                    bad.add(b)
        return bad

    @staticmethod
    def _margin_events(data, channels, frac, eps=1e-9):
        """Boolean per-event mask of margin/saturation events: those at a
        channel's ceiling (its max) when that ceiling is *piled up* — i.e.
        at least `frac` of events share the max value. A single off-scale
        max (continuous data) isn't a pile-up and is left alone."""
        n = len(data)
        bad = np.zeros(n, dtype=bool)
        if n == 0:
            return bad
        thresh = max(2, int(np.ceil(frac * n)))
        for ch in channels:
            col = np.asarray(data[ch].values, dtype=float)
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                continue
            ceiling = finite.max()
            at_ceiling = np.isfinite(col) & (np.abs(col - ceiling) <= eps)
            if int(at_ceiling.sum()) >= thresh:
                bad |= at_ceiling
        return bad

    def run(self, n_bins=200, threshold: float = 5, channels=None,
            drift=True, flow_rate=True, margins=True, flow_rate_threshold=5.0,
            margin_frac=0.01):
        n = len(self.data)
        keep = np.ones(n, dtype=bool)
        report = {'drift': 0, 'flow_rate': 0, 'margin': 0}

        if channels is None:
            channels = [c for c in self.data.columns
                        if c != self.time_channel
                        and not _is_excluded(c)
                        and pd.api.types.is_numeric_dtype(self.data[c])]

        bins = None
        if self.time_channel is not None and n_bins > 0 and n > 0:
            t    = np.asarray(self.data[self.time_channel].values)
            bins = pd.cut(t, bins=n_bins, labels=False)
            bin_arr = np.asarray(bins)

            if drift:
                drift_bad = self._drift_bad_bins(
                    bins, channels, n_bins, threshold)
                if drift_bad:
                    drift_evt = np.isin(bin_arr, list(drift_bad))
                    report['drift'] = int(drift_evt.sum())
                    keep &= ~drift_evt

            if flow_rate:
                flow_bad = self._flowrate_bad_bins(
                    bins, n_bins, flow_rate_threshold)
                if flow_bad:
                    flow_evt = np.isin(bin_arr, list(flow_bad))
                    report['flow_rate'] = int(flow_evt.sum())
                    keep &= ~flow_evt
        else:
            log.info("  [QC] No time channel — time-based detectors skipped.")

        if margins:
            margin_evt = self._margin_events(self.data, channels, margin_frac)
            report['margin'] = int(margin_evt.sum())
            keep &= ~margin_evt

        report['total'] = int((~keep).sum())
        self.report = report
        self.flag   = pd.Series(keep, index=self.data.index)
        pct_rem     = (report['total'] / n * 100.0) if n else 0.0
        log.info(
            "  [QC] Removed %.1f%% events (%s total: drift %s, flow-rate %s, "
            "margin %s).",
            pct_rem, f"{report['total']:,}", f"{report['drift']:,}",
            f"{report['flow_rate']:,}", f"{report['margin']:,}")
        return self.data.index[keep]

    def plot(self):
        if self.bin_stats is None:
            log.info("  [QC] Run .run() first.")
            return
        import matplotlib.pyplot as plt  # lazy: see module-top comment
        t    = self.data[self.time_channel].values
        bins = pd.cut(t, bins=len(self.bin_stats), labels=False)
        cnts = pd.Series(bins).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(np.asarray(cnts.index), np.asarray(cnts.values),
                lw=0.8, color='steelblue')
        ax.set_xlabel('Time bin')
        ax.set_ylabel('Event count')
        ax.set_title('Acquisition QC — event rate over time')
        plt.tight_layout()
        return ax


# ══════════════════════════════════════════════════════════════════════════════
# CELL CYCLE
# ══════════════════════════════════════════════════════════════════════════════
#
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


def _robust_sd(arr):
    """1.4826 * MAD — outlier-resistant SD estimate. NaN on empty."""
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float('nan')
    med = np.median(a)
    return 1.4826 * float(np.median(np.abs(a - med)))


def find_dna_channel(sample):
    """Best-guess DNA-content detector for `sample`, or None.

    Matches known DNA-dye tokens against each channel's antibody label
    first, then its detector name. Prefers an Area (`-A`) channel. The
    short token 'pi' only matches as a whole word so it doesn't fire on
    'PE' / 'APC' / 'PI3K' etc."""
    labels = getattr(sample, 'channel_labels', {}) or {}
    cols = list(sample.data.columns)

    def matches(text):
        t = str(text).lower()
        for dye in DNA_DYES:
            if dye == 'pi':
                if re.search(r'(?<![a-z])pi(?![a-z])', t):
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
        ok      (bool — False when no usable peak was found)
    Use assign_phase(values, model) to label arbitrary arrays with the
    same boundaries."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    nan = float('nan')
    model = {'g1_mean': nan, 'g1_sd': nan, 'g2_mean': nan, 'g2_sd': nan,
             'g1_hi': nan, 'g2_lo': nan, 'pct_g1': nan, 'pct_s': nan,
             'pct_g2m': nan, 'counts': {}, 'n_cycling': 0, 'n': 0, 'ok': False}

    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    model['n'] = int(v.size)
    if v.size < 50:
        return model

    lo, hi = np.percentile(v, [0.5, 99.5])
    if hi <= lo:
        return model
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    sm = gaussian_filter1d(hist.astype(float), sigma=2.0)
    if sm.max() <= 0:
        return model

    peaks, _ = find_peaks(sm, prominence=sm.max() * 0.05)
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(sm))])
    # G1 = the most prominent (tallest) peak.
    order = peaks[np.argsort(sm[peaks])[::-1]]
    g1_mean = float(centers[order[0]])
    if g1_mean <= 0:
        g1_mean = float(np.median(v))

    # G2/M ≈ 2× G1: nearest remaining peak in [1.7×, 2.3×]; else nominal 2×.
    g2_mean = None
    for pk in order[1:]:
        c = float(centers[pk])
        if 1.7 * g1_mean <= c <= 2.3 * g1_mean:
            g2_mean = c
            break
    if g2_mean is None:
        g2_mean = 2.0 * g1_mean

    g1_win = v[(v >= 0.85 * g1_mean) & (v <= 1.15 * g1_mean)]
    g1_sd = _robust_sd(g1_win)
    if not np.isfinite(g1_sd) or g1_sd <= 0:
        g1_sd = 0.05 * g1_mean
    g2_win = v[(v >= 0.9 * g2_mean) & (v <= 1.1 * g2_mean)]
    g2_sd = _robust_sd(g2_win)
    if not np.isfinite(g2_sd) or g2_sd <= 0:
        g2_sd = g1_sd * 1.4   # G2 CV ~ G1 CV; width scales with mean

    g1_hi = g1_mean + k * g1_sd
    g2_lo = g2_mean - k * g2_sd
    if g1_hi >= g2_lo:                      # peaks overlap → split at midpoint
        mid = 0.5 * (g1_mean + g2_mean)
        g1_hi = min(g1_hi, mid)
        g2_lo = max(g2_lo, mid)

    model.update(g1_mean=g1_mean, g1_sd=g1_sd, g2_mean=g2_mean, g2_sd=g2_sd,
                 g1_hi=g1_hi, g2_lo=g2_lo, ok=True)

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


# ══════════════════════════════════════════════════════════════════════════════
# FMO GATE THRESHOLDER
# ══════════════════════════════════════════════════════════════════════════════

class FMOGater:
    """
    Calculate positive thresholds from FMO control files.

    Usage
    -----
        gater = FMOGater()
        gater.add_fmo('Comp-BV421-A', 'fmo_bv421.fcs')  # CD11b
        gater.add_fmo('Comp-APC-A',           'fmo_apc.fcs')   # CD34
        gater.add_fmo('Comp-PE-Cy7-A',        'fmo_cy7.fcs')   # CD45
        thresholds = gater.compute(percentile=99.5)
        sample.apply_threshold_gates(thresholds)
    """

    def __init__(self):
        self.fmos        = {}
        self._is_fallback = {}   # channel -> True if using unstained instead of FMO

    def add_fmo(self, channel_name, fcs_path, is_fallback=False):
        """
        channel_name: detector name this FMO (or fallback unstained) controls.
        is_fallback:  True when the file is an unstained control rather than a
                      proper FMO — noted in compute() output.
        """
        self.fmos[channel_name]         = FlowSample(fcs_path)
        self._is_fallback[channel_name] = is_fallback
        return self

    def add_fmos_from_dir(self, directory, pattern=r'fmo[_-]?(\w+)'):
        rx = re.compile(pattern, re.IGNORECASE)
        for fname in os.listdir(directory):
            if not fname.lower().endswith('.fcs'):
                continue
            m = rx.search(fname)
            if m:
                ch  = m.group(1)
                self.fmos[ch] = FlowSample(os.path.join(directory, fname))
                log.info(f"  [FMO] '{ch}' ← {fname}")
        return self

    def prepare(self, wsp_path=None, transform_method='logicle'):
        """
        Compensate and transform all FMO samples so thresholds are in the
        same data space as the experimental samples.  Call before compute().
        """
        for _ch, sample in self.fmos.items():
            if wsp_path:
                sample.compensate_from_wsp(wsp_path)
            else:
                sample.auto_compensate()
            sample.apply_transform(method=transform_method)
        return self

    def compute(self, percentile=99.5):
        """
        Returns {channel_name: threshold} where channel_name is the key
        passed to add_fmo() — typically the compensated detector name.
        The threshold is the p`percentile` of that channel in the FMO.
        """
        thresholds = {}
        for ch, sample in self.fmos.items():
            col = self._find_col(sample, ch)
            if col is None:
                log.info(f"  [FMO] '{ch}' not found in {sample.name} — skipped.")
                continue
            val = float(np.percentile(sample.data[col].dropna(), percentile))
            thresholds[ch] = val
            tag = ' [UNSTAINED fallback]' if self._is_fallback.get(ch) else ''
            log.info(f"  [FMO] {ch}: threshold={val:.3f} (p{percentile}){tag}")
        return thresholds

    @staticmethod
    def _find_col(sample, hint):
        """Match by exact name, then by stripping common comp prefixes, then substring."""
        if hint in sample.data.columns:
            return hint
        stripped = re.sub(r'^[Cc]omp-', '', hint)
        for col in sample.data.columns:
            if col == stripped or col.lower() == stripped.lower():
                return col
        for col in sample.data.columns:
            if hint.lower() in col.lower() or stripped.lower() in col.lower():
                return col
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CORE SAMPLE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class FlowSample:
    """
    Single FCS file with full analysis pipeline.

    Typical workflow
    ----------------
        s = FlowSample('file.fcs')
        s.run_qc()
        s.auto_compensate()           # OR s.compensate_from_wsp('exp.wsp')
        s.apply_transform()
        s.apply_threshold_gates(thresholds)   # optional FMO gates
        s.cluster(k=30)
        s.run_umap()
        s.plot('CD11b', 'CD45', color_by='cluster')
        s.cluster_heatmap()
        s.export_csv()
    """

    def __init__(self, fcs_path):
        self.path            = fcs_path
        self.name            = os.path.splitext(os.path.basename(fcs_path))[0]
        self.metadata        = {}
        self.channel_names   = []
        self.channel_labels  = {}
        self.scatter_channels = []
        self.fluor_channels   = []
        self.thresholds       = {}
        # Compensation matrix actually applied to this sample (post-channel-
        # intersection). Populated by `_apply_comp`, read by the workspace
        # exporters so the .wsp round-trips the spillover.
        self.comp_matrix:   np.ndarray | None = None
        self.comp_channels: list[str] = []
        # data / raw are populated unconditionally by _load() below. We
        # initialise with empty DataFrames (rather than None) so static
        # analysers can see them as pd.DataFrame everywhere downstream
        # without needing per-call narrowing.
        self.data: pd.DataFrame = pd.DataFrame()
        self.raw:  pd.DataFrame = pd.DataFrame()
        self.clusters         = None
        self.umap_coords      = None
        self.trimap_coords    = None
        self.pacmap_coords    = None
        self.cell_cycle_result = None
        self.flowsom_result   = None

        self._load()
        self._classify_channels()
        self._print_summary()

    @classmethod
    def from_dataframe(cls, df, name='sample', labels=None, metadata=None,
                       path=''):
        """Build a FlowSample from an in-memory DataFrame instead of an FCS
        file — e.g. to re-open a pipeline ``*_processed.csv`` (which carries
        cluster / UMAP / flowsom columns) in the editor.

        `labels` is an optional ``{column: antibody label}`` map; columns
        without one use the column name. Derived/analysis columns (cluster,
        UMAP*, flowsom*, cell_cycle, Time) are auto-excluded from the marker
        lists by ``_classify_channels``."""
        import pandas as pd
        s = cls.__new__(cls)
        s.path            = path
        s.name            = name
        s.metadata        = dict(metadata or {})
        s.channel_names   = list(df.columns)
        labels = labels or {}
        s.channel_labels  = {c: (labels.get(c) or c) for c in s.channel_names}
        s.scatter_channels = []
        s.fluor_channels   = []
        s.thresholds       = {}
        s.comp_matrix      = None
        s.comp_channels    = []
        s.data = pd.DataFrame(df).reset_index(drop=True).copy()
        s.raw  = s.data.copy()
        s.clusters         = None
        s.umap_coords      = None
        s.trimap_coords    = None
        s.pacmap_coords    = None
        s.cell_cycle_result = None
        s.flowsom_result   = None
        s._classify_channels()
        return s

    # ── Load ──────────────────────────────────────────────────────────────────

    def _load(self):
        try:
            fcs = flowio.FlowData(self.path)
        except FileNotFoundError as e:
            raise FcsParseError(f"FCS file not found: {self.path}") from e
        except Exception as e:
            raise FcsParseError(
                f"could not parse FCS {self.path}: "
                f"{type(e).__name__}: {e}") from e
        self.metadata = dict(fcs.text)
        # FlowIO ≥1.0 uses integer keys and lowercase field names (pnn/pns)
        for i in range(1, fcs.channel_count + 1):
            ch    = fcs.channels[i]
            name  = ch.get('pnn') or ch.get('PnN') or f'Ch{i}'
            label = (ch.get('pns') or ch.get('PnS') or '').strip()
            self.channel_names.append(name)
            self.channel_labels[name] = label if label else name
        if not any(self.channel_labels[n] != n for n in self.channel_names):
            log.info("  (no PnS labels in FCS — use detector names for plotting)")
        # flowio types `events` as Optional but in practice it's always
        # populated after a successful FlowData() construction; assert so
        # static analysis follows.
        assert fcs.events is not None
        events   = np.reshape(np.asarray(fcs.events), (-1, fcs.channel_count))
        self.raw  = pd.DataFrame(events, columns=pd.Index(self.channel_names))
        self.data = self.raw.copy()

    def _classify_channels(self):
        for ch in self.channel_names:
            if _is_scatter(ch):
                self.scatter_channels.append(ch)
            elif not _is_excluded(ch):
                self.fluor_channels.append(ch)

    def _print_summary(self):
        labels = [self.channel_labels[c] for c in self.fluor_channels]
        log.info(
            "[%s]  %s events  |  fluor channels: %s",
            self.name, f"{len(self.data):,}", labels)

    def set_labels(self, mapping):
        """
        Assign antibody names to detector channels after loading.
        mapping: {detector_name: antibody_label}
        e.g. {'BV421-A': 'CD11b', 'APC-A': 'CD34', 'PE-Cy7-A': 'CD45'}
        After calling this, plot('CD11b', 'CD45') and axis labels both work.
        """
        applied = {}
        for det, label in mapping.items():
            if det in self.channel_labels:
                self.channel_labels[det] = label
                applied[det] = label
        if applied:
            log.info(f"  Labels: { {d: lbl for d, lbl in applied.items()} }")
        return self

    # ── QC ────────────────────────────────────────────────────────────────────

    def run_qc(self, n_bins=200, threshold=5, plot=False):
        qc        = AcquisitionQC(self.data)
        clean_idx = qc.run(n_bins=n_bins, threshold=threshold)
        self.data = self.data.loc[clean_idx].reset_index(drop=True)
        if plot:
            qc.plot()
        return self

    # ── Debris + doublet filtering ────────────────────────────────────────────

    def _find_scatter_col(self, prefix, suffix='-A'):
        """Return the column matching e.g. FSC-A or FSC-H, case-insensitive,
        preferring exact -A / -H endings. None if not found."""
        prefix_u = prefix.upper()
        suffix_u = suffix.upper()
        for c in self.data.columns:
            cu = c.upper()
            if cu.startswith(prefix_u) and cu.endswith(suffix_u):
                return c
        for c in self.data.columns:
            if c.upper().startswith(prefix_u):
                return c
        return None

    def filter_debris(self, fsc_channel=None, min_fsc=None):
        """Drop events whose FSC-A is below `min_fsc`. No-op if `min_fsc`
        is None or the FSC-A column can't be located."""
        if min_fsc is None:
            return self
        if fsc_channel is None:
            fsc_channel = self._find_scatter_col('FSC', '-A')
        if not fsc_channel or fsc_channel not in self.data.columns:
            log.info("  [Debris] No FSC-A channel found — debris filter skipped.")
            return self
        before = len(self.data)
        keep   = self.data[fsc_channel] >= float(min_fsc)
        self.data = cast(pd.DataFrame, self.data[keep]).reset_index(drop=True)
        kept = len(self.data)
        pct  = (kept / before * 100.0) if before else 0.0
        log.info(
            "  [Debris] Kept %s / %s events (%.1f%%) — FSC-A >= %.0f",
            f"{kept:,}", f"{before:,}", pct, float(min_fsc))
        return self

    def filter_doublets(self, fsc_a_channel=None, fsc_h_channel=None,
                        tol=0.25):
        """Drop doublets via the FSC-A / FSC-H ratio. Singlets fall along
        FSC-A ≈ k·FSC-H; doublets push FSC-A high relative to FSC-H, so
        their ratio sits well outside the population median. We keep
        events whose ratio is within ±`tol` of the median ratio.

        `tol` defaults to 0.25 (a relatively wide window) because
        polyploid cells are intrinsically more variable in
        FSC-A / FSC-H than typical leukocytes. Tighten to 0.15 for
        diploid samples.
        """
        if tol is None or tol <= 0:
            return self
        if fsc_a_channel is None:
            fsc_a_channel = self._find_scatter_col('FSC', '-A')
        if fsc_h_channel is None:
            fsc_h_channel = self._find_scatter_col('FSC', '-H')
        if (not fsc_a_channel or fsc_a_channel not in self.data.columns
                or not fsc_h_channel or fsc_h_channel not in self.data.columns):
            log.warning(
                "  [Doublets] FSC-A or FSC-H channel missing — "
                "doublet filter skipped.")
            return self
        before  = len(self.data)
        fsc_a   = self.data[fsc_a_channel].astype(float)
        fsc_h   = self.data[fsc_h_channel].astype(float)
        ratio   = np.where(fsc_h > 0, fsc_a / fsc_h, np.nan)
        valid   = np.isfinite(ratio)
        median  = float(np.nanmedian(ratio[valid])) if valid.any() else 0.0
        lo, hi  = median * (1.0 - tol), median * (1.0 + tol)
        keep    = valid & (ratio >= lo) & (ratio <= hi)
        self.data = cast(pd.DataFrame, self.data[keep]).reset_index(drop=True)
        kept = len(self.data)
        pct  = (kept / before * 100.0) if before else 0.0
        log.info(
            "  [Doublets] Kept %s / %s events (%.1f%%) — "
            "FSC-A/FSC-H in [%.3f, %.3f] (median %.3f, ±%.0f%%)",
            f"{kept:,}", f"{before:,}", pct, lo, hi, median, tol * 100)
        return self

    # ── Compensation ──────────────────────────────────────────────────────────

    def auto_compensate(self):
        """Apply spillover matrix embedded in FCS metadata."""
        spill = self._parse_spillover()
        if spill is None:
            log.warning(f"  [!] No spillover in FCS metadata for {self.name}.")
            return self
        self._apply_comp(spill['matrix'], spill['channels'])
        return self

    def compensate_from_wsp(self, wsp_path, matrix_name=None):
        """Apply compensation from a FlowJo .wsp file."""
        m = WspReader(wsp_path).get_matrix(matrix_name)
        self._apply_comp(m['matrix'], m['channels'])
        return self

    def manual_compensate(self, matrix, channels):
        self._apply_comp(matrix, channels)
        return self

    def _apply_comp(self, matrix, channels):
        idx   = [i for i, c in enumerate(channels) if c in self.data.columns]
        avail = [channels[i] for i in idx]
        if not avail:
            log.warning("  [!] No matching channels for compensation.")
            return
        sub = matrix[np.ix_(idx, idx)]
        inv = np.linalg.inv(sub)
        from . import gpu_accel
        # Spillover is source->dest (measured = true @ M), so un-mixing is
        # `data @ inv(M)` — NO transpose. (gpu_accel.compensate computes
        # `values @ arg`.) The historical `inv.T` here silently left asymmetric
        # spillover uncorrected AND corrupted clean channels — i.e. every real
        # compensated dataset. See test_compensation_recovers_true_signal.
        self.data[avail] = gpu_accel.compensate(self.data[avail].values, inv)
        # Persist the matrix that was ACTUALLY applied (post-channel-
        # intersection, so it matches the channels in self.data) so the
        # GUI / CLI workspace export can round-trip it.
        self.comp_matrix   = np.asarray(sub, dtype=float).copy()
        self.comp_channels = list(avail)
        log.info(f"  Compensation applied: {avail}")

    def _parse_spillover(self):
        for key in ['SPILL','SPILLOVER','$SPILL','$SPILLOVER','spill','spillover']:
            val = self.metadata.get(key)
            if val:
                try:
                    parts = val.split(',')
                    n     = int(parts[0])
                    chans = [p.strip() for p in parts[1:n+1]]
                    vals  = [float(p) for p in parts[n+1:n+1+n*n]]
                    return dict(channels=chans, matrix=np.array(vals).reshape(n,n))
                except Exception as e:
                    log.warning(f"  [!] Spillover parse error: {e}")
        return None

    # ── Transform ─────────────────────────────────────────────────────────────

    def apply_transform(self, channels=None, method='logicle',
                        t=262144, m=4.5, w=0.5, a=0, cofactor=150.0):
        if channels is None:
            channels = self.fluor_channels
        avail = [c for c in channels if c in self.data.columns]
        for ch in avail:
            vals = np.asarray(self.data[ch].values, dtype=float).copy()
            try:
                self.data[ch] = transform_values(
                    vals, method=method, t=t, m=m, w=w, a=a, cofactor=cofactor)
            except Exception as e:
                log.warning(f"  [!] Transform failed {ch}: {e}")
        log.info(f"  {method} transform applied to {len(avail)} channels.")
        return self

    # ── FMO gating ────────────────────────────────────────────────────────────

    def apply_threshold_gates(self, thresholds):
        """
        Add boolean '<channel>_pos' columns from FMO-derived thresholds.
        thresholds : {channel_name: cutoff_value}
        """
        self.thresholds = thresholds
        for ch, val in thresholds.items():
            col = self._resolve(ch)
            if col in self.data.columns:
                self.data[f'{col}_pos'] = self.data[col] > val
                pct = self.data[f'{col}_pos'].mean() * 100
                log.info(f"  Gate {col} > {val:.2f}  ->  {pct:.1f}% positive")
        return self

    def apply_region_gates(self, gates):
        """Filter `self.data` to events satisfying every (enabled) gate in
        `gates`, combined via logical AND.

        Memory-aware: we maintain one bool `keep` mask over the original
        rows and evaluate each gate ONLY against the still-active rows
        (`np.where(keep)`). The DataFrame is sliced exactly once at the
        end, so we never pay the cost of duplicating wide intermediate
        DataFrames. Polygon gates use float32 coords so a 10 M-event
        sample's pts array is 80 MB instead of 160 MB; subsequent
        polygons after earlier gates have narrowed the data are much
        smaller still.

        Disabled gates (gate['enabled'] is False) are skipped.

        Call AFTER `apply_transform()` so gate coordinates (typically
        logicle) match the data scale.
        """
        if not gates:
            return self
        active = [g for g in gates if g.get('enabled', True)]
        if not active:
            return self

        n_total = len(self.data)
        keep = np.ones(n_total, dtype=bool)
        per_gate = []                  # (label, kept_after_this_gate)

        for g in active:
            n_active = int(keep.sum())
            if n_active == 0:
                per_gate.append((describe_gate(g), 0))
                continue
            active_idx = np.where(keep)[0]
            sub_mask = self._evaluate_gate_on(g, active_idx)
            # Write the gate's verdict back into the full-length keep
            # mask: rows the gate excluded become False; rows it kept
            # stay True (they were already True).
            keep[active_idx] = sub_mask
            per_gate.append((describe_gate(g), int(keep.sum())))

        kept = int(keep.sum())
        pct  = (kept / n_total * 100.0) if n_total else 0.0
        self.data = cast(pd.DataFrame, self.data[keep]).reset_index(drop=True)
        log.info(
            "  [Gates] Kept %s / %s events (%.1f%%) after %d region gate(s):",
            f"{kept:,}", f"{n_total:,}", pct, len(active))
        for lbl, n in per_gate:
            this_pct = (n / n_total * 100.0) if n_total else 0.0
            # ASCII-only — stdout may be cp1252 in library callers.
            log.info(f"    -> {n:>10,d} ({this_pct:5.1f}%)   {lbl}")
        return self

    def _evaluate_gate_on(self, gate, active_idx):
        """Evaluate one gate against `self.data` restricted to the rows at
        `active_idx`. Returns a bool mask of length len(active_idx).

        Allocates ONLY the columns the gate actually touches (and only the
        active rows of them), so memory scales with how much earlier
        gates have already narrowed the candidate set, not with the
        original sample size.
        """
        kind = gate.get('kind')
        n_act = len(active_idx)

        def _col(ch, dtype):
            if ch not in self.data.columns:
                return None
            # Pull only the active rows of this one column, in compact dtype.
            return np.asarray(self.data[ch].values[active_idx], dtype=dtype)

        if kind == 'threshold':
            vals = _col(gate['channel'], np.float64)
            if vals is None:
                log.warning(
                    "  [gate] threshold: channel %r not in data — skipped",
                    gate['channel'])
                return np.ones(n_act, dtype=bool)
            return vals > float(gate['value'])

        if kind == 'interval':
            vals = _col(gate['channel'], np.float64)
            if vals is None:
                log.warning(
                    "  [gate] interval: channel %r not in data — skipped",
                    gate['channel'])
                return np.ones(n_act, dtype=bool)
            return (vals > float(gate['lo'])) & (vals < float(gate['hi']))

        if kind == 'rect':
            xs = _col(gate['x_channel'], np.float64)
            ys = _col(gate['y_channel'], np.float64)
            if xs is None or ys is None:
                log.warning(
                    "  [gate] rect: channel(s) %r / %r missing — skipped",
                    gate.get('x_channel'), gate.get('y_channel'))
                return np.ones(n_act, dtype=bool)
            return ((xs > float(gate['x0'])) & (xs < float(gate['x1'])) &
                    (ys > float(gate['y0'])) & (ys < float(gate['y1'])))

        if kind == 'polygon':
            from matplotlib.path import Path as _MplPath
            xs = _col(gate['x_channel'], np.float32)
            ys = _col(gate['y_channel'], np.float32)
            if xs is None or ys is None:
                log.warning(
                    "  [gate] polygon: channel(s) %r / %r missing — skipped",
                    gate.get('x_channel'), gate.get('y_channel'))
                return np.ones(n_act, dtype=bool)
            verts = np.asarray(gate['vertices'], dtype=np.float32)
            if verts.ndim != 2 or verts.shape[1] != 2 or len(verts) < 3:
                log.warning(
                    "  [gate] polygon: malformed vertices (shape=%s) — skipped",
                    verts.shape)
                return np.ones(n_act, dtype=bool)
            pts = np.column_stack([xs, ys])
            result = _MplPath(verts).contains_points(pts)
            del pts, xs, ys     # release the temps before the next gate
            return result

        log.info(f"  [gate] unknown kind {kind!r} — skipped")
        return np.ones(n_act, dtype=bool)

    # ── Clustering ────────────────────────────────────────────────────────────

    def cluster(self, channels=None, k=30, n_jobs=1, max_events=None,
                use_gpu='auto', vram_admission_gb=1.0, random_state=42,
                reproducible=False):
        """Phenograph-style clustering.

        max_events
            If set and the sample exceeds it, cluster a random sub-sample and
            assign the remaining events to their nearest labelled neighbour
            via a KD-tree (caps peak memory regardless of sample size).

        use_gpu
            'auto' (default): use the GPU clustering path when RAPIDS is
            available *and* free VRAM ≥ `vram_admission_gb`; otherwise CPU.
            True : force GPU (fall back to CPU only on exception).
            False: never attempt GPU.

        vram_admission_gb
            Minimum free VRAM (GB) required to take the GPU branch.

        reproducible
            Off by default: PhenoGraph's default Louvain community detection is
            NOT seed-reproducible (its Blondel binary is time-seeded), so labels
            can differ run-to-run for the same input + ``random_state``. When
            True, cluster with PhenoGraph's *seeded Leiden* backend on the CPU
            path (GPU cuGraph Louvain is likewise unseeded, so it's skipped) —
            identical input + ``random_state`` → identical labels. This changes
            the algorithm, so labels differ from a default Louvain run: it buys
            reproducibility, not label-compatibility with prior Louvain results.

        The GPU branch uses cuML NearestNeighbors + cuGraph Louvain on the
        kNN graph. Any GPU failure (OOM, CUDA error, missing kit) falls back
        to CPU Phenograph for the same call — the result is always written.
        """
        if channels is None:
            channels = self.fluor_channels
        avail = [c for c in channels if c in self.data.columns]
        X     = self.data[avail].values.astype(float)
        mask  = np.all(np.isfinite(X), axis=1)
        Xc    = X[mask]
        n     = len(Xc)

        # Pick the events that will actually be clustered (full or subsample).
        if max_events and n > max_events:
            rng        = np.random.default_rng(random_state)
            sub_idx    = np.sort(rng.choice(n, max_events, replace=False))
            X_cluster  = Xc[sub_idx]
            subsampled = True
        else:
            sub_idx    = None
            X_cluster  = Xc
            subsampled = False

        # Decide which backend to attempt first. Reproducible mode forces the
        # CPU seeded-Leiden path: Louvain (CPU Blondel and GPU cuGraph alike) is
        # time-seeded and cannot be pinned.
        try_gpu = False
        if not reproducible and use_gpu is not False and GPU_CLUSTERING_AVAILABLE:
            free_vram = _vram_free_gb()
            if use_gpu is True or free_vram is None or free_vram >= vram_admission_gb:
                try_gpu = True
            else:
                log.info(
                    "  [VRAM admission] %.1f GB < %.1f GB — clustering on CPU",
                    free_vram, vram_admission_gb)

        size_label = (f"{max_events:,} (subsample of {n:,})"
                      if subsampled else f"{n:,}")

        sub_comm = None
        Q        = -1.0
        backend  = 'CPU'

        if try_gpu:
            try:
                log.info(f"  GPU cluster: {size_label} × {len(avail)}, k={k} …")
                sub_comm, Q = self._cluster_gpu(X_cluster, k)
                backend = 'GPU'
            except Exception as e:
                log.warning(
                    "  [!] GPU clustering failed (%s: %s) — CPU fallback",
                    type(e).__name__, e)
                sub_comm = None

        if sub_comm is None:
            log.info(f"  Phenograph: {size_label} × {len(avail)}, k={k} …")
            # Lazy import — drags in igraph + sklearn.community, ~1 s and
            # ~200 MB. Only relevant for callers that actually cluster.
            try:
                import phenograph
            except ImportError as e:
                raise ClusteringError(
                    "phenograph is required for CPU clustering "
                    "(pip install phenograph)") from e
            # Phenograph writes scratch files (kNN graph, .tree, _graph.weights)
            # into CWD — redirect into the project-local cache folder.
            os.makedirs(_PHENOGRAPH_CACHE_DIR, exist_ok=True)
            pg_kwargs: dict = dict(k=k, n_jobs=n_jobs)
            if reproducible:
                # PhenoGraph's default Louvain is non-deterministic; its Leiden
                # backend accepts an explicit seed. Feature-detect so an older
                # build degrades gracefully (a warning, not a crash).
                import inspect
                _pg = inspect.signature(phenograph.cluster).parameters
                if 'clustering_algo' in _pg and 'seed' in _pg:
                    pg_kwargs['clustering_algo'] = 'leiden'
                    pg_kwargs['seed'] = int(random_state)
                else:
                    log.warning(
                        "  [!] reproducible clustering needs a phenograph build "
                        "with seeded Leiden; using (non-deterministic) Louvain.")
            try:
                with contextlib.chdir(_PHENOGRAPH_CACHE_DIR):
                    sub_comm, _, Q = phenograph.cluster(X_cluster, **pg_kwargs)
            except Exception as e:
                raise ClusteringError(
                    f"Phenograph CPU clustering failed: {e}") from e
            backend = 'CPU'

        # Expand sub-sample labels back to all events (KD-tree assign for rest).
        if subsampled:
            communities          = np.full(n, -1, dtype=int)
            communities[sub_idx] = sub_comm
            good = sub_comm >= 0
            if good.sum() > 0:
                from sklearn.neighbors import NearestNeighbors
                nn_clf = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
                nn_clf.fit(X_cluster[good])
                ref_labels         = sub_comm[good]
                rest_mask          = np.ones(n, dtype=bool)
                rest_mask[sub_idx] = False
                n_rest             = int(rest_mask.sum())
                log.info(
                    "  Assigning %s remaining events to nearest cluster "
                    "(KD-tree, 1-NN) …", f"{n_rest:,}")
                _, nbr = nn_clf.kneighbors(Xc[rest_mask])
                communities[rest_mask] = ref_labels[nbr[:, 0]]
            else:
                log.warning(
                    "  [!] All sub-sample events were marked as noise — "
                    "the rest cannot be assigned.")
        else:
            communities = sub_comm

        labels       = np.full(len(self.data), -1, dtype=int)
        labels[mask] = communities
        self.data['cluster'] = labels
        self.clusters        = labels
        valid = communities[communities >= 0]
        log.info(
            "  → %d clusters, Q=%.3f [%s]",
            len(np.unique(valid)) if len(valid) else 0, Q, backend)
        return self

    def _cluster_gpu(self, X, k):
        """GPU clustering: cuML kNN graph + cuGraph Louvain community detection.

        Returns (communities ndarray of length len(X), modularity float).
        Raises on any RAPIDS / VRAM failure — the caller is responsible
        for catching and falling back to CPU.
        """
        kit     = _GPU_CLUSTER_KIT
        # caller only invokes us when GPU_CLUSTERING_AVAILABLE is True,
        # which implies kit is non-None — assert so pyright can narrow.
        assert kit is not None
        cupy    = kit['cupy']
        cudf    = kit['cudf']
        cugraph = kit['cugraph']
        CuNN    = kit['cunn']

        n     = X.shape[0]
        X_gpu = cupy.asarray(X, dtype=cupy.float32)

        # k+1 neighbours so we can drop the self-link (column 0).
        nn = CuNN(n_neighbors=k + 1)
        nn.fit(X_gpu)
        _, indices = nn.kneighbors(X_gpu)

        if hasattr(indices, 'values'):          # cuDF DataFrame on some versions
            idx_gpu = cupy.asarray(indices.values)
        else:
            idx_gpu = cupy.asarray(indices)
        idx_gpu = idx_gpu[:, 1:]                 # drop self

        src = cupy.repeat(cupy.arange(n, dtype=cupy.int32), k)
        dst = idx_gpu.flatten().astype(cupy.int32)

        edges = cudf.DataFrame({'src': src, 'dst': dst})
        G = cugraph.Graph()
        G.from_cudf_edgelist(edges, source='src', destination='dst', renumber=False)

        parts, modularity = cugraph.louvain(G)
        parts_sorted = parts.sort_values('vertex')
        communities  = parts_sorted['partition'].to_numpy().astype(np.int64)
        return communities, float(modularity)

    # ── UMAP ──────────────────────────────────────────────────────────────────

    def _embedding_input(self, channels, sample_n, random_state):
        """Shared front-end for the dimensionality-reduction backends
        (UMAP / TriMap / PaCMAP).

        Returns ``(X, sub_mask, avail)`` where ``X`` is the full float
        matrix of the available channels, ``sub_mask`` is a boolean row
        selector (finite rows, optionally sub-sampled to ``sample_n``),
        and ``avail`` is the list of channels actually used. Pure: no
        embedding is computed and nothing is written back.
        """
        if channels is None:
            channels = self.fluor_channels
        avail = [c for c in channels if c in self.data.columns]
        X     = self.data[avail].values.astype(float)
        mask  = np.all(np.isfinite(X), axis=1)

        if sample_n and mask.sum() > sample_n:
            idx      = np.where(mask)[0]
            chosen   = np.random.default_rng(random_state).choice(
                           idx, sample_n, replace=False)
            sub_mask = np.zeros(len(X), dtype=bool)
            sub_mask[chosen] = True
        else:
            sub_mask = mask
        return X, sub_mask, avail

    def _store_embedding(self, emb, sub_mask, prefix):
        """Write a 2-D embedding back as ``{prefix}1`` / ``{prefix}2``
        columns, NaN where the row wasn't embedded. Returns the embedding."""
        emb = np.asarray(emb)
        self.data[f'{prefix}1'] = np.nan
        self.data[f'{prefix}2'] = np.nan
        self.data.loc[sub_mask, f'{prefix}1'] = emb[:, 0]
        self.data.loc[sub_mask, f'{prefix}2'] = emb[:, 1]
        return emb

    def run_umap(self, channels=None, n_neighbors=30, min_dist=0.3,
                 sample_n=100_000, random_state=42):
        X, sub_mask, avail = self._embedding_input(
            channels, sample_n, random_state)

        log.info(f"  UMAP: {sub_mask.sum():,} events × {len(avail)} channels …")
        emb = None

        if GPU_AVAILABLE:
            try:
                import cudf  # type: ignore[import-not-found]
                assert _CumlUMAP is not None  # gated above by GPU_AVAILABLE
                X_gpu = cudf.DataFrame(X[sub_mask].astype(np.float32))
                emb   = np.array(
                    _CumlUMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                              random_state=random_state).fit_transform(X_gpu)
                )
                log.info(f"  UMAP complete [GPU — {GPU_NAME}].")
            except Exception as gpu_err:
                log.warning(f"  [!] GPU UMAP failed ({type(gpu_err).__name__}: {gpu_err})")
                log.warning("  [!] Retrying on CPU …")
                emb = None

        if emb is None:
            try:
                import umap as umap_lib
            except ImportError:
                log.warning("  [!] pip install umap-learn")
                return self
            # umap-learn forces n_jobs=1 when random_state is set (because
            # parallel UMAP isn't deterministic) and emits a UserWarning
            # every call. That's exactly the trade-off we want — same
            # seed → same embedding across runs / samples / restarts —
            # so silence the noise.
            warnings.filterwarnings(
                'ignore',
                message='.*n_jobs value.*overridden.*',
                category=UserWarning)
            emb = np.asarray(
                umap_lib.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                              random_state=random_state).fit_transform(X[sub_mask])
            )
            log.info("  UMAP complete [CPU].")

        self.umap_coords = self._store_embedding(emb, sub_mask, 'UMAP')
        return self

    def run_trimap(self, channels=None, sample_n=100_000, random_state=42,
                   **trimap_kwargs):
        """TriMap embedding — a triplet-constraint alternative to UMAP that
        tends to preserve global structure better. CPU-only (the ``trimap``
        package has no GPU path). Writes ``TRIMAP1`` / ``TRIMAP2``.

        Install with ``pip install openflo[embed]`` (or ``pip install
        trimap``). Extra keyword args pass through to ``trimap.TRIMAP``.
        """
        X, sub_mask, avail = self._embedding_input(
            channels, sample_n, random_state)
        try:
            import trimap  # type: ignore[import-not-found]
        except ImportError:
            log.warning("  [!] TriMap not installed — pip install openflo[embed]")
            return self
        log.info(f"  TriMap: {sub_mask.sum():,} events × {len(avail)} channels …")
        emb = trimap.TRIMAP(**trimap_kwargs).fit_transform(X[sub_mask])
        self.trimap_coords = self._store_embedding(emb, sub_mask, 'TRIMAP')
        log.info("  TriMap complete [CPU].")
        return self

    def run_pacmap(self, channels=None, n_neighbors=None, sample_n=100_000,
                   random_state=42, **pacmap_kwargs):
        """PaCMAP embedding — another global-structure-preserving
        alternative to UMAP. CPU-only. Writes ``PACMAP1`` / ``PACMAP2``.

        Install with ``pip install openflo[embed]`` (or ``pip install
        pacmap``). Extra keyword args pass through to ``pacmap.PaCMAP``.
        """
        X, sub_mask, avail = self._embedding_input(
            channels, sample_n, random_state)
        try:
            import pacmap  # type: ignore[import-not-found]
        except ImportError:
            log.warning("  [!] PaCMAP not installed — pip install openflo[embed]")
            return self
        log.info(f"  PaCMAP: {sub_mask.sum():,} events × {len(avail)} channels …")
        reducer = pacmap.PaCMAP(  # pyright: ignore[reportArgumentType]  # untyped optional dep; None = auto
            n_neighbors=n_neighbors, random_state=random_state, **pacmap_kwargs)
        emb = reducer.fit_transform(X[sub_mask])
        self.pacmap_coords = self._store_embedding(emb, sub_mask, 'PACMAP')
        log.info("  PaCMAP complete [CPU].")
        return self

    def run_tsne(self, channels=None, perplexity=30.0, sample_n=50_000,
                 random_state=42, **tsne_kwargs):
        """t-SNE embedding (scikit-learn, a core dependency). Writes
        ``TSNE1`` / ``TSNE2``. t-SNE is O(n log n) but still heavier than UMAP,
        so it subsamples to ``sample_n`` events; ``perplexity`` is clamped below
        the sample size. Extra kwargs pass through to ``sklearn.manifold.TSNE``.
        """
        from sklearn.manifold import TSNE
        X, sub_mask, avail = self._embedding_input(
            channels, sample_n, random_state)
        n = int(sub_mask.sum())
        if n < 5:
            log.warning("  [t-SNE] too few events — skipped.")
            return self
        perp = float(min(perplexity, max(5.0, (n - 1) / 3.0)))
        log.info(f"  t-SNE: {n:,} events × {len(avail)} channels "
                 f"(perplexity {perp:.0f}) …")
        emb = TSNE(n_components=2, perplexity=perp,
                   random_state=random_state, init='pca',
                   **tsne_kwargs).fit_transform(X[sub_mask])
        self.tsne_coords = self._store_embedding(np.asarray(emb), sub_mask,
                                                 'TSNE')
        log.info("  t-SNE complete [CPU].")
        return self

    def run_phate(self, channels=None, sample_n=50_000, random_state=42,
                  **phate_kwargs):
        """PHATE embedding — a diffusion-based method that preserves continuous
        / trajectory structure especially well (complements the trajectory
        tool). Writes ``PHATE1`` / ``PHATE2``. Optional dependency: install with
        ``pip install openflo[embed]`` (or ``pip install phate``). Extra kwargs
        pass through to ``phate.PHATE``."""
        X, sub_mask, avail = self._embedding_input(
            channels, sample_n, random_state)
        try:
            import phate  # type: ignore[import-not-found]
        except ImportError:
            log.warning("  [!] PHATE not installed — pip install openflo[embed]")
            return self
        log.info(f"  PHATE: {sub_mask.sum():,} events × {len(avail)} channels …")
        emb = phate.PHATE(random_state=random_state,
                          verbose=False, **phate_kwargs).fit_transform(
                              X[sub_mask])
        self.phate_coords = self._store_embedding(np.asarray(emb), sub_mask,
                                                  'PHATE')
        log.info("  PHATE complete [CPU].")
        return self

    # ── Plotting ──────────────────────────────────────────────────────────────

    def _resolve(self, name):
        if name in self.data.columns:
            return name
        for det, lbl in self.channel_labels.items():
            if lbl.lower() == name.lower():
                return det
        raise KeyError(f"Channel '{name}' not found.")

    def plot(self, x, y, color_by='density', sample_n=50_000,
             ax=None, title=None, s=1.0, alpha=0.5):
        """
        Scatter plot. x/y accept detector name, stain label, UMAP1, UMAP2.
        color_by: 'density' | 'cluster' | channel name/label
        """
        import matplotlib.pyplot as plt  # lazy: see module-top comment
        xch = self._resolve(x)
        ych = self._resolve(y)
        df  = self.data.dropna(subset=[xch, ych])
        if sample_n and len(df) > sample_n:
            df = df.sample(sample_n, random_state=42)
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))
        xv = np.asarray(df[xch].values)
        yv = np.asarray(df[ych].values)

        if color_by == 'cluster' and 'cluster' in df.columns:
            sc = ax.scatter(xv, yv, c=np.asarray(df['cluster'].values),
                            cmap='tab20', s=s, alpha=alpha, linewidths=0)
            plt.colorbar(sc, ax=ax, label='Cluster')
        elif color_by == 'density':
            # FlowJo-style O(n) histogram density: bin events into a
            # 256x256 grid, smooth the bin counts, look up each event's
            # density by its bin index. Replaces a gaussian_kde call
            # that was O(n^2) and took minutes on >100k events.
            try:
                from scipy.ndimage import gaussian_filter
                xv_f = np.asarray(xv, dtype=float)
                yv_f = np.asarray(yv, dtype=float)
                finite = np.isfinite(xv_f) & np.isfinite(yv_f)
                xv_f = xv_f[finite]; yv_f = yv_f[finite]
                if xv_f.size == 0:
                    raise ValueError("no finite points")
                BINS = 256
                hist, x_edges, y_edges = np.histogram2d(xv_f, yv_f, bins=BINS)
                hist = gaussian_filter(hist, sigma=1.5)
                ix = np.clip(np.searchsorted(x_edges, xv_f, side='right') - 1,
                             0, BINS - 1)
                iy = np.clip(np.searchsorted(y_edges, yv_f, side='right') - 1,
                             0, BINS - 1)
                z   = hist[ix, iy]
                idx = z.argsort()
                ax.scatter(xv_f[idx], yv_f[idx], c=z[idx],
                           cmap='jet', s=s, alpha=alpha, linewidths=0,
                           rasterized=True)
            except Exception:
                ax.scatter(xv, yv, s=s, alpha=alpha, color='steelblue')
        else:
            try:
                cch = self._resolve(color_by)
                col = df[cch]
                # Categorical (string) column — e.g. 'sample_origin' on a
                # concatenated dataset. matplotlib can't take string values
                # for `c`; factorise to integer codes and draw a discrete
                # legend instead of a colorbar.
                if (col.dtype == object
                        or pd.api.types.is_string_dtype(col)
                        or isinstance(col.dtype, pd.CategoricalDtype)):
                    codes, uniques = pd.factorize(col, sort=True)
                    n_groups = max(1, len(uniques))
                    # tab10 has 10 maximally-distinct colours; tab20
                    # alternates same-hue light/dark pairs so its first
                    # two entries are both blues — bad default when the
                    # user only has 2 samples / conditions on one plot.
                    palette_name = getattr(self, '_palette_name', None) \
                                    or _DEFAULT_CATEGORICAL_PALETTE
                    if palette_name == 'auto':
                        palette_name = ('tab10' if n_groups <= 10
                                        else 'tab20' if n_groups <= 20
                                        else 'gist_ncar')
                    cmap = plt.get_cmap(palette_name)
                    sc = ax.scatter(xv, yv, c=codes, cmap=cmap,
                                    vmin=0, vmax=max(n_groups - 1, 1),
                                    s=s, alpha=alpha, linewidths=0)
                    import matplotlib.patches as mpatches
                    handles = [
                        mpatches.Patch(color=cmap((i / max(n_groups - 1, 1))
                                                  if n_groups > 1 else 0.0),
                                       label=str(u))
                        for i, u in enumerate(uniques)
                    ]
                    ax.legend(handles=handles, title=color_by, loc='best',
                              fontsize=8, framealpha=0.8)
                else:
                    sc = ax.scatter(xv, yv, c=np.asarray(col.values),
                                    cmap='viridis',
                                    s=s, alpha=alpha, linewidths=0)
                    plt.colorbar(sc, ax=ax,
                                 label=self.channel_labels.get(cch, cch))
            except KeyError:
                ax.scatter(xv, yv, s=s, alpha=alpha, color='steelblue')

        ax.set_xlabel(self.channel_labels.get(xch, xch))
        ax.set_ylabel(self.channel_labels.get(ych, ych))
        ax.set_title(title or self.name)
        plt.tight_layout()
        return ax

    def plot_umap(self, color_by='cluster', **kwargs):
        return self.plot('UMAP1', 'UMAP2', color_by=color_by, **kwargs)

    def cluster_heatmap(self, channels=None):
        if 'cluster' not in self.data.columns:
            log.warning("  [!] Run .cluster() first.")
            return
        import matplotlib.pyplot as plt  # lazy: see module-top comment
        if channels is None:
            channels = self.fluor_channels
        avail  = [c for c in channels if c in self.data.columns]
        labels = [self.channel_labels.get(c, c) for c in avail]
        # set_axis avoids the .rename(columns=Mapping) overload that pandas-stubs
        # can't resolve cleanly off a chained groupby().median() expression.
        med    = (self.data.groupby('cluster')[avail].median()
                           .set_axis(labels, axis=1))
        _, ax = plt.subplots(figsize=(max(6, len(avail)),
                                      max(4, len(med) * 0.4)))
        import seaborn as sns  # lazy: only when plotting
        sns.heatmap(med, cmap='vlag', center=0, ax=ax,
                    linewidths=0.3, linecolor='grey')
        ax.set_title(f'{self.name} — Cluster Median Expression')
        plt.tight_layout()
        return ax

    # ── Stats & export ────────────────────────────────────────────────────────

    def cluster_frequencies(self):
        if 'cluster' not in self.data.columns:
            return pd.DataFrame()
        total  = len(self.data)
        counts = self.data['cluster'].value_counts().sort_index()
        meds   = self.data.groupby('cluster')[self.fluor_channels].median()
        meds.columns = [f'median_{self.channel_labels.get(c,c)}' for c in meds.columns]
        count_arr = np.asarray(counts.values)
        df = pd.DataFrame(dict(sample=self.name, cluster=counts.index,
                               count=count_arr,
                               pct_total=count_arr/total*100)).set_index('cluster')
        return df.join(meds).reset_index()

    # ── FlowSOM ───────────────────────────────────────────────────────────────

    def run_flowsom(self, channels=None, grid=(10, 10), n_metaclusters=10,
                    iters=10, max_events=50_000, seed=42):
        """FlowSOM clustering: train a SOM over the marker space, assign each
        event to a node, then agglomerate nodes into metaclusters. Writes a
        ``flowsom`` (node id) and ``flowsom_meta`` (metacluster id) column;
        non-finite rows get -1. Stores the model in ``self.flowsom_result``.

        Lighter than the R FlowSOM but the same structure — fast, CPU-only,
        good for very large files. Returns self."""
        channels = channels or self.fluor_channels
        avail = [c for c in channels if c in self.data.columns]
        if not avail:
            log.warning("  [FlowSOM] no usable channels — skipped.")
            return self
        X = self.data[avail].values.astype(float)
        mask = np.all(np.isfinite(X), axis=1)
        Xf = X[mask]
        if Xf.shape[0] < max(n_metaclusters, np.prod(grid)):
            log.warning("  [FlowSOM] too few events — skipped.")
            return self

        log.info("  FlowSOM: %s events × %d channels, grid %dx%d …",
                 f"{Xf.shape[0]:,}", len(avail), grid[0], grid[1])
        W, _coords = _som_train(Xf, grid=grid, iters=iters,
                                max_events=max_events, seed=seed)
        nodes = _som_assign(Xf, W)
        meta_of_node = _som_metacluster(W, n_metaclusters)
        meta = meta_of_node[nodes]

        node_col = np.full(len(self.data), -1, dtype=int)
        meta_col = np.full(len(self.data), -1, dtype=int)
        node_col[mask] = nodes
        meta_col[mask] = meta
        self.data['flowsom'] = node_col
        self.data['flowsom_meta'] = meta_col
        n_meta = len(np.unique(meta))
        self.flowsom_result = {
            'grid': grid, 'n_nodes': int(np.prod(grid)),
            'n_metaclusters': int(n_meta), 'channels': avail, 'weights': W}
        log.info("  FlowSOM complete → %d nodes, %d metaclusters.",
                 int(np.prod(grid)), n_meta)
        return self

    def run_leiden(self, channels=None, n_neighbors=15, resolution=1.0,
                   max_events=200_000, random_state=42):
        """Leiden community detection — the current standard for high-dimensional
        spectral cytometry.

        Builds a symmetric k-nearest-neighbour graph over the marker space and
        partitions it with the Leiden algorithm (RBConfiguration objective;
        ``resolution`` tunes granularity — higher gives more, smaller clusters).
        Writes a ``leiden`` column (non-finite rows → -1). Like ``cluster``,
        very large samples are graph-partitioned on a random subsample of up to
        ``max_events`` events and the rest assigned to their nearest labelled
        neighbour (KD-tree). Requires ``igraph`` + ``leidenalg`` (declared
        dependencies). Returns self."""
        try:
            import igraph as ig
            import leidenalg
        except ImportError as e:
            raise ClusteringError(
                "igraph + leidenalg are required for Leiden clustering "
                "(pip install igraph leidenalg)") from e
        from sklearn.neighbors import NearestNeighbors, kneighbors_graph

        channels = channels or self.fluor_channels
        avail = [c for c in channels if c in self.data.columns]
        if not avail:
            log.warning("  [Leiden] no usable channels — skipped.")
            return self
        X = self.data[avail].values.astype(float)
        mask = np.all(np.isfinite(X), axis=1)
        Xc = X[mask]
        n = len(Xc)
        if n < 3:
            log.warning("  [Leiden] too few finite events — skipped.")
            return self

        if max_events and n > max_events:
            rng = np.random.default_rng(random_state)
            sub_idx = np.sort(rng.choice(n, max_events, replace=False))
            X_cluster = Xc[sub_idx]
            subsampled = True
        else:
            sub_idx = None
            X_cluster = Xc
            subsampled = False

        k = int(max(1, min(n_neighbors, len(X_cluster) - 1)))
        log.info("  Leiden: %s events × %d channels, k=%d, resolution=%.2f …",
                 f"{len(X_cluster):,}", len(avail), k, resolution)
        # Shared-nearest-neighbour (Jaccard) edge weights — the Phenograph /
        # Seurat construction. A plain binary kNN graph makes modularity-style
        # objectives over-split uniform blobs; Jaccard weighting sharpens real
        # communities so the partition tracks the populations.
        a = kneighbors_graph(X_cluster, k, mode='connectivity',
                             include_self=True)
        inter = (a @ a.T).tocoo()                # |N(i) ∩ N(j)|
        deg = np.asarray(a.sum(axis=1)).ravel()  # |N(i)| = k+1
        keep = inter.row < inter.col             # upper triangle, no self loops
        ii = inter.row[keep]
        jj = inter.col[keep]
        shared = inter.data[keep]
        union = deg[ii] + deg[jj] - shared
        w = shared / np.maximum(union, 1e-9)
        good = w > 0
        edges = list(zip(ii[good].tolist(), jj[good].tolist(), strict=True))
        g = ig.Graph(n=int(X_cluster.shape[0]), edges=edges, directed=False)
        g.es['weight'] = w[good].tolist()
        part = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition, weights='weight',
            resolution_parameter=float(resolution), seed=int(random_state))
        sub_comm = np.asarray(part.membership, dtype=int)

        if subsampled:
            communities = np.full(n, -1, dtype=int)
            communities[sub_idx] = sub_comm
            nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(
                X_cluster)
            rest_mask = np.ones(n, dtype=bool)
            rest_mask[sub_idx] = False
            if rest_mask.any():
                _, nbr = nn.kneighbors(Xc[rest_mask])
                communities[rest_mask] = sub_comm[nbr[:, 0]]
        else:
            communities = sub_comm

        labels = np.full(len(self.data), -1, dtype=int)
        labels[mask] = communities
        self.data['leiden'] = labels
        n_clusters = len(np.unique(communities[communities >= 0]))
        log.info("  Leiden complete → %d clusters (resolution=%.2f).",
                 n_clusters, resolution)
        return self

    # ── Cell cycle ──────────────────────────────────────────────────────────

    def _dna_width_partner(self, dna_channel):
        """The width (`-W`) partner of an area (`-A`) DNA channel, if the
        FCS carries one (used for doublet exclusion). None otherwise."""
        cu = dna_channel.upper()
        if not cu.endswith('-A'):
            return None
        stem = dna_channel[:-2]
        for cand in (stem + '-W', stem + '-H'):
            if cand in self.data.columns:
                return cand
        return None

    def cell_cycle(self, dna_channel=None, k=1.5, singlet_channel=None,
                   singlet_tol=0.25):
        """DNA-content cell-cycle analysis.

        Auto-detects a DNA-stain channel (PI / DAPI / FxCycle / 7-AAD /
        Hoechst / DRAQ5 / …) when `dna_channel` is None; a label is
        accepted too. Doublet exclusion matters here (a G1 doublet lands at
        the G2/M position), so when a `-W`/`-H` partner exists (or
        `singlet_channel` is given) we pre-gate singlets on the DNA-A vs
        width ratio before modelling.

        Writes a categorical ``cell_cycle`` column (G1 / S / G2M / sub-G1 /
        >G2M / NA) and stores the model in ``self.cell_cycle_result``.
        Returns self."""
        col = dna_channel
        if col is None:
            col = find_dna_channel(self)
        elif col not in self.data.columns:
            try:
                col = self._resolve(col)
            except KeyError:
                col = None
        if not col or col not in self.data.columns:
            log.warning("  [cell-cycle] no DNA channel found — skipped.")
            self.cell_cycle_result = None
            return self

        vals = np.asarray(self.data[col].values, dtype=float)
        singlet = np.ones(len(self.data), dtype=bool)
        wcol = singlet_channel or self._dna_width_partner(col)
        if wcol and wcol in self.data.columns:
            w = np.asarray(self.data[wcol].values, dtype=float)
            ratio = np.where(w > 0, vals / w, np.nan)
            good  = np.isfinite(ratio)
            med   = float(np.nanmedian(ratio[good])) if good.any() else 0.0
            lo, hi = med * (1 - singlet_tol), med * (1 + singlet_tol)
            singlet = good & (ratio >= lo) & (ratio <= hi)

        model  = analyze_dna(vals[singlet], k=k)
        phases = np.full(len(self.data), 'NA', dtype=object)
        phases[singlet] = assign_phase(vals[singlet], model)
        self.data['cell_cycle'] = phases

        model = dict(model)
        model['channel'] = col
        model['width_channel'] = wcol
        model['n_singlet'] = int(singlet.sum())
        self.cell_cycle_result = model
        log.info(
            "  [cell-cycle] %s: G1 %.1f%% / S %.1f%% / G2M %.1f%% "
            "(singlets=%s/%s)",
            col, model.get('pct_g1', float('nan')),
            model.get('pct_s', float('nan')), model.get('pct_g2m', float('nan')),
            f"{model['n_singlet']:,}", f"{len(self.data):,}")
        return self

    def export_csv(self, path=None):
        if path is None:
            path = f"{self.name}_processed.csv"
        self.data.to_csv(path, index=False)
        log.info(f"  Exported → {path}")
        return self

    def export_stats(self, path=None):
        freq = self.cluster_frequencies()
        if path is None:
            path = f"{self.name}_cluster_stats.csv"
        freq.to_csv(path, index=False)
        log.info(f"  Stats → {path}")
        return freq


def write_fcs(path, df, channels=None, channel_labels=None):
    """Write a DataFrame of events to an FCS 3.1 file (via FlowIO).

    ``channels`` selects/orders the parameters written (default: every column).
    ``channel_labels`` (``{column: antibody}``) populates the per-parameter
    ``$PnS`` marker names, so the file re-opens in FlowJo / FCS Express with
    labels intact. Non-finite cells are zeroed (FCS stores finite floats).
    Returns the number of events written.

    Used to export a gated subpopulation as a standalone, re-importable FCS —
    pass the events you want (e.g. ``sample.raw.iloc[mask]`` for raw detector
    values) and the channels to keep."""
    channels = [str(c) for c in (channels if channels else list(df.columns))]
    mat = np.nan_to_num(df[channels].to_numpy(dtype=float),
                        nan=0.0, posinf=0.0, neginf=0.0)
    opt = ([str(channel_labels.get(c, '') or '') for c in channels]
           if channel_labels else None)
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, mat.flatten().tolist(), channels,
                          opt_channel_names=opt)
    return len(mat)


# ══════════════════════════════════════════════════════════════════════════════
# CONCATENATION
# ══════════════════════════════════════════════════════════════════════════════

def concatenate(samples, label_col='sample_origin'):
    """
    Merge multiple FlowSample.data DataFrames, tagging each row with
    the sample name. Returns a single DataFrame.
    """
    frames = []
    for s in samples:
        df = s.data.copy()
        df[label_col] = s.name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    log.info(f"Concatenated {len(samples)} samples → {len(out):,} events.")
    return out


# ── Cross-sample label alignment ───────────────────────────────────────────────
#
# The same antibody (e.g. CD11b) can sit on a different fluorophore /
# detector in different samples or on different days. Comparing or
# clustering ACROSS samples by detector name therefore mis-aligns
# phenotypes. These helpers align by the antibody LABEL instead, while
# compensation stays keyed on detectors (each sample compensates its own
# $SPILL upstream — labels never touch the comp math).

def _sample_fluor_labels(sample):
    """{label: detector} for one sample's fluor channels. The label is
    the antibody name (channel_labels), falling back to the detector
    name when no label is set. Later detectors win on a label clash
    (rare; logged by callers if it matters)."""
    labels = getattr(sample, 'channel_labels', {}) or {}
    out = {}
    for det in getattr(sample, 'fluor_channels', []) or []:
        lbl = labels.get(det, det) or det
        out[lbl] = det
    return out


def align_fluor_labels(samples):
    """Align a set of samples by antibody label.

    Parameters
    ----------
    samples : iterable of FlowSample-like
        Each needs ``.name``, ``.fluor_channels`` (detector names) and
        ``.channel_labels`` ({detector: antibody label}).

    Returns
    -------
    dict with:
      ``common``      ordered list of labels present as a fluor in EVERY
                      sample (ordered by the first sample's channel order)
      ``per_sample``  {sample_name: {label: detector}}
      ``missing``     {label: [sample_names lacking it]} for any label
                      present in some-but-not-all samples
      ``all_labels``  ordered union of every label seen
    """
    samples = list(samples)
    per_sample = {}
    order = []                       # first-seen label order
    seen = set()
    label_to_samples = {}            # label -> set(names) that have it
    for s in samples:
        name = getattr(s, 'name', None) or f'sample{len(per_sample)}'
        l2d = _sample_fluor_labels(s)
        per_sample[name] = l2d
        for lbl in l2d:
            if lbl not in seen:
                seen.add(lbl)
                order.append(lbl)
            label_to_samples.setdefault(lbl, set()).add(name)

    n = len(samples)
    common = [lbl for lbl in order if len(label_to_samples.get(lbl, ())) == n]
    missing = {
        lbl: sorted(set(per_sample) - label_to_samples[lbl])
        for lbl in order
        if 0 < len(label_to_samples[lbl]) < n
    }
    return {'common': common, 'per_sample': per_sample,
            'missing': missing, 'all_labels': order}


def relabel_gate_for_sample(gate, label_to_detector):
    """Retarget a gate's channel fields to a specific sample's detectors
    by antibody label.

    A template gate carries both a detector channel (e.g. ``BV421-A``)
    and, when authored in a labelled editor, the antibody label it stood
    for (``x_label`` / ``y_label`` / ``label``). Applied to a sample
    where that marker sits on a DIFFERENT detector, we rewrite the
    channel to that sample's detector so one template ties phenotypes
    across panels. Compensation is unaffected — this only swaps which
    column the gate reads.

    label_to_detector : {antibody label: detector} for the target sample
        (e.g. from ``_sample_fluor_labels``).

    Returns a shallow copy with channel fields remapped where a stored
    label resolves in the target sample; fields without a label, or
    whose label isn't present in the sample, are left as-is (the gate
    then reads its original detector, or no-ops with a warning if that
    detector is also absent).
    """
    if not label_to_detector:
        return dict(gate)
    g = dict(gate)
    for chan_field, label_field in (('channel', 'label'),
                                    ('x_channel', 'x_label'),
                                    ('y_channel', 'y_label')):
        lbl = g.get(label_field)
        if lbl and lbl in label_to_detector:
            g[chan_field] = label_to_detector[lbl]
    return g


def common_fluor_warning(samples):
    """Human-readable warning when samples don't share a common fluor
    label set, or '' when they're all consistent. Lists which labels are
    missing from which samples and notes that cross-sample analysis uses
    the common (intersection) set."""
    info = align_fluor_labels(samples)
    if not info['missing']:
        return ''
    lines = [f"  • {lbl}: missing from {', '.join(names)}"
             for lbl, names in info['missing'].items()]
    common = ', '.join(info['common']) or '(none)'
    return ("Samples don't share a common fluor panel. Cross-sample "
            "analysis will use only the common labels:\n"
            f"  common: {common}\n"
            "Non-common labels:\n" + '\n'.join(lines))


def concatenate_by_label(samples, label_col='sample_origin'):
    """Like :func:`concatenate`, but first renames each sample's fluor
    columns to their antibody LABEL and keeps ONLY the labels common to
    every sample — so a marker on different fluors across samples lines
    up into one column. Scatter / non-fluor columns are dropped from the
    merged frame (clustering uses fluor labels). Compensation already
    happened per sample upstream on detectors, so this is purely a
    rename-and-intersect for cross-sample clustering.

    Returns (merged_df, common_labels). merged_df has the common label
    columns + `label_col`; common_labels is the ordered label list used.
    """
    info = align_fluor_labels(samples)
    common = info['common']
    frames = []
    for s in samples:
        l2d = info['per_sample'].get(getattr(s, 'name', ''), {})
        # detector for each common label in THIS sample
        cols = {lbl: l2d[lbl] for lbl in common if lbl in l2d}
        sub = s.data[list(cols.values())].copy()
        sub.columns = list(cols.keys())          # rename detector → label
        sub[label_col] = s.name
        frames.append(sub)
    if not frames:
        return pd.DataFrame(), common
    out = pd.concat(frames, ignore_index=True)
    log.info("Concatenated %d samples by label → %d events, %d common "
             "fluor label(s): %s", len(frames), len(out), len(common),
             ', '.join(common) or '(none)')
    return out, common


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT (multi-sample)
# ══════════════════════════════════════════════════════════════════════════════

class FlowExperiment:
    """
    Batch-process multiple FCS files.

    Usage
    -----
        exp = FlowExperiment('/path/to/fcs/')
        exp.exclude_pattern('unstained|bead|fmo')
        exp.run_all(k=30, umap=True)
        exp.compare_conditions(
            groupA=['sample_1','sample_2'], groupB=['sample_3','sample_4'],
            label_a='Group A',              label_b='Group B')
        exp.export_all('results/')
    """

    def __init__(self, source):
        self.samples = {}
        for p in self._resolve(source):
            try:
                s = FlowSample(p)
                self.samples[s.name] = s
            except Exception as e:
                log.warning(f"  [!] {p}: {e}")
        log.info(f"\nLoaded {len(self.samples)} sample(s).")

    @staticmethod
    def _resolve(source):
        if isinstance(source, (list, tuple)):
            return list(source)
        if os.path.isdir(source):
            return sorted([os.path.join(source, f)
                           for f in os.listdir(source)
                           if f.lower().endswith('.fcs')])
        raise ValueError("source must be a directory or list of paths.")

    def exclude_pattern(self, pattern):
        rx  = re.compile(pattern, re.IGNORECASE)
        rem = [n for n in self.samples if rx.search(n)]
        for n in rem:
            del self.samples[n]
        log.info(f"Excluded {len(rem)}: {rem}")
        return self

    def keep_pattern(self, pattern):
        rx  = re.compile(pattern, re.IGNORECASE)
        rem = [n for n in self.samples if not rx.search(n)]
        for n in rem:
            del self.samples[n]
        log.info(f"Kept {len(self.samples)}, removed {len(rem)}.")
        return self

    def run_all(self, qc=True, compensate=True, wsp_path=None,
                transform=True, transform_method='logicle',
                cluster=True, k=30, umap=False):
        for name, s in self.samples.items():
            log.info(f"\n── {name}")
            if qc:           s.run_qc()
            if compensate:
                if wsp_path: s.compensate_from_wsp(wsp_path)
                else:        s.auto_compensate()
            if transform:    s.apply_transform(method=transform_method)
            if cluster:      s.cluster(k=k)
            if umap:         s.run_umap()
        return self

    def combined_frequencies(self):
        frames = [s.cluster_frequencies() for s in self.samples.values()
                  if 'cluster' in s.data.columns]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def compare_conditions(self, groupA, groupB,
                            label_a='A', label_b='B', plot=True):
        freq = self.combined_frequencies()
        if freq.empty:
            log.warning("  [!] No cluster data — run clustering first.")
            return freq
        freq['condition'] = freq['sample'].apply(
            lambda n: label_a if n in groupA else (label_b if n in groupB else 'other'))
        freq = freq[freq['condition'] != 'other']
        summary = (freq.groupby(['condition','cluster'])['pct_total']
                       .agg(['mean','std']).reset_index()
                       .rename(columns={'mean':'mean_pct','std':'sd_pct'}))
        if plot:
            import matplotlib.pyplot as plt  # lazy: see module-top comment
            clusters = sorted(summary['cluster'].unique())
            a_v = (summary[summary.condition==label_a]
                          .set_index('cluster')['mean_pct']
                          .reindex(clusters, fill_value=0))
            b_v = (summary[summary.condition==label_b]
                          .set_index('cluster')['mean_pct']
                          .reindex(clusters, fill_value=0))
            x, w = np.arange(len(clusters)), 0.35
            fig, ax = plt.subplots(figsize=(max(8, len(clusters)*0.5), 5))
            ax.bar(x-w/2, a_v, w, label=label_a, color='steelblue', alpha=0.8)
            ax.bar(x+w/2, b_v, w, label=label_b, color='coral',     alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([f'C{c}' for c in clusters], rotation=45)
            ax.set_ylabel('% of total events')
            ax.set_title(f'{label_a} vs {label_b} — cluster frequencies')
            ax.legend()
            plt.tight_layout()
        return summary

    def plot_all(self, x, y, color_by='cluster', ncols=3, sample_n=30_000):
        import matplotlib.pyplot as plt  # lazy: see module-top comment
        names  = list(self.samples.keys())
        nrows  = (len(names)+ncols-1)//ncols
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(6*ncols, 5*nrows))
        axes = np.array(axes).flatten()
        # i defaults to -1 so that when `names` is empty, the post-loop
        # range(i+1, len(axes)) hides every axis (otherwise i is unbound).
        i = -1
        for i, name in enumerate(names):
            self.samples[name].plot(x, y, color_by=color_by,
                                    sample_n=sample_n, ax=axes[i])
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        return fig

    def concatenate_group(self, names):
        return concatenate([self.samples[n] for n in names if n in self.samples])

    def export_all(self, out_dir='.'):
        os.makedirs(out_dir, exist_ok=True)
        for name, s in self.samples.items():
            s.export_csv(os.path.join(out_dir, f"{name}_processed.csv"))
            if 'cluster' in s.data.columns:
                s.export_stats(os.path.join(out_dir, f"{name}_stats.csv"))
        return self


# ══════════════════════════════════════════════════════════════════════════════
# LAZY IMPORTS  (PEP 562)
# ══════════════════════════════════════════════════════════════════════════════
#
# Old code wrote ``sns.heatmap(...)`` and ``gaussian_kde(...)`` after the
# module-level ``import seaborn as sns`` and ``from scipy.stats import
# gaussian_kde``. Those imports are now deferred — the hook below resolves
# them the first time a function inside this module touches the name.
#
# Phenograph isn't exposed this way; FlowSample.cluster() imports it
# locally, which is fine because it's a single call site.

_LAZY = {
    'sns':           ('seaborn',          None),
    'gaussian_kde':  ('scipy.stats',      'gaussian_kde'),
    'phenograph':    ('phenograph',       None),
    'plt':           ('matplotlib.pyplot', None),
}


def __getattr__(name):
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = spec
    import importlib
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, attr) if attr else mod
    globals()[name] = obj      # cache so subsequent lookups skip the hook
    return obj


# ══════════════════════════════════════════════════════════════════════════════
# QUICK-START
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── Single sample ──────────────────────────────────────────────────────
    # s = FlowSample('sample_1.fcs')
    # s.run_qc()
    # s.auto_compensate()            # spillover from FCS
    # s.apply_transform()
    # s.cluster(k=30)
    # s.run_umap()
    # s.plot_umap(color_by='cluster')
    # s.cluster_heatmap()
    # plt.show()

    # ── WSP compensation ───────────────────────────────────────────────────
    # s = FlowSample('sample_1.fcs')
    # s.compensate_from_wsp('experiment.wsp')

    # ── FMO gating ─────────────────────────────────────────────────────────
    # gater = FMOGater()
    # gater.add_fmo('Comp-BV421-A', 'fmo_bv421.fcs')  # CD11b
    # gater.add_fmo('Comp-APC-A',           'fmo_apc.fcs')   # CD34
    # gater.add_fmo('Comp-PE-Cy7-A',        'fmo_cy7.fcs')   # CD45
    # gater.prepare()          # compensate + transform FMOs before thresholding
    # thresholds = gater.compute(percentile=99.5)
    # s.apply_threshold_gates(thresholds)

    # ── Full experiment ────────────────────────────────────────────────────
    # exp = FlowExperiment('/path/to/fcs/')
    # exp.exclude_pattern('unstained|bead|fmo')
    # exp.run_all(k=30, umap=True)
    # exp.compare_conditions(
    #     groupA=['sample_1','sample_2'],
    #     groupB=['sample_3','sample_4'],
    #     label_a='Group A', label_b='Group B')
    # exp.export_all('results/')
    # plt.show()

    # ── WSP reader standalone ──────────────────────────────────────────────
    # reader = WspReader('experiment.wsp')
    # reader.print_matrices()
    # m = reader.get_matrix()

    print("Import this module or uncomment example blocks to run.")
