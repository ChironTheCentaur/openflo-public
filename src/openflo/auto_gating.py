"""Automated density-based gating: suggest a 1-D threshold (``auto_threshold``),
a polygon around the main 2-D density mode (``auto_polygon_gate``), a singlet
gate (``auto_singlet_gate``) or Gaussian-mixture ellipses
(``gmm_ellipse_gates``) from the data, plus ``resolution_bins``, the
histogram bin count the auto-clean filters share.

Core module: numpy, with scipy, contourpy, matplotlib and scikit-learn
imported where they are used. No Tk, and it does not import
``openflo.pipeline`` (pipeline imports it).
"""

import numpy as np

from .errors import note
from .robust import MAD_TO_SD
from .robust import mad as _mad


def resolution_bins(v, lo, hi, bins):
    """Bin count capped at the number of DISTINCT values in range.

    A channel cannot fill more bins than it has values. An integer detector
    ($DATATYPE I) whose bulk spans a few tens of ADC steps, spread over 256
    bins, becomes a COMB: most bins are structurally empty, the smoothing —
    which is measured in BINS, not data units — cannot close gaps several bins
    wide, and peak finding reads two adjacent teeth as two modes with a zero
    between them. That empty bin trivially satisfies any "is the valley deep
    enough" test.

    Measured: an all-live, unimodal channel of 43 integer levels produced a
    valley at its own median, and the auto-clean viability filter deleted 46%
    of the sample. The identical data unrounded returned no valley at all.
    """
    in_range = v[(v >= lo) & (v <= hi)]
    distinct = int(np.unique(in_range).size)
    return int(max(8, min(int(bins), distinct)))


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
    bins = resolution_bins(v, lo, hi, bins)
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
    mad = _mad(ratio, med)
    sigma = MAD_TO_SD * mad if mad > 0 else float(np.std(ratio))
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
        except Exception as exc:  # noqa: BLE001 - noted; a component count the mixture cannot fit is skipped
            note("fitting a mixture model (skipping this component count)", exc)
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
