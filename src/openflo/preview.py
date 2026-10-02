"""
preview.py — interactive scatter preview for a single sample.

Usage:
    python -m openflo.preview --fcs <fcs-dir> --sample <sample-name>
    python -m openflo.preview --fcs <fcs-dir> --sample <sample-name> --n 30000 --group late
"""

import argparse
import os
import sys

# Force UTF-8 stdout/stderr — avoid cp1252 crashes on Windows when printing
# non-ASCII characters such as ≥, →, …, em-dashes.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[attr-defined]
except Exception:
    pass

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .cli import parse_labels
from .pipeline import FlowSample, FMOGater

# ── channel → display label (since FCS has no PnS) ────────────────────────────
LABELS = {
    'BV421-A': 'CD11b (BV421)',
    'APC-A':           'CD34 (APC)',
    'PE-Cy7-A':        'CD45 (PE-Cy7)',
}

# Example FMO-control basenames per detector (replace with your own).
EARLY_FMOS = {
    'BV421-A': 'fmo_bv421_early',
    'APC-A':           'fmo_apc_early',
    'PE-Cy7-A':        'fmo_cy7_early',
}
LATE_FMOS = {
    'BV421-A': 'fmo_bv421_late',
    'APC-A':           'fmo_apc_late',
    'PE-Cy7-A':        'fmo_cy7_late',
}


def find_fcs(fcs_dir, name):
    name_l = name.lower()
    for f in sorted(os.listdir(fcs_dir)):
        if not f.lower().endswith('.fcs'):
            continue
        base = f.lower()
        if f'_{name_l}_' in base or base.endswith(f'_{name_l}.fcs'):
            return os.path.join(fcs_dir, f)
    raise FileNotFoundError(f"No FCS found for '{name}' in {fcs_dir}")


def load_fmo_thresholds(fcs_dir, fmo_map):
    gater = FMOGater()
    for ch, name in fmo_map.items():
        try:
            gater.add_fmo(ch, find_fcs(fcs_dir, name))
        except FileNotFoundError:
            pass
    if not gater.fmos:
        return {}
    gater.prepare()
    return gater.compute(percentile=99.5)


def density_scatter(ax, x, y, s=1.5, cmap='jet', **kwargs):
    from scipy.stats import gaussian_kde
    try:
        xy = np.vstack([x, y])
        z  = gaussian_kde(xy)(xy)
        idx = z.argsort()
        ax.scatter(x[idx], y[idx], c=z[idx], s=s, cmap=cmap,
                   linewidths=0, **kwargs)
    except Exception:
        ax.scatter(x, y, s=s, color='steelblue', alpha=0.3, linewidths=0)


def gate_quadrant_counts(df, xcol, ycol, xthresh, ythresh):
    """Return % in each quadrant: (lo/lo, hi/lo, lo/hi, hi/hi)."""
    n   = len(df)
    if n == 0:                          # an empty gate/sample → no division
        return {'lo/lo': 0.0, 'hi/lo': 0.0, 'lo/hi': 0.0, 'hi/hi': 0.0}
    xhi = df[xcol] > xthresh
    yhi = df[ycol] > ythresh
    q   = {
        'lo/lo': (~xhi & ~yhi).sum() / n * 100,
        'hi/lo': ( xhi & ~yhi).sum() / n * 100,
        'lo/hi': (~xhi &  yhi).sum() / n * 100,
        'hi/hi': ( xhi &  yhi).sum() / n * 100,
    }
    return q


def add_threshold_lines(ax, xthresh=None, ythresh=None, color='red', lw=1.2):
    if xthresh is not None:
        ax.axvline(xthresh, color=color, lw=lw, ls='--', alpha=0.8)
    if ythresh is not None:
        ax.axhline(ythresh, color=color, lw=lw, ls='--', alpha=0.8)


def annotate_quadrants(ax, df, xcol, ycol, xthresh, ythresh):
    """Label the regions the thresholds actually define.

    A missing threshold does not define a boundary, and substituting a
    sentinel invents one. This used to pass ``-999`` for an absent FMO
    threshold, so a plot with only an x threshold still displayed four
    quadrant percentages — and on a compensated channel whose dim population
    sits near -2000, the sentinel fell INSIDE the data and split it
    67.9 / 7.0 / 23.2 / 2.0: arbitrary numbers with the appearance of a
    gating result. ``None`` now means what it says, and only the axis that
    has a threshold is annotated.
    """
    n = len(df)
    if n == 0 or (xthresh is None and ythresh is None):
        return
    xl, xh = ax.get_xlim()
    yl, yh = ax.get_ylim()
    pad = 0.03
    fs  = 7.5

    def put(fx, fy, pct, **kw):
        ax.text(xl + fx*(xh-xl), yl + fy*(yh-yl), f"{pct:.1f}%",
                fontsize=fs, color='0.3', **kw)

    if xthresh is not None and ythresh is not None:
        q = gate_quadrant_counts(df, xcol, ycol, xthresh, ythresh)
        put(pad,     pad,     q['lo/lo'], va='bottom')
        put(1 - pad, pad,     q['hi/lo'], va='bottom', ha='right')
        put(pad,     1 - pad, q['lo/hi'], va='top')
        put(1 - pad, 1 - pad, q['hi/hi'], va='top', ha='right')
    elif xthresh is not None:                      # a left/right split only
        hi = float((df[xcol] > xthresh).sum()) / n * 100
        put(pad,     1 - pad, 100.0 - hi, va='top')
        put(1 - pad, 1 - pad, hi,         va='top', ha='right')
    else:                                          # a low/high split only
        hi = float((df[ycol] > ythresh).sum()) / n * 100
        put(pad, pad,     100.0 - hi, va='bottom')
        put(pad, 1 - pad, hi,         va='top')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fcs',    required=True, help='FCS directory')
    ap.add_argument('--sample', required=True,
                    help='Sample name (matched against FCS basenames)')
    ap.add_argument('--n',      type=int, default=50_000,
                    help='Events to display (default 50000)')
    ap.add_argument('--group',  choices=['early', 'late'], default=None,
                    help='Which FMO set to use (auto-detected from sample name if omitted)')
    ap.add_argument('--labels', default='',
                    help='Channel labels e.g. "BV421-A=CD11b;APC-A=CD34;PE-Cy7-A=CD45"')
    args = ap.parse_args()

    fcs_dir = args.fcs
    if not os.path.isdir(fcs_dir):
        sys.exit(f"Directory not found: {fcs_dir}")

    # Auto-detect FMO group
    group = args.group
    if group is None:
        group = 'late' if args.sample.lower().startswith('late') else 'early'
    fmo_map = LATE_FMOS if group == 'late' else EARLY_FMOS
    print(f"Using {group} FMOs for gating.")

    # Load and process sample
    print(f"[STEP 1/3] Loading and preprocessing {args.sample} ...", flush=True)
    s = FlowSample(find_fcs(fcs_dir, args.sample))
    s.run_qc()
    s.auto_compensate()
    s.apply_transform()
    labels = parse_labels(args.labels)
    if labels:
        s.set_labels(labels)

    # Downsample for display
    df = s.data.copy()
    if len(df) > args.n:
        df = df.sample(args.n, random_state=42)
    print(f"Displaying {len(df):,} events.")

    # FMO thresholds
    print("[STEP 2/3] Computing FMO thresholds ...", flush=True)
    thresh = load_fmo_thresholds(fcs_dir, fmo_map)

    cd11b = 'BV421-A'
    cd34 = 'APC-A'
    cd45 = 'PE-Cy7-A'

    # Use assigned labels for display if available
    def disp(det):
        return s.channel_labels.get(det, LABELS.get(det, det))

    t41  = thresh.get(cd11b)
    t34  = thresh.get(cd34)
    t45  = thresh.get(cd45)

    # ── Figure ────────────────────────────────────────────────────────────────
    print("[STEP 3/3] Rendering figure ...", flush=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle(f"{args.sample}  |  {len(df):,} events  |  {group} FMOs",
                 fontsize=12, fontweight='bold')

    plots = [
        (axes[0], cd11b, cd45, t41, t45),   # primary gate
        (axes[1], cd11b, cd34, t41, t34),   # progenitor proxy
        (axes[2], cd45, cd34, t45, t34),   # bulk vs progenitor
    ]

    for ax, xcol, ycol, xth, yth in plots:
        x = df[xcol].values
        y = df[ycol].values
        density_scatter(ax, x, y, alpha=0.5)
        ax.set_xlabel(disp(xcol), fontsize=10)
        ax.set_ylabel(disp(ycol), fontsize=10)
        if xth is not None or yth is not None:
            add_threshold_lines(ax, xth, yth)
            annotate_quadrants(ax, df, xcol, ycol, xth, yth)

    # Threshold legend
    if any(t is not None for t in [t41, t34, t45]):
        patch = mpatches.Patch(color='red', linestyle='--',
                               fill=False, label='FMO threshold (p99.5)')
        axes[2].legend(handles=[patch], fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
