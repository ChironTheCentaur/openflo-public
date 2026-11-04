"""openflo-run's plots: the per-sample cluster scatters, FSC/SSC view and
cluster heatmap, the per-group marker-pair overlays, grids and UMAP, and
--show-plots' re-opening of the saved figures.

Moved out of cli.py unchanged; the old cli names still work, deprecated.
"""
from __future__ import annotations

import copy
import os

import matplotlib.pyplot as plt
import numpy as np

from .autoclean import find_scatter_channel
from .cli_args import DEFAULT_DPI
from .cli_groups import output_name
from .export_stamp import background_kwargs, savefig_metadata
from .label_alignment import concatenate

# Default marker pairs for the group pair-scatter outputs. CD labels are
# resolved per sample via channel_labels, so these only render when the
# panel / --labels actually assigned them.
DEFAULT_SCATTER_PAIRS = [('CD902', 'CD901b'), ('CD901b', 'CD903'), ('CD902', 'CD903')]

# When --show-plots is on, the final group-level figures (UMAP + condition
# comparison) are *not* closed after saving so that the trailing plt.show()
# in main() actually has something to display. Per-sample figures are still
# always closed — 4 samples × 3 figures each is too many windows. run()
# passes its show_plots down as keep_open; this default applies only to a
# script that calls the group plots itself without keep_open. run() never
# sets it, so one run cannot change what another keeps.
KEEP_FINAL_PLOTS = False


def keeps_open(keep_open=None):
    """Whether a group-level figure stays open after saving: keep_open
    when given, else the KEEP_FINAL_PLOTS default."""
    return KEEP_FINAL_PLOTS if keep_open is None else bool(keep_open)

# The name prefix of the comparison staging folders that cli_compare's
# compare-set publishing makes; the re-opening of saved figures skips them.
STAGING_PREFIX = '.compare-staging-'


def parse_pairs(pairs_str):
    """Parse 'CD902/CD901b,CD901b/CD903' → [('CD902','CD901b'), ('CD901b','CD903')]."""
    if not pairs_str:
        return None
    out = []
    for chunk in pairs_str.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        for sep in ('/', 'vs', ':', '-'):
            if sep in chunk:
                x, y = chunk.split(sep, 1)
                out.append((x.strip(), y.strip()))
                break
    return out or None


def resolve_pair(s, xlabel, ylabel):
    """Resolve a CD-label pair to (x_detector, y_detector) for one sample,
    or (None, None) if either channel isn't present."""
    try:
        return s._resolve(xlabel), s._resolve(ylabel)
    except KeyError:
        return None, None


def intensity_axes(ax, resolved):
    """Tick the overlay's axes at round intensities. The samples' STORED
    values are drawn (logicle for fluorescence), which matplotlib ticked
    0.2 .. 1.0 where the intensities are 100 .. 10^5. One axis takes the
    transform its samples share; if they differ (no shared intensity for a
    position), it keeps the stored ticks and its label says so."""
    from .scales import intensity_ticker
    for i, axis in enumerate((ax.xaxis, ax.yaxis)):
        specs = [(getattr(s, 'data_transforms', None) or {}).get(chans[i])
                 for s, *chans in resolved]
        if any(sp != specs[0] for sp in specs):
            axis.set_label_text(f'{axis.get_label_text()} '
                                '(stored display units)')
            continue
        ticker = intensity_ticker(specs[0], 'stored') if specs[0] else None
        if ticker is not None:
            axis.set_major_locator(ticker[0])
            axis.set_major_formatter(ticker[1])


def save_group_pair_scatters(samples, label, out_dir, pairs=None,
                             max_points=20_000, random_state=42,
                             dpi=DEFAULT_DPI):
    """Per marker pair, write two figures for a *group* of samples:

      • ``<group>_<X>_<Y>_overlay.png`` — every sample on one axes, coloured
        by sample, with a legend (the 'stacked' overlay view).
      • ``<group>_<X>_<Y>_grid.png`` — one density panel per sample on shared
        axis limits (small-multiples comparison).

    Pairs that no sample can resolve (panel labels absent) are skipped.
    """
    if not samples:
        return
    pairs = pairs or DEFAULT_SCATTER_PAIRS
    os.makedirs(out_dir, exist_ok=True)
    token = output_name(label)
    from .pipeline import categorical_cmap
    base_cmap = categorical_cmap('auto', len(samples))

    for xlabel, ylabel in pairs:
        resolved = []
        for s in samples:
            xch, ych = resolve_pair(s, xlabel, ylabel)
            if xch and ych:
                resolved.append((s, xch, ych))
        if not resolved:
            print(f"  [pairs] {label}: '{xlabel} vs {ylabel}' skipped "
                  f"(channels not labelled on any sample).", flush=True)
            continue
        pair_token = f'{output_name(xlabel)}_{output_name(ylabel)}'

        # Shared, outlier-robust axis limits across all samples in the group.
        xc = np.concatenate([s.data[xch].to_numpy(dtype=float)
                             for s, xch, _ in resolved])
        yc = np.concatenate([s.data[ych].to_numpy(dtype=float)
                             for s, _, ych in resolved])
        xc = xc[np.isfinite(xc)]
        yc = yc[np.isfinite(yc)]
        xlim = (float(np.percentile(xc, 0.5)),
                float(np.percentile(xc, 99.5))) if xc.size else None
        ylim = (float(np.percentile(yc, 0.5)),
                float(np.percentile(yc, 99.5))) if yc.size else None

        # ── Overlay (all samples, coloured by sample) ──────────────────────
        try:
            import matplotlib.patches as mpatches
            fig, ax = plt.subplots(figsize=(7, 6))
            handles = []
            for i, (s, xch, ych) in enumerate(resolved):
                d = s.data.dropna(subset=[xch, ych])
                if max_points and len(d) > max_points:
                    d = d.sample(max_points, random_state=random_state)
                color = base_cmap(i % base_cmap.N)
                ax.scatter(d[xch].to_numpy(dtype=float),
                           d[ych].to_numpy(dtype=float),
                           s=1.0, alpha=0.4, color=color, linewidths=0,
                           rasterized=True)
                handles.append(mpatches.Patch(color=color, label=s.name))
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            intensity_axes(ax, resolved)
            if xlim:
                ax.set_xlim(*xlim)
            if ylim:
                ax.set_ylim(*ylim)
            ax.legend(handles=handles, fontsize=8, framealpha=0.8, loc='best')
            ax.set_title(f'{label} — {xlabel} vs {ylabel} (overlay)')
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir,
                        f'{token}_{pair_token}_overlay.png'), dpi=dpi,
                        metadata=savefig_metadata(fmt='png'),
                        **background_kwargs(fig, fmt='png'))
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001 - reported; other plots go on
            print(f"  [!] overlay {xlabel}/{ylabel} ({label}) failed: "
                  f"{type(exc).__name__}: {exc}", flush=True)

        # ── Grid (one density panel per sample, shared limits) ─────────────
        try:
            n = len(resolved)
            ncols = min(3, n)
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(5 * ncols, 4.5 * nrows),
                                     squeeze=False)
            flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]
            for i, (s, xch, ych) in enumerate(resolved):
                s.plot(xch, ych, color_by='density', ax=flat[i], title=s.name)
                if xlim:
                    flat[i].set_xlim(*xlim)
                if ylim:
                    flat[i].set_ylim(*ylim)
            for j in range(n, len(flat)):
                flat[j].set_visible(False)
            fig.suptitle(f'{label} — {xlabel} vs {ylabel} (per sample)',
                         fontsize=11)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir,
                        f'{token}_{pair_token}_grid.png'), dpi=dpi,
                        metadata=savefig_metadata(fmt='png'),
                        **background_kwargs(fig, fmt='png'))
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001 - reported; other plots go on
            print(f"  [!] grid {xlabel}/{ylabel} ({label}) failed: "
                  f"{type(exc).__name__}: {exc}", flush=True)


def save_plots(s, out_dir, dpi=DEFAULT_DPI):
    """Write a sample's pairwise scatter, FSC vs SSC and cluster heatmap PNGs.

    Files go into `out_dir`; a plot that fails is reported and skipped so the
    others are still saved.
    """
    from itertools import combinations
    os.makedirs(out_dir, exist_ok=True)
    # Sample name may include a subfolder prefix (e.g. 'day1/m1'); flatten
    # it for use in PNG filenames so we don't accidentally create a
    # 'day1/' directory inside out_dir.
    fname_token = output_name(s.name)

    channels = s.fluor_channels
    pairs    = list(combinations(channels, 2))
    if pairs:
        try:
            ncols = min(3, len(pairs))
            nrows = (len(pairs) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(6 * ncols, 5.5 * nrows),
                                     squeeze=False)
            flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]
            for i, (xcol, ycol) in enumerate(pairs):
                s.plot(xcol, ycol, color_by='cluster', ax=flat[i])
            for j in range(len(pairs), len(flat)):
                flat[j].set_visible(False)
            fig.suptitle(f'{s.name} — pairwise scatter (cluster)', fontsize=11)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f'{fname_token}_scatter.png'),
                        dpi=dpi, metadata=savefig_metadata(fmt='png'),
                        **background_kwargs(fig, fmt='png'))
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001 - reported; other plots go on
            print(f"  [!] scatter save failed for {s.name}: "
                  f"{type(exc).__name__}: {exc}", flush=True)

    # FSC vs SSC scatter, colored by cluster — separate file so it's easy to
    # find. Uses the raw (un-transformed) scatter channels, which is the
    # standard flow-cytometry view for population gating.
    fsc = find_scatter_channel(s.scatter_channels, 'FSC')
    ssc = find_scatter_channel(s.scatter_channels, 'SSC')
    if fsc and ssc and 'cluster' in s.data.columns:
        try:
            fig, ax = plt.subplots(figsize=(7, 6))
            s.plot(fsc, ssc, color_by='cluster', ax=ax,
                   title=f'{s.name} — FSC vs SSC (cluster)')
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f'{fname_token}_fsc_ssc.png'),
                        dpi=dpi, metadata=savefig_metadata(fmt='png'),
                        **background_kwargs(fig, fmt='png'))
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001 - reported; other plots go on
            print(f"  [!] FSC/SSC save failed for {s.name}: "
                  f"{type(exc).__name__}: {exc}", flush=True)

    # Cluster-median heatmap. Wrapped so a missing 'cluster' column or any
    # other matplotlib hiccup doesn't bring down the rest of save_plots
    # (the scatter / fsc_ssc above) — that asymmetric failure mode was the
    # cause of the "compare CSV exists but heatmap doesn't" report.
    try:
        ax = s.cluster_heatmap()
        if ax is not None:
            out_pth = os.path.join(out_dir, f'{fname_token}_heatmap.png')
            ax.figure.savefig(out_pth, dpi=dpi,
                              metadata=savefig_metadata(out_pth),
                              **background_kwargs(ax.figure, out_pth))
            plt.close(ax.figure)
            print(f"  [save] heatmap: {out_pth}", flush=True)
        else:
            print(f"  [!] heatmap skipped for {s.name} — "
                  f"cluster_heatmap() returned None (no 'cluster' column?)",
                  flush=True)
    except Exception as exc:  # noqa: BLE001 - reported; other plots go on
        print(f"  [!] heatmap save failed for {s.name}: "
              f"{type(exc).__name__}: {exc}", flush=True)


def run_group_umap(samples, label, out_dir, random_state=42,
                   dpi=DEFAULT_DPI, keep_open=None):
    """Pool `samples`, run UMAP on the pooled events and save ``<label>_umap.png``.

    The figure shows UMAP by cluster and by sample, plus FSC vs SSC when both
    scatter channels exist. Returns the pooled sample.
    """
    print(f"\n-- UMAP: {label} ({len(samples)} samples)")
    combined_data = concatenate(samples)
    ref = copy.copy(samples[0])
    ref.data = combined_data
    ref.name = label
    ref.umap_coords = None
    ref.run_umap(random_state=random_state)

    # Locate FSC/SSC channels — add as a third panel when present so we get
    # both the UMAP view *and* the classic flow-cytometry FSC vs SSC view of
    # the same cluster assignment.
    fsc = find_scatter_channel(ref.scatter_channels, 'FSC')
    ssc = find_scatter_channel(ref.scatter_channels, 'SSC')
    have_scatter = bool(fsc and ssc)

    n_panels = 3 if have_scatter else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    axes = list(axes) if n_panels > 1 else [axes]

    ref.plot_umap(color_by='cluster',       ax=axes[0],
                  title=f'{label} -- UMAP (cluster)')
    ref.plot_umap(color_by='sample_origin', ax=axes[1],
                  title=f'{label} -- UMAP (sample)')
    if have_scatter:
        ref.plot(fsc, ssc, color_by='cluster', ax=axes[2],
                 title=f'{label} -- FSC vs SSC (cluster)')

    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{label}_umap.png'), dpi=dpi,
                metadata=savefig_metadata(fmt='png'),
                **background_kwargs(fig, fmt='png'))
    if not keeps_open(keep_open):
        plt.close(fig)
    return ref


def show_saved_group_plots(out_dir):
    """Re-open every previously-saved figure as a matplotlib window so
    --show-plots is useful on a re-run that hit --skip-existing for
    everything (and therefore drew nothing fresh).

    Includes group-level outputs (UMAP, condition comparison) AND
    per-sample outputs (cluster scatter, FSC/SSC scatter, cluster-median
    heatmap). A big run produces dozens of per-sample files; if you need
    the lean view, run without --show-plots and open files from the
    explorer instead.
    Returns the number of figures opened.
    """
    from pathlib import Path
    p = Path(out_dir)
    if not p.exists():
        return 0
    seen = []
    for pat in ('*_umap.png',
                '*_heatmap.png',
                '*_scatter.png',
                '*_fsc_ssc.png',
                'compare_*.png',
                'condition_comparison.png'):
        # Not from a comparison staging folder: its PNGs are not this
        # run's set (a locked run leaves its new set there).
        seen.extend(sorted(f for f in p.rglob(pat)
                           if not any(part.startswith(STAGING_PREFIX)
                                      for part in f.relative_to(p).parts)))
    if not seen:
        return 0
    print(f"\n[Show plots] Re-opening {len(seen)} saved figure(s) "
          f"from {out_dir}", flush=True)
    for png in seen:
        try:
            img = plt.imread(str(png))
            h, w = img.shape[:2]
            # Render at a reasonable size while keeping aspect ratio.
            target_w = 13.0
            fig = plt.figure(figsize=(target_w, target_w * h / max(w, 1)))
            ax  = fig.add_subplot(1, 1, 1)
            ax.imshow(img)
            ax.set_axis_off()
            fig.suptitle(str(png.relative_to(p)), fontsize=9)
            fig.tight_layout()
        except Exception as e:  # noqa: BLE001 - reported; the other figures still reopen
            print(f"  [!] Could not load {png}: {e}", flush=True)
    return len(seen)
