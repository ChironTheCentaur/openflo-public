"""openflo-run's per-trial setup: a trial folder's label, the bead-based
µm-per-FSC calibration, the FMO thresholds of each FMO set the groups use,
and whether --skip-existing finds a sample already done.

Moved out of cli.py unchanged; the old cli names still work, deprecated.
"""
from __future__ import annotations

import os

import numpy as np

from .autoclean import find_scatter_channel
from .cli_groups import output_name
from .cli_samples import fcs_path, find_unstained
from .flow_sample import FlowSample
from .fmo_gater import FMOGater


def trial_label(trial_dir):
    """Return the trial folder's base name, for labels in output and logs."""
    return os.path.basename(trial_dir.rstrip('/\\')) or trial_dir


def calibrate_bead_micron(trial_dir, fmo_sets, bead_um=8.0):
    """Try every FMO control file in `fmo_sets` for `trial_dir`; the
    first one we can read gives us the bead population's median FSC-A,
    which we use to derive the µm-per-FSC scale factor.

    Returns (factor: float | None, bead_path: str | None). `factor` is
    `bead_um / median_FSC_A` — multiply any raw FSC-A value by `factor`
    to get its physical size in µm. None when no readable bead file
    could be found.
    """
    for set_name, mapping in (fmo_sets or {}).items():
        for ch, fname in (mapping or {}).items():
            try:
                path = fcs_path(trial_dir, fname)
            except FileNotFoundError:
                continue
            try:
                bead = FlowSample(path)
            except Exception as exc:  # noqa: BLE001 - reported and skipped
                print(f"  [Calibration] Could not read {path}: {exc}")
                continue
            fsc = find_scatter_channel(bead.data.columns, 'FSC', '-A',
                                       fallback=False)
            if fsc is None:
                continue
            vals = bead.data[fsc].astype(float)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals) < 100:
                continue
            median = float(np.median(vals))
            if median <= 0:
                continue
            factor = bead_um / median
            print(f"[Calibration] Bead file: {os.path.basename(path)} "
                  f"({set_name} / {ch})", flush=True)
            print(f"[Calibration] Median {fsc} = {median:,.0f} → "
                  f"{factor*1e6:.3f} µm per 10^6 FSC units "
                  f"(assuming {bead_um} µm beads).", flush=True)
            return factor, path
    return None, None


def build_fmo_thresholds(fcs_dir, fmo_map, label, fmo_percentile=99.0):
    """Compute ``{channel: threshold}`` from the FMO files named in `fmo_map`.

    A channel with no FMO falls back to the unstained control (with a
    warning); returns ``{}`` when no control is found at all.
    """
    print(f"\n-- FMO thresholds ({label})")
    gater   = FMOGater()
    missing = []

    for ch, name in fmo_map.items():
        try:
            gater.add_fmo(ch, fcs_path(fcs_dir, name))
        except FileNotFoundError:
            missing.append(ch)

    if missing:
        group = 'late' if 'late' in label.lower() else 'early'
        print(f"\n  [!] No FMO found for: {missing}")
        unstained = find_unstained(fcs_dir, group)
        if unstained:
            print(f"  [!] WARNING: Falling back to unstained control for {missing}")
            print(f"      File : {os.path.basename(unstained)}")
            print("      Valid only when those channels have no spectral spillover.")
            for ch in missing:
                gater.add_fmo(ch, unstained, is_fallback=True)
        else:
            print(f"  [!] No unstained control found either. Channels will not be gated: {missing}")

    if not gater.fmos:
        return {}
    gater.prepare()
    return gater.compute(percentile=fmo_percentile)


def already_done(out_dir, group, name):
    """True when this sample's ``_stats.csv`` already exists under `out_dir`/`group`."""
    return os.path.exists(os.path.join(out_dir,
                                       output_name(group),
                                       f'{output_name(name)}_stats.csv'))


def build_fmo_thresholds_for_groups(trial_dir, groups, fmo_sets,
                                     fmo_percentile, gate_overrides, report,
                                     trial_label_prefix=''):
    """Build the FMO threshold dict for each FMO set that's actually used by
    at least one group. Returns {fmo_set_name: thresholds_dict}.
    `report` is called once per unique set (for accurate step counting)."""
    # Collect every FMO set referenced by any group default OR any
    # per-sample override. Skip the empty set ('' = ungated): the task
    # lookup returns {} for it without needing an entry here.
    used_sets = []
    for g in groups:
        for s in [g.get('fmo_set', '')] + list(g.get('sample_fmo', {}).values()):
            if s and s not in used_sets:
                used_sets.append(s)

    fmo_thresh = {}
    prefix = f'[{trial_label_prefix}] ' if trial_label_prefix else ''
    for set_name in used_sets:
        if set_name not in fmo_sets:
            report(f'{prefix}FMO set "{set_name}" not defined — '
                   f'groups using it will run ungated')
            fmo_thresh[set_name] = {}
            continue
        report(f'{prefix}Building FMO thresholds ({set_name}) …')
        thresh = build_fmo_thresholds(trial_dir, fmo_sets[set_name],
                                      set_name, fmo_percentile)
        if gate_overrides:
            thresh.update(gate_overrides)
        fmo_thresh[set_name] = thresh
    # --gates threshold overrides must apply to EVERY sample, including ungated
    # ones (fmo_set='') — which are the norm in by-day auto-grouping. Without a
    # '' entry the task lookup returns {} and the thresholds are silently dropped
    # (yet --export-wsp would still record them). Seed the empty set here.
    if gate_overrides:
        fmo_thresh.setdefault('', {}).update(gate_overrides)
    return fmo_thresh
