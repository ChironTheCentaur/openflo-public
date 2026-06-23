"""
openflo command-line runner.
----------------------------
Panel-agnostic flow-cytometry analysis: point it at one or more FCS trials,
describe your groups / samples / FMO controls, and it compensates, gates,
clusters, and exports per-sample / per-condition stats, plots and a `.wsp`.

The constants below (example panel, FMO sets, groups) are just defaults used
when you don't pass --groups / --samples / --fmo-sets — replace them with your
own, or supply the flags per run.

Usage (single trial):
    openflo-run --trials /path/to/fcs/ --samples sample_1,sample_2 --out results/

Usage (multiple trials — independent):
    openflo-run --trials /trial1,/trial2,/trial3 --samples sample_1 --batch-mode independent

Usage (multiple trials — concatenated, cells retain trial origin):
    openflo-run --trials /trial1,/trial2 --samples sample_1 --batch-mode concatenate
"""

import argparse
import copy
import json
import logging
import os
import re
import sys
from typing import Literal, overload

import numpy as np


# Cap each process's BLAS thread pool to a fair share of cores. Must run
# BEFORE any numpy / scipy / sklearn import — those libraries read these
# env vars at load time and never re-check. Each parallel sample worker
# inherits this via os.environ, so on a 24-core box with --workers 4 each
# worker gets 6 BLAS threads instead of all spawning 24 (= 96 total threads,
# which is what was triggering the 'OpenBLAS error after 10 retries' kills).
def _cap_blas_threads():
    try:
        workers = 1
        if '--workers' in sys.argv:
            i = sys.argv.index('--workers')
            if i + 1 < len(sys.argv):
                workers = max(1, int(sys.argv[i + 1]))
        cores = os.cpu_count() or 2
        n = max(1, cores // max(1, workers))
        for var in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                    'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            os.environ.setdefault(var, str(n))
    except Exception:
        pass
_cap_blas_threads()

# Force UTF-8 stdout/stderr so non-ASCII characters (≥, →, …, em-dashes, etc.)
# don't crash under the Windows default cp1252 codepage. Safe no-op on
# platforms / streams that don't support reconfigure.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[attr-defined]
except Exception:
    pass

# E402 noqa on matplotlib imports: backend selection MUST happen after
# `import matplotlib` and BEFORE `import matplotlib.pyplot`, so these
# three statements have to be interleaved with logic.
import matplotlib  # noqa: E402

# Headless backend for the main run (unless --show-plots) and for every
# multiprocessing child ('__mp_main__'). Left untouched when imported elsewhere
# (e.g. preview.py) so that interactive previews still open a window.
if __name__ in ('__main__', '__mp_main__') and '--show-plots' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from .pipeline import FlowExperiment, FlowSample, FMOGater, concatenate  # noqa: E402

# ── Gate JSON parsing ─────────────────────────────────────────────────────────
#
# `--gates` accepts a JSON list of gate dicts (see flow_pipeline.gate_to_mask).
# 'threshold' kinds feed gate_overrides (which hooks into the existing FMO
# threshold merge); everything else feeds region_gates and is applied as a
# compound mask via FlowSample.apply_region_gates.

def _parse_gates_arg(s):
    """Returns (gate_overrides_dict, region_gates_list). Empty input → ({}, [])."""
    if not s:
        return {}, []
    try:
        data = json.loads(s)
    except json.JSONDecodeError as e:
        print(f"[gates] JSON parse error: {e}", flush=True)
        return {}, []
    if not isinstance(data, list):
        print(f"[gates] expected a JSON list of gate dicts, got "
              f"{type(data).__name__}", flush=True)
        return {}, []

    overrides = {}
    region_gates = []
    for g in data:
        if not isinstance(g, dict):
            continue
        kind = g.get('kind')
        if kind == 'threshold':
            try:
                overrides[str(g['channel'])] = float(g['value'])
            except (KeyError, TypeError, ValueError):
                continue
        elif kind in ('interval', 'rect', 'polygon'):
            region_gates.append(g)
        else:
            print(f"[gates] unknown gate kind {kind!r} — skipped", flush=True)
    return overrides, region_gates


# ── Example panel + controls (EDIT for your experiment) ──────────────────────
# A generic 3-colour example mapping standard CD markers to detector channels.
# The pipeline is panel-agnostic — nothing here is required; override per run
# with --channels / --groups / --samples / --fmo-sets (or via the GUI).
CD11b = 'BV421-A'
CD34 = 'APC-A'
CD45 = 'PE-Cy7-A'

# Example FMO control sets: each maps a detector channel to the FCS file
# basename holding that channel's single-stain / FMO control. Replace the
# basenames with your own (or pass --fmo-sets). The 'A'/'B' sets illustrate
# two staining runs; groups reference a set by name.
FMOS_SET_A = {
    CD11b: 'fmo_bv421_a',
    CD34: 'fmo_apc_a',
    CD45: 'fmo_cy7_a',
}
FMOS_SET_B = {
    CD11b: 'fmo_bv421_b',
    CD34: 'fmo_apc_b',
    CD45: 'fmo_cy7_b',
}

# Named FMO sets. Each set maps a detector channel to the FCS file basename
# that contains the single-stained / no-stain control for that channel. Groups
# reference these by name; extend this dict (via the GUI or by editing) for
# more staining runs without touching any other code.
DEFAULT_FMO_SETS = {
    'Set A': dict(FMOS_SET_A),
    'Set B': dict(FMOS_SET_B),
}

# Example default groups — used ONLY when neither --groups nor --samples is
# supplied. Replace with your own group / sample names.
DEFAULT_GROUPS = [
    {'name': 'Group A', 'samples': ['sample_1', 'sample_2'], 'fmo_set': 'Set A'},
    {'name': 'Group B', 'samples': ['sample_3', 'sample_4'], 'fmo_set': 'Set B'},
]


def _safe_filename(name):
    """Strip path-hostile characters so a group name can be used as a
    folder / file component."""
    return (re.sub(r'[<>:"/\\|?*\s]+', '_', str(name)).strip('_')
            or 'unnamed')


def _build_export_gate_list(region_gates, gate_overrides):
    """Combine region gates (with their parent_id structure preserved)
    and threshold overrides (added as roots) into a flat list with
    fresh consecutive ids. Used by the --export-wsp path."""
    out = []
    seq = [0]
    def alloc():
        seq[0] += 1
        return f'g{seq[0]}'
    old_to_new = {}
    for g in (region_gates or []):
        d = dict(g)
        src_id = d.pop('_import_id', None) or d.pop('id', None)
        d['id'] = alloc()
        if src_id is not None:
            old_to_new[src_id] = d['id']
        pid = g.get('parent_id')
        d['parent_id'] = old_to_new.get(pid) if pid else None
        out.append(d)
    for ch, val in (gate_overrides or {}).items():
        out.append({'kind': 'threshold',
                    'channel': ch,
                    'value': float(val),
                    'id': alloc(),
                    'parent_id': None})
    return out


def _export_pipeline_workspace(out_path, trial_dirs, groups,
                                gate_overrides, region_gates):
    """Emit a FlowJo-compatible .wsp containing the gates that were
    applied during the run, one SampleNode per FCS file processed.

    Resolves each sample name to its FCS path via fcs_path(); samples
    that can't be located are skipped (the rest still write). FMO-derived
    thresholds are NOT included — those are recomputed from the FMO
    control FCS files on every run, so they belong with the inputs the
    pipeline derives them from, not with the per-sample workspace
    export."""
    try:
        from .pipeline import WspWriter
    except ImportError as exc:
        print(f"[--export-wsp] openflo.pipeline import failed: {exc}", flush=True)
        return

    w = WspWriter(cytometer='OpenFlo-pipeline')
    samples_added = 0
    first_resolved_fcs = None
    for trial_dir in trial_dirs:
        for grp in (groups or []):
            for sample_name in grp.get('samples', []):
                try:
                    fp = fcs_path(trial_dir, sample_name)
                except FileNotFoundError as exc:
                    print(f"[--export-wsp] {sample_name}: {exc}", flush=True)
                    continue
                if first_resolved_fcs is None:
                    first_resolved_fcs = fp
                w.add_sample(
                    name=sample_name,
                    fcs_path=fp,
                    channels=[],     # WspWriter doesn't currently need these
                    gates=_build_export_gate_list(region_gates, gate_overrides))
                samples_added += 1
    if not samples_added:
        print("[--export-wsp] no samples resolved; nothing to write",
              flush=True)
        return

    # Compensation matrix: WspWriter stores one workspace-wide matrix.
    # The pipeline runs auto_compensate (FCS $SPILL) per sample, so the
    # source of truth is the FCS metadata. Pull the first resolved
    # sample's $SPILL and write it — every sample in a trial shares the
    # same spillover keyword in practice.
    comp_note = ''
    # Guarded by the samples_added > 0 check above, so first_resolved_fcs
    # is non-None here — assert to narrow for pyright.
    assert first_resolved_fcs is not None
    try:
        from .pipeline import read_compensation_matrix
        chans, mat = read_compensation_matrix(first_resolved_fcs)
        if chans is not None and mat is not None:
            w.set_compensation(chans, mat)
            comp_note = ' + spillover'
        else:
            print(
                f"[--export-wsp] {os.path.basename(first_resolved_fcs)}: "
                "no $SPILL in FCS metadata — exporting without spillover",
                flush=True)
    except Exception as exc:
        print(f"[--export-wsp] spillover read failed: {exc}", flush=True)

    try:
        w.write(out_path)
        print(
            f"[--export-wsp] wrote {samples_added} sample(s){comp_note} "
            f"-> {out_path}", flush=True)
    except Exception as exc:
        print(f"[--export-wsp] write failed: {exc}", flush=True)


def _normalise_groups(groups):
    """Coerce a raw groups spec into a list of canonical dicts.

    Each normalised dict has:
      name       (str)
      samples    (list[str])          — sample tokens, in order
      fmo_set    (str)                — the group's DEFAULT FMO set
      sample_fmo ({name: fmo_set})    — per-sample resolved FMO set

    Per-sample FMO assignment: a `samples` entry may be a plain string
    (uses the group's default `fmo_set`) OR a dict
    ``{'name': ..., 'fmo_set': ...}`` to override just that sample. This
    keeps the common case terse while letting any sample point at a
    different FMO control set than its group-mates. Compensation and
    antibody labels are resolved per sample automatically from each
    FCS ($SPILL / $PnS) in the pipeline, so they don't need a slot here.

    Drops any group whose name or sample list is empty.
    """
    norm = []
    for g in groups or []:
        if not isinstance(g, dict):
            continue
        name = str(g.get('name', '')).strip()
        raw_samples = g.get('samples', [])
        if isinstance(raw_samples, str):
            raw_samples = [s.strip() for s in raw_samples.split(',') if s.strip()]
        group_fmo = str(g.get('fmo_set', '')).strip()

        sample_names = []
        sample_fmo = {}
        for s in raw_samples:
            if isinstance(s, dict):
                sname = str(s.get('name', '')).strip()
                if not sname:
                    continue
                sfmo = str(s.get('fmo_set', '') or group_fmo).strip()
            else:
                sname = str(s).strip()
                if not sname:
                    continue
                sfmo = group_fmo
            sample_names.append(sname)
            sample_fmo[sname] = sfmo

        if not name or not sample_names:
            continue
        entry = {'name': name, 'samples': sample_names,
                 'fmo_set': group_fmo, 'sample_fmo': sample_fmo}
        # By-day auto-groups tag each group with its source folder so the
        # run resolves that group's samples there (rather than the run's
        # single trial dir). Preserve it through normalisation.
        if g.get('trial_dir'):
            entry['trial_dir'] = str(g['trial_dir'])
        norm.append(entry)
    return norm


# ── By-day auto-grouping ───────────────────────────────────────────────────────

def _discover_day_folders(trial_dirs):
    """Every directory at/under the given paths that DIRECTLY contains at
    least one .fcs file. Lets the user point at a single PARENT folder
    and have each sub-folder auto-become its own day/group, sampled
    independently. De-duplicated, sorted."""
    found, seen = [], set()
    for d in trial_dirs:
        if not d or not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            if any(f.lower().endswith('.fcs') for f in files):
                rp = os.path.normpath(root)
                if rp not in seen:
                    seen.add(rp)
                    found.append(rp)
    return sorted(found)


def _auto_groups_by_day(trial_dirs):
    """Build one group per discovered day-folder (a folder that directly
    holds FCS files). Each group's samples are that folder's FCS files
    (by filename stem); each group is tagged with its source folder via
    `trial_dir` so the run resolves its samples there. Group name is the
    folder basename, tidied to 'Day N' when a day token is present;
    duplicate names are disambiguated with the parent folder.

    Returns [] when no FCS are found anywhere (caller falls back to the
    legacy default groups)."""
    folders = _discover_day_folders(trial_dirs)
    groups = []
    for d in folders:
        try:
            fcs = sorted(f for f in os.listdir(d)
                         if f.lower().endswith('.fcs'))
        except OSError:
            continue
        if not fcs:
            continue
        samples = [os.path.splitext(f)[0] for f in fcs]
        base = os.path.basename(d.rstrip('/\\')) or d
        m = re.search(r'day\s*[-_ ]?([0-9]+)', base, re.IGNORECASE)
        name = f'Day {m.group(1)}' if m else base
        groups.append({'name': name, 'samples': samples,
                       'fmo_set': '', 'trial_dir': d})

    # Disambiguate duplicate group names (e.g. two "Day 3" folders under
    # different parents) by appending the parent folder name.
    counts = {}
    for g in groups:
        counts[g['name']] = counts.get(g['name'], 0) + 1
    for g in groups:
        if counts[g['name']] > 1:
            parent = os.path.basename(os.path.dirname(g['trial_dir']))
            if parent:
                g['name'] = f"{g['name']} ({parent})"
    return groups


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_labels(labels_str):
    """Parse 'DetA=LabelA;DetB=LabelB' into {DetA: LabelA, ...}."""
    if not labels_str:
        return {}
    result = {}
    for pair in labels_str.split(';'):
        if '=' in pair:
            det, lbl = pair.split('=', 1)
            result[det.strip()] = lbl.strip()
    return result


def _norm_token(s):
    """Lower-case and strip all non-alphanumerics — so 'PE/Cy7',
    'PE-Cy7-A' and 'pecy7' all compare equal at the token level."""
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def read_staining_panel(path, channels):
    """Read a staining-panel spreadsheet → ``{detector_channel: CD_label}``.

    The sheet pairs a CD/marker token (e.g. ``CD11b``) with a fluorophore
    token (e.g. ``BV421``, ``PE/Cy7``, ``APC``) on the same row, in two
    columns, in any order, header optional. Each fluorophore is matched to
    the detector channel whose name *contains* it (normalised, case- and
    punctuation-insensitive), so ``BV421`` → ``BV421-A`` and
    ``PE/Cy7`` → ``PE-Cy7-A``. Returns ``{}`` when nothing matched (the run
    then falls back to whatever ``--labels`` provided, or the raw detector
    names).

    `channels` is the list of real detector channel names to match against
    (typically a sample's ``channel_names`` / DataFrame columns).
    """
    try:
        import pandas as pd
        sheets = pd.read_excel(path, header=None, sheet_name=None)
    except Exception as exc:
        print(f"  [panel] Could not read {path}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return {}

    norm_channels = {ch: _norm_token(ch) for ch in channels}

    def match_fluor(fluor):
        nf = _norm_token(fluor)
        if not nf:
            return None
        cands = [ch for ch, nc in norm_channels.items() if nf and nf in nc]
        if not cands:
            return None
        # Prefer area ('-A') detectors and the shortest channel name.
        cands.sort(key=lambda c: ('-a' not in c.lower(), len(c)))
        return cands[0]

    mapping = {}
    cd_re = re.compile(r'^cd\d', re.I)
    for df in sheets.values():
        for _, row in df.iterrows():
            cells = [str(v).strip() for v in row.tolist()
                     if str(v).strip() and str(v).strip().lower() != 'nan']
            cd = next((c for c in cells if cd_re.match(c)), None)
            if not cd:
                continue
            for fl in (c for c in cells if c != cd):
                det = match_fluor(fl)
                if det and det not in mapping:
                    mapping[det] = cd
                    break
    if mapping:
        print(f"  [panel] {os.path.basename(path)} → "
              f"{dict(mapping)}", flush=True)
    else:
        print(f"  [panel] {os.path.basename(path)}: no CD↔fluorophore "
              f"rows matched the data channels.", flush=True)
    return mapping


def find_panel_xlsx(dirs):
    """Locate a staining-panel spreadsheet for the given trial dir(s).

    Searches each dir's descendants first, then walks up a few ancestor
    levels (the panel commonly sits in a study-root folder above the day
    sub-folders). Files whose name mentions 'panel' or 'stain' win ties.
    Returns a path or None.
    """
    candidates = []
    seen = set()

    def add(p):
        ap = os.path.abspath(p)
        if ap not in seen and os.path.isfile(ap):
            base = os.path.basename(ap)
            if base.lower().endswith(('.xlsx', '.xls')) \
                    and not base.startswith('~$'):
                seen.add(ap)
                candidates.append(ap)

    for d in dirs:
        if not d:
            continue
        d = os.path.abspath(d)
        # Descendants of the trial dir.
        for root, _dn, files in os.walk(d):
            for f in files:
                add(os.path.join(root, f))
        # Direct contents of up to 4 ancestor levels.
        cur = d
        for _ in range(4):
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            try:
                for f in os.listdir(parent):
                    add(os.path.join(parent, f))
            except OSError:
                pass
            cur = parent

    candidates.sort(key=lambda p: (
        'panel' not in os.path.basename(p).lower()
        and 'stain' not in os.path.basename(p).lower(),
        len(p)))
    return candidates[0] if candidates else None


# Default marker pairs for the group pair-scatter outputs. CD labels are
# resolved per sample via channel_labels, so these only render when the
# panel / --labels actually assigned them.
DEFAULT_SCATTER_PAIRS = [('CD34', 'CD11b'), ('CD11b', 'CD45'), ('CD34', 'CD45')]


def parse_pairs(pairs_str):
    """Parse 'CD34/CD11b,CD11b/CD45' → [('CD34','CD11b'), ('CD11b','CD45')]."""
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


def _resolve_pair(s, xlabel, ylabel):
    """Resolve a CD-label pair to (x_detector, y_detector) for one sample,
    or (None, None) if either channel isn't present."""
    try:
        return s._resolve(xlabel), s._resolve(ylabel)
    except KeyError:
        return None, None


def save_group_pair_scatters(samples, label, out_dir, pairs=None,
                             max_points=20_000, random_state=42):
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
    token = _safe_filename(label)
    base_cmap = plt.get_cmap('tab10' if len(samples) <= 10 else 'tab20')

    for xlabel, ylabel in pairs:
        resolved = []
        for s in samples:
            xch, ych = _resolve_pair(s, xlabel, ylabel)
            if xch and ych:
                resolved.append((s, xch, ych))
        if not resolved:
            print(f"  [pairs] {label}: '{xlabel} vs {ylabel}' skipped "
                  f"(channels not labelled on any sample).", flush=True)
            continue
        pair_token = f'{_safe_filename(xlabel)}_{_safe_filename(ylabel)}'

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
            if xlim:
                ax.set_xlim(*xlim)
            if ylim:
                ax.set_ylim(*ylim)
            ax.legend(handles=handles, fontsize=8, framealpha=0.8, loc='best')
            ax.set_title(f'{label} — {xlabel} vs {ylabel} (overlay)')
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir,
                        f'{token}_{pair_token}_overlay.png'), dpi=150)
            plt.close(fig)
        except Exception as exc:
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
                        f'{token}_{pair_token}_grid.png'), dpi=150)
            plt.close(fig)
        except Exception as exc:
            print(f"  [!] grid {xlabel}/{ylabel} ({label}) failed: "
                  f"{type(exc).__name__}: {exc}", flush=True)


def _trial_label(trial_dir):
    return os.path.basename(trial_dir.rstrip('/\\')) or trial_dir


def _calibrate_bead_micron(trial_dir, fmo_sets, bead_um=8.0):
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
            except Exception as exc:
                print(f"  [Calibration] Could not read {path}: {exc}")
                continue
            fsc = next((c for c in bead.data.columns
                        if c.upper().startswith('FSC')
                        and c.upper().endswith('-A')), None)
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


def _gpu_present():
    """True if an NVIDIA GPU is visible via nvidia-smi (driver-level check).
    Independent of whether RAPIDS cuML is installed."""
    try:
        import subprocess
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        out = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=6,
                             creationflags=flags)
        return out.returncode == 0 and bool(out.stdout.strip())
    except Exception:
        return False


def _find_unstained(fcs_dir, group='early'):
    prefer = 'lateunstained' if group == 'late' else 'unstained'
    for f in sorted(os.listdir(fcs_dir)):
        base = f.lower()
        if not base.endswith('.fcs'):
            continue
        if f'_{prefer}_' in base or base.endswith(f'_{prefer}.fcs'):
            return os.path.join(fcs_dir, f)
    for f in sorted(os.listdir(fcs_dir)):
        base = f.lower()
        if 'unstained' in base and base.endswith('.fcs'):
            if group == 'early' and 'late' in base:
                continue
            return os.path.join(fcs_dir, f)
    return None


@overload
def _scan_for_token(search_dir: str, token: str, recurse: bool,
                    all_matches: Literal[False] = False) -> str | None: ...
@overload
def _scan_for_token(search_dir: str, token: str, recurse: bool,
                    all_matches: Literal[True]) -> list[str]: ...
def _scan_for_token(search_dir, token, recurse, all_matches=False):
    """Find FCS files whose name matches `token`. A match is any of:
      • exact filename stem  (`<token>.fcs`)        — by-day auto-groups
        hand the whole stem here, so this is the primary path for them;
      • `_<token>_` substring                        — short logical names
      • ends with `_<token>.fcs`                     — trailing short name
    Returns the first match (or every match when `all_matches=True`);
    None / [] when nothing matches."""
    token_l = token.lower()
    matches: list[str] = []
    try:
        if recurse:
            walker = os.walk(search_dir)
        else:
            walker = [(search_dir, [], os.listdir(search_dir))]
    except OSError:
        return matches if all_matches else None
    for root, _dirs, files in walker:
        for f in sorted(files):
            if not f.lower().endswith('.fcs'):
                continue
            b = f.lower()
            stem = b[:-4]   # strip '.fcs'
            if (stem == token_l
                    or f'_{token_l}_' in b
                    or b.endswith(f'_{token_l}.fcs')):
                path = os.path.join(root, f)
                if not all_matches:
                    return path
                matches.append(path)
    return matches if all_matches else None


def fcs_path(fcs_dir: str, name: str) -> str:
    """Locate the FCS file for a logical sample name inside `fcs_dir`.

    `name` accepts a forward-slash subfolder prefix:
      • 'm1'         → search fcs_dir directly
      • 'day1/m1'    → search fcs_dir/day1/ directly (no fallback)
      • 'study/day1/m1' → search fcs_dir/study/day1/ directly

    When `name` has no slash and isn't found at the root of fcs_dir, we
    fall back to a recursive search. If that recursive search hits more
    than one match the first one is used and a warning lists the rest —
    the user can disambiguate by adding a subfolder prefix to the sample
    name in their group definition.
    """
    name = str(name).strip().replace('\\', '/')

    # Explicit subfolder path → search ONLY there
    if '/' in name:
        subfolder, token = name.rsplit('/', 1)
        search_dir = os.path.join(fcs_dir, subfolder)
        path = _scan_for_token(search_dir, token, recurse=False)
        if path:
            return path
        raise FileNotFoundError(
            f"Could not find FCS file for '{name}' in {search_dir}")

    # Bare token → try root first
    path = _scan_for_token(fcs_dir, name, recurse=False)
    if path:
        return path

    # Recursive fallback (handles FMO files placed in a controls/
    # subfolder, etc.).
    matches = _scan_for_token(fcs_dir, name, recurse=True, all_matches=True)
    if not matches:
        raise FileNotFoundError(
            f"Could not find FCS file for '{name}' in {fcs_dir}")
    if len(matches) > 1:
        print(f"  [!] '{name}' matched {len(matches)} files; using "
              f"{os.path.relpath(matches[0], fcs_dir)}")
        for extra in matches[1:]:
            print(f"      also: {os.path.relpath(extra, fcs_dir)}")
        print(f"      Disambiguate by using a subfolder prefix in the "
              f"sample name (e.g. 'day1/{name}').")
    return matches[0]


def build_fmo_thresholds(fcs_dir, fmo_map, label, fmo_percentile=99.5):
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
        unstained = _find_unstained(fcs_dir, group)
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


def _find_scatter_channel(s, prefix):
    """Pick the area-version scatter channel matching prefix (FSC/SSC).
    Prefers '-A' channels; falls back to the first match."""
    prefix_u = prefix.upper()
    area = [c for c in s.scatter_channels
            if c.upper().startswith(prefix_u) and '-A' in c.upper()]
    if area:
        return area[0]
    other = [c for c in s.scatter_channels if c.upper().startswith(prefix_u)]
    return other[0] if other else None


def save_plots(s, out_dir):
    from itertools import combinations
    os.makedirs(out_dir, exist_ok=True)
    # Sample name may include a subfolder prefix (e.g. 'day1/m1'); flatten
    # it for use in PNG filenames so we don't accidentally create a
    # 'day1/' directory inside out_dir.
    fname_token = _safe_filename(s.name)

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
                        dpi=150)
            plt.close(fig)
        except Exception as exc:
            print(f"  [!] scatter save failed for {s.name}: "
                  f"{type(exc).__name__}: {exc}", flush=True)

    # FSC vs SSC scatter, colored by cluster — separate file so it's easy to
    # find. Uses the raw (un-transformed) scatter channels, which is the
    # standard flow-cytometry view for population gating.
    fsc = _find_scatter_channel(s, 'FSC')
    ssc = _find_scatter_channel(s, 'SSC')
    if fsc and ssc and 'cluster' in s.data.columns:
        try:
            fig, ax = plt.subplots(figsize=(7, 6))
            s.plot(fsc, ssc, color_by='cluster', ax=ax,
                   title=f'{s.name} — FSC vs SSC (cluster)')
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f'{fname_token}_fsc_ssc.png'),
                        dpi=150)
            plt.close(fig)
        except Exception as exc:
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
            ax.figure.savefig(out_pth, dpi=150)
            plt.close(ax.figure)
            print(f"  [save] heatmap: {out_pth}", flush=True)
        else:
            print(f"  [!] heatmap skipped for {s.name} — "
                  f"cluster_heatmap() returned None (no 'cluster' column?)",
                  flush=True)
    except Exception as exc:
        print(f"  [!] heatmap save failed for {s.name}: "
              f"{type(exc).__name__}: {exc}", flush=True)


def run_group_umap(samples, label, out_dir, random_state=42):
    print(f"\n-- UMAP: {label} ({len(samples)} samples)")
    combined_data = concatenate(samples)
    ref = copy.copy(samples[0])
    ref.data = combined_data
    ref.name = label
    ref.umap_coords = None
    ref.run_umap(sample_n=80_000, random_state=random_state)

    # Locate FSC/SSC channels — add as a third panel when present so we get
    # both the UMAP view *and* the classic flow-cytometry FSC vs SSC view of
    # the same cluster assignment.
    fsc = _find_scatter_channel(ref, 'FSC')
    ssc = _find_scatter_channel(ref, 'SSC')
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
    fig.savefig(os.path.join(out_dir, f'{label}_umap.png'), dpi=150)
    if not _KEEP_FINAL_PLOTS:
        plt.close(fig)
    return ref


def _already_done(out_dir, group, name):
    return os.path.exists(os.path.join(out_dir,
                                       _safe_filename(group),
                                       f'{_safe_filename(name)}_stats.csv'))


# When --show-plots is on, the final group-level figures (UMAP + condition
# comparison) are *not* closed after saving so that the trailing plt.show()
# in main() actually has something to display. Per-sample figures are still
# always closed — 4 samples × 3 figures each is too many windows.
_KEEP_FINAL_PLOTS = False


def _show_saved_group_plots(out_dir):
    """Re-open every previously-saved figure as a matplotlib window so
    --show-plots is useful on a re-run that hit --skip-existing for
    everything (and therefore drew nothing fresh).

    Includes group-level outputs (UMAP, condition comparison) AND
    per-sample outputs (cluster scatter, FSC/SSC scatter, cluster-median
    heatmap). The per-sample set was historically excluded because a
    big run produces dozens of files; if you need the lean view, run
    without --show-plots and open files from the explorer instead.
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
        seen.extend(sorted(p.rglob(pat)))
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
        except Exception as e:
            print(f"  [!] Could not load {png}: {e}", flush=True)
    return len(seen)


# ── Parallel sample processing ────────────────────────────────────────────────

def _process_sample_task(task):
    """Top-level worker (picklable — runs in a child process).

    Loads every input FCS for one output sample, concatenates them when there
    is more than one (tagging each cell with its trial-of-origin), gates and
    clusters. Returns (key, FlowSample | None, error_msg | None) where
    `key` uniquely identifies the task (group + name) so the dispatcher
    can bucket results correctly even when two groups (e.g. two days)
    contain identically-named FCS files.
    """
    name = task['name']
    key  = task.get('key', name)
    try:
        pieces = []
        for origin, path in task['paths']:
            s = FlowSample(path)
            s.run_qc()
            # Debris + doublet filtering, applied to raw (un-transformed)
            # FSC channels. Order matters: drop debris first so the
            # doublet ratio's median is computed on cell-sized events.
            if task.get('debris_min_fsc') is not None:
                s.filter_debris(min_fsc=task['debris_min_fsc'])
            if task.get('doublet_tol'):
                s.filter_doublets(tol=float(task['doublet_tol']))
            s.auto_compensate()
            s.apply_transform()
            if task['labels']:
                s.set_labels(task['labels'])
            s.apply_threshold_gates(task['thresholds'])
            # Compound region gates (rect/polygon/interval) — filter events,
            # applied after transform so coordinates match post-logicle space.
            if task.get('region_gates'):
                s.apply_region_gates(task['region_gates'])
            s.name = origin                     # → sample_origin column
            pieces.append(s)
        if not pieces:
            return (key, None, 'no input files found')
        combined = copy.copy(pieces[0])
        if len(pieces) > 1:
            combined.data = concatenate(pieces)
        combined.name = name      # display / filename stem
        # Batch correction (CytoNorm) — apply the pre-fitted model after
        # transform, before clustering, so clusters/UMAP/stats use the
        # batch-aligned values.
        if task.get('cytonorm'):
            from .pipeline import CytoNorm
            combined.data = CytoNorm.from_dict(task['cytonorm']).apply(
                combined.data, task.get('batch_id', ''))
        # The seed is held constant across every sample in a trial so the
        # subsampling cuts (when max_events truncates) and any subsequent
        # UMAP are reproducible. Workers can run in parallel safely —
        # each spawns its own RNG from the same seed value.
        combined.cluster(k=task['k'], n_jobs=task['n_jobs'],
                         max_events=task.get('max_events'),
                         vram_admission_gb=task.get('vram_admission_gb', 1.0),
                         random_state=task.get('random_state', 42))
        return (key, combined, None)
    except Exception as e:
        return (key, None, f'{type(e).__name__}: {e}')


def _run_sample_tasks(tasks, workers, admission_gb, report):
    """Process sample tasks — in parallel when workers > 1, with admission
    control: a new worker is only submitted when free RAM ≥ `admission_gb`.
    In-flight workers are NEVER paused (suspending would not free RAM) — they
    complete naturally; only the *launcher* waits. Each completed sample's
    plots + stats are written immediately so --skip-existing works after a
    watchdog restart. Returns {group_name: [FlowSample, ...]} keyed by the
    'group' field on each task."""
    results  = {}                                  # group_name -> [samples]
    # Dispatch key MUST be unique per task. Bare sample name collides
    # when two groups (e.g. two day-folders) hold identically-named FCS
    # — that silently bucketed both into one group. Key by group+name.
    for t in tasks:
        t['key'] = f"{t['group']}\x1f{t['name']}"
    by_key   = {t['key']: t for t in tasks}

    def _handle(result):
        key, s, err = result
        t = by_key[key]
        nm = t['name']                              # for display + filenames
        if s is None:
            report(f"Failed: {nm} ({err})")
            # Detect the unmistakable Windows commit-charge / pagefile signature
            # and tell the user how to fix it. RAM might look fine in psutil
            # while commit is exhausted — the watchdog won't catch this.
            e = (err or '').lower()
            if ('paging file' in e or 'pagefile' in e
                    or 'winerror 1455' in e or 'winerror 8' in e
                    or 'not enough memory resources' in e):
                print(
                    "\n  [!] WINDOWS PAGEFILE TOO SMALL.\n"
                    "      Your physical RAM is fine but the OS commit charge\n"
                    "      ran out. Increase the pagefile:\n"
                    "        Win → View advanced system settings\n"
                    "        Performance → Settings → Advanced → Virtual memory → Change\n"
                    "        Uncheck 'Automatically manage' → pick C: → Custom size\n"
                    "        Initial + Maximum ≥ 32768 MB (32 GB) → Set → OK → restart\n",
                    flush=True)
            return
        group_dir = _safe_filename(t['group'])
        file_nm   = _safe_filename(nm)
        save_plots(s, os.path.join(t['out'], group_dir))
        s.export_stats(os.path.join(t['out'], group_dir, f"{file_nm}_stats.csv"))
        results.setdefault(t['group'], []).append(s)
        report(f"Processed {nm}  ({len(s.data):,} events)")

    if not tasks:
        return results

    if workers <= 1 or len(tasks) == 1:
        for t in tasks:
            _handle(_process_sample_task(t))
        return results

    # Parallel path with admission control.
    import time
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    try:
        import psutil
    except ImportError:
        psutil = None  # type: ignore[assignment]

    n = min(workers, len(tasks))
    print(f"\n-- Processing {len(tasks)} sample(s) on {n} parallel worker(s) "
          f"({tasks[0]['n_jobs']} clustering thread(s) each).", flush=True)
    if admission_gb > 0 and psutil is not None:
        print(f"   Admission: a new worker starts only when ≥ {admission_gb:.1f} GB RAM is free.",
              flush=True)
    else:
        print("   Admission: disabled (psutil unavailable or threshold ≤ 0).",
              flush=True)

    waiting     = False
    no_progress = 0       # consecutive 2 s polls with no completion

    # ProcessPoolExecutor workers are NON-daemonic, so Phenograph can still
    # spawn its own Jaccard-kernel pool inside each worker. (multiprocessing.Pool
    # workers are daemonic and would crash with "daemonic processes are not
    # allowed to have children".)
    with ProcessPoolExecutor(max_workers=n) as ex:
        pending   = list(tasks)
        in_flight = {}                            # future → sample name

        while pending or in_flight:
            # ── Top up workers, admission-gated ──────────────────────────
            while pending and len(in_flight) < n:
                if admission_gb > 0 and psutil is not None:
                    free_gb = psutil.virtual_memory().available / (1024 ** 3)
                else:
                    free_gb = float('inf')
                if free_gb < admission_gb:
                    if not in_flight:
                        if not waiting:
                            print(f"   [Admission] Free RAM {free_gb:.1f} GB "
                                  f"< {admission_gb:.1f} GB — waiting …",
                                  flush=True)
                            waiting = True
                        time.sleep(3)
                        continue
                    break
                if waiting:
                    print(f"   [Admission] Free RAM {free_gb:.1f} GB ≥ "
                          f"{admission_gb:.1f} GB — resuming submission.",
                          flush=True)
                    waiting = False
                t = pending.pop(0)
                fut = ex.submit(_process_sample_task, t)
                in_flight[fut] = t['name']   # display name for status only

            if not in_flight:
                continue

            # ── Drain completions (timeout lets us re-poll RAM regularly) ─
            try:
                done, _ = wait(in_flight.keys(),
                               timeout=2.0, return_when=FIRST_COMPLETED)
            except Exception as e:
                # Pool broken etc. — surface and stop, don't hang silently.
                for nm in in_flight.values():
                    report(f"Failed: {nm} (pool broken: {e})")
                return results

            if done:
                no_progress = 0
                for fut in done:
                    nm = in_flight.pop(fut)
                    try:
                        _handle(fut.result())
                    except Exception as e:
                        report(f"Failed: {nm} ({type(e).__name__}: {e})")
            else:
                no_progress += 1
                # 90 polls × 2 s = ~3 minutes of silence → log a warning.
                # Repeats every 3 minutes so a hang can't go undetected.
                if no_progress % 90 == 0:
                    free_gb = (psutil.virtual_memory().available / (1024**3)
                               if psutil is not None else 0)
                    stuck = ', '.join(in_flight.values())
                    print(f"   [Stuck?] {no_progress*2//60} min with no "
                          f"completions. In flight: {stuck}. "
                          f"Free RAM {free_gb:.1f} GB. "
                          f"If this persists, Cancel and retry with fewer "
                          f"workers or a smaller --max-events cap.",
                          flush=True)

    return results


# ── Independent mode ──────────────────────────────────────────────────────────

def _build_fmo_thresholds_for_groups(trial_dir, groups, fmo_sets,
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
    return fmo_thresh


def _compare_group_pair(g_a, g_b, results, trial_out, report,
                        trial_label_prefix=''):
    """Run condition-comparison between two groups. Called for every pair."""
    prefix = f'[{trial_label_prefix}] ' if trial_label_prefix else ''
    report(f'{prefix}Compare: {g_a["name"]} vs {g_b["name"]} …')
    sa = results.get(g_a['name'], [])
    sb = results.get(g_b['name'], [])
    if not sa or not sb:
        print(f'  [!] {g_a["name"]} or {g_b["name"]} has no processed '
              f'samples — comparison skipped')
        return
    all_s = sa + sb
    exp = FlowExperiment.__new__(FlowExperiment)
    exp.samples = {s.name: s for s in all_s}
    try:
        summary = exp.compare_conditions(
            groupA=[s.name for s in sa],
            groupB=[s.name for s in sb],
            label_a=g_a['name'], label_b=g_b['name'],
        )
    except Exception as exc:
        print(f'  [!] Comparison failed: {exc}')
        return
    if summary.empty:
        return
    base = _safe_filename(f'compare_{g_a["name"]}_vs_{g_b["name"]}')
    summary.to_csv(os.path.join(trial_out, f'{base}.csv'), index=False)
    plt.savefig(os.path.join(trial_out, f'{base}.png'), dpi=150)
    if not _KEEP_FINAL_PLOTS:
        plt.close()


def _run_independent(trial_dirs, out_dir, groups, fmo_sets,
                     k, fmo_percentile, labels, gate_overrides, region_gates,
                     skip_existing,
                     workers, n_jobs, max_events, admission_gb,
                     vram_admission_gb, filter_debris_um, bead_um,
                     doublet_tol, report, random_state=42, pairs=None,
                     cytonorm=None):
    """Process each trial folder as a separate analysis; results go into
    per-trial sub-dirs. Samples within a trial are processed in parallel
    when workers > 1. Every group with >= 2 samples gets a UMAP; every
    pair of groups gets a comparison."""
    from itertools import combinations

    n_trials = len(trial_dirs)

    for trial_dir in trial_dirs:
        tl = _trial_label(trial_dir)
        trial_out = os.path.join(out_dir, tl) if n_trials > 1 else out_dir

        # FMO thresholds for each set actually referenced by the groups
        fmo_thresh = _build_fmo_thresholds_for_groups(
            trial_dir, groups, fmo_sets, fmo_percentile, gate_overrides,
            report, trial_label_prefix=tl)

        # Bead-based µm/FSC calibration (used only if debris filter is on).
        debris_min_fsc = None
        if filter_debris_um and filter_debris_um > 0:
            factor, _ = _calibrate_bead_micron(trial_dir, fmo_sets, bead_um)
            if factor:
                debris_min_fsc = float(filter_debris_um) / factor
                print(f"[Calibration] Debris cut-off: {filter_debris_um:.1f} µm "
                      f"→ FSC-A >= {debris_min_fsc:,.0f}", flush=True)
            else:
                print(f"[Calibration] No readable bead file in {tl} — "
                      f"debris filter disabled for this trial.", flush=True)

        # Build per-sample tasks
        tasks = []
        for g in groups:
            for name in g['samples']:
                # Per-sample FMO set (override → group default → ungated).
                sfmo = g.get('sample_fmo', {}).get(name, g['fmo_set'])
                thresh = fmo_thresh.get(sfmo, {})
                if skip_existing and _already_done(trial_out, g['name'], name):
                    report(f'Skipping {name} (output already exists)')
                    continue
                # By-day groups carry their own source folder; fall back to
                # the run's trial_dir for the classic single-folder layout.
                src_dir = g.get('trial_dir') or trial_dir
                try:
                    path = fcs_path(src_dir, name)
                except FileNotFoundError as e:
                    report(f'[{tl}] {name}: {e}')
                    continue
                tasks.append({'name': name, 'group': g['name'],
                              'out': trial_out,
                              'paths': [(name, path)], 'thresholds': thresh,
                              'region_gates': region_gates,
                              'k': k, 'labels': labels, 'n_jobs': n_jobs,
                              'max_events': max_events,
                              'vram_admission_gb': vram_admission_gb,
                              'debris_min_fsc': debris_min_fsc,
                              'doublet_tol': doublet_tol,
                              'random_state': random_state,
                              'cytonorm': cytonorm,
                              'batch_id': src_dir})

        results = _run_sample_tasks(tasks, workers, admission_gb, report)

        # UMAP per group with >= 1 sample. Single-sample groups still get
        # the per-group UMAP rendered (the by-sample_origin panel is just
        # one colour in that case, but the by-cluster + FSC/SSC panels
        # are still meaningful and were previously being dropped).
        for g in groups:
            samples = results.get(g['name'], [])
            if not samples:
                continue
            report(f'[{tl}] Running UMAP ({g["name"]}, {len(samples)} sample(s)) …')
            run_group_umap(samples,
                           _safe_filename(f'{g["name"]}_{tl}'),
                           trial_out,
                           random_state=random_state)
            # Marker-pair scatters: per-sample grid + cross-sample overlay.
            save_group_pair_scatters(samples, f'{g["name"]}_{tl}',
                                     trial_out, pairs=pairs)

        # Pairwise comparisons across every pair of groups
        for g_a, g_b in combinations(groups, 2):
            _compare_group_pair(g_a, g_b, results, trial_out, report,
                                trial_label_prefix=tl)


# ── Concatenate mode ──────────────────────────────────────────────────────────

def _run_concatenated(trial_dirs, out_dir, groups, fmo_sets,
                      k, fmo_percentile, labels, gate_overrides, region_gates,
                      skip_existing,
                      workers, n_jobs, max_events, admission_gb,
                      vram_admission_gb, filter_debris_um, bead_um,
                      doublet_tol, report, random_state=42, pairs=None,
                      cytonorm=None):
    """Load each sample from every trial, concatenate with trial-origin labels,
    then cluster / UMAP once on the merged data per sample. Different
    samples are processed in parallel when workers > 1. FMO thresholds
    come from the first (reference) trial only."""
    from itertools import combinations

    ref_dir = trial_dirs[0]

    # FMO thresholds for each set actually referenced by the groups
    fmo_thresh = _build_fmo_thresholds_for_groups(
        ref_dir, groups, fmo_sets, fmo_percentile, gate_overrides,
        report, trial_label_prefix='ref trial')

    # Bead-based µm/FSC calibration from the reference trial.
    debris_min_fsc = None
    if filter_debris_um and filter_debris_um > 0:
        factor, _ = _calibrate_bead_micron(ref_dir, fmo_sets, bead_um)
        if factor:
            debris_min_fsc = float(filter_debris_um) / factor
            print(f"[Calibration] Debris cut-off: {filter_debris_um:.1f} µm "
                  f"→ FSC-A >= {debris_min_fsc:,.0f}", flush=True)
        else:
            print("[Calibration] No readable bead file in reference trial — "
                  "debris filter disabled.", flush=True)

    # Build per-sample tasks — paths span every trial folder
    tasks = []
    for g in groups:
        for name in g['samples']:
            # Per-sample FMO set (override → group default → ungated).
            sfmo = g.get('sample_fmo', {}).get(name, g['fmo_set'])
            thresh = fmo_thresh.get(sfmo, {})
            if skip_existing and _already_done(out_dir, g['name'], name):
                report(f'Skipping {name} (output already exists)')
                continue
            paths = []
            for trial_dir in trial_dirs:
                try:
                    paths.append((_trial_label(trial_dir),
                                   fcs_path(trial_dir, name)))
                except FileNotFoundError as e:
                    print(f"  [!] {e}")
            if not paths:
                report(f'{name}: not found in any trial folder')
                continue
            tasks.append({'name': name, 'group': g['name'], 'out': out_dir,
                          'paths': paths, 'thresholds': thresh,
                          'region_gates': region_gates,
                          'k': k, 'labels': labels, 'n_jobs': n_jobs,
                          'max_events': max_events,
                          'vram_admission_gb': vram_admission_gb,
                          'debris_min_fsc': debris_min_fsc,
                          'doublet_tol': doublet_tol,
                          'random_state': random_state,
                          'cytonorm': cytonorm,
                          'batch_id': g.get('trial_dir') or ref_dir})

    results = _run_sample_tasks(tasks, workers, admission_gb, report)

    # UMAP per group with >= 1 sample (see _run_independent for rationale).
    for g in groups:
        samples = results.get(g['name'], [])
        if not samples:
            continue
        report(f'Running UMAP ({g["name"]}, {len(samples)} sample(s)) …')
        run_group_umap(samples, _safe_filename(g['name']), out_dir,
                       random_state=random_state)
        save_group_pair_scatters(samples, g['name'], out_dir, pairs=pairs)

    # Pairwise comparisons across every pair of groups
    for g_a, g_b in combinations(groups, 2):
        _compare_group_pair(g_a, g_b, results, out_dir, report)


# ── Programmatic entry point ──────────────────────────────────────────────────
# `run(...)` takes keyword arguments and is suitable for importing from other
# Python code. `main()` (below) is the no-arg argparse-driven entry point
# that the `openflo-run` console script invokes.

def _proper_markers(sample):
    """Fluor marker channels with -H/-W detector versions dropped."""
    fl = list(getattr(sample, 'fluor_channels', None) or [])
    prim = [c for c in fl
            if not (c.endswith('-H') or c.endswith('-W')
                    or c.endswith(' H') or c.endswith(' W'))]
    return prim or fl


def _fit_cytonorm(groups, default_trial, mode='goal', n_metaclusters=10,
                  control_token='', max_events_per_batch=20_000, report=None):
    """Fit a CytoNorm model across batches (each group's ``trial_dir``).
    'goal' pools all samples per batch; 'controls' pools only samples whose
    name contains ``control_token``. Returns ``(model_dict, channels)`` or
    ``(None, None)`` when it can't (fewer than 2 usable batches/markers)."""
    import numpy as np
    import pandas as pd

    from .pipeline import CytoNorm
    say = report or (lambda m: print(m, flush=True))
    by_batch = {}
    for g in groups:
        tdir = g.get('trial_dir') or default_trial
        for raw in g['samples']:
            nm = raw.get('name') if isinstance(raw, dict) else raw
            if not nm:
                continue
            nm = str(nm)
            if (mode == 'controls' and control_token
                    and control_token.lower() not in nm.lower()):
                continue
            try:
                by_batch.setdefault(tdir, []).append(fcs_path(tdir, nm))
            except FileNotFoundError:
                continue
    if len(by_batch) < 2:
        say(f"[batch-correct] need ≥2 batches with samples (got "
            f"{len(by_batch)}) — skipped.")
        return None, None

    loaded, shared, ref = {}, None, None
    for batch, paths in by_batch.items():
        frames = []
        for path in paths:
            try:
                s = FlowSample(path)
                s.run_qc()
                s.auto_compensate()
                s.apply_transform()
            except Exception as exc:
                say(f"[batch-correct] skip {os.path.basename(path)}: "
                    f"{type(exc).__name__}: {exc}")
                continue
            ch = set(_proper_markers(s))
            shared = ch if shared is None else (shared & ch)
            ref = ref or s
            d = s.data
            if len(d) > max_events_per_batch:
                d = d.sample(max_events_per_batch, random_state=42)
            frames.append(d)
        if frames:
            loaded[batch] = frames
    if ref is None or shared is None or len(shared) < 2 or len(loaded) < 2:
        say("[batch-correct] insufficient shared markers / batches — skipped.")
        return None, None

    channels = [c for c in _proper_markers(ref) if c in shared]
    events_by_batch = {
        batch: pd.concat([d[channels] for d in frames], ignore_index=True)
        for batch, frames in loaded.items()}
    cn = CytoNorm(channels, n_metaclusters=n_metaclusters, mode=mode).fit(
        events_by_batch)
    qc = cn.qc(events_by_batch)
    b = float(np.mean([v['before'] for v in qc.values()]))
    a = float(np.mean([v['after'] for v in qc.values()]))
    say(f"[batch-correct] CytoNorm ({mode}): {len(channels)} markers, "
        f"{len(events_by_batch)} batches, mean batch→goal "
        f"{b:.3f} → {a:.3f}.")
    return cn.to_dict(), channels


def run(trial_dirs, out_dir, groups, fmo_sets=None, batch_mode='independent',
        k=30, fmo_percentile=99.5, labels=None, gate_overrides=None,
        region_gates=None,
        show_plots=False, skip_existing=False, workers=1, max_events=None,
        admission_gb=4.0, vram_admission_gb=1.0,
        filter_debris_um=0.0, bead_um=8.0, doublet_tol=0.0,
        random_state=42, palette_name='auto', pairs=None,
        batch_correct=False, cytonorm_mode='goal', cytonorm_metaclusters=10,
        cytonorm_control=''):
    """Run the full pipeline for a list of user-defined sample groups.

    groups
        List of dicts: ``[{'name': 'Day 1', 'samples': ['d1m1','d1m2'],
        'fmo_set': 'standard'}, ...]``. Supports any number of named groups
        (group 1 vs group 2, day 1 .. day 8, treated vs control, etc).

    fmo_sets
        Dict mapping a set name to a ``{detector_channel: fcs_basename}``
        dict that the FMO threshold builder needs. Defaults to
        ``DEFAULT_FMO_SETS`` (the Late / Early pair).
    """

    global _KEEP_FINAL_PLOTS
    _KEEP_FINAL_PLOTS = bool(show_plots)

    if isinstance(trial_dirs, str):
        trial_dirs = [trial_dirs]
    trial_dirs = [d for d in trial_dirs if d]

    if fmo_sets is None:
        fmo_sets = DEFAULT_FMO_SETS
    groups = _normalise_groups(groups)
    if not groups:
        print("[!] No groups defined — nothing to do.", flush=True)
        return

    os.makedirs(out_dir, exist_ok=True)

    # Batch correction (CytoNorm) — fit once across batches up front; the
    # serialized model rides along on each task and is applied in the worker.
    cytonorm_dict = None
    if batch_correct:
        default_trial = trial_dirs[0] if trial_dirs else '.'
        cytonorm_dict, _ = _fit_cytonorm(
            groups, default_trial, mode=cytonorm_mode,
            n_metaclusters=cytonorm_metaclusters,
            control_token=cytonorm_control)

    from .pipeline import GPU_AVAILABLE, GPU_CLUSTERING_AVAILABLE, GPU_NAME
    if GPU_AVAILABLE:
        clu = "+ clustering (cuML + cuGraph)" if GPU_CLUSTERING_AVAILABLE else "(UMAP only)"
        print(f"[GPU] {GPU_NAME} detected — running UMAP {clu}", flush=True)
        if GPU_CLUSTERING_AVAILABLE:
            print(f"[GPU] Per-worker VRAM admission: ≥ {vram_admission_gb:.1f} GB free "
                  "to use GPU clustering (else CPU Phenograph).", flush=True)
    elif _gpu_present():
        print("[GPU] NVIDIA GPU detected, but RAPIDS cuML is not installed — "
              "UMAP & clustering run on CPU. Install cuML+cuGraph (Linux/WSL2) "
              "for GPU acceleration.", flush=True)
    else:
        print("[GPU] No CUDA GPU detected — UMAP will run on CPU (numba JIT)", flush=True)

    # Parallel sample processing.
    #
    # When running multiple outer workers in parallel we deliberately set
    # n_jobs=1 inside Phenograph instead of `cores // workers`. The reason:
    # phenograph's Jaccard step spawns a multiprocessing.Pool of size n_jobs,
    # and on Windows each of those inner children must re-import the full
    # numpy/scipy/sklearn stack (~250 MB) AND receive a fresh copy of the
    # events array — so n_jobs=6 inside each outer worker means ~7× the
    # per-worker RAM footprint. Forcing n_jobs=1 keeps every outer worker
    # to a single process; total parallelism is simply `workers` (each one
    # runs Louvain on its own core, which is the single-threaded bottleneck
    # anyway).
    workers = max(1, int(workers))
    cores   = os.cpu_count() or 2
    n_jobs  = -1 if workers <= 1 else 1

    # RAM-aware throttling — cap workers to what free RAM can actually hold.
    try:
        import psutil
        free_gb  = psutil.virtual_memory().available / (1024 ** 3)
        # Per-worker peak estimate (calibrated against observed Windows runs).
        # Each outer worker holds, at peak:
        #   • imports (numpy/scipy/sklearn/phenograph/igraph)         ~0.5 GB
        #   • kNN graph (indices + distances)                          events × ~30 B
        #   • Jaccard sparse graph + intermediate copies              events × ~80 B
        #   • Louvain native binary-file scratch + igraph state       events × ~30 B
        #   • Phenograph's pickled return buffers + igraph leiden     events × ~15 B
        # → ≈ 0.5 GB + events × 4 µ-GB.  For 500 k events that's ~2.5 GB,
        # which matched the working-set we saw before the pagefile gave out.
        events_per_worker = max_events if max_events else 3_000_000
        per_worker_gb     = max(1.0, 0.5 + events_per_worker * 4.0e-6)
        # Reserve 2 GB for the parent process + OS headroom.
        budget_gb         = max(1.0, free_gb - 2.0)
        max_safe          = max(1, int(budget_gb / per_worker_gb))
        if workers > max_safe:
            print(f"[RAM-throttle] Free RAM {free_gb:.1f} GB, per-worker estimate "
                  f"~{per_worker_gb:.1f} GB → reducing workers {workers} → {max_safe}",
                  flush=True)
            workers = max_safe
    except ImportError:
        pass

    n_trials = len(trial_dirs)
    cap_msg  = f"  |  cluster cap: {max_events:,} events" if max_events else ""
    group_msg = " / ".join(f"{g['name']}({len(g['samples'])})" for g in groups)
    print(f"[Trials] {n_trials} trial folder(s)  |  mode: {batch_mode}  |  "
          f"parallel workers: {workers}  ({cores} cores){cap_msg}", flush=True)
    print(f"[Groups] {group_msg}", flush=True)

    # Step total — one [STEP N/M] line per unit of work the runners will do:
    #   • one per unique FMO set built  (per trial in independent mode,
    #     once in concatenate mode)
    #   • one per sample processed (or skipped)
    #   • one per group that gets a UMAP run (any group with >= 2 samples)
    #   • one per pair of groups (pairwise condition comparison)
    # Unique FMO sets across group defaults + per-sample overrides
    # (one build step each; '' = ungated builds nothing).
    _all_sets = set()
    for g in groups:
        _all_sets.add(g.get('fmo_set', ''))
        _all_sets.update(g.get('sample_fmo', {}).values())
    n_used_sets = len({s for s in _all_sets if s})
    n_samples   = sum(len(g['samples']) for g in groups)
    n_umaps     = sum(1 for g in groups if len(g['samples']) >= 2)
    n_compares  = (len(groups) * (len(groups) - 1)) // 2
    per_trial_steps = n_used_sets + n_samples + n_umaps + n_compares
    total = (per_trial_steps if batch_mode == 'concatenate'
             else n_trials * per_trial_steps)
    total = max(total, 1)   # avoid 0/0 in the progress bar

    step = [0]

    def report(msg):
        step[0] += 1
        print(f'[STEP {step[0]}/{total}] {msg}', flush=True)

    # Propagate the palette choice to the worker side too — if main()
    # is being called from a script that didn't go through __main__,
    # the palette set earlier in __main__ won't apply here.
    if palette_name and palette_name != 'auto':
        from .pipeline import set_default_palette
        set_default_palette(palette_name)

    if batch_mode == 'concatenate':
        _run_concatenated(trial_dirs, out_dir, groups, fmo_sets,
                          k, fmo_percentile, labels, gate_overrides,
                          region_gates,
                          skip_existing, workers, n_jobs, max_events,
                          admission_gb, vram_admission_gb,
                          filter_debris_um, bead_um, doublet_tol, report,
                          random_state=random_state, pairs=pairs,
                          cytonorm=cytonorm_dict)
    else:
        _run_independent(trial_dirs, out_dir, groups, fmo_sets,
                         k, fmo_percentile, labels, gate_overrides,
                         region_gates,
                         skip_existing, workers, n_jobs, max_events,
                         admission_gb, vram_admission_gb,
                         filter_debris_um, bead_um, doublet_tol, report,
                         random_state=random_state, pairs=pairs,
                         cytonorm=cytonorm_dict)

    print(f"\nDone. Results in: {out_dir}")
    if show_plots:
        # If nothing fresh was drawn (e.g. every sample was --skip-existing'd
        # and no group-UMAPs ran because there weren't enough live samples
        # to UMAP), reload the saved group-level PNGs from disk so the user
        # can still review them interactively.
        if not plt.get_fignums():
            opened = _show_saved_group_plots(out_dir)
            if opened == 0:
                print("[Show plots] No figures to display — nothing fresh "
                      "this run and no saved group-level PNGs found in "
                      f"{out_dir}.", flush=True)
        plt.show()


def _load_unmix_controls(spec):
    """Resolve the ``--unmix-controls`` spec (a JSON file path or inline JSON)
    to a ``{fluor: fcs_path}`` dict (with an optional ``unstained`` key)."""
    spec = (spec or '').strip()
    if not spec:
        raise ValueError("--unmix needs --unmix-controls (fluor→FCS mapping).")
    if os.path.isfile(spec):
        with open(spec, encoding='utf-8') as f:
            mapping = json.load(f)
    else:
        mapping = json.loads(spec)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("--unmix-controls must be a non-empty JSON object "
                         "mapping fluor names to FCS paths.")
    return {str(k): str(v) for k, v in mapping.items()}


def run_batch_unmix(args) -> int:
    """Spectral batch-unmixing CLI mode. Builds reference spectra from the
    single-stain controls, unmixes each input FCS into per-fluor abundance
    columns (written as CSV), and writes a QC report (similarity + spillover
    spread, Markdown + JSON) plus a reference-spectra plot. Returns 0 on
    success, non-zero on a setup error."""
    import glob

    from .pipeline import FlowSample
    from .spectral import (
        apply_unmixing,
        build_reference_spectra,
        unmixing_qc,
    )

    out_dir = args.out or 'outputs'
    os.makedirs(out_dir, exist_ok=True)

    try:
        controls = _load_unmix_controls(args.unmix_controls)
    except (ValueError, json.JSONDecodeError, OSError) as exc:
        print(f"[unmix] bad --unmix-controls: {exc}", flush=True)
        return 2

    unstained_path = controls.pop('unstained', None)
    if not controls:
        print("[unmix] no single-stain controls given (only 'unstained').",
              flush=True)
        return 2

    # Load the single-stain controls (raw detector signal — spectral data is
    # unmixed, not compensated).
    def _load_raw(path):
        s = FlowSample(path)
        s.run_qc()
        return s

    single_samples = {}
    for fluor, path in controls.items():
        if not os.path.isfile(path):
            print(f"[unmix] control for {fluor!r} not found: {path}",
                  flush=True)
            return 2
        single_samples[fluor] = _load_raw(path)
    first = next(iter(single_samples.values()))

    # Detectors: 'auto' → the first control's fluorescence channels.
    if args.unmix_detectors and args.unmix_detectors != 'auto':
        detectors = [d.strip() for d in args.unmix_detectors.split(',')
                     if d.strip()]
    else:
        detectors = list(getattr(first, 'fluor_channels', []) or [])
    if len(detectors) < 2:
        print("[unmix] need ≥2 detector channels (got "
              f"{len(detectors)}). Use --unmix-detectors.", flush=True)
        return 2

    stains = {}
    for fluor, s in single_samples.items():
        cols = [d for d in detectors if d in s.raw.columns]
        if len(cols) != len(detectors):
            print(f"[unmix] control {fluor!r} is missing some detectors — "
                  "skipped.", flush=True)
            continue
        stains[fluor] = s.raw[detectors].to_numpy(dtype=float)
    if len(stains) < 1:
        print("[unmix] no usable single-stain controls.", flush=True)
        return 2

    un = None
    if unstained_path and os.path.isfile(unstained_path):
        us = _load_raw(unstained_path)
        cols = [d for d in detectors if d in us.raw.columns]
        if len(cols) == len(detectors):
            un = us.raw[detectors].to_numpy(dtype=float)

    spectra, fluors = build_reference_spectra(stains, unstained=un)
    print(f"[unmix] built {len(fluors)} reference spectra over "
          f"{len(detectors)} detectors.", flush=True)

    # Resolve the inputs to unmix.
    raw_in = args.unmix_input or args.fcs or args.trials or ''
    inputs = []
    for tok in raw_in.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if os.path.isdir(tok):
            inputs.extend(sorted(glob.glob(os.path.join(tok, '*.fcs'))))
        elif os.path.isfile(tok):
            inputs.append(tok)
    # Don't unmix the control files themselves.
    control_paths = {os.path.abspath(p) for p in controls.values()}
    if unstained_path:
        control_paths.add(os.path.abspath(unstained_path))
    inputs = [p for p in inputs if os.path.abspath(p) not in control_paths]

    qc_stains = dict(stains)
    if un is not None and 'Autofluorescence' in fluors:
        qc_stains['Autofluorescence'] = un
    qc = unmixing_qc(qc_stains, spectra, fluors, nonneg=bool(args.unmix_nonneg))

    # Write the QC report (Markdown + JSON) and the reference-spectra plot.
    _write_unmix_qc(out_dir, qc, spectra, fluors, detectors)

    n_done = 0
    for path in inputs:
        try:
            s = _load_raw(path)
            cols = [d for d in detectors if d in s.raw.columns]
            if len(cols) != len(detectors):
                print(f"[unmix] {os.path.basename(path)}: missing detectors — "
                      "skipped.", flush=True)
                continue
            # Unmix the RAW detector matrix; FlowSample.apply_unmixing reads
            # from .data, so point a lightweight view at .raw.
            import types as _types
            view = _types.SimpleNamespace(data=s.raw)
            apply_unmixing(view, spectra, fluors, detectors,
                           nonneg=bool(args.unmix_nonneg))
            stem = os.path.splitext(os.path.basename(path))[0]
            out_csv = os.path.join(out_dir, f'{stem}_unmixed.csv')
            ucols = [f'U:{f}' for f in fluors]
            view.data[ucols].to_csv(out_csv, index=False)
            n_done += 1
        except Exception as exc:
            print(f"[unmix] {os.path.basename(path)}: "
                  f"{type(exc).__name__}: {exc}", flush=True)

    cond = qc['condition_number']
    cond_txt = "inf" if cond == float('inf') else f"{cond:.1f}"
    print(f"[unmix] done: {n_done} sample(s) unmixed → {out_dir} "
          f"(condition number {cond_txt}, "
          f"{len(qc['similar_pairs'])} similar pair(s)).", flush=True)
    if not inputs:
        print("[unmix] (no input FCS to unmix — built spectra + QC only; "
              "pass --unmix-input).", flush=True)
    return 0


def _write_unmix_qc(out_dir, qc, spectra, fluors, detectors):
    """Write the spectral-QC Markdown + JSON report and a reference-spectra
    PNG into ``out_dir``."""
    import numpy as np

    cond = qc['condition_number']
    cond_txt = "inf" if cond == float('inf') else f"{cond:.2f}"
    md = ["# Spectral unmixing QC", "",
          f"- **fluors**: {len(fluors)}",
          f"- **detectors**: {len(detectors)}",
          f"- **condition_number**: {cond_txt}", "",
          "## Spectrally-similar pairs"]
    if qc['similar_pairs']:
        md += ["| Fluor A | Fluor B | Cosine similarity |", "|---|---|---|"]
        md += [f"| {d['fluor_a']} | {d['fluor_b']} | {d['similarity']:.4f} |"
               for d in qc['similar_pairs']]
    else:
        md.append("None above threshold.")
    md += ["", "## Largest spillover spread"]
    if qc['worst_spread']:
        md += ["| Into | From | Spread |", "|---|---|---|"]
        md += [f"| {d['into']} | {d['from']} | {d['spread']:.4g} |"
               for d in qc['worst_spread']]
    else:
        md.append("No measured spread (single-stain controls missing).")
    with open(os.path.join(out_dir, 'spectral_qc.md'), 'w',
              encoding='utf-8') as f:
        f.write("\n".join(md) + "\n")

    payload = {
        'format': 'openflo-spectral-qc', 'version': 1,
        'fluors': list(fluors), 'detectors': list(detectors),
        'condition_number': (None if cond == float('inf') else cond),
        'similarity': np.asarray(qc['similarity']).tolist(),
        'ssm': [[None if not np.isfinite(v) else float(v) for v in row]
                for row in np.asarray(qc['ssm'])],
        'similar_pairs': qc['similar_pairs'],
        'worst_spread': qc['worst_spread'],
    }
    with open(os.path.join(out_dir, 'spectral_qc.json'), 'w',
              encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, fl in enumerate(fluors):
            ax.plot(range(spectra.shape[1]), spectra[i], marker='o', ms=2,
                    lw=1.2, label=fl)
        ax.set_xlabel('detector')
        ax.set_ylabel('normalized signal')
        ax.set_title('Reference spectra')
        ax.legend(fontsize=7, loc='best')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'reference_spectra.png'), dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[unmix] spectra plot skipped: {exc}", flush=True)


def main() -> int:
    """argparse-driven entry point for the ``openflo-run`` console script.

    Mirrors the historical ``python run_analysis.py ...`` invocation
    exactly. Returns 0 on clean exit; non-zero from argparse or an
    unhandled exception bubbles via sys.exit / SystemExit.
    """
    import multiprocessing
    multiprocessing.freeze_support()

    ap = argparse.ArgumentParser()
    ap.add_argument('--trials',      default='',
                    help='Comma-separated list of FCS trial directories')
    ap.add_argument('--fcs',         default='',
                    help='Single FCS directory (alias for --trials with one folder)')
    ap.add_argument('--out',         default='outputs', help='Output directory')
    ap.add_argument('--groups',      default='',
                    help='JSON list of groups: '
                         '\'[{"name":"Day1","samples":["d1m1","d1m2"],'
                         '"fmo_set":"standard"}, ...]\'. Replaces --samples '
                         'when given.')
    ap.add_argument('--fmo-sets',    default='', dest='fmo_sets',
                    help='JSON dict of FMO sets: '
                         '\'{"standard": {"BV421-A":"fmo_bv421", ...}}\'. '
                         'Defaults to the built-in example sets.')
    ap.add_argument('--samples',     default='',
                    help='LEGACY: comma-separated sample names; names starting '
                         'with "late" go to "Group B", others to "Group A". '
                         'Use --groups for full control.')
    ap.add_argument('--batch-mode',  default='independent',
                    choices=['independent', 'concatenate'], dest='batch_mode',
                    help='independent: one analysis per trial | '
                         'concatenate: merge trials, cells retain trial-of-origin label')
    ap.add_argument('--workers',     type=int, default=1,
                    help='Number of samples to process in parallel (default 1)')
    ap.add_argument('--max-events',  type=int, default=0, dest='max_events',
                    help='Cap clustering at this many events per sample '
                         '(0 = no cap). When exceeded, Phenograph runs on a '
                         'random sub-sample and the rest are assigned via '
                         'nearest-neighbour. Recommended: ~500000.')
    ap.add_argument('--admission-ram', type=float, default=4.0, dest='admission_gb',
                    help='Free RAM (GB) required before submitting a new '
                         'parallel worker. 0 disables admission control.')
    ap.add_argument('--admission-vram', type=float, default=1.0, dest='vram_admission_gb',
                    help='Free VRAM (GB) required for a worker to take the GPU '
                         'clustering branch (cuML+cuGraph). Below this it uses '
                         'CPU Phenograph. 0 = always try GPU when available.')
    ap.add_argument('--filter-debris-um', type=float, default=0.0,
                    dest='filter_debris_um',
                    help='Drop events whose FSC-A falls below this size in µm. '
                         '0 disables the debris filter. The conversion µm→FSC '
                         'is calibrated per trial from the comp-bead controls '
                         '(see --bead-um). Recommended: 4.0')
    ap.add_argument('--bead-um', type=float, default=8.0, dest='bead_um',
                    help='Diameter (µm) of the comp-bead control used for FSC '
                         'calibration. Default 8.0 (BD CompBeads / Invitrogen '
                         'UltraComp). Spherotech is 7.5.')
    ap.add_argument('--doublet-tol', type=float, default=0.0, dest='doublet_tol',
                    help='Doublet exclusion: keep events whose FSC-A/FSC-H is '
                         'within ±tol of the population median. 0 disables. '
                         'Recommended: 0.25 for polyploid samples, 0.15 '
                         'for typical diploid leukocytes.')
    ap.add_argument('--k',              type=int,   default=30,   help='Phenograph k')
    ap.add_argument('--fmo-percentile', type=float, default=99.5, dest='fmo_percentile')
    ap.add_argument('--labels', default='',
                    help='Channel labels: "DetA=LabelA;DetB=LabelB"')
    ap.add_argument('--panel', default='',
                    help='Path to a staining-panel .xlsx (CD↔fluorophore '
                         'rows). Resolved to {detector: CD} labels and '
                         'merged with --labels (--labels wins on conflict). '
                         'Pass "auto" to search the trial folder(s) and a '
                         'few ancestor levels for one.')
    ap.add_argument('--pairs', default='',
                    help='Marker pairs for the per-group overlay + grid '
                         'scatters, e.g. "CD34/CD11b,CD11b/CD45,CD34/CD45". '
                         'Default: those three. Labels must be assigned '
                         '(via --panel / --labels) for a pair to render.')
    ap.add_argument('--batch-correct', action='store_true', dest='batch_correct',
                    help='CytoNorm batch-normalize across batches (each '
                         'trial/day folder) before clustering.')
    ap.add_argument('--cytonorm-mode', default='goal',
                    choices=['goal', 'controls'], dest='cytonorm_mode',
                    help="'goal' (CytoNorm 2.0, control-free, default) fits on "
                         "all samples; 'controls' (classic) fits only on "
                         "per-batch control samples — see --cytonorm-control.")
    ap.add_argument('--cytonorm-control', default='', dest='cytonorm_control',
                    help="Filename token identifying the per-batch control "
                         "sample(s) for --cytonorm-mode controls (e.g. 'ref').")
    ap.add_argument('--cytonorm-metaclusters', type=int, default=10,
                    dest='cytonorm_metaclusters',
                    help='FlowSOM metaclusters for CytoNorm (default 10).')
    # ── Spectral batch-unmixing mode (full-spectrum cytometers) ──────────
    ap.add_argument('--unmix', action='store_true',
                    help='Batch spectral-unmixing mode: build reference '
                         'spectra from single-stain controls and unmix a set '
                         'of FCS into per-fluor abundances + a QC report. '
                         'Bypasses the standard analysis pipeline.')
    ap.add_argument('--unmix-controls', default='', dest='unmix_controls',
                    help='Single-stain controls for --unmix: a path to a JSON '
                         'file OR inline JSON mapping fluor→FCS path, e.g. '
                         '\'{"FITC":"ss_fitc.fcs","PE":"ss_pe.fcs",'
                         '"unstained":"unstained.fcs"}\'. The optional '
                         '"unstained" key adds an autofluorescence endmember.')
    ap.add_argument('--unmix-input', default='', dest='unmix_input',
                    help='FCS to unmix in --unmix mode: a directory or a '
                         'comma-separated list of files. Defaults to '
                         '--fcs / --trials.')
    ap.add_argument('--unmix-detectors', default='auto', dest='unmix_detectors',
                    help="Detector channels for --unmix: 'auto' (default — the "
                         "fluorescence channels of the first control) or a "
                         "comma-separated channel list.")
    ap.add_argument('--unmix-nonneg', action='store_true', dest='unmix_nonneg',
                    help='Clip unmixed abundances at 0 (non-negative).')
    ap.add_argument('--show-plots',  action='store_true', dest='show_plots')
    ap.add_argument('--gates',       default='',
                    help='JSON gate overrides e.g. \'{"BV421-A": 0.5}\'')
    ap.add_argument('--skip-existing', action='store_true', dest='skip_existing')
    ap.add_argument('--export-wsp', default='', dest='export_wsp',
                    help='After the run, write a FlowJo-compatible .wsp '
                         'workspace at this path. Contains the gates '
                         'that were applied (FMO thresholds + --gates) '
                         'and one SampleNode per input FCS.')
    ap.add_argument('--seed', type=int, default=42,
                    help='Random seed used for sample subsampling and '
                         'UMAP. Held constant across every sample in a '
                         'trial so results are reproducible across runs.')
    ap.add_argument('--palette', default='auto',
                    help='Categorical palette for per-sample / per-condition '
                         'plot colouring. Default: auto (tab10 for ≤10 groups, '
                         'tab20 for 11-20, gist_ncar above). Common '
                         'alternatives: tab10, Set1, Set2, Dark2, Paired.')
    ap.add_argument('-v', '--verbose', action='count', default=0,
                    help='Increase log verbosity (repeat for more: -v=INFO '
                         '[default], -vv=DEBUG). Mutually exclusive with -q.')
    ap.add_argument('-q', '--quiet', action='store_true',
                    help='Suppress INFO messages — only WARNING and above.')
    args = ap.parse_args()

    # Configure the root logger ONCE based on -v / -q. Done before any
    # `from .pipeline import ...` runs in this process so worker subprocesses
    # inherit a sensible default via env. The actual handlers / format are
    # left at defaults — users wanting JSON / file output should configure
    # `logging` themselves before importing openflo.
    if args.quiet:
        _log_level = logging.WARNING
    elif args.verbose >= 2:
        _log_level = logging.DEBUG
    else:
        _log_level = logging.INFO
    logging.basicConfig(
        level=_log_level,
        format='%(asctime)s %(levelname)-7s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
        force=True,    # override any prior configuration
    )

    # Spectral batch-unmixing is a self-contained mode — run it and exit
    # before the standard analysis-pipeline argument resolution.
    if args.unmix:
        return run_batch_unmix(args)

    # Resolve trial dirs: --trials takes precedence, else --fcs
    raw_trials = args.trials or args.fcs or '.'
    trial_dirs = [t.strip() for t in raw_trials.split(',') if t.strip()]

    # Resolve groups / fmo_sets — four paths in order of preference:
    #   1. --groups (explicit) → use as-is. Pair with --fmo-sets if given.
    #   2. --samples (legacy) → auto-split by "late" prefix into two groups.
    #   3. Neither, but folders contain FCS → by-DAY auto-grouping: each
    #      folder (incl. sub-folders of a parent you point at) becomes a
    #      day/group, sampled independently, compared across days in one
    #      analysis. This is the new default.
    #   4. Fallback → a single generic group over the named samples.
    if args.groups:
        groups = json.loads(args.groups)
        fmo_sets = (json.loads(args.fmo_sets)
                    if args.fmo_sets else DEFAULT_FMO_SETS)
    elif args.samples:
        names = [s.strip() for s in args.samples.split(',') if s.strip()]
        # Optional two-group split on a name prefix (override with --groups).
        group_b = [n for n in names if n.lower().startswith('late')]
        group_a = [n for n in names if not n.lower().startswith('late')]
        groups = []
        if group_a:
            groups.append({'name': 'Group A',
                           'samples': group_a, 'fmo_set': 'Set A'})
        if group_b:
            groups.append({'name': 'Group B',
                           'samples': group_b, 'fmo_set': 'Set B'})
        fmo_sets = (json.loads(args.fmo_sets)
                    if args.fmo_sets else DEFAULT_FMO_SETS)
    else:
        auto = _auto_groups_by_day(trial_dirs)
        if auto:
            groups = auto
            fmo_sets = (json.loads(args.fmo_sets)
                        if args.fmo_sets else DEFAULT_FMO_SETS)
            print(f"[grouping] By-day: {len(auto)} folder(s) → "
                  f"{', '.join(g['name'] for g in auto)}", flush=True)
            # Each group carries its own trial_dir, so collapse the run to
            # a SINGLE pass (one combined analysis comparing days). The
            # representative trial_dir below is only used for labels /
            # bead calibration fallbacks.
            trial_dirs = [trial_dirs[0] if trial_dirs else '.']
        else:
            groups   = DEFAULT_GROUPS
            fmo_sets = DEFAULT_FMO_SETS

    gate_overrides, region_gates = _parse_gates_arg(args.gates)
    parsed_overrides = gate_overrides   # keep the dict form for the wsp export
    if not gate_overrides and not region_gates:
        gate_overrides = None   # preserve the original "no gates" semantics

    # Channel labels: staining-panel xlsx first, then --labels overrides.
    labels = {}
    panel_path = args.panel
    if panel_path == 'auto':
        panel_path = find_panel_xlsx(trial_dirs)
        if panel_path:
            print(f"[panel] Auto-detected: {panel_path}", flush=True)
    if panel_path:
        try:
            probe_dir = groups[0].get('trial_dir') or trial_dirs[0]
            probe = fcs_path(probe_dir, groups[0]['samples'][0])
            channels = list(FlowSample(probe).data.columns)
            labels.update(read_staining_panel(panel_path, channels))
        except Exception as exc:
            print(f"[panel] Could not apply panel: "
                  f"{type(exc).__name__}: {exc}", flush=True)
    labels.update(parse_labels(args.labels))   # explicit --labels win
    pairs = parse_pairs(args.pairs)

    # Apply the user's palette choice process-wide before any plot is
    # rendered. Worker processes inherit via os.environ-free fork so the
    # function reset below is enough for the parent; per-worker callers
    # also re-set after their imports (see _process_sample_task).
    if args.palette and args.palette != 'auto':
        from .pipeline import set_default_palette
        set_default_palette(args.palette)

    run(
        trial_dirs     = trial_dirs,
        out_dir        = args.out,
        groups         = groups,
        fmo_sets       = fmo_sets,
        batch_mode     = args.batch_mode,
        k              = args.k,
        fmo_percentile = args.fmo_percentile,
        labels         = labels,
        gate_overrides = gate_overrides,
        region_gates   = region_gates,
        show_plots     = args.show_plots,
        skip_existing  = args.skip_existing,
        workers        = args.workers,
        max_events     = args.max_events or None,
        admission_gb   = args.admission_gb,
        vram_admission_gb = args.vram_admission_gb,
        filter_debris_um = args.filter_debris_um,
        bead_um          = args.bead_um,
        doublet_tol      = args.doublet_tol,
        random_state     = args.seed,
        palette_name     = args.palette,
        pairs            = pairs,
        batch_correct    = args.batch_correct,
        cytonorm_mode    = args.cytonorm_mode,
        cytonorm_control = args.cytonorm_control,
        cytonorm_metaclusters = args.cytonorm_metaclusters,
    )

    if args.export_wsp:
        _export_pipeline_workspace(
            out_path       = args.export_wsp,
            trial_dirs     = trial_dirs,
            groups         = groups,
            gate_overrides = parsed_overrides,
            region_gates   = region_gates)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
