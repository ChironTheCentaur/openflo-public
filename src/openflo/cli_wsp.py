"""``openflo-run --export-wsp``: a FlowJo-compatible .wsp holding the gates
the run applied, one SampleNode per FCS file, each with its own spillover.

Moved out of cli.py unchanged; the old cli names still work, deprecated.
"""
from __future__ import annotations

import os

from .cli_gates import pipeline_gate_transforms
from .cli_samples import fcs_path, normalise_groups


def build_export_gate_list(region_gates, gate_overrides):
    """Combine region gates (with their parent_id structure preserved)
    and threshold overrides (added as roots) into a flat list with
    fresh consecutive ids. Used by the --export-wsp path."""
    out = []
    seq = [0]
    def alloc():
        seq[0] += 1
        return f'g{seq[0]}'
    # Two passes so a child gate that appears BEFORE its parent in the list
    # still re-maps correctly: allocate every new id first, then resolve each
    # parent_id against the complete map (a single pass would leave a
    # forward-referencing child rooted at None → wrong hierarchy in FlowJo).
    old_to_new = {}
    prepared = []
    for g in (region_gates or []):
        d = dict(g)
        src_id = d.pop('_import_id', None) or d.pop('id', None)
        d['id'] = alloc()
        if src_id is not None:
            old_to_new[src_id] = d['id']
        prepared.append((d, g.get('parent_id')))
    for d, pid in prepared:
        d['parent_id'] = old_to_new.get(pid) if pid else None
        out.append(d)
    for ch, val in (gate_overrides or {}).items():
        out.append({'kind': 'threshold',
                    'channel': ch,
                    'value': float(val),
                    'id': alloc(),
                    'parent_id': None})
    return out


def own_spillover(fcs):
    """`(channels, matrix)` of the SPILL the pipeline applied to `fcs`
    (auto_compensate reads the same keyword), or None. It goes inside that
    sample in the .wsp, so FlowJo applies each file's own matrix."""
    try:
        from .compensation_io import read_compensation_matrix
        chans, mat = read_compensation_matrix(fcs)
    except Exception as exc:  # noqa: BLE001 - reported; no SPILL in .wsp
        print(f"[--export-wsp] {os.path.basename(fcs)}: spillover read failed: {exc}",
              flush=True)
        return None
    return None if chans is None or mat is None else (chans, mat)


def export_pipeline_workspace(out_path, trial_dirs, groups,
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
        from .wsp_writer import WspWriter
    except ImportError as exc:
        print(f"[--export-wsp] openflo.pipeline import failed: {exc}", flush=True)
        return

    # Normalise first: the raw --groups spec may carry `samples` as a comma
    # string or per-sample dicts ({'name':…,'fmo_set':…}). Without this the loop
    # would iterate a string char-by-char or hit dict sample "names" and resolve
    # ZERO samples (the run itself succeeds because run() normalises its copy).
    groups = normalise_groups(groups)
    w = WspWriter(cytometer='OpenFlo-pipeline')
    samples_added = 0
    seen = set()
    for grp in (groups or []):
        # Honour a per-group trial_dir (by-day grouping) so filename collisions
        # across day folders resolve to the RIGHT day's FCS; fall back to
        # searching the run's trial dirs for groups without one.
        grp_dir = grp.get('trial_dir')
        search_dirs = [grp_dir] if grp_dir else list(trial_dirs or ['.'])
        for sample_name in grp.get('samples', []):
            fp = None
            last_exc = 'not found'
            for d in search_dirs:
                try:
                    fp = fcs_path(d, sample_name)
                    break
                except FileNotFoundError as exc:
                    last_exc = exc
            if fp is None:
                print(f"[--export-wsp] {sample_name}: {last_exc}", flush=True)
                continue
            if (sample_name, fp) in seen:        # avoid duplicate SampleNodes
                continue
            seen.add((sample_name, fp))
            gates = build_export_gate_list(region_gates, gate_overrides)
            w.add_sample(
                name=sample_name,
                fcs_path=fp,
                channels=[],     # WspWriter doesn't currently need these
                gates=gates,
                compensation=own_spillover(fp),
                transforms=pipeline_gate_transforms(gates))
            samples_added += 1
    if not samples_added:
        print("[--export-wsp] no samples resolved; nothing to write",
              flush=True)
        return

    # Compensation: each sample already carries its own FCS SPILL (the
    # matrix auto_compensate applied to it); WspWriter lists the distinct
    # ones in <Matrices>. A sample without SPILL carries none.
    n_comp = sum(1 for s in w.samples if s.get('compensation') is not None)
    comp_note = ' + spillover' if n_comp else ''
    if n_comp < samples_added:
        print(f"[--export-wsp] {samples_added - n_comp} of {samples_added} "
              "sample(s) have no usable SPILL in FCS metadata — exported without "
              "a spillover matrix", flush=True)

    try:
        w.write(out_path)
        print(
            f"[--export-wsp] wrote {samples_added} sample(s){comp_note} "
            f"-> {out_path}", flush=True)
        # Where FlowJo will count differently from this run (see WspWriter).
        for msg in w.warnings:
            print(f"[--export-wsp] FlowJo will differ: {msg}", flush=True)
    except Exception as exc:  # noqa: BLE001 - reported; run results stand
        print(f"[--export-wsp] write failed: {exc}", flush=True)
