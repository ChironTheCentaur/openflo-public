"""openflo-run's batch correction: one CytoNorm model fitted across the
trial folders (batches) before the samples run, as the app's Normalize
batches does.

Moved out of cli.py unchanged; the old cli name still works, deprecated.
"""
from __future__ import annotations

import os

import numpy as np

from .cli_samples import fcs_path
from .pipeline import FlowSample


def fit_cytonorm(groups, default_trial, mode='goal', n_metaclusters=10,
                  control_token='', max_events_per_batch=20_000, report=None,
                  qc=False):
    """Fit a CytoNorm model across batches (each group's ``trial_dir``)
    with batch_normalize.fit_batch_model, as the app's Normalize batches
    does: each batch's samples pooled and capped at `max_events_per_batch`
    events with the same seeded draw. 'goal' pools all samples per batch;
    'controls' pools only samples whose name contains ``control_token``,
    and every batch must have one (ValueError otherwise, as in the app).
    Returns ``(model_dict, channels)`` or ``(None, None)`` when it can't
    (fewer than 2 usable batches/markers)."""
    import pandas as pd

    from .batch_normalize import fit_batch_model
    from .channel_roles import default_cluster_channels
    from .cytonorm import cap_events_per_batch
    say = report or (lambda m: print(m, flush=True))
    token = (control_token or '').strip().lower()
    if mode == 'controls' and not token:
        raise ValueError(
            "--cytonorm-mode controls needs --cytonorm-control: the name "
            "token of the reference sample run in every batch, e.g. ref")
    by_batch, seen = {}, []
    for g in groups:
        tdir = g.get('trial_dir') or default_trial
        for raw in g['samples']:
            nm = raw.get('name') if isinstance(raw, dict) else raw
            if not nm:
                continue
            nm = str(nm)
            try:
                path = fcs_path(tdir, nm)
            except FileNotFoundError:
                continue
            seen.append(tdir)
            if mode == 'controls' and token not in nm.lower():
                continue
            by_batch.setdefault(tdir, []).append(path)
    no_control = [b for b in dict.fromkeys(seen) if b not in by_batch]
    if no_control:
        raise ValueError(
            f"--cytonorm-mode controls: no sample containing "
            f"{control_token.strip()!r} in batch(es) "
            f"{', '.join(os.path.basename(str(b)) or str(b) for b in no_control)}"
            "; 'controls' mode needs the reference run in every batch")
    if len(by_batch) < 2:
        say(f"[batch-correct] need ≥2 batches with samples (got "
            f"{len(by_batch)}) — skipped.")
        return None, None

    frames, batch_of, shared, ref = {}, {}, None, None
    for batch, paths in by_batch.items():
        for path in paths:
            try:
                s = FlowSample(path)
                if qc:
                    s.run_qc()
                s.auto_compensate()
                s.apply_transform()
            except Exception as exc:  # noqa: BLE001 - bad file reported, skipped
                say(f"[batch-correct] skip {os.path.basename(path)}: "
                    f"{type(exc).__name__}: {exc}")
                continue
            ch = set(default_cluster_channels(
                getattr(s, 'fluor_channels', None)))
            shared = ch if shared is None else (shared & ch)
            ref = ref or s
            key = f'{batch}{os.sep}{os.path.basename(path)}'
            frames[key] = s.data
            batch_of[key] = str(batch)
    n_loaded = len(set(batch_of.values()))
    if ref is None or shared is None or len(shared) < 2 or n_loaded < 2:
        say("[batch-correct] insufficient shared markers / batches — skipped.")
        return None, None

    channels = [c for c in default_cluster_channels(
                    getattr(ref, 'fluor_channels', None)) if c in shared]
    references = None
    if mode == 'controls':
        references = {}
        for key, b in batch_of.items():
            references.setdefault(b, []).append(key)
    cn = fit_batch_model(frames, batch_of, channels, mode=mode,
                         references=references,
                         n_metaclusters=n_metaclusters,
                         max_events_per_batch=max_events_per_batch)
    # Scored on the events it was fitted on (the same capped draw).
    events_by_batch = cap_events_per_batch(
        {b: pd.concat([frames[k][channels] for k, kb in batch_of.items()
                       if kb == b], ignore_index=True)
         for b in cn.batches},
        max_events_per_batch, seed=cn.seed)
    qc_scores = cn.qc(events_by_batch)
    # Channels that could not be evaluated report NaN, so average across the
    # ones that were actually measured and say how many were not.
    b, a, skipped = cn.qc_means(qc_scores)
    note = f", {skipped} marker(s) not evaluable" if skipped else ""
    scores = "not evaluable" if not np.isfinite(b) else f"{b:.3f} → {a:.3f}"
    say(f"[batch-correct] CytoNorm ({mode}): {len(channels)} markers, "
        f"{len(events_by_batch)} batches, mean batch→goal "
        f"{scores}{note}.")
    return cn.to_dict(), channels
