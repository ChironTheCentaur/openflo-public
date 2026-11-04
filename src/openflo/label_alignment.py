"""Merging samples into one events table: ``concatenate`` by detector, and
``concatenate_by_label`` by antibody label after ``align_fluor_labels`` has
paired each sample's channels; ``relabel_gate_for_sample`` and
``common_fluor_warning`` use the same pairing.

Core module: pandas and the standard library. No Tk, and it does not import
``openflo.pipeline``.
"""

import logging

import pandas as pd

# Name kept as 'openflo.pipeline' so existing log filters and caplog tests see
# these messages.
log = logging.getLogger('openflo.pipeline')


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
# The same antibody (e.g. CD901b) can sit on a different fluorophore /
# detector in different samples or on different days. Comparing or
# clustering ACROSS samples by detector name therefore mis-aligns
# phenotypes. These helpers align by the antibody LABEL instead, while
# compensation stays keyed on detectors (each sample compensates its own
# $SPILL upstream — labels never touch the comp math).

def sample_fluor_labels(sample):
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
    """Align a set of samples by antibody label, with the CLI's matcher.

    The pairing is :func:`openflo.shared_clustering.align_shared_channels`,
    the one the shared clustering uses, so the library and the CLI agree on
    which channels are one marker: labels compare without case, spacing,
    punctuation or a fluor word at either end, Greek letters as Latin; the
    -A/-H/-W suffix is part of the match; an unlabelled channel pairs by
    detector; two different labels on one detector are never pooled. See
    that function for the full rules.

    Parameters
    ----------
    samples : iterable of FlowSample-like
        Each needs ``.name``, ``.fluor_channels`` (detector names) and
        ``.channel_labels`` ({detector: antibody label}). ``.data`` may be
        None (every fluor channel then counts as present); otherwise only
        fluor channels that are columns of ``.data`` take part.

    Returns
    -------
    dict with:
      ``common``      the matched columns' names, one per column, in the
                      first sample's channel order (the matcher's display
                      name: the first sample's label, or the detector when
                      unlabelled)
      ``per_sample``  {sample_name: {name: detector}}: each matched column
                      under its common name, plus every unmatched channel
                      under its own label (its detector when unlabelled;
                      'label (detector)' where the bare label would clash
                      or is held, unmatched, by every sample)
      ``missing``     {name: [sample_names lacking it]} for the unmatched
                      channels
      ``all_labels``  every name above, in first-seen order over the
                      samples
    """
    samples = list(samples)
    names = [getattr(s, 'name', None) or f'sample{i}'
             for i, s in enumerate(samples)]
    if not samples:
        return {'common': [], 'per_sample': {}, 'missing': {},
                'all_labels': []}
    from .shared_clustering import align_shared_channels
    channels, columns, _notes, display = align_shared_channels(samples, names)
    common = [display[d] for d in channels]
    # Each sample's channels in its own order: (detector, matched name or
    # None, own label).
    rows = []
    for s, cols in zip(samples, columns, strict=True):
        matched = dict(zip(cols, common, strict=True))   # detector -> name
        labels = getattr(s, 'channel_labels', None) or {}
        data = getattr(s, 'data', None)
        rows.append([(det, matched.get(det), labels.get(det, det) or det)
                     for det in getattr(s, 'fluor_channels', None) or []
                     if data is None or det in data.columns])
    # An unmatched label every sample holds (CD3 on FL1-A in one, on FL1-H
    # in the other) is not one marker: name it with its detector there.
    held = {}
    for nm, row in zip(names, rows, strict=True):
        for _det, m, lbl in row:
            if m is None:
                held.setdefault(lbl, set()).add(nm)
    n = len(set(names))
    everywhere = {lbl for lbl, ns in held.items() if len(ns) == n}
    per_sample = {}
    order = []                       # first-seen name order
    have = {}                        # unmatched name -> names that have it
    for nm, row in zip(names, rows, strict=True):
        l2d = per_sample.setdefault(nm, {})
        for det, m, lbl in row:
            if m is None:
                if lbl in everywhere or lbl in l2d or lbl in common:
                    lbl = f'{lbl} ({det})'
                have.setdefault(lbl, set()).add(nm)
            else:
                lbl = m
            l2d[lbl] = det
            if lbl not in order:
                order.append(lbl)
    missing = {lbl: sorted(set(per_sample) - have[lbl])
               for lbl in order if lbl in have}
    return {'common': common, 'per_sample': per_sample,
            'missing': missing, 'all_labels': order}


def relabel_gate_for_sample(gate, label_to_detector):
    """Retarget a gate's channel fields to a specific sample's detectors
    by antibody label.

    A template gate carries both a detector channel (e.g. ``FL1-A``)
    and, when authored in a labelled editor, the antibody label it stood
    for (``x_label`` / ``y_label`` / ``label``). Applied to a sample
    where that marker sits on a DIFFERENT detector, we rewrite the
    channel to that sample's detector so one template ties phenotypes
    across panels. Compensation is unaffected — this only swaps which
    column the gate reads.

    label_to_detector : {antibody label: detector} for the target sample
        (e.g. from ``sample_fluor_labels``).

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
    panel, or '' when every channel pairs across them. Channels pair as
    :func:`align_fluor_labels` pairs them (so 'CD3' and 'cd-3' are one
    marker); lists which unmatched labels are missing from which samples
    and notes that cross-sample analysis uses the matched (common) set."""
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
    columns to their antibody LABEL and keeps ONLY the channels that pair
    across every sample (:func:`align_fluor_labels`, the CLI's matcher) —
    so a marker on different fluors, or spelt 'CD3' in one sample and
    'cd-3' in another, lines up into one column named by the first
    sample's label. Scatter / non-fluor columns are dropped from the
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
