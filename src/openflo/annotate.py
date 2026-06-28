"""Automated population annotation.

Two complementary, dependency-light ways to turn numeric cluster IDs into
biological labels:

  * **MEM** (Marker Enrichment Modeling; Diggins et al. 2017) — a quantitative
    enrichment score per marker per population, against a reference (the rest of
    the cells by default), yielding labels like ``CD3⁺⁸ CD4⁺⁶ CD8⁻⁴``. Captures
    both the median shift and the spread (IQR) change.
  * **Reference-table** annotation (ACDC / Scyan style) — match each
    population's +/- marker pattern to a user-supplied ``name: CD3+ CD4+ CD8-``
    table and assign the best-fitting cell type.

Pure numpy / pandas — no deep learning. The GUI runs these on a clustered
sample and writes the names back onto the populations.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Subscript/superscript-free signed format keeps cp1252 stdout happy.
_SIGN = {1: '+', -1: '-'}


def scale_markers(X, lo_pct=1.0, hi_pct=99.0, out_max=10.0):
    """Scale each marker (column) to ``[0, out_max]`` using robust percentiles
    (``lo_pct``→0, ``hi_pct``→out_max), clipped — so the MEM magnitude term is
    comparable across markers on different intensity scales."""
    X = np.asarray(X, dtype=float)
    lo = np.nanpercentile(X, lo_pct, axis=0)
    hi = np.nanpercentile(X, hi_pct, axis=0)
    rng = np.where(hi > lo, hi - lo, 1.0)
    return np.clip((X - lo) / rng, 0.0, 1.0) * out_max


def _iqr(a):
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    return float(np.percentile(a, 75) - np.percentile(a, 25))


def mem_scores(data, labels, markers, reference=None, scale=True, eps=0.5):
    """Marker Enrichment Modeling scores.

    ``data`` : DataFrame or array (cells × len(markers)).
    ``labels`` : per-cell population label (array-like).
    ``markers`` : marker names (columns / order of ``data``).
    ``reference`` : a label to use as the reference population for every
        comparison; ``None`` (default) uses "all other cells" per population.

    For each population *p* and marker *m*::

        MAG = median_p − median_ref
        raw = |MAG| + IQR_ref / IQR_p − 1          (floored at 0)
        MEM = sign(MAG) · raw

    Scores are then globally rescaled so the largest |MEM| maps to 10 (the MEM
    convention). Returns a DataFrame indexed by population, columns = markers,
    of (rounded) MEM values in ``[-10, 10]``."""
    if isinstance(data, pd.DataFrame):
        X = data[list(markers)].to_numpy(dtype=float)
    else:
        X = np.asarray(data, dtype=float)
    labels = np.asarray(labels)
    if scale:
        X = scale_markers(X)
    pops = [p for p in pd.unique(labels) if p is not None and p == p]  # noqa: PLR0124
    pops = sorted(pops, key=str)

    rows = {}
    for p in pops:
        in_p = labels == p
        if int(in_p.sum()) == 0:
            continue
        ref_mask = (labels == reference) if reference is not None else ~in_p
        if int(ref_mask.sum()) == 0:
            ref_mask = ~in_p
        vals = np.empty(len(markers))
        for j in range(len(markers)):
            pj, rj = X[in_p, j], X[ref_mask, j]
            mag = float(np.nanmedian(pj) - np.nanmedian(rj))
            iqr_p = _iqr(pj) + eps
            iqr_r = _iqr(rj) + eps
            raw = abs(mag) + iqr_r / iqr_p - 1.0
            vals[j] = np.sign(mag) * max(raw, 0.0)
        rows[p] = vals

    mem = pd.DataFrame(rows, index=pd.Index(markers)).T  # pops × markers
    mx = float(np.nanmax(np.abs(mem.to_numpy()))) if mem.size else 0.0
    if mx > 0:
        mem = (mem / mx * 10.0).round(0)
    return mem


def mem_label(mem_row, markers=None, threshold=2.0, max_markers=8):
    """Build a MEM text label from one population's score row (a Series or
    array): markers sorted by descending |score|, kept above ``threshold``,
    formatted ``CD3+8 CD4+6 CD8-4``. Returns '' when nothing is enriched."""
    if isinstance(mem_row, pd.Series):
        items = [(m, float(v)) for m, v in mem_row.items()]
    else:
        items = list(zip(markers or [], [float(v) for v in mem_row],
                         strict=True))
    items = [(m, v) for m, v in items if abs(v) >= threshold]
    items.sort(key=lambda t: abs(t[1]), reverse=True)
    items = items[:max_markers]
    return ' '.join(f"{m}{_SIGN[int(np.sign(v))]}{abs(int(round(v)))}"
                    for m, v in items)


def population_states(mem, threshold=3.0):
    """Reduce a MEM DataFrame to per-population +/- marker states
    (``{population: {marker: +1 | -1}}``) for reference matching: a marker is
    ``+1`` above ``+threshold``, ``-1`` below ``-threshold``, omitted if
    ambiguous."""
    states = {}
    for pop, row in mem.iterrows():
        st = {}
        for m, v in row.items():
            if v >= threshold:
                st[m] = 1
            elif v <= -threshold:
                st[m] = -1
        states[pop] = st
    return states


def parse_signature_table(text):
    """Parse a reference cell-type table into ``{name: {marker: +1 | -1}}``.

    One cell type per line, ``Name: CD3+ CD4+ CD8-`` (or comma/whitespace
    separated). A marker token ending in ``+``/``hi`` is required-positive,
    ``-``/``lo`` required-negative. Blank lines and ``#`` comments are ignored.
    """
    table = {}
    for line in (text or '').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if ':' in line:
            name, rest = line.split(':', 1)
        else:
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            name, rest = parts
        name = name.strip()
        sig = {}
        for tok in re.split(r'[,\s]+', rest.strip()):
            if not tok:
                continue
            m = re.match(r'^(.*?)(\+\+?|--?|hi|lo|high|low)$', tok,
                         flags=re.IGNORECASE)
            if not m:
                continue
            marker, sign = m.group(1).strip(), m.group(2).lower()
            sig[marker] = -1 if sign.startswith(('-', 'lo', 'low')) else 1
        if name and sig:
            table[name] = sig
    return table


def annotate_by_reference(states, table, min_score=1):
    """Assign each population the best-matching cell type from ``table``.

    Scoring weights the *defining* (positive) markers: a matched required-
    positive marker is worth +2, a matched required-negative +1, and a
    contradiction (the population has the opposite sign) −2. A cell type is only
    eligible if at least one of its positive markers is present in the
    population (so a type can't win on a shared negative like ``CD3-`` alone).
    Returns ``{population: {name, score, n_match, n_required}}``; ``name`` is
    ``'unknown'`` when no eligible type scores ≥ ``min_score``. ``states`` comes
    from :func:`population_states`."""
    out = {}
    for pop, st in states.items():
        best = ('unknown', 0, 0, 0)
        for name, sig in table.items():
            pos = [m for m, w in sig.items() if w == 1]
            score = 0
            n_match = 0
            pos_hit = 0
            for marker, want in sig.items():
                have = st.get(marker, 0)
                if have == want:
                    n_match += 1
                    score += 2 if want == 1 else 1
                    pos_hit += (want == 1)
                elif have == -want:
                    score -= 2
            # A type with positive requirements must hit at least one of them.
            if pos and pos_hit == 0:
                continue
            if score > best[1]:
                best = (name, score, n_match, len(sig))
        name, score, n_match, n_req = best
        out[pop] = {'name': name if score >= min_score else 'unknown',
                    'score': int(score), 'n_match': int(n_match),
                    'n_required': int(n_req)}
    return out
