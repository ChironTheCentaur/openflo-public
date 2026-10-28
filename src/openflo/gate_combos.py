"""Combination (Boolean) gates and polyfunctionality -- Tk-free.

A cytokine or activation panel is read as every Boolean combination of a set
of positive gates: with N gates there are 2^N populations, from all positive
to all negative, each a fraction of the population the gates were drawn in.
That is the standard summary for polyfunctional T-cell responses (SPICE;
Roederer, Nozzi and Nason, Cytometry A 2011), shown as bars per combination
with a +/- grid under them and pies of how many functions each cell has.

The combinations are ordinary boolean gates, so they appear in the gate
tree, the statistics, every export and the FlowJo workspace like any other:

* for each chosen gate A, one ``NOT A`` gate (OpenFlo's NOT: in the parent,
  outside A, and measured on A's channels -- an event with no value is not
  negative);
* for each sign pattern, one ``AND`` gate over A or ``NOT A`` per marker.

All of them sit under the lowest common parent of the chosen gates, so a
negative means "in the parent and not in A", not "anywhere outside A": the
all-negative combination of two cytokine gates drawn on CD4 T cells is the
CD4 T cells making neither, not the debris too. Gates already there with the
same meaning (a ``NOT A`` or an ``AND`` over the same operands under the same
parent) are reused, so making the combinations twice adds nothing.
"""
from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence

import numpy as np

# More than this many gates gives more combinations than anyone reads
# (2^6 = 64 already), and the bars stop fitting a page.
MAX_COMBINATION_GATES = 6


def ancestors(gates: Mapping[str, Mapping], gid) -> list:
    """``gid``'s ancestors, nearest first, ending at a root gate. Cycle-safe."""
    out, seen = [], {gid}
    cur = gates.get(gid, {}).get('parent_id')
    while cur is not None and cur in gates and cur not in seen:
        seen.add(cur)
        out.append(cur)
        cur = gates[cur].get('parent_id')
    return out


def common_parent(gates: Mapping[str, Mapping], ids: Sequence):
    """The deepest gate every gate in ``ids`` lies under (not one of them),
    or None when the only thing they share is the whole sample."""
    if not ids:
        return None
    shared = None
    for gid in ids:
        chain = ancestors(gates, gid)
        shared = chain if shared is None else [a for a in shared if a in chain]
    shared = [a for a in (shared or []) if a not in ids]
    return shared[0] if shared else None


def gate_label(gates: Mapping[str, Mapping], gid, transforms=None) -> str:
    """What a combination calls a gate: its name, else what it gates on."""
    from .pipeline import describe_gate
    g = gates[gid]
    name = g.get('label') or g.get('name')
    if name:
        return str(name).strip()
    return describe_gate(g, transforms=transforms).split('  ', 1)[-1].strip()


def sign_patterns(n: int) -> list[tuple[bool, ...]]:
    """Every +/- pattern over ``n`` gates in the order SPICE draws them: most
    positives first, then by the patterns themselves (the first gate varies
    slowest), so all positive comes first and all negative last."""
    pats = list(itertools.product((True, False), repeat=n))
    return sorted(pats, key=lambda p: -sum(p))


def combination_name(labels: Sequence[str], signs: Sequence[bool]) -> str:
    """'A+ B- C+': each gate with its sign. ASCII, so every export keeps it."""
    return ' '.join(f"{lab}{'+' if s else '-'}"
                    for lab, s in zip(labels, signs, strict=True))


def _find_not(gates, gid, parent):
    for k, g in gates.items():
        if (g.get('kind') == 'boolean' and (g.get('op') or '').lower() == 'not'
                and list(g.get('operands') or []) == [gid]
                and g.get('parent_id') == parent):
            return k
    return None


def _find_and(gates, operands, parent):
    want = sorted(operands)
    for k, g in gates.items():
        if (g.get('kind') == 'boolean' and (g.get('op') or 'and').lower() == 'and'
                and sorted(g.get('operands') or []) == want
                and g.get('parent_id') == parent):
            return k
    return None


def check_combinable(gates: Mapping[str, Mapping], ids: Sequence) -> str | None:
    """Why ``ids`` cannot be combined, or None when they can."""
    if len(ids) < 2:
        return "Choose at least two gates."
    if len(ids) > MAX_COMBINATION_GATES:
        return (f"Choose at most {MAX_COMBINATION_GATES} gates "
                f"({2 ** MAX_COMBINATION_GATES} combinations).")
    if len(set(ids)) != len(ids):
        return "A gate is chosen twice."
    for gid in ids:
        if gid not in gates:
            return f"Gate {gid} is not in this sample."
    for a in ids:
        for b in ids:
            if a != b and a in ancestors(gates, b):
                return ("A chosen gate lies under another chosen gate; its "
                        "combinations with that gate would be empty.")
    return None


def build_combinations(gates: dict, ids: Sequence, new_id, labels=None):
    """Add the NOT and AND gates for every combination of ``ids`` to
    ``gates`` (one sample's ``{gid: gate}``, changed in place).

    ``new_id(gate)`` stores a gate dict and returns its id -- the editor's
    ``_add_gate``, or a plain counter in tests. ``labels`` overrides the
    gates' own labels (the same per sample). Gates already present with the
    same meaning are reused, never duplicated. Returns
    ``{'parent': pid, 'labels': [...], 'combos': [(signs, gid), ...],
    'added': n}`` with ``combos`` in sign_patterns order."""
    why = check_combinable(gates, ids)
    if why:
        raise ValueError(why)
    parent = common_parent(gates, ids)
    labels = list(labels) if labels else [gate_label(gates, g) for g in ids]
    added = 0
    negs = {}
    for gid, lab in zip(ids, labels, strict=True):
        hit = _find_not(gates, gid, parent)
        if hit is None:
            hit = new_id({'kind': 'boolean', 'op': 'not', 'operands': [gid],
                          'name': f"{lab}-", 'parent_id': parent})
            added += 1
        negs[gid] = hit
    combos = []
    for signs in sign_patterns(len(ids)):
        ops = [g if s else negs[g] for g, s in zip(ids, signs, strict=True)]
        hit = _find_and(gates, ops, parent)
        if hit is None:
            hit = new_id({'kind': 'boolean', 'op': 'and', 'operands': ops,
                          'name': combination_name(labels, signs),
                          'parent_id': parent, 'combination': True})
            added += 1
        combos.append((signs, hit))
    return {'parent': parent, 'labels': labels, 'combos': combos,
            'added': added}


def combination_counts(gates: Mapping[str, Mapping], df, combos, parent):
    """``(n_parent, [n per combination])`` for one sample, the counts the
    gate tree shows for those boolean gates; a combination whose population
    cannot be computed (pipeline.unknown_population_reason) is NaN, as is
    every combination when the parent's population cannot be.

    Each operand's population is computed once (a cache shared across the
    combinations) instead of once per combination it appears in."""
    from .pipeline import cumulative_gate_mask, unknown_population_reason
    cache: dict = {}
    cols = df.columns

    def cum(gid):
        return np.asarray(cumulative_gate_mask(gates, gid, df, cache=cache),
                          dtype=bool)
    if parent is not None and unknown_population_reason(gates, parent,
                                                        columns=cols):
        return float('nan'), [float('nan')] * len(combos)
    base = cum(parent) if parent is not None else np.ones(len(df), dtype=bool)
    out = []
    for _signs, gid in combos:
        g = gates.get(gid)
        if g is None or unknown_population_reason(gates, gid, columns=cols):
            out.append(float('nan'))
            continue
        m = base.copy()
        for op in g.get('operands') or []:
            m &= cum(op)
        out.append(int(m.sum()))
    return int(base.sum()), out


def frequency_rows(per_sample, labels, groups=None):
    """Long-format rows for the frequency table: one per sample and
    combination. ``per_sample`` is ``{sample: (n_parent, combos, counts)}``
    with ``combos`` as build_combinations returns them; ``groups`` maps a
    sample to its group. '% of parent' is NaN when the parent is empty (no
    frequency is not 0%)."""
    rows = []
    for sample, (n_parent, combos, counts) in per_sample.items():
        for (signs, _gid), n in zip(combos, counts, strict=True):
            pct = (100.0 * n / n_parent
                   if n_parent and n_parent == n_parent and n == n
                   else float('nan'))
            rows.append({'Sample': sample,
                         'Group': (groups or {}).get(sample, ''),
                         'Combination': combination_name(labels, signs),
                         'Positives': int(sum(signs)),
                         'Count': n, 'Parent count': n_parent,
                         '% of parent': pct})
    return rows


def degree_fractions(rows, n_gates):
    """``{sample: [fraction with 0, 1, ..., n positives]}`` of the parent,
    for the pies. NaN when a sample's frequencies cannot be computed."""
    out: dict = {}
    for r in rows:
        acc = out.setdefault(r['Sample'], [0.0] * (n_gates + 1))
        acc[r['Positives']] += r['% of parent'] / 100.0
    return out


def group_means(rows):
    """``{group: {combination: mean % of parent}}`` over the samples of each
    group (a sample whose frequency is NaN is left out of the mean)."""
    acc: dict = {}
    for r in rows:
        v = r['% of parent']
        if v != v:
            continue
        acc.setdefault(r['Group'], {}).setdefault(r['Combination'], []).append(v)
    return {g: {c: float(np.mean(v)) for c, v in d.items()}
            for g, d in acc.items()}
