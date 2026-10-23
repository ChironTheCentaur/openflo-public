"""Gating computations lifted out of the GUI — Tk-free, headless-testable.

Step of decomposing ``gui.py``: the selected-gate event-count / %-of-parent
readout is correctness-critical (it's the headline number in cytometry), so it
belongs where it can be unit-tested directly against a DataFrame rather than
only through a constructed editor window.

The cumulative-mask maths stays in :mod:`openflo.pipeline`; this just composes
it into the counts and the human-readable summary.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

from .robust import MAD_TO_SD


def gate_counts(sample_gates: Mapping[str, Mapping], gid: str, df,
                overrides=None) -> tuple[int, int, str]:
    """``(n_gate, n_parent, of_label)`` for a gate:

    - ``n_gate``   — events inside the gate's cumulative mask (gate + ancestors)
    - ``n_parent`` — its parent population's cumulative count, or the total
      event count for a root gate
    - ``of_label`` — ``'parent'`` for a child gate, ``'all'`` for a root gate
    """
    import numpy as np

    from .pipeline import cumulative_gate_mask

    def _n(target):
        return int(np.asarray(
            cumulative_gate_mask(sample_gates, target, df, overrides=overrides),
            dtype=bool).sum())

    n_gate = _n(gid)
    pid = sample_gates[gid].get('parent_id') if gid in sample_gates else None
    if pid and pid in sample_gates:
        return n_gate, _n(pid), 'parent'
    return n_gate, len(df), 'all'


def format_percent(n_gate, n_parent) -> str:
    """A population frequency as every OpenFlo surface shows it: one
    decimal ('45.2%'), and 'NA' when there is no parent to be a percent of
    (an empty parent has no frequency, which is not the same as 0%)."""
    if not n_parent:
        return 'NA'
    return f"{100.0 * n_gate / n_parent:.1f}%"


def format_gate_count(name: str, n_gate: int, n_parent: int,
                      of_label: str) -> str:
    """Status-bar string: ``"CD3+:  n = 12,345   (45.2% of parent)"``."""
    return (f"{name}:  n = {n_gate:,}   "
            f"({format_percent(n_gate, n_parent)} of {of_label})")


def gate_label_text(name, n_gate, n_parent, why=None) -> str:
    """The label printed on a gate in a plot: its name (when it has one) over
    its frequency of the parent, one decimal -- '45.2%', 'Lymphocytes\n93.0%'.
    `why` is set when the population cannot be computed (see
    pipeline.unknown_population_reason): the frequency then reads 'NA',
    never a 0 that was not measured."""
    pct = 'NA' if why else format_percent(n_gate, n_parent)
    name = str(name or '').strip()
    return f"{name}\n{pct}" if name else pct


def gate_channels(gate: Mapping) -> set:
    """The set of FCS channel names a gate dict references (``channel`` for a
    1-D gate, ``x_channel`` / ``y_channel`` for a 2-D one)."""
    chs = set()
    for k in ('channel', 'x_channel', 'y_channel'):
        v = gate.get(k)
        if v:
            chs.add(v)
    return chs


def population_path(gates: Mapping[str, Mapping], gid: str,
                    transforms: Mapping[str, Mapping] | None = None) -> str:
    """Human-readable population path, e.g. ``'Cells/Singlets/CD901b+'``, built
    by walking ``parent_id`` to the root. Cycle-safe.

    An unnamed gate is named by its description, which carries its bounds.
    ``transforms`` (the sample's ``data_transforms``) has those said as
    intensities, as the gate list says them ('T  CD3-A >= 987', not the
    stored logicle '>= 0.453'); give it wherever the path is SHOWN. Without
    it the bounds are stored coordinates, as before. Never an identity: a
    population is matched across samples by ``population_keys``."""
    from .pipeline import describe_gate
    names, seen, cur = [], set(), gid
    while cur and cur in gates and cur not in seen:
        seen.add(cur)
        g = gates[cur]
        names.append(g.get('label') or g.get('name')
                     or describe_gate(g, transforms=transforms) or cur)
        cur = g.get('parent_id')
    return '/'.join(reversed(names)) if names else str(gid)


# ── Population identity across samples ───────────────────────────────────────
#
# Frequencies, Compare-all and differential abundance line populations up
# across samples by their path. An unnamed gate's path component is its
# description, which carries its coordinates ('T  CD3-A >= 0.453'), so a gate
# nudged in one sample, or a marker read on another detector in a second
# panel, split one population into several, each "missing" from every sample
# but one -- and missing was then counted as 0 events. The key below names a
# gate by what it IS rather than where it was drawn: its name when it has
# one, else its kind and the markers (antibody labels, not detectors) it is
# drawn on, with an ordinal among same-identity siblings so two bands on one
# channel stay two populations.

def _identity_text(s) -> str:
    return ' '.join(str(s).split()).casefold()


def _gate_base_identity(gates: Mapping[str, Mapping], gid: str,
                        channel_labels: Mapping[str, str]) -> str:
    g = gates[gid]
    name = g.get('label') or g.get('name')
    if name:
        return 'n:' + _identity_text(name)
    kind = g.get('kind') or '?'

    def marker(ch):
        return _identity_text(channel_labels.get(ch, ch)) if ch else '?'
    if 'channel' in g:
        return f"{kind}:{marker(g.get('channel'))}"
    if kind in ('rect', 'polygon', 'ellipsoid', 'flowjo_ellipse'):
        return f"{kind}:{marker(g.get('x_channel'))}x{marker(g.get('y_channel'))}"
    if kind == 'cluster':
        return f"cluster:{g.get('label_col', '')}={g.get('cluster_id')}"
    if kind == 'category':
        return f"category:{g.get('column', '')}={g.get('value')}"
    if kind == 'boolean':
        ops = ','.join(sorted(
            _gate_base_identity(gates, o, channel_labels)
            for o in (g.get('operands') or []) if o in gates))
        return f"boolean:{(g.get('op') or 'and').lower()}({ops})"
    return kind


def _gate_position(g: Mapping) -> tuple:
    """Where a gate sits, to order siblings that share an identity (two
    unnamed bands on one channel): low to high on each axis."""
    def f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0
    if 'channel' in g:
        return (f(g.get('value', g.get('lo'))), f(g.get('hi')))
    if g.get('vertices'):
        v = g['vertices']
        return (sum(f(p[0]) for p in v) / len(v), sum(f(p[1]) for p in v) / len(v))
    if g.get('mean') is not None:
        return (f(g['mean'][0]), f(g['mean'][1]))
    return (f(g.get('x0')), f(g.get('y0')))


def population_keys(gates: Mapping[str, Mapping],
                    channel_labels: Mapping[str, str] | None = None) -> dict:
    """``{gid: key}`` identifying each population across samples (see the
    note above). Two samples' populations share a key when they sit at the
    same place in the tree with the same name -- or, unnamed, the same kind
    of gate on the same markers in the same sibling rank -- however their
    coordinates or detectors differ."""
    labels = channel_labels or {}
    base = {gid: _gate_base_identity(gates, gid, labels) for gid in gates}
    # Ordinal among siblings that share parent and identity, so they stay
    # distinct; a lone gate gets none, keeping keys stable when a sibling of
    # another identity is added.
    by_slot: dict = {}
    for gid, g in gates.items():
        by_slot.setdefault((g.get('parent_id'), base[gid]), []).append(gid)
    local = {}
    for (_pid, ident), gids in by_slot.items():
        if len(gids) == 1:
            local[gids[0]] = ident
            continue
        ranked = sorted(gids, key=lambda k: (_gate_position(gates[k]), str(k)))
        for i, k in enumerate(ranked, 1):
            local[k] = f"{ident}#{i}"
    keys = {}
    for gid in gates:
        parts, seen, cur = [], set(), gid
        while cur and cur in gates and cur not in seen:
            seen.add(cur)
            parts.append(local[cur])
            cur = gates[cur].get('parent_id')
        keys[gid] = '/'.join(reversed(parts))
    return keys


# Keys that say where a gate sits or how it is shown, not what it selects.
_PLACEMENT_KEYS = frozenset({'id', 'parent_id', 'operands', 'enabled', 'open'})


def _gate_shape(value):
    """Gate content as plain, comparable data: arrays and tuples as lists,
    floats to 12 significant digits (so a JSON round trip of a gate is still
    the same gate)."""
    if isinstance(value, dict):
        return {k: _gate_shape(v) for k, v in value.items()}
    tolist = getattr(value, 'tolist', None)
    if callable(tolist):                   # a numpy array or scalar
        value = tolist()
    if isinstance(value, (list, tuple)):
        return [_gate_shape(v) for v in value]
    if isinstance(value, float):
        return float(f"{value:.12g}")
    return value


def same_gate(a: dict, b: dict) -> bool:
    """True when two gates select the same events from the same parent
    population: same kind, channels, geometry and name, ignoring ids, the
    on/off state and anything private. The caller compares parents and
    boolean operands, which are ids local to each sample."""
    def content(g):
        return _gate_shape({k: v for k, v in g.items()
                            if k not in _PLACEMENT_KEYS
                            and not str(k).startswith('_')})
    return content(a) == content(b)


_IDENTITY_KEYS = ('kind', 'x_channel', 'y_channel', 'channel', 'name',
                  'label', 'quad_set')


def same_gate_identity(a: dict, b: dict) -> bool:
    """True when two gates are the same population drawn differently: same
    kind, axes and name, whatever the geometry. Booleans never qualify (their
    meaning is their operands, compared by the caller)."""
    if a.get('kind') == 'boolean' or b.get('kind') == 'boolean':
        return False
    return all(a.get(k) == b.get(k) for k in _IDENTITY_KEYS)


def remap_operands(gate: dict, id_map: Mapping[str, str], keep=()) -> list:
    """Point a boolean gate's ``operands`` at the ids its gates were given.

    Every path that gives gates fresh ids (copy to samples, template apply,
    move, paste, session restore) remaps ``parent_id`` through an old -> new
    map. Operands are gate ids too, and left alone they name whatever gate
    holds that id where the boolean lands. Measured: a copied AND(CD3+, CD4+)
    combined the target's own 'CD3 bright' and 'CD4 not-dim', 1,233 events
    where 5,021 is right.

    An operand in ``id_map`` takes its new id; one in ``keep`` (still valid
    where the gate lands) stays. Any other operand cannot be resolved. It is
    not dropped, which silently widens an AND or a NOT, and not kept, since
    its id may be another gate's there: the operands are emptied, and
    gate_to_mask fails closed. Returns the unresolved ids. A non-boolean gate
    is left untouched.
    """
    ops = gate.get('operands')
    if gate.get('kind') != 'boolean' or not ops:
        return []
    new, missing = [], []
    for op in ops:
        if op in id_map:
            new.append(id_map[op])
        elif op in keep:
            new.append(op)
        else:
            missing.append(op)
    gate['operands'] = [] if missing else new
    return missing


def booleans_using(gates: Mapping[str, Mapping], ids) -> list:
    """Ids of the boolean gates in ``gates``, other than ``ids`` themselves,
    that combine any gate in ``ids``."""
    ids = set(ids)
    return [gid for gid, g in gates.items()
            if gid not in ids and g.get('kind') == 'boolean'
            and ids.intersection(g.get('operands') or ())]


# Tag on the statistic keys of a channel whose transform was guessed (see
# population_stats); the Statistics window turns it into a heading.
GUESSED_TAG = '[guessed logicle]'

# Which axis a FlowJo-form GeoMean was averaged on (a row's
# '__geomean_axis__'): the stored display transform, or a FlowJo workspace's
# (geomean_axis_text).
OPENFLO_AXIS_TEXT = 'OpenFlo display axis'


def geomean_axis_text(axis):
    """Short name of a FlowJo axis (a pipeline.wsp_stat_axes spec), as the
    GeoMean column names it: 'FlowJo axis: logicle W=1 M=4.42'. T and A are
    named only when not FlowJo's defaults (262144, 0)."""
    def g(v, digits=3):
        return f'{float(v):.{digits}g}'
    kind = (axis or {}).get('kind')
    try:
        if kind == 'logicle':
            parts = ([f"T={g(axis['T'], 7)}"] if float(axis['T']) != 262144
                     else [])
            parts += [f"W={g(axis['W'])}", f"M={g(axis['M'])}"]
            if float(axis['A']):
                parts.append(f"A={g(axis['A'])}")
            body = 'logicle ' + ' '.join(parts)
        elif kind == 'biex':
            body = f"biex width={g(axis['width'])} pos={g(axis['pos'])}"
            if float(axis['neg']):
                body += f" neg={g(axis['neg'])}"
        elif kind == 'log':
            body = f"log {g(axis['decades'])} decades"
        elif kind == 'fasinh':
            body = (f"arcsinh T={g(axis['T'], 7)} M={g(axis['M'])} "
                    f"A={g(axis['A'])}")
        else:
            body = str(kind)
    except (KeyError, TypeError, ValueError):
        body = str(kind)
    return f'FlowJo axis: {body}'


def geomean_axis_note(rows, label):
    """What a GeoMean column of channel `label` says about its axis, over
    the stats `rows` it shows: the FlowJo axis when every row's GeoMean was
    taken on that one, 'axis differs by sample' when rows used different
    axes, and None when all used OpenFlo's display axis (the column keeps
    the name it always had) or the textbook form, which has no axis."""
    used = {r['__geomean_axis__'][label] for r in rows or ()
            if label in (r.get('__geomean_axis__') or {})}
    if not used or used == {OPENFLO_AXIS_TEXT}:
        return None
    if len(used) > 1:
        return 'axis differs by sample'
    return used.pop()


def population_stats(sample_name, df, gates, order, channel_labels, channels,
                     want, stat_chan, select=None, transforms=None,
                     stat_forms=None, stat_axes=None):
    """Statistic rows for ONE sample's populations (pure — no Tk).

    ``want`` is the set of selected stat names; ``stat_chan`` the per-channel
    subset (Median/Mean/GeoMean/CV/rCV). Counts are computed over the FULL
    gate tree (so
    %Parent stays correct) even when ``select`` restricts which gates emit
    rows. Each row carries a hidden ``__gid__``. Empty populations yield NaN
    per-channel and 0 counts. A population that cannot be computed
    (pipeline.unknown_population_reason: it rests on a gate whose region is
    unknown, which selects nothing) is missing, not empty: every value is
    NaN and the row's hidden ``__unknown__`` says why.

    ``transforms`` maps a channel to the transform_spec baked into ``df``
    (a FlowSample's ``data_transforms``). Gates are evaluated on ``df`` as
    stored, because gate coordinates live in that display space, but the
    per-channel statistics are computed on the compensated LINEAR values, as
    FlowJo reports MFI, GeoMean and CV. On logicle values a lognormal(7, 0.5)
    channel measured GeoMean 0.46 and CV 11 % where its linear values give
    1,094 and 53 %, and only the median can be recovered from such a number
    afterwards.

    ``stat_forms`` picks the GeoMean and rCV forms; None reads them from
    Preferences (``prefs.load_stat_forms``), whose defaults are FlowJo's.

    ``stat_axes`` maps a channel to the FlowJo axis a workspace declared for
    it (a FlowSample's ``stat_axes``, see pipeline.wsp_stat_axes). FlowJo's
    GeoMean is a mean on the display axis, so it depends on that axis: on
    such a channel it is taken on the workspace's axis, elsewhere on the
    stored display transform. Each row's hidden ``__geomean_axis__`` says,
    per channel label, which axis its GeoMean used (geomean_axis_text).
    """
    import numpy as np

    from .parallel import cpu_threads
    from .pipeline import (
        cumulative_gate_mask,
        is_guessed_spec,
        is_unknown_spec,
        to_linear,
        unknown_population_reason,
    )
    from .prefs import load_stat_forms
    forms = stat_forms if stat_forms is not None else load_stat_forms()
    total = len(df)
    order = order or list(gates)
    counts, masks = {}, {}
    # One cache for this whole collection pass: `gates`/`df` are fixed here, so
    # a child reuses its parent's cumulative mask instead of re-walking (and
    # re-evaluating) the entire ancestor chain. Masks are read-only below —
    # only boolean-indexed and summed.
    mask_cache = {}
    for gid in order:
        if gid not in gates:
            continue
        m = cumulative_gate_mask(gates, gid, df, cache=mask_cache)
        masks[gid] = m
        counts[gid] = int(np.asarray(m).sum())

    emit = select if select is not None else order
    keys = population_keys(gates, channel_labels)
    rows = []
    row_gids = []
    for gid in emit:
        if gid not in gates or gid not in counts:
            continue
        g = gates[gid]
        cnt = counts[gid]
        parent = g.get('parent_id')
        parent_cnt = counts.get(parent, total) if parent else total
        row = {'Sample': sample_name,
               # Displayed and exported: an unnamed gate's bounds as the
               # intensities the gate list shows, not stored coordinates.
               'Population': population_path(gates, gid,
                                             transforms=transforms),
               '__gid__': gid,
               # Identity across samples (population_keys); the path above is
               # for display and changes with a gate's coordinates.
               '__key__': keys.get(gid, str(gid))}
        # Its gates select nothing, but that 0 was never measured: a
        # missing population, as one a sample does not have.
        why = unknown_population_reason(gates, gid, columns=df.columns)
        if why:
            row['__unknown__'] = why
        if 'Count' in want:
            row['Count'] = float('nan') if why else cnt
        if '%Parent' in want:
            # An empty parent has no fraction to give: NaN, not a 0 % that
            # reads as "measured, none of them".
            row['%Parent'] = ((cnt / parent_cnt * 100.0)
                              if parent_cnt and not why else float('nan'))
        if '%Grandparent' in want:
            # Percent of the parent's parent. A gate drawn on all events has
            # no grandparent (its parent IS all events): blank, not a
            # number. One drawn on such a gate has all events as its
            # grandparent.
            gp_cnt = float('nan')
            if parent and parent in gates:
                gp = gates[parent].get('parent_id')
                gp_cnt = counts.get(gp, total) if gp else total
            row['%Grandparent'] = ((cnt / gp_cnt * 100.0)
                                   if gp_cnt and gp_cnt == gp_cnt
                                   and not why else float('nan'))
        if '%Total' in want:
            row['%Total'] = ((cnt / total * 100.0) if total and not why
                             else float('nan'))
        rows.append(row)
        row_gids.append(gid)

    pcts = percentile_names(want)
    ratio = MFI_RATIO in want
    need_chan = (want & set(stat_chan)) | set(pcts)
    if ratio:
        # The ratio is taken later, across samples (add_mfi_ratio), from
        # each row's medians; the Median column itself shows only if asked.
        need_chan |= {'Median'}
    if not (need_chan and channels):
        return rows
    calc = want | need_chan
    # Channel by channel, so only ONE linearised column is alive at a time
    # (holding them all would be 8 MB per channel per million events), and
    # each column is inverted once rather than once per population. Every
    # row still receives its channels in `channels` order, so the column
    # order the callers derive from first-seen keys is unchanged.
    #
    # Within a channel the populations are independent, so a large table
    # spreads them over threads (the sorts and sums inside release the GIL).
    # Each population's rows are found once for all channels, and only the
    # statistics asked for are computed. The numbers are the same either
    # way: the same values reach the same numpy calls.
    transforms = transforms or {}
    stat_axes = stat_axes or {}
    from .pipeline import _edge_axis_fns
    rows_of = {gid: np.flatnonzero(np.asarray(masks[gid], dtype=bool))
               for gid in row_gids if counts[gid]}
    work = sum(counts[gid] for gid in row_gids) * len(channels)
    n_threads = (min(cpu_threads(), len(row_gids))
                 if work >= _PARALLEL_MIN_VALUES else 1)
    pool = None
    if n_threads > 1:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=n_threads)
    try:
        for ch in channels:
            if ch not in df.columns:
                continue
            lbl = channel_labels.get(ch, ch)
            # A column whose display transform is unknown (an old OpenFlo
            # CSV) has no linear values: its statistics are left out (NaN,
            # to_linear) and every row names it, so the window can say why
            # they are blank.
            unknown = is_unknown_spec(transforms.get(ch))
            # A column whose transform was GUESSED (no record, values in the
            # logicle range) gets its own keys, so its numbers never sit
            # under the certain '(linear)' heading: right for a
            # default-logicle old session, ~20x off for a dim channel the old
            # editor had on asinh.
            if is_guessed_spec(transforms.get(ch)):
                lbl = f'{lbl} {GUESSED_TAG}'
            spec = transforms.get(ch)
            lin = (to_linear(df[ch].values, spec) if rows_of else None)
            # The FlowJo form on the axis a FlowJo workspace declared for
            # this channel, when there is one (geo_axis); else on the stored
            # display transform, as before.
            geo_axis = geo_note = None
            if 'GeoMean' in want and forms.get('geomean') != 'log':
                ax = stat_axes.get(ch)
                geo_axis = _edge_axis_fns(ax) if ax else None
                if geo_axis is not None:
                    geo_note = geomean_axis_text(ax)
                elif spec and spec.get('method', 'linear') != 'linear':
                    geo_note = OPENFLO_AXIS_TEXT
            display_geo = bool('GeoMean' in want and spec and geo_axis is None
                               and forms.get('geomean') != 'log'
                               and spec.get('method', 'linear') != 'linear')
            stored = (np.asarray(df[ch].values, dtype=float)
                      if display_geo and rows_of else None)

            def one(gid, lin=lin, stored=stored, spec=spec, geo_axis=geo_axis):
                idx = rows_of.get(gid)
                if idx is None or lin is None:
                    return _NAN_STATS
                return _channel_stats(lin[idx], stored, idx, spec, calc,
                                      to_linear, pcts, forms,
                                      geo_axis=geo_axis)

            results = (list(pool.map(one, row_gids)) if pool is not None
                       else [one(gid) for gid in row_gids])
            for row, res in zip(rows, results, strict=True):
                if unknown:
                    row.setdefault('__scale_unknown__', []).append(lbl)
                if geo_note:
                    row.setdefault('__geomean_axis__', {})[lbl] = geo_note
                for stat in CHANNEL_STAT_ORDER:
                    if stat in want and stat in stat_chan:
                        row[f'{stat} {lbl}'] = res.get(stat, _NAN)
                for name, _q in pcts:
                    row[f'{name} {lbl}'] = res.get(name, _NAN)
                if ratio:
                    row.setdefault('__median__', {})[lbl] = res.get(
                        'Median', _NAN)
    finally:
        if pool is not None:
            pool.shutdown()
    return rows


_NAN = float('nan')
_NAN_STATS: dict = {}            # every statistic missing (read as NaN)

# Per-channel statistics in the order their columns appear. SD and rSD sit
# next to the CVs they divide into the mean or median.
CHANNEL_STAT_ORDER = ('Median', 'Mean', 'GeoMean', 'SD', 'rSD', 'CV', 'rCV',
                      'MAD')

# A percentile is asked for by name, 'P<q>' (P5, P95, P2.5): one column per
# channel each, so a window can offer any set without a fixed list here.
_PCT_RE = re.compile(r'^P(\d+(?:\.\d+)?)$')

# The MFI ratio: a population's median over the same population's median in
# a reference sample (an unstained or FMO control), per channel.
MFI_RATIO = 'MFI ratio'


def percentile_names(want):
    """The ``P<q>`` percentile names in ``want`` as ``[(name, q)]``, by q.
    A q outside 0..100 is not a percentile and is ignored."""
    out = []
    for w in want:
        m = _PCT_RE.match(str(w))
        if m and 0.0 <= float(m.group(1)) <= 100.0:
            out.append((str(w), float(m.group(1))))
    return sorted(out, key=lambda t: t[1])


def parse_percentiles(text):
    """``'5, 25, 75, 95'`` -> ``['P5', 'P25', 'P75', 'P95']``. Raises
    ValueError naming the first entry that is not a number in 0..100."""
    names = []
    for part in re.split(r'[,;\s]+', str(text).strip()):
        if not part:
            continue
        try:
            q = float(part.rstrip('%'))
        except ValueError:
            raise ValueError(f"'{part}' is not a number") from None
        if not 0.0 <= q <= 100.0:
            raise ValueError(f"{part} is outside 0 to 100")
        name = f'P{q:g}'
        if name not in names:
            names.append(name)
    return names


def is_channel_stat_column(col, stat_chan):
    """Whether table column ``col`` is a per-channel statistic computed on
    the compensated linear values (one of ``stat_chan`` or a percentile),
    as opposed to a count, a frequency or an MFI ratio."""
    head = str(col).split(' ', 1)[0]
    return head in stat_chan or bool(_PCT_RE.match(head))


def add_mfi_ratio(rows, reference, reference_rows=None):
    """Give each row an ``MFI ratio <channel>`` column: its median over the
    median of the same population (matched by ``__key__``) in sample
    ``reference``. The reference's own rows are taken from ``rows`` unless
    ``reference_rows`` is given (a curated table may leave them out).

    A ratio needs both medians and a positive reference: otherwise it is NaN
    (blank). A population the reference lacks has no ratio, never 0 or 1.
    Rows must come from population_stats with MFI_RATIO wanted, which keeps
    their medians in the hidden ``__median__``."""
    import math
    ref_rows = reference_rows if reference_rows is not None else [
        r for r in rows if r.get('Sample') == reference]
    ref = {r.get('__key__'): r.get('__median__', {}) for r in ref_rows
           if r.get('Sample') == reference and not r.get('__unknown__')}
    for r in rows:
        meds = r.get('__median__', {})
        base = ref.get(r.get('__key__'))
        for lbl, med in meds.items():
            den = base.get(lbl, _NAN) if base is not None else _NAN
            ok = (med == med and den == den and den > 0
                  and not math.isinf(den))
            r[f'{MFI_RATIO} {lbl}'] = (med / den) if ok else _NAN
    return rows

# Below this many (event x channel) values, thread start-up costs more than
# it saves; the answer is the same either way.
_PARALLEL_MIN_VALUES = 2_000_000


def _channel_stats(vals, stored, idx, spec, want, to_linear, pcts=(),
                   forms=None, geo_axis=None):
    """``{statistic: value}`` of one population on one channel. ``vals`` are
    its linear values, ``stored`` the whole channel as stored (needed only
    for a display-space GeoMean) and ``idx`` the population's rows. ``pcts``
    are the ``(name, q)`` percentiles wanted, ``forms`` the GeoMean and rCV
    forms (``prefs.load_stat_forms``; FlowJo's when None). ``geo_axis`` is
    ``(forward, inverse)`` of the axis a FlowJo workspace declared for the
    channel: FlowJo's GeoMean is then taken there instead of on the stored
    display transform. Statistics not in ``want`` are left out (read as
    NaN) rather than computed."""
    import numpy as np

    forms = forms or {}
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return _NAN_STATS
    med = mean = cv = gmean = rcv = sd = rsd = mad = _NAN
    if want & {'Median', 'rCV', 'MAD'}:
        med = float(np.median(vals))
    if want & {'Mean', 'CV', 'SD'}:
        mean = float(np.mean(vals))
        # Sample SD (divides by n - 1), the standard estimate, and
        # CV = SD / Mean. The n form read low on small gates (about 5% at
        # 10 events). One event has no SD, so its SD and CV read NA.
        if vals.size > 1:
            sd = float(np.std(vals, ddof=1))
            cv = (sd / mean * 100.0) if mean else _NAN
    if 'GeoMean' in want:
        # Geometric mean, as FlowJo defines it: the mean in the channel's
        # display (transform) space, back-transformed. That is what makes it
        # defined for the negative values compensation produces. The classic
        # exp(mean(log x)) had to drop them, which biased a dim population
        # upward (measured: median 151, GeoMean 211.7 from 69% of its events,
        # where FlowJo's reads 83.5). A channel stored linear has no display
        # space here, so it keeps the classic form over its positive values.
        #
        # The mean depends on the axis: on public data, OpenFlo's default
        # logicle put a dim population's GeoMean 30-65% below the one on the
        # workspace's own axis (W=1, M=4.42). So a channel a FlowJo
        # workspace declared an axis for averages on THAT axis (geo_axis).
        if forms.get('geomean') != 'log' and geo_axis is not None:
            fwd, inv = geo_axis
            gmean = float(inv(np.array([np.mean(fwd(vals))]))[0])
        elif (forms.get('geomean') != 'log' and spec
                and spec.get('method', 'linear') != 'linear'):
            st = stored[idx]
            st = st[np.isfinite(st)]
            gmean = (float(to_linear(np.array([np.mean(st)]), spec)[0])
                     if st.size else _NAN)
        else:
            pos = vals[vals > 0]
            gmean = (float(np.exp(np.mean(np.log(pos))))
                     if pos.size else _NAN)
    if want & {'rCV', 'rSD'}:
        # Robust SD and robust CV = 100 * rSD / median. By default FlowJo's
        # rSD: 1/2 (P84.13 - P15.87), the half-width that holds one SD
        # either side of the mean for a normal population. The 'mad' form
        # is the robust-statistics one, MAD_TO_SD * MAD: it tolerates up to
        # half the events being outliers, but on a skewed population it
        # differs from FlowJo's (log-normal sigma 0.8: 74% where FlowJo
        # reports 89%).
        if forms.get('rcv') == 'mad':
            if med != med:
                med = float(np.median(vals))
            rsd = MAD_TO_SD * float(np.median(np.abs(vals - med)))
        else:
            p_lo, p_hi = np.percentile(vals, [15.87, 84.13])
            rsd = 0.5 * float(p_hi - p_lo)
        if 'rCV' in want:
            rcv = (100.0 * rsd / med if med else _NAN)
    if 'MAD' in want:
        # Median absolute deviation from the median, unscaled (multiply by
        # 1.4826 for an SD estimate under normality).
        mad = float(np.median(np.abs(vals - med)))
    out = {'Median': med, 'Mean': mean, 'GeoMean': gmean, 'SD': sd,
           'rSD': rsd, 'CV': cv, 'rCV': rcv, 'MAD': mad}
    if pcts:
        qs = np.percentile(vals, [q for _n, q in pcts])
        out.update({n: float(v) for (n, _q), v in zip(pcts, qs, strict=True)})
    return out
