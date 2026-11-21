"""Run templates: a Pipeline Workspace saved without its sample files, so a
repeating experiment can be run on each new day's files with the same
groups, gates, compensation, FMO sets and run settings.

A template holds *slots* instead of samples. Each slot is one workspace
group (or the ungrouped rows) with a filename pattern and the populations
(gate chains) its rows ran on. Starting a run from a template matches the
day's FCS files to slots (:func:`match_files`), the user checks and fixes
the match in a preview, and :func:`instantiate` builds a fresh
:class:`~openflo.workspace_run.WorkspaceModel` with one row per file and
population, each reloaded from its path at Run.

Core module: no Tk. The panel (``workspace.py``) owns the dialogs.
"""
from __future__ import annotations

import copy
import fnmatch
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from .cluster_labels import GATE_PALETTE
from .paths import fcs_stem
from .workspace_run import (
    WorkspaceModel,
    default_run_cfg,
    derive_trial_name,
    item_from_json,
    item_to_json,
    natural_sort_key,
)

TEMPLATE_FORMAT = 'openflo-run-template'
TEMPLATE_SCHEMA = 1                # bump when the template dict changes shape
UNGROUPED = 'Ungrouped'

# What a population row keeps from a workspace item: everything but which
# sample it was (sample, trial, path, colour) and the per-row run flags the
# instantiated row gets afresh.
_POPULATION_KEYS = ('gate_id', 'gate_path', 'is_leaf', 'gate_chain', 'gates',
                    'qc_applied',
                    'channel_labels', 'comp_override', 'fmo', 'display',
                    'transforms')


def is_template_dict(d: Any) -> bool:
    """Whether ``d`` (a loaded JSON value) is a saved run template."""
    return isinstance(d, dict) and d.get('format') == TEMPLATE_FORMAT


# ── Patterns ───────────────────────────────────────────────────────────────

def name_tokens(name: str) -> list[str]:
    """The words of a file or sample name, lower case, in order: runs of
    letters and digits ('Ctrl_2h-B.fcs' -> ['ctrl', '2h', 'b', 'fcs'])."""
    return [t for t in re.split(r'[^0-9a-z]+', str(name).casefold()) if t]


def _stem(path_or_name: str) -> str:
    return fcs_stem(path_or_name)


def suggest_pattern(slot_names: Sequence[str],
                    all_names: Sequence[str]) -> str:
    """A filename pattern for a slot whose rows came from samples
    `slot_names`, among all the workspace's samples `all_names`: the words
    every one of the slot's names shares and not every sample shares (a
    day or date word every sample carries would stop the template matching
    the next day's files). Empty (match any file) when there are none."""
    if not slot_names:
        return ''
    toks = [name_tokens(_stem(n)) for n in slot_names]
    common = set(toks[0]).intersection(*map(set, toks[1:]))
    everywhere = set(name_tokens(_stem(all_names[0]))).intersection(
        *(set(name_tokens(_stem(n))) for n in all_names[1:])) if all_names else set()
    keep = common - everywhere
    return ' '.join(t for t in dict.fromkeys(toks[0]) if t in keep)


def _is_glob(pattern: str) -> bool:
    return any(c in pattern for c in '*?[')


def pattern_matches(pattern: str, path: str) -> bool:
    """Whether file `path` fits a slot's `pattern`. A pattern with ``*``,
    ``?`` or ``[`` is a filename glob (case ignored); otherwise every word
    of it must be a word of the file name. An empty pattern fits any
    file."""
    pattern = (pattern or '').strip()
    if not pattern:
        return True
    base = os.path.basename(str(path))
    if _is_glob(pattern):
        return (fnmatch.fnmatchcase(base.casefold(), pattern.casefold())
                or fnmatch.fnmatchcase(_stem(base).casefold(), pattern.casefold()))
    words = set(name_tokens(_stem(base)))
    return all(t in words for t in name_tokens(pattern))


def _specificity(pattern: str) -> int:
    """How much a pattern says: its word count, a glob's counted from its
    literal parts (``*_ctrl_*`` says one word, as ``ctrl`` does). A file
    fitting two slots goes to the more specific one; an empty pattern is
    the catch-all."""
    return len(name_tokens(re.sub(r'\[[^\]]*\]|[*?]', ' ', pattern or '')))


# ── Template from a workspace ──────────────────────────────────────────────

def _population_key(it: Mapping[str, Any]) -> tuple:
    """Rows of one group that ran the same population on different samples
    share this: the gate path as the tree shows it (None for a whole
    sample), with the gate id as the tie-break for two unnamed gates."""
    if not it.get('gate_id'):
        return (None,)
    return (it.get('gate_path') or '', tuple(
        (g or {}).get('name') or (g or {}).get('label') or gid
        for gid, g in ((gid, (it.get('gates') or {}).get(gid))
                       for gid in it.get('gate_chain') or [])))


# Takes a workspace row (a dict); Any keeps a dict-typed reader assignable.
TransformsOf = Callable[[Any], 'Mapping[str, Any] | None']



def _population(it: Mapping[str, Any], transforms_of: TransformsOf | None
                ) -> dict:
    pop = {k: it.get(k) for k in _POPULATION_KEYS if k in it}
    rec = transforms_of(it) if transforms_of is not None else None
    if rec is not None:
        pop['transforms'] = rec
    return pop


def _slot(name: str, grouped: bool, items: Sequence[Mapping[str, Any]],
          all_names: Sequence[str], group: Mapping[str, Any] | None,
          transforms_of: TransformsOf | None = None) -> dict:
    pops: dict[tuple, dict] = {}
    for it in items:
        key = _population_key(it)
        if key not in pops:
            pops[key] = _population(it, transforms_of)
    names = list(dict.fromkeys(str(it.get('sample') or '') for it in items))
    g = group or {}
    return {'name': name, 'grouped': grouped,
            'pattern': suggest_pattern(names, all_names),
            'display': g.get('display', True), 'comp': g.get('comp'),
            'fmo': g.get('fmo'), 'cfg': dict(g.get('cfg') or {}),
            'populations': list(pops.values())}


def template_from_model(model: WorkspaceModel,
                        transforms_of: TransformsOf | None = None) -> dict:
    """The run template of a workspace: one slot per group (and one for the
    ungrouped rows, when there are any), each with a suggested filename
    pattern and the distinct populations its rows ran. Each population
    keeps the scales its gates were drawn on (``transforms``, a
    ``data_transforms`` record: no entry = linear) as `transforms_of` gives
    them for its row, else as the row already carries them. Raises
    ValueError for a workspace with no rows."""
    all_names = list(dict.fromkeys(str(it.get('sample') or '')
                                   for _m, it, _g in model.all_items()))
    if not all_names:
        raise ValueError('the workspace has no rows to make a template from')
    slots = [_slot(g['name'], True, list(g['items'].values()), all_names, g,
                   transforms_of)
             for g in model.groups.values() if g['items']]
    if model.loose:
        slots.append(_slot(UNGROUPED, False, list(model.loose.values()),
                           all_names, None, transforms_of))
    return {'format': TEMPLATE_FORMAT, 'schema': TEMPLATE_SCHEMA,
            'title': model.title, 'run_cfg': dict(model.run_cfg),
            'slots': slots}


def slots_without_scales(template: Mapping[str, Any]) -> list[str]:
    """The slots holding a population with no ``transforms`` record: a
    run could not put a new file on the scales its gates were drawn on."""
    return [s['name'] for s in template.get('slots', [])
            if any(not isinstance(p.get('transforms'), dict)
                   for p in s.get('populations', []))]


def template_to_json(template: Mapping[str, Any]) -> dict:
    """``template`` with every population's comp matrix as nested lists, so
    the whole thing is JSON-serialisable (``app_version`` stamped)."""
    from .update import current_version
    out = dict(template)
    out['app_version'] = current_version()
    out['slots'] = [dict(s, populations=[item_to_json(p)
                                          for p in s.get('populations', [])])
                    for s in template.get('slots', [])]
    return out


def template_from_json(d: Mapping[str, Any]) -> dict:
    """Inverse of :func:`template_to_json`. Raises ValueError when `d` is
    not a run template or has no slots."""
    if not is_template_dict(d):
        raise ValueError('not an OpenFlo run template')
    slots = []
    for s in d.get('slots') or []:
        if not isinstance(s, dict) or not s.get('populations'):
            continue
        pops = [item_from_json(p) for p in s['populations']]
        slots.append(dict(s, populations=pops,
                          name=str(s.get('name') or f'Slot {len(slots) + 1}'),
                          pattern=str(s.get('pattern') or '')))
    if not slots:
        raise ValueError('the run template has no slots')
    run_cfg = default_run_cfg()
    if isinstance(d.get('run_cfg'), dict):
        run_cfg.update(d['run_cfg'])
    return dict(d, slots=slots, run_cfg=run_cfg)


# ── Matching the day's files to slots ──────────────────────────────────────

def match_files(template: Mapping[str, Any], paths: Iterable[str]) -> dict:
    """Which slot each of `paths` goes to, by the slots' patterns.

    Returns ``{'assign': {path: slot index}, 'unmatched': [path],
    'ambiguous': {path: [slot index]}, 'empty': [slot index]}``. A file
    fitting several slots goes to the most specific pattern; a tie between
    the most specific is ambiguous and is NOT assigned (the preview asks).
    A slot no file went to is in ``empty``."""
    slots = list(template.get('slots') or [])
    assign: dict[str, int] = {}
    unmatched: list[str] = []
    ambiguous: dict[str, list[int]] = {}
    for p in paths:
        fits = [i for i, s in enumerate(slots)
                if pattern_matches(s.get('pattern', ''), p)]
        if not fits:
            unmatched.append(p)
            continue
        best = max(_specificity(slots[i].get('pattern', '')) for i in fits)
        top = [i for i in fits
               if _specificity(slots[i].get('pattern', '')) == best]
        if len(top) == 1:
            assign[p] = top[0]
        else:
            ambiguous[p] = top
    used = set(assign.values())
    return {'assign': assign, 'unmatched': unmatched, 'ambiguous': ambiguous,
            'empty': [i for i in range(len(slots)) if i not in used]}


def file_channels(path: str) -> list[str]:
    """The channel (column) names of FCS file `path`, reading only its TEXT
    segment, named as the loader names them. Raises as the loader does."""
    import functools

    from . import fcs_read
    from .fcs_quirks import read_fcs_tolerant
    from .flow_sample import FlowSample
    fcs = read_fcs_tolerant(path, functools.partial(fcs_read.FlowData,
                                                    only_text=True), [])
    return [n for n, _label in FlowSample.fcs_channel_names(fcs)[0]]


def has_spillover(path: str) -> bool:
    """Whether FCS file `path` carries its own $SPILLOVER (SPILL), the
    matrix a template row is compensated with. Raises when the file
    cannot be read."""
    from .compensation_io import read_compensation_matrix
    return read_compensation_matrix(str(path))[1] is not None


def missing_channels(template: Mapping[str, Any], slot_index: int,
                     columns: Iterable[str]) -> list[str]:
    """The channels slot `slot_index`'s gates read that a file with
    `columns` lacks, sorted. Non-empty means that file must not run on this
    slot: its gates would select nothing."""
    from .gating import gate_channels
    cols = set(columns)
    need: set[str] = set()
    for pop in template['slots'][slot_index].get('populations', []):
        for g in (pop.get('gates') or {}).values():
            need |= set(gate_channels(g))
    return sorted(need - cols, key=natural_sort_key)


# ── Building the day's workspace ───────────────────────────────────────────

def _unique_names(paths: Sequence[str]) -> dict[str, str]:
    """A sample name per path: the file name without .fcs, with ' (2)',
    ' (3)' ... for a name two files share (same name, other folders)."""
    out: dict[str, str] = {}
    seen: dict[str, int] = {}
    for p in paths:
        base = _stem(p)
        n = seen.get(base.casefold(), 0) + 1
        seen[base.casefold()] = n
        out[p] = base if n == 1 else f'{base} ({n})'
    return out


def instantiate(template: Mapping[str, Any], assignment: Mapping[str, int],
                *, trial_of: Callable[[str], str] = derive_trial_name
                ) -> WorkspaceModel:
    """A new workspace for the files in `assignment` (path -> slot index):
    each slot becomes a group (the ungrouped slot, loose rows) with the
    template's compensation, FMO set and run settings, holding one row per
    file and population. Rows carry the file's path and no live sample, so
    Run reloads each file and puts the template's gates and compensation on
    it. A file assigned to a slot index that does not exist is skipped."""
    slots = list(template.get('slots') or [])
    m = WorkspaceModel(str(template.get('title') or 'Pipeline'))
    m.run_cfg.update(dict(template.get('run_cfg') or {}))
    ordered = sorted(assignment, key=lambda p: natural_sort_key(_stem(p)))
    names = _unique_names(ordered)
    # One colour per file, as each loaded sample gets its own.
    colours = {p: GATE_PALETTE[i % len(GATE_PALETTE)]
               for i, p in enumerate(ordered)}
    for i, slot in enumerate(slots):
        files = [p for p in ordered if assignment[p] == i]
        if not files:
            continue
        gid = None
        if slot.get('grouped', True):
            gid = m.new_group(slot['name'])
            g = m.groups[gid]
            g['display'] = slot.get('display', True)
            g['comp'] = slot.get('comp')
            g['fmo'] = slot.get('fmo')
            g['cfg'] = dict(slot.get('cfg') or {})
        for path in files:
            for pop in slot.get('populations', []):
                # Copies: a row's gates or matrix edited in place leave the
                # template and the other rows alone.
                payload = copy.deepcopy({k: v for k, v in pop.items()
                                         if k in _POPULATION_KEYS})
                payload.update(sample=names[path], trial=trial_of(path),
                               path=os.path.abspath(path),
                               color=colours[path],
                               template_slot=slot['name'])
                m.add_item(payload, gid)
    return m


# ── The preview's rows (Tk-free; the dialog shows and edits them) ──────────

SKIP = -1      # a row's slot when the user leaves that file out of the run


def preview_rows(template: Mapping[str, Any], paths: Sequence[str],
                 channels_of: Callable[[str], list[str]] = file_channels,
                 spill_of: Callable[[str], bool] = has_spillover
                 ) -> list[dict]:
    """One row per file for the preview: ``{'path', 'slot', 'fits',
    'columns', 'spill', 'error'}``. ``slot`` is the matched slot index, or
    None when the file fits no slot or several (``fits`` lists those).
    ``columns`` is the file's channels and ``spill`` whether it carries its
    own $SPILLOVER (None with ``error`` when it cannot be read)."""
    match = match_files(template, paths)
    rows = []
    for p in paths:
        try:
            cols, spill, err = list(channels_of(p)), bool(spill_of(p)), None
        except Exception as exc:  # noqa: BLE001 - any unreadable file is shown on its row (and must be skipped), never raised out of the preview
            cols, spill, err = None, None, f'{type(exc).__name__}: {exc}'
        slot = match['assign'].get(p)
        fits = match['ambiguous'].get(p) or ([slot] if slot is not None else [])
        rows.append({'path': p, 'slot': slot, 'fits': fits,
                     'columns': cols, 'spill': spill, 'error': err})
    return rows


def row_problem(template: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    """What stops `row` running as set, or '' when it is ready (or
    skipped). Said so the user knows what to do about it."""
    slots = template['slots']
    if row['slot'] == SKIP:
        return ''
    if row.get('error'):
        return f"cannot be read ({row['error']}); skip it"
    if row['slot'] is None:
        if len(row['fits']) > 1:
            names = ' / '.join(slots[i]['name'] for i in row['fits'])
            return f"fits {names}; choose one"
        return 'fits no slot; choose one or skip it'
    miss = missing_channels(template, row['slot'], row['columns'] or [])
    if miss:
        more = f" and {len(miss) - 3} more" if len(miss) > 3 else ''
        return (f"has no {', '.join(miss[:3])}{more}, which "
                f"{slots[row['slot']]['name']} gates on; "
                "choose another slot or skip it")
    return ''


def _comp_word(comp: Mapping[str, Any], where: str) -> str:
    return (f"the {where}'s comp beads" if comp.get('kind') == 'beads'
            else f"the {where}'s matrix")


def row_comp_words(template: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    """How the file of `row` is compensated, for the preview, in the order
    a run takes it (``resolve_comp``): a population's own beads or matrix,
    else its slot's, else the file's own $SPILLOVER (a template carries no
    day's matrix), or none when the file has none. Populations of one slot
    compensated differently are each named."""
    if row['slot'] in (None, SKIP) or row.get('error'):
        return ''
    slot = template['slots'][row['slot']]
    own = ("the file's own $SPILLOVER" if row.get('spill')
           else 'none: the file has no $SPILLOVER')
    words = [_comp_word(p['comp_override'], 'row') if p.get('comp_override')
             else _comp_word(slot['comp'], 'slot') if slot.get('comp')
             else own
             for p in slot.get('populations', [])]
    kinds = list(dict.fromkeys(words))      # a slot has populations
    return kinds[0] if len(kinds) == 1 else 'by population: ' + '; '.join(kinds)


def row_status(template: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    """The preview's Status column for `row`. A file that only the
    catch-all slot (no pattern) took says so: it may be a stray file."""
    if row['slot'] == SKIP:
        return 'skipped'
    problem = row_problem(template, row)
    if problem:
        return problem
    notes = []
    if not (template['slots'][row['slot']].get('pattern') or '').strip():
        notes.append(f"only {template['slots'][row['slot']]['name']}, the "
                     "slot with no pattern, takes it; skip it if it is not "
                     "part of this run")
    if 'the file has no $SPILLOVER' in row_comp_words(template, row):
        notes.append('it has no $SPILLOVER, so it runs uncompensated')
    return f"ready ({'; '.join(notes)})" if notes else 'ready'


def preview_assignment(template: Mapping[str, Any],
                       rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """``{path: slot index}`` for the rows ready to run. Raises ValueError,
    naming the first file in the way, while any row still has a problem or
    when every file is skipped: a run never starts on a guess."""
    for r in rows:
        why = row_problem(template, r)
        if why:
            raise ValueError(f"{os.path.basename(r['path'])} {why}")
    out = {r['path']: r['slot'] for r in rows if r['slot'] not in (None, SKIP)}
    if not out:
        raise ValueError('no file is set to run')
    return out
