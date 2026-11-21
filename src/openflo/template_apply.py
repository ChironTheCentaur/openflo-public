"""Putting a gating template's gates (a JSON template, a FlowJo .wsp or a
Gating-ML file) onto a sample, and what is said about it: the per-sample
steps of the desktop's "Apply template" (``editor_template``) and of the
phone's Gating-ML import (``document.Document.apply_gatingml``).

Core module: no Tk. Both apps take the steps and the words from here, so
one template puts the same gates on the same sample and says the same
thing about it.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

Gate = dict[str, Any]


def template_source(path: str) -> str:
    """'<kind> (<file name>)' for the status line."""
    low = path.lower()
    kind_label = ('FlowJo .wsp' if low.endswith('.wsp') else
                  'Gating-ML' if low.endswith('.xml') else 'JSON template')
    return f'{kind_label} ({os.path.basename(path)})'


def retarget_template(gate_dicts: Iterable[Mapping[str, Any]], sample: Any,
                      legacy_label: bool = True) -> list[Gate]:
    """Copies of a template's ``gate_dicts`` retargeted to `sample`'s
    detectors by antibody label (label_alignment.relabel_gate_for_sample):
    a gate on CD901b reads wherever CD901b sits in this sample.
    ``legacy_label`` is False for .wsp and Gating-ML gates, whose ``label``
    names the population. `sample` None retargets nothing."""
    from .label_alignment import relabel_gate_for_sample, sample_fluor_labels
    label_to_det = sample_fluor_labels(sample) if sample is not None else {}
    return [relabel_gate_for_sample(g, label_to_det, legacy_label=legacy_label)
            for g in gate_dicts]


def note_uncompensated(name: str, sample: Any,
                       gate_dicts: Iterable[Mapping[str, Any]],
                       notes: list[str]) -> None:
    """Add a note to `notes` for each Gating-ML gate drawn on a channel's
    uncompensated values that sample `name` (`sample`) has compensated: it
    counts compensated ones there."""
    from .wsp_gate_coords import wsp_uncompensated_channels
    comp = list(getattr(sample, 'comp_channels', None) or [])
    for g in gate_dicts:
        for ch in wsp_uncompensated_channels(g, comp):
            msg = (f"'{g.get('label') or '?'}' is drawn on "
                   f"uncompensated {ch}; {name} is compensated "
                   "there, so it counts compensated values")
            if msg not in notes:
                notes.append(msg)


def onto_sample_scales(gate_dicts: Iterable[Mapping[str, Any]],
                       stored: Mapping[str, Any] | None, *,
                       from_wsp: bool = False, from_gml: bool = False
                       ) -> list[Gate]:
    """The template's gates on the scales the sample stores its channels on
    (`stored`, ``sample_prep.stored_transforms``). A .wsp holds FlowJo's
    linear units; a Gating-ML file names each axis's scale (a polygon on
    the same scale keeps its vertices exactly). A channel of unknown scale
    takes linear values and reads NA (mark_unknown_scale). A JSON
    template's gates are on the stored scales already."""
    from .wsp_gate_coords import gate_scale_fns, map_gate_coords, mark_unknown_scale
    if from_gml:
        from .gatingml import gates_to_stored
        return [mark_unknown_scale(g, stored)
                for g in gates_to_stored(gate_dicts, stored)]
    if from_wsp:
        from_flowjo = gate_scale_fns(stored, 'from_linear')
        return [mark_unknown_scale(map_gate_coords(dict(g), from_flowjo or {}),
                                   stored) for g in gate_dicts]
    return [dict(g) for g in gate_dicts]


def install_template(gate_dicts: Iterable[Gate],
                     add: Callable[[Gate], str]) -> list[str]:
    """Add the template's gates (on the sample's scales, parents first)
    with `add` (a gate -> its new id), each source parent id (.wsp
    ``_import_id`` or template ``id``) rewritten to the new id of its
    parent, and a boolean's operands too once every id is known. Returns
    the new ids in order."""
    from .gating import remap_operands
    old_to_new: dict[str, str] = {}
    added: list[Gate] = []
    ids: list[str] = []
    for g in gate_dicts:
        src_id = g.pop('_import_id', None) or g.pop('id', None)
        parent = g.get('parent_id')
        if parent is not None:
            g['parent_id'] = old_to_new.get(parent)
        gid = add(g)
        added.append(g)
        ids.append(gid)
        if src_id is not None:
            old_to_new[src_id] = gid
    # Boolean operands are template ids as well. Rewritten after the loop,
    # when every id is known; left as they are, with 'Add to existing' they
    # would name the sample's own gates (measured: 88 events, 5,035 right).
    for g in added:
        remap_operands(g, old_to_new)
    return ids


def broken_template_booleans(gates: Mapping[str, Mapping[str, Any]],
                             added_ids: Iterable[str]
                             ) -> tuple[list[str], list[str]]:
    """``(lost, built_on)``: the names (as the gate list shows them) of the
    template's boolean gates (`added_ids`, in one sample's `gates`) installed with an operand that
    could not be resolved, and of those built on one, all of which select
    nothing (gate_eval.broken_booleans)."""
    from .gate_eval import broken_booleans
    from .gate_text import listed_name
    lost_ids, built_ids = broken_booleans(gates)
    mine = set(added_ids)

    def names(ids: Iterable[str]) -> list[str]:
        return [listed_name(gates[i]) for i in ids if i in mine]
    return names(lost_ids), names(built_ids)


def channel_mismatches(columns: Iterable[str],
                       gate_dicts: Iterable[Mapping[str, Any]]) -> int:
    """How many of `gate_dicts` read a channel not among `columns`."""
    from .gating import gate_channels
    cols = set(columns)
    return sum(1 for g in gate_dicts if set(gate_channels(g)) - cols)


def relabel_note(relabelled: Mapping[str, str] | None) -> str:
    """The status note for the channel labels a template's Apply changed
    (``relabelled``: ``{detector: label}``). Labels are the session's, so
    the note says so: they change for every sample, not only the ticked
    ones."""
    if not relabelled:
        return ''
    shown = ', '.join(f"{d} as {v}" for d, v in list(relabelled.items())[:3])
    more = len(relabelled) - 3
    if more > 0:
        shown += f" and {more} more"
    return f" Relabeled {shown} for the whole session."


def applied_status(gate_dicts: Sequence[Mapping[str, Any]], source: str,
                   overwrite: bool, n_targets: int, lost: Sequence[str],
                   built_on: Sequence[str],
                   relabelled: Mapping[str, str] | None = None) -> str:
    """The status line after a template is applied to `n_targets`
    samples."""
    from .flowjo_placeholders import flowjo_placeholder_note
    from .gate_text import lost_operand_note
    kinds = ', '.join(sorted({g.get('kind', '?') for g in gate_dicts})) \
        or 'none'
    verb = 'overwrote' if overwrite else 'added to'
    return (f"Applied {len(gate_dicts)} gate(s) [{kinds}] from {source}: "
            f"{verb} {n_targets} sample(s)"
            f"{lost_operand_note(lost, built_on)}."
            + flowjo_placeholder_note(gate_dicts)
            + relabel_note(relabelled))


MISMATCH_TITLE = "Template channels missing in some samples"


MISMATCH_HEAD = ("Some gates reference channels that aren't present in "
                 "these samples — they'll be inactive (no-op) there until "
                 "the channels exist:")


def mismatch_lines(mismatches: Mapping[str, int]) -> list[str]:
    """One line per sample with gates on channels it lacks (`mismatches`:
    per sample, how many): they sit inert there."""
    return [f"{n}: {c} gate(s)" for n, c in mismatches.items()]


def mismatch_text(mismatches: Mapping[str, int]) -> str:
    """The warning for gates on channels some target samples lack."""
    return (MISMATCH_HEAD + "\n\n"
            + '\n'.join(f"  • {line}" for line in mismatch_lines(mismatches)))
