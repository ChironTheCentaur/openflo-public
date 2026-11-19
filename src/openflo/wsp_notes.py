"""What a FlowJo workspace import says to the user: the one-line summary
(``wsp_summary``), the load notes (``wsp_load_notes``), each sample's own
load warnings (``sample_load_notes``, for any load), the boolean gates
built on an unconverted ellipse (``placeholder_boolean_note``) and the
words for gates FlowJo evaluates on an uncompensated parameter
(``uncompensated_summary``, ``uncompensated_gate_note``).

Core module: no Tk. The desktop editor (``editor_load``) and the phone's
engine (``document.Document.import_wsp``) both take their text from here, so
the two say the same thing about the same workspace.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def wsp_summary(wsp: str, plan: Any) -> str:
    """The status line after staging workspace ``wsp`` (a basename) from
    its plan (wsp_open.WspPlan)."""
    from .flowjo_placeholders import flowjo_placeholder_note

    unresolved, unread = plan.unresolved, plan.unread
    summary = (f"[WSP] {wsp}: queued "
               f"{len(plan.samples)} sample(s), "
               f"{len(plan.imported)} gate(s)")
    if unresolved:
        summary += f" — couldn't locate FCS for: {', '.join(unresolved[:3])}"
        if len(unresolved) > 3:
            summary += f" (+{len(unresolved) - 3})"
    # An ellipse the import could not convert counts nothing; say so where
    # the user looks, not only in the log.
    summary += flowjo_placeholder_note(plan.imported)
    if plan.mismatched:
        summary += (f" — {len(plan.mismatched)} sample(s) matched to an FCS "
                    "file with a different number of events than the "
                    "workspace records (see the load notes)")
    if unread:
        # Left out with everything under them, not lifted to the
        # grandparent (see WspReader.extract_gates).
        summary += (f" — {len(unread)} population(s) with a gate type "
                    "OpenFlo cannot read were not imported, nor were the "
                    f"{sum(n for _, _, n in unread)} under them: "
                    + ', '.join(f"'{nm}'" for nm, _, _ in unread[:3])
                    + (f" (+{len(unread) - 3})" if len(unread) > 3 else ''))
    return summary


def wsp_load_notes(wsp: str, plan: Any,
                   in_several: Iterable[tuple[str, Sequence[str]]]
                   ) -> list[str]:
    """The load notes for workspace ``wsp`` (a basename) from its plan
    (wsp_open.WspPlan); `in_several` holds (sample, its groups) for each
    sample in more than one group."""
    notes = []
    if plan.unresolved:
        notes.append(f"{wsp}: FCS file not found for "
                     + ', '.join(plan.unresolved))
    for fname, n in plan.ambiguous:
        notes.append(
            f"{wsp}: {n} files named {fname} in its folders; none was "
            "picked. Point the FCS folder at the right one.")
    for nm, why in plan.mismatched:
        notes.append(f"{wsp}: '{nm}': {why}")
    for nm, mine in in_several:
        notes.append(
            f"{wsp}: '{nm}' is in the groups {', '.join(mine)}; it was "
            f"put in '{mine[0]}'")
    for nm, tags, n_under in plan.unread:
        notes.append(
            f"{wsp}: population '{nm}' ({', '.join(tags)}) not imported"
            + (f", nor the {n_under} under it" if n_under else ''))
    notes.extend(f"{wsp}: {n}" for n in getattr(plan, 'notes', None) or ())
    for g in plan.imported:
        if g.get('kind') == 'flowjo_ellipse':
            notes.append(
                f"{wsp}: ellipse '{g.get('label') or '?'}' could not be "
                "converted and reads NA")
    return notes


def uncompensated_summary(pairs: Sequence[tuple[str, str]]) -> str:
    """Status-line text for imported .wsp gates FlowJo evaluates on a
    channel's UNcompensated parameter (`pairs` of (gate, channel)), or ''.
    The sample holds that channel compensated only, so these gates cannot
    count what FlowJo counts; openflo-compare can."""
    if not pairs:
        return ''
    gates = list(dict.fromkeys(g for g, _ in pairs))
    chans = sorted({ch for _, ch in pairs})
    shown = ', '.join(f"'{g}'" for g in gates[:3]) + (
        f' (+{len(gates) - 3})' if len(gates) > 3 else '')
    return (f" — {len(gates)} imported gate(s) are on FlowJo's "
            f"UNcompensated {', '.join(chans)} (no 'Comp-' in the "
            f"workspace) but this sample holds it compensated, so their "
            f"counts will differ from FlowJo's: {shown}")


def uncompensated_gate_note(gate: str, channel: str) -> str:
    """The load note for one imported gate on FlowJo's uncompensated
    `channel`."""
    return (f"imported gate '{gate}' is on FlowJo's uncompensated "
            f"{channel}, but this sample holds it compensated: its count "
            "will differ from FlowJo's")


def sample_load_notes(name: str, sample: Any) -> list[str]:
    """The load notes for sample `name` from what reading its file warned
    (``load_warnings``: a matrix that could not be used, an event count
    other than $TOT, a channel named twice, ...), one per warning."""
    return [f"{name}: {w}"
            for w in getattr(sample, 'load_warnings', None) or ()]


def placeholder_boolean_note(wsp: str, name: str) -> str:
    """The load note for boolean gate `name` of workspace ``wsp``, built on
    a FlowJo ellipse the import could not convert: it selects nothing."""
    return (f"{wsp}: boolean gate '{name}' is built on an ellipse that was "
            "not converted and reads NA")
