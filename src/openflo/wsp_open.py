"""Importing a FlowJo workspace (.wsp) -- Tk-free.

A workspace lists its samples (``<Sample>``, with the FCS file's uri and
the sample's own compensation matrix), each with its gate tree
(``<SampleNode>``), and the groups the samples are in. ``plan_wsp`` reads
all of it into plain records, one per sample whose FCS file was found, so
the Tk gate editor and a headless document import a workspace the same
way. Finding the files is the caller's: it passes a locator, either an
object with ``find(uri) -> path or None`` (by default
``compare.FcsLocator``, which looks beside the workspace and in its
subfolders) or a plain function ``uri -> path or None``. A phone app passes
one that looks where it copied the files.

``gates_on_sample`` then puts a sample's imported gates on the units its
data is stored in; ``gate_store.install_gates`` adds them to a sample's
gate set with fresh ids, its parents and boolean operands remapped.

Core module: no Tk.
"""
from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .exceptions import WspParseError
from .gate_store import Gate

if TYPE_CHECKING:
    import xml.etree.ElementTree as ET

    from .wsp_reader import WspReader



@runtime_checkable
class FcsFinder(Protocol):
    """Finds the FCS file a workspace ``<DataSet uri>`` names."""

    def find(self, uri: str | None) -> str | None:
        """The local path of the file `uri` names, or None."""
        ...


class WspUnreadable(WspParseError):
    """The workspace could not be read or parsed; the message says why."""


@dataclass(frozen=True)
class WspSample:
    """One workspace sample whose FCS file was found.

    ``node_name`` is the name FlowJo shows (None when the workspace gives
    none), ``gates`` its gate tree as OpenFlo gate dicts in FlowJo's linear
    units (see ``gates_on_sample``), ``stat_axes`` the axes the workspace
    declares for its statistics, ``groups`` the workspace groups it is in
    (the first is the one it goes into) and ``comp`` its own compensation
    matrix as ``{'channels', 'matrix', 'source'}``, or None when it has
    none of its own (the file's $SPILL applies, as in FlowJo).
    ``set_number`` is the data set it is of a file holding several, and
    ``set_name`` that data set's name (``fcs_data_sets.data_set_names``);
    both None for a sample opened as the whole file."""

    node_name: str | None
    fcs_path: str
    gates: list[Gate]
    stat_axes: dict[str, Any]
    groups: list[str]
    comp: dict[str, Any] | None
    # The data set of an FCS file holding several (fcs_data_sets) the
    # sample is, matched by its <Keywords> (_wsp_data_set), and the name
    # that data set is listed under; None for the file as one sample.
    set_number: int | None = None
    set_name: str | None = None
    # How many data sets its <Keywords> matched (0 or several when
    # set_number is None), for the note that says why it opens whole.
    set_matches: int = 0


@dataclass
class WspPlan:
    """What a workspace holds and what of it was found.

    ``samples`` are those whose file was found, in workspace order;
    ``unresolved`` names the samples whose file was not; ``mismatched``
    holds ``(sample, why)`` for a file found with another event count than
    the workspace records; ``unread`` holds ``(population, tags, n under)``
    for each population whose gate type OpenFlo cannot read (left out with
    everything under it); ``ambiguous`` holds ``(file name, n files)`` for
    a file name the locator found several times and did not pick from;
    ``notes`` holds any other load note (a FACSDiva tube opened
    uncompensated, see diva_reader)."""

    samples: list[WspSample] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    mismatched: list[tuple[str, str]] = field(default_factory=list)
    unread: list[tuple[str, list[str], int]] = field(default_factory=list)
    ambiguous: list[tuple[str, int]] = field(default_factory=list)
    # Anything else the import says, each a load note as it is.
    notes: list[str] = field(default_factory=list)

    @property
    def imported(self) -> list[Gate]:
        """Every gate imported, sample by sample."""
        return [g for s in self.samples for g in s.gates]


def plan_wsp(wsp_path: str,
             locator: FcsFinder | Callable[[str | None], str | None] | None
             = None) -> WspPlan:
    """Read the workspace at `wsp_path`: per sample, its file (found by
    `locator`, an ``FcsFinder`` or a plain function of the uri; None uses
    ``compare.FcsLocator`` on the workspace's folder), gates, axes, groups
    and own matrix. A locator's ``ambiguous`` list, when it has one, is
    reported. A BD FACSDiva experiment XML is read the same way
    (``diva_reader.plan_diva``). Raises ``WspUnreadable`` when the file
    cannot be read."""
    from .diva_reader import DivaUnreadable, is_diva_xml, plan_diva
    if is_diva_xml(wsp_path):
        try:
            return plan_diva(wsp_path, locator)
        except DivaUnreadable as exc:
            raise WspUnreadable(str(exc)) from exc
    reader, root = _parse_wsp(wsp_path)
    if locator is None:
        from .compare import FcsLocator
        locator = FcsLocator(wsp_path)
    find = locator.find if isinstance(locator, FcsFinder) else locator
    plan = WspPlan()
    groups = reader.sample_groups()
    placed: set[tuple[str, int | None]] = set()
    for sample_elem in root.iter('Sample'):
        rec = _plan_sample(reader, find, groups, sample_elem, plan)
        if rec is not None and _place(rec, placed, plan):
            plan.samples.append(rec)
    plan.ambiguous = list(getattr(locator, 'ambiguous', None) or [])
    return plan


def _place(rec: WspSample, placed: set[tuple[str, int | None]],
           plan: WspPlan) -> bool:
    """Whether workspace sample `rec` is imported: not when another sample
    of the workspace already took its file (or its data set), since both
    would load as one sample and the second's gates would replace the
    first's. That is said in `plan`'s notes, as is a sample of a file
    holding several data sets that none of them matched."""
    from .fcs_data_sets import sample_data_sets
    from .paths import is_fcs_container, sample_key
    key = (sample_key(rec.fcs_path), rec.set_number)
    name = rec.node_name or '(unnamed)'
    fname = os.path.basename(rec.fcs_path)
    if key in placed:
        what = (fname if rec.set_number is None
                else f'data set {rec.set_number} of {fname}')
        plan.notes.append(
            f"'{name}' was not imported: another sample of the workspace "
            f"was already put on {what}, and its gates would have replaced "
            "that one's")
        return False
    placed.add(key)
    if rec.set_number is None:
        sets, _note = sample_data_sets(rec.fcs_path)
        if sets:
            which = 'several have' if rec.set_matches > 1 else 'none has'
            unit = ('well' if is_fcs_container(rec.fcs_path)
                    else 'data set')
            plan.notes.append(
                f"'{name}': {fname} holds {len(sets)} {unit}s and {which} "
                "this sample's event count and name ($TOT and "
                "$FIL or the like), so it opens as the whole file, on its "
                f"first {unit}; check its gates")
    return True


def _wsp_data_set(sample_elem: ET.Element,
                  fcs_path: str) -> tuple[int | None, str | None, int]:
    """``(set number, set name, matches)`` of the data set of `fcs_path`
    the workspace ``<Sample>`` `sample_elem` is, matched by its <Keywords>
    (``fcs_data_sets.data_set_matches``); number and name are None for a
    file opened as one sample, or unless exactly one data set matches, and
    `matches` says how many did."""
    from .fcs_data_sets import data_set_matches, data_set_names, sample_data_sets
    sets, _note = sample_data_sets(fcs_path)
    if not sets:
        return None, None, 0
    kws = {(kw.get('name') or '').upper(): kw.get('value') or ''
           for kw in sample_elem.findall('Keywords/Keyword')}
    hits = data_set_matches(sets, kws)
    if len(hits) != 1:
        return None, None, len(hits)
    number = hits[0]
    return number, data_set_names(sets, fcs_path)[number - 1], 1


def _parse_wsp(wsp_path: str) -> tuple[WspReader, ET.Element]:
    """(WspReader, namespace-stripped XML root) for a workspace."""
    from xml.etree import ElementTree as ET

    from .wsp_reader import WspReader

    try:
        reader = WspReader(wsp_path)
    except Exception as exc:  # noqa: BLE001 - any failure to read the workspace is reported to the caller
        raise WspUnreadable(str(exc)) from exc
    # Parsed again here to walk it per sample: the reader's extract_gates()
    # flattens, and each sample needs its own gate subtree.
    try:
        ns_re = re.compile(r'\{.*?\}')
        root = ET.parse(wsp_path).getroot()
        for elem in root.iter():
            elem.tag = ns_re.sub('', elem.tag)
            if elem.attrib:
                elem.attrib = {ns_re.sub('', k): v
                               for k, v in elem.attrib.items()}
    except Exception as exc:  # noqa: BLE001 - an unparsable workspace is reported to the caller
        raise WspUnreadable(str(exc)) from exc
    return reader, root


def _plan_sample(reader: WspReader,
                 find: Callable[[str | None], str | None],
                 groups: Mapping[str, list[str]], sample_elem: ET.Element,
                 plan: WspPlan) -> WspSample | None:
    """The record of one workspace ``<Sample>``, or None when it has no
    gate tree or its file was not found (then named in `plan`)."""
    from .wsp_gate_coords import wsp_stat_axes
    from .wsp_reader import (
        WspReader,
        fcs_event_total,
        wsp_event_mismatch,
        wsp_sample_events,
    )

    ds = sample_elem.find('DataSet')
    uri = ds.get('uri') if ds is not None else None
    sn = sample_elem.find('SampleNode')
    if sn is None:
        return None
    fcs_path = find(uri)
    if fcs_path is None:
        plan.unresolved.append(sn.get('name') or '(unnamed)')
        return None
    # A workspace can hold several files of one name; the one found may not
    # be the one FlowJo gated. Imported anyway, and said.
    set_number, set_name, matches = _wsp_data_set(sample_elem, fcs_path)
    why = wsp_event_mismatch(wsp_sample_events(sample_elem),
                             fcs_event_total(fcs_path, set_number))
    if why:
        plan.mismatched.append((sn.get('name') or '(unnamed)', why))
    # This <SampleNode>'s gates only. FlowJo writes Time gates in seconds,
    # scaled with the FCS's own timestep: the one of the file found.
    gates = reader.extract_gates(
        sample_node=sn,
        timestep=WspReader.sample_timestep(sample_elem, fcs_path,
                                           set_number))
    plan.unread.extend(reader.unread_populations)
    sid = sn.get('sampleID') or (ds.get('sampleID')
                                 if ds is not None else None)
    # Compensated as the workspace does: these gates were drawn on values
    # compensated with the sample's OWN matrix there, which may be a
    # bead-derived or edited one. A sample with none of its own gets
    # FlowJo's acquisition-defined one, the file's $SPILL.
    own = reader.sample_matrix(sample_elem)
    comp = None if own is None else {
        'channels': list(own['channels']), 'matrix': own['matrix'],
        'source': f"wsp:{own.get('name') or 'sample'}"}
    return WspSample(node_name=sn.get('name'), fcs_path=fcs_path,
                     gates=gates, stat_axes=wsp_stat_axes(sample_elem),
                     groups=list((groups.get(sid) if sid is not None else None)
                                 or []),
                     comp=comp, set_number=set_number, set_name=set_name,
                     set_matches=matches)


def gates_on_sample(gates: list[Gate],
                    stored_transforms: Mapping[str, Any], sample: Any
                    ) -> tuple[list[Gate], list[tuple[str, str]]]:
    """A sample's imported `gates` on the units its data is stored in, and
    ``(gate, channel)`` for each gate FlowJo evaluates on a channel's
    UNcompensated parameter while `sample` holds it compensated.

    Workspace gates are in FlowJo's linear units; `stored_transforms`
    (channel -> the transform its stored values are on) says how to carry
    them over. Unconverted, "FL1-A > 1000" would compare logicle values
    (below ~1.2) with 1000 and hold no events. A gate on a channel of
    unknown scale is kept but reads NA (``mark_unknown_scale``). A gate on
    a bare channel name (no 'Comp-') counts measured values in FlowJo,
    where this sample holds compensated ones, so its count differs."""
    from .wsp_gate_coords import (
        gate_scale_fns,
        map_gate_coords,
        mark_unknown_scale,
        wsp_uncompensated_channels,
    )
    fns = gate_scale_fns(stored_transforms, 'from_linear')
    out = [mark_unknown_scale(map_gate_coords(g, fns), stored_transforms)
           for g in gates]
    comp = list(getattr(sample, 'comp_channels', None) or [])
    uncompensated = [(g.get('label') or g.get('name') or '?', ch)
                     for g in out
                     for ch in wsp_uncompensated_channels(g, comp)]
    return out, uncompensated
