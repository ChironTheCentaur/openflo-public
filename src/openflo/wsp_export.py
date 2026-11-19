"""What a FlowJo .wsp export writes for one sample, shared by the editor's
File > Export FlowJo .wsp and ``Document.export_wsp`` (expanding a project
archive back to FCS files and a workspace). Tk-free."""
from __future__ import annotations

import logging
import os
from typing import Any

from .gate_lineage import ancestors
from .paths import is_fcs_file

log = logging.getLogger(__name__)


def exported_parent(gates, gid, skipped):
    """The parent ``gid`` is written under when the ``skipped`` gates stay
    out of the export: its nearest ancestor that is kept. When every
    ancestor is skipped, the last one's own parent_id (None for a root)."""
    up = ancestors(gates, gid)
    kept = next((a for a in up if a not in skipped), None)
    if kept is not None:
        return kept
    return (gates[up[-1]] if up else gates[gid]).get('parent_id')


def wsp_gate_list(gates, order):
    """Sample gates `gates` (``{id: gate}``, `order` their ids) as the
    writer takes them: each with its ``id`` and the ``parent_id`` it is
    written under. Gates with no FlowJo geometry, and boolean gates FlowJo
    cannot express (``gate_text.wsp_unexportable``), are left out; a gate
    under one of them is put under its nearest ancestor that is kept, so it
    is not orphaned."""
    from .gate_text import wsp_unexportable
    skipped = wsp_unexportable(gates)
    out = []
    for gid in order:
        if gid in skipped or gid not in gates:
            continue
        g = dict(gates[gid])
        g['parent_id'] = exported_parent(gates, gid, skipped)
        g['id'] = gid
        out.append(g)
    return out


def match_unknown_scales(sample, stored, gates, comp_channels, comp_matrix):
    """``{channel: f}`` (see wsp_gate_coords.rank_matched_fn) for each channel of
    unknown scale that one of `gates` is on, matched to the same channel
    of the sample's raw FCS as FlowJo will gate it: compensated with the
    matrix written for the sample, after QC when the sample had it. A
    channel without such an FCS, or one that cannot be matched, is left
    out, and the writer exports its gates as stored (with a warning)."""
    from .channel_transforms import is_unknown_spec
    from .flowjo_events import fcs_values_as_flowjo_gates_them
    from .gating import gate_channels
    from .wsp_gate_coords import rank_matched_fn
    unknown = {ch for ch, sp in (stored or {}).items()
               if is_unknown_spec(sp)}
    used = sorted({ch for g in gates if not g.get('scale_unknown')
                   for ch in gate_channels(g)} & unknown)
    fcs = sample_fcs(sample)
    data = getattr(sample, 'data', None)
    if (not used or data is None or not is_fcs_file(fcs)
            or not os.path.isfile(fcs)):
        return {}
    try:
        raw = fcs_values_as_flowjo_gates_them(
            fcs, qc=bool(getattr(sample, 'qc_applied', False)),
            compensation=((comp_channels, comp_matrix)
                          if comp_matrix is not None and comp_channels
                          else None),
            set_number=getattr(sample, 'set_number', None))
    except Exception as exc:  # noqa: BLE001 - logged; unknown-scale gates of an unreadable FCS are exported as stored (the writer warns)
        log.warning("[wsp export] %s: could not read %s to match its "
                    "unknown-scale channels (%s)",
                    getattr(sample, 'name', ''), os.path.basename(fcs), exc)
        return {}
    out = {}
    for ch in used:
        if ch in data.columns and ch in raw.columns:
            f = rank_matched_fn(data[ch].to_numpy(), raw[ch].to_numpy())
            if f is not None:
                out[ch] = f
    return out


def sample_fcs(sample: Any) -> str:
    """The raw FCS a sample came from, also for one restored from a
    processed CSV (FlowJo cannot open the CSV); '' when unknown."""
    return str(getattr(sample, 'source_fcs', '')
               or getattr(sample, 'path', '') or '')


def own_compensation(sample: Any):
    """``(channels, matrix)`` the sample is compensated with, or None:
    FlowJo applies a per-sample matrix as given."""
    m = getattr(sample, 'comp_matrix', None)
    c = getattr(sample, 'comp_channels', None)
    return (c, m) if m is not None and c else None


def _same_bytes(a: str, b: str) -> bool:
    """True when files `a` and `b` hold the same bytes."""
    import filecmp
    try:
        return filecmp.cmp(a, b, shallow=False)
    except OSError:
        return False


def _free_name(folder: str, name: str, taken: set[str]) -> str:
    """`name`, or 'stem (2).ext', 'stem (3).ext', ... : the first that is
    neither in `taken` (case-insensitive, as Windows and macOS compare
    names) nor a file in `folder`."""
    stem, ext = os.path.splitext(name)
    out, i = name, 2
    while (out.lower() in taken
           or os.path.lexists(os.path.join(folder, out))):
        out = f"{stem} ({i}){ext}"
        i += 1
    return out


def copy_fcs_beside(sources: dict[str, str], wsp_path: str):
    """Copy each sample's FCS file into the folder of the workspace at
    `wsp_path`, so the workspace and its data travel together: the
    workspace then names each file alone, relative to itself
    (``WspWriter(file_names_only=True)``), and FlowJo finds it there on any
    computer.

    `sources` maps sample name to the FCS file it was read from
    (``sample_fcs``). The originals are only read, never changed. A file
    already in that folder is used as it is; a different file of the same
    name there, or another sample's file of that name, is never
    overwritten: the copy gets a free name ('x (2).fcs'). Samples of one
    file (wells of one Accuri file) share one copy. Each copy is written
    under a new temporary name ('.x.fcs.<random>.part') and renamed when
    complete, so a failed copy leaves no half file.

    Returns ``(paths, missing)``: ``{sample: path of its file beside the
    workspace}`` and the samples whose FCS file is not there to copy (in
    `sources` order); those keep their original path."""
    import shutil
    import tempfile
    folder = os.path.dirname(os.path.abspath(wsp_path))
    paths: dict[str, str] = {}
    missing: list[str] = []
    done: dict[str, str] = {}      # realpath of an original -> its copy
    taken: set[str] = set()        # names given out in this export
    for name, src in sources.items():
        if not src or not os.path.isfile(src):
            missing.append(name)
            paths[name] = src
            continue
        real = os.path.realpath(src)
        if real in done:
            paths[name] = done[real]
            continue
        base = os.path.basename(src)
        here = os.path.join(folder, base)
        if os.path.isfile(here) and (os.path.realpath(here) == real
                                     or _same_bytes(here, src)):
            dest = here                      # already beside the workspace
        else:
            dest = os.path.join(folder, _free_name(folder, base, taken))
            # A temporary name no file has: never one of the user's.
            fd, part = tempfile.mkstemp(dir=folder, prefix=f'.{base}.',
                                        suffix='.part')
            os.close(fd)
            try:
                shutil.copyfile(src, part)
                try:                       # dates kept where the disk can
                    shutil.copystat(src, part)
                except OSError:
                    pass
                os.replace(part, dest)
            except BaseException:
                try:
                    os.remove(part)
                except OSError:
                    pass
                raise
        taken.add(os.path.basename(dest).lower())
        done[real] = dest
        paths[name] = dest
    return paths, missing
