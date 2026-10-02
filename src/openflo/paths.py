"""Filesystem path helpers lifted out of the GUI — Tk-free, testable.

Drop-target expansion and sidecar-name sanitisation: pure path logic the
editor uses for drag-and-drop import and processed-data sidecars.
"""
from __future__ import annotations

import hashlib
import os


def safe_sidecar_name(name) -> str:
    """Filesystem-safe stem for a sample's processed-data sidecar. The same
    mapping is used by the session writer and the loader's fallback location,
    so a sidecar is found even if the recorded pointer is missing.

    Mapping every character outside ``[alnum-_]`` to ``_`` is not injective.
    ``Tube 01 Rep A`` and ``Tube_01_Rep_A`` produced the SAME stem, as did
    ``Day 1``/``Day.1``, ``Patient 01``/``Patient#01`` and
    ``Live/Dead``/``Live Dead``. The session writer then handed both samples
    that one file and recorded it on both entries, and restore PREFERS the
    sidecar over the raw FCS — so one sample came back holding the other's
    events, clusters and compensated values, with nothing said.

    A short digest of the ORIGINAL name is therefore appended whenever
    sanitisation actually changed it. A name that was already filesystem-safe
    keeps its plain stem, so sidecars written by earlier versions are still
    found by the loader's fallback; and two names can now share a stem only if
    they were identical to begin with.
    """
    text = str(name)
    safe = ''.join(ch if (ch.isalnum() or ch in '-_') else '_' for ch in text)
    if not safe:
        return 'sample'          # one empty name, so nothing to disambiguate
    if safe == text:
        return safe
    return f'{safe}_{hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]}'


def legacy_sidecar_name(name) -> str:
    """The pre-digest sidecar stem: a plain character substitution.

    Kept ONLY so the loader can still find sidecars written by earlier
    versions when the recorded pointer is missing. It is not injective — which
    is why it was replaced — so nothing should ever WRITE with it.
    """
    return ''.join(ch if (ch.isalnum() or ch in '-_') else '_'
                   for ch in str(name)) or 'sample'


def expand_dropped_paths(paths) -> tuple[list[str], list[str]]:
    """Resolve dropped paths (files and/or folders) into the flat ``.fcs`` and
    ``.wsp`` files they contain. Folders are walked recursively, so dropping a
    trial folder (or a parent of several) surfaces every sample inside.

    Returns ``(fcs_paths, wsp_paths)``, each de-duplicated and sorted for a
    deterministic load order.
    """
    fcs, wsp = set(), set()

    def _add_file(fp):
        low = fp.lower()
        if low.endswith('.fcs'):
            fcs.add(fp)
        elif low.endswith('.wsp'):
            wsp.add(fp)

    for p in paths:
        p = (p or '').strip().strip('"').strip("'")
        if not p:
            continue
        if os.path.isdir(p):
            for dirpath, _dirs, files in os.walk(p):
                for fn in files:
                    _add_file(os.path.join(dirpath, fn))
        elif os.path.isfile(p):
            _add_file(p)
    return sorted(fcs), sorted(wsp)
