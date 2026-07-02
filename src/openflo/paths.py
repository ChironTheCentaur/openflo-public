"""Filesystem path helpers lifted out of the GUI — Tk-free, testable.

Drop-target expansion and sidecar-name sanitisation: pure path logic the
editor uses for drag-and-drop import and processed-data sidecars.
"""
from __future__ import annotations

import os


def safe_sidecar_name(name) -> str:
    """Filesystem-safe stem for a sample's processed-data sidecar. The same
    mapping is used by the session writer and the loader's fallback location,
    so a sidecar is found even if the recorded pointer is missing."""
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
