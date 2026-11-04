"""openflo-run's channel labels: --labels pairs and the staining-panel
spreadsheet (found next to the trials with --panel auto) that names each
detector channel by its CD marker.

Moved out of cli.py unchanged; parse_labels, read_staining_panel and
find_panel_xlsx are still public cli names, and the old private cli name of
the token normaliser still works, deprecated.
"""
from __future__ import annotations

import os
import re


def parse_labels(labels_str):
    """Parse 'DetA=LabelA;DetB=LabelB' into {DetA: LabelA, ...}."""
    if not labels_str:
        return {}
    result = {}
    for pair in labels_str.split(';'):
        if '=' in pair:
            det, lbl = pair.split('=', 1)
            result[det.strip()] = lbl.strip()
    return result


def norm_token(s):
    """Lower-case and strip all non-alphanumerics — so 'Dye/X7',
    'Dye-X7-A' and 'dyex7' all compare equal at the token level."""
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def read_staining_panel(path, channels):
    """Read a staining-panel spreadsheet → ``{detector_channel: CD_label}``.

    The sheet pairs a CD/marker token (e.g. ``CD901b``) with a fluorophore
    token (e.g. ``FL1``, ``Dye/X7``, ``FL2``) on the same row, in two
    columns, in any order, header optional. Each fluorophore is matched to
    the detector channel whose name *contains* it (normalised, case- and
    punctuation-insensitive), so ``FL1`` → ``FL1-A`` and
    ``Dye/X7`` → ``Dye-X7-A``. Returns ``{}`` when nothing matched (the run
    then falls back to whatever ``--labels`` provided, or the raw detector
    names).

    `channels` is the list of real detector channel names to match against
    (typically a sample's ``channel_names`` / DataFrame columns).
    """
    try:
        import pandas as pd
        sheets = pd.read_excel(path, header=None, sheet_name=None)
    except Exception as exc:  # noqa: BLE001 - reported, gives no labels
        print(f"  [panel] Could not read {path}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return {}

    norm_channels = {ch: norm_token(ch) for ch in channels}

    def match_fluor(fluor):
        nf = norm_token(fluor)
        if not nf:
            return None
        cands = [ch for ch, nc in norm_channels.items() if nf and nf in nc]
        if not cands:
            return None
        # Prefer area ('-A') detectors and the shortest channel name.
        cands.sort(key=lambda c: ('-a' not in c.lower(), len(c)))
        return cands[0]

    mapping = {}
    cd_re = re.compile(r'^cd\d', re.I)
    for df in sheets.values():
        for _, row in df.iterrows():
            cells = [str(v).strip() for v in row.tolist()
                     if str(v).strip() and str(v).strip().lower() != 'nan']
            cd = next((c for c in cells if cd_re.match(c)), None)
            if not cd:
                continue
            for fl in (c for c in cells if c != cd):
                det = match_fluor(fl)
                if det and det not in mapping:
                    mapping[det] = cd
                    break
    if mapping:
        print(f"  [panel] {os.path.basename(path)} → "
              f"{dict(mapping)}", flush=True)
    else:
        print(f"  [panel] {os.path.basename(path)}: no CD↔fluorophore "
              f"rows matched the data channels.", flush=True)
    return mapping


def find_panel_xlsx(dirs):
    """Locate a staining-panel spreadsheet for the given trial dir(s).

    Searches each dir's descendants first, then walks up a few ancestor
    levels (the panel commonly sits in a study-root folder above the day
    sub-folders). Files whose name mentions 'panel' or 'stain' win ties.
    Returns a path or None.
    """
    candidates = []
    seen = set()

    def add(p):
        ap = os.path.abspath(p)
        if ap not in seen and os.path.isfile(ap):
            base = os.path.basename(ap)
            if base.lower().endswith(('.xlsx', '.xls')) \
                    and not base.startswith('~$'):
                seen.add(ap)
                candidates.append(ap)

    for d in dirs:
        if not d:
            continue
        d = os.path.abspath(d)
        # Descendants of the trial dir.
        for root, _dn, files in os.walk(d):
            for f in files:
                add(os.path.join(root, f))
        # Direct contents of up to 4 ancestor levels.
        cur = d
        for _ in range(4):
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            try:
                for f in os.listdir(parent):
                    add(os.path.join(parent, f))
            except OSError:
                pass
            cur = parent

    candidates.sort(key=lambda p: (
        'panel' not in os.path.basename(p).lower()
        and 'stain' not in os.path.basename(p).lower(),
        len(p)))
    return candidates[0] if candidates else None
