"""openflo-run's example panel, FMO control sets and groups: the defaults
used when no --groups / --samples / --fmo-sets are given.

Moved out of cli.py unchanged; cli still serves these names.
"""
from __future__ import annotations

# ── Example panel + controls (EDIT for your experiment) ──────────────────────
# A generic 3-colour example mapping standard CD markers to detector channels.
# The pipeline is panel-agnostic — nothing here is required; override per run
# with --channels / --groups / --samples / --fmo-sets (or via the GUI).
CD901b = 'FL1-A'
CD902 = 'FL2-A'
CD903 = 'FL3-A'

# Example FMO control sets: each maps a detector channel to the FCS file
# basename holding that channel's single-stain / FMO control. Replace the
# basenames with your own (or pass --fmo-sets). The 'A'/'B' sets illustrate
# two staining runs; groups reference a set by name.
FMOS_SET_A = {
    CD901b: 'fmo_fl1_a',
    CD902: 'fmo_fl2_a',
    CD903: 'fmo_fl3_a',
}
FMOS_SET_B = {
    CD901b: 'fmo_fl1_b',
    CD902: 'fmo_fl2_b',
    CD903: 'fmo_fl3_b',
}

# Named FMO sets. Each set maps a detector channel to the FCS file basename
# that contains the single-stained / no-stain control for that channel. Groups
# reference these by name; extend this dict (via the GUI or by editing) for
# more staining runs without touching any other code.
DEFAULT_FMO_SETS = {
    'Set A': dict(FMOS_SET_A),
    'Set B': dict(FMOS_SET_B),
}

# Example default groups — used ONLY when neither --groups nor --samples is
# supplied. Replace with your own group / sample names.
DEFAULT_GROUPS = [
    {'name': 'Group A', 'samples': ['sample_1', 'sample_2'], 'fmo_set': 'Set A'},
    {'name': 'Group B', 'samples': ['sample_3', 'sample_4'], 'fmo_set': 'Set B'},
]
