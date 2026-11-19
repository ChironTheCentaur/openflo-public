"""Names and colours shared by gates and clusters: ``GATE_PALETTE``, the noise
cluster sentinel (``NOISE_CLUSTER_ID``, ``NOISE_CLUSTER_LABEL``,
``NOISE_CLUSTER_COLOR``) and ``cluster_label``.

Core module: the standard library only. No Tk, and it does not import
``openflo.pipeline``.
"""

# Categorical palette used by the GUI editor (and any other gate authors)
# to auto-assign a colour when a gate is created without one. Same order
# as the well-known "20 distinct colours" list (Sasha Trubetskoy 2017).
GATE_PALETTE = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
    '#008080', '#e6beff', '#9a6324', '#800000', '#aaffc3',
    '#808000', '#ffd8b1', '#000075', '#808080', '#000000',
]

# The clustering sentinel for events that were never assigned to a cluster —
# non-finite values in a clustering channel, PhenoGraph outliers, or sub-sample
# "rest" events that couldn't be back-assigned. It is NOT a biological
# population; it is kept in stats/exports/displays but ALWAYS labelled
# explicitly (never silently dropped, never shown as a bare "Cluster -1") so it
# can't be mistaken for one.
NOISE_CLUSTER_ID = -1
NOISE_CLUSTER_LABEL = 'Unclustered (noise)'
NOISE_CLUSTER_COLOR = '#9aa0a8'          # neutral grey — reads as "not a pop"


def cluster_label(cid):
    """Display name for a cluster id: ``Cluster N`` for a real cluster, or the
    explicit noise label for the -1 sentinel. Shared by the pipeline stats
    exports and the GUI so both name the noise bucket identically."""
    try:
        cid = int(cid)
    except (TypeError, ValueError):
        return str(cid)
    return NOISE_CLUSTER_LABEL if cid == NOISE_CLUSTER_ID else f'Cluster {cid}'
