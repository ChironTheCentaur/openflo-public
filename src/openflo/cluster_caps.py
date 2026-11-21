"""How many events each clustering method clusters by default, and how much
memory a graph clustering of more needs. Tk-free and import-light, so the
library (FlowSample.cluster), openflo-run and the Cluster dialog share them.

Above its cap a method clusters a seeded random subsample and gives every
other event the label of its nearest clustered event (1-NN on a KD-tree);
FlowSOM instead trains its SOM on the subsample and maps every event to it.
"""
from __future__ import annotations

# PhenoGraph had no cap: a sample of about a million events built its
# k-nearest-neighbor graph on all of them and took several GB on one core
# (about 10 GB at k=30 by the fit below, more at a higher k), freezing an
# 8 GB laptop while the system swapped. It now caps at Leiden's 200,000,
# the other graph method.
PHENOGRAPH_DEFAULT_MAX_EVENTS = 200_000
LEIDEN_DEFAULT_MAX_EVENTS = 200_000
FLOWSOM_DEFAULT_MAX_EVENTS = 50_000

METHOD_EVENT_CAP = {'phenograph': PHENOGRAPH_DEFAULT_MAX_EVENTS,
                    'leiden': LEIDEN_DEFAULT_MAX_EVENTS,
                    'flowsom': FLOWSOM_DEFAULT_MAX_EVENTS}

# The graph methods whose memory grows with the events clustered, and so
# get a warning before a run on more than their cap.
GRAPH_METHODS = ('phenograph', 'leiden')

# Leiden's graph: True links only each event's own nearest neighbors
# (cluster_graph.snn_jaccard_graph prune=True, PhenoGraph's and Seurat's
# form), the default since the dense graph needed about 21 GB at Leiden's
# own 200,000-event default. False is the older dense graph, kept to
# compare with runs made before.
LEIDEN_FAST_GRAPH_DEFAULT = True

# Peak memory of one graph clustering, in GB, fitted (non-negative least
# squares) to the peak RSS of one process clustering generated data (12
# channels, 8 populations, one core; scripts/bench_cluster_memory.py) at
# k = 30, 60, 100 and 200, the Cluster dialog's range:
#
#     GB = MARGIN * (base + a*n + b*n*k + c*n**1.5*k)
#
#                         n      k=30   k=60   k=100   k=200
#     PhenoGraph         50k     0.61   0.96   1.41    2.63
#                       100k     1.07   1.76   2.71    5.13
#                       200k     1.93     -      -       -
#     Leiden, fast       50k     0.70   1.48   2.35    3.75
#      (the default)    100k     1.27     -    5.37    9.95
#                       200k     2.38   6.23     -       -
#     Leiden, dense      25k     2.46   5.42     -       -
#      (fast_graph off) 100k    10.31     -      -       -
#
# (MEMORY_MEASURED below holds every point the fit used.) PhenoGraph
# grows linearly in n and k. Leiden grows a little faster than linearly in
# n at a high k; the dense graph, which links every pair of events sharing
# a neighbor, about 9 times the fast graph at k=30, ran out of 15 GB at
# 50k events and k=60 (fitted as at least 12.5 GB). MARGIN covers more channels (30 channels measured
# 13% above 12) and keeps the estimate at or above every measured point.
MEMORY_MARGIN = 1.15
_MEMORY_FIT = {'phenograph': (0.171, 1.542e-6, 2.387e-7, 2.258e-12),
               'leiden': (0.0, 0.0, 3.335e-7, 4.477e-10),
               'leiden-dense': (0.0, 0.0, 3.658e-6, 0.0)}

# (method key, events, k, measured peak GB): the measurements behind
# _MEMORY_FIT, which tests/test_phenograph_event_cap.py holds it to.
MEMORY_MEASURED = (
    ('phenograph', 25_000, 30, 0.40), ('phenograph', 50_000, 30, 0.61),
    ('phenograph', 100_000, 30, 1.07), ('phenograph', 200_000, 30, 1.93),
    ('phenograph', 400_000, 30, 3.66), ('phenograph', 50_000, 60, 0.96),
    ('phenograph', 100_000, 60, 1.76), ('phenograph', 50_000, 100, 1.41),
    ('phenograph', 100_000, 100, 2.71), ('phenograph', 50_000, 200, 2.63),
    ('phenograph', 100_000, 200, 5.13),
    ('leiden', 50_000, 30, 0.70), ('leiden', 100_000, 30, 1.27),
    ('leiden', 200_000, 30, 2.38), ('leiden', 50_000, 60, 1.48),
    ('leiden', 200_000, 60, 6.23), ('leiden', 25_000, 100, 1.02),
    ('leiden', 50_000, 100, 2.35), ('leiden', 100_000, 100, 5.37),
    ('leiden', 25_000, 200, 1.36), ('leiden', 50_000, 200, 3.75),
    ('leiden', 100_000, 200, 9.95),
    ('leiden-dense', 25_000, 30, 2.46), ('leiden-dense', 50_000, 30, 5.06),
    ('leiden-dense', 75_000, 30, 7.68), ('leiden-dense', 100_000, 30, 10.31),
    ('leiden-dense', 10_000, 60, 1.55), ('leiden-dense', 25_000, 60, 5.42),
    # Ran out of memory at 11.9 GB (a 15 GB machine): counted as 12.5.
    ('leiden-dense', 50_000, 60, 12.5),
)


def default_cap(method: str) -> int | None:
    """The events `method` clusters when no cap of the user's own is set."""
    return METHOD_EVENT_CAP.get(method)


def clustering_memory_gb(method: str, n_events: int, k: int = 30,
                         fast_graph: bool = LEIDEN_FAST_GRAPH_DEFAULT
                         ) -> float | None:
    """About how many GB of memory one `method` clustering of `n_events`
    events with `k` neighbors needs at its peak (margin included), or None
    for a method not estimated. `fast_graph` picks Leiden's graph (ignored
    for the others)."""
    key = 'leiden-dense' if method == 'leiden' and not fast_graph else method
    fit = _MEMORY_FIT.get(key)
    if fit is None or n_events <= 0:
        return None
    base, a, b, c = fit
    n, kk = float(n_events), float(max(int(k), 1))
    return MEMORY_MARGIN * (base + a * n + b * n * kk + c * n ** 1.5 * kk)


def available_memory_gb() -> float | None:
    """Free RAM in GB (psutil's 'available'), or None when unknown."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        return psutil.virtual_memory().available / 1024 ** 3
    except (OSError, RuntimeError):
        return None


def large_run_warning(method: str, n_events: int, free_gb: float | None,
                      fast_graph: bool = LEIDEN_FAST_GRAPH_DEFAULT,
                      k: int = 30) -> str | None:
    """The warning to show before a graph clustering of `n_events` events
    in one sample, or None. It warns above the method's default cap, and
    at any size whose memory estimate is more than the `free_gb` free now
    (Leiden's dense graph needs about 21 GB at 200,000). Not for
    FlowSOM. `fast_graph` is Leiden's graph choice, `k` the neighbors."""
    cap = default_cap(method)
    need = clustering_memory_gb(method, n_events, k, fast_graph)
    if method not in GRAPH_METHODS or cap is None or need is None:
        return None
    above_cap = n_events > cap
    too_big = free_gb is not None and need > free_gb
    if not (above_cap or too_big):
        return None
    name = {'phenograph': 'PhenoGraph', 'leiden': 'Leiden'}[method]
    text = f"{name} will cluster {n_events:,} events in one sample"
    if above_cap:
        text += f", more than its default of {cap:,}"
    text += f". That needs about {need:.1f} GB of memory"
    if free_gb is not None:
        text += f" ({free_gb:.1f} GB is free now)"
    text += " and can take a long time."
    if too_big:
        text += (" It needs more memory than is free, so the computer may "
                 "slow down badly while it runs.")
    if above_cap:
        text += (f"\n\nWith Events / sample set to Default, it clusters "
                 f"{cap:,} events and gives every other event the cluster "
                 "of its nearest clustered event, which is much faster.")
    else:
        text += ("\n\nFewer events (Events / sample: Custom) or FlowSOM "
                 "need much less memory.")
    return text + "\n\nRun anyway?"
