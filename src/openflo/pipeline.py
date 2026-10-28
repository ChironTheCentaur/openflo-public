"""
flow_pipeline.py
----------------
Generalized flow cytometry analysis pipeline.

Features
--------
- Auto-reads FCS metadata (channels, labels, spillover)
- FlowJo .wsp compensation matrix reader
- Time-based QC (acquisition anomaly detection)
- FMO-based gate threshold calculation
- Logicle / log transform
- Phenograph clustering
- UMAP dimensionality reduction
- Sample concatenation with origin labeling
- Condition-level frequency comparison
- Statistics export (FlowJo Table-style)

Requirements
------------
    pip install flowio flowutils numpy pandas matplotlib seaborn \
                phenograph scikit-learn umap-learn scipy
"""

import contextlib
import copy
import csv
import functools
import hashlib
import logging
import os
import re
import tempfile
import threading
import uuid
import warnings
import xml.etree.ElementTree as ET
from typing import Any, cast

import flowio
import numpy as np
import pandas as pd
from flowutils import transforms

from . import fcs_read as _fcs_read
from . import numba_threads
from .errors import note
from .fcs_export import fcs_event_block
from .parallel import tree_kneighbors
from .robust import MAD_TO_SD
from .som_kernel import FLOWSOM_DEFAULT_METACLUSTERS

# Heavy / optional deps are loaded on demand via the PEP-562 ``__getattr__``
# hook at the bottom of this module:
#   - ``phenograph``       — only needed by FlowSample.cluster()
#   - ``seaborn``          — only by heatmap-style plots
#   - ``gaussian_kde``     — only by density plot paths
#   - ``matplotlib.pyplot`` — only by the plot methods (~370 ms saved;
#     re-measured 2026-09-10, `python scripts/bench_import_cost.py`)
# Importing the module no longer pulls in igraph + scikit-learn's community
# detection OR matplotlib's Tk backend probing. Matters for the gate
# editor / compare tool / WSP-only callers — `import openflo.pipeline`
# drops from ~1050 ms to ~400 ms.
#
# Plot methods inside this module add a local `import matplotlib.pyplot
# as plt` at the top because PEP-562 __getattr__ only fires for OTHER
# modules accessing `pipeline.plt`; bare-name lookups inside this module
# follow normal scoping rules and would NameError without the local.

warnings.filterwarnings('ignore', category=FutureWarning)


# Module logger. Configure the root logger via the CLI (see openflo.cli)
# or programmatically with ``logging.basicConfig(level=logging.INFO)`` to
# see pipeline progress. Levels in use here:
#   DEBUG    — fine-grained per-event diagnostics (none currently)
#   INFO     — normal progress: load, QC, comp, transform, cluster, UMAP
#   WARNING  — recoverable issues that the pipeline routed around
#              (missing channels, GPU fallback, malformed gates, …)
#   ERROR    — only used by exception-raising code paths (rare; we mostly
#              raise OpenFloError subclasses instead)
log = logging.getLogger(__name__)


# ── Exception hierarchy ───────────────────────────────────────────────────────
# All recoverable errors raised by the pipeline are subclasses of
# OpenFloError. Top-level callers can ``except OpenFloError`` to surface a
# user-friendly message, then fall through to ``except Exception`` for
# unexpected bugs (which deserve a full traceback).

class OpenFloError(Exception):
    """Base class for every error raised intentionally by OpenFlo."""


class FcsParseError(OpenFloError):
    """Raised when an FCS file can't be read or its metadata is malformed."""


class CompensationError(OpenFloError):
    """Raised when a compensation matrix can't be parsed or applied."""


class WspParseError(OpenFloError):
    """Raised when a FlowJo .wsp can't be parsed."""


class GateError(OpenFloError):
    """Raised when a gate definition is invalid or can't be applied."""


class ClusteringError(OpenFloError):
    """Raised when clustering (Phenograph CPU or RAPIDS GPU) fails."""


# ── Constants ─────────────────────────────────────────────────────────────────

# Kept for callers that read it; classification uses _is_scatter, which
# matches these (and Beckman / Sony / CyTOF names) case-insensitively.
SCATTER_KEYWORDS = ['FSC', 'SSC', 'Time', 'time', 'Width', 'width']
# Derived / analysis columns — never treated as markers for clustering,
# stats, channel classification or acquisition QC. Keep this in step with
# every column the app WRITES back onto sample.data: three of the five
# embeddings were listed and TSNE/PHATE were not, and `leiden` and
# `pseudotime` were missing entirely. Acquisition QC selects channels as
# "every numeric column minus this list", so an omission here is not
# cosmetic — the margin detector read a cluster label as a detector at its
# ceiling and deleted the whole top-numbered cluster (measured: 12.4% of a
# clean sample). tests/test_qc_ignores_derived_columns.py pins the list
# against what the code actually writes.
#
# '__group__' / '__sample__' are the workspace's per-event source tags,
# written into every run's _events.csv; counted as fluor channels they made
# the Statistics window take the median of '(ungrouped)' (ValueError).
EXCLUDE_CLUSTER  = ['Time', 'time',
                    'cluster', 'flowsom', 'flowsom_meta', 'cell_cycle',
                    'debarcode', 'leiden', 'pseudotime',
                    'UMAP1', 'UMAP2', 'TSNE1', 'TSNE2', 'TRIMAP1', 'TRIMAP2',
                    'PACMAP1', 'PACMAP2', 'PHATE1', 'PHATE2',
                    '__group__', '__sample__']

# Phenograph (native Louvain backend) writes scratch files — kNN graph
# `*.bin`, dendrogram `*.tree`, and `*_graph.weights` — into the current
# working directory, and deletes them only when the run succeeds. Each call
# runs in its own temporary folder, removed afterwards whatever happens. It
# used to be a `.phenograph_cache` folder inside the package, which a pip
# install puts in site-packages (read-only on a system Python) and which
# kept the scratch files of every failed run.

# phenograph 1.5.7's cluster() ends with ``sort_by_size(communities,
# min_cluster_size)`` and does not pass its n_jobs on. sort_by_size then uses
# its own default of -1 and opens ``mp.Pool(cpu_count())``, so every call
# spawned a full-width Pool even at n_jobs=1. A spy on multiprocessing.Pool saw
# Pool(24) per cluster(n_jobs=1) call on a 24-core host. Under the CLI's
# parallel workers that is workers x cpu_count extra interpreters, each
# re-importing the numeric stack. To fix this without forking phenograph, its
# module-level sort_by_size is swapped (once) for a wrapper. The wrapper applies
# the n_jobs of the OpenFlo call running on the same thread. Any other caller
# gets phenograph's own behaviour, and n_jobs=1 runs in-process with no Pool.
_PG_SORT_JOBS = threading.local()
_PG_SORT_INSTALL_LOCK = threading.Lock()


def _sort_by_size_serial(clusters, min_size=10):
    """phenograph.cluster.sort_by_size, without its process Pool.

    The Pool only counts each cluster's events, in np.unique order. Counting
    in-process gives the same sizes in the same order, and the rest is
    phenograph's own code verbatim (its argsort, its position-keyed mapping,
    its np.vectorize), so the labels come out identical."""
    _, counts = np.unique(clusters, return_counts=True)
    sizes = [int(c) for c in counts]
    o = np.argsort(sizes)[::-1]
    my_dict = {c: i for i, c in enumerate(o) if sizes[c] > min_size}
    my_dict.update({c: -1 for i, c in enumerate(o) if sizes[c] <= min_size})
    return np.vectorize(my_dict.get)(clusters)


def _install_phenograph_sort_cap():
    """Route phenograph.cluster's sort_by_size through the n_jobs cap. Runs
    once; a phenograph without that function is left untouched."""
    import importlib
    try:
        pgc = importlib.import_module('phenograph.cluster')
    except ImportError:
        return
    with _PG_SORT_INSTALL_LOCK:
        orig = getattr(pgc, 'sort_by_size', None)
        if orig is None or getattr(orig, '_openflo_n_jobs_cap', False):
            return

        @functools.wraps(orig)
        def sort_by_size(clusters, min_size=10, n_jobs=-1):
            want = getattr(_PG_SORT_JOBS, 'n_jobs', None)
            if want is None:                  # not an OpenFlo cluster() call
                return orig(clusters, min_size, n_jobs)
            if want == 1:
                return _sort_by_size_serial(clusters, min_size)
            return orig(clusters, min_size, want)

        sort_by_size._openflo_n_jobs_cap = True  # type: ignore[attr-defined]
        pgc.sort_by_size = sort_by_size  # pyright: ignore[reportAttributeAccessIssue]


@contextlib.contextmanager
def _phenograph_sort_jobs(n_jobs):
    """While active on this thread, phenograph's size sort honours ``n_jobs``."""
    _install_phenograph_sort_cap()
    prev = getattr(_PG_SORT_JOBS, 'n_jobs', None)
    _PG_SORT_JOBS.n_jobs = n_jobs
    try:
        yield
    finally:
        _PG_SORT_JOBS.n_jobs = prev

# Default categorical palette for per-sample / per-condition colouring on
# plots that group by `sample_origin` etc. `'auto'` picks the colour-blind
# safe Okabe-Ito colours for ≤8 groups and falls back to tab20 / evenly
# spaced hues as the group count grows. Override via `set_default_palette('Set2')` or by
# passing `--palette` / the GUI palette dropdown to the pipeline.
_DEFAULT_CATEGORICAL_PALETTE = 'auto'


def categorical_cmap(name, n_groups):
    """A colormap whose first ``n_groups`` steps are the group colours.

    'auto' and 'okabe_ito' give the colour-blind-safe Okabe-Ito colours
    (distinct under colour-vision deficiency up to 8 groups), 'classic'
    tab10's; any other name is a matplotlib
    colormap. The map is listed with exactly ``n_groups`` colours, so with
    ``vmin=0, vmax=n_groups-1`` group i gets colour i."""
    import matplotlib
    from matplotlib.colors import ListedColormap

    from .theme import categorical_colors
    n = max(1, int(n_groups))
    key = {'auto': 'okabe_ito', 'okabe-ito': 'okabe_ito',
           'tab10': 'classic'}.get(str(name), str(name))
    if key in ('okabe_ito', 'classic'):
        return ListedColormap(categorical_colors(n, key), name=key)
    return matplotlib.colormaps[str(name)]


def set_default_palette(name):
    """Process-wide default for the categorical colour palette used by
    FlowSample.plot when colouring by a string column (sample_origin,
    condition, etc.). Common picks: 'auto', 'tab10', 'Set1', 'Set2',
    'Dark2', 'Paired'. Anything matplotlib's get_cmap accepts works."""
    global _DEFAULT_CATEGORICAL_PALETTE
    _DEFAULT_CATEGORICAL_PALETTE = str(name) or 'auto'


# ── Gate model & evaluator ────────────────────────────────────────────────────
#
# A gate is a JSON-friendly dict describing a region in event space. The same
# schema is consumed by the pipeline (this module) and authored by the GUI
# editor. Five kinds today:
#
#   {"kind": "threshold", "channel": str, "value": float}
#       1D one-sided:  x >= value  (a Gating-ML min-only RectangleGate)
#
#   {"kind": "interval",  "channel": str, "lo": float, "hi": float}
#       1D two-sided:  lo <= x < hi  (half-open)
#
#   {"kind": "rect",      "x_channel": str, "y_channel": str,
#                          "x0": float, "x1": float, "y0": float, "y1": float}
#       2D axis-aligned rectangle, half-open on both axes.
#
#   An open side is the sentinel -1e12 / +1e12 (JSON cannot carry +-inf; see
#   _finite_bound), not +-inf. Data never comes near 1e12, so for every
#   finite event an open side is unbounded. A non-finite event stays outside
#   every bounded comparison: +inf fails `x < 1e12`, as it fails any finite
#   upper bound, and is inside only a threshold (tests/
#   test_gate_nonfinite_events.py pins this).
#
#   {"kind": "polygon",   "x_channel": str, "y_channel": str,
#                          "vertices": [[x, y], ...]}
#       2D polygon (>=3 vertices). Membership via matplotlib.path.Path.
#
# Coordinates are in the SAME space as the data being gated (typically
# post-transform — logicle / log). FlowJo .wsp gates round-trip in this space
# already, matching apply_threshold_gates() semantics.

def depends_on_flowjo_placeholder(gates_by_id, gid, _depth=0):
    """True when gate `gid`'s population rests on a 'flowjo_ellipse'
    placeholder (a FlowJo ellipse OpenFlo could not convert): `gid` is one,
    one is among its ancestors, or, for a boolean, one of its operands rests
    on one. Such a population is unknown, not empty. Cycle-safe: booleans
    nested past gate_to_mask's own depth limit cannot be followed, so they
    count as unknown too (True), never as shown not to rest on one."""
    if _depth > 20:
        return True
    gates_by_id = gates_by_id or {}
    seen = set()
    while gid in gates_by_id and gid not in seen:
        seen.add(gid)
        g = gates_by_id[gid]
        if g.get('kind') == 'flowjo_ellipse':
            return True
        if g.get('kind') == 'boolean' and any(
                depends_on_flowjo_placeholder(gates_by_id, op, _depth + 1)
                for op in g.get('operands') or ()):
            return True
        gid = g.get('parent_id')
    return False


def rests_on_broken_boolean(gates_by_id, gid, _depth=0):
    """True when gate `gid`'s population rests on a boolean gate that cannot
    be evaluated -- one with no operands, or naming a gate not in
    `gates_by_id`, which gate_to_mask makes select nothing: `gid` is one, one
    is among its ancestors, or, for a boolean, one of its operands rests on
    one. remap_operands leaves exactly that shape when it cannot resolve an
    operand. Such a population is unknown, not empty: NOT over it admitted
    every event, and OR with it was silently the other operand alone.
    Cycle-safe; nested past gate_to_mask's depth limit it counts as unknown
    (True), as in depends_on_flowjo_placeholder."""
    if _depth > 20:
        return True
    gates_by_id = gates_by_id or {}
    seen = set()
    while gid in gates_by_id and gid not in seen:
        seen.add(gid)
        g = gates_by_id[gid]
        if g.get('kind') == 'boolean':
            ops = g.get('operands') or []
            unresolved = not ops or any(op not in gates_by_id for op in ops)
            if unresolved or any(
                    rests_on_broken_boolean(gates_by_id, op, _depth + 1)
                    for op in ops):
                return True
        gid = g.get('parent_id')
    return False


def broken_booleans(gates_by_id):
    """``(lost, built_on)``: ids of the boolean gates in `gates_by_id` that
    select nothing because they cannot be evaluated. `lost` have no operands
    or name a gate that is not there; `built_on` rest on such a gate
    (rests_on_broken_boolean)."""
    lost, built_on = [], []
    for gid, g in gates_by_id.items():
        if g.get('kind') != 'boolean':
            continue
        ops = g.get('operands') or []
        if not ops or any(op not in gates_by_id for op in ops):
            lost.append(gid)
        elif rests_on_broken_boolean(gates_by_id, gid):
            built_on.append(gid)
    return lost, built_on


def _own_unknown_reason(gates_by_id, g, columns=None):
    """Why gate `g` itself cannot be evaluated (None when it can): the
    regions gate_to_mask fails closed on because they are unknown. With
    `columns` (the sample's), a gate on a channel the sample lacks is one."""
    kind = g.get('kind')
    if g.get('scale_unknown'):
        return 'imported onto a channel of unknown scale'
    if columns is not None:
        gone = missing_gate_channels(g, columns)
        if gone:
            return ('channel ' + ', '.join(gone) + ' is not in this sample')
        if kind in ('cluster', 'category'):
            # A population of a label column the sample does not have (its
            # clustering was not restored with a session, say) selects
            # nothing, but nobody measured that 0: every cluster of a
            # reopened autosave read 0 events under a group of 20,000.
            col = g.get('channel') or ('cluster' if kind == 'cluster' else '')
            if col not in set(columns):
                return f"column '{col}' is not in this sample"
    if kind == 'flowjo_ellipse':
        return 'FlowJo ellipse not converted'
    if kind == 'boolean':
        ops = g.get('operands') or []
        if not ops or any(op not in gates_by_id for op in ops):
            return 'boolean gate lost an operand'
    if kind == 'ellipsoid':
        mean, cov = g.get('mean'), g.get('cov')
        if mean is not None and cov is not None:
            cov = np.asarray(cov, dtype=float)
            if cov.shape == (2, 2) and not _positive_definite(cov):
                return 'ellipse has no valid covariance'
    return None


def unknown_population_reason(gates_by_id, gid, _depth=0, columns=None):
    """Why gate `gid`'s population cannot be computed, or None when it can.

    A population is unknown when it, an ancestor, or (for a boolean) an
    operand rests on a gate whose region is unknown: a FlowJo ellipse
    OpenFlo could not convert, a boolean that lost an operand, or an ellipse
    whose covariance describes no ellipse. gate_to_mask makes such gates
    select nothing (fail-closed), so their count is 0 events -- but that 0 is
    not a measurement, and statistics report the population as missing (NA)
    with this reason. Cycle-safe; nested past gate_to_mask's depth limit a
    population counts as unknown too, as in depends_on_flowjo_placeholder.

    `columns`, the sample's columns, adds gates on a channel the sample
    lacks (missing_gate_channels), and cluster / category populations of a
    label column it lacks; without it they are not judged."""
    if _depth > 20:
        return 'gates nest too deep to follow'
    gates_by_id = gates_by_id or {}
    seen = set()
    while gid in gates_by_id and gid not in seen:
        seen.add(gid)
        g = gates_by_id[gid]
        why = _own_unknown_reason(gates_by_id, g, columns)
        if why:
            return why
        if g.get('kind') == 'boolean':
            for op in g.get('operands') or ():
                why = unknown_population_reason(gates_by_id, op, _depth + 1,
                                                columns=columns)
                if why:
                    return why
        gid = g.get('parent_id')
    return None


def _measured_in(df, gates_by_id, operand_ids, _depth=0):
    """Events that HAVE a measurement in every channel the given gates touch.

    Used by boolean NOT: an event with no value in the operand's channel is
    not "negative for it", it is unmeasured, and negating a mask would
    otherwise adopt it. Recurses through nested boolean operands. Channels
    that are absent or non-numeric (a cluster or category label) place no
    requirement — there is nothing to be non-finite about.

    The channel keys are read inline rather than via gating.gate_channels:
    gating imports from here, and one shared helper is not worth the cycle.
    """
    n = len(df)
    valid = np.ones(n, dtype=bool)
    if gates_by_id is None or _depth > 20:
        return valid
    for gid in operand_ids or ():
        gate = gates_by_id.get(gid)
        if gate is None:
            continue
        if gate.get('kind') == 'boolean':
            valid &= _measured_in(df, gates_by_id, gate.get('operands'),
                                  _depth + 1)
            continue
        for key in ('channel', 'x_channel', 'y_channel'):
            ch = gate.get(key)
            if not ch:
                continue
            if ch not in df.columns:
                continue
            col = df[ch]
            if not pd.api.types.is_numeric_dtype(col):
                continue
            valid &= np.isfinite(np.asarray(col.values, dtype=float))

    return valid


def _finite_bound(value, sentinel):
    """A gate bound that JSON can carry.

    Gating-ML expresses an open side as "-INF"/"INF"; float() makes that ±inf,
    and json.dump writes `Infinity`, which RFC 8259 forbids. Python's reader
    accepts it, so such a session file is valid locally and unreadable to
    JSON.parse, jq, or any other consumer. ±1e12 is the sentinel this module
    already uses for a MISSING bound, for the same stated reason.
    """
    if value is None:
        return None
    v = float(value)
    if np.isnan(v):
        return None
    if np.isinf(v):
        return float(sentinel if v < 0 else abs(sentinel))
    return v


def resolve_seed(value, default=42):
    """Turn a seed setting into an int, accepting a word or phrase as well.

    ``42`` and ``"42"`` both mean 42. Anything else — ``"pilot run 3"``,
    ``"donor A rerun"`` — is hashed to an int, so a lab can name a run
    instead of remembering a number and still get the identical answer back
    on any machine, any OS, any Python build, forever.

    That last part is why this uses blake2b rather than the obvious
    ``hash(value)``. Python randomises string hashing per process (PEP 456,
    on by default since 3.3): ``hash("pilot")`` differs between two runs of
    the same script on the same machine. A seed derived that way would look
    reproducible in one session and silently stop being reproducible in the
    next — the failure would appear as unexplained drift in results, not as
    an error. A cryptographic digest has no such freedom.

    Returns an int in [0, 2**32) — the range numpy's legacy ``RandomState``
    and every downstream library accept.
    """
    if value is None:
        return int(default) % (2 ** 32)
    if isinstance(value, bytes):
        value = value.decode('utf-8', 'replace')
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value) % (2 ** 32)
    # A whole-numbered float means the number, not a phrase: JSON has no int
    # type, so a recipe or session round-tripped through it can hand us 42.0
    # where 42 was written. Hashing that would silently give a different run
    # the same settings were supposed to reproduce.
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value) % (2 ** 32)
    text = str(value).strip()
    if not text:
        return int(default) % (2 ** 32)
    try:
        return int(text) % (2 ** 32)
    except ValueError:
        pass
    try:
        as_float = float(text)
    except ValueError:
        pass
    else:
        if as_float.is_integer():
            return int(as_float) % (2 ** 32)
    digest = hashlib.blake2b(text.encode('utf-8'), digest_size=4).digest()
    return int.from_bytes(digest, 'big')


def _snn_jaccard_graph(X, k, prune=False):
    """Shared-nearest-neighbour graph with Jaccard edge weights, as igraph.

    A plain binary kNN graph makes modularity-style objectives over-split
    uniform blobs; Jaccard weighting sharpens real communities so the
    partition tracks the populations.

    NOTE this is NOT PhenoGraph's or Seurat's construction, though it is
    closely related and was described as theirs here until it was measured.
    They compute Jaccard only between i and the members of i's own kNN list
    (phenograph.core.calc_jaccard: "between i and i's direct neighbors"), so
    the edge set is bounded by n*k. ``a @ a.T`` below instead gives an edge to
    EVERY pair sharing at least one neighbour, which is far denser: on 20k
    events at k=30, 6,106,544 edges against PhenoGraph's 297,138 — 20.6x — and
    16x slower to partition (14.9s vs 0.9s per Louvain run).

    The denser graph is not obviously worse: it scored better against planted
    truth (ARI 0.488 vs 0.378). But it under-splits, finding 4 communities
    where 8 were planted and PhenoGraph found 11, which for cytometry means
    distinct populations merged. Pruning to kNN pairs would change every
    existing Leiden and Louvain result, so it has not been changed here.

    Extracted so ``run_leiden`` and ``run_louvain`` share ONE graph: they
    differ only in how the graph is partitioned, and a difference in how it is
    BUILT would silently make their results incomparable.

    ``prune`` selects between two edge sets, measured on 20,000 events x 10
    channels, 8 planted populations, k=30 (regenerate with
    ``scripts/bench_snn_prune.py``):

                        well separated            heavily overlapping
                     clusters  ARI    homog     clusters  ARI    homog
        prune=False     6     0.9487  0.8964       4     0.4933  0.3891
        prune=True      8     0.9545  0.9220       8     0.3405  0.3973

    with Leiden taking 5.0s vs 0.8s and 9.8s vs 1.1s respectively.

    Pruned is faster, recovers the planted cluster COUNT in both cases, and has
    higher homogeneity in both — the dense graph is the one that merges
    populations, finding 4 where 8 were planted. On separable data pruned is
    better on every measure and gives nearly the same partition
    (ARI 0.97 between them).

    It is nonetheless NOT the default, for two reasons. On heavily overlapping
    data the two disagree substantially (ARI 0.53) and pruned scores worse on
    ARI, AMI and completeness — it splits populations the dense graph merges,
    and when populations genuinely overlap it is not clear that is wrong so
    much as different. And switching would silently change every Leiden and
    Louvain result anyone has already produced.
    """
    import igraph as ig
    from scipy.sparse import csr_matrix
    from sklearn.neighbors import NearestNeighbors

    # kneighbors_graph(X, k, mode='connectivity', include_self=True), built
    # the way scikit-learn builds it but with the queries spread over threads
    # (tree_kneighbors returns the same neighbours, in the same order).
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    ind = tree_kneighbors(nn, nn._fit_X, k)
    n_q = ind.shape[0]
    a = csr_matrix((np.ones(n_q * k), ind.ravel(), np.arange(0, n_q * k + 1, k)),
                   shape=(n_q, nn.n_samples_fit_))
    inter = (a @ a.T)                        # |N(i) ∩ N(j)|
    if prune:
        # "Fast mode": keep an edge only where one point is in the other's kNN
        # list, which is what PhenoGraph and Seurat do. Without this every pair
        # sharing a single neighbour gets an edge, and the graph is ~14x denser
        # (20k events, k=30: 6,106,544 edges against 424,349).
        inter = inter.multiply((a + a.T) > 0)
    inter = inter.tocoo()
    deg = np.asarray(a.sum(axis=1)).ravel()  # |N(i)| = k+1
    keep = inter.row < inter.col             # upper triangle, no self loops
    ii = inter.row[keep]
    jj = inter.col[keep]
    shared = inter.data[keep]
    union = deg[ii] + deg[jj] - shared
    w = shared / np.maximum(union, 1e-9)
    good = w > 0
    edges = list(zip(ii[good].tolist(), jj[good].tolist(), strict=True))
    g = ig.Graph(n=int(X.shape[0]), edges=edges, directed=False)
    g.es['weight'] = w[good].tolist()
    return g


def _points_in_polygon(verts, pts):
    """Point-in-polygon for gating, via flowutils' compiled ``gating_c``.

    Was ``matplotlib.path.Path.contains_points``. Swapped after testing which
    rule is actually CORRECT on the boundary, not merely faster -- a point
    exactly on an edge is neither strictly inside nor outside, so the tie-break
    is a choice, and matplotlib's choice is wrong for gating:

    * **It double-counts a shared edge.** Two gates meeting on a line both
      claim every point on it. Across a quadrant, 902 of 20 000 integer events
      landed in two populations at once, so the four quadrants summed above
      100%. flowutils assigns each such point to exactly one gate.
    * **It depends on WINDING.** The same square drawn clockwise and
      anticlockwise gives different answers on its boundary. Users draw gates
      in both directions, so the same shape could gate differently depending
      on the mouse path that produced it. flowutils is winding-invariant.
    * **Its rule is not even self-consistent**: for an axis-aligned square it
      includes three edges and excludes the bottom; for the same square rotated
      45 degrees it includes all four.

    flowutils implements the standard half-open convention -- a point on a
    "lower/left" edge is inside, one on an "upper/right" edge is outside -- so
    adjacent gates tile without gaps or overlaps, which is the property gating
    actually needs.

    Impact: on continuous float data the two agree EXACTLY (0 of 200 000
    events differed). They diverge only where events land on exact
    coordinates, i.e. integer ``$DATATYPE I`` channels (6.96% of events), and
    there matplotlib was the one double-counting. The golden baseline is
    unchanged by the swap.

    It is also 5-12x faster, which is a side benefit rather than the reason:
    at 1M events a normal gate went 172 -> 32 ms and a tight gate 163 -> 14 ms.
    A bounding-box prefilter was tried on top and REMOVED -- building the mask
    costs more than the crossing test it saves now (48 vs 32 ms).
    """
    verts = np.asarray(verts, dtype=float)
    pts = np.asarray(pts, dtype=float)
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    try:
        from flowutils import gating as _fu_gating
        return np.asarray(_fu_gating.points_in_polygon(verts, pts), dtype=bool)
    except Exception:                                        # noqa: BLE001
        # flowutils is a hard pinned dependency, so this should not happen;
        # falling back keeps gating working rather than failing the run, at
        # the cost of the boundary behaviour documented above.
        from matplotlib.path import Path as _MplPath
        log.warning("  [gate] flowutils polygon test unavailable — "
                    "falling back to matplotlib (boundary events may differ)")
        return _MplPath(verts).contains_points(pts)


# The channel keys of the geometric gate kinds: a gate naming a channel the
# data lacks cannot be evaluated on it.
_GEOMETRIC_CHANNEL_KEYS = {'threshold': ('channel',), 'interval': ('channel',),
                           'rect': ('x_channel', 'y_channel'),
                           'polygon': ('x_channel', 'y_channel'),
                           'ellipsoid': ('x_channel', 'y_channel')}


def missing_gate_channels(gate, columns):
    """The channels geometric gate `gate` names that `columns` lacks."""
    cols = set(columns)
    return [gate.get(k) for k in _GEOMETRIC_CHANNEL_KEYS.get(gate.get('kind'), ())
            if gate.get(k) and gate.get(k) not in cols]


def _channel_missing(gate, channels, n, df=None):
    """All-False mask for a geometric gate whose channel the data lacks.

    It used to admit every event, so a sample whose FL1-A was named FL1-H
    reported the gated population at 100 % and gated its children on all
    events, with only an INFO line. The region was never evaluated: fail
    closed, and statistics show the population as NA with the reason
    (unknown_population_reason, given the sample's columns)."""
    if df is not None:
        channels = [c for c in channels if c not in df.columns]
    log.warning("  [gate] %s: channel(s) %s not in this sample -- admitting "
                "no events (fail-closed)", gate.get('name') or gate.get('kind'),
                ", ".join(map(repr, channels)))
    return np.zeros(n, dtype=bool)


def gate_to_mask(gate, df, gates_by_id=None, _depth=0):
    """Evaluate one gate dict against a DataFrame. Returns a 1-D bool ndarray
    aligned to df. A gate whose channel the data lacks selects nothing
    (fail-closed) and its population is unknown (unknown_population_reason
    with `columns`); a geometric gate also logs a warning. Unknown kinds
    no-op.

    `gates_by_id` is only needed for 'boolean' gates, whose operands are
    OTHER gates in the same sample (resolved via their cumulative masks);
    `_depth` guards against operand cycles."""
    kind = gate.get('kind')
    n    = len(df)
    if gate.get('scale_unknown'):
        # Imported in FlowJo's linear units onto a channel whose display
        # scale is unknown: its coordinates could not be converted, and
        # compared as stored they name some other region. Fail closed; the
        # population is unknown (unknown_population_reason), not empty.
        log.warning("  [gate] %s: imported onto a channel of unknown scale "
                    "-- admitting no events (fail-closed)",
                    gate.get('name') or kind)
        return np.zeros(n, dtype=bool)
    if kind == 'threshold':
        ch = gate['channel']
        if ch not in df.columns:
            return _channel_missing(gate, [ch], n)
        # AT OR ABOVE the cut. A threshold is what a min-only Gating-ML
        # RectangleGate imports as and what WspWriter writes back as
        # `gating:min`; Gating-ML's rectangle is min <= x, and FlowJo counts
        # it so. `x > value` left out every event sitting on the cut:
        # integer events 995..1005 with a minimum of 1000 gave 5 per value
        # where FlowJo gives 6. It also agrees now with the interval
        # [value, open) of the same gate.
        return np.asarray(df[ch].values >= float(gate['value']))
    if kind == 'interval':
        ch = gate['channel']
        if ch not in df.columns:
            return _channel_missing(gate, [ch], n)
        vals = np.asarray(df[ch].values, dtype=float)
        # HALF-OPEN [lo, hi): the lower bound is inside, the upper bound is
        # not. Fully-open bounds silently LOST every event sitting exactly on
        # a boundary -- adjacent intervals [0,10] and [10,20] put a value of
        # exactly 10 in NEITHER, so those events vanished from both
        # populations and the percentages did not sum. Measured at 14.3% of
        # events on integer $DATATYPE I data; float data is unaffected (0 of
        # 200k), because continuous values never land exactly on a bound.
        # Also matches Gating-ML's RectangleGate convention and the polygon
        # rule in _points_in_polygon, so the same region gates identically
        # however it is expressed.
        return (vals >= float(gate['lo'])) & (vals < float(gate['hi']))
    if kind == 'rect':
        xc, yc = gate['x_channel'], gate['y_channel']
        if xc not in df.columns or yc not in df.columns:
            return _channel_missing(gate, [xc, yc], n, df)
        xs = np.asarray(df[xc].values, dtype=float)
        ys = np.asarray(df[yc].values, dtype=float)
        # Half-open on both axes -- see the reasoning on `interval` above.
        # Adjacent rectangles must partition, and a rect must agree with the
        # identical region drawn as a polygon.
        return ((xs >= float(gate['x0'])) & (xs < float(gate['x1'])) &
                (ys >= float(gate['y0'])) & (ys < float(gate['y1'])))
    if kind == 'polygon':
        xc, yc = gate['x_channel'], gate['y_channel']
        if xc not in df.columns or yc not in df.columns:
            return _channel_missing(gate, [xc, yc], n, df)
        verts = np.asarray(gate['vertices'], dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 2 or len(verts) < 3:
            log.info(f"  [gate] polygon: malformed vertices (shape={verts.shape}) — skipped")
            return np.ones(n, dtype=bool)
        pts = np.column_stack([
            np.asarray(df[xc].values, dtype=float),
            np.asarray(df[yc].values, dtype=float),
        ])
        edge_fns = _polygon_edge_fns(gate)
        if edge_fns is not None:
            # Imported from FlowJo: the edges are straight on FlowJo's axes
            # (see _flowjo_edge_axis), so vertices and events go there first.
            verts = verts.copy()
            for i, f in enumerate(edge_fns):
                if f is not None:
                    verts[:, i] = f[0](np.clip(verts[:, i], -_GATE_OPEN, _GATE_OPEN))
                    pts[:, i] = f[0](pts[:, i])
        return _points_in_polygon(verts, pts)
    if kind == 'ellipsoid':
        # Gating-ML 2.0 EllipsoidGate: an event is inside when its
        # squared Mahalanobis distance from the mean is within
        # `distance_sq`:  (p-µ)ᵀ Σ⁻¹ (p-µ) ≤ distance_sq.
        xc, yc = gate['x_channel'], gate['y_channel']
        if xc not in df.columns or yc not in df.columns:
            return _channel_missing(gate, [xc, yc], n, df)
        mean = np.asarray(gate['mean'], dtype=float)
        cov  = np.asarray(gate['cov'], dtype=float)
        dist_sq = float(gate.get('distance_sq', 4.0))
        if mean.shape != (2,) or cov.shape != (2, 2):
            log.info("  [gate] ellipsoid: malformed mean/cov — skipped")
            return np.ones(n, dtype=bool)
        # A covariance that is singular, not positive definite or not finite
        # describes no ellipse. A singular one was 'skipped' and admitted
        # every event, and so did a negative-definite one, whose quadratic
        # form is never positive. Its region is unknown: fail closed, as a
        # FlowJo ellipse that cannot be converted does (the WSP exporter
        # says so for both kinds, _wsp_export_gate).
        if not _positive_definite(cov):
            log.warning("  [gate] ellipsoid: covariance is singular or not "
                        "positive definite — admitting no events (fail-closed)")
            return np.zeros(n, dtype=bool)
        pts = np.column_stack([
            np.asarray(df[xc].values, dtype=float),
            np.asarray(df[yc].values, dtype=float),
        ])
        try:
            # flowutils' compiled gating_c, the same extension backing
            # _points_in_polygon. Agrees with the hand-rolled einsum quadratic
            # form on real data — 0 of 1,000,000 random events differ — and is
            # ~1.4x faster: re-measured 2026-09-10, 42 -> 31 ms at 1M events.
            #
            # They are NOT bit-identical on the boundary itself, as this
            # comment used to claim: of 512 points constructed to sit exactly
            # on the ellipse, the two disagree on 92. Those points compute d^2
            # in the range 3.99999999999999867 .. 4.00000000000000178, i.e.
            # they straddle dist_sq by one or two ulp, and the two orderings of
            # the same arithmetic break the tie differently.
            #
            # That is harmless HERE, and it is worth being clear why, because
            # the identical situation was NOT harmless for rect and interval
            # gates. An ellipse contour is not lattice-aligned: integer
            # $DATATYPE I events land exactly on an interval or rectangle
            # bound constantly (14.3% of events, which is why those became
            # half-open), but on 200,000 integer events the two ellipsoid
            # paths disagreed 0 times, and a 13x13 integer lattice holds no
            # point at exactly d^2 == dist_sq at all. So the fallback below
            # cannot silently move a real event across this boundary. It takes the
            # covariance itself and inverts internally, so the singular case
            # still surfaces as LinAlgError and is handled below exactly as
            # before. Using one library for both polygon and ellipsoid keeps
            # gate geometry semantics in a single place.
            from flowutils import gating as _fu_gating
            return np.asarray(
                _fu_gating.points_in_ellipsoid(cov, mean, dist_sq, pts),
                dtype=bool)
        except np.linalg.LinAlgError:
            log.warning("  [gate] ellipsoid: singular covariance — admitting "
                        "no events (fail-closed)")
            return np.zeros(n, dtype=bool)
        except Exception:                                    # noqa: BLE001
            # Never lose a gate to an unexpected backend failure: fall back to
            # the quadratic form, which needs only numpy.
            inv = np.linalg.inv(cov)
            d = pts - mean
            return np.einsum('ij,jk,ik->i', d, inv, d) <= dist_sq
    if kind == 'cluster':
        # Membership in one clustering label. Unlike the geometric gates,
        # a missing column means the population is undefined for this
        # sample, so it selects NOTHING (empty) rather than no-op all-True
        # — an unclustered sample shouldn't masquerade as "all events".
        ch = gate.get('channel', 'cluster')
        if ch not in df.columns:
            log.info(f"  [gate] cluster: column '{ch}' not in data — empty")
            return np.zeros(n, dtype=bool)
        return np.asarray(df[ch].values == gate.get('cluster_id'))
    if kind == 'category':
        # Membership in a categorical label column (e.g. cell-cycle phase
        # in a 'cell_cycle' column). Like 'cluster', a missing column means
        # the population is undefined → selects nothing.
        ch = gate.get('channel')
        if not ch or ch not in df.columns:
            log.info(f"  [gate] category: column '{ch}' not in data — empty")
            return np.zeros(n, dtype=bool)
        return np.asarray(df[ch].values == gate.get('value'))
    if kind == 'flowjo_ellipse':
        # An ellipse drawn in FlowJo on an axis OpenFlo cannot convert (see
        # _wsp_ellipse_gate). Its region is unknown, so it selects nothing:
        # the unknown-kind default (every event) would hand each population
        # under it every event the ellipse excluded.
        log.info("  [gate] flowjo_ellipse: FlowJo ellipse not converted — empty")
        return np.zeros(n, dtype=bool)
    if kind == 'boolean':
        # Combine OTHER gates' cumulative masks. op ∈ {and, or, not}; 'not'
        # negates the OR of its operands (so a single operand → plain NOT).
        op = gate.get('op', 'and')
        operands = gate.get('operands', []) or []
        # A boolean gate that cannot resolve its operands must NOT admit every
        # event. Returning all-True made a "NOT X" population the ENTIRE
        # sample — measured, `NOT (CD3 > 2.0)` kept 20,000 events where the
        # correct answer was 10,002, so every CD3-positive event was admitted
        # into the CD3-negative population. Fail CLOSED, matching both the
        # 'cluster'/'category' missing-column branches above and the
        # fail-closed exception handler in apply_region_gates: an empty
        # population is loud, a superset reads as success.
        if gates_by_id is None or not operands or _depth > 20:
            log.warning("  [gate] boolean: operands could not be resolved — "
                        "admitting no events (fail-closed)")
            return np.zeros(n, dtype=bool)
        # ONE missing operand is enough to fail closed. Dropping it and
        # combining the rest changes the population without a word: measured,
        # deleting CD4+ turned AND(CD3+, CD4+) from 4,983 events into CD3+
        # alone, 10,044; a NOT over two operands becomes the wider NOT of one,
        # and an OR quietly narrows.
        missing = [gid for gid in operands if gid not in gates_by_id]
        if missing:
            log.warning(f"  [gate] boolean: operand(s) {missing} no longer "
                        "exist — admitting no events (fail-closed)")
            return np.zeros(n, dtype=bool)
        # An operand resting on a FlowJo ellipse OpenFlo could not convert
        # (the ellipse, a gate under it, or a boolean built on one) has an
        # unknown region, and so does every combination of it. The ellipse
        # selects nothing, which keeps it and its subpopulations empty, but
        # its complement is not "everything" nor its union "the other
        # operand": measured, NOT(ellipse) admitted all 20,000 events and
        # OR(CD3+, ellipse) all of CD3+.
        unknown = [gid for gid in operands
                   if depends_on_flowjo_placeholder(gates_by_id, gid)]
        if unknown:
            log.warning(f"  [gate] boolean: operand(s) {unknown} rest on a "
                        "FlowJo ellipse that was not converted (or nest too "
                        "deep to follow) — admitting no events (fail-closed)")
            return np.zeros(n, dtype=bool)
        # Likewise an operand resting on a boolean that cannot be evaluated
        # (no operands, or one deleted): that boolean fails closed, and its
        # emptiness is unknown, not a measured zero. Measured: with B empty,
        # NOT(B) admitted all 10,000 events, OR(CD3+, B) was CD3+ alone and
        # NOT(CD3+, B) was NOT CD3+.
        broken = [gid for gid in operands
                  if rests_on_broken_boolean(gates_by_id, gid)]
        if broken:
            log.warning(f"  [gate] boolean: operand(s) {broken} rest on a "
                        "boolean gate that cannot be evaluated — admitting "
                        "no events (fail-closed)")
            return np.zeros(n, dtype=bool)
        # And an operand resting on an ellipse whose covariance describes no
        # ellipse: that ellipse selects nothing for the same reason, so NOT
        # over it admitted every event (unknown_population_reason).
        unknown = [gid for gid in operands
                   if unknown_population_reason(gates_by_id, gid,
                                                columns=df.columns)]
        if unknown:
            log.warning(f"  [gate] boolean: operand(s) {unknown} rest on a "
                        "gate whose region is unknown — admitting no events "
                        "(fail-closed)")
            return np.zeros(n, dtype=bool)
        masks = [cumulative_gate_mask(gates_by_id, gid, df, _depth + 1)
                 for gid in operands]
        out = masks[0].copy()
        if op == 'or':
            for m in masks[1:]:
                out |= m
            return out
        if op == 'not':
            for m in masks[1:]:
                out |= m
            # Negation turns "could not be measured" into "belongs here". An
            # event whose value is NaN fails `> threshold`, so it is correctly
            # OUTSIDE the positive gate — and then the complement swept it in.
            # Measured: 2,000 unmeasurable events made up 20% of a CD3-negative
            # population, and because the two gates still summed to the sample
            # total the result looked self-consistent. Non-finite is not
            # negative, so those events belong to neither side.
            return ~out & _measured_in(df, gates_by_id, operands)
        for m in masks[1:]:            # default: and
            out &= m
        return out
    if kind == 'autoclean':
        # Recipe gate (no coordinates): the AND of every enabled cleaning
        # method, recomputed from THIS df. See autoclean_keep_mask.
        return autoclean_keep_mask(gate, df)
    if kind == 'group':
        # Pure organisational container (e.g. a 'Phenograph (N)' folder over
        # cluster populations): no geometry, never filters — children carry
        # the actual masks.
        return np.ones(n, dtype=bool)
    log.info(f"  [gate] unknown kind {kind!r} — skipped")
    return np.ones(n, dtype=bool)


# ── Auto-clean (acquisition-cleaning) gate ─────────────────────────────────
#
# An 'autoclean' gate stores a RECIPE — a list of cleaning METHODS — not
# coordinates. Each method recomputes its keep-mask from whatever sample the
# gate is evaluated against, so copying the gate to other samples re-runs the
# calculations rather than reusing one sample's geometry (bubbles/debris/clogs
# sit in different places per sample). The gate's mask is the AND of every
# ENABLED method's keep-mask: events clean of ALL selected anomaly types.

AUTOCLEAN_METHODS = [
    {'key': 'debris',    'label': 'Debris (size: beads; valley opt-in)', 'params': {'mode': 'bead', 'bead_um': 8.0, 'min_um': 4.0}},
    {'key': 'viability', 'label': 'Dead cells (viability dye)',    'params': {}},
    {'key': 'doublets',  'label': 'Doublets (FSC-A/FSC-H)',        'params': {'tol': 0.25}},
    {'key': 'margin',    'label': 'Margin (saturation)',           'params': {'margin_frac': 0.01}},
    {'key': 'flow_rate', 'label': 'Flow rate (bubbles/clogs)',     'params': {'n_bins': 200, 'flow_rate_threshold': 5.0}},
    {'key': 'drift',     'label': 'Signal drift',                  'params': {'n_bins': 200, 'threshold': 5}},
]


def default_autoclean_methods():
    """A fresh, all-enabled copy of the standard cleaning recipe."""
    return [{'key': m['key'], 'label': m['label'], 'enabled': True,
             'params': copy.deepcopy(m['params'])} for m in AUTOCLEAN_METHODS]


def autoclean_methods_signature(gate):
    """A hashable signature of an autoclean gate's recipe (each method's key,
    enabled flag, and sorted params). Two gates with the same signature produce
    the same mask on the same data — used as a mask-cache key by the GUI."""
    out = []
    for m in gate.get('methods') or []:
        params = m.get('params') or {}
        out.append((m.get('key'), bool(m.get('enabled', True)),
                    tuple(sorted(params.items()))))
    return tuple(out)


def _autoclean_find_scatter(df, prefix, suffix='-A'):
    pu, su = prefix.upper(), suffix.upper()
    for c in df.columns:
        cu = c.upper()
        if cu.startswith(pu) and cu.endswith(su):
            return c
    for c in df.columns:
        if c.upper().startswith(pu):
            return c
    return None


# Dye name tokens we recognise as viability / live-dead stains (lowercased
# substrings, matched against antibody label first, then detector name).
# Dead cells take up the dye and read HIGH; live cells exclude it and read low.
# Overlaps with DNA_DYES (7-AAD, PI, DAPI, SYTOX, TO-PRO double as viability).
VIABILITY_DYES = (
    'live/dead', 'livedead', 'live-dead', 'l/d', 'viability', 'viadye',
    'viable', 'zombie', 'ghost dye', 'ghost', 'fixable viability',
    'fixable viable', 'fixable live', 'fvs', 'fvd', 'efluor 506',
    'efluor 780', 'ef506', 'ef780', 'aqua', 'near-ir', 'sytox', 'to-pro',
    'topro', '7-aad', '7aad', 'propidium', 'dapi', 'pi',
)


def find_viability_channel(columns, channel_labels=None):
    """Best-guess viability / live-dead detector among ``columns``, or None.

    Matches known viability-dye tokens against each channel's antibody label
    first (from ``channel_labels``, a ``{detector: label}`` dict), then its
    detector name. Prefers an Area (``-A``) channel. The short tokens 'pi' /
    'l/d' only match as whole words so they don't fire inside longer names."""
    labels = channel_labels or {}
    cols = list(columns)

    def matches(text):
        t = str(text).lower()
        for dye in VIABILITY_DYES:
            if dye in ('pi', 'l/d'):
                if re.search(r'(?<![a-z0-9/])' + re.escape(dye) + r'(?![a-z0-9])', t):
                    return True
            elif dye in t:
                return True
        return False

    candidates = [det for det in cols
                  if matches(labels.get(det, det)) or matches(det)]
    if not candidates:
        return None
    for c in candidates:
        if str(c).upper().endswith('-A'):
            return c
    return candidates[0]


def _debris_mode(params):
    """The debris method's mode as the user meant it: trimmed and lower case,
    'bead' when unset. The parameters dialog stored it as typed, so 'Valley'
    must still mean the valley cut. Anything else is not a mode and cuts
    nothing (see :func:`autoclean_method_diagnostic`)."""
    return str((params or {}).get('mode') or 'bead').strip().lower()


def _autoclean_debris_mask(df, params):
    """Drop debris, following the standard manual gating hierarchy as closely
    as the mode allows.

    Resolution order (first applicable wins):
      1. **manual** — an explicit ``min_fsc`` FSC-A floor is used verbatim.
      2. **bead-calibrated absolute size** (the default, ``mode='bead'``) —
         when a bead anchor ``bead_fsc`` (the median FSC-A of size-calibration
         beads of diameter ``bead_um`` µm) is present and a target ``min_um``
         is set, keep events whose implied size is ≥ ``min_um`` µm, i.e.
         ``FSC-A >= min_um * bead_fsc / bead_um``. A pure 1-D size ruler — the
         most reproducible cut and, with a sub-cell ``min_um`` (≈4 µm), the most
         conservative (it removes only genuine sub-cellular fragments, never
         small-but-real cells such as lymphocytes).
      3. **2-D scatter gate** (``mode='valley'`` only — opt-in) — emulates the
         manual **FSC-A × SSC-A** debris polygon: an event is debris only when
         it is low on BOTH FSC-A AND SSC-A (the bottom-left cloud), each
         boundary being that channel's density valley below its median. This
         keeps low-FSC / high-SSC granular cells (granulocytes, etc.) that a
         1-D FSC cut would wrongly discard. Degrades to a 1-D FSC valley when
         there's no usable SSC-A. Its limits are in :func:`_debris_fsc_valley`.
    No-op when there's no FSC-A column or no usable cutoff. In particular
    ``mode='bead'`` (the default) with no bead anchor cuts NOTHING: it used to
    fall back to the valley cut, and the valley cut cannot tell debris from
    small real cells by scatter alone (it took all the platelets of a
    platelet assay, all the bacteria of a bacterial-load assay, and the
    lymphocytes under large tumour cells), so it runs only when asked for."""
    n = len(df)
    fsc = _autoclean_find_scatter(df, 'FSC', '-A')
    if fsc is None:
        return np.ones(n, dtype=bool)
    vals = np.asarray(df[fsc].values, dtype=float)
    fin  = np.isfinite(vals)
    params = params or {}
    # (1) explicit FSC-A floor — deterministic, applies regardless of N.
    manual = params.get('min_fsc')
    if manual is not None:
        keep = fin & (vals >= float(manual))
        # A FROZEN valley gate pins the SSC-granular threshold too: add back the
        # low-FSC / high-SSC granulocytes the 2-D valley gate rescued, so a
        # frozen gate replays the full 2-D cut instead of a lossy 1-D floor.
        gthr = params.get('min_ssc_granular')
        if gthr is not None:
            ssc = params.get('ssc_channel') or _autoclean_find_scatter(
                df, 'SSC', '-A')
            if ssc is not None and ssc in df.columns:
                svals = np.asarray(df[ssc].values, dtype=float)
                low_fsc = fin & (vals < float(manual))
                granular = (low_fsc & np.isfinite(svals)
                            & (svals >= float(gthr)))
                keep = keep | granular
        return keep
    # (2) bead-calibrated absolute size — deterministic, applies regardless of N.
    mode     = _debris_mode(params)
    bead_fsc = params.get('bead_fsc')
    min_um   = params.get('min_um')
    bead_um  = params.get('bead_um', 8.0)
    if (mode == 'bead' and bead_fsc and min_um
            and float(bead_fsc) > 0 and float(bead_um) > 0):
        thr = float(min_um) * float(bead_fsc) / float(bead_um)
        return fin & (vals >= thr)
    # (3) the 2-D scatter gate, only when asked for (needs enough events to
    # estimate). Uses the strict bimodal valley (not Otsu) so a UNIMODAL
    # FSC-A is never bisected — only a separate low-FSC mode below the median
    # is cut.
    finite = vals[fin]
    if mode != 'valley' or finite.size < 50:
        return np.ones(n, dtype=bool)
    fthr = _debris_fsc_cut(finite)
    if fthr is None:
        return np.ones(n, dtype=bool)
    low_fsc = fin & (vals < float(fthr))
    ssc = _autoclean_find_scatter(df, 'SSC', '-A')
    if ssc is not None and ssc != fsc and params.get('use_ssc', True):
        # Rescue granular cells: among the small (low-FSC) events, is there a
        # genuinely separate HIGH-SSC subpopulation (granulocytes)? Look for a
        # bimodal SSC split WITHIN those events only — a global SSC valley
        # tends to split off the high-SSC tail, not the debris floor. Only when
        # such a split exists do we keep the high-SSC side (= the manual
        # polygon's upper-left lobe); otherwise the small events are all debris.
        svals = np.asarray(df[ssc].values, dtype=float)
        ss_lo = svals[low_fsc & np.isfinite(svals)]
        sthr  = _bimodal_valley(ss_lo) if ss_lo.size >= 50 else None
        if sthr is not None:
            granular = low_fsc & np.isfinite(svals) & (svals >= float(sthr))
            return ~(low_fsc & ~granular)      # debris = small AND non-granular
    return ~low_fsc                            # 1-D FSC fallback (no granular lobe)


def _resolution_bins(v, lo, hi, bins):
    """Bin count capped at the number of DISTINCT values in range.

    A channel cannot fill more bins than it has values. An integer detector
    ($DATATYPE I) whose bulk spans a few tens of ADC steps, spread over 256
    bins, becomes a COMB: most bins are structurally empty, the smoothing —
    which is measured in BINS, not data units — cannot close gaps several bins
    wide, and peak finding reads two adjacent teeth as two modes with a zero
    between them. That empty bin trivially satisfies any "is the valley deep
    enough" test.

    Measured: an all-live, unimodal channel of 43 integer levels produced a
    valley at its own median, and the auto-clean viability filter deleted 46%
    of the sample. The identical data unrounded returned no valley at all.
    """
    in_range = v[(v >= lo) & (v <= hi)]
    distinct = int(np.unique(in_range).size)
    return int(max(8, min(int(bins), distinct)))


# Fewest events a histogram bin may average in _bimodal_valley.
_VALLEY_MIN_PER_BIN = 10


def _bimodal_valley(values, bins=256, smooth=2.0, lowest=False):
    """The valley between the two most prominent peaks of a smoothed
    histogram, but ONLY when the data is genuinely bimodal. Returns None for
    unimodal data — unlike :func:`auto_threshold`, it does NOT fall back to
    Otsu, so a caller can treat 'no valley' as 'do not split'.

    With ``lowest`` it is the first genuine valley scanning up from the low
    end instead: the valley above the LOWEST mode, whatever else is taller.
    The debris cut needs that (see :func:`_debris_fsc_cut`).

    Three things kept this from returning None for unimodal data, or from
    finding a real second mode:

    * A bin must average ``_VALLEY_MIN_PER_BIN`` events. The bins were
      capped only by the distinct values, so a small continuous sample got
      ~n bins of ~1 event each, and 2-bin smoothing left noise peaks above
      the 5% prominence bar. Single normal samples of 200 events were
      split in 43 of 100 seeds, uniform ones of 1000 in 71; now 0 and 2.
      The granular rescue calls this on the low-FSC subset (>= 50 events),
      where a false split kept debris as "granulocytes". From 2560 events
      the cap does not bind.
    * The pair of peaks is the two most PROMINENT, not the two tallest.
      Below ~50k events, noise bumps on the major mode pass the 5% bar
      and are taller than a minor mode, and the valley between two of
      them is shallow. A 70/30 mixture of modes 4 SD apart was split in 39
      of 100 seeds at 5000 events; now 100.
    * The smoothed histogram has a zero on each side, so a mode at the edge
      of the 0.5-99.5 percentile window (debris piled at the origin, a
      flat debris plateau) is a peak. ``find_peaks`` cannot return an end
      point, and an edge plateau had almost no prominence.

    Measured over 100 seeds each: unimodal normal, uniform, log-normal and
    exponential samples of 60 to 5000 events are split in at most 3 (were
    up to 92); 2-mode mixtures (minor mode 10-50%, 3.5-5 SD apart) are
    split between the modes in 57-100 of 100 from 500 events (were 0-80).
    Modes 3 SD apart are not split, as before. At 200 events and fewer a
    real minor mode is found in 0-17 of 100: at this smoothing the data
    cannot tell it from noise, and 'do not split' is the safe answer. (The
    old code found it in 1-83 of 100 there, but also split 11-90 of 100
    unimodal samples of those sizes.)"""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 50:
        return None
    lo, hi = np.percentile(v, [0.5, 99.5])
    if hi <= lo:
        return None
    bins = max(8, min(_resolution_bins(v, lo, hi, bins),
                      v.size // _VALLEY_MIN_PER_BIN))
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    sm = gaussian_filter1d(hist.astype(float), smooth)
    if sm.max() <= 0:
        return None
    sm = np.r_[0.0, sm, 0.0]              # sm[i] is the bin at centers[i - 1]
    peaks, props = find_peaks(sm, prominence=sm.max() * 0.05)
    if peaks.size < 2:
        return None
    if lowest:
        pairs = list(zip(peaks[:-1], peaks[1:], strict=True))
    else:
        top = peaks[np.argsort(props['prominences'])[::-1][:2]]
        pairs = [(min(top), max(top))]
    for a, b in pairs:
        valley = a + int(np.argmin(sm[a:b + 1]))
        # Demand a genuine separation: the valley must dip to ≤ half the
        # SHORTER of the two peaks. Sampling-noise bumps on a single mode
        # leave a shallow 'valley' near the peak height — not a split.
        if sm[valley] <= 0.5 * float(min(sm[a], sm[b])):
            return float(centers[valley - 1])
    return None


# The lowest FSC-A mode is debris only when its events' median FSC-A is at
# most this fraction of the larger events' median. See _debris_fsc_valley.
_DEBRIS_MAX_FSC_RATIO = 0.25


def _debris_fsc_valley(finite):
    """``(cut, why_not)`` for the 2-D scatter gate's FSC-A cut. ``cut`` is
    the valley above the LOWEST FSC-A mode, or None (do not cut) with
    ``why_not`` 'unimodal' (no such valley below the median) or
    'cell-sized' (the lowest mode is too large to be debris). ``finite``
    is the finite FSC-A values, at least 50.

    The lowest mode, not the two most prominent: doublets and large cells
    are modes too, and when the second most prominent sat ABOVE the cells
    the valley landed above the median and nothing was cut. With 1000
    debris spread over U(2k, 15k), 8000 cells and 1000 doublets, the cut
    fell between cells and doublets, 0 of 1000 debris were removed and the
    diagnostic called FSC-A unimodal.

    Being the lowest mode below the median does not make a population
    debris. In lysed whole blood with no debris (lymphocytes 30%, monocytes
    8%, granulocytes 62%) the lowest mode is the lymphocytes, and their SSC
    is unimodal, so the granular rescue cannot save them: 99.9% of them
    were cut (the two-tallest rule before it cut 65%). Debris is small:
    the cut is made only when the lowest mode's median FSC-A is at most
    ``_DEBRIS_MAX_FSC_RATIO`` of the median of the events above it.
    Measured over 20 seeds a case, on linear FSC-A: a debris mode (piled at
    the origin, uniform, or a peak at 8% of the cells' FSC-A; 1-45% of
    events; under PBMC or whole blood) gives 0.036-0.12; the lymphocytes
    under granulocytes give 0.48. On transformed scatter the ratio is not
    a size ratio and errs toward 0.5 and above, toward not cutting. When
    the evidence is not there, not cutting is the safe answer: a debris
    mode too weak to form the lowest valley (50 events in 10000 whole
    blood cells) is left in, and so is a debris majority (the valley is
    then above the median).

    KNOWN LIMITS — why this runs only when ``mode='valley'`` is chosen.
    Scatter alone cannot tell debris from small real cells; this rule was
    kept over the two-tallest rule (28641c3) and the lowest-valley rule
    without the size test (d883512) because, over two independent attack
    batteries, it removed the fewest real cells (mean 12.7% against 13.9%
    and 16.7% over 72 class-scenarios; lymphocytes in whole blood with 0
    to 100 debris events: none, against 65-95% and 90-99.9%), at the price
    of leaving more debris in (53% against 88% and 25%). Measured, 20
    seeds a case:

    * It removes small real cells under a much larger population:
      lymphocytes (45k FSC-A) under tumour cells, macrophages or other
      large cells whose median FSC-A is 190k or more are all cut (50-65% at
      180k, none at 150k).
    * It removes small particles that are the point of the assay: all the
      platelets of an unlysed platelet sample, all the free bacteria of a
      bacterial-load sample.
    * It keeps debris that piles up at a high instrument FSC threshold, or
      overlaps the cells: of debris piled at a 12k threshold under PBMC
      (60k) 0-9% is cut, of overlapping debris 0-27%.
    * Linear FSC-A only: on log-amplified or arcsinh-scaled scatter it cuts
      nothing.

    A bead anchor (``mode='bead'``) or a manual ``min_fsc`` has none of
    these limits."""
    thr = _bimodal_valley(finite, lowest=True)
    if thr is None or thr >= float(np.median(finite)):
        return None, 'unimodal'
    small = float(np.median(finite[finite < thr]))
    rest = float(np.median(finite[finite >= thr]))
    if not (rest > 0 and small <= _DEBRIS_MAX_FSC_RATIO * rest):
        return None, 'cell-sized'
    return thr, ''


def _debris_fsc_cut(finite):
    """The 2-D scatter gate's FSC-A cut, or None: see
    :func:`_debris_fsc_valley`."""
    return _debris_fsc_valley(finite)[0]


def _autoclean_viability_mask(df, params):
    """Keep LIVE cells — drop the high-signal dead population on a viability
    dye. The detector is ``params['channel']`` when set (and present), else
    auto-detected by dye-name tokens among the columns. A manual
    ``max_signal`` ceiling is used verbatim; otherwise the live/dead split is
    the density valley, applied only when a dead mode sits ABOVE the median
    (so an all-live, unimodal sample is never bisected). No-op when no
    viability channel is found or there's too little data."""
    n = len(df)
    params = params or {}
    ch = params.get('channel')
    if not ch or ch not in df.columns:
        ch = find_viability_channel(list(df.columns))   # labels unavailable here
    if not ch or ch not in df.columns:
        return np.ones(n, dtype=bool)
    vals   = np.asarray(df[ch].values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size < 50:
        return np.ones(n, dtype=bool)
    manual = params.get('max_signal')
    if manual is not None:
        return np.isfinite(vals) & (vals <= float(manual))
    # Require a genuine bimodal live/dead split (no Otsu fallback) and the
    # dead mode must sit ABOVE the median — so an all-live, unimodal sample
    # is never bisected.
    thr = _bimodal_valley(finite)
    if thr is None or thr <= float(np.median(finite)):
        return np.ones(n, dtype=bool)
    return np.isfinite(vals) & (vals <= float(thr))


# How many robust SDs of the FSC-A/FSC-H ratio the singlet window may span.
# See _autoclean_doublets_mask: generous on purpose.
_DOUBLET_SIGMA_K = 10.0


def _autoclean_doublets_mask(df, params, ref_mask=None):
    """Keep singlets via the FSC-A/FSC-H ratio (within ±tol of the median
    ratio). No-op if FSC-A or FSC-H is missing.

    ``ref_mask`` (optional) restricts which events the *median* ratio is taken
    over — the standard gating hierarchy removes debris BEFORE the singlet gate,
    so the window is centred on cell-sized events, not debris. The keep
    predicate is still applied to every event (AND-of-methods semantics); only
    the reference population for the median changes."""
    n   = len(df)
    tol = float(params.get('tol', 0.25))
    if tol <= 0:
        return np.ones(n, dtype=bool)
    fa = _autoclean_find_scatter(df, 'FSC', '-A')
    fh = _autoclean_find_scatter(df, 'FSC', '-H')
    if fa is None or fh is None or fa == fh:
        return np.ones(n, dtype=bool)
    a = np.asarray(df[fa].values, dtype=float)
    h = np.asarray(df[fh].values, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(h > 0, a / h, np.nan)
    valid = np.isfinite(ratio)
    if not valid.any():
        return np.ones(n, dtype=bool)
    ref = valid
    if ref_mask is not None:
        ref = valid & np.asarray(ref_mask, dtype=bool)
        if not ref.any():                    # debris removed everything → fall back
            ref = valid
    med = float(np.nanmedian(ratio[ref]))
    if not med > 0:
        # A non-positive median ratio inverts the multiplicative window
        # (lo > hi), which matches nothing and would delete the entire
        # sample. Doublets cannot be identified from such a ratio at all.
        log.warning("doublet filter skipped: the median FSC-A/FSC-H ratio is "
                    "%.4g, so no acceptance window can be formed.", med)
        return np.ones(n, dtype=bool)

    # `tol` is a FRACTION OF THE MEDIAN, but how far the singlet ratio
    # actually scatters depends on the scale the scatter channels are on.
    # After an arcsinh or logicle bake the ratio's median barely moves while
    # its spread collapses — measured, a CV of 14.5% became 1.7% — so a fixed
    # ±25% window swallowed the doublet population whole: 4,000 of 4,000
    # doublets retained, where the same events on linear scatter lost all
    # 4,000. The filter silently became a no-op on transformed data.
    #
    # Same shape as the median-ratio bug in filter_doublets: a window derived
    # from a scale-dependent quantity. The band is now also bounded by the
    # ratio's OWN robust spread, which is what auto_singlet_gate already
    # uses. `tol` keeps its meaning as the widest the window may be; the
    # dispersion can only narrow it. With no measurable spread there is
    # nothing to narrow by, so `tol` stands alone.
    #
    # k is deliberately generous. At 5 robust SDs the bound also trimmed
    # 0.2% of genuine singlets from the golden sample — the golden baseline
    # caught it — so it is set where a well-behaved linear panel is left
    # exactly as it was while a collapsed spread (8x tighter after arcsinh)
    # is still caught with room to spare.
    dev = np.abs(ratio[ref] - med)
    sigma = MAD_TO_SD * float(np.nanmedian(dev))
    half = tol * med
    if sigma > 0:
        half = min(half, _DOUBLET_SIGMA_K * sigma)
    lo, hi = med - half, med + half
    return valid & (ratio >= lo) & (ratio <= hi)


def autoclean_debris_threshold(df, params):
    """The scalar FSC-A keep-threshold the debris method resolves to on ``df``
    (manual ``min_fsc`` → bead absolute size → density valley when
    ``mode='valley'``), or None when it wouldn't cut — as for bead mode with
    no bead anchor. Used to FREEZE an auto cut to a fixed value when
    copying."""
    params = params or {}
    fsc = _autoclean_find_scatter(df, 'FSC', '-A')
    if fsc is None:
        return None
    manual = params.get('min_fsc')
    if manual is not None:
        return float(manual)
    mode = _debris_mode(params)
    bead_fsc, min_um = params.get('bead_fsc'), params.get('min_um')
    bead_um = params.get('bead_um', 8.0)
    if (mode == 'bead' and bead_fsc and min_um
            and float(bead_fsc) > 0 and float(bead_um) > 0):
        return float(min_um) * float(bead_fsc) / float(bead_um)
    if mode != 'valley':
        return None
    vals = np.asarray(df[fsc].values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size < 50:
        return None
    thr = _debris_fsc_cut(finite)
    return None if thr is None else float(thr)


def autoclean_debris_freeze(df, params):
    """Frozen params that reproduce this sample's debris cut on any target:
    ``{'min_fsc': ...}`` for a manual / bead / 1-D-valley cut, plus
    ``{'min_ssc_granular', 'ssc_channel'}`` when the valley path found a
    high-SSC granulocyte lobe — so a frozen valley gate replays the full 2-D
    rescue instead of collapsing to a lossy 1-D floor. ``{}`` if it wouldn't cut."""
    params = params or {}
    thr = autoclean_debris_threshold(df, params)
    if thr is None:
        return {}
    out = {'min_fsc': float(thr)}
    # A manual floor or a bead absolute-size cut is a genuine 1-D threshold —
    # only the 2-D valley cut carries an SSC granulocyte rescue to pin.
    if params.get('min_fsc') is not None:
        return out
    bead_fsc = params.get('bead_fsc')
    if (_debris_mode(params) == 'bead' and bead_fsc
            and params.get('min_um') and float(bead_fsc) > 0):
        return out
    fsc = _autoclean_find_scatter(df, 'FSC', '-A')
    ssc = _autoclean_find_scatter(df, 'SSC', '-A')
    if (fsc is None or ssc is None or ssc == fsc
            or not params.get('use_ssc', True)):
        return out
    vals = np.asarray(df[fsc].values, dtype=float)
    low_fsc = np.isfinite(vals) & (vals < float(thr))
    svals = np.asarray(df[ssc].values, dtype=float)
    ss_lo = svals[low_fsc & np.isfinite(svals)]
    sthr = _bimodal_valley(ss_lo) if ss_lo.size >= 50 else None
    if sthr is not None:
        out['min_ssc_granular'] = float(sthr)
        out['ssc_channel'] = ssc
    return out


def autoclean_viability_threshold(df, params):
    """The scalar dye-signal ceiling the viability method resolves to (events
    above it are dead), or None when it wouldn't cut. For freezing on copy."""
    params = params or {}
    ch = params.get('channel')
    if not ch or ch not in df.columns:
        ch = find_viability_channel(list(df.columns))
    if not ch or ch not in df.columns:
        return None
    manual = params.get('max_signal')
    if manual is not None:
        return float(manual)
    vals = np.asarray(df[ch].values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size < 50:
        return None
    thr = _bimodal_valley(finite)
    if thr is None or thr <= float(np.median(finite)):
        return None
    return float(thr)


def freeze_autoclean_gate(gate, df, channel_labels=None):
    """Return a deep copy of an ``autoclean`` ``gate`` with its auto-derived
    cuts PINNED to the values computed from ``df`` (one specific sample), so the
    copy applies identical thresholds everywhere instead of recomputing per
    sample. Debris → fixed ``min_fsc`` (plus ``min_ssc_granular`` + ``ssc_channel``
    when the valley path has a 2-D granulocyte rescue, so freezing keeps those
    cells instead of collapsing to a lossy 1-D floor); viability → resolved
    ``channel`` + fixed ``max_signal``. Methods without a single threshold
    (doublets, margin, flow-rate, drift) are left untouched (still per-sample).
    Non-autoclean gates are returned unchanged."""
    g = copy.deepcopy(gate)
    if g.get('kind') != 'autoclean':
        return g
    for m in g.get('methods') or []:
        key = m.get('key')
        mp = m.setdefault('params', {})
        if key == 'debris':
            mp.update(autoclean_debris_freeze(df, mp))
        elif key == 'viability':
            ch = mp.get('channel') or find_viability_channel(
                list(df.columns), channel_labels)
            if ch:
                mp['channel'] = ch
            thr = autoclean_viability_threshold(df, mp)
            if thr is not None:
                mp['max_signal'] = float(thr)
    return g


def autoclean_method_diagnostic(key, df, params, channel_labels=None):
    """A short, human reason a cleaning method removed **nothing** — or None
    when it's operating normally. Lets the GUI explain a silent 0-drop ("no
    viability dye detected", "FSC-A is unimodal — no debris mode") instead of
    leaving the user guessing whether the method is broken."""
    params = params or {}
    if len(df) == 0:
        return "no events"

    if key == 'debris':
        fsc = _autoclean_find_scatter(df, 'FSC', '-A')
        if fsc is None:
            return "no FSC-A channel"
        if params.get('min_fsc') is not None:
            return "nothing below the manual min_fsc"
        mode = _debris_mode(params)
        has_bead = bool(params.get('bead_fsc')) and bool(params.get('min_um'))
        if mode == 'bead' and has_bead:
            return None        # deterministic bead cut — if it dropped 0, that's real
        if mode == 'bead':
            return ("no bead size anchor — nothing cut. Load beads, set a "
                    "manual FSC floor, or switch to 'valley' to cut by "
                    "scatter")
        if mode != 'valley':
            return (f"unknown debris mode {params.get('mode')!r} — nothing "
                    "cut. Use 'bead' or 'valley'")
        vals = np.asarray(df[fsc].values, dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size < 50:
            return "too few events"
        cut, why_not = _debris_fsc_valley(finite)
        if cut is None:
            return ("the lowest FSC-A mode is cell-sized, not debris — "
                    "nothing cut" if why_not == 'cell-sized' else
                    "FSC-A is unimodal — no low-debris mode to cut")
        return None

    if key == 'viability':
        ch = params.get('channel')
        if not ch or ch not in df.columns:
            ch = find_viability_channel(list(df.columns), channel_labels)
        if not ch or ch not in df.columns:
            return ("no viability dye detected — set the channel via right-click "
                    "(panel has none?)")
        vals = np.asarray(df[ch].values, dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size < 50:
            return "too few events"
        if params.get('max_signal') is not None:
            return f"nothing above the manual ceiling on {ch}"
        thr = _bimodal_valley(finite)
        if thr is None:
            return f"no bimodal live/dead split on {ch}"
        if thr <= float(np.median(finite)):
            return (f"the high-signal population is the majority on {ch} — "
                    "not treated as dead")
        return None

    if key == 'doublets':
        fa = _autoclean_find_scatter(df, 'FSC', '-A')
        fh = _autoclean_find_scatter(df, 'FSC', '-H')
        if fa is None or fh is None or fa == fh:
            return "needs both FSC-A and FSC-H"
        if float(params.get('tol', 0.25)) <= 0:
            return "tolerance is 0"
        return None

    if key == 'gaussian':
        from .gaussian_cleanup import parse_channels
        if not parse_channels(params.get('channels'), df.columns):
            return "no Gaussian parameter (Event_length, Center, Offset, Width, Residual)"
        return None

    if key == 'beads':
        want = [c.strip() for c in str(params.get('channels') or '').split(',')
                if c.strip()]
        if not want or any(c not in df.columns for c in want):
            return "a bead channel is missing from this sample"
        return None

    return None


def autoclean_keep_mask(gate, df):
    """Boolean keep-mask for an 'autoclean' gate: the AND of every ENABLED
    method's per-sample keep-mask, recomputed from ``df``. A group with no
    enabled methods (or an empty df) is a no-op (all-True)."""
    n    = len(df)
    keep = np.ones(n, dtype=bool)
    if n == 0:
        return keep
    enabled = [m for m in (gate.get('methods') or []) if m.get('enabled', True)]
    if not enabled:
        return keep
    want = {m.get('key') for m in enabled}

    def _p(key):
        for m in enabled:
            if m.get('key') == key:
                return m.get('params', {}) or {}
        return {}

    # Time-binned (drift, flow-rate) + margin detectors share AcquisitionQC's
    # pd.cut binning and MAD logic — call it once with per-method flags.
    if want & {'drift', 'flow_rate', 'margin'}:
        dp, fp, mp = _p('drift'), _p('flow_rate'), _p('margin')
        qc  = AcquisitionQC(df.reset_index(drop=True))
        idx = qc.run(
            n_bins=int(dp.get('n_bins', fp.get('n_bins', 200))),
            threshold=float(dp.get('threshold', 5)),
            drift=('drift' in want),
            flow_rate=('flow_rate' in want),
            margins=('margin' in want),
            flow_rate_threshold=float(fp.get('flow_rate_threshold', 5.0)),
            margin_frac=float(mp.get('margin_frac', 0.01)))
        m = np.zeros(n, dtype=bool)
        m[np.asarray(idx, dtype=int)] = True   # idx are positions (reset index)
        keep &= m
    debris_keep = None
    if 'debris' in want:
        debris_keep = _autoclean_debris_mask(df, _p('debris'))
        keep &= debris_keep
    if 'viability' in want:
        keep &= _autoclean_viability_mask(df, _p('viability'))
    if 'doublets' in want:
        # Standard gating order removes debris first, so the singlet-ratio median
        # is centred on cell-sized events, not debris — pass the debris keep-mask
        # as the reference population when debris cleaning is also enabled.
        keep &= _autoclean_doublets_mask(df, _p('doublets'), ref_mask=debris_keep)
    # Mass cytometry: Gaussian discrimination parameters and bead events
    # (added by Tools > Mass cytometry; not in the default recipe).
    if 'gaussian' in want:
        from .gaussian_cleanup import gaussian_keep_mask
        keep &= gaussian_keep_mask(df, _p('gaussian'))
    if 'beads' in want:
        from .bead_normalize import bead_keep_mask
        keep &= bead_keep_mask(df, _p('beads'))
    return keep


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


def cumulative_gate_mask(gates_by_id, gid, df, _depth=0, overrides=None,
                         cache=None):
    """AND every gate's mask from `gid` up the parent chain to the root.
    Cycle-safe.

    The chain ALWAYS includes every ancestor regardless of each one's
    `enabled` flag: the toggle controls visibility ("draw this gate's
    highlight overlay") and pipeline inclusion separately, not whether a
    gate participates in defining the population at this node. So the
    cumulative meaning of leaf C inside parent P stays `P AND C` even
    when P's highlight is hidden.

    `gates_by_id` is a dict of gate_id -> gate_dict where each gate_dict
    may carry a 'parent_id' field naming another key (None = root).

    `overrides` (optional) maps gate_id -> a precomputed df-aligned bool mask
    used INSTEAD of evaluating that gate. The GUI uses it to inject cached
    auto-clean masks (whose recompute is expensive) so a chain that nests
    populations under an auto-clean root doesn't re-run the cleaning per node.

    `cache` (optional) memoises the CUMULATIVE mask per gate id. Walking a
    chain stops at the nearest cached ancestor, so evaluating a whole tree
    costs one `gate_to_mask` per gate instead of one per gate *per descendant*
    — the difference between O(gates x depth) and O(gates). This matters
    because a polygon gate is ~250x the cost of a threshold gate
    (`Path.contains_points` over every event), and real gating hierarchies are
    polygon-based: a 60-gate panel over 200k events drops from ~4.7 s to
    ~0.7 s.

    The caller OWNS the dict and MUST discard it whenever `df`, `gates_by_id`
    or `overrides` change — nothing here can detect that. Masks returned from
    a cache are SHARED; treat them as read-only. Omit `cache` (the default) and
    behaviour is exactly as before, with a freshly allocated mask every call.
    """
    if cache is not None:
        hit = cache.get(gid)
        if hit is not None:
            return hit

    # Walk up to the root (or to the nearest cached ancestor), recording the
    # chain. Re-applying it top-down is what lets every intermediate level be
    # cached on the way back — AND is associative, so the result is unchanged.
    chain = []
    seen = set()
    base = None
    cur = gid
    while cur is not None and cur not in seen:
        if cache is not None:
            hit = cache.get(cur)
            if hit is not None:
                base = hit
                break
        seen.add(cur)
        g = gates_by_id.get(cur)
        if g is None:
            break
        chain.append((cur, g))
        cur = g.get('parent_id')

    mask = np.ones(len(df), dtype=bool) if base is None else base
    for cur_id, g in reversed(chain):
        if overrides is not None and cur_id in overrides:
            mask = mask & overrides[cur_id]
        else:
            mask = mask & gate_to_mask(g, df, gates_by_id, _depth)
        if cache is not None:
            cache[cur_id] = mask
    return mask


# ── FlowJo .wsp writer ────────────────────────────────────────────────────────
#
# Emits Gating-ML v2 XML that FlowJo v10 reads. Mirrors the structure WspReader
# expects (and that the real workspaces in our test set use), so a round-trip
# `extract_gates → write → extract_gates` preserves every gate.
#
# Usage:
#     w = WspWriter(cytometer='LSRFortessa')
#     w.set_compensation(['FL1-A', 'FL2-A', 'FL3-A'], spillover_matrix)
#     w.add_sample('sample_1', '/path/to/sample_1.fcs', channels=[...],
#                  gates=[{'kind': 'polygon', 'id': 'g1', ...}, ...])
#     w.write('out.wsp')

_WSP_NS = {
    'gating':     'http://www.isac-net.org/std/Gating-ML/v2.0/gating',
    'transforms': 'http://www.isac-net.org/std/Gating-ML/v2.0/transformations',
    'data-type':  'http://www.isac-net.org/std/Gating-ML/v2.0/datatypes',
    'xsi':        'http://www.w3.org/2001/XMLSchema-instance',
}


def _q(prefix, local):
    """Build a Clark-notation tag/attr name (`{namespace}localname`) for ET."""
    return f'{{{_WSP_NS[prefix]}}}{local}'


def _wsp_dataset_uri(path):
    """`<DataSet uri>` in the form FlowJo writes: 'file:/' + the absolute
    path with forward slashes, percent-encoded except '/', ':' and ','
    (Windows: 'file:/C:/a%20b,c/x.fcs'; POSIX: 'file:/data/a%20b/x.fcs')."""
    from urllib.parse import quote
    p = os.path.abspath(path).replace('\\', '/')
    if not p.startswith('/'):
        p = '/' + p
    return 'file:' + quote(p, safe='/:,')


def _wsp_uri_path(uri):
    """Local path of a <DataSet uri> ('file:/C:/a%20b/x.fcs', 'file:C:/...',
    '/data/x.fcs' or a bare path), or None when it is empty."""
    if not uri:
        return None
    from urllib.parse import unquote
    p = unquote(uri[len('file:'):]) if uri.startswith('file:') else uri
    p = p.replace('\\', '/')
    if p.startswith('//'):                       # file:///C:/... or file:///data
        p = p[2:]
    if len(p) > 2 and p[0] == '/' and p[2] == ':':  # /C:/... on Windows
        p = p[1:]
    return p


# ── FCS TEXT and Time units at the .wsp boundary ─────────────────────────────
#
# OpenFlo keeps the Time channel as the FCS stores it: raw ticks. FlowJo
# 10.10.2 holds Time in SECONDS (measured with synthetic files that differ
# only in that keyword): seconds = ticks x $TIMESTEP, with $TIMESTEP read from
# the FCS file -- a different value in the workspace <Keywords> is ignored.
# A file with no $TIMESTEP gets ($ETIM - $BTIM) / last tick. So the writer
# multiplies Time coordinates by that factor and the reader divides by it.

def _positive_float(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) and f > 0 else None


def _fcs_text_pairs(path):
    """The FCS TEXT segment as (key, value) pairs in file order, each key in
    its original case with its '$' kept: the form FlowJo 10 writes into a
    sample's <Keywords>. flowio cannot supply this (it lower-cases keys and
    strips every '$'). A doubled delimiter is an escaped literal delimiter."""
    with open(path, 'rb') as fh:
        hdr = fh.read(58)
        t0, t1 = int(hdr[10:18]), int(hdr[18:26])
        fh.seek(t0)
        raw = fh.read(t1 - t0 + 1)
    try:
        s = raw.decode('utf-8')
    except UnicodeDecodeError:
        s = raw.decode('latin-1')
    delim, parts, cur, i = s[0], [], [], 1
    while i < len(s):
        c = s[i]
        if c == delim:
            if i + 1 < len(s) and s[i + 1] == delim:     # escaped delimiter
                cur.append(delim)
                i += 2
                continue
            parts.append(''.join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    if cur:
        parts.append(''.join(cur))
    # strict=False: a malformed TEXT with a trailing key and no value drops it
    return list(zip(parts[0::2], parts[1::2], strict=False))


def _xml_safe(v):
    """Drop the characters XML 1.0 cannot carry (C0 controls other than tab,
    newline and CR); an FCS TEXT value may contain them."""
    return ''.join(ch for ch in v
                   if (ch in '\t\n\r' or ord(ch) >= 0x20)
                   and ch not in '\ufffe\uffff')


def _clock_seconds(v):
    """$BTIM/$ETIM -> seconds since midnight. FCS 3.1 writes hh:mm:ss[.cc],
    FCS 3.0 hh:mm:ss[:tt] with tt in 1/60 s. None if unparseable."""
    try:
        f = str(v).strip().split(':')
        sec = int(f[0]) * 3600 + int(f[1]) * 60 + float(f[2])
        if len(f) == 4:
            sec += int(f[3]) / 60.0
        return sec
    except (ValueError, IndexError):
        return None


def _fcs_timestep(path, pairs=None):
    """Seconds per Time tick for the FCS at `path`, as FlowJo derives it:
    $TIMESTEP; failing that ($ETIM - $BTIM) / last Time tick. None when the
    file is missing or neither is usable. `pairs` = its _fcs_text_pairs."""
    if not path or not os.path.isfile(path):
        return None
    try:
        kw = {k.upper(): v for k, v in (pairs if pairs is not None
                                         else _fcs_text_pairs(path))}
    except (OSError, ValueError, IndexError):
        return None
    ts = _positive_float(kw.get('$TIMESTEP'))
    if ts:
        return ts
    b, e = _clock_seconds(kw.get('$BTIM')), _clock_seconds(kw.get('$ETIM'))
    if b is None or e is None:
        return None
    span = e - b if e > b else e - b + 86400.0     # acquisition over midnight
    try:
        fd = read_fcs_tolerant(path, _fcs_read.FlowData, [])
        names = [fd.channels[k]['pnn'] for k in sorted(fd.channels, key=int)]
        ti = [n.lower() for n in names].index('time')
        if fd.events is None:
            raise ValueError('no DATA segment')
        events = np.asarray(fd.events, dtype=float).reshape(-1, fd.channel_count)
        last = float(events[:, ti].max())
    except Exception as exc:
        note("reading the time channel from an FCS file", exc)
        return None
    return _positive_float(span / last) if last > 0 else None


def _wsp_time_gain(sample_elem):
    """The `gain` of the linear axis a (namespace-stripped) FlowJo <Sample>
    declares for its Time parameter in <Transformations>: the seconds per
    tick FlowJo shows Time in. None when there is no such axis, or its gain
    is missing, not a positive number, or 1 (which says nothing about
    seconds; the FCS decides, see WspReader.sample_timestep)."""
    if sample_elem is None:
        return None
    for block in sample_elem.findall('Transformations'):
        for t in block:
            p = t.find('parameter')
            if t.tag != 'linear' or p is None or (p.get('name') or '').lower() != 'time':
                continue
            g = _positive_float(t.get('gain'))
            return g if g and g != 1.0 else None
    return None


def _scale_time_coords(g, factor):
    """Copy of gate dict `g` with every coordinate on a Time axis multiplied
    by `factor`; `g` itself when there is no factor or no Time axis. The
    ±1e12 open-bound sentinel stays the sentinel."""
    if not factor or factor == 1.0:
        return g

    def k(ch):
        return factor if isinstance(ch, str) and ch.lower() == 'time' else 1.0

    def sc(v, f):
        v = float(v)
        return v if f == 1.0 or abs(v) >= 1e12 else v * f

    if 'channel' in g:
        f = k(g['channel'])
        if f == 1.0:
            return g
        g = dict(g)
        for key in ('value', 'lo', 'hi'):
            if g.get(key) is not None:
                g[key] = sc(g[key], f)
        return g
    fx, fy = k(g.get('x_channel')), k(g.get('y_channel'))
    if fx == 1.0 and fy == 1.0:
        return g
    g = dict(g)
    for key, f in (('x0', fx), ('x1', fx), ('quad_origin_x', fx),
                   ('y0', fy), ('y1', fy), ('quad_origin_y', fy)):
        if g.get(key) is not None:
            g[key] = sc(g[key], f)
    if g.get('vertices') is not None:
        g['vertices'] = [[sc(vx, fx), sc(vy, fy)] for vx, vy in g['vertices']]
    if g.get('mean') is not None:
        g['mean'] = [sc(g['mean'][0], fx), sc(g['mean'][1], fy)]
    if g.get('cov') is not None:
        f2 = (fx, fy)
        g['cov'] = [[float(g['cov'][i][j]) * f2[i] * f2[j] for j in range(2)]
                    for i in range(2)]
    return g


# ── Gate coordinates across the .wsp boundary ───────────────────────────────
#
# A FlowJo workspace holds gate coordinates in the parameter's SCALE units:
# compensated linear intensity (compare.py evaluates them on linear data and
# agrees with FlowJo's counts). OpenFlo's editor and CLI store fluorescence
# logicle-transformed in place and keep gates in those stored coordinates. The
# two were passed across unconverted in both directions: an imported FlowJo
# "FL1-A > 1000" was evaluated against logicle values that never exceed ~1.2
# (7,653 events in FlowJo, 0 in OpenFlo), and an exported OpenFlo threshold of
# 0.454 (logicle) reached FlowJo as an intensity of 0.454 (7,653 events in
# OpenFlo, 14,087 in FlowJo). `map_gate_coords` converts a gate between the
# two, given a function per transformed channel.

_GATE_OPEN = 1e12          # |coordinate| at or beyond this is an open side
_POLYGON_EDGE_STEPS = 16   # sub-segments per polygon edge when re-mapped


def map_gate_coords(g, fns):
    """Copy of gate dict `g` with every coordinate on a channel in `fns` passed
    through that channel's function; `g` itself when no axis maps.

    `fns` is ``{channel: f}`` with `f` a vectorised, increasing map (e.g.
    ``transform_values`` / ``inverse_transform_values`` for that channel's
    method). Bounds map exactly because the transforms are monotone, and the
    ±1e12 open-bound sentinels stay open. A non-finite image (a non-positive
    bound under 'log') becomes the open lower sentinel, as it lies below every
    value the channel can hold.

    A straight polygon edge in one space is a curve in the other, so each
    edge is subdivided before its vertices are mapped; an ellipse is not an
    ellipse in the other space, so it is traced as a polygon first (area-
    preserving, see `_ellipse_polygon`) and leaves as kind 'polygon'.

    A polygon imported from FlowJo with 'flowjo_edge_axes' has its edges
    straight on FlowJo's axes, not in `g`'s units: each edge is traced there
    instead (even when no axis maps), and the copy, straight in the units it
    leaves in, drops the key."""
    edge_fns = _polygon_edge_fns(g) if g.get('kind') == 'polygon' else None
    if not fns and edge_fns is None:
        return g
    fns = fns or {}

    def fn_for(ch):
        return fns.get(ch) if isinstance(ch, str) else None

    def m(f, vals):
        v = np.asarray(vals, dtype=float).copy()
        live = np.abs(v) < _GATE_OPEN
        if f is not None and live.any():
            got = np.asarray(f(v[live]), dtype=float).reshape(-1)
            v[live] = np.where(np.isfinite(got), got, -_GATE_OPEN)
        return v

    if 'channel' in g:                              # threshold / interval
        f = fn_for(g['channel'])
        if f is None:
            return g
        g = dict(g)
        for key in ('value', 'lo', 'hi'):
            if g.get(key) is not None:
                g[key] = float(m(f, [g[key]])[0])
        return g
    fx, fy = fn_for(g.get('x_channel')), fn_for(g.get('y_channel'))
    if fx is None and fy is None and edge_fns is None:
        return g
    g = dict(g)
    kind = g.get('kind')
    if edge_fns is not None:
        g.pop('flowjo_edge_axes', None)
        g['vertices'] = _trace_edges_on_axes(g['vertices'], edge_fns)
    for key, f in (('x0', fx), ('x1', fx), ('quad_origin_x', fx),
                   ('y0', fy), ('y1', fy), ('quad_origin_y', fy)):
        if g.get(key) is not None:
            g[key] = float(m(f, [g[key]])[0])
    if kind == 'ellipsoid' and g.get('mean') is not None:
        verts = _ellipse_polygon(g['mean'], g['cov'],
                                 g.get('distance_sq', 4.0))
        if verts is None:
            log.warning("  [gate] %s: ellipse with a singular covariance "
                        "cannot be converted between scales — left as is.",
                        g.get('name') or g.get('id') or 'ellipse')
            return g
        for key in ('mean', 'cov', 'distance_sq'):
            g.pop(key, None)
        g['kind'] = kind = 'polygon'
        g['vertices'] = verts
    if kind == 'polygon' and g.get('vertices') is not None:
        v = np.asarray(g['vertices'], dtype=float)
        if v.ndim == 2 and v.shape[1] == 2 and len(v) >= 3:
            pts = []
            for i in range(len(v)):
                a, b = v[i], v[(i + 1) % len(v)]
                pts.append(a)
                if edge_fns is None and np.all(
                        np.abs(np.concatenate([a, b])) < _GATE_OPEN):
                    t = np.arange(1, _POLYGON_EDGE_STEPS) / _POLYGON_EDGE_STEPS
                    pts.extend(a + (b - a) * t[:, None])
            p = np.asarray(pts, dtype=float)
            xs, ys = m(fx, p[:, 0]), m(fy, p[:, 1])
            g['vertices'] = [[float(x), float(y)] for x, y in zip(xs, ys, strict=True)]
    return g


def _trace_edges_on_axes(vertices, edge_fns, steps=None):
    """Polygon `vertices` (data units) with each edge traced as `steps`
    segments straight on the axes `edge_fns` names (see _polygon_edge_fns);
    an edge with an open (±1e12) end is kept as it is."""
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] != 2 or len(v) < 3:
        return vertices
    steps = steps or _EDGE_AXIS_STEPS
    t = (np.arange(steps) / steps)[:, None]
    out = []
    for i in range(len(v)):
        a, b = v[i], v[(i + 1) % len(v)]
        if not np.all(np.abs(np.concatenate([a, b])) < _GATE_OPEN):
            out.append(a[None, :])
            continue
        ea, eb = a.copy(), b.copy()
        for k, f in enumerate(edge_fns):
            if f is not None:
                ea[k], eb[k] = (float(x) for x in f[0](np.array([a[k], b[k]])))
        seg = ea + (eb - ea) * t
        for k, f in enumerate(edge_fns):
            if f is not None:
                seg[:, k] = f[1](seg[:, k])
        seg[0] = a                     # the drawn vertex exactly
        out.append(seg)
    p = np.concatenate(out)
    return [[float(x), float(y)] for x, y in p]


def gate_scale_fns(transforms, direction):
    """``{channel: f}`` for `map_gate_coords` from ``{channel: method}`` (or
    ``{channel: params dict}`` as FlowSample._transformed_channels returns):
    `direction` 'to_linear' inverts the stored transform (export to FlowJo),
    'from_linear' applies it (import from FlowJo). Linear channels are left
    out, so gates on them pass through untouched."""
    if direction not in ('to_linear', 'from_linear'):
        raise ValueError(f"direction must be 'to_linear' or 'from_linear', "
                         f"not {direction!r}")
    out = {}
    for ch, spec in (transforms or {}).items():
        params = dict(spec) if isinstance(spec, dict) else {'method': spec}
        if (params.get('method') or 'linear') == 'linear':
            continue
        if is_unknown_spec(params):
            # No transform to invert or apply: the gate stays as it is, which
            # is wrong in one of the two spaces, so say so.
            log.warning("  [gate] %s: display scale unknown -- gate "
                        "coordinates on it are NOT converted for FlowJo.", ch)
            continue
        params = transform_params(params)   # not 'origin' / 'guessed'
        fn = (inverse_transform_values if direction == 'to_linear'
              else transform_values)
        out[ch] = (lambda v, _p=params, _f=fn:
                   np.asarray(_f(np.asarray(v, dtype=float), **_p)))
    return out


def gate_rescale_fns(before, after):
    """``{channel: f}`` for `map_gate_coords` that moves gate coordinates
    from the transforms `before` to `after` (two ``data_transforms``
    records): undo the old transform, apply the new one. Only channels whose
    parameters differ are listed; a channel of unknown scale on either side
    is left out, as nothing converts to or from it. Used when a sample's data
    is re-transformed under gates drawn on its old transform (a cofactor-5
    asinh sample put on the editor's cofactor 150): unconverted, a cut that
    held 240 of 800 events held none."""
    before, after = before or {}, after or {}
    lin = {'method': 'linear'}
    out = {}
    for ch in set(before) | set(after):
        b, a = before.get(ch) or lin, after.get(ch) or lin
        if is_unknown_spec(b) or is_unknown_spec(a):
            continue
        if transform_params(b) == transform_params(a):
            continue
        a_params = transform_params(a)
        if a.get('method', 'linear') == 'linear':
            out[ch] = (lambda v, _b=b: to_linear(v, _b))
        else:
            out[ch] = (lambda v, _b=b, _p=a_params: np.asarray(
                transform_values(to_linear(v, _b), **_p)))
    return out


def rank_matched_fn(stored, linear, n=1024):
    """``f`` for `map_gate_coords` taking a channel of UNKNOWN scale
    (`stored`, its values as kept) to the linear units FlowJo will see
    (`linear`, the same channel of the events read from the FCS): the
    stored value at each quantile goes to the linear value at the same
    quantile. The display transforms are increasing, so when both hold the
    same events this is the transform itself and a cut keeps exactly its
    events either side; past the data it carries on with the end slopes, so
    a bound beyond every event stays beyond every event. ``None`` when
    either side has fewer than two distinct finite values."""
    s = np.asarray(stored, dtype=float)
    x = np.asarray(linear, dtype=float)
    s, x = s[np.isfinite(s)], x[np.isfinite(x)]
    if np.unique(s).size < 2 or np.unique(x).size < 2:
        return None
    q = np.linspace(0.0, 1.0, int(n))
    sq, xq = np.quantile(s, q), np.quantile(x, q)
    sq, keep = np.unique(sq, return_index=True)     # np.interp needs xp rising
    xq = xq[keep]
    if sq.size < 2:
        return None

    span = (xq[-1] - xq[0]) / (sq[-1] - sq[0])      # > 0: both sides vary

    def _slope(i, j):
        d = (xq[j] - xq[i]) / (sq[j] - sq[i])
        return d if d > 0 else span

    lo_k, hi_k = _slope(0, 1), _slope(-2, -1)

    def f(v):
        v = np.asarray(v, dtype=float)
        out = np.interp(v, sq, xq)
        lo, hi = v < sq[0], v > sq[-1]
        out[lo] = xq[0] + (v[lo] - sq[0]) * lo_k
        out[hi] = xq[-1] + (v[hi] - sq[-1]) * hi_k
        return out
    return f


def fcs_values_as_flowjo_gates_them(fcs_path, *, qc=False,
                                    compensation=None):
    """The events of `fcs_path` on the linear scale FlowJo gates them on in
    an OpenFlo export: compensated with `compensation` (``(channels,
    matrix)``, the one written for the sample) or not at all, and after QC
    when the sample had it, so they are the same events. A fresh FlowSample;
    nothing the editor holds is touched."""
    raw = FlowSample(fcs_path)
    if qc:
        raw.run_qc()
    if compensation is not None:
        raw.manual_compensate(compensation[1], list(compensation[0]))
    return raw.data


def mark_unknown_scale(gate: dict, transforms) -> dict:
    """`gate` (a copy) flagged ``scale_unknown`` when it is in FlowJo's
    linear units on a channel whose display scale is unknown
    (``transforms``: the sample's stored transforms), which gate_scale_fns
    cannot convert: compared as stored, its coordinates name another region.
    gate_to_mask then fails closed and statistics say NA. A boolean or
    placeholder gate has no coordinates of its own and is returned as is."""
    from .gating import gate_channels
    if gate.get('kind') in ('boolean', 'flowjo_ellipse'):
        return gate
    if any(is_unknown_spec((transforms or {}).get(ch))
           for ch in gate_channels(gate)):
        out: dict = dict(gate)
        out['scale_unknown'] = True
        return out
    return gate


# The FlowJo version an export says it is in the file's flowJoVersion: the
# version whose own files the export's form follows (full-path <Dependent>
# names, a <Transformations> block per sample). Other readers (FlowKit) parse
# it as a number; OpenFlo's 'OpenFlo-export-1.0' stopped them. Which OpenFlo
# wrote the file is in the root's WSP_OPENFLO_ATTR attribute instead.
WSP_FLOWJO_VERSION = '10.8.1'
WSP_OPENFLO_ATTR = 'openfloVersion'


def wsp_written_by_openflo(root):
    """Whether a parsed workspace `root` is one OpenFlo exported: it carries
    WSP_OPENFLO_ATTR, or (exports before that attribute) a flowJoVersion
    that starts 'OpenFlo'."""
    return root is not None and (bool(root.get(WSP_OPENFLO_ATTR)) or str(
        root.get('flowJoVersion') or '').startswith('OpenFlo'))


class WspWriter:
    """Build a FlowJo-compatible .wsp workspace from in-memory gate dicts.

    Round-trips through WspReader: every gate kind we author (threshold,
    interval, rect, polygon) survives extract → write → extract.

    Out of scope: boolean gates only (ellipsoid DOES round-trip).
    A 'flowjo_ellipse' placeholder (a FlowJo-drawn ellipse OpenFlo could not
    convert) is written back as the EllipsoidGate FlowJo drew, with the axes
    it was drawn on, so FlowJo counts it as before.
    An editor quadrant (4 rects sharing a ``quad_set``) is written as 4
    sibling RectangleGate populations — the form FlowJo itself writes — so
    a child stays under its own sector. Re-import gives 4 plain rects.
    Cells / event counts are emitted as 0 — FlowJo recomputes them.
    """

    def __init__(self, *, cytometer='Generic',
                 flowjo_version=WSP_FLOWJO_VERSION):
        self.cytometer       = cytometer
        self.flowjo_version  = flowjo_version
        self.samples         = []   # list of dicts (see add_sample)
        self.matrix          = None # (name, channels, np.ndarray)
        # Filled by write(): where FlowJo will not count what OpenFlo counts.
        self.warnings        = []
        # Filled by add_sample: gates on a channel of unknown scale.
        self.scale_notes     = []
        # Fix ids / labels the stamp comment names (export_stamp).
        self.corrections: list[str] | None = None

    @staticmethod
    def _check_matrix(channels, matrix):
        m = np.asarray(matrix, dtype=float)
        if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] != len(channels):
            raise ValueError(
                f"matrix shape {m.shape} doesn't match {len(channels)} channels")
        return m

    def set_compensation(self, channels, matrix, name='Acquisition-defined'):
        """Register a workspace-level spillover matrix, listed in <Matrices>
        only. `matrix` is an NxN numpy array whose rows are source channels
        in `channels` order, columns destination channels in the same order.
        Diagonals are usually 1.0.

        It is NOT copied into the samples: a matrix inside a <Sample> is what
        FlowJo applies to that sample, as given, so a copy of one file's
        matrix would override every other file's own. A sample with no
        matrix of its own (see `add_sample`) gets FlowJo's
        "Acquisition-defined" matrix from its own FCS (SPILL)."""
        self.matrix = (name, list(channels),
                       self._check_matrix(channels, matrix))

    def add_sample(self, name, fcs_path, channels, gates, compensation=None,
                   timestep=None, transforms=None, unknown_fns=None,
                   group=None):
        """Register one sample's gating tree. `gates` is a list of gate
        dicts in the shared schema (see `gate_to_mask`) with `id` and
        `parent_id` fields resolved within the list (use
        `read_template_gates` if you have a .wsp / template to convert).
        `compensation` is the `(channels, matrix)` OpenFlo applies to THIS
        sample; it is written inside the sample, where FlowJo applies it.

        `transforms` is ``{channel: method}`` for the channels whose gate
        coordinates are in a transformed space (the editor's logicle, say).
        Those coordinates are converted to the linear scale units a FlowJo
        workspace holds (see `map_gate_coords`); without it every coordinate
        is taken to be linear already.

        A channel of UNKNOWN scale in `transforms` has no transform to undo.
        `unknown_fns` (``{channel: f}``, see `rank_matched_fn`) converts it
        where the caller could match it to the FCS; a gate on one it cannot
        is written as stored. Either way it is listed in `self.scale_notes`
        for the user. A gate flagged ``scale_unknown`` was imported from
        FlowJo and is still in FlowJo's units on such a channel, so only its
        other channels are converted.

        `timestep` is seconds per Time tick, used only when the FCS gives
        none: FlowJo scales Time with the file's own value (see
        `_fcs_timestep`), so that wins, and a different `timestep` is logged.
        Time gates are written in seconds with it. The FCS TEXT goes into
        the sample's <Keywords>. `group` names a FlowJo group the sample is
        put in, besides 'All Samples' (OpenFlo's group of the sample)."""
        comp = None
        if compensation is not None:
            c_chans, c_matrix = compensation
            comp = (list(c_chans), self._check_matrix(c_chans, c_matrix))
        keywords = None
        if fcs_path and os.path.isfile(fcs_path):
            try:
                keywords = _fcs_text_pairs(fcs_path)
            except (OSError, ValueError, IndexError) as exc:
                log.info(f"[WspWriter] {fcs_path}: FCS TEXT unreadable "
                         f"({exc}); sample written without <Keywords>")
        ts = _fcs_timestep(fcs_path, keywords)
        given = _positive_float(timestep) if timestep is not None else None
        if ts and given and not np.isclose(ts, given, rtol=1e-9):
            log.warning(f"[WspWriter] {name}: timestep {given:g} differs from "
                        f"the FCS's {ts:g}; FlowJo uses the file's, so it is used")
        transforms = transforms or {}
        unknown = {ch for ch, sp in transforms.items()
                   if is_unknown_spec(sp if isinstance(sp, dict)
                                      else {'method': sp})}
        unknown_fns = {ch: f for ch, f in (unknown_fns or {}).items()
                       if ch in unknown}
        to_linear = gate_scale_fns(
            {ch: sp for ch, sp in transforms.items() if ch not in unknown},
            'to_linear')
        gates = [self._gate_to_linear(name, g, to_linear, unknown_fns, unknown)
                 for g in gates]
        self.samples.append({
            'name':         name,
            'fcs_path':     fcs_path or '',
            'channels':     list(channels),
            'gates':        list(gates),
            'compensation': comp,
            'keywords':     keywords,
            'timestep':     ts or given,
            'group':        str(group) if group else None,
        })

    def _gate_to_linear(self, sample, g, to_linear, unknown_fns, unknown):
        """`g` in FlowJo's linear units (see `add_sample`), noting in
        `scale_notes` a gate that rests on a channel of unknown scale."""
        from .gating import gate_channels
        # map_gate_coords also traces a polygon imported with FlowJo's edge
        # axes onto straight edges in data units, which is how the workspace
        # written here declares them (no <Transformations>), so it is called
        # even when no channel maps.
        if (g.get('scale_unknown')
                or g.get('kind') in ('boolean', 'flowjo_ellipse')):
            return map_gate_coords(g, to_linear)
        label = g.get('name') or g.get('id') or 'gate'
        for ch in sorted(set(gate_channels(g)) & unknown):
            how = ('converted by matching its values to the FCS rank for rank'
                   if ch in unknown_fns else
                   'written as stored (no FCS to match it to), so FlowJo '
                   'will likely count another region')
            note = f"{sample} / {label}: {ch} has an unknown scale; {how}."
            log.warning("[WspWriter] %s", note)
            self.scale_notes.append(note)
        fns = dict(to_linear, **unknown_fns)
        return map_gate_coords(g, fns)

    def write(self, out_path):
        """Render the workspace and write it to `out_path`."""
        xml_str = self._build_xml()
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(xml_str)

    # ── XML construction (private) ────────────────────────────────────────

    def _build_xml(self):
        self.warnings = []
        for prefix, uri in _WSP_NS.items():
            ET.register_namespace(prefix, uri)

        from .export_stamp import version as openflo_version
        root = ET.Element('Workspace', {
            'version':       '20.0',
            'modDate':       _now_for_wsp(),
            'flowJoVersion': self.flowjo_version,
            WSP_OPENFLO_ATTR: openflo_version(),
            _q('xsi', 'schemaLocation'): (
                f"{_WSP_NS['gating']} {_WSP_NS['gating']}/Gating-ML.v2.0.xsd "
                f"{_WSP_NS['transforms']} "
                f"{_WSP_NS['transforms']}/Transformations.v2.0.xsd "
                f"{_WSP_NS['data-type']} "
                f"{_WSP_NS['data-type']}/DataTypes.v2.0.xsd"),
        })

        # ── Matrices ──────────────────────────────────────────────────────
        # FlowJo lists each matrix here AND puts a copy (same transforms:id)
        # inside every <Sample> it compensates, right after <DataSet>.
        plan, sample_matrix = self._matrix_plan()
        matrices = ET.SubElement(root, 'Matrices')
        for entry in plan:
            self._emit_matrix(matrices, *entry)

        # ── Cytometers (minimal — FlowJo wants the element present) ───────
        cyts = ET.SubElement(root, 'Cytometers')
        ET.SubElement(cyts, 'Cytometer', {
            'name':          self.cytometer,
            'cyt':           self.cytometer,
            'transformType': 'BIEX',
            'manufacturer':  '',
            'serialnumber':  '',
        })

        # ── Groups (one default group, sample refs only) ──────────────────
        groups = ET.SubElement(root, 'Groups')
        grp_node = ET.SubElement(groups, 'GroupNode', {
            'name':         'All Samples',
            'owningGroup':  'All Samples',
            'expanded':     '1',
            'sortPriority': '10',
        })
        grp = ET.SubElement(grp_node, 'Group', {'name': 'All Samples'})
        refs = ET.SubElement(grp, 'SampleRefs')
        for i in range(1, len(self.samples) + 1):
            ET.SubElement(refs, 'SampleRef', {'sampleID': str(i)})
        # One more group per OpenFlo group, so FlowJo (and OpenFlo reading
        # the workspace back) keeps the samples grouped.
        named = {}
        for i, s in enumerate(self.samples, 1):
            if s.get('group') and s['group'].lower() not in \
                    WspReader._DEFAULT_GROUPS:
                named.setdefault(s['group'], []).append(i)
        for gname, ids in named.items():
            node = ET.SubElement(groups, 'GroupNode', {
                'name': gname, 'owningGroup': gname, 'expanded': '1',
                'sortPriority': '10'})
            g_el = ET.SubElement(node, 'Group', {'name': gname})
            g_refs = ET.SubElement(g_el, 'SampleRefs')
            for i in ids:
                ET.SubElement(g_refs, 'SampleRef', {'sampleID': str(i)})

        # ── SampleList ────────────────────────────────────────────────────
        sample_list = ET.SubElement(root, 'SampleList')
        for i, s in enumerate(self.samples, 1):
            sample = ET.SubElement(sample_list, 'Sample')
            if s['fcs_path']:
                ET.SubElement(sample, 'DataSet', {
                    'uri':      _wsp_dataset_uri(s['fcs_path']),
                    'sampleID': str(i),
                })
            own = sample_matrix[i - 1]
            if own is not None:
                self._emit_matrix(sample, *own)
            gates = [self._flowjo_form(g, s) for g in s['gates']]
            self._emit_axes(sample, s, gates)
            self._emit_keywords(sample, s)
            sn_attrs = {
                'name':         s['name'],
                'sampleID':     str(i),
                'owningGroup':  '',
                'expanded':     '1',
                'sortPriority': '10',
                'count':        '0',
            }
            sample_node = ET.SubElement(sample, 'SampleNode', sn_attrs)
            self._emit_gate_tree(sample_node, gates)

        # Pretty-print (Python 3.9+).
        if hasattr(ET, 'indent'):
            ET.indent(root, space='  ')
        body = ET.tostring(root, encoding='unicode')
        # Which OpenFlo wrote it, and with which corrections, as a comment
        # too (the root's WSP_OPENFLO_ATTR names the version alone).
        from .export_stamp import xml_comment
        return ('<?xml version="1.0" encoding="UTF-8"?>\n'
                + xml_comment(self.corrections) + '\n' + body)

    def _flowjo_form(self, g, s):
        """Gate `g` of sample `s` as FlowJo must receive it, recording in
        `self.warnings` every place FlowJo will count differently:

        - an ellipse becomes a polygon: FlowJo 10.10.2 silently drops the
          whole sample for a Gating-ML (mean/covariance) EllipsoidGate;
        - Time coordinates become seconds (see `_scale_time_coords`);
        - a channel this sample's own matrix compensates is named
          'Comp-<channel>': OpenFlo gated the compensated values, and in
          FlowJo the bare name is the raw parameter;
        - a lower edge FlowJo opens (or may open) is reported, not changed."""
        name = g.get('label') or g.get('name') or g.get('id')
        if g.get('kind') == 'ellipsoid':
            base = {k: g[k] for k in ('id', 'parent_id', 'name', 'label', '_import_id')
                    if k in g}
            verts = _ellipse_polygon(g['mean'], g['cov'], g.get('distance_sq', 4.0))
            if verts is not None:
                g = dict(base, kind='polygon', x_channel=g['x_channel'],
                         y_channel=g['y_channel'], vertices=verts)
                self._warn(f"{s['name']}: ellipse '{name}' exported as a "
                           f"{len(verts)}-vertex polygon (FlowJo drops a sample "
                           "with an ellipse gate); FlowJo decides polygon "
                           "membership on a display grid, so counts near the "
                           "edge can differ")
            else:
                # No ellipse, so gate_to_mask admits no events through it
                # (fail-closed). No gate shape is known that FlowJo counts as
                # empty, so it still goes out as the all-pass rectangle, and
                # the warning says the counts will differ.
                try:
                    np.linalg.inv(np.asarray(g['cov'], dtype=float))
                    why = "a covariance that is not positive definite"
                except np.linalg.LinAlgError:
                    why = "a singular covariance"
                g = dict(base, kind='rect', x_channel=g['x_channel'],
                         y_channel=g['y_channel'], x0=-1e6, x1=1e6, y0=-1e6, y1=1e6)
                self._warn(f"{s['name']}: ellipse '{name}' has {why}, so OpenFlo "
                           "admits no events through it; exported as an "
                           "all-pass rectangle, so FlowJo will differ and count "
                           "every event")
        elif g.get('kind') == 'flowjo_ellipse':
            self._warn(f"{s['name']}: FlowJo ellipse '{name}' written back as "
                       "FlowJo drew it, on the axes it was drawn on; OpenFlo "
                       "could not convert it and counts it, and every "
                       "population under it, as empty, but FlowJo will count "
                       "them")
        if s['timestep'] and _touches_time(g):
            self._check_time_edge(g, s, name)
        g = _scale_time_coords(g, s['timestep'])
        if not s['timestep'] and _touches_time(g):
            self._warn(f"{s['name']}: gate '{name}' is on Time but the FCS gives "
                       "no $TIMESTEP or $BTIM/$ETIM; exported in ticks, and "
                       "FlowJo's Time scale for it is unknown")
        for ch, edge, certain in _edges_flowjo_opens(g, self._display(s)):
            self._warn(f"{s['name']}: gate '{name}' lower edge {edge:g} on {ch} "
                       + ("is at the bottom of FlowJo's axis: FlowJo treats it "
                          "as open and also counts every event below it" if certain
                          else "is near the bottom of FlowJo's axis, where FlowJo "
                          "may treat it as open (not measured)"))
        comp = set(s['compensation'][0]) if s.get('compensation') else set()
        named = {k: 'Comp-' + g[k] for k in ('channel', 'x_channel', 'y_channel')
                 if g.get(k) in comp}
        return dict(g, **named) if named else g

    def _check_time_edge(self, g, s, name):
        """Warn when a Time gate's upper edge looks converted to seconds
        before it reached the writer, which keeps Time in ticks and converts
        again (so FlowJo gets it `timestep` times too small). The mark: the
        edge (in ticks) ends the gate within the first `timestep` fraction
        of the run (the first 1% at the usual 0.01 s per tick), exactly where
        the run's length in seconds lands. A deliberate Time cut trims the
        start or the end of a run and keeps most of it, so it never lands
        there."""
        hi = _time_upper_edge(g)
        ts = float(s['timestep'])
        if hi is None or not (0 < ts < 0.5):
            return
        times = self._time_ticks(s)
        if times is None or not times.size:
            return
        last = float(np.nanmax(times))
        if not (0 <= hi < last and hi <= last * ts * 1.05):
            return
        cut = int(np.count_nonzero(times >= hi))
        self._warn(f"{s['name']}: Time gate '{name}' ends at {hi:g} ticks, "
                   f"where the run's last event is at {last:g}, so it cuts "
                   f"{cut:,} of {times.size:,} events "
                   f"({100.0 * cut / times.size:.1f}%). The run is "
                   f"{last * ts:g} s long, "
                   "so the edge was probably given in seconds: OpenFlo keeps "
                   "Time in ticks and multiplies by the timestep "
                   f"({ts:g} s) for FlowJo, which then gets {hi * ts:g} s "
                   f"where {hi:g} s was meant. Check this gate before "
                   "trusting FlowJo's counts.")

    @staticmethod
    def _time_ticks(s):
        """The Time channel of sample `s`'s FCS (raw ticks), read once and
        kept on the sample; None without a readable FCS or Time channel."""
        if '_time_ticks' in s:
            return s['_time_ticks']
        out = None
        path = s.get('fcs_path')
        if path and os.path.isfile(path):
            try:
                fd = _fcs_read.FlowData(path)
                names = [fd.channels[k]['pnn'].lower()
                         for k in sorted(fd.channels, key=int)]
                if 'time' in names and fd.events is not None:
                    ev = np.asarray(fd.events, dtype=float).reshape(
                        -1, fd.channel_count)
                    out = ev[:, names.index('time')]
            except Exception as exc:                  # noqa: BLE001
                log.info(f"[WspWriter] {path}: Time not read ({exc})")
        s['_time_ticks'] = out
        return out

    def _warn(self, msg):
        self.warnings.append(msg)
        log.warning(f"[WspWriter] {msg}")

    @staticmethod
    def _polygon_channels(s):
        """The channels (bare names) of sample `s`'s polygon gates, ellipses
        included (they go out as polygons): their axes are written linear,
        see `_emit_axes`."""
        out = set()
        for g in s['gates']:
            if g.get('kind') in ('polygon', 'ellipsoid'):
                out.update(c for c in (g.get('x_channel'), g.get('y_channel'))
                           if isinstance(c, str) and c.lower() != 'time')
        return out

    def _display(self, s):
        """{channel: 'LIN'/'LOG'}: the FCS's P<n>DISPLAY (see
        `_flowjo_axis_is_linear`), with the polygon channels `_emit_axes`
        writes linear marked 'LIN', so the warnings follow the axes FlowJo
        will be given."""
        out = _fcs_display(s.get('keywords'))
        ellipse = {c for g in s['gates'] if g.get('kind') == 'flowjo_ellipse'
                   for c, ax in zip((g.get('x_channel'), g.get('y_channel')),
                                    (g.get('flowjo') or {}).get('axes') or (),
                                    strict=False) if ax}
        out.update({c: 'LIN' for c in self._polygon_channels(s) - ellipse})
        return out

    @staticmethod
    def _max_range(s, ch):
        """$PnR of channel `ch` (bare name) in sample `s`'s FCS TEXT, the
        top of FlowJo's linear axis for it; 262144 when not known."""
        kw = {str(k).upper().lstrip('$'): v for k, v in (s.get('keywords') or [])}
        for k, v in kw.items():
            m = re.fullmatch(r'P(\d+)N', k)
            if m and str(v).strip() == ch:
                r = _positive_float(kw.get(f'P{m.group(1)}R'))
                if r:
                    return r
        return 262144.0

    def _emit_axes(self, sample, s, gates):
        """<Transformations>, which FlowJo writes in every sample (other
        readers, FlowKit among them, require it). `gates` are in FlowJo form
        (`_flowjo_form`), so a parameter is named as its gate names it
        ('Comp-' where this sample is compensated). Declared:

        - the axes a 'flowjo_ellipse' is drawn on: its points are fractions
          of those axes, so FlowJo must get the axes it was drawn on, as
          FlowJo writes them (every attribute in the transforms namespace but
          linear's `gain`);
        - a linear axis (0 to the FCS's $PnR, gain 1) for each other gated
          parameter FlowJo shows linear anyway (scatter, a P<n>DISPLAY of
          LIN) and for each polygon's parameters. A polygon goes out with
          edges straight in data units (`_gate_to_linear` traces one drawn on
          another scale), which is what FlowJo does on a linear axis; on its
          default biex axis FlowJo would bend them. OpenFlo reads a linear or
          undeclared axis alike, so the export reads back as written.

        Time is left to FlowJo, which puts it in seconds from the FCS; other
        parameters keep FlowJo's defaults, as before."""
        axes = {}
        for g in gates:
            if g.get('kind') != 'flowjo_ellipse':
                continue
            fj = g.get('flowjo') or {}
            for ch, ax in zip((g.get('x_channel'), g.get('y_channel')),
                              fj.get('axes') or (), strict=False):
                if not ch or not ax:
                    continue      # drawn on FlowJo's default axis: keep it
                if ch in axes and axes[ch] != ax:
                    self._warn(f"{s['name']}: FlowJo ellipses on {ch} were drawn on "
                               "different axes; the first one's axis is written, "
                               "so FlowJo will place the others differently")
                    continue
                axes[ch] = ax
        display = self._display(s)
        for g in gates:
            if g.get('kind') in ('boolean', 'flowjo_ellipse'):
                continue
            for ch in (g.get('channel'), g.get('x_channel'), g.get('y_channel')):
                if (not isinstance(ch, str) or ch in axes
                        or ch.lower() == 'time'
                        or not _flowjo_axis_is_linear(ch, display)):
                    continue
                bare = ch[5:] if ch.startswith('Comp-') else ch
                axes[ch] = {'kind': 'linear', 'attrs': {
                    'minRange': '0', 'maxRange': f'{self._max_range(s, bare):g}',
                    'gain': '1'}}
        block = ET.SubElement(sample, 'Transformations')
        for ch, ax in axes.items():
            t = ET.SubElement(block, _q('transforms', str(ax['kind'])), {
                (k if k == 'gain' else _q('transforms', k)): str(v)
                for k, v in (ax.get('attrs') or {}).items()})
            ET.SubElement(t, _q('data-type', 'parameter'), {
                _q('data-type', 'name'): ch})

    @staticmethod
    def _emit_keywords(sample, s):
        """<Keywords>: FlowJo's FJ_FCS_VERSION, then the FCS TEXT in file
        order with the original key case. FlowJo 10.10.2 computes no
        population of a sample without them (measured: it keeps the
        sample and shows "parameters missing"). With no readable FCS, only a
        known $TIMESTEP is recorded, so OpenFlo's reader can convert Time
        back; it is never added next to an FCS TEXT that lacks it (FlowJo
        reads $TIMESTEP from the file, not from here)."""
        if s.get('keywords'):
            kel = ET.SubElement(sample, 'Keywords')
            ET.SubElement(kel, 'Keyword', {'name': 'FJ_FCS_VERSION', 'value': '3'})
            for k, v in s['keywords']:
                ET.SubElement(kel, 'Keyword', {
                    'name': _xml_safe(k), 'value': _xml_safe(v)})
        elif s.get('timestep'):
            kel = ET.SubElement(sample, 'Keywords')
            ET.SubElement(kel, 'Keyword', {
                'name': '$TIMESTEP', 'value': repr(s['timestep'])})

    def _matrix_plan(self):
        """Return `(plan, per_sample)`. `plan` lists one
        `(name, id, channels, matrix)` per distinct matrix for <Matrices>:
        the `set_compensation` matrix first, then each sample's own.
        `per_sample[k]` is sample k's OWN entry, or None -- never the
        workspace matrix (see `set_compensation`)."""
        plan = []

        def entry(name, channels, matrix):
            for e in plan:
                if e[2] == channels and np.array_equal(e[3], matrix):
                    return e
            if any(e[0] == name for e in plan):
                name = f'{name}-{len(plan) + 1}'
            e = (name, str(uuid.uuid4()), channels, matrix)
            plan.append(e)
            return e

        if self.matrix is not None:
            entry(*self.matrix)
        base = self.matrix[0] if self.matrix is not None else 'Acquisition-defined'
        per_sample = [entry(base, *s['compensation'])
                      if s.get('compensation') is not None else None
                      for s in self.samples]
        return plan, per_sample

    def _emit_matrix(self, parent, name, mid, channels, matrix):
        sm = ET.SubElement(parent, _q('transforms', 'spilloverMatrix'), {
            'spectral':               '0',
            'weightOptAlgorithmType': 'OLS',
            'prefix':                 'Comp-',
            'name':                   name,
            'editable':               '0',
            'color':                  '#c0c0c0',
            'version':                f'FlowJo-{self.flowjo_version}',
            'status':                 'FINALIZED',
            _q('transforms', 'id'):   mid,
            'suffix':                 '',
        })
        params = ET.SubElement(sm, _q('data-type', 'parameters'))
        for ch in channels:
            ET.SubElement(params, _q('data-type', 'parameter'), {
                _q('data-type', 'name'):  ch,
                'userProvidedCompInfix':  f'Comp-{ch}',
            })
        for i, src in enumerate(channels):
            sp = ET.SubElement(sm, _q('transforms', 'spillover'), {
                _q('data-type', 'parameter'): src,
                'userProvidedCompInfix':      f'Comp-{src}',
            })
            for j, dst in enumerate(channels):
                ET.SubElement(sp, _q('transforms', 'coefficient'), {
                    _q('data-type', 'parameter'): dst,
                    _q('transforms', 'value'):    repr(float(matrix[i, j])),
                })

    def _emit_gate_tree(self, sample_node, gates):
        """Build <Subpopulations><Population><Gate>...</Gate><Subpopulations>...
        recursively from a flat list of gate dicts linked by parent_id.

        Quadrant rects (`quad_set`) are NOT folded into a QuadrantGate:
        FlowJo writes a quadrant as 4 RectangleGate populations, and folding
        moved a child of one sector onto the whole quadrant."""
        if not gates:
            return
        # Tolerate gates that carry only the reader's `_import_id` (a direct
        # WspReader→WspWriter round-trip, with no GUI id-assignment in between):
        # synthesise an `id` from it so the hierarchy still builds. parent_id in
        # reader output references `_import_id`, so this stays consistent.
        for _i, g in enumerate(gates):
            if 'id' not in g:
                g['id'] = g.get('_import_id') or f'_g{_i}'
        by_id    = {g['id']: g for g in gates}
        children = {}
        for g in gates:
            children.setdefault(g.get('parent_id'), []).append(g['id'])
        roots = children.get(None, [])
        if not roots:
            return
        sub = ET.SubElement(sample_node, 'Subpopulations')
        taken = set()
        # Population paths (names from the sample down) by gate id, and the
        # boolean nodes whose <Dependent> names wait for them: an operand may
        # be written after the boolean that names it.
        paths, pending = {}, []
        for gid in roots:
            self._emit_population(sub, gid, by_id, children, taken, (),
                                  paths, pending)
        # Each operand by its path from the sample ('A/B/C'), as FlowJo 10
        # writes it. The '../C' form (relative to the boolean) is read back
        # by OpenFlo but not by other readers (FlowKit).
        for deps, operands, _own in pending:
            for op in operands:
                ET.SubElement(deps, 'Dependent', {'name': '/'.join(paths[op])})

    def _emit_population(self, parent, gid, by_id, children, taken,
                         parent_path=(), paths=None, pending=None):
        g = by_id[gid]
        paths = {} if paths is None else paths
        pending = [] if pending is None else pending
        # FlowJo ties a population across samples by its name, and needs it
        # unique among siblings: an unnamed gate is named for what it gates
        # on, not for the editor's per-sample id ('g2' here, 'g3' there).
        pop_name = unique_population_name(wsp_population_name(g), taken)
        own = tuple(parent_path) + (str(pop_name),)
        paths[gid] = own
        attrs = {
            'name':         str(pop_name),
            'count':        '0',
            'owningGroup':  '',
            'expanded':     '1',
            'sortPriority': '10',
        }
        if g.get('kind') == 'boolean':
            # FlowJo's boolean populations: AndNode / OrNode / NotNode, each
            # naming its operands as <Dependent> population paths.
            pop = ET.SubElement(parent, _WSP_BOOLEAN_TAGS[
                str(g.get('op', 'and')).lower()], attrs)
            deps = ET.SubElement(pop, 'Dependents')
            pending.append((deps, [op for op in g.get('operands') or []
                                   if op in by_id], own))
        else:
            # owningGroup="" = the sample's own gate. Naming "All Samples"
            # here would claim a gate the (empty) group node does not carry.
            pop = ET.SubElement(parent, 'Population', attrs)
            gate_wrap = ET.SubElement(pop, 'Gate', {
                _q('gating', 'id'): gid,
            })
            self._emit_gate_xml(gate_wrap, g)
        child_taken = set()
        for child in children.get(gid, []):
            sub = pop.find('Subpopulations')
            if sub is None:
                sub = ET.SubElement(pop, 'Subpopulations')
            self._emit_population(sub, child, by_id, children, child_taken,
                                  own, paths, pending)

    def _emit_gate_xml(self, gate_wrap, g):
        kind = g.get('kind')
        if kind == 'threshold':
            rect = ET.SubElement(gate_wrap, _q('gating', 'RectangleGate'))
            dim  = ET.SubElement(rect, _q('gating', 'dimension'), {
                _q('gating', 'min'): repr(float(g['value'])),
            })
            ET.SubElement(dim, _q('data-type', 'fcs-dimension'), {
                _q('data-type', 'name'): g['channel'],
            })
        elif kind == 'interval':
            rect = ET.SubElement(gate_wrap, _q('gating', 'RectangleGate'))
            dim  = ET.SubElement(rect, _q('gating', 'dimension'),
                                 _bound_attrs(float(g['lo']), float(g['hi'])))
            ET.SubElement(dim, _q('data-type', 'fcs-dimension'), {
                _q('data-type', 'name'): g['channel'],
            })
        elif kind == 'rect':
            rect = ET.SubElement(gate_wrap, _q('gating', 'RectangleGate'))
            x0, x1 = sorted([float(g['x0']), float(g['x1'])])
            y0, y1 = sorted([float(g['y0']), float(g['y1'])])
            for ch, lo, hi in (
                    (g['x_channel'], x0, x1),
                    (g['y_channel'], y0, y1)):
                dim = ET.SubElement(rect, _q('gating', 'dimension'),
                                    _bound_attrs(lo, hi))
                ET.SubElement(dim, _q('data-type', 'fcs-dimension'), {
                    _q('data-type', 'name'): ch,
                })
        elif kind == 'polygon':
            poly = ET.SubElement(gate_wrap, _q('gating', 'PolygonGate'))
            for ch in (g['x_channel'], g['y_channel']):
                dim = ET.SubElement(poly, _q('gating', 'dimension'))
                ET.SubElement(dim, _q('data-type', 'fcs-dimension'), {
                    _q('data-type', 'name'): ch,
                })
            for vx, vy in g.get('vertices', []):
                vert = ET.SubElement(poly, _q('gating', 'vertex'))
                ET.SubElement(vert, _q('gating', 'coordinate'), {
                    _q('data-type', 'value'): repr(float(vx)),
                })
                ET.SubElement(vert, _q('gating', 'coordinate'), {
                    _q('data-type', 'value'): repr(float(vy)),
                })
        elif kind == 'ellipsoid':
            ell = ET.SubElement(gate_wrap, _q('gating', 'EllipsoidGate'))
            for ch in (g['x_channel'], g['y_channel']):
                dim = ET.SubElement(ell, _q('gating', 'dimension'))
                ET.SubElement(dim, _q('data-type', 'fcs-dimension'), {
                    _q('data-type', 'name'): ch,
                })
            mean = ET.SubElement(ell, _q('gating', 'mean'))
            for mv in g['mean']:
                ET.SubElement(mean, _q('gating', 'coordinate'), {
                    _q('data-type', 'value'): repr(float(mv)),
                })
            cov = ET.SubElement(ell, _q('gating', 'covarianceMatrix'))
            for row_vals in g['cov']:
                row = ET.SubElement(cov, _q('gating', 'row'))
                for entry_val in row_vals:
                    ET.SubElement(row, _q('gating', 'entry'), {
                        _q('data-type', 'value'): repr(float(entry_val)),
                    })
            ET.SubElement(ell, _q('gating', 'distanceSquare'), {
                _q('data-type', 'value'): repr(float(g.get('distance_sq', 4.0))),
            })
        elif kind == 'flowjo_ellipse':
            # The element as FlowJo wrote it (read namespace-stripped), with
            # its namespaces put back and its two dimensions named as this
            # export names them ('Comp-' where the sample is compensated).
            raw = ET.fromstring((g.get('flowjo') or {}).get('xml') or '<EllipsoidGate/>')
            _emit_flowjo_element(gate_wrap, raw, [g['x_channel'], g['y_channel']])
        else:
            log.info(f"[WspWriter] unknown gate kind {kind!r} — skipped")


def _emit_flowjo_element(parent, el, names):
    """Re-namespace `el`, a FlowJo gate element read namespace-stripped, under
    `parent` as FlowJo writes it: <fcs-dimension> and the `name`/`value` of
    <fcs-dimension>/<coordinate> in data-type, FlowJo's gating attributes
    (distance, min, max, id, parent_id) and every other tag in gating, the
    rest of the attributes plain. The <fcs-dimension>s take `names` in order."""
    tag = el.tag
    attrs = {}
    for k, v in el.attrib.items():
        if (tag, k) in (('fcs-dimension', 'name'), ('coordinate', 'value')):
            attrs[_q('data-type', k)] = v
        elif k in ('distance', 'min', 'max', 'id', 'parent_id'):
            attrs[_q('gating', k)] = v
        else:
            attrs[k] = v
    if tag == 'fcs-dimension' and names:
        attrs[_q('data-type', 'name')] = names.pop(0)
    new = ET.SubElement(parent, _q('data-type' if tag == 'fcs-dimension' else 'gating', tag),
                        attrs)
    if el.text and el.text.strip():
        new.text = el.text
    for child in el:
        _emit_flowjo_element(new, child, names)


_ELLIPSE_VERTICES = 64
# How FlowJo 10.10.2 treats the bottom of an axis. Measured by having FlowJo
# open, compute and save synthetic probe workspaces whose probe values each
# carried a distinct power-of-two number of events, so every saved count
# decodes to an exact set of events:
# - A rectangle's lower edge at or below a floor is open: FlowJo also counts
#   every event below the axis, however far, whatever the upper edge. On the
#   linear 0..262144 axis FlowJo gives a parameter when a workspace has no
#   <Transformations> (as OpenFlo exports) -- _flowjo_axis_is_linear says
#   which parameters get one -- an edge at 20.125 or below was opened and one
#   at 64 or above kept; in between is not measured. On its default biex
#   (width -100) an edge at -100 was kept and the floor lies in [-2000, -1000).
# - A polygon opens from higher up: its bottom at 1000 was opened on an axis
#   starting at 500, where a rectangle edge at 1000 was kept. FlowJo decides
#   polygon membership on a display grid (256 cells per axis by default,
#   1024 units on 0..262144); the polygon cases fit "bottom inside the first
#   cell", which is what the export warns about.
# Edges at or below _OPEN_BOUND are OpenFlo's own "open" sentinels.
_LINEAR_OPENED_UPTO = 20.125
_LINEAR_KEPT_FROM = 64.0
_POLYGON_LINEAR_OPENED_UPTO = 1.0     # bottoms at 1 and at -2000 were opened
_POLYGON_LINEAR_KEPT_FROM = 1024.0
_BIEX_OPENED_UPTO = -2000.0
_BIEX_KEPT_FROM = -1000.0
_OPEN_BOUND = -1e6


def _positive_definite(cov):
    """Whether `cov` is a covariance that describes an ellipse: finite,
    symmetric and positive definite (it has a Cholesky factor)."""
    cov = np.asarray(cov, dtype=float)
    finite_symmetric = (bool(np.isfinite(cov).all())
                        and bool(np.allclose(cov, cov.T)))
    try:
        np.linalg.cholesky(cov)
        factored = True
    except np.linalg.LinAlgError:
        factored = False
    return finite_symmetric and factored


def _ellipse_polygon(mean, cov, distance_sq, n=_ELLIPSE_VERTICES):
    """Vertices of an n-gon tracing {p : (p-mean)' cov^-1 (p-mean) = distance_sq},
    scaled so its area equals the ellipse's (it neither systematically gains
    nor loses events at the edge). None when `cov` is not positive definite."""
    try:
        L = np.linalg.cholesky(np.asarray(cov, dtype=float))
    except np.linalg.LinAlgError:
        return None
    r = np.sqrt(float(distance_sq)) * np.sqrt(2 * np.pi / (n * np.sin(2 * np.pi / n)))
    t = 2 * np.pi * np.arange(n) / n
    pts = np.asarray(mean, dtype=float)[:, None] + r * (L @ np.vstack([np.cos(t), np.sin(t)]))
    return [[float(x), float(y)] for x, y in pts.T]


def _bound_attrs(lo, hi):
    """gating:min/max for one dimension. A bound at OpenFlo's open sentinel
    (|v| >= 1e12) is left out -- FlowJo's own form for an open side, which it
    accepted for quadrant sectors; WspReader reads a missing bound back as
    the sentinel. Both are kept when both are open (a dimension needs one)."""
    lo_open, hi_open = lo <= -1e12, hi >= 1e12
    if lo_open and hi_open:
        lo_open = hi_open = False
    a = {}
    if not lo_open:
        a[_q('gating', 'min')] = repr(lo)
    if not hi_open:
        a[_q('gating', 'max')] = repr(hi)
    return a


def _time_upper_edge(g):
    """The highest bounded coordinate gate `g` puts on a Time axis, or None
    (no Time axis, only a lower bound, or an open side)."""
    vals = []
    if isinstance(g.get('channel'), str) and g['channel'].lower() == 'time':
        if g.get('hi') is not None:
            vals.append(g['hi'])
    else:
        for axis, keys, i in (('x_channel', ('x0', 'x1'), 0),
                              ('y_channel', ('y0', 'y1'), 1)):
            ch = g.get(axis)
            if not (isinstance(ch, str) and ch.lower() == 'time'):
                continue
            vals.extend(g[k] for k in keys if g.get(k) is not None)
            vals.extend(v[i] for v in g.get('vertices') or ())
    vals = [float(v) for v in vals]
    vals = [v for v in vals if np.isfinite(v) and abs(v) < _GATE_OPEN]
    return max(vals) if vals else None


def _touches_time(g):
    return any(isinstance(g.get(k), str) and g[k].lower() == 'time'
               for k in ('channel', 'x_channel', 'y_channel'))


def _flowjo_axis_is_linear(ch, display=None):
    """Whether FlowJo 10.10.2 shows `ch` on a linear axis in a workspace with
    no <Transformations> (as OpenFlo exports). It follows the FCS's
    P<n>DISPLAY keyword (BD FACSDiva writes LIN or LOG per parameter):
    measured, scatter marked LOG got a biex axis and scatter marked LIN a
    linear one. Without the keyword, scatter and Time are linear and the rest
    biex. `display` is {channel name: DISPLAY value} from the FCS TEXT;
    `ch` is a channel name (str)."""
    d = (display or {}).get(ch[5:] if ch.startswith('Comp-') else ch)
    if ch.lower() != 'time' and d in ('LIN', 'LOG'):
        return d == 'LIN'
    return ch.lower() == 'time' or ch.upper().startswith(('FSC', 'SSC'))


def _fcs_display(keywords):
    """{channel name: 'LIN'/'LOG'} from FCS TEXT pairs ($PnN + PnDISPLAY)."""
    kw = {k.upper().lstrip('$'): v for k, v in (keywords or [])}
    out = {}
    for k, v in kw.items():
        m = re.fullmatch(r'P(\d+)N', k)
        if m:
            d = kw.get(f'P{m.group(1)}DISPLAY', '').strip().upper()
            if d in ('LIN', 'LOG'):
                out[v] = d
    return out


def _edges_flowjo_opens(g, display=None):
    """(channel, lower edge, certain) for each lower edge of gate `g` (in
    FlowJo units) that FlowJo opens (certain) or may open (not measured),
    while OpenFlo keeps it literal. See the thresholds above."""
    kind = g.get('kind')
    edges = []
    if kind == 'threshold':
        edges.append((g.get('channel'), g.get('value')))
    elif kind == 'interval':
        edges.append((g.get('channel'), g.get('lo')))
    elif kind == 'rect':
        edges.append((g.get('x_channel'), min(float(g['x0']), float(g['x1']))))
        edges.append((g.get('y_channel'), min(float(g['y0']), float(g['y1']))))
    elif kind == 'polygon' and g.get('vertices'):
        vs = np.asarray(g['vertices'], dtype=float)
        edges.append((g.get('x_channel'), float(vs[:, 0].min())))
        edges.append((g.get('y_channel'), float(vs[:, 1].min())))
    out = []
    for ch, v in edges:
        if v is None or not isinstance(ch, str):
            continue
        v = float(v)
        if v <= _OPEN_BOUND or (ch.lower() == 'time' and v <= 0):
            continue                     # meant open / nothing lies below 0 s
        if _flowjo_axis_is_linear(ch, display):
            opened, kept = ((_POLYGON_LINEAR_OPENED_UPTO, _POLYGON_LINEAR_KEPT_FROM)
                            if kind == 'polygon' else (_LINEAR_OPENED_UPTO, _LINEAR_KEPT_FROM))
        elif kind == 'polygon':
            opened, kept = -np.inf, _BIEX_KEPT_FROM      # polygons on biex: not measured
        else:
            opened, kept = _BIEX_OPENED_UPTO, _BIEX_KEPT_FROM
        if v < kept:
            out.append((ch, v, v <= opened))
    return out


# Import side of the same convention, for RECTANGLES only (1-D and 2-D),
# keyed by the axis each FlowJo <Sample> declares. Measured on linear axes of
# maxRange 262144, gain 1, for these minRange values -> (edge opened at or
# below, edge kept from):
_LINEAR_FLOORS_MEASURED = {
    0.0: (20.125, 64.0),       # 20.125 s on Time and 10 on SSC opened; 64 kept
    3.0: (10.0, 64.0),
    500.0: (500.0, 1000.0),
    -2000.0: (-2000.0, -1500.0),
}
# Every measured axis opened an edge AT its minRange; for another minRange
# only that much is applied and anything within 64 above it is logged.
# Polygons stay exact on import: FlowJo opens them from a different height
# than rectangles, and its in-range polygon rule is not identified (it is
# not exact point-in-polygon). On the six real round trips, clamping polygon
# events to FlowJo's axis range moved the largest FlowJo-OpenFlo gap only
# from 5.41% to 5.39%.


def _wsp_axis_floors(sample_elem):
    """{FlowJo parameter name: (open_at_or_below, kept_from)} for the axes a
    FlowJo ``<Sample>`` declares in ``<Transformations>`` (namespace-stripped
    tree), for the axis kinds the floor was measured on. Empty without such a
    block. A workspace OpenFlo exported (`wsp_written_by_openflo`) is not
    floored by the callers: it keeps OpenFlo's literal gate semantics on
    re-import, though it now declares linear axes."""
    out = {}
    if sample_elem is None:
        return out
    for block in sample_elem.findall('Transformations'):
        for t in block:
            p = t.find('parameter')
            name = p.get('name') if p is not None else None
            if not name:
                continue
            try:
                if (t.tag == 'linear' and float(t.get('maxRange', 'nan')) == 262144
                        and float(t.get('gain', 1)) == 1):
                    m = float(t.get('minRange', 0))
                    out[name] = _LINEAR_FLOORS_MEASURED.get(m, (m, m + 64.0))
                elif (t.tag == 'biex' and float(t.get('width', 'nan')) == -100
                      and float(t.get('maxRange', 'nan')) == 262144
                      and float(t.get('neg', 'nan')) == 0
                      and abs(float(t.get('pos', 'nan')) - 4.418539922) < 1e-6):
                    out[name] = (_BIEX_OPENED_UPTO, _BIEX_KEPT_FROM)
            except ValueError:
                continue
    return out


def _wsp_axis_floor_lo(param, lo, floors):
    """A FlowJo rectangle lower bound as OpenFlo must apply it: -1e12 (open)
    when FlowJo opens it, else unchanged. Between the measured bounds the
    edge stays literal and the uncertainty is logged."""
    f = floors.get(param) if param else None
    if lo is None or f is None or lo <= -1e12:
        return lo
    open_at, kept_from = f
    if lo <= open_at:
        return -1e12
    if lo < kept_from:
        log.info(f"[WspReader] {param}: lower edge {lo:g} lies where FlowJo's "
                 f"axis floor is not measured ({open_at:g}..{kept_from:g}); kept literal")
    return lo


# An ellipse drawn in FlowJo is not a Gating-ML mean/covariance ellipse: FlowJo
# writes two <foci> and four <edge> points in its 256 x 256 display space, and
# each display coordinate / 256 is the fraction of that axis's TRANSFORMED range
# (the form FlowKit's WSPEllipsoidGate documents; FlowKit, BSD-3-Clause, is the
# reference for the geometry below). Converting one to data units therefore
# needs the axis transform its <Sample> declares. Measured against FlowJo's own
# counts in two FlowJo-written workspaces (FlowKit's test data):
# - linear 0..262144 axes: 51 of FlowJo's 51 events. A linear axis maps display
#   to data affinely, so the ellipse is exactly an ellipse in data units too.
# - logicle axes (T 262144, W 1, M 4.42, A 0): 5869 where FlowJo counts 6002,
#   under a parent OpenFlo counts 0.18% below FlowJo. Matching that parent
#   closed 27 of the 133 events; the rest is not explained, so the import says
#   so. FlowKit's own test of the same workspace (another sample) counts 7023
#   where FlowJo counts 7256. 64, 128 and 512 polygon vertices gave the same
#   count.
# - biex axes (FlowJo's default for fluorescence) use FlowJo's biex lookup
#   table, see _flowjo_biex_lut. No FlowJo-drawn ellipse on a biex axis was
#   available to count against FlowJo. Against FlowKit, event for event on
#   the real ICS sample, with the ellipse at 20 placements on biex axes of
#   several parameter sets (some at the axis bottom or a corner): up to 139
#   events differ, at most 0.18% of the gate, all at the outline, with
#   OpenFlo slightly higher (a 64-gon of equal area here, FlowKit's 128-gon
#   inscribed).
# Any other axis -- log, fasinh, a linear gain other than 1, or no declared
# transform -- is not converted, and the gate becomes a 'flowjo_ellipse'
# placeholder: it admits no events, keeps its children under it, and keeps
# FlowJo's own points and axes so the export can write it back unchanged.
_FLOWJO_DISPLAY_BINS = 256.0
# FlowJo's biex lookup table has 4096 channels across the 256 display bins.
_FLOWJO_BIEX_CHANNELS = 4096


def _flowjo_log_root(b, w):
    """Root of 2 ln(d) + w d = 2 ln(b) - w b on (0, b], by Newton steps
    guarded by bisection. Ported line for line from FlowKit's `_log_root`
    (FlowKit, BSD-3-Clause, Copyright (c) 2018 Scott White), which ports
    cytolib's port of FlowJo's legacy Java code; kept identical so the table
    below matches FlowKit's and FlowJo's (see _flowjo_biex_lut)."""
    x_lo, x_hi = 0.0, b
    d = (x_lo + x_hi) / 2
    dx = abs(int(x_lo - x_hi))
    dx_last = dx
    fb = -2 * np.log(b) + w * b
    f = 2. * np.log(d) + w * b + fb
    df = 2 / d + w
    if w == 0:
        return b
    for _ in range(100):
        if (((d - x_hi) * df - f) - ((d - x_lo) * df - f)) > 0 \
                or abs(2 * f) > abs(dx_last * df):
            dx = (x_hi - x_lo) / 2
            d = x_lo + dx
            if d == x_lo:
                return d
        else:
            dx = f / df
            t = d
            d -= dx
            if d == t:
                return d
        if abs(dx) < 1.0E-12:
            return d
        dx_last = dx
        f = 2 * np.log(d) + w * d + fb
        df = 2 / d + w
        if f < 0:
            x_lo = d
        else:
            x_hi = d
    return d


@functools.lru_cache(maxsize=16)
def _flowjo_biex_lut(neg, width, pos, max_range, channels=_FLOWJO_BIEX_CHANNELS):
    """FlowJo 10's biex axis as a lookup table: (data value at each channel,
    channel numbers 0..channels). FlowJo maps between them by linear
    interpolation, and so does `_flowjo_display_axis`. `neg`, `width`, `pos`
    and `max_range` are the <transforms:biex> attributes of the same names
    (`width` is FlowJo's negative "width basis", eg -100).

    The arithmetic is FlowKit's `generate_biex_lut` (FlowKit, BSD-3-Clause,
    Copyright (c) 2018 Scott White; ported from cytolib, itself ported from
    FlowJo's legacy Java code), reproduced step for step: bit for bit equal
    to FlowKit's table for five parameter sets. It matches the lookup table
    FlowJo itself exports for width -7.943282, neg 1 (FlowKit's test data) at
    every one of its 4096 entries within one unit of the sixth significant
    digit FlowJo prints (largest relative difference 4.8e-6; FlowKit's table
    differs identically). Returns None for parameters it is not defined for."""
    if not (width < 0 and max_range > 0 and np.isfinite([neg, pos]).all()):
        return None
    ln10 = np.log(10.0)
    w = np.log10(-width)
    decades = pos - w / 2
    extra = max(neg, 0.0) + w / 2
    zero_point = int((extra * channels) / (extra + decades))
    zero_point = int(min(zero_point, channels / 2))
    if zero_point > 0:
        decades = extra * channels / zero_point
    w = w / (2 * decades)
    positive_range = ln10 * decades
    minimum = max_range / np.exp(positive_range)
    negative_range = _flowjo_log_root(positive_range, w)
    n = channels + 1
    idx = np.arange(n)
    positive = np.exp(idx / float(n) * positive_range)
    negative = np.exp(idx / float(n) * -negative_range)
    negative *= np.exp((positive_range + negative_range) * (w + extra / decades))
    s = positive[zero_point] - negative[zero_point]
    positive[zero_point:] = minimum * (positive[zero_point:] - negative[zero_point:] - s)
    below = np.arange(zero_point)
    positive[below] = -positive[2 * zero_point - below]
    if not (np.isfinite(positive).all() and (np.diff(positive) > 0).all()):
        return None
    return positive, idx.astype(float)


def _wsp_axis_elem(sample_elem, param):
    """The <Transformations> child that sets FlowJo parameter `param`'s axis
    in a (namespace-stripped) <Sample>, or None."""
    for block in (sample_elem.findall('Transformations') if sample_elem is not None else []):
        for t in block:
            p = t.find('parameter')
            if p is not None and p.get('name') == param:
                return t
    return None


def _flowjo_display_axis(sample_elem, param):
    """(kind, display -> data function, why) for one axis of FlowJo's display
    space: kind 'linear', 'logicle' or 'biex', or None with `why` when this
    axis cannot be converted. `param` is FlowJo's parameter name ('Comp-'
    kept), as the sample's <Transformations> names it."""
    t = _wsp_axis_elem(sample_elem, param)
    if t is None:
        return None, None, f"the sample declares no axis transform for {param}"
    try:
        if t.tag == 'linear' and float(t.get('gain', 1)) == 1:
            lo = float(t.get('minRange', 'nan'))
            hi = float(t.get('maxRange', 'nan'))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                # Display 0 is minRange. FlowJo 10.10.2 measured: on linear
                # axes it saved with minRange 500 and -2000, a rectangle edge
                # AT minRange was the axis bottom (opened) and one 500 above
                # was kept (see _LINEAR_FLOORS_MEASURED). FlowKit's
                # (x + minRange) / (maxRange + minRange) would put those axes'
                # bottoms at -500 and +2000 instead. With minRange 0 the two
                # agree.
                return ('linear',
                        lambda d: lo + (np.asarray(d, dtype=float)
                                        / _FLOWJO_DISPLAY_BINS) * (hi - lo),
                        '')
        if t.tag == 'logicle':
            lp: dict[str, Any] = {k.lower(): float(t.get(k, 'nan'))
                                  for k in ('T', 'W', 'M', 'A')}
            if all(np.isfinite(v) for v in lp.values()):
                return ('logicle',
                        lambda d: inverse_transform_values(
                            np.asarray(d, dtype=float) / _FLOWJO_DISPLAY_BINS,
                            'logicle', **lp),
                        '')
        if t.tag == 'biex' and float(t.get('length', 'nan')) == _FLOWJO_DISPLAY_BINS:
            lut = _flowjo_biex_lut(*(float(t.get(k, 'nan'))
                                     for k in ('neg', 'width', 'pos', 'maxRange')))
            if lut is not None:
                data, chan = lut
                # Off the axis, the table's end value: FlowJo has no table
                # entry beyond either end.
                return ('biex',
                        lambda d: np.interp(np.asarray(d, dtype=float)
                                            * (_FLOWJO_BIEX_CHANNELS / _FLOWJO_DISPLAY_BINS),
                                            chan, data),
                        '')
    except (TypeError, ValueError):
        pass
    kind = t.tag + (f" (gain {t.get('gain')})" if t.tag == 'linear' else '')
    return None, None, f"{param} is on a FlowJo {kind} axis OpenFlo cannot convert"


# A FlowJo workspace stores a polygon's vertices in data units, but FlowJo
# draws and evaluates its edges straight on the plot's axes: the transform
# each <Sample> declares per parameter in <Transformations>. On a logicle or
# biex axis a straight edge there is a curve in data units, so evaluating the
# same vertices with straight edges in data units counts another region
# (measured on public FlowJo 10 workspaces with logicle axes: 6 to 114 events
# per polygon differed from FlowKit, which evaluates on the axes; with the
# edges on the axes, none). An imported polygon therefore carries
# 'flowjo_edge_axes': per axis (x, y) the FlowJo axis its edges are straight
# on, or None for data units. gate_to_mask evaluates it there;
# map_gate_coords, which puts a gate on another scale, traces each edge on
# those axes first and drops the key. A linear axis, at any gain, keeps edges
# straight in data units; an undeclared axis (OpenFlo's own exports declare
# none) keeps them so too, so a workspace OpenFlo wrote reads back as written.
_EDGE_AXIS_STEPS = 64      # sub-segments per edge when a polygon is re-mapped


def _flowjo_edge_axis(sample_elem, param):
    """The axis FlowJo draws `param`'s polygon edges straight on, as a JSON
    spec (see `_edge_axis_fns`), or None when they are straight in data units
    (a linear axis, or no or an unreadable declaration). `param` is FlowJo's
    parameter name ('Comp-' kept), as <Transformations> names it."""
    t = _wsp_axis_elem(sample_elem, param) if param else None
    if t is None or t.tag == 'linear':
        return None

    def num(*keys):
        vals = [float(t.get(k, 'nan')) for k in keys]
        if not all(np.isfinite(v) for v in vals):
            raise ValueError(f"{t.tag} axis of {param}: missing attribute")
        return vals

    try:
        if t.tag == 'logicle':
            T, W, M, A = num('T', 'W', 'M', 'A')
            return {'kind': 'logicle', 'T': T, 'W': W, 'M': M, 'A': A}
        if t.tag == 'biex':
            neg, width, pos, max_range = num('neg', 'width', 'pos', 'maxRange')
            if _flowjo_biex_lut(neg, width, pos, max_range) is not None:
                return {'kind': 'biex', 'neg': neg, 'width': width,
                        'pos': pos, 'maxRange': max_range}
        elif t.tag == 'log':
            offset, decades = num('offset', 'decades')
            if offset > 0 and decades > 0:
                return {'kind': 'log', 'offset': offset, 'decades': decades}
        elif t.tag == 'fasinh':
            T, M, A = num('T', 'M', 'A')
            if T > 0 and M > 0:
                return {'kind': 'fasinh', 'T': T, 'M': M, 'A': A}
    except ValueError:
        pass
    log.warning("[WspReader] %s: FlowJo %s axis not understood; polygon "
                "edges on it are taken as straight in data units", param, t.tag)
    return None


def wsp_stat_axes(sample_elem):
    """{data column: FlowJo axis} that a (namespace-stripped) workspace
    <Sample> declares in <Transformations>, as `_flowjo_edge_axis` specs.
    FlowJo's GeoMean is a mean on this axis (gating.population_stats).

    A compensated parameter ('Comp-FL1-A', '<FL1-A>') wins over the bare,
    uncompensated one, as the data column holds compensated values. A linear
    axis, and one not understood, give no entry: that channel keeps OpenFlo's
    own display transform."""
    names = {}
    for block in (sample_elem.findall('Transformations')
                  if sample_elem is not None else []):
        for t in block:
            p = t.find('parameter')
            name = p.get('name') if p is not None else None
            if not name:
                continue
            col, marked = _strip_comp_marker(name)
            if marked or col not in names:
                names[col] = name
    out = {}
    for col, name in names.items():
        axis = _flowjo_edge_axis(sample_elem, name)
        if axis is not None:
            out[col] = axis
    return out


def stat_axes_from_json(rec):
    """A saved ``stat_axes`` record (see wsp_stat_axes) as {channel: axis},
    keeping only the axes `_edge_axis_fns` reads; {} for anything else."""
    if not isinstance(rec, dict):
        return {}
    return {str(c): dict(a) for c, a in rec.items()
            if isinstance(a, dict) and _edge_axis_fns(a) is not None}


def _edge_axis_fns(spec):
    """(forward, inverse) between data units and a 'flowjo_edge_axes' axis
    (see `_flowjo_edge_axis`), both vectorised and increasing; None for a
    spec that names no axis this reads."""
    if not isinstance(spec, dict):
        return None
    kind = spec.get('kind')
    try:
        if kind == 'logicle':
            p: dict[str, Any] = {'t': float(spec['T']), 'w': float(spec['W']),
                                 'm': float(spec['M']), 'a': float(spec['A'])}
            return ((lambda v: transform_values(v, 'logicle', **p)),
                    (lambda y: inverse_transform_values(y, 'logicle', **p)))
        if kind == 'biex':
            lut = _flowjo_biex_lut(float(spec['neg']), float(spec['width']),
                                   float(spec['pos']), float(spec['maxRange']))
            if lut is None:
                return None
            data, chan = lut
            # FlowJo's table; an event off either end sits at that end, as
            # FlowJo and FlowKit put it.
            return ((lambda v: np.interp(np.asarray(v, dtype=float), data, chan)),
                    (lambda y: np.interp(np.asarray(y, dtype=float), chan, data)))
        if kind == 'log':
            off, dec = float(spec['offset']), float(spec['decades'])
            # FlowJo's log axis: a value below the offset sits at the bottom.
            return ((lambda v: np.log10(np.maximum(np.asarray(v, dtype=float), off)
                                        / off) / dec),
                    (lambda y: off * np.power(10.0, np.asarray(y, dtype=float) * dec)))
        if kind == 'fasinh':
            from .gatingml import _fasinh, _fasinh_inv
            T, M, A = float(spec['T']), float(spec['M']), float(spec['A'])
            return ((lambda v: _fasinh(v, T, M, A)),
                    (lambda y: _fasinh_inv(y, T, M, A)))
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _wsp_polygon_edge_axes(gate_elem, sample_elem):
    """'flowjo_edge_axes' for a FlowJo <PolygonGate> in `sample_elem` (both
    namespace-stripped): [x axis, y axis], or None when its edges are
    straight in data units on both."""
    if sample_elem is None:
        return None
    params = []
    for d in list(gate_elem.iter('dimension'))[:2]:
        fd = next(d.iter('fcs-dimension'), None)
        params.append((fd.get('name') or fd.get('PnN')) if fd is not None else None)
    axes = [_flowjo_edge_axis(sample_elem, p) for p in params]
    return axes if any(a is not None for a in axes) else None


def _polygon_edge_fns(gate):
    """[(forward, inverse) or None for x, the same for y] for a polygon
    carrying 'flowjo_edge_axes' (None: that axis is in data units); None
    when the gate carries none this reads."""
    axes = gate.get('flowjo_edge_axes')
    if not isinstance(axes, (list, tuple)) or len(axes) != 2:
        return None
    fns = [_edge_axis_fns(a) if a is not None else None for a in axes]
    return fns if any(f is not None for f in fns) else None


def _flowjo_ellipse_points(ell_elem):
    """(foci, edge) of a FlowJo-drawn EllipsoidGate as (n, 2) float arrays in
    display units, or None when they are missing or not numbers."""
    def points(tag):
        el = ell_elem.find(tag)
        if el is None:
            return None
        out = []
        for v in el.findall('vertex'):
            cs = [c.get('value') for c in v.findall('coordinate')]
            if len(cs) != 2:
                return None
            try:
                out.append([float(cs[0]), float(cs[1])])
            except (TypeError, ValueError):
                return None
        return np.asarray(out, dtype=float).reshape(-1, 2)

    foci, edge = points('foci'), points('edge')
    if foci is None or edge is None:
        return None
    return foci, edge


def _flowjo_ellipse_display(foci, edge):
    """(centre, major-axis unit vector, a, b) of a FlowJo-drawn ellipse in
    display units, or None when its foci/edge points do not define one.

    The centre is mid-foci and the major axis runs through the foci. An edge
    point's distance from the centre is at most `a`, reached at the ends of the
    major axis, which FlowJo stores first; `b` follows from the foci
    (b^2 = a^2 - c^2), as in FlowKit, which found FlowJo's stored minor-axis
    points unreliable."""
    foci, edge = np.asarray(foci, dtype=float), np.asarray(edge, dtype=float)
    if (foci.shape != (2, 2) or edge.ndim != 2 or len(edge) == 0
            or not np.isfinite(foci).all() or not np.isfinite(edge).all()):
        return None
    centre = foci.mean(axis=0)
    half = (foci[1] - foci[0]) / 2.0
    c = float(np.hypot(*half))
    a = float(np.max(np.hypot(*(edge - centre).T)))
    if not a > c:
        return None
    u = half / c if c > 0 else np.array([1.0, 0.0])
    return centre, u, a, float(np.sqrt(a * a - c * c))


def _flowjo_native_ellipse(foci, edge, sample_elem, x_param, y_param):
    """A FlowJo-drawn EllipsoidGate (foci + edge, no mean) as a gate body in
    data units: {'kind': 'ellipsoid', ...} on two linear axes (exact),
    {'kind': 'polygon', ...} when an axis is logicle or biex, else None.
    Returns (body, why): `why` says what was approximated, or why there is no
    body."""
    geom = _flowjo_ellipse_display(foci, edge)
    if geom is None:
        return None, "its foci/edge points do not define an ellipse"
    centre, u, a, b = geom
    v = np.array([-u[1], u[0]])
    axes = [_flowjo_display_axis(sample_elem, p) for p in (x_param, y_param)]
    for kind, _f, why in axes:
        if kind is None:
            return None, why
    disp_cov = a * a * np.outer(u, u) + b * b * np.outer(v, v)
    (kx, fx, _), (ky, fy, _) = axes
    if kx == ky == 'linear':
        # Affine per axis, so exact: data = lo + s * display.
        lo = np.array([float(fx(0.0)), float(fy(0.0))])
        s = np.array([float(fx(1.0)), float(fy(1.0))]) - lo
        cov = np.outer(s, s) * disp_cov
        return {'kind': 'ellipsoid',
                'mean': [float(v_) for v_ in lo + s * centre],
                'cov': [[float(e) for e in row] for row in cov],
                'distance_sq': 1.0}, ''
    verts = _ellipse_polygon(centre, disp_cov, 1.0)
    if verts is None:
        return None, "its foci/edge points do not define an ellipse"
    vs = np.asarray(verts, dtype=float)
    n_drawn = len(vs)
    # A biex axis is a table: an event off either end sits at that end of the
    # axis (FlowKit clamps it there, as FlowJo puts off-scale events on the
    # axis edge). So the part of the ellipse beyond an end is cut off at the
    # end, and the cut is extended straight out to +-1e12, taking in every
    # event beyond that end between the two cut points. Mapping the off-axis
    # vertices to the table's end value instead left those events out: 352
    # where FlowKit counts 359 on an ellipse crossing the bottom of a width
    # -10 axis.
    opened = []
    for i, kind in enumerate((kx, ky)):
        if kind == 'biex':
            vs = _clip_polygon_axis(vs, i, 0.0, _FLOWJO_DISPLAY_BINS)
            opened.append(i)
    if len(vs) < 3:
        return None, "its outline lies entirely off the axis"
    data = np.column_stack([fx(vs[:, 0]), fy(vs[:, 1])])
    if not np.isfinite(data).all():
        return None, "its outline falls outside the axis transform's range"
    for i in opened:
        for bound, far in ((0.0, -1e12), (_FLOWJO_DISPLAY_BINS, 1e12)):
            vs, data = _open_polygon_at(vs, data, i, bound, far)
    return ({'kind': 'polygon', 'vertices': data.tolist()},
            f"converted from FlowJo's display space on {kx}/{ky} axes as a "
            f"{n_drawn}-vertex polygon; FlowJo's own count can differ (on the "
            "real FlowJo logicle ellipses measured, OpenFlo counted a few "
            "percent fewer events than FlowJo)")


def _open_polygon_at(vs, data, axis, bound, far):
    """Extend polygon `data` (data units; `vs` the same vertices in display
    units) beyond a cut at display coordinate `bound` on `axis`: each run of
    vertices on the cut keeps its two ends and gains copies moved to `far`
    on that axis, so the region beyond the axis end runs straight out from
    the cut points."""
    on = vs[:, axis] == bound
    if not on.any():
        return vs, data
    n = len(vs)
    out_v, out_d = [], []
    for k in range(n):
        if not on[k]:
            out_v.append(vs[k])
            out_d.append(data[k])
            continue
        moved = data[k].copy()
        moved[axis] = far
        pts = ([data[k]] if not on[k - 1] else []) + [moved] \
            + ([data[k]] if not on[(k + 1) % n] else [])
        out_v.extend([vs[k]] * len(pts))
        out_d.extend(pts)
    return np.asarray(out_v, dtype=float), np.asarray(out_d, dtype=float)


def _clip_polygon_axis(vs, axis, lo, hi):
    """Polygon `vs` ((n, 2) array) cut to lo <= coordinate `axis` <= hi
    (Sutherland-Hodgman, one axis); the cut points lie exactly on lo / hi."""
    def cut(pts, keep, bound):
        out = []
        for k in range(len(pts)):
            p, q = pts[k - 1], pts[k]
            if keep(q) != keep(p):
                r = p + (bound - p[axis]) / (q[axis] - p[axis]) * (q - p)
                r[axis] = bound
                out.append(r)
            if keep(q):
                out.append(q)
        return out
    pts = cut(list(np.asarray(vs, dtype=float)), lambda p: p[axis] >= lo, lo)
    pts = cut(pts, lambda p: p[axis] <= hi, hi) if pts else pts
    return np.asarray(pts, dtype=float).reshape(-1, 2)


def _strip_comp_marker(name):
    """(data column, compensated?) for a FlowJo parameter name. FlowJo names
    a legacy-compensated parameter 'Comp-FL5-A' and a natively compensated
    one '<FL5-A>' (angle brackets); both strip to the FCS column 'FL5-A'. The
    bare name is the UNcompensated parameter of a compensated sample, which
    FlowJo keeps beside the compensated one. (A fully configurable comp
    prefix/suffix is a rarer case not handled here.)"""
    marked = False
    if name.startswith('Comp-'):
        name, marked = name[len('Comp-'):], True
    if len(name) >= 2 and name[0] == '<' and name[-1] == '>':
        name, marked = name[1:-1], True
    return name, marked


def _wsp_comp_channels(gate_elem):
    """Sorted data columns that gate element `gate_elem` (a RectangleGate,
    PolygonGate, EllipsoidGate or QuadrantGate, namespace-stripped) names
    with FlowJo's compensated marker. Every gate read from a .wsp carries
    this as 'wsp_comp_channels' (its presence says the gate came from a
    workspace, where a bare name means something): stripping the marker merges
    'Comp-FL2-A' and the uncompensated 'FL2-A' into one column, and FlowJo
    evaluates the two on different values (a gate at 2,500: 5,216 events
    compensated, 8,413 not). openflo-compare evaluates the unmarked ones on
    the uncompensated data; the editor, which holds only compensated
    values, warns about them."""
    out = set()
    for fd in gate_elem.iter('fcs-dimension'):
        ch, marked = _strip_comp_marker(fd.get('name') or fd.get('PnN') or '')
        if marked and ch:
            out.add(ch)
    return sorted(out)


def wsp_uncompensated_channels(gate, compensated):
    """The channels of imported gate `gate` that FlowJo evaluates on the
    UNcompensated parameter although `compensated` (the sample's compensated
    channels) holds them: named without 'Comp-' / <...> in the workspace.
    [] for a gate without 'wsp_comp_channels' -- one not read from a .wsp
    (drawn, from a template, hand-built), whose names mean the data's
    columns (see _wsp_comp_channels)."""
    if 'wsp_comp_channels' not in gate:
        return []
    marked = set(gate.get('wsp_comp_channels') or ())
    comp = set(compensated or ())
    return sorted({gate.get(k) for k in ('channel', 'x_channel', 'y_channel')
                   if gate.get(k) in comp and gate.get(k) not in marked})


def _unread_tags(unread):
    """Gate tags named by `unread` (see _log_unread)."""
    return {t for _, tags, _ in unread for t in tags}


def _log_unread(where, unread):
    """Warn about the populations WspReader.extract_gates left out because
    their gate could not be read (see walk_population)."""
    if not unread:
        return
    under = sum(n for _, _, n in unread)
    shown = ', '.join(f"'{name}' ({'/'.join(tags)})"
                      for name, tags, _ in unread[:5])
    more = f' (+{len(unread) - 5})' if len(unread) > 5 else ''
    log.warning(
        "[WspReader] %s: %d population(s) with a gate OpenFlo cannot read "
        "were not imported: %s%s%s. They are left out rather than counted "
        "without their gate.", where, len(unread), shown, more,
        f"; nor were the {under} population(s) under them" if under else '')


def placeholder_booleans(gates):
    """Names of the boolean gates in `gates` (gate dicts of ONE id namespace:
    a template, or one sample's saved gates; ids in '_import_id' or 'id')
    that rest on a 'flowjo_ellipse' placeholder and so select nothing."""
    by_id = {}
    for g in gates:
        gid = g.get('_import_id') or g.get('id')
        if gid is not None:
            by_id[gid] = g
    return [g.get('name') or describe_gate(g) for g in gates
            if g.get('kind') == 'boolean' and any(
                depends_on_flowjo_placeholder(by_id, op)
                for op in g.get('operands') or ())]


def _flowjo_placeholder_note(gates, booleans=None):
    """Status-bar note for the 'flowjo_ellipse' placeholders in `gates`
    (gate dicts), or '' when there are none. It also names the boolean gates
    built on them, which select nothing too: `booleans`, or when None those
    `placeholder_booleans` finds in `gates`."""
    names = [g.get('label') or g.get('name') or '?' for g in gates
             if g.get('kind') == 'flowjo_ellipse']
    if not names:
        return ''
    shown = ', '.join(names[:3]) + (f' (+{len(names) - 3})' if len(names) > 3 else '')
    note = (f" — {len(names)} FlowJo ellipse(s) not converted ({shown}): "
            "they and their subpopulations count no events (see log)")
    booleans = list(dict.fromkeys(placeholder_booleans(gates)
                                  if booleans is None else booleans))
    if booleans:
        bshown = ', '.join(f"'{b}'" for b in booleans[:3]) + (
            f' (+{len(booleans) - 3})' if len(booleans) > 3 else '')
        note += f"; so do the boolean gate(s) built on them: {bshown}"
    return note


def _wsp_ellipse_gate(ell_elem, sample_elem):
    """A <EllipsoidGate> of a (namespace-stripped) FlowJo workspace as a gate
    body without ids: (body, why), body None when the element is unreadable.

    - Gating-ML 2.0 form (<mean>, <covarianceMatrix>, <distanceSquare>, a
      squared Mahalanobis radius): 'ellipsoid'.
    - Drawn in FlowJo (two <foci>, four <edge> points in display space):
      converted with the sample's axes by _flowjo_native_ellipse, `why`
      naming any approximation; where it cannot be converted, a
      'flowjo_ellipse' placeholder that admits no events and keeps FlowJo's
      points, element attributes and the two axes (`why` says why). Dropping
      it instead attached its children to the grandparent, so each child
      silently counted every event the ellipse had excluded (12,358 where
      2,405 were right).

    `sample_elem` is the <Sample> holding the gate (its <Transformations>);
    None converts nothing."""
    dims = list(ell_elem.findall('dimension'))
    if len(dims) != 2:
        return None, "it does not have two dimensions"
    xc = WspReader._channel_name(dims[0])
    yc = WspReader._channel_name(dims[1])
    if not xc or not yc:
        return None, "a dimension names no parameter"
    mean_elem = ell_elem.find('mean')
    if mean_elem is None and ell_elem.find('foci') is not None:
        # Keyed by FlowJo's own parameter names ('Comp-' kept), as
        # <Transformations> names them.
        params = []
        for d in dims:
            fd = next(d.iter('fcs-dimension'), None)
            params.append((fd.get('name') or fd.get('PnN')) if fd is not None else None)
        pts = _flowjo_ellipse_points(ell_elem)
        if pts is None:
            # Unreadable, but still FlowJo's ellipse: dropping it attached
            # its children to the grandparent, every event of 40,000 inside.
            body, why = None, "its foci/edge points cannot be read"
        else:
            body, why = _flowjo_native_ellipse(*pts, sample_elem, *params)
        if body is not None:
            return {**body, 'x_channel': xc, 'y_channel': yc}, why
        axes = []
        for p in params:
            t = _wsp_axis_elem(sample_elem, p)
            axes.append(None if t is None else
                        {'kind': t.tag, 'attrs': {k: v for k, v in t.attrib.items()}})
        raw = copy.copy(ell_elem)
        raw.tail = None
        return {'kind': 'flowjo_ellipse', 'x_channel': xc, 'y_channel': yc,
                'flowjo': {'parameters': params,
                           'foci': None if pts is None else pts[0].tolist(),
                           'edge': None if pts is None else pts[1].tolist(),
                           'axes': axes, 'why': why,
                           # For the export: the element exactly as read.
                           'xml': ET.tostring(raw, encoding='unicode')}}, why
    if mean_elem is None:
        return None, "it has neither a mean nor foci"
    try:
        mean_vals = [float(c.get('value')) for c in mean_elem.findall('coordinate')]
        cov = [[float(e.get('value')) for e in row.findall('entry')]
               for row in ell_elem.find('covarianceMatrix').findall('row')]
    except (AttributeError, TypeError, ValueError):
        return None, "its mean or covariance is not numbers"
    if len(mean_vals) != 2 or len(cov) != 2 or any(len(r) != 2 for r in cov):
        return None, "its mean or covariance is not 2-D"
    dsq_elem = ell_elem.find('distanceSquare')
    try:
        dist_sq = (float(dsq_elem.get('value'))
                   if dsq_elem is not None else 4.0)
    except (TypeError, ValueError):
        dist_sq = 4.0
    return {'kind': 'ellipsoid', 'x_channel': xc, 'y_channel': yc,
            'mean': mean_vals, 'cov': cov, 'distance_sq': dist_sq}, ''


def _now_for_wsp():
    """FlowJo's modDate format, eg 'Fri Mar 20 14:22:19 EDT 2026'."""
    import time
    return time.strftime('%a %b %d %H:%M:%S %Z %Y')


# ── Compensation matrix IO ────────────────────────────────────────────────────
#
# Polymorphic read / write so the GUI's matrix editor and the pipeline can
# treat all of these as a single concept:
#   .wsp   FlowJo workspace (any number of matrices; we return the first)
#   .csv / .tsv  Header row = destination channels, first column of each
#                data row = source channel (rows are put in header order by
#                that label). Also accepted: a header row with no corner cell
#                over unlabelled rows (FlowJo's export), and header-less
#                (all cells numeric; `channels` is then None and the caller
#                supplies the names).
#   .fcs   Reads the SPILL (BD FACSDiva) / $SPILLOVER (FCS 3.1) keyword.
#
# Every reader returns spillover as FRACTIONS (diagonal 1). A matrix written in
# percent (diagonal 100, as some exports are) is converted with a warning:
# taken as fractions it divides every compensated channel by ~100. Measured: a
# percent CSV left the compensated data at 0.01x its true value, in silence.

def _is_number(cell):
    try:
        float(str(cell).strip())
    except ValueError:
        return False
    return True


# How far a diagonal may sit from 1 (fractions) or 100 (percent) and still be
# read as that unit. Acquisition software writes exactly 1 or 100; this only
# absorbs rounding in a hand-edited or re-exported file.
_SPILL_DIAG_TOL = 0.05


def _spill_units_to_fractions(matrix, source='spillover matrix'):
    """`matrix` as FRACTIONS (diagonal ~1).

    A diagonal of ~100 in every row is a percent matrix and is divided by 100,
    with a warning. Any other diagonal (neither ~1 nor ~100) is returned as
    given but warned about loudly: compensation divides each channel by its
    diagonal entry, which is almost never what the file meant."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] \
            or not matrix.size:
        return matrix
    d = np.diag(matrix)
    if not np.all(np.isfinite(d)):
        return matrix
    if np.all(np.abs(d - 100.0) <= 100.0 * _SPILL_DIAG_TOL):
        log.warning(
            "  [!] %s is in PERCENT (diagonal %s) -- converted to fractions "
            "(divided by 100). Applied as fractions it would have scaled "
            "every compensated channel by ~0.01.",
            source, np.array2string(d, precision=4, separator=', '))
        return matrix / 100.0
    if not np.all(np.abs(d - 1.0) <= _SPILL_DIAG_TOL):
        log.warning(
            "  [!] %s has diagonal %s -- neither 1 (fractions) nor 100 "
            "(percent). It is used AS GIVEN, which divides each compensated "
            "channel by its diagonal entry: check the file's units before "
            "trusting the compensated values.",
            source, np.array2string(d, precision=4, separator=', '))
    return matrix


def _spill_channel_refs(chans, param_names, source, markers=None):
    """The channels a SPILL / $SPILLOVER keyword names, as $PnN names.

    FCS 3.1 says the keyword lists $PnN names, but some acquisition software
    writes the 1-based parameter NUMBERS instead (``3,3,4,5,...``). Those
    matched no column, so the matrix was skipped with only "No matching
    channels for compensation" in the log and the sample stayed
    uncompensated (measured: 2-21% median error on a 3-colour file). When
    every reference is an integer in 1..$PAR and none of them is itself a
    channel name, they are parameter numbers and are mapped to $PnN.

    Others write the MARKER names ($PnS) instead. When no reference is a
    $PnN name and each names exactly one parameter's $PnS (`markers`, in
    parameter order), they are mapped to those parameters' $PnN."""
    chans = [str(c).strip() for c in chans]
    names = [str(c) for c in (param_names or [])]
    if not chans or not names or any(c in names for c in chans):
        return chans
    marks = [str(m).strip() for m in (markers or [])]
    if marks and not all(c.isdigit() for c in chans):
        hits = []
        for c in chans:
            at = [k for k, m in enumerate(marks) if m and m == c]
            if len(at) != 1 or at[0] >= len(names):
                hits = []
                break
            hits.append(names[at[0]])
        if hits and len(set(hits)) == len(hits):
            log.info("  %s: SPILL names its channels by marker %s -> %s.",
                     source, chans, hits)
            return hits
    idx = []
    for c in chans:
        if not c.isdigit() or not 1 <= int(c) <= len(names):
            return chans
        idx.append(int(c))
    if len(set(idx)) != len(idx):
        return chans
    mapped = [names[k - 1] for k in idx]
    log.info("  %s: SPILL names its channels by parameter number %s -> %s.",
             source, chans, mapped)
    return mapped


def _parse_spill_keyword(spill, param_names, source, markers=None):
    """``N,ch1,...,chN,v11,...,vNN`` as ``(channels, matrix)``: channels as
    $PnN names (see `_spill_channel_refs`), matrix in fractions (see
    `_spill_units_to_fractions`). `param_names` are the file's $PnN names in
    parameter order. Raises CompensationError on a malformed value.

    Shared by FlowSample._parse_spillover and read_compensation_matrix, so
    auto_compensate, the compensation editor and the workspace export can
    never disagree about what a file carries."""
    parts = [x.strip() for x in str(spill).split(',')]
    try:
        n = int(parts[0])
    except ValueError as e:
        raise CompensationError(
            f"{source}: SPILL keyword header is not an integer: "
            f"{parts[0]!r}") from e
    if n < 1 or len(parts) < 1 + n + n * n:
        raise CompensationError(
            f"{source}: SPILL keyword promises {n} channels + "
            f"{n*n} values but only {len(parts)-1} entries provided")
    extra = [x for x in parts[1 + n + n * n:] if x]
    if extra:
        log.warning("  [!] %s: SPILL keyword has %d value(s) past its %dx%d "
                    "matrix -- ignored (%s)", source, len(extra), n, n,
                    ', '.join(extra[:4]) + (' ...' if len(extra) > 4 else ''))
    channels = _spill_channel_refs(parts[1:1 + n], param_names, source,
                                   markers)
    try:
        vals = [float(x) for x in parts[1 + n: 1 + n + n * n]]
    except ValueError as e:
        raise CompensationError(
            f"{source}: non-numeric value in SPILL matrix") from e
    matrix = np.asarray(vals, dtype=float).reshape(n, n)
    return channels, _spill_units_to_fractions(matrix, f"{source} SPILL")


def _spill_text(metadata):
    """The SPILL / $SPILLOVER value in an FCS TEXT dict, or None. FlowIO strips
    every '$' and lower-cases every keyword; other sources may not, so both
    spellings are normalised. SPILL wins when a file carries both, as FlowJo
    10.10.2 does (tests/test_compensation_fcs_spill.py)."""
    text = {str(k).lstrip('$').lower(): v for k, v in (metadata or {}).items()}
    return text.get('spill') or text.get('spillover') or None


def _fcs_param_markers(metadata):
    """$PnS marker names in parameter order ('' where none)."""
    text = {str(k).lstrip('$').lower(): v for k, v in (metadata or {}).items()}
    try:
        npar = int(str(text.get('par', '')).strip())
    except ValueError:
        return []
    return [str(text.get(f'p{k}s') or '').strip() for k in range(1, npar + 1)]


def _fcs_param_names(metadata, channel_names):
    """$PnN names in parameter order: from the TEXT keywords when present,
    else the loaded channel names (which FlowSample reads from $PnN)."""
    text = {str(k).lstrip('$').lower(): v for k, v in (metadata or {}).items()}
    names = list(channel_names or [])
    try:
        npar = int(str(text.get('par', '')).strip())
    except ValueError:
        npar = len(names)
    out = []
    for k in range(1, npar + 1):
        nm = text.get(f'p{k}n')
        if not nm and k <= len(names):
            nm = names[k - 1]
        out.append(str(nm).strip() if nm else f'Ch{k}')
    return out


def _read_matrix_csv(path):
    """`(channels, matrix)` from a CSV/TSV; channels None when header-less.

    Accepted layouts (row = source fluor, column = detector):
      * a header row with a corner cell (blank or a caption such as
        "Source/Detector") over one labelled row per source -- what
        write_compensation_matrix writes;
      * a header row of channel names only over rows of numbers only
        (FlowJo's export) -- this used to fail as "not square (shape
        (3, 2))", the first number of each row taken for its label;
      * numbers only (header-less).
    Row labels are honoured: rows are put in header order by label. They used
    to be ignored, so a file listing its rows in another order loaded the FL2
    row as the FL4 row without a word. Labels that are not exactly the
    header's channels are refused rather than guessed at."""
    sep = '\t' if path.lower().endswith('.tsv') else ','
    with open(path, encoding='utf-8', newline='') as f:
        # csv.reader, not str.split: a channel name containing the
        # separator — a $PnS description like "CD4 FL3, clone SK3" is
        # ordinary — split into two fields, and the read then failed with
        # "could not convert string to float: ' clone SK3'". Quoting is
        # what the format is for. Unquoted files written by earlier
        # versions parse identically.
        #
        # Don't strip the whole file: TSVs start with a leading tab to
        # align the header row above the row-label column, and that tab
        # is significant (an empty first cell ↔ "no row label here"). A
        # row of only empty cells is blank and skipped.
        rows = [r for r in csv.reader(f, delimiter=sep)
                if any(c.strip() for c in r)]
    if not rows:
        return None, None

    def cells(r):
        r = [c.strip() for c in r]
        while r and r[-1] == '':          # a trailing separator is not a cell
            r.pop()
        return r

    def numbers(r, where):
        try:
            return [float(c) for c in r]
        except ValueError as e:
            raise CompensationError(
                f"{path}: non-numeric value in {where}: {e}") from e

    source = os.path.basename(path)
    rows = [cells(r) for r in rows]
    if _is_number(rows[0][0]):
        widths = sorted({len(r) for r in rows})
        if widths != [len(rows)]:
            raise CompensationError(
                f"matrix in {path} is not square ({len(rows)} rows of "
                f"{'/'.join(map(str, widths))} values)")
        matrix = np.asarray([numbers(r, f'row {i + 1}')
                             for i, r in enumerate(rows)], dtype=float)
        return None, _spill_units_to_fractions(matrix, source)

    header, body = rows[0], rows[1:]
    if header[0].startswith('#'):
        # numpy.savetxt(header=...) writes '# FL1-A,FL2-A,...': the '#' is
        # its comment mark, not part of the first channel's name. Kept, it
        # named a channel no sample has ('# FL1-A'), and that channel was
        # left out of compensation without a word.
        header = [header[0][1:].strip()] + header[1:]
    widths = {len(r) for r in body}
    # A first column of row labels: the header's first cell is blank, or a
    # data row starts with a non-number. Exception: a blank corner over rows
    # exactly one cell narrower than the header, all numeric, carries none.
    labelled = header[0] == '' or any(r and not _is_number(r[0])
                                      for r in body)
    if (labelled and header[0] == '' and widths == {len(header) - 1}
            and all(_is_number(r[0]) for r in body if r)):
        labelled = False
        header = header[1:]
    channels = header[1:] if labelled else header
    if any(c == '' for c in channels):
        raise CompensationError(
            f"{path}: blank channel name in the header row {rows[0]}")
    n = len(channels)
    if len(body) != n:
        raise CompensationError(
            f"matrix in {path} is not square: {n} channel(s) in the header "
            f"but {len(body)} data row(s)")
    labels, mat = [], []
    for i, r in enumerate(body):
        vals = r[1:] if labelled else r
        if len(vals) != n:
            raise CompensationError(
                f"matrix in {path} is not square: row {i + 2} has "
                f"{len(vals)} value(s) for {n} channel(s)")
        if labelled:
            labels.append(r[0])
        mat.append(numbers(vals, f'row {i + 2}'))
    matrix = np.asarray(mat, dtype=float)
    if labelled and any(labels):
        if sorted(labels) != sorted(channels) or len(set(labels)) != n:
            raise CompensationError(
                f"{path}: the row labels {labels} are not the header's "
                f"channels {channels}. Each row is the spillover FROM the "
                "channel it is labelled with; a matrix whose rows and "
                "columns name different channels cannot be applied.")
        if labels != channels:
            log.info("  %s: rows listed in the order %s -- reordered to the "
                     "header's %s.", source, labels, channels)
            matrix = matrix[[labels.index(c) for c in channels]]
    return channels, _spill_units_to_fractions(matrix, source)


def read_compensation_matrix(path):
    """Returns `(channels, matrix)`. `channels` is a list[str] or None
    (only None for header-less CSVs); `matrix` is an NxN numpy array of
    FRACTIONS (a percent matrix is converted, with a warning), or
    ``(None, None)`` when the file is a legitimate FCS/WSP/CSV that simply
    has no spillover defined.

    Raises:
        CompensationError: malformed input (non-square matrix, channel /
            shape mismatch, row labels that are not the header's channels,
            unparseable SPILL keyword, unsupported extension).
    """
    p = path.lower()

    if p.endswith('.wsp'):
        reader = WspReader(path)
        # get_matrix() raises on a workspace without a matrix; this function
        # documents (None, None) for "a legitimate file with no spillover".
        if not reader.matrices:
            return None, None
        m = reader.get_matrix()
        return list(m['channels']), _spill_units_to_fractions(
            m['matrix'], f"{os.path.basename(path)} '{m.get('name', '')}'")

    if p.endswith(('.csv', '.tsv')):
        return _read_matrix_csv(path)

    if p.endswith('.fcs'):
        # "N,ch1,ch2,...,chN,v11,v12,...,vNN" under the bare SPILL keyword
        # (BD FACSDiva) or $SPILLOVER (FCS 3.1). FlowIO strips every '$' and
        # lower-cases every keyword, so they arrive as 'spill'/'spillover';
        # the upper-case lookups that were here never matched, and every FCS
        # read as "no spillover". The same keyword choice and parser as
        # FlowSample._parse_spillover, so this agrees with auto_compensate.
        try:
            fcs = read_fcs_tolerant(path, _fcs_read.FlowData, [])
        except Exception as e:
            raise FcsParseError(f"could not read FCS {path}: {e}") from e
        spill = _spill_text(fcs.text)
        if not spill:
            return None, None
        names = [fcs.channels[k].get('pnn') or fcs.channels[k].get('PnN')
                 or f'Ch{k}' for k in range(1, fcs.channel_count + 1)]
        return _parse_spill_keyword(
            spill, _fcs_param_names(fcs.text, names), path,
            markers=_fcs_param_markers(fcs.text))

    raise CompensationError(
        f"unsupported compensation file format: {path} "
        "(expected .wsp / .csv / .tsv / .fcs)")


def write_compensation_matrix(path, matrix, channels):
    """Write `matrix` (NxN numpy) labelled by `channels` (list[str]) to
    `path`. Format dispatched on extension:
      .wsp        -> WspWriter (matrix only; no samples / gates)
      .csv / .tsv -> header row + row-labelled rows
      .fcs        -> not supported (FCS spillover lives inside an FCS,
                     no use-case for writing one as a standalone file)
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square, got {matrix.shape}")
    if len(channels) != matrix.shape[0]:
        raise ValueError(
            f"{len(channels)} channels vs {matrix.shape} matrix")
    p = path.lower()
    if p.endswith('.wsp'):
        w = WspWriter()
        w.set_compensation(list(channels), matrix)
        w.write(path)
        return
    if p.endswith(('.csv', '.tsv')):
        sep = '\t' if p.endswith('.tsv') else ','
        with open(path, 'w', encoding='utf-8', newline='') as f:
            # csv.writer quotes any channel name containing the separator, so
            # the file survives a name like "CD4 FL3, clone SK3". Joining
            # by hand produced a file this module could not read back.
            w = csv.writer(f, delimiter=sep, lineterminator='\n')
            w.writerow([''] + list(channels))
            for i, ch in enumerate(channels):
                w.writerow([ch] + [repr(float(matrix[i, j]))
                                   for j in range(matrix.shape[1])])
        return
    raise ValueError(f"unsupported compensation file format for writing: {path}")


# ── Compensation matrix optimizer ─────────────────────────────────────────────
#
# Single-stain estimate, as the FlowJo / Diva compensation wizards make it:
# split each control's primary channel into its negative and positive
# populations and take, for every other detector,
#
#     spill = (median_pos(dst) - median_neg(dst)) / (median_pos(src) - median_neg(src))
#
# Diagonals stay 1.0; rows are the source fluor, columns the detector.
#
# This replaced a least-squares slope of dst on src fitted within the
# brightest 5% of src, which was biased two ways (measured, true 0.20):
# * Errors in variables. Within a tight bead peak most of the src spread is
#   noise, which flattens the slope: 0.158 (and 0.124 with 90% positive
#   beads). The medians use the whole distance from the negative to the
#   positive population, where the noise is negligible: 0.1998.
# * Saturation. Events piled at the detector ceiling sit at the top of the
#   tail with their src clipped and dst not: 0.274 with 0.9% of events
#   saturated, 0.566 with 2.8%, and 0 with 9.9%. They are dropped first.
# The negative population is the WHOLE of it, not the dimmest tail: with
# autofluorescence correlated across detectors, the dimmest events by src are
# also dim in dst, which read a true 0.010 as 0.020 (bottom 5%) or 0.016
# (bottom 25%). Taken whole, 0.0105. The split is spectral's
# _split_negative_positive, which the spillover spread matrix uses too.


def optimize_compensation(channels, single_stain_paths,
                          positive_percentile=95.0, min_events=100):
    """Estimate an NxN spillover matrix from per-channel single-stain controls.

    Parameters
    ----------
    channels : list[str]
        Fluor channel names in the order the matrix rows / columns will use.
    single_stain_paths : dict[str, str]
        Maps each source channel name to an FCS file in which only that
        fluor is bright. Channels with no entry contribute their identity
        row only.
    positive_percentile : float
        Used only for a control whose primary channel shows no separate
        negative and positive populations: the events above this percentile
        are then compared with those below ``100 - positive_percentile``, and
        a warning says the estimate is approximate.
    min_events : int
        Minimum number of events required in each of the negative and the
        positive population to estimate a row.

    Each row ``i`` is ``(median_pos - median_neg)`` of every detector over
    that of the source channel, the populations split on the source channel
    (see the module comment above). Events piled at any matrix channel's
    ceiling (saturated) are left out first. Returns (channels, matrix);
    diagonal entries are always 1.0, negative estimates are clamped to 0.
    """
    from .spectral import _split_negative_positive
    n = len(channels)
    matrix = np.eye(n, dtype=float)
    diag_report = {}
    for i, src_ch in enumerate(channels):
        path = single_stain_paths.get(src_ch)
        if not path:
            continue
        try:
            fcs = read_fcs_tolerant(path, _fcs_read.FlowData, [])
        except Exception as exc:
            log.info(f"[optimize] {src_ch}: load failed ({exc})")
            continue
        ch_names = [fcs.channels[k].get('pnn') or fcs.channels[k].get('PnN')
                    or f'Ch{k}' for k in range(1, fcs.channel_count + 1)]
        events = np.reshape(np.asarray(fcs.events),
                            (-1, fcs.channel_count)).astype(float)
        if src_ch not in ch_names:
            log.info(f"[optimize] {src_ch}: not present in {os.path.basename(path)}")
            continue
        used = [c for c in channels if c in ch_names]
        cols = [ch_names.index(c) for c in used]
        frame = pd.DataFrame(events[:, cols], columns=pd.Index(used))
        ok = np.isfinite(frame.to_numpy()).all(axis=1)
        saturated = AcquisitionQC._margin_events(frame, used, 0.0)
        if saturated.any():
            log.info("[optimize] %s: %d of %d events saturated (at a "
                     "detector's ceiling) are left out.", src_ch,
                     int(saturated.sum()), len(frame))
        ok &= ~saturated
        x = frame[src_ch].to_numpy()[ok]
        if x.size < 2 * min_events:
            log.info(f"[optimize] {src_ch}: only {x.size} usable events; skipped")
            continue
        neg, pos, separated = _split_negative_positive(x)
        if not separated:
            log.warning(
                "[optimize] %s: %s shows no separate negative and positive "
                "population on %s, so its spillover is estimated from the "
                "brightest vs the dimmest %g%% of events and is approximate. "
                "A control with a distinct negative gives an exact estimate.",
                src_ch, os.path.basename(path), src_ch,
                100.0 - positive_percentile)
            pos = x > np.percentile(x, positive_percentile)
            neg = x < np.percentile(x, 100.0 - positive_percentile)
        n_pos, n_neg = int(pos.sum()), int(neg.sum())
        if n_pos < min_events or n_neg < min_events:
            log.info(f"[optimize] {src_ch}: {n_pos} positive / {n_neg} "
                     "negative events; skipped")
            continue
        dx = float(np.median(x[pos]) - np.median(x[neg]))
        if not dx > 0:
            log.info(f"[optimize] {src_ch}: positive and negative do not "
                     "differ on the source channel; skipped")
            continue
        diag_report[src_ch] = (n_pos, n_neg)
        for j, dst_ch in enumerate(channels):
            if j == i or dst_ch not in used:
                continue
            y = frame[dst_ch].to_numpy()[ok]
            slope = float(np.median(y[pos]) - np.median(y[neg])) / dx
            matrix[i, j] = max(0.0, slope)   # clamp negatives to 0
    log.info(f"[optimize] (positive, negative) events per source: {diag_report}")
    return list(channels), matrix


def _template_ids(raw):
    """Imported gates (`_import_id` keys, see WspReader.extract_gates) with
    ids rewritten to the gN form, so the returned shape is consistent
    regardless of source."""
    from .gating import remap_operands
    imp_to_id = {g.get('_import_id'): f'g{i}' for i, g in enumerate(raw, 1)}
    gates = []
    for g in raw:
        d = dict(g)
        d.pop('_import_id', None)
        d['id'] = imp_to_id[g.get('_import_id')]
        pid = g.get('parent_id')
        d['parent_id'] = imp_to_id.get(pid) if pid else None
        if d.get('kind') == 'boolean':
            remap_operands(d, imp_to_id)
        gates.append(d)
    return gates


def _wsp_template_node(reader):
    """(gates, <SampleNode>) of the sample a `.wsp` template takes its gate
    tree from: the first that has gates; ([], None) when none has."""
    for sn in (reader.root.iter('SampleNode')
               if reader.root is not None else ()):
        raw = reader.extract_gates(sample_node=sn)
        if raw:
            return raw, sn
    return [], None


def wsp_template_stat_axes(path):
    """`wsp_stat_axes` of the sample a `.wsp` template's gates come from
    (see read_template_gates), so a sample the template is applied to takes
    its GeoMean on the same axes; {} when no sample holds gates."""
    reader = WspReader(path)
    _raw, used = _wsp_template_node(reader)
    if used is None or reader.root is None:
        return {}
    for sample in reader.root.iter('Sample'):
        if any(sn is used for sn in sample.iter('SampleNode')):
            return wsp_stat_axes(sample)
    return {}


def read_template_gates(path):
    """Read gates from a v2 JSON template, a FlowJo `.wsp` workspace or a
    standalone Gating-ML 2.0 `.xml` file.

    Returns `(gates, labels)` where:
      * `gates` is a list[dict] of gate definitions. Each entry has an
        `id` field (allocated `g1`, `g2`, … as needed) and a `parent_id`
        that references another id within the same list (or None for a
        root). Schema otherwise matches `gate_to_mask`.
      * `labels` is a {detector: antibody_label} dict from the template,
        or None if the source doesn't carry labels.

    Callers that want their own id namespace (the editor allocates ids
    per sample) can remap by walking the list in order and substituting
    parent_id pointers via an old->new map.

    A `.wsp` holds a gate tree per sample; the template is ONE of them, the
    first sample's that has gates (the log names it). Concatenating every
    sample's tree, as this used to, gave a 2-sample workspace of 5 gates as
    10, and each target sample got every population twice.

    `.wsp` gates are in linear units. Gating-ML gates stay on the file's own
    scales, each with its '_gml_axes': put them on a sample's stored scales
    with `gatingml.gates_to_stored`.
    """
    import json as _json
    p = path.lower()
    if p.endswith('.xml'):
        # Standalone Gating-ML 2.0: one gate tree, kept on the file's own
        # scales (each gate's '_gml_axes'); gatingml.gates_to_stored puts it
        # on a sample's.
        from .gatingml import read_gatingml
        raw, _warnings = read_gatingml(path, raw=True)
        return _template_ids(raw), None
    if p.endswith('.wsp'):
        reader = WspReader(path)
        nodes = (list(reader.root.iter('SampleNode'))
                 if reader.root is not None else [])
        raw, used = _wsp_template_node(reader)
        if used is None:
            # No sample holds a <Population> tree: the reader's flat scan
            # of bare gates, as before.
            raw = reader.extract_gates()
        else:
            others = sum(1 for sn in nodes if sn is not used
                         and next(sn.iter('Population'), None) is not None)
            log.info("[template] %s: the gate tree of sample '%s' (%d "
                     "gate(s))%s", os.path.basename(path),
                     used.get('name') or '?', len(raw),
                     f"; {others} other sample(s) hold their own tree, not "
                     "used" if others else '')
        return _template_ids(raw), None

    if p.endswith('.json'):
        with open(path, encoding='utf-8') as f:
            data = _json.load(f)
        if not isinstance(data, dict):
            raise ValueError("template JSON must be an object")
        gates_field = data.get('gates')
        if not isinstance(gates_field, list):
            raise ValueError(
                "template JSON must have a 'gates' field of type list")
        labels = (data.get('labels')
                  if isinstance(data.get('labels'), dict) else None)
        out = [dict(g) for g in gates_field
               if isinstance(g, dict) and g.get('kind')]
        return out, labels

    raise ValueError(f"unsupported template format: {path}")


# ── Transforms ─────────────────────────────────────────────────────────────
#
# Display/analysis transforms for fluorescence channels. logicle + log were
# the originals; asinh and hyperlog round out FlowJo parity, plus a 'linear'
# pass-through. asinh is parametrised by the intuitive `cofactor`
# (arcsinh(x / cofactor)); the others share FlowJo's t/m/w/a knobs.
TRANSFORM_METHODS = ('logicle', 'hyperlog', 'asinh', 'log', 'linear')


@functools.lru_cache(maxsize=8)
def _biexp_lut(method, t, m, w, a, n=16384):
    """Monotone (linear-value -> transformed-scale) lookup table for the GPU
    forward logicle / hyperlog transform.

    Built from flowutils' EXACT inverse sampled uniformly in SCALE, so it's dense
    exactly where the transform is steep (the linear region near 0). Interpolating
    against it reproduces flowutils' forward transform to ~1e-8 of scale (see the
    Phase-1 spike), which is far below any gating tolerance. Cached per parameter
    set. The scale span [-1.0, 1.1] maps to data ~[-2.6e6 .. 7.4e5]; a value
    outside it is transformed exactly by transform_values, not clamped."""
    inv = (transforms.logicle_inverse if method == 'logicle'
           else transforms.hyperlog_inverse)
    s = np.linspace(-1.0, 1.1, n)
    d = inv(s.reshape(-1, 1), channel_indices=[0],
            t=t, m=m, w=w, a=a).flatten()
    return d, s


def transform_values(values, method='logicle', t=262144, m=4.5, w=0.5, a=0,
                     cofactor=150.0):
    """Transform a 1-D array by `method`. Pure; returns a new array.

    logicle / hyperlog : FlowJo biexponential family (t/m/w/a).
    asinh              : arcsinh(x / cofactor) — cofactor ~150 (fluor),
                         ~5 (mass cytometry).
    log                : log10, clamped at >0 (0 elsewhere).
    linear             : pass-through.

    When GPU acceleration is enabled (Preferences), logicle / hyperlog use a
    flowutils-derived LUT + GPU interp (~1e-8 match); otherwise the exact
    flowutils path runs — so the default (GPU off) is bitwise unchanged.
    """
    v = np.asarray(values, dtype=float)
    if method in ('logicle', 'hyperlog'):
        from . import gpu_accel
        # The biexponential backends map every NON-FINITE input to the BOTTOM
        # of the scale (-1.0) — NaN, +inf and absurd magnitudes alike. That is
        # silent corruption twice over: a maximally bright reading is rendered
        # and gated as maximally DIM, and, worse, a NaN becomes a finite
        # coordinate, so the `dropna` that protects the plot and the gate
        # masks no longer removes it. The event then counts as a real, very
        # negative measurement in every population, median and frequency.
        # asinh already propagates NaN correctly; this makes the family
        # consistent. Finite in-range values are untouched.
        finite = np.isfinite(v)
        vv = np.where(finite, v, 0.0)                 # placeholder, masked out
        fn = (transforms.logicle if method == 'logicle'
              else transforms.hyperlog)
        if gpu_accel.enabled():
            d, s = _biexp_lut(method, t, m, w, a)
            out = np.array(gpu_accel.interp(vv, d, s), dtype=float)
            # The table spans scale [-1, 1.1], data ~[-2.6e6, 7.4e5] at the
            # defaults; interp clamps outside it, so every value above 7.4e5
            # read 1.1 (2e6 is 1.196 exactly, 4e6 1.263) and its inverse came
            # back as 7.4e5. A 22-bit spectral cytometer reaches 4.2e6 and
            # compensation can push past range, so those few events take the
            # exact path instead of a clamped coordinate.
            off = (vv < d[0]) | (vv > d[-1])
            if off.any():
                out[off] = fn(vv[off].reshape(-1, 1), channel_indices=[0],
                              t=t, m=m, w=w, a=a).flatten()
        else:
            out = fn(vv.reshape(-1, 1), channel_indices=[0],
                     t=t, m=m, w=w, a=a).flatten()
        if not finite.all():
            out = np.where(finite, out, np.nan)
        return out
    if method == 'asinh':
        from . import gpu_accel
        return gpu_accel.arcsinh(v, cofactor)
    if method == 'log':
        # A non-positive value has no logarithm. Folding it to 0.0 put it at
        # raw intensity 1.0 — ABOVE every genuinely positive event dimmer than
        # 1.0 — and piled a third of a compensated channel onto that single
        # coordinate. Measured: a gate at log > -1.0 selected 99,995 of
        # 100,000 events where 67,375 was correct, so 32,620 negative events
        # were called positive; the artificial spike is also what
        # auto_threshold and _bimodal_valley lock onto. NaN instead, matching
        # what the logicle/hyperlog branch above already does for input it
        # cannot place, so `dropna` keeps these events out of gates and
        # medians rather than inventing a coordinate for them.
        pos = v > 0
        if not pos.all():
            log.warning(
                "log transform: %d of %d value(s) are non-positive and have "
                "no logarithm (NaN) — use logicle or hyperlog to represent "
                "negative compensated values.", int((~pos).sum()), v.size)
        safe = np.log10(np.clip(np.where(pos, v, 1.0), 1e-6, None))
        return np.where(pos, safe, np.nan)
    if method == 'linear':
        return v
    raise ValueError(f"Unknown transform method '{method}'.")


def inverse_transform_values(values, method='logicle', t=262144, m=4.5,
                             w=0.5, a=0, cofactor=150.0):
    """Invert `transform_values` — map a transformed channel back to its
    (compensated) linear scale. Used to re-transform a channel from one
    method to another without re-running compensation. 'log' is not
    perfectly invertible where it clamped to 0; that region maps to ~0."""
    v = np.asarray(values, dtype=float)
    if method == 'logicle':
        return transforms.logicle_inverse(v.reshape(-1, 1), channel_indices=[0],
                                          t=t, m=m, w=w, a=a).flatten()
    if method == 'hyperlog':
        return transforms.hyperlog_inverse(v.reshape(-1, 1),
                                           channel_indices=[0],
                                           t=t, m=m, w=w, a=a).flatten()
    if method == 'asinh':
        return np.sinh(v) * float(cofactor)
    if method == 'log':
        # `v` here is ALREADY log-scaled, and a log-scaled value is negative
        # for every raw intensity below 1.0 — log10(0.5) is -0.301. The old
        # `v > 0` guard confused the transformed value with the raw one and
        # sent all of those to 0.0, so a round trip destroyed every event
        # dimmer than raw 1.0 (measured: 0.01, 0.1, 0.5 and 1.0 all came back
        # as 0.0). That runs whenever a channel's scale is changed, which does
        # inverse-then-forward. 10**v is the inverse on the whole real line;
        # NaN stays NaN.
        return np.power(10.0, v)
    if method == 'linear':
        return v
    raise ValueError(f"Unknown transform method '{method}'.")


def transform_spec(method='logicle', t=262144, m=4.5, w=0.5, a=0,
                   cofactor=150.0):
    """The arguments of one `transform_values` call, as the dict both it and
    `inverse_transform_values` accept. A FlowSample records one per transformed
    channel (``data_transforms``) so the transform baked into its data can be
    undone exactly, parameters included."""
    return {'method': method, 't': t, 'm': m, 'w': w, 'a': a,
            'cofactor': cofactor}


# A column known to hold display-TRANSFORMED values whose transform is not
# known: an OpenFlo CSV written before transform records existed, outside
# the logicle range. Its values are kept exactly as stored (so gates select
# what they did), never re-transformed, and no linear statistic is reported.
UNKNOWN_SPEC = {'method': 'unknown'}


class UnknownScaleError(ValueError):
    """A channel's linear values were asked for, but its display transform is
    unknown (see UNKNOWN_SPEC), so they cannot be recovered."""


def is_unknown_spec(spec):
    """True for a record entry saying the transform is unknown."""
    return bool(spec) and spec.get('method') == 'unknown'


def unknown_spec(origin=None):
    """An UNKNOWN_SPEC naming where the unknown scale was found. Two columns
    of unknown scale are only known to share it when their origins match:
    ``'session:<data folder>'`` for one older session's sidecars (one 2.6.1
    editor, one transform per channel), ``'file:<csv>'`` for anything else.
    The origin rides along through re-saves, so it keeps meaning that."""
    return dict(UNKNOWN_SPEC, **({'origin': str(origin)} if origin else {}))


# The keys of a record entry that ARE transform parameters. An entry may also
# carry 'guessed' (below) or an unknown one's 'origin'; neither may reach a
# transform call.
_SPEC_PARAMS = ('method', 't', 'm', 'w', 'a', 'cofactor')


def transform_params(spec):
    """`spec`'s transform parameters only -- what transform_values /
    inverse_transform_values / FlowSample.retransform take."""
    return {k: spec[k] for k in _SPEC_PARAMS if k in (spec or {})}


def is_guessed_spec(spec):
    """True for a record entry whose transform was GUESSED: a column of an
    OpenFlo CSV with no record whose values all lie in the logicle range.
    Its linear values are a guess too -- right for the default-logicle
    case, ~20x wrong for a dim channel the old editor had on asinh -- so
    nothing may present them as certain."""
    return bool(spec) and spec.get('guessed') is True


def guessed_spec(spec):
    """`spec`, marked as guessed (see is_guessed_spec)."""
    return dict(spec, guessed=True)


def to_linear(values, spec):
    """Undo the transform `spec` (a `transform_spec` dict) on `values`.
    ``None`` or a 'linear' spec means the values are already linear; an
    unknown transform gives NaN (nothing linear can be said about it)."""
    v = np.asarray(values, dtype=float)
    if not spec or spec.get('method', 'linear') == 'linear':
        return v
    if is_unknown_spec(spec):
        return np.full(v.shape, np.nan)
    return inverse_transform_values(v, **transform_params(spec))


def linear_values(sample, channel, values=None):
    """`channel`'s compensated LINEAR values in `sample`.

    The editor's loader transforms each fluor channel IN PLACE (QC ->
    compensate -> logicle), so ``sample.data`` holds display coordinates. A
    tool that models intensity -- a G2 = 2 x G1 DNA fit, a least-squares
    unmix, a straight MESF line, a mean or a CV -- needs the linear values
    those coordinates encode. This undoes the transform the sample recorded
    for the channel (``data_transforms``); a channel with no record, or a
    sample without one, is already linear. `values` replaces the column (e.g.
    a row subset of it).

    Inverting on demand rather than keeping a linear copy: logicle_inverse
    measured 0.04 s per 2M events and round-trips to 2e-9 absolute, while a
    kept copy would double the resident data the loader works to halve."""
    if values is None:
        values = sample.data[channel].values
    spec = (getattr(sample, 'data_transforms', None) or {}).get(channel)
    if is_unknown_spec(spec):
        raise UnknownScaleError(
            f"'{channel}' in '{getattr(sample, 'name', '?')}' holds values on "
            "an unknown display scale (an OpenFlo CSV written before transform "
            "records), so its linear values cannot be recovered.")
    return to_linear(values, spec)


def _matmul_known(values, matrix):
    """``values @ matrix`` where a non-finite input spoils only the outputs
    it feeds. A plain product spreads it through zero coefficients too, as
    NaN * 0 is NaN: one unknown value on a channel with no spillover would
    blank every channel of its event."""
    values = np.asarray(values, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    bad = ~np.isfinite(values)
    out = np.where(bad, 0.0, values) @ matrix
    if bad.any():
        out[(bad.astype(float) @ (matrix != 0).astype(float)) > 0] = np.nan
    return out


# ── Transform records beside a written CSV ────────────────────────────────────
#
# A CSV carries values, not the transform baked into them. Every event-table
# CSV OpenFlo writes (session sidecar, workspace _events.csv, pipeline
# _processed.csv, CLI _unmixed.csv) gets a small JSON record beside it so a
# reader inverts exactly what was applied instead of guessing.

TRANSFORMS_FORMAT = 'openflo-data-transforms'
TRANSFORMS_VERSION = 1

# Headroom past the logicle's top of scale for the no-record rule, in decades.
# 16 x T is a 22-bit spectral cytometer's range, and compensation can push a
# value a few-fold past range; two decades (100 x T) covers both.
_LOGICLE_HEADROOM_DECADES = 2


def transforms_sidecar_path(csv_path):
    """``<stem>_transforms.json`` beside `csv_path`.

    Named from the CSV's FULL stem (``X_events.csv`` ->
    ``X_events_transforms.json``). Stripping ``_events`` / ``_processed``, as
    the ``_labels.json`` lookup does, would hand two CSVs of one folder the
    same record: a session's samples 'a' and 'a_events' both map to
    ``a_transforms.json``."""
    return os.path.splitext(csv_path)[0] + '_transforms.json'


def write_transforms_sidecar(csv_path, transforms, columns, n_rows,
                             corrections=None):
    """Record, beside the CSV at `csv_path`, the transform each of its
    `columns` carries. `transforms` is a ``data_transforms`` dict (partial
    specs are completed with `transform_spec`'s defaults); a column it does
    not name is linear. The column list and the row count `n_rows` are
    stored too, so a reader can tell a record that belongs to another CSV --
    another sample's export with the same panel matches the columns alone.
    The record also names the OpenFlo version that wrote the CSV (and
    ``corrections``): the CSV itself stays a plain table, as every reader
    of it, OpenFlo's own included, expects. Returns the record's path."""
    import json

    from .export_stamp import export_metadata
    stamp = export_metadata(corrections)
    cols = [str(c) for c in columns]
    keep = {}
    for ch, spec in (transforms or {}).items():
        # A linear entry says nothing -- unless it is a guess, which the
        # next reader must still be told.
        if (str(ch) in cols and spec
                and (spec.get('method', 'linear') != 'linear'
                     or is_guessed_spec(spec))):
            if is_unknown_spec(spec):
                keep[str(ch)] = unknown_spec(spec.get('origin'))
                continue
            full = transform_spec(**transform_params(spec))
            keep[str(ch)] = guessed_spec(full) if is_guessed_spec(spec) else full
    path = transforms_sidecar_path(csv_path)
    tmp = path + '.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump({'format': TRANSFORMS_FORMAT,
                       'version': TRANSFORMS_VERSION,
                       'columns': cols, 'rows': int(n_rows),
                       'transforms': keep,
                       'openflo_version': stamp['OpenFlo version'],
                       **({'corrections': stamp['Corrections']}
                          if 'Corrections' in stamp else {})},
                      fh, indent=1, default=float)
        os.replace(tmp, path)          # atomic: the record is whole or absent
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise
    return path


def discard_transforms_sidecar(csv_path):
    """Remove the record beside `csv_path`. Every writer calls this BEFORE
    it rewrites the CSV: if the new record then fails to write, no record is
    better than the previous one, which describes other data (measured: an
    asinh column re-saved and then inverted with its old logicle record,
    median 123.5 where 19,922 is right)."""
    with contextlib.suppress(FileNotFoundError):
        os.remove(transforms_sidecar_path(csv_path))


def recorded_sidecar_rows(csv_path):
    """The row count the transform record beside `csv_path` was written
    for, or None when there is no readable record. A sidecar holding fewer
    rows than that was cut short (a crash mid-write)."""
    import json
    try:
        with open(transforms_sidecar_path(csv_path), encoding='utf-8') as fh:
            rec = json.load(fh)
        return int(rec['rows'])
    except (OSError, ValueError, TypeError, KeyError):
        return None


def read_transforms_sidecar(csv_path, columns, n_rows):
    """The ``data_transforms`` dict recorded beside `csv_path` by
    `write_transforms_sidecar`, or ``None`` when there is no record.

    Raises ValueError for a record that cannot be trusted: unreadable, not
    this format, from a newer OpenFlo, naming a transform this build does not
    know, or written for a column list other than `columns` or a row count
    other than `n_rows` (a stale record beside an edited or replaced CSV)."""
    import json
    path = transforms_sidecar_path(csv_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding='utf-8') as fh:
            rec = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ValueError(f"unreadable ({exc})") from exc
    if not isinstance(rec, dict) or rec.get('format') != TRANSFORMS_FORMAT:
        raise ValueError("not an OpenFlo transform record")
    if int(rec.get('version') or 0) > TRANSFORMS_VERSION:
        raise ValueError("written by a newer OpenFlo")
    if list(rec.get('columns') or []) != [str(c) for c in columns]:
        raise ValueError("written for different columns")
    if rec.get('rows') != int(n_rows):
        raise ValueError(f"written for {rec.get('rows')} rows; the CSV has "
                         f"{int(n_rows)}")
    return transform_record_from_json(rec.get('transforms') or {})


def transform_record_from_json(obj):
    """A ``data_transforms`` dict rebuilt from its JSON form
    (``{column: spec}``): each spec checked -- a known method, numeric
    parameters, 'guessed' a boolean -- and completed with `transform_spec`'s
    defaults; 'linear' entries dropped, as a FlowSample records none, unless
    guessed. Raises ValueError for anything else (a crafted or corrupt
    file)."""
    if not isinstance(obj, dict):
        raise ValueError("not a column -> transform mapping")
    out = {}
    for ch, spec in obj.items():
        if isinstance(spec, dict) and is_unknown_spec(spec):
            out[str(ch)] = unknown_spec(spec.get('origin'))
            continue
        if (not isinstance(spec, dict)
                or spec.get('method') not in TRANSFORM_METHODS):
            raise ValueError(f"unknown transform for {ch!r}")
        guessed = spec.get('guessed', False)
        if not isinstance(guessed, bool):
            raise ValueError(f"bad 'guessed' flag for {ch!r}")
        try:
            full = transform_spec(**{k: v for k, v in spec.items()
                                     if k != 'guessed'})
        except TypeError as exc:
            raise ValueError(f"bad transform for {ch!r} ({exc})") from exc
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               for k, v in full.items() if k != 'method'):
            raise ValueError(f"non-numeric parameter for {ch!r}")
        if guessed:
            out[str(ch)] = guessed_spec(full)
        elif full['method'] != 'linear':
            out[str(ch)] = full
    return out


def logicle_display_bounds(t=262144, m=4.5, w=0.5, a=0):
    """The range the loader's logicle maps cytometer data into: its image of
    raw values within +-100 x `t` (the top of scale plus two decades of
    headroom). [-1.22, 1.44] at the defaults.

    The rule for a CSV column with no transform record: it is logicle only
    if every finite value lies inside. Linear intensities run from the
    hundreds up and essentially never do. Computed with flowutils directly,
    so the GPU LUT's clamping cannot narrow it."""
    r = float(t) * 10.0 ** _LOGICLE_HEADROOM_DECADES
    lo, hi = transforms.logicle(np.array([[-r], [r]]), channel_indices=[0],
                                t=t, m=m, w=w, a=a).flatten()
    return float(lo), float(hi)


def gate_value_text(value, spec=None):
    """A gate coordinate as a user should read it, ASCII.

    Gates live in the channel's STORED coordinates, which for fluorescence
    are logicle: a cut at an intensity of ~1,000 is stored as 0.453, and the
    gate list, the population names and the slider printed exactly that.
    With the channel's ``transform_spec`` (`spec`, a ``data_transforms``
    entry) this is the linear intensity the coordinate stands for, written as
    the axis ticks write it (scales.format_intensity_text: '987', '1.23K',
    '50K'); a linear channel's value (`spec` None) is its intensity already
    and is written alike. A channel whose transform is unknown has no
    intensity to give: its stored value, '.3g', which the caller marks."""
    v = float(value)
    if is_unknown_spec(spec):
        return f'{v:.3g}'
    from .scales import format_intensity_text
    return format_intensity_text(float(to_linear(np.array([v]), spec)[0]))


def _gate_axis_text(channel, lo, hi, spec):
    """One axis of a gate, as `describe_gate` says it with transforms:
    'CD3-A >= 1K', 'CD3-A < 2.5K', 'CD3-A in [1K, 2.5K)' (gates are half
    open, lower bound inside); '' when both ends are open (the axis does not
    constrain). An end at the +-1e12 sentinel (or None) is open: printed as a
    number it read '[-1e+12, 0.453]'."""
    lo_open = lo is None or lo <= -_GATE_OPEN
    hi_open = hi is None or hi >= _GATE_OPEN
    if (lo is not None and lo >= _GATE_OPEN) or (hi is not None
                                                 and hi <= -_GATE_OPEN):
        return f"{channel} (no events)"
    if lo_open and hi_open:
        return ''
    if lo_open:
        text = f"{channel} < {gate_value_text(hi, spec)}"
    elif hi_open:
        text = f"{channel} >= {gate_value_text(lo, spec)}"
    else:
        text = (f"{channel} in [{gate_value_text(lo, spec)}, "
                f"{gate_value_text(hi, spec)})")
    if is_unknown_spec(spec):
        text += " (stored, scale unknown)"
    return text


def wsp_population_name(gate):
    """The name a gate's population takes in an exported FlowJo workspace:
    the user's name or label when it has one, else what it gates on
    ('FSC-A x SSC-A', 'CD3-A'), so the same gate is named the same in every
    sample. Never the editor's gate id, which differs between samples."""
    nm = gate.get('label') or gate.get('name')
    if nm:
        return str(nm)
    x, y = gate.get('x_channel'), gate.get('y_channel')
    if x and y:
        return f"{x} x {y}"
    if gate.get('channel'):
        return str(gate['channel'])
    return str(gate.get('kind') or 'Population').replace('_', ' ').title()


# FlowJo's element for each boolean operation, and back.
_WSP_BOOLEAN_TAGS = {'and': 'AndNode', 'or': 'OrNode', 'not': 'NotNode'}
_WSP_BOOLEAN_OPS = {v: k for k, v in _WSP_BOOLEAN_TAGS.items()}


# Gate kinds with no FlowJo geometry: left out of a .wsp export.
WSP_NO_GEOMETRY = ('cluster', 'category', 'autoclean', 'group')


def wsp_unexportable(gates_by_id):
    """Ids of the gates a FlowJo .wsp cannot carry: the kinds with no
    geometry (WSP_NO_GEOMETRY), and the boolean gates FlowJo cannot express
    -- a NOT over more than one operand (FlowJo's NotNode takes one), an
    unknown operation, or an operand that is missing or itself left out."""
    out = {gid for gid, g in gates_by_id.items()
           if g.get('kind') in WSP_NO_GEOMETRY}
    changed = True
    while changed:
        changed = False
        for gid, g in gates_by_id.items():
            if gid in out or g.get('kind') != 'boolean':
                continue
            op = str(g.get('op', 'and')).lower()
            ops = g.get('operands') or []
            unwritable = (op not in _WSP_BOOLEAN_TAGS or not ops
                          or (op == 'not' and len(ops) != 1)
                          or any(o not in gates_by_id or o in out
                                 for o in ops))
            if unwritable:
                out.add(gid)
                changed = True
    return out


def wsp_dependent_name(own_path, target_path):
    """The <Dependent> name of a FlowJo boolean node at `own_path` for the
    population at `target_path` (both tuples of population names from the
    sample down): '..' steps up from the boolean node itself, so a sibling
    operand reads '../CD4+', as FlowJo writes it."""
    own, target = list(own_path), list(target_path)
    common = 0
    while (common < len(own) - 1 and common < len(target)
           and own[common] == target[common]):
        common += 1
    return '/'.join(['..'] * (len(own) - common) + target[common:])


def resolve_wsp_dependent(name, own_path, known):
    """The population path a FlowJo <Dependent> `name` names, or None.

    `own_path` is the boolean node's path and `known` the population paths
    read for its sample. FlowJo writes a path relative to the boolean node
    ('../CD4+' is a sibling); a name with no '..' is tried below the
    boolean's parent, then from the sample, then as a unique population
    name anywhere in the sample."""
    parts = [p for p in str(name or '').split('/') if p not in ('', '.')]
    if not parts:
        return None
    base = list(own_path)
    for p in parts:
        if p == '..':
            if not base:
                return None
            base.pop()
        else:
            base.append(p)
    tried = [tuple(base)]
    plain = [p for p in parts if p != '..']
    tried += [tuple(own_path[:-1]) + tuple(plain), tuple(plain)]
    for t in tried:
        if t in known:
            return t
    # A population name may itself hold a '/' ('CD3+ / CD4+'), which the
    # split above cut apart: match the written path as a whole.
    written = '/'.join(plain)
    whole = [k for k in known
             if '/'.join(k) == written or '/'.join(k).endswith('/' + written)]
    if len(whole) == 1:
        return whole[0]
    last = [k for k in known if k and k[-1] == plain[-1]] if plain else []
    return last[0] if len(last) == 1 else None


def unique_population_name(name, taken):
    """`name`, or `name (2)`, `name (3)`... when a sibling already holds it
    (FlowJo population paths must be unique). Adds the result to `taken`."""
    out, n = name, 2
    while out in taken:
        out, n = f"{name} ({n})", n + 1
    taken.add(out)
    return out


def describe_gate(gate, transforms=None):
    """Short human-readable label, for logs and the GUI gate list.
    ASCII-only so it survives cp1252 stdout when callers haven't
    reconfigured (e.g. ad-hoc scripts importing flow_pipeline).

    `transforms` ({channel: transform_spec}, the gated sample's
    ``data_transforms``) makes it the user's text: the bounds of a
    threshold, interval or rectangle are the intensities they stand for
    (``gate_value_text``) rather than stored logicle coordinates, open sides
    are said as open ('T  CD3-A >= 987', 'I  CD3-A < 2.5K',
    'R  Q1  FSC-A x CD3-A  FSC-A < 52.3K, CD3-A >= 1K'), and a channel of
    unknown scale keeps its stored value, marked so. Without it the text is
    as it always was (stored coordinates), for logs and library callers."""
    k = gate.get('kind')
    if transforms is not None and k in ('threshold', 'interval', 'rect'):
        return _describe_bounds(gate, k, transforms)
    if k == 'threshold':
        return f"T  {gate['channel']} >= {float(gate['value']):.3g}"
    if k == 'interval':
        return (f"I  {gate['channel']} in "
                f"[{float(gate['lo']):.3g}, {float(gate['hi']):.3g}]")
    if k == 'rect':
        nm = gate.get('name') or gate.get('label')
        pre = f"{nm}  " if nm else ""
        return (f"R  {pre}{gate['x_channel']} x {gate['y_channel']}  "
                f"[{float(gate['x0']):.3g},{float(gate['x1']):.3g}] x "
                f"[{float(gate['y0']):.3g},{float(gate['y1']):.3g}]")
    if k == 'polygon':
        nm = gate.get('name') or gate.get('label')
        pre = f"{nm}  " if nm else ""
        return (f"P  {pre}{gate['x_channel']} x {gate['y_channel']}  "
                f"({len(gate.get('vertices', []))} verts)")
    if k == 'ellipsoid':
        nm = gate.get('name') or gate.get('label')
        pre = f"{nm}  " if nm else ""
        return f"E  {pre}{gate.get('x_channel')} x {gate.get('y_channel')}"
    if k == 'flowjo_ellipse':
        nm = gate.get('name') or gate.get('label')
        pre = f"{nm}  " if nm else ""
        return (f"E!  {pre}{gate.get('x_channel')} x {gate.get('y_channel')}  "
                "(FlowJo ellipse, not converted: admits no events)")
    if k == 'cluster':
        nm = gate.get('name') or gate.get('label')
        return f"C  {nm}" if nm else f"C  cluster {gate.get('cluster_id')}"
    if k == 'category':
        nm = gate.get('name') or gate.get('label') or gate.get('value')
        return f"=  {nm}"
    if k == 'boolean':
        nm = gate.get('name')
        if nm:
            return f"B  {nm}"
        op = (gate.get('op') or 'and').upper()
        return f"B  {op}({len(gate.get('operands', []))})"
    if k == 'autoclean':
        nm = gate.get('name') or 'autocleaned sample'
        on = sum(1 for m in (gate.get('methods') or [])
                 if m.get('enabled', True))
        return f"AC  {nm}  ({on} on)"
    if k == 'group':
        return gate.get('name') or 'group'
    return f"?  {k}"


def _describe_bounds(gate, kind, transforms):
    """`describe_gate` of a threshold / interval / rect in intensity."""
    def axis(ch, lo, hi):
        return _gate_axis_text(ch, None if lo is None else float(lo),
                               None if hi is None else float(hi),
                               transforms.get(ch))
    if kind in ('threshold', 'interval'):
        ch = gate['channel']
        text = (axis(ch, gate['value'], None) if kind == 'threshold'
                else axis(ch, gate['lo'], gate['hi']))
        return f"{'T' if kind == 'threshold' else 'I'}  {text or ch + ' (all events)'}"
    nm = gate.get('name') or gate.get('label')
    pre = f"{nm}  " if nm else ""
    xc, yc = gate['x_channel'], gate['y_channel']
    sides = [t for t in (axis(xc, gate['x0'], gate['x1']),
                         axis(yc, gate['y0'], gate['y1'])) if t]
    return f"R  {pre}{xc} x {yc}  {', '.join(sides) or '(all events)'}"


# ── Automated density-based gating ─────────────────────────────────────────────
#
# Lightweight "suggest a gate" helpers. They don't replace expert gating —
# they propose a sensible threshold or polygon from the data's density so the
# user can accept/adjust. Pure (numpy/scipy/contourpy), no Tk.

def _otsu_threshold(hist, centers):
    """Otsu's between-class-variance threshold on a 1-D histogram."""
    hist = np.asarray(hist, dtype=float)
    total = hist.sum()
    if total <= 0:
        return float(centers[len(centers) // 2])
    p = hist / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    k = int(np.nanargmax(sigma_b))
    return float(centers[k])


def auto_threshold(values, bins=256, smooth=2.0):
    """Suggest a 1-D split point for `values`.

    Bimodal data → the deepest valley between the two tallest peaks of the
    smoothed histogram. Unimodal data → Otsu's threshold. None when there's
    too little data. Returns a value in the data's own units."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 50:
        return None
    lo, hi = np.percentile(v, [0.5, 99.5])
    if hi <= lo:
        return None
    bins = _resolution_bins(v, lo, hi, bins)
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    sm = gaussian_filter1d(hist.astype(float), smooth)
    if sm.max() <= 0:
        return None
    peaks, _ = find_peaks(sm, prominence=sm.max() * 0.05)
    if peaks.size >= 2:
        a, b = sorted(peaks[np.argsort(sm[peaks])[::-1]][:2])
        valley = a + int(np.argmin(sm[a:b + 1]))
        return float(centers[valley])
    return _otsu_threshold(hist, centers)


def _polygon_area(verts):
    """Absolute shoelace area of an Nx2 polygon."""
    v = np.asarray(verts, dtype=float)
    if len(v) < 3:
        return 0.0
    x, y = v[:, 0], v[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) -
                          np.dot(y, np.roll(x, -1))))


def auto_polygon_gate(x, y, bins=128, level_frac=0.2, smooth=2.0,
                      max_verts=40):
    """Suggest a polygon around the dominant 2-D density mode of (x, y).

    Builds a smoothed 2-D histogram, then takes the density contour at
    `level_frac` of the peak density that encloses the global-max bin (the
    main population), simplified to at most `max_verts` vertices. Returns a
    list of ``[x, y]`` vertices in data coords, or None when it can't."""
    import contourpy
    from scipy.ndimage import gaussian_filter
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 50:
        return None
    xlo, xhi = np.percentile(x, [0.5, 99.5])
    ylo, yhi = np.percentile(y, [0.5, 99.5])
    if xhi <= xlo or yhi <= ylo:
        return None
    H, xe, ye = np.histogram2d(x, y, bins=bins, range=[[xlo, xhi], [ylo, yhi]])
    H = gaussian_filter(H, smooth)
    if H.max() <= 0:
        return None
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    # contourpy wants Z indexed [row=y, col=x] over a meshgrid.
    Z = H.T
    X, Y = np.meshgrid(xc, yc)
    peak_ix, peak_iy = np.unravel_index(int(np.argmax(H)), H.shape)
    peak_xy = (xc[peak_ix], yc[peak_iy])
    cg = contourpy.contour_generator(X, Y, Z)
    lines = cg.lines(level_frac * float(H.max()))
    polys = [np.asarray(p, dtype=float) for p in lines
             if p is not None and len(p) >= 3]
    if not polys:
        return None

    # Prefer the polygon that contains the density peak; among those (or all
    # if none contain it) take the largest by area.
    from matplotlib.path import Path as _MplPath
    containing = [p for p in polys if _MplPath(p).contains_point(peak_xy)]
    pool = containing or polys
    best = max(pool, key=_polygon_area)
    if len(best) > max_verts:
        idx = np.linspace(0, len(best) - 1, max_verts).astype(int)
        best = best[idx]
    return [[float(a), float(b)] for a, b in best]


def auto_singlet_gate(area, height, k=3.0, h_pct=(0.1, 99.9)):
    """Singlet-discrimination gate from an area channel (e.g. FSC-A, ``x``)
    vs its height channel (FSC-H, ``y``).

    Singlets satisfy ``area ≈ slope·height`` — a tight diagonal — while
    doublets / aggregates carry more area per unit height and sit *above* it.
    Keeps events whose ``area/height`` ratio lies within a robust band around
    the population median (``median ± k·1.4826·MAD``), returned as a polygon
    in ``(area, height)`` data coordinates (x = area, y = height) so it drops
    straight into a 'polygon' gate.

    Returns ``(vertices, quality)`` or ``(None, None)`` when undefined.
    ``quality = {'frac_kept', 'ratio_cv', 'slope'}``: a clean singlet gate
    keeps ~85–99 % of events with a low ratio CV; a high CV or low keep
    fraction is the signal NOT to trust it blindly.
    """
    area = np.asarray(area, dtype=float)
    height = np.asarray(height, dtype=float)
    m = np.isfinite(area) & np.isfinite(height) & (height > 0)
    a, h = area[m], height[m]
    if a.size < 50:
        return None, None
    ratio = a / h
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size < 50:
        return None, None
    med = float(np.median(ratio))
    if not np.isfinite(med) or med <= 0:
        return None, None
    mad = float(np.median(np.abs(ratio - med)))
    sigma = MAD_TO_SD * mad if mad > 0 else float(np.std(ratio))
    if sigma <= 0:
        return None, None
    r_lo = max(med - k * sigma, 1e-9)
    r_hi = med + k * sigma
    h_lo, h_hi = np.percentile(h, list(h_pct))
    if h_hi <= h_lo:
        return None, None
    # Band between the two rays area = r_lo·height and area = r_hi·height,
    # clipped to the populated height range — a quadrilateral (x = area).
    verts = [[r_lo * h_lo, h_lo],
             [r_lo * h_hi, h_hi],
             [r_hi * h_hi, h_hi],
             [r_hi * h_lo, h_lo]]
    a_ratio = a / h
    inside = ((a_ratio >= r_lo) & (a_ratio <= r_hi)
              & (h >= h_lo) & (h <= h_hi))
    quality = {'frac_kept': float(inside.mean()),
               'ratio_cv': float(sigma / med),
               'slope': med}
    return verts, quality


def gmm_ellipse_gates(x, y, max_components=6, min_weight=0.02,
                      coverage=0.90, max_events=20_000, seed=42):
    """Decompose a 2-D ``(x, y)`` distribution into Gaussian populations and
    return one ellipsoid gate per component — a principled replacement for a
    single arbitrary density contour.

    Fits ``sklearn.mixture.GaussianMixture`` (full covariance) for
    ``k = 1..max_components`` on STANDARDIZED coordinates and selects ``k`` by
    minimum BIC. Each retained component (mixing weight ≥ ``min_weight``)
    becomes an ellipsoid gate ``{mean, cov, distance_sq}`` in the ORIGINAL
    data coordinates, with ``distance_sq`` set to the chi-square quantile
    (df = 2) so the ellipse encloses ``coverage`` of a bivariate-normal
    component (e.g. coverage 0.90 → distance_sq ≈ 4.605).

    Returns a list of ``(gate, info)`` tuples sorted by descending weight,
    where ``gate`` is a partial ellipsoid-gate dict WITHOUT
    ``x_channel`` / ``y_channel`` (the caller fills those in), and
    ``info = {'weight', 'n_events', 'separation', 'n_components'}``.
    ``separation`` is the Mahalanobis distance (component metric) to the
    nearest other component mean — larger means a cleaner, more trustworthy
    split (< ~2 means the components overlap heavily). Empty list when the
    fit is undefined.
    """
    from scipy.stats import chi2
    from sklearn.mixture import GaussianMixture
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 100:
        return []
    P = np.column_stack([x, y])
    rng = np.random.default_rng(seed)
    if len(P) > max_events:
        P = P[rng.choice(len(P), max_events, replace=False)]
    # Standardize so the GMM fit is scale-invariant and numerically stable
    # (FSC counts and a logicle marker differ by orders of magnitude).
    mu = P.mean(0)
    sd = P.std(0)
    sd[sd == 0] = 1.0
    Z = (P - mu) / sd

    best = None
    kmax = min(int(max_components), len(Z))
    for k in range(1, max(1, kmax) + 1):
        gm = GaussianMixture(n_components=k, covariance_type='full',
                             random_state=seed, reg_covar=1e-6, n_init=1)
        try:
            gm.fit(Z)
            bic = float(gm.bic(Z))
        except Exception as exc:
            note("fitting a mixture model (skipping this component count)", exc)
            continue
        if best is None or bic < best[0]:
            best = (bic, gm)
    if best is None:
        return []
    gm = best[1]
    dist_sq = float(chi2.ppf(coverage, df=2))
    labels = gm.predict(Z)
    # Local ndarray copies of the fitted parameters (sklearn types them
    # Optional; they're always populated post-fit).
    weights = np.asarray(gm.weights_, dtype=float)
    means = np.asarray(gm.means_, dtype=float)
    covs = np.asarray(gm.covariances_, dtype=float)
    n_comp = int(gm.n_components)
    S = np.diag(sd)   # cov back-transform: Cov(P) = S · Cov(Z) · S
    out = []
    for i in range(n_comp):
        w = float(weights[i])
        if w < min_weight:
            continue
        mean_z = means[i]
        cov_z = covs[i]
        mean = mean_z * sd + mu
        cov = S @ cov_z @ S
        # Separation: nearest other component mean in THIS component's metric.
        sep = np.inf
        try:
            inv_z = np.linalg.inv(cov_z)
            for j in range(n_comp):
                if j == i:
                    continue
                d = mean_z - means[j]
                sep = min(sep, float(np.sqrt(max(d @ inv_z @ d, 0.0))))
        except np.linalg.LinAlgError:
            sep = np.inf
        gate = {'kind': 'ellipsoid',
                'mean': [float(mean[0]), float(mean[1])],
                'cov': [[float(cov[0, 0]), float(cov[0, 1])],
                        [float(cov[1, 0]), float(cov[1, 1])]],
                'distance_sq': dist_sq}
        info = {'weight': w,
                'n_events': int(np.sum(labels == i)),
                'separation': (None if not np.isfinite(sep) else float(sep)),
                'n_components': n_comp}
        out.append((gate, info))
    out.sort(key=lambda t: t[1]['weight'], reverse=True)
    return out


# ── FlowSOM (self-organizing map + metaclustering) ────────────────────────────
#
# A compact, dependency-free FlowSOM: train a rectangular SOM over the marker
# space, assign each event to its best-matching unit (node), then agglomerate
# the node prototypes into a handful of metaclusters. Pure numpy + sklearn
# (already a dependency). Not as tuned as the R FlowSOM, but the same shape:
# nodes capture fine structure, metaclusters give interpretable populations.

def _som_train(X, grid=(10, 10), iters=10, max_events=50_000, seed=42,
               use_kernel=True):
    """Train a rectangular SOM. Returns (weights[n_nodes, d],
    coords[n_nodes, 2]). Online updates over a (sub-sampled) event stream
    with linearly decaying neighbourhood + learning rate. ``use_kernel=False``
    forces the pure-numpy loop (the reference the compiled one must match)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    if len(X) > max_events:
        X = X[rng.choice(len(X), max_events, replace=False)]
    gx, gy = grid
    n_nodes = gx * gy
    init = rng.choice(len(X), n_nodes, replace=(len(X) < n_nodes))
    W = X[init].astype(float).copy()
    coords = np.array([(i // gy, i % gy) for i in range(n_nodes)], dtype=float)
    sigma0 = max(gx, gy) / 2.0
    lr0 = 0.5
    # The per-event loop runs compiled when numba is there (about 10x faster,
    # identical weights; see openflo.som_kernel). Its neighbourhood table
    # holds the same numbers the loop below computes per event: gd2[b] is
    # ((coords - coords[b]) ** 2).sum(1).
    from .som_kernel import som_epoch_kernel
    epoch = som_epoch_kernel() if use_kernel else None
    if epoch is not None:
        X = np.ascontiguousarray(X)
    gd2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(2)
    for t in range(iters):
        frac = t / max(1, iters)
        sigma = sigma0 * (1.0 - frac) + 0.5
        lr = lr0 * (1.0 - frac)
        two_sig2 = 2.0 * sigma * sigma
        if epoch is not None:
            epoch(X, W, rng.permutation(len(X)),
                  lr * np.exp(-gd2 / two_sig2))
            continue
        for i in rng.permutation(len(X)):
            x = X[i]
            bmu = int(np.argmin(((W - x) ** 2).sum(1)))
            gd = ((coords - coords[bmu]) ** 2).sum(1)
            h = np.exp(-gd / two_sig2)
            W += (lr * h)[:, None] * (x - W)
    return W, coords


def _som_assign(X, W):
    """Best-matching-unit node index for each row of X (chunked)."""
    X = np.asarray(X, dtype=float)
    out = np.empty(len(X), dtype=int)
    step = 10_000
    for s in range(0, len(X), step):
        chunk = X[s:s + step]
        # squared distances to every node: (chunk·node) expansion
        d = (np.sum(chunk ** 2, 1)[:, None]
             - 2.0 * chunk @ W.T
             + np.sum(W ** 2, 1)[None, :])
        out[s:s + step] = np.argmin(d, axis=1)
    return out


def _som_metacluster(W, n_metaclusters):
    """Agglomerate node prototypes into `n_metaclusters` labels (one per
    node). Falls back to one cluster when there are too few nodes."""
    from sklearn.cluster import AgglomerativeClustering
    k = max(1, min(int(n_metaclusters), len(W)))
    if k == 1:
        return np.zeros(len(W), dtype=int)
    return AgglomerativeClustering(n_clusters=k).fit_predict(W)


def flowsom_mst(weights):
    """Minimal-spanning-tree edges over FlowSOM node prototypes — the backbone
    of the classic FlowSOM star-tree plot.

    ``weights`` : ``(n_nodes, n_markers)`` SOM prototypes (``flowsom_result
    ['weights']``). Returns ``(edges, dist)`` where ``edges`` is a list of
    ``(i, j)`` node-index pairs (``n_nodes − 1`` of them, forming a tree) and
    ``dist`` is the full pairwise Euclidean distance matrix (so the caller can
    weight or lay out the tree)."""
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import pdist, squareform
    W = np.asarray(weights, dtype=float)
    n = len(W)
    if n < 2:
        return [], np.zeros((n, n))
    dist = squareform(pdist(W))
    mst = minimum_spanning_tree(dist).toarray()
    i, j = np.nonzero(mst)
    edges = list(zip(i.tolist(), j.tolist(), strict=True))
    return edges, dist


@contextlib.contextmanager
def igraph_seeded(seed):
    """Run igraph's randomised routines on ``random.Random(seed)``, then hand
    igraph back its default generator, the ``random`` module itself.

    igraph has no integer seed argument. ``layout_fruchterman_reingold(seed=)``
    takes a matrix of STARTING POSITIONS, not a number, and even with one the
    layout still draws random moves from igraph's generator. So the only way
    to make a layout or a Louvain run repeat is to swap the generator in. The
    restore must be the ``random`` module and not a fresh ``random.Random()``:
    a fresh instance cuts igraph off from ``random.seed()`` for the rest of
    the process."""
    import random as _random

    import igraph as ig
    ig.set_random_number_generator(_random.Random(int(seed)))
    try:
        yield
    finally:
        ig.set_random_number_generator(_random)


def flowsom_layout(n_nodes, edges, seed=42):
    """2-D layout for the FlowSOM MST. Uses igraph's Fruchterman-Reingold on
    the tree; falls back to a circle if igraph isn't available. Returns an
    ``(n_nodes, 2)`` float array of positions; the same ``seed`` gives the
    same layout."""
    if n_nodes == 0:
        return np.zeros((0, 2))
    try:
        import igraph as ig
    except ImportError:
        ig = None
    if ig is not None:
        g = ig.Graph(n=int(n_nodes), edges=[(int(a), int(b)) for a, b in edges])
        with igraph_seeded(seed):
            lay = g.layout_fruchterman_reingold(niter=500)
        return np.asarray(lay.coords, dtype=float)
    ang = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


# ── CytoNorm batch normalization ──────────────────────────────────────────────
#
# CytoNorm (Van Gassen 2020; CytoNorm 2.0, Quintelier 2025): the standard for
# removing technical batch/acquisition variation in cytometry. FlowSOM-cluster
# the pooled data into metaclusters, then for each metacluster and channel,
# quantile-normalize every batch's intensity distribution onto a shared GOAL
# distribution via a monotone (PCHIP) spline. Population-aware, so it doesn't
# smear distinct populations together the way a global quantile norm can.
#
# Two modes (same engine — only the events you fit on differ):
#   • 'goal'     (CytoNorm 2.0, default): fit on ALL samples; goal = pooled
#                aggregate. No dedicated control samples required.
#   • 'controls' (classic CytoNorm): fit on per-batch CONTROL samples; the
#                fitted per-batch transform is then applied to every sample in
#                that batch. Most rigorous when you ran a shared control aliquot
#                in each batch.
# The fitted model serializes (to_dict/from_dict) so it can be applied to new
# samples later — which cyCombine-style methods can't do.

def _strict_increasing(x):
    """Nudge ties so `x` is strictly increasing (PCHIP needs strictly
    increasing knots). Tiny relative epsilon, preserves order/scale."""
    x = np.asarray(x, dtype=float).copy()
    for i in range(1, len(x)):
        if x[i] <= x[i - 1]:
            x[i] = x[i - 1] + 1e-9 * (abs(x[i - 1]) + 1.0)
    return x


def cap_events_per_batch(events_by_batch, max_events_per_batch=20_000,
                         seed=42):
    """``{batch: events}`` with each batch's events subsampled, without
    replacement, to at most `max_events_per_batch` (0 or None keeps all).
    One generator seeded with `seed` draws for the batches in order, and the
    kept rows stay in their original order, so the same input always gives
    the same events. A DataFrame stays a DataFrame (index reset); an array
    stays an array. The draw CytoNorm.fit (and so the app's batch
    normalization and ``openflo-run --batch-correct``) fits on."""
    rng = np.random.default_rng(seed)
    out = {}
    for b, ev in events_by_batch.items():
        n = len(ev)
        if max_events_per_batch and n > max_events_per_batch:
            idx = np.sort(rng.choice(n, int(max_events_per_batch),
                                     replace=False))
            ev = (ev.iloc[idx].reset_index(drop=True) if hasattr(ev, 'iloc')
                  else np.asarray(ev)[idx])
        out[b] = ev
    return out


class CytoNorm:
    """FlowSOM + per-metacluster per-channel quantile normalization.

    Usage::

        cn = CytoNorm(channels=fluor_markers, mode='goal')
        cn.fit({batch_id: events_df, ...})       # events_df = a sample (or pooled)
        corrected = cn.apply(sample.data, batch_id)
        report = cn.qc({batch_id: events_df, ...})

    ``events_by_batch`` maps a batch label to that batch's events (a DataFrame
    carrying ``channels``, or an ndarray in channel order). In 'goal' mode pass
    all samples per batch (concatenate per batch); in 'controls' mode pass the
    control sample per batch, then ``apply`` to every sample of that batch.
    ``fit`` uses at most 20,000 events a batch by default
    (cap_events_per_batch, seeded with ``seed``).
    """

    def __init__(self, channels, n_metaclusters=10, grid=(10, 10),
                 n_quantiles=101, min_cell_events=50, mode='goal', seed=42):
        self.channels = [str(c) for c in channels]
        self.n_metaclusters = int(n_metaclusters)
        self.grid = (int(grid[0]), int(grid[1]))
        self.n_quantiles = int(n_quantiles)
        if self.n_quantiles < 2:
            raise ValueError(
                "n_quantiles must be >= 2 for spline fitting, got "
                f"{self.n_quantiles}")
        self.min_cell_events = int(min_cell_events)
        self.mode = str(mode)
        self.seed = int(seed)
        self.batches = []
        self._mean = None
        self._std = None
        self._W = None
        self._node_meta = None
        self._qs = None
        self._goal_q = {}        # (meta, ch_idx) -> goal quantile array | None
        self._batch_q = {}       # (meta, batch, ch_idx) -> quantile array | None

    # -- helpers --
    def _events(self, ev):
        if hasattr(ev, 'columns'):
            missing = [c for c in self.channels if c not in ev.columns]
            if missing:
                raise ValueError(
                    f"CytoNorm: batch data is missing channel(s) {missing}; "
                    f"expected {list(self.channels)}")
            return ev[self.channels].to_numpy(dtype=float)
        return np.asarray(ev, dtype=float)

    def _z(self, X):
        return (X - self._mean) / self._std

    def _meta_of(self, X):
        assert self._node_meta is not None and self._W is not None
        return self._node_meta[_som_assign(self._z(X), self._W)]

    # -- fit / apply / qc --
    def fit(self, events_by_batch, max_events_per_batch=20_000):
        """Fit on ``{batch: events}``. Each batch is first subsampled to at
        most `max_events_per_batch` events (0 or None: all) with the model's
        ``seed`` (cap_events_per_batch), so a re-fit is identical and a big
        batch doesn't outweigh the others in the goal distribution."""
        self.batches = [str(b) for b in events_by_batch]
        capped = cap_events_per_batch(events_by_batch, max_events_per_batch,
                                      seed=self.seed)
        finite = {}
        pools = []
        for b, ev in capped.items():
            X = self._events(ev)
            X = X[np.isfinite(X).all(1)]
            finite[b] = X
            if len(X):
                pools.append(X)
        if not pools:
            raise ValueError("CytoNorm.fit: no finite events to fit on.")
        pooled = np.vstack(pools)
        self._mean = pooled.mean(0)
        self._std = pooled.std(0)
        self._std[self._std == 0] = 1.0
        self._W, _coords = _som_train(self._z(pooled), self.grid, seed=self.seed)
        self._node_meta = _som_metacluster(self._W, self.n_metaclusters)
        self._qs = np.linspace(0.0, 1.0, self.n_quantiles)

        # Goal quantiles per (metacluster, channel) — the pooled aggregate.
        pooled_meta = self._meta_of(pooled)
        for m in np.unique(self._node_meta):
            sub = pooled[pooled_meta == m]
            for j in range(len(self.channels)):
                col = sub[:, j][np.isfinite(sub[:, j])]
                self._goal_q[(int(m), j)] = (
                    np.quantile(col, self._qs)
                    if col.size >= self.min_cell_events else None)

        # Per-batch quantiles per (metacluster, channel).
        for b, X in finite.items():
            bmeta = self._meta_of(X) if len(X) else np.array([], dtype=int)
            for m in np.unique(self._node_meta):
                sub = X[bmeta == m] if len(X) else X
                for j in range(len(self.channels)):
                    col = sub[:, j][np.isfinite(sub[:, j])] if len(sub) else sub
                    self._batch_q[(int(m), str(b), j)] = (
                        np.quantile(col, self._qs)
                        if col.size >= self.min_cell_events else None)
        return self

    def apply(self, df, batch_id):
        """Return a corrected copy of ``df`` for ``batch_id``. Rows/channels
        without a usable transform pass through unchanged."""
        from scipy.interpolate import PchipInterpolator
        out = df.copy()
        if self._W is None or not all(c in out.columns for c in self.channels):
            return out
        X = out[self.channels].to_numpy(dtype=float)
        finite = np.isfinite(X).all(1)
        meta = np.full(len(X), -1, dtype=int)
        if finite.any():
            meta[finite] = self._meta_of(X[finite])
        bkey = str(batch_id)
        for j, ch in enumerate(self.channels):
            vals = out[ch].to_numpy(dtype=float).copy()
            for m in np.unique(meta[meta >= 0]):
                bq = self._batch_q.get((int(m), bkey, j))
                gq = self._goal_q.get((int(m), j))
                if bq is None or gq is None:
                    continue
                sel = (meta == m) & np.isfinite(vals)
                if not sel.any():
                    continue
                xq = _strict_increasing(bq)
                v = np.clip(vals[sel], xq[0], xq[-1])
                nv = PchipInterpolator(xq, gq, extrapolate=False)(v)
                vals[sel] = np.where(np.isfinite(nv), nv, vals[sel])
            out[ch] = vals
        return out

    @staticmethod
    def qc_means(qc):
        """``(before, after, n_not_evaluable)`` for a :meth:`qc` result: the
        mean distances over the channels that could be measured, and how
        many could not. A channel that could not be measured is NaN in
        ``qc``; averaging it in would hide every real score behind one
        unmeasurable marker. Both means are NaN when no channel was
        measurable."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN mean
            before = float(np.nanmean([v['before'] for v in qc.values()]))
            after = float(np.nanmean([v['after'] for v in qc.values()]))
        skipped = sum(1 for v in qc.values() if not np.isfinite(v['before']))
        return before, after, skipped

    def qc(self, events_by_batch):
        """Per-channel mean Wasserstein distance batch→goal, before vs after.
        ``{channel: {'before': x, 'after': y}}`` — lower 'after' = better.

        A channel that could not be measured reports NaN, never 0.0. Zero is
        the BEST possible score on this scale, so a channel nobody could
        evaluate used to be indistinguishable from a perfectly aligned one.
        An empty goal distribution is likewise reported rather than raising:
        when no event has every channel finite, the pooled reference is empty
        and scipy's wasserstein_distance raised "Distribution can't be
        empty" out of a QC call.
        """
        import pandas as pd
        from scipy.stats import wasserstein_distance
        nan_result = {ch: {'before': float('nan'), 'after': float('nan')}
                      for ch in self.channels}
        pooled = np.vstack([self._events(ev) for ev in events_by_batch.values()])
        pooled = pooled[np.isfinite(pooled).all(1)]
        if pooled.size == 0:
            log.warning(
                "CytoNorm.qc: no event has a finite value in every channel, "
                "so there is no goal distribution to compare against — "
                "reporting the batch→goal distances as undefined.")
            return nan_result
        res = {}
        for j, ch in enumerate(self.channels):
            goal = pooled[:, j]
            goal = goal[np.isfinite(goal)]
            if goal.size == 0:
                res[ch] = dict(nan_result[ch])
                continue
            before, after = [], []
            for b, ev in events_by_batch.items():
                X = self._events(ev)
                col = X[:, j][np.isfinite(X[:, j])]
                if col.size:
                    before.append(wasserstein_distance(col, goal))
                cor = self.apply(pd.DataFrame(X, columns=pd.Index(self.channels)), b)
                cc = cor[ch].to_numpy(dtype=float)
                cc = cc[np.isfinite(cc)]
                if cc.size:
                    after.append(wasserstein_distance(cc, goal))
            res[ch] = {
                'before': float(np.mean(before)) if before else float('nan'),
                'after': float(np.mean(after)) if after else float('nan')}
        return res

    # -- serialization --
    def to_dict(self):
        def qmap(d):
            return {f'{k[0]}|{k[1]}' if len(k) == 2 else f'{k[0]}|{k[1]}|{k[2]}':
                    (None if v is None else [float(x) for x in v])
                    for k, v in d.items()}
        return {
            'format': 'openflo-cytonorm', 'version': 1,
            'channels': self.channels, 'n_metaclusters': self.n_metaclusters,
            'grid': list(self.grid), 'n_quantiles': self.n_quantiles,
            'min_cell_events': self.min_cell_events, 'mode': self.mode,
            'seed': self.seed, 'batches': self.batches,
            'mean': None if self._mean is None else self._mean.tolist(),
            'std': None if self._std is None else self._std.tolist(),
            'W': None if self._W is None else self._W.tolist(),
            'node_meta': None if self._node_meta is None
            else [int(x) for x in self._node_meta],
            'qs': None if self._qs is None else self._qs.tolist(),
            'goal_q': qmap(self._goal_q),
            'batch_q': qmap(self._batch_q),
        }

    @classmethod
    def from_dict(cls, d):
        cn = cls(d['channels'], d.get('n_metaclusters', 10),
                 tuple(d.get('grid', (10, 10))), d.get('n_quantiles', 101),
                 d.get('min_cell_events', 50), d.get('mode', 'goal'),
                 d.get('seed', 42))
        cn.batches = list(d.get('batches', []))
        cn._mean = None if d.get('mean') is None else np.asarray(d['mean'])
        cn._std = None if d.get('std') is None else np.asarray(d['std'])
        cn._W = None if d.get('W') is None else np.asarray(d['W'])
        cn._node_meta = (None if d.get('node_meta') is None
                         else np.asarray(d['node_meta'], dtype=int))
        cn._qs = None if d.get('qs') is None else np.asarray(d['qs'])

        def unqmap(m):
            out = {}
            for k, v in (m or {}).items():
                parts = k.split('|')
                # batch_q keys are (meta, batch, channel); a batch label may
                # itself contain '|' (POSIX path), so take meta from the first
                # segment, channel from the last, and rejoin the middle as the
                # batch name. goal_q keys are 2-part and parsed separately.
                key = ((int(parts[0]), int(parts[1])) if len(parts) == 2
                       else (int(parts[0]), '|'.join(parts[1:-1]), int(parts[-1])))
                out[key] = None if v is None else np.asarray(v, dtype=float)
            return out
        cn._goal_q = unqmap(d.get('goal_q'))
        cn._batch_q = unqmap(d.get('batch_q'))
        return cn


# ── GPU probe ─────────────────────────────────────────────────────────────────

def _probe_gpu():
    """Return (available, display_name, CumlUMAP_class, cluster_kit).

    `cluster_kit` is a dict {cunn, cugraph, cupy, cudf} if the RAPIDS pieces
    needed for GPU clustering are present, else None. cuML UMAP can work
    standalone, but clustering needs the full RAPIDS stack.
    """
    try:
        from cuml.manifold import UMAP as _CU  # type: ignore[import-not-found]
        name = 'GPU'
        try:
            import pynvml  # type: ignore[import-not-found]
            pynvml.nvmlInit()
            h   = pynvml.nvmlDeviceGetHandleByIndex(0)
            raw = pynvml.nvmlDeviceGetName(h)
            name = raw.decode() if isinstance(raw, bytes) else raw
        except Exception:  # noqa: BLE001 - GPU name is informational
            pass

        cluster_kit = None
        try:
            import cudf as _cudf  # type: ignore[import-not-found]
            import cugraph as _cugraph  # type: ignore[import-not-found]
            import cupy as _cupy  # type: ignore[import-not-found]
            from cuml.neighbors import NearestNeighbors as _CuNN  # type: ignore[import-not-found]
            cluster_kit = {
                'cunn':    _CuNN,
                'cugraph': _cugraph,
                'cupy':    _cupy,
                'cudf':    _cudf,
            }
        except ImportError:
            pass

        return True, name, _CU, cluster_kit
    except ImportError:
        return False, '', None, None

GPU_AVAILABLE, GPU_NAME, _CumlUMAP, _GPU_CLUSTER_KIT = _probe_gpu()
GPU_CLUSTERING_AVAILABLE = _GPU_CLUSTER_KIT is not None


def _vram_free_gb():
    """Free VRAM in GB via pynvml → nvidia-smi fallback. None if no GPU."""
    try:
        import pynvml  # type: ignore[import-not-found]
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(h).free / (1024 ** 3)
    except Exception:  # noqa: BLE001 - GPU memory probe is best-effort
        pass
    try:
        import subprocess
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=4,
            creationflags=flags)
        if out.returncode == 0:
            return float(out.stdout.strip().splitlines()[0]) / 1024.0
    except Exception:  # noqa: BLE001 - GPU memory probe is best-effort
        pass
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────

# Scatter, time, pulse-width and instrument parameters stay LINEAR and are
# never markers. SCATTER_KEYWORDS matched case-sensitive substrings, so every
# name below went to the fluorescence list, was logicle-transformed by
# apply_transform and clustered on as a marker: Beckman's 'FS INT' / 'SS INT'
# / 'TIME', a lower-case 'fsc-a', Sony's back-scatter 'BSC-A', and CyTOF's
# Event_length. Measured: TIME 0..1000 s became 0.11..0.45 logicle units. The
# match is now on the upper-cased name with token boundaries, so a scatter
# name is recognised whatever its case or separator and a fluor name is not
# caught by a stray substring.
_SCATTER_RE = re.compile(
    # FSC / SSC with up to two letters before: VSSC1-A (violet SSC),
    # BSSC-A (Attune), SSC-B-A (Cytek), 'Violet SSC-A', FSC-HLin (Guava);
    # BSC is Sony's back-scatter.
    r'(?<![A-Z0-9])[A-Z]{0,2}(?:FSC|SSC|BSC)(?![A-Z])'
    # Beckman Gallios / Navios / FC500: 'FS INT', 'SS PEAK', 'FS TOF', 'FS Lin'.
    r'|^(?:FS|SS)(?:[^A-Z0-9]|$)')
_TIME_RE = re.compile(r'(?<![A-Z0-9])TIME')     # Time, TIME, Time (s), TIMESTAMP
# Mass-cytometry acquisition parameters (Fluidigm / Standard BioTools).
_INSTRUMENT_PARAMS = frozenset({'EVENT_LENGTH', 'EVENT LENGTH', 'CENTER',
                                'OFFSET', 'RESIDUAL'})


def _is_time(name):
    """True for an acquisition-time parameter, in any case."""
    return bool(_TIME_RE.search(str(name).upper()))


def _is_scatter(name):
    """True for a parameter kept linear and never used as a marker: scatter,
    time, pulse width and instrument parameters (see _SCATTER_RE)."""
    u = str(name).upper()
    return (bool(_SCATTER_RE.search(u)) or _is_time(u) or 'WIDTH' in u
            or u in _INSTRUMENT_PARAMS)


def _is_excluded(name):
    return any(k in name for k in EXCLUDE_CLUSTER) or _is_time(name)


# A detector's pulse measurement as the last token of its name: 'FL1-A',
# 'FL1 H', 'FL1_W', 'FL3 Area', 'FL3-Height'. Group 1 is the detector, group 2
# the measurement.
_PULSE_RE = re.compile(r'^(.*?\S)[\s_-]+(A|AREA|H|HEIGHT|W|WIDTH)$',
                       re.IGNORECASE)


def default_cluster_channels(channels):
    """The channels clustering and embedding use when none are given: ONE
    column per marker, out of ``channels`` (a sample's fluorescence list).

    A marker exported as area, height and width counted about three times in
    every neighbour distance, because the three are collinear. So each
    detector keeps its area (``-A`` / ``Area``) column; its height (``-H`` /
    ``Height``) only when it has no area, so a file with FL1-A and FL2-H
    keeps FL2; its width (``-W`` / ``Width``) never. A name without one of
    these suffixes (a mass channel, a label) is kept. Order is kept. Falls
    back to every channel if the rule would leave none."""
    fl = [str(c) for c in (channels or [])]
    kinds: dict[str, set] = {}
    parsed = []
    for c in fl:
        m = _PULSE_RE.match(c.strip())
        if m is None:
            parsed.append((c, None, None))
            continue
        base, kind = m.group(1).strip().upper(), m.group(2).upper()[0]
        parsed.append((c, base, kind))
        kinds.setdefault(base, set()).add(kind)
    keep = [c for c, base, kind in parsed
            if kind is None or kind == 'A'
            or (kind == 'H' and 'A' not in kinds[base])]
    return keep or fl


# ══════════════════════════════════════════════════════════════════════════════
# WSP COMPENSATION READER
# ══════════════════════════════════════════════════════════════════════════════

def wsp_sample_events(sample_elem):
    """The event count a (namespace-stripped) FlowJo <Sample> records for
    its file (FlowJo writes it as the <SampleNode>'s `count`), or None when
    it records none. 0 and -1 are not counts: OpenFlo's own exports write
    0, and FlowJo -1 for a sample it did not compute."""
    sn = sample_elem.find('SampleNode') if sample_elem is not None else None
    if sn is None:
        return None
    try:
        n = int(sn.get('count'))
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def fcs_event_total(path):
    """$TOT of the FCS file at `path`, or None when it cannot be read."""
    try:
        kw = {k.upper(): v for k, v in _fcs_text_pairs(path)}
        n = int(str(kw.get('$TOT', '')).strip())
    except (OSError, ValueError, IndexError):
        return None
    return n if n >= 0 else None


def wsp_event_mismatch(expected, actual):
    """'' when a workspace sample's recorded event count `expected` (see
    wsp_sample_events) agrees with the `actual` events of the FCS file it
    was matched to, or either is unknown; else what to tell the user. A
    workspace can hold several samples whose files share a name, so a file
    found by name may be another sample's."""
    if expected is None or actual is None or int(expected) == int(actual):
        return ''
    return (f"the workspace records {int(expected)} events for this sample "
            f"but the FCS file has {int(actual)}: it may be a different "
            "file of the same name")


_DTD_SUBSET = re.compile(rb'<!doctype[^>\[]*\[', re.IGNORECASE)


def xml_declares_entities(head):
    """Whether the start of an XML file (bytes) declares a DTD internal
    subset (``<!DOCTYPE x [ ... ]>``) or an entity, the vehicles of an
    entity-expansion attack. A bare ``<!DOCTYPE name>`` declares neither."""
    return bool(_DTD_SUBSET.search(head)) or b'<!entity' in head.lower()


def _pre_flowjo10_version(root):
    """The FlowJo version of a parsed workspace root when it is older than
    FlowJo 10 (its ``flowJoVersion``, else its ``version``, below 10; FlowJo
    9 writes version "2.0" and flowJoVersion "9.x", FlowJo 10 "20.0" and
    "10.x"), else ''. A version that is not a number (earlier OpenFlo
    exports say "OpenFlo-export-1.0") is not judged."""
    tag = re.sub(r'\{.*?\}', '', root.tag)
    if tag != 'Workspace':
        return ''
    for key in ('flowJoVersion', 'version'):
        v = (root.get(key) or '').strip()
        m = re.match(r'(\d+)(?:\.\d+)*', v)
        if m:
            return v if int(m.group(1)) < 10 else ''
    return ''


class WspReader:
    """
    Parse compensation matrices from a FlowJo v10 .wsp workspace file.

    Usage
    -----
        reader = WspReader('experiment.wsp')
        reader.print_matrices()
        m = reader.get_matrix()           # first matrix
        m = reader.get_matrix('My Comp') # by name
        # m keys: matrix (ndarray), channels (list), prefix, suffix
    """

    def __init__(self, wsp_path):
        self.path     = wsp_path
        self.matrices = {}
        self.root     = None        # populated by _parse; needed for gate extraction
        # Populations the last extract_gates() left out (unreadable gate).
        self.unread_populations = []
        self._parse()

    def _parse(self):
        try:
            # DTD/entity guard: stdlib etree doesn't resolve external entities
            # (no XXE file-read), but IS vulnerable to internal entity-expansion
            # (billion-laughs / quadratic-blowup) amplification — a few-KB .wsp
            # ballooning to GBs. Reject an internal DTD subset or any ENTITY
            # declaration before parsing (no new dep). A bare
            # `<!DOCTYPE Workspace>`, as FlowJo 9 writes, declares nothing and
            # is let through so the version check below can name the file.
            with open(self.path, 'rb') as _fh:
                _head = _fh.read(1 << 20)        # DOCTYPE must be in the prolog
            if xml_declares_entities(_head):
                raise WspParseError(
                    "refusing to parse a WSP with a DTD/entity declaration "
                    f"(possible entity-expansion attack): {self.path}")
            tree = ET.parse(self.path)
        except FileNotFoundError as e:
            raise WspParseError(f"WSP file not found: {self.path}") from e
        except ET.ParseError as e:
            raise WspParseError(
                f"WSP file is not valid XML ({self.path}): {e}") from e
        root = tree.getroot()
        old = _pre_flowjo10_version(root)
        if old:
            raise WspParseError(
                f"{os.path.basename(self.path)} is a FlowJo {old} workspace. "
                "FlowJo 9 (and older) workspaces are not supported yet; "
                "re-save it in FlowJo 10 and open that file.")
        # Strip XML namespace from BOTH element tags AND attribute keys.
        # FlowJo v10's Gating-ML v2 .wsp files namespace attributes too
        # (gating:min, data-type:name, data-type:value, ...), and
        # ElementTree expands those to {namespace_uri}localname. Without
        # this attribute-side strip, our find/get calls all miss and the
        # reader silently extracts zero matrices and zero gates.
        ns_re = re.compile(r'\{.*?\}')
        for elem in root.iter():
            elem.tag = ns_re.sub('', elem.tag)
            if elem.attrib:
                stripped = {ns_re.sub('', k): v for k, v in elem.attrib.items()}
                elem.attrib.clear()
                elem.attrib.update(stripped)
        self.root = root
        for tag in self._MATRIX_TAGS:
            for node in root.iter(tag):
                self._extract_matrix(node)
        if not self.matrices:
            log.info("[WspReader] No compensation matrices found.")

    # FlowJo's own groups: every sample is in 'All Samples', and FlowJo puts
    # compensation controls in 'Compensation'. Neither is a user's grouping.
    _DEFAULT_GROUPS = frozenset({'all samples', 'compensation'})

    def sample_groups(self):
        """``{sampleID: [group name, ...]}`` from the workspace's <Groups>,
        in the order the workspace lists them, leaving out FlowJo's own
        'All Samples' and 'Compensation' groups."""
        out = {}
        if self.root is None:
            return out
        for gn in self.root.iter('GroupNode'):
            name = (gn.get('name') or '').strip()
            if not name or name.lower() in self._DEFAULT_GROUPS:
                continue
            for ref in gn.iter('SampleRef'):
                sid = ref.get('sampleID')
                if sid is not None and name not in out.setdefault(sid, []):
                    out[sid].append(name)
        return out

    @staticmethod
    def _channel_name(dim_elem):
        """Pull the FCS channel name from a gating <dimension>, stripping
        FlowJo's 'Comp-' prefix so it matches FlowSample.data columns."""
        ch_elem = dim_elem.find('fcs-dimension')
        if ch_elem is None:
            ch_elem = next(dim_elem.iter('fcs-dimension'), None)
        if ch_elem is None:
            return None
        ch_name = ch_elem.get('name') or ch_elem.get('PnN')
        if not ch_name:
            return None
        return _strip_comp_marker(ch_name)[0]

    @staticmethod
    def sample_timestep(sample_elem, fcs_path=None):
        """Seconds per Time tick for a <Sample>.

        When the sample's <Transformations> puts Time on a linear axis with
        a `gain` other than 1, that gain: FlowJo writes there the seconds
        per tick it used. FlowJo 10.6 and 10.8 used ($ETIM - $BTIM) / last
        tick, not the file's $TIMESTEP, and wrote that as the gain (0.0100853
        where $TIMESTEP said 0.01); with the gain, FlowJo's Time counts come
        out exactly, with $TIMESTEP they do not.

        Otherwise as FlowJo 10.10.2 derives it, measured on workspaces
        without such a gain: from the FCS file -- its $TIMESTEP, else
        ($ETIM - $BTIM) / last tick -- ignoring a different $TIMESTEP in the
        workspace <Keywords> (FlowJo applied the file's 0.05 where the
        workspace said 0.01). `fcs_path` is the sample's FCS; None resolves
        the <DataSet> uri. Only when the file cannot be read does the
        <Keywords> value stand in. None when nothing gives one."""
        if sample_elem is None:
            return None
        gain = _wsp_time_gain(sample_elem)
        if gain:
            return gain
        if fcs_path is None:
            ds = sample_elem.find('DataSet')
            fcs_path = _wsp_uri_path(ds.get('uri') if ds is not None else None)
        ts = _fcs_timestep(fcs_path)
        if ts:
            return ts
        if fcs_path and os.path.isfile(fcs_path):
            return None              # the file itself gives FlowJo no factor
        for kw in sample_elem.findall('Keywords/Keyword'):
            if (kw.get('name') or '').upper() == '$TIMESTEP':
                return _positive_float(kw.get('value'))
        return None

    def extract_gates(self, *, sample_node=None, timestep=None):
        """Extract supported gates from the .wsp as a list of gate dicts in
        topological order (parents before children).

        Parameters
        ----------
        sample_node : ET.Element | None
            When None (default), walk every ``<SampleNode>`` in the
            document and return a single flattened list — preserves the
            historical behaviour used by :func:`read_template_gates` and
            the GUI's per-template loader.
            When given a specific ``<SampleNode>`` element, walk only
            that sample's gate tree. Used by the gate editor's
            ``Add FCS / Workspace`` to attach the right gate tree to
            the right sample when opening a multi-sample .wsp.
        timestep : float | None
            Seconds per Time tick for `sample_node`. FlowJo writes Time gate
            coordinates in seconds; they are divided by this so they come
            back in the raw ticks OpenFlo's data holds. None derives it as
            FlowJo does, from the sample's FCS (see `sample_timestep`); pass
            it when the caller located the FCS somewhere else. No factor
            means no conversion.

        Each returned gate carries an `_import_id` (a temporary string, stable
        within this call) and a `parent_id` referencing another gate's
        `_import_id` (or None for roots). The GUI editor remaps these to its
        own gate_ids on load.

        Hierarchy comes from FlowJo's nested <Population>/<Subpopulations>
        structure. If no such hierarchy is found, falls back to a flat scan
        of every RectangleGate / PolygonGate in the document.

        Supports:
          • 1-D RectangleGate (min only)   → 'threshold'
          • 1-D RectangleGate (min + max)  → 'interval'
          • 2-D RectangleGate              → 'rect'
          • PolygonGate                    → 'polygon'
          • EllipsoidGate (Gating-ML)      → 'ellipsoid'
          • EllipsoidGate (drawn in FlowJo: foci + edge in display space)
                                           → 'ellipsoid' on linear axes,
                                             'polygon' on logicle / biex axes;
                                             on any other axis a
                                             'flowjo_ellipse' placeholder, which
                                             admits no events (see
                                             _wsp_ellipse_gate)
          • QuadrantGate                   → 4 linked 'rect' (quad_set)
          • AndNode / OrNode / NotNode     → 'boolean' over the populations
                                             its <Dependent>s name
        Skipped (with a warning): CurlyQuad, Gating-ML BooleanGate, and any
        EllipsoidGate / QuadrantGate that could not be read. A population
        whose gate is skipped is left out together with the populations
        under it (`self.unread_populations` lists them for this call); a
        population with no <Gate> at all passes its children to its parent.

        Each gate carries 'wsp_comp_channels': its channels FlowJo names with
        the compensated marker ('Comp-' or <...>; see _wsp_comp_channels).
        """
        if self.root is None:
            return []

        gates = []
        next_id = [0]
        # SampleNode -> its <Sample> (ET elements have no parent pointer). A
        # caller may pass a SampleNode from its own parse of the same file
        # (the editor does), so fall back to matching on sampleID.
        owner = {}
        for s_el in self.root.iter('Sample'):
            for sn_el in s_el.findall('SampleNode'):
                owner[sn_el] = s_el
                if sn_el.get('sampleID') is not None:
                    owner.setdefault(('sampleID', sn_el.get('sampleID')), s_el)
        floors = [{}]
        # The <Sample> being walked: a FlowJo-drawn ellipse needs its axis
        # transforms. Gate tags walked but not imported, for the warning, and
        # the populations imported as 'flowjo_ellipse' placeholders.
        cur_sample = [None]
        not_imported = set()
        placeholders = []
        # (population, gate tags, populations under it) whose gate could not
        # be read: left out with their subtree. Also kept on the reader for
        # a caller's status line.
        unread = []
        self.unread_populations = unread

        def sample_of(sn):
            s_el = owner.get(sn)
            return s_el if s_el is not None else owner.get(('sampleID', sn.get('sampleID')))

        literal = wsp_written_by_openflo(self.root)

        def floors_for(sn):
            # OpenFlo's own export declares linear axes FlowJo would floor;
            # OpenFlo counted its edges as written, so they read back so.
            return {} if literal else _wsp_axis_floors(sample_of(sn))

        def timestep_for(sn):
            s_el = sample_of(sn)
            return self.sample_timestep(s_el) if s_el is not None else None

        def to_ticks(start, ts):
            """Gates appended since `start`: FlowJo's Time seconds -> ticks."""
            if ts:
                gates[start:] = [_scale_time_coords(g, 1.0 / ts)
                                 for g in gates[start:]]

        def make_id():
            i = next_id[0]
            next_id[0] += 1
            return f'imp_{i}'

        def parse_rect(rect_elem, parent_imp_id):
            parsed = []
            for dim in rect_elem.iter('dimension'):
                ch = self._channel_name(dim)
                if not ch:
                    continue
                min_attr = dim.get('min')
                max_attr = dim.get('max')
                try:
                    lo = float(min_attr) if min_attr is not None else None
                    hi = float(max_attr) if max_attr is not None else None
                except (TypeError, ValueError):
                    continue
                # Gating-ML writes an open bound as "-INF"/"INF", which float()
                # turns into ±inf — and json.dump then writes `-Infinity`, which
                # RFC 8259 does not allow. Python's own reader accepts it, so
                # the session file looked fine locally while being unreadable
                # by JSON.parse, jq, or anything else. The MISSING-bound case a
                # few lines below already uses ±1e12 for exactly this reason
                # ("JSON-safe, unlike -inf"); the EXPLICIT-INF case did not.
                lo = _finite_bound(lo, -1e12)
                hi = _finite_bound(hi, 1e12)
                # Keyed by FlowJo's own parameter name (before _channel_name
                # strips 'Comp-'), as <Transformations> names it.
                fd = next(dim.iter('fcs-dimension'), None)
                lo = _wsp_axis_floor_lo(fd.get('name') if fd is not None else None,
                                        lo, floors[0])
                parsed.append((ch, lo, hi))
            if len(parsed) == 1:
                ch, lo, hi = parsed[0]
                if lo is not None and hi is not None:
                    return {'kind': 'interval', 'channel': ch,
                            'lo': lo, 'hi': hi,
                            'parent_id': parent_imp_id,
                            '_import_id': make_id()}
                if lo is not None:
                    return {'kind': 'threshold', 'channel': ch, 'value': lo,
                            'parent_id': parent_imp_id,
                            '_import_id': make_id()}
                if hi is not None:
                    # Max-only 1-D rect (x < hi): represent as an interval with
                    # a large negative sentinel lo (JSON-safe, unlike -inf) so
                    # the constraint — and this population's children — stay in
                    # the tree instead of being silently re-parented up to the
                    # grandparent (which drops `x < hi` from every descendant).
                    return {'kind': 'interval', 'channel': ch,
                            'lo': -1e12, 'hi': hi,
                            'parent_id': parent_imp_id,
                            '_import_id': make_id()}
            elif len(parsed) == 2:
                (xc, x0, x1), (yc, y0, y1) = parsed
                # A missing bound is an open side: FlowJo writes each of a
                # quadrant's four sectors with only its divider bound per
                # dimension. Skipping them dropped every FlowJo quadrant and
                # lifted its children out of their sector.
                return {'kind': 'rect',
                        'x_channel': xc, 'y_channel': yc,
                        'x0': -1e12 if x0 is None else x0,
                        'x1': 1e12 if x1 is None else x1,
                        'y0': -1e12 if y0 is None else y0,
                        'y1': 1e12 if y1 is None else y1,
                        'parent_id': parent_imp_id,
                        '_import_id': make_id()}
            return None

        def parse_polygon(poly_elem, parent_imp_id):
            dims = list(poly_elem.iter('dimension'))
            if len(dims) != 2:
                return None
            xc = self._channel_name(dims[0])
            yc = self._channel_name(dims[1])
            if not xc or not yc:
                return None
            verts = []
            for v in poly_elem.iter('vertex'):
                coords = list(v.iter('coordinate'))
                if len(coords) < 2:
                    continue
                vx_attr = coords[0].get('value')
                vy_attr = coords[1].get('value')
                if vx_attr is None or vy_attr is None:
                    continue
                try:
                    vx = float(vx_attr)
                    vy = float(vy_attr)
                except (TypeError, ValueError):
                    continue
                # A vertex at infinity carries the same JSON hazard as an open
                # rectangle bound, and a NaN vertex has no place in a polygon
                # at all — skip it rather than writing an unreadable session.
                vx = _finite_bound(vx, -1e12 if vx < 0 else 1e12)
                vy = _finite_bound(vy, -1e12 if vy < 0 else 1e12)
                if vx is None or vy is None:
                    continue
                verts.append([vx, vy])
            if len(verts) >= 3:
                g = {'kind': 'polygon',
                     'x_channel': xc, 'y_channel': yc,
                     'vertices': verts,
                     'parent_id': parent_imp_id,
                     '_import_id': make_id()}
                axes = _wsp_polygon_edge_axes(poly_elem, cur_sample[0])
                if axes is not None:
                    g['flowjo_edge_axes'] = axes
                return g
            return None

        def parse_ellipsoid(ell_elem, parent_imp_id, pop_name=None):
            """EllipsoidGate, in either form (see _wsp_ellipse_gate). An
            event is inside a Gating-ML one when
            (x-µ)ᵀ Σ⁻¹ (x-µ) ≤ distanceSquare. A FlowJo-drawn one that cannot
            be converted is a 'flowjo_ellipse' placeholder, reported here and
            in the walk's summary as imported, not skipped."""
            body, why = _wsp_ellipse_gate(ell_elem, cur_sample[0])
            if body is None:
                return None
            name = pop_name or 'unnamed'
            if body['kind'] == 'flowjo_ellipse':
                placeholders.append(name)
                log.warning(
                    f"[WspReader] FlowJo ellipse '{name}' could not be "
                    f"converted ({why}); imported as a placeholder that "
                    "admits no events, so it and every population under it "
                    "stay empty. Redraw it in OpenFlo; a FlowJo export "
                    "writes it back as FlowJo drew it.")
            elif why:
                log.warning(f"[WspReader] FlowJo ellipse '{name}': {why}.")
            return {**body, 'parent_id': parent_imp_id, '_import_id': make_id()}

        def parse_quadrant(quad_elem, parent_imp_id):
            """Gating-ML 2.0 QuadrantGate: two <divider> elements, each
            naming a dimension + a threshold value. Expanded into FOUR
            rect gates sharing a `quad_set` id + `quad_origin`, spanning
            ±1e12 on their open sides — matching the editor's internal
            quadrant representation so the round-trip stays in one model.

            Returns a LIST of 4 gate dicts (not a single dict)."""
            divs = quad_elem.findall('divider')
            if len(divs) != 2:
                return None
            parsed = []
            for d in divs:
                ch = self._channel_name(d)
                # Divider threshold is a <value> child element (Gating-ML)
                # or a value= attribute (defensive fallback).
                val = None
                v_elem = d.find('value')
                if v_elem is not None and (v_elem.text or '').strip():
                    val = v_elem.text.strip()
                if val is None:
                    val = d.get('value')
                try:
                    val = float(val) if val is not None else None
                except (TypeError, ValueError):
                    val = None
                if not ch or val is None:
                    return None
                parsed.append((ch, val))
            (xc, xdiv), (yc, ydiv) = parsed
            big = 1e12
            # Rect bounds are half-open [lo, hi), so a divider used as the
            # LOWER bound of the upper quadrant and the UPPER bound of the
            # lower one already assigns an event sitting exactly on it to
            # exactly one side. The old code nudged the lower bound down by
            # one ULP (nextafter) to work around the previous strict `>`;
            # that is now not just unnecessary but harmful -- an event at
            # exactly nextafter(div, -inf) would satisfy both `>= xlo` above
            # and `< xdiv` below, i.e. be double-counted.
            xlo, ylo = xdiv, ydiv
            qs_id = make_id()
            out = []
            for label, x0, x1, y0, y1 in [
                    ('Q++ (x>, y>)',  xlo,   big,  ylo,   big),
                    ('Q+- (x>, y<)',  xlo,   big, -big,  ydiv),
                    ('Q-+ (x<, y>)', -big,  xdiv,  ylo,   big),
                    ('Q-- (x<, y<)', -big,  xdiv, -big,  ydiv)]:
                out.append({'kind': 'rect',
                            'x_channel': xc, 'y_channel': yc,
                            'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1,
                            'label': label,
                            'quad_set': qs_id,
                            'quad_origin_x': xdiv, 'quad_origin_y': ydiv,
                            'parent_id': parent_imp_id,
                            '_import_id': make_id()})
            return out

        # Per sample: population path -> import id, and the boolean nodes
        # read, whose operands are resolved when the sample is done.
        pop_paths = {}
        booleans = []

        def walk_population(pop_elem, parent_imp_id, path=()):
            """Process a <Population>: parse its <Gate> (if any), then recurse
            into its <Subpopulations>. A population with no gate passes its
            children to its parent. One whose gate could not be read is left
            out WITH everything under it (named in the warning): attaching
            its children to the grandparent, as this did, made each count
            every event the unread gate excluded, a plausible wrong number.
            """
            this_id = None
            # The population's NAME lives on the <Population> element, not on
            # the <Gate> inside it — so parsing only the gate discarded every
            # label in the file. A 40-population FlowJo workspace imported
            # fully unnamed, and nothing downstream put the names back.
            # `setdefault` so a parser that already derived one (the quadrant
            # gates name their own four sectors) keeps it.
            pop_name = pop_elem.get('name')
            for gate_wrapper in pop_elem.findall('Gate'):
                for child in gate_wrapper:
                    g = None
                    if child.tag == 'RectangleGate':
                        g = parse_rect(child, parent_imp_id)
                    elif child.tag == 'PolygonGate':
                        g = parse_polygon(child, parent_imp_id)
                    elif child.tag == 'EllipsoidGate':
                        g = parse_ellipsoid(child, parent_imp_id, pop_name)
                    elif child.tag == 'QuadrantGate':
                        quad = parse_quadrant(child, parent_imp_id)
                        if quad:
                            # parse_quadrant returns 4 linked rects.
                            cc = _wsp_comp_channels(child)
                            for q in quad:
                                q['wsp_comp_channels'] = list(cc)
                            gates.extend(quad)
                            this_id = quad[0]['_import_id']
                            break
                    if g is not None:
                        if pop_name:
                            g.setdefault('label', pop_name)
                        g['wsp_comp_channels'] = _wsp_comp_channels(child)
                        gates.append(g)
                        this_id = g['_import_id']
                        break
                if this_id is not None:
                    break
            gate_tags = [c.tag for gw in pop_elem.findall('Gate') for c in gw]
            if this_id is None and gate_tags:
                not_imported.update(gate_tags)
                unread.append((pop_name or 'unnamed', sorted(set(gate_tags)),
                               sum(1 for _ in pop_elem.iter('Population')) - 1))
                return
            attach_to = this_id if this_id is not None else parent_imp_id
            own = tuple(path) + (pop_name or '',)
            pop_paths[own] = attach_to
            walk_children(pop_elem, attach_to, own)

        def walk_boolean(node, parent_imp_id, path):
            """A FlowJo AndNode / OrNode / NotNode: a boolean gate over the
            populations its <Dependent> elements name, resolved once the
            whole sample is read (an operand may come after it). These used
            to be skipped without a word, together with everything under
            them."""
            name = node.get('name') or ''
            g = {'kind': 'boolean', 'op': _WSP_BOOLEAN_OPS[node.tag],
                 'operands': [], 'parent_id': parent_imp_id,
                 '_import_id': make_id()}
            if name:
                g['label'] = name
            gates.append(g)
            own = tuple(path) + (name,)
            pop_paths[own] = g['_import_id']
            deps = [d.get('name') for dn in node.findall('Dependents')
                    for d in dn.findall('Dependent')]
            booleans.append((g, own, deps))
            walk_children(node, g['_import_id'], own)

        def walk_children(elem, parent_imp_id, path):
            for sub in elem.findall('Subpopulations'):
                for child in sub:
                    if child.tag == 'Population':
                        walk_population(child, parent_imp_id, path)
                    elif child.tag in _WSP_BOOLEAN_OPS:
                        walk_boolean(child, parent_imp_id, path)

        def resolve_booleans(sample_label):
            """Point each boolean read since the last call at its operands'
            import ids. An operand that cannot be found leaves the boolean
            with an operand no gate has, so it reads NA (fail-closed), and
            is named in the log."""
            for g, own, deps in booleans:
                for dep in deps:
                    hit = resolve_wsp_dependent(dep, own, pop_paths)
                    target = pop_paths.get(hit) if hit is not None else None
                    if target is None or target == g['_import_id']:
                        log.warning("[WspReader] %s: boolean %r: operand %r "
                                    "not found -- the population reads NA",
                                    sample_label, g.get('label') or g['op'],
                                    dep)
                        g['operands'].append(f'unresolved:{dep}')
                    else:
                        g['operands'].append(target)
                if not deps:
                    log.warning("[WspReader] %s: boolean %r names no operand "
                                "-- the population reads NA", sample_label,
                                g.get('label') or g['op'])
            booleans.clear()
            pop_paths.clear()

        if sample_node is not None:
            # Per-sample walk: just this <SampleNode>'s subpopulations.
            # Skip the fallback flat scan + the cross-document unsupported-
            # kinds warning (extract_gates() with no sample_node still runs
            # both for the full-document case), but name what this sample's
            # walk could not import: this is the editor's import path.
            floors[0] = floors_for(sample_node)
            cur_sample[0] = sample_of(sample_node)
            walk_children(sample_node, None, ())
            resolve_booleans(sample_node.get('name') or 'sample')
            to_ticks(0, _positive_float(timestep) if timestep is not None
                     else timestep_for(sample_node))
            # One warning per skipped gate: the unread populations' names
            # their gate types (_log_unread); any other type is named here.
            rest = not_imported - _unread_tags(unread)
            if rest:
                log.warning(
                    "[WspReader] %s: skipped unsupported gate types: %s",
                    sample_node.get('name') or 'sample', sorted(rest))
            _log_unread(sample_node.get('name') or 'sample', unread)
            if placeholders:
                log.warning(
                    "[WspReader] %s: %d FlowJo ellipse(s) imported as empty "
                    "placeholders (not converted): %s",
                    sample_node.get('name') or 'sample', len(placeholders),
                    ', '.join(placeholders))
        else:
            # Walk hierarchy starting at every SampleNode/Subpopulations/
            # Population.
            walked = False
            for sn in self.root.iter('SampleNode'):
                start = len(gates)
                floors[0] = floors_for(sn)
                cur_sample[0] = sample_of(sn)
                walk_children(sn, None, ())
                walked = walked or len(gates) > start or any(
                    True for sub in sn.findall('Subpopulations') for _ in sub)
                resolve_booleans(sn.get('name') or 'sample')
                to_ticks(start, timestep_for(sn))

            # Fallback flat scan when the .wsp has gates but no Population
            # wrappers.
            if not walked:
                floors[0] = {}          # no owning sample for a flat scan
                for rect in self.root.iter('RectangleGate'):
                    g = parse_rect(rect, None)
                    if g is not None:
                        gates.append(g)
                for poly in self.root.iter('PolygonGate'):
                    g = parse_polygon(poly, None)
                    if g is not None:
                        gates.append(g)
                # The flat scan reads neither of these.
                not_imported.update(gt for gt in ('EllipsoidGate', 'QuadrantGate')
                                    if any(True for _ in self.root.iter(gt)))

            # Unsupported kinds: report once. EllipsoidGate and QuadrantGate
            # are parsed, so they are named only when one was not imported;
            # listing them whenever present called a correctly imported
            # ellipse "skipped".
            # The unread populations' own warning names their gate types.
            covered = _unread_tags(unread)
            skipped = set(not_imported) - covered
            for gt in ('CurlyQuad', 'BooleanGate'):
                if gt not in covered and any(True for _ in self.root.iter(gt)):
                    skipped.add(gt)
            if skipped:
                log.warning(
                    "[WspReader] Skipped unsupported gate types: %s",
                    sorted(skipped))
            _log_unread('workspace', unread)

        if gates:
            # Indented log so the hierarchy is visible.
            id_to_depth = {}
            for g in gates:
                pid = g.get('parent_id')
                id_to_depth[g['_import_id']] = (
                    id_to_depth.get(pid, -1) + 1 if pid else 0)
            log.info(f"[WspReader] Extracted {len(gates)} gate(s):")
            for g in gates:
                d = id_to_depth.get(g['_import_id'], 0)
                log.info(f"   {'  ' * d}{describe_gate(g)}")
        else:
            log.info("[WspReader] No supported gates found.")

        return gates

    def _extract_matrix(self, node):
        """Parse a matrix element and register it in `self.matrices` under its
        name. A name already holding a DIFFERENT matrix gets a numbered key
        instead: FlowJo calls every file's own SPILL "Acquisition-defined",
        and keying by bare name kept only the last of them, so a lookup by
        name handed every sample the last file's matrix."""
        m = self._parse_matrix(node)
        if m is None:
            return
        name, key, k = m['name'], m['name'], 1
        while key in self.matrices:
            old = self.matrices[key]
            if (old['channels'] == m['channels']
                    and np.array_equal(old['matrix'], m['matrix'])):
                return                   # a sample's copy of a listed matrix
            k += 1
            key = f'{name} ({k})'
        self.matrices[key] = m

    def _parse_matrix(self, node):
        """A compensation-matrix element as dict(matrix, channels, prefix,
        suffix, name), or None when it holds no usable values."""
        name   = node.get('name') or node.get('matrixName') or 'unnamed'
        prefix = node.get('prefix', 'Comp-')
        suffix = node.get('suffix', '')
        channels = [p.get('name') or p.get('PnN')
                    for p in node.iter('parameter')
                    if p.get('name') or p.get('PnN')]
        if not channels:
            return
        n      = len(channels)
        values = []
        for vn in node.iter('spilloverValues'):
            try:
                values = [float(v) for v in vn.get('values', '').split(',')]
            except ValueError:
                pass
        matrix = None
        if len(values) == n * n:
            matrix = np.array(values).reshape(n, n)
        else:
            rows = []
            for row in node.iter('spilloverRow'):
                try:
                    rows.append([float(v) for v in row.get('values','').split(',')])
                except ValueError:
                    pass
            if len(rows) == n:
                matrix = np.array(rows)
        # Gating-ML v2 layout (FlowJo v10 .wsp):
        #   <spillover parameter="src"><coefficient parameter="dst" value="x"/></spillover>
        # Build the matrix row-by-row from the source/destination channel
        # pairs against our channel index. Identity diagonal as the default.
        if matrix is None:
            ch_idx = {c: i for i, c in enumerate(channels)}
            m = np.eye(n)
            seen = 0
            declared = set()
            for sp in node.iter('spillover'):
                src = sp.get('parameter')
                if src not in ch_idx:
                    continue
                declared.add(src)
                i = ch_idx[src]
                for coef in sp.iter('coefficient'):
                    dst = coef.get('parameter')
                    val = coef.get('value')
                    if dst in ch_idx and val is not None:
                        try:
                            m[i, ch_idx[dst]] = float(val)
                            seen += 1
                        except ValueError:
                            pass
            if seen >= n:    # at minimum the diagonal entries should show up
                # `seen` counts COEFFICIENTS, not channels, so a file in which
                # a couple of detectors carry a full row clears the bar while
                # others are never mentioned at all. Those keep the identity
                # row they were initialised with, which silently means "this
                # detector has no spillover" — and spillover from every other
                # channel then leaks into it uncorrected. Verified: 2 of 4
                # channels declaring 8 coefficients was accepted with the
                # other 2 left uncompensated. Still accepted, because the
                # declared rows are usable and a partial matrix beats none,
                # but no longer in silence.
                missing = [c for c in channels if c not in declared]
                if missing:
                    log.warning(
                        "[WspReader] '%s': no spillover declared for %d of %d "
                        "channels (%s). They are left UNCOMPENSATED — check "
                        "the workspace if you expected them to be corrected.",
                        name, len(missing), n, ', '.join(missing))
                matrix = m
        if matrix is None:
            log.info(f"[WspReader] Could not parse values for '{name}'.")
            return None
        return dict(matrix=matrix, channels=channels, prefix=prefix,
                    suffix=suffix, name=name)

    _MATRIX_TAGS = ('CompensationMatrix', 'spilloverMatrix', 'Compensation')

    def sample_matrix(self, sample_elem):
        """The matrix a <Sample> carries for itself (FlowJo writes a copy of
        it directly inside the Sample, after <DataSet>), parsed; None when
        the sample has none -- FlowJo then applies its Acquisition-defined
        matrix, i.e. the FCS file's own $SPILL."""
        if sample_elem is None:
            return None
        for child in sample_elem:
            if child.tag in self._MATRIX_TAGS:
                m = self._parse_matrix(child)
                if m is not None:
                    return m
        return None

    def matrix_for_fcs(self, fcs_path):
        """`(listed, matrix)` for the FCS at `fcs_path`: `listed` is True when
        a <Sample> of this workspace has that file as its DataSet (matched by
        full path, else by an unambiguous file name), and `matrix` is that
        sample's own matrix (see `sample_matrix`) or None when it has none."""
        if self.root is None or not fcs_path:
            return False, None
        want = os.path.normcase(os.path.abspath(str(fcs_path)))
        base = os.path.basename(want)
        exact, by_name = [], []
        for sample_elem in self.root.iter('Sample'):
            ds = sample_elem.find('DataSet')
            p = _wsp_uri_path(ds.get('uri') if ds is not None else None)
            if not p:
                continue
            p = os.path.normcase(os.path.abspath(p))
            if p == want:
                exact.append(sample_elem)
            elif os.path.basename(p) == base:
                by_name.append(sample_elem)
        hits = exact or by_name
        if len(hits) != 1:
            if len(hits) > 1:
                log.warning("[WspReader] %d samples in %s point at %s -- not "
                            "choosing between them.", len(hits),
                            os.path.basename(str(self.path)), base)
            return False, None
        return True, self.sample_matrix(hits[0])

    def distinct_matrix_count(self):
        return len(self.matrices)

    def get_matrix(self, name=None):
        if not self.matrices:
            raise RuntimeError("No matrices available.")
        if name is None:
            return next(iter(self.matrices.values()))
        if name not in self.matrices:
            raise KeyError(f"'{name}' not found. Available: {list(self.matrices)}")
        return self.matrices[name]

    def print_matrices(self):
        for name, m in self.matrices.items():
            log.info(f"  '{name}' — {len(m['channels'])} ch: {m['channels']}")


# ══════════════════════════════════════════════════════════════════════════════
# QC MODULE
# ══════════════════════════════════════════════════════════════════════════════

class AcquisitionQC:
    """
    Acquisition QC. Three independent anomaly detectors, combined into one
    clean-event index:

      1. **Signal drift** — time bins where any channel's events sit higher
         or lower than in its other bins: the bin's mean rank is more than
         ``threshold`` SDs from the median bin's (sensor drift, settling,
         sustained instability).
      2. **Flow-rate anomalies** — time bins whose event count is a robust
         (MAD) outlier, or interior bins that are empty: clogs (rate
         collapses) and bubbles (a gap or a burst).
      3. **Margin / saturation events** — events piled up at a channel's
         ceiling (off-scale), the classic signature of a bubble/clog or
         electronic saturation. These are dropped per-event, not per-bin.

    Detectors 1–2 need a Time channel and no-op without one; detector 3
    does not. A clean acquisition trips none of them.

    Usage
    -----
        qc = AcquisitionQC(sample.data)
        clean_idx = qc.run(n_bins=200, threshold=5)
        sample.data = sample.data.loc[clean_idx].reset_index(drop=True)
        qc.plot()
        qc.report   # {'drift': n, 'flow_rate': n, 'margin': n, 'total': n}
    """

    def __init__(self, data, time_channel='Time'):
        self.data         = data
        self.time_channel = self._find_time(data, time_channel)
        self.flag         = None
        self.bin_stats    = None
        self.bin_counts   = None
        self.report       = {}

    @staticmethod
    def _find_time(data, hint):
        if hint in data.columns:
            return hint
        for col in data.columns:
            if 'time' in col.lower():
                return col
        return None

    @staticmethod
    def _mad_outliers(vals, threshold, min_scale=0.0):
        """Boolean mask of robust (median ± threshold·MAD) outliers. The
        flow-rate detector's test; drift has its own, :meth:`_drift_outliers`.

        A MAD of exactly 0 is routine here: event counts tie exactly on a
        steady acquisition. (Per-bin medians of an integer channel tied too,
        when drift used them.) The previous ``+ 1e-10`` epsilon turned that
        into a band of width ~1e-10, so any value differing by a single
        quantisation step was an outlier. Measured: a clean, drift-free
        integer channel lost 38% of its events to "drift" while the identical
        data as float lost none.

        A zero MAD carries no scale, so one must come from outside — and it
        must NOT be derived from the deviations themselves, or a lone genuine
        outlier sets the very band meant to catch it (a 350-event burst among
        uniform 50s made its own band 1750 and escaped). ``min_scale`` is
        therefore supplied by the caller from the statistics of its own data:
        the Poisson noise for a count. It is a floor, not a replacement — a
        real MAD still wins.

        With no scale from either source there is no spread to speak of, and
        nothing can be called an outlier.
        """
        vals = np.asarray(vals, dtype=float)
        med   = np.median(vals)
        dev   = np.abs(vals - med)
        scale = max(float(np.median(dev)), float(min_scale))
        if not scale > 0:
            return np.zeros(vals.shape, dtype=bool)
        return dev > threshold * scale

    @staticmethod
    def _mid_ranks(vals, cells=65536):
        """Each value's mid-rank in its channel, in [0, 1]: the fraction of
        events below it plus half of those level with it. ``vals`` finite.

        Values are placed on a grid of ``cells`` equal steps across roughly
        the 0.1 to 99.9 percentile range (the tails go to the end cells), and
        a cell's events share its mid-rank. That is exact for an integer
        channel spanning fewer steps, and any monotone grid still gives a
        valid rank test. Placement is arithmetic, and the range comes from a
        fixed stride of ~20k values: measured on 1M events, the whole
        function takes 10 ms a channel, where a binary search over the cells
        took 120 ms and a full percentile alone 13 ms."""
        v = np.asarray(vals, dtype=float)
        if v.size == 0:
            return v
        lo, hi = np.percentile(v[::max(1, v.size // 20_000)], [0.1, 99.9])
        if not hi > lo:
            return np.full(v.size, 0.5)
        cell = v - lo
        cell *= (cells - 1) / (hi - lo)
        np.clip(cell, 0, cells - 1, out=cell)
        cell = cell.astype(np.intp)
        n_at = np.bincount(cell, minlength=cells)
        return ((np.cumsum(n_at) - 0.5 * n_at) / v.size)[cell]

    @staticmethod
    def _drift_outliers(mr, n, threshold, floor, skew, kurt, clip=3.0):
        """Boolean mask of the bins whose mean rank ``mr`` (of ``n`` events)
        is more than ``threshold`` SDs from the centre. See
        :meth:`_drift_bad_bins` for why.

        A bin's deviation is rescaled to the median bin's event count, so
        every bin has the same null SD. The SD is 1.4826·MAD of those
        deviations, at least ``floor``. The bins within ``clip`` SDs of the
        median bin are kept and the rest set aside, re-estimated until that
        set is stable, so a fault spanning many bins does not widen the band
        meant to catch it. The centre the cut is measured from is then the
        mean rank of the kept bins' events: under the null that is every
        bin's expected mean rank, and on a skewed channel the median bin sits
        below it.

        The band is widened by Cornish-Fisher terms about the mean, never
        narrowed: on its long side by (t²-1)·skew/6, ``skew`` being the null
        skewness of a bin's mean rank, and on both sides by (t³-3t)·kurt/24,
        ``kurt`` its excess kurtosis beyond what that skewness brings in a
        Poisson count (the second-order terms that would narrow the band
        are left out). ``floor``, ``skew`` and ``kurt`` may be per bin."""
        mr = np.asarray(mr, dtype=float)
        n = np.asarray(n, dtype=float)
        w = np.sqrt(n / float(np.median(n)))
        floor = np.broadcast_to(np.asarray(floor, dtype=float), mr.shape)
        skew = np.broadcast_to(np.asarray(skew, dtype=float), mr.shape)
        kurt = np.broadcast_to(np.asarray(kurt, dtype=float), mr.shape)

        def spread(d, keep):
            kept = d[keep]
            return np.maximum(MAD_TO_SD * float(
                np.median(np.abs(kept - np.median(kept)))), floor)

        def band(t, sd):
            g = (t * t - 1.0) * skew / 6.0
            k = t + (t ** 3 - 3.0 * t) * np.maximum(kurt, 0.0) / 24.0
            return (k + np.maximum(g, 0.0)) * sd, (k + np.maximum(-g, 0.0)) * sd

        inl = np.ones(mr.size, dtype=bool)
        for _ in range(10):
            d = (mr - float(np.median(mr[inl]))) * w
            above, below = band(clip, spread(d, inl))
            nxt = (d <= above) & (-d <= below)
            if nxt.sum() < 0.5 * mr.size or bool((nxt == inl).all()):
                break
            inl = nxt
        d = (mr - float(np.sum(n[inl] * mr[inl]) / np.sum(n[inl]))) * w
        above, below = band(threshold, spread(d, inl))
        return (d > above) | (-d > below)

    def _drift_bad_bins(self, bins, channels, n_bins, threshold):
        """Time bins whose events sit higher or lower in a channel than that
        channel's other bins do: the bin's MEAN RANK is more than
        ``threshold`` SDs from the centre. Also populates self.bin_stats
        (per-bin mean ranks).

        Every (bin, channel) pair is a test, so on a clean, stationary
        acquisition the statistic must have the same, known spread in every
        channel and bin, and the cut must hold for the whole family. The
        per-bin RAW MEDIAN at 5 unscaled MADs had neither:

        * 5 MAD is only 3.4 SD, a two-sided tail of 7.5e-4: 200 bins x 7
          channels expected about one false bin per clean sample. At 5 SD
          the tail is 5.7e-7, under 0.01 false bins even at 200 x 64.
        * A bin's raw median depends on the channel's shape. On a bimodal
          marker whose median sits at the edge of the gap between negative
          and positive events, it jumps across the gap whenever chance puts
          the bin's positive fraction past one half. The shuffled synthetic
          PBMC sample lost 7556 of 20000 events to "drift" (CD3 flagged 75
          of 200 bins).

        Measured at the loader defaults on clean, stationary data, 240
        synthetic samples (normal and log-normal, 10k-100k events, 4-16
        fluors, float and integer, a constant and a 3x falling event rate):
        the raw median removed events from 228 of them, a mean of 0.8% to
        3.7% per group; this removes none.

        A rank does not depend on the channel's shape, and the mean rank
        uses every event, so a composition change moves it smoothly. The
        mean of n ranks has a sampling SD of sqrt(var(rank) / n), so each
        bin's deviation is rescaled to the median bin's event count: a bin
        that is sparse because the rate fell is noisier, not drifting.

        :meth:`_drift_outliers` makes the cut. Each part below was needed,
        and removing any one of them fails a test in
        tests/test_acquisition_qc_ground_truth.py:

        * The SD is 1.4826·MAD across bins, so bin-to-bin spread beyond the
          sampling noise (2% gain jitter per bin) is the channel's spread,
          not drift. Unscaled, 9 of 10 such samples lost bins.
        * The SD is at least the sampling SD above, a figure from the
          events, not from the deviations being judged. The MAD of 200 bins
          is itself off by ~8%; clean log-normal samples lost a bin where it
          fell low (3 of 30 at 100k x 10 fluors).
        * Bins more than 3 SDs out are set aside before the centre and SD
          are taken, re-estimated until stable. A bin's mean rank can move
          by at most ~0.5 however large the fault, and a fault over 40% of a
          short run inflated the MAD until the 5-SD band passed that limit:
          one channel shifted by 1000 SD lost 29% of its faulty events at
          5000 events (none on one seed) and 6% at 3000. Now 99.9% and 96%,
          as with the raw median. A smooth trend is not set aside and is
          tolerated as before.
        * The centre is the mean rank of the kept bins' events, not the
          median bin: under the null that mean is every bin's expectation,
          and on a skewed channel the median bin sits below it.
        * On a channel mostly tied at one value, a bin's mean rank counts
          its rare events, and 5 SDs is no longer a 5.7e-7 tail. One-sided
          (0 but for 2% of events, 40 channels, 50k events), 11 of 50 clean
          samples lost a bin; two-sided (0.1% below, 0.1% above), 10 of 20.
          The band is widened by Cornish-Fisher skew and kurtosis terms,
          from the events' own moments over that bin's event count: taken
          at the median bin, sparse bins where the rate fell (the tube
          running dry) cost 12 of 20 clean samples a bin.

        Sensitivity, one fluor shifted by k SEs of a bin's median over 10%
        of a 50k-event run (20 seeds): 40% of the shifted events removed at
        k=4, 82% at k=5, 98% at k=6, all from k=8; outside the shift, 5
        events in all 20 runs, from the bin straddling its edge. The raw
        median took 54%, 84% and 96%, and removed 11067 events outside it.

        The widening is the price of holding the 5-SD false-positive rate on
        tied channels, and it is paid in sensitivity there (10 seeds, a 50k
        run, removed share of the faulty events; in brackets, an earlier
        version without the kurtosis term or per-bin moments, which lost
        events from up to 12 of 20 CLEAN samples of these kinds):

        * one-sided sparse channel, positive share 2% -> 10% over 10% of
          the run: 71%; 10% -> none: 57%;
        * two-sided (0, with rare events below and above), the upper share
          0.5% -> 4% over 10% of the run: 14% (34%); 1% -> 8%: 66% (77%);
        * an unused integer detector at 37 whose +1 glitches go 0.5% -> 5%
          over 2% of the run: 55% (82%).

        The raw median caught none of the last three (its removals there
        were its 0.5-0.7% background on clean data).

        Bins with < 10 events contribute nothing (NaN in bin_stats, which
        keeps a row per bin, len == n_bins, for the QC plot)."""
        b = np.asarray(bins)
        valid = ~pd.isna(b)
        all_valid = bool(valid.all())
        bi = (b if all_valid else b[valid]).astype(np.intp)
        n_all = np.bincount(bi, minlength=n_bins)[:n_bins]

        stats = {}
        bad = set()
        for ch in channels:
            if not pd.api.types.is_numeric_dtype(self.data[ch]):
                continue
            col = np.asarray(self.data[ch].to_numpy(), dtype=float)
            if not all_valid:
                col = col[valid]
            fin = np.isfinite(col)
            if fin.all():                    # the usual case: no copies
                vals, b_fin, n_in = col, bi, n_all
            else:
                vals, b_fin = col[fin], bi[fin]
                n_in = np.bincount(b_fin, minlength=n_bins)[:n_bins]
            rank = self._mid_ranks(vals)
            ok = n_in >= 10
            mean_rank = np.full(n_bins, np.nan)
            mean_rank[ok] = (np.bincount(b_fin, weights=rank,
                                         minlength=n_bins)[:n_bins][ok]
                             / n_in[ok])
            stats[ch] = mean_rank
            if not ok.any():
                continue
            n_ok = n_in[ok].astype(float)
            n_ref = float(np.median(n_ok))
            dev = rank - rank.mean()
            sq = dev * dev
            var = float(sq.mean())
            if not var > 0:              # every event level: nothing to drift
                continue
            # Per bin: the skewness and excess kurtosis of a mean of that
            # bin's events, the events' own over sqrt(n) and n, so larger in
            # a sparse bin. Moments by dot product: ** 3 on 1M events cost
            # ~31 ms a channel.
            g1 = float(np.dot(sq, dev)) / dev.size / var ** 1.5
            k1 = float(np.dot(sq, sq)) / dev.size / var ** 2 - 3.0 - g1 * g1
            out = self._drift_outliers(
                mean_rank[ok], n_ok, threshold, float(np.sqrt(var / n_ref)),
                g1 / np.sqrt(n_ok), k1 / n_ok)
            bad.update(np.flatnonzero(ok)[out].tolist())
        self.bin_stats = pd.DataFrame(stats, index=range(n_bins))
        return bad

    def _flowrate_bad_bins(self, bins, n_bins, threshold):
        """Time bins whose event count is a MAD outlier, plus empty
        interior bins (a gap = bubble; a collapse = clog). Edge bins are
        exempt from the 'empty' rule — acquisitions routinely start/stop
        mid-bin. Also populates self.bin_counts."""
        # Vectorised per-bin counts (bincount) — identical to the former
        # per-bin count_nonzero loop, O(N) instead of O(N·n_bins).
        b = np.asarray(bins)
        valid = ~pd.isna(b)
        counts = np.bincount(b[valid].astype(int),
                             minlength=n_bins)[:n_bins].astype(int)
        self.bin_counts = counts
        nonempty = counts[counts > 0]
        if nonempty.size < 3:
            return set()
        bad = set()
        # Count outliers among bins that actually have events.
        nz_idx = np.where(counts > 0)[0]
        # Counting statistics: a bin holding ~N events carries sqrt(N) noise
        # of its own, so that is the floor when the counts tie exactly.
        out    = self._mad_outliers(
            counts[nz_idx], threshold,
            min_scale=float(np.sqrt(max(np.median(counts[nz_idx]), 1.0))))
        bad.update(nz_idx[out].tolist())
        # Empty interior bins (gaps), ignoring leading/trailing empties.
        # Only meaningful when bins are densely populated — on a sparse
        # file an empty interior bin is expected, not an anomaly.
        first, last = int(nz_idx[0]), int(nz_idx[-1])
        if np.median(nonempty) >= 20:
            for b in range(first + 1, last):
                if counts[b] == 0:
                    bad.add(b)
        return bad

    @staticmethod
    def _margin_events(data, channels, frac, eps=1e-9):
        """Boolean per-event mask of margin/saturation events: those at a
        channel's ceiling (its max) when that ceiling is *piled up* — i.e.
        at least `frac` of events share the max value. A single off-scale
        max (continuous data) isn't a pile-up and is left alone."""
        n = len(data)
        bad = np.zeros(n, dtype=bool)
        if n == 0:
            return bad
        thresh = max(2, int(np.ceil(frac * n)))
        for ch in channels:
            col = np.asarray(data[ch].values, dtype=float)
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                continue
            # A channel with no range at all (a dead or disabled detector,
            # constant across the file) has no ceiling to pile up against:
            # every event is trivially "at the max". Flagging them deleted
            # 100% of an otherwise healthy sample because of ONE unused
            # channel, since margins are OR-ed across the panel.
            if float(finite.max() - finite.min()) <= eps:
                continue
            ceiling = finite.max()
            at_ceiling = np.isfinite(col) & (np.abs(col - ceiling) <= eps)
            if int(at_ceiling.sum()) >= thresh:
                bad |= at_ceiling
        return bad

    @staticmethod
    def _time_bins(t, n_bins):
        """Each event's equal-width Time bin (``pd.cut`` labels), NaN where
        Time is not finite; None when no Time value is. ``pd.cut`` raises on
        ±inf ("cannot specify integer `bins` when input data contains
        infinity"), so one infinite Time crashed QC, and with it the GUI load
        that runs QC on every file. Such events, like a NaN Time, get no bin:
        the time-based detectors cannot judge them, and they are kept."""
        t = np.asarray(t, dtype=float)
        fin = np.isfinite(t)
        if not fin.any():
            return None
        if not fin.all():
            t = np.where(fin, t, np.nan)
        return pd.cut(t, bins=n_bins, labels=False)

    def run(self, n_bins=200, threshold: float = 5, channels=None,
            drift=True, flow_rate=True, margins=True, flow_rate_threshold=5.0,
            margin_frac=0.01):
        n = len(self.data)
        keep = np.ones(n, dtype=bool)
        report = {'drift': 0, 'flow_rate': 0, 'margin': 0}

        if channels is None:
            # Boolean columns are never detectors. The FMO step writes one
            # `<channel>_pos` flag per marker, and numpy treats bools as 0/1 —
            # so the margin detector saw "every event piled up at the ceiling"
            # and deleted every marker-POSITIVE event (measured: 49.8% of a
            # clean sample). Name-matching those away would be fragile; the
            # dtype is the honest signal.
            channels = [c for c in self.data.columns
                        if c != self.time_channel
                        and not _is_excluded(c)
                        and not pd.api.types.is_bool_dtype(self.data[c])
                        and pd.api.types.is_numeric_dtype(self.data[c])]

        bins = None
        if self.time_channel is not None and n_bins > 0 and n > 0:
            bins = self._time_bins(self.data[self.time_channel].values,
                                   n_bins)
        if bins is not None:
            bin_arr = np.asarray(bins)
            unbinned = int(pd.isna(bin_arr).sum())
            if unbinned:
                log.info("  [QC] %s events with no finite Time are not "
                         "time-binned: drift and flow-rate cannot judge "
                         "them, so they are kept.", f"{unbinned:,}")

            if drift:
                drift_bad = self._drift_bad_bins(
                    bins, channels, n_bins, threshold)
                if drift_bad:
                    drift_evt = np.isin(bin_arr, list(drift_bad))
                    report['drift'] = int(drift_evt.sum())
                    keep &= ~drift_evt

            if flow_rate:
                flow_bad = self._flowrate_bad_bins(
                    bins, n_bins, flow_rate_threshold)
                if flow_bad:
                    flow_evt = np.isin(bin_arr, list(flow_bad))
                    report['flow_rate'] = int(flow_evt.sum())
                    keep &= ~flow_evt
        else:
            log.info("  [QC] No time channel with a finite value — "
                     "time-based detectors skipped.")

        if margins:
            margin_evt = self._margin_events(self.data, channels, margin_frac)
            report['margin'] = int(margin_evt.sum())
            keep &= ~margin_evt

        report['total'] = int((~keep).sum())
        self.report = report
        self.flag   = pd.Series(keep, index=self.data.index)
        pct_rem     = (report['total'] / n * 100.0) if n else 0.0
        log.info(
            "  [QC] Removed %.1f%% events (%s total: drift %s, flow-rate %s, "
            "margin %s).",
            pct_rem, f"{report['total']:,}", f"{report['drift']:,}",
            f"{report['flow_rate']:,}", f"{report['margin']:,}")
        return self.data.index[keep]

    def plot(self):
        if self.bin_stats is None:
            log.info("  [QC] Run .run() first.")
            return
        import matplotlib.pyplot as plt  # lazy: see module-top comment
        bins = self._time_bins(self.data[self.time_channel].values,
                               len(self.bin_stats))
        cnts = pd.Series(bins).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(np.asarray(cnts.index), np.asarray(cnts.values),
                lw=0.8, color='steelblue')
        ax.set_xlabel('Time bin')
        ax.set_ylabel('Event count')
        ax.set_title('Acquisition QC — event rate over time')
        plt.tight_layout()
        return ax


# ══════════════════════════════════════════════════════════════════════════════
# CELL CYCLE
# ══════════════════════════════════════════════════════════════════════════════
#
# DNA-content cell-cycle modelling. Given a DNA-stain intensity (PI, DAPI,
# FxCycle, 7-AAD, Hoechst, DRAQ5, …), the histogram is bimodal: a G0/G1
# peak and a G2/M peak at ~2× the DNA content, with S phase spread between.
# We locate the two peaks, estimate each peak's spread robustly, and assign
# every event to a phase by intensity boundaries — a pragmatic, explainable
# alternative to the Dean-Jett-Fox / Watson deconvolution that doesn't need
# a curve-fitter and degrades gracefully on non-cycling samples.

# Dye name tokens we recognise as DNA-content stains (lowercased substrings,
# matched against antibody label first, then detector name).
DNA_DYES = (
    'fxcycle', 'propidium iodide', 'propidium', 'hoechst', 'draq5', 'draq7',
    'vybrant dyecycle', 'dyecycle', 'sytox', 'to-pro', 'topro', 'dapi',
    '7-aad', '7aad', 'pi',
)

# Ordered phase labels. 'cycling' = G1+S+G2M; sub-G1 (apoptotic/debris) and
# >G2M (aggregates/polyploid) are reported but excluded from the cycle %.
CELL_CYCLE_PHASES = ('sub-G1', 'G1', 'S', 'G2M', '>G2M')


def _robust_sd(arr):
    """1.4826 * MAD — outlier-resistant SD estimate. NaN on empty."""
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float('nan')
    med = np.median(a)
    return MAD_TO_SD * float(np.median(np.abs(a - med)))


def find_dna_channel(sample):
    """Best-guess DNA-content detector for `sample`, or None.

    Matches known DNA-dye tokens against each channel's antibody label
    first, then its detector name. Prefers an Area (`-A`) channel. The
    short token 'pi' only matches as a whole word so it doesn't fire on
    'PI3K' etc."""
    labels = getattr(sample, 'channel_labels', {}) or {}
    cols = list(sample.data.columns)

    def matches(text):
        t = str(text).lower()
        for dye in DNA_DYES:
            if dye == 'pi':
                # Exclude digits from the boundaries too, else 'PI3K' matches
                # ('3' satisfies a bare (?![a-z]) lookahead) and gets mis-tagged
                # as the PI DNA stain — matching find_viability_channel's guard.
                if re.search(r'(?<![a-z0-9])pi(?![a-z0-9])', t):
                    return True
            elif dye in t:
                return True
        return False

    candidates = [det for det in cols
                  if matches(labels.get(det, det)) or matches(det)]
    if not candidates:
        return None
    for c in candidates:
        if c.upper().endswith('-A'):
            return c
    return candidates[0]


def analyze_dna(values, k=1.5, bins=256):
    """Model the DNA-content histogram → cell-cycle phase boundaries + %.

    Pure. `values` is the DNA-stain intensity (linear scale). `k` sets how
    many robust SDs around each peak count as G1 / G2M (the rest, between,
    is S). Returns a model dict:
        g1_mean, g1_sd, g2_mean, g2_sd, g1_hi, g2_lo  (phase boundaries)
        pct_g1, pct_s, pct_g2m   (% of cycling events)
        counts  {phase: n}, n_cycling, n  (totals)
        ok      (bool — False when no usable peak was found)
    Use assign_phase(values, model) to label arbitrary arrays with the
    same boundaries."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    nan = float('nan')
    model = {'g1_mean': nan, 'g1_sd': nan, 'g2_mean': nan, 'g2_sd': nan,
             'g1_hi': nan, 'g2_lo': nan, 'pct_g1': nan, 'pct_s': nan,
             'pct_g2m': nan, 'counts': {}, 'n_cycling': 0, 'n': 0, 'ok': False}

    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    model['n'] = int(v.size)
    if v.size < 50:
        return model

    lo, hi = np.percentile(v, [0.5, 99.5])
    if hi <= lo:
        return model
    bins = _resolution_bins(v, lo, hi, bins)
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    sm = gaussian_filter1d(hist.astype(float), sigma=2.0)
    if sm.max() <= 0:
        return model

    peaks, _ = find_peaks(sm, prominence=sm.max() * 0.05)
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(sm))])
    # G1 = the most prominent (tallest) peak.
    order = peaks[np.argsort(sm[peaks])[::-1]]
    g1_mean = float(centers[order[0]])
    if g1_mean <= 0:
        g1_mean = float(np.median(v))

    # G2/M ≈ 2× G1: nearest remaining peak in [1.7×, 2.3×]; else nominal 2×.
    g2_mean = None
    for pk in order[1:]:
        c = float(centers[pk])
        if 1.7 * g1_mean <= c <= 2.3 * g1_mean:
            g2_mean = c
            break
    if g2_mean is None:
        g2_mean = 2.0 * g1_mean

    g1_win = v[(v >= 0.85 * g1_mean) & (v <= 1.15 * g1_mean)]
    g1_sd = _robust_sd(g1_win)
    if not np.isfinite(g1_sd) or g1_sd <= 0:
        g1_sd = 0.05 * g1_mean
    g2_win = v[(v >= 0.9 * g2_mean) & (v <= 1.1 * g2_mean)]
    g2_sd = _robust_sd(g2_win)
    if not np.isfinite(g2_sd) or g2_sd <= 0:
        g2_sd = g1_sd * 1.4   # G2 CV ~ G1 CV; width scales with mean

    g1_hi = g1_mean + k * g1_sd
    g2_lo = g2_mean - k * g2_sd
    if g1_hi >= g2_lo:                      # peaks overlap → split at midpoint
        mid = 0.5 * (g1_mean + g2_mean)
        g1_hi = min(g1_hi, mid)
        g2_lo = max(g2_lo, mid)

    model.update(g1_mean=g1_mean, g1_sd=g1_sd, g2_mean=g2_mean, g2_sd=g2_sd,
                 g1_hi=g1_hi, g2_lo=g2_lo, ok=True)

    phases = assign_phase(v, model)
    counts = {p: int(np.count_nonzero(phases == p)) for p in CELL_CYCLE_PHASES}
    cyc = counts['G1'] + counts['S'] + counts['G2M']
    model['counts'] = counts
    model['n_cycling'] = cyc
    if cyc > 0:
        model['pct_g1'] = 100.0 * counts['G1'] / cyc
        model['pct_s'] = 100.0 * counts['S'] / cyc
        model['pct_g2m'] = 100.0 * counts['G2M'] / cyc
    return model


def assign_phase(values, model):
    """Label each value with its cell-cycle phase using `model`'s
    boundaries. Non-finite values and any input when the model is invalid
    become 'NA'. Returns an object ndarray of CELL_CYCLE_PHASES (+ 'NA')."""
    v = np.asarray(values, dtype=float)
    out = np.full(v.size, 'NA', dtype=object)
    if not model.get('ok'):
        return out
    g1_lo = model['g1_mean'] - (model['g1_hi'] - model['g1_mean'])
    g1_hi = model['g1_hi']
    g2_lo = model['g2_lo']
    g2_hi = model['g2_mean'] + (model['g2_mean'] - model['g2_lo'])
    finite = np.isfinite(v)
    out[finite] = '>G2M'                         # default for finite > g2_hi
    out[finite & (v < g1_lo)] = 'sub-G1'
    out[finite & (v >= g1_lo) & (v <= g1_hi)] = 'G1'
    out[finite & (v > g1_hi) & (v < g2_lo)] = 'S'
    out[finite & (v >= g2_lo) & (v <= g2_hi)] = 'G2M'
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FMO GATE THRESHOLDER
# ══════════════════════════════════════════════════════════════════════════════

class FMOGater:
    """
    Calculate positive thresholds from FMO control files.

    Usage
    -----
        gater = FMOGater()
        gater.add_fmo('Comp-FL1-A', 'fmo_fl1.fcs')  # CD901b
        gater.add_fmo('Comp-FL2-A',           'fmo_fl2.fcs')   # CD902
        gater.add_fmo('Comp-FL3-A',        'fmo_fl3.fcs')   # CD903
        thresholds = gater.compute(percentile=99.0)
        sample.apply_threshold_gates(thresholds)
    """

    def __init__(self):
        self.fmos        = {}
        self._is_fallback = {}   # channel -> True if using unstained instead of FMO

    def add_fmo(self, channel_name, fcs_path, is_fallback=False):
        """
        channel_name: detector name this FMO (or fallback unstained) controls.
        is_fallback:  True when the file is an unstained control rather than a
                      proper FMO — noted in compute() output.
        """
        self.fmos[channel_name]         = FlowSample(fcs_path)
        self._is_fallback[channel_name] = is_fallback
        return self

    def add_fmos_from_dir(self, directory, pattern=r'fmo[_-]?(\w+)'):
        rx = re.compile(pattern, re.IGNORECASE)
        for fname in os.listdir(directory):
            if not fname.lower().endswith('.fcs'):
                continue
            m = rx.search(fname)
            if m:
                ch  = m.group(1)
                self.fmos[ch] = FlowSample(os.path.join(directory, fname))
                log.info(f"  [FMO] '{ch}' ← {fname}")
        return self

    def prepare(self, wsp_path=None, transform_method='logicle'):
        """
        Compensate and transform all FMO samples so thresholds are in the
        same data space as the experimental samples.  Call before compute().
        """
        for _ch, sample in self.fmos.items():
            done = list(sample._transformed_channels())
            if done:
                # Prepared before: back to linear, so the compensation below
                # replaces the earlier one instead of meeting logicle values.
                sample.retransform(done, method='linear')
            if wsp_path:
                sample.compensate_from_wsp(wsp_path)
            else:
                sample.auto_compensate()
            sample.apply_transform(method=transform_method)
        return self

    def compute(self, percentile=99.0):
        """
        Returns {channel_name: threshold} where channel_name is the key
        passed to add_fmo() — typically the compensated detector name.
        The threshold is the p`percentile` of that channel in the FMO:
        99 by default (1 % of FMO events above the gate, the usual choice
        and the app's FMO dialog default).
        """
        thresholds = {}
        for ch, sample in self.fmos.items():
            col = self._find_col(sample, ch)
            if col is None:
                log.info(f"  [FMO] '{ch}' not found in {sample.name} — skipped.")
                continue
            val = float(np.percentile(sample.data[col].dropna(), percentile))
            thresholds[ch] = val
            tag = ' [UNSTAINED fallback]' if self._is_fallback.get(ch) else ''
            # The cut is in the transformed space the samples are gated in;
            # logged bare ('threshold=0.487', at 1,369 linear) it read as an
            # intensity. The linear value beside it is the one to compare
            # with an MFI.
            spec = (getattr(sample, 'data_transforms', None) or {}).get(col)
            scale = ''
            if spec and spec.get('method', 'linear') != 'linear':
                lin = float(to_linear(np.array([val]), spec)[0])
                scale = f" {spec['method']} (= {lin:,.0f} linear)"
            log.info(f"  [FMO] {ch}: threshold={val:.3f}{scale} "
                     f"(p{percentile}){tag}")
        return thresholds

    @staticmethod
    def _find_col(sample, hint):
        """Match by exact name, then by stripping common comp prefixes, then substring."""
        if hint in sample.data.columns:
            return hint
        stripped = re.sub(r'^[Cc]omp-', '', hint)
        for col in sample.data.columns:
            if col == stripped or col.lower() == stripped.lower():
                return col
        for col in sample.data.columns:
            if hint.lower() in col.lower() or stripped.lower() in col.lower():
                return col
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CORE SAMPLE CLASS
# ══════════════════════════════════════════════════════════════════════════════

def read_events_csv(path):
    """``(frame, notes)``: an events table from a CSV a user picked, and
    what had to be worked out to read it (as sentences for the user).

    ``pd.read_csv(path)`` alone read common exports wrong without a word:
    a file with no header row lost its first event to the column names; a
    semicolon-separated file with decimal commas (as European Excel writes)
    or a tab-separated one came back as ONE column; quoted decimal commas
    made every column text; and '#' comment lines at the top raised a raw
    ParserError. The delimiter is now detected, leading comment lines are
    skipped, decimal commas are read as decimals, an all-number first row
    is taken as data, and a file with no numeric column is refused."""
    import re as _re

    import pandas as pd
    with open(path, encoding='utf-8-sig', errors='replace') as fh:
        head = fh.read(1 << 16)
    lines = head.splitlines()
    skip = 0
    while skip < len(lines) and (not lines[skip].strip()
                                 or lines[skip].lstrip().startswith('#')):
        skip += 1
    body = [ln for ln in lines[skip:skip + 50] if ln.strip()]
    if not body:
        raise ValueError(f"{os.path.basename(path)} has no rows")
    sample = '\n'.join(body)
    notes = []
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=',;\t|').delimiter
    except csv.Error:
        sep = ','
    decimal = '.'
    if (sep != ',' and _re.search(r'\d,\d', sample)
            and not _re.search(r'\d\.\d', sample)):
        decimal = ','
    if sep != ',':
        notes.append({';': 'semicolon', '\t': 'tab', '|': 'bar'}[sep]
                     + '-separated' + (', with decimal commas'
                                       if decimal == ',' else ''))
    if skip:
        notes.append(f"{skip} comment line(s) at the top skipped")
    def _read(header: int | None = 0) -> pd.DataFrame:
        return pd.read_csv(path, sep=sep, decimal=decimal, skiprows=skip,
                           encoding='utf-8-sig', header=header)
    df = _read()

    def _number(x):
        try:
            float(str(x).strip())
            return True
        except ValueError:
            return False
    if len(df.columns) and all(_number(c) for c in df.columns):
        df = _read(header=None)
        # numpy.savetxt(header=...) puts the column names on a '#' line
        # just above the numbers. Skipped as a comment, its names were lost.
        last = lines[skip - 1].lstrip()[1:] if skip else ''
        names = [c.strip() for c in next(csv.reader([last], delimiter=sep), [])]
        if (len(names) == len(df.columns) and all(names)
                and not any(_number(c) for c in names)
                and len(set(names)) == len(names)):
            df.columns = names
            notes.append("column names read from the '#' line above the "
                         "numbers")
        else:
            df.columns = [f'Ch{i}' for i in range(1, len(df.columns) + 1)]
            notes.append("no header row: columns named Ch1 to "
                         f"Ch{len(df.columns)}")
    # Quoted decimal commas ("1,5") in a comma-separated file read as text.
    fixed = []
    for c in df.columns:
        col = df[c]
        # Text columns are object dtype, or str under pandas 3.
        if (pd.api.types.is_object_dtype(col)
                or pd.api.types.is_string_dtype(col)):
            v = col.dropna().astype(str).str.strip()
            if len(v) and v.str.fullmatch(r'-?\d+,\d+|-?\d+').all() \
                    and v.str.contains(',').any():
                df[c] = pd.to_numeric(col.astype(str).str.replace(
                    ',', '.', regex=False), errors='coerce')
                fixed.append(str(c))
    if fixed:
        notes.append("decimal commas read as decimals in "
                     + ', '.join(fixed[:6]) + (' ...' if len(fixed) > 6 else ''))
    if not any(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns):
        raise ValueError(f"{os.path.basename(path)} has no numeric column to "
                         "read as events")
    return df, notes


@contextlib.contextmanager
def _collect_warnings():
    """The WARNING messages this module logs on THIS thread inside the
    block, as a list (the pool loads several samples at once)."""
    import threading
    me = threading.get_ident()
    said = []

    class _Grab(logging.Handler):
        def emit(self, record):
            if record.thread == me and record.levelno >= logging.WARNING:
                said.append(record.getMessage().strip().lstrip('[!] ')
                            .strip())

    h = _Grab()
    log.addHandler(h)
    try:
        yield said
    finally:
        log.removeHandler(h)


# Writer quirks flowio can read past when asked, each with the option that
# reads it and the note the user gets. Tried in this order, adding options,
# because one file can carry more than one quirk.
_FCS_QUIRKS = (
    ('off_by_one', {'ignore_offset_error': True},
     "its data end is recorded one byte off (a known writer bug); "
     "the data itself is intact and was read in full"),
    ('discrepancy', {'ignore_offset_discrepancy': True},
     "its HEADER and TEXT disagree on where the data starts; "
     "read at the TEXT position, as the FCS standard says"),
    ('multiple', {'nextdata_offset': 0},
     "it holds more than one data set; only the first was loaded"),
)


def _fcs_quirk(exc):
    """Which entry of _FCS_QUIRKS ``exc`` (raised by flowio) names, or None."""
    name = type(exc).__name__
    text = str(exc).lower()
    if name == 'MultipleDataSetsError':
        return 'multiple'
    if name == 'DataOffsetDiscrepancyError':
        return 'discrepancy'
    if name == 'FCSParsingError' and 'off by 1' in text:
        return 'off_by_one'
    return None


def _quirk_note(note, exc):
    """`note` for the quirk `exc` names, saying where the data ENDS when
    that is what HEADER and TEXT disagree on (flowio raises one error for
    either end; the note always said "starts")."""
    if _fcs_data_end_discrepancy(exc):
        return note.replace('where the data starts', 'where the data ends')
    return note


def _fcs_data_end_discrepancy(exc):
    """Whether flowio's offset-discrepancy error `exc` is about the data's
    end ('DATA end byte location'), not its start."""
    return (type(exc).__name__ == 'DataOffsetDiscrepancyError'
            and ' end ' in f' {str(exc).lower()} ')


def read_fcs_tolerant(path, reader, notes):
    """``reader(path)`` (a FlowData class), retried with flowio's options for
    the writer quirks in _FCS_QUIRKS that other readers accept. Each quirk
    read past appends a plain-language note to ``notes``; anything else, or
    a quirk that persists, raises as before."""
    kw: dict = {}
    while True:
        try:
            if not kw:
                return reader(path)
            # flowio warns about the quirk it is reading past; the note
            # below says the same thing in the user's terms.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                return reader(path, **kw)
        except Exception as e:
            q = _fcs_quirk(e)
            entry = next((x for x in _FCS_QUIRKS if x[0] == q), None)
            if entry is None or all(kw.get(k) == v for k, v in entry[1].items()):
                raise
            kw.update(entry[1])
            notes.append(f"{os.path.basename(str(path))}: "
                         f"{_quirk_note(entry[2], e)}")


def plain_fcs_problem(exc):
    """A plain-language reason an FCS file could not be read, from the
    exception flowio raised: the raw text ("read() didn't return enough
    bytes", "invalid literal for int()") says nothing to a user."""
    import re as _re
    text = str(exc)
    m = _re.fullmatch(r"'?p(\d+)([a-z])'?", text.strip().lower())
    if isinstance(exc, KeyError) and m:
        what = {'r': 'channel range', 'b': 'bits per value',
                'n': 'channel name', 'e': 'amplification'}.get(m.group(2), '')
        return (f"required keyword $P{m.group(1)}{m.group(2).upper()}"
                + (f" ({what})" if what else '') + " is missing")
    if isinstance(exc, KeyError):
        key = text.strip("'\"").upper()
        return f"required keyword ${key} is missing"
    if isinstance(exc, (EOFError, ValueError, IndexError, struct_error_types())):
        return "the file is cut short, damaged, or not an FCS file"
    low = text.lower()
    q = _fcs_quirk(exc)
    if q == 'off_by_one':
        return "its data end is recorded one byte off, and reading past that failed"
    if q == 'discrepancy':
        where = 'ends' if _fcs_data_end_discrepancy(exc) else 'starts'
        return (f"its HEADER and TEXT disagree on where the data {where}, and "
                "the data could not be read at either position")
    if q == 'multiple':
        return "it holds more than one data set, and the first could not be read"
    if 'greater than file size' in low or 'offset' in low:
        return "the file is cut short (its data section runs past the end)"
    return "the file is damaged or not an FCS file"


def struct_error_types():
    import struct
    return struct.error


class _RangeFillingFlowData(_fcs_read.FlowData):
    """openflo.fcs_read.FlowData for a file whose TEXT lacks a required $PnR: each
    missing range is taken as 2^$PnB (the most an n-bit value can hold), as
    FlowJo does, and a sentence saying so is appended to `filled`."""

    def __init__(self, path, filled, **kw):
        self._openflo_filled = filled
        super().__init__(path, **kw)

    def _FlowData__parse_text(self, *args, **kwargs):   # flowio's private
        text = super()._FlowData__parse_text(*args, **kwargs)  # type: ignore[attr-defined]
        try:
            npar = int(str(text.get('par', '')).strip())
        except ValueError:
            return text
        for i in range(1, npar + 1):
            if f'p{i}r' in text:
                continue
            try:
                bits = int(str(text.get(f'p{i}b', '')).strip())
            except ValueError:
                continue
            if not 0 < bits <= 64:
                continue
            text[f'p{i}r'] = str(2 ** bits)
            name = text.get(f'p{i}n') or f'Ch{i}'
            self._openflo_filled.append(
                f"$P{i}R (range of {name}) is missing: taken as 2^{bits} "
                f"= {2 ** bits:,} from its $P{i}B")
        return text


class FlowSample:
    """
    Single FCS file with full analysis pipeline.

    Typical workflow
    ----------------
        s = FlowSample('file.fcs')
        s.run_qc()
        s.auto_compensate()           # OR s.compensate_from_wsp('exp.wsp')
        s.apply_transform()
        s.apply_threshold_gates(thresholds)   # optional FMO gates
        s.cluster(k=30)
        s.run_umap()
        s.plot('CD901b', 'CD903', color_by='cluster')
        s.cluster_heatmap()
        s.export_csv()
    """

    def __init__(self, fcs_path):
        self.path            = fcs_path
        self.name            = os.path.splitext(os.path.basename(fcs_path))[0]
        self.metadata        = {}
        self.channel_names   = []
        self.channel_labels  = {}
        self.scatter_channels = []
        self.fluor_channels   = []
        self.thresholds       = {}
        # Compensation matrix actually applied to this sample (post-channel-
        # intersection). Populated by `_apply_comp`, read by the workspace
        # exporters so the .wsp round-trips the spillover.
        self.comp_matrix:   np.ndarray | None = None
        self.comp_channels: list[str] = []
        # Per channel, the transform_spec currently baked into self.data (no
        # entry = linear). Written by apply_transform / retransform /
        # mark_transformed and read by linear_values. Always REBOUND, never
        # mutated, so a copy.copy of a sample cannot change the original's.
        self.data_transforms: dict[str, dict] = {}
        # Per channel, the axis a FlowJo workspace declared for this sample
        # (wsp_stat_axes), which FlowJo's GeoMean is averaged on; empty for a
        # sample not imported from a workspace (gating.population_stats).
        self.stat_axes: dict[str, dict] = {}
        # Where comp_matrix came from ('$SPILL', 'wsp:<name>', 'manual'), or
        # None while uncompensated. Recorded in the editor's provenance log.
        self.compensation_source: str | None = None
        # The raw FCS behind a sample rebuilt from a processed CSV (whose
        # `path` is the CSV); None for a sample read from its FCS.
        self.source_fcs: str | None = None
        # Events run_qc removed from `data` (FlowJo would count them).
        self.qc_removed: int = 0
        # Whether run_qc has run on this sample: a rebuild (re-compensation,
        # session reload from the raw FCS) repeats the same choice, so the
        # event set under existing gates does not change.
        self.qc_applied: bool = False
        # Channels whose stored integers _load turned into intensities, as
        # {channel: what was done} ('log, 4 decades' / 'gain 2'), and each
        # such channel's linear $PnR equivalent ({channel: max value}).
        self.fcs_decoding: dict[str, str] = {}
        self.linear_ranges: dict[str, float] = {}
        # What the loader had to work around in the file (a duplicate
        # channel name, a wrong $TOT), for the user, not just the log.
        self.load_warnings: list[str] = []
        # data / raw are populated unconditionally by _load() below. We
        # initialise with empty DataFrames (rather than None) so static
        # analysers can see them as pd.DataFrame everywhere downstream
        # without needing per-call narrowing.
        self.data: pd.DataFrame = pd.DataFrame()
        self.raw:  pd.DataFrame = pd.DataFrame()
        self.clusters         = None
        # What cluster() last ran with (k, route, applied resolution).
        self.cluster_settings: dict | None = None
        self.umap_coords      = None
        self.trimap_coords    = None
        self.pacmap_coords    = None
        self.cell_cycle_result = None
        self.flowsom_result   = None

        self._load()
        self._classify_channels()
        self._print_summary()

    @classmethod
    def from_dataframe(cls, df, name='sample', labels=None, metadata=None,
                       path=''):
        """Build a FlowSample from an in-memory DataFrame instead of an FCS
        file — e.g. to re-open a pipeline ``*_processed.csv`` (which carries
        cluster / UMAP / flowsom columns) in the editor.

        `labels` is an optional ``{column: antibody label}`` map; columns
        without one use the column name. Derived/analysis columns (cluster,
        UMAP*, flowsom*, cell_cycle, Time) are auto-excluded from the marker
        lists by ``_classify_channels``."""
        import pandas as pd
        s = cls.__new__(cls)
        s.path            = path
        s.name            = name
        s.metadata        = dict(metadata or {})
        s.channel_names   = list(df.columns)
        labels = labels or {}
        s.channel_labels  = {c: (labels.get(c) or c) for c in s.channel_names}
        s.scatter_channels = []
        s.fluor_channels   = []
        s.thresholds       = {}
        s.comp_matrix      = None
        s.comp_channels    = []
        # The frame's values are taken as given (linear). A caller holding
        # already-transformed data says so with mark_transformed().
        s.data_transforms  = {}
        s.stat_axes        = {}
        s.compensation_source = None
        s.source_fcs       = None
        s.qc_removed       = 0
        s.qc_applied       = False
        s.fcs_decoding     = {}
        s.linear_ranges    = {}
        s.load_warnings    = []
        s.data = pd.DataFrame(df).reset_index(drop=True).copy()
        s.raw  = s.data.copy()
        s.clusters         = None
        s.cluster_settings = None
        s.umap_coords      = None
        s.trimap_coords    = None
        s.pacmap_coords    = None
        s.cell_cycle_result = None
        s.flowsom_result   = None
        s._classify_channels()
        return s

    # ── Load ──────────────────────────────────────────────────────────────────

    def _load(self):
        filled: list[str] = []
        quirks: list[str] = []
        try:
            try:
                fcs = read_fcs_tolerant(self.path, _fcs_read.FlowData, quirks)
            except KeyError as e:
                # A file without a required $PnR was refused outright, where
                # FlowJo reads it with the range its $PnB allows.
                if not re.fullmatch(r"p\d+r", str(e).strip("'\"").lower()):
                    raise
                quirks.clear()
                fcs = read_fcs_tolerant(
                    self.path, functools.partial(_RangeFillingFlowData,
                                                 filled=filled), quirks)
        except FileNotFoundError as e:
            raise FcsParseError(f"FCS file not found: {self.path}") from e
        except Exception as e:
            raise FcsParseError(
                f"could not read {os.path.basename(self.path)}: "
                f"{plain_fcs_problem(e)} ({type(e).__name__}: {e})") from e
        self.metadata = dict(fcs.text)
        # FlowIO ≥1.0 uses integer keys and lowercase field names (pnn/pns).
        # A parameter with no $PnN is not in flowio's channel map at all: it
        # used to escape as a bare KeyError ("Failed to load X: 3"); it is
        # named Ch<n>, as the fallback always meant.
        named, renamed = self._fcs_channel_names(fcs)
        for name, label in named:
            self.channel_names.append(name)
            self.channel_labels[name] = label if label else name
        if renamed:
            log.warning("  [%s] duplicate channel names renamed: %s",
                        self.name, ", ".join(f"{a} -> {b}" for a, b in renamed))
        if quirks:
            log.warning("  [%s] %s", self.name, '; '.join(quirks))
        self.load_warnings = quirks + filled + [
            f"duplicate channel name {a!r}: the second is {b!r}"
            for a, b in renamed]
        if filled:
            log.warning("  [%s] %s", self.name, '; '.join(filled))
        if not any(self.channel_labels[n] != n for n in self.channel_names):
            log.info("  (no PnS labels in FCS — use detector names for plotting)")
        # flowio types `events` as Optional but in practice it's always
        # populated after a successful FlowData() construction; assert so
        # static analysis follows.
        assert fcs.events is not None
        events   = np.reshape(np.asarray(fcs.events), (-1, fcs.channel_count))
        events   = self._decode_stored_values(fcs, events)
        try:
            tot = int(str(fcs.text.get('tot', '')).strip())
        except ValueError:
            tot = None
        if tot is not None and tot != len(events):
            # The events are what the data segment holds; the header's count
            # disagreeing says the file was written or copied wrongly.
            msg = (f"$TOT says {tot:,} events; the data holds "
                   f"{len(events):,}")
            log.warning("  [%s] %s", self.name, msg)
            self.load_warnings.append(msg)
        # Two copies were being made here that nothing needed, and at the real
        # workload -- 20-25 multimillion-event files -- they dominated both
        # load time and memory. Measured on 8 x 2M events x 18 channels:
        # 2.25s / +2304 MB before, 0.75s / +1152 MB after; projected to 25
        # files, 7.0s / 7.2 GB -> 2.3s / 3.6 GB. The memory halving matters
        # more than the seconds: 7 GB of resident frames is where a machine
        # starts paging, and paging is what makes loading *feel* slow.
        #
        # `copy=False`: flowio hands back an `array.array` supporting the
        # buffer protocol, so `np.asarray` is already a free view -- the
        # 175 ms/file went on pandas materialising its own copy of a buffer
        # nothing else references.
        #
        # `copy(deep=False)`: `raw` must stay the pristine detector values
        # (the .fcs export reads it), and `self.data` is written IN PLACE in
        # a dozen places -- compensation and transforms among them -- so
        # aliasing the two would corrupt `raw`. Under pandas >= 3 copy-on-write
        # a shallow copy is a distinct object that duplicates a block only
        # when written, which gives independence without paying for it up
        # front. Verified: `data[col] = ...` and `data.loc[...] = ...` both
        # leave `raw` untouched, and no code path writes through `.values`,
        # which would bypass CoW.
        self.raw  = pd.DataFrame(events, columns=pd.Index(self.channel_names),
                                 copy=False)
        self.data = self.raw.copy(deep=False)

    @staticmethod
    def _fcs_channel_names(fcs):
        """[(column name, $PnS label)] for each parameter of the flowio
        ``fcs``, in order, and the duplicate names renamed as [(old, new)]."""
        seen: dict[str, int] = {}
        renamed = []
        named = []
        for i in range(1, fcs.channel_count + 1):
            ch    = fcs.channels.get(i) or {}
            name  = str(ch.get('pnn') or ch.get('PnN') or f'Ch{i}')
            label = (ch.get('pns') or ch.get('PnS') or '').strip()
            # Two detectors with one name loaded as two same-named columns,
            # and the second could not be reached by name: the transform
            # then failed on both. Keep both, the second as 'FL1-A (2)'.
            if name in seen:
                seen[name] += 1
                new = f'{name} ({seen[name]})'
                while new in seen:
                    seen[name] += 1
                    new = f'{name} ({seen[name]})'
                renamed.append((name, new))
                name = new
            seen.setdefault(name, 1)
            named.append((name, label))
        return named, renamed

    def restore_fcs_header(self, fcs_path):
        """Put back what this sample's raw FCS says about it -- its TEXT
        keywords (`metadata`) and how its stored values were decoded
        (`fcs_decoding`, `linear_ranges`) -- reading the TEXT segment only.
        For a sample rebuilt from a processed CSV, which carries the events
        but none of these: without them it lost its instrument, date,
        voltages and $TIMESTEP on every reopen. Raises as FlowSample does
        when the file cannot be read."""
        quirks: list[str] = []
        fcs = read_fcs_tolerant(
            fcs_path, functools.partial(_fcs_read.FlowData, only_text=True),
            quirks)
        self.metadata = dict(fcs.text)
        names = [n for n, _ in self._fcs_channel_names(fcs)[0]]
        _plan, decoding, ranges = self._decoding_plan(fcs, names)
        self.fcs_decoding = decoding
        self.linear_ranges = ranges

    @staticmethod
    def _decoding_plan(fcs, channel_names):
        """What _decode_stored_values does to each channel of the flowio
        ``fcs`` (its columns named `channel_names`): the plan
        [(column index, decades, log0, range, gain)], and the
        {channel: what was done} and {channel: linear $PnR} it records."""
        datatype = str(fcs.text.get('datatype', '')).strip().upper()
        plan = []                      # (column, decades, log0, range, gain)
        decoding: dict[str, str] = {}
        ranges: dict[str, float] = {}
        for i in range(1, fcs.channel_count + 1):
            ch = fcs.channels.get(i) or {}
            name = channel_names[i - 1]
            if str(name).strip().lower() == 'time':
                continue
            decades, log0 = ch.get('pne') or (0.0, 0.0)
            gain = float(ch.get('png') or 1.0)
            rng = float(ch.get('pnr') or 0.0)
            if datatype != 'I':
                if gain not in (0.0, 1.0):
                    log.info("  %s: $PnG %g kept as acquisition metadata "
                             "(float data is stored as intensities)",
                             name, gain)
                continue
            if decades > 0 and rng > 0:
                log0 = float(log0 or 1.0)
                plan.append((i - 1, float(decades), log0, rng, 1.0))
                decoding[name] = f"log, {float(decades):g} decades"
                ranges[name] = log0 * 10.0 ** float(decades)
            elif gain not in (0.0, 1.0):
                plan.append((i - 1, 0.0, 0.0, rng, gain))
                decoding[name] = f"gain {gain:g}"
                if rng > 0:
                    ranges[name] = rng / gain
        return plan, decoding, ranges

    def _decode_stored_values(self, fcs, events):
        """`events` as intensities. An integer FCS stores a log-amplified
        channel ($PnE f1,f2 with f1 > 0) as channel numbers 0..$PnR-1, whose
        intensity is f2 * 10**(f1 * x / $PnR), and an amplified linear one
        ($PnG g) as g times the intensity (FCS 3.1, section 3.3.29 / 3.3.31).
        Both used to load unchanged, so gates, transforms and statistics ran
        on channel numbers: a 4-decade channel's 1023 read 1023, not 9,910.

        Only integer data is decoded: the standard requires $PnE 0,0 for
        float data, and a float file's $PnG describes acquisition, not how
        its values are stored, so it is left alone (and said so). Time is
        never rescaled. A file needing nothing keeps the zero-copy view."""
        plan, decoding, ranges = self._decoding_plan(fcs, self.channel_names)
        if not plan:
            return events
        self.fcs_decoding.update(decoding)
        self.linear_ranges.update(ranges)
        out = np.array(events, dtype=np.float64)
        for idx, decades, log0, rng, gain in plan:
            if decades > 0:
                out[:, idx] = log0 * 10.0 ** (decades * out[:, idx] / rng)
            else:
                out[:, idx] = out[:, idx] / gain
        log.warning("  [%s] stored values converted to intensities: %s",
                    self.name, ", ".join(f"{c} ({d})"
                                         for c, d in self.fcs_decoding.items()))
        return out

    def _classify_channels(self):
        for ch in self.channel_names:
            if _is_scatter(ch):
                self.scatter_channels.append(ch)
            elif not _is_excluded(ch) and not self._derived_column(ch):
                self.fluor_channels.append(ch)

    def _derived_column(self, ch):
        """True for a column a sample rebuilt from a CSV carries that is not
        a measurement: a flag or label (bool / text: '<ch>_pos',
        'sample_origin') or a MESF: calibration of a detector. A live sample
        never lists these as fluorescence (they are added after loading), but
        from_dataframe did, so a restored session clustered on them by
        default: a MESF: column (linear, ~1e4-1e5) beside logicle detectors
        (0-1) is all a kNN distance sees."""
        if str(ch).startswith(('MESF:', 'D:')):
            return True
        from .result_stamp import is_original_column
        if is_original_column(ch):
            return True        # a corrected column's kept original (fixes)
        data = getattr(self, 'data', None)
        col = data.get(ch) if isinstance(data, pd.DataFrame) else None
        if isinstance(col, pd.DataFrame):          # duplicate column names
            col = col.iloc[:, 0]
        # A name with no column has no dtype to judge: its name decides.
        return col is not None and (pd.api.types.is_bool_dtype(col)
                                    or not pd.api.types.is_numeric_dtype(col))

    def _print_summary(self):
        labels = [self.channel_labels[c] for c in self.fluor_channels]
        log.info(
            "[%s]  %s events  |  fluor channels: %s",
            self.name, f"{len(self.data):,}", labels)

    def set_labels(self, mapping):
        """
        Assign antibody names to detector channels after loading.
        mapping: {detector_name: antibody_label}
        e.g. {'FL1-A': 'CD901b', 'FL2-A': 'CD902', 'FL3-A': 'CD903'}
        After calling this, plot('CD901b', 'CD903') and axis labels both work.
        """
        applied = {}
        for det, label in mapping.items():
            if det in self.channel_labels:
                self.channel_labels[det] = label
                applied[det] = label
        if applied:
            log.info(f"  Labels: { {d: lbl for d, lbl in applied.items()} }")
        return self

    # ── QC ────────────────────────────────────────────────────────────────────

    def run_qc(self, n_bins=200, threshold=5, plot=False):
        """Acquisition QC: drop the events AcquisitionQC flags from `data`
        AND `raw`. `raw` used to keep them, so its rows no longer matched
        `data`'s, and the CLI's --unmix, which read `raw`, unmixed every
        event QC had removed (QC had no effect there)."""
        qc        = AcquisitionQC(self.data)
        clean_idx = qc.run(n_bins=n_bins, threshold=threshold)
        if len(clean_idx) != len(self.data) and len(self.raw) == len(self.data):
            keep = self.data.index.isin(clean_idx)
            self.raw = self.raw.loc[keep].reset_index(drop=True)
        # Recorded, not only logged: these events are gone before any gate,
        # and FlowJo, which has no such step, counts them (20,000 events ->
        # 19,614; a gate at 6,371 in FlowJo read 5,985 after QC).
        self.qc_removed = (getattr(self, 'qc_removed', 0)
                           + len(self.data) - len(clean_idx))
        self.qc_applied = True
        self.data = self.data.loc[clean_idx].reset_index(drop=True)
        if plot:
            qc.plot()
        return self

    # ── Debris + doublet filtering ────────────────────────────────────────────

    def _find_scatter_col(self, prefix, suffix='-A'):
        """Return the column matching e.g. FSC-A or FSC-H, case-insensitive,
        preferring exact -A / -H endings. None if not found."""
        prefix_u = prefix.upper()
        suffix_u = suffix.upper()
        for c in self.data.columns:
            cu = c.upper()
            if cu.startswith(prefix_u) and cu.endswith(suffix_u):
                return c
        for c in self.data.columns:
            if c.upper().startswith(prefix_u):
                return c
        return None

    def filter_debris(self, fsc_channel=None, min_fsc=None):
        """Drop events whose FSC-A is below `min_fsc`. No-op if `min_fsc`
        is None or the FSC-A column can't be located."""
        if min_fsc is None:
            return self
        if fsc_channel is None:
            fsc_channel = self._find_scatter_col('FSC', '-A')
        if not fsc_channel or fsc_channel not in self.data.columns:
            log.info("  [Debris] No FSC-A channel found — debris filter skipped.")
            return self
        before = len(self.data)
        keep   = self.data[fsc_channel] >= float(min_fsc)
        self.data = cast(pd.DataFrame, self.data[keep]).reset_index(drop=True)
        kept = len(self.data)
        pct  = (kept / before * 100.0) if before else 0.0
        log.info(
            "  [Debris] Kept %s / %s events (%.1f%%) — FSC-A >= %.0f",
            f"{kept:,}", f"{before:,}", pct, float(min_fsc))
        return self

    def filter_doublets(self, fsc_a_channel=None, fsc_h_channel=None,
                        tol=0.25):
        """Drop doublets via the FSC-A / FSC-H ratio. Singlets fall along
        FSC-A ≈ k·FSC-H; doublets push FSC-A high relative to FSC-H, so
        their ratio sits well outside the population median. We keep
        events whose ratio is within ±`tol` of the median ratio.

        `tol` defaults to 0.25 (a relatively wide window) because
        polyploid cells are intrinsically more variable in
        FSC-A / FSC-H than typical leukocytes. Tighten to 0.15 for
        diploid samples.
        """
        if tol is None or tol <= 0:
            return self
        if fsc_a_channel is None:
            fsc_a_channel = self._find_scatter_col('FSC', '-A')
        if fsc_h_channel is None:
            fsc_h_channel = self._find_scatter_col('FSC', '-H')
        if (not fsc_a_channel or fsc_a_channel not in self.data.columns
                or not fsc_h_channel or fsc_h_channel not in self.data.columns):
            log.warning(
                "  [Doublets] FSC-A or FSC-H channel missing — "
                "doublet filter skipped.")
            return self
        before  = len(self.data)
        fsc_a   = self.data[fsc_a_channel].astype(float)
        fsc_h   = self.data[fsc_h_channel].astype(float)
        ratio   = np.where(fsc_h > 0, fsc_a / fsc_h, np.nan)
        valid   = np.isfinite(ratio)
        if not valid.any():
            # No usable FSC-A/FSC-H ratio anywhere. The old code substituted a
            # median of 0.0, which made the acceptance window [0, 0] and
            # dropped EVERY event — a whole sample deleted in silence. Doublets
            # cannot be identified without the ratio, so remove nothing and say
            # so, matching the missing-channel branch above.
            log.warning(
                "  [Doublets] No usable FSC-A/FSC-H ratio (FSC-H is "
                "non-positive for every event) — doublet filter skipped, "
                "all %s events kept.", f"{before:,}")
            return self
        median  = float(np.nanmedian(ratio[valid]))
        lo, hi  = median * (1.0 - tol), median * (1.0 + tol)
        keep    = valid & (ratio >= lo) & (ratio <= hi)
        self.data = cast(pd.DataFrame, self.data[keep]).reset_index(drop=True)
        kept = len(self.data)
        pct  = (kept / before * 100.0) if before else 0.0
        log.info(
            "  [Doublets] Kept %s / %s events (%.1f%%) — "
            "FSC-A/FSC-H in [%.3f, %.3f] (median %.3f, ±%.0f%%)",
            f"{kept:,}", f"{before:,}", pct, lo, hi, median, tol * 100)
        return self

    # ── Compensation ──────────────────────────────────────────────────────────
    #
    # Compensation is a STATE of the sample, not an operation on whatever
    # `self.data` holds at the moment: every entry point below sets "the
    # matrix applied to this sample". A second call replaces the first --
    # its effect is undone before the new matrix is applied -- so it can
    # never compound. It used to: `_apply_comp` multiplied the current data
    # by inv(M) every time, so auto_compensate() followed by
    # compensate_from_wsp() (or the same call twice, or FMOGater.prepare()
    # run twice) compensated already-compensated values. Measured: a second
    # auto_compensate() moved a 0.30-spillover channel by ~15,000 units.
    #
    # Compensation is linear algebra on LINEAR values, so it is refused on a
    # channel that has already been transformed (logicle etc.) rather than
    # silently mixing transformed coordinates.

    def auto_compensate(self):
        """Apply the spillover matrix embedded in the FCS metadata ($SPILL /
        $SPILLOVER) -- what FlowJo calls the "Acquisition-defined" matrix.
        Replaces any matrix applied earlier."""
        with _collect_warnings() as said:
            spill = self._parse_spillover()
            if spill is not None:
                self._apply_comp(spill['matrix'], spill['channels'],
                                 source='$SPILL')
        # What went wrong with the file's matrix (malformed, singular,
        # channels it names that the data lacks, a percent matrix) used to
        # reach only the log, which the window never shows.
        warnings = getattr(self, 'load_warnings', None)
        if warnings is not None:
            warnings.extend(f"compensation: {m}" for m in said)
        if spill is None and not said:
            log.warning(f"  [!] No spillover in FCS metadata for {self.name}.")
        return self

    def compensate_from_wsp(self, wsp_path, matrix_name=None):
        """Apply compensation from a FlowJo .wsp file.

        With `matrix_name`, that named matrix. Without it, the matrix the
        workspace applies to THIS file, as FlowJo does: the spillover matrix
        inside the <Sample> whose DataSet is this FCS or, when that sample
        carries none, FlowJo's Acquisition-defined matrix -- this file's own
        $SPILL. The first matrix in the workspace is used only for a file the
        workspace does not list (an FMO or control analysed against it).

        This used to take the workspace's FIRST matrix for every file, so in
        a workspace whose samples carry different matrices all but one were
        compensated with someone else's, and a workspace with no matrix at
        all raised instead of using each file's $SPILL as FlowJo does."""
        reader = WspReader(wsp_path)
        if matrix_name is not None:
            m = reader.get_matrix(matrix_name)
            self._apply_comp(m['matrix'], m['channels'],
                             source=f'wsp:{matrix_name}')
            return self
        listed, m = reader.matrix_for_fcs(self.path)
        if listed:
            if m is None:
                log.info("  [%s] the workspace gives this sample no matrix of "
                         "its own -- FlowJo's Acquisition-defined ($SPILL) "
                         "applies.", self.name)
                return self.auto_compensate()
            self._apply_comp(m['matrix'], m['channels'],
                             source=f"wsp:{m.get('name', 'sample')}")
            return self
        m = reader.get_matrix()          # raises when the workspace has none
        distinct = reader.distinct_matrix_count()
        if distinct > 1:
            log.warning(
                "  [!] %s is not a sample of %s, which holds %d different "
                "matrices -- applying the first ('%s'). Pass matrix_name= to "
                "choose.", os.path.basename(str(self.path)),
                os.path.basename(str(wsp_path)), distinct,
                m.get('name', '?'))
        self._apply_comp(m['matrix'], m['channels'],
                         source=f"wsp:{m.get('name', 'first')}")
        return self

    def manual_compensate(self, matrix, channels):
        """Apply a user-supplied spillover `matrix` over `channels` (rows =
        source fluorochrome, columns = detector). Replaces any matrix applied
        earlier."""
        self._apply_comp(matrix, channels)
        return self

    def recompensate(self, matrix, channels, source='manual'):
        """Replace the compensation this sample's events carry with `matrix`
        (spillover, source -> destination) over `channels`, keeping every
        event and every other column. Returns the number of events that now
        have an unknown value they did not have before.

        For the channels of the old and the new matrix it undoes each
        channel's recorded display transform (``data_transforms``), puts the
        recorded spillover (``comp_matrix``) back to get the measured values,
        un-mixes those with the new matrix, and re-applies each channel's own
        transform. A fresh load of the FCS would give the same values but
        none of the computed columns, and the whole file where the sample
        was downsampled.

        A value with no linear equivalent (NaN: <= 0 on a 'log' channel)
        spoils only the outputs it feeds. Raises UnknownScaleError when a
        channel's display scale is unknown, ValueError when no channel of
        the matrix is in the data or a recorded comp channel is missing, and
        numpy.linalg.LinAlgError for a singular matrix, CompensationError
        when the matrix's shape does not match `channels`.

        The compensation record (``comp_matrix``, ``comp_channels``,
        ``compensation_source``) is the one _apply_comp keeps: the old
        matrix is undone the same way (``compensated @ M = measured``), here
        through each channel's transform, which _apply_comp refuses."""
        matrix   = np.asarray(matrix, dtype=float)
        channels = [str(c) for c in channels]
        n = len(channels)
        # The same checks as _apply_comp: a matrix that does not fit its
        # channel list is refused, and a percent matrix becomes fractions.
        if matrix.ndim != 2 or matrix.shape != (n, n):
            raise CompensationError(
                f"spillover matrix is {matrix.shape} but names {n} "
                f"channel(s): {channels}")
        matrix = _spill_units_to_fractions(
            matrix, f"{self.name}: the {source} spillover matrix")
        idx = [i for i, c in enumerate(channels) if c in self.data.columns]
        avail = [channels[i] for i in idx]
        if not avail:
            raise ValueError("none of the matrix's channels are in the data")
        if len(avail) < n:
            log.warning(
                "  [!] %s: %d of %d spillover channel(s) are not in the data "
                "(%s) -- compensating the other %d only.", self.name,
                n - len(avail), n,
                ', '.join(c for c in channels if c not in avail), len(avail))
        sub = matrix[np.ix_(idx, idx)]
        inv = np.linalg.inv(sub)
        old_ch = list(self.comp_channels or [])
        missing = [c for c in old_ch if c not in self.data.columns]
        if missing:
            raise ValueError(
                f"the compensation recorded for {missing} cannot be undone: "
                "the channel is no longer in the data")
        old_m = (None if self.comp_matrix is None
                 else np.asarray(self.comp_matrix, dtype=float))
        if old_ch and (old_m is None or old_m.shape != (len(old_ch),) * 2):
            raise ValueError("the compensation recorded for this sample has "
                             "no matching matrix, so it cannot be undone")
        involved = list(dict.fromkeys([*old_ch, *avail]))
        pos = {c: j for j, c in enumerate(involved)}
        lin = np.column_stack([linear_values(self, c) for c in involved])
        measured = lin.copy()
        if old_ch:
            j = [pos[c] for c in old_ch]
            measured[:, j] = _matmul_known(lin[:, j], old_m)
        comp = measured.copy()
        j = [pos[c] for c in avail]
        comp[:, j] = _matmul_known(measured[:, j], inv)
        before = np.isfinite(self.data[involved].to_numpy(dtype=float))
        recs = self.data_transforms or {}
        data = self.data.copy(deep=False)
        for c in involved:
            spec = recs.get(c)
            v = comp[:, pos[c]]
            data[c] = (transform_values(v, **spec)
                       if spec and spec.get('method', 'linear') != 'linear'
                       else v)
        after = np.isfinite(data[involved].to_numpy(dtype=float))
        # A new frame, not an in-place write: caches keyed by the data's
        # identity (auto-clean masks) must see that the values changed.
        self.data = data
        self.comp_matrix = sub.copy()
        self.comp_channels = list(avail)
        self.compensation_source = source
        log.info(f"  Re-compensated ({source}): {avail}")
        return int((before & ~after).any(axis=1).sum())

    def is_compensated(self):
        """True when a spillover matrix is currently applied to `data`."""
        return (getattr(self, 'comp_matrix', None) is not None
                and bool(getattr(self, 'comp_channels', None)))

    def _transformed_channels(self):
        """{channel: transform_spec} for channels whose stored values are not
        linear (``data_transforms``; unknown-scale entries included)."""
        return {ch: spec for ch, spec in
                (getattr(self, 'data_transforms', None) or {}).items()
                if spec and spec.get('method', 'linear') != 'linear'}

    def _apply_comp(self, matrix, channels, source='manual'):
        matrix   = np.asarray(matrix, dtype=float)
        channels = [str(c) for c in channels]
        n = len(channels)
        if matrix.ndim != 2 or matrix.shape != (n, n):
            raise CompensationError(
                f"spillover matrix is {matrix.shape} but names {n} "
                f"channel(s): {channels}")
        # Every route in (a .wsp sample's matrix, manual_compensate, a staged
        # session matrix) gets the readers' unit check: a percent matrix
        # applied as fractions leaves every channel at 0.01x.
        matrix = _spill_units_to_fractions(
            matrix, f"{self.name}: the {source} spillover matrix")
        idx   = [i for i, c in enumerate(channels) if c in self.data.columns]
        avail = [channels[i] for i in idx]
        if not avail:
            log.warning("  [!] No matching channels for compensation.")
            return
        missing = [c for c in channels if c not in self.data.columns]
        if missing:
            # Spillover FROM a missing detector cannot be removed, so the
            # remaining channels are only partly corrected. Not fatal (a
            # matrix written for a larger panel is common), but never silent.
            log.warning(
                "  [!] %s: %d of %d spillover channel(s) are not in the data "
                "(%s) -- compensating the other %d only.", self.name,
                len(missing), n, ', '.join(missing), len(avail))
        prev_chans = list(getattr(self, 'comp_channels', None) or [])
        done = self._transformed_channels()
        on_curve = sorted(c for c in set(avail) | set(prev_chans) if c in done)
        if on_curve:
            raise CompensationError(
                f"{self.name}: cannot compensate {on_curve} -- already "
                "transformed. Compensation is linear and must be applied "
                "before apply_transform().")
        sub = matrix[np.ix_(idx, idx)]
        from . import gpu_accel
        try:
            # Loader threads run this concurrently; see gpu_accel.cpu_blas_lock.
            with gpu_accel.cpu_blas_lock():
                inv = np.linalg.inv(sub)
        except np.linalg.LinAlgError:
            log.warning("  [!] Spillover matrix is singular — "
                        "compensation skipped%s.",
                        ' (the matrix applied earlier is kept)'
                        if self.is_compensated() else '')
            return
        if self.is_compensated():
            # Undo the matrix applied earlier (compensated @ M = measured)
            # so the new one starts from the measured values. Exact float64
            # on purpose: this restores data, it is not the hot load path.
            prev = [c for c in prev_chans if c in self.data.columns]
            if prev != prev_chans:
                raise CompensationError(
                    f"{self.name}: the previously compensated channel(s) "
                    f"{sorted(set(prev_chans) - set(prev))} are gone from the "
                    "data, so the earlier compensation cannot be undone.")
            prev_m = np.asarray(self.comp_matrix, dtype=float)
            with gpu_accel.cpu_blas_lock():
                undone = np.asarray(self.data[prev].values, dtype=float) @ prev_m
            self.data[prev] = undone
            log.info("  [%s] replacing the compensation applied earlier (%s).",
                     self.name, getattr(self, 'compensation_source', None)
                     or 'manual')
        # Spillover is source->dest (measured = true @ M), so un-mixing is
        # `data @ inv(M)` — NO transpose. (gpu_accel.compensate computes
        # `values @ arg`.) The historical `inv.T` here silently left asymmetric
        # spillover uncorrected AND corrupted clean channels — i.e. every real
        # compensated dataset. See test_compensation_recovers_true_signal.
        self.data[avail] = gpu_accel.compensate(self.data[avail].values, inv)
        # Persist the matrix that was ACTUALLY applied (post-channel-
        # intersection, so it matches the channels in self.data) so the
        # GUI / CLI workspace export can round-trip it.
        self.comp_matrix   = np.asarray(sub, dtype=float).copy()
        self.comp_channels = list(avail)
        self.compensation_source = source
        log.info(f"  Compensation applied ({source}): {avail}")

    def _parse_spillover(self):
        """The file's SPILL / $SPILLOVER matrix as ``{'channels', 'matrix'}``,
        or None. The same keyword choice and parser as
        read_compensation_matrix (channels written as parameter numbers are
        mapped to $PnN; a percent matrix becomes fractions), so
        auto_compensate applies exactly what the compensation editor and the
        workspace export read from this file."""
        spill = _spill_text(self.metadata)
        if not spill:
            return None
        try:
            chans, matrix = _parse_spill_keyword(
                spill, _fcs_param_names(self.metadata, self.channel_names),
                self.name, markers=_fcs_param_markers(self.metadata))
        except CompensationError as e:
            log.warning("  [!] Malformed $SPILL ignored (not treated as 'no "
                        "spillover'): %s", e)
            return None
        return dict(channels=chans, matrix=matrix)

    # ── Transform ─────────────────────────────────────────────────────────────

    def apply_transform(self, channels=None, method='logicle',
                        t=262144, m=4.5, w=0.5, a=0, cofactor=150.0):
        if channels is None:
            channels = self.fluor_channels
        avail = [c for c in channels if c in self.data.columns]
        failed = []
        spec = transform_spec(method, t=t, m=m, w=w, a=a, cofactor=cofactor)
        done = dict(getattr(self, 'data_transforms', None) or {})
        # This transforms the values that are there. On a channel that already
        # carries a transform it would stack a second one on the first and
        # record only the second (measured: linear_values 0.446 where 924 is
        # true); 'linear' would drop the record and leave the data as it is.
        # Refuse before touching anything -- retransform undoes first.
        stacked = [str(c) for c in avail
                   if (done.get(c) or {}).get('method', 'linear') != 'linear']
        if stacked:
            raise ValueError(
                f"apply_transform: {', '.join(stacked)} already carr"
                f"{'ies' if len(stacked) == 1 else 'y'} a transform; use "
                "retransform() to change a channel's method.")
        for ch in avail:
            vals = np.asarray(self.data[ch].values, dtype=float).copy()
            try:
                self.data[ch] = transform_values(vals, **spec)
            except Exception as e:
                log.warning(f"  [!] Transform failed {ch}: {e}")
                failed.append(str(ch))
                continue
            if method == 'linear':
                done.pop(ch, None)
            else:
                done[ch] = spec
        self.data_transforms = done
        # Record the failed channels every call (empty when all succeeded, so a
        # later successful re-transform clears a prior failure) for any consumer
        # that wants to flag them.
        self._transform_failed = failed
        if failed:
            # A channel left in RAW scale among logicle-transformed siblings is a
            # silent trap: gates/plots authored in transformed space compare
            # against raw values and select the wrong events. Surface it loudly.
            log.error(
                "  [!] %d channel(s) left in RAW scale (transform failed): %s — "
                "gates/plots on them will be WRONG until re-transformed.",
                len(failed), ', '.join(failed))
        log.info(f"  {method} transform applied to {len(avail) - len(failed)} "
                 f"channel(s).")
        return self

    def retransform(self, channels, method='logicle', t=262144, m=4.5, w=0.5,
                    a=0, cofactor=150.0):
        """Put each of `channels` on `method`, first undoing the transform its
        data already carries (``data_transforms``). apply_transform instead
        transforms whatever values are there, so calling it on a logicle
        channel would stack a second transform on the first. A channel whose
        transform is unknown cannot be undone and is left as stored."""
        spec = transform_spec(method, t=t, m=m, w=w, a=a, cofactor=cofactor)
        done = dict(getattr(self, 'data_transforms', None) or {})
        for ch in channels:
            if ch not in self.data.columns:
                continue
            if is_unknown_spec(done.get(ch)):
                log.warning(f"  [!] {ch}: display scale unknown, left as "
                            "stored (not re-transformed).")
                continue
            try:
                lin = to_linear(self.data[ch].values, done.get(ch))
                self.data[ch] = transform_values(lin, **spec)
            except Exception as e:
                log.warning(f"  [!] Re-transform failed {ch}: {e}")
                continue
            if is_guessed_spec(done.get(ch)):
                # Re-transformed from guessed linear values: still a guess.
                done[ch] = guessed_spec(spec)
            elif method == 'linear':
                done.pop(ch, None)
            else:
                done[ch] = spec
        self.data_transforms = done
        return self

    def mark_transformed(self, channels=None, method='logicle', t=262144,
                         m=4.5, w=0.5, a=0, cofactor=150.0):
        """Record that `channels` (default: the fluor channels) ALREADY hold
        `method`-transformed values, without touching them -- for data that
        was transformed before it reached this sample, such as a processed CSV
        exported from transformed data."""
        if channels is None:
            channels = self.fluor_channels
        spec = transform_spec(method, t=t, m=m, w=w, a=a, cofactor=cofactor)
        done = dict(getattr(self, 'data_transforms', None) or {})
        for ch in channels:
            if ch not in self.data.columns:
                continue
            if method == 'linear':
                done.pop(ch, None)
            else:
                done[ch] = spec
        self.data_transforms = done
        return self

    # ── FMO gating ────────────────────────────────────────────────────────────

    def apply_threshold_gates(self, thresholds):
        """
        Add boolean '<channel>_pos' columns from FMO-derived thresholds.
        thresholds : {channel_name: cutoff_value}
        """
        self.thresholds = thresholds
        for ch, val in thresholds.items():
            col = self._resolve(ch)
            if col in self.data.columns:
                # At or above the cut, as a 'threshold' gate counts: a
                # --gates threshold lands here, and --export-wsp writes it
                # back to FlowJo as a min-only gate (min <= x).
                self.data[f'{col}_pos'] = self.data[col] >= val
                pct = self.data[f'{col}_pos'].mean() * 100
                log.info(f"  Gate {col} >= {val:.2f}  ->  {pct:.1f}% positive")
        return self

    def apply_region_gates(self, gates):
        """Filter `self.data` to events satisfying every (enabled) gate in
        `gates`, combined via logical AND.

        Memory-aware: we maintain one bool `keep` mask over the original
        rows and evaluate each gate ONLY against the still-active rows
        (`np.where(keep)`). The DataFrame is sliced exactly once at the
        end, so we never pay the cost of duplicating wide intermediate
        DataFrames. Polygon gates use float32 coords so a 10 M-event
        sample's pts array is 80 MB instead of 160 MB; subsequent
        polygons after earlier gates have narrowed the data are much
        smaller still.

        Disabled gates (gate['enabled'] is False) are skipped.

        Call AFTER `apply_transform()` so gate coordinates (typically
        logicle) match the data scale.
        """
        if not gates:
            return self
        active = [g for g in gates if g.get('enabled', True)]
        if not active:
            return self

        n_total = len(self.data)
        keep = np.ones(n_total, dtype=bool)
        per_gate = []                  # (label, kept_after_this_gate)
        # The run log names each gate by the intensities it cuts at, as the
        # editor's gate list does, not by its stored logicle coordinates.
        tf = dict(getattr(self, 'data_transforms', None) or {})
        # Boolean gates reference other gates by id, and we have the whole
        # list right here — without it they cannot resolve their operands and
        # used to admit every event, turning a "NOT X" population into the
        # entire sample. Ancestors are included (not just `active`) so a
        # disabled parent still defines the population, as cumulative_gate_mask
        # documents.
        gates_by_id = {g['id']: g for g in gates if g.get('id') is not None}

        for g in active:
            n_active = int(keep.sum())
            if n_active == 0:
                per_gate.append((describe_gate(g, transforms=tf), 0))
                continue
            active_idx = np.where(keep)[0]
            sub_mask = self._evaluate_gate_on(g, active_idx, gates_by_id)
            # Write the gate's verdict back into the full-length keep
            # mask: rows the gate excluded become False; rows it kept
            # stay True (they were already True).
            keep[active_idx] = sub_mask
            per_gate.append((describe_gate(g, transforms=tf), int(keep.sum())))

        kept = int(keep.sum())
        pct  = (kept / n_total * 100.0) if n_total else 0.0
        self.data = cast(pd.DataFrame, self.data[keep]).reset_index(drop=True)
        log.info(
            "  [Gates] Kept %s / %s events (%.1f%%) after %d region gate(s):",
            f"{kept:,}", f"{n_total:,}", pct, len(active))
        for lbl, n in per_gate:
            this_pct = (n / n_total * 100.0) if n_total else 0.0
            # ASCII-only — stdout may be cp1252 in library callers.
            log.info(f"    -> {n:>10,d} ({this_pct:5.1f}%)   {lbl}")
        return self

    def _evaluate_gate_on(self, gate, active_idx, gates_by_id=None):
        """Evaluate one gate against `self.data` restricted to the rows at
        `active_idx`. Returns a bool mask of length len(active_idx).

        Allocates ONLY the columns the gate actually touches (and only the
        active rows of them), so memory scales with how much earlier
        gates have already narrowed the candidate set, not with the
        original sample size.
        """
        kind = gate.get('kind')
        n_act = len(active_idx)

        def _col(ch, dtype):
            if ch not in self.data.columns:
                return None
            # Pull only the active rows of this one column, in compact dtype.
            return np.asarray(self.data[ch].values[active_idx], dtype=dtype)

        if kind == 'threshold':
            vals = _col(gate['channel'], np.float64)
            if vals is None:
                log.warning(
                    "  [gate] threshold: channel %r not in data — skipped",
                    gate['channel'])
                return np.ones(n_act, dtype=bool)
            # At or above, matching gate_to_mask -- see there for why.
            return vals >= float(gate['value'])

        if kind == 'interval':
            vals = _col(gate['channel'], np.float64)
            if vals is None:
                log.warning(
                    "  [gate] interval: channel %r not in data — skipped",
                    gate['channel'])
                return np.ones(n_act, dtype=bool)
            # Half-open, matching gate_to_mask -- see there for why.
            return (vals >= float(gate['lo'])) & (vals < float(gate['hi']))

        if kind == 'rect':
            xs = _col(gate['x_channel'], np.float64)
            ys = _col(gate['y_channel'], np.float64)
            if xs is None or ys is None:
                log.warning(
                    "  [gate] rect: channel(s) %r / %r missing — skipped",
                    gate.get('x_channel'), gate.get('y_channel'))
                return np.ones(n_act, dtype=bool)
            # Half-open, matching gate_to_mask -- see there for why.
            return ((xs >= float(gate['x0'])) & (xs < float(gate['x1'])) &
                    (ys >= float(gate['y0'])) & (ys < float(gate['y1'])))

        if kind == 'polygon':
            xs = _col(gate['x_channel'], np.float32)
            ys = _col(gate['y_channel'], np.float32)
            if xs is None or ys is None:
                log.warning(
                    "  [gate] polygon: channel(s) %r / %r missing — skipped",
                    gate.get('x_channel'), gate.get('y_channel'))
                return np.ones(n_act, dtype=bool)
            verts = np.asarray(gate['vertices'], dtype=np.float32)
            if verts.ndim != 2 or verts.shape[1] != 2 or len(verts) < 3:
                log.warning(
                    "  [gate] polygon: malformed vertices (shape=%s) — skipped",
                    verts.shape)
                return np.ones(n_act, dtype=bool)
            pts = np.column_stack([xs, ys])
            result = _points_in_polygon(verts, pts)
            del pts, xs, ys     # release the temps before the next gate
            return result

        # Kinds without a memory-lean path here (ellipsoid / cluster / category /
        # boolean / autoclean): delegate to the full gate_to_mask on just the
        # active rows, rather than silently passing everything through.
        sub = self.data.iloc[active_idx] if n_act else self.data.iloc[:0]
        try:
            return np.asarray(gate_to_mask(gate, sub, gates_by_id),
                              dtype=bool)
        except Exception as exc:                       # noqa: BLE001
            # A gate that ERRORS out must NOT silently pass every event through
            # (all-True) — that inflates the population with a superset and reads
            # as success. Fail CLOSED (admit nothing) so the failure is loud (the
            # population empties) rather than silently wrong. This differs from
            # the "channel missing → skip" cases above, which are a gate that
            # legitimately doesn't apply to this sample.
            log.warning("  [gate] kind %r via gate_to_mask FAILED (%s) — admitting "
                        "no events (fail-closed)", kind, exc)
            return np.zeros(n_act, dtype=bool)

    # ── Clustering ────────────────────────────────────────────────────────────

    def cluster(self, channels=None, k=30, n_jobs=1, max_events=None,
                use_gpu='auto', vram_admission_gb=1.0, random_state=42,
                reproducible=True, resolution=1.0):
        """Phenograph-style clustering.

        channels
            None (default): one column per marker, see
            ``default_cluster_channels``. ``run_leiden``, ``run_louvain``,
            ``run_flowsom`` and the embeddings use the same default.

        max_events
            If set and the sample exceeds it, cluster a random sub-sample and
            assign the remaining events to their nearest labelled neighbour
            via a KD-tree (caps peak memory regardless of sample size).

        use_gpu
            'auto' (default): use the GPU clustering path when RAPIDS is
            available *and* free VRAM ≥ `vram_admission_gb`; otherwise CPU.
            True : force GPU (fall back to CPU only on exception).
            False: never attempt GPU.

        vram_admission_gb
            Minimum free VRAM (GB) required to take the GPU branch.

        reproducible
            **On by default since 2.4.9.** PhenoGraph's Louvain community
            detection is NOT seed-reproducible: it shells out to the Blondel
            reference binaries, which seed with
            ``srand(time(NULL) + getpid())`` and use that RNG to shuffle the
            node traversal order. The PID half is assigned by the OS, so no
            injected clock can pin it -- 20 runs on identical input inside one
            second gave 20 distinct partitions. Measured on 50k events with
            overlapping populations: two runs of the default agreed only to
            ARI 0.76, while the seeded Leiden backend was bit-identical.
            An analysis that changes when you re-run it is hard to defend in a
            methods section, so the reproducible path is now the default.

            With True, clustering uses PhenoGraph's *seeded Leiden* backend on
            the CPU path — identical input + ``random_state`` → identical
            labels. It was also ~1.5x faster in that measurement (63s vs 95s).

            Two consequences to be aware of:

            * **Labels differ from a Louvain run.** Leiden is the corrected
              form of Louvain (it cannot emit internally disconnected
              communities), not a reimplementation of it: on ambiguous data the
              two agreed only to ARI 0.74, with 7 clusters against 9. Accuracy
              against planted ground truth was equivalent (0.4905 vs 0.4990).
              Set ``reproducible=False`` to reproduce prior Louvain output or
              to match published PhenoGraph results.
            * **The GPU path is skipped**, because cuGraph's Louvain is
              likewise unseeded. ``reproducible=False`` restores it.

        resolution
            Granularity of the community detection: lower merges clusters
            (fewer, larger), higher splits them (more, smaller). 1.0, the
            default, is the standard PhenoGraph result and passes exactly
            what earlier versions passed. PhenoGraph over-splits data with
            few markers: four well-separated populations in three channels
            (300 events each) came out as 8 clusters at 1.0 and as the 4
            populations at 0.5. Only the reproducible (seeded Leiden) route
            has a resolution: PhenoGraph's Louvain runs the Blondel binaries,
            which take none, so with ``reproducible=False`` (and on the GPU
            Louvain path) a value other than 1.0 is not applied and a warning
            says so. ``self.cluster_settings`` records the resolution that
            was applied (None when it was not).

        The GPU branch uses cuML NearestNeighbors + cuGraph Louvain on the
        kNN graph. Any GPU failure (OOM, CUDA error, missing kit) falls back
        to CPU Phenograph for the same call — the result is always written.
        """
        resolution = float(resolution)
        if not (np.isfinite(resolution) and resolution > 0):
            raise ValueError(f"resolution must be > 0, got {resolution}")
        if channels is None:
            channels = default_cluster_channels(self.fluor_channels)
        avail = [c for c in channels if c in self.data.columns]
        X     = self.data[avail].values.astype(float)
        mask  = np.all(np.isfinite(X), axis=1)
        Xc    = X[mask]
        n     = len(Xc)

        # Pick the events that will actually be clustered (full or subsample).
        if max_events and n > max_events:
            rng        = np.random.default_rng(random_state)
            sub_idx    = np.sort(rng.choice(n, max_events, replace=False))
            X_cluster  = Xc[sub_idx]
            subsampled = True
        else:
            sub_idx    = None
            X_cluster  = Xc
            subsampled = False

        # Decide which backend to attempt first. Reproducible mode forces the
        # CPU seeded-Leiden path: Louvain (CPU Blondel and GPU cuGraph alike) is
        # non-deterministic per process and cannot be pinned from outside.
        try_gpu = False
        if not reproducible and use_gpu is not False and GPU_CLUSTERING_AVAILABLE:
            free_vram = _vram_free_gb()
            if use_gpu is True or free_vram is None or free_vram >= vram_admission_gb:
                try_gpu = True
            else:
                log.info(
                    "  [VRAM admission] %.1f GB < %.1f GB — clustering on CPU",
                    free_vram, vram_admission_gb)

        size_label = (f"{max_events:,} (subsample of {n:,})"
                      if subsampled else f"{n:,}")

        sub_comm = None
        Q        = -1.0
        backend  = 'CPU'
        route    = 'louvain'      # the community detection that ran

        if try_gpu:
            try:
                log.info(f"  GPU cluster: {size_label} × {len(avail)}, k={k} …")
                sub_comm, Q = self._cluster_gpu(X_cluster, k)
                backend = 'GPU'
            except Exception as e:
                log.warning(
                    "  [!] GPU clustering failed (%s: %s) — CPU fallback",
                    type(e).__name__, e)
                sub_comm = None

        if sub_comm is None:
            log.info(f"  Phenograph: {size_label} × {len(avail)}, k={k} …")
            # Lazy import — drags in igraph + sklearn.community, ~1 s and
            # ~200 MB. Only relevant for callers that actually cluster.
            try:
                import phenograph
            except ImportError as e:
                raise ClusteringError(
                    "phenograph is required for CPU clustering "
                    "(pip install phenograph)") from e
            pg_kwargs: dict = dict(k=k, n_jobs=n_jobs)
            if reproducible:
                # PhenoGraph's default Louvain is non-deterministic; its Leiden
                # backend accepts an explicit seed. Feature-detect so an older
                # build degrades gracefully (a warning, not a crash).
                import inspect
                _pg = inspect.signature(phenograph.cluster).parameters
                if 'clustering_algo' in _pg and 'seed' in _pg:
                    pg_kwargs['clustering_algo'] = 'leiden'
                    pg_kwargs['seed'] = int(random_state)
                    route = 'leiden'
                    # Only a non-default value is passed, so 1.0 sends what
                    # earlier versions sent (phenograph's own default is 1).
                    if resolution != 1.0:
                        pg_kwargs['resolution_parameter'] = resolution
                else:
                    log.warning(
                        "  [!] reproducible clustering needs a phenograph build "
                        "with seeded Leiden; using (non-deterministic) Louvain.")
            try:
                with tempfile.TemporaryDirectory(
                        prefix='openflo-phenograph-',
                        ignore_cleanup_errors=True) as scratch, \
                        contextlib.chdir(scratch), \
                        _phenograph_sort_jobs(n_jobs):
                    sub_comm, _, Q = phenograph.cluster(X_cluster, **pg_kwargs)
            except Exception as e:
                raise ClusteringError(
                    f"Phenograph CPU clustering failed: {e}") from e
            backend = 'CPU'

        if route != 'leiden' and resolution != 1.0:
            log.warning(
                "  [!] resolution %.2f not applied: PhenoGraph's Louvain has "
                "no resolution; it needs reproducible (seeded Leiden) "
                "clustering.", resolution)
        self.cluster_settings = {
            'method': 'phenograph', 'k': int(k), 'route': route,
            'backend': backend, 'random_state': random_state,
            'resolution': resolution if route == 'leiden' else None}

        # Expand sub-sample labels back to all events (KD-tree assign for rest).
        if subsampled:
            communities          = np.full(n, -1, dtype=int)
            communities[sub_idx] = sub_comm
            good = sub_comm >= 0
            if good.sum() > 0:
                from sklearn.neighbors import NearestNeighbors
                nn_clf = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
                nn_clf.fit(X_cluster[good])
                ref_labels         = sub_comm[good]
                rest_mask          = np.ones(n, dtype=bool)
                rest_mask[sub_idx] = False
                n_rest             = int(rest_mask.sum())
                log.info(
                    "  Assigning %s remaining events to nearest cluster "
                    "(KD-tree, 1-NN) …", f"{n_rest:,}")
                nbr = tree_kneighbors(nn_clf, Xc[rest_mask], 1)
                communities[rest_mask] = ref_labels[nbr[:, 0]]
            else:
                log.warning(
                    "  [!] All sub-sample events were marked as noise — "
                    "the rest cannot be assigned.")
        else:
            communities = sub_comm

        labels       = np.full(len(self.data), -1, dtype=int)
        labels[mask] = communities
        self.data['cluster'] = labels
        self.clusters        = labels
        valid = communities[communities >= 0]
        log.info(
            "  → %d clusters, Q=%.3f [%s%s]",
            len(np.unique(valid)) if len(valid) else 0, Q, backend,
            f", resolution {resolution:g}" if route == 'leiden' else '')
        return self

    def _cluster_gpu(self, X, k):
        """GPU clustering: cuML kNN graph + cuGraph Louvain community detection.

        Returns (communities ndarray of length len(X), modularity float).
        Raises on any RAPIDS / VRAM failure — the caller is responsible
        for catching and falling back to CPU.
        """
        kit     = _GPU_CLUSTER_KIT
        # caller only invokes us when GPU_CLUSTERING_AVAILABLE is True,
        # which implies kit is non-None — assert so pyright can narrow.
        assert kit is not None
        cupy    = kit['cupy']
        cudf    = kit['cudf']
        cugraph = kit['cugraph']
        CuNN    = kit['cunn']

        n     = X.shape[0]
        X_gpu = cupy.asarray(X, dtype=cupy.float32)

        # k+1 neighbours so we can drop the self-link (column 0).
        nn = CuNN(n_neighbors=k + 1)
        nn.fit(X_gpu)
        _, indices = nn.kneighbors(X_gpu)

        if hasattr(indices, 'values'):          # cuDF DataFrame on some versions
            idx_gpu = cupy.asarray(indices.values)
        else:
            idx_gpu = cupy.asarray(indices)
        idx_gpu = idx_gpu[:, 1:]                 # drop self

        src = cupy.repeat(cupy.arange(n, dtype=cupy.int32), k)
        dst = idx_gpu.flatten().astype(cupy.int32)

        edges = cudf.DataFrame({'src': src, 'dst': dst})
        G = cugraph.Graph()
        G.from_cudf_edgelist(edges, source='src', destination='dst', renumber=False)

        parts, modularity = cugraph.louvain(G)
        parts_sorted = parts.sort_values('vertex')
        communities  = parts_sorted['partition'].to_numpy().astype(np.int64)
        return communities, float(modularity)

    # ── UMAP ──────────────────────────────────────────────────────────────────

    def _embedding_input(self, channels, sample_n, random_state):
        """Shared front-end for the dimensionality-reduction backends
        (UMAP / TriMap / PaCMAP).

        Returns ``(X, sub_mask, avail)`` where ``X`` is the full float
        matrix of the available channels, ``sub_mask`` is a boolean row
        selector (finite rows, optionally sub-sampled to ``sample_n``),
        and ``avail`` is the list of channels actually used. Pure: no
        embedding is computed and nothing is written back.
        """
        if channels is None:
            channels = default_cluster_channels(self.fluor_channels)
        avail = [c for c in channels if c in self.data.columns]
        X     = self.data[avail].values.astype(float)
        mask  = np.all(np.isfinite(X), axis=1)

        if sample_n and mask.sum() > sample_n:
            idx      = np.where(mask)[0]
            chosen   = np.random.default_rng(random_state).choice(
                           idx, sample_n, replace=False)
            sub_mask = np.zeros(len(X), dtype=bool)
            sub_mask[chosen] = True
        else:
            sub_mask = mask
        return X, sub_mask, avail

    def _store_embedding(self, emb, sub_mask, prefix):
        """Write a 2-D embedding back as ``{prefix}1`` / ``{prefix}2``
        columns, NaN where the row wasn't embedded. Returns the embedding."""
        emb = np.asarray(emb)
        self.data[f'{prefix}1'] = np.nan
        self.data[f'{prefix}2'] = np.nan
        self.data.loc[sub_mask, f'{prefix}1'] = emb[:, 0]
        self.data.loc[sub_mask, f'{prefix}2'] = emb[:, 1]
        return emb

    def run_umap(self, channels=None, n_neighbors=30, min_dist=0.3,
                 sample_n=100_000, random_state=42):
        X, sub_mask, avail = self._embedding_input(
            channels, sample_n, random_state)

        log.info(f"  UMAP: {sub_mask.sum():,} events × {len(avail)} channels …")
        emb = None

        if GPU_AVAILABLE:
            try:
                import cudf  # type: ignore[import-not-found]
                assert _CumlUMAP is not None  # gated above by GPU_AVAILABLE
                X_gpu = cudf.DataFrame(X[sub_mask].astype(np.float32))
                emb   = np.array(
                    _CumlUMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                              random_state=random_state).fit_transform(X_gpu)
                )
                log.info(f"  UMAP complete [GPU — {GPU_NAME}].")
            except Exception as gpu_err:
                log.warning(f"  [!] GPU UMAP failed ({type(gpu_err).__name__}: {gpu_err})")
                log.warning("  [!] Retrying on CPU …")
                emb = None

        if emb is None:
            try:
                import umap as umap_lib
            except ImportError:
                log.warning("  [!] pip install umap-learn")
                return self
            # umap-learn forces n_jobs=1 when random_state is set (because
            # its layout optimisation is a lock-free parallel SGD: threads
            # race on the shared embedding array, and thread interleaving is
            # not an RNG, so no seed can pin it). We want the determinism —
            # same seed → same embedding across runs / samples / restarts —
            # so we take the override and silence its per-call UserWarning.
            warnings.filterwarnings(
                'ignore',
                message='.*n_jobs value.*overridden.*',
                category=UserWarning)

            # Do NOT "recover" the cores by building the kNN yourself with
            # pynndescent(n_jobs=-1) and passing precomputed_knn. It was tried
            # and reverted. pynndescent's parallel NN-descent is deterministic
            # only for a FIXED numba thread count: with random_state=42 on
            # 5000x8, thread counts 1 / 4 / 24 give three DIFFERENT neighbour
            # graphs (each perfectly reproducible at its own count). That makes
            # a seeded embedding machine-dependent — a 24-core workstation, a
            # 4-core laptop and a CI runner would produce different coordinates
            # from identical data and seed, which is exactly the guarantee this
            # function exists to provide. Measured gain was ~10% (119.3s ->
            # 108.7s at 150k events); the price was cross-machine
            # reproducibility of published figures. Bad trade.
            # tests/test_umap_embedding.py::test_embedding_does_not_depend_on
            # _thread_count pins it.
            # Serialized only where numba's threading layer needs it (macOS):
            # see openflo/numba_threads.py.
            with numba_threads.parallel_kernels():
                emb = np.asarray(
                    umap_lib.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                                  random_state=random_state).fit_transform(
                                      X[sub_mask]))
            log.info("  UMAP complete [CPU].")

        self.umap_coords = self._store_embedding(emb, sub_mask, 'UMAP')
        return self

    def run_trimap(self, channels=None, sample_n=100_000, random_state=42,
                   **trimap_kwargs):
        """TriMap embedding — a triplet-constraint alternative to UMAP that
        tends to preserve global structure better. CPU-only (the ``trimap``
        package has no GPU path). Writes ``TRIMAP1`` / ``TRIMAP2``.

        Install with ``pip install openflo[embed]`` (or ``pip install
        trimap``). Extra keyword args pass through to ``trimap.TRIMAP``.
        """
        X, sub_mask, avail = self._embedding_input(
            channels, sample_n, random_state)
        try:
            import trimap  # type: ignore[import-not-found]
        except ImportError:
            log.warning("  [!] TriMap not installed — pip install openflo[embed]")
            return self
        log.info(f"  TriMap: {sub_mask.sum():,} events × {len(avail)} channels …")
        with numba_threads.parallel_kernels():
            emb = trimap.TRIMAP(**trimap_kwargs).fit_transform(X[sub_mask])
        self.trimap_coords = self._store_embedding(emb, sub_mask, 'TRIMAP')
        log.info("  TriMap complete [CPU].")
        return self

    def run_pacmap(self, channels=None, n_neighbors=None, sample_n=100_000,
                   random_state=42, **pacmap_kwargs):
        """PaCMAP embedding — another global-structure-preserving
        alternative to UMAP. CPU-only. Writes ``PACMAP1`` / ``PACMAP2``.

        Install with ``pip install openflo[embed]`` (or ``pip install
        pacmap``). Extra keyword args pass through to ``pacmap.PaCMAP``.
        """
        X, sub_mask, avail = self._embedding_input(
            channels, sample_n, random_state)
        try:
            import pacmap  # type: ignore[import-not-found]
        except ImportError:
            log.warning("  [!] PaCMAP not installed — pip install openflo[embed]")
            return self
        log.info(f"  PaCMAP: {sub_mask.sum():,} events × {len(avail)} channels …")
        # n_neighbors=None means "auto" to PaCMAP, but its signature types the
        # parameter as int. The ignore must sit on the ARGUMENT line, not the
        # call line, or pyright still reports it.
        with numba_threads.parallel_kernels():
            reducer = pacmap.PaCMAP(
                n_neighbors=n_neighbors,  # pyright: ignore[reportArgumentType]
                random_state=random_state, **pacmap_kwargs)
            emb = reducer.fit_transform(X[sub_mask])
        self.pacmap_coords = self._store_embedding(emb, sub_mask, 'PACMAP')
        log.info("  PaCMAP complete [CPU].")
        return self

    def run_tsne(self, channels=None, perplexity=30.0, sample_n=50_000,
                 random_state=42, **tsne_kwargs):
        """t-SNE embedding (scikit-learn, a core dependency). Writes
        ``TSNE1`` / ``TSNE2``. t-SNE is O(n log n) but still heavier than UMAP,
        so it subsamples to ``sample_n`` events; ``perplexity`` is clamped below
        the sample size. Extra kwargs pass through to ``sklearn.manifold.TSNE``.
        """
        from sklearn.manifold import TSNE
        X, sub_mask, avail = self._embedding_input(
            channels, sample_n, random_state)
        n = int(sub_mask.sum())
        if n < 5:
            log.warning("  [t-SNE] too few events — skipped.")
            return self
        perp = float(min(perplexity, max(5.0, (n - 1) / 3.0)))
        log.info(f"  t-SNE: {n:,} events × {len(avail)} channels "
                 f"(perplexity {perp:.0f}) …")
        emb = TSNE(n_components=2, perplexity=perp,
                   random_state=random_state, init='pca',
                   **tsne_kwargs).fit_transform(X[sub_mask])
        self.tsne_coords = self._store_embedding(np.asarray(emb), sub_mask,
                                                 'TSNE')
        log.info("  t-SNE complete [CPU].")
        return self

    def run_phate(self, channels=None, sample_n=50_000, random_state=42,
                  **phate_kwargs):
        """PHATE embedding — a diffusion-based method that preserves continuous
        / trajectory structure especially well (complements the trajectory
        tool). Writes ``PHATE1`` / ``PHATE2``. Optional dependency: install with
        ``pip install openflo[embed]`` (or ``pip install phate``). Extra kwargs
        pass through to ``phate.PHATE``."""
        X, sub_mask, avail = self._embedding_input(
            channels, sample_n, random_state)
        try:
            import phate  # type: ignore[import-not-found]
        except ImportError:
            log.warning("  [!] PHATE not installed — pip install openflo[embed]")
            return self
        log.info(f"  PHATE: {sub_mask.sum():,} events × {len(avail)} channels …")
        emb = phate.PHATE(random_state=random_state,
                          verbose=False, **phate_kwargs).fit_transform(
                              X[sub_mask])
        self.phate_coords = self._store_embedding(np.asarray(emb), sub_mask,
                                                  'PHATE')
        log.info("  PHATE complete [CPU].")
        return self

    # ── Plotting ──────────────────────────────────────────────────────────────

    def _resolve(self, name):
        if name in self.data.columns:
            return name
        for det, lbl in self.channel_labels.items():
            if lbl.lower() == name.lower():
                return det
        raise KeyError(f"Channel '{name}' not found.")

    def _intensity_ticker(self, channel):
        """``(locator, formatter)`` labelling an axis of `channel`'s stored
        values in intensity (scales.intensity_ticker, 'stored'), or None for
        a channel stored linear or on an unknown scale."""
        from .scales import intensity_ticker
        spec = (getattr(self, 'data_transforms', None) or {}).get(channel)
        return intensity_ticker(spec, 'stored') if spec else None

    def plot(self, x, y, color_by='density', sample_n=50_000,
             ax=None, title=None, s=1.0, alpha=0.5):
        """
        Scatter plot. x/y accept detector name, stain label, UMAP1, UMAP2.
        color_by: 'density' | 'cluster' | channel name/label
        """
        import matplotlib.pyplot as plt  # lazy: see module-top comment
        xch = self._resolve(x)
        ych = self._resolve(y)
        df  = self.data.dropna(subset=[xch, ych])
        if sample_n and len(df) > sample_n:
            df = df.sample(sample_n, random_state=42)
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))
        xv = np.asarray(df[xch].values)
        yv = np.asarray(df[ych].values)

        if color_by == 'cluster' and 'cluster' in df.columns:
            sc = ax.scatter(xv, yv, c=np.asarray(df['cluster'].values),
                            cmap='tab20', s=s, alpha=alpha, linewidths=0)
            plt.colorbar(sc, ax=ax, label='Cluster')
        elif color_by == 'density':
            # FlowJo-style O(n) histogram density: bin events into a
            # 256x256 grid, smooth the bin counts, look up each event's
            # density by its bin index. Replaces a gaussian_kde call
            # that was O(n^2) and took minutes on >100k events.
            try:
                from scipy.ndimage import gaussian_filter
                xv_f = np.asarray(xv, dtype=float)
                yv_f = np.asarray(yv, dtype=float)
                finite = np.isfinite(xv_f) & np.isfinite(yv_f)
                xv_f = xv_f[finite]; yv_f = yv_f[finite]
                if xv_f.size == 0:
                    raise ValueError("no finite points")
                BINS = 256
                hist, x_edges, y_edges = np.histogram2d(xv_f, yv_f, bins=BINS)
                hist = gaussian_filter(hist, sigma=1.5)
                ix = np.clip(np.searchsorted(x_edges, xv_f, side='right') - 1,
                             0, BINS - 1)
                iy = np.clip(np.searchsorted(y_edges, yv_f, side='right') - 1,
                             0, BINS - 1)
                z   = hist[ix, iy]
                idx = z.argsort()
                ax.scatter(xv_f[idx], yv_f[idx], c=z[idx],
                           cmap='turbo', s=s, alpha=alpha, linewidths=0,
                           rasterized=True)
            except Exception:
                ax.scatter(xv, yv, s=s, alpha=alpha, color='steelblue')
        else:
            try:
                cch = self._resolve(color_by)
                col = df[cch]
                # Categorical (string) column — e.g. 'sample_origin' on a
                # concatenated dataset. matplotlib can't take string values
                # for `c`; factorise to integer codes and draw a discrete
                # legend instead of a colorbar.
                if (col.dtype == object
                        or pd.api.types.is_string_dtype(col)
                        or isinstance(col.dtype, pd.CategoricalDtype)):
                    codes, uniques = pd.factorize(col, sort=True)
                    n_groups = max(1, len(uniques))
                    # 'auto' is colour-blind safe: Okabe-Ito's first n
                    # colours, then tab20, then evenly spaced hues (see
                    # theme.categorical_colors). tab20 alone alternates
                    # same-hue light/dark pairs, so its first two entries
                    # are both blues.
                    palette_name = getattr(self, '_palette_name', None) \
                                    or _DEFAULT_CATEGORICAL_PALETTE
                    cmap = categorical_cmap(palette_name, n_groups)
                    sc = ax.scatter(xv, yv, c=codes, cmap=cmap,
                                    vmin=0, vmax=max(n_groups - 1, 1),
                                    s=s, alpha=alpha, linewidths=0)
                    import matplotlib.patches as mpatches
                    handles = [
                        mpatches.Patch(color=cmap((i / max(n_groups - 1, 1))
                                                  if n_groups > 1 else 0.0),
                                       label=str(u))
                        for i, u in enumerate(uniques)
                    ]
                    ax.legend(handles=handles, title=color_by, loc='best',
                              fontsize=8, framealpha=0.8)
                else:
                    sc = ax.scatter(xv, yv, c=np.asarray(col.values),
                                    cmap='viridis',
                                    s=s, alpha=alpha, linewidths=0)
                    cbar = plt.colorbar(sc, ax=ax,
                                        label=self.channel_labels.get(cch, cch))
                    ticker = self._intensity_ticker(cch)
                    if ticker is not None:
                        cbar.locator, cbar.formatter = ticker
            except KeyError:
                ax.scatter(xv, yv, s=s, alpha=alpha, color='steelblue')

        ax.set_xlabel(self.channel_labels.get(xch, xch))
        ax.set_ylabel(self.channel_labels.get(ych, ych))
        # The values drawn are STORED: a logicle axis was ticked 0.2 .. 1.0
        # where the intensities are 100 .. 10^5. Tick each transformed axis
        # at round intensities, as the editor's axes are.
        for axis, ch in ((ax.xaxis, xch), (ax.yaxis, ych)):
            ticker = self._intensity_ticker(ch)
            if ticker is not None:
                axis.set_major_locator(ticker[0])
                axis.set_major_formatter(ticker[1])
        ax.set_title(title or self.name)
        plt.tight_layout()
        return ax

    def plot_umap(self, color_by='cluster', **kwargs):
        return self.plot('UMAP1', 'UMAP2', color_by=color_by, **kwargs)

    def cluster_heatmap(self, channels=None, label_col='cluster'):
        if label_col not in self.data.columns:
            log.warning("  [!] Run .cluster() first.")
            return
        import matplotlib.pyplot as plt  # lazy: see module-top comment
        if channels is None:
            channels = self.fluor_channels
        avail  = [c for c in channels if c in self.data.columns]
        labels = [self.channel_labels.get(c, c) for c in avail]
        # set_axis avoids the .rename(columns=Mapping) overload that pandas-stubs
        # can't resolve cleanly off a chained groupby().median() expression.
        med    = (self.data.groupby(label_col)[avail].median()
                           .set_axis(labels, axis=1))
        # Label each row by cluster name so the -1 noise/unclustered row is drawn
        # AND clearly named "Unclustered (noise)" rather than a bare -1.
        med.index = [cluster_label(c) for c in med.index]
        _, ax = plt.subplots(figsize=(max(6, len(avail)),
                                      max(4, len(med) * 0.4)))
        import seaborn as sns  # lazy: only when plotting
        # Colours are medians of the stored (display-transformed) values, as
        # cluster heatmaps conventionally are; the colour bar says so, since
        # its 0-1 ticks are not intensities (<name>_stats.csv has the linear
        # medians).
        sns.heatmap(med, cmap='vlag', center=0, ax=ax,
                    linewidths=0.3, linecolor='grey',
                    cbar_kws={'label': 'median, display scale'})
        ax.set_title(f'{self.name} — Cluster Median Expression')
        plt.tight_layout()
        return ax

    # ── Stats & export ────────────────────────────────────────────────────────

    def cluster_frequencies(self, label_col='cluster'):
        # ``label_col`` defaults to the canonical 'cluster' column but accepts any
        # per-event label column (e.g. 'leiden', 'flowsom_meta') so each has the
        # same noise-labelled frequency export rather than a bare -1.
        if label_col not in self.data.columns:
            return pd.DataFrame()
        total  = len(self.data)
        labels = self.data[label_col]
        counts = labels.value_counts().sort_index()
        # Medians of the compensated LINEAR values, as the Statistics window
        # reports them. The pipeline logicle-transforms fluor channels before
        # clustering, and the median was taken of those coordinates: two
        # clusters at linear ~800 and ~20,000 were written to <sample>_stats
        # .csv as 0.43 and 0.75. One channel is inverted at a time; a column
        # of unknown display scale has no linear median (NaN).
        chans = [c for c in self.fluor_channels if c in self.data.columns]
        cols = []
        for c in chans:
            try:
                lin = linear_values(self, c)
            except UnknownScaleError:
                lin = np.full(total, np.nan)
            cols.append(pd.Series(lin, index=self.data.index)
                        .groupby(labels).median().reindex(counts.index))
        meds = (pd.concat(cols, axis=1) if cols
                else pd.DataFrame(index=counts.index))
        meds.columns = [f'median_{self.channel_labels.get(c, c)}'
                        for c in chans]
        count_arr = np.asarray(counts.values)
        df = pd.DataFrame(dict(sample=self.name, cluster=counts.index,
                               count=count_arr,
                               pct_total=count_arr/total*100)).set_index('cluster')
        df = df.join(meds).reset_index()
        # Keep EVERY cluster (including the -1 noise/unclustered sentinel) but
        # give each an explicit 'population' name, so the noise bucket is
        # unmistakable in the exported CSV / Prism sheet rather than a bare -1 or
        # silently dropped. cluster()'s count log still excludes -1; the noise
        # row here simply carries the "Unclustered (noise)" label.
        df.insert(1, 'population',
                  np.array([cluster_label(c) for c in df['cluster']]))
        return df

    # ── FlowSOM ───────────────────────────────────────────────────────────────

    def run_flowsom(self, channels=None, grid=(10, 10),
                    n_metaclusters=FLOWSOM_DEFAULT_METACLUSTERS,
                    iters=10, max_events=50_000, seed=42):
        """FlowSOM clustering: train a SOM over the marker space, assign each
        event to a node, then agglomerate nodes into metaclusters. Writes a
        ``flowsom`` (node id) and ``flowsom_meta`` (metacluster id) column;
        non-finite rows get -1. Stores the model in ``self.flowsom_result``.

        `n_metaclusters` defaults to FLOWSOM_DEFAULT_METACLUSTERS (20), as in
        the Cluster dialog and openflo-run: set it above the number of
        populations you expect, since asked for exactly that many, FlowSOM
        tends to merge a rare population into a neighbour.

        Lighter than the R FlowSOM but the same structure — fast, CPU-only,
        good for very large files. Returns self."""
        channels = channels or default_cluster_channels(self.fluor_channels)
        avail = [c for c in channels if c in self.data.columns]
        if not avail:
            log.warning("  [FlowSOM] no usable channels — skipped.")
            return self
        X = self.data[avail].values.astype(float)
        mask = np.all(np.isfinite(X), axis=1)
        Xf = X[mask]
        if Xf.shape[0] < max(n_metaclusters, np.prod(grid)):
            log.warning("  [FlowSOM] too few events — skipped.")
            return self

        log.info("  FlowSOM: %s events × %d channels, grid %dx%d …",
                 f"{Xf.shape[0]:,}", len(avail), grid[0], grid[1])
        W, _coords = _som_train(Xf, grid=grid, iters=iters,
                                max_events=max_events, seed=seed)
        nodes = _som_assign(Xf, W)
        meta_of_node = _som_metacluster(W, n_metaclusters)
        meta = meta_of_node[nodes]

        node_col = np.full(len(self.data), -1, dtype=int)
        meta_col = np.full(len(self.data), -1, dtype=int)
        node_col[mask] = nodes
        meta_col[mask] = meta
        self.data['flowsom'] = node_col
        self.data['flowsom_meta'] = meta_col
        n_meta = len(np.unique(meta))
        # meta_of_node maps a node to its metacluster, so events this run did
        # not see can be labelled the way FlowSOM labels its own: the
        # metacluster of their best-matching node (_som_assign).
        self.flowsom_result = {
            'grid': grid, 'n_nodes': int(np.prod(grid)),
            'n_metaclusters': int(n_meta), 'channels': avail, 'weights': W,
            'meta_of_node': meta_of_node}
        log.info("  FlowSOM complete → %d nodes, %d metaclusters.",
                 int(np.prod(grid)), n_meta)
        return self

    def run_leiden(self, channels=None, n_neighbors=30, resolution=1.0,
                   max_events=200_000, random_state=42, fast_graph=False):
        """Leiden community detection — the current standard for high-dimensional
        spectral cytometry.

        Builds a symmetric k-nearest-neighbour graph over the marker space and
        partitions it with the Leiden algorithm (RBConfiguration objective;
        ``resolution`` tunes granularity — higher gives more, smaller clusters).
        Writes a ``leiden`` column (non-finite rows → -1). Like ``cluster``,
        very large samples are graph-partitioned on a random subsample of up to
        ``max_events`` events and the rest assigned to their nearest labelled
        neighbour (KD-tree). ``channels=None`` clusters one column per marker
        (``default_cluster_channels``). ``n_neighbors`` defaults to 30, as in
        the app, the CLI and ``cluster_jointly`` (15 up to 2.7.0). Requires
        ``igraph`` + ``leidenalg`` (declared dependencies). Returns self."""
        try:
            # igraph is an availability probe here: the graph itself is built
            # by _snn_jaccard_graph, but failing early gives a better message
            # than an ImportError from inside it.
            import igraph  # noqa: F401
            import leidenalg
        except ImportError as e:
            raise ClusteringError(
                "igraph + leidenalg are required for Leiden clustering "
                "(pip install igraph leidenalg)") from e
        from sklearn.neighbors import NearestNeighbors

        channels = channels or default_cluster_channels(self.fluor_channels)
        avail = [c for c in channels if c in self.data.columns]
        if not avail:
            log.warning("  [Leiden] no usable channels — skipped.")
            return self
        X = self.data[avail].values.astype(float)
        mask = np.all(np.isfinite(X), axis=1)
        Xc = X[mask]
        n = len(Xc)
        if n < 3:
            log.warning("  [Leiden] too few finite events — skipped.")
            return self

        if max_events and n > max_events:
            rng = np.random.default_rng(random_state)
            sub_idx = np.sort(rng.choice(n, max_events, replace=False))
            X_cluster = Xc[sub_idx]
            subsampled = True
        else:
            sub_idx = None
            X_cluster = Xc
            subsampled = False

        k = int(max(1, min(n_neighbors, len(X_cluster) - 1)))
        log.info("  Leiden: %s events × %d channels, k=%d, resolution=%.2f …",
                 f"{len(X_cluster):,}", len(avail), k, resolution)
        g = _snn_jaccard_graph(X_cluster, k, prune=fast_graph)
        part = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition, weights='weight',
            resolution_parameter=float(resolution), seed=int(random_state))
        sub_comm = np.asarray(part.membership, dtype=int)

        if subsampled:
            communities = np.full(n, -1, dtype=int)
            communities[sub_idx] = sub_comm
            nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(
                X_cluster)
            rest_mask = np.ones(n, dtype=bool)
            rest_mask[sub_idx] = False
            if rest_mask.any():
                nbr = tree_kneighbors(nn, Xc[rest_mask], 1)
                communities[rest_mask] = sub_comm[nbr[:, 0]]
        else:
            communities = sub_comm

        labels = np.full(len(self.data), -1, dtype=int)
        labels[mask] = communities
        self.data['leiden'] = labels
        n_clusters = len(np.unique(communities[communities >= 0]))
        log.info("  Leiden complete → %d clusters (resolution=%.2f).",
                 n_clusters, resolution)
        return self

    def run_louvain(self, channels=None, n_neighbors=30, restarts=5,
                    max_events=200_000, random_state=42, fast_graph=False):
        """Louvain community detection — deterministic, unlike PhenoGraph's.

        Same algorithm as ``cluster(reproducible=False)``, but it cannot give a
        different answer twice. PhenoGraph shells out to the Blondel reference
        binaries, which seed themselves ``srand(time(NULL) + getpid())``; the
        PID half is assigned by the OS, so that path is unpinnable from
        outside. igraph implements Louvain in-process with a settable RNG, so
        here the seed is ours.

        ``restarts`` reproduces what makes PhenoGraph's Louvain good rather
        than merely fast. A single Louvain run lands in whatever local optimum
        its node ordering leads to, so the reference implementation re-runs it
        and keeps the highest modularity. This does the same with a
        DETERMINISTIC seed sequence (``random_state``, ``random_state + 1``,
        …), so the exploration is preserved and the answer is still identical
        every time.

        One difference worth knowing. PhenoGraph's restart count is ADAPTIVE,
        not fixed: ``phenograph.core.runlouvain(max_runs=100,
        time_limit=2000, tol=1e-3)`` keeps going until modularity has not
        improved for 20 consecutive runs, then stops — capped at 100 runs or
        2000 seconds. So it spends restarts where they help and stops where
        they do not, while ``restarts`` here is a flat count paid in full
        every time. Matching that stopping rule would be a better design than
        any fixed number.

        Re-measured 2026-09-09 on 20,000 events x 10 channels, 8 overlapping
        populations (centres uniform in [3.5, 6.5], sizes Dirichlet(0.6),
        noise sd 1.4, seed 0), k=30, on a 24-core Windows box:

            restarts=1     24.1s   ARI 0.4880   4 clusters
            restarts=5    121.0s   ARI 0.4867   4 clusters   <- default
            restarts=20   496.5s   ARI 0.4881   4 clusters
            restarts=40  1019.4s   ARI 0.4929   4 clusters

            PhenoGraph binary  41.6s   ARI 0.3784   11 clusters, NOT reproducible

        What holds: the result is identical across runs (verified), and it
        scores better than the binary against the planted truth (0.488 vs
        0.378).

        Read the sweep honestly: ARI barely moves across it. 1, 5 and 20 sit
        within 0.0014 of each other — restarts=1 actually edged restarts=5,
        which is noise, not a finding — and only 40 pulls clearly ahead, at
        forty times the cost of 1. So no restart count in this range is a
        measured optimum, and the sweep has NOT converged where the old
        default of 20 claimed it had.

        5 is therefore a compromise, chosen and labelled as one: it keeps some
        of the exploration that restarts exist for, at a quarter of what 20
        cost, on a method that is already about 12x SLOWER than the PhenoGraph
        binary it replaces (121s against 41.6s). Raise it if you want the
        extra 0.005 of ARI that 40 bought and can spend 17 minutes on 20k
        events; drop it to 1 for interactive work and lose almost nothing
        measurable.

        Modularity is deliberately not compared against the binary. Q is a
        property OF A GRAPH, and the two methods partition different graphs
        (see ``_snn_jaccard_graph``), so comparing their Q values says nothing.
        ARI against known truth is the comparable measure.

        Writes a ``louvain`` column (non-finite rows → -1). Very large samples
        are partitioned on a random subsample of up to ``max_events`` and the
        rest assigned to their nearest labelled neighbour, as ``run_leiden``
        does; ``channels`` and ``n_neighbors`` default as there (30, 15 up to
        2.7.0). Returns self.
        """
        try:
            import igraph  # noqa: F401  (availability check; igraph_seeded uses it)
        except ImportError as e:
            raise ClusteringError(
                "igraph is required for Louvain clustering "
                "(pip install igraph)") from e
        from sklearn.neighbors import NearestNeighbors

        channels = channels or default_cluster_channels(self.fluor_channels)
        avail = [c for c in channels if c in self.data.columns]
        if not avail:
            log.warning("  [Louvain] no usable channels — skipped.")
            return self
        X = self.data[avail].values.astype(float)
        mask = np.all(np.isfinite(X), axis=1)
        Xc = X[mask]
        n = len(Xc)
        if n < 3:
            log.warning("  [Louvain] too few finite events — skipped.")
            return self

        if max_events and n > max_events:
            rng = np.random.default_rng(random_state)
            sub_idx = np.sort(rng.choice(n, max_events, replace=False))
            X_cluster = Xc[sub_idx]
            subsampled = True
        else:
            sub_idx = None
            X_cluster = Xc
            subsampled = False

        k = int(max(1, min(n_neighbors, len(X_cluster) - 1)))
        n_restarts = int(max(1, restarts))
        log.info("  Louvain: %s events × %d channels, k=%d, %d restarts …",
                 f"{len(X_cluster):,}", len(avail), k, n_restarts)
        g = _snn_jaccard_graph(X_cluster, k, prune=fast_graph)

        # Deterministic restart sequence: seed s, s+1, … keep the best Q.
        best, best_q = None, -np.inf
        for step in range(n_restarts):
            # igraph_seeded hands igraph back the random module afterwards,
            # so nothing else in the process stays pinned to this seed.
            with igraph_seeded(int(random_state) + step):
                # community_multilevel returns a VertexClustering; the union
                # with list[...] in its signature only arises with
                # return_levels=True, which we never pass. cast so the
                # attribute access type-checks against the real return.
                part = cast(Any, g.community_multilevel(weights='weight'))
            if float(part.modularity) > best_q:
                best, best_q = np.asarray(part.membership, dtype=int), \
                    part.modularity
        sub_comm = best if best is not None else np.zeros(len(X_cluster), int)

        if subsampled:
            communities = np.full(n, -1, dtype=int)
            communities[sub_idx] = sub_comm
            nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(
                X_cluster)
            rest_mask = np.ones(n, dtype=bool)
            rest_mask[sub_idx] = False
            if rest_mask.any():
                nbr = tree_kneighbors(nn, Xc[rest_mask], 1)
                communities[rest_mask] = sub_comm[nbr[:, 0]]
        else:
            communities = sub_comm

        labels = np.full(len(self.data), -1, dtype=int)
        labels[mask] = communities
        self.data['louvain'] = labels
        n_clusters = len(np.unique(communities[communities >= 0]))
        log.info("  Louvain complete → %d clusters (Q=%.4f, %d restarts).",
                 n_clusters, best_q, n_restarts)
        return self

    # ── Cell cycle ──────────────────────────────────────────────────────────

    def _dna_width_partner(self, dna_channel):
        """The width (`-W`) partner of an area (`-A`) DNA channel, if the
        FCS carries one (used for doublet exclusion). None otherwise."""
        cu = dna_channel.upper()
        if not cu.endswith('-A'):
            return None
        stem = dna_channel[:-2]
        cols = {str(c).upper(): c for c in self.data.columns}
        for suf in ('-W', '-H'):          # case-insensitive: match -w/-h too
            hit = cols.get((stem + suf).upper())
            if hit is not None:
                return hit
        return None

    def cell_cycle(self, dna_channel=None, k=1.5, singlet_channel=None,
                   singlet_tol=0.25, method='boundaries'):
        """DNA-content cell-cycle analysis.

        Auto-detects a DNA-stain channel (PI / DAPI / FxCycle / 7-AAD /
        Hoechst / DRAQ5 / …) when `dna_channel` is None; a label is
        accepted too. Doublet exclusion matters here (a G1 doublet lands at
        the G2/M position), so when a `-W`/`-H` partner exists (or
        `singlet_channel` is given) we pre-gate singlets on the DNA-A vs
        width ratio before modelling.

        Writes a categorical ``cell_cycle`` column (G1 / S / G2M / sub-G1 /
        >G2M / NA) and stores the model in ``self.cell_cycle_result``.
        ``method`` is 'boundaries' (counts between intensity cut-offs) or
        'dean-jett-fox' (the fitted model's areas; see cell_cycle_model).
        Returns self."""
        found = self.cell_cycle_model(dna_channel=dna_channel, k=k,
                                      singlet_channel=singlet_channel,
                                      singlet_tol=singlet_tol, method=method)
        if found is None:
            self.cell_cycle_result = None
            return self
        phases, model = found
        self.data['cell_cycle'] = phases
        self.cell_cycle_result = model
        return self

    def cell_cycle_model(self, dna_channel=None, k=1.5, singlet_channel=None,
                         singlet_tol=0.25, method='boundaries'):
        """What `cell_cycle` writes, computed without writing it: ``(phases,
        model)``, or None when no DNA channel is found. Reads the sample
        only, so the editor runs it off the window thread.

        With ``method='dean-jett-fox'`` the phase percents are the areas of
        the fitted Dean-Jett-Fox model (cellcycle_djf), started from the
        boundaries model's peaks, and each cycling event takes the phase
        whose fitted curve is highest at its DNA content. The boundaries
        percents stay in ``model['boundaries']``. A failed fit keeps the
        boundaries result and says so (``model['method']``)."""
        if method not in ('boundaries', 'dean-jett-fox'):
            raise ValueError(f"Unknown cell-cycle method {method!r}: use "
                             "'boundaries' or 'dean-jett-fox'.")
        col = dna_channel
        if col is None:
            col = find_dna_channel(self)
        elif col not in self.data.columns:
            try:
                col = self._resolve(col)
            except KeyError:
                col = None
        if not col or col not in self.data.columns:
            log.warning("  [cell-cycle] no DNA channel found — skipped.")
            return None

        # DNA content is modelled on the LINEAR scale (G2 = 2 x G1, the
        # A/W singlet ratio). On the editor's logicle values G2/G1 measured
        # 1.07-1.11 for any G1 from 5,000 to 100,000, outside the 1.7-2.3
        # window analyze_dna searches, so every G2/M cell was called S.
        vals = linear_values(self, col)
        singlet = np.ones(len(self.data), dtype=bool)
        wcol = singlet_channel or self._dna_width_partner(col)
        if wcol and wcol in self.data.columns:
            w = linear_values(self, wcol)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(w > 0, vals / w, np.nan)
            good  = np.isfinite(ratio)
            if good.any():
                med = float(np.nanmedian(ratio[good]))
                lo, hi = med * (1 - singlet_tol), med * (1 + singlet_tol)
                singlet = good & (ratio >= lo) & (ratio <= hi)
            else:
                # Same trap as filter_doublets: a median of 0.0 gives the
                # window [0, 0], so NO event is a singlet and every cell is
                # scored 'NA' — the cell-cycle result silently becomes empty.
                # Without a usable width ratio there is no singlet
                # discrimination to apply, so keep every event and say so.
                log.warning(
                    "  [CellCycle] No usable %s/%s ratio — singlet "
                    "discrimination skipped; all events scored.", col, wcol)

        model  = analyze_dna(vals[singlet], k=k)
        phases = np.full(len(self.data), 'NA', dtype=object)
        phases[singlet] = assign_phase(vals[singlet], model)

        model = dict(model)
        model['channel'] = col
        model['width_channel'] = wcol
        model['n_singlet'] = int(singlet.sum())
        model['method'] = 'boundaries'
        if method == 'dean-jett-fox' and model.get('ok'):
            from .cellcycle_djf import fit_dean_jett_fox, phase_by_model
            sv = vals[singlet]
            fit = fit_dean_jett_fox(sv, model['g1_mean'], model.get('g2_mean'),
                                    model.get('g1_sd'), model.get('g2_sd'))
            model['boundaries'] = {p: model.get(p) for p in (
                'pct_g1', 'pct_s', 'pct_g2m', 'g1_mean', 'g2_mean')}
            if fit['ok']:
                phases[singlet] = phase_by_model(sv, fit, phases[singlet])
                model.update(
                    method='Dean-Jett-Fox', pct_g1=float(fit['pct_g1']),
                    pct_s=float(fit['pct_s']), pct_g2m=float(fit['pct_g2m']),
                    g1_mean=fit['g1_mean'], g2_mean=fit['g2_mean'],
                    g1_cv=fit['g1_cv'], g2_g1_ratio=fit['g2_g1_ratio'],
                    rcs=fit['rcs'],
                    djf_params=[float(x) for x in fit['params']])
            else:
                model['method'] = 'boundaries (Dean-Jett-Fox fit failed)'
        log.info(
            "  [cell-cycle] %s: G1 %.1f%% / S %.1f%% / G2M %.1f%% "
            "(singlets=%s/%s)",
            col, model.get('pct_g1', float('nan')),
            model.get('pct_s', float('nan')), model.get('pct_g2m', float('nan')),
            f"{model['n_singlet']:,}", f"{len(self.data):,}")
        return phases, model

    def export_csv(self, path=None):
        if path is None:
            path = f"{self.name}_processed.csv"
        discard_transforms_sidecar(path)    # never beside the new CSV
        self.data.to_csv(path, index=False)
        # Beside it, the transform each column carries, so re-opening the CSV
        # (File > Load CSV) inverts exactly that instead of guessing.
        write_transforms_sidecar(path, self.data_transforms, self.data.columns,
                                 len(self.data))
        log.info(f"  Exported → {path}")
        return self

    def export_stats(self, path=None):
        freq = self.cluster_frequencies()
        if path is None:
            path = f"{self.name}_cluster_stats.csv"
        freq.to_csv(path, index=False)
        log.info(f"  Stats → {path}")
        return freq


def write_fcs(path, df, channels=None, channel_labels=None, corrections=None):
    """Write a DataFrame of events to an FCS 3.1 file (via FlowIO).

    ``channels`` selects/orders the parameters written (default: every column).
    ``channel_labels`` (``{column: antibody}``) populates the per-parameter
    ``$PnS`` marker names, so the file re-opens in FlowJo / FCS Express with
    labels intact. Non-finite cells are zeroed (FCS stores finite floats).
    OPENFLO_VERSION (and OPENFLO_CORRECTIONS) record which OpenFlo wrote it.
    Returns the number of events written.

    Used to export a gated subpopulation as a standalone, re-importable FCS —
    pass the events you want and the channels to keep. For a population of a
    loaded sample, ``fcs_export.population_export_frame(sample, mask)`` gives
    them as an FCS should hold them (compensated, linear). Not
    ``sample.raw.iloc[mask]``: a mask is computed on ``sample.data``, whose
    rows the debris, doublet and region-gate filters drop without touching
    ``raw``, so the rows do not correspond."""
    channels = [str(c) for c in (channels if channels else list(df.columns))]
    mat = np.nan_to_num(df[channels].to_numpy(dtype=float),
                        nan=0.0, posinf=0.0, neginf=0.0)
    opt = ([str(channel_labels.get(c, '') or '') for c in channels]
           if channel_labels else None)
    from .export_stamp import fcs_keywords
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, fcs_event_block(mat), channels,
                          opt_channel_names=opt,
                          metadata_dict=fcs_keywords(corrections))
    return len(mat)


# ══════════════════════════════════════════════════════════════════════════════
# CONCATENATION
# ══════════════════════════════════════════════════════════════════════════════

def concatenate(samples, label_col='sample_origin'):
    """
    Merge multiple FlowSample.data DataFrames, tagging each row with
    the sample name. Returns a single DataFrame.
    """
    frames = []
    for s in samples:
        df = s.data.copy()
        df[label_col] = s.name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    log.info(f"Concatenated {len(samples)} samples → {len(out):,} events.")
    return out


# ── Cross-sample label alignment ───────────────────────────────────────────────
#
# The same antibody (e.g. CD901b) can sit on a different fluorophore /
# detector in different samples or on different days. Comparing or
# clustering ACROSS samples by detector name therefore mis-aligns
# phenotypes. These helpers align by the antibody LABEL instead, while
# compensation stays keyed on detectors (each sample compensates its own
# $SPILL upstream — labels never touch the comp math).

def _sample_fluor_labels(sample):
    """{label: detector} for one sample's fluor channels. The label is
    the antibody name (channel_labels), falling back to the detector
    name when no label is set. Later detectors win on a label clash
    (rare; logged by callers if it matters)."""
    labels = getattr(sample, 'channel_labels', {}) or {}
    out = {}
    for det in getattr(sample, 'fluor_channels', []) or []:
        lbl = labels.get(det, det) or det
        out[lbl] = det
    return out


def align_fluor_labels(samples):
    """Align a set of samples by antibody label, with the CLI's matcher.

    The pairing is :func:`openflo.cli.align_shared_channels`, the one the
    shared clustering uses, so the library and the CLI agree on which
    channels are one marker: labels compare without case, spacing,
    punctuation or a fluor word at either end, Greek letters as Latin; the
    -A/-H/-W suffix is part of the match; an unlabelled channel pairs by
    detector; two different labels on one detector are never pooled. See
    that function for the full rules.

    Parameters
    ----------
    samples : iterable of FlowSample-like
        Each needs ``.name``, ``.fluor_channels`` (detector names) and
        ``.channel_labels`` ({detector: antibody label}). ``.data`` may be
        None (every fluor channel then counts as present); otherwise only
        fluor channels that are columns of ``.data`` take part.

    Returns
    -------
    dict with:
      ``common``      the matched columns' names, one per column, in the
                      first sample's channel order (the matcher's display
                      name: the first sample's label, or the detector when
                      unlabelled)
      ``per_sample``  {sample_name: {name: detector}}: each matched column
                      under its common name, plus every unmatched channel
                      under its own label (its detector when unlabelled;
                      'label (detector)' where the bare label would clash
                      or is held, unmatched, by every sample)
      ``missing``     {name: [sample_names lacking it]} for the unmatched
                      channels
      ``all_labels``  every name above, in first-seen order over the
                      samples
    """
    samples = list(samples)
    names = [getattr(s, 'name', None) or f'sample{i}'
             for i, s in enumerate(samples)]
    if not samples:
        return {'common': [], 'per_sample': {}, 'missing': {},
                'all_labels': []}
    from .cli import align_shared_channels
    channels, columns, _notes, display = align_shared_channels(samples, names)
    common = [display[d] for d in channels]
    # Each sample's channels in its own order: (detector, matched name or
    # None, own label).
    rows = []
    for s, cols in zip(samples, columns, strict=True):
        matched = dict(zip(cols, common, strict=True))   # detector -> name
        labels = getattr(s, 'channel_labels', None) or {}
        data = getattr(s, 'data', None)
        rows.append([(det, matched.get(det), labels.get(det, det) or det)
                     for det in getattr(s, 'fluor_channels', None) or []
                     if data is None or det in data.columns])
    # An unmatched label every sample holds (CD3 on FL1-A in one, on FL1-H
    # in the other) is not one marker: name it with its detector there.
    held = {}
    for nm, row in zip(names, rows, strict=True):
        for _det, m, lbl in row:
            if m is None:
                held.setdefault(lbl, set()).add(nm)
    n = len(set(names))
    everywhere = {lbl for lbl, ns in held.items() if len(ns) == n}
    per_sample = {}
    order = []                       # first-seen name order
    have = {}                        # unmatched name -> names that have it
    for nm, row in zip(names, rows, strict=True):
        l2d = per_sample.setdefault(nm, {})
        for det, m, lbl in row:
            if m is None:
                if lbl in everywhere or lbl in l2d or lbl in common:
                    lbl = f'{lbl} ({det})'
                have.setdefault(lbl, set()).add(nm)
            else:
                lbl = m
            l2d[lbl] = det
            if lbl not in order:
                order.append(lbl)
    missing = {lbl: sorted(set(per_sample) - have[lbl])
               for lbl in order if lbl in have}
    return {'common': common, 'per_sample': per_sample,
            'missing': missing, 'all_labels': order}


def relabel_gate_for_sample(gate, label_to_detector):
    """Retarget a gate's channel fields to a specific sample's detectors
    by antibody label.

    A template gate carries both a detector channel (e.g. ``FL1-A``)
    and, when authored in a labelled editor, the antibody label it stood
    for (``x_label`` / ``y_label`` / ``label``). Applied to a sample
    where that marker sits on a DIFFERENT detector, we rewrite the
    channel to that sample's detector so one template ties phenotypes
    across panels. Compensation is unaffected — this only swaps which
    column the gate reads.

    label_to_detector : {antibody label: detector} for the target sample
        (e.g. from ``_sample_fluor_labels``).

    Returns a shallow copy with channel fields remapped where a stored
    label resolves in the target sample; fields without a label, or
    whose label isn't present in the sample, are left as-is (the gate
    then reads its original detector, or no-ops with a warning if that
    detector is also absent).
    """
    if not label_to_detector:
        return dict(gate)
    g = dict(gate)
    for chan_field, label_field in (('channel', 'label'),
                                    ('x_channel', 'x_label'),
                                    ('y_channel', 'y_label')):
        lbl = g.get(label_field)
        if lbl and lbl in label_to_detector:
            g[chan_field] = label_to_detector[lbl]
    return g


def common_fluor_warning(samples):
    """Human-readable warning when samples don't share a common fluor
    panel, or '' when every channel pairs across them. Channels pair as
    :func:`align_fluor_labels` pairs them (so 'CD3' and 'cd-3' are one
    marker); lists which unmatched labels are missing from which samples
    and notes that cross-sample analysis uses the matched (common) set."""
    info = align_fluor_labels(samples)
    if not info['missing']:
        return ''
    lines = [f"  • {lbl}: missing from {', '.join(names)}"
             for lbl, names in info['missing'].items()]
    common = ', '.join(info['common']) or '(none)'
    return ("Samples don't share a common fluor panel. Cross-sample "
            "analysis will use only the common labels:\n"
            f"  common: {common}\n"
            "Non-common labels:\n" + '\n'.join(lines))


def concatenate_by_label(samples, label_col='sample_origin'):
    """Like :func:`concatenate`, but first renames each sample's fluor
    columns to their antibody LABEL and keeps ONLY the channels that pair
    across every sample (:func:`align_fluor_labels`, the CLI's matcher) —
    so a marker on different fluors, or spelt 'CD3' in one sample and
    'cd-3' in another, lines up into one column named by the first
    sample's label. Scatter / non-fluor columns are dropped from the
    merged frame (clustering uses fluor labels). Compensation already
    happened per sample upstream on detectors, so this is purely a
    rename-and-intersect for cross-sample clustering.

    Returns (merged_df, common_labels). merged_df has the common label
    columns + `label_col`; common_labels is the ordered label list used.
    """
    info = align_fluor_labels(samples)
    common = info['common']
    frames = []
    for s in samples:
        l2d = info['per_sample'].get(getattr(s, 'name', ''), {})
        # detector for each common label in THIS sample
        cols = {lbl: l2d[lbl] for lbl in common if lbl in l2d}
        sub = s.data[list(cols.values())].copy()
        sub.columns = list(cols.keys())          # rename detector → label
        sub[label_col] = s.name
        frames.append(sub)
    if not frames:
        return pd.DataFrame(), common
    out = pd.concat(frames, ignore_index=True)
    log.info("Concatenated %d samples by label → %d events, %d common "
             "fluor label(s): %s", len(frames), len(out), len(common),
             ', '.join(common) or '(none)')
    return out, common


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT (multi-sample)
# ══════════════════════════════════════════════════════════════════════════════

class FlowExperiment:
    """
    Batch-process multiple FCS files.

    Usage
    -----
        exp = FlowExperiment('/path/to/fcs/')
        exp.exclude_pattern('unstained|bead|fmo')
        exp.run_all(k=30, umap=True)
        exp.compare_conditions(
            groupA=['sample_1','sample_2'], groupB=['sample_3','sample_4'],
            label_a='Group A',              label_b='Group B')
        exp.export_all('results/')
    """

    def __init__(self, source):
        self.samples = {}
        self.cluster_params = None       # set by run_all
        self.joint_clustering = None     # set by cluster_jointly
        for p in self._resolve(source):
            try:
                s = FlowSample(p)
                self.samples[s.name] = s
            except Exception as e:
                log.warning(f"  [!] {p}: {e}")
        log.info(f"\nLoaded {len(self.samples)} sample(s).")

    @staticmethod
    def _resolve(source):
        if isinstance(source, (list, tuple)):
            return list(source)
        if os.path.isdir(source):
            return sorted([os.path.join(source, f)
                           for f in os.listdir(source)
                           if f.lower().endswith('.fcs')])
        raise ValueError("source must be a directory or list of paths.")

    def exclude_pattern(self, pattern):
        rx  = re.compile(pattern, re.IGNORECASE)
        rem = [n for n in self.samples if rx.search(n)]
        for n in rem:
            del self.samples[n]
        log.info(f"Excluded {len(rem)}: {rem}")
        return self

    def keep_pattern(self, pattern):
        rx  = re.compile(pattern, re.IGNORECASE)
        rem = [n for n in self.samples if not rx.search(n)]
        for n in rem:
            del self.samples[n]
        log.info(f"Kept {len(self.samples)}, removed {len(rem)}.")
        return self

    def run_all(self, qc=False, compensate=True, wsp_path=None,
                transform=True, transform_method='logicle',
                cluster=True, k=30, umap=False, resolution=1.0):
        for name, s in self.samples.items():
            log.info(f"\n── {name}")
            if qc:           s.run_qc()
            if compensate:
                if wsp_path: s.compensate_from_wsp(wsp_path)
                else:        s.auto_compensate()
            if transform:    s.apply_transform(method=transform_method)
            if cluster:      s.cluster(k=k, resolution=resolution)
            if umap:         s.run_umap()
        if cluster:
            # What compare_conditions clusters the compared samples with.
            self.cluster_params = {'method': 'phenograph', 'k': k,
                                   'resolution': float(resolution)}
        return self

    def combined_frequencies(self, label_col='cluster', names=None):
        """Every sample's (or the named samples') frequencies per label of
        `label_col`, one table."""
        pick = (self.samples.values() if names is None
                else [self.samples[n] for n in names])
        frames = [s.cluster_frequencies(label_col) for s in pick
                  if label_col in s.data.columns]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def cluster_jointly(self, names=None, method=None, k=None, max_events=None,
                        seed=42, n_jobs=1, resolution=None,
                        n_metaclusters=FLOWSOM_DEFAULT_METACLUSTERS,
                        label_col='shared_cluster'):
        """Cluster the named samples (default: all) TOGETHER, once, and
        write each sample's label from that one clustering to its
        `label_col` column. Returns ``{name: labels}``.

        Each sample's own clustering (FlowSample.cluster, run_all) numbers
        its clusters arbitrarily, so one id names different populations in
        different samples and cannot be compared across them. This pools an
        equal share of every sample's events (at most `max_events` in all;
        by default as many as the largest sample holds, or the method's own
        cap), clusters the pool with `method` / `k` / `resolution` (default:
        what run_all used, else PhenoGraph k=30 at resolution 1.0), and gives every other event its
        nearest pooled event's label (FlowSOM: its SOM node's), exactly as
        openflo-run's group comparisons do (cli._shared_cluster_labels).
        Channels are matched across samples by antibody label.

        Records the clustering in ``self.joint_clustering`` (samples,
        label_col, settings), which compare_conditions reads."""
        from .cli import _cluster_settings, _shared_cluster_labels
        names = (list(self.samples) if names is None
                 else list(dict.fromkeys(names)))
        missing = [n for n in names if n not in self.samples]
        if missing:
            raise KeyError(f"not in this experiment: {missing}")
        if not names:
            raise ValueError("no samples to cluster")
        p = dict(getattr(self, 'cluster_params', None) or {})
        method = method or p.get('method', 'phenograph')
        k = int(k or p.get('k', 30))
        resolution = float(resolution if resolution is not None
                           else p.get('resolution', 1.0))
        task = _cluster_settings(k, n_jobs, max_events, 1.0, seed,
                                 {'method': method, 'resolution': resolution,
                                  'n_metaclusters': n_metaclusters})
        samples = [self.samples[n] for n in names]
        log.info("Clustering %d sample(s) together (%s, k=%d) so cluster "
                 "ids mean the same population in every sample.",
                 len(samples), method, k)
        labels, channels, _cols, _display = _shared_cluster_labels(
            samples, task, names=names)
        for s, lab in zip(samples, labels, strict=True):
            s.data[label_col] = lab
        self.joint_clustering = {'samples': tuple(names),
                                 'label_col': label_col, 'method': method,
                                 'k': k, 'resolution': resolution,
                                 'channels': list(channels)}
        return dict(zip(names, labels, strict=True))

    def compare_conditions(self, groupA, groupB,
                            label_a='A', label_b='B', plot=True,
                            **cluster_kw):
        """Mean % of events per cluster in each condition (+ sd, n).

        A cluster id is compared ACROSS samples here, so it must come from
        one clustering of all the compared samples. The ids of each
        sample's own clustering do not: PhenoGraph, Leiden and FlowSOM
        number clusters arbitrarily, and with CD4 at 80 % in A and 20 % in
        B, each sample's largest cluster was id 0, so the table read 80 % vs
        80 % for "cluster 0" (CD4 in A, CD8 in B). So unless
        ``self.joint_clustering`` (cluster_jointly, or openflo-run's shared
        clustering) already covers every compared sample, the compared
        samples are first clustered together with cluster_jointly
        (`cluster_kw`: method, k, max_events, seed, ...; passing any forces
        a new joint clustering), and the frequencies come from those
        labels. To compare several pairs on one set of ids, call
        ``cluster_jointly()`` once on all samples first."""
        names = [n for n in dict.fromkeys(list(groupA) + list(groupB))
                 if n in self.samples]
        jc = getattr(self, 'joint_clustering', None)
        if (jc and not cluster_kw and set(names) <= set(jc['samples'])
                and all(jc['label_col'] in self.samples[n].data.columns
                        for n in names)):
            label_col = jc['label_col']
        else:
            if not names:
                log.warning("  [!] None of the compared samples is in this "
                            "experiment.")
                return pd.DataFrame()
            self.cluster_jointly(names, **cluster_kw)
            label_col = cluster_kw.get('label_col', 'shared_cluster')
        freq = self.combined_frequencies(label_col, names)
        if freq.empty:
            log.warning("  [!] No cluster data — run clustering first.")
            return freq
        freq['condition'] = freq['sample'].apply(
            lambda n: label_a if n in groupA else (label_b if n in groupB else 'other'))
        freq = freq[freq['condition'] != 'other']
        # A sample where a cluster has no events contributes no ROW: the
        # frequencies come from value_counts(), which omits empty categories.
        # Averaging the rows directly therefore averaged only the samples the
        # cluster appeared in — a population found in 1 of 3 samples at 10%
        # was reported as the group's 10%, when the group mean is 3.3%. Absent
        # means 0%, not "not measured", so the table is squared off first and
        # every sample in the condition contributes a value.
        wide = freq.pivot_table(index=['condition', 'sample'],
                                columns='cluster', values='pct_total',
                                aggfunc='sum', fill_value=0.0)
        tall: pd.DataFrame
        if wide.empty:
            tall = pd.DataFrame(
                freq[['condition', 'sample', 'cluster', 'pct_total']])
        else:
            tall = pd.DataFrame(
                {'pct_total': wide.stack()}).reset_index()      # type: ignore[arg-type]
        summary = (tall.groupby(['condition','cluster'])['pct_total']
                       .agg(['mean','std','size']).reset_index()
                       .rename(columns={'mean':'mean_pct', 'std':'sd_pct',
                                        'size':'n_samples'}))
        # Name each cluster (incl. the -1 noise bucket) so the comparison table
        # never carries a bare -1 next to real populations.
        summary.insert(2, 'population',
                       np.array([cluster_label(c) for c in summary['cluster']]))
        if plot:
            import matplotlib.pyplot as plt  # lazy: see module-top comment
            clusters = sorted(summary['cluster'].unique())
            a_v = (summary[summary.condition==label_a]
                          .set_index('cluster')['mean_pct']
                          .reindex(clusters, fill_value=0))
            b_v = (summary[summary.condition==label_b]
                          .set_index('cluster')['mean_pct']
                          .reindex(clusters, fill_value=0))
            x, w = np.arange(len(clusters)), 0.35
            fig, ax = plt.subplots(figsize=(max(8, len(clusters)*0.5), 5))
            ax.bar(x-w/2, a_v, w, label=label_a, color='steelblue', alpha=0.8)
            ax.bar(x+w/2, b_v, w, label=label_b, color='coral',     alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([cluster_label(c) for c in clusters], rotation=45,
                               ha='right')
            ax.set_ylabel('% of total events')
            ax.set_title(f'{label_a} vs {label_b} — cluster frequencies')
            ax.legend()
            plt.tight_layout()
        return summary

    def plot_all(self, x, y, color_by='cluster', ncols=3, sample_n=30_000):
        import matplotlib.pyplot as plt  # lazy: see module-top comment
        names  = list(self.samples.keys())
        nrows  = (len(names)+ncols-1)//ncols
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(6*ncols, 5*nrows))
        axes = np.array(axes).flatten()
        # i defaults to -1 so that when `names` is empty, the post-loop
        # range(i+1, len(axes)) hides every axis (otherwise i is unbound).
        i = -1
        for i, name in enumerate(names):
            self.samples[name].plot(x, y, color_by=color_by,
                                    sample_n=sample_n, ax=axes[i])
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        return fig

    def concatenate_group(self, names):
        return concatenate([self.samples[n] for n in names if n in self.samples])

    def export_all(self, out_dir='.'):
        os.makedirs(out_dir, exist_ok=True)
        for name, s in self.samples.items():
            s.export_csv(os.path.join(out_dir, f"{name}_processed.csv"))
            if 'cluster' in s.data.columns:
                s.export_stats(os.path.join(out_dir, f"{name}_stats.csv"))
        return self


# ══════════════════════════════════════════════════════════════════════════════
# LAZY IMPORTS  (PEP 562)
# ══════════════════════════════════════════════════════════════════════════════
#
# Old code wrote ``sns.heatmap(...)`` and ``gaussian_kde(...)`` after the
# module-level ``import seaborn as sns`` and ``from scipy.stats import
# gaussian_kde``. Those imports are now deferred — the hook below resolves
# them the first time a function inside this module touches the name.
#
# Phenograph isn't exposed this way; FlowSample.cluster() imports it
# locally, which is fine because it's a single call site.

_LAZY = {
    'sns':           ('seaborn',          None),
    'gaussian_kde':  ('scipy.stats',      'gaussian_kde'),
    'phenograph':    ('phenograph',       None),
    'plt':           ('matplotlib.pyplot', None),
}


def __getattr__(name):
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = spec
    import importlib
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, attr) if attr else mod
    globals()[name] = obj      # cache so subsequent lookups skip the hook
    return obj


# ══════════════════════════════════════════════════════════════════════════════
# QUICK-START
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── Single sample ──────────────────────────────────────────────────────
    # s = FlowSample('sample_1.fcs')
    # s.run_qc()
    # s.auto_compensate()            # spillover from FCS
    # s.apply_transform()
    # s.cluster(k=30)
    # s.run_umap()
    # s.plot_umap(color_by='cluster')
    # s.cluster_heatmap()
    # plt.show()

    # ── WSP compensation ───────────────────────────────────────────────────
    # s = FlowSample('sample_1.fcs')
    # s.compensate_from_wsp('experiment.wsp')

    # ── FMO gating ─────────────────────────────────────────────────────────
    # gater = FMOGater()
    # gater.add_fmo('Comp-FL1-A', 'fmo_fl1.fcs')  # CD901b
    # gater.add_fmo('Comp-FL2-A',           'fmo_fl2.fcs')   # CD902
    # gater.add_fmo('Comp-FL3-A',        'fmo_fl3.fcs')   # CD903
    # gater.prepare()          # compensate + transform FMOs before thresholding
    # thresholds = gater.compute(percentile=99.5)
    # s.apply_threshold_gates(thresholds)

    # ── Full experiment ────────────────────────────────────────────────────
    # exp = FlowExperiment('/path/to/fcs/')
    # exp.exclude_pattern('unstained|bead|fmo')
    # exp.run_all(k=30, umap=True)
    # exp.compare_conditions(
    #     groupA=['sample_1','sample_2'],
    #     groupB=['sample_3','sample_4'],
    #     label_a='Group A', label_b='Group B')
    # exp.export_all('results/')
    # plt.show()

    # ── WSP reader standalone ──────────────────────────────────────────────
    # reader = WspReader('experiment.wsp')
    # reader.print_matrices()
    # m = reader.get_matrix()

    print("Import this module or uncomment example blocks to run.")
