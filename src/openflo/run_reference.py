"""Frozen reference clusters for a run template: one FlowSOM map every
later run labels its events with, so a cluster id means the same
population on every day.

A run clustered with FlowSOM writes its map beside its outputs
(``<label>_flowsom_map.json``): the channels, each node's prototype, the
metacluster of each node, the transform each channel was on, and how far
the run's own events lay from their nearest node (the 95th percentile).
Freezing copies one such map into the template's run settings
(``run_cfg['reference']``). A run with a reference does not cluster: each
event takes the metacluster of its nearest node, as FlowSOM labels its own
events, and the run says what share of its events lie farther from every
node than 95% of the reference run's did (about 5% when nothing moved).
Each member's share of each reference cluster, on all its events (before
the run's max-events cap), joins the run's population table under its
population's name, so the Run history window compares clusters across
days.

Core module: numpy, no Tk.
"""
from __future__ import annotations

import glob
import json
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

MAP_SUFFIX = '_flowsom_map.json'
MAP_FORMAT = 'openflo-flowsom-map'
# The reference run's own distance percentile: the share of its events
# beyond it is 100 - FAR_PERCENTILE by construction.
FAR_PERCENTILE = 95.0
# Warn when more of a run's events than this lie beyond it: twice what an
# unchanged sample gives.
DRIFT_WARN_PCT = 10.0
REFERENCE_PREFIX = 'Reference '
REFERENCE_SEP = ' · '


def node_distances(X, W, nodes) -> np.ndarray:
    """Each row of `X`'s Euclidean distance to its node `nodes` in `W`."""
    return np.sqrt(((np.asarray(X, float) - W[nodes]) ** 2).sum(1))


def _finite_rows(data, channels: Sequence[str]):
    X = data[list(channels)].to_numpy(dtype=float)
    return X, np.all(np.isfinite(X), axis=1)


def build_map(som: Mapping[str, Any], data, transforms: Mapping[str, Any],
              source: str) -> dict[str, Any]:
    """The map of a FlowSOM run (``FlowSample.flowsom_result`` `som`, on
    the run's events `data` whose channels carry `transforms`), named
    `source`: JSON-ready."""
    from .flowsom import som_assign
    channels = [str(c) for c in som['channels']]
    W = np.asarray(som['weights'], dtype=float)
    X, ok = _finite_rows(data, channels)
    X = X[ok]
    d = node_distances(X, W, som_assign(X, W))
    return {'format': MAP_FORMAT, 'source': source, 'channels': channels,
            'weights': W.tolist(),
            'meta_of_node': [int(m) for m in som['meta_of_node']],
            'transforms': {c: transforms.get(c) for c in channels},
            'far': float(np.percentile(d, FAR_PERCENTILE)),
            'n_events': int(len(X))}


def write_map(out_dir: str, label: str, m: Mapping[str, Any]) -> str:
    """Write map `m` as ``<label>_flowsom_map.json`` in `out_dir`."""
    from .atomic_write import atomic_write_json
    path = os.path.join(out_dir, label + MAP_SUFFIX)
    atomic_write_json(path, m, indent=None)
    return path


def check_map(m: Any) -> dict[str, Any]:
    """`m` when it is a whole map; raises ValueError naming what is
    wrong."""
    if not isinstance(m, dict) or m.get('format') != MAP_FORMAT:
        raise ValueError('not an OpenFlo FlowSOM map')
    channels = m.get('channels')
    if not channels or not all(isinstance(c, str) for c in channels):
        raise ValueError('the map names no channels')
    try:
        W = np.asarray(m.get('weights'), dtype=float)
        meta = np.asarray(m.get('meta_of_node'), dtype=int)
        far = float(m.get('far', 'nan'))
    except (TypeError, ValueError) as exc:
        raise ValueError(f'the map is damaged ({exc})') from exc
    if W.ndim != 2 or W.shape[1] != len(channels) \
            or not np.all(np.isfinite(W)):
        raise ValueError('the map has no node for each of its channels')
    if meta.shape != (len(W),):
        raise ValueError('the map has no metacluster for each node')
    if not math.isfinite(far) or far < 0:
        raise ValueError('the map has no reference distance')
    return m


def read_map(path: str) -> dict[str, Any]:
    """The map in `path`. Raises OSError or ValueError."""
    with open(path, encoding='utf-8') as fh:
        return check_map(json.load(fh))


def _scale(spec: Any) -> str:
    return str((spec or {}).get('method', 'linear'))


def label_events(data, ref: Mapping[str, Any],
                 transforms: Mapping[str, Any]) -> tuple[np.ndarray, float]:
    """``(labels, far_pct)``: each event of `data` labelled with the
    metacluster of its nearest node of map `ref` (-1 for a row with a
    non-finite value), and the % of labelled events farther from it than
    the reference's own 95th percentile. Raises ValueError when `data`
    lacks a map channel, or holds one on a scale other than the map's:
    a distance across two scales means nothing."""
    from .flowsom import som_assign
    check_map(ref)
    channels = ref['channels']
    missing = [c for c in channels if c not in data.columns]
    if missing:
        raise ValueError('the frozen reference needs channel(s) this run '
                         'lacks: ' + ', '.join(missing))
    want = ref.get('transforms') or {}
    for c in channels:
        if json.dumps(want.get(c), sort_keys=True) != \
                json.dumps(transforms.get(c), sort_keys=True):
            raise ValueError(
                f'{c} is on {_scale(transforms.get(c))} here but on '
                f'{_scale(want.get(c))} in the frozen reference: put it back '
                'on that scale, or freeze a new reference')
    W = np.asarray(ref['weights'], dtype=float)
    meta = np.asarray(ref['meta_of_node'], dtype=int)
    X, ok = _finite_rows(data, channels)
    labels = np.full(len(X), -1, dtype=int)
    if not ok.any():
        return labels, 0.0
    nodes = som_assign(X[ok], W)
    labels[ok] = meta[nodes]
    far = node_distances(X[ok], W, nodes) > float(ref['far'])
    return labels, float(100.0 * far.mean())


def drift_warning(far_pct: float) -> str | None:
    """The run's warning when `far_pct` % of its events lie beyond the
    reference's 95th percentile, or None when that is no more than
    :data:`DRIFT_WARN_PCT`."""
    if far_pct <= DRIFT_WARN_PCT:
        return None
    return (f"{far_pct:.0f}% of events lie farther from the frozen "
            f"reference clusters than 95% of the reference run's events "
            f"did (about {100 - FAR_PERCENTILE:.0f}% expected): a population "
            "may have moved or appeared, or the instrument drifted. Check "
            "before comparing clusters across runs")


def member_rows(labels, ref: Mapping[str, Any], key: Mapping[str, str],
                population: str) -> list[dict]:
    """Population-table rows of one run-unit member's share of each
    reference cluster (``%Parent``, of all that member's events), zero
    included so an emptied cluster reads 0 and not missing. `labels` are
    :func:`label_events` labels of the member's own events (all of them,
    not the run's subsample); `key` is its Group, Trial and Sample. Rows
    are named by the member's `population` (``<population> · Reference
    Cluster N``): one file can be two members of a unit (a parent and its
    child), and its shares of each must not be pooled. The noise row (-1)
    is there only when some event is in it."""
    from .cluster_labels import cluster_label
    labels = np.asarray(labels, dtype=int)
    if not len(labels):
        return []
    ids = sorted({int(m) for m in ref['meta_of_node']})
    if (labels < 0).any():
        ids = [-1, *ids]
    return [{**key, 'Population': f'{population}{REFERENCE_SEP}'
                                  f'{REFERENCE_PREFIX}{cluster_label(cid)}',
             '%Parent': 100.0 * float((labels == cid).mean())} for cid in ids]


def run_maps(hdir: str) -> list[tuple[str, str]]:
    """``(title, path)`` of each FlowSOM map among the active runs of
    history folder `hdir`, newest run first."""
    from .run_history import load_runs, mark_active
    out = []
    for r in reversed(mark_active(load_runs(hdir))):
        if not r['active']:                 # only finished runs are active
            continue
        for p in sorted(glob.glob(os.path.join(glob.escape(r['dir']),
                                               '*' + MAP_SUFFIX))):
            label = os.path.basename(p)[:-len(MAP_SUFFIX)]
            out.append((f"{r['run_id']} ({r.get('experiment_date')}) · "
                        f"{label}", p))
    return out


def set_template_reference(tpath: str, ref: Mapping[str, Any] | None) -> dict:
    """Put map `ref` in the run settings of the template at `tpath` (or
    take the reference out, with None) and save it, whole or not at all
    (it is the user's only copy). Returns the template."""
    from .atomic_write import atomic_write_json
    from .run_template import template_from_json, template_to_json
    with open(tpath, encoding='utf-8') as fh:
        tpl = template_from_json(json.load(fh))
    if ref is None:
        tpl['run_cfg'].pop('reference', None)
    else:
        tpl['run_cfg']['reference'] = dict(check_map(ref))
    atomic_write_json(tpath, template_to_json(tpl))
    return tpl
