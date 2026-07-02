"""Provenance & reproducibility helpers (headless, stdlib-only).

Two complementary outputs are produced from a finished analysis session:

* :func:`methods_paragraph` — a paper-ready *Methods* paragraph in prose,
  generated from an :class:`~openflo.audit.AuditLog` (or a plain list of its
  entry dicts). It names only the analytical methods that actually appear in
  the audit trail and attaches the matching literature citation for each.
* :func:`run_manifest` — a JSON-able dict capturing the runtime environment
  (OpenFlo / Python / platform versions, key scientific-stack package
  versions), the parameters of the run, and the samples it operated on. It is
  deterministic: it does **not** read the wall clock, so the caller is
  responsible for stamping the time (see ``generated_hint``).

This module is intentionally free of Tk / matplotlib / numpy imports so it can
be exercised in CI and unit tests without a display or the heavy optional
dependencies. Version lookups go through :mod:`importlib.metadata` and degrade
to ``'unknown'`` when a package is not installed.

Audit ``action`` strings are the ones emitted by the application, e.g.
``'sample.load'``, ``'transform'``, ``'compensate'`` / ``'cytonorm'`` /
``'unmix'``, ``'cluster'`` (with ``details['method']`` in
{phenograph, leiden, flowsom} and ``details['embedding']`` in
{UMAP, TSNE, TRIMAP, PACMAP, PHATE, none}), ``'trajectory'``, ``'calibration'``,
``'session.load'``, ``'autogate.*'``, etc.
"""
from __future__ import annotations

import importlib.metadata
import os
import platform

__all__ = ["methods_paragraph", "run_manifest", "OPENFLO_VERSION"]


def _version(dist):
    """``importlib.metadata.version`` that never raises; -> 'unknown'."""
    try:
        return importlib.metadata.version(dist)
    except Exception:
        return "unknown"


OPENFLO_VERSION = _version("openflo")


# ── citations ─────────────────────────────────────────────────────────────
# Short inline-citation strings keyed by a stable method id. These mirror the
# `references` block in CITATION.cff at the repo root.
CITATIONS = {
    "phenograph": "Levine et al., 2015",
    "umap": "McInnes et al., 2018",
    "tsne": "van der Maaten & Hinton, 2008",
    "leiden": "Traag et al., 2019",
    "flowsom": "Van Gassen et al., 2015",
    "phate": "Moon et al., 2019",
    "trimap": "Amid & Warmuth, 2019",
    "pacmap": "Wang et al., 2021",
    "logicle": "Parks et al., 2006",
    "cytonorm": "Van Gassen et al., 2020",
}

# Human-readable display names for the embeddings, keyed by the lowercased
# `details['embedding']` prefix the app records.
_EMBED_NAMES = {
    "umap": "UMAP",
    "tsne": "t-SNE",
    "trimap": "TriMap",
    "pacmap": "PaCMAP",
    "phate": "PHATE",
}

# Display names for the clustering algorithms (lowercased `details['method']`).
_CLUSTER_NAMES = {
    "phenograph": "PhenoGraph",
    "leiden": "Leiden community detection",
    "flowsom": "FlowSOM",
}


def _cite(method_id):
    cit = CITATIONS.get(method_id)
    return f" ({cit})" if cit else ""


def _iter_entries(audit_entries):
    """Yield ``(action, details)`` tuples from an AuditLog or list of dicts."""
    if audit_entries is None:
        return
    # AuditLog exposes .entries(); a plain list is iterated directly.
    entries = audit_entries.entries() if hasattr(audit_entries, "entries") else audit_entries
    for e in entries:
        if not isinstance(e, dict):
            continue
        action = str(e.get("action", "")).lower()
        details = e.get("details") or {}
        if not isinstance(details, dict):
            details = {}
        yield action, details


def methods_paragraph(audit_entries, samples=None):
    """Build a paper-ready Methods paragraph from an audit trail.

    Parameters
    ----------
    audit_entries : AuditLog | list[dict] | None
        The session's audit log, or its ``.entries()`` list of
        ``{'action', 'details', ...}`` dicts.
    samples : sequence | None
        Optional sample objects/dicts (with ``name`` / ``n_events``) used only
        to phrase the opening sentence (sample & event counts). Each may be a
        mapping or an object with ``.name`` / ``.n_events`` attributes.

    Returns
    -------
    str
        A single paragraph naming only the methods that appear in the trail,
        each with its literature citation, ending with an OpenFlo + version
        attribution. Returns a minimal attribution sentence if nothing maps.
    """
    actions = []      # ordered, de-duplicated method ids that have prose
    seen = set()

    def _add(mid):
        if mid and mid not in seen:
            seen.add(mid)
            actions.append(mid)

    has_compensate = has_cytonorm = has_unmix = has_logicle = False
    has_trajectory = has_calibration = False
    n_samples_loaded = 0
    events_loaded = 0

    for action, details in _iter_entries(audit_entries):
        if action == "sample.load":
            n_samples_loaded += 1
            ne = details.get("n_events")
            if isinstance(ne, (int, float)):
                events_loaded += int(ne)
        elif action in ("compensate", "compensation"):
            has_compensate = True
        elif action == "cytonorm":
            has_cytonorm = True
            _add("cytonorm")
        elif action == "unmix":
            has_unmix = True
        elif action == "transform":
            # transform details carry a `changes` map {channel: method}.
            changes = details.get("changes") or {}
            methods = list(changes.values()) if isinstance(changes, dict) else []
            if any(str(m).lower() == "logicle" for m in methods):
                has_logicle = True
                _add("logicle")
        elif action == "cluster":
            method = str(details.get("method", "")).lower()
            if method in _CLUSTER_NAMES:
                _add(method)
            emb = str(details.get("embedding", "")).lower()
            if emb in _EMBED_NAMES:
                _add(emb)
        elif action == "trajectory":
            has_trajectory = True
        elif action == "calibration":
            has_calibration = True

    sentences = []

    # Opening: sample/event accounting.
    samples = list(samples) if samples else []
    n = len(samples) or n_samples_loaded
    if n:
        evt = _total_events(samples) or events_loaded
        if evt:
            sentences.append(
                f"Flow-cytometry data from {n} sample(s) "
                f"({evt:,} total events) were analyzed in OpenFlo "
                f"(v{OPENFLO_VERSION})."
            )
        else:
            sentences.append(
                f"Flow-cytometry data from {n} sample(s) were analyzed in "
                f"OpenFlo (v{OPENFLO_VERSION})."
            )
    else:
        sentences.append(f"Data were analyzed in OpenFlo (v{OPENFLO_VERSION}).")

    # Preprocessing.
    pre = []
    if has_compensate:
        pre.append("spectral spillover was compensated")
    if has_unmix:
        pre.append("spectral data were unmixed")
    if has_logicle:
        pre.append(f"fluorescence channels were transformed with the logicle method{_cite('logicle')}")
    if has_cytonorm:
        pre.append(f"batch effects were normalized with CytoNorm{_cite('cytonorm')}")
    if has_calibration:
        pre.append("fluorescence intensities were converted to MESF units by bead calibration")
    if pre:
        clause = _join_clauses(pre)
        sentences.append(clause[:1].upper() + clause[1:] + ".")

    # Clustering.
    cluster_ids = [m for m in actions if m in _CLUSTER_NAMES]
    if cluster_ids:
        parts = [f"{_CLUSTER_NAMES[m]}{_cite(m)}" for m in cluster_ids]
        sentences.append(f"Cells were clustered with {_join_clauses(parts)}.")

    # Embeddings.
    embed_ids = [m for m in actions if m in _EMBED_NAMES]
    if embed_ids:
        parts = [f"{_EMBED_NAMES[m]}{_cite(m)}" for m in embed_ids]
        sentences.append(f"Single-cell data were embedded with {_join_clauses(parts)}.")

    # Trajectory.
    if has_trajectory:
        sentences.append(
            "Developmental trajectories were inferred from the single-cell data."
        )

    return " ".join(sentences)


def _total_events(samples):
    total = 0
    for s in samples:
        n = _attr(s, "n_events")
        if isinstance(n, (int, float)):
            total += int(n)
    return total


def _join_clauses(parts):
    """Oxford-style 'a, b and c' joining."""
    parts = [p for p in parts if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def _attr(obj, name, default=None):
    """Read ``name`` from a mapping or an object, tolerantly."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def run_manifest(params=None, samples=None):
    """Capture a deterministic, JSON-able reproducibility manifest.

    Records the OpenFlo / Python / platform versions, the versions of the key
    scientific-stack packages, the run ``params``, and the ``samples`` (name,
    file basename, and event count). It deliberately does **not** stamp the
    time — see ``generated_hint`` — so callers get reproducible output and own
    the timestamp.

    Parameters
    ----------
    params : dict | None
        Arbitrary run parameters to embed verbatim. Defaults to ``{}``.
    samples : sequence | None
        Sample objects/dicts. Each contributes ``{'name', 'file', 'n_events'}``
        where ``file`` is reduced to a basename to avoid leaking paths.

    Returns
    -------
    dict
        ``{'openflo_version', 'python', 'platform', 'packages', 'params',
        'samples', 'generated_hint'}``.
    """
    pkgs = {
        "numpy": _version("numpy"),
        "pandas": _version("pandas"),
        "scipy": _version("scipy"),
        "scikit-learn": _version("scikit-learn"),
        "umap-learn": _version("umap-learn"),
    }

    sample_rows = []
    for s in (samples or []):
        f = _attr(s, "file") or _attr(s, "path") or _attr(s, "filename")
        sample_rows.append(
            {
                "name": _attr(s, "name"),
                "file": os.path.basename(f) if f else None,
                "n_events": _attr(s, "n_events"),
            }
        )

    return {
        "openflo_version": OPENFLO_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": pkgs,
        "params": dict(params) if params else {},
        "samples": sample_rows,
        "generated_hint": "stamp at call site",
    }
