"""Tests for openflo.provenance (Methods paragraph + run manifest).

Pure stdlib; headless (no Tk / matplotlib / numpy required)."""
from __future__ import annotations

from openflo.audit import AuditLog
from openflo.provenance import (
    OPENFLO_VERSION,
    methods_paragraph,
    run_manifest,
)


# ── methods_paragraph ───────────────────────────────────────────────────────
def test_paragraph_names_methods_and_citations():
    entries = [
        {"action": "sample.load", "details": {"name": "s1", "n_events": 1000}},
        {"action": "sample.load", "details": {"name": "s2", "n_events": 500}},
        {"action": "transform", "details": {"changes": {"FITC-A": "logicle"}}},
        {"action": "cytonorm", "details": {"mode": "goal"}},
        {"action": "cluster", "details": {"method": "phenograph", "embedding": "UMAP"}},
    ]
    p = methods_paragraph(entries)
    # Methods present in the trail are named with their citations.
    assert "PhenoGraph" in p and "Levine et al., 2015" in p
    assert "UMAP" in p and "McInnes et al., 2018" in p
    assert "logicle" in p and "Parks et al., 2006" in p
    assert "CytoNorm" in p and "Van Gassen et al., 2020" in p
    # OpenFlo attribution + version.
    assert "OpenFlo" in p and OPENFLO_VERSION in p
    # Sample/event accounting from the two loads.
    assert "2 sample(s)" in p
    assert "1,500" in p


def test_paragraph_omits_absent_methods():
    entries = [
        {"action": "sample.load", "details": {"name": "s1", "n_events": 10}},
        {"action": "cluster", "details": {"method": "leiden", "embedding": "none"}},
    ]
    p = methods_paragraph(entries)
    assert "Leiden" in p and "Traag et al., 2019" in p
    # Methods that never ran must not appear.
    assert "PhenoGraph" not in p
    assert "UMAP" not in p
    assert "FlowSOM" not in p
    assert "PHATE" not in p
    assert "CytoNorm" not in p


def test_paragraph_tsne_phate_flowsom_and_citations():
    entries = [
        {"action": "cluster", "details": {"method": "flowsom", "embedding": "TSNE"}},
        {"action": "cluster", "details": {"method": "phenograph", "embedding": "PHATE"}},
    ]
    p = methods_paragraph(entries)
    assert "FlowSOM" in p and "Van Gassen et al., 2015" in p
    assert "t-SNE" in p and "van der Maaten & Hinton, 2008" in p
    assert "PHATE" in p and "Moon et al., 2019" in p


def test_paragraph_accepts_auditlog_object():
    log = AuditLog()
    log.record("sample.load", name="s1", n_events=100)
    log.record("cluster", method="phenograph", embedding="UMAP")
    p = methods_paragraph(log)
    assert "PhenoGraph" in p and "UMAP" in p


def test_paragraph_empty_trail_is_just_attribution():
    p = methods_paragraph([])
    assert "OpenFlo" in p and OPENFLO_VERSION in p
    assert "clustered" not in p
    assert "embedded" not in p


def test_paragraph_samples_argument_drives_counts():
    class _S:
        def __init__(self, name, n):
            self.name = name
            self.n_events = n

    p = methods_paragraph([], samples=[_S("a", 30), _S("b", 70)])
    assert "2 sample(s)" in p
    assert "100" in p


def test_paragraph_trajectory_and_compensate():
    entries = [
        {"action": "compensate", "details": {}},
        {"action": "trajectory", "details": {"samples": ["s1"]}},
    ]
    p = methods_paragraph(entries)
    assert "compensated" in p
    assert "trajector" in p.lower()


# ── run_manifest ─────────────────────────────────────────────────────────────
def test_manifest_has_expected_keys():
    m = run_manifest()
    for key in (
        "openflo_version",
        "python",
        "platform",
        "packages",
        "params",
        "samples",
        "generated_hint",
    ):
        assert key in m
    for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "umap-learn"):
        assert pkg in m["packages"]
    assert m["openflo_version"] == OPENFLO_VERSION
    assert m["params"] == {}
    assert m["samples"] == []


def test_manifest_reflects_params_and_samples():
    samples = [
        {"name": "s1", "file": "/data/run/s1.fcs", "n_events": 1000},
        {"name": "s2", "file": "s2.fcs", "n_events": 500},
    ]
    m = run_manifest(params={"k": 30, "method": "phenograph"}, samples=samples)
    assert m["params"] == {"k": 30, "method": "phenograph"}
    assert len(m["samples"]) == 2
    # File is reduced to a basename (no path leak).
    assert m["samples"][0] == {"name": "s1", "file": "s1.fcs", "n_events": 1000}
    assert m["samples"][1]["file"] == "s2.fcs"


def test_manifest_is_deterministic_no_clock():
    # No wall-clock dependency -> two calls are identical and carry a hint
    # telling the caller they own the timestamp.
    a = run_manifest(params={"x": 1})
    b = run_manifest(params={"x": 1})
    assert a == b
    assert a["generated_hint"] == "stamp at call site"


def test_manifest_handles_object_samples():
    class _S:
        def __init__(self, name, path, n):
            self.name = name
            self.path = path
            self.n_events = n

    m = run_manifest(samples=[_S("s1", "C:/x/y/s1.fcs", 42)])
    assert m["samples"][0] == {"name": "s1", "file": "s1.fcs", "n_events": 42}
