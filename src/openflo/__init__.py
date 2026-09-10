"""OpenFlo — flow cytometry analysis pipeline.

Public surface re-exported here so ``from openflo import FlowSample`` works
without callers needing to know which submodule the symbol lives in. The
heavyweight submodules (``openflo.gui``, ``openflo.preview``) are NOT
imported here — importing the package shouldn't load Tk or matplotlib's
Tk backend.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Static type checkers see the public surface up-front. At runtime the
    # ``__getattr__`` hook below resolves these lazily without paying the
    # phenograph / pandas import cost just to ``import openflo``.
    from .annotate import (
        annotate_by_reference,
        mem_label,
        mem_scores,
        parse_signature_table,
        population_states,
    )
    from .audit import AuditLog
    from .calibration import (
        apply_calibration,
        detect_bead_peaks,
        fit_mesf_calibration,
    )
    from .compliance import (
        build_manifest,
        record_to_markdown,
        sign_manifest,
        verify_record,
    )
    from .diffexp import (
        cluster_abundance,
        differential_abundance,
        differential_test,
        marker_expression,
    )
    from .interop import (
        mds_embed,
        sample_distance_matrix,
        to_anndata,
        write_h5ad,
    )
    from .pipeline import (
        ClusteringError,
        CompensationError,
        CytoNorm,
        FcsParseError,
        FlowExperiment,
        FlowSample,
        FMOGater,
        GateError,
        OpenFloError,
        WspParseError,
        WspReader,
        WspWriter,
        concatenate,
        cumulative_gate_mask,
        describe_gate,
        flowsom_layout,
        flowsom_mst,
        gate_to_mask,
        optimize_compensation,
        read_compensation_matrix,
        read_template_gates,
        set_default_palette,
        write_compensation_matrix,
        write_fcs,
    )
    from .report import (
        build_html_report,
        df_to_html_table,
        figure_to_data_uri,
    )
    from .spectral import (
        apply_unmixing,
        build_reference_spectra,
        spectral_condition_number,
        spectral_similarity_matrix,
        spillover_spread_matrix,
        unmix,
        unmixing_qc,
    )
    from .stats import (
        compare_all_features,
        compare_groups,
        group_kde,
        p_to_stars,
        to_prism_column,
        to_prism_grouped,
        volcano_data,
    )
    from .trajectory import (
        compute_pseudotime,
        pseudotime_trends,
        robust_root,
    )
    from .voltage import VoltageTitration

try:
    __version__ = _pkg_version("openflo")
except PackageNotFoundError:
    # Package not installed (e.g. running from a source checkout without
    # `pip install -e .`). Fall back to a sentinel so callers can still
    # introspect.
    __version__ = "0.0.0+unknown"

# Public surface — submodules are imported lazily on first attribute access
# (PEP 562) so just `import openflo` stays cheap (~150 ms for numpy+pandas).
_PUBLIC = {
    # Calibration
    "detect_bead_peaks":     "openflo.calibration",
    "fit_mesf_calibration":  "openflo.calibration",
    "apply_calibration":     "openflo.calibration",
    # Compliance
    "build_manifest":        "openflo.compliance",
    "sign_manifest":         "openflo.compliance",
    "verify_record":         "openflo.compliance",
    "record_to_markdown":    "openflo.compliance",
    # Annotation
    "mem_scores":            "openflo.annotate",
    "mem_label":             "openflo.annotate",
    "population_states":     "openflo.annotate",
    "parse_signature_table": "openflo.annotate",
    "annotate_by_reference": "openflo.annotate",
    # Provenance
    "AuditLog":              "openflo.audit",
    # Core pipeline
    "FlowSample":            "openflo.pipeline",
    "FMOGater":              "openflo.pipeline",
    "FlowExperiment":        "openflo.pipeline",
    "CytoNorm":              "openflo.pipeline",
    "concatenate":           "openflo.pipeline",
    # IO
    "WspReader":             "openflo.pipeline",
    "WspWriter":             "openflo.pipeline",
    "read_template_gates":   "openflo.pipeline",
    "read_compensation_matrix":  "openflo.pipeline",
    "write_compensation_matrix": "openflo.pipeline",
    "optimize_compensation": "openflo.pipeline",
    # Gates
    "describe_gate":         "openflo.pipeline",
    "gate_to_mask":          "openflo.pipeline",
    "cumulative_gate_mask":  "openflo.pipeline",
    # Defaults / config
    "set_default_palette":   "openflo.pipeline",
    "write_fcs":             "openflo.pipeline",
    "flowsom_mst":           "openflo.pipeline",
    "flowsom_layout":        "openflo.pipeline",
    # Tools
    "VoltageTitration":      "openflo.voltage",
    "differential_test":     "openflo.diffexp",
    "differential_abundance": "openflo.diffexp",
    "cluster_abundance":     "openflo.diffexp",
    "marker_expression":     "openflo.diffexp",
    "sample_distance_matrix": "openflo.interop",
    "mds_embed":             "openflo.interop",
    "to_anndata":            "openflo.interop",
    "write_h5ad":            "openflo.interop",
    "build_reference_spectra": "openflo.spectral",
    "unmix":                 "openflo.spectral",
    "apply_unmixing":        "openflo.spectral",
    "spectral_similarity_matrix": "openflo.spectral",
    "spectral_condition_number":  "openflo.spectral",
    "spillover_spread_matrix":    "openflo.spectral",
    "unmixing_qc":           "openflo.spectral",
    "build_html_report":     "openflo.report",
    "df_to_html_table":      "openflo.report",
    "figure_to_data_uri":    "openflo.report",
    "compare_groups":        "openflo.stats",
    "compare_all_features":  "openflo.stats",
    "volcano_data":          "openflo.stats",
    "group_kde":             "openflo.stats",
    "to_prism_column":       "openflo.stats",
    "to_prism_grouped":      "openflo.stats",
    "p_to_stars":            "openflo.stats",
    "compute_pseudotime":    "openflo.trajectory",
    "pseudotime_trends":     "openflo.trajectory",
    "robust_root":           "openflo.trajectory",
    # Exception hierarchy
    "OpenFloError":          "openflo.pipeline",
    "FcsParseError":         "openflo.pipeline",
    "CompensationError":     "openflo.pipeline",
    "WspParseError":         "openflo.pipeline",
    "GateError":             "openflo.pipeline",
    "ClusteringError":       "openflo.pipeline",
}


def __getattr__(name):
    mod_name = _PUBLIC.get(name)
    if mod_name is None:
        raise AttributeError(f"module 'openflo' has no attribute {name!r}")
    import importlib
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, name)
    globals()[name] = obj   # cache
    return obj


# Build __all__ as a plain literal so pyright can verify it. Keep in sync
# with _PUBLIC above.
__all__ = [
    "AuditLog",
    "ClusteringError",
    "annotate_by_reference",
    "apply_calibration",
    "build_manifest",
    "CompensationError",
    "CytoNorm",
    "FcsParseError",
    "FlowExperiment",
    "FlowSample",
    "FMOGater",
    "GateError",
    "OpenFloError",
    "VoltageTitration",
    "WspParseError",
    "WspReader",
    "WspWriter",
    "apply_unmixing",
    "build_html_report",
    "build_reference_spectra",
    "cluster_abundance",
    "compare_groups",
    "compare_all_features",
    "volcano_data",
    "compute_pseudotime",
    "concatenate",
    "cumulative_gate_mask",
    "describe_gate",
    "detect_bead_peaks",
    "df_to_html_table",
    "differential_abundance",
    "differential_test",
    "figure_to_data_uri",
    "fit_mesf_calibration",
    "flowsom_layout",
    "flowsom_mst",
    "gate_to_mask",
    "group_kde",
    "marker_expression",
    "mds_embed",
    "mem_label",
    "mem_scores",
    "optimize_compensation",
    "p_to_stars",
    "parse_signature_table",
    "population_states",
    "pseudotime_trends",
    "read_compensation_matrix",
    "read_template_gates",
    "record_to_markdown",
    "robust_root",
    "sample_distance_matrix",
    "set_default_palette",
    "sign_manifest",
    "spectral_condition_number",
    "spectral_similarity_matrix",
    "spillover_spread_matrix",
    "to_anndata",
    "to_prism_column",
    "to_prism_grouped",
    "unmix",
    "unmixing_qc",
    "verify_record",
    "write_compensation_matrix",
    "write_fcs",
    "write_h5ad",
    "__version__",
]
