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

import logging
import warnings

from .acquisition_qc import AcquisitionQC as AcquisitionQC
from .autoclean import find_scatter_channel as find_scatter_channel
from .cell_cycle import analyze_dna as analyze_dna
from .cell_cycle import assign_phase as assign_phase
from .cell_cycle import find_dna_channel as find_dna_channel
from .channel_names import is_time_channel as is_time_channel
from .channel_roles import default_cluster_channels as default_cluster_channels
from .channel_transforms import UnknownScaleError as UnknownScaleError
from .channel_transforms import guessed_spec as guessed_spec
from .channel_transforms import inverse_transform_values as inverse_transform_values
from .channel_transforms import is_guessed_spec as is_guessed_spec
from .channel_transforms import is_unknown_spec as is_unknown_spec
from .channel_transforms import linear_values as linear_values
from .channel_transforms import to_linear as to_linear
from .channel_transforms import transform_spec as transform_spec
from .channel_transforms import transform_values as transform_values
from .cluster_labels import cluster_label as cluster_label
from .exceptions import ClusteringError as ClusteringError
from .exceptions import CompensationError as CompensationError
from .exceptions import FcsParseError as FcsParseError
from .exceptions import GateError as GateError
from .exceptions import OpenFloError as OpenFloError
from .exceptions import WspParseError as WspParseError
from .fcs_export import df_path_args as df_path_args
from .fcs_quirks import plain_fcs_problem as plain_fcs_problem
from .fcs_quirks import read_fcs_tolerant as read_fcs_tolerant
from .fcs_write import write_fcs as write_fcs
from .flow_experiment import FlowExperiment as FlowExperiment
from .flow_sample import FlowSample
from .flow_sample import categorical_cmap as categorical_cmap
from .flow_sample import set_default_palette as set_default_palette
from .flowsom import igraph_seeded as igraph_seeded
from .fmo_gater import FMOGater as FMOGater
from .gate_eval import depends_on_flowjo_placeholder as depends_on_flowjo_placeholder
from .gate_eval import gate_to_mask as gate_to_mask
from .gate_text import WSP_BOOLEAN_OPS as WSP_BOOLEAN_OPS
from .gate_text import (
    describe_gate,
)
from .gate_text import resolve_wsp_dependent as resolve_wsp_dependent
from .gating_helpers import fmo_threshold as fmo_threshold
from .gpu_probe import GPU_AVAILABLE as GPU_AVAILABLE
from .gpu_probe import GPU_CLUSTERING_AVAILABLE as GPU_CLUSTERING_AVAILABLE
from .gpu_probe import GPU_NAME as GPU_NAME
from .label_alignment import concatenate as concatenate
from .parallel import tree_kneighbors as tree_kneighbors
from .scales import ASINH_DEFAULT_COFACTOR as ASINH_DEFAULT_COFACTOR
from .scales import LOGICLE_A as LOGICLE_A
from .scales import LOGICLE_M as LOGICLE_M
from .scales import LOGICLE_T as LOGICLE_T
from .scales import LOGICLE_W as LOGICLE_W
from .som_kernel import FLOWSOM_DEFAULT_METACLUSTERS as FLOWSOM_DEFAULT_METACLUSTERS
from .transform_sidecar import discard_transforms_sidecar as discard_transforms_sidecar
from .transform_sidecar import write_transforms_sidecar as write_transforms_sidecar
from .wsp_reader import WspReader as WspReader
from .wsp_writer import WspWriter as WspWriter

# Heavy / optional deps are loaded on demand via the PEP-562 ``__getattr__``
# hook at the bottom of this module:
#   - ``phenograph``       — only needed by FlowSample.cluster()
#   - ``seaborn``          — only by heatmap-style plots
#   - ``gaussian_kde``     — only by density plot paths
#   - ``matplotlib.pyplot`` — only by the plot methods (~370 ms saved;
#     re-measured 2026-09-10, `python scripts/bench_import_cost.py`)
# Importing the module does not pull in igraph + scikit-learn's community
# detection OR matplotlib's Tk backend probing. Matters for the gate
# editor / compare tool / WSP-only callers — `import openflo.pipeline`
# drops from ~1050 ms to ~400 ms.
#
# Plot methods inside this module add a local `import matplotlib.pyplot
# as plt` at the top because PEP-562 __getattr__ only fires for OTHER
# modules accessing `pipeline.plt`; bare-name lookups inside this module
# follow normal scoping rules and would NameError without the local.


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
#
# OpenFloError and its subclasses live in exceptions; they are imported at
# the top, so pipeline.OpenFloError and the rest are the same classes.


# ── Constants ─────────────────────────────────────────────────────────────────

# Deprecated public SCATTER_KEYWORDS (served by __getattr__ below with a
# DeprecationWarning); classification uses _is_scatter, which matches these
# (and Beckman / Sony / CyTOF names) case-insensitively.
_SCATTER_KEYWORDS = ['FSC', 'SSC', 'Time', 'time', 'Width', 'width']
# EXCLUDE_CLUSTER, the derived / analysis columns never treated as markers,
# lives in channel_roles; pipeline no longer uses it, so it is served,
# deprecated, by __getattr__ at the bottom.

# Phenograph (native Louvain backend) writes scratch files — kNN graph
# `*.bin`, dendrogram `*.tree`, and `*_graph.weights` — into the current
# working directory, and deletes them only when the run succeeds. Each call
# runs in its own temporary folder, removed afterwards whatever happens. A
# fixed folder inside the package would not do: a pip install puts it in
# site-packages (read-only on a system Python), and it would keep the scratch
# files of every failed run.
#
# The cap that keeps phenograph's size sort to the caller's n_jobs
# (phenograph_sort_jobs) lives in cluster_graph.


# The default categorical palette (categorical_cmap, set_default_palette)
# lives in flow_sample with FlowSample, whose plots read it; both are imported
# at the top.


# ── Gate model & evaluator ────────────────────────────────────────────────────
#
# The gate schema, gate_to_mask, cumulative_gate_mask and the unknown-population
# checks (unknown_population_reason, broken_booleans, ...) live in gate_eval;
# the names used here are imported at the top under their old names, and the
# rest are served, deprecated, through _MOVED at the bottom.


# ── Auto-clean (acquisition-cleaning) gate ─────────────────────────────────
#
# The auto-clean recipe (AUTOCLEAN_METHODS), each method's keep-mask and
# autoclean_keep_mask live in autoclean; find_scatter_channel is imported at
# the top, and the rest are served, deprecated, through _MOVED at the bottom.


# ── Seeds, the SNN graph and cluster labels ───────────────────────────────────
#
# resolve_seed lives in seeds, the SNN graph and the phenograph size-sort cap
# in cluster_graph, and GATE_PALETTE, the noise-cluster sentinel and
# cluster_label in cluster_labels. The names used here are imported at the top
# under their old names; the rest are served, deprecated, through _MOVED at
# the bottom.


# ── FlowJo .wsp writer ────────────────────────────────────────────────────────
#
# WspWriter and its export checks live in wsp_writer; WspWriter is imported at
# the top under its old name, and the rest are served, deprecated, through
# _MOVED at the bottom.


# ── FCS TEXT and Time units at the .wsp boundary ─────────────────────────────
#
# The FCS TEXT keywords and Time units at the .wsp boundary (fcs_text_pairs,
# fcs_timestep, wsp_time_gain, scale_time_coords, ...) live in fcs_text; the
# names used here are imported at the top under their old names, and
# _clock_seconds is served, deprecated, by __getattr__ at the bottom.


# ── Gate coordinates across the .wsp boundary ───────────────────────────────
#
# map_gate_coords and the scale functions it takes live in wsp_gate_coords;
# the names used here are imported at the top under their old names, and the
# rest are served, deprecated, through _MOVED at the bottom. What stays here
# needs FlowSample or the gate evaluator.


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


# ── Compensation matrix IO ────────────────────────────────────────────────────
#
# Reading and writing compensation matrix files (read_compensation_matrix,
# write_compensation_matrix and the SPILL keyword parser) lives in
# compensation_io; the names used here are imported at the top under their old
# names, and the rest are served, deprecated, through _MOVED at the bottom.


# ── Compensation matrix optimizer and gate templates ──────────────────────────
#
# The single-stain spillover estimate (optimize_compensation) lives in
# compensation_optimizer, and the template readers (read_template_gates,
# wsp_template_stat_axes and their helpers) in gate_templates; pipeline uses
# none of them, so they are served, deprecated, through _MOVED at the bottom.


# ── Transforms ─────────────────────────────────────────────────────────────
#
# The channel transforms (transform_values, linear_values, the transform
# records) live in channel_transforms; the ones used here are imported at
# the top, the rest are served, deprecated, by __getattr__ at the bottom.
# The unknown-aware matrix product compensation uses (_matmul_known) lives in
# compensation_math; FlowSample, in flow_sample, imports it.


# ── Transform records beside a written CSV ────────────────────────────────────
#
# The transform records beside a written CSV (write_transforms_sidecar,
# read_transforms_sidecar, logicle_display_bounds, ...) live in
# transform_sidecar; the gate text and FlowJo population naming
# (describe_gate, gate_value_text, wsp_population_name, WSP_BOOLEAN_OPS, ...)
# live in gate_text. The names used here are imported at the top, the rest are
# served, deprecated, by __getattr__ at the bottom.


# ── Automated density-based gating ─────────────────────────────────────────────
#
# The density-based gate suggestions (auto_threshold, auto_polygon_gate,
# auto_singlet_gate, gmm_ellipse_gates) live in auto_gating; pipeline uses
# none of them, so the old names are served, deprecated, by __getattr__ at the
# bottom.


# ── FlowSOM (self-organizing map + metaclustering) ────────────────────────────
#
# FlowSOM (som_train, som_assign, som_metacluster, flowsom_mst, igraph_seeded,
# flowsom_layout) lives in flowsom; the names used here are imported at the
# top, under their old names, and flowsom_mst / flowsom_layout are served,
# deprecated, by __getattr__ at the bottom.


# ── CytoNorm batch normalization ──────────────────────────────────────────────
#
# CytoNorm (CytoNorm, cap_events_per_batch, _strict_increasing) lives in
# cytonorm; pipeline uses none of it, so the old names are served, deprecated,
# by __getattr__ at the bottom.


# ── GPU probe ─────────────────────────────────────────────────────────────────
#
# The GPU probe (GPU_AVAILABLE, GPU_NAME, the cuML UMAP class, the RAPIDS
# clustering kit, the free-VRAM query) lives in gpu_probe; the names used
# here are imported at the top, _probe_gpu is served, deprecated, by
# __getattr__ at the bottom.


# ── Helpers ───────────────────────────────────────────────────────────────────
#
# Scatter / time / pulse-width classification (is_scatter, is_excluded,
# default_cluster_channels, ...) lives in channel_roles; the names used here
# are imported at the top under their old names, the rest are served,
# deprecated, by __getattr__ at the bottom.


# ── FlowJo .wsp reader ────────────────────────────────────────────────────────
#
# The .wsp reader (WspReader, wsp_sample_events, wsp_ellipse_gate, the
# FlowJo axis floors and display axes, ...) lives in wsp_reader; the names
# used here are imported at the top under their old names, the rest are
# served, deprecated, by __getattr__ at the bottom.


# ── FMO gate thresholder ──────────────────────────────────────────────────────
#
# FMOGater lives in fmo_gater and is imported at the top, so pipeline.FMOGater
# is that class.

# ══════════════════════════════════════════════════════════════════════════════
# CORE SAMPLE CLASS
# ══════════════════════════════════════════════════════════════════════════════

# The single-sample class, FlowSample, lives in flow_sample and is imported
# at the top, so pipeline.FlowSample is that class. The public names it uses
# are still imported at the top, so pipeline.transform_values and the rest keep
# working; the private ones (_collect_warnings, _RangeFillingFlowData,
# _vram_free_gb, ...) are served, deprecated, through _MOVED at the bottom.
#
# Reading an events table from a CSV a user picked (read_events_csv) lives in
# events_csv; it is served, deprecated, by __getattr__ at the bottom.
# Reading past FCS writer quirks (read_fcs_tolerant, plain_fcs_problem) lives
# in fcs_quirks; the quirk table and its helpers are served, deprecated, by
# __getattr__ at the bottom.


# write_fcs (the write_fcs(df, path) wrapper over fcs_export.write_fcs) lives
# in fcs_write and is imported at the top, so pipeline.write_fcs is that
# function; the private alias _export_write_fcs is served, deprecated, through
# _MOVED at the bottom.


# ── Concatenation and cross-sample label alignment ────────────────────────────
#
# concatenate, concatenate_by_label and the antibody-label alignment
# (align_fluor_labels, relabel_gate_for_sample, common_fluor_warning, ...) live
# in label_alignment; concatenate is imported at the top, the rest are served,
# deprecated, by __getattr__ at the bottom.


# ── Experiment (multi-sample) ─────────────────────────────────────────────────
#
# FlowExperiment lives in flow_experiment and is imported at the top, so
# pipeline.FlowExperiment is that class.

# ══════════════════════════════════════════════════════════════════════════════
# LAZY IMPORTS  (PEP 562)
# ══════════════════════════════════════════════════════════════════════════════
#
# ``seaborn``, ``gaussian_kde``, ``phenograph`` and ``matplotlib.pyplot`` are
# not imported at module level. The hook below serves them to OTHER modules
# (``pipeline.sns``) on first access; a function inside this module imports
# what it needs locally, because a bare name here never reaches the hook
# (see the note at the top).
#
# Phenograph's one call site, FlowSample.cluster(), imports it locally.

_LAZY = {
    'sns':           ('seaborn',          None),
    'gaussian_kde':  ('scipy.stats',      'gaussian_kde'),
    'phenograph':    ('phenograph',       None),
    'plt':           ('matplotlib.pyplot', None),
}


# Private aliases of public helpers compare.py shares, still served, with a
# DeprecationWarning, by __getattr__ below. (The aliases of the .wsp helpers
# that moved to wsp_reader are served through _MOVED.)
_RENAMED = {'_WSP_BOOLEAN_OPS': 'WSP_BOOLEAN_OPS'}

# Names that moved to another module, possibly under a new name, and that
# this module no longer uses itself: old name -> (module, new name). Still
# served here, with a DeprecationWarning, by __getattr__ below. (The moved
# names pipeline does use are imported at the top under their old names.)
_MOVED = {'UNKNOWN_SPEC': ('channel_transforms', 'UNKNOWN_SPEC'),
          '_SPEC_PARAMS': ('channel_transforms', '_SPEC_PARAMS'),
          '_biexp_lut': ('channel_transforms', '_biexp_lut'),
          '_probe_gpu': ('gpu_probe', 'probe_gpu'),
          '_strict_increasing': ('cytonorm', '_strict_increasing'),
          'CytoNorm': ('cytonorm', 'CytoNorm'),
          'cap_events_per_batch': ('cytonorm', 'cap_events_per_batch'),
          'flowsom_layout': ('flowsom', 'flowsom_layout'),
          'flowsom_mst': ('flowsom', 'flowsom_mst'),
          'TRANSFORM_METHODS': ('channel_transforms', 'TRANSFORM_METHODS'),
          'unknown_spec': ('channel_transforms', 'unknown_spec'),
          'TRANSFORMS_FORMAT': ('transform_sidecar', 'TRANSFORMS_FORMAT'),
          'TRANSFORMS_VERSION': ('transform_sidecar', 'TRANSFORMS_VERSION'),
          '_LOGICLE_HEADROOM_DECADES': ('transform_sidecar', '_LOGICLE_HEADROOM_DECADES'),
          'transforms_sidecar_path': ('transform_sidecar', 'transforms_sidecar_path'),
          'recorded_sidecar_rows': ('transform_sidecar', 'recorded_sidecar_rows'),
          'read_transforms_sidecar': ('transform_sidecar', 'read_transforms_sidecar'),
          'transform_record_from_json': ('transform_sidecar', 'transform_record_from_json'),
          'logicle_display_bounds': ('transform_sidecar', 'logicle_display_bounds'),
          'gate_value_text': ('gate_text', 'gate_value_text'),
          '_gate_axis_text': ('gate_text', '_gate_axis_text'),
          'WSP_NO_GEOMETRY': ('gate_text', 'WSP_NO_GEOMETRY'),
          'wsp_unexportable': ('gate_text', 'wsp_unexportable'),
          'wsp_dependent_name': ('gate_text', 'wsp_dependent_name'),
          '_describe_bounds': ('gate_text', '_describe_bounds'),
          '_otsu_threshold': ('auto_gating', '_otsu_threshold'),
          'auto_threshold': ('auto_gating', 'auto_threshold'),
          '_polygon_area': ('auto_gating', '_polygon_area'),
          'auto_polygon_gate': ('auto_gating', 'auto_polygon_gate'),
          'auto_singlet_gate': ('auto_gating', 'auto_singlet_gate'),
          'gmm_ellipse_gates': ('auto_gating', 'gmm_ellipse_gates'),
          '_clock_seconds': ('fcs_text', '_clock_seconds'),
          '_FCS_QUIRKS': ('fcs_quirks', '_FCS_QUIRKS'),
          '_fcs_quirk': ('fcs_quirks', '_fcs_quirk'),
          '_quirk_note': ('fcs_quirks', '_quirk_note'),
          '_fcs_data_end_discrepancy': ('fcs_quirks', '_fcs_data_end_discrepancy'),
          'struct_error_types': ('fcs_quirks', 'struct_error_types'),
          'EXCLUDE_CLUSTER': ('channel_roles', 'EXCLUDE_CLUSTER'),
          '_SCATTER_RE': ('channel_roles', '_SCATTER_RE'),
          '_TIME_RE': ('channel_roles', '_TIME_RE'),
          '_INSTRUMENT_PARAMS': ('channel_roles', '_INSTRUMENT_PARAMS'),
          '_is_time': ('channel_roles', '_is_time'),
          '_PULSE_RE': ('channel_roles', '_PULSE_RE'),
          'DNA_DYES': ('cell_cycle', 'DNA_DYES'),
          'CELL_CYCLE_PHASES': ('cell_cycle', 'CELL_CYCLE_PHASES'),
          'robust_sd': ('robust', 'robust_sd'),
          '_wsp_time_gain': ('fcs_text', 'wsp_time_gain'),
          'wsp_sample_events': ('wsp_reader', 'wsp_sample_events'),
          'fcs_event_total': ('wsp_reader', 'fcs_event_total'),
          'wsp_event_mismatch': ('wsp_reader', 'wsp_event_mismatch'),
          '_DTD_SUBSET': ('wsp_reader', '_DTD_SUBSET'),
          'xml_declares_entities': ('wsp_reader', 'xml_declares_entities'),
          '_pre_flowjo10_version': ('wsp_reader', '_pre_flowjo10_version'),
          '_finite_bound': ('wsp_reader', '_finite_bound'),
          '_wsp_uri_path': ('wsp_reader', '_wsp_uri_path'),
          'wsp_written_by_openflo': ('wsp_reader', 'wsp_written_by_openflo'),
          '_ELLIPSE_VERTICES': ('wsp_reader', '_ELLIPSE_VERTICES'),
          '_LINEAR_FLOORS_MEASURED': ('wsp_reader', '_LINEAR_FLOORS_MEASURED'),
          'wsp_axis_floors': ('wsp_reader', 'wsp_axis_floors'),
          'wsp_axis_floor_lo': ('wsp_reader', 'wsp_axis_floor_lo'),
          '_FLOWJO_DISPLAY_BINS': ('wsp_reader', '_FLOWJO_DISPLAY_BINS'),
          '_FLOWJO_BIEX_CHANNELS': ('wsp_reader', '_FLOWJO_BIEX_CHANNELS'),
          '_flowjo_log_root': ('wsp_reader', '_flowjo_log_root'),
          '_wsp_axis_elem': ('wsp_reader', '_wsp_axis_elem'),
          '_flowjo_display_axis': ('wsp_reader', '_flowjo_display_axis'),
          'wsp_polygon_edge_axes': ('wsp_reader', 'wsp_polygon_edge_axes'),
          '_flowjo_ellipse_points': ('wsp_reader', '_flowjo_ellipse_points'),
          '_flowjo_ellipse_display': ('wsp_reader', '_flowjo_ellipse_display'),
          '_flowjo_native_ellipse': ('wsp_reader', '_flowjo_native_ellipse'),
          '_open_polygon_at': ('wsp_reader', 'open_polygon_at'),
          '_clip_polygon_axis': ('wsp_reader', '_clip_polygon_axis'),
          'wsp_comp_channels': ('wsp_reader', 'wsp_comp_channels'),
          '_unread_tags': ('wsp_reader', '_unread_tags'),
          '_log_unread': ('wsp_reader', '_log_unread'),
          'wsp_ellipse_gate': ('wsp_reader', 'wsp_ellipse_gate'),
          '_wsp_axis_floor_lo': ('wsp_reader', 'wsp_axis_floor_lo'),
          '_wsp_axis_floors': ('wsp_reader', 'wsp_axis_floors'),
          '_wsp_comp_channels': ('wsp_reader', 'wsp_comp_channels'),
          '_wsp_ellipse_gate': ('wsp_reader', 'wsp_ellipse_gate'),
          '_wsp_polygon_edge_axes': ('wsp_reader', 'wsp_polygon_edge_axes'),
          '_WSP_NS': ('wsp_writer', '_WSP_NS'),
          '_q': ('wsp_writer', '_q'),
          '_wsp_dataset_uri': ('wsp_writer', '_wsp_dataset_uri'),
          '_emit_flowjo_element': ('wsp_writer', '_emit_flowjo_element'),
          '_LINEAR_OPENED_UPTO': ('wsp_writer', '_LINEAR_OPENED_UPTO'),
          '_LINEAR_KEPT_FROM': ('wsp_writer', '_LINEAR_KEPT_FROM'),
          '_POLYGON_LINEAR_OPENED_UPTO': ('wsp_writer', '_POLYGON_LINEAR_OPENED_UPTO'),
          '_POLYGON_LINEAR_KEPT_FROM': ('wsp_writer', '_POLYGON_LINEAR_KEPT_FROM'),
          '_OPEN_BOUND': ('wsp_writer', '_OPEN_BOUND'),
          '_bound_attrs': ('wsp_writer', '_bound_attrs'),
          '_time_upper_edge': ('wsp_writer', '_time_upper_edge'),
          '_touches_time': ('wsp_writer', '_touches_time'),
          '_flowjo_axis_is_linear': ('wsp_writer', '_flowjo_axis_is_linear'),
          '_fcs_display': ('wsp_writer', '_fcs_display'),
          '_edges_flowjo_opens': ('wsp_writer', '_edges_flowjo_opens'),
          '_now_for_wsp': ('wsp_writer', '_now_for_wsp'),
          '_POLYGON_EDGE_STEPS': ('wsp_gate_coords', 'POLYGON_EDGE_STEPS'),
          'map_gate_coords': ('wsp_gate_coords', 'map_gate_coords'),
          '_trace_edges_on_axes': ('wsp_gate_coords', '_trace_edges_on_axes'),
          'gate_scale_fns': ('wsp_gate_coords', 'gate_scale_fns'),
          'gate_rescale_fns': ('wsp_gate_coords', 'gate_rescale_fns'),
          'rank_matched_fn': ('wsp_gate_coords', 'rank_matched_fn'),
          'mark_unknown_scale': ('wsp_gate_coords', 'mark_unknown_scale'),
          '_EDGE_AXIS_STEPS': ('wsp_gate_coords', '_EDGE_AXIS_STEPS'),
          'stat_axes_from_json': ('wsp_gate_coords', 'stat_axes_from_json'),
          '_edge_axis_fns': ('wsp_gate_coords', 'edge_axis_fns'),
          'wsp_uncompensated_channels': ('wsp_gate_coords', 'wsp_uncompensated_channels'),
          'wsp_stat_axes': ('wsp_gate_coords', 'wsp_stat_axes'),
          '_polygon_edge_fns': ('wsp_gate_coords', 'polygon_edge_fns'),
          '_SPILL_DIAG_TOL': ('compensation_io', '_SPILL_DIAG_TOL'),
          '_spill_channel_refs': ('compensation_io', '_spill_channel_refs'),
          '_read_matrix_csv': ('compensation_io', '_read_matrix_csv'),
          'read_compensation_matrix': ('compensation_io', 'read_compensation_matrix'),
          'write_compensation_matrix': ('compensation_io', 'write_compensation_matrix'),
          'optimize_compensation': ('compensation_optimizer', 'optimize_compensation'),
          '_template_ids': ('gate_templates', 'template_ids'),
          '_wsp_template_node': ('gate_templates', '_wsp_template_node'),
          'wsp_template_stat_axes': ('gate_templates', 'wsp_template_stat_axes'),
          'read_template_gates': ('gate_templates', 'read_template_gates'),
          'transform_params': ('channel_transforms', 'transform_params'),
          '_fcs_text_pairs': ('fcs_text', 'fcs_text_pairs'),
          '_fcs_timestep': ('fcs_text', 'fcs_timestep'),
          '_positive_float': ('fcs_text', 'positive_float'),
          'scale_time_coords': ('fcs_text', 'scale_time_coords'),
          '_scale_time_coords': ('fcs_text', 'scale_time_coords'),
          '_xml_safe': ('fcs_text', 'xml_safe'),
          '_WSP_BOOLEAN_TAGS': ('gate_text', 'WSP_BOOLEAN_TAGS'),
          'unique_population_name': ('gate_text', 'unique_population_name'),
          'wsp_population_name': ('gate_text', 'wsp_population_name'),
          '_BIEX_KEPT_FROM': ('wsp_reader', 'BIEX_KEPT_FROM'),
          '_BIEX_OPENED_UPTO': ('wsp_reader', 'BIEX_OPENED_UPTO'),
          'WSP_FLOWJO_VERSION': ('wsp_reader', 'WSP_FLOWJO_VERSION'),
          'WSP_OPENFLO_ATTR': ('wsp_reader', 'WSP_OPENFLO_ATTR'),
          '_ellipse_polygon': ('wsp_reader', 'ellipse_polygon'),
          '_flowjo_biex_lut': ('wsp_reader', 'flowjo_biex_lut'),
          '_flowjo_edge_axis': ('wsp_reader', 'flowjo_edge_axis'),
          '_strip_comp_marker': ('wsp_reader', 'strip_comp_marker'),
          'rests_on_broken_boolean': ('gate_eval', 'rests_on_broken_boolean'),
          'broken_booleans': ('gate_eval', 'broken_booleans'),
          '_own_unknown_reason': ('gate_eval', '_own_unknown_reason'),
          'unknown_population_reason': ('gate_eval', 'unknown_population_reason'),
          '_measured_in': ('gate_eval', '_measured_in'),
          '_GEOMETRIC_CHANNEL_KEYS': ('gate_eval', '_GEOMETRIC_CHANNEL_KEYS'),
          'missing_gate_channels': ('gate_eval', 'missing_gate_channels'),
          '_channel_missing': ('gate_eval', '_channel_missing'),
          'cumulative_gate_mask': ('gate_eval', 'cumulative_gate_mask'),
          '_positive_definite': ('gate_eval', '_positive_definite'),
          'AUTOCLEAN_METHODS': ('autoclean', 'AUTOCLEAN_METHODS'),
          'default_autoclean_methods': ('autoclean', 'default_autoclean_methods'),
          'autoclean_methods_signature': ('autoclean', 'autoclean_methods_signature'),
          'VIABILITY_DYES': ('autoclean', 'VIABILITY_DYES'),
          'find_viability_channel': ('autoclean', 'find_viability_channel'),
          '_debris_mode': ('autoclean', 'debris_mode'),
          '_autoclean_debris_mask': ('autoclean', '_autoclean_debris_mask'),
          '_VALLEY_MIN_PER_BIN': ('autoclean', '_VALLEY_MIN_PER_BIN'),
          '_bimodal_valley': ('autoclean', '_bimodal_valley'),
          '_DEBRIS_MAX_FSC_RATIO': ('autoclean', '_DEBRIS_MAX_FSC_RATIO'),
          '_debris_fsc_valley': ('autoclean', '_debris_fsc_valley'),
          '_debris_fsc_cut': ('autoclean', '_debris_fsc_cut'),
          '_autoclean_viability_mask': ('autoclean', '_autoclean_viability_mask'),
          '_DOUBLET_SIGMA_K': ('autoclean', '_DOUBLET_SIGMA_K'),
          '_autoclean_doublets_mask': ('autoclean', '_autoclean_doublets_mask'),
          'autoclean_debris_threshold': ('autoclean', 'autoclean_debris_threshold'),
          'autoclean_debris_freeze': ('autoclean', 'autoclean_debris_freeze'),
          'autoclean_viability_threshold': ('autoclean', 'autoclean_viability_threshold'),
          'freeze_autoclean_gate': ('autoclean', 'freeze_autoclean_gate'),
          'autoclean_method_diagnostic': ('autoclean', 'autoclean_method_diagnostic'),
          'autoclean_keep_mask': ('autoclean', 'autoclean_keep_mask'),
          '_is_number': ('compensation_io', 'is_number'),
          'read_events_csv': ('events_csv', 'read_events_csv'),
          '_sample_fluor_labels': ('label_alignment', 'sample_fluor_labels'),
          'align_fluor_labels': ('label_alignment', 'align_fluor_labels'),
          'relabel_gate_for_sample': ('label_alignment', 'relabel_gate_for_sample'),
          'common_fluor_warning': ('label_alignment', 'common_fluor_warning'),
          'concatenate_by_label': ('label_alignment', 'concatenate_by_label'),
          '_PG_SORT_JOBS': ('cluster_graph', '_PG_SORT_JOBS'),
          '_PG_SORT_INSTALL_LOCK': ('cluster_graph', '_PG_SORT_INSTALL_LOCK'),
          '_sort_by_size_serial': ('cluster_graph', 'sort_by_size_serial'),
          '_install_phenograph_sort_cap': ('cluster_graph', 'install_phenograph_sort_cap'),
          'resolve_seed': ('seeds', 'resolve_seed'),
          'GATE_PALETTE': ('cluster_labels', 'GATE_PALETTE'),
          'NOISE_CLUSTER_ID': ('cluster_labels', 'NOISE_CLUSTER_ID'),
          'NOISE_CLUSTER_LABEL': ('cluster_labels', 'NOISE_CLUSTER_LABEL'),
          'NOISE_CLUSTER_COLOR': ('cluster_labels', 'NOISE_CLUSTER_COLOR'),
          '_is_excluded': ('channel_roles', 'is_excluded'),
          '_is_scatter': ('channel_roles', 'is_scatter'),
          '_phenograph_sort_jobs': ('cluster_graph', 'phenograph_sort_jobs'),
          '_snn_jaccard_graph': ('cluster_graph', 'snn_jaccard_graph'),
          '_fcs_param_markers': ('compensation_io', 'fcs_param_markers'),
          '_fcs_param_names': ('compensation_io', 'fcs_param_names'),
          '_parse_spill_keyword': ('compensation_io', 'parse_spill_keyword'),
          '_spill_text': ('compensation_io', 'spill_text'),
          '_spill_units_to_fractions': ('compensation_io', 'spill_units_to_fractions'),
          '_matmul_known': ('compensation_math', 'matmul_known'),
          '_RangeFillingFlowData': ('fcs_range_fill', 'RangeFillingFlowData'),
          '_som_assign': ('flowsom', 'som_assign'),
          '_som_metacluster': ('flowsom', 'som_metacluster'),
          '_som_train': ('flowsom', 'som_train'),
          '_points_in_polygon': ('gate_eval', 'points_in_polygon'),
          '_CumlUMAP': ('gpu_probe', 'CUML_UMAP'),
          '_GPU_CLUSTER_KIT': ('gpu_probe', 'GPU_CLUSTER_KIT'),
          '_vram_free_gb': ('gpu_probe', 'vram_free_gb'),
          '_collect_warnings': ('log_capture', 'collect_warnings'),
          '_export_write_fcs': ('fcs_export', 'write_fcs')}


def __getattr__(name):
    if name == 'SCATTER_KEYWORDS':
        warnings.warn('SCATTER_KEYWORDS is deprecated and will be removed in '
                      'a future release', DeprecationWarning, stacklevel=2)
        return _SCATTER_KEYWORDS
    if name in _RENAMED:
        new = _RENAMED[name]
        warnings.warn(f'{name} is deprecated and will be removed in a future '
                      f'release; use {new}', DeprecationWarning, stacklevel=2)
        return globals()[new]
    if name in _MOVED:
        home, new = _MOVED[name]
        warnings.warn(f'pipeline.{name} is deprecated and will be removed in a '
                      f'future release; use openflo.{home}.{new}', DeprecationWarning,
                      stacklevel=2)
        import importlib
        return getattr(importlib.import_module(f'.{home}', __package__), new)
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = spec
    import importlib
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, attr) if attr else mod
    globals()[name] = obj      # cache so subsequent lookups skip the hook
    return obj
