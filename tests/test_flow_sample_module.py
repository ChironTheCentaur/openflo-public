"""The single-sample class, FlowSample, and the categorical palette its plots
read (categorical_cmap, set_default_palette) live in flow_sample, not pipeline.

They moved unchanged and pipeline imports them back, so pipeline.FlowSample
and ``from openflo import FlowSample`` are the same class. These tests pin the
move:

* flow_sample does not import pipeline (or Tk);
* pipeline.FlowSample, categorical_cmap and set_default_palette are the moved
  objects, without a warning, and pipeline defines no copy of them;
* set_default_palette still steers the palette FlowSample's plots use;
* the public names FlowSample uses are still pipeline attributes, without a
  warning, and the private ones resolve through pipeline._MOVED with one;
* FlowSample still logs to the 'openflo.pipeline' logger.
"""
import ast
import importlib
import logging
import pathlib
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd

import openflo
from openflo import flow_sample, pipeline

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'openflo'
MOVED = {'FlowSample', 'categorical_cmap', 'set_default_palette',
         '_DEFAULT_CATEGORICAL_PALETTE'}

# Public names pipeline imported for FlowSample and no longer uses itself:
# still pipeline attributes, with no warning.
STILL_PUBLIC = ['AcquisitionQC', 'analyze_dna', 'assign_phase', 'find_dna_channel',
                'is_time_channel', 'default_cluster_channels', 'UnknownScaleError',
                'guessed_spec', 'is_guessed_spec', 'is_unknown_spec',
                'linear_values', 'transform_spec', 'transform_values',
                'ClusteringError', 'CompensationError', 'FcsParseError',
                'plain_fcs_problem', 'read_fcs_tolerant', 'igraph_seeded',
                'GPU_AVAILABLE', 'GPU_CLUSTERING_AVAILABLE', 'GPU_NAME',
                'tree_kneighbors', 'ASINH_DEFAULT_COFACTOR', 'LOGICLE_A',
                'LOGICLE_M', 'LOGICLE_T', 'LOGICLE_W',
                'discard_transforms_sidecar', 'write_transforms_sidecar',
                'WspReader']

# Private names pipeline imported for FlowSample: old name -> (home, name).
NOW_DEPRECATED = {
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
}


def test_flow_sample_does_not_import_pipeline_or_tk():
    code = ('import sys, openflo.flow_sample; '
            "print('openflo.pipeline' in sys.modules, 'tkinter' in sys.modules)")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ['False', 'False'], out.stdout + out.stderr


def test_pipeline_binds_the_moved_objects_and_defines_no_copy():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert pipeline.FlowSample is flow_sample.FlowSample
        assert pipeline.categorical_cmap is flow_sample.categorical_cmap
        assert pipeline.set_default_palette is flow_sample.set_default_palette
        assert openflo.FlowSample is flow_sample.FlowSample
    assert flow_sample.FlowSample.__module__ == 'openflo.flow_sample'
    tree = ast.parse((PACKAGE / 'pipeline.py').read_text(encoding='utf-8'))
    defined = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            defined.update(t.id for t in node.targets if isinstance(t, ast.Name))
    assert defined & MOVED == set()


def test_set_default_palette_still_steers_the_plots_palette():
    before = flow_sample._DEFAULT_CATEGORICAL_PALETTE
    try:
        pipeline.set_default_palette('Set2')
        assert flow_sample._DEFAULT_CATEGORICAL_PALETTE == 'Set2'
        pipeline.set_default_palette('')
        assert flow_sample._DEFAULT_CATEGORICAL_PALETTE == 'auto'
    finally:
        flow_sample._DEFAULT_CATEGORICAL_PALETTE = before


def test_public_names_stay_on_pipeline_without_a_warning():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for name in STILL_PUBLIC:
            assert name in vars(pipeline), name
            assert getattr(pipeline, name) is getattr(flow_sample, name), name


def test_private_names_resolve_with_a_warning_naming_their_home():
    for old, (home, new) in NOW_DEPRECATED.items():
        assert old not in vars(pipeline), old
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            obj = getattr(pipeline, old)
        target = getattr(importlib.import_module(f'openflo.{home}'), new)
        assert obj is target, old
        assert getattr(flow_sample, old) is target, old
        hits = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert hits and f'use openflo.{home}.{new}' in str(hits[0].message), old


def test_flow_sample_logs_to_the_pipeline_logger(caplog):
    rng = np.random.default_rng(7)
    df = pd.DataFrame({'FSC-A': rng.uniform(1e4, 2e5, 200),
                       'SSC-A': rng.uniform(1e4, 2e5, 200),
                       'FL1-A': rng.lognormal(6, 1, 200)})
    s = flow_sample.FlowSample.from_dataframe(df, name='synthetic_a')
    with caplog.at_level(logging.INFO, logger='openflo.pipeline'):
        s.apply_transform(['FL1-A'], method='asinh')
    assert flow_sample.log.name == 'openflo.pipeline'
    assert any(r.name == 'openflo.pipeline' for r in caplog.records)
