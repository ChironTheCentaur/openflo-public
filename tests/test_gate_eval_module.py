"""Gate evaluation lives in gate_eval and the auto-clean gate in autoclean,
not pipeline.

The gate-model and auto-clean blocks of pipeline.py (gate_to_mask,
cumulative_gate_mask, the unknown-population checks, _positive_definite, the
auto-clean recipe and its keep-masks) moved unchanged into the two modules,
which do not import pipeline. Seeds, the SNN graph and the cluster labels in
the same blocks stayed. These tests pin the move:

* neither new module imports pipeline (or Tk);
* the names pipeline still uses are the moved objects, without a warning;
* every other old name still resolves as ``pipeline.X`` to the same object,
  with a DeprecationWarning naming its home;
* the package's public API serves gate_to_mask and cumulative_gate_mask from
  gate_eval, without a warning;
* no module in the package imports a moved name from pipeline any more.
"""
import ast
import pathlib
import subprocess
import sys
import warnings

import pytest

import openflo
from openflo import autoclean, flow_sample, gate_eval, pipeline

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'openflo'

HOMES = {'gate_eval': gate_eval, 'autoclean': autoclean}
# Old names served by pipeline.__getattr__ (old name -> (module, new name)).
DEPRECATED = {old: home for old, home in pipeline._MOVED.items()
              if home[0] in HOMES}
# Moved names pipeline still uses, imported back under their old names.
KEPT = {'gate_to_mask': (gate_eval, 'gate_to_mask'),
        'depends_on_flowjo_placeholder': (gate_eval,
                                          'depends_on_flowjo_placeholder'),
        'find_scatter_channel': (autoclean, 'find_scatter_channel')}


def test_the_new_modules_do_not_import_pipeline_or_tk():
    code = ('import sys, openflo.gate_eval, openflo.autoclean; '
            "print('openflo.pipeline' in sys.modules, 'tkinter' in sys.modules)")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ['False', 'False'], out.stdout + out.stderr


def test_the_moved_set_is_the_whole_section():
    assert DEPRECATED == {
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
        'autoclean_methods_signature': ('autoclean',
                                        'autoclean_methods_signature'),
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
        'autoclean_viability_threshold': ('autoclean',
                                          'autoclean_viability_threshold'),
        'freeze_autoclean_gate': ('autoclean', 'freeze_autoclean_gate'),
        'autoclean_method_diagnostic': ('autoclean',
                                        'autoclean_method_diagnostic'),
        'autoclean_keep_mask': ('autoclean', 'autoclean_keep_mask'),
        # Used by FlowSample, which moved to flow_sample and imports it.
        '_points_in_polygon': ('gate_eval', 'points_in_polygon')}
    for old in DEPRECATED:
        assert old not in vars(pipeline), old


@pytest.mark.parametrize('old', sorted(KEPT))
def test_the_names_pipeline_uses_are_the_moved_objects(old):
    module, new = KEPT[old]
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert getattr(pipeline, old) is getattr(module, new)


def test_flow_sample_binds_the_polygon_test_it_uses():
    assert flow_sample._points_in_polygon is gate_eval.points_in_polygon


@pytest.mark.parametrize('old', sorted(DEPRECATED))
def test_the_old_pipeline_name_still_works_with_a_warning(old):
    home, new = pipeline._MOVED[old]
    with pytest.warns(DeprecationWarning, match=f'use openflo.{home}.{new}'):
        obj = getattr(pipeline, old)
    assert obj is getattr(HOMES[home], new)


@pytest.mark.parametrize('name', ['gate_to_mask', 'cumulative_gate_mask'])
def test_the_public_api_serves_the_new_home_without_a_warning(name):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert getattr(openflo, name) is getattr(gate_eval, name)


def test_no_module_imports_a_moved_name_from_pipeline():
    moved = set(DEPRECATED) | (set(KEPT) - {'_points_in_polygon'})
    hits = []
    for path in sorted(PACKAGE.rglob('*.py')):
        if path.name == 'pipeline.py':
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (
                    (node.level and node.module == 'pipeline')
                    or node.module == 'openflo.pipeline'):
                hits.extend(f'{path.name}:{node.lineno} {a.name}'
                            for a in node.names if a.name in moved)
    assert hits == []
