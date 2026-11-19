"""Compensation matrix files are read and written by compensation_io, not
pipeline.

The compensation-matrix IO block of pipeline.py (read_compensation_matrix,
write_compensation_matrix, the CSV reader and the SPILL keyword parser) moved
unchanged into compensation_io, which does not import pipeline. These tests
pin the move:

* the new module does not import pipeline (or Tk);
* the names FlowSample uses for SPILL parsing are the SAME objects in
  flow_sample (FlowSample's home since it left pipeline), under their old
  names;
* every other old name still resolves as ``pipeline.X`` to the same object,
  with a DeprecationWarning naming its home;
* the package's public API serves the readers from their new home, without a
  warning;
* no module in the package imports a moved name from pipeline any more.
"""
import ast
import pathlib
import subprocess
import sys
import warnings

import pytest

import openflo
from openflo import compensation_io, flow_sample, pipeline

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'openflo'

# Moved names FlowSample uses: imported into flow_sample (old name -> new name).
IMPORTED = {'_spill_units_to_fractions': 'spill_units_to_fractions',
            '_parse_spill_keyword': 'parse_spill_keyword',
            '_spill_text': 'spill_text',
            '_fcs_param_markers': 'fcs_param_markers',
            '_fcs_param_names': 'fcs_param_names'}
# Old names served by pipeline.__getattr__ (old name -> (module, new name)).
DEPRECATED = {old: home for old, home in pipeline._MOVED.items()
              if home[0] == 'compensation_io'}


def test_the_new_module_does_not_import_pipeline_or_tk():
    code = ('import sys, openflo.compensation_io; '
            "print('openflo.pipeline' in sys.modules, 'tkinter' in sys.modules)")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ['False', 'False'], out.stdout + out.stderr


def test_pipeline_uses_the_same_objects_not_copies():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for old, new in IMPORTED.items():
            assert old in vars(flow_sample), old
            assert getattr(flow_sample, old) is getattr(compensation_io, new), old
    assert flow_sample._parse_spill_keyword.__module__ == 'openflo.compensation_io'


def test_the_moved_set_is_the_whole_section():
    moved = {new for _home, new in DEPRECATED.values()} | set(IMPORTED.values())
    assert moved == {'is_number', '_SPILL_DIAG_TOL', 'spill_units_to_fractions',
                     '_spill_channel_refs', 'parse_spill_keyword', 'spill_text',
                     'fcs_param_markers', 'fcs_param_names', '_read_matrix_csv',
                     'read_compensation_matrix', 'write_compensation_matrix'}
    for old in DEPRECATED:
        assert old not in vars(pipeline), old


@pytest.mark.parametrize('old', sorted(DEPRECATED))
def test_the_old_pipeline_name_still_works_with_a_warning(old):
    home, new = pipeline._MOVED[old]
    with pytest.warns(DeprecationWarning, match=f'use openflo.{home}.{new}'):
        obj = getattr(pipeline, old)
    assert obj is getattr(compensation_io, new)


@pytest.mark.parametrize('name', ['read_compensation_matrix',
                                  'write_compensation_matrix'])
def test_the_public_api_serves_the_new_home_without_a_warning(name):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert getattr(openflo, name) is getattr(compensation_io, name)


def test_no_module_imports_a_moved_name_from_pipeline():
    hits = []
    for path in sorted(PACKAGE.rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (
                    (node.level and node.module == 'pipeline')
                    or node.module == 'openflo.pipeline'):
                hits.extend(f'{path.name}:{node.lineno} {a.name}'
                            for a in node.names if a.name in DEPRECATED)
    assert hits == []
