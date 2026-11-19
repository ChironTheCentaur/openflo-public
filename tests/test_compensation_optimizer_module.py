"""The compensation optimizer lives in compensation_optimizer and the gate
template readers in gate_templates, not pipeline.

The compensation-optimizer block of pipeline.py (optimize_compensation, then
the template readers read_template_gates, wsp_template_stat_axes and their
helpers) moved unchanged into the two modules, which do not import pipeline.
Pipeline used none of these names. These tests pin the move:

* neither new module imports pipeline (or Tk);
* every old name still resolves as ``pipeline.X`` to the same object, with a
  DeprecationWarning naming its home;
* the package's public API serves optimize_compensation and
  read_template_gates from their new homes, without a warning;
* no module in the package imports a moved name from pipeline any more.
"""
import ast
import pathlib
import subprocess
import sys
import warnings

import pytest

import openflo
from openflo import compensation_optimizer, gate_templates, pipeline

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'openflo'

HOMES = {'compensation_optimizer': compensation_optimizer,
         'gate_templates': gate_templates}
# Old names served by pipeline.__getattr__ (old name -> (module, new name)).
DEPRECATED = {old: home for old, home in pipeline._MOVED.items()
              if home[0] in HOMES}


def test_the_new_modules_do_not_import_pipeline_or_tk():
    code = ('import sys, openflo.compensation_optimizer, '
            'openflo.gate_templates; '
            "print('openflo.pipeline' in sys.modules, 'tkinter' in sys.modules)")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ['False', 'False'], out.stdout + out.stderr


def test_the_moved_set_is_the_whole_section():
    assert DEPRECATED == {
        'optimize_compensation': ('compensation_optimizer',
                                  'optimize_compensation'),
        '_template_ids': ('gate_templates', 'template_ids'),
        '_wsp_template_node': ('gate_templates', '_wsp_template_node'),
        'wsp_template_stat_axes': ('gate_templates', 'wsp_template_stat_axes'),
        'read_template_gates': ('gate_templates', 'read_template_gates')}
    for old in DEPRECATED:
        assert old not in vars(pipeline), old


@pytest.mark.parametrize('old', sorted(DEPRECATED))
def test_the_old_pipeline_name_still_works_with_a_warning(old):
    home, new = pipeline._MOVED[old]
    with pytest.warns(DeprecationWarning, match=f'use openflo.{home}.{new}'):
        obj = getattr(pipeline, old)
    assert obj is getattr(HOMES[home], new)


@pytest.mark.parametrize('name, module', [
    ('optimize_compensation', compensation_optimizer),
    ('read_template_gates', gate_templates)])
def test_the_public_api_serves_the_new_home_without_a_warning(name, module):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert getattr(openflo, name) is getattr(module, name)


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
