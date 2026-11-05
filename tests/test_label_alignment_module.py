"""Reading an events CSV lives in events_csv, and concatenation and the
cross-sample label alignment in label_alignment, not pipeline.

read_events_csv, concatenate and the cross-sample label alignment block of
pipeline.py (sample_fluor_labels, align_fluor_labels, relabel_gate_for_sample,
common_fluor_warning, concatenate_by_label) moved unchanged into the two
modules, which do not import pipeline. FlowSample, FMOGater and write_fcs in
the same blocks stayed. These tests pin the move:

* neither new module imports pipeline (or Tk);
* concatenate, which FlowExperiment uses, is the moved object, without a
  warning;
* every other old name still resolves as ``pipeline.X`` to the same object,
  with a DeprecationWarning naming its home;
* the package's public API serves concatenate from label_alignment, without a
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
from openflo import compensation_io, events_csv, label_alignment, pipeline

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'openflo'

HOMES = {'events_csv': events_csv, 'label_alignment': label_alignment,
         'compensation_io': compensation_io}
# Old names served by pipeline.__getattr__ (old name -> (module, new name)).
DEPRECATED = {old: home for old, home in pipeline._MOVED.items()
              if home[0] in ('events_csv', 'label_alignment')}


def test_the_new_modules_do_not_import_pipeline_or_tk():
    code = ('import sys, openflo.events_csv, openflo.label_alignment; '
            "print('openflo.pipeline' in sys.modules, 'tkinter' in sys.modules)")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ['False', 'False'], out.stdout + out.stderr


def test_the_moved_set_is_the_whole_section():
    assert DEPRECATED == {
        'read_events_csv': ('events_csv', 'read_events_csv'),
        '_sample_fluor_labels': ('label_alignment', 'sample_fluor_labels'),
        'align_fluor_labels': ('label_alignment', 'align_fluor_labels'),
        'relabel_gate_for_sample': ('label_alignment',
                                    'relabel_gate_for_sample'),
        'common_fluor_warning': ('label_alignment', 'common_fluor_warning'),
        'concatenate_by_label': ('label_alignment', 'concatenate_by_label')}
    for old in [*DEPRECATED, '_is_number']:
        assert old not in vars(pipeline), old


def test_concatenate_is_the_moved_object():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert pipeline.concatenate is label_alignment.concatenate


@pytest.mark.parametrize('old', sorted([*DEPRECATED, '_is_number']))
def test_the_old_pipeline_name_still_works_with_a_warning(old):
    home, new = pipeline._MOVED[old]
    with pytest.warns(DeprecationWarning, match=f'use openflo.{home}.{new}'):
        obj = getattr(pipeline, old)
    assert obj is getattr(HOMES[home], new)


def test_the_public_api_serves_the_new_home_without_a_warning():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert openflo.concatenate is label_alignment.concatenate


def test_no_module_imports_a_moved_name_from_pipeline():
    moved = {*DEPRECATED, '_is_number', 'concatenate'}
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
