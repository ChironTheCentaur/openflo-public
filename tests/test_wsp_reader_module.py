"""The FlowJo .wsp reader lives in wsp_reader, the exception hierarchy in
exceptions, not pipeline.

The reader block of pipeline.py (WspReader and the gate-element helpers it and
openflo-compare share) moved unchanged into wsp_reader, and the exception
classes into exceptions, so the reader no longer needs pipeline. These tests
pin the move:

* neither new module imports pipeline (or Tk);
* the names pipeline still uses, and the public exceptions, are the SAME
  objects in pipeline (imported, not copied), under their old names;
* every other old name, and every earlier private alias, still resolves as
  ``pipeline.X`` to the moved object, with a DeprecationWarning naming the
  new home;
* no module in the package imports a moved name from pipeline any more.
"""
import ast
import pathlib
import subprocess
import sys
import warnings

import pytest

from openflo import exceptions, pipeline, wsp_reader

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'openflo'

EXCEPTIONS = ('OpenFloError', 'FcsParseError', 'CompensationError',
              'WspParseError', 'GateError', 'ClusteringError')
# Moved names pipeline still uses: imported there (old name -> new name).
IMPORTED = {'WspReader': 'WspReader'}
# Old names served by pipeline.__getattr__ (old name -> (module, new name)).
DEPRECATED = {old: home for old, home in pipeline._MOVED.items()
              if home[0] == 'wsp_reader'}
DEPRECATED['_wsp_time_gain'] = ('fcs_text', 'wsp_time_gain')


def test_the_new_modules_do_not_import_pipeline_or_tk():
    code = ('import sys, openflo.wsp_reader, openflo.exceptions; '
            "print('openflo.pipeline' in sys.modules, 'tkinter' in sys.modules)")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ['False', 'False'], out.stdout + out.stderr


def test_pipeline_uses_the_same_objects_not_copies():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for name in EXCEPTIONS:
            assert name in vars(pipeline), name
            assert getattr(pipeline, name) is getattr(exceptions, name), name
        for old, new in IMPORTED.items():
            assert old in vars(pipeline), old
            assert getattr(pipeline, old) is getattr(wsp_reader, new), old
    assert issubclass(wsp_reader.WspParseError, pipeline.OpenFloError)
    assert pipeline.WspReader.__module__ == 'openflo.wsp_reader'


def test_the_moved_set_is_the_whole_reader():
    # A slice that misses a name would leave it, or a copy, in pipeline.
    moved = {new for _home, new in DEPRECATED.values()} | set(IMPORTED.values())
    assert {'WspReader', 'wsp_sample_events', 'xml_declares_entities',
            'wsp_ellipse_gate', 'wsp_axis_floors', 'wsp_comp_channels',
            'open_polygon_at', '_finite_bound', '_wsp_uri_path'} <= moved
    for old in DEPRECATED:
        assert old not in vars(pipeline), old


@pytest.mark.parametrize('old', sorted(DEPRECATED))
def test_the_old_pipeline_name_still_works_with_a_warning(old):
    home, new = DEPRECATED[old]
    module = sys.modules[f'openflo.{home}']
    with pytest.warns(DeprecationWarning, match=f'use openflo.{home}.{new}'):
        obj = getattr(pipeline, old)
    assert obj is getattr(module, new)


@pytest.mark.parametrize('old', sorted(pipeline._RENAMED))
def test_the_remaining_renamed_aliases_still_work(old):
    with pytest.warns(DeprecationWarning, match=pipeline._RENAMED[old]):
        obj = getattr(pipeline, old)
    assert obj is vars(pipeline)[pipeline._RENAMED[old]]


def test_no_module_imports_a_moved_name_from_pipeline():
    moved = set(DEPRECATED)
    hits = []
    for path in sorted(PACKAGE.rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (
                    (node.level and node.module == 'pipeline')
                    or node.module == 'openflo.pipeline'):
                hits.extend(f'{path.name}:{node.lineno} {a.name}'
                            for a in node.names if a.name in moved)
    assert hits == []
