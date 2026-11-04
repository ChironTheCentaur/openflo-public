"""The transform records beside a CSV live in transform_sidecar, the gate text
and FlowJo population naming in gate_text, not pipeline.

The section of pipeline.py that wrote and read the ``<stem>_transforms.json``
record beside an event CSV, and described gates as text and named their
FlowJo populations, moved unchanged into two core modules so pipeline.py can
be split up (the boolean-tag map is public there as WSP_BOOLEAN_TAGS; pipeline
binds it under its old private name). These tests pin the move:

* every moved name lives in its new module, and the ones pipeline still uses
  are the SAME objects there (imported, not copied);
* the names pipeline no longer uses still resolve as ``pipeline.X``, to the
  same objects, with a DeprecationWarning naming the new home; so do
  TRANSFORM_METHODS and unknown_spec, which only the moved code used;
* the package-level ``openflo.describe_gate`` comes from gate_text;
* neither module imports pipeline;
* no module in the package imports a moved name from pipeline any more, and
  pipeline defines no copy of one.
"""
import ast
import pathlib
import subprocess
import sys
import warnings

import pytest

import openflo
from openflo import channel_transforms, gate_text, pipeline, transform_sidecar

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'src' / 'openflo'

SIDECAR = ('TRANSFORMS_FORMAT', 'TRANSFORMS_VERSION', '_LOGICLE_HEADROOM_DECADES',
           'transforms_sidecar_path', 'write_transforms_sidecar',
           'discard_transforms_sidecar', 'recorded_sidecar_rows',
           'read_transforms_sidecar', 'transform_record_from_json',
           'logicle_display_bounds')
GATE_TEXT = ('gate_value_text', '_gate_axis_text', 'wsp_population_name',
             'WSP_BOOLEAN_TAGS', 'WSP_BOOLEAN_OPS', 'WSP_NO_GEOMETRY',
             'wsp_unexportable', 'wsp_dependent_name', 'resolve_wsp_dependent',
             'unique_population_name', 'describe_gate', '_describe_bounds')
HOME = {**dict.fromkeys(SIDECAR, transform_sidecar),
        **dict.fromkeys(GATE_TEXT, gate_text)}

# Moved names pipeline still uses: imported there (old name -> new name).
IMPORTED = {'write_transforms_sidecar': 'write_transforms_sidecar',
            'discard_transforms_sidecar': 'discard_transforms_sidecar',
            'WSP_BOOLEAN_OPS': 'WSP_BOOLEAN_OPS',
            'resolve_wsp_dependent': 'resolve_wsp_dependent',
            'describe_gate': 'describe_gate'}
# Old names pipeline does not use: served by pipeline.__getattr__, deprecated.
# (WSP_BOOLEAN_TAGS was only ever pipeline._WSP_BOOLEAN_TAGS, served the same
# way; see test_wsp_writer_module.)
DEPRECATED = tuple(n for n in HOME
                   if n not in IMPORTED.values() and n != 'WSP_BOOLEAN_TAGS')
# Imported by pipeline from channel_transforms for the moved code only.
UNUSED_NOW = ('TRANSFORM_METHODS', 'unknown_spec')
MOVED = tuple(IMPORTED) + DEPRECATED + ('WSP_BOOLEAN_TAGS',)


def test_every_moved_name_lives_in_its_new_module():
    for name, mod in HOME.items():
        assert name in vars(mod), name


def test_pipeline_uses_the_same_objects_not_copies():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for old, new in IMPORTED.items():
            assert old in vars(pipeline), old
            assert getattr(pipeline, old) is getattr(HOME[new], new), old


@pytest.mark.parametrize('name', DEPRECATED)
def test_the_old_pipeline_name_still_works_with_a_warning(name):
    assert name not in vars(pipeline)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        obj = getattr(pipeline, name)
    assert obj is getattr(HOME[name], name)
    hits = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert hits, f'pipeline.{name} gave no DeprecationWarning'
    assert f'use openflo.{HOME[name].__name__.rsplit(".", 1)[-1]}.{name}' in str(
        hits[0].message)


@pytest.mark.parametrize('name', UNUSED_NOW)
def test_a_transform_name_only_the_moved_code_used_warns(name):
    assert name not in vars(pipeline)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        obj = getattr(pipeline, name)
    assert obj is getattr(channel_transforms, name)
    assert any(f'use openflo.channel_transforms.{name}' in str(w.message)
               for w in caught if issubclass(w.category, DeprecationWarning))


def test_the_package_serves_describe_gate_from_gate_text():
    assert openflo._PUBLIC['describe_gate'] == 'openflo.gate_text'
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert openflo.describe_gate is gate_text.describe_gate


@pytest.mark.parametrize('module', ['transform_sidecar', 'gate_text'])
def test_importing_the_module_does_not_import_pipeline(module):
    code = (f"import sys, openflo.{module}; "
            "print('openflo.pipeline' in sys.modules)")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, check=True, timeout=120)
    assert out.stdout.strip() == 'False', out.stdout + out.stderr


def _imports_of_moved_names_from_pipeline(tree):
    """``from .pipeline import X`` / ``from openflo.pipeline import X`` and
    ``<pipeline alias>.X`` for a moved X, as 'line: text' strings."""
    aliases = set()
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ''
            if (node.level and mod == 'pipeline') or mod == 'openflo.pipeline':
                found.extend(f'{node.lineno}: from {mod} import {a.name}'
                             for a in node.names if a.name in MOVED + UNUSED_NOW)
            elif (node.level and not mod) or mod == 'openflo':
                aliases.update(a.asname or a.name for a in node.names
                               if a.name == 'pipeline')
        elif isinstance(node, ast.Import):
            aliases.update(a.asname for a in node.names
                           if a.name == 'openflo.pipeline' and a.asname)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and node.attr in MOVED + UNUSED_NOW
                and isinstance(node.value, ast.Name) and node.value.id in aliases):
            found.append(f'{node.lineno}: {node.value.id}.{node.attr}')
    return found


def test_no_module_imports_a_moved_name_from_pipeline():
    hits = []
    for path in sorted(PACKAGE.rglob('*.py')):
        if path == PACKAGE / 'pipeline.py':
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        hits.extend(f'{path.relative_to(ROOT).as_posix()}:{h}'
                    for h in _imports_of_moved_names_from_pipeline(tree))
    assert not hits, ('import these from .transform_sidecar / .gate_text:\n'
                      + '\n'.join(hits))


def test_pipeline_defines_no_copy_of_a_moved_name():
    tree = ast.parse((PACKAGE / 'pipeline.py').read_text(encoding='utf-8'))
    defined = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                elts = target.elts if isinstance(target, ast.Tuple) else [target]
                defined.update(t.id for t in elts if isinstance(t, ast.Name))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            if isinstance(node.target, ast.Name):
                defined.add(node.target.id)
    assert not defined & set(MOVED), sorted(defined & set(MOVED))


def test_a_sidecar_round_trips_through_the_new_module(tmp_path):
    csv = tmp_path / 'synthetic_events.csv'
    csv.write_text('FL1-A,FL2-A\n1,2\n', encoding='utf-8')
    tf = {'FL1-A': {'method': 'asinh', 'cofactor': 150.0}}
    path = transform_sidecar.write_transforms_sidecar(
        str(csv), tf, ['FL1-A', 'FL2-A'], 1)
    assert path == transform_sidecar.transforms_sidecar_path(str(csv))
    back = transform_sidecar.read_transforms_sidecar(str(csv), ['FL1-A', 'FL2-A'], 1)
    assert back == {'FL1-A': channel_transforms.transform_spec('asinh', cofactor=150.0)}
    assert transform_sidecar.recorded_sidecar_rows(str(csv)) == 1
    transform_sidecar.discard_transforms_sidecar(str(csv))
    assert transform_sidecar.read_transforms_sidecar(str(csv), ['FL1-A', 'FL2-A'], 1) is None
