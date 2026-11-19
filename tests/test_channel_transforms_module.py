"""The channel transforms live in channel_transforms, not pipeline.

transform_values, inverse_transform_values, the transform records
(transform_spec, unknown_spec, guessed_spec, ...) and linear_values were
defined in pipeline.py and moved, unchanged, into their own core module so
pipeline.py can be split up. These tests pin the move:

* every moved name lives in channel_transforms, and the ones pipeline still
  uses are the SAME objects there (imported, not copied), so a fix in one
  place reaches every caller;
* the moved names pipeline no longer uses still resolve as ``pipeline.X``,
  with a DeprecationWarning naming the new home;
* channel_transforms does not import pipeline, so a caller that needs only
  the transforms does not pay for the whole pipeline module;
* no module in the package imports a moved name from pipeline any more;
* the log transform's warning keeps its 'openflo.pipeline' logger name, so
  existing log filters still see it.
"""
import ast
import logging
import pathlib
import subprocess
import sys
import warnings

import numpy as np
import pytest

from openflo import channel_transforms, pipeline

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'src' / 'openflo'

# Moved names pipeline still uses itself: imported there from channel_transforms.
IMPORTED = (
    'UnknownScaleError', 'guessed_spec',
    'inverse_transform_values', 'is_guessed_spec', 'is_unknown_spec',
    'linear_values', 'to_linear', 'transform_spec',
    'transform_values',
)
# Moved names pipeline does not use: served by pipeline.__getattr__, deprecated.
# TRANSFORM_METHODS and unknown_spec joined them when the transform records
# moved to transform_sidecar, the only pipeline code that used them.
DEPRECATED = ('UNKNOWN_SPEC', '_SPEC_PARAMS', '_biexp_lut',
              'TRANSFORM_METHODS', 'unknown_spec', 'transform_params')
MOVED = IMPORTED + DEPRECATED


def test_every_moved_name_lives_in_channel_transforms():
    for name in MOVED:
        assert name in vars(channel_transforms), name


def test_pipeline_uses_the_same_objects_not_copies():
    tree = ast.parse((PACKAGE / 'pipeline.py').read_text(encoding='utf-8'))
    defined = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            defined.update(t.id for t in node.targets if isinstance(t, ast.Name))
    assert not defined & set(MOVED), sorted(defined & set(MOVED))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for name in IMPORTED:
            assert getattr(pipeline, name) is getattr(channel_transforms, name), name


@pytest.mark.parametrize('name', DEPRECATED)
def test_a_name_pipeline_no_longer_uses_still_works_with_a_warning(name):
    assert name not in vars(pipeline)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        obj = getattr(pipeline, name)
    assert obj is getattr(channel_transforms, name)
    hits = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert hits, f'pipeline.{name} gave no DeprecationWarning'
    assert f'use openflo.channel_transforms.{name}' in str(hits[0].message)


def test_importing_channel_transforms_does_not_import_pipeline():
    code = ('import sys, openflo.channel_transforms; '
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
                             for a in node.names if a.name in MOVED)
            elif (node.level and not mod) or mod == 'openflo':
                aliases.update(a.asname or a.name for a in node.names
                               if a.name == 'pipeline')
        elif isinstance(node, ast.Import):
            aliases.update(a.asname for a in node.names
                           if a.name == 'openflo.pipeline' and a.asname)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and node.attr in MOVED
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
    assert not hits, 'import these from .channel_transforms:\n' + '\n'.join(hits)


def test_the_log_warning_keeps_the_pipeline_logger_name(caplog):
    rng = np.random.default_rng(7)
    values = rng.normal(100.0, 300.0, 500)          # FL1-A-like, some negative
    assert (values <= 0).any()
    with caplog.at_level(logging.WARNING, logger='openflo.pipeline'):
        out = channel_transforms.transform_values(values, method='log')
    assert np.isnan(out[values <= 0]).all()
    records = [r for r in caplog.records if 'log transform' in r.getMessage()]
    assert records, 'no log-transform warning was logged'
    assert {r.name for r in records} == {'openflo.pipeline'}
