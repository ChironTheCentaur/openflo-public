"""CytoNorm lives in cytonorm, not pipeline.

The CytoNorm model, cap_events_per_batch and the spline-knot helper
_strict_increasing were defined in pipeline.py and moved, unchanged apart from
calling the FlowSOM helpers by their public flowsom names, into their own core
module so pipeline.py can be split up. These tests pin the move:

* every moved name lives in cytonorm, and CytoNorm calls flowsom's SOM
  helpers (the same objects pipeline uses);
* pipeline no longer uses any of them, so ``pipeline.CytoNorm`` and the rest
  still resolve, to the SAME objects, with a DeprecationWarning naming
  ``openflo.cytonorm``;
* the package-level ``openflo.CytoNorm`` comes from cytonorm without one;
* cytonorm does not import pipeline;
* no module in the package imports a moved name from pipeline any more, and
  pipeline defines no copy of one;
* CytoNorm's QC warning keeps its 'openflo.pipeline' logger name, so existing
  log filters still see it.
"""
import ast
import logging
import pathlib
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

import openflo
from openflo import cytonorm, flowsom, pipeline

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'src' / 'openflo'

# Old names pipeline does not use: served by pipeline.__getattr__, deprecated.
# Each keeps its name in cytonorm.
DEPRECATED = ('CytoNorm', 'cap_events_per_batch', '_strict_increasing')
# The FlowSOM helpers CytoNorm calls, by the name cytonorm binds them under.
SOM_HELPERS = ('som_train', 'som_assign', 'som_metacluster')
CHANNELS = ['FL1-A', 'FL2-A']


def test_every_moved_name_lives_in_cytonorm():
    for name in DEPRECATED:
        assert name in vars(cytonorm), name


def test_cytonorm_calls_the_flowsom_helpers_pipeline_uses():
    for name in SOM_HELPERS:
        assert getattr(cytonorm, name) is getattr(flowsom, name), name
        with pytest.warns(DeprecationWarning, match=f'flowsom.{name}'):
            alias = getattr(pipeline, f'_{name}')
        assert alias is getattr(cytonorm, name), name


@pytest.mark.parametrize('name', DEPRECATED)
def test_the_old_pipeline_name_still_works_with_a_warning(name):
    assert name not in vars(pipeline)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        obj = getattr(pipeline, name)
    assert obj is getattr(cytonorm, name)
    hits = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert hits, f'pipeline.{name} gave no DeprecationWarning'
    assert f'use openflo.cytonorm.{name}' in str(hits[0].message)


def test_the_package_serves_cytonorm_from_cytonorm():
    assert openflo._PUBLIC['CytoNorm'] == 'openflo.cytonorm'
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert openflo.CytoNorm is cytonorm.CytoNorm


def test_importing_cytonorm_does_not_import_pipeline():
    code = ("import sys, openflo.cytonorm; "
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
                             for a in node.names if a.name in DEPRECATED)
            elif (node.level and not mod) or mod == 'openflo':
                aliases.update(a.asname or a.name for a in node.names
                               if a.name == 'pipeline')
        elif isinstance(node, ast.Import):
            aliases.update(a.asname for a in node.names
                           if a.name == 'openflo.pipeline' and a.asname)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and node.attr in DEPRECATED
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
    assert not hits, 'import these from .cytonorm:\n' + '\n'.join(hits)


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
    assert not defined & set(DEPRECATED), sorted(defined & set(DEPRECATED))


def test_the_qc_warning_keeps_the_pipeline_logger_name(caplog):
    rng = np.random.default_rng(0)
    batches = {b: pd.DataFrame(rng.normal(shift, 1.0, (400, 2)),
                               columns=pd.Index(CHANNELS))
               for b, shift in (('b1', 0.0), ('b2', 0.5))}
    model = cytonorm.CytoNorm(CHANNELS, n_metaclusters=2, grid=(3, 3)).fit(batches)
    # Every event lacks one channel: no goal distribution to compare against.
    holes = pd.DataFrame({'FL1-A': [1.0, np.nan], 'FL2-A': [np.nan, 1.0]})
    with caplog.at_level(logging.WARNING, logger='openflo.pipeline'):
        model.qc({'b1': holes})
    records = [r for r in caplog.records if 'goal distribution' in r.getMessage()]
    assert records, 'no QC warning was logged'
    assert {r.name for r in records} == {'openflo.pipeline'}
