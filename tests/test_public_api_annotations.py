"""The public FlowSample and FlowExperiment methods, every private FlowSample
helper, the public functions of the gating module, and every function of the
gate evaluator, the group statistics, the FlowJo workspace reader, the
Gating-ML reader and writer and the Pipeline Workspace window, and every
function the openflo package re-exports, carry full type hints
(best-practices survey R10).

These are the library API scripts and notebooks call (gating.py holds the
population counts and statistics), so their parameters and return types are
annotated and pyright checks every caller in src/ against them. This scan
stops an unannotated public method or function coming back.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / 'src' / 'openflo'

CLASSES = [
    ('flow_sample.py', 'FlowSample'),
    ('flow_experiment.py', 'FlowExperiment'),
]


def _public_methods(module, cls):
    tree = ast.parse((SRC / module).read_text(encoding='utf-8'))
    node = next(n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == cls)
    return [n for n in node.body
            if isinstance(n, ast.FunctionDef)
            and (not n.name.startswith('_') or n.name == '__init__')]


def _missing(fn):
    a = fn.args
    params = a.posonlyargs + a.args + a.kwonlyargs
    if params and params[0].arg in ('self', 'cls'):
        params = params[1:]
    params += [p for p in (a.vararg, a.kwarg) if p is not None]
    gaps = [p.arg for p in params if p.annotation is None]
    if fn.returns is None:
        gaps.append('return')
    return gaps


@pytest.mark.parametrize('module, cls', CLASSES)
def test_every_public_method_is_fully_annotated(module, cls):
    methods = _public_methods(module, cls)
    assert len(methods) > 5, f'{cls}: methods not found'
    gaps = {fn.name: _missing(fn) for fn in methods if _missing(fn)}
    assert gaps == {}, f'{cls} methods missing type hints: {gaps}'


def test_every_private_flowsample_method_is_fully_annotated():
    # The loading, compensation, gate and embedding helpers the public
    # methods call; with hints, pyright checks those calls too.
    tree = ast.parse((SRC / 'flow_sample.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == 'FlowSample')
    private = [n for n in node.body
               if isinstance(n, ast.FunctionDef) and n.name.startswith('_')
               and not n.name.startswith('__')]
    assert len(private) > 10, 'FlowSample private methods not found'
    gaps = {fn.name: _missing(fn) for fn in private if _missing(fn)}
    assert gaps == {}, f'FlowSample private methods missing type hints: {gaps}'


MODULES = ['gating.py', 'gate_eval.py', 'stats.py', 'wsp_reader.py']
# Fully annotated modules with too few public functions for the check above
# (gatingml has four, workspace none); their every-function test covers them.
WHOLE_MODULES = ['gatingml.py', 'workspace.py']


def _public_functions(module):
    tree = ast.parse((SRC / module).read_text(encoding='utf-8'))
    return [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and not n.name.startswith('_')]


@pytest.mark.parametrize('module', MODULES)
def test_every_public_function_is_fully_annotated(module):
    functions = _public_functions(module)
    assert len(functions) > 5, f'{module}: functions not found'
    gaps = {fn.name: _missing(fn) for fn in functions if _missing(fn)}
    assert gaps == {}, f'{module} functions missing type hints: {gaps}'


def test_every_gate_eval_function_is_fully_annotated():
    # gate_eval turns every gate into its event mask; the per-kind evaluators
    # and the unknown-population helpers are private but carry the same
    # gate / DataFrame / mask contract, so pyright checks them too.
    tree = ast.parse((SRC / 'gate_eval.py').read_text(encoding='utf-8'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert len(functions) > 20, 'gate_eval functions not found'
    gaps = {fn.name: _missing(fn) for fn in functions if _missing(fn)}
    assert gaps == {}, f'gate_eval functions missing type hints: {gaps}'


def test_every_stats_function_is_fully_annotated():
    # stats holds the group tests behind every comparison window and the
    # differential export; its private helpers and the steps nested in
    # compare_groups carry the same values / p-value contract, so pyright
    # checks them too.
    tree = ast.parse((SRC / 'stats.py').read_text(encoding='utf-8'))
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert len(functions) > 20, 'stats functions not found'
    gaps = {fn.name: _missing(fn) for fn in functions if _missing(fn)}
    assert gaps == {}, f'stats functions missing type hints: {gaps}'


def test_every_wsp_reader_function_is_fully_annotated():
    # wsp_reader turns a FlowJo workspace into gates and compensation
    # matrices; its gate walk (_GateWalk), the WspReader methods and the
    # helpers nested in them carry the same element / gate-dict contract, so
    # pyright checks them too.
    tree = ast.parse((SRC / 'wsp_reader.py').read_text(encoding='utf-8'))
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert len(functions) > 60, 'wsp_reader functions not found'
    gaps = {fn.name: _missing(fn) for fn in functions if _missing(fn)}
    assert gaps == {}, f'wsp_reader functions missing type hints: {gaps}'


def test_every_gatingml_function_is_fully_annotated():
    # gatingml reads and writes standalone Gating-ML 2.0 files; its gate
    # readers and writers, the scale classes (_ExportScales, _DimReader) and
    # the helpers nested in them carry the same element / gate-dict / scale
    # contract, so pyright checks them too.
    tree = ast.parse((SRC / 'gatingml.py').read_text(encoding='utf-8'))
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert len(functions) > 50, 'gatingml functions not found'
    gaps = {fn.name: _missing(fn) for fn in functions if _missing(fn)}
    assert gaps == {}, f'gatingml functions missing type hints: {gaps}'


def test_every_workspace_function_is_fully_annotated():
    # workspace is the Pipeline Workspace window: the tree view over a
    # WorkspaceModel, the results viewer and the panel that runs each unit
    # in a child process. Its drop routing, callbacks and job records carry
    # the same model / tree-row / job contract, so pyright checks them too
    # (ast.walk: methods and the helpers nested in them count).
    tree = ast.parse((SRC / 'workspace.py').read_text(encoding='utf-8'))
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert len(functions) > 100, 'workspace functions not found'
    gaps = {fn.name: _missing(fn) for fn in functions if _missing(fn)}
    assert gaps == {}, f'workspace functions missing type hints: {gaps}'


def _package_exports():
    # The (module, name) pairs openflo/__init__.py re-exports; its
    # TYPE_CHECKING block imports each one from its home module.
    tree = ast.parse((SRC / '__init__.py').read_text(encoding='utf-8'))
    return [(n.module, a.name) for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.level == 1
            for a in n.names]


def _definition(module, name, _seen=()):
    """``(module file, node)`` for the def or class `name` that `module`
    defines, following ``from .x import name`` re-exports (pipeline.py
    re-exports names defined elsewhere). None when it cannot be found."""
    if (module, name) in _seen:
        return None
    tree = ast.parse((SRC / f'{module}.py').read_text(encoding='utf-8'))
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name:
            return f'{module}.py', n
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module:
            for a in n.names:
                if (a.asname or a.name) == name:
                    return _definition(n.module, a.name,
                                       (*_seen, (module, name)))
    return None


def _exported_functions():
    found = []
    for module, name in _package_exports():
        hit = _definition(module, name)
        assert hit is not None, f'openflo.{name}: definition not found'
        if isinstance(hit[1], ast.FunctionDef):
            found.append(hit)
    return found


EXPORT_MODULES = sorted({m for m, _ in _exported_functions()})


def test_every_function_exported_from_openflo_is_fully_annotated():
    # `from openflo import ...` is the documented way in; every name it
    # exports must resolve to a def or class (an unresolved one fails here,
    # not silently). The classes are covered above or by later slices.
    functions = _exported_functions()
    assert len(functions) > 50, 'exported functions not found'
    gaps = {f'{module}:{fn.name}': _missing(fn)
            for module, fn in functions if _missing(fn)}
    assert gaps == {}, f'exported functions missing type hints: {gaps}'


@pytest.mark.parametrize('module', sorted(
    {m for m, _ in CLASSES} | set(MODULES) | set(WHOLE_MODULES)
    | set(EXPORT_MODULES)))
def test_annotations_are_never_evaluated_at_import(module):
    # Hints name pandas, numpy and matplotlib types; postponed evaluation keeps
    # them free at import time (matplotlib loads only inside plot methods).
    tree = ast.parse((SRC / module).read_text(encoding='utf-8'))
    futures = {a.name for n in tree.body
               if isinstance(n, ast.ImportFrom) and n.module == '__future__'
               for a in n.names}
    assert 'annotations' in futures
