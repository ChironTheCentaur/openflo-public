"""FlowSOM lives in flowsom, not pipeline.

The SOM training, node assignment and metaclustering helpers, the MST /
layout helpers behind the FlowSOM tree, and igraph_seeded were defined in
pipeline.py and moved, unchanged apart from public names, into their own core
module so pipeline.py can be split up. These tests pin the move:

* every moved name lives in flowsom, and pipeline binds the ones it uses,
  under their old names, to the SAME objects (imported, not copied), with no
  warning;
* ``pipeline.flowsom_mst`` / ``pipeline.flowsom_layout``, which pipeline no
  longer uses, still resolve, with a DeprecationWarning naming the new home,
  and the package-level ``openflo.flowsom_mst`` / ``flowsom_layout`` come
  from flowsom without one;
* flowsom does not import pipeline, and it still loads the compiled SOM loop
  (som_kernel, numba), scikit-learn and igraph only when they are used;
* no module in the package imports a moved name from pipeline any more, and
  pipeline defines no copy of one.
"""
import ast
import pathlib
import subprocess
import sys
import warnings

import openflo
from openflo import flow_sample, flowsom, pipeline

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'src' / 'openflo'

# pipeline's old name -> flowsom's name, for the names FlowSample uses (bound
# in flow_sample, FlowSample's home; the public igraph_seeded also in pipeline).
IMPORTED = {
    '_som_train': 'som_train',
    '_som_assign': 'som_assign',
    '_som_metacluster': 'som_metacluster',
    'igraph_seeded': 'igraph_seeded',
}
# Old names pipeline does not use: served by pipeline.__getattr__, deprecated.
DEPRECATED = {'flowsom_mst': 'flowsom_mst', 'flowsom_layout': 'flowsom_layout'}
OLD_TO_NEW = {**IMPORTED, **DEPRECATED}
# Any spelling, old or new, a module might import from pipeline.
MOVED = set(OLD_TO_NEW) | set(OLD_TO_NEW.values())


def test_pipeline_binds_its_old_names_to_the_flowsom_objects():
    for new in OLD_TO_NEW.values():
        assert new in vars(flowsom), new
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for old, new in IMPORTED.items():
            assert old in vars(flow_sample), old
            assert getattr(flow_sample, old) is getattr(flowsom, new), old
        assert pipeline.igraph_seeded is flowsom.igraph_seeded


def test_the_old_tree_helper_names_still_work_with_a_warning():
    for old, new in DEPRECATED.items():
        assert old not in vars(pipeline)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            obj = getattr(pipeline, old)
        assert obj is getattr(flowsom, new)
        hits = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert hits, f'pipeline.{old} gave no DeprecationWarning'
        assert f'use openflo.flowsom.{new}' in str(hits[0].message)


def test_the_package_serves_the_tree_helpers_from_flowsom():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for name in DEPRECATED.values():
            assert openflo._PUBLIC[name] == 'openflo.flowsom', name
            assert getattr(openflo, name) is getattr(flowsom, name), name


def test_importing_flowsom_loads_neither_pipeline_nor_the_heavy_deps():
    heavy = ('openflo.pipeline', 'openflo.som_kernel', 'numba', 'sklearn', 'igraph')
    code = ('import sys, openflo.flowsom; '
            f'print([m for m in {heavy!r} if m in sys.modules])')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, check=True, timeout=120)
    assert out.stdout.strip() == '[]', out.stdout + out.stderr


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
    assert not hits, 'import these from .flowsom:\n' + '\n'.join(hits)


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
    assert not defined & MOVED, sorted(defined & MOVED)
