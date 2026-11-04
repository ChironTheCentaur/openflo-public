"""The GPU probe lives in gpu_probe, not pipeline.

The probe (GPU_AVAILABLE, GPU_NAME, the cuML UMAP class, the RAPIDS clustering
kit, GPU_CLUSTERING_AVAILABLE) and the free-VRAM query were defined in
pipeline.py and moved, unchanged apart from public names, into their own core
module so pipeline.py can be split up. These tests pin the move:

* every moved name lives in gpu_probe, and pipeline binds the ones it uses,
  under their old names, to the SAME objects (imported, not copied);
* ``pipeline._probe_gpu``, which pipeline no longer uses, still resolves, with
  a DeprecationWarning naming the new home;
* gpu_probe does not import pipeline, so a caller that only asks "is there a
  GPU?" does not pay for the whole pipeline module;
* no module in the package imports a moved name from pipeline any more, and
  pipeline defines no copy of one.

Since FlowSample moved to flow_sample, it is flow_sample that binds these
names; pipeline keeps the public ones. That monkeypatching
``flow_sample._vram_free_gb`` / ``GPU_CLUSTERING_AVAILABLE`` /
``_GPU_CLUSTER_KIT`` steers FlowSample.cluster()'s GPU path is pinned by
tests/test_cluster_gpu_fallback.py, which patches exactly those names.
"""
import ast
import pathlib
import subprocess
import sys
import warnings

from openflo import flow_sample, gpu_probe, pipeline

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'src' / 'openflo'

# pipeline's old name -> gpu_probe's name, for the names pipeline still uses.
IMPORTED = {
    'GPU_AVAILABLE': 'GPU_AVAILABLE',
    'GPU_NAME': 'GPU_NAME',
    'GPU_CLUSTERING_AVAILABLE': 'GPU_CLUSTERING_AVAILABLE',
    '_CumlUMAP': 'CUML_UMAP',
    '_GPU_CLUSTER_KIT': 'GPU_CLUSTER_KIT',
    '_vram_free_gb': 'vram_free_gb',
}
# Old names pipeline does not use: served by pipeline.__getattr__, deprecated.
DEPRECATED = {'_probe_gpu': 'probe_gpu'}
OLD_TO_NEW = {**IMPORTED, **DEPRECATED}
# Any spelling, old or new, a module might import from pipeline.
MOVED = set(OLD_TO_NEW) | set(OLD_TO_NEW.values())


def test_pipeline_binds_its_old_names_to_the_gpu_probe_objects():
    for new in OLD_TO_NEW.values():
        assert new in vars(gpu_probe), new
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for old, new in IMPORTED.items():
            assert old in vars(flow_sample), old
            assert getattr(flow_sample, old) is getattr(gpu_probe, new), old
            if not old.startswith('_'):
                assert getattr(pipeline, old) is getattr(gpu_probe, new), old


def test_the_old_probe_name_still_works_with_a_warning():
    for old, new in DEPRECATED.items():
        assert old not in vars(pipeline)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            obj = getattr(pipeline, old)
        assert obj is getattr(gpu_probe, new)
        hits = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert hits, f'pipeline.{old} gave no DeprecationWarning'
        assert f'use openflo.gpu_probe.{new}' in str(hits[0].message)


def test_importing_gpu_probe_does_not_import_pipeline():
    code = ('import sys, openflo.gpu_probe; '
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
    assert not hits, 'import these from .gpu_probe:\n' + '\n'.join(hits)


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
