"""Seeds live in seeds, the SNN graph and the phenograph size-sort cap in
cluster_graph, and the gate palette, noise-cluster sentinel and cluster_label
in cluster_labels, not pipeline.

The "Seeds, the SNN graph and cluster labels" block of pipeline.py and the
phenograph size-sort cap above it moved unchanged into the three modules,
which do not import pipeline. The phenograph scratch-folder note, the
categorical palette helpers and FlowSample stayed. These tests pin the move:

* none of the new modules imports pipeline (or Tk);
* _snn_jaccard_graph, _phenograph_sort_jobs and cluster_label, which
  FlowSample uses, are the moved objects, without a warning;
* every other old name still resolves as ``pipeline.X`` to the same object,
  with a DeprecationWarning naming its home;
* no module in the package imports a moved name from pipeline any more.
"""
import ast
import pathlib
import subprocess
import sys
import warnings

import pytest

from openflo import cluster_graph, cluster_labels, flow_sample, pipeline, seeds

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'openflo'

HOMES = {'seeds': seeds, 'cluster_graph': cluster_graph,
         'cluster_labels': cluster_labels}
# Old names served by pipeline.__getattr__ (old name -> (module, new name)).
DEPRECATED = {old: home for old, home in pipeline._MOVED.items()
              if home[0] in HOMES}


def test_the_new_modules_do_not_import_pipeline_or_tk():
    code = ('import sys, openflo.seeds, openflo.cluster_graph, '
            'openflo.cluster_labels; '
            "print('openflo.pipeline' in sys.modules, 'tkinter' in sys.modules)")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ['False', 'False'], out.stdout + out.stderr


def test_the_moved_set_is_the_whole_section():
    assert DEPRECATED == {
        '_PG_SORT_JOBS': ('cluster_graph', '_PG_SORT_JOBS'),
        '_PG_SORT_INSTALL_LOCK': ('cluster_graph', '_PG_SORT_INSTALL_LOCK'),
        '_sort_by_size_serial': ('cluster_graph', 'sort_by_size_serial'),
        '_install_phenograph_sort_cap': ('cluster_graph',
                                         'install_phenograph_sort_cap'),
        'resolve_seed': ('seeds', 'resolve_seed'),
        'GATE_PALETTE': ('cluster_labels', 'GATE_PALETTE'),
        'NOISE_CLUSTER_ID': ('cluster_labels', 'NOISE_CLUSTER_ID'),
        'NOISE_CLUSTER_LABEL': ('cluster_labels', 'NOISE_CLUSTER_LABEL'),
        'NOISE_CLUSTER_COLOR': ('cluster_labels', 'NOISE_CLUSTER_COLOR'),
        # Used by FlowSample, which moved to flow_sample and imports them.
        '_snn_jaccard_graph': ('cluster_graph', 'snn_jaccard_graph'),
        '_phenograph_sort_jobs': ('cluster_graph', 'phenograph_sort_jobs')}
    for old in DEPRECATED:
        assert old not in vars(pipeline), old


def test_the_names_pipeline_uses_are_the_moved_objects():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert flow_sample._snn_jaccard_graph is cluster_graph.snn_jaccard_graph
        assert (flow_sample._phenograph_sort_jobs
                is cluster_graph.phenograph_sort_jobs)
        assert pipeline.cluster_label is cluster_labels.cluster_label


@pytest.mark.parametrize('old', sorted(DEPRECATED))
def test_the_old_pipeline_name_still_works_with_a_warning(old):
    home, new = pipeline._MOVED[old]
    with pytest.warns(DeprecationWarning, match=f'use openflo.{home}.{new}'):
        obj = getattr(pipeline, old)
    assert obj is getattr(HOMES[home], new)


def test_no_module_imports_a_moved_name_from_pipeline():
    moved = {*DEPRECATED, '_snn_jaccard_graph', '_phenograph_sort_jobs',
             'cluster_label'}
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
