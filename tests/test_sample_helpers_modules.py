"""The three small helpers FlowSample uses live outside pipeline: the
unknown-aware matrix product in compensation_math, the per-thread warning
collector in log_capture, and the FCS reader that fills a missing $PnR in
fcs_range_fill.

They moved unchanged and pipeline imports them back under their old names,
so nothing is deprecated. These tests pin the move:

* none of the new modules imports pipeline (or Tk);
* _matmul_known, _collect_warnings and _RangeFillingFlowData in pipeline are
  the moved objects, without a warning, and pipeline no longer defines them;
* the collector still hears what pipeline logs, and only on its own thread;
* the matrix product still keeps an unknown value to the outputs it feeds;
* no module in the package imports a moved name from pipeline.
"""
import ast
import logging
import pathlib
import subprocess
import sys
import threading
import warnings

import numpy as np

from openflo import compensation_math, fcs_range_fill, flow_sample, log_capture, pipeline

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'openflo'
MOVED = {'_matmul_known', '_collect_warnings', '_RangeFillingFlowData'}


def test_the_new_modules_do_not_import_pipeline_or_tk():
    code = ('import sys, openflo.compensation_math, openflo.log_capture, '
            'openflo.fcs_range_fill; '
            "print('openflo.pipeline' in sys.modules, 'tkinter' in sys.modules)")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ['False', 'False'], out.stdout + out.stderr


def test_the_names_pipeline_uses_are_the_moved_objects():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        # FlowSample, their user, now lives in flow_sample; pipeline serves
        # the old names through _MOVED (tests/test_flow_sample_module.py).
        assert flow_sample._matmul_known is compensation_math.matmul_known
        assert flow_sample._collect_warnings is log_capture.collect_warnings
        assert (flow_sample._RangeFillingFlowData
                is fcs_range_fill.RangeFillingFlowData)
    tree = ast.parse((PACKAGE / 'pipeline.py').read_text(encoding='utf-8'))
    defined = {n.name for n in tree.body
               if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert defined & MOVED == set()


def test_the_collector_hears_pipeline_warnings_on_its_own_thread_only():
    other = threading.Thread(
        target=lambda: pipeline.log.warning('  [!] from another thread'))
    with log_capture.collect_warnings() as said:
        pipeline.log.warning('  [!] spillover names FL9-A')
        pipeline.log.info('  progress, not a warning')
        logging.getLogger('openflo.elsewhere').warning('another logger')
        other.start()
        other.join()
    pipeline.log.warning('  [!] after the block')
    assert said == ['spillover names FL9-A']


def test_an_unknown_value_spoils_only_the_outputs_it_feeds():
    matrix = np.array([[1.0, 0.2, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])
    values = np.array([[np.nan, 2.0, 3.0],
                       [1.0, 2.0, 3.0]])
    out = compensation_math.matmul_known(values, matrix)
    assert np.isnan(out[0, :2]).all()
    assert out[0, 2] == 3.0
    np.testing.assert_array_equal(out[1], values[1] @ matrix)


def test_no_module_imports_a_moved_name_from_pipeline():
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
                            for a in node.names if a.name in MOVED)
    assert hits == []
