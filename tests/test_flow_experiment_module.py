"""The last classes and function of pipeline.py live in their own core
modules: FMOGater in fmo_gater, write_fcs (the ``write_fcs(df, path)`` wrapper
over fcs_export.write_fcs) in fcs_write, and FlowExperiment in
flow_experiment.

They moved unchanged and pipeline imports them back, so pipeline.FMOGater and
``from openflo import FMOGater`` (and the rest) are the moved objects. These
tests pin the move:

* the new modules do not import pipeline (or Tk);
* pipeline.FMOGater, write_fcs and FlowExperiment, and openflo's, are the
  moved objects, without a warning, and pipeline defines no copy of them;
* no other module imports them from pipeline;
* the public names they use are still pipeline attributes, without a
  warning, and the private alias pipeline no longer uses resolves through
  pipeline._MOVED with exactly one warning;
* FMOGater and FlowExperiment still log to the 'openflo.pipeline' logger.
"""
import ast
import importlib
import logging
import pathlib
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd

import openflo
from openflo import fcs_write, flow_experiment, flow_sample, fmo_gater, pipeline

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'openflo'
HOMES = {'FMOGater': fmo_gater, 'write_fcs': fcs_write,
         'FlowExperiment': flow_experiment}

# Public names pipeline imported for the moved code and no longer uses
# itself: still pipeline attributes, with no warning. name -> (home, the new
# module that uses it).
STILL_PUBLIC = {
    'to_linear': ('channel_transforms', fmo_gater),
    'fmo_threshold': ('gating_helpers', fmo_gater),
    'df_path_args': ('fcs_export', fcs_write),
    'cluster_label': ('cluster_labels', flow_experiment),
    'FLOWSOM_DEFAULT_METACLUSTERS': ('som_kernel', flow_experiment),
    'concatenate': ('label_alignment', flow_experiment),
}

# Private names pipeline imported for the moved code: old -> (home, name).
NOW_DEPRECATED = {'_export_write_fcs': ('fcs_export', 'write_fcs')}


def _frame(n=300, seed=5):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'FSC-A': rng.uniform(1e4, 2e5, n),
                         'SSC-A': rng.uniform(1e4, 2e5, n),
                         'FL1-A': rng.lognormal(6, 1, n),
                         'FL2-A': rng.lognormal(5, 1, n)})


def test_the_new_modules_do_not_import_pipeline_or_tk():
    code = ('import sys, openflo.fmo_gater, openflo.fcs_write, '
            'openflo.flow_experiment; '
            "print('openflo.pipeline' in sys.modules, 'tkinter' in sys.modules)")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ['False', 'False'], out.stdout + out.stderr


def test_pipeline_and_openflo_bind_the_moved_objects_without_a_warning():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for name, home in HOMES.items():
            obj = vars(home)[name]
            assert obj.__module__ == home.__name__, name
            assert vars(pipeline)[name] is obj, name
            assert getattr(pipeline, name) is obj, name
            assert getattr(openflo, name) is obj, name
            assert openflo._PUBLIC[name] == home.__name__, name


def test_pipeline_defines_no_copy_of_a_moved_name():
    tree = ast.parse((PACKAGE / 'pipeline.py').read_text(encoding='utf-8'))
    defined = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            defined.update(t.id for t in node.targets if isinstance(t, ast.Name))
    assert defined & set(HOMES) == set()


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
                            for a in node.names if a.name in HOMES)
    assert hits == []


def test_the_moved_code_uses_flow_sample_directly():
    assert fmo_gater.FlowSample is flow_sample.FlowSample
    assert flow_experiment.FlowSample is flow_sample.FlowSample


def test_public_names_stay_on_pipeline_without_a_warning():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for name, (home, user) in STILL_PUBLIC.items():
            target = getattr(importlib.import_module(f'openflo.{home}'), name)
            assert name in vars(pipeline), name
            assert getattr(pipeline, name) is target, name
            assert vars(user)[name] is target, name


def test_private_names_resolve_with_exactly_one_warning():
    for old, (home, new) in NOW_DEPRECATED.items():
        assert old not in vars(pipeline), old
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            obj = getattr(pipeline, old)
        target = getattr(importlib.import_module(f'openflo.{home}'), new)
        assert obj is target, old
        assert vars(fcs_write)[old] is target, old
        hits = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(hits) == 1, [str(w.message) for w in caught]
        assert f'use openflo.{home}.{new}' in str(hits[0].message), old


def test_write_fcs_still_writes_through_the_export(tmp_path):
    df = _frame(120)
    path = tmp_path / 'synthetic_a.fcs'
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        n = pipeline.write_fcs(df, str(path))
    assert n == 120
    back = flow_sample.FlowSample(str(path))
    assert len(back.raw) == 120
    assert {'FSC-A', 'SSC-A', 'FL1-A', 'FL2-A'} <= set(back.raw.columns)


def test_fmo_gater_logs_to_the_pipeline_logger(caplog):
    s = flow_sample.FlowSample.from_dataframe(_frame(), name='synthetic_fmo')
    s.apply_transform(['FL1-A'], method='asinh')
    g = fmo_gater.FMOGater()
    g.fmos['FL1-A'] = s
    with caplog.at_level(logging.INFO, logger='openflo.pipeline'):
        cut = g.compute(percentile=99.0)['FL1-A']
    assert cut == float(np.percentile(s.data['FL1-A'], 99.0))
    assert fmo_gater.log.name == 'openflo.pipeline'
    lines = [r for r in caplog.records if '[FMO] FL1-A' in r.getMessage()]
    assert len(lines) == 1
    assert lines[0].name == 'openflo.pipeline'
    assert lines[0].module == 'fmo_gater'


def test_flow_experiment_logs_to_the_pipeline_logger(tmp_path, caplog):
    for name, seed in (('synthetic_a', 1), ('synthetic_b', 2)):
        fcs_write.write_fcs(_frame(seed=seed), str(tmp_path / f'{name}.fcs'))
    with caplog.at_level(logging.INFO, logger='openflo.pipeline'):
        exp = flow_experiment.FlowExperiment(str(tmp_path))
        exp.exclude_pattern('_b$')
    assert list(exp.samples) == ['synthetic_a']
    assert flow_experiment.log.name == 'openflo.pipeline'
    loaded = [r for r in caplog.records if 'Loaded 2 sample(s).' in r.getMessage()]
    excluded = [r for r in caplog.records if r.getMessage().startswith('Excluded 1')]
    for found in (loaded, excluded):
        assert len(found) == 1
        assert found[0].name == 'openflo.pipeline'
        assert found[0].module == 'flow_experiment'
