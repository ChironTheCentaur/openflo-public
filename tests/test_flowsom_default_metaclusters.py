"""FlowSOM's default metacluster count is one shared constant, 20, everywhere.

The Cluster dialog started at 20 while openflo-run, FlowSample.run_flowsom
and the workspace batch still started at 10, so the same data clustered
from the GUI and from a script came out with different populations. These
tests pin every entry point to som_kernel.FLOWSOM_DEFAULT_METACLUSTERS, check
a run with the default really makes 20 metaclusters (10 when 10 is passed),
and check CytoNorm's own count stays at 10.
"""
from __future__ import annotations

import inspect
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pandas as pd

import openflo.cli as cli
import openflo.workspace as W
from openflo import som_kernel
from openflo.batch_normalize import fit_batch_model
from openflo.cli_args import build_parser
from openflo.pipeline import CytoNorm, FlowExperiment, FlowSample

CH = ['FL1-A', 'FL2-A', 'FL3-A']


def _default(fn, name):
    return inspect.signature(fn).parameters[name].default


def _cube_sample(n=6000, seed=0):
    """Events spread evenly through a cube in three channels. Every SOM node
    then wins events, so each of the k metaclusters the nodes are grouped
    into holds events and the labels seen are exactly 0..k-1. (Separate
    blobs leave dead nodes between them, and a metacluster of dead nodes
    labels no event.)"""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 100, size=(n, len(CH)))
    s = FlowSample.from_dataframe(pd.DataFrame(X, columns=CH), name='SampleA')
    s.fluor_channels = list(CH)
    return s


def test_the_shared_default_is_20():
    assert som_kernel.FLOWSOM_DEFAULT_METACLUSTERS == 20


def test_cli_parses_n_metaclusters_default_as_20():
    args = build_parser().parse_args([])
    assert args.n_metaclusters == 20
    assert '20' in next(a.help for a in build_parser()._actions
                        if a.dest == 'n_metaclusters')
    # Passing a value still wins.
    assert build_parser().parse_args(['--n-metaclusters', '10']).n_metaclusters == 10


def test_api_signatures_default_to_20():
    assert _default(FlowSample.run_flowsom, 'n_metaclusters') == 20
    assert _default(FlowExperiment.cluster_jointly, 'n_metaclusters') == 20
    assert _default(cli.run, 'n_metaclusters') == 20
    assert W.DEFAULT_RUN_CFG['n_metaclusters'] == 20
    assert W.default_run_cfg()['n_metaclusters'] == 20
    # A run's shared settings with no n_metaclusters given.
    assert cli._cluster_settings(30, 1, None, 1.0, 42,
                                 {'method': 'flowsom'})['n_metaclusters'] == 20


def test_run_flowsom_default_makes_20_metaclusters_and_10_when_asked():
    s = _cube_sample()
    s.run_flowsom(grid=(8, 8), iters=4, seed=0)
    meta = s.data['flowsom_meta'].to_numpy()
    assert s.flowsom_result['n_metaclusters'] == 20
    assert len(np.unique(meta[meta >= 0])) == 20
    assert set(np.unique(meta)) == set(range(20))

    s10 = _cube_sample()
    s10.run_flowsom(grid=(8, 8), n_metaclusters=10, iters=4, seed=0)
    meta10 = s10.data['flowsom_meta'].to_numpy()
    assert s10.flowsom_result['n_metaclusters'] == 10
    assert set(np.unique(meta10)) == set(range(10))


def test_cli_task_without_n_metaclusters_runs_flowsom_with_20(monkeypatch):
    seen = []
    real = FlowSample.run_flowsom

    def spy(self, **kw):
        seen.append(kw.get('n_metaclusters'))
        return real(self, **kw)

    monkeypatch.setattr(FlowSample, 'run_flowsom', spy)
    s = _cube_sample()
    cli._cluster_by_method(s, {'method': 'flowsom', 'k': 30, 'n_jobs': 1,
                               'max_events': None, 'random_state': 0})
    assert seen == [20]
    assert set(np.unique(s.data['cluster'])) == set(range(20))


def test_workspace_cfg_without_n_metaclusters_runs_flowsom_with_20(
        monkeypatch, tmp_path):
    seen = []
    real = FlowSample.run_flowsom

    def spy(self, **kw):
        seen.append(kw.get('n_metaclusters'))
        return real(self, **kw)

    monkeypatch.setattr(FlowSample, 'run_flowsom', spy)
    s = _cube_sample()
    editor = SimpleNamespace(_samples={'SampleA': s})
    cfg = {'method': 'flowsom', 'k': 10, 'max_events': 0, 'seed': 0,
           'umap': False, 'trimap': False}
    res = W.run_workspace_item(editor, {'sample': 'SampleA', 'path': None,
                                        'gate_id': None, 'gates': {}},
                               cfg, str(tmp_path))
    assert res['ok'] is True, res
    assert seen == [20]
    assert res['n_clusters'] == 20


def test_dialog_cli_and_api_share_one_constant():
    """Change the constant before anything reads it: every default follows,
    so no entry point carries its own copy of the number."""
    code = textwrap.dedent("""
        import inspect
        from openflo import som_kernel
        som_kernel.FLOWSOM_DEFAULT_METACLUSTERS = 7
        from openflo import cli, cli_args, editor_analysis, pipeline, workspace
        d = lambda f: inspect.signature(f).parameters['n_metaclusters'].default
        print(cli_args.build_parser().parse_args([]).n_metaclusters,
              editor_analysis.FLOWSOM_DIALOG_METACLUSTERS,
              d(pipeline.FlowSample.run_flowsom),
              d(pipeline.FlowExperiment.cluster_jointly),
              d(cli.run),
              workspace.DEFAULT_RUN_CFG['n_metaclusters'],
              cli._cluster_settings(30, 1, None, 1.0, 42, {})['n_metaclusters'],
              cli_args.build_parser().parse_args([]).cytonorm_metaclusters)
    """)
    env = dict(os.environ, MPLBACKEND='Agg')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ['7'] * 7 + ['10']


def test_cytonorm_keeps_its_own_10():
    assert build_parser().parse_args([]).cytonorm_metaclusters == 10
    assert _default(cli.run, 'cytonorm_metaclusters') == 10
    assert _default(cli._fit_cytonorm, 'n_metaclusters') == 10
    assert _default(fit_batch_model, 'n_metaclusters') == 10
    assert _default(CytoNorm.__init__, 'n_metaclusters') == 10
    assert CytoNorm(CH).n_metaclusters == 10
