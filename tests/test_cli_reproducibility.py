"""openflo-run reproducibility (Complaints item 23): a run manifest, gates
from a GUI session / template / .wsp / Gating-ML file, gates in data units,
and Benjamini-Hochberg p-values in the comparison CSV.

In-process, as test_cli_run_options.py: clustering, UMAP and plots are
stubbed; every expected count is computed here from the generated events.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

import openflo.cli as cli
from openflo.pipeline import FlowSample, transform_spec, transform_values

sys.path.insert(0, os.path.dirname(__file__))
from test_cli_run_options import _fluor, _scatter, _write  # noqa: E402


@pytest.fixture
def stubbed(monkeypatch):
    def fake_cluster(self, **kw):
        self.data['cluster'] = np.where(
            (self.data[self._resolve('FL1-A')]
             > self.data[self._resolve('FL2-A')]).to_numpy(), 0, 1)
        return self
    monkeypatch.setattr(FlowSample, 'cluster', fake_cluster)
    monkeypatch.setattr(cli, 'save_plots',
                        lambda s, out_dir, **k: os.makedirs(out_dir,
                                                         exist_ok=True))
    monkeypatch.setattr(cli, 'run_group_umap', lambda *a, **k: None)
    monkeypatch.setattr(cli, 'save_group_pair_scatters', lambda *a, **k: None)
    monkeypatch.setattr(cli, '_gpu_present', lambda: False)


def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, 'argv', ['openflo-run', *argv])
    return cli.main()


def _trial(tmp_path, fracs=(('a', (.70, .72, .74)), ('b', (.30, .32, .34))),
           n=1000):
    trial = tmp_path / 'trial'
    for k, (g, fs) in enumerate(fracs):
        for i, f in enumerate(fs):
            rng = np.random.default_rng(10 * k + i)
            _write(str(trial / f'{g}{i}.fcs'),
                   np.column_stack([_scatter(rng, n),
                                    _fluor(rng, n, frac_bv=f)]))
    return trial


GROUPS = json.dumps([{'name': 'A', 'samples': ['a0', 'a1', 'a2'], 'fmo_set': ''},
                     {'name': 'B', 'samples': ['b0', 'b1', 'b2'], 'fmo_set': ''}])


def _kept(out, group, name):
    return int(pd.read_csv(out / group / f'{name}_stats.csv')['count'].sum())


def _fl1(trial, name):
    return FlowSample(str(trial / f'{name}.fcs')).data['FL1-A'].to_numpy()


def test_comparison_csv_has_logit_t_and_bh_p_values(
        tmp_path, stubbed, monkeypatch):
    """A (70-74% FL1+) against B (30-34%): every A sample beats every B
    sample. Mann-Whitney could only ever say p = 0.1 here (2 of 20
    orderings at 3 vs 3); Welch's t on the logit scale gives the real
    strength of the difference, well under 0.05, and BH over two clusters
    with equal p keeps it."""
    trial = _trial(tmp_path)
    out = tmp_path / 'out'
    assert _main(monkeypatch, '--trials', str(trial), '--out', str(out),
                 '--groups', GROUPS, '--fmo-sets', '{}', '-q') == 0
    df = pd.read_csv(out / 'compare_A_vs_B.csv')
    p = df['p'].tolist()
    assert all(x < 0.001 for x in p)
    assert p == pytest.approx([p[0]] * 4)
    assert df['p_adj'].tolist() == pytest.approx(p)


def test_bh_adjusts_across_clusters():
    """Unit check of the column builder with three clusters of known p."""
    class S:
        def __init__(self, labels):
            self.data = pd.DataFrame({'cluster': labels})
    # Cluster 0: A all higher (p 0.1); cluster 1: tied shares (p 1.0);
    # cluster 2: B higher (p 0.1).
    sa = [S([0] * 6 + [1] * 2 + [2] * 2), S([0] * 7 + [1] * 2 + [2] * 1),
          S([0] * 8 + [1] * 2)]
    sb = [S([0] * 2 + [1] * 2 + [2] * 6), S([0] * 1 + [1] * 2 + [2] * 7),
          S([1] * 2 + [2] * 8)]
    summary = pd.DataFrame({'condition': ['A'] * 3 + ['B'] * 3,
                            'cluster': [0, 1, 2] * 2})
    cli._add_pvalues(summary, sa, sb)
    p = dict(zip(summary['cluster'], summary['p'], strict=False))
    q = dict(zip(summary['cluster'], summary['p_adj'], strict=False))
    # Welch's t on the logit of each sample's %.
    from scipy.stats import ttest_ind

    from openflo.stats import transform_values as tv
    exp0 = ttest_ind(tv([60, 70, 80], 'logit'), tv([20, 10, 0], 'logit'),
                     equal_var=False).pvalue
    exp2 = ttest_ind(tv([20, 10, 0], 'logit'), tv([60, 70, 80], 'logit'),
                     equal_var=False).pvalue
    assert p[0] == pytest.approx(exp0) and p[2] == pytest.approx(exp2)
    assert p[1] == pytest.approx(1.0)        # the same share everywhere
    # BH: two equal p below 1.0 -> each times 3/2; 1.0 stays.
    assert q == pytest.approx({0: exp0 * 1.5, 1: 1.0, 2: exp2 * 1.5})


def test_manifest_records_inputs_settings_software_and_outputs(
        tmp_path, stubbed, monkeypatch):
    trial = _trial(tmp_path, fracs=(('a', (.7,)),))
    out = tmp_path / 'out'
    grp = json.dumps([{'name': 'A', 'samples': ['a0'], 'fmo_set': ''}])
    assert _main(monkeypatch, '--trials', str(trial), '--out', str(out),
                 '--groups', grp, '--fmo-sets', '{}', '--seed', '7',
                 '-q') == 0
    m = json.loads((out / cli.MANIFEST_NAME).read_text(encoding='utf-8'))
    assert m['format'] == 'openflo-run-manifest' and m['exit_code'] == 0
    assert m['settings']['seed'] == 7
    assert m['command'][0] == 'openflo-run' and '--seed' in m['command']
    assert 'numpy' in m['packages'] and m['openflo']
    (inp,) = m['inputs']
    data = (trial / 'a0.fcs').read_bytes()
    assert inp['sha256'] == hashlib.sha256(data).hexdigest()
    assert inp['bytes'] == len(data)
    outs = {o['path']: o for o in m['outputs']}
    assert 'A/a0_stats.csv' in outs
    assert outs['A/a0_stats.csv']['sha256'] == hashlib.sha256(
        (out / 'A' / 'a0_stats.csv').read_bytes()).hexdigest()
    assert cli.MANIFEST_NAME not in outs


def test_gates_in_data_units_cut_where_the_intensity_says(
        tmp_path, stubbed, monkeypatch):
    """--gate-units data: an FL1-A interval [5000, 1e6) in intensities keeps
    exactly the events whose FL1-A lies there. Read as stored units (the
    default) the same numbers lie far above the logicle scale's top (1.0)
    and keep nothing."""
    trial = _trial(tmp_path, fracs=(('a', (.6,)),))
    grp = json.dumps([{'name': 'A', 'samples': ['a0'], 'fmo_set': ''}])
    gate = json.dumps([{'kind': 'interval', 'channel': 'FL1-A',
                        'lo': 5000, 'hi': 1e6, 'id': 'g1'}])
    fl1 = _fl1(trial, 'a0')
    want = int(((fl1 >= 5000) & (fl1 < 1e6)).sum())
    assert 300 < want < 900
    out = tmp_path / 'data'
    _main(monkeypatch, '--trials', str(trial), '--out', str(out), '--groups',
          grp, '--fmo-sets', '{}', '-q', '--gates', gate,
          '--gate-units', 'data')
    assert _kept(out, 'A', 'a0') == want
    out = tmp_path / 'stored'
    _main(monkeypatch, '--trials', str(trial), '--out', str(out), '--groups',
          grp, '--fmo-sets', '{}', '-q', '--gates', gate)
    assert not (out / 'A' / 'a0_stats.csv').exists() or \
        _kept(out, 'A', 'a0') == 0


def _tree_stored():
    """FSC gate, then an FL1-A+ child and an FL2-A+ child, on the run's own
    scale (logicle fluorescence)."""
    lg = transform_spec('logicle')
    cut = float(transform_values(np.array([5000.0]), **lg)[0])
    return [
        {'id': 'g1', 'parent_id': None, 'kind': 'interval', 'name': 'Cells',
         'channel': 'FSC-A', 'lo': 90000.0, 'hi': 1e12},
        {'id': 'g2', 'parent_id': 'g1', 'kind': 'interval', 'name': 'FL1 pos',
         'channel': 'FL1-A', 'lo': cut, 'hi': 1e12},
        {'id': 'g3', 'parent_id': 'g1', 'kind': 'interval', 'name': 'FL2 pos',
         'channel': 'FL2-A', 'lo': cut, 'hi': 1e12},
    ]


def _write_gate_file(tmp_path, kind):
    """The same tree as each kind of gate file."""
    gates = _tree_stored()
    if kind == 'template':
        path = tmp_path / 't.json'
        path.write_text(json.dumps({'version': 2, 'gates': gates}),
                        encoding='utf-8')
    elif kind == 'gatingml':
        from openflo.gatingml import write_gatingml
        path = tmp_path / 't.xml'
        write_gatingml(str(path), gates,
                       {'FL1-A': transform_spec('logicle'),
                        'FL2-A': transform_spec('logicle')})
    elif kind == 'wsp':
        from openflo.pipeline import WspWriter
        path = tmp_path / 't.wsp'
        w = WspWriter()
        w.add_sample(name='x', fcs_path='', channels=[], gates=gates,
                     transforms={'FL1-A': 'logicle', 'FL2-A': 'logicle'})
        w.write(str(path))
    else:                                      # a GUI session, on asinh
        asinh = transform_spec('asinh', cofactor=150.0)
        lg = transform_spec('logicle')
        moved = []
        for g in gates:
            g = dict(g)
            if g['channel'] != 'FSC-A':
                from openflo.pipeline import inverse_transform_values
                lin = inverse_transform_values(np.array([g['lo']]), **lg)
                g['lo'] = float(transform_values(lin, **asinh)[0])
            moved.append(g)
        path = tmp_path / 's.flowsession'
        path.write_text(json.dumps({
            'format': 'openflo-session', 'version': 2, 'active_sample': 'x',
            'samples': [{'name': 'x', 'transforms': {
                'FL1-A': asinh, 'FL2-A': asinh}}],
            'sample_gates': {'x': moved}}), encoding='utf-8')
    return str(path)


@pytest.mark.parametrize('kind', ['template', 'gatingml', 'wsp', 'session'])
def test_gates_from_a_file_gate_on_the_named_population(
        tmp_path, stubbed, monkeypatch, kind):
    """Each kind of gate file, holding Cells > FL1 pos and Cells > FL2 pos,
    gates every sample on the population named: FSC-A >= 90,000 and FL1-A
    >= 5,000 (intensities). A session saved on asinh is put on the run's
    logicle first."""
    trial = _trial(tmp_path, fracs=(('a', (.6,)),))
    grp = json.dumps([{'name': 'A', 'samples': ['a0'], 'fmo_set': ''}])
    s = FlowSample(str(trial / 'a0.fcs'))
    want = int(((s.data['FSC-A'] >= 90000) & (s.data['FL1-A'] >= 5000)).sum())
    assert 200 < want < 800
    path = _write_gate_file(tmp_path, kind)
    out = tmp_path / 'out'
    assert _main(monkeypatch, '--trials', str(trial), '--out', str(out),
                 '--groups', grp, '--fmo-sets', '{}', '-q',
                 '--gates-from', path, '--gate-population', 'fl1 POS') == 0
    assert abs(_kept(out, 'A', 'a0') - want) <= 1


def test_gates_from_a_tree_needs_a_population(
        tmp_path, stubbed, monkeypatch, capsys):
    trial = _trial(tmp_path, fracs=(('a', (.6,)),))
    grp = json.dumps([{'name': 'A', 'samples': ['a0'], 'fmo_set': ''}])
    path = _write_gate_file(tmp_path, 'template')
    out = tmp_path / 'out'
    rc = _main(monkeypatch, '--trials', str(trial), '--out', str(out),
               '--groups', grp, '--fmo-sets', '{}', '-q', '--gates-from', path)
    assert rc == 2
    msg = capsys.readouterr().out
    assert 'several populations' in msg and 'FL1 pos' in msg and 'FL2 pos' in msg
    rc = _main(monkeypatch, '--trials', str(trial), '--out', str(out),
               '--groups', grp, '--fmo-sets', '{}', '-q', '--gates-from', path,
               '--gate-population', 'CD4')
    assert rc == 2 and "no population named 'CD4'" in capsys.readouterr().out


def test_population_path_brings_boolean_operands_disabled():
    gates = [
        {'id': 'a', 'parent_id': None, 'kind': 'threshold', 'channel': 'FSC-A',
         'value': 1, 'name': 'Cells'},
        {'id': 'b', 'parent_id': 'a', 'kind': 'threshold', 'channel': 'FL1-A',
         'value': 1, 'name': 'FL1+'},
        {'id': 'c', 'parent_id': 'a', 'kind': 'boolean', 'op': 'not',
         'operands': ['b'], 'name': 'FL1-'},
    ]
    got = cli._population_gates(gates, 'FL1-')
    assert [(g['id'], g.get('enabled', True)) for g in got] == \
        [('b', False), ('a', True), ('c', True)]
