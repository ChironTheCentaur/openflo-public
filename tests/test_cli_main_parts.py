"""What each part of cli_main.main does, pinned through main().

main is a short caller of named helpers (_parse_args, _configure_logging,
_resolve_groups, _refuse_bad_groups, _resolve_gates, _channel_labels,
_run_analysis, _write_outputs). These tests drive main() with an argv and
record what reaches the functions it dispatches to (run, run_batch_unmix,
export_pipeline_workspace, write_tables_workbook, write_run_manifest), so a
helper call dropped, moved or handed the wrong value fails here. Generated
data only: small FCS files written by tests/_fcs_writer.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import pytest

import openflo.cli_main as cli_main
from openflo import flow_sample
from openflo.cli_defaults import DEFAULT_FMO_SETS, DEFAULT_GROUPS
from tests._fcs_writer import write_float_fcs

CH = ['FSC-A', 'FSC-H', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A']


def _fcs(path, seed=0, n=50):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    write_float_fcs(str(path), rng.lognormal(6, 1, (n, len(CH))), CH)


@pytest.fixture
def root_logging():
    """main configures the root logger with force=True; put it back."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield root
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in handlers:
        root.addHandler(h)
    root.setLevel(level)


@pytest.fixture
def calls(monkeypatch, tmp_path, root_logging):
    """Record what main hands on; run returns calls['rc'] (default 0)."""
    log = {'rc': 0, 'order': []}

    def rec(name, ret=None):
        def f(*a, **k):
            log['order'].append(name)
            log.setdefault(name, []).append((a, k))
            if name == 'run':
                return log['rc']
            return ret
        return f
    for name in ('run', 'export_pipeline_workspace',
                 'write_tables_workbook', 'write_run_manifest'):
        monkeypatch.setattr(cli_main, name, rec(name))
    monkeypatch.setattr(cli_main, 'run_batch_unmix', rec('run_batch_unmix', 5))
    monkeypatch.setattr(flow_sample, '_DEFAULT_CATEGORICAL_PALETTE',
                        flow_sample._DEFAULT_CATEGORICAL_PALETTE)
    monkeypatch.chdir(tmp_path)
    return log


def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, 'argv', ['openflo-run', *argv])
    return cli_main.main()


def _run_kw(calls):
    (_a, kw), = calls['run']
    return kw


def _manifest(calls):
    (a, _kw), = calls['write_run_manifest']
    return a


@pytest.fixture
def trial(tmp_path):
    for i, nm in enumerate(('a0', 'a1', 'late_b0')):
        _fcs(tmp_path / 'trial' / f'{nm}.fcs', seed=i)
    return 'trial'


G1 = json.dumps([{'name': 'A', 'samples': ['a0'], 'fmo_set': ''}])


# ── _parse_args ─────────────────────────────────────────────────────────────
def test_help_exits_0_before_anything_runs(calls, monkeypatch, capsys):
    with pytest.raises(SystemExit) as e:
        _main(monkeypatch, '--help')
    assert e.value.code == 0
    assert 'usage: openflo-run' in capsys.readouterr().out
    assert calls['order'] == []


def test_check_args_refusal_exits_2_before_anything_runs(
        calls, monkeypatch, capsys):
    with pytest.raises(SystemExit) as e:
        _main(monkeypatch, '--batch-correct', '--cytonorm-mode', 'controls')
    assert e.value.code == 2
    assert '--cytonorm-control' in capsys.readouterr().err
    assert calls['order'] == []


def test_the_parsed_options_reach_the_manifest(calls, monkeypatch, trial):
    assert _main(monkeypatch, '--trials', trial, '--out', 'o', '--groups', G1,
                 '--k', '11', '--seed', '7') == 0
    args = _manifest(calls)[0]
    assert (args.k, args.seed, args.out, args.groups) == (11, 7, 'o', G1)


# ── _configure_logging ──────────────────────────────────────────────────────
@pytest.mark.parametrize('flags, level', [
    ((), logging.INFO), (('-v',), logging.INFO), (('-vv',), logging.DEBUG),
    (('-q',), logging.WARNING), (('-q', '-vv'), logging.WARNING)])
def test_log_level_from_verbose_and_quiet(calls, monkeypatch, trial,
                                          root_logging, flags, level):
    stale = logging.NullHandler()
    root_logging.addHandler(stale)
    _main(monkeypatch, '--trials', trial, '--out', 'o', '--groups', G1, *flags)
    assert root_logging.level == level
    assert stale not in root_logging.handlers           # force=True
    (h,) = root_logging.handlers
    assert h.formatter._fmt == '%(asctime)s %(levelname)-7s %(name)s: %(message)s'
    assert h.formatter.datefmt == '%H:%M:%S'


# ── --unmix ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize('flag, level', [('-q', logging.WARNING),
                                         ('-vv', logging.DEBUG)])
def test_unmix_returns_its_code_after_logging_and_runs_nothing_else(
        calls, monkeypatch, root_logging, flag, level):
    root_logging.setLevel(logging.CRITICAL)
    assert _main(monkeypatch, '--unmix', flag, '--out', 'u') == 5
    assert calls['order'] == ['run_batch_unmix']
    (args,), _kw = calls['run_batch_unmix'][0]
    assert args.unmix and args.out == 'u'
    assert root_logging.level == level


def test_unmix_exits_before_the_groups_are_resolved(calls, monkeypatch,
                                                    tmp_path, capsys):
    _fcs(tmp_path / 'day1' / 'x1.fcs')        # would make a by-day group
    assert _main(monkeypatch, '--unmix', '--out', 'u') == 5
    assert '[grouping]' not in capsys.readouterr().out


# ── _resolve_groups ─────────────────────────────────────────────────────────
def test_explicit_groups_and_fmo_sets_pass_through(calls, monkeypatch, trial):
    fmo = {'S': {'FL1-A': 'a1'}}
    _main(monkeypatch, '--trials', f' {trial} , ,', '--fcs', 'ignored',
          '--out', 'o', '--groups', G1, '--fmo-sets', json.dumps(fmo))
    kw = _run_kw(calls)
    assert kw['trial_dirs'] == [trial]
    assert kw['groups'] == json.loads(G1)
    assert kw['fmo_sets'] == fmo


def test_explicit_groups_default_the_fmo_sets(calls, monkeypatch, trial):
    _main(monkeypatch, '--fcs', trial, '--out', 'o', '--groups', G1)
    kw = _run_kw(calls)
    assert kw['trial_dirs'] == [trial]
    assert kw['fmo_sets'] == DEFAULT_FMO_SETS


def test_samples_split_on_the_late_prefix(calls, monkeypatch, trial):
    _main(monkeypatch, '--trials', trial, '--out', 'o',
          '--samples', 'a0, LATE_b0,,a1')
    kw = _run_kw(calls)
    assert kw['groups'] == [
        {'name': 'Group A', 'samples': ['a0', 'a1'], 'fmo_set': 'Set A'},
        {'name': 'Group B', 'samples': ['LATE_b0'], 'fmo_set': 'Set B'}]
    assert kw['fmo_sets'] == DEFAULT_FMO_SETS
    assert kw['trial_dirs'] == [trial]


@pytest.mark.parametrize('names, want', [
    ('late_b0', ['Group B']), ('a0,a1', ['Group A'])])
def test_samples_make_only_the_groups_they_fill(calls, monkeypatch, trial,
                                                names, want):
    fmo = {'X': {}}
    _main(monkeypatch, '--trials', trial, '--out', 'o', '--samples', names,
          '--fmo-sets', json.dumps(fmo))
    kw = _run_kw(calls)
    assert [g['name'] for g in kw['groups']] == want
    assert kw['fmo_sets'] == fmo


def test_folders_make_one_group_per_day_and_one_pass(
        calls, monkeypatch, tmp_path, capsys):
    _fcs(tmp_path / 'study' / 'day1' / 'x1.fcs')
    _fcs(tmp_path / 'study' / 'day2' / 'y1.fcs', seed=1)
    _main(monkeypatch, '--trials', 'study,elsewhere', '--out', 'o')
    kw = _run_kw(calls)
    assert [g['name'] for g in kw['groups']] == ['Day 1', 'Day 2']
    assert kw['trial_dirs'] == ['study']
    assert kw['fmo_sets'] == DEFAULT_FMO_SETS
    assert '[grouping] By-day: 2 folder(s) → Day 1, Day 2' in \
        capsys.readouterr().out


def test_folders_take_given_fmo_sets(calls, monkeypatch, tmp_path):
    _fcs(tmp_path / 'day1' / 'x1.fcs')
    _main(monkeypatch, '--out', 'o', '--fmo-sets', '{"Q": {}}')
    kw = _run_kw(calls)
    assert kw['fmo_sets'] == {'Q': {}}
    assert kw['trial_dirs'] == ['.']


def test_no_fcs_anywhere_falls_back_to_the_default_groups(
        calls, monkeypatch, tmp_path, capsys):
    (tmp_path / 'empty').mkdir()
    _main(monkeypatch, '--trials', 'empty', '--out', 'o',
          '--fmo-sets', '{"Q": {}}')
    kw = _run_kw(calls)
    assert kw['groups'] is DEFAULT_GROUPS
    assert kw['fmo_sets'] is DEFAULT_FMO_SETS
    assert kw['trial_dirs'] == ['empty']
    assert '[grouping]' not in capsys.readouterr().out


# ── _refuse_bad_groups ──────────────────────────────────────────────────────
def test_bad_explicit_groups_exit_2_naming_each_problem(
        calls, monkeypatch, trial, capsys):
    spec = json.dumps([{'name': 'A', 'samples': ['a0', 'a0']},
                       {'name': 'A', 'samples': ['a1']}])
    assert _main(monkeypatch, '--trials', trial, '--out', 'o',
                 '--groups', spec) == 2
    out = [ln for ln in capsys.readouterr().out.splitlines() if ln]
    assert len(out) == 2 and all(ln.startswith('[!] --groups: ') for ln in out)
    assert calls['order'] == []


def test_bad_samples_exit_2_naming_samples(calls, monkeypatch, trial,
                                           tmp_path, capsys):
    # run from another folder: the check must look in the --trials folder
    (tmp_path / 'cwd').mkdir()
    monkeypatch.chdir(tmp_path / 'cwd')
    assert _main(monkeypatch, '--trials', str(tmp_path / trial), '--out', 'o',
                 '--samples', 'a0,A0') == 2
    assert capsys.readouterr().out.startswith('[!] --samples: group ')
    assert calls['order'] == []


def test_bad_folder_groups_exit_2_naming_the_folders(
        calls, monkeypatch, tmp_path, capsys):
    _fcs(tmp_path / 'clash' / 'C1.fcs')
    _fcs(tmp_path / 'clash' / 'c1.fcs')
    if len(os.listdir(tmp_path / 'clash')) == 1:
        # Windows and macOS file systems ignore case: c1.fcs was written over
        # C1.fcs, so the folder holds one file and nothing clashes. The name
        # check that does run there is test_bad_samples_exit_2_naming_samples.
        pytest.skip('this file system ignores case, so C1.fcs and c1.fcs '
                    'cannot both exist')
    assert _main(monkeypatch, '--trials', 'clash', '--out', 'o') == 2
    assert '[!] groups from the folders: group ' in capsys.readouterr().out
    assert calls['order'] == []


def test_sound_groups_print_nothing(calls, monkeypatch, trial, capsys):
    assert _main(monkeypatch, '--trials', trial, '--out', 'o',
                 '--groups', G1) == 0
    assert '[!]' not in capsys.readouterr().out


def test_bad_groups_are_refused_before_the_gate_file_is_read(
        calls, monkeypatch, trial, tmp_path, capsys):
    path = _tree(tmp_path)
    assert _main(monkeypatch, '--trials', trial, '--out', 'o',
                 '--samples', 'a0,A0', '--gates-from', path,
                 '--gate-population', 'Pos') == 2
    assert '[gates]' not in capsys.readouterr().out


# ── _resolve_gates ──────────────────────────────────────────────────────────
TREE = [
    {'id': 'g1', 'parent_id': None, 'kind': 'interval', 'name': 'Cells',
     'channel': 'FSC-A', 'lo': 1.0, 'hi': 1e12},
    {'id': 'g2', 'parent_id': 'g1', 'kind': 'interval', 'name': 'Pos',
     'channel': 'FL1-A', 'lo': 0.5, 'hi': 1e12},
    {'id': 'g3', 'parent_id': 'g1', 'kind': 'interval', 'name': 'Other',
     'channel': 'FL2-A', 'lo': 0.5, 'hi': 1e12},
    {'id': 'g4', 'parent_id': 'g1', 'kind': 'boolean', 'op': 'not',
     'operands': ['g2'], 'name': 'Neg'},
]
THR = json.dumps([{'kind': 'threshold', 'channel': 'FL1-A', 'value': 0.5},
                  {'kind': 'interval', 'channel': 'FSC-A', 'lo': 2.0,
                   'hi': 3.0, 'name': 'big'}])


def _tree(tmp_path):
    (tmp_path / 'tree.json').write_text(
        json.dumps({'version': 2, 'gates': TREE}), encoding='utf-8')
    return 'tree.json'


def test_no_gates_mean_none_for_the_run_and_a_dict_for_the_outputs(
        calls, monkeypatch, trial):
    _main(monkeypatch, '--trials', trial, '--out', 'o', '--groups', G1,
          '--export-wsp', 'w.wsp')
    kw = _run_kw(calls)
    assert kw['gate_overrides'] is None and kw['region_gates'] == []
    assert _manifest(calls)[3] == {}
    (_a, wk), = calls['export_pipeline_workspace']
    assert wk['gate_overrides'] == {} and wk['region_gates'] == []


def test_gates_split_into_thresholds_and_regions(calls, monkeypatch, trial):
    _main(monkeypatch, '--trials', trial, '--out', 'o', '--groups', G1,
          '--gates', THR)
    kw = _run_kw(calls)
    assert kw['gate_overrides'] == {'FL1-A': 0.5}
    assert [g['name'] for g in kw['region_gates']] == ['big']
    assert _manifest(calls)[3] == {'FL1-A': 0.5}


def test_gate_units_reach_the_gate_parser(calls, monkeypatch, trial):
    _main(monkeypatch, '--trials', trial, '--out', 'o', '--groups', G1,
          '--gates', json.dumps([{'kind': 'threshold', 'channel': 'FL1-A',
                                  'value': 1000}]), '--gate-units', 'data')
    got = _run_kw(calls)['gate_overrides']['FL1-A']
    assert 0 < got < 1          # put on the stored (logicle) scale


def test_regions_alone_keep_the_empty_threshold_dict(calls, monkeypatch,
                                                     trial):
    _main(monkeypatch, '--trials', trial, '--out', 'o', '--groups', G1,
          '--gates', json.dumps([{'kind': 'interval', 'channel': 'FSC-A',
                                  'lo': 2.0, 'hi': 3.0}]))
    assert _run_kw(calls)['gate_overrides'] == {}


def test_gates_from_a_file_get_their_own_ids_after_the_cli_gates(
        calls, monkeypatch, trial, tmp_path, capsys):
    path = _tree(tmp_path)
    assert _main(monkeypatch, '--trials', trial, '--out', 'o', '--groups', G1,
                 '--gates', THR, '--gates-from', path,
                 '--gate-population', 'neg', '--export-wsp', 'w.wsp') == 0
    kw = _run_kw(calls)
    regions = kw['region_gates']
    assert regions[0]['name'] == 'big'
    picked = {g['id']: g for g in regions[1:]}
    assert set(picked) == {'file:g1', 'file:g2', 'file:g4'}
    assert picked['file:g1']['parent_id'] is None
    assert picked['file:g2']['parent_id'] == 'file:g1'
    assert picked['file:g4']['parent_id'] == 'file:g1'
    assert picked['file:g4']['operands'] == ['file:g2']
    assert kw['gate_overrides'] == {'FL1-A': 0.5}
    assert _manifest(calls)[4] is regions
    (_a, wk), = calls['export_pipeline_workspace']
    assert wk['region_gates'] is regions
    # the disabled operand is left out of the list printed
    assert '[gates] tree.json (template): Cells > Neg' in \
        capsys.readouterr().out


def test_gates_from_alone_keep_the_empty_threshold_dict(
        calls, monkeypatch, trial, tmp_path):
    _main(monkeypatch, '--trials', trial, '--out', 'o', '--groups', G1,
          '--gates-from', _tree(tmp_path), '--gate-population', 'Pos')
    kw = _run_kw(calls)
    assert kw['gate_overrides'] == {}
    assert [g['id'] for g in kw['region_gates']] == ['file:g1', 'file:g2']


def test_a_gates_from_tree_with_no_gate_says_so(calls, monkeypatch, trial,
                                                tmp_path, capsys):
    (tmp_path / 'none.json').write_text(
        json.dumps({'version': 2, 'gates': []}), encoding='utf-8')
    _main(monkeypatch, '--trials', trial, '--out', 'o', '--groups', G1,
          '--gates-from', 'none.json')
    assert '[gates] none.json (template): no gates' in capsys.readouterr().out
    assert _run_kw(calls)['gate_overrides'] is None


@pytest.mark.parametrize('file, pop, says', [
    ('tree.json', '', 'several populations'),
    ('tree.json', 'nothing', "no population named 'nothing'"),
    ('missing.json', '', 'No such file'),
])
def test_unreadable_gates_from_exits_2(calls, monkeypatch, trial, tmp_path,
                                       capsys, file, pop, says):
    _tree(tmp_path)
    assert _main(monkeypatch, '--trials', trial, '--out', 'o', '--groups', G1,
                 '--gates-from', str(tmp_path / file),
                 '--gate-population', pop) == 2
    out = capsys.readouterr().out
    assert out.startswith(f'[!] --gates-from {file}: ') and says in out
    assert calls['order'] == []


def test_a_bad_gate_file_is_refused_before_the_panel_is_read(
        calls, monkeypatch, trial, tmp_path, capsys):
    _panel(tmp_path / 'p.xlsx')
    assert _main(monkeypatch, '--trials', trial, '--out', 'o', '--groups', G1,
                 '--gates-from', 'missing.json', '--panel', 'p.xlsx') == 2
    assert '[panel]' not in capsys.readouterr().out


# ── _channel_labels, pairs, fast graph, palette ─────────────────────────────
def _panel(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([['Marker', 'Fluor'], ['CD901b', 'FL1'], ['CD902', 'FL2']]) \
        .to_excel(path, header=False, index=False)


def test_panel_labels_then_explicit_labels_win(calls, monkeypatch, tmp_path,
                                               capsys):
    _fcs(tmp_path / 'study' / 'day1' / 'x1.fcs')
    _panel(tmp_path / 'study' / 'staining panel.xlsx')
    # run from another folder: the panel is found from the --trials folder
    (tmp_path / 'cwd').mkdir()
    monkeypatch.chdir(tmp_path / 'cwd')
    _main(monkeypatch, '--trials', str(tmp_path / 'study' / 'day1'),
          '--out', 'o', '--groups',
          json.dumps([{'name': 'G', 'samples': 'x1'}]), '--panel', 'auto',
          '--labels', 'FL1-A=L1; FL3-A=L3')
    assert _run_kw(calls)['labels'] == {'FL1-A': 'L1', 'FL2-A': 'CD902',
                                        'FL3-A': 'L3'}
    assert '[panel] Auto-detected: ' in capsys.readouterr().out


def test_panel_probe_reads_a_by_day_groups_own_folder(calls, monkeypatch,
                                                      tmp_path):
    _fcs(tmp_path / 'study' / 'day1' / 'x1.fcs')
    _panel(tmp_path / 'p.xlsx')
    _main(monkeypatch, '--trials', 'study', '--out', 'o', '--panel', 'p.xlsx')
    assert _run_kw(calls)['labels'] == {'FL1-A': 'CD901b', 'FL2-A': 'CD902'}


def test_no_panel_found_leaves_only_explicit_labels(calls, monkeypatch, trial,
                                                    capsys):
    _main(monkeypatch, '--trials', trial, '--out', 'o', '--groups', G1,
          '--panel', 'auto', '--labels', 'FL3-A=L3')
    assert _run_kw(calls)['labels'] == {'FL3-A': 'L3'}
    assert '[panel]' not in capsys.readouterr().out


def test_a_failed_panel_probe_is_reported_and_labels_still_apply(
        calls, monkeypatch, trial, tmp_path, capsys):
    _panel(tmp_path / 'p.xlsx')
    _main(monkeypatch, '--trials', trial, '--out', 'o', '--groups',
          json.dumps([{'name': 'A', 'samples': ['nope']}]),
          '--panel', 'p.xlsx', '--labels', 'FL3-A=L3')
    assert _run_kw(calls)['labels'] == {'FL3-A': 'L3'}
    assert '[panel] Could not apply panel: FileNotFoundError: ' in \
        capsys.readouterr().out


def test_pairs_reach_the_run(calls, monkeypatch, trial):
    _main(monkeypatch, '--trials', trial, '--out', 'o', '--groups', G1,
          '--pairs', 'FL1-A:FL2-A')
    assert _run_kw(calls)['pairs'] == [('FL1-A', 'FL2-A')]


@pytest.mark.parametrize('method, warned', [('phenograph', True),
                                            ('leiden', False)])
def test_fast_graph_note_only_off_leiden(calls, monkeypatch, trial, capsys,
                                         method, warned):
    _main(monkeypatch, '--trials', trial, '--out', 'o', '--groups', G1,
          '--fast-graph', '--cluster-method', method)
    out = capsys.readouterr().out
    assert (f'[!] --fast-graph applies only to --cluster-method leiden; '
            f'{method} runs on its usual graph.' in out) == warned


def test_no_fast_graph_no_note(calls, monkeypatch, trial, capsys):
    _main(monkeypatch, '--trials', trial, '--out', 'o', '--groups', G1)
    assert '--fast-graph' not in capsys.readouterr().out


@pytest.mark.parametrize('palette, want', [('tab10', 'tab10'),
                                           ('auto', 'unset'), ('', 'unset')])
def test_palette_is_set_process_wide_before_the_run(calls, monkeypatch, trial,
                                                    palette, want):
    monkeypatch.setattr(flow_sample, '_DEFAULT_CATEGORICAL_PALETTE', 'unset')
    seen = []
    monkeypatch.setattr(cli_main, 'run', lambda **kw: seen.append(
        flow_sample._DEFAULT_CATEGORICAL_PALETTE) or 0)
    _main(monkeypatch, '--trials', trial, '--out', 'o', '--groups', G1,
          '--palette', palette)
    assert seen == [want]


# ── _run_analysis ───────────────────────────────────────────────────────────
def test_every_option_reaches_run(calls, monkeypatch, trial):
    groups = [{'name': 'A', 'samples': ['a0'], 'fmo_set': ''},
              {'name': 'B', 'samples': ['a1'], 'fmo_set': ''}]
    _main(monkeypatch, '--trials', trial, '--out', 'o',
          '--groups', json.dumps(groups), '--fmo-sets', '{"S": {}}',
          '--batch-mode', 'concatenate', '--workers', '3', '--k', '11',
          '--cluster-method', 'leiden', '--resolution', '0.5',
          '--n-metaclusters', '4', '--no-reproducible', '--fast-graph',
          '--dpi', '90', '--fmo-percentile', '97.5', '--labels', 'FL1-A=L1',
          '--gates', THR, '--show-plots', '--skip-existing',
          '--max-events', '123', '--admission-ram', '2.5',
          '--admission-vram', '0.5', '--filter-debris-um', '3',
          '--bead-um', '6', '--doublet-tol', '0.1', '--seed', '7',
          '--palette', 'tab10', '--pairs', 'FL1-A:FL2-A', '--batch-correct',
          '--cytonorm-mode', 'controls', '--cytonorm-control', 'a0',
          '--cytonorm-metaclusters', '5', '--qc', '--umap-components', '3')
    kw = _run_kw(calls)
    region = kw.pop('region_gates')
    assert [g['name'] for g in region] == ['big']
    assert kw == {
        'trial_dirs': [trial], 'out_dir': 'o', 'groups': groups,
        'fmo_sets': {'S': {}}, 'batch_mode': 'concatenate', 'k': 11,
        'cluster_method': 'leiden', 'resolution': 0.5, 'n_metaclusters': 4,
        'reproducible': False, 'fast_graph': True, 'dpi': 90,
        'fmo_percentile': 97.5, 'labels': {'FL1-A': 'L1'},
        'gate_overrides': {'FL1-A': 0.5}, 'show_plots': True,
        'skip_existing': True, 'workers': 3, 'max_events': 123,
        'admission_gb': 2.5, 'vram_admission_gb': 0.5,
        'filter_debris_um': 3.0, 'bead_um': 6.0, 'doublet_tol': 0.1,
        'random_state': 7, 'palette_name': 'tab10',
        'pairs': [('FL1-A', 'FL2-A')], 'batch_correct': True,
        'cytonorm_mode': 'controls', 'cytonorm_control': 'a0',
        'cytonorm_metaclusters': 5, 'qc': True, 'umap_components': 3}
    assert calls['run'][0][0] == ()


def test_default_options_reach_run(calls, monkeypatch, trial):
    _main(monkeypatch, '--trials', trial, '--out', 'o', '--groups', G1)
    kw = _run_kw(calls)
    assert kw['max_events'] is None          # 0 means every event
    assert kw['reproducible'] is True
    assert kw['fast_graph'] is True and kw['qc'] is False   # Leiden fast graph
    assert kw['umap_components'] == 2


# ── _write_outputs and the return code ──────────────────────────────────────
def test_outputs_follow_the_run_with_its_code_and_start(
        calls, monkeypatch, trial):
    calls['rc'] = 3
    before = _dt.datetime.now(_dt.UTC)
    assert _main(monkeypatch, '--trials', trial, '--out', 'o', '--groups', G1,
                 '--gates', THR, '--export-wsp', 'w.wsp', '--xlsx') == 3
    after = _dt.datetime.now(_dt.UTC)
    assert calls['order'] == ['run', 'export_pipeline_workspace',
                              'write_tables_workbook', 'write_run_manifest']
    args, dirs, groups, overrides, regions, rc, started = _manifest(calls)
    assert args.out == 'o' and dirs == [trial] and rc == 3
    assert groups == json.loads(G1)
    assert overrides == {'FL1-A': 0.5} and regions[0]['name'] == 'big'
    assert before <= started <= after and started.tzinfo is _dt.UTC
    (a, kw), = calls['write_tables_workbook']
    assert kw == {} and a == ('o', started.timestamp() - 2)
    (a, kw), = calls['export_pipeline_workspace']
    assert a == ()
    assert kw == {'out_path': 'w.wsp', 'trial_dirs': [trial],
                  'groups': json.loads(G1), 'gate_overrides': {'FL1-A': 0.5},
                  'region_gates': regions}
    assert _run_kw(calls)['region_gates'] is regions


def test_no_wsp_or_xlsx_unless_asked(calls, monkeypatch, trial):
    assert _main(monkeypatch, '--trials', trial, '--out', 'o',
                 '--groups', G1) == 0
    assert calls['order'] == ['run', 'write_run_manifest']


def test_an_unwritable_manifest_is_reported_and_the_code_kept(
        calls, monkeypatch, trial, capsys):
    calls['rc'] = 4

    def refuse(*a):
        raise OSError('disk full')
    monkeypatch.setattr(cli_main, 'write_run_manifest', refuse)
    assert _main(monkeypatch, '--trials', trial, '--out', 'o',
                 '--groups', G1) == 4
    assert '[manifest] not written: disk full' in capsys.readouterr().out


def test_a_generated_run_writes_its_workbook_and_manifest(
        monkeypatch, tmp_path, trial, root_logging):
    """The real writers, with run() writing one table."""
    monkeypatch.chdir(tmp_path)

    def fake_run(**kw):
        (tmp_path / 'o' / 'A').mkdir(parents=True)
        pd.DataFrame({'x': [1, 2]}).to_csv(tmp_path / 'o' / 'A' / 't.csv',
                                           index=False)
        return 0
    monkeypatch.setattr(cli_main, 'run', fake_run)
    assert _main(monkeypatch, '--trials', trial, '--out', 'o', '--groups', G1,
                 '--xlsx', '-q') == 0
    assert (tmp_path / 'o' / 'openflo_tables.xlsx').is_file()
    m = json.loads((tmp_path / 'o' / 'run_manifest.json').read_text('utf-8'))
    assert m['exit_code'] == 0
    assert sorted(o['path'] for o in m['outputs']) == [
        'A/t.csv', 'openflo_tables.xlsx']
    assert [i['path'].replace('\\', '/') for i in m['inputs']] == \
        ['trial/a0.fcs']
