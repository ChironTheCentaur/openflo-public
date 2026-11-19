"""cli_run.run() after its split into helpers (R6 slice D3): each step it
hands to a helper still happens, in the same order, with the same values,
and the exit codes and early exits are unchanged.

The mode runners, fit_cytonorm, the GPU probe, psutil and the plot calls
are recorders here, so these tests run no pipeline; the sample files are
small generated FCS files (synthetic channels and labels only).
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

import openflo.cli_run as cli_run
from openflo import flow_sample, gpu_probe
from openflo.cli_defaults import DEFAULT_FMO_SETS
from tests._fcs_writer import write_float_fcs

CH = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A']
MK = ['', '', 'MkA', 'MkB']
G2 = [{'name': 'Ctrl', 'samples': ['c1', 'c2'], 'fmo_set': ''},
      {'name': 'Treat', 'samples': ['t1'], 'fmo_set': ''}]


def _write(path, seed):
    rng = np.random.default_rng(seed)
    ev = rng.lognormal(8, .3, size=(50, len(CH))).astype(np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_float_fcs(path, ev, CH, opt_channel_names=MK)


@pytest.fixture
def trials(tmp_path):
    out = []
    for t in ('trialA', 'trialB'):
        for i, nm in enumerate(('c1', 'c2', 't1')):
            _write(str(tmp_path / t / f'expt_{nm}.fcs'), i)
        out.append(str(tmp_path / t))
    return out


@pytest.fixture
def rec(monkeypatch):
    """Recorders for everything run() calls out to. `rec['published']` is
    what the mode runner returns; `rec['reports']` how often it reports."""
    r = {'calls': [], 'published': True, 'reports': 1, 'saved': 0}

    def runner(name):
        def fn(*a, **k):
            r['calls'].append((name, a, k))
            for i in range(r['reports']):
                a[18](f'unit {i}')
            return r['published']
        return fn

    def fit(*a, **k):
        print('FIT', flush=True)
        r['calls'].append(('fit_cytonorm', a, k))
        return {'model': 1}, None

    def saved(out_dir):
        r['calls'].append(('show_saved_group_plots', (out_dir,), {}))
        return r['saved']

    monkeypatch.setattr(cli_run, 'run_independent', runner('independent'))
    monkeypatch.setattr(cli_run, 'run_concatenated', runner('concatenate'))
    monkeypatch.setattr(cli_run, 'fit_cytonorm', fit)
    monkeypatch.setattr(cli_run, 'gpu_present', lambda: False)
    monkeypatch.setattr(cli_run, 'show_saved_group_plots', saved)
    monkeypatch.setattr(cli_run.plt, 'show',
                        lambda: r['calls'].append(('show', (), {})))
    monkeypatch.setattr(flow_sample, 'set_default_palette',
                        lambda p: r['calls'].append(('palette', (p,), {})))
    monkeypatch.setattr(gpu_probe, 'GPU_AVAILABLE', False)
    monkeypatch.setattr(gpu_probe, 'GPU_CLUSTERING_AVAILABLE', False)
    monkeypatch.setitem(sys.modules, 'psutil', _fake_psutil(512.0))
    monkeypatch.setattr(os, 'cpu_count', lambda: 4)
    cli_run.plt.close('all')
    yield r
    cli_run.plt.close('all')


def _fake_psutil(free_gb):
    return types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(
        available=free_gb * 1024 ** 3))


def _names(rec):
    return [c[0] for c in rec['calls']]


def _runner(rec):
    (call,) = [c for c in rec['calls'] if c[0] in ('independent', 'concatenate')]
    return call


# ── trial folders and groups ─────────────────────────────────────────────

def test_clean_trial_dirs_makes_a_list_without_blanks():
    assert cli_run._clean_trial_dirs('a') == ['a']
    assert cli_run._clean_trial_dirs(['', 'a', None, 'b']) == ['a', 'b']
    assert cli_run._clean_trial_dirs(()) == []


def test_a_str_trial_reaches_the_runner_as_a_list(trials, tmp_path, rec):
    assert cli_run.run(trials[0], str(tmp_path / 'o'), G2) == 0
    assert _runner(rec)[1][0] == [trials[0]]


def test_blank_trials_are_dropped_before_the_runner(trials, tmp_path, rec):
    cli_run.run(['', trials[0], ''], str(tmp_path / 'o'), G2)
    assert _runner(rec)[1][0] == [trials[0]]


def test_no_groups_returns_0_before_anything_else(trials, tmp_path, rec,
                                                  capsys):
    out = tmp_path / 'o'
    rc = cli_run.run(trials, str(out), [{'name': '', 'samples': ['c1']}])
    assert rc == 0 and type(rc) is int
    assert capsys.readouterr().out == '[!] No groups defined — nothing to do.\n'
    assert rec['calls'] == [] and not out.exists()


def test_invalid_groups_raise_before_the_out_folder_is_made(trials, tmp_path,
                                                            rec):
    out = tmp_path / 'o'
    with pytest.raises(ValueError, match="invalid groups: 2 groups are named"):
        cli_run.run(trials, str(out), G2 + [G2[0]])
    assert rec['calls'] == [] and not out.exists()


def test_check_groups_names_a_file_listed_twice(trials):
    with pytest.raises(ValueError, match="lists c1 and expt_c1"):
        cli_run._check_groups([{'name': 'X', 'samples': ['c1', 'expt_c1'],
                                'fmo_set': ''}], trials[:1])
    cli_run._check_groups([{'name': 'X', 'samples': ['c1'], 'fmo_set': ''}],
                          trials[:1])


def test_fmo_sets_default_and_given(trials, tmp_path, rec):
    cli_run.run(trials[:1], str(tmp_path / 'o'), G2)
    assert _runner(rec)[1][3] is DEFAULT_FMO_SETS
    rec['calls'].clear()
    mine = {'s': {'FL1-A': 'x'}}
    cli_run.run(trials[:1], str(tmp_path / 'o'), G2, fmo_sets=mine)
    assert _runner(rec)[1][3] is mine


# ── batch correction ─────────────────────────────────────────────────────

@pytest.mark.parametrize('mode', ['independent', 'concatenate'])
def test_batch_correction_is_fitted_once_and_handed_on(trials, tmp_path, rec,
                                                       capsys, mode):
    cli_run.run(trials, str(tmp_path / 'o'), G2, batch_mode=mode,
                batch_correct=True, cytonorm_mode='mq',
                cytonorm_metaclusters=7, cytonorm_control='ctl', qc='QC')
    assert _names(rec) == ['fit_cytonorm', mode]
    _n, a, k = rec['calls'][0]
    assert [g['name'] for g in a[0]] == ['Ctrl', 'Treat']
    assert a[1] == trials[0]
    assert k == {'mode': 'mq', 'n_metaclusters': 7, 'control_token': 'ctl',
                 'qc': 'QC'}
    assert _runner(rec)[2]['cytonorm'] == {'model': 1}
    out = capsys.readouterr().out
    assert out.index('FIT') < out.index('[GPU]')


def test_batch_correction_with_no_trials_uses_the_current_folder(tmp_path,
                                                                 rec):
    cli_run.run([], str(tmp_path / 'o'), G2, batch_correct=True)
    assert rec['calls'][0][1][1] == '.'
    assert cli_run._fit_batch_correction(
        G2, ['t1', 't2'], 'm', 3, 'c', False) == {'model': 1}
    assert rec['calls'][-1][1][1] == 't1'


def test_no_batch_correction_hands_none_on(trials, tmp_path, rec):
    cli_run.run(trials[:1], str(tmp_path / 'o'), G2)
    assert 'fit_cytonorm' not in _names(rec)
    assert _runner(rec)[2]['cytonorm'] is None


# ── GPU line ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize('avail, clus, present, want', [
    (True, True, False,
     ['[GPU] GPU-X detected — running UMAP + clustering (cuML + cuGraph)',
      '[GPU] Per-worker VRAM admission: ≥ 2.5 GB free to use GPU clustering '
      '(else CPU Phenograph).']),
    (True, False, False,
     ['[GPU] GPU-X detected — running UMAP (UMAP only)']),
    (False, False, True,
     ['[GPU] NVIDIA GPU detected, but RAPIDS cuML is not installed — UMAP & '
      'clustering run on CPU. Install cuML+cuGraph (Linux/WSL2) for GPU '
      'acceleration.']),
    (False, False, False,
     ['[GPU] No CUDA GPU detected — UMAP will run on CPU (numba JIT)']),
])
def test_gpu_status_lines(trials, tmp_path, rec, capsys, monkeypatch, avail,
                          clus, present, want):
    monkeypatch.setattr(gpu_probe, 'GPU_AVAILABLE', avail)
    monkeypatch.setattr(gpu_probe, 'GPU_CLUSTERING_AVAILABLE', clus)
    monkeypatch.setattr(gpu_probe, 'GPU_NAME', 'GPU-X')
    monkeypatch.setattr(cli_run, 'gpu_present', lambda: present)
    cli_run.run(trials[:1], str(tmp_path / 'o'), G2, vram_admission_gb=2.5)
    lines = capsys.readouterr().out.splitlines()
    assert [ln for ln in lines if ln.startswith('[GPU]')] == want
    assert lines[:len(want)] == want


def test_gpu_line_prints_before_the_worker_count_is_read(trials, tmp_path,
                                                         rec, capsys):
    with pytest.raises(ValueError):
        cli_run.run(trials[:1], str(tmp_path / 'o'), G2, workers='many')
    assert capsys.readouterr().out.startswith('[GPU] No CUDA GPU')


# ── workers, RAM throttle, header ────────────────────────────────────────

@pytest.mark.parametrize('workers, free_gb, max_events, want, n_jobs', [
    (1, 512.0, None, 1, -1),
    (4, 512.0, None, 4, 1),
    (0, 512.0, None, 1, -1),
    ('3', 512.0, None, 3, 1),
    (8, 6.0, None, 1, 1),         # 12.5 GB a worker, 4 GB budget
    (8, 6.0, 100_000, 4, 1),      # 1 GB a worker (the floor)
    (8, 24.0, 2_500_000, 2, 1),   # 10.5 GB a worker, 22 GB budget
    (3, 0.5, None, 1, 1),         # budget floor of 1 GB
])
def test_workers_throttle_and_n_jobs(trials, tmp_path, rec, capsys,
                                     monkeypatch, workers, free_gb,
                                     max_events, want, n_jobs):
    monkeypatch.setitem(sys.modules, 'psutil', _fake_psutil(free_gb))
    cli_run.run(trials[:1], str(tmp_path / 'o'), G2, workers=workers,
                max_events=max_events)
    a = _runner(rec)[1]
    assert (a[10], a[11], a[12]) == (want, n_jobs, max_events)
    out = capsys.readouterr().out
    assert ('[RAM-throttle]' in out) == (want < max(1, int(workers)))


def test_throttle_message_and_return():
    sys_mod = sys.modules.get('psutil')
    sys.modules['psutil'] = _fake_psutil(6.0)
    try:
        assert cli_run._throttle_workers(8, 100_000) == 4
        assert cli_run._throttle_workers(2, 100_000) == 2
    finally:
        if sys_mod is None:
            sys.modules.pop('psutil', None)
        else:
            sys.modules['psutil'] = sys_mod


def test_throttle_is_skipped_without_psutil(trials, tmp_path, rec,
                                            monkeypatch):
    monkeypatch.setitem(sys.modules, 'psutil', None)
    cli_run.run(trials[:1], str(tmp_path / 'o'), G2, workers=6)
    assert _runner(rec)[1][10] == 6


def test_cores_fall_back_to_2(trials, tmp_path, rec, capsys, monkeypatch):
    monkeypatch.setattr(os, 'cpu_count', lambda: None)
    cli_run.run(trials[:1], str(tmp_path / 'o'), G2)
    assert '(2 cores)' in capsys.readouterr().out


def test_header_lines(trials, tmp_path, rec, capsys):
    cli_run.run(trials, str(tmp_path / 'o'), G2, batch_mode='concatenate',
                workers=2, max_events=1_234_567)
    lines = capsys.readouterr().out.splitlines()
    i = next(i for i, ln in enumerate(lines) if ln.startswith('[Trials]'))
    assert lines[i:i + 2] == [
        '[Trials] 2 trial folder(s)  |  mode: concatenate  |  parallel '
        'workers: 2  (4 cores)  |  cluster cap: 1,234,567 events',
        '[Groups] Ctrl(2) / Treat(1)']
    assert lines[i - 1].startswith('[GPU]')
    assert lines[i + 2].startswith('[STEP 1/')


def test_print_run_header_without_a_cap(capsys):
    cli_run._print_run_header(1, None, [{'name': 'A', 'samples': ['x']}],
                              'independent', 3, 8)
    assert capsys.readouterr().out == (
        '[Trials] 1 trial folder(s)  |  mode: independent  |  parallel '
        'workers: 3  (8 cores)\n[Groups] A(1)\n')


# ── step total ───────────────────────────────────────────────────────────

FMO_GROUPS = [{'name': 'A', 'samples': ['c1', 'c2'], 'fmo_set': 's1',
               'sample_fmo': {'c2': 's2'}},
              {'name': 'B', 'samples': ['t1'], 'fmo_set': 's1'}]


@pytest.mark.parametrize('mode, n_trials, groups, total', [
    ('independent', 2, FMO_GROUPS, 18),   # (2 sets + 3 + 2 + 1 + 1) x 2
    ('concatenate', 2, FMO_GROUPS, 9),
    ('independent', 1, G2, 7),            # 0 sets + 3 + 2 + 1 + 1
    ('independent', 2, G2[:1], 6),        # (2 + 1) x 2: no shared, no compare
    ('independent', 0, G2, 1),            # never 0
])
def test_step_total(trials, tmp_path, rec, capsys, mode, n_trials, groups,
                    total):
    rec['reports'] = 2
    cli_run.run(trials[:n_trials], str(tmp_path / 'o'), groups,
                batch_mode=mode)
    lines = capsys.readouterr().out.splitlines()
    assert [ln for ln in lines if ln.startswith('[STEP')] == [
        f'[STEP 1/{total}] unit 0', f'[STEP 2/{total}] unit 1']


def test_count_steps_direct():
    from openflo.cli_samples import normalise_groups
    g = normalise_groups(FMO_GROUPS)
    assert cli_run._count_steps(g, 'independent', 3) == 27
    assert cli_run._count_steps(g, 'concatenate', 3) == 9
    assert cli_run._count_steps([], 'independent', 3) == 1


# ── palette, cluster settings, every option handed on ───────────────────

@pytest.mark.parametrize('name, want', [('Set1', [('palette', ('Set1',), {})]),
                                        ('auto', []), ('', []), (None, [])])
def test_palette(trials, tmp_path, rec, name, want):
    cli_run.run(trials[:1], str(tmp_path / 'o'), G2, palette_name=name)
    assert [c for c in rec['calls'] if c[0] == 'palette'] == want
    assert _names(rec)[-1] == 'independent'


@pytest.mark.parametrize('mode', ['independent', 'concatenate', 'other'])
def test_every_option_reaches_the_runner(trials, tmp_path, rec, mode):
    s = {n: object() for n in (
        'k', 'fmo_percentile', 'labels', 'gate_overrides', 'region_gates',
        'skip_existing', 'admission_gb', 'vram_admission_gb',
        'filter_debris_um', 'bead_um', 'doublet_tol', 'random_state', 'pairs',
        'qc', 'dpi', 'cluster_method', 'resolution', 'n_metaclusters',
        'reproducible')}
    fmo = {'x': {}}
    out = str(tmp_path / 'o')
    cli_run.run(trials, out, G2, fmo_sets=fmo, batch_mode=mode, workers=2,
                max_events=777, fast_graph='yes', show_plots=0, **s)
    name, a, k = _runner(rec)
    assert name == ('concatenate' if mode == 'concatenate' else 'independent')
    assert a[0] == trials and a[1] == out
    assert [g['name'] for g in a[2]] == ['Ctrl', 'Treat']
    assert a[3] is fmo
    assert a[4:10] == (s['k'], s['fmo_percentile'], s['labels'],
                       s['gate_overrides'], s['region_gates'],
                       s['skip_existing'])
    assert a[10:13] == (2, 1, 777)
    assert a[13:18] == (s['admission_gb'], s['vram_admission_gb'],
                        s['filter_debris_um'], s['bead_um'], s['doublet_tol'])
    assert callable(a[18]) and len(a) == 19
    assert k == {'random_state': s['random_state'], 'pairs': s['pairs'],
                 'cytonorm': None,
                 'cluster_cfg': {'method': s['cluster_method'],
                                 'resolution': s['resolution'],
                                 'n_metaclusters': s['n_metaclusters'],
                                 'reproducible': s['reproducible'],
                                 'fast_graph': True},
                 'qc': s['qc'], 'dpi': s['dpi'], 'keep_open': False}
    assert k['keep_open'] is False and k['cluster_cfg']['fast_graph'] is True


# ── the end of the run: messages, plots, exit code ───────────────────────

@pytest.mark.parametrize('mode', ['independent', 'concatenate'])
@pytest.mark.parametrize('published, rc', [(True, 0), (False, 1)])
def test_exit_code_and_closing_lines(trials, tmp_path, rec, capsys, mode,
                                     published, rc):
    rec['published'] = published
    out = str(tmp_path / 'o')
    got = cli_run.run(trials[:1], out, G2, batch_mode=mode)
    assert got == rc and type(got) is int
    tail = capsys.readouterr().out.splitlines()
    want = ['', f'Done. Results in: {out}']
    if not published:
        want.append('[!] Some group comparisons were not written; see the '
                    'messages above.')
    assert tail[-len(want):] == want
    assert 'show' not in _names(rec)


@pytest.mark.parametrize('mode', ['independent', 'concatenate'])
@pytest.mark.parametrize('saved, note', [(0, True), (2, False)])
def test_show_plots_with_nothing_drawn(trials, tmp_path, rec, capsys, mode,
                                       saved, note):
    rec['saved'] = saved
    out = str(tmp_path / 'o')
    assert cli_run.run(trials[:1], out, G2, batch_mode=mode,
                       show_plots=1) == 0
    assert _names(rec) == [mode, 'show_saved_group_plots', 'show']
    assert rec['calls'][1][1] == (out,)
    assert _runner(rec)[2]['keep_open'] is True
    text = capsys.readouterr().out
    assert ('[Show plots] No figures to display — nothing fresh this run and '
            f'no saved group-level PNGs found in {out}.' in text) == note
    assert text.index('unit 0') < text.index('Done. Results in')


def test_show_plots_with_a_figure_open(trials, tmp_path, rec):
    cli_run.plt.figure()
    cli_run.run(trials[:1], str(tmp_path / 'o'), G2, show_plots=True)
    assert _names(rec) == ['independent', 'show']


def test_finish_run_direct(tmp_path, rec, capsys):
    cli_run._finish_run('OUT', False, False)
    assert capsys.readouterr().out == (
        '\nDone. Results in: OUT\n[!] Some group comparisons were not '
        'written; see the messages above.\n')
    assert rec['calls'] == []


def test_unpublished_note_comes_before_the_plot_lines(trials, tmp_path, rec,
                                                      capsys):
    rec['published'] = False
    out = str(tmp_path / 'o')
    assert cli_run.run(trials[:1], out, G2, show_plots=True) == 1
    tail = capsys.readouterr().out.splitlines()[-4:]
    assert tail == [
        '', f'Done. Results in: {out}',
        '[!] Some group comparisons were not written; see the messages above.',
        '[Show plots] No figures to display — nothing fresh this run and no '
        f'saved group-level PNGs found in {out}.']
