"""Batch G (Speed 9): the CLI's batch work is spread over processes, with the
same files out.

- Each sample's plots and stats CSV are drawn in the worker that clustered it,
  not one at a time in the main process while the workers wait.
- Group UMAPs (independent, each seeded) are drawn in a process pool.
- ``compare`` runs its samples in a process pool, and reports them in
  workspace order.

Every check runs the same generated data both ways and compares the files
byte for byte (the positive half), and records which process did the work so
a run that quietly fell back to the main process fails (the negative half).
Synthetic sample names and generic channels only.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')

from openflo import cli, compare  # noqa: E402
from openflo.fcs_export import write_fcs  # noqa: E402


def _fcs(path, seed, n=600):
    rng = np.random.default_rng(seed)
    half = n // 2
    df = pd.DataFrame({
        'FSC-A': rng.uniform(2e4, 9e4, n),
        'SSC-A': rng.uniform(1e4, 5e4, n),
        'FL1-A': np.r_[rng.normal(200, 40, half), rng.normal(5000, 500,
                                                             n - half)],
        'FL2-A': np.r_[rng.normal(3000, 300, half), rng.normal(150, 30,
                                                               n - half)],
        'FL3-A': rng.exponential(400, n) + 50})
    write_fcs(df, str(path))
    return str(path)


def _task(path, name, group, out):
    return {'name': name, 'group': group, 'out': str(out),
            'paths': [(name, path)], 'labels': {}, 'thresholds': {},
            'k': 10, 'n_jobs': 1, 'method': 'leiden', 'max_events': 0,
            'random_state': 42}


def _files(root):
    out = {}
    for dirpath, _dirs, names in os.walk(root):
        for nm in names:
            p = os.path.join(dirpath, nm)
            out[os.path.relpath(p, root)] = open(p, 'rb').read()
    return out


def _pid_recorder(monkeypatch, tmp_path, name):
    """Wrap ``cli.<name>`` so every call appends its process id to a file.
    A forked worker inherits the wrapper (and records its own pid); a
    spawned one runs the original (and records nothing). Either way, the
    main process's pid in the file means the main process did the work."""
    log = tmp_path / f'{name}.pids'
    real = getattr(cli, name)

    def wrapper(*a, **k):
        with open(log, 'a', encoding='utf-8') as f:
            f.write(f'{os.getpid()}\n')
        return real(*a, **k)

    monkeypatch.setattr(cli, name, wrapper)
    return lambda: ({int(x) for x in log.read_text().split()}
                    if log.exists() else set())


def _run(tasks, workers):
    return cli._run_sample_tasks([dict(t) for t in tasks], workers, 0.0,
                                 lambda msg: None)


@pytest.fixture(scope='module')
def fcs_files(tmp_path_factory):
    d = tmp_path_factory.mktemp('fcs')
    return [_fcs(d / f'synthetic_{i}.fcs', seed=i) for i in (1, 2, 3)]


def test_parallel_samples_write_the_same_files_from_the_workers(
        fcs_files, tmp_path, monkeypatch):
    serial, parallel = tmp_path / 'serial', tmp_path / 'parallel'
    names = ['synthetic_1', 'synthetic_2', 'synthetic_3']
    groups = ['G1', 'G1', 'G2']
    res1 = _run([_task(p, n, g, serial) for p, n, g
                 in zip(fcs_files, names, groups, strict=True)], 1)
    pids = _pid_recorder(monkeypatch, tmp_path, '_write_sample_outputs')
    res2 = _run([_task(p, n, g, parallel) for p, n, g
                 in zip(fcs_files, names, groups, strict=True)], 2)
    a, b = _files(serial), _files(parallel)
    assert sorted(a) == sorted(b)
    assert any(k.endswith('_stats.csv') for k in a)
    assert any(k.endswith('_scatter.png') for k in a)
    for k in a:
        assert a[k] == b[k], f'{k} differs between 1 and 2 workers'
    # The same samples come back for the group outputs.
    assert ({g: sorted(s.name for s in ss) for g, ss in res1.items()}
            == {g: sorted(s.name for s in ss) for g, ss in res2.items()})
    assert os.getpid() not in pids(), 'the main process drew the plots'


def test_one_worker_still_writes_in_the_main_process(fcs_files, tmp_path,
                                                     monkeypatch):
    pids = _pid_recorder(monkeypatch, tmp_path, '_write_sample_outputs')
    _run([_task(fcs_files[0], 'synthetic_1', 'G1', tmp_path)], 1)
    assert pids() == {os.getpid()}
    assert (tmp_path / 'G1' / 'synthetic_1_stats.csv').is_file()


def test_a_failed_sample_writes_nothing_and_is_reported(tmp_path):
    seen = []
    bad = _task(str(tmp_path / 'missing.fcs'), 'synthetic_9', 'G1', tmp_path)
    good = _task(_fcs(tmp_path / 'synthetic_1.fcs', 1), 'synthetic_1', 'G1',
                 tmp_path)
    failures = {}
    res = cli._run_sample_tasks([bad, good], 2, 0.0, seen.append, failures)
    assert [s.name for s in res['G1']] == ['synthetic_1']
    assert list(failures) == ['G1\x1fsynthetic_9']
    assert any(m.startswith('Failed: synthetic_9') for m in seen)
    assert not (tmp_path / 'G1' / 'synthetic_9_stats.csv').exists()


def _clustered(fcs_files, tmp_path):
    res = _run([_task(p, f'synthetic_{i}', f'G{i}', tmp_path / 'samples')
                for i, p in enumerate(fcs_files[:2], 1)], 1)
    return [('G1', res['G1'], 'G1', 'G1'), ('G2', res['G2'], 'G2', 'G2')]


def test_group_umaps_drawn_in_parallel_match_one_at_a_time(
        fcs_files, tmp_path, monkeypatch):
    jobs = _clustered(fcs_files, tmp_path)
    a_dir, b_dir = tmp_path / 'a', tmp_path / 'b'
    seen = []
    cli._draw_group_plots(jobs, str(a_dir), 42, None, 1, seen.append)
    pids = _pid_recorder(monkeypatch, tmp_path, 'run_group_umap')
    cli._draw_group_plots(jobs, str(b_dir), 42, None, 2, seen.append)
    a, b = _files(a_dir), _files(b_dir)
    assert {'G1_umap.png', 'G2_umap.png'} <= set(a)
    assert sorted(a) == sorted(b)
    for k in a:
        assert a[k] == b[k], f'{k} differs between 1 and 2 workers'
    # Reported once per group, in group order, both ways.
    assert seen[:2] == seen[2:] and 'G1' in seen[0] and 'G2' in seen[1]
    assert os.getpid() not in pids(), 'the main process drew the UMAPs'


def test_shown_figures_keep_the_group_plots_in_this_process(
        fcs_files, tmp_path, monkeypatch):
    jobs = _clustered(fcs_files, tmp_path)
    monkeypatch.setattr(cli, '_KEEP_FINAL_PLOTS', True)
    pids = _pid_recorder(monkeypatch, tmp_path, 'run_group_umap')
    cli._draw_group_plots(jobs, str(tmp_path / 'c'), 42, None, 2,
                          lambda m: None)
    assert pids() == {os.getpid()}
    import matplotlib.pyplot as plt
    plt.close('all')


def _workspace(d, n=3):
    body = []
    for i in range(1, n + 1):
        rng = np.random.default_rng(i)
        write_fcs(pd.DataFrame({'FSC-A': rng.uniform(0, 10, 400),
                                'SSC-A': rng.uniform(0, 10, 400)}),
                  str(d / f'synthetic_{i}.fcs'))
        body.append(
            f'<Sample><DataSet uri="synthetic_{i}.fcs"/>'
            f'<SampleNode name="synthetic_{i}"><Subpopulations>'
            f'<Population name="Cells" count="{100 + i}"><Gate>'
            '<RectangleGate>'
            '<dimension min="1" max="9"><fcs-dimension name="FSC-A"/>'
            '</dimension><dimension min="1" max="9">'
            '<fcs-dimension name="SSC-A"/></dimension>'
            '</RectangleGate></Gate></Population></Subpopulations>'
            '</SampleNode></Sample>')
    wsp = d / 'synthetic.wsp'
    wsp.write_text('<Workspace><SampleList>' + ''.join(body)
                   + '</SampleList></Workspace>', encoding='utf-8')
    return str(wsp)


def test_compare_in_parallel_gives_the_same_report_in_order(tmp_path):
    wsp = _workspace(tmp_path)
    reader = compare.WspReader(wsp)
    loc = compare.FcsLocator(wsp, str(tmp_path))
    inv = compare._per_sample_inventory(reader, str(tmp_path),
                                        locate=loc.find)
    one = list(compare._compare_samples(wsp, str(tmp_path), inv, loc, 1))
    two = list(compare._compare_samples(wsp, str(tmp_path), inv, loc, 2))
    assert one == two
    assert [rows[0]['sample'] for _line, rows in two] == [
        'synthetic_1', 'synthetic_2', 'synthetic_3']
    # Counted, not copied: each sample's own count.
    counts = [rows[0]['openflo_count'] for _line, rows in two]
    assert len(set(counts)) == 3


def test_compare_worker_count(monkeypatch):
    monkeypatch.setattr(compare.os, 'cpu_count', lambda: 16)
    assert compare._compare_workers(0, 100) == 4     # capped by default
    monkeypatch.setattr(compare.os, 'cpu_count', lambda: 2)
    assert compare._compare_workers(0, 100) == 2     # never past the cores
    assert compare._compare_workers(3, 10) == 3
    assert compare._compare_workers(8, 2) == 2       # never past the samples
    assert compare._compare_workers(0, 1) == 1
    assert 1 <= compare._compare_workers(0, 100) <= 4
    assert compare._compare_workers(-2, 5) >= 1


def test_compare_pool_does_the_samples_in_workers(tmp_path, monkeypatch):
    wsp = _workspace(tmp_path)
    reader = compare.WspReader(wsp)
    loc = compare.FcsLocator(wsp, str(tmp_path))
    inv = compare._per_sample_inventory(reader, str(tmp_path),
                                        locate=loc.find)
    log = tmp_path / 'pids'
    real = compare._compare_one_sample

    def wrapper(*a, **k):
        with open(log, 'a', encoding='utf-8') as f:
            f.write(f'{os.getpid()}\n')
        return real(*a, **k)

    monkeypatch.setattr(compare, '_compare_one_sample', wrapper)
    list(compare._compare_samples(wsp, str(tmp_path), inv, loc, 2))
    pids = {int(x) for x in log.read_text().split()} if log.exists() else set()
    assert os.getpid() not in pids


def test_compare_skips_a_missing_file_without_stopping(tmp_path):
    wsp = _workspace(tmp_path)
    os.remove(tmp_path / 'synthetic_2.fcs')
    reader = compare.WspReader(wsp)
    loc = compare.FcsLocator(wsp, str(tmp_path))
    inv = compare._per_sample_inventory(reader, str(tmp_path),
                                        locate=loc.find)
    out = list(compare._compare_samples(wsp, str(tmp_path), inv, loc, 2))
    assert '[skip] synthetic_2' in out[1][0] and out[1][1] == []
    assert out[0][1] and out[2][1]
