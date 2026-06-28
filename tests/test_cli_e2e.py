"""End-to-end CLI tests.

Invoke ``python -m openflo.cli`` as a real subprocess. Catches the
class of regression unit tests miss: argument-parsing breakage,
console-script wiring, subprocess exits with traceback mid-pipeline,
missing output files, encoding issues, runtime-only dependency gaps.

Two tiers:

  - ``--help``-based wiring checks — fast (~3 s), always run. These
    catch the most common regression ("the console script doesn't even
    start") without paying for a pipeline run.

  - The full-pipeline smoke run — runs Phenograph + UMAP on the
    synthetic FCS. That's ~35 s warm but balloons unpredictably under
    a loaded parent process (numba JIT, matplotlib font cache, CPU
    contention). Opt-in via ``OPENFLO_RUN_SLOW_TESTS=1`` so it doesn't
    make the default suite flaky. Run it locally before a release:
        OPENFLO_RUN_SLOW_TESTS=1 pytest tests/test_cli_e2e.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

_SLOW = pytest.mark.skipif(
    os.environ.get('OPENFLO_RUN_SLOW_TESTS') != '1',
    reason="full-pipeline subprocess run — set OPENFLO_RUN_SLOW_TESTS=1")


def _run_cli(args, env_extra=None, timeout=300):
    """Run ``python -m openflo.cli ARGS`` and return the CompletedProcess.

    Forces ``MPLBACKEND=Agg`` so plot saves don't require a display on
    Linux CI runners. Decodes stdout/stderr as utf-8 with `replace`
    errors so log output containing non-ASCII characters (→, ≥, em-
    dashes) doesn't blow up on Windows cp1252 terminals.
    """
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, '-m', 'openflo.cli'] + list(args),
        env=env,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=timeout,
    )


def test_cli_help_exits_zero():
    """The cheapest possible sanity check — argparse can render --help
    without a traceback. Catches console-script wiring regressions
    (entry point name typos, missing main() function, broken module
    init)."""
    result = _run_cli(['--help'], timeout=30)
    assert result.returncode == 0, (
        f"--help exited {result.returncode}\n{result.stderr}")
    # Must mention every documented flag a user might grep for.
    for flag in ('--trials', '--out', '--groups', '--seed',
                 '--verbose', '--quiet'):
        assert flag in result.stdout, (
            f"--help output missing {flag}:\n{result.stdout}")


@_SLOW
def test_cli_smoke_run_against_synthetic_fcs(synthetic_fcs, tmp_path):
    """Full pipeline run against the synthetic FCS. Verifies:

      - The CLI parses our argv without complaint
      - The pipeline completes (load → QC → comp → transform →
        gates → cluster → stats → export)
      - Exit code is 0
      - Output directory is created with at least one CSV inside
      - stderr contains no Python traceback

    Empty FMO set is intentional — the synthetic fixture has no
    matching FMO controls; the build_fmo_thresholds helper handles
    that gracefully by returning {} (no threshold gates applied).
    """
    trial_dir = tmp_path / 'trial'
    trial_dir.mkdir()
    # The CLI's fcs_path() helper looks for `_<sample>_` or
    # `_<sample>.fcs` patterns — name the file accordingly.
    sample_name = 'smoke'
    shutil.copy(synthetic_fcs, trial_dir / f'expt_{sample_name}.fcs')

    out_dir = tmp_path / 'out'
    groups = [{
        'name': 'smoke-group',
        'samples': [sample_name],
        'fmo_set': 'none',
    }]
    fmo_sets = {'none': {}}     # no FMO controls — gracefully no-op

    result = _run_cli([
        '--trials',   str(trial_dir),
        '--out',      str(out_dir),
        '--groups',   json.dumps(groups),
        '--fmo-sets', json.dumps(fmo_sets),
        '--k',        '5',           # tiny k for 1000-event synthetic
        '--workers',  '1',
        '--seed',     '42',
        '--quiet',                   # silence INFO; we only want errors
    ])

    # Diagnostic dump if anything failed.
    diag = (f"exit={result.returncode}\n"
            f"--- stdout (last 2000 chars) ---\n"
            f"{result.stdout[-2000:]}\n"
            f"--- stderr (last 2000 chars) ---\n"
            f"{result.stderr[-2000:]}")

    assert result.returncode == 0, f"CLI failed:\n{diag}"

    # Nothing looking like a Python traceback in stderr.
    assert 'Traceback (most recent call last)' not in result.stderr, (
        f"CLI emitted a traceback to stderr:\n{diag}")

    # The pipeline writes per-sample stats CSVs + group-level outputs.
    # Don't pin the exact filenames (they may evolve) — just verify
    # SOMETHING landed.
    assert out_dir.is_dir(), f"output dir not created: {out_dir}"
    artifacts = list(out_dir.rglob('*'))
    assert artifacts, f"output dir is empty: {out_dir}"
    csvs = [p for p in artifacts if p.suffix == '.csv']
    pngs = [p for p in artifacts if p.suffix == '.png']
    assert csvs or pngs, (
        f"no .csv or .png outputs landed under {out_dir}; "
        f"got: {[str(p) for p in artifacts]}")


def test_cli_verbose_quiet_flags_parse():
    """Cheap argparse check — ``-v`` and ``-q`` must both be recognised.

    Used to be a full pipeline-twice comparison, but the wiring is
    ``logging.basicConfig(level=...)`` against a fixed argparse
    ``count`` / ``store_true`` action — verifying it parses is enough
    end-to-end coverage. The runtime effect of changing log level is
    a logging.basicConfig contract, not an OpenFlo behaviour to test.
    """
    # Each flag should be listed in --help.
    result = _run_cli(['--help'], timeout=30)
    assert result.returncode == 0
    assert '-v, --verbose' in result.stdout or '--verbose' in result.stdout
    assert '-q, --quiet' in result.stdout or '--quiet' in result.stdout


@_SLOW
def test_cli_by_day_grouping_no_sample_name_collision(synthetic_fcs, tmp_path):
    """Point the CLI at a PARENT folder with two day sub-folders that
    each contain an identically-named FCS. By-day grouping must process
    BOTH days (not bucket both into one), produce per-day output dirs,
    and run the cross-day comparison.

    Regression guard for the task-dispatch key collision: tasks used to
    be keyed by bare sample name, so two days' 'sample_1.fcs'
    collided and one day silently vanished.
    """
    import shutil
    parent = tmp_path / 'expt'
    for day in ('day0', 'day3'):
        d = parent / day
        d.mkdir(parents=True)
        # SAME filename in both days — this is what triggered the bug.
        shutil.copy(synthetic_fcs, d / 'sample_1.fcs')

    out_dir = tmp_path / 'out'
    result = _run_cli([
        '--trials',   str(parent),
        '--out',      str(out_dir),
        '--fmo-sets', '{}',           # no FMO controls → ungated, fast
        '--k',        '5',
        '--workers',  '1',
        '--seed',     '42',
        '--quiet',
    ])
    diag = (f"exit={result.returncode}\n--- stderr ---\n{result.stderr[-2000:]}\n"
            f"--- stdout ---\n{result.stdout[-2000:]}")
    assert result.returncode == 0, diag
    assert 'Traceback (most recent call last)' not in result.stderr, diag

    # Both day groups must have produced their own output dir + stats.
    day0 = out_dir / 'Day_0'
    day3 = out_dir / 'Day_3'
    assert day0.is_dir(), f"Day 0 output missing — collision regression\n{diag}"
    assert day3.is_dir(), f"Day 3 output missing — collision regression\n{diag}"
    assert list(day0.glob('*_stats.csv')), "Day 0 produced no stats"
    assert list(day3.glob('*_stats.csv')), "Day 3 produced no stats"
    # Cross-day comparison should have run.
    assert list(out_dir.glob('compare_Day_0_vs_Day_3*')), (
        f"cross-day comparison missing\n{diag}")
