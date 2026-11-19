"""Comparisons over a run template's history (``history.csv``, see
``run_history``): within one run, and across every run.

Within a run, each sample is a replicate: the chosen statistic of one
population is compared across the run's groups (or trials, or samples)
with ``stats.compare_groups``. Across runs, each run DAY is the
replicate: the samples of a group in one run are averaged, and the groups
are compared with the runs paired by experiment date (a repeated-measures
design: every run measured every group). Frequencies are tested on the
logit scale, counts on log(1 + n) and intensities on arcsinh, as the
Statistics window does.

Only active runs are used unless asked: a day run again counts once.

Core module: no Tk. ``ui_run_compare`` shows it.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from .run_history import HISTORY_CSV, read_history_table

SERIES = ('Group', 'Trial', 'Sample')
_KEY_COLS = ('Run', 'Experiment date', 'Active', 'Template version', 'Group',
             'Trial', 'Sample', 'Population')
_STAT_PREFIXES = ('Median ', 'GeoMean ')


def load_history(hdir: str, include_superseded: bool = False):
    """``history.csv`` of history folder `hdir` as a DataFrame (empty when
    there is none), its active runs only unless `include_superseded`."""
    import pandas as pd
    path = os.path.join(hdir, HISTORY_CSV)
    if not os.path.exists(path):
        return pd.DataFrame(columns=list(_KEY_COLS))
    df = read_history_table(path)
    for c in ('Group', 'Trial', 'Sample', 'Population'):
        if c in df.columns:
            df[c] = df[c].fillna('').astype(str)
    if 'Active' in df.columns:
        df['Active'] = df['Active'].astype(str).str.lower().eq('true')
        if not include_superseded:
            df = df[df['Active']]
    return df.reset_index(drop=True)


def stat_names(df) -> list[str]:
    """The statistics the history holds, in table order: Count, %Parent,
    %Total, then each channel's Median and GeoMean."""
    return [c for c in df.columns if c in ('Count', '%Parent', '%Total')
            or str(c).startswith(_STAT_PREFIXES)]


def populations(df) -> list[str]:
    """The populations in the history, first seen first."""
    return list(dict.fromkeys(df['Population'])) if 'Population' in df else []


def scale_for(stat: str) -> str:
    """The scale a statistic is tested on: logit for a frequency, log(1+n)
    for a count, arcsinh for an intensity."""
    if stat.startswith('%'):
        return 'logit'
    return 'log1p' if stat == 'Count' else 'asinh'


def _rows(df, population: str, stat: str):
    if stat not in df.columns:
        raise ValueError(f"the history has no {stat}")
    return df[df['Population'] == population]


RUN_SEP = ' · '


def per_run_means(df, population: str, stat: str, by: str = 'Group'):
    """One value per run and series (`by`: Group, Trial or Sample): the
    mean of that series' samples in that run. Rows are the runs by
    experiment date (oldest first), each labelled with its date; a date
    held by more than one run (superseded runs shown) labels each of them
    ``<date> · <run>``, so two runs of a day are never averaged into a
    value neither gave. Active rows only are one day's values even when a
    re-run of some groups took them from two runs (``run_history``: each
    group's rows come from one run), so they are one row per date.
    Columns are the series in first-seen order."""
    import pandas as pd
    sub = _rows(df, population, stat)
    order = list(dict.fromkeys(sub[by]))
    if 'Active' in sub.columns and bool(sub['Active'].all()):
        table = sub.pivot_table(index='Experiment date', columns=by,
                                values=stat, aggfunc='mean')
        return table.reindex(columns=[c for c in order if c in table.columns])
    table = sub.pivot_table(index=['Experiment date', 'Run'], columns=by,
                            values=stat, aggfunc='mean')
    # pivot_table sorts by date, then run; the series keep first-seen order.
    dates = [d for d, _r in table.index]
    table.index = pd.Index([f'{d}{RUN_SEP}{r}' if dates.count(d) > 1 else d
                            for d, r in table.index], name='Experiment date')
    return table.reindex(columns=[c for c in order if c in table.columns])


def label_date(label: Any) -> str:
    """The experiment date of a :func:`per_run_means` row label."""
    return str(label).split(RUN_SEP)[0]


def compare_within(df, run: str, population: str, stat: str,
                   by: str = 'Group') -> dict[str, Any]:
    """`stat` of `population` compared across the `by` series of one run
    (its samples the replicates): ``stats.compare_groups`` on the stat's
    test scale. For an active run that is its day: every active row of
    its experiment date, which a re-run of some groups took from two runs
    (:func:`within_rows`)."""
    from .stats import compare_groups
    sub = within_rows(_rows(df, population, stat), run)
    groups = {k: list(v[stat]) for k, v in sub.groupby(by, sort=False)}
    return compare_groups(groups, transform=scale_for(stat))


def within_rows(df, run: str):
    """The rows of `df` that make up run `run` for a within-run comparison:
    when any of its rows is active, every active row of its experiment
    date (a re-run of one group leaves the day's other groups in the
    earlier run); else (a superseded run, shown on request) its own rows."""
    mine = df[df['Run'] == run]
    if 'Active' not in df.columns or not bool(mine['Active'].any()):
        return mine
    dates = set(mine['Experiment date'])
    return df[df['Active'] & df['Experiment date'].isin(dates)]


def compare_across(df, population: str, stat: str,
                   by: str = 'Group') -> dict[str, Any]:
    """`stat` of `population` compared across the `by` series over every
    run, each run day one replicate and the days paired
    (``stats.compare_groups`` with ``pair_keys``): the per-day means of
    :func:`per_run_means` over the active runs only, so a day run again
    counts once whatever the window shows."""
    from .stats import compare_groups
    if 'Active' in df.columns:
        df = df[df['Active']]
    table = per_run_means(df, population, stat, by)
    groups, keys = {}, {}
    for c in table.columns:
        # A day a series lacks is NaN here; compare_groups leaves it out
        # of that series (and of the pairing) with its key.
        groups[c] = list(table[c].values)
        keys[c] = list(table.index)
    return compare_groups(groups, transform=scale_for(stat), pair_keys=keys)


def version_changes(df) -> list[str]:
    """The experiment dates whose template versions differ from the date
    before (a day run partly again may hold two): where an across-run
    trend may jump for a reason other than biology."""
    if df.empty or 'Template version' not in df:
        return []
    if 'Active' in df.columns:          # a date's versions are its active rows'
        df = df[df['Active']]
    by_date = (df.groupby('Experiment date', sort=True)['Template version']
               .agg(lambda v: frozenset(map(str, v))))
    out, prev = [], None
    for date, v in by_date.items():
        if prev is not None and v != prev:
            out.append(str(date))
        prev = v
    return out


def _fmt_p(p: float | None) -> str:
    import math
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return 'n/a'
    return f'{p:.3g}'


def result_lines(res: Mapping[str, Any]) -> list[str]:
    """A comparison (``compare_groups``) as lines for the window and the
    exported summary: the test and its p, each group's n and mean, then
    the pairwise p's (BH-adjusted)."""
    lines = []
    if res.get('test'):
        paired = ', paired by run day' if res.get('paired') else ''
        lines.append(f"{res['test']}{paired}: p = {_fmt_p(res.get('p'))}"
                     + (' (cannot reach p < 0.05 with these numbers)'
                        if res.get('underpowered') else ''))
    else:
        lines.append('Not enough data to test (each series needs 2 or more '
                     'values).')
    for name, g in (res.get('groups') or {}).items():
        lines.append(f"  {name}: n = {g.get('n')}, mean = {g.get('mean'):.4g}")
    for ph in res.get('posthoc') or []:
        lines.append(f"  {ph['a']} vs {ph['b']}: p(adj) = {_fmt_p(ph.get('p_adj'))}")
    return lines


def runs(df) -> list[tuple[str, str]]:
    """``(run, experiment date)`` of each run in the history, newest first."""
    if df.empty:
        return []
    pairs = df[['Run', 'Experiment date']].drop_duplicates()
    return [(str(r), str(d)) for r, d in
            sorted(pairs.itertuples(index=False), reverse=True)]


def export_table(df, population: str, stat: str, by: str,
                 path: str) -> None:
    """Write :func:`per_run_means` to `path` as CSV (dates down, series
    across)."""
    from .table_export import write_frame_csv
    write_frame_csv(path, per_run_means(df, population, stat, by), index=True)


def series_choices(df) -> Sequence[str]:
    """The ways to split the history into series that it can do: Group,
    Trial and Sample, each only when it holds more than one value."""
    return [c for c in SERIES if c in df.columns and df[c].nunique() > 1] \
        or ['Group']
