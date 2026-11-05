"""openflo-run's group-level outputs: a UMAP and marker-pair scatters per
group, and every pairwise group comparison from one clustering shared by
every sample, written to a staging folder and published as one set.

Moved out of cli.py unchanged; the old cli names still work, deprecated.
"""
from __future__ import annotations

import copy
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from . import cli_parallel, cli_plots
from ._console import start_worker, worker_settings
from .cli_args import DEFAULT_DPI
from .cli_groups import compare_base, output_name
from .cli_plots import STAGING_PREFIX, run_group_umap, save_group_pair_scatters
from .export_stamp import background_kwargs, savefig_metadata
from .flow_experiment import FlowExperiment
from .shared_clustering import shared_cluster_labels, shared_pool_cap

SHARED_IDS_NOTE = ('shared: one clustering of every sample; phenotypes in '
                    'compare_shared_clusters.csv; NOT the <name>_stats.csv '
                    'cluster ids')


# The only names openflo-run's comparison step writes (or wrote: the
# manifest of an earlier build). Anything else called compare_* is not
# ours: matching compare_*.csv deleted openflo-compare's compare_report.csv.
COMPARE_OWN = re.compile(r'compare_(shared_clusters\.csv|manifest\.json'
                          r'|.+_vs_.+\.(csv|png))', re.IGNORECASE)


def compare_files(out):
    """The comparison files openflo-run made in `out`: compare_shared_
    clusters.csv and compare_<A>_vs_<B>.csv / .png (plus an old
    compare_manifest.json), matched without case, as Windows and macOS
    file names are."""
    return sorted(f for f in os.listdir(out)
                  if COMPARE_OWN.fullmatch(f)
                  and os.path.isfile(os.path.join(out, f)))


def compare_group_pair(g_a, g_b, results, trial_out, report,
                        trial_label_prefix='', dpi=DEFAULT_DPI,
                        keep_open=None):
    """Run condition-comparison between two groups. Called for every pair.
    `results` maps group name → samples labelled by ONE clustering shared
    by every sample (see shared_cluster_labels). Returns None when it
    wrote compare_<A>_vs_<B>.csv / .png, else why it wrote nothing.
    `keep_open` keeps the figure open for --show-plots (see
    cli_plots.keeps_open)."""
    prefix = f'[{trial_label_prefix}] ' if trial_label_prefix else ''
    report(f'{prefix}Compare: {g_a["name"]} vs {g_b["name"]} …')
    sa = results.get(g_a['name'], [])
    sb = results.get(g_b['name'], [])
    if not sa or not sb:
        print(f'  [!] {g_a["name"]} or {g_b["name"]} has no samples to '
              'compare — comparison skipped')
        return 'had no samples'
    # Key each sample by side + position, never by its name: by-day groups
    # hold the SAME filename in every day folder, and a name-keyed dict kept
    # only the last one, so the CSV had one condition holding the other
    # day's data. compare_conditions matches samples by `.name`, so each
    # gets a shallow copy carrying its key.
    exp = FlowExperiment.__new__(FlowExperiment)
    exp.samples = {}
    keys = {'A': [], 'B': []}
    for side, members in (('A', sa), ('B', sb)):
        for i, s in enumerate(members):
            v = copy.copy(s)
            v.name = f'{side}{i}:{s.name}'
            exp.samples[v.name] = v
            keys[side].append(v.name)
    # Their 'cluster' labels ARE one clustering of every sample; without
    # this record compare_conditions would cluster them together again.
    exp.joint_clustering = {'samples': tuple(exp.samples),
                            'label_col': 'cluster',
                            'method': 'openflo-run shared clustering'}
    try:
        summary = exp.compare_conditions(
            groupA=keys['A'],
            groupB=keys['B'],
            label_a=g_a['name'], label_b=g_b['name'],
        )
    except Exception as exc:  # noqa: BLE001 - reported; the run goes on
        print(f'  [!] Comparison failed: {exc}')
        return f'failed ({type(exc).__name__}: {exc})'
    if summary.empty:
        print(f"  [!] {g_a['name']} vs {g_b['name']}: comparison produced no "
              "rows (no shared populations) — nothing written", flush=True)
        return 'had no rows'
    add_pvalues(summary, sa, sb)
    # The ids are the shared clustering's, not those of the <name>_stats.csv
    # beside it; say so in the file, where a reader comparing the two looks.
    summary['cluster_ids'] = SHARED_IDS_NOTE
    base = compare_base(g_a, g_b)
    summary.to_csv(os.path.join(trial_out, f'{base}.csv'), index=False)
    plt.gca().set_title(f"{g_a['name']} vs {g_b['name']} — shared clusters "
                        "(compare_shared_clusters.csv)")
    plt.savefig(os.path.join(trial_out, f'{base}.png'), dpi=dpi,
                metadata=savefig_metadata(fmt='png'),
                **background_kwargs(plt.gcf(), fmt='png'))
    if not cli_plots.keeps_open(keep_open):
        plt.close()
    return None


def add_pvalues(summary, sa, sb):
    """Add 'p' and 'p_adj' to a comparison table (diffexp.differential_test):
    per cluster, Welch's t-test of the samples' % of events in A against B
    on the logit scale, as the GUI's default, and the Benjamini-Hochberg
    adjustment across the clusters. A cluster absent from a sample is 0 %
    there. It was a Mann-Whitney U test, whose smallest p at 3 samples a
    group is 0.1, so no cluster could ever come out significant at the
    sizes runs use. Identical shares in every sample give p = 1; NaN where
    the test cannot run."""
    from .diffexp import differential_test
    clusters = sorted(summary['cluster'].unique())

    def pcts(samples):
        out = {c: [] for c in clusters}
        for s in samples:
            lab = s.data['cluster'] if 'cluster' in s.data.columns else None
            n = len(s.data)
            vc = lab.value_counts() if lab is not None else {}
            for c in clusters:
                out[c].append(100.0 * float(vc.get(c, 0)) / n if n else np.nan)
        return out

    rows = {r['feature']: r for r in differential_test(
        pcts(sa), pcts(sb), parametric=True, transform='logit')}
    summary['p'] = [rows.get(c, {}).get('p', np.nan) for c in summary['cluster']]
    summary['p_adj'] = [rows.get(c, {}).get('p_adj', np.nan)
                        for c in summary['cluster']]


def shared_n_jobs(n_jobs, pool):
    """PhenoGraph n_jobs for the shared clustering of `pool` events: -1
    (every core) when free RAM covers it, else 1, whatever --workers set
    (`n_jobs`, the per-sample runs' value, is used only when psutil is
    missing). The shared run starts after every per-sample run has
    finished and runs alone, so with --workers > 1 (per-sample n_jobs=1) it
    may use every core; with --workers 1 (per-sample n_jobs=-1) its pool
    can still be bigger than any per-sample run's (--max-events above the
    largest sample: 3 x 100k samples with a 500k cap make a 300k pool from
    100k-event runs), so the gate applies there too. Measured on 14
    channels, seeded Leiden, identical labels both ways (peak = RSS of the
    process and its children, 24 cores):

        pool 200k   n_jobs=1  356.6 s, 2.0 GB    n_jobs=-1  87.5 s, 5.7 GB
        pool  60k   n_jobs=1   45.6 s, 1.1 GB    n_jobs=-1  24.5 s, 4.6 GB
        pool 500k                                n_jobs=-1 382.5 s, 7.4 GB

    The extra memory is the one process per core that PhenoGraph's Jaccard
    step starts: about 163 MB plus 240 B per pooled event each (starmap
    sends every child the kNN index; review-cli's measurement). Every core
    is used only when free RAM covers 2 GB plus that per core (8.96 GB for
    the 500k pool above, against its 7.4 GB peak); otherwise one core. A
    flat 0.25 GB a core let a 1M-event pool start children needing about
    12 GB with 10 GB free."""
    try:
        import psutil
    except ImportError:
        return n_jobs
    free_gb = psutil.virtual_memory().available / 1024 ** 3
    need_gb = 2.0 + (os.cpu_count() or 1) * (0.17 + pool * 2.4e-7)
    if free_gb < need_gb:
        print(f"  Shared clustering runs PhenoGraph on one core: "
              f"{free_gb:.1f} GB RAM free, every core would need about "
              f"{need_gb:.1f} GB for a {pool:,}-event pool.", flush=True)
        return 1
    return -1


STAGING_MARK = '.openflo-compare-staging'


def is_staging(path):
    """True for a staging folder OpenFlo made: the exact name prefix AND the
    mark file written when it was made, so a group's output folder that
    happens to share the prefix is never taken for one."""
    return (os.path.basename(path).startswith(STAGING_PREFIX)
            and os.path.isfile(os.path.join(path, STAGING_MARK)))


def clear_stale_staging(out, prefix=''):
    """Remove staging folders an earlier run left in `out`: one killed
    while publishing, or one whose new set could not be moved in (a file in
    use). They were never removed, and --show-plots re-opened their PNGs."""
    import shutil
    if not os.path.isdir(out):
        return
    for name in sorted(os.listdir(out)):
        path = os.path.join(out, name)
        if os.path.isdir(path) and is_staging(path):
            shutil.rmtree(path, ignore_errors=True)
            print(f'  {prefix}Removed {path}, left by an earlier run.',
                  flush=True)


def remove_earlier_set(out, prefix, why):
    """Remove openflo-run's own comparison files from `out` (the
    compare_files rules), naming them and `why`, for a run that makes no
    comparison. Returns False when one could not be removed (in use), so
    the run can exit non-zero: the folder then does not match the run."""
    if not os.path.isdir(out):
        return True
    gone, stuck = [], []
    for f in compare_files(out):
        try:
            os.remove(os.path.join(out, f))
            gone.append(f)
        except OSError as exc:
            stuck.append(f'{f} ({type(exc).__name__}: {exc.strerror or exc})')
    if gone:
        print(f"  [!] {prefix}{', '.join(gone)} from an earlier run removed: "
              f'{why}.', flush=True)
    if stuck:
        print(f"  [!] {prefix}could not remove {', '.join(stuck)}: from an "
              f'earlier run, not this one ({why}).', flush=True)
    return not stuck


def publish_compare_set(out, staging, written, no_file, prefix):
    """Replace the comparison set in `out` with the one in `staging`: every
    compare_* file already in `out` is moved aside, the new files are moved
    in, and the old ones are deleted. So no old file stays beside the new
    set: not a pair of a group no longer in the run, not a pair that failed
    or had no rows this time. File names compare without case, as Windows
    and macOS do: comparing them with case deleted a just-written
    compare_Ctrl_vs_Treat.csv whose old name was compare_ctrl_vs_Treat.csv.
    `no_file` maps a pair's file stem to why it has no file this time.

    Returns None, or why the set could not be moved in: a file that cannot
    be moved (open in Excel: PermissionError). Then everything is put back
    as it was, the old set stays in `out`, and the new one in `staging`."""
    import shutil
    old = compare_files(out)
    aside = os.path.join(staging, 'previous')
    os.makedirs(aside)
    moved, placed = [], []
    try:
        for f in old:
            os.replace(os.path.join(out, f), os.path.join(aside, f))
            moved.append(f)
        for f in written:
            os.replace(os.path.join(staging, f), os.path.join(out, f))
            placed.append(f)
    except OSError as exc:
        stuck = []
        for src, dst in ([(os.path.join(out, f), os.path.join(staging, f))
                          for f in reversed(placed)]
                         + [(os.path.join(aside, f), os.path.join(out, f))
                            for f in reversed(moved)]):
            try:
                os.replace(src, dst)
            except OSError:
                stuck.append(src)
        why = f'{exc.filename} cannot be moved ({type(exc).__name__}: ' \
              f'{exc.strerror or exc}), e.g. it is open in Excel'
        if stuck:
            why += ('; these could not be put back and are still where they '
                    'were moved: ' + ', '.join(stuck))
        else:
            os.rmdir(aside)                         # empty again
        return why
    new = {f.casefold() for f in written}
    why_of = {stem.casefold(): why for stem, why in no_file.items()}
    gone = {}
    for f in old:
        if f.casefold() not in new:
            why = why_of.get(os.path.splitext(f)[0].casefold(),
                             "not part of this run's set")
            gone.setdefault(why, []).append(f)
    for why, files in gone.items():
        print(f"  [!] {prefix}{', '.join(files)} from an earlier run "
              f'removed: {why}.', flush=True)
    shutil.rmtree(staging, ignore_errors=True)   # ours: made by mkdtemp
    return None


def not_in_this_run(groups, results, reused, tasks=(), failures=None):
    """One 'group/sample: why' line per listed sample that is missing from
    the comparison: neither processed in this run nor kept from an earlier
    one (--skip-existing) -- it failed (with its error, from `failures`),
    or no task was made for it because its FCS file was not found -- or
    whose task is missing an input file (`missing_inputs`: a kept sample's
    file is gone, or a trial folder of a concatenated sample lacks it).
    Checked before any kept sample is reloaded."""
    failures = failures or {}
    tried = {(t['group'], t['name']) for t in tasks}
    kept = {(g, t['name']) for g, ts in reused.items() for t in ts}
    missing = [f"{t['group']}/{t['name']}: "
               + ('kept from an earlier run, but its input is missing'
                  if (t['group'], t['name']) in kept
                  else 'its input is missing')
               + f" ({'; '.join(t['missing_inputs'])})"
               for t in tasks if t.get('missing_inputs')]
    for g in groups:
        have = ({s.name for s in results.get(g['name'], [])}
                | {t['name'] for t in reused.get(g['name'], [])})
        for nm in g['samples']:
            if nm in have:
                continue
            err = failures.get(f"{g['name']}\x1f{nm}")
            if err:
                why = f'failed in this run ({err})'
            elif (g['name'], nm) in tried:
                why = f"failed in this run (see 'Failed: {nm}' above)"
            else:
                why = 'its FCS file was not found'
            missing.append(f"{g['name']}/{nm}: {why}")
    return missing


def samples_to_compare(groups, results, reused, tasks=(), failures=None):
    """Every sample of every group, for the group comparisons. Returns
    (by_group, missing): {group name: [FlowSample]} with the samples
    processed in this run plus, under --skip-existing, the ones kept from
    an earlier run, whose events are reloaded (not re-clustered — the
    shared clustering does not need their own clusters), ordered as the
    group lists them so the shared clustering's subsample does not depend
    on the order parallel workers finished in; and `missing`, one
    'group/sample (why)' line per listed sample that is in neither. The
    caller writes no comparison when `missing` is not empty, so a sample
    not in this run is named BEFORE any reload, which would be wasted."""
    missing = not_in_this_run(groups, results, reused, tasks, failures)
    if missing:
        return {}, missing
    by_group = {}
    for g in groups:
        name = g['name']
        if name in by_group:
            continue
        members = list(results.get(name, []))
        kept = reused.get(name, [])
        for i, t in enumerate(kept, 1):
            print(f"  Reloading {name}/{t['name']} (kept from an earlier run) "
                  f"for the group comparisons, {i}/{len(kept)} …", flush=True)
            try:
                s = cli_parallel.load_task_sample(t)
                err = None if s is not None else 'no input files found'
            except Exception as exc:  # noqa: BLE001 - listed as missing below
                s, err = None, f'{type(exc).__name__}: {exc}'
            if s is None:
                missing.append(f"{name}/{t['name']}: kept from an earlier "
                               f"run, but cannot be reloaded ({err})")
                continue
            members.append(s)
        rank = {n: i for i, n in enumerate(g['samples'])}
        by_group[name] = sorted(members,
                                key=lambda s: rank.get(s.name, len(rank)))
    return by_group, missing


def group_plots_job(samples, label, pair_label, out, random_state, pairs,
                     dpi=DEFAULT_DPI, keep_open=None):
    """One group's UMAP and marker-pair scatters (picklable: runs in a
    worker process when groups are drawn in parallel)."""
    run_group_umap(samples, label, out, random_state=random_state, dpi=dpi,
                   keep_open=keep_open)
    save_group_pair_scatters(samples, pair_label, out, pairs=pairs, dpi=dpi)


def draw_group_plots(jobs, out, random_state, pairs, workers, report,
                      prefix='', dpi=DEFAULT_DPI, keep_open=None):
    """Each group's UMAP and pair scatters. ``jobs`` holds ``(name,
    samples, label, pair_label)`` per group. With ``workers`` > 1 and more
    than one group the groups run in a process pool (each is independent and
    its UMAP seeded, so the files are the same); not while figures are kept
    open to be shown (`keep_open`), which needs this process's figures."""
    keep_open = cli_plots.keeps_open(keep_open)
    if workers > 1 and len(jobs) > 1 and not keep_open:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        # Fresh processes, as on Windows: a forked copy of a process whose
        # numba threads have run (UMAP, clustering) can die on its first
        # parallel loop.
        with ProcessPoolExecutor(
                max_workers=min(workers, len(jobs)),
                mp_context=multiprocessing.get_context('spawn'),
                initializer=start_worker,
                initargs=(worker_settings(),)) as ex:
            futs = [ex.submit(group_plots_job, samples, label, pair_label,
                              out, random_state, pairs, dpi)
                    for _name, samples, label, pair_label in jobs]
            for (name, samples, _l, _p), fut in zip(jobs, futs, strict=True):
                report(f'{prefix}Running UMAP ({name}, {len(samples)} '
                       'sample(s)) …')
                fut.result()
        return
    for name, samples, label, pair_label in jobs:
        report(f'{prefix}Running UMAP ({name}, {len(samples)} '
               'sample(s)) …')
        group_plots_job(samples, label, pair_label, out, random_state, pairs,
                         dpi, keep_open)


def write_group_outputs(groups, results, reused, out, report, cluster_task,
                         random_state=42, pairs=None, tl='', tasks=(),
                         failures=None, workers=1, keep_open=None):
    """Group-level outputs of one analysis: a UMAP + marker-pair scatters per
    group, then every pairwise group comparison. `results` holds the samples
    processed in this run; `reused` maps a group to the tasks of its samples
    that --skip-existing kept from an earlier run; `tasks` is every
    sample's task, processed or kept, and `failures` maps a failed task's
    key to its error (to say why a sample is missing).

    No group output is computed from part of a group. A resumed run must not
    rewrite them from the re-run samples alone: a 2-sample comparison would
    be overwritten by a 1-sample one, and the UMAP redrawn from one sample.

    Returns False when the comparisons asked for were not written (a sample
    missing, the shared clustering failed, or a file in use kept the new
    set out of `out`), so the run can end with a non-zero exit; True when
    they were, or when none was due (one group). `keep_open` keeps the
    group figures open for --show-plots (see cli_plots.keeps_open)."""
    from itertools import combinations

    import pandas as pd
    prefix = f'[{tl}] ' if tl else ''
    suffix = f'_{tl}' if tl else ''
    clear_stale_staging(out, prefix)

    # UMAP per group with >= 1 sample. Single-sample groups still get
    # the per-group UMAP rendered (the by-sample_origin panel is just
    # one colour in that case, but the by-cluster + FSC/SSC panels
    # are still meaningful).
    # With --workers > 1 the groups are drawn in parallel (draw_group_plots).
    jobs = []
    for g in groups:
        samples = results.get(g['name'], [])
        if not samples:
            continue
        if reused.get(g['name']):
            # The UMAP colours by each sample's own clusters, which a kept
            # sample no longer has without clustering it again.
            print(f"  [skip-existing] {g['name']}: "
                  f"{len(reused[g['name']])} sample(s) kept from an earlier "
                  "run, so its UMAP and pair scatters are left as they are. "
                  "Re-run without --skip-existing to redraw them.", flush=True)
            continue
        # Marker-pair scatters: per-sample grid + cross-sample overlay.
        jobs.append((g['name'], samples,
                     output_name(f'{g["name"]}{suffix}'),
                     f'{g["name"]}{suffix}'))
    dpi = cluster_task.get('dpi', DEFAULT_DPI)
    draw_group_plots(jobs, out, random_state, pairs, workers, report,
                      prefix, dpi=dpi, keep_open=keep_open)

    group_pairs = list(combinations(groups, 2))
    if not group_pairs:
        # One group, no comparison. An earlier run's comparison files would
        # sit in the folder, unmarked, as if this run had made them.
        return remove_earlier_set(out, prefix, 'this run has one group, so '
                                   'it makes no comparison')
    # The comparisons are recomputed on EVERY run, resumes included:
    # --skip-existing skips only the per-sample work. Deciding when an
    # earlier set could be kept would take a record of everything that made
    # it (groups, input files, settings, code) and is easy to get wrong; the
    # shared clustering costs 10 s to a few minutes, small next to the samples.
    by_group, missing = samples_to_compare(groups, results, reused, tasks,
                                            failures)
    if missing:
        # All the comparison files come from one shared clustering, so they
        # are written together or not at all (the user's rule, fresh runs
        # included). A resume that cannot reload one kept sample must not
        # leave its group out and rewrite the rest: compare_shared_clusters
        # .csv and the other pairs would come from a new clustering while
        # that group's pairs kept the old one, and one cluster id could read
        # 30 % in one file and 70 % in the next.
        print(f"  [!] {prefix}No group comparisons were written: "
              f"{len(missing)} sample(s) of the groups are missing.",
              flush=True)
        for line in missing:
            print(f'      {line}', flush=True)
        print("      A comparison from part of a group would not be the "
              "group's. Fix these samples or remove them from the groups, "
              "then run again (--skip-existing reuses the finished samples). "
              f"Any compare_* files in {out} are from an earlier run, not "
              "this one.", flush=True)
        return False
    import shutil
    import tempfile

    # Every file of the set is written to a staging folder and moved into
    # `out` only once the whole set exists (publish_compare_set). Writing
    # in place left a run killed after its first pair with
    # compare_shared_clusters.csv and that pair from the new clustering and
    # the other pairs from the old one. The folder is a NEW one (mkdtemp),
    # in `out` so the moves stay on one drive: a fixed name could be a
    # group's output folder, and a group named '.compare_staging' had its
    # stats deleted when that folder was cleared.
    staging = tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=out)
    with open(os.path.join(staging, STAGING_MARK), 'w', encoding='utf-8'):
        pass                    # marks it ours, for clear_stale_staging
    shared = {}
    flat = [(g, s) for g, members in by_group.items() for s in members]
    report(f'{prefix}Clustering {len(flat)} sample(s) together for the '
           'group comparisons …')
    if cluster_task.get('method', 'phenograph') == 'phenograph':
        sizes = [len(s.data) for _, s in flat]
        pool = min(sum(sizes), shared_pool_cap(sizes, cluster_task))
        cluster_task = dict(cluster_task, n_jobs=shared_n_jobs(
            cluster_task.get('n_jobs', 1), pool))
    try:
        labels, channels, columns, display = shared_cluster_labels(
            [s for _, s in flat], cluster_task,
            names=[f'{g}/{s.name}' for g, s in flat])
    except Exception as exc:  # noqa: BLE001 - reported; no comparisons
        # Falling back to each sample's own cluster ids would compare
        # unrelated populations; write no comparison rather than that.
        shutil.rmtree(staging, ignore_errors=True)
        print(f"  [!] {prefix}Shared clustering failed ({type(exc).__name__}"
              f": {exc}). No group comparisons were written. Any compare_* "
              f"files in {out} are from an earlier run, not this one.",
              flush=True)
        return False
    # Medians are named by the shared columns' names for every sample:
    # the columns are matched by label, so one name is one marker.
    phenotypes = []
    for (g, s), lab, cols in zip(flat, labels, columns, strict=True):
        v = copy.copy(s)
        v.data = pd.DataFrame({'cluster': lab})
        v.fluor_channels = []
        shared.setdefault(g, []).append(v)
        m = copy.copy(s)
        m.data = s.data[cols].set_axis(channels, axis=1).assign(cluster=lab)
        m.fluor_channels = list(channels)
        m.channel_labels = display
        # The record goes with each column to its shared name, so the
        # medians are inverted with this sample's own transform.
        rec = getattr(s, 'data_transforms', None) or {}
        m.data_transforms = {ch: rec[col] for ch, col
                             in zip(channels, cols, strict=True) if col in rec}
        freq = m.cluster_frequencies()
        freq.insert(0, 'group', g)
        phenotypes.append(freq)
    table = 'compare_shared_clusters.csv'
    pd.concat(phenotypes, ignore_index=True).to_csv(
        os.path.join(staging, table), index=False)
    written, no_file = [table], {}
    for g_a, g_b in group_pairs:
        base = compare_base(g_a, g_b)
        why = compare_group_pair(g_a, g_b, shared, staging, report,
                                  trial_label_prefix=tl, dpi=dpi,
                                  keep_open=keep_open)
        if why:
            no_file[base] = (f"this run's {g_a['name']} vs {g_b['name']} "
                             f'comparison {why}')
        else:
            written += [f'{base}.csv', f'{base}.png']
    locked = publish_compare_set(out, staging, written, no_file, prefix)
    if locked:
        print(f"  [!] {prefix}Group comparisons NOT updated: {locked}. The "
              f"comparison files in {out} are left as they were, from an "
              f"earlier run. This run's comparisons are in {staging}. Close "
              "the file and run again, or copy them from there before the "
              "next run, which removes that folder.", flush=True)
        return False
    print(f"  Group comparisons use clusters found by clustering all "
          f"{len(flat)} samples together, so an id names the same "
          "population in every sample. They are NOT the ids in each "
          "<name>_stats.csv. Each shared cluster's size and marker "
          f"medians per sample: {os.path.join(out, table)}", flush=True)
    return True
