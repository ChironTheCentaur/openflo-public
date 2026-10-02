"""Config-driven batch runner for OpenFlo.

Reads a JSON config describing one or more analyses and, for each group,
writes per-sample stats + plots, a group UMAP, and overlay/grid marker-pair
scatters, plus pairwise group comparisons. The pipeline runs samples
sequentially in the main process (phenograph always spawns a multiprocessing
Pool, which deadlocks inside a daemon worker — see the n_jobs note below).

NO data paths are baked in. Keep your real paths in a git-ignored config
(default ``private/analysis_config.json``); see
``scripts/analysis_config.example.json`` for the format and the supported
``group_by`` strategies.

Usage:
    python scripts/run_analyses.py [CONFIG] [--dry-run] [NAME ...]

    CONFIG     path to the JSON config (default: $OPENFLO_ANALYSIS_CONFIG or
               private/analysis_config.json)
    --dry-run  resolve + print the groups (and verify every file is found)
               without clustering — fast sanity check
    NAME ...   only run the named analyses (default: all in the config)
"""
import argparse
import glob
import json
import os
import re
import sys

# Headless backend BEFORE matplotlib is imported anywhere downstream.
os.environ.setdefault('MPLBACKEND', 'Agg')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from openflo import cli  # noqa: E402
from openflo.pipeline import FlowSample  # noqa: E402

# Phenograph ALWAYS spawns a multiprocessing Pool. A daemon process (what a
# ProcessPoolExecutor worker is) cannot spawn children → deadlock. So we
# process samples sequentially in the MAIN (non-daemon) process and keep
# phenograph's own pool tiny (n_jobs=1 → Pool(1), one short-lived child) to
# avoid the all-cores re-import memory storm.
DEFAULTS = {
    'max_events': 12_000, 'k': 30, 'doublet_tol': 0.2, 'seed': 42, 'njobs': 1,
    'pairs': [['CD34', 'CD11b'], ['CD11b', 'CD45'], ['CD34', 'CD45']],
}


# ── Config + path helpers ─────────────────────────────────────────────────────

def _load_config(path):
    with open(path, encoding='utf-8') as fh:
        cfg = json.load(fh)
    cfg.setdefault('data_root', '.')
    cfg.setdefault('out_dir', 'outputs/analyses')
    d = dict(DEFAULTS)
    d.update(cfg.get('defaults', {}))
    cfg['defaults'] = d
    return cfg


def _resolve(data_root, p):
    """A config path → absolute, relative to data_root (which is itself
    relative to the repo root unless absolute)."""
    if os.path.isabs(p):
        return p
    base = data_root if os.path.isabs(data_root) \
        else os.path.join(ROOT, data_root)
    return os.path.normpath(os.path.join(base, p))


def _specimens(folder, exclude=('compensation control', 'unstained'),
               require_prefix='specimen'):
    """FCS stems in `folder`, excluding control/blank tubes by name."""
    out = []
    for f in sorted(glob.glob(os.path.join(folder, '*.fcs'))):
        b = os.path.basename(f).lower()
        if any(tok in b for tok in exclude):
            continue
        if require_prefix and not b.startswith(require_prefix):
            continue
        out.append(os.path.splitext(os.path.basename(f))[0])
    return out


def _panel_labels(cfg, probe_fcs):
    panel = cfg.get('panel')
    if not panel:
        return {}
    if panel == 'auto':
        panel = cli.find_panel_xlsx([os.path.dirname(probe_fcs)])
        if not panel:
            return {}
    else:
        panel = _resolve(cfg['data_root'], panel)
    channels = list(FlowSample(probe_fcs).data.columns)
    return cli.read_staining_panel(panel, channels)


# ── group_by strategies (parameterised by the config, not the data) ───────────

def _groups_split_by_prefix(cfg, a):
    """Two groups split by a filename prefix token (e.g. an enrichment
    label). `require_regex` keeps only biological specimens; `exclude_regex`
    drops re-stains / QC tubes."""
    trial = _resolve(cfg['data_root'], a['trial_dir'])
    prefix = a.get('prefix', '').lower()
    req = re.compile(a['require_regex'], re.I) if a.get('require_regex') else None
    exc = re.compile(a['exclude_regex'], re.I) if a.get('exclude_regex') else None
    g_with, g_without = [], []
    for f in sorted(glob.glob(os.path.join(trial, '*.fcs'))):
        b = os.path.basename(f).lower()
        if 'compensation control' in b or 'unstained' in b:
            continue
        stem = os.path.splitext(os.path.basename(f))[0]
        tail = stem.lower().split('sample_', 1)[-1]
        if req and not req.search(tail):
            continue
        if exc and exc.search(tail):
            continue
        (g_with if prefix and tail.startswith(prefix) else g_without).append(stem)
    groups = [
        {'name': a.get('with_name', 'Group A'), 'samples': g_with},
        {'name': a.get('without_name', 'Group B'), 'samples': g_without},
    ]
    return [g for g in groups if g['samples']], trial


def _groups_by_folder(cfg, a):
    """One group per listed folder (e.g. a day/time-point series). Each
    group carries its own source folder."""
    groups = []
    for label, folder in a['folders']:
        fdir = _resolve(cfg['data_root'], folder)
        s = _specimens(fdir)
        if s:
            groups.append({'name': label, 'samples': s, 'trial_dir': fdir})
    return groups, _resolve(cfg['data_root'], a.get('root', '.'))


def _groups_split_by_token(cfg, a):
    """Two groups split by a name token across several folders, pooled.
    Sample display names are prefixed with their folder label so the pooled
    origin stays clear. `control_regex` matches the control condition."""
    root = _resolve(cfg['data_root'], a['root'])
    ctrl_re = re.compile(a['control_regex'], re.I)
    treated, control = [], []
    for label, folder in a['folders']:
        fdir = _resolve(cfg['data_root'], folder)
        tag = re.sub(r'\s+', '', label)
        for stem in _specimens(fdir):
            rel = os.path.relpath(os.path.join(fdir, stem), root) \
                .replace('\\', '/')
            disp = f"{tag} {stem.split('_', 2)[-1]}"
            (control if ctrl_re.search(stem) else treated).append((rel, disp))
    groups = [
        {'name': a.get('treated_name', 'Treated'), 'samples': treated},
        {'name': a.get('control_name', 'Control'), 'samples': control},
    ]
    return [g for g in groups if g['samples']], root


_STRATEGIES = {
    'split_by_prefix': _groups_split_by_prefix,
    'by_folder': _groups_by_folder,
    'split_by_token': _groups_split_by_token,
}


# ── Engine (sequential, main-process) ─────────────────────────────────────────

def _process_one(path, display_name, labels, d):
    s = FlowSample(path)
    s.run_qc()                                   # acquisition QC (cleaning)
    if d['doublet_tol']:
        s.filter_doublets(tol=d['doublet_tol'])
    s.auto_compensate()                          # embedded $SPILL
    s.apply_transform()                          # logicle
    if labels:
        s.set_labels(labels)
    s.name = display_name
    s.cluster(k=d['k'], n_jobs=d['njobs'], max_events=d['max_events'],
              random_state=d['seed'])
    return s


def _drive(groups, out_dir, labels, default_trial, d):
    from itertools import combinations
    pairs = [tuple(p) for p in d['pairs']]
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for g in groups:
        tdir = g.get('trial_dir') or default_trial
        gdir = os.path.join(out_dir, cli._safe_filename(g['name']))
        samples = []
        for entry in g['samples']:
            name, disp = entry if isinstance(entry, (tuple, list)) else (
                entry, entry.split('/')[-1].replace('Sample_', ''))
            try:
                path = cli.fcs_path(tdir, name)
                s = _process_one(path, disp, labels, d)
                cli.save_plots(s, gdir)
                s.export_stats(os.path.join(
                    gdir, cli._safe_filename(disp) + '_stats.csv'))
                samples.append(s)
                print(f"  [ok] {g['name']} / {disp}: {len(s.data):,} events, "
                      f"{s.data['cluster'].nunique()} clusters", flush=True)
            except Exception as exc:
                print(f"  [FAIL] {g['name']} / {name}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
        if samples:
            cli.run_group_umap(samples, cli._safe_filename(g['name']),
                               out_dir, random_state=d['seed'])
            cli.save_group_pair_scatters(samples, g['name'], out_dir,
                                         pairs=pairs)
            results[g['name']] = samples
    for ga, gb in combinations(groups, 2):
        try:
            cli._compare_group_pair(ga, gb, results, out_dir,
                                    lambda m: print('   ' + m, flush=True))
        except Exception as exc:
            print(f"  [compare FAIL] {ga['name']} vs {gb['name']}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
    print(f"[done] {out_dir}", flush=True)
    return results


def _build_groups(cfg, a):
    strat = _STRATEGIES.get(a.get('group_by'))
    if strat is None:
        raise ValueError(
            f"unknown group_by {a.get('group_by')!r}; "
            f"expected one of {sorted(_STRATEGIES)}")
    return strat(cfg, a)


def _first_sample_path(groups, default_trial):
    for g in groups:
        for entry in g['samples']:
            name = entry[0] if isinstance(entry, (tuple, list)) else entry
            tdir = g.get('trial_dir') or default_trial
            try:
                return cli.fcs_path(tdir, name)
            except FileNotFoundError:
                continue
    return None


def run_analysis(cfg, a, dry_run=False):
    name = a.get('name', a.get('group_by', 'analysis'))
    groups, default_trial = _build_groups(cfg, a)
    summary = {g['name']: len(g['samples']) for g in groups}
    print(f"\n[{name}] {a.get('group_by')} → {summary}", flush=True)
    if dry_run:
        for g in groups:
            tdir = g.get('trial_dir') or default_trial
            for entry in g['samples']:
                nm = entry[0] if isinstance(entry, (tuple, list)) else entry
                try:
                    cli.fcs_path(tdir, nm)
                    status = 'OK'
                except Exception as exc:
                    status = f'MISS ({exc})'
                print(f"    [{status}] {g['name']} / {nm}", flush=True)
        return
    probe = _first_sample_path(groups, default_trial)
    labels = _panel_labels(cfg, probe) if probe else {}
    out_dir = _resolve(cfg['data_root'], cfg['out_dir'])
    _drive(groups, os.path.join(out_dir, cli._safe_filename(name)),
           labels, default_trial, cfg['defaults'])


def main():
    default_cfg = (os.environ.get('OPENFLO_ANALYSIS_CONFIG')
                   or os.path.join(ROOT, 'private', 'analysis_config.json'))
    ap = argparse.ArgumentParser(description='Config-driven OpenFlo runner.')
    ap.add_argument('-c', '--config', default=default_cfg,
                    help='JSON config (default: %(default)s)')
    ap.add_argument('--dry-run', action='store_true',
                    help='resolve + print groups without clustering')
    ap.add_argument('names', nargs='*',
                    help='only run these analysis names (default: all)')
    args = ap.parse_args()

    if not os.path.isfile(args.config):
        ap.error(f"config not found: {args.config}\n"
                 f"Copy scripts/analysis_config.example.json to "
                 f"private/analysis_config.json and edit it.")
    cfg = _load_config(args.config)
    analyses = cfg.get('analyses', [])
    if args.names:
        analyses = [a for a in analyses if a.get('name') in args.names]
    if not analyses:
        ap.error('no matching analyses in config.')
    for a in analyses:
        run_analysis(cfg, a, dry_run=args.dry_run)
    print('\n[ALL DONE]', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
