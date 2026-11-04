"""How openflo-run names and checks its groups: a group, sample or
comparison name as an output file name, the checks that refuse a group spec
that cannot be run, and the by-day grouping that makes one group per folder
of FCS files.

Moved out of cli.py unchanged; the old cli names still work, deprecated.
"""
from __future__ import annotations

import os
import re

from .cli_samples import fcs_path


def output_name(name):
    """A group, sample or comparison name as a folder / file name in the
    CLI's output tree: runs of whitespace and path-hostile characters
    become one ``_``; ``unnamed`` when nothing usable remains."""
    return (re.sub(r'[<>:"/\\|?*\s]+', '_', str(name)).strip('_')
            or 'unnamed')


def compare_base(g_a, g_b):
    """File stem of one pair's comparison: compare_<A>_vs_<B>."""
    return output_name(f'compare_{g_a["name"]}_vs_{g_b["name"]}')


def group_spec_problems(groups):
    """Why a normalised group spec cannot be run: one line per problem, []
    when it can. A group's name is one side of every comparison and its
    output folder, so two groups with one name had their samples merged
    into one side, a missing sample of the second went unreported, and
    compare_Ctrl_vs_Ctrl was written. Names that differ only in case or in
    characters output_name strips share a folder on Windows / macOS. A
    sample listed twice in a group was processed and counted twice."""
    out = []
    names = [g['name'] for g in groups]
    for nm in dict.fromkeys(n for n in names if names.count(n) > 1):
        out.append(f'{names.count(nm)} groups are named {nm!r}; group names '
                   'must be unique.')
    folder = {}
    for nm in dict.fromkeys(names):
        folder.setdefault(output_name(nm).casefold(), []).append(nm)
    for same in folder.values():
        if len(same) > 1:
            out.append(' and '.join(repr(n) for n in same) + ' would share '
                       f'the output folder {output_name(same[0])!r}; '
                       'rename one.')
    for g in groups:
        twice = list(dict.fromkeys(s for s in g['samples']
                                   if g['samples'].count(s) > 1))
        if twice:
            out.append(f"group {g['name']!r} lists {', '.join(twice)} more "
                       'than once.')
    # Two pairs whose comparison files share a name ('A' vs 'B vs C' and
    # 'A vs B' vs 'C' both make compare_A_vs_B_vs_C): one overwrote the
    # other.
    from itertools import combinations
    by_file = {}
    for a, b in combinations(groups, 2):
        by_file.setdefault(compare_base(a, b).casefold(), []).append((a, b))
    for pairs in by_file.values():
        if len(pairs) > 1:
            out.append(' and '.join(f"{a['name']!r} vs {b['name']!r}"
                                    for a, b in pairs)
                       + f' would both write {compare_base(*pairs[0])}.csv;'
                       ' rename a group.')
    return out


def sample_file_problems(groups, trial_dirs):
    """One line per group that lists one FCS file under two names. fcs_path
    ignores case and also matches a name's tail ('c1' finds expt_c1.fcs),
    so 'c1' and 'C1', or 'c1' and 'expt_c1', passed the name check and the
    file was processed, and counted in the comparison, twice. Files are
    compared by resolved path; a name with no file is left to the run."""
    import contextlib
    import io
    out = []
    for g in groups:
        seen = {}                        # resolved path -> names
        for d in ([g['trial_dir']] if g.get('trial_dir') else trial_dirs):
            for nm in g['samples']:
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        path = fcs_path(d, nm)
                except OSError:          # incl. FileNotFoundError
                    continue
                names = seen.setdefault(
                    os.path.normcase(os.path.realpath(path)), [])
                if nm not in names:
                    names.append(nm)
        for path, names in seen.items():
            if len(names) > 1:
                out.append(f"group {g['name']!r} lists "
                           f"{' and '.join(names)}, which are one file: "
                           f'{path}.')
    return out


# ── By-day auto-grouping ───────────────────────────────────────────────────────

def discover_day_folders(trial_dirs):
    """Every directory at/under the given paths that DIRECTLY contains at
    least one .fcs file. Lets the user point at a single PARENT folder
    and have each sub-folder auto-become its own day/group, sampled
    independently. De-duplicated, sorted."""
    found, seen = [], set()
    for d in trial_dirs:
        if not d or not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            if any(f.lower().endswith('.fcs') for f in files):
                rp = os.path.normpath(root)
                if rp not in seen:
                    seen.add(rp)
                    found.append(rp)
    return sorted(found)


def auto_groups_by_day(trial_dirs):
    """Build one group per discovered day-folder (a folder that directly
    holds FCS files). Each group's samples are that folder's FCS files
    (by filename stem); each group is tagged with its source folder via
    `trial_dir` so the run resolves its samples there. Group name is the
    folder basename, tidied to 'Day N' when a day token is present;
    duplicate names are disambiguated with the parent folder.

    Returns [] when no FCS are found anywhere (caller falls back to the
    legacy default groups)."""
    folders = discover_day_folders(trial_dirs)
    groups = []
    for d in folders:
        try:
            fcs = sorted(f for f in os.listdir(d)
                         if f.lower().endswith('.fcs'))
        except OSError:
            continue
        if not fcs:
            continue
        samples = [os.path.splitext(f)[0] for f in fcs]
        base = os.path.basename(d.rstrip('/\\')) or d
        m = re.search(r'day\s*[-_ ]?([0-9]+)', base, re.IGNORECASE)
        name = f'Day {m.group(1)}' if m else base
        groups.append({'name': name, 'samples': samples,
                       'fmo_set': '', 'trial_dir': d})

    # Disambiguate clashing group names. Names clash as output folders do:
    # by output_name, without case ('ctrl' and 'Ctrl' were left as they
    # were, and the run refused them). First by the folder's own name where
    # it says more than the group name (exp/day3 and exp/day_3 share every
    # parent, so walking up never told them apart); then by the parent
    # folder, and more of its path while names still clash (X/exp/day3 and
    # Y/exp/day3 both gave 'Day 3 (exp)').
    def key(name):
        return output_name(name).casefold()

    base = [g['name'] for g in groups]
    own = [os.path.basename(g['trial_dir'].rstrip('/\\')) for g in groups]
    parents = [[p for p in os.path.dirname(g['trial_dir'].rstrip('/\\'))
                .replace('\\', '/').split('/') if p and not p.endswith(':')]
               for g in groups]
    keys = [key(b) for b in base]
    clash = {i for i, k in enumerate(keys) if keys.count(k) > 1}
    depth = 0
    while clash and depth <= max(map(len, parents), default=0):
        for i in clash:
            if depth == 0:
                if key(own[i]) != key(base[i]):
                    groups[i]['name'] = f'{base[i]} ({own[i]})'
            elif parents[i][-depth:]:
                groups[i]['name'] = f"{base[i]} ({'/'.join(parents[i][-depth:])})"
        keys = [key(g['name']) for g in groups]
        clash = {i for i in clash if keys.count(keys[i]) > 1}
        depth += 1
    return groups
