"""Where openflo-run finds its samples: the ``--groups`` spec normalised to
canonical group dicts, and each sample name resolved to its FCS file
(including a group's unstained control).

Moved out of cli.py unchanged; the old cli names still work, deprecated.
"""
from __future__ import annotations

import os
from typing import Literal, overload


def normalise_groups(groups):
    """Coerce a raw groups spec into a list of canonical dicts.

    Each normalised dict has:
      name       (str)
      samples    (list[str])          — sample tokens, in order
      fmo_set    (str)                — the group's DEFAULT FMO set
      sample_fmo ({name: fmo_set})    — per-sample resolved FMO set

    Per-sample FMO assignment: a `samples` entry may be a plain string
    (uses the group's default `fmo_set`) OR a dict
    ``{'name': ..., 'fmo_set': ...}`` to override just that sample. This
    keeps the common case terse while letting any sample point at a
    different FMO control set than its group-mates. Compensation and
    antibody labels are resolved per sample automatically from each
    FCS ($SPILL / $PnS) in the pipeline, so they don't need a slot here.

    Drops any group whose name or sample list is empty.
    """
    norm = []
    for g in groups or []:
        if not isinstance(g, dict):
            continue
        name = str(g.get('name', '')).strip()
        raw_samples = g.get('samples', [])
        if isinstance(raw_samples, str):
            raw_samples = [s.strip() for s in raw_samples.split(',') if s.strip()]
        group_fmo = str(g.get('fmo_set', '')).strip()
        # Per-sample assignments already resolved by an earlier pass, if any.
        prior = g.get('sample_fmo')
        prior_fmo = dict(prior) if isinstance(prior, dict) else {}

        sample_names = []
        sample_fmo = {}
        for s in raw_samples:
            if isinstance(s, dict):
                sname = str(s.get('name', '')).strip()
                if not sname:
                    continue
                sfmo = str(s.get('fmo_set', '') or group_fmo).strip()
            else:
                sname = str(s).strip()
                if not sname:
                    continue
                # Idempotence: a per-sample override survives the first pass
                # only inside `sample_fmo`, because that pass flattens the
                # dict form to a plain name. Re-normalising an already-
                # normalised group would then reassign the GROUP default and
                # silently discard the override — which would gate that sample
                # against the wrong FMO control. Callers are told to
                # "normalise first"; this makes doing it twice harmless.
                sfmo = str(prior_fmo.get(sname, group_fmo) or '').strip()
            sample_names.append(sname)
            sample_fmo[sname] = sfmo

        if not name or not sample_names:
            continue
        entry = {'name': name, 'samples': sample_names,
                 'fmo_set': group_fmo, 'sample_fmo': sample_fmo}
        # By-day auto-groups tag each group with its source folder so the
        # run resolves that group's samples there (rather than the run's
        # single trial dir). Preserve it through normalisation.
        if g.get('trial_dir'):
            entry['trial_dir'] = str(g['trial_dir'])
        norm.append(entry)
    return norm


def find_unstained(fcs_dir, group='early'):
    """Path of the unstained control FCS file in ``fcs_dir``, or None.

    A ``_lateunstained`` file is preferred for ``group='late'`` and an
    ``_unstained`` one otherwise; failing that, any file with 'unstained' in its
    name is taken (skipping 'late' files for the early group)."""
    prefer = 'lateunstained' if group == 'late' else 'unstained'
    for f in sorted(os.listdir(fcs_dir)):
        base = f.lower()
        if not base.endswith('.fcs'):
            continue
        if f'_{prefer}_' in base or base.endswith(f'_{prefer}.fcs'):
            return os.path.join(fcs_dir, f)
    for f in sorted(os.listdir(fcs_dir)):
        base = f.lower()
        if 'unstained' in base and base.endswith('.fcs'):
            if group == 'early' and 'late' in base:
                continue
            return os.path.join(fcs_dir, f)
    return None


@overload
def scan_for_token(search_dir: str, token: str, recurse: bool,
                    all_matches: Literal[False] = False) -> str | None: ...
@overload
def scan_for_token(search_dir: str, token: str, recurse: bool,
                    all_matches: Literal[True]) -> list[str]: ...
def scan_for_token(search_dir, token, recurse, all_matches=False):
    """Find FCS files whose name matches `token`. A match is any of:
      • exact filename stem  (`<token>.fcs`)        — by-day auto-groups
        hand the whole stem here, so this is the primary path for them;
      • `_<token>_` substring                        — short logical names
      • ends with `_<token>.fcs`                     — trailing short name
    Returns the first match (or every match when `all_matches=True`);
    None / [] when nothing matches."""
    token_l = token.lower()
    matches: list[str] = []
    try:
        if recurse:
            walker = os.walk(search_dir)
        else:
            walker = [(search_dir, [], os.listdir(search_dir))]
    except OSError:
        return matches if all_matches else None
    for root, _dirs, files in walker:
        for f in sorted(files):
            if not f.lower().endswith('.fcs'):
                continue
            b = f.lower()
            stem = b[:-4]   # strip '.fcs'
            if (stem == token_l
                    or f'_{token_l}_' in b
                    or b.endswith(f'_{token_l}.fcs')):
                path = os.path.join(root, f)
                if not all_matches:
                    return path
                matches.append(path)
    return matches if all_matches else None


def fcs_path(fcs_dir: str, name: str) -> str:
    """Locate the FCS file for a logical sample name inside `fcs_dir`.

    `name` accepts a forward-slash subfolder prefix:
      • 'm1'         → search fcs_dir directly
      • 'day1/m1'    → search fcs_dir/day1/ directly (no fallback)
      • 'study/day1/m1' → search fcs_dir/study/day1/ directly

    When `name` has no slash and isn't found at the root of fcs_dir, we
    fall back to a recursive search. If that recursive search hits more
    than one match the first one is used and a warning lists the rest —
    the user can disambiguate by adding a subfolder prefix to the sample
    name in their group definition.
    """
    name = str(name).strip().replace('\\', '/')

    # Explicit subfolder path → search ONLY there
    if '/' in name:
        subfolder, token = name.rsplit('/', 1)
        search_dir = os.path.join(fcs_dir, subfolder)
        path = scan_for_token(search_dir, token, recurse=False)
        if path:
            return path
        raise FileNotFoundError(
            f"Could not find FCS file for '{name}' in {search_dir}")

    # Bare token → try root first
    path = scan_for_token(fcs_dir, name, recurse=False)
    if path:
        return path

    # Recursive fallback (handles FMO files placed in a controls/
    # subfolder, etc.).
    matches = scan_for_token(fcs_dir, name, recurse=True, all_matches=True)
    if not matches:
        raise FileNotFoundError(
            f"Could not find FCS file for '{name}' in {fcs_dir}")
    if len(matches) > 1:
        print(f"  [!] '{name}' matched {len(matches)} files; using "
              f"{os.path.relpath(matches[0], fcs_dir)}")
        for extra in matches[1:]:
            print(f"      also: {os.path.relpath(extra, fcs_dir)}")
        print(f"      Disambiguate by using a subfolder prefix in the "
              f"sample name (e.g. 'day1/{name}').")
    return matches[0]
