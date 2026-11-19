"""``openflo-run``'s ``run_manifest.json``: what a run ran, on which files,
with which software, and what it wrote, each file with its size and SHA-256.

Moved out of cli.py unchanged; the old cli names still work, deprecated.
"""
from __future__ import annotations

import json
import os
import sys

from .cli_samples import fcs_path, normalise_groups

MANIFEST_NAME = 'run_manifest.json'

# The packages whose versions decide what a run computes.
MANIFEST_PACKAGES = ('numpy', 'pandas', 'scipy', 'scikit-learn', 'flowio',
                     'flowutils', 'matplotlib', 'umap-learn', 'phenograph',
                     'leidenalg', 'igraph', 'FlowSOM', 'anndata')


def sha256(path, chunk=1 << 20):
    """Hex SHA-256 digest of the file at ``path``, read in ``chunk``-byte blocks."""
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def file_entry(path, root=None):
    """Manifest entry ``{'path', 'bytes', 'sha256'}`` for one file; the path is
    made relative to ``root`` when given and always uses ``/`` separators."""
    return {'path': (os.path.relpath(path, root) if root else path)
            .replace(os.sep, '/'),
            'bytes': os.path.getsize(path), 'sha256': sha256(path)}


def run_input_files(trial_dirs, groups):
    """The FCS files the run's groups name, located as the run locates
    them; a sample it cannot locate is left out (the run said so)."""
    out, seen = [], set()
    for grp in normalise_groups(groups) or []:
        dirs = [grp['trial_dir']] if grp.get('trial_dir') else list(
            trial_dirs or ['.'])
        for name in grp.get('samples', []):
            for d in dirs:
                try:
                    fp = fcs_path(d, name)
                except FileNotFoundError:
                    continue
                key = os.path.normcase(os.path.abspath(fp))
                if key not in seen:
                    seen.add(key)
                    out.append(fp)
                break
    return out


def write_run_manifest(args, trial_dirs, groups, gate_overrides,
                       region_gates, rc, started):
    """Write ``run_manifest.json`` in the output folder: what was run, on
    what, with which software, and what it wrote, so a run can be checked
    and repeated. Inputs and outputs carry their size and SHA-256."""
    import datetime as _dt
    import platform
    from importlib import metadata

    from .result_stamp import app_version
    out = args.out
    os.makedirs(out, exist_ok=True)
    packages = {}
    for pkg in MANIFEST_PACKAGES:
        try:
            packages[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    gate_files = [p for p in (args.gates_from, args.panel)
                  if p and p != 'auto' and os.path.isfile(p)]
    outputs = []
    for dirpath, dirnames, filenames in os.walk(out):
        dirnames.sort()
        for f in sorted(filenames):
            full = os.path.join(dirpath, f)
            if os.path.abspath(full) == os.path.abspath(
                    os.path.join(out, MANIFEST_NAME)):
                continue
            outputs.append(file_entry(full, out))
    manifest = {
        'format': 'openflo-run-manifest',
        'version': 1,
        'openflo': app_version(),
        'started': started.isoformat(timespec='seconds'),
        'finished': _dt.datetime.now(_dt.UTC).isoformat(
            timespec='seconds'),
        'exit_code': rc,
        'command': ['openflo-run'] + sys.argv[1:],
        'python': platform.python_version(),
        'platform': platform.platform(),
        'packages': packages,
        'settings': dict(sorted(vars(args).items())),
        'groups': normalise_groups(groups),
        'gates': {'thresholds': gate_overrides or {},
                  'region_gates': region_gates or []},
        'inputs': [file_entry(p) for p in run_input_files(trial_dirs, groups)],
        'gate_files': [file_entry(p) for p in gate_files],
        'outputs': outputs,
    }
    path = os.path.join(out, MANIFEST_NAME)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[manifest] {path}", flush=True)
    return path
