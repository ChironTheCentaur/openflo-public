"""Optional-engine probe + build provenance.

Tk-free so the GUI's Help → Environment panel, the provenance footer stamped on
exported figures, and the tests can all share it. Many OpenFlo features degrade
gracefully when an optional backend isn't installed (a method greys out, a run
skips); this surfaces *which* engines are present so a user isn't left
wondering why a button did nothing.
"""
from __future__ import annotations

import importlib.metadata as _md
import importlib.util

# (import_name, distribution_name, label, what it powers, pip extra or '' for core)
_ENGINES = [
    ('flowio', 'FlowIO', 'FlowIO', 'Reading / writing FCS files', ''),
    ('flowutils', 'FlowUtils', 'FlowUtils', 'Compensation + transforms', ''),
    ('umap', 'umap-learn', 'UMAP', 'UMAP embedding', ''),
    ('phenograph', 'PhenoGraph', 'PhenoGraph', 'PhenoGraph clustering', ''),
    ('igraph', 'igraph', 'python-igraph', 'Graph backend for Leiden', ''),
    ('leidenalg', 'leidenalg', 'Leiden', 'Leiden clustering', ''),
    ('trimap', 'trimap', 'TriMap', 'TriMap embedding', 'embed'),
    ('pacmap', 'pacmap', 'PaCMAP', 'PaCMAP embedding', 'embed'),
    ('phate', 'phate', 'PHATE', 'PHATE embedding', 'embed'),
    ('anndata', 'anndata', 'AnnData', '.h5ad export (scanpy interop)', 'interop'),
    ('tkinterdnd2', 'tkinterdnd2', 'tkinterdnd2',
     'Drag-and-drop FCS into the window', 'gui'),
]


def _version_of(import_name: str, dist_name: str) -> str | None:
    """Best-effort installed version, or None if the engine isn't present.

    Uses ``find_spec`` (locates the module without running its ``__init__``)
    plus distribution metadata — importing engines like umap/phate here would
    block the Tk thread for seconds, which froze Help → Environment."""
    try:
        spec = importlib.util.find_spec(import_name)
    except (ImportError, ValueError, ModuleNotFoundError):
        spec = None
    if spec is None:
        return None
    try:
        return _md.version(dist_name)
    except Exception:
        return 'installed'          # present but version metadata unavailable


def probe_capabilities() -> list[dict]:
    """Status of every optional/core engine OpenFlo uses. Each entry is
    ``{key, label, powers, extra, available, version}``. ``extra`` is the pip
    extra that installs it (``''`` for a core dependency)."""
    out = []
    for import_name, dist, label, powers, extra in _ENGINES:
        version = _version_of(import_name, dist)
        out.append({
            'key': import_name, 'label': label, 'powers': powers,
            'extra': extra, 'available': version is not None,
            'version': version or '',
        })
    return out


def install_hint(extra: str) -> str:
    """How to install a missing engine for the given pip extra."""
    if extra:
        return f"pip install 'openflo[{extra}]'"
    return "pip install --upgrade openflo   # core dependency — reinstall"


def openflo_version() -> str:
    try:
        return _md.version('openflo')
    except Exception:
        try:
            from . import __version__
            return str(__version__)
        except Exception:
            return '0'


def build_provenance(extra: str = '') -> str:
    """One-line build stamp for exported figures: version (+ short git SHA when
    running from a checkout). ``extra`` is appended verbatim when given."""
    stamp = f"OpenFlo {openflo_version()}"
    sha = _git_sha()
    if sha:
        stamp += f" ({sha})"
    if extra:
        stamp += f" · {extra}"
    return stamp


_GIT_SHA_CACHE: list = []          # [] = not probed; [None] / [sha] = result


def _git_sha() -> str | None:
    """Short git SHA of the OpenFlo source checkout, or None for a pip install
    (no .git). Best-effort, silent, and cached so repeated figure exports don't
    re-spawn git."""
    if _GIT_SHA_CACHE:
        return _GIT_SHA_CACHE[0]
    import os
    import subprocess
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    sha = None
    try:
        out = subprocess.run(
            ['git', '-C', pkg_dir, 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=3)
        if out.returncode == 0 and out.stdout.strip():
            sha = out.stdout.strip()
    except Exception:
        sha = None
    _GIT_SHA_CACHE.append(sha)
    return sha
