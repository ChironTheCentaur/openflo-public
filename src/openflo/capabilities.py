"""Optional-engine probe + build provenance.

Tk-free so the GUI's Help → Environment panel, the provenance footer stamped on
exported figures, and the tests can all share it. Many OpenFlo features degrade
gracefully when an optional backend isn't installed (a method greys out, a run
skips); this surfaces *which* engines are present so a user isn't left
wondering why a button did nothing.
"""
from __future__ import annotations

import importlib
import importlib.metadata as _md
import importlib.util
import threading

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


# ── Can an engine actually be used? ─────────────────────────────────────────
#
# One answer for every place that asks: Help → Environment, Compare
# embeddings, the Cluster dialog and the Pipeline. They used to ask in two
# ways, finding the package (find_spec) or importing it, and disagreed about
# a package that is installed but fails to import (a missing dependency of
# its own): Environment and the Pipeline offered TriMap, Compare embeddings
# said it was not installed, and a run with it ticked made nothing.
#
# Importing is the only real test, and it is slow for some engines (seconds
# for umap or phate). It runs in a separate Python process: in a thread of
# the GUI process, an import that big set off garbage collection there, which
# finalised Tk objects off the Tk thread ("main thread is not in main loop").
# A process also leaves the GUI without the engine's memory until it is used.
# The answer is cached for the session, and the windows ask for it off the
# Tk thread (``check_engines`` under run_async), using the quick find_spec
# answer only until it has come back.

NOT_INSTALLED = 'not installed'

_IMPORT_RESULT: dict[str, str | None] = {}     # module -> None (works) / why not
_IMPORT_LOCK = threading.Lock()

# Run as ``python -c``: imports each module named in argv and prints
# {module: null | reason} as JSON on the last line of stdout.
_CHECK_SCRIPT = r"""
import importlib, json, sys
out = {}
for m in sys.argv[1:]:
    try:
        importlib.import_module(m)
        out[m] = None
    except BaseException as exc:
        msg = str(exc).strip().splitlines()
        out[m] = ("fails to load (" + type(exc).__name__
                  + (": " + msg[0] if msg else "") + ")")
sys.stdout.write("\n" + json.dumps(out) + "\n")
"""


def _import_check(modules: list[str]) -> dict[str, str | None]:
    """``{module: None or why not}`` from importing each in a fresh Python.

    If that process can't be run or answer (no interpreter at
    sys.executable, a timeout), the modules are imported here instead."""
    import json
    import subprocess
    import sys
    flags = 0x08000000 if sys.platform == 'win32' else 0   # CREATE_NO_WINDOW
    try:
        proc = subprocess.run(
            # -P: the working folder is not put on the path, so a stray
            # file there can't stand in for an engine.
            [sys.executable, '-P', '-c', _CHECK_SCRIPT, *modules],
            capture_output=True, text=True, timeout=180,
            creationflags=flags)
        lines = (proc.stdout or '').strip().splitlines()
        found = json.loads(lines[-1]) if lines else None
        if isinstance(found, dict) and set(found) == set(modules):
            return found
    except Exception as exc:
        from .errors import note
        note("checking which optional engines are installed", exc)
    out: dict[str, str | None] = {}
    for m in modules:
        try:
            importlib.import_module(m)
            out[m] = None
        except BaseException as exc:              # noqa: BLE001
            msg = str(exc).strip().splitlines()
            out[m] = (f"fails to load ({type(exc).__name__}"
                      + (f": {msg[0]}" if msg else '') + ")")
    return out


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def engine_unavailable(module: str, load: bool = True) -> str | None:
    """Why ``module`` cannot be used here, or None when it can.

    ``NOT_INSTALLED`` when the package is not there; ``'fails to load
    (Error: message)'`` when it is there but importing it raises. With
    ``load=False`` an engine whose import has not been tried yet is judged
    by its presence alone (quick, for building a window); the import's
    answer replaces that as soon as anything has run it."""
    if module in _IMPORT_RESULT:
        return _IMPORT_RESULT[module]
    if not _installed(module):
        _IMPORT_RESULT[module] = NOT_INSTALLED
        return NOT_INSTALLED
    if not load:
        return None
    return check_engines([module])[module]


def engine_checked(module: str) -> bool:
    """Whether ``engine_unavailable(module)`` already has its final answer."""
    return module in _IMPORT_RESULT


def check_engines(modules=None) -> dict[str, str | None]:
    """Import-check ``modules`` (default: every engine listed here), all in
    one separate process, and return ``{module: why not, or None}``. Slow
    the first time (seconds); run it off the Tk thread."""
    if modules is None:
        modules = [e[0] for e in _ENGINES]
    with _IMPORT_LOCK:
        todo = [m for m in dict.fromkeys(modules) if m not in _IMPORT_RESULT]
        for m in [m for m in todo if not _installed(m)]:
            _IMPORT_RESULT[m] = NOT_INSTALLED
            todo.remove(m)
        if todo:
            _IMPORT_RESULT.update(_import_check(todo))
    return {m: _IMPORT_RESULT[m] for m in modules}


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
    except Exception:  # noqa: BLE001 - version metadata is informational
        return 'installed'          # present but version metadata unavailable


def probe_capabilities(load: bool = True) -> list[dict]:
    """Status of every optional/core engine OpenFlo uses. Each entry is
    ``{key, label, powers, extra, available, version, problem, checked}``.
    ``extra`` is the pip extra that installs it (``''`` for a core
    dependency); ``problem`` is why it can't be used ('' when it can).

    ``available`` is ``engine_unavailable`` (an import) — the same answer the
    Pipeline and Compare embeddings act on. ``load=False`` skips imports not
    yet tried, for a window that must open at once; ``checked`` says whether
    an entry's answer is final."""
    out = []
    for import_name, dist, label, powers, extra in _ENGINES:
        version = _version_of(import_name, dist)
        why = engine_unavailable(import_name, load=load)
        out.append({
            'key': import_name, 'label': label, 'powers': powers,
            'extra': extra, 'available': why is None,
            'version': version or '', 'problem': why or '',
            'checked': engine_checked(import_name),
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
        except Exception:  # noqa: BLE001 - version metadata is informational
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
