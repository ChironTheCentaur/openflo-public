"""Update checking — tell the user when a newer OpenFlo release exists on
GitHub, and (when they ask) run the right upgrade command.

Pure, testable core (version parsing/compare, release-JSON parsing, install-kind
detection, command construction) + thin network/subprocess wrappers. A silent
auto-update of a running GUI is deliberately NOT done — you can't safely
hot-swap a running Python process and it risks the user's environment; instead
this offers a *user-instigated* upgrade that runs `pip install -U git+…`
(pip installs) or `git pull` (source checkouts) and asks the user to restart.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request

# The public release repo the check queries.
PUBLIC_REPO = "ChironTheCentaur/openflo-public"
_RELEASES_API = "https://api.github.com/repos/{repo}/releases/latest"
_TAGS_API = "https://api.github.com/repos/{repo}/tags"


def current_version():
    """Installed OpenFlo version string (falls back to the package attr)."""
    try:
        import importlib.metadata as _m
        return _m.version("openflo")
    except Exception:
        try:
            from . import __version__
            return str(__version__)
        except Exception:
            return "0"


def parse_release_tag(release_json):
    """Extract a bare version (no leading ``v``) from a GitHub release JSON
    dict, or None when absent/malformed."""
    if not isinstance(release_json, dict):
        return None
    tag = str(release_json.get("tag_name") or "").strip()
    tag = tag.lstrip("vV")
    return tag or None


def is_newer(latest, current):
    """True if ``latest`` is a strictly newer version than ``current``. Uses
    :mod:`packaging` when available, else a numeric-tuple fallback; any parse
    failure returns False (never nags on garbage)."""
    if not latest or not current:
        return False
    try:
        from packaging.version import Version
        return Version(str(latest)) > Version(str(current))
    except Exception:
        def _t(v):
            out = []
            for part in str(v).split("."):
                num = "".join(c for c in part if c.isdigit())
                out.append(int(num) if num else 0)
            return tuple(out)
        try:
            return _t(latest) > _t(current)
        except Exception:
            return False


def fetch_latest_release(repo=PUBLIC_REPO, timeout=4.0):
    """GET the latest release JSON from GitHub (public, unauthenticated), or
    None on any network/HTTP/parse error — callers must treat None as
    'couldn't check', never as 'up to date'."""
    url = _RELEASES_API.format(repo=repo)
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "openflo-update-check"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def latest_tag(tags_json):
    """Highest version-like tag name (no leading ``v``) from a GitHub tags
    JSON list, or None. Used as a fallback for repos that push tags but don't
    cut formal Releases."""
    if not isinstance(tags_json, list):
        return None
    names = [str((t or {}).get("name") or "").strip().lstrip("vV")
             for t in tags_json]
    names = [n for n in names if n]
    if not names:
        return None
    try:
        from packaging.version import Version
        return str(max(names, key=Version))
    except Exception:
        def _t(v):
            out = []
            for part in str(v).split("."):
                num = "".join(c for c in part if c.isdigit())
                out.append(int(num) if num else 0)
            return tuple(out)
        try:
            return max(names, key=_t)
        except Exception:
            return names[0]


def fetch_latest_tag(repo=PUBLIC_REPO, timeout=4.0):
    """Newest version-like tag on the repo, or None on any error. Fallback when
    the repo has no GitHub Releases (e.g. a force-pushed snapshot mirror)."""
    url = _TAGS_API.format(repo=repo)
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "openflo-update-check"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return latest_tag(json.loads(resp.read().decode("utf-8")))
    except Exception:
        return None


def check_for_update(repo=PUBLIC_REPO, timeout=4.0):
    """Check GitHub for a newer version. Returns
    ``{current, latest, available, url}`` or None if the check couldn't run
    (offline, rate-limited, no releases AND no tags). Prefers a formal Release;
    falls back to the newest tag so a tag-only mirror still works.
    ``available`` is True only when a strictly-newer version exists."""
    data = fetch_latest_release(repo=repo, timeout=timeout)
    latest = parse_release_tag(data)
    url = (data or {}).get("html_url") if isinstance(data, dict) else None
    if latest is None:                       # no Release → try tags
        latest = fetch_latest_tag(repo=repo, timeout=timeout)
        url = f"https://github.com/{repo}/tags" if latest else url
    if latest is None:
        return None
    cur = current_version()
    return {"current": cur, "latest": latest,
            "available": is_newer(latest, cur),
            "url": url or f"https://github.com/{repo}/releases/latest"}


def _package_git_root():
    """The git work-tree root the installed package lives in, or None when it
    isn't a git checkout (i.e. a normal pip install)."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(
            ["git", "-C", pkg_dir, "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    root = out.stdout.strip()
    return root if out.returncode == 0 and root else None


def detect_install_kind():
    """``'git'`` when OpenFlo runs from a source checkout, else ``'pip'``."""
    return "git" if _package_git_root() else "pip"


def update_command(kind=None, repo=PUBLIC_REPO):
    """The argv to upgrade OpenFlo for the given install ``kind`` (auto-detected
    when None): ``git pull`` for a checkout, ``pip install -U git+…`` for a pip
    install."""
    kind = kind or detect_install_kind()
    if kind == "git":
        root = _package_git_root() or "."
        return ["git", "-C", root, "pull", "--ff-only"]
    return [sys.executable, "-m", "pip", "install", "--upgrade",
            f"git+https://github.com/{repo}.git"]


def run_update(kind=None, repo=PUBLIC_REPO, timeout=600):
    """Run the upgrade command, returning ``(ok: bool, output: str)``. The
    caller should prompt the user to restart OpenFlo afterwards — a running
    process keeps the old code loaded."""
    cmd = update_command(kind=kind, repo=repo)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout)
    except Exception as exc:                                  # noqa: BLE001
        return False, f"Update failed to start: {exc}\n\nCommand: {' '.join(cmd)}"
    log = (out.stdout or "") + (out.stderr or "")
    return out.returncode == 0, log.strip() or f"(exit {out.returncode})"
