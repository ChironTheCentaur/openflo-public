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
import re
import subprocess
import sys
import urllib.request

# The public release repo the check queries.
PUBLIC_REPO = "ChironTheCentaur/openflo-public"
# Links the app opens for users. They must name the PUBLIC mirror: the
# development repository is private, so a link to it is a 404 for everyone
# else (tests/test_public_links.py fails on any link to it under src/).
PUBLIC_URL = f"https://github.com/{PUBLIC_REPO}"
ISSUES_URL = f"{PUBLIC_URL}/issues"
DOCS_URL = f"{PUBLIC_URL}#readme"
CHANGELOG_URL = f"{PUBLIC_URL}/blob/main/CHANGELOG.md"
# The one-click Windows installer ships its own Python and writes this file
# into that Python's sys.prefix: {"kind": "installer", "version": "<ver>"}.
INSTALL_MARKER = "openflo-install.json"
# Release notes longer than this are cut (at a line end) in the update offer;
# the full notes are a link away.
NOTES_LIMIT = 1500
_RELEASES_API = "https://api.github.com/repos/{repo}/releases/latest"
_TAGS_API = "https://api.github.com/repos/{repo}/tags"


def current_version():
    """Running OpenFlo version string.

    For a SOURCE CHECKOUT, read the live pyproject.toml version — the
    editable-install metadata is frozen at `pip install -e` time, so after a
    version bump it goes stale and the update check would falsely report an
    update is available (e.g. metadata stuck at 1.5.0 while the source is 1.7.x).
    Falls back to the installed metadata, then the package attr."""
    root = _package_git_root()
    if root:
        try:
            import tomllib
            with open(os.path.join(root, "pyproject.toml"), "rb") as f:
                v = tomllib.load(f).get("project", {}).get("version")
            if v:
                return str(v)
        except (ImportError, OSError, ValueError):
            pass
    try:
        import importlib.metadata as _m
        return _m.version("openflo")
    except Exception:  # noqa: BLE001 - no installed metadata falls back to __version__
        try:
            from . import __version__
            return str(__version__)
        except ImportError:
            return "0"


def parse_release_tag(release_json):
    """Extract a bare version (no leading ``v``) from a GitHub release JSON
    dict, or None when absent/malformed."""
    if not isinstance(release_json, dict):
        return None
    tag = str(release_json.get("tag_name") or "").strip()
    tag = tag.lstrip("vV")
    return tag or None


def _fallback_key(v):
    """Sort key for a version when :mod:`packaging` is missing: the release
    numbers, then 0 for a pre-release (``2.7.0b1``) and 1 for a final, so a
    beta sorts below its own release. Reading every digit would make
    ``2.7.0b1`` (2, 7, 1) newer than ``2.7.0``."""
    m = re.match(r"\s*(\d+(?:\.\d+)*)(.*)$", str(v))
    if not m:
        raise ValueError(v)
    nums = tuple(int(x) for x in m.group(1).split("."))
    nums = nums + (0,) * (4 - len(nums))
    return nums + (0 if m.group(2).strip() else 1,)


def _is_prerelease(v):
    try:
        from packaging.version import Version
        return Version(str(v)).is_prerelease
    except Exception:  # noqa: BLE001 - no packaging or an unparseable version uses the regex
        return bool(re.search(r"(a|b|rc|dev|alpha|beta)\d*", str(v), re.I))


def is_newer(latest, current):
    """True if ``latest`` is a strictly newer version than ``current``. Uses
    :mod:`packaging` when available, else ``_fallback_key``; any parse
    failure returns False (never nags on garbage). A pre-release (a beta)
    counts only for someone already on a pre-release: a final release is
    never told to update to a beta."""
    if not latest or not current:
        return False
    if _is_prerelease(latest) and not _is_prerelease(current):
        return False
    try:
        from packaging.version import Version
        return Version(str(latest)) > Version(str(current))
    except Exception:  # noqa: BLE001 - no packaging or an unparseable version uses _fallback_key
        try:
            return _fallback_key(latest) > _fallback_key(current)
        except (ValueError, TypeError):
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
    except Exception:  # noqa: BLE001 - offline or bad reply just means no update info
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
    except Exception:  # noqa: BLE001 - no packaging or an unparseable tag sorts by _fallback_key
        def _key(v):
            try:
                return _fallback_key(v)
            except ValueError:               # a non-version tag sorts last
                return (-1,)
        return max(names, key=_key)


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
    except Exception:  # noqa: BLE001 - offline or bad reply just means no update info
        return None


_MD_LINK = re.compile(r"!?\[([^\]]*)\]\(([^)\s]*)[^)]*\)")
# An autolink (<https://...>) keeps its address.
_MD_AUTOLINK = re.compile(r"<([A-Za-z][A-Za-z0-9+.-]{1,31}:[^\s<>]*)>")
# Only comments and tag-shaped markup (<br>, </details>, <img src=...>) go;
# a lone "<" or ">" in text ("FL1-A < 1000 and FL2-A > 50") stays.
_MD_HTML = re.compile(r"<!--.*?-->|</?[A-Za-z][A-Za-z0-9-]*(?:[\s/][^>\n]*)?>",
                      re.S)
# A comment that fills its own lines goes with them, leaving no blank line.
_MD_COMMENT_LINES = re.compile(r"^[ \t]*<!--.*?-->[ \t]*(?:\n|\Z)", re.S | re.M)


def _plain_line(line):
    """One Markdown line as plain text: no heading hashes, emphasis marks,
    inline code ticks or link syntax (``[text](url)`` keeps the text)."""
    line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
    line = re.sub(r"^(\s*)[*+]\s+", r"\1- ", line)
    line = _MD_LINK.sub(lambda m: m.group(1) or m.group(2), line)
    for mark in ("**", "__", "`"):
        line = line.replace(mark, "")
    return line.rstrip()


def plain_notes(body, limit: int | None = NOTES_LIMIT):
    """Release notes (GitHub Markdown ``body``) as short plain text: markup
    removed, runs of blank lines folded to one, and, past ``limit``
    characters, cut at the last whole line with "…" added (``limit=None``:
    never cut). '' when there are none."""
    if not isinstance(body, str):
        return ""
    text = _MD_COMMENT_LINES.sub("", body.replace("\r\n", "\n"))
    text = _MD_AUTOLINK.sub(r"\1", text)
    text = _MD_HTML.sub("", text)
    lines, blank = [], True
    for raw in text.split("\n"):
        line = _plain_line(raw)
        if not line.strip():
            if not blank:
                lines.append("")
            blank = True
            continue
        lines.append(line)
        blank = False
    out = "\n".join(lines).strip()
    if limit is None or len(out) <= limit:
        return out
    cut = out[:limit]
    nl = cut.rfind("\n")
    if nl > limit // 2:
        cut = cut[:nl]
    return cut.rstrip() + "\n…"


def check_for_update(repo=PUBLIC_REPO, timeout=4.0):
    """Check GitHub for a newer version. Returns
    ``{current, latest, available, url, notes, has_release}`` or None if the
    check couldn't run (offline, rate-limited, no releases AND no tags).
    Prefers a formal Release; falls back to the newest tag so a tag-only
    mirror still works (``has_release`` False, ``notes`` ''). ``available``
    is True only when a strictly-newer version exists; ``notes`` is the
    Release's own notes as plain text (:func:`plain_notes`)."""
    data = fetch_latest_release(repo=repo, timeout=timeout)
    latest = parse_release_tag(data)
    url = (data or {}).get("html_url") if isinstance(data, dict) else None
    has_release = latest is not None
    notes = plain_notes(data.get("body")) if isinstance(data, dict) and has_release else ""
    if latest is None:                       # no Release → try tags
        latest = fetch_latest_tag(repo=repo, timeout=timeout)
        url = f"https://github.com/{repo}/tags" if latest else url
    if latest is None:
        return None
    cur = current_version()
    return {"current": cur, "latest": latest,
            "available": is_newer(latest, cur),
            "url": url or f"https://github.com/{repo}/releases/latest",
            "notes": notes, "has_release": has_release}


def _package_git_root():
    """The root of the OpenFlo source checkout the running package lives in,
    or None when it isn't one (i.e. a normal pip install).

    Being inside *some* git work tree is not enough. A pip install into a
    project's own venv (myproject/.venv/Lib/site-packages/openflo) sits in
    myproject's work tree, and taking that for a checkout reported
    myproject's pyproject version and made Update run `git pull` on
    myproject. So the root must be OpenFlo's own, and the package must not
    be an installed copy under a site-packages inside it."""
    pkg_dir = os.path.dirname(os.path.realpath(__file__))
    try:
        out = subprocess.run(
            ["git", "-C", pkg_dir, "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    root = out.stdout.strip()
    if out.returncode != 0 or not root:
        return None
    return root if _is_openflo_checkout(root, pkg_dir) else None


def _is_openflo_checkout(root, pkg_dir):
    """True when ``root``'s pyproject names the openflo project and
    ``pkg_dir`` is its source tree, not an installed copy within it."""
    try:
        import tomllib
        with open(os.path.join(root, "pyproject.toml"), "rb") as f:
            name = tomllib.load(f).get("project", {}).get("name")
        rel = os.path.relpath(pkg_dir, root)
    except (ImportError, OSError, ValueError):
        return False
    if str(name or "").strip().lower().replace("_", "-") != "openflo":
        return False
    parts = {p.lower() for p in rel.replace("\\", "/").split("/")}
    return not parts & {"site-packages", "dist-packages"}


def installer_marker(prefix=None):
    """The one-click installer's marker (a dict with ``kind`` 'installer')
    read from ``prefix`` (default ``sys.prefix``), or None when there is no
    such file or it is not an installer marker."""
    path = os.path.join(prefix or sys.prefix, INSTALL_MARKER)
    try:
        # utf-8-sig: a marker re-saved by an editor that adds a byte-order
        # mark still reads.
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if isinstance(data, dict) and data.get("kind") == "installer":
        return data
    return None


def detect_install_kind():
    """``'installer'`` when the Windows installer's own Python runs OpenFlo
    (its marker is in ``sys.prefix``; checked first), ``'git'`` for a
    source checkout, else ``'pip'``."""
    if installer_marker() is not None:
        return "installer"
    return "git" if _package_git_root() else "pip"


INSTALLER_ADVICE = (
    "OpenFlo was set up with the Windows installer. To update, download the "
    "new installer from the release page and run it: it upgrades OpenFlo in "
    "place and keeps your settings. Close OpenFlo first.")


def update_command(kind=None, repo=PUBLIC_REPO):
    """The argv to upgrade OpenFlo for the given install ``kind`` (auto-detected
    when None): ``git pull`` for a checkout, ``pip install -U git+…`` for a pip
    install."""
    kind = kind or detect_install_kind()
    if kind == "installer":
        raise ValueError("an installer install updates with the next "
                         "installer, not a command")
    if kind == "git":
        root = _package_git_root() or "."
        return ["git", "-C", root, "pull", "--ff-only"]
    return [sys.executable, "-m", "pip", "install", "--upgrade",
            f"git+https://github.com/{repo}.git"]


def _git_current_branch(root):
    try:
        out = subprocess.run(["git", "-C", root, "rev-parse",
                              "--abbrev-ref", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _git_has_upstream(root):
    """True if the current branch tracks an upstream, so `git pull` knows what
    to pull. False on a development / feature checkout (or detached HEAD) with
    no tracking — where a blind `git pull` errors with a cryptic message."""
    try:
        out = subprocess.run(
            ["git", "-C", root, "rev-parse", "--abbrev-ref",
             "--symbolic-full-name", "@{u}"],
            capture_output=True, text=True, timeout=5)
        return out.returncode == 0 and bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def _git(root, *args, timeout=60):
    """Run ``git -C root <args>``; the CompletedProcess, or None when git
    could not be started or timed out."""
    try:
        return subprocess.run(["git", "-C", root, *args],
                              capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None


def _git_ok(root, *args, timeout=60):
    out = _git(root, *args, timeout=timeout)
    return out is not None and out.returncode == 0


def _git_log(out, lines=6):
    """The last few lines of a git command's output, for a failure message
    (the dialog shows only the end of the message)."""
    if out is None:
        return ""
    text = ((out.stdout or "") + (out.stderr or "")).strip()
    return "\n".join(text.splitlines()[-lines:])


def _git_default_branch(root, remote):
    """The remote's main branch name (``master`` or ``main``), from
    ``<remote>/HEAD`` when set, else whichever of the two exists; None when
    neither does."""
    out = _git(root, "symbolic-ref", "--short", f"refs/remotes/{remote}/HEAD",
               timeout=5)
    if out is not None and out.returncode == 0:
        ref = out.stdout.strip()
        if ref.startswith(remote + "/"):
            return ref[len(remote) + 1:]
    for name in ("master", "main"):
        if _git_ok(root, "rev-parse", "--verify", "--quiet",
                   f"refs/remotes/{remote}/{name}", timeout=5):
            return name
    return None


def _git_tracked_changes(root):
    """Tracked files with changes not committed ('' when none). Untracked
    files (a local launcher, a scratch venv) do not count: switching
    branches keeps them. When git cannot say, it counts as changed, so
    nothing is switched on a guess."""
    out = _git(root, "status", "--porcelain", "--untracked-files=no",
               timeout=30)
    if out is None or out.returncode != 0:
        return "(git status failed)"
    return out.stdout.strip()


def _git_operation_in_progress(root):
    """The git operation left unfinished in this checkout (a rebase,
    merge, cherry-pick or revert), or '' when there is none."""
    out = _git(root, "rev-parse", "--git-dir", timeout=5)
    if out is None or out.returncode != 0:
        return ""
    git_dir = out.stdout.strip()
    if not os.path.isabs(git_dir):
        git_dir = os.path.join(root, git_dir)
    for name, what in (("rebase-merge", "rebase"), ("rebase-apply", "rebase"),
                       ("MERGE_HEAD", "merge"),
                       ("CHERRY_PICK_HEAD", "cherry-pick"),
                       ("REVERT_HEAD", "revert")):
        if os.path.exists(os.path.join(git_dir, name)):
            return what
    return ""


def _git_branch_in_other_worktree(root, branch):
    """True when `branch` is checked out in a worktree of this repository.
    Only asked about a branch this checkout is not on, so a hit is another
    worktree. Moving the branch there (``checkout -B``, which git 2.43
    allows) would leave that worktree's files behind its branch."""
    out = _git(root, "worktree", "list", "--porcelain", timeout=10)
    if out is None or out.returncode != 0:
        return False
    return f"branch refs/heads/{branch}" in out.stdout.splitlines()


def _run_git_update(root, repo=PUBLIC_REPO, remote="origin"):
    """Update a source checkout to the newest ``<remote>/<main branch>``.

    Fetches (pruning branches deleted on GitHub), then:

    - on the main branch: fast-forward it;
    - on another branch (or a detached HEAD) whose work is all in the main
      branch, and whose upstream is gone, missing, or also merged: switch to
      the main branch and fast-forward it. This is the checkout left on a
      merged pull request's branch after GitHub deleted it, where a plain
      ``git pull`` fails with "no such ref was fetched";
    - on a branch with work not yet in the main branch: pull that branch
      when its upstream still exists, else stop with a plain message.

    Nothing is switched while tracked files have changes."""
    if not _git_ok(root, "remote", "get-url", remote, timeout=5):
        return False, (
            f"This OpenFlo folder has no link to GitHub (no '{remote}' "
            "remote), so there is nothing to update from.\n\n"
            "Install the released build instead:\n"
            f"    pip install -U git+https://github.com/{repo}.git")
    fetch = _git(root, "fetch", "--prune", remote, timeout=300)
    if fetch is None or fetch.returncode != 0:
        return False, ("Could not download the update from GitHub. Check the "
                       "internet connection and try again.\n\n"
                       + _git_log(fetch))
    main = _git_default_branch(root, remote)
    if main is None:
        return False, (f"Could not find the main branch on '{remote}'. "
                       "Nothing was changed.")
    target = f"{remote}/{main}"
    busy = _git_operation_in_progress(root)
    if busy:
        return False, (f"A git {busy} is not finished in this OpenFlo "
                       "folder, so Update left it alone.\n\n"
                       f"Finish or abort the {busy} first, then press Update "
                       "again.")
    branch = _git_current_branch(root) or "HEAD"
    if branch != main:
        return _update_off_main(root, main, target, branch)
    return _fast_forward_main(root, main, target)


def _update_off_main(root, main, target, branch):
    """:func:`_run_git_update` for a checkout on another branch or a
    detached HEAD: pull a live development branch, switch a merged one to
    `main`, or say why neither is done."""
    merged = _git_ok(root, "merge-base", "--is-ancestor", "HEAD", target,
                     timeout=30)
    upstream_live = _git_has_upstream(root)
    upstream_merged = upstream_live and _git_ok(
        root, "merge-base", "--is-ancestor", "@{u}", target, timeout=30)
    where = ("a detached commit" if branch == "HEAD"
             else f"branch '{branch}'")
    if upstream_live and not (merged and upstream_merged):
        # A development branch still on GitHub: bring it up to date.
        out = _git(root, "pull", "--ff-only", timeout=300)
        if out is not None and out.returncode == 0:
            return True, _git_log(out, 12) or "Up to date."
        return False, (f"Could not update {where}. Nothing was "
                       "switched.\n\n" + _git_log(out))
    if not merged:
        # An upstream is set but no longer exists after the prune.
        gone = ("was deleted on GitHub and " if branch != "HEAD"
                and _git_ok(root, "config", f"branch.{branch}.merge",
                            timeout=5) else "")
        save = ("Save that work on a branch first:\n"
                "    git switch -c my-work\n"
                if branch == "HEAD" else "")
        return False, (
            f"This OpenFlo folder is on {where}, which {gone}has work "
            f"that is not in {main} yet, so Update left it alone.\n\n"
            f"{save}To use the released code, switch by hand:\n"
            f"    git checkout {main}\n"
            "then press Update again.")
    return _switch_to_main(root, main, target, where)


def _switch_to_main(root, main, target, where):
    """Switch a checkout whose work is all in `target` to the local `main`
    branch set to `target`, unless something there would be lost."""
    changes = _git_tracked_changes(root)
    if changes:
        return False, (
            f"This OpenFlo folder is on {where} (its work is already in "
            f"{main}), but OpenFlo's own files have changes that are not "
            "saved in git, so Update did not switch.\n\n"
            "Put the changes aside with:\n"
            "    git stash\n"
            "then press Update again, and bring them back with:\n"
            "    git stash pop")
    if _git_ok(root, "rev-parse", "--verify", "--quiet",
               f"refs/heads/{main}", timeout=5) and not _git_ok(
            root, "merge-base", "--is-ancestor", f"refs/heads/{main}",
            target, timeout=30):
        return False, (
            f"This OpenFlo folder is on {where}, and its {main} has "
            "commits that are not on GitHub, so Update did not switch.\n\n"
            f"Switch by hand with:\n    git checkout {main}")
    if _git_branch_in_other_worktree(root, main):
        return False, (
            f"This OpenFlo folder is on {where} (its work is already in "
            f"{main}), but {main} is open in another folder (a git "
            "worktree), so Update did not switch.\n\n"
            f"Update that folder instead, or switch here by hand.")
    # One step: point the local main branch at the fetched one (it has
    # nothing of its own, checked above) and check it out. git refuses
    # before changing anything when a file is in the way.
    out = _git(root, "checkout", "-B", main, "--track", target,
               timeout=120)
    if out is None or out.returncode != 0:
        return False, (f"Could not switch to {main}. Nothing was "
                       "changed.\n\n" + _git_log(out))
    return True, (f"Switched from {where} (its work is already in "
                  f"{main}) to {main}, now up to date.")


def _fast_forward_main(root, main, target):
    """Fast-forward the checked-out `main` to `target`."""
    out = _git(root, "merge", "--ff-only", target, timeout=300)
    if out is not None and out.returncode == 0:
        return True, _git_log(out, 10) or "Up to date."
    if not _git_ok(root, "merge-base", "--is-ancestor", "HEAD", target,
                   timeout=30):
        return False, (f"Could not bring {main} up to date: it has commits "
                       "that are not on GitHub. Nothing was changed.")
    return False, (f"Could not bring {main} up to date. Nothing was "
                   "changed.\n\n" + _git_log(out))


def run_update(kind=None, repo=PUBLIC_REPO, timeout=600):
    """Run the upgrade, returning ``(ok: bool, output: str)``. The
    caller should prompt the user to restart OpenFlo afterwards — a running
    process keeps the old code loaded. A source checkout updates through
    :func:`_run_git_update`."""
    kind = kind or detect_install_kind()
    if kind == "installer":
        # The installer's Python is upgraded by the next installer, never by
        # pip: a pip upgrade there would leave the install out of step with
        # its marker and its shortcuts.
        return False, INSTALLER_ADVICE
    if kind == "git":
        return _run_git_update(_package_git_root() or ".", repo=repo)
    cmd = update_command(kind=kind, repo=repo)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout)
    except Exception as exc:                                  # noqa: BLE001 - any failure to start the updater comes back as the failure message, with the command to run by hand
        return False, f"Update failed to start: {exc}\n\nCommand: {' '.join(cmd)}"
    log = (out.stdout or "") + (out.stderr or "")
    return out.returncode == 0, log.strip() or f"(exit {out.returncode})"
