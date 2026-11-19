"""What's new: this version's CHANGELOG section, and whether to show it.

The CHANGELOG ships inside the package (``openflo/CHANGELOG.md``, copied in
by setup.py's build step); a source checkout reads the repository's own.
Preferences remember the last version the user started (``last_seen_version``)
so the window opens once, on the first start of a newer version. OpenFlo
3.0.0 did not record that key, so an upgrade from it is recognised by the
preferences it did save (it saves the window size on every close): the
window opens then too. A fresh install records its version silently and
shows nothing.
"""
from __future__ import annotations

import os
import re

from .first_run import PREF_KEY as WELCOME_KEY

LAST_SEEN_KEY = 'last_seen_version'
# Preferences a fresh install's first start can write before (or while) the
# What's new check runs (the welcome window's flag): they are no sign an
# earlier OpenFlo ran.
_FIRST_START_KEYS = frozenset({LAST_SEEN_KEY, WELCOME_KEY})
_HEADING = re.compile(r'^##\s+\[([^\]]+)\]', re.MULTILINE)


def _same_version(a, b):
    a, b = str(a).strip().lstrip('vV'), str(b).strip().lstrip('vV')
    if a == b:
        return True
    try:
        from packaging.version import Version
        return Version(a) == Version(b)
    except Exception:  # noqa: BLE001 - no packaging or an unparseable heading is simply not this version
        return False


def changelog_section(text, version):
    """The body of ``version``'s ``## [version]`` section of a Keep a
    Changelog text (the lines up to the next ``## [`` heading), stripped, or
    None when the text has no such section."""
    if not isinstance(text, str):
        return None
    heads = list(_HEADING.finditer(text))
    for i, m in enumerate(heads):
        if not _same_version(m.group(1), version):
            continue
        start = text.find('\n', m.end())
        start = len(text) if start < 0 else start + 1
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        # A trailing link-reference block ("[3.0.0]: https://...") is not notes.
        body = re.sub(r'(?m)^\[[^\]]+\]:\s*\S+\s*$', '', text[start:end])
        return body.strip()
    return None


def _changelog_paths():
    pkg = os.path.dirname(os.path.abspath(__file__))
    # Installed: shipped in the package. Source checkout (src layout): the
    # repository's own CHANGELOG two levels up.
    return [os.path.join(pkg, 'CHANGELOG.md'),
            os.path.join(pkg, os.pardir, os.pardir, 'CHANGELOG.md')]


def load_changelog():
    """The CHANGELOG text this copy of OpenFlo shipped with, or None."""
    for path in _changelog_paths():
        try:
            with open(path, encoding='utf-8') as f:
                return f.read()
        except OSError:
            continue
    return None


def whats_new_text(version, text=None):
    """``version``'s notes from the changelog (``text``, default the shipped
    one) as plain text (update.plain_notes, never cut), or a short plain
    sentence saying there are none."""
    if text is None:
        text = load_changelog()
    from .update import plain_notes
    section = plain_notes(changelog_section(text, version), limit=None)
    if section:
        return section
    return (f"The changelog shipped with OpenFlo {version} has no notes for "
            "this version. The full changelog is online.")


def ran_before(prefs):
    """Whether the stored preferences show that an earlier OpenFlo ran: a
    dict holding any key a fresh install's first start does not write
    itself. OpenFlo 3.0.0 saved the window size on every close but never
    ``last_seen_version``; a damaged (non-dict) file is no sign."""
    return (isinstance(prefs, dict)
            and any(k not in _FIRST_START_KEYS for k in prefs))


def first_start_of_new_version(current, read, write):
    """Record ``current`` as the last-seen version and say whether the What's
    new window should open: True when ``current`` is newer than the version
    recorded before, or when nothing is recorded but the preferences show an
    earlier OpenFlo ran (an upgrade from 3.0.0, which did not record it). A
    fresh install (no preferences of its own yet), a damaged preferences
    file and a downgrade record silently; the same version changes nothing.
    ``read()`` returns the stored preferences and ``write(key, value)``
    stores one (the GUI passes prefs.read_prefs and prefs.write_pref)."""
    prefs = read()
    if not isinstance(prefs, dict):
        prefs = {}
    last = prefs.get(LAST_SEEN_KEY)
    recorded = isinstance(last, str) and bool(last.strip())
    if recorded and _same_version(last, current):
        return False
    write(LAST_SEEN_KEY, str(current))
    if not recorded:
        return ran_before(prefs)
    return _later(current, last)


def _later(a, b):
    """True when version ``a`` is later than ``b``. Unlike update.is_newer,
    a beta counts: a tester who moves from 3.0.0 to 3.1.0b1 sees its notes."""
    try:
        from packaging.version import Version
        return Version(str(a)) > Version(str(b))
    except Exception:  # noqa: BLE001 - no packaging or an unparseable version uses update's own order
        from .update import is_newer
        return is_newer(a, b)
