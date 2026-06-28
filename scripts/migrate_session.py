#!/usr/bin/env python
"""Upgrade an OpenFlo ``.flowsession`` file to the current save format.

The GUI auto-upgrades sessions when you open them; this script does the same
for batch / headless use (e.g. refreshing a folder of saved sessions after a
schema change). It is Tk-free.

    python scripts/migrate_session.py old.flowsession            # -> old_upgraded.flowsession
    python scripts/migrate_session.py old.flowsession -o new.flowsession
    python scripts/migrate_session.py *.flowsession --in-place   # overwrite each

Exit status: 0 if all files are now current (upgraded or already current),
1 if any file could not be migrated (e.g. written by a newer OpenFlo).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from openflo.session_format import (
    SESSION_VERSION,
    SessionVersionError,
    migrate_session,
)


def _process(path: str, out: str | None, in_place: bool) -> bool:
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        data, notes = migrate_session(data)
    except SessionVersionError as exc:
        print(f"[skip] {path}: {exc}", file=sys.stderr)
        return False
    except Exception as exc:                           # noqa: BLE001
        print(f"[fail] {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False

    if not notes:
        print(f"[ok]   {path}: already current (v{SESSION_VERSION})")
        return True

    dest = (path if in_place else
            out or (os.path.splitext(path)[0] + '_upgraded' +
                    os.path.splitext(path)[1]))
    with open(dest, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[up]   {path}: {'; '.join(notes)} -> {dest}")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='+', help='.flowsession file(s) to upgrade')
    ap.add_argument('-o', '--out', default=None,
                    help='output path (single input only; default <name>_upgraded)')
    ap.add_argument('--in-place', action='store_true',
                    help='overwrite each input file in place')
    args = ap.parse_args(argv)
    if args.out and (len(args.paths) > 1 or args.in_place):
        ap.error('--out works with a single input and not with --in-place')

    ok = all(_process(p, args.out, args.in_place) for p in args.paths)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
