"""One-line guard against the Windows console killing a CLI.

Every command-line entry point here prints characters outside cp1252 — check
and cross marks in the self-test table, arrows in the titration output, `Δ` in
the comparison report, superscripts in the synthetic marker names. A legacy
Windows console encodes stdout as cp1252, so printing any of them raises
``UnicodeEncodeError`` and the command dies having reported nothing.

That is worst on exactly the commands a new user runs first: `openflo-selftest`
crashed mid-table instead of saying whether the install reproduces reference
behaviour, and `openflo-doctor --help` crashed on the arrow in its own epilog.

Call this FIRST in ``main()`` — before building the argument parser. Argparse
prints ``--help`` and exits from inside ``parse_args``, so a guard placed after
that line does not protect the help text (which is how the doctor's crash
survived having the guard at all).

`openflo.cli` does the same thing at import time, which is fine for a module
that is only ever a CLI. This is a function so that importing, say,
``openflo.selftest`` as a library does not silently reconfigure the caller's
streams.
"""
from __future__ import annotations

import sys


def force_utf8_streams() -> None:
    """Re-encode stdout/stderr as UTF-8, replacing anything unencodable.

    Deliberately best-effort: a stream that does not support ``reconfigure``
    (a pytest capture buffer, a pipe wrapper, an embedded interpreter) is left
    alone rather than failing the command that was about to run.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[union-attr]
        except Exception:                                            # noqa: BLE001
            pass
