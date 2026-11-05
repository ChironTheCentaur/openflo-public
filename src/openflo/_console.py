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
        except Exception:  # noqa: BLE001 - stream may not support reconfigure
            pass


def hide_future_warnings() -> None:
    """Keep library FutureWarnings (pandas, scikit-learn, ...) off the console.

    They warn about a dependency's next release, which a user running an
    analysis cannot act on. Call this in an application's ``main()``, never
    at import time: doing it on import would hide every FutureWarning from
    any program that imported the library (BP3).
    """
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)


def worker_settings() -> dict:
    """This process's console settings that a spawned worker starts without:
    the root logger's level and format, and the categorical palette.

    Pass it to `start_worker` through ``initargs`` when building a process
    pool, so a worker logs what the parent would (``-v``, ``-q``) and draws
    with the palette the user picked."""
    import logging
    root = logging.getLogger()
    fmt = next((h.formatter for h in root.handlers if h.formatter), None)
    flow_sample = sys.modules.get('openflo.flow_sample')
    # Exact type: the '{' and '$' style classes subclass the '%' one.
    style = next((key for key, (cls, _default) in logging._STYLES.items()
                  if fmt and type(fmt._style) is cls), '%')
    return {'handled': bool(root.handlers), 'level': root.level,
            'format': fmt._fmt if fmt else None,
            'datefmt': fmt.datefmt if fmt else None, 'style': style,
            'palette': getattr(flow_sample, '_DEFAULT_CATEGORICAL_PALETTE',
                               'auto')}


def start_worker(settings: dict | None = None) -> None:
    """Initializer for a spawned process-pool worker.

    A spawned worker is a fresh interpreter: it keeps none of the parent's
    stream encoding, warning filters, logging set-up, plotting backend or
    palette. This sets them as `worker_settings` recorded them in the parent.
    Workers only write files, so plots use the headless Agg backend."""
    force_utf8_streams()
    hide_future_warnings()
    import matplotlib
    matplotlib.use('Agg')
    settings = settings or {}
    if settings.get('handled'):
        import logging
        try:
            logging.basicConfig(level=settings['level'],
                                format=settings['format'],
                                datefmt=settings['datefmt'],
                                style=settings.get('style', '%'), force=True)
        except Exception:  # noqa: BLE001 - an initializer that raises breaks the whole pool
            logging.basicConfig(level=settings['level'], force=True)
    if settings.get('palette', 'auto') != 'auto':
        from .flow_sample import set_default_palette
        set_default_palette(settings['palette'])
