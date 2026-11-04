"""Collecting the WARNING messages logged on one thread inside a block:
``collect_warnings``.

Core module: the standard library only. No Tk, and it does not import
``openflo.pipeline``.
"""

import contextlib
import logging

# Name kept as 'openflo.pipeline': the handler listens on the logger pipeline
# and the modules split from it log to, as it did when this lived there.
log = logging.getLogger('openflo.pipeline')


@contextlib.contextmanager
def collect_warnings():
    """The WARNING messages this module logs on THIS thread inside the
    block, as a list (the pool loads several samples at once)."""
    import threading
    me = threading.get_ident()
    said = []

    class _Grab(logging.Handler):
        def emit(self, record):
            if record.thread == me and record.levelno >= logging.WARNING:
                said.append(record.getMessage().strip().lstrip('[!] ')
                            .strip())

    h = _Grab()
    log.addHandler(h)
    try:
        yield said
    finally:
        log.removeHandler(h)
