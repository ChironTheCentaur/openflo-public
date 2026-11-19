"""The one place that decides whether to use FlowUtils or its numpy ports.

FlowUtils is a pinned hard dependency on desktop; only a platform where it
cannot be installed at all (a browser runtime such as Pyodide) runs on the
numpy ports ``biexp_np`` and ``polygon_np``. ``flowutils_module`` tells
those two cases apart from a broken install: it returns None only when the
``flowutils`` package itself is not found, and lets any other import failure
(a missing or ABI-broken compiled extension, a missing dependency of
FlowUtils) raise, so a broken desktop install fails loudly instead of
quietly running on the ports.

Core module: the standard library only. No Tk.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def flowutils_module(submodule: str) -> ModuleType | None:
    """``flowutils.<submodule>``, or None when the ``flowutils`` package is
    not installed. Any other ImportError (FlowUtils installed but broken)
    propagates."""
    try:
        importlib.import_module('flowutils')
    except ModuleNotFoundError as exc:
        # Only ModuleNotFoundError: a broken extension imported as
        # ``from . import logicle_c`` raises a plain ImportError that also
        # carries the name 'flowutils'.
        if exc.name != 'flowutils':
            raise
        return None
    return importlib.import_module(f'flowutils.{submodule}')
