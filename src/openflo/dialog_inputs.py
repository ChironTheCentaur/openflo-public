"""Read a number the user typed into a dialog, or say plainly what is wrong.

A dialog that reads a number with ``float(var.get())`` fails in one of two
bad ways when the text is not a number: the error escapes into the generic
"Something went wrong" window (a ``DoubleVar`` or ``IntVar`` raises
``TclError``, a ``StringVar`` read through ``float`` raises ``ValueError``),
or a ``try`` quietly swaps in a default the user never chose, and the
analysis runs with it. ``read_number`` does neither: it returns the number,
or shows which setting is wrong and what it accepts and returns None, so the
caller stops.
"""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import messagebox


def _fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def _accepts(integer, lo, hi):
    kind = "a whole number" if integer else "a number"
    if lo is not None and hi is not None:
        return f"{kind} from {_fmt(lo)} to {_fmt(hi)}"
    if lo is not None:
        return f"{kind} of at least {_fmt(lo)}"
    if hi is not None:
        return f"{kind} no larger than {_fmt(hi)}"
    return kind


def parse_number(text, *, integer=False, lo=None, hi=None):
    """``text`` as a number within [lo, hi], or None. Whole numbers may be
    written as ``8`` or ``8.0``; blanks, words, inf and nan are refused."""
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    if integer:
        if value != int(value):
            return None
        value = int(value)
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        return None
    return value


def _text(var):
    """What ``var`` (a Tk variable or an entry) holds, as typed."""
    try:
        return var.get()
    except (tk.TclError, ValueError):
        # A DoubleVar/IntVar holding text that is not a number: read the text.
        try:
            return var._tk.globalgetvar(var._name)
        except (AttributeError, tk.TclError):
            return None


def number_or(var, default, *, integer=False, lo=None, hi=None):
    """The number in ``var``, or ``default`` without a word: only for a
    setting that is hidden and so not in use."""
    text = _text(var)
    value = None if text is None else parse_number(
        text, integer=integer, lo=lo, hi=hi)
    return default if value is None else value


def read_number(var, label, *, integer=False, lo=None, hi=None, parent=None,
                title="Check the settings"):
    """The number held by ``var`` (a Tk variable or an entry), or None after
    telling the user that ``label`` must be a number within [lo, hi]."""
    text = _text(var)
    value = None if text is None else parse_number(
        text, integer=integer, lo=lo, hi=hi)
    if value is None:
        shown = '' if text is None else str(text).strip()
        got = f" (it says “{shown}”)" if shown else " (it is empty)"
        message = f"{label} must be {_accepts(integer, lo, hi)}{got}."
        if parent is None:
            messagebox.showerror(title, message)
        else:
            messagebox.showerror(title, message, parent=parent)
    return value


def read_int(var, label, *, lo=None, hi=None, parent=None,
             title="Check the settings") -> int | None:
    """``read_number`` for a whole-number setting."""
    value = read_number(var, label, integer=True, lo=lo, hi=hi,
                        parent=parent, title=title)
    return None if value is None else int(value)
