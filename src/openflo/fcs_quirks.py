"""Reading FCS files past the writer quirks other readers accept
(``read_fcs_tolerant``), and a plain-language reason when a file cannot be
read at all (``plain_fcs_problem``).

Core module: standard library only; the reader class is passed in. No Tk, and
it does not import ``openflo.pipeline`` (pipeline imports it).
"""

import os
import warnings

# Writer quirks flowio can read past when asked, each with the option that
# reads it and the note the user gets. Tried in this order, adding options,
# because one file can carry more than one quirk.
_FCS_QUIRKS = (
    ('off_by_one', {'ignore_offset_error': True},
     "its data end is recorded one byte off (a known writer bug); "
     "the data itself is intact and was read in full"),
    ('discrepancy', {'ignore_offset_discrepancy': True},
     "its HEADER and TEXT disagree on where the data starts; "
     "read at the TEXT position, as the FCS standard says"),
    ('multiple', {'nextdata_offset': 0},
     "it holds more than one data set; only the first was loaded"),
)


def _fcs_quirk(exc):
    """Which entry of _FCS_QUIRKS ``exc`` (raised by flowio) names, or None."""
    name = type(exc).__name__
    text = str(exc).lower()
    if name == 'MultipleDataSetsError':
        return 'multiple'
    if name == 'DataOffsetDiscrepancyError':
        return 'discrepancy'
    if name == 'FCSParsingError' and 'off by 1' in text:
        return 'off_by_one'
    return None


def _quirk_note(note, exc):
    """`note` for the quirk `exc` names, saying where the data ENDS when
    that is what HEADER and TEXT disagree on (flowio raises one error for
    either end; the note always said "starts")."""
    if _fcs_data_end_discrepancy(exc):
        return note.replace('where the data starts', 'where the data ends')
    return note


def _fcs_data_end_discrepancy(exc):
    """Whether flowio's offset-discrepancy error `exc` is about the data's
    end ('DATA end byte location'), not its start."""
    return (type(exc).__name__ == 'DataOffsetDiscrepancyError'
            and ' end ' in f' {str(exc).lower()} ')


def read_fcs_tolerant(path, reader, notes):
    """``reader(path)`` (a FlowData class), retried with flowio's options for
    the writer quirks in _FCS_QUIRKS that other readers accept. Each quirk
    read past appends a plain-language note to ``notes``; anything else, or
    a quirk that persists, raises."""
    kw: dict = {}
    while True:
        try:
            if not kw:
                return reader(path)
            # flowio warns about the quirk it is reading past; the note
            # below says the same thing in the user's terms.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                return reader(path, **kw)
        except Exception as e:
            q = _fcs_quirk(e)
            entry = next((x for x in _FCS_QUIRKS if x[0] == q), None)
            if entry is None or all(kw.get(k) == v for k, v in entry[1].items()):
                raise
            kw.update(entry[1])
            notes.append(f"{os.path.basename(str(path))}: "
                         f"{_quirk_note(entry[2], e)}")


def plain_fcs_problem(exc):
    """A plain-language reason an FCS file could not be read, from the
    exception flowio raised: the raw text ("read() didn't return enough
    bytes", "invalid literal for int()") says nothing to a user."""
    import re as _re
    text = str(exc)
    m = _re.fullmatch(r"'?p(\d+)([a-z])'?", text.strip().lower())
    if isinstance(exc, KeyError) and m:
        what = {'r': 'channel range', 'b': 'bits per value',
                'n': 'channel name', 'e': 'amplification'}.get(m.group(2), '')
        return (f"required keyword $P{m.group(1)}{m.group(2).upper()}"
                + (f" ({what})" if what else '') + " is missing")
    if isinstance(exc, KeyError):
        key = text.strip("'\"").upper()
        return f"required keyword ${key} is missing"
    if isinstance(exc, (EOFError, ValueError, IndexError, struct_error_types())):
        return "the file is cut short, damaged, or not an FCS file"
    low = text.lower()
    q = _fcs_quirk(exc)
    if q == 'off_by_one':
        return "its data end is recorded one byte off, and reading past that failed"
    if q == 'discrepancy':
        where = 'ends' if _fcs_data_end_discrepancy(exc) else 'starts'
        return (f"its HEADER and TEXT disagree on where the data {where}, and "
                "the data could not be read at either position")
    if q == 'multiple':
        return "it holds more than one data set, and the first could not be read"
    if 'greater than file size' in low or 'offset' in low:
        return "the file is cut short (its data section runs past the end)"
    return "the file is damaged or not an FCS file"


def struct_error_types():
    """The ``struct.error`` exception class, imported lazily, for use in
    ``isinstance`` checks on read failures."""
    import struct
    return struct.error
