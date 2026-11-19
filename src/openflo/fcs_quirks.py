"""Reading FCS files past the writer quirks other readers accept, and the
full-resolution data set of a Beckman Coulter ``.lmd`` file
(``read_fcs_tolerant``), and a plain-language reason when a file cannot be
read at all (``plain_fcs_problem``).

Core module: standard library only; the reader class is passed in. No Tk, and
it does not import ``openflo.pipeline`` (pipeline imports it).
"""

import os
import re
import warnings

from .exceptions import FcsParseError
from .fcs_containers import ContainerError
from .fcs_data_sets import MissingDataSetError, data_set_offset
from .paths import is_fcs_container
from .vendor_formats import vendor_message

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


def read_fcs_tolerant(path, reader, notes, set_number=None):
    """``reader(path)`` (a FlowData class), retried with flowio's options for
    the writer quirks in _FCS_QUIRKS that other readers accept. Each quirk
    read past appends a plain-language note to ``notes``; anything else, or
    a quirk that persists, raises.

    A Beckman Coulter list-mode file (``.lmd``) holds an FCS 2.0 data set
    followed by an FCS 3.0 copy; the FCS 3.0 copy is read, because it holds
    the full-resolution values (see ``_read_lmd_second``). Which copy was
    read is set on the result as ``lmd_data_set`` (``LMD_FCS3`` or
    ``LMD_FCS2``). Any other file with more than one data set loads its
    first, with a note saying so, unless `set_number` names the data set
    to read (from 1, as ``fcs_data_sets.list_data_sets`` counts them; the
    editor opens each as a sample of its own). That data set is read
    without the note. A file that no longer holds it raises
    ``fcs_data_sets.MissingDataSetError``. An .lmd file ignores
    `set_number`: its copies are chosen as above.

    A BD Accuri file (``paths.is_fcs_container``) is read from a copy of its
    FCS member `set_number` (``fcs_containers.member_file``); without one,
    its first, with a note when it holds more.

    An instrument file OpenFlo cannot read (``vendor_formats``, decided by
    extension) raises ``FcsParseError`` with the message that says how to
    export FCS from it, before any byte is read."""
    vendor = vendor_message(path)
    if vendor:
        raise FcsParseError(vendor)
    if is_fcs_container(path):
        return _read_container_member(path, reader, notes, set_number)
    if set_number is not None and not is_lmd_path(path):
        offset = data_set_offset(path, set_number)
        return _read_with_quirks(path, reader, notes,
                                 {'nextdata_offset': offset})
    if is_lmd_path(path):
        fcs, fell_back = _read_lmd_second(path, reader, notes)
        if fcs is not None:
            return fcs
        if fell_back:                   # its note already says which was read
            return read_lmd_first(path, reader, notes)
    return _read_with_quirks(path, reader, notes, {})


def _read_container_member(path, reader, notes, set_number):
    """``read_fcs_tolerant`` for FCS member `set_number` (None: the first)
    of the BD Accuri file at `path`."""
    from .fcs_containers import member_file
    copy, member, count = member_file(path, set_number)
    if set_number is None and count > 1:
        notes.append(
            f"{os.path.basename(str(path))}: it holds {count} samples, one "
            f"per well; only the first ({member.label}) was read. Open it "
            "with Add FCS in the editor to get one sample per well.")
    return read_fcs_tolerant(copy, reader, notes)


def read_lmd_first(path, reader, notes):
    """``reader(path)`` for the first (FCS 2.0) data set of the ``.lmd``
    file at ``path``, read past the quirks in _FCS_QUIRKS as
    ``read_fcs_tolerant`` reads, and marked ``LMD_FCS2``. Used when the FCS
    3.0 copy cannot be read, and to read again the copy a sample was loaded
    from (``FlowSample.restore_fcs_header``)."""
    fcs = _read_with_quirks(path, reader, notes, {'nextdata_offset': 0})
    _mark_lmd_set(fcs, LMD_FCS2)
    return fcs


# Which copy of an .lmd file was read (``lmd_data_set`` on the reader's
# result): the FCS 3.0 copy holds raw detector values, with the instrument's
# matrix (when there is one) as a keyword; the FCS 2.0 copy was already
# compensated by the instrument.
LMD_FCS3 = 'fcs3'
LMD_FCS2 = 'fcs2'
LMD_COPIES = (LMD_FCS3, LMD_FCS2)

_LMD_FALLBACK = ("its full-resolution (FCS 3.0) data set could not be read, "
                 "so its first (FCS 2.0) data set was loaded instead. That "
                 "copy was already compensated by the instrument, so OpenFlo "
                 "will not compensate it again")


def is_lmd_path(path):
    """Whether ``path`` names a Beckman Coulter list-mode (``.lmd``) file."""
    return str(path).lower().endswith('.lmd')


def lmd_data_set(fcs):
    """Which copy of an ``.lmd`` file the reader's result ``fcs`` holds
    (``LMD_FCS3`` or ``LMD_FCS2``), or None for any other file."""
    return getattr(fcs, 'lmd_data_set', None)


def _mark_lmd_set(fcs, which):
    """Record on the reader's result ``fcs`` which .lmd copy it holds."""
    try:
        fcs.lmd_data_set = which
    except AttributeError:          # a result that takes no attributes
        pass


def _missing_range(exc):
    """Whether ``exc`` is flowio's KeyError for a missing $PnR, which the
    caller reads past by filling the range (FlowSample's retry)."""
    return (isinstance(exc, KeyError)
            and re.fullmatch(r'p\d+r', str(exc).strip('\'"').lower())
            is not None)


def _read_lmd_second(path, reader, notes):
    """``(fcs, fell_back)``: the second data set of the ``.lmd`` file at
    ``path`` (its FCS 3.0 copy, which $NEXTDATA in the first data set points
    to), or ``(None, False)`` when its TEXT cannot be read or names no second
    data set (reading it as a plain file then reports why, or reads the one
    it has). When the second data set cannot be read, a note says the first
    is loaded instead and ``(None, True)`` is returned; the caller then reads
    the first. A missing $PnR in the second data set raises its KeyError, so
    the caller can fill the range and read that data set again."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            head = reader(path, only_text=True, nextdata_offset=0)
        nxt = int(str(head.text.get('nextdata', '0')).strip() or 0)
    except Exception:  # noqa: BLE001 - reading the first data set reports it
        return None, False
    if nxt <= 0:
        return None, False
    second_notes: list = []
    try:
        fcs = _read_with_quirks(path, reader, second_notes,
                                {'nextdata_offset': nxt})
    except Exception as e:  # noqa: BLE001 - the first data set is loaded instead
        if _missing_range(e):
            raise
        notes.append(f"{os.path.basename(str(path))}: {_LMD_FALLBACK}")
        return None, True
    notes.extend(second_notes)
    _mark_lmd_set(fcs, LMD_FCS3)
    return fcs, False


def _read_with_quirks(path, reader, notes, kw):
    """``reader(path, **kw)``, retried with the options in _FCS_QUIRKS (see
    ``read_fcs_tolerant``). A file that still cannot be read is read again
    through ``fcs_repair`` when it carries a HEADER or TEXT fault that
    module mends; its notes say what was repaired. A note also says how
    many BD Accuri values above the range were pinned at its top
    (``fcs_read.FlowData.over_range_pinned``)."""
    fcs = _read_mended(path, reader, notes, kw)
    note = over_range_note(path, getattr(fcs, 'over_range_pinned', None))
    if note:
        notes.append(note)
    return fcs


def over_range_note(path, pinned) -> str:
    """The loading note for `pinned` (``{channel: values pinned}``) of the
    file at `path`; '' when none was."""
    if not pinned:
        return ''
    total = sum(pinned.values())
    each = ', '.join(f'{ch} {n:,}' for ch, n in pinned.items())
    return (f"{os.path.basename(str(path))}: {total:,} value(s) above "
            f"the channel's range ({each}) were set to the top of the "
            f"scale instead of wrapping round to small values")


def _read_mended(path, reader, notes, kw):
    """``_read_with_quirks`` without the over-range note."""
    tried: list = []
    try:
        fcs = _read_quirks_only(path, reader, tried, kw)
    except Exception:
        fcs = _read_repaired(path, reader, notes, kw)
        if fcs is None:
            raise
        return fcs
    if getattr(fcs, 'events', ()) is None:
        # FlowIO gives no events, without an error, for integer data whose
        # $PnB is not 8, 16 or 32 (and for a TEXT-only read).
        mended = _read_repaired(path, reader, notes, kw)
        if mended is not None:
            return mended
    return _read_text_end_mended(path, reader, notes, kw, fcs, tried)


def _read_text_end_mended(path, reader, notes, kw, fcs, tried):
    """`fcs` (read from `path` with the quirk notes `tried`), read again
    through ``fcs_repair`` when TEXT's last value has no closing delimiter
    and FlowIO cut its last character; a note says so either way."""
    from .fcs_repair import CUT_LAST_VALUE_NOTE, repair_fcs
    name = os.path.basename(str(path))
    try:
        fixed = repair_fcs(path, kw.get('nextdata_offset') or 0,
                           text_end_only=True)
    except (OSError, ValueError):
        fixed = None
    if fixed is None:
        notes.extend(tried)
        return fcs
    src, repairs = fixed
    more: list = []
    try:
        mended = None if src is None else _read_quirks_only(src, reader,
                                                            more, kw)
    except Exception:  # noqa: BLE001 - keep what was read, and say so
        mended = None
    notes.extend(tried if mended is None else ())
    notes.extend(f"{name}: {r}" for r in (
        repairs if mended is not None else [CUT_LAST_VALUE_NOTE]))
    notes.extend(more if mended is not None else ())
    return fcs if mended is None else mended


def _read_repaired(path, reader, notes, kw):
    """``_read_quirks_only`` on the view of `path` that ``repair_fcs``
    mends, its notes added to `notes`; None when nothing is mended or the
    mended file cannot be read either (the caller raises the first error).
    A missing $PnR is raised, so FlowSample can fill the range."""
    from .fcs_repair import repair_fcs
    try:
        fixed = repair_fcs(path, kw.get('nextdata_offset') or 0)
    except (OSError, ValueError):
        return None
    if fixed is None:
        return None
    src, repairs = fixed
    more: list = []
    try:
        fcs = _read_quirks_only(src, reader, more, kw)
    except Exception as e:  # noqa: BLE001 - the first error is the one to report
        if _missing_range(e):
            raise
        return None
    name = os.path.basename(str(path))
    notes.extend(f"{name}: {r}" for r in repairs)
    notes.extend(more)
    return fcs


def _read_quirks_only(path, reader, notes, kw):
    """``reader(path, **kw)``, retried with the options in _FCS_QUIRKS."""
    kw = dict(kw)
    first = True
    while True:
        try:
            if first:
                return reader(path, **kw)
            # flowio warns about the quirk it is reading past; the note
            # below says the same thing in the user's terms.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                return reader(path, **kw)
        except Exception as e:
            first = False
            q = _fcs_quirk(e)
            entry = next((x for x in _FCS_QUIRKS if x[0] == q), None)
            if entry is None or all(kw.get(k) == v for k, v in entry[1].items()):
                raise
            kw.update(entry[1])
            notes.append(f"{os.path.basename(str(path))}: "
                         f"{_quirk_note(entry[2], e)}")


def unread_values_problem(text):
    """Why FlowIO read the FCS TEXT `text` (its keyword dict) but gave no
    events, in plain words. It does that, without an error, for integer
    data whose $PnB is not 8, 16 or 32, which fcs_repair could not confirm
    from the file's own sizes."""
    kw = {str(k).lstrip('$').lower(): str(v).strip() for k, v in text.items()}
    try:
        n_par = int(kw.get('par', ''))
    except ValueError:
        n_par = 0
    for i in range(1, n_par + 1):
        bits = kw.get(f'p{i}b', '')
        if bits not in ('8', '16', '32'):
            return (f"its bits per value $P{i}B is {bits or 'blank'}, and the "
                    "data size does not say how the values are stored")
    return "its values could not be read"


def plain_fcs_problem(exc):
    """A plain-language reason an FCS file could not be read, from the
    exception flowio raised: the raw text ("read() didn't return enough
    bytes", "invalid literal for int()") says nothing to a user."""
    import re as _re
    text = str(exc)
    if isinstance(exc, (MissingDataSetError, ContainerError)):
        return text
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
