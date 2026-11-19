"""The data sets of an FCS file that holds more than one ($NEXTDATA).

Some instruments and programs write several data sets into one FCS file:
Guava / easyCyte plate runs, Kaluza, and files
concatenated by other tools. Each data set has its own HEADER, TEXT and DATA,
and the TEXT keyword $NEXTDATA gives the byte offset of the next one,
counted from the start of the data set that names it (FCS 2.0 says so; the
FCS 3.x texts do not say, and some writers count it from the start of the
file instead, which is the same thing for the second data set).

``list_data_sets`` walks that chain, reading only each HEADER and TEXT, and
stops where it goes wrong: an offset that points back to an earlier data set
(which would loop: the chain only ever moves forward), past the end of the
file, or at no FCS HEADER. ``sample_data_sets``
says which data sets the editor opens as samples of their own, and
``data_set_names`` the names they are listed under. A Beckman Coulter
``.lmd`` file is not split: its two data sets hold the same events (see
``fcs_quirks.read_fcs_tolerant``), and neither is a pair laid out as one in
a file named ``.fcs``.

Core module: standard library only. No Tk.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .paths import FCS_EXTENSIONS, is_fcs_container

# Bytes in an FCS HEADER (the version and six offsets).
HEADER_SIZE = 58
# The most data sets read from one file: a chain longer than this is damaged.
MAX_DATA_SETS = 10000

# The TEXT keywords a data set's sample name is taken from, in the order
# tried: the file name it was acquired to, the sample number, then the
# names instruments write for the tube or sample.
NAME_KEYWORDS = ('$FIL', '$SMNO', 'TUBE NAME', 'SAMPLE ID', '$SRC')
# The longest name taken from a keyword.
_MAX_NAME = 80


class MissingDataSetError(ValueError):
    """A sample was read from a data set its FCS file no longer holds."""


@dataclass(frozen=True)
class DataSet:
    """One data set of an FCS file: its place in the file (``number``,
    from 1), the byte ``offset`` of its HEADER, its FCS ``version`` and its
    TEXT ``keywords`` (keys upper-case, '$' kept)."""

    number: int
    offset: int
    version: str
    keywords: dict[str, str] = field(default_factory=dict)


def parse_text(raw: bytes) -> list[tuple[str, str]]:
    """The (key, value) pairs of the TEXT segment bytes `raw`, in file
    order, keys in their original case. A doubled delimiter is an escaped
    literal delimiter. Raises IndexError for an empty segment."""
    try:
        s = raw.decode('utf-8')
    except UnicodeDecodeError:
        s = raw.decode('latin-1')
    delim, parts, cur, i = s[0], [], [], 1
    while i < len(s):
        c = s[i]
        if c == delim:
            if i + 1 < len(s) and s[i + 1] == delim:     # escaped delimiter
                cur.append(delim)
                i += 2
                continue
            parts.append(''.join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    if cur:
        parts.append(''.join(cur))
    # strict=False: a malformed TEXT with a trailing key and no value drops it
    return list(zip(parts[0::2], parts[1::2], strict=False))


def read_header_text(fh, offset: int = 0) -> tuple[str, list[tuple[str, str]]]:
    """``(version, TEXT pairs)`` of the data set whose HEADER starts at byte
    `offset` of the open binary file `fh`. Raises ValueError (an offset that
    is not a number) or IndexError (no TEXT) for one that cannot be read."""
    fh.seek(offset)
    hdr = fh.read(HEADER_SIZE)
    if len(hdr) < 26 or not hdr.startswith(b'FCS'):
        raise ValueError('no FCS HEADER there')
    t0, t1 = int(hdr[10:18]), int(hdr[18:26])
    if t0 <= 0 or t1 < t0:
        raise ValueError('its TEXT offsets are not usable')
    fh.seek(offset + t0)
    raw = fh.read(t1 - t0 + 1)
    return hdr[3:10].decode('latin-1').strip(), parse_text(raw)


def _is_header(fh, offset: int, size: int, whole: bool = True) -> bool:
    """Whether an FCS HEADER starts at byte `offset` of `fh` (`size`
    bytes): with `whole`, one whose TEXT reads, else any that starts with
    the bytes 'FCS' (which the DATA of another data set can hold)."""
    if offset < 0 or offset + HEADER_SIZE > size:
        return False
    if not whole:
        fh.seek(offset)
        return fh.read(3) == b'FCS'
    try:
        read_header_text(fh, offset)
    except (ValueError, IndexError):
        return False
    return True


def _next_offset(fh, here: int, nextdata: int,
                 size: int) -> tuple[int | None, str]:
    """``(offset, '')`` of the data set $NEXTDATA `nextdata` names, from the
    data set at `here`, or ``(None, why)`` when it names none that can be
    read. The offset is counted from `here` (the FCS 2.0 rule), else from
    the start of the file (what some writers do); either must move forward
    to a HEADER whose TEXT reads, so a chain never comes back to a data set
    it has read and cannot loop, even when the offset names the data set
    it is in."""
    # A HEADER that reads wins; failing that, one that only starts like
    # one (whose data set is then said to be unreadable).
    for whole, cand in ((True, here + nextdata), (True, nextdata),
                        (False, here + nextdata), (False, nextdata)):
        if cand > here and _is_header(fh, cand, size, whole):
            return cand, ''
    if nextdata == here:              # counted from the file start: itself
        return None, 'names its own data set'
    if nextdata <= 0:
        return None, 'points back to an earlier data set'
    if here + nextdata + HEADER_SIZE > size and nextdata + HEADER_SIZE > size:
        return None, 'points past the end of the file'
    return None, 'does not point to a data set'


def list_data_sets(path) -> tuple[list[DataSet], str]:
    """``(data sets, note)`` of the FCS file at `path`, in file order, read
    from their HEADER and TEXT only. `note` is '' when the $NEXTDATA chain
    ends cleanly (a value of 0), else says, for the user, where it went
    wrong; the data sets before that are still listed. Raises OSError,
    ValueError or IndexError when the first data set cannot be read (the
    loader then reports why)."""
    size = os.path.getsize(path)
    name = os.path.basename(str(path))
    sets: list[DataSet] = []
    offset = 0
    with open(path, 'rb') as fh:
        while True:
            try:
                version, pairs = read_header_text(fh, offset)
            except (ValueError, IndexError):
                if not sets:
                    raise
                return sets, (f"{name}: data set {len(sets) + 1} could not "
                              "be read, so only the data sets before it "
                              "were opened")
            kw = {str(k).upper(): str(v) for k, v in pairs}
            sets.append(DataSet(len(sets) + 1, offset, version, kw))
            try:
                nextdata = int(kw.get('$NEXTDATA', '0').strip() or 0)
            except ValueError:
                nextdata = 0
            if nextdata == 0:
                return sets, ''
            if len(sets) >= MAX_DATA_SETS:
                return sets, (f"{name}: it names more than {MAX_DATA_SETS} "
                              "data sets; only the first "
                              f"{MAX_DATA_SETS} were opened")
            nxt, why = _next_offset(fh, offset, nextdata, size)
            if nxt is None:
                return sets, (f"{name}: the next-data-set offset ($NEXTDATA) "
                              f"of data set {len(sets)} {why}, so only "
                              f"{_count(len(sets))} were opened")
            offset = nxt


def _count(n: int) -> str:
    """'the first data set' / 'the first 3 data sets'."""
    return 'the first data set' if n == 1 else f'the first {n} data sets'


def same_events_copies(sets: list[DataSet]) -> bool:
    """Whether `sets` is the layout of a Beckman Coulter list-mode file: an
    FCS 2.0 data set followed by an FCS 3.x copy of the same events (the
    same $PAR and $TOT). Such a file is never split into two samples, even
    when it is named ``.fcs``."""
    if len(sets) != 2:
        return False
    a, b = sets
    return (a.version.startswith('2') and b.version.startswith('3')
            and a.keywords.get('$PAR') == b.keywords.get('$PAR')
            and a.keywords.get('$TOT') == b.keywords.get('$TOT'))


def sample_data_sets(path) -> tuple[list[DataSet], str]:
    """``(data sets, note)``: the data sets of the FCS file at `path` that
    open as samples of their own, and the note ``list_data_sets`` gave
    about its chain. Empty for a file opened as one sample: a file with one
    data set, an ``.lmd`` file, a pair of copies of the same events
    (``same_events_copies``), or a file whose first data set cannot be read
    (the loader says why).

    A BD Accuri file (``paths.is_fcs_container``) lists its FCS members
    instead (``fcs_containers.list_members``), even just one: each opens as
    a sample named after its well. It is empty when the file is not such an
    archive or holds no FCS member; the loader then says why."""
    if is_fcs_container(path):
        from .fcs_containers import list_members
        try:
            members, note = list_members(path)
        except (OSError, ValueError):
            return [], ''
        return list(members), note
    if str(path).lower().endswith('.lmd'):
        return [], ''
    try:
        sets, note = list_data_sets(path)
    except (OSError, ValueError, IndexError):
        return [], ''
    if len(sets) < 2 or same_events_copies(sets):
        return [], ''
    return sets, note


def data_set_location(path, number: int | None) -> tuple[str, int]:
    """``(file, offset)``: where the HEADER of data set `number` (from 1;
    None: the first) of the file at `path` is. For a BD Accuri file that is
    the start of a copy of its FCS member `number`, in OpenFlo's cache
    (``fcs_containers.member_file``). Raises MissingDataSetError when the
    file holds fewer, and ``fcs_containers.ContainerError`` (a ValueError)
    for an Accuri file that cannot be opened."""
    if is_fcs_container(path):
        from .fcs_containers import member_file
        return member_file(path, number)[0], 0
    return str(path), (data_set_offset(path, number) if number else 0)


def data_set_offset(path, number: int) -> int:
    """The byte offset of data set `number` (from 1) of the FCS file at
    `path`. Raises MissingDataSetError when the file holds fewer."""
    sets, _note = list_data_sets(path)
    if not 1 <= int(number) <= len(sets):
        held = ('one data set' if len(sets) == 1
                else f'{len(sets)} data sets')
        raise MissingDataSetError(
            f"its data set {number} was not found: the file now holds "
            f"{held}")
    return sets[int(number) - 1].offset


def _stem(path) -> str:
    """The file name of `path` (either separator) without .fcs / .lmd."""
    base = re.split(r'[\\/]', str(path))[-1]
    root, ext = os.path.splitext(base)
    return root if ext.lower() in FCS_EXTENSIONS else base


def _clean(key: str, value: str | None) -> str:
    """Keyword `key`'s `value` as a sample name: a file name ($FIL) without
    its folder and extension, control characters dropped, white space
    collapsed and the length capped; '' when nothing is left."""
    if value is None:
        return ''
    v = _stem(value) if key == '$FIL' else str(value)
    v = ''.join(ch if ch.isprintable() else ' ' for ch in v)
    return ' '.join(v.split())[:_MAX_NAME].strip()


def data_set_names(sets: list[DataSet], path) -> list[str]:
    """The name each of `sets` (from the file at `path`) is listed under,
    before the editor makes it unique among its samples: the first keyword
    of NAME_KEYWORDS that every data set has, with a different value in
    each; else ``'<file name> #<n>'``. The members of a BD Accuri file keep
    the labels they were listed with (``fcs_containers.Member``)."""
    labels = [getattr(s, 'label', '') for s in sets]
    if sets and all(labels):
        return labels
    for key in NAME_KEYWORDS:
        vals = [_clean(key, s.keywords.get(key)) for s in sets]
        if all(vals) and len(set(vals)) == len(vals):
            return vals
    stem = _stem(path)
    return [f'{stem} #{s.number}' for s in sets]


def match_data_set(sets: list[DataSet],
                   keywords: dict[str, str]) -> int | None:
    """The number of the one data set of `sets` whose TEXT agrees with
    `keywords` (``data_set_matches``), or None when no data set or more
    than one agrees: a workspace names only the file, so these keywords
    are all that says which data set a sample was."""
    hits = data_set_matches(sets, keywords)
    return hits[0] if len(hits) == 1 else None


def data_set_matches(sets: list[DataSet],
                     keywords: dict[str, str]) -> list[int]:
    """The numbers of the data sets of `sets` whose TEXT agrees with
    `keywords` (a FlowJo workspace sample's <Keywords>, keys upper-case):
    the same $TOT and the same value of the first of NAME_KEYWORDS the
    workspace records that tells the data sets apart (a keyword with the
    same value in every data set, such as a $FIL naming the whole file,
    says nothing, so the next is tried). Empty when the workspace records
    none of these."""
    want = {str(k).upper(): str(v).strip() for k, v in keywords.items()}
    keys = ['$TOT'] if want.get('$TOT') else []
    keys += [k for k in NAME_KEYWORDS if want.get(k)
             and len({s.keywords.get(k, '').strip() for s in sets}) > 1][:1]
    if not keys:
        return []
    return [s.number for s in sets
            if all(s.keywords.get(k, '').strip() == want[k] for k in keys)]
