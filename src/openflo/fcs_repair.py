"""Mending the HEADER and TEXT faults that make FlowIO refuse an FCS file
other programs open (``repair_fcs``), read through a view of the file that
carries the repairs (``RepairedFcs``); the file on disk is never changed.

Each repair follows the FCS 3.1 / 3.2 standard and comes with a note that
says plainly what was repaired:

* HEADER data offsets left blank: the position is taken from $BEGINDATA /
  $ENDDATA in TEXT (0 in the HEADER when the data runs past byte
  99,999,999, as the standard writes it).
* $BEGINDATA / $ENDDATA missing from an FCS 3.x TEXT: taken from the HEADER.
* An ANALYSIS segment marked empty in a non-standard way (``0,-1``,
  ``1000,999``, blank): read as empty, which is what the writer meant.
* A $PnB that is not a whole number of bytes on integer data (``10`` for
  values stored in 16-bit words): read as the word it is stored in, keeping
  only the bits $PnR allows, as the standard says for integer data.
* $TOT missing or not a number: counted from the size of the DATA segment.
* TEXT's last value running to its recorded end with no closing delimiter,
  whose last character FlowIO would drop without a word: read up to the
  delimiter when it lies a few bytes on (the HEADER's end was short), else
  kept as written. This one is checked on every read, since such a file
  reads without an error.

The other repairs are only tried after the file failed to read as it is (see
``fcs_quirks._read_with_quirks``), so a file that reads today reads exactly
as before. Repaired TEXT is placed after the file's last byte in the view,
and the HEADER's TEXT offsets point to it, so every other offset is kept.

Core module: standard library only. No Tk, and it does not import
``openflo.pipeline``.
"""

import os

_HEADER_LEN = 58
_MAX_HEADER_OFFSET = 99_999_999       # 8 digits: the most a HEADER can hold
_WORDS = (8, 16, 32)                  # integer word sizes FlowIO reads
_FLOAT_BYTES = {'F': 4, 'D': 8}
_BLANK = b' \0\r\n\t'
_DELIM_REACH = 8          # bytes past TEXT's end searched for its delimiter
# Said when TEXT's last value is cut and the closed TEXT cannot be placed
# (the file is past what an 8-digit HEADER can point to).
CUT_LAST_VALUE_NOTE = ("its TEXT has no closing delimiter, so its last "
                       "keyword may be missing its last character")


class RepairedFcs:
    """A read-only file view of the FCS file at `path` with `patches`
    (``[(byte position, bytes)]``) written over it and `tail` added after
    its last byte. FlowIO reads it as it reads an open file; it may be read
    again after FlowIO closes it. ``str()`` gives the path, so notes name
    the file."""

    def __init__(self, path, patches, tail=b''):
        self.path = str(path)
        self.name = os.path.basename(self.path)
        self._patches = list(patches)
        self._tail = bytes(tail)
        self._size = os.path.getsize(self.path)
        self._fh = None
        self._pos = 0

    def __str__(self):
        return self.path

    def seek(self, pos, whence=os.SEEK_SET):
        """Move to `pos` (from the start, here, or the end); returns it.
        The end is the file's own end, not the repaired TEXT's: FlowIO
        sizes the file from it and refuses DATA that runs past it."""
        base = {os.SEEK_SET: 0, os.SEEK_CUR: self._pos,
                os.SEEK_END: self._size}[whence]
        self._pos = base + pos
        return self._pos

    def tell(self):
        """The current position."""
        return self._pos

    def read(self, n=-1):
        """Up to `n` bytes from the current position (all with -1). A read
        that starts in the file stops at its end, so a DATA segment that
        runs past the end is short, as it is in the file itself, and never
        takes in the repaired TEXT."""
        if self._pos >= self._size:
            at = self._pos - self._size
            out = self._tail[at:] if n is None or n < 0 else \
                self._tail[at:at + n]
        else:
            if self._fh is None:
                self._fh = open(self.path, 'rb')
            self._fh.seek(self._pos)
            out = self._fh.read(self._size - self._pos if n is None or n < 0
                                else min(n, self._size - self._pos))
        out = self._overlay(out)
        self._pos += len(out)
        return out

    def _overlay(self, data):
        """`data` (read at the current position) with the patches on it."""
        buf = None
        for at, new in self._patches:
            lo, hi = max(at, self._pos), min(at + len(new), self._pos + len(data))
            if lo < hi:
                buf = bytearray(data) if buf is None else buf
                buf[lo - self._pos:hi - self._pos] = new[lo - at:hi - at]
        return data if buf is None else bytes(buf)

    def close(self):
        """Close the file; a later read opens it again."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class _Text:
    """One TEXT segment's keywords, with value edits and added keywords
    kept apart until ``render``. Keys match without '$' and in any case,
    as FlowIO matches them."""

    def __init__(self, raw):
        self.raw = raw
        self.delim = raw[:1]
        spans = _token_spans(raw)
        self._pairs = {}
        self.last_key = ''
        for k, v in zip(spans[0::2], spans[1::2], strict=False):
            key = raw[k[0]:k[1]].decode('latin-1').replace('$', '').upper()
            self.last_key = key.strip()
            self._pairs[self.last_key] = v      # the last wins, as in FlowIO
        self._end = spans[len(spans) // 2 * 2 - 1][1] if len(spans) > 1 else 1
        self._edits = {}
        self._added = {}

    def get(self, key):
        """The value of `key` (stripped), the edited one if edited, or None."""
        if key in self._edits:
            return self._edits[key]
        if key in self._added:
            return self._added[key]
        span = self._pairs.get(key)
        if span is None:
            return None
        d = self.delim
        return self.raw[span[0]:span[1]].replace(d + d, d).decode(
            'latin-1').strip()

    def int(self, key):
        """``get(key)`` as an int, or None when missing or not a number."""
        try:
            return int(self.get(key) or '')
        except ValueError:
            return None

    def set(self, key, value):
        """Give `key` the value `value` (digits), adding it if missing."""
        (self._edits if key in self._pairs else self._added)[key] = str(value)

    @property
    def changed(self):
        """Whether any value was edited or keyword added."""
        return bool(self._edits or self._added)

    def render(self):
        """The TEXT with the edits: ends at its last value's closing
        delimiter (one is added when the writer left it out; padding and
        junk after it are dropped), then the added keywords."""
        body = bytearray(self.raw[:self._end])
        for key, value in sorted(self._edits.items(),
                                 key=lambda kv: -self._pairs[kv[0]][0]):
            s, e = self._pairs[key]
            body[s:e] = value.encode('ascii')
        d = self.delim
        body += d
        for key, value in self._added.items():
            body += b'$' + key.encode('ascii') + d + value.encode('ascii') + d
        return bytes(body)


def _token_spans(raw):
    """``[(start, end)]`` of each key and value in the TEXT bytes `raw`
    (first byte the delimiter; a doubled delimiter is an escaped one). A
    last value whose closing delimiter is missing ends at the last byte
    that is not padding."""
    d, spans, start, i = raw[:1], [], 1, 1
    while i < len(raw):
        if raw[i:i + 1] == d:
            if raw[i + 1:i + 2] == d:
                i += 2
                continue
            spans.append((start, i))
            start = i + 1
        i += 1
    rest = raw[start:].rstrip(_BLANK)
    if len(spans) % 2 and rest:
        spans.append((start, start + len(rest)))
    return spans


def _cut_last_value(raw):
    """Whether the TEXT bytes `raw` end inside a value, with no closing
    delimiter and no padding: FlowIO drops the last byte of TEXT, which is
    then that value's last character."""
    spans = _token_spans(raw)
    return len(spans) >= 2 and not len(spans) % 2 and spans[-1][1] == len(raw)


def _reach_delimiter(path, offset, head, text, t1):
    """How many bytes past TEXT's recorded end (`t1`) its closing delimiter
    lies: 1 to _DELIM_REACH, before the DATA segment, with only printable,
    non-blank bytes before it; None when it is not there."""
    d = text.delim
    more = _read_at(path, offset + t1 + 1, _DELIM_REACH + 1)
    k = more.find(d)
    while 0 <= k < _DELIM_REACH and more[k + 1:k + 2] == d:   # escaped
        k = more.find(d, k + 2)
    data = _header_int(head, 26) or text.int('BEGINDATA') or 0
    if not 0 <= k < _DELIM_REACH or (data and t1 + 1 + k >= data):
        return None
    if not all(0x21 <= c <= 0x7E for c in more[:k]):
        return None             # not the rest of a value: junk or padding
    return k + 1


def _mend_text_end(path, offset, head, raw, t1, patches):
    """``(raw, note, rewrite)`` for TEXT whose last value runs to its
    recorded end with no closing delimiter (the FCS standard ends TEXT with
    one). A delimiter a few bytes on, before DATA, means the recorded end is
    short: TEXT is read up to it (`raw` grows; the HEADER's end is patched).
    Otherwise the value is kept as written and `rewrite` asks for TEXT to be
    written again, closed. ``(raw, None, False)`` when TEXT ends properly."""
    if not _cut_last_value(raw):
        return raw, None, False
    text = _Text(raw)
    key = f"${text.last_key}" if text.last_key else "its last keyword"
    k = _reach_delimiter(path, offset, head, text, t1)
    if k:
        patches.append((offset + 18, str(t1 + k).rjust(8).encode()))
        raw += _read_at(path, offset + t1 + 1, k)
        return raw, (f"its HEADER ends TEXT {k} byte(s) too early, before "
                     "its closing delimiter; TEXT was read up to the "
                     f"delimiter, so the last keyword ({key}) keeps its "
                     "full value"), False
    return raw, (f"its TEXT has no closing delimiter; the last keyword "
                 f"({key}) was kept as written"), True


def _header_int(head, at):
    """The HEADER's 8-digit field at byte `at`, or None when not a number."""
    try:
        return int(head[at:at + 8])
    except ValueError:
        return None


def _fix_blank_header_data(head, text, patches, offset):
    """A HEADER with no usable DATA offsets: take them from TEXT."""
    if _header_int(head, 26) is not None and _header_int(head, 34) is not None:
        return None
    begin, end = text.int('BEGINDATA'), text.int('ENDDATA')
    if begin is None or end is None:
        return None
    if end > _MAX_HEADER_OFFSET:            # the standard writes 0 then
        begin = end = 0
    patches.append((offset + 26, str(begin).rjust(8).encode()
                    + str(end).rjust(8).encode()))
    return ("its HEADER does not say where the data is; it was read at "
            "the position its TEXT gives ($BEGINDATA, $ENDDATA)")


def _fix_missing_text_data(head, text):
    """An FCS 3.x TEXT with no usable $BEGINDATA / $ENDDATA: take them from
    the HEADER (FCS 2.0 needs neither; FlowIO reads its HEADER)."""
    if head[3:6] == b'2.0':
        return None
    if text.int('BEGINDATA') is not None and text.int('ENDDATA') is not None:
        return None
    begin, end = _header_int(head, 26), _header_int(head, 34)
    if not begin or not end:
        return None
    text.set('BEGINDATA', begin)
    text.set('ENDDATA', end)
    return ("its TEXT does not say where the data is ($BEGINDATA, "
            "$ENDDATA); it was read at the position its HEADER gives")


def _fix_pseudo_empty_analysis(text):
    """$BEGINANALYSIS / $ENDANALYSIS that mean 'empty' in a way the
    standard does not allow (end before start, or blank): set to 0, 0."""
    if text.get('BEGINANALYSIS') is None or text.get('ENDANALYSIS') is None:
        return None
    begin, end = text.int('BEGINANALYSIS'), text.int('ENDANALYSIS')
    if begin is not None and end is not None and end >= begin:
        return None
    text.set('BEGINANALYSIS', 0)
    text.set('ENDANALYSIS', 0)
    return ("its TEXT gives no usable position for the ANALYSIS segment "
            "(results some software stores beside the events); it was read "
            "as empty. The events are not affected")


def _data_length(head, text):
    """Bytes in the DATA segment, from TEXT (HEADER for FCS 2.0, or when
    TEXT does not say), or None."""
    begin, end = text.int('BEGINDATA'), text.int('ENDDATA')
    if head[3:6] == b'2.0' or begin is None or end is None:
        begin, end = _header_int(head, 26), _header_int(head, 34)
    if begin is None or end is None or end < begin:
        return None
    return end - begin + 1


def _fits(n_bytes, event_bytes, tot):
    """Whether `n_bytes` of DATA hold whole events of `event_bytes` (or one
    byte more: the end FlowIO's off-by-one option reads past), `tot` of
    them when the count is known."""
    if event_bytes <= 0 or n_bytes % event_bytes not in (0, 1):
        return False
    return tot is None or n_bytes // event_bytes == tot


def _widths(text) -> list[int] | None:
    """Each parameter's $PnB as an int, or None when any is not a number."""
    found = [text.int(f'P{i}B') for i in range(1, (text.int('PAR') or 0) + 1)]
    widths = [b for b in found if b is not None]
    return widths if len(widths) == len(found) else None


def _fix_bit_mask_widths(head, text):
    """Integer data whose $PnB is not a whole number of bytes (a bit count,
    10 for 10-bit values in 16-bit words): set $PnB to the word the values
    are stored in, when the DATA size and $TOT confirm it. FlowIO keeps only
    the bits $PnR allows, as the standard says for integer data."""
    if (text.get('DATATYPE') or '').upper() != 'I':
        return None
    widths, n_bytes = _widths(text), _data_length(head, text)
    if not widths or n_bytes is None:
        return None
    odd = {i: b for i, b in enumerate(widths, 1) if b not in _WORDS}
    if not odd or any(b > _WORDS[-1] or b < 1 for b in odd.values()):
        return None
    word = {i: next(w for w in _WORDS if w >= b) for i, b in odd.items()}
    stored = [word[i] if i in word else b for i, b in enumerate(widths, 1)]
    # Only the file's own $TOT can confirm the word: without it a packed
    # layout (10 bits after 10 bits) can fit the DATA size just as well.
    tot = text.int('TOT')
    if tot is None or not _fits(n_bytes, sum(stored) // 8, tot):
        return None
    for i, w in word.items():
        text.set(f'P{i}B', w)
    said = ', '.join(sorted({str(b) for b in odd.values()}))
    words = ', '.join(sorted({str(w) for w in word.values()}))
    return (f"{len(odd)} of its channels give {said} bits per value ($PnB), "
            f"which is not a whole number of bytes; the values are stored in "
            f"{words}-bit words. They were read as such, keeping only the "
            "bits each channel's range ($PnR) allows, as the FCS standard "
            "says")


def _event_bytes(text):
    """Bytes per event from $DATATYPE and $PnB, or None."""
    n_par = text.int('PAR') or 0
    kind = (text.get('DATATYPE') or '').upper()
    if n_par < 1:
        return None
    if kind in _FLOAT_BYTES:
        return n_par * _FLOAT_BYTES[kind]
    if kind != 'I':
        return None
    widths = _widths(text)
    if not widths or any(w not in _WORDS for w in widths):
        return None
    return sum(widths) // 8


def _fix_missing_event_count(head, text):
    """$TOT missing or not a number: the events the DATA segment holds."""
    if text.int('TOT') is not None:
        return None
    n_bytes, event_bytes = _data_length(head, text), _event_bytes(text)
    if n_bytes is None or event_bytes is None or not _fits(
            n_bytes, event_bytes, None):
        return None
    tot = n_bytes // event_bytes
    text.set('TOT', tot)
    return (f"its event count ($TOT) is missing; {tot:,} events were "
            "counted from the size of its data section")


def _read_at(path, pos, n):
    """`n` bytes of the file at `path` from byte `pos`."""
    with open(path, 'rb') as fh:
        fh.seek(pos)
        return fh.read(n)


def repair_fcs(path, offset=0, *, text_end_only=False):
    """``(RepairedFcs, notes)`` for the data set at byte `offset` of the FCS
    file at `path` when it carries a fault this module mends (see the module
    docstring), else None. The notes say what was repaired, in the user's
    terms. `text_end_only` mends only a last TEXT value FlowIO would cut
    (``_mend_text_end``), the one fault a file that reads can carry; when
    the mended TEXT cannot be placed, ``(None, [CUT_LAST_VALUE_NOTE])``."""
    head = _read_at(path, offset, _HEADER_LEN)
    t0, t1 = _header_int(head, 10), _header_int(head, 18)
    if (len(head) < _HEADER_LEN or not head.startswith(b'FCS')
            or t0 is None or t1 is None or t1 <= t0):
        return None
    raw = _read_at(path, offset + t0, t1 - t0 + 1)
    if len(raw) < 2:
        return None
    patches: list = []
    raw, end_note, rewrite = _mend_text_end(path, offset, head, raw, t1,
                                            patches)
    text = _Text(raw)
    notes = [end_note] if end_note else []
    if not text_end_only:
        notes += [n for n in (
            _fix_blank_header_data(head, text, patches, offset),
            _fix_missing_text_data(head, text),
            _fix_pseudo_empty_analysis(text),
            _fix_bit_mask_widths(head, text),
            _fix_missing_event_count(head, text)) if n]
    if not notes:
        return None
    tail = b''
    if text.changed or rewrite:
        tail = text.render()
        start = os.path.getsize(path) - offset
        if start + len(tail) - 1 > _MAX_HEADER_OFFSET:
            # the HEADER cannot point past 8 digits
            return (None, [CUT_LAST_VALUE_NOTE]) if text_end_only else None
        patches.append((offset + 10, str(start).rjust(8).encode()
                        + str(start + len(tail) - 1).rjust(8).encode()))
    return RepairedFcs(path, patches, tail), notes
