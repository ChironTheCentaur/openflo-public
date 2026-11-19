"""Pocket Lab Table v1 (PLT): read and write the results table toolboxes send
to Statistics. The format is defined in ../SPEC.md; section numbers point there.

This is the Python copy of ../src/plt.ts (standard library only, Python 3.10+),
for desktop apps and Pyodide toolboxes. Copy the file as it is; both copies
pass the same files in ../vectors/, so their output is byte-identical.

    table, notices, extra = read_plt(data)      # bytes or str
    text = write_plt(table, bom=True)           # str

A table is a plain dict shaped like the TypeScript PltTable (same keys).
"""
from __future__ import annotations

import math
import re
import unicodedata
from decimal import Decimal

PLT_MAJOR = 1
PLT_MINOR = 0

LIMITS = {
    'fileBytes': 16 * 1024 * 1024,
    'lineBytes': 64 * 1024,
    'cellBytes': 4 * 1024,
    'preambleLines': 2000,
    'rows': 200_000,
    'columns': 512,
}

KNOWN_FEATURES: frozenset[str] = frozenset()

ROLES = ('id', 'group', 'factor', 'bio_rep', 'tech_rep', 'analyte', 'block', 'well_type',
         'x', 'dilution', 'measure', 'flag', 'limit', 'text')
NUMBER_ROLES = {'x', 'dilution', 'measure', 'limit'}
TEXT_ROLES = {'id', 'group', 'factor', 'bio_rep', 'tech_rep', 'analyte', 'block', 'well_type', 'flag'}
TRIMMED_ROLES = {'id', 'group', 'factor', 'bio_rep', 'tech_rep', 'analyte', 'block', 'well_type'}
SINGLE_ROLES = ('group', 'bio_rep', 'tech_rep', 'analyte', 'block', 'well_type')
UNIT_ROLES = {'x', 'measure', 'limit'}

ASSAYS = ('generic', 'dose-response', 'qpcr', 'elisa')
REPLICATES = ('biological', 'technical', 'unknown')
DESIGNS = ('independent', 'paired', 'repeated', 'unknown')
SCALES = ('linear', 'log2', 'log10', 'proportion', 'percent', 'correlation', 'count')
WELL_TYPES = ('standard', 'unknown', 'blank', 'control')
KNOWN_FLAGS = ('saturated', 'below_lod', 'above_range', 'undetermined', 'not_converged',
               'excluded', 'edited', 'low_quality', 'outlier_marked')

# Python's re is not JavaScript's; every pattern below matches the same
# strings as its TypeScript twin (ASCII classes only).
INVISIBLE = re.compile('[\u00ad\u200b\u200e\u200f\u202a-\u202e\u2060\u2066-\u2069\ufeff]')
# JavaScript's whitespace (String.prototype.trim, \\s), spelled out: Python's \\s differs.
WS = '\\t\\n\\v\\f\\r \\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000\\ufeff'
TRIM_RE = re.compile(f'^[{WS}]+|[{WS}]+\\Z')
CONTROL = re.compile('[\u0000-\u001f\u007f-\u009f]')
FORMULA_START = re.compile(r"[=+\-@']")
NUMBER_RE = re.compile(r'-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?\Z')
NAME_RE = re.compile(r'[A-Za-z][A-Za-z0-9_.-]{0,63}\Z')
ID_RE = re.compile(r'[a-z][a-z0-9-]{0,31}\Z')
SOURCE_VERSION_RE = re.compile(r'[0-9A-Za-z][0-9A-Za-z.+-]{0,31}\Z')
CREATED_RE = re.compile(r'\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])T([01]\d|2[0-3]):[0-5]\dZ\Z', re.ASCII)
VERSION_RE = re.compile(r'(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})\Z')
SHA_RE = re.compile(r'[0-9a-f]{64}\Z')
FLAG_RE = re.compile(r'[a-z0-9_]+\Z')
KEY_RE = re.compile(r'#[a-z]+\Z')
INDEX_RE = re.compile(r'[1-9][0-9]{0,5}\Z')
UNKNOWN_ROLE_RE = re.compile(r'[a-z_]{1,32}\Z')
UNKNOWN_TYPE_RE = re.compile(r'[a-z]{1,16}\Z')
PATH_RE = re.compile(r'\\|[a-z][a-z0-9+.-]*://|(?:^|[' + WS + r'"\'(=:])(?:~/|[a-z]:/|/[^' + WS + r'/]+/)'
                     r'|(?:^|[^a-z])(?:documents|desktop|downloads|pictures|dcim|dropbox|onedrive|icloud|library|users|home)/',
                     re.IGNORECASE | re.ASCII)
FILE_NAME_RE = re.compile(r'(?:^|[' + WS + r'"\'(=:/])[^' + WS + r'"\'(),;=:/]*[^' + WS + r'"\'(),;=:/.]'
                          r'\.[A-Za-z][A-Za-z0-9]{1,4}(?=\Z|[' + WS + r'"\'),;])')


def looks_like_path(s: str) -> bool:
    return bool(PATH_RE.search(s) or FILE_NAME_RE.search(s))


class PltError(ValueError):
    def __init__(self, code: str, line: int, message: str):
        super().__init__(f'Line {line}: {message}' if line else message)
        self.code = code
        self.line = line


def utf8_length(s: str) -> int:
    return len(s.encode('utf-8', 'surrogatepass'))


def _trim(s: str) -> str:
    # JavaScript's String.prototype.trim: its whitespace set, not Python's.
    return TRIM_RE.sub('', s)


def _clean(s: str) -> str:
    return INVISIBLE.sub('', unicodedata.normalize('NFC', s))


def _escape(s: str) -> str:
    return "'" + s if s and FORMULA_START.match(s) else s


def _unescape(s: str) -> str:
    return s[1:] if len(s) >= 2 and s[0] == "'" and FORMULA_START.match(s[1]) else s


def _fold_micro(unit: str) -> str:
    return unit.replace('\u03bc', '\u00b5')


def _short(s: str) -> str:
    return s[:40] + '…' if len(s) > 40 else s


# ---------------------------------------------------------------- numbers

def format_number(x) -> str:
    """ECMAScript Number.prototype.toString of x, with -0 written 0."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise PltError('bad_number', 0, 'a number cell holds a non-number.')
    if isinstance(x, int):
        if abs(x) > 2 ** 53:
            raise PltError('bad_number', 0, 'a whole number above 2^53 loses precision; write it as text.')
        x = float(x)
    if not math.isfinite(x):
        raise PltError('bad_number', 0, 'a number is not finite (write an empty cell instead).')
    if x == 0:
        return '0'
    sign = '-' if x < 0 else ''
    d = Decimal(repr(abs(x)))
    _, digits, exp = d.as_tuple()
    ds = ''.join(map(str, digits)).rstrip('0')
    k = len(ds)
    # n: position of the decimal point relative to the digit string.
    n = exp + len(digits)
    if k <= n <= 21:
        return sign + ds + '0' * (n - k)
    if 0 < n <= 21:
        return sign + ds[:n] + '.' + ds[n:]
    if -6 < n <= 0:
        return sign + '0.' + '0' * (-n) + ds
    e = n - 1
    es = ('+' if e >= 0 else '-') + str(abs(e))
    return sign + (ds if k == 1 else ds[0] + '.' + ds[1:]) + 'e' + es


def _parse_number(s: str, line: int) -> float:
    if not NUMBER_RE.match(s):
        raise PltError('bad_number', line,
                       f'"{_short(s)}" is not a number in the table\'s form (digits, "." decimal, optional exponent).')
    v = float(s)
    if not math.isfinite(v):
        raise PltError('number_overflow', line, f'{_short(s)} is too large.')
    return v


# ---------------------------------------------------------------- CSV fields

def split_fields(line: str, n: int) -> list[str]:
    out: list[str] = []
    size = len(line)
    i = 0
    while True:
        if i < size and line[i] == '"':
            cell = []
            i += 1
            while True:
                if i >= size:
                    raise PltError('bad_quote', n, 'a quote is never closed.')
                c = line[i]
                if c == '"':
                    if i + 1 < size and line[i + 1] == '"':
                        cell.append('"')
                        i += 2
                        continue
                    i += 1
                    break
                cell.append(c)
                i += 1
            if i < size and line[i] != ',':
                raise PltError('bad_quote', n, 'text after a closing quote.')
            out.append(''.join(cell))
        else:
            comma = line.find(',', i)
            end = size if comma < 0 else comma
            cell = line[i:end]
            if '"' in cell:
                raise PltError('bad_quote', n, 'a quote inside an unquoted field.')
            out.append(cell)
            i = end
        if i >= size:
            return out
        i += 1


def join_fields(cells) -> str:
    return ','.join('"' + s.replace('"', '""') + '"' if ('"' in s or ',' in s) else s for s in cells)


# ---------------------------------------------------------------- reading

def sniff_plt(data: bytes) -> bool:
    if data.startswith(b'\xef\xbb\xbf'):
        data = data[3:]
    return data.startswith(b'#plt,')


def _decode(data) -> str:
    if isinstance(data, str):
        return data[1:] if data.startswith('\ufeff') else data
    if len(data) > LIMITS['fileBytes']:
        raise PltError('too_large', 0, 'the table is larger than 16 MB.')
    if data[:2] in (b'\xff\xfe', b'\xfe\xff'):
        raise PltError('utf16', 0, 'the table is saved as UTF-16; save it as CSV (UTF-8).')
    try:
        text = bytes(data).decode('utf-8')
    except UnicodeDecodeError:
        raise PltError('not_utf8', 0, 'the table is not valid UTF-8 text.') from None
    return text[1:] if text.startswith('\ufeff') else text


def _cell_text(raw: str, line: int) -> str:
    if utf8_length(raw) > LIMITS['cellBytes']:
        raise PltError('cell_too_long', line, 'a cell is longer than 4 KB.')
    return _clean(_unescape(raw))


def read_plt(data):
    """(table, notices, extra_keys) from PLT bytes or text; PltError if it breaks a rule."""
    text = _decode(data)
    if isinstance(data, str) and utf8_length(text) > LIMITS['fileBytes']:
        raise PltError('too_large', 0, 'the table is larger than 16 MB.')
    if not text.startswith('#plt,'):
        raise PltError('not_plt', 1, 'this is not a Pocket Lab Table (it must start with "#plt,").')
    lines = text.split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    notices: list[str] = []
    extra: dict[str, list[list[str]]] = {}
    seen: dict[str, list[str]] = {}
    settings: list[list[str]] = []
    methods: list[str] = []
    analytes: list[dict] = []
    inputs: list[dict] = []
    cols: list[dict] = []
    version = (0, 0)

    def physical(k: int) -> str:
        line = lines[k]
        if line.endswith('\r'):
            line = line[:-1]
        n = k + 1
        if '\r' in line:
            raise PltError('bare_cr', n, 'a line ends with CR alone.')
        if line == '':
            raise PltError('blank_line', n, 'blank lines are not allowed.')
        if utf8_length(line) > LIMITS['lineBytes']:
            raise PltError('line_too_long', n, 'the line is longer than 64 KB.')
        if CONTROL.search(line):
            raise PltError('control_char', n, 'the line holds a control character (tab or similar).')
        return line

    def singleton(key: str, fields: list[str], n: int) -> None:
        if key in seen:
            raise PltError('duplicate_key', n, f'{key} appears twice.')
        seen[key] = fields

    i = 0
    while i < len(lines):
        n = i + 1
        raw = physical(i)
        if not raw.startswith('#'):
            break
        if i >= LIMITS['preambleLines']:
            raise PltError('too_many_preamble', n, 'more than 2,000 lines before the header.')
        comma = raw.find(',')
        key = raw if comma < 0 else raw[:comma]
        if not KEY_RE.match(key):
            raise PltError('bad_key', n, f'"{_short(key)}" is not a key (# then lowercase letters).')
        fields = [] if comma < 0 else split_fields(raw[comma + 1:], n)
        while fields and fields[-1] == '':
            fields.pop()

        def f(k: int, fields=fields, n=n) -> str:
            return _cell_text(fields[k] if k < len(fields) else '', n)

        def g(k: int, fields=fields) -> str:
            return fields[k] if k < len(fields) else ''

        def bad(what: str, key=key, n=n) -> PltError:
            return PltError('bad_field', n, f'{key}: {what}')

        if i == 0:
            if key != '#plt':
                raise PltError('not_plt', n, 'this is not a Pocket Lab Table (it must start with "#plt,").')
            m = VERSION_RE.match(g(0))
            if not m:
                raise PltError('bad_version', n, f'"{_short(g(0))}" is not a version (major.minor).')
            version = (int(m.group(1)), int(m.group(2)))
            if version[0] > PLT_MAJOR:
                raise PltError('newer_major', n,
                               'This table was made by a newer version of Pocket Lab. Update the app to open it.')
            if version[0] < 1:
                raise PltError('bad_version', n, f'version {g(0)} is not one this app reads.')
            if version[1] > PLT_MINOR:
                notices.append('This table was made by a newer version of Pocket Lab; some details may not show.')
            seen['#plt'] = fields
            i += 1
            continue

        if key == '#plt':
            raise PltError('duplicate_key', n, '#plt appears twice.')
        if key == '#requires':
            singleton(key, fields, n)
            for feat in (_trim(x) for x in fields):
                if feat and feat not in KNOWN_FEATURES:
                    raise PltError('unknown_requires', n, 'This table needs a newer version of Pocket Lab to be '
                                   'read correctly. Update the app to open it.')
        elif key == '#source':
            singleton(key, fields, n)
            if not ID_RE.match(g(0)):
                raise bad('toolbox id must be lowercase letters, digits and -.')
            if not SOURCE_VERSION_RE.match(g(1)):
                raise bad('version is missing or malformed.')
            if not ID_RE.match(g(2)):
                raise bad('analysis id must be lowercase letters, digits and -.')
        elif key == '#created':
            singleton(key, fields, n)
            if not CREATED_RE.match(g(0)):
                raise bad('time must be YYYY-MM-DDTHH:MMZ.')
        elif key in ('#title', '#rows', '#assay', '#replicates', '#design'):
            singleton(key, fields, n)
        elif key == '#control':
            singleton(key, fields, n)
            if not g(0) or not _trim(f(1)):
                raise bad('needs a column name and a level.')
        elif key == '#analyte':
            value = _trim(f(0))
            kind = g(1)
            if not value:
                raise bad('needs a value.')
            if kind not in ('target', 'reference'):
                raise bad('kind must be target or reference.')
            if any(a['value'] == value for a in analytes):
                raise PltError('duplicate_key', n, f'#analyte {_short(value)} appears twice.')
            amp = None
            if g(2):
                amp = _parse_number(g(2), n)
                if not amp > 1:
                    raise bad('amplification factor must be above 1 (2 = 100 % efficient).')
            analytes.append({'value': value, 'kind': kind, 'amplification': amp})
        elif key == '#input':
            if not INDEX_RE.match(g(0)):
                raise bad('index must be a whole number from 1.')
            index = int(g(0))
            if any(x['index'] == index for x in inputs):
                raise PltError('duplicate_key', n, f'#input {index} appears twice.')
            label = _trim(f(1))
            if not label:
                raise bad('needs a label.')
            sha = g(2)
            if sha and not SHA_RE.match(sha):
                raise bad('SHA-256 must be 64 lowercase hex digits.')
            inputs.append({'index': index, 'label': label, 'sha256': sha or None})
        elif key == '#setting':
            if not _trim(f(0)):
                raise bad('needs a name.')
            settings.append([_trim(f(0)), f(1)])
        elif key == '#methods':
            methods.append(f(0))
        elif key == '#col':
            if len(cols) >= LIMITS['columns']:
                raise PltError('too_many_cols', n, 'more than 512 columns.')
            name, role, typ, of, scale, relative = g(0), g(1), g(2), g(5), g(6), g(7)
            if not NAME_RE.match(name):
                raise PltError('bad_name', n, f'"{_short(name)}" is not a column name (a letter, then letters, '
                               'digits, _ . -; at most 64).')
            if any(c['name'] == name for c in cols):
                raise PltError('duplicate_name', n, f'column {name} is declared twice.')
            if role in ROLES:
                r = role
            else:
                if not UNKNOWN_ROLE_RE.match(role):
                    raise PltError('bad_role', n, f'"{_short(role)}" is not a role.')
                notices.append(f'Column {name} has a role this version does not know ({role}); it is read as text.')
                r = 'text'
            if typ in ('number', 'text'):
                t = typ
            else:
                if not UNKNOWN_TYPE_RE.match(typ):
                    raise PltError('bad_type', n, f'"{_short(typ)}" is not a type.')
                notices.append(f'Column {name} has a type this version does not know ({typ}); it is read as text.')
                t = 'text'
                if role in ROLES and role != 'text':
                    notices.append(f'Column {name} is read as plain text.')
                    r = 'text'
            if r == 'text':
                t = 'text'
            if r in NUMBER_ROLES and t != 'number':
                raise PltError('bad_type', n, f'column {name} ({r}) must be a number column.')
            if r in TEXT_ROLES and t != 'text':
                raise PltError('bad_type', n, f'column {name} ({r}) must be a text column.')
            unit = _fold_micro(_trim(f(3)))
            if r in UNIT_ROLES and not unit:
                raise PltError('missing_unit', n, f'column {name} needs a unit ("1" if it has none).')
            sc = 'linear'
            if r == 'measure' and scale:
                if scale in SCALES:
                    sc = scale
                else:
                    notices.append(f'Column {name} has a scale this version does not know ({_short(scale)}); '
                                   'it is read as linear.')
            if relative not in ('', 'yes'):
                raise bad('relative must be "yes" or empty.')
            cols.append({'name': name, 'role': r, 'type': t, 'unit': unit, 'label': _trim(f(4)), 'of': of,
                         'scale': sc, 'relative': r == 'measure' and relative == 'yes'})
        else:
            if key not in extra:
                notices.append(f'The table holds details this version does not show ({key}).')
                extra[key] = []
            extra[key].append([_cell_text(x, n) for x in fields])
        i += 1

    for req in ('#source', '#created', '#assay', '#rows', '#replicates', '#design'):
        if req not in seen:
            raise PltError('missing_key', 0, f'the table has no {req} line.')
    if not cols:
        raise PltError('missing_key', 0, 'the table has no #col lines.')
    if i >= len(lines):
        raise PltError('no_header', 0, 'the table has no header line.')

    header_line = i + 1
    header = split_fields(physical(i), header_line)
    if len(header) != len(cols) or any(h != c['name'] for h, c in zip(header, cols, strict=True)):
        raise PltError('header_mismatch', header_line, 'the header does not match the #col lines (names and order).')
    i += 1

    def pick(key: str, options, fallback: str, what: str) -> str:
        v = _trim(seen[key][0] if seen[key] else '')
        if v in options:
            return v
        notices.append(f'Unknown {what} "{_short(v)}"; read as {fallback}.')
        return fallback

    def text_of(key: str) -> str:
        v = seen.get(key)
        return _trim(_clean(_unescape(v[0] if v else ''))) if v is not None else ''

    src = seen['#source']
    table = {
        'source': {'toolbox': src[0], 'version': src[1], 'analysis': src[2]},
        'created': seen['#created'][0],
        'title': text_of('#title'),
        'assay': pick('#assay', ASSAYS, 'generic', 'assay'),
        'rows': text_of('#rows'),
        'replicates': pick('#replicates', REPLICATES, 'unknown', 'replicates value'),
        'design': pick('#design', DESIGNS, 'unknown', 'design'),
        'control': None,
        'analytes': analytes,
        'inputs': inputs,
        'settings': settings,
        'methods': methods,
        'columns': cols,
        'data': [],
    }
    if not table['rows']:
        raise PltError('bad_field', 0, '#rows: needs a value.')
    if '#control' in seen:
        c = seen['#control']
        table['control'] = {'column': c[0], 'level': _trim(_clean(_unescape(c[1] if len(c) > 1 else '')))}

    while i < len(lines):
        n = i + 1
        if len(table['data']) >= LIMITS['rows']:
            raise PltError('too_many_rows', n, 'more than 200,000 rows.')
        raw = physical(i)
        if raw.startswith('#'):
            raise PltError('bad_row', n, 'a # line after the header.')
        cells = split_fields(raw, n)
        if len(cells) != len(cols):
            raise PltError('row_length', n, f'the row has {len(cells)} cells; the header has {len(cols)}.')
        row = []
        for c, col in zip(cells, cols, strict=True):
            if col['type'] == 'number':
                if utf8_length(c) > LIMITS['cellBytes']:
                    raise PltError('cell_too_long', n, 'a cell is longer than 4 KB.')
                row.append(None if c == '' else _parse_number(c, n))
            else:
                v = _cell_text(c, n)
                row.append(_trim(v) if col['role'] in TRIMMED_ROLES else v)
        table['data'].append(row)
        i += 1

    _validate(table, 'read')
    return table, notices, extra


# ---------------------------------------------------------------- shared checks

def _validate(t: dict, mode: str) -> None:
    cols = t['columns']
    names = {c['name']: c for c in cols}
    counts: dict[str, int] = {}
    for c in cols:
        counts[c['role']] = counts.get(c['role'], 0) + 1
    for r in SINGLE_ROLES:
        if counts.get(r, 0) > 1:
            raise PltError('duplicate_role', 0, f'more than one {r} column.')
    if counts.get('group') and counts.get('factor'):
        raise PltError('group_and_factor', 0, 'a table has either one group column or factor columns, not both.')
    for c in cols:
        if c['role'] in ('flag', 'limit'):
            if c['role'] == 'limit' and not c['of']:
                raise PltError('bad_of', 0, f'limit column {c["name"]} must name its measure.')
            if c['of'] and (names.get(c['of']) or {}).get('role') != 'measure':
                raise PltError('bad_of', 0, f'column {c["name"]} belongs to "{_short(c["of"])}", which is not a '
                               'measure column.')
    ctl = t['control']
    if ctl:
        c = names.get(ctl['column'])
        if not c or c['role'] not in ('group', 'factor'):
            raise PltError('bad_control', 0, f'#control names "{_short(ctl["column"])}", which is not a group or '
                           'factor column.')
        k = cols.index(c)
        if not any(row[k] == ctl['level'] for row in t['data']):
            raise PltError('bad_control', 0, f'#control level "{_short(ctl["level"])}" is not in column {c["name"]}.')
    for k, c in enumerate(cols):
        if c['role'] == 'flag':
            for row in t['data']:
                if row[k] == '':
                    continue
                for flag in row[k].split(';'):
                    if not FLAG_RE.match(flag):
                        raise PltError('bad_flag', 0, f'"{_short(flag)}" in column {c["name"]} is not a flag '
                                       '(lowercase letters, digits, _).')
        if c['role'] == 'well_type':
            for row in t['data']:
                if row[k] != '' and row[k] not in WELL_TYPES:
                    raise PltError('bad_well_type', 0, f'"{_short(row[k])}" in column {c["name"]} is not standard, '
                                   'unknown, blank or control.')
    if mode == 'write':
        for name, value in t['settings']:
            if looks_like_path(name) or looks_like_path(value):
                raise PltError('path_in_setting', 0, f'setting "{_short(name)}" holds a file path; give the setting '
                               'without it.')
        for inp in t['inputs']:
            if looks_like_path(inp['label']):
                raise PltError('file_name_label', 0, f'input label "{_short(inp["label"])}" is a file name; use the '
                               f'user\'s own label or "Input {inp["index"]}".')


# ---------------------------------------------------------------- writing

def _out_text(s: str, field: str) -> str:
    v = _clean(s)
    if CONTROL.search(v):
        raise PltError('control_char', 0, f'{field} holds a control character or line break.')
    return _escape(v)


def _normalise(t: dict) -> dict:
    cols = t['columns']

    def trimmed(s: str) -> str:
        return _trim(_clean(s))

    def cell(v, k):
        if not isinstance(v, str):
            return v
        role = cols[k]['role'] if k < len(cols) else ''
        return trimmed(v) if role in TRIMMED_ROLES else _clean(v)

    return {
        **t,
        'title': trimmed(t.get('title', '')),
        'rows': trimmed(t['rows']),
        'control': {'column': t['control']['column'], 'level': trimmed(t['control']['level'])}
        if t.get('control') else None,
        'analytes': [{**a, 'value': trimmed(a['value'])} for a in t.get('analytes', [])],
        'inputs': [{**x, 'label': trimmed(x['label'])} for x in t.get('inputs', [])],
        'settings': [[trimmed(a), _clean(b)] for a, b in t.get('settings', [])],
        'methods': [_clean(m) for m in t.get('methods', [])],
        'columns': [_column(c, trimmed) for c in cols],
        'data': [[cell(v, k) for k, v in enumerate(row)] for row in t['data']],
    }


def _column(c: dict, trimmed) -> dict:
    out = {'of': '', 'scale': 'linear', 'relative': False}
    out.update(c)
    out['unit'] = _fold_micro(trimmed(c.get('unit', '')))
    out['label'] = trimmed(c.get('label', ''))
    return out


def write_plt(table: dict, bom: bool = False, hashes: bool = True) -> str:
    """The table as PLT text (CRLF line ends); PltError when it breaks a rule."""
    t = _normalise(table)
    _validate(t, 'write')
    out: list[str] = []

    def rec(key: str, fields: list[str]) -> None:
        while fields and fields[-1] == '':
            fields.pop()
        out.append(f'{key},{join_fields(fields)}' if fields else key)

    rec('#plt', [f'{PLT_MAJOR}.{PLT_MINOR}'])
    s = t['source']
    if not (ID_RE.match(s['toolbox']) and ID_RE.match(s['analysis']) and SOURCE_VERSION_RE.match(s['version'])):
        raise PltError('bad_field', 0, '#source: toolbox and analysis ids are lowercase letters, digits and -; '
                       'version is required.')
    rec('#source', [s['toolbox'], s['version'], s['analysis']])
    if not CREATED_RE.match(t['created']):
        raise PltError('bad_field', 0, '#created: time must be YYYY-MM-DDTHH:MMZ.')
    rec('#created', [t['created']])
    if t['title']:
        rec('#title', [_out_text(t['title'], 'title')])
    if t['assay'] not in ASSAYS:
        raise PltError('bad_field', 0, '#assay: unknown assay.')
    rec('#assay', [t['assay']])
    if not t['rows']:
        raise PltError('bad_field', 0, '#rows: needs a value.')
    rec('#rows', [_out_text(t['rows'], 'rows')])
    if t['replicates'] not in REPLICATES:
        raise PltError('bad_field', 0, '#replicates: unknown value.')
    rec('#replicates', [t['replicates']])
    if t['design'] not in DESIGNS:
        raise PltError('bad_field', 0, '#design: unknown value.')
    rec('#design', [t['design']])
    if t['control']:
        rec('#control', [t['control']['column'], _out_text(t['control']['level'], 'control level')])
    seen_analytes: set[str] = set()
    for a in t['analytes']:
        if not a['value'] or a['value'] in seen_analytes:
            raise PltError('bad_field', 0, '#analyte: values must be given and different.')
        seen_analytes.add(a['value'])
        amp = a.get('amplification')
        if amp is not None and not amp > 1:
            raise PltError('bad_field', 0, '#analyte: amplification factor must be above 1.')
        rec('#analyte', [_out_text(a['value'], 'analyte'), a['kind'], '' if amp is None else format_number(amp)])
    seen_idx: set[int] = set()
    for inp in t['inputs']:
        idx = inp['index']
        if isinstance(idx, bool) or not isinstance(idx, int) or not 1 <= idx <= 999999 or idx in seen_idx:
            raise PltError('bad_field', 0, '#input: indexes are different whole numbers from 1.')
        seen_idx.add(idx)
        if not inp['label']:
            raise PltError('bad_field', 0, '#input: needs a label.')
        sha = inp.get('sha256')
        if sha is not None and not SHA_RE.match(sha):
            raise PltError('bad_field', 0, '#input: SHA-256 must be 64 lowercase hex digits.')
        rec('#input', [str(idx), _out_text(inp['label'], 'input label'), sha if hashes and sha else ''])
    for name, value in t['settings']:
        if not name:
            raise PltError('bad_field', 0, '#setting: needs a name.')
        rec('#setting', [_out_text(name, 'setting name'), _out_text(value, 'setting value')])
    for m in t['methods']:
        rec('#methods', [_out_text(m, 'methods')])
    cols = t['columns']
    if not cols:
        raise PltError('missing_key', 0, 'the table has no columns.')
    if len(cols) > LIMITS['columns']:
        raise PltError('too_many_cols', 0, 'more than 512 columns.')
    names: set[str] = set()
    for c in cols:
        if not NAME_RE.match(c['name']):
            raise PltError('bad_name', 0, f'"{_short(c["name"])}" is not a column name.')
        if c['name'] in names:
            raise PltError('duplicate_name', 0, f'column {c["name"]} appears twice.')
        names.add(c['name'])
        r = c['role']
        if r not in ROLES:
            raise PltError('bad_role', 0, f'"{_short(r)}" is not a role.')
        if r in NUMBER_ROLES and c['type'] != 'number':
            raise PltError('bad_type', 0, f'column {c["name"]} ({r}) must be a number column.')
        if (r in TEXT_ROLES or r == 'text') and c['type'] != 'text':
            raise PltError('bad_type', 0, f'column {c["name"]} ({r}) must be a text column.')
        if r in UNIT_ROLES and not c['unit']:
            raise PltError('missing_unit', 0, f'column {c["name"]} needs a unit ("1" if it has none).')
        if r == 'measure' and c['scale'] not in SCALES:
            raise PltError('bad_field', 0, f'column {c["name"]}: unknown scale.')
        rec('#col', [c['name'], r, c['type'], _out_text(c['unit'], 'unit'), _out_text(c['label'], 'label'), c['of'],
                     c['scale'] if r == 'measure' and c['scale'] != 'linear' else '',
                     'yes' if r == 'measure' and c['relative'] else ''])
    out.append(join_fields([c['name'] for c in cols]))
    if len(t['data']) > LIMITS['rows']:
        raise PltError('too_many_rows', 0, 'more than 200,000 rows.')
    for row in t['data']:
        if len(row) != len(cols):
            raise PltError('row_length', 0, f'a row has {len(row)} cells; the table has {len(cols)} columns.')
        cells = []
        for v, c in zip(row, cols, strict=True):
            if c['type'] == 'number':
                cells.append('' if v is None else format_number(v))
            else:
                if not isinstance(v, str):
                    raise PltError('bad_field', 0, f'column {c["name"]} holds a non-text value.')
                s2 = _out_text(v, f'column {c["name"]}')
                if utf8_length(s2) > LIMITS['cellBytes']:
                    raise PltError('cell_too_long', 0, 'a cell is longer than 4 KB.')
                cells.append(s2)
        line = join_fields(cells)
        if (len(cells) == 1 and cells[0] == '') or line.startswith('#'):
            first = join_fields([cells[0]])
            line = '"' + cells[0].replace('"', '""') + '"' + line[len(first):]
        out.append(line)
    for line in out:
        if utf8_length(line) > LIMITS['lineBytes']:
            raise PltError('line_too_long', 0, 'a line would be longer than 64 KB.')
    body = '\r\n'.join(out) + '\r\n'
    if utf8_length(body) > LIMITS['fileBytes']:
        raise PltError('too_large', 0, 'the table would be larger than 16 MB.')
    return ('\ufeff' if bom else '') + body


def share_file_name(title: str) -> str:
    """A file name for sharing: letters, digits, space, - and _, at most 60 code points, then .plt.csv."""
    kept = ''.join(ch if ch in ' _-' or unicodedata.category(ch)[0] in 'LN' else ' ' for ch in _clean(title))
    kept = _trim(re.sub(' +', ' ', kept))
    base = _trim(kept[:60])
    return f'{base or "results"}.plt.csv'
