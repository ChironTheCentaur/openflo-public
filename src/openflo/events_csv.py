"""Reading an events table from a CSV a user picked: ``read_events_csv``
detects the delimiter and decimal commas, skips leading comment lines and
reads a header-less file's first row as data.

Core module: pandas (imported when called) and the standard library. No Tk, and it does not import
``openflo.pipeline``.
"""

import csv
import os
import re

from .compensation_io import is_number as _is_number


def read_events_csv(path):
    """``(frame, notes)``: an events table from a CSV a user picked, and
    what had to be worked out to read it (as sentences for the user).

    ``pd.read_csv(path)`` alone reads common exports wrong without a word:
    a file with no header row loses its first event to the column names; a
    semicolon-separated file with decimal commas (as European Excel writes)
    or a tab-separated one comes back as ONE column; quoted decimal commas
    make every column text; and '#' comment lines at the top raise a raw
    ParserError. So the delimiter is detected, leading comment lines are
    skipped, decimal commas are read as decimals, an all-number first row
    is taken as data, and a file with no numeric column is refused."""
    import pandas as pd
    with open(path, encoding='utf-8-sig', errors='replace') as fh:
        lines = fh.read(1 << 16).splitlines()
    skip = _leading_comment_lines(lines)
    body = [ln for ln in lines[skip:skip + 50] if ln.strip()]
    if not body:
        raise ValueError(f"{os.path.basename(path)} has no rows")
    sep, decimal = _sniff_format('\n'.join(body))
    notes = _format_notes(sep, decimal, skip)

    def _read(header: int | None = 0) -> pd.DataFrame:
        return pd.read_csv(path, sep=sep, decimal=decimal, skiprows=skip,
                           encoding='utf-8-sig', header=header)
    df = _read()
    if len(df.columns) and all(_is_number(c) for c in df.columns):
        df = _read(header=None)
        notes.append(_name_headerless_columns(df, lines, skip, sep))
    fixed = _read_decimal_comma_columns(df)
    if fixed:
        notes.append("decimal commas read as decimals in "
                     + ', '.join(fixed[:6]) + (' ...' if len(fixed) > 6 else ''))
    if not any(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns):
        raise ValueError(f"{os.path.basename(path)} has no numeric column to "
                         "read as events")
    return df, notes


def _leading_comment_lines(lines):
    """How many lines at the top are blank or '#' comments."""
    skip = 0
    while skip < len(lines) and (not lines[skip].strip()
                                 or lines[skip].lstrip().startswith('#')):
        skip += 1
    return skip


def _sniff_format(sample):
    """``(sep, decimal)`` detected from the first rows of data."""
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=',;\t|').delimiter
    except csv.Error:
        sep = ','
    decimal = '.'
    if (sep != ',' and re.search(r'\d,\d', sample)
            and not re.search(r'\d\.\d', sample)):
        decimal = ','
    return sep, decimal


def _format_notes(sep, decimal, skip):
    """The notes saying which separator and how many comment lines."""
    notes = []
    if sep != ',':
        notes.append({';': 'semicolon', '\t': 'tab', '|': 'bar'}[sep]
                     + '-separated' + (', with decimal commas'
                                       if decimal == ',' else ''))
    if skip:
        notes.append(f"{skip} comment line(s) at the top skipped")
    return notes


def _name_headerless_columns(df, lines, skip, sep):
    """Name a header-less frame's columns in place; returns the note.

    numpy.savetxt(header=...) puts the column names on a '#' line just
    above the numbers. Skipped as a comment, its names were lost."""
    last = lines[skip - 1].lstrip()[1:] if skip else ''
    names = [c.strip() for c in next(csv.reader([last], delimiter=sep), [])]
    if (len(names) == len(df.columns) and all(names)
            and not any(_is_number(c) for c in names)
            and len(set(names)) == len(names)):
        df.columns = names
        return "column names read from the '#' line above the numbers"
    df.columns = [f'Ch{i}' for i in range(1, len(df.columns) + 1)]
    return f"no header row: columns named Ch1 to Ch{len(df.columns)}"


def _read_decimal_comma_columns(df):
    """Convert text columns holding quoted decimal commas ("1,5" in a
    comma-separated file) to numbers, in place; returns their names."""
    import pandas as pd
    fixed = []
    for c in df.columns:
        col = df[c]
        # Text columns are object dtype, or str under pandas 3.
        if (pd.api.types.is_object_dtype(col)
                or pd.api.types.is_string_dtype(col)):
            v = col.dropna().astype(str).str.strip()
            if len(v) and v.str.fullmatch(r'-?\d+,\d+|-?\d+').all() \
                    and v.str.contains(',').any():
                df[c] = pd.to_numeric(col.astype(str).str.replace(
                    ',', '.', regex=False), errors='coerce')
                fixed.append(str(c))
    return fixed
