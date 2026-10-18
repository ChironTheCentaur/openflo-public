"""Saving tables so Excel opens them correctly in any locale.

Two things go wrong when a flow user opens OpenFlo's tables in Excel:

* In most of continental Europe, Latin America and elsewhere, the system's
  list separator is ``;`` and the decimal mark is ``,``. Excel reads a CSV
  with those, so a comma-separated file with decimal points opens as one
  column of text. ``csv_format('system')`` asks the operating system for
  its separator and decimal mark and writes the CSV that way.
* A table with several statistics is easier to use as a workbook, one sheet
  per statistic, than as one wide CSV. ``write_workbook`` writes an .xlsx
  through openpyxl (already a dependency, for panel sheets).

The default stays the plain comma-and-point CSV every analysis tool reads;
the system style is a preference (``csv_style``).
"""
from __future__ import annotations

import csv
import math
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .prefs import read_prefs

# Preference values for CSV output, with the label the Preferences dialog
# shows for each.
CSV_STYLES = (
    ('standard', 'Comma and decimal point (any analysis tool)'),
    ('system', "This computer's separator and decimal mark (Excel)"),
)
DEFAULT_CSV_STYLE = 'standard'

# Missing values read NA, as everywhere else in OpenFlo: a population that
# cannot be computed is not a blank that reads as forgotten.
NA = 'NA'


def _windows_number_format() -> tuple[str, str] | None:
    """(list separator, decimal mark) from the Windows regional settings,
    which is what Excel uses; None if they cannot be read."""
    if sys.platform != 'win32':
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r'Control Panel\International') as key:
            sep = str(winreg.QueryValueEx(key, 'sList')[0])
            dec = str(winreg.QueryValueEx(key, 'sDecimal')[0])
    except Exception:                                    # noqa: BLE001
        return None
    if len(sep) != 1 or len(dec) != 1 or sep == dec:
        return None
    return sep, dec


def _locale_decimal_mark() -> str:
    """The decimal mark of the user's locale (macOS and Linux)."""
    import locale
    try:
        prev = locale.setlocale(locale.LC_NUMERIC)
    except locale.Error:
        return '.'
    try:
        locale.setlocale(locale.LC_NUMERIC, '')
        return str(locale.localeconv().get('decimal_point') or '.')
    except locale.Error:
        return '.'
    finally:
        try:
            locale.setlocale(locale.LC_NUMERIC, prev)
        except locale.Error:
            pass


def system_csv_format() -> tuple[str, str]:
    """(separator, decimal mark) Excel expects on this computer.

    Windows reports both. Elsewhere only the decimal mark is known, and
    Excel's rule is used: where the decimal mark is a comma, the list
    separator is a semicolon."""
    win = _windows_number_format()
    if win is not None:
        return win
    dec = _locale_decimal_mark()
    return (';', ',') if dec == ',' else (',', '.')


def csv_style() -> str:
    """The CSV style preference ('standard' or 'system')."""
    style = str(read_prefs().get('csv_style', DEFAULT_CSV_STYLE))
    return style if style in dict(CSV_STYLES) else DEFAULT_CSV_STYLE


def csv_format(style: str | None = None) -> tuple[str, str]:
    """(separator, decimal mark) for a CSV style; the preference if None."""
    style = csv_style() if style is None else style
    if style == 'system':
        return system_csv_format()
    return ',', '.'


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return isinstance(value, float) and math.isnan(value)
    except TypeError:
        return False


def format_cell(value: Any, decimal: str = '.', na_rep: str = NA) -> str:
    """One cell as text, with the given decimal mark on real numbers.

    Numbers keep full precision (``repr``), so nothing is rounded away on
    the way to Excel; integers and text are written unchanged."""
    try:
        import numpy as np
        if isinstance(value, np.generic):
            value = value.item()
    except ImportError:                                  # pragma: no cover
        pass
    if _is_missing(value):
        return na_rep
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        text = repr(value)
        return text.replace('.', decimal) if decimal != '.' else text
    return str(value)


def write_rows_csv(path: str, header: Sequence[Any],
                   rows: Iterable[Sequence[Any]], *,
                   style: str | None = None, na_rep: str = NA) -> None:
    """Write a header and rows as CSV in the given style.

    The system style is written with a UTF-8 byte-order mark: without it
    Excel reads the file in the legacy code page and mangles any character
    outside ASCII (a µ in a channel label, an accented sample name)."""
    sep, dec = csv_format(style)
    encoding = 'utf-8-sig' if (style or csv_style()) == 'system' else 'utf-8'
    with open(path, 'w', newline='', encoding=encoding) as f:
        w = csv.writer(f, delimiter=sep)
        w.writerow([format_cell(h, dec, na_rep) for h in header])
        for r in rows:
            w.writerow([format_cell(v, dec, na_rep) for v in r])


def pandas_csv_format(style: str | None = None) -> tuple[str, str, str]:
    """(sep, decimal, encoding) for ``DataFrame.to_csv`` in a CSV style, for
    frames whose layout pandas writes best itself (a two-level heading)."""
    sep, dec = csv_format(style)
    system = (style or csv_style()) == 'system'
    return sep, dec, 'utf-8-sig' if system else 'utf-8'


def write_frame_csv(path: str, df, *, style: str | None = None,
                    na_rep: str = NA, index: bool = False) -> None:
    """Write a pandas DataFrame as CSV in the given style."""
    if index:
        # As pandas writes it: the index name heads its column, or nothing.
        header = [df.index.name or ''] + list(df.columns)
        rows: Iterable[Sequence[Any]] = df.itertuples(index=True, name=None)
    else:
        header = list(df.columns)
        rows = df.itertuples(index=False, name=None)
    write_rows_csv(path, header, rows, style=style, na_rep=na_rep)


_BAD_SHEET_CHARS = re.compile(r'[\[\]:*?/\\]')


def sheet_name(name: str, taken: Iterable[str] = ()) -> str:
    """A valid, unique Excel sheet name: at most 31 characters, none of
    ``[]:*?/\\``, not blank, and not already used (case-insensitively)."""
    base = _BAD_SHEET_CHARS.sub('_', str(name)).strip().strip("'") or 'Sheet'
    base = base[:31]
    used = {t.lower() for t in taken}
    out, n = base, 2
    while out.lower() in used:
        tail = f" ({n})"
        out = base[:31 - len(tail)] + tail
        n += 1
    return out


def write_workbook(path: str, sheets: Mapping[str, Any], *,
                   na_rep: str = NA,
                   about: Mapping[str, Any] | None = None) -> list[str]:
    """Write DataFrames to one .xlsx, a sheet each, in the order given.

    Numbers are stored as numbers (Excel shows them in the reader's own
    locale, so a workbook needs no separator choice). Missing values read
    NA. ``about`` adds a last sheet of key / value rows (OpenFlo version,
    when it was saved), so a workbook can be traced back to what made it.
    Returns the sheet names used."""
    import pandas as pd
    names: list[str] = []
    with pd.ExcelWriter(path, engine='openpyxl') as xw:
        for title, df in sheets.items():
            nm = sheet_name(title, names)
            names.append(nm)
            df.to_excel(xw, sheet_name=nm, index=False, na_rep=na_rep)
            ws = xw.sheets[nm]
            ws.freeze_panes = 'A2'
            for i, col in enumerate(df.columns, start=1):
                width = max([len(str(col))]
                            + [len(format_cell(v)) for v in df[col].head(200)])
                ws.column_dimensions[_col_letter(i)].width = min(
                    60, max(8, width + 2))
        if about:
            nm = sheet_name('About', names)
            names.append(nm)
            pd.DataFrame({'Item': list(about.keys()),
                          'Value': [str(v) for v in about.values()]}
                         ).to_excel(xw, sheet_name=nm, index=False)
    return names


def _col_letter(i: int) -> str:
    out = ''
    while i:
        i, r = divmod(i - 1, 26)
        out = chr(65 + r) + out
    return out


def statistics_sheets(header: Sequence[str], rows: Sequence[Sequence[Any]],
                      id_columns: int = 2) -> dict[str, Any]:
    """Sheets for a statistics table: everything on one sheet, then one
    sheet per statistic with the populations down the side and the samples
    across, the shape most people paste into a figure or a stats package.

    ``header`` and ``rows`` are the table as exported; the first
    ``id_columns`` columns identify a row (sample, then population, then
    anything else that is not a statistic). A blank cell stays empty."""
    import pandas as pd
    clean = [[None if v == '' else v for v in r] for r in rows]
    df = pd.DataFrame(clean, columns=list(header))
    sheets: dict[str, Any] = {'All statistics': df}
    if id_columns < 2 or len(header) <= id_columns or df.empty:
        return sheets
    sample_col, pop_col = header[0], header[1]
    # Populations and samples in table order, not alphabetical.
    pops = list(dict.fromkeys(df[pop_col].tolist()))
    samples = list(dict.fromkeys(df[sample_col].tolist()))
    for stat in header[id_columns:]:
        wide = pd.DataFrame(
            df.groupby([pop_col, sample_col], sort=False, dropna=False)
            [stat].first().unstack(sample_col))
        wide = wide.reindex(index=pops, columns=samples)
        wide.columns.name = None
        sheets[str(stat)] = wide.reset_index()
    return sheets


def about_rows() -> dict[str, str]:
    """Provenance rows for a workbook's About sheet."""
    from datetime import datetime

    from . import __version__
    return {'Saved by': f"OpenFlo {__version__}",
            'Saved at': datetime.now().isoformat(timespec='seconds')}


def is_excel_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in ('.xlsx', '.xlsm')


# Excel's hard limit; a longer table cannot go in a sheet.
EXCEL_MAX_ROWS = 1_048_576


def csvs_to_workbook(out_dir: str, path: str, *,
                     since: float | None = None,
                     max_bytes: int = 50 * 1024 * 1024
                     ) -> tuple[list[str], list[str]]:
    """Collect the CSV tables under ``out_dir`` into one workbook, a sheet
    each, named after the file (with its folder when two share a name).

    ``since`` (a ``time.time()`` value) keeps only files written since then,
    so a re-run does not sweep in old tables. Per-event exports are too big
    for a sheet and are left out by size (``max_bytes``) or by Excel's row
    limit. Returns (files included, files left out with the reason)."""
    import pandas as pd
    found: list[str] = []
    for root, _dirs, files in os.walk(out_dir):
        for fn in sorted(files):
            if fn.lower().endswith('.csv'):
                full = os.path.join(root, fn)
                if since is None or os.path.getmtime(full) >= since:
                    found.append(full)
    found.sort()
    stems: dict[str, int] = {}
    for f in found:
        stem = os.path.splitext(os.path.basename(f))[0]
        stems[stem] = stems.get(stem, 0) + 1
    sheets: dict[str, Any] = {}
    included: list[str] = []
    skipped: list[str] = []
    for f in found:
        rel = os.path.relpath(f, out_dir)
        if os.path.getsize(f) > max_bytes:
            skipped.append(f"{rel}: too large for a sheet")
            continue
        try:
            df = pd.read_csv(f)
        except Exception as exc:                         # noqa: BLE001
            skipped.append(f"{rel}: could not be read ({exc})")
            continue
        if len(df) + 1 > EXCEL_MAX_ROWS:
            skipped.append(f"{rel}: more rows than Excel allows")
            continue
        stem = os.path.splitext(os.path.basename(f))[0]
        title = stem
        if stems[stem] > 1:
            parent = os.path.basename(os.path.dirname(f))
            title = f"{parent} {stem}" if parent else stem
        sheets[title] = df
        included.append(rel)
    if sheets:
        write_workbook(path, sheets, about=about_rows())
    return included, skipped
