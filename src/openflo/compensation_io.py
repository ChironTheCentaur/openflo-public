"""Compensation (spillover) matrix files: ``read_compensation_matrix`` and
``write_compensation_matrix`` for .wsp, .csv / .tsv and .fcs, plus the SPILL /
$SPILLOVER keyword parser FlowSample shares with them.

Core module: numpy and the standard library. No Tk, and it does not import
``openflo.pipeline`` (pipeline imports it).
"""

import csv
import logging
import os

import numpy as np

from . import fcs_read as _fcs_read
from .exceptions import CompensationError, FcsParseError
from .fcs_quirks import read_fcs_tolerant
from .wsp_reader import WspReader
from .wsp_writer import WspWriter

# Name kept as 'openflo.pipeline' so existing log filters and caplog tests see
# these messages.
log = logging.getLogger('openflo.pipeline')


# ── Compensation matrix IO ────────────────────────────────────────────────────
#
# Polymorphic read / write so the GUI's matrix editor and the pipeline can
# treat all of these as a single concept:
#   .wsp   FlowJo workspace (any number of matrices; we return the first)
#   .csv / .tsv  Header row = destination channels, first column of each
#                data row = source channel (rows are put in header order by
#                that label). Also accepted: a header row with no corner cell
#                over unlabelled rows (FlowJo's export), and header-less
#                (all cells numeric; `channels` is then None and the caller
#                supplies the names).
#   .fcs   Reads the SPILL (BD FACSDiva) / $SPILLOVER (FCS 3.1) keyword.
#
# Every reader returns spillover as FRACTIONS (diagonal 1). A matrix written in
# percent (diagonal 100, as some exports are) is converted with a warning:
# taken as fractions it divides every compensated channel by ~100. Measured: a
# percent CSV leaves the compensated data at 0.01x its true value, in silence.

def is_number(cell):
    """True when ``cell`` (any value, stripped as text) parses as a float."""
    try:
        float(str(cell).strip())
    except ValueError:
        return False
    return True


# How far a diagonal may sit from 1 (fractions) or 100 (percent) and still be
# read as that unit. Acquisition software writes exactly 1 or 100; this only
# absorbs rounding in a hand-edited or re-exported file.
_SPILL_DIAG_TOL = 0.05


def non_finite_cells(matrix, channels):
    """The cells of a spillover matrix that are not finite numbers, named by
    channel ("FL1-A -> FL2-A = nan"), or '' when every cell is finite."""
    matrix = np.asarray(matrix, dtype=float)
    bad = np.argwhere(~np.isfinite(matrix))
    return ', '.join(f"{channels[i]} -> {channels[j]} = {matrix[i, j]}"
                     for i, j in bad[:3]) + (' ...' if len(bad) > 3 else '')


def spill_units_to_fractions(matrix, source='spillover matrix'):
    """`matrix` as FRACTIONS (diagonal ~1).

    A diagonal of ~100 in every row is a percent matrix and is divided by 100,
    with a warning. Any other diagonal (neither ~1 nor ~100) is returned as
    given but warned about loudly: compensation divides each channel by its
    diagonal entry, which is almost never what the file meant."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] \
            or not matrix.size:
        return matrix
    d = np.diag(matrix)
    if not np.all(np.isfinite(d)):
        return matrix
    if np.all(np.abs(d - 100.0) <= 100.0 * _SPILL_DIAG_TOL):
        log.warning(
            "  [!] %s is in PERCENT (diagonal %s) -- converted to fractions "
            "(divided by 100). Applied as fractions it would have scaled "
            "every compensated channel by ~0.01.",
            source, np.array2string(d, precision=4, separator=', '))
        return matrix / 100.0
    if not np.all(np.abs(d - 1.0) <= _SPILL_DIAG_TOL):
        log.warning(
            "  [!] %s has diagonal %s -- neither 1 (fractions) nor 100 "
            "(percent). It is used AS GIVEN, which divides each compensated "
            "channel by its diagonal entry: check the file's units before "
            "trusting the compensated values.",
            source, np.array2string(d, precision=4, separator=', '))
    return matrix


def _spill_channel_refs(chans, param_names, source, markers=None):
    """The channels a SPILL / $SPILLOVER keyword names, as $PnN names.

    FCS 3.1 says the keyword lists $PnN names, but some acquisition software
    writes the 1-based parameter NUMBERS instead (``3,3,4,5,...``). Those
    matched no column, so the matrix was skipped with only "No matching
    channels for compensation" in the log and the sample stayed
    uncompensated (measured: 2-21% median error on a 3-colour file). When
    every reference is an integer in 1..$PAR and none of them is itself a
    channel name, they are parameter numbers and are mapped to $PnN.

    Others write the MARKER names ($PnS) instead. When no reference is a
    $PnN name and each names exactly one parameter's $PnS (`markers`, in
    parameter order), they are mapped to those parameters' $PnN."""
    chans = [str(c).strip() for c in chans]
    names = [str(c) for c in (param_names or [])]
    if not chans or not names or any(c in names for c in chans):
        return chans
    marks = [str(m).strip() for m in (markers or [])]
    if marks and not all(c.isdigit() for c in chans):
        hits = []
        for c in chans:
            at = [k for k, m in enumerate(marks) if m and m == c]
            if len(at) != 1 or at[0] >= len(names):
                hits = []
                break
            hits.append(names[at[0]])
        if hits and len(set(hits)) == len(hits):
            log.info("  %s: SPILL names its channels by marker %s -> %s.",
                     source, chans, hits)
            return hits
    idx = []
    for c in chans:
        if not c.isdigit() or not 1 <= int(c) <= len(names):
            return chans
        idx.append(int(c))
    if len(set(idx)) != len(idx):
        return chans
    mapped = [names[k - 1] for k in idx]
    log.info("  %s: SPILL names its channels by parameter number %s -> %s.",
             source, chans, mapped)
    return mapped


def parse_spill_keyword(spill, param_names, source, markers=None):
    """``N,ch1,...,chN,v11,...,vNN`` as ``(channels, matrix)``: channels as
    $PnN names (see `_spill_channel_refs`), matrix in fractions (see
    `spill_units_to_fractions`). `param_names` are the file's $PnN names in
    parameter order. Raises CompensationError on a malformed value,
    including a cell that is not a finite number (NaN/inf would make every
    compensated value unknown, and would be written into a .wsp as is).

    Shared by FlowSample._parse_spillover and read_compensation_matrix, so
    auto_compensate, the compensation editor and the workspace export can
    never disagree about what a file carries."""
    parts = [x.strip() for x in str(spill).split(',')]
    try:
        n = int(parts[0])
    except ValueError as e:
        raise CompensationError(
            f"{source}: SPILL keyword header is not an integer: "
            f"{parts[0]!r}") from e
    if n < 1 or len(parts) < 1 + n + n * n:
        raise CompensationError(
            f"{source}: SPILL keyword promises {n} channels + "
            f"{n*n} values but only {len(parts)-1} entries provided")
    extra = [x for x in parts[1 + n + n * n:] if x]
    if extra:
        log.warning("  [!] %s: SPILL keyword has %d value(s) past its %dx%d "
                    "matrix -- ignored (%s)", source, len(extra), n, n,
                    ', '.join(extra[:4]) + (' ...' if len(extra) > 4 else ''))
    channels = _spill_channel_refs(parts[1:1 + n], param_names, source,
                                   markers)
    try:
        vals = [float(x) for x in parts[1 + n: 1 + n + n * n]]
    except ValueError as e:
        raise CompensationError(
            f"{source}: non-numeric value in SPILL matrix") from e
    matrix = np.asarray(vals, dtype=float).reshape(n, n)
    bad = non_finite_cells(matrix, channels)
    if bad:
        raise CompensationError(
            f"{source}: a SPILL cell is not a finite number ({bad})")
    return channels, spill_units_to_fractions(matrix, f"{source} SPILL")


def spill_text(metadata):
    """The SPILL / $SPILLOVER value in an FCS TEXT dict, or None. FlowIO strips
    every '$' and lower-cases every keyword; other sources may not, so both
    spellings are normalised. SPILL wins when a file carries both, as FlowJo
    10.10.2 does (tests/test_compensation_fcs_spill.py)."""
    text = {str(k).lstrip('$').lower(): v for k, v in (metadata or {}).items()}
    return text.get('spill') or text.get('spillover') or None


def fcs_param_markers(metadata):
    """$PnS marker names in parameter order ('' where none)."""
    text = {str(k).lstrip('$').lower(): v for k, v in (metadata or {}).items()}
    try:
        npar = int(str(text.get('par', '')).strip())
    except ValueError:
        return []
    return [str(text.get(f'p{k}s') or '').strip() for k in range(1, npar + 1)]


def fcs_param_names(metadata, channel_names):
    """$PnN names in parameter order: from the TEXT keywords when present,
    else the loaded channel names (which FlowSample reads from $PnN)."""
    text = {str(k).lstrip('$').lower(): v for k, v in (metadata or {}).items()}
    names = list(channel_names or [])
    try:
        npar = int(str(text.get('par', '')).strip())
    except ValueError:
        npar = len(names)
    out = []
    for k in range(1, npar + 1):
        nm = text.get(f'p{k}n')
        if not nm and k <= len(names):
            nm = names[k - 1]
        out.append(str(nm).strip() if nm else f'Ch{k}')
    return out


def _csv_rows(path):
    """The non-blank rows of a matrix CSV/TSV, each cell stripped and a
    trailing separator dropped."""
    sep = '\t' if path.lower().endswith('.tsv') else ','
    with open(path, encoding='utf-8', newline='') as f:
        # csv.reader, not str.split: a channel name containing the
        # separator — a $PnS description like "CD4 FL3, clone SK3" is
        # ordinary — would split into two fields, and the read would fail with
        # "could not convert string to float: ' clone SK3'". Quoting is
        # what the format is for. Unquoted files written by earlier
        # versions parse identically.
        #
        # Don't strip the whole file: TSVs start with a leading tab to
        # align the header row above the row-label column, and that tab
        # is significant (an empty first cell ↔ "no row label here"). A
        # row of only empty cells is blank and skipped.
        rows = [r for r in csv.reader(f, delimiter=sep)
                if any(c.strip() for c in r)]
    return [_cells(r) for r in rows]


def _cells(r):
    r = [c.strip() for c in r]
    while r and r[-1] == '':          # a trailing separator is not a cell
        r.pop()
    return r


def _numbers(r, where, path):
    try:
        return [float(c) for c in r]
    except ValueError as e:
        raise CompensationError(
            f"{path}: non-numeric value in {where}: {e}") from e


def _headerless_matrix(rows, path):
    """The matrix of a numbers-only file, which must be square."""
    widths = sorted({len(r) for r in rows})
    if widths != [len(rows)]:
        raise CompensationError(
            f"matrix in {path} is not square ({len(rows)} rows of "
            f"{'/'.join(map(str, widths))} values)")
    return np.asarray([_numbers(r, f'row {i + 1}', path)
                       for i, r in enumerate(rows)], dtype=float)


def _csv_header(rows, path):
    """`(channels, labelled)` from the header row: whether the data rows
    carry a first column of row labels, and the channels the columns name."""
    header, body = rows[0], rows[1:]
    if header[0].startswith('#'):
        # numpy.savetxt(header=...) writes '# FL1-A,FL2-A,...': the '#' is
        # its comment mark, not part of the first channel's name. Kept, it
        # named a channel no sample has ('# FL1-A'), and that channel was
        # left out of compensation without a word.
        header = [header[0][1:].strip()] + header[1:]
    # A first column of row labels: the header's first cell is blank, or a
    # data row starts with a non-number. Exception: a blank corner over rows
    # exactly one cell narrower than the header, all numeric, carries none.
    labelled = header[0] == '' or any(r and not is_number(r[0])
                                      for r in body)
    if labelled and _unlabelled_under_blank_corner(header, body):
        labelled = False
        header = header[1:]
    channels = header[1:] if labelled else header
    if any(c == '' for c in channels):
        raise CompensationError(
            f"{path}: blank channel name in the header row {rows[0]}")
    return channels, labelled


def _unlabelled_under_blank_corner(header, body):
    """True for a blank corner over all-numeric rows one cell narrower than
    the header: the corner is padding, not a row-label column."""
    return (header[0] == '' and {len(r) for r in body} == {len(header) - 1}
            and all(is_number(r[0]) for r in body if r))


def _csv_body(body, n, labelled, path):
    """`(labels, matrix)` from the data rows; labels empty when unlabelled."""
    if len(body) != n:
        raise CompensationError(
            f"matrix in {path} is not square: {n} channel(s) in the header "
            f"but {len(body)} data row(s)")
    labels, mat = [], []
    for i, r in enumerate(body):
        vals = r[1:] if labelled else r
        if len(vals) != n:
            raise CompensationError(
                f"matrix in {path} is not square: row {i + 2} has "
                f"{len(vals)} value(s) for {n} channel(s)")
        if labelled:
            labels.append(r[0])
        mat.append(_numbers(vals, f'row {i + 2}', path))
    return labels, np.asarray(mat, dtype=float)


def _rows_in_header_order(matrix, labels, channels, path, source):
    """The matrix with its rows put in header order by their labels."""
    if not any(labels):
        return matrix
    if sorted(labels) != sorted(channels) or len(set(labels)) != len(channels):
        raise CompensationError(
            f"{path}: the row labels {labels} are not the header's "
            f"channels {channels}. Each row is the spillover FROM the "
            "channel it is labelled with; a matrix whose rows and "
            "columns name different channels cannot be applied.")
    if labels != channels:
        log.info("  %s: rows listed in the order %s -- reordered to the "
                 "header's %s.", source, labels, channels)
        matrix = matrix[[labels.index(c) for c in channels]]
    return matrix


def _read_matrix_csv(path):
    """`(channels, matrix)` from a CSV/TSV; channels None when header-less.

    Accepted layouts (row = source fluor, column = detector):
      * a header row with a corner cell (blank or a caption such as
        "Source/Detector") over one labelled row per source -- what
        write_compensation_matrix writes;
      * a header row of channel names only over rows of numbers only
        (FlowJo's export) -- the first number of each row must not be taken
        for its label, which fails as "not square (shape (3, 2))";
      * numbers only (header-less).
    Row labels are honoured: rows are put in header order by label. Ignoring
    them would load, from a file listing its rows in another order, the FL2
    row as the FL4 row without a word. Labels that are not exactly the
    header's channels are refused rather than guessed at."""
    rows = _csv_rows(path)
    if not rows:
        return None, None
    source = os.path.basename(path)
    if is_number(rows[0][0]):
        return None, spill_units_to_fractions(_headerless_matrix(rows, path),
                                              source)
    channels, labelled = _csv_header(rows, path)
    labels, matrix = _csv_body(rows[1:], len(channels), labelled, path)
    matrix = _rows_in_header_order(matrix, labels, channels, path, source)
    return channels, spill_units_to_fractions(matrix, source)


def read_compensation_matrix(path):
    """Returns `(channels, matrix)`. `channels` is a list[str] or None
    (only None for header-less CSVs); `matrix` is an NxN numpy array of
    FRACTIONS (a percent matrix is converted, with a warning), or
    ``(None, None)`` when the file is a legitimate FCS/WSP/CSV that simply
    has no spillover defined.

    Raises:
        CompensationError: malformed input (non-square matrix, channel /
            shape mismatch, row labels that are not the header's channels,
            unparseable SPILL keyword, unsupported extension).
    """
    p = path.lower()

    if p.endswith('.wsp'):
        reader = WspReader(path)
        # get_matrix() raises on a workspace without a matrix; this function
        # documents (None, None) for "a legitimate file with no spillover".
        if not reader.matrices:
            return None, None
        m = reader.get_matrix()
        return list(m['channels']), spill_units_to_fractions(
            m['matrix'], f"{os.path.basename(path)} '{m.get('name', '')}'")

    if p.endswith(('.csv', '.tsv')):
        return _read_matrix_csv(path)

    if p.endswith('.fcs'):
        # "N,ch1,ch2,...,chN,v11,v12,...,vNN" under the bare SPILL keyword
        # (BD FACSDiva) or $SPILLOVER (FCS 3.1). FlowIO strips every '$' and
        # lower-cases every keyword, so they arrive as 'spill'/'spillover';
        # the upper-case lookups that were here never matched, and every FCS
        # read as "no spillover". The same keyword choice and parser as
        # FlowSample._parse_spillover, so this agrees with auto_compensate.
        try:
            fcs = read_fcs_tolerant(path, _fcs_read.FlowData, [])
        except Exception as e:
            raise FcsParseError(f"could not read FCS {path}: {e}") from e
        spill = spill_text(fcs.text)
        if not spill:
            return None, None
        names = [fcs.channels[k].get('pnn') or fcs.channels[k].get('PnN')
                 or f'Ch{k}' for k in range(1, fcs.channel_count + 1)]
        return parse_spill_keyword(
            spill, fcs_param_names(fcs.text, names), path,
            markers=fcs_param_markers(fcs.text))

    raise CompensationError(
        f"unsupported compensation file format: {path} "
        "(expected .wsp / .csv / .tsv / .fcs)")


def write_compensation_matrix(path, matrix, channels):
    """Write `matrix` (NxN numpy) labelled by `channels` (list[str]) to
    `path`. Format dispatched on extension:
      .wsp        -> WspWriter (matrix only; no samples / gates)
      .csv / .tsv -> header row + row-labelled rows
      .fcs        -> not supported (FCS spillover lives inside an FCS,
                     no use-case for writing one as a standalone file)
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square, got {matrix.shape}")
    if len(channels) != matrix.shape[0]:
        raise ValueError(
            f"{len(channels)} channels vs {matrix.shape} matrix")
    p = path.lower()
    if p.endswith('.wsp'):
        w = WspWriter()
        w.set_compensation(list(channels), matrix)
        w.write(path)
        return
    if p.endswith(('.csv', '.tsv')):
        sep = '\t' if p.endswith('.tsv') else ','
        with open(path, 'w', encoding='utf-8', newline='') as f:
            # csv.writer quotes any channel name containing the separator, so
            # the file survives a name like "CD4 FL3, clone SK3". Joining
            # by hand produced a file this module could not read back.
            w = csv.writer(f, delimiter=sep, lineterminator='\n')
            w.writerow([''] + list(channels))
            for i, ch in enumerate(channels):
                w.writerow([ch] + [repr(float(matrix[i, j]))
                                   for j in range(matrix.shape[1])])
        return
    raise ValueError(f"unsupported compensation file format for writing: {path}")
