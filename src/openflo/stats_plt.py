"""Population statistics as a Pocket Lab Table, for "Send to Statistics app".

The Statistics app (Pocket Lab Essentials) opens a Pocket Lab Table
(``.plt.csv``, format in pocketlab_table.py) with its columns already set:
one row per sample and population, the population as the analyte, the
sample's group (a workspace's group) as the group when there are two or
more, and every count, frequency and per-channel statistic as a measure
with its unit. Only the result table travels: no events, no file paths.

The action stays hidden until the Statistics app is in the stores
(``send_to_statistics_enabled``): a developer turns it on with the
environment variable ``OPENFLO_SEND_TO_STATISTICS=1``.
"""
from __future__ import annotations

import math
import numbers
import os
import re
from collections.abc import Collection, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from .pocketlab_table import SOURCE_VERSION_RE, share_file_name, write_plt

# Where the not-installed note sends the user: the Statistics app's page on
# the Pocket Lab Essentials site, until its store listings are live.
STATISTICS_APP_URL = 'https://pocketlabessentials.com/stats'

# Off until the Statistics app is approved in the stores (Sky, 2026-11-18).
# Turn on when its listing goes live; until then only the variable below
# shows the action, for testing.
RELEASED = False
ENV_FLAG = 'OPENFLO_SEND_TO_STATISTICS'

# The table's columns that say which row it is, not what was measured.
SAMPLE_COL = 'Sample'
POPULATION_COL = 'Population'
TEXT_COLS = ('Source', 'Set aside')

# Measures with their own unit and scale; any other per-channel statistic
# is an intensity ('AU', linear), any other number dimensionless.
_KNOWN = {
    'Count': ('counts', 'count'),
    '%Parent': ('%', 'percent'),
    '%Total': ('%', 'percent'),
}
_PERCENT_STATS = ('CV', 'rCV')
_NAME_BAD = re.compile(r'[^A-Za-z0-9_.-]+')


def send_to_statistics_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the "Send to Statistics app" action is offered."""
    env = os.environ if environ is None else environ
    return RELEASED or env.get(ENV_FLAG, '') == '1'


def _finite_float(v: Any) -> float | None:
    """A finite number (numpy's included), else None (missing)."""
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _is_number_cell(v: Any) -> bool:
    """Whether `v` is a number or missing (None, empty, NaN)."""
    return v is None or v == '' or isinstance(v, numbers.Real) and not isinstance(v, bool)


def _column_name(title: str, taken: set[str]) -> str:
    """A PLT column name ([A-Za-z][A-Za-z0-9_.-]{0,63}) for `title`, unique
    in `taken`."""
    base = _NAME_BAD.sub('_', title).strip('_') or 'col'
    if not base[0].isalpha():
        base = f'c_{base}'
    base = base[:60]
    name, k = base, 2
    while name.lower() in taken:
        name = f'{base[:60 - len(str(k))]}_{k}'
        k += 1
    taken.add(name.lower())
    return name


def _measure_kind(col: str, stat_chan: Collection[str]) -> tuple[str, str] | None:
    """(unit, scale) of a measure column, or None for a non-statistic."""
    from .gating import is_channel_stat_column
    if col in _KNOWN:
        return _KNOWN[col]
    if is_channel_stat_column(col, stat_chan):
        head = col.split(' ', 1)[0]
        return ('%', 'percent') if head in _PERCENT_STATS else ('AU', 'linear')
    return None


def stats_table(rows: Sequence[Mapping[str, Any]], cols: Sequence[str],
                headings: Mapping[str, str], stat_chan: Collection[str], *,
                groups: Mapping[str, str | None] | None = None,
                version: str = '0', created: datetime | None = None) -> dict[str, Any]:
    """The PLT table (a dict for pocketlab_table.write_plt) of statistics
    rows as the Statistics window and Document.export_stats_csv make them:
    `cols` in order (``Sample`` and ``Population`` among them), `headings`
    each column's title (the CSV's), `groups` each sample's group. A row
    whose ``__unknown__`` is set has every measure missing."""
    taken: set[str] = set()
    columns: list[dict[str, Any]] = []
    getters: list[Any] = []

    def add(name: str, role: str, typ: str, label: str, get: Any, unit: str = '',
            scale: str = 'linear') -> None:
        columns.append({'name': _column_name(name, taken), 'role': role, 'type': typ,
                        'unit': unit, 'label': label, 'scale': scale})
        getters.append(get)

    def text(v: Any) -> str:
        return '' if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)

    add('sample', 'id', 'text', 'Sample', lambda r: text(r.get(SAMPLE_COL)))
    levels = {} if groups is None else {
        str(r.get(SAMPLE_COL)): text(groups.get(str(r.get(SAMPLE_COL)))) for r in rows}
    if len(set(levels.values())) >= 2:
        add('group', 'group', 'text', 'Group',
            lambda r: levels.get(str(r.get(SAMPLE_COL)), '') or 'No group')
    add('population', 'analyte', 'text', 'Population', lambda r: text(r.get(POPULATION_COL)))
    for col in cols:
        if col in (SAMPLE_COL, POPULATION_COL):
            continue
        label = headings.get(col, col)
        kind = _measure_kind(col, stat_chan)
        if kind is None and col not in TEXT_COLS and all(_is_number_cell(r.get(col)) for r in rows):
            kind = ('1', 'linear')
        if kind is None:
            add(col, 'text', 'text', label, lambda r, c=col: text(r.get(c)))
            continue
        unit, scale = kind
        add(col, 'measure', 'number', label,
            lambda r, c=col: None if r.get('__unknown__') else _finite_float(r.get(c)),
            unit=unit, scale=scale)
    when = (created or datetime.now(UTC)).astimezone(UTC)
    return {
        'source': {'toolbox': 'openflo', 'version': version if SOURCE_VERSION_RE.match(version) else '0',
                   'analysis': 'population-stats'},
        'created': when.strftime('%Y-%m-%dT%H:%MZ'),
        'assay': 'generic',
        'rows': 'population',
        'replicates': 'unknown',
        'design': 'independent',
        'settings': [['Channel statistics', 'Computed on linear values']],
        'columns': columns,
        'data': [[g(r) for g in getters] for r in rows],
    }


def write_stats_plt(path: str, table: Mapping[str, Any]) -> None:
    """Write `table` as a NEW Pocket Lab Table file at `path` (UTF-8 with a
    byte-order mark, as a file for sharing)."""
    text = write_plt(dict(table), bom=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(text)


def plt_file_name(title: str = 'population_stats') -> str:
    """The name the table is sent under: ``<title>.plt.csv``."""
    return share_file_name(title)
