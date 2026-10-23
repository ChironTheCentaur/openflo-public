"""Batch N: Excel and locale-aware CSV export (12), samples set aside with a
reason (13), and colour-blind-safe group colours plus arrow-key gate moves
(14).

All data are generated here (synthetic sample names, generic FL channels).
Each check has a negative partner: the standard CSV that must NOT change,
a colour picked by hand that a palette switch must keep, a gate on other
axes that the arrow keys must not move, a sample that is not set aside and
must stay in its group.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')

from openflo import clipboard_export, table_export, theme  # noqa: E402
from openflo.editor_keyboard import nudge_gate  # noqa: E402
from tests.conftest import gui_unavailable  # noqa: E402

# ── Item 12: locale-aware CSV ───────────────────────────────────────────────


@pytest.fixture
def europe(monkeypatch):
    """This computer uses ';' and ',' (as Excel does across most of
    Europe)."""
    monkeypatch.setattr(table_export, 'system_csv_format',
                        lambda: (';', ','))


def test_format_cell_uses_the_decimal_mark_on_real_numbers_only():
    assert table_export.format_cell(0.25, ',') == '0,25'
    assert table_export.format_cell(np.float64(1234.5), ',') == '1234,5'
    # Integers and text are not numbers with a decimal mark.
    assert table_export.format_cell(1000, ',') == '1000'
    assert table_export.format_cell('FL1-A', ',') == 'FL1-A'
    # Missing reads NA, never 0 or a blank.
    assert table_export.format_cell(float('nan'), ',') == 'NA'
    assert table_export.format_cell(None) == 'NA'
    # Full precision: nothing rounded away on the way to Excel.
    assert table_export.format_cell(1 / 3) == repr(1 / 3)


def test_system_csv_opens_as_columns_in_a_semicolon_locale(tmp_path, europe):
    df = pd.DataFrame({'Sample': ['synthetic_1', 'synthetic_2'],
                       'Count': [120, 80], '%Parent': [12.5, float('nan')]})
    path = tmp_path / 'stats.csv'
    table_export.write_frame_csv(str(path), df, style='system')
    raw = path.read_bytes()
    assert raw.startswith(b'\xef\xbb\xbf')            # BOM, so Excel reads UTF-8
    text = raw.decode('utf-8-sig')
    assert text.splitlines()[0] == 'Sample;Count;%Parent'
    assert text.splitlines()[1] == 'synthetic_1;120;12,5'
    assert text.splitlines()[2] == 'synthetic_2;80;NA'
    back = pd.read_csv(path, sep=';', decimal=',', encoding='utf-8-sig')
    assert back['%Parent'].iloc[0] == 12.5
    assert back['Count'].tolist() == [120, 80]


def test_standard_csv_is_unchanged_whatever_the_locale(tmp_path, europe):
    df = pd.DataFrame({'Sample': ['synthetic_1'], '%Parent': [12.5]})
    path = tmp_path / 'stats.csv'
    table_export.write_frame_csv(str(path), df, style='standard')
    raw = path.read_bytes()
    assert not raw.startswith(b'\xef\xbb\xbf')
    assert raw.decode() == 'Sample,%Parent\r\nsynthetic_1,12.5\r\n'


def test_csv_style_follows_the_preference(monkeypatch, europe):
    monkeypatch.setattr(table_export, 'read_prefs',
                        lambda: {'csv_style': 'system'})
    assert table_export.csv_format() == (';', ',')
    monkeypatch.setattr(table_export, 'read_prefs', lambda: {})
    assert table_export.csv_format() == (',', '.')
    # An unknown value falls back to the standard style.
    monkeypatch.setattr(table_export, 'read_prefs',
                        lambda: {'csv_style': 'bogus'})
    assert table_export.csv_style() == 'standard'
    assert table_export.csv_format() == (',', '.')


def test_system_format_reads_windows_then_the_locale(monkeypatch):
    monkeypatch.setattr(table_export, '_windows_number_format',
                        lambda: ('\t', ','))
    assert table_export.system_csv_format() == ('\t', ',')
    monkeypatch.setattr(table_export, '_windows_number_format', lambda: None)
    monkeypatch.setattr(table_export, '_locale_decimal_mark', lambda: ',')
    assert table_export.system_csv_format() == (';', ',')
    monkeypatch.setattr(table_export, '_locale_decimal_mark', lambda: '.')
    assert table_export.system_csv_format() == (',', '.')


def test_index_column_is_headed_as_pandas_heads_it(tmp_path):
    df = pd.DataFrame({'a': [1.5]}, index=pd.Index(['d1'], name='Day'))
    path = tmp_path / 'g.csv'
    table_export.write_frame_csv(str(path), df, style='standard', index=True)
    assert path.read_text().splitlines() == ['Day,a', 'd1,1.5']


def test_copied_table_uses_the_decimal_mark():
    lines = [['Sample', 'Mean'], ['synthetic_1', 0.5], ['synthetic_2', 7]]
    assert clipboard_export.table_text(lines, decimal=',') == (
        'Sample\tMean\nsynthetic_1\t0,5\nsynthetic_2\t7\n')
    assert clipboard_export.table_text(lines, decimal='.') == (
        'Sample\tMean\nsynthetic_1\t0.5\nsynthetic_2\t7\n')


# ── Item 12: Excel workbooks ────────────────────────────────────────────────

_HEADER = ['Sample', 'Population', 'Count', '%Parent']
# Populations and samples deliberately out of alphabetical order.
_ROWS = [['synthetic_2', 'Singlets', 900, 90.0],
         ['synthetic_2', 'Lymphs', 810, 90.0],
         ['synthetic_1', 'Singlets', 500, 50.0],
         ['synthetic_1', 'Lymphs', 'NA', 'NA'],
         ['synthetic_3', 'Singlets', 300, '']]


def test_statistics_sheets_one_per_statistic_in_table_order():
    sheets = table_export.statistics_sheets(_HEADER, _ROWS, 2)
    assert list(sheets) == ['All statistics', 'Count', '%Parent']
    count = sheets['Count']
    # Populations down the side, samples across, both in table order.
    assert count.columns.tolist() == ['Population', 'synthetic_2',
                                      'synthetic_1', 'synthetic_3']
    assert count['Population'].tolist() == ['Singlets', 'Lymphs']
    assert count.loc[0, 'synthetic_1'] == 500
    assert count.loc[1, 'synthetic_2'] == 810
    # A population that cannot be computed keeps reading NA...
    assert count.loc[1, 'synthetic_1'] == 'NA'
    # ...a blank stays blank, and a missing combination is empty.
    pct = sheets['%Parent']
    assert pd.isna(pct.loc[0, 'synthetic_3'])
    assert pd.isna(pct.loc[1, 'synthetic_3'])


def test_workbook_round_trips_with_numbers_as_numbers(tmp_path):
    path = tmp_path / 'stats.xlsx'
    sheets = table_export.statistics_sheets(_HEADER, _ROWS, 2)
    names = table_export.write_workbook(
        str(path), sheets, na_rep='', about={'Saved by': 'OpenFlo test'})
    assert names == ['All statistics', 'Count', '%Parent', 'About']
    back = pd.read_excel(path, sheet_name=None, keep_default_na=False)
    assert back['Count'].loc[0, 'synthetic_1'] == 500
    assert isinstance(back['Count'].loc[0, 'synthetic_1'], (int, np.integer))
    assert back['Count'].loc[1, 'synthetic_1'] == 'NA'
    assert back['About'].loc[0, 'Value'] == 'OpenFlo test'


def test_sheet_names_are_valid_and_unique():
    assert table_export.sheet_name('Median FL1-A [linear]') == \
        'Median FL1-A _linear_'
    long = 'x' * 40
    a = table_export.sheet_name(long)
    b = table_export.sheet_name(long, [a])
    assert len(a) == 31 and len(b) == 31 and a != b
    assert table_export.sheet_name('Count', ['count']) == 'Count (2)'
    assert table_export.sheet_name('  ') == 'Sheet'


def test_csvs_to_workbook_collects_new_tables_only(tmp_path):
    import time
    out = tmp_path / 'run'
    (out / 'trial_a').mkdir(parents=True)
    (out / 'trial_b').mkdir()
    old = out / 'old.csv'
    old.write_text('a\n1\n')
    os.utime(old, (1_000_000, 1_000_000))
    since = time.time() - 1
    (out / 'trial_a' / 'stats.csv').write_text('Sample,Count\ns1,5\n')
    (out / 'trial_b' / 'stats.csv').write_text('Sample,Count\ns2,7\n')
    big = out / 'events.csv'
    big.write_text('x\n' + '1\n' * 50)
    book = out / 'tables.xlsx'
    inc, skipped = table_export.csvs_to_workbook(
        str(out), str(book), since=since, max_bytes=60)
    assert sorted(inc) == [os.path.join('trial_a', 'stats.csv'),
                           os.path.join('trial_b', 'stats.csv')]
    assert skipped == ['events.csv: too large for a sheet']
    sheets = pd.read_excel(book, sheet_name=None)
    # Two files share a name: each sheet says which folder it came from.
    assert set(sheets) == {'trial_a stats', 'trial_b stats', 'About'}
    assert sheets['trial_b stats'].loc[0, 'Count'] == 7


def test_cli_accepts_xlsx_and_writes_the_workbook(tmp_path, capsys):
    from openflo import cli
    from openflo.cli_args import build_parser
    assert build_parser().parse_args(['--xlsx']).xlsx is True
    assert build_parser().parse_args([]).xlsx is False
    (tmp_path / 't.csv').write_text('a,b\n1,2.5\n')
    cli._write_tables_workbook(str(tmp_path), 0)
    book = tmp_path / 'openflo_tables.xlsx'
    assert pd.read_excel(book, sheet_name='t').loc[0, 'b'] == 2.5
    assert '1 table(s)' in capsys.readouterr().out


# ── Item 14: colour-blind-safe group colours ────────────────────────────────

def _srgb_to_lin(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_to_lab(rgb):
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = rgb @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz),
                 (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


# Deuteranopia (the commonest red-green deficiency), Machado et al. 2009,
# severity 1.0, applied to linear RGB.
_DEUTAN = np.array([[0.367322, 0.860646, -0.227968],
                    [0.280085, 0.672501, 0.047413],
                    [-0.011820, 0.042940, 0.968881]])


def _min_deutan_distance(colors):
    from matplotlib.colors import to_rgb
    lin = _srgb_to_lin([to_rgb(c) for c in colors])
    lab = _lin_to_lab(np.clip(lin @ _DEUTAN.T, 0, 1))
    d = np.linalg.norm(lab[:, None, :] - lab[None, :, :], axis=-1)
    return d[np.triu_indices(len(colors), 1)].min()


def test_default_colours_stay_apart_for_a_red_green_deficiency():
    # Okabe-Ito's eight stay well apart as a deuteranope sees them; tab10's
    # ten include pairs that close up, its green and red among them.
    # (Delta E, CIELAB.)
    assert _min_deutan_distance(theme.OKABE_ITO) > 12
    assert _min_deutan_distance(theme.TAB10) < 8
    assert _min_deutan_distance(['#2ca02c', '#d62728']) < 8


def test_group_colours_default_to_okabe_ito(monkeypatch):
    monkeypatch.setattr(theme, 'read_prefs', lambda: {})
    assert theme.group_palette_name() == 'okabe_ito'
    assert theme.categorical_colors(3) == list(theme.OKABE_ITO[:3])
    monkeypatch.setattr(theme, 'read_prefs',
                        lambda: {'group_palette': 'classic'})
    assert theme.categorical_colors(3) == list(theme.TAB10[:3])


def test_many_groups_still_get_distinct_colours():
    for n in (8, 9, 14, 15, 20, 25):
        cols = theme.categorical_colors(n, 'okabe_ito')
        assert len(cols) == n and len(set(cols)) == n


def test_pipeline_auto_palette_is_okabe_ito_in_order():
    from matplotlib.colors import to_hex

    from openflo.pipeline import categorical_cmap
    cm = categorical_cmap('auto', 3)
    # With vmin=0, vmax=n-1 group i gets colour i (not first, middle, last).
    got = [to_hex(cm(i / 2)) for i in range(3)]
    assert got == [c.lower() for c in theme.OKABE_ITO[:3]]
    tab = categorical_cmap('tab10', 2)
    assert [to_hex(tab(i)) for i in (0.0, 1.0)] == list(theme.TAB10[:2])
    # Any other name is still a matplotlib map.
    assert categorical_cmap('viridis', 4).name == 'viridis'


# ── Item 14: arrow-key gate moves (pure) ────────────────────────────────────

def test_nudge_moves_each_gate_kind_on_its_axes():
    r = {'kind': 'rect', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
         'x0': 0.0, 'x1': 1.0, 'y0': 0.0, 'y1': 2.0}
    assert nudge_gate(r, 0.5, -1.0, 'FL1-A', 'FL2-A')
    assert (r['x0'], r['x1'], r['y0'], r['y1']) == (0.5, 1.5, -1.0, 1.0)
    p = {'kind': 'polygon', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
         'vertices': [[0, 0], [1, 0], [0, 1]]}
    assert nudge_gate(p, 1.0, 2.0, 'FL1-A', 'FL2-A')
    assert p['vertices'] == [[1, 2], [2, 2], [1, 3]]
    e = {'kind': 'ellipsoid', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
         'mean': [1.0, 1.0]}
    assert nudge_gate(e, -1.0, 0.0, 'FL1-A', 'FL2-A')
    assert e['mean'] == [0.0, 1.0]
    t = {'kind': 'threshold', 'channel': 'FL2-A', 'value': 3.0}
    assert nudge_gate(t, 5.0, 0.25, 'FL1-A', 'FL2-A')       # y channel: dy
    assert t['value'] == 3.25
    i = {'kind': 'interval', 'channel': 'FL1-A', 'lo': 1.0, 'hi': 2.0}
    assert nudge_gate(i, 0.5, 9.0, 'FL1-A', 'FL2-A')
    assert (i['lo'], i['hi']) == (1.5, 2.5)


def test_nudge_leaves_a_gate_on_other_axes_alone():
    r = {'kind': 'rect', 'x_channel': 'FL3-A', 'y_channel': 'FL2-A',
         'x0': 0.0, 'x1': 1.0, 'y0': 0.0, 'y1': 2.0}
    assert not nudge_gate(r, 1.0, 1.0, 'FL1-A', 'FL2-A')
    assert r['x0'] == 0.0
    t = {'kind': 'threshold', 'channel': 'FL3-A', 'value': 3.0}
    assert not nudge_gate(t, 1.0, 1.0, 'FL1-A', 'FL2-A')
    assert t['value'] == 3.0
    q = {'kind': 'quadrant', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A'}
    assert not nudge_gate(q, 1.0, 1.0, 'FL1-A', 'FL2-A')


# ── The editor: set aside, colours, keys, statistics export ─────────────────

@pytest.fixture(scope='module')
def ed():
    try:
        import tkinter as tk
    except ImportError:
        gui_unavailable("tkinter not available — headless environment")
    try:
        root = tk.Tk()
    except Exception as e:                              # noqa: BLE001
        gui_unavailable(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    import importlib
    gui = importlib.import_module('openflo.gui')
    editor = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                      on_save=None, primary=False)
    editor.withdraw()
    editor._finish_plot_canvas()
    rng = np.random.default_rng(7)
    cols = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A']
    for k, (name, trial) in enumerate((('synthetic_a1', 'Day1'),
                                       ('synthetic_a2', 'Day1'),
                                       ('synthetic_b1', 'Day2'),
                                       ('synthetic_b2', 'Day2'))):
        df = pd.DataFrame(rng.normal(1000 + 100 * k, 50, (2000, 4)),
                          columns=cols)
        editor._samples[name] = SimpleNamespace(
            name=name, path=f'/data/{name}.fcs', data=df,
            fluor_channels=['FL1-A', 'FL2-A'],
            channel_labels={c: c for c in cols})
        editor._sample_order.append(name)
        editor._sample_trial[name] = trial
        if trial not in editor._trial_order:
            editor._trial_order.append(trial)
        editor._sample_plot_enabled[name] = True
        editor._sample_gates.setdefault(name, {})
        editor._sample_gate_seq.setdefault(name, 0)
        editor._sample_gate_order.setdefault(name, [])
    editor._channels = cols
    editor._channel_labels = {c: c for c in cols}
    editor._populate_channel_combos()
    editor._set_active_sample('synthetic_a1')
    editor.x_combo.set('FL1-A')
    editor.y_combo.set('FL2-A')
    editor.mode_var.set('dot')
    editor._replot()
    yield editor
    root.destroy()


def test_set_aside_sample_is_in_no_group_under_any_factor(ed):
    try:
        assert ed._set_sample_aside(['synthetic_a2'], 'clogged') == 1
        for factor in ('Trial / day', 'Comp vs Samples', 'Name token'):
            assert ed._sample_group_label('synthetic_a2', factor,
                                          ['synthetic']) is None
            # The sample beside it is still grouped.
            assert ed._sample_group_label('synthetic_a1', factor,
                                          ['synthetic']) is not None
        assert ed._set_aside_lines() == ['synthetic_a2: clogged']
        # An empty reason sets nothing aside.
        assert ed._set_sample_aside(['synthetic_b1'], '  ') == 0
        assert 'synthetic_b1' not in ed._sample_set_aside
    finally:
        ed._return_sample(['synthetic_a2'])
    assert ed._sample_group_label('synthetic_a2', 'Trial / day') == 'Day1'


def test_set_aside_is_greyed_in_the_tree_and_audited(ed):
    try:
        ed._set_sample_aside(['synthetic_b2'], 'too few events')
        ed._set_sample_note('synthetic_b2', 'rerun next week')
        text = ed.gate_tv.item(ed._sample_iid('synthetic_b2'), 'text')
        assert 'set aside: too few events' in text and '✎' in text
        assert text.startswith('⊘')
        other = ed.gate_tv.item(ed._sample_iid('synthetic_b1'), 'text')
        assert 'set aside' not in other and other.startswith('■')
        acts = [e['action'] for e in ed._audit_log.entries()]
        assert 'sample.set_aside' in acts
    finally:
        ed._return_sample(['synthetic_b2'])
        ed._set_sample_note('synthetic_b2', '')
    assert 'synthetic_b2' not in ed._sample_notes


def test_set_aside_and_note_survive_save_and_reload(ed):
    try:
        ed._set_sample_aside(['synthetic_b1'], 'bad stain')
        ed._set_sample_note('synthetic_b1', 'check lot')
        entry = next(e for e in ed._session_state()['samples']
                     if e['name'] == 'synthetic_b1')
        assert entry['set_aside'] == 'bad stain'
        assert entry['note'] == 'check lot'
        plain = next(e for e in ed._session_state()['samples']
                     if e['name'] == 'synthetic_a1')
        assert 'set_aside' not in plain and 'note' not in plain
        # Reload: the loader restores both from the saved entry.
        sample = ed._samples['synthetic_b1']
        ed._sample_set_aside.clear()
        ed._sample_notes.clear()
        pkey = os.path.normcase(os.path.abspath(sample.path))
        ed._pending_sample_meta[pkey] = entry
        ed._on_loaded('synthetic_b1', sample)
        assert ed._sample_set_aside == {'synthetic_b1': 'bad stain'}
        assert ed._sample_notes == {'synthetic_b1': 'check lot'}
    finally:
        ed._sample_set_aside.clear()
        ed._sample_notes.clear()


def test_group_stats_and_methods_name_the_set_aside_sample(ed):
    from openflo.ui_group_stats import GroupStatsWindow
    from openflo.ui_methods import MethodsWindow
    try:
        ed._set_sample_aside(['synthetic_a2'], 'clogged')
        w = GroupStatsWindow(ed)
        w._ch.set('FL1-A')
        w._run()
        text = w._txt.get('1.0', 'end')
        w.destroy()
        assert 'Set aside: synthetic_a2: clogged' in text
        assert 'Day1: n=1' in text and 'Day2: n=2' in text
        m = MethodsWindow(ed)
        assert ('Samples set aside from group comparisons, with the reason: '
                'synthetic_a2 (clogged).') in m._para
        m.destroy()
    finally:
        ed._return_sample(['synthetic_a2'])
    w = GroupStatsWindow(ed)
    w._ch.set('FL1-A')
    w._run()
    text = w._txt.get('1.0', 'end')
    w.destroy()
    assert 'Day1: n=2' in text and 'Set aside' not in text


def test_samples_take_okabe_ito_and_a_switch_keeps_hand_picked(ed, monkeypatch):
    prefs = {}
    monkeypatch.setattr(theme, 'read_prefs', lambda: dict(prefs))
    import openflo.prefs as prefs_mod
    monkeypatch.setattr(prefs_mod, 'write_pref',
                        lambda k, v: prefs.__setitem__(k, v))
    saved = dict(ed._sample_colors)
    try:
        ed._sample_colors.clear()
        got = [ed._color_for(n) for n in ed._sample_order[:3]]
        assert got == list(theme.OKABE_ITO[:3])
        ed._sample_colors['synthetic_b2'] = '#123456'      # picked by hand
        ed._group_palette_var.set('classic')
        ed._set_group_palette()
        assert prefs['group_palette'] == 'classic'
        assert [ed._sample_colors[n] for n in ed._sample_order[:3]] == \
            list(theme.TAB10[:3])
        assert ed._sample_colors['synthetic_b2'] == '#123456'
        ed._group_palette_var.set('okabe_ito')
        ed._set_group_palette()
        assert [ed._sample_colors[n] for n in ed._sample_order[:3]] == \
            list(theme.OKABE_ITO[:3])
    finally:
        ed._sample_colors.clear()
        ed._sample_colors.update(saved)
        ed._group_palette_var.set('okabe_ito')


def test_view_menu_offers_group_colours(ed):
    btn = next(b for b in ed._menubar_buttons if b.cget('text') == 'View')
    viewm = ed.nametowidget(btn.cget('menu'))
    labels = [viewm.entrycget(j, 'label')
              for j in range(viewm.index('end') + 1)
              if viewm.type(j) != 'separator']
    assert 'Group colors' in labels and 'Density colors' in labels


def _select_gate(ed, gid):
    ed._refresh_gate_list()
    iid = next(i for i in _all_iids(ed.gate_tv)
               if ed._parse_iid(i) == ('gate', 'synthetic_a1', gid))
    ed.gate_tv.selection_set(iid)


def _all_iids(tv, parent=''):
    for i in tv.get_children(parent):
        yield i
        yield from _all_iids(tv, i)


def test_arrow_keys_move_the_selected_gate_with_one_undo(ed):
    ed._set_active_sample('synthetic_a1')
    gid = ed._next_gate_id()
    ed._gates[gid] = {'kind': 'rect', 'x_channel': 'FL1-A',
                      'y_channel': 'FL2-A', 'x0': 900.0, 'x1': 1100.0,
                      'y0': 900.0, 'y1': 1100.0, 'parent_id': None,
                      'name': 'Box', 'enabled': True}
    ed._sample_gate_order['synthetic_a1'].append(gid)
    try:
        ed._replot()
        _select_gate(ed, gid)
        xl = ed.ax.get_xlim()
        span = xl[1] - xl[0]
        undo_before = len(ed._undo_stack)
        w = ed.canvas.get_tk_widget()
        assert w.bind('<Right>') and w.bind('<Shift-Up>')
        ed._on_arrow_key('Right', False)
        ed._undo_pending = False          # as if the Tk event loop ran
        ed._on_arrow_key('Right', False)
        g = ed._gates[gid]
        assert g['x0'] == pytest.approx(900 + 2 * 0.005 * span)
        assert g['y0'] == pytest.approx(900.0)            # y untouched
        assert len(ed._undo_stack) == undo_before + 1     # one undo step
        ed._undo_pending = False
        yl = ed.ax.get_ylim()
        ed._on_arrow_key('Up', True)
        assert g['y0'] == pytest.approx(900 + 0.05 * (yl[1] - yl[0]))
        ed._undo()
        assert ed._gates[gid]['x0'] == pytest.approx(900.0)
    finally:
        ed._gates.pop(gid, None)
        ed._sample_gate_order['synthetic_a1'].remove(gid)
        ed._refresh_gate_list()


def test_arrow_keys_do_nothing_without_a_selected_gate(ed):
    ed._refresh_gate_list()
    ed.gate_tv.selection_set(ed._sample_iid('synthetic_a1'))
    assert ed._nudge_selected_gate(0.01, 0) is False
    assert 'Select a gate' in ed.status_var.get()


def test_statistics_window_saves_a_workbook_with_set_aside(ed, tmp_path,
                                                           monkeypatch):
    from openflo import ui_statistics
    gid = ed._next_gate_id()
    ed._gates[gid] = {'kind': 'threshold', 'channel': 'FL1-A', 'op': '>',
                      'value': -1e12, 'parent_id': None, 'name': 'All',
                      'enabled': True}
    ed._sample_gate_order['synthetic_a1'].append(gid)
    try:
        ed._set_sample_aside(['synthetic_a1'], 'clogged')
        w = ui_statistics.StatisticsWindow(ed)
        rows = [{'Sample': 'synthetic_a1', 'Population': 'All',
                 'Count': 2000, '%Parent': 100.0},
                {'Sample': 'synthetic_b1', 'Population': 'All',
                 'Count': 1500, '%Parent': 100.0}]
        w._populate(rows, ['Sample', 'Population', 'Count', '%Parent'],
                    False)
        assert w._cols[:3] == ['Sample', 'Population', 'Set aside']
        path = tmp_path / 'stats.xlsx'
        monkeypatch.setattr(ui_statistics.filedialog, 'asksaveasfilename',
                            lambda **k: str(path))
        w._export_xlsx()
        book = pd.read_excel(path, sheet_name=None, keep_default_na=False)
        assert list(book)[:3] == ['All statistics', 'Count', '%Parent']
        assert book['Count'].columns.tolist() == ['Population',
                                                  'synthetic_a1',
                                                  'synthetic_b1']
        assert book['Count'].loc[0, 'synthetic_b1'] == 1500
        assert book['All statistics'].loc[0, 'Set aside'] == 'clogged'
        assert book['All statistics'].loc[1, 'Set aside'] == ''
        # The CSV follows the system style when asked.
        csv_path = tmp_path / 'stats.csv'
        monkeypatch.setattr(ui_statistics.filedialog, 'asksaveasfilename',
                            lambda **k: str(csv_path))
        monkeypatch.setattr(table_export, 'csv_style', lambda: 'system')
        monkeypatch.setattr(table_export, 'system_csv_format',
                            lambda: (';', ','))
        w._export_csv()
        first = csv_path.read_text(encoding='utf-8-sig').splitlines()[1]
        assert first == 'synthetic_a1;All;clogged;2000;100,0'
        w.destroy()
    finally:
        ed._return_sample(['synthetic_a1'])
        ed._gates.pop(gid, None)
        ed._sample_gate_order['synthetic_a1'].remove(gid)
