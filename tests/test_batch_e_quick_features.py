"""Batch E: more statistics, editable vector figures and the clipboard, and
a per-sample saturation report (complaints 16, 18 and 22).

Every number is checked against a value worked out here from generated data
(synthetic sample names, generic FL channels), and each check has a negative
partner: a statistic that must NOT move (a robust spread under outliers), a
setting switched off (text drawn as shapes), an event one unit under the
ceiling (not counted), a channel with no known range (NA, not 0).
"""
from __future__ import annotations

import io
import os
import re
import struct
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(__file__))

from openflo import clipboard_export, saturation, theme  # noqa: E402
from openflo.gating import (  # noqa: E402
    CHANNEL_STAT_ORDER,
    MFI_RATIO,
    add_mfi_ratio,
    parse_percentiles,
    population_stats,
)

CH = ['FL1-A']


def _gate(parent, value, name):
    return {'kind': 'threshold', 'channel': 'FL1-A', 'op': '>',
            'value': value, 'parent_id': parent, 'name': name,
            'enabled': True}


def _stats(vals, want, gates=None, order=None, name='synthetic_1'):
    df = pd.DataFrame({'FL1-A': np.asarray(vals, dtype=float)})
    gates = gates or {'g1': _gate(None, -np.inf, 'All')}
    order = order or list(gates)
    return population_stats(name, df, gates, order, {}, CH, set(want),
                            CHANNEL_STAT_ORDER)


# ── Complaint 16: more statistics ───────────────────────────────────────────

def _tree():
    # 0..999: 900 events at 100 or over, 500 at 500, 100 at 900.
    return ({'g1': _gate(None, 100, 'A'), 'g2': _gate('g1', 500, 'B'),
             'g3': _gate('g2', 900, 'C')}, ['g1', 'g2', 'g3'])


def test_grandparent_percent_counts_against_the_parents_parent():
    gates, order = _tree()
    rows = {r['__gid__']: r for r in _stats(
        np.arange(1000), {'Count', '%Parent', '%Grandparent'}, gates, order)}
    # A gate on all events has no grandparent: blank, not a number.
    assert np.isnan(rows['g1']['%Grandparent'])
    # B's grandparent is all events; C's is A.
    assert rows['g2']['%Grandparent'] == pytest.approx(500 / 1000 * 100)
    assert rows['g3']['%Grandparent'] == pytest.approx(100 / 900 * 100)
    # And %Parent is unchanged beside it.
    assert rows['g3']['%Parent'] == pytest.approx(100 / 500 * 100)


def test_grandparent_is_absent_unless_asked():
    gates, order = _tree()
    rows = _stats(np.arange(1000), {'Count', '%Parent'}, gates, order)
    assert all('%Grandparent' not in r for r in rows)


def test_sd_rsd_and_mad_match_their_definitions():
    rng = np.random.default_rng(1)
    vals = rng.normal(1000.0, 100.0, 200_000)
    r = _stats(vals, {'Mean', 'SD', 'rSD', 'CV', 'rCV', 'MAD',
                      'Median'})[0]
    p_lo, p_hi = np.percentile(vals, [15.87, 84.13])
    med = np.median(vals)
    assert r['SD FL1-A'] == pytest.approx(np.std(vals, ddof=1), rel=1e-12)
    assert r['rSD FL1-A'] == pytest.approx(0.5 * (p_hi - p_lo), rel=1e-12)
    assert r['MAD FL1-A'] == pytest.approx(np.median(np.abs(vals - med)),
                                           rel=1e-12)
    # CV and rCV are these spreads over the centre.
    assert r['CV FL1-A'] == pytest.approx(
        r['SD FL1-A'] / r['Mean FL1-A'] * 100, rel=1e-12)
    assert r['rCV FL1-A'] == pytest.approx(
        r['rSD FL1-A'] / r['Median FL1-A'] * 100, rel=1e-12)
    # For a normal population all three estimate sigma = 100.
    assert r['SD FL1-A'] == pytest.approx(100, rel=0.02)
    assert r['rSD FL1-A'] == pytest.approx(100, rel=0.02)
    assert r['MAD FL1-A'] * 1.4826 == pytest.approx(100, rel=0.02)


def test_robust_spreads_ignore_outliers_that_inflate_the_sd():
    rng = np.random.default_rng(2)
    clean = rng.normal(1000.0, 100.0, 100_000)
    dirty = np.concatenate([clean, np.full(1_000, 1e6)])     # 1 % off scale
    a = _stats(clean, {'SD', 'rSD', 'MAD'})[0]
    b = _stats(dirty, {'SD', 'rSD', 'MAD'})[0]
    assert b['SD FL1-A'] > 50 * a['SD FL1-A']
    assert b['rSD FL1-A'] == pytest.approx(a['rSD FL1-A'], rel=0.05)
    assert b['MAD FL1-A'] == pytest.approx(a['MAD FL1-A'], rel=0.05)


def test_percentiles_give_one_column_each():
    vals = np.arange(1, 1001, dtype=float)
    r = _stats(vals, {'Median', 'P5', 'P50', 'P97.5'})[0]
    for name, q in (('P5', 5), ('P50', 50), ('P97.5', 97.5)):
        assert r[f'{name} FL1-A'] == pytest.approx(np.percentile(vals, q))
    assert r['P50 FL1-A'] == r['Median FL1-A']
    assert not any(k.startswith('P25') for k in r)


def test_percentile_text_is_parsed_and_checked():
    assert parse_percentiles('5, 25,75 95') == ['P5', 'P25', 'P75', 'P95']
    assert parse_percentiles('2.5;97.5%, 2.5') == ['P2.5', 'P97.5']
    assert parse_percentiles('  ') == []
    for bad in ('101', '-1', 'five'):
        with pytest.raises(ValueError):
            parse_percentiles(bad)


def _two_samples(want):
    gates = {'g1': _gate(None, -np.inf, 'All'), 'g2': _gate('g1', 0, 'Pos')}
    ctrl = np.linspace(90.0, 110.0, 201)              # median 100
    rows = _stats(ctrl, want, gates, name='control_1')
    rows += _stats(ctrl * 8.0, want, gates, name='stained_1')
    return rows


def test_mfi_ratio_divides_by_the_same_population_in_the_reference():
    rows = add_mfi_ratio(_two_samples({'Median', MFI_RATIO}), 'control_1')
    by = {(r['Sample'], r['Population']): r for r in rows}
    assert by[('stained_1', 'All')]['MFI ratio FL1-A'] == pytest.approx(8.0)
    assert by[('control_1', 'All')]['MFI ratio FL1-A'] == pytest.approx(1.0)
    # A different reference gives a different ratio.
    rows = add_mfi_ratio(_two_samples({MFI_RATIO}), 'stained_1')
    by = {(r['Sample'], r['Population']): r for r in rows}
    assert by[('control_1', 'All')]['MFI ratio FL1-A'] == pytest.approx(
        1 / 8.0)


def test_mfi_ratio_shows_without_the_median_column():
    rows = add_mfi_ratio(_two_samples({MFI_RATIO}), 'control_1')
    assert all('Median FL1-A' not in r for r in rows)
    assert all('MFI ratio FL1-A' in r for r in rows)


def test_mfi_ratio_is_blank_without_a_usable_reference():
    rows = _two_samples({MFI_RATIO})
    # Reference rows taken from elsewhere (a curated table): only 'All'.
    ref = [r for r in rows if r['Sample'] == 'control_1'
           and r['Population'] == 'All']
    stained = [r for r in rows if r['Sample'] == 'stained_1']
    add_mfi_ratio(stained, 'control_1', reference_rows=ref)
    by = {r['__gid__']: r for r in stained}
    assert by['g1']['MFI ratio FL1-A'] == pytest.approx(8.0)
    assert np.isnan(by['g2']['MFI ratio FL1-A'])       # reference lacks it
    # A reference median at or below zero has no ratio.
    gates = {'g1': _gate(None, -np.inf, 'All')}
    neg = _stats(np.full(11, -5.0), {MFI_RATIO}, gates, name='control_1')
    pos = _stats(np.full(11, 50.0), {MFI_RATIO}, gates, name='stained_1')
    add_mfi_ratio(neg + pos, 'control_1')
    assert np.isnan(pos[0]['MFI ratio FL1-A'])
    # No reference sample loaded at all: blank, never 1 or 0.
    rows = add_mfi_ratio(_two_samples({MFI_RATIO}), 'missing_1')
    assert all(np.isnan(r['MFI ratio FL1-A']) for r in rows)


# ── Complaint 18: editable vector figures, DPI, font, clipboard ────────────

@pytest.fixture
def prefs(monkeypatch):
    """theme reads preferences from this dict instead of the user's file."""
    store = {'export_provenance': False}
    monkeypatch.setattr(theme, 'read_prefs', lambda: dict(store))
    return store


def _figure():
    from matplotlib.figure import Figure
    fig = Figure(figsize=(3, 2), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1])
    ax.set_title('GATELABEL')
    return fig


def _save(fig, fmt, **kw):
    buf = io.BytesIO()
    theme.savefig_background(fig, buf, fmt=fmt, **kw)
    return buf.getvalue()


def test_pdf_text_is_embedded_as_truetype(prefs):
    pdf = _save(_figure(), 'pdf')
    assert b'/FontFile2' in pdf and b'/Type3' not in pdf
    prefs['export_editable_text'] = False
    pdf = _save(_figure(), 'pdf')
    assert b'/Type3' in pdf and b'/FontFile2' not in pdf


def test_svg_text_stays_text(prefs):
    svg = _save(_figure(), 'svg').decode()
    assert re.search(r'<text[^>]*>GATELABEL</text>', svg)
    prefs['export_editable_text'] = False
    svg = _save(_figure(), 'svg').decode()
    assert not re.search(r'<text[ >]', svg)       # glyphs drawn as paths


def test_export_settings_do_not_leak_into_other_plots(prefs):
    import matplotlib
    before = matplotlib.rcParams['pdf.fonttype']
    _save(_figure(), 'pdf')
    assert matplotlib.rcParams['pdf.fonttype'] == before


def _png_size(png):
    return struct.unpack('>II', png[16:24])


def test_export_dpi_follows_the_preference(prefs):
    fig = _figure()                        # 3 x 2 inches
    assert _png_size(_save(fig, 'png', tight=False)) == (900, 600)  # 300
    prefs['export_dpi'] = 600
    assert _png_size(_save(fig, 'png', tight=False)) == (1800, 1200)
    prefs['export_dpi'] = 'junk'
    assert theme.export_dpi() == theme.EXPORT_DPI == 300
    # A caller's explicit DPI (the layout editor's own) still wins.
    assert _png_size(_save(fig, 'png', tight=False, dpi=100)) == (300, 200)


def test_figure_font_preference_sets_and_restores_the_plot_font(prefs):
    import matplotlib
    keys = ('font.family', 'font.sans-serif')
    original = {k: list(matplotlib.rcParams[k]) for k in keys}
    try:
        prefs['figure_font'] = 'DejaVu Serif'
        theme.apply_figure_font()
        assert matplotlib.rcParams['font.sans-serif'][0] == 'DejaVu Serif'
        fig = _figure()                    # drawn after the change
        pdf = _save(fig, 'pdf')
        assert b'DejaVuSerif' in pdf
        prefs['figure_font'] = ''
        theme.apply_figure_font()
        assert {k: list(matplotlib.rcParams[k]) for k in keys} == original
        assert b'DejaVuSerif' not in _save(_figure(), 'pdf')
    finally:
        for k, v in original.items():
            matplotlib.rcParams[k] = v


def test_installed_fonts_are_a_subset_of_the_offer():
    fonts = theme.installed_figure_fonts()
    assert fonts and set(fonts) <= set(theme.FIGURE_FONTS)
    assert 'DejaVu Sans' in fonts           # matplotlib's own, always there


def test_table_text_is_tab_separated_with_safe_cells():
    text = clipboard_export.table_text(
        [['Sample', 'Population', 'Median FL1-A (linear)'],
         ['synthetic_1', 'A\tB', 12.5],
         ['synthetic_2', 'line\nbreak', float('nan')],
         ['synthetic_3', None, 'NA']])
    lines = text.splitlines()
    assert lines[0] == 'Sample\tPopulation\tMedian FL1-A (linear)'
    assert lines[1] == 'synthetic_1\tA B\t12.5'
    assert lines[2] == 'synthetic_2\tline break\t'
    assert lines[3] == 'synthetic_3\t\tNA'
    assert len(lines) == 4


def test_png_becomes_a_windows_bitmap():
    from PIL import Image
    img = Image.new('RGB', (5, 3), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    dib = clipboard_export.png_to_dib(buf.getvalue())
    size, w, h, planes, bits = struct.unpack('<IiiHH', dib[:16])
    assert (size, w, h, planes, bits) == (40, 5, 3, 1, 24)
    stride = (5 * 3 + 3) // 4 * 4
    assert len(dib) == 40 + stride * 3
    # Bottom-up BGR: the first pixel is blue 0, green 0, red 255.
    assert dib[40:43] == bytes([0, 0, 255])


def test_linux_picks_the_clipboard_tool_that_is_there(monkeypatch):
    have = set()
    monkeypatch.setattr(clipboard_export.shutil, 'which',
                        lambda t: f'/usr/bin/{t}' if t in have else None)
    monkeypatch.delenv('WAYLAND_DISPLAY', raising=False)
    assert clipboard_export.linux_command() is None
    have.add('xclip')
    assert clipboard_export.linux_command()[0] == 'xclip'
    have.add('wl-copy')
    assert clipboard_export.linux_command()[0] == 'xclip'   # X11 session
    monkeypatch.setenv('WAYLAND_DISPLAY', 'wayland-0')
    assert clipboard_export.linux_command()[0] == 'wl-copy'


def test_copy_figure_sends_a_png_and_says_when_it_cannot(monkeypatch, prefs):
    calls = []

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(clipboard_export.sys, 'platform', 'linux')
    monkeypatch.setattr(clipboard_export.subprocess, 'run', fake_run)
    monkeypatch.delenv('WAYLAND_DISPLAY', raising=False)
    monkeypatch.setattr(clipboard_export.shutil, 'which', lambda t: None)
    ok, msg = clipboard_export.copy_figure(_figure())
    assert not ok and 'xclip' in msg and not calls
    monkeypatch.setattr(clipboard_export.shutil, 'which',
                        lambda t: '/usr/bin/xclip' if t == 'xclip' else None)
    ok, msg = clipboard_export.copy_figure(_figure())
    assert ok, msg
    cmd, kw = calls[-1]
    assert cmd[0] == 'xclip' and 'image/png' in cmd
    assert kw['input'][:8] == b'\x89PNG\r\n\x1a\n'
    # Not captured: xclip keeps serving the clipboard in the background,
    # and waiting on its pipes would hang.
    assert kw.get('stdout') == subprocess.DEVNULL
    assert not kw.get('capture_output')


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows clipboard')
def test_copy_figure_reaches_the_windows_clipboard(prefs):  # pragma: no cover
    import ctypes
    user32 = ctypes.windll.user32
    ok, msg = clipboard_export.copy_figure(_figure())
    assert ok, msg
    assert user32.IsClipboardFormatAvailable(8)              # CF_DIB
    assert user32.IsClipboardFormatAvailable(
        user32.RegisterClipboardFormatW('PNG'))


# ── Complaint 22: saturation report ─────────────────────────────────────────

def _fcs(tmp_path, data, names, **kw):
    from openflo.pipeline import FlowSample
    from tests._fcs_writer import write_fcs
    path = write_fcs(tmp_path / 'synthetic_1.fcs', np.asarray(data), names,
                     **kw)
    return FlowSample(str(path))


def test_float_file_reports_events_at_the_top_of_each_channel(tmp_path):
    n = 1000
    rng = np.random.default_rng(3)
    fsc = rng.uniform(1e4, 1e5, n)
    fl1 = rng.uniform(10, 1e4, n)
    fl2 = rng.uniform(10, 1e4, n)
    fsc[:10] = 262143.0                      # 1 % at the top
    fl1[:30] = 262143.0                      # 3 %, 10 of them shared
    fl1[30:35] = 262142.0                    # one unit under: not clipped
    fl2[:2] = 262144.0                       # past the top counts too
    s = _fcs(tmp_path, np.column_stack([fsc, fl1, fl2]),
             ['FSC-A', 'FL1-A', 'FL2-A'], datatype='F')
    r = saturation.sample_saturation('synthetic_1', s)
    assert r['Events'] == n
    assert r['FSC-A'] == pytest.approx(1.0)
    assert r['FL1-A'] == pytest.approx(3.0)
    assert r['FL2-A'] == pytest.approx(0.2)
    assert r['Any channel'] == pytest.approx(3.0)   # FSC's 10 are FL1's
    assert r['__flagged__'] == ['FSC-A', 'FL1-A']
    assert r['__unknown__'] == []


def test_a_clean_sample_reports_zero(tmp_path):
    data = np.column_stack([np.linspace(1e3, 2e5, 500)] * 2)
    s = _fcs(tmp_path, data, ['FSC-A', 'FL1-A'], datatype='F')
    r = saturation.sample_saturation('synthetic_1', s)
    assert r['FSC-A'] == 0.0 and r['FL1-A'] == 0.0
    assert r['Any channel'] == 0.0 and r['__flagged__'] == []


def test_integer_log_and_gain_channels_use_their_converted_top(tmp_path):
    stored = np.array([[1023, 1023, 1023, 1],
                       [1023, 1022, 1022, 2],
                       [500, 100, 100, 3],
                       [10, 1023, 1023, 4]])
    s = _fcs(tmp_path, stored, ['FSC-A', 'FL1-A', 'FL2-A', 'Time'],
             pne=['0,0', '4,1', '0,0', '0,0'], png=[1, 1, 2, 1])
    # Loaded as intensities: FL1-A's top is 10 ** (4 * 1023 / 1024).
    assert s.raw['FL1-A'].max() == pytest.approx(10 ** (4 * 1023 / 1024))
    r = saturation.sample_saturation('synthetic_1', s)
    assert r['FSC-A'] == pytest.approx(50.0)
    assert r['FL1-A'] == pytest.approx(50.0)     # 1022 is under the top
    assert r['FL2-A'] == pytest.approx(50.0)     # gain 2: top 511.5
    assert 'Time' not in r


def test_compensation_and_transforms_do_not_change_the_report(tmp_path):
    data = np.column_stack([np.linspace(1e3, 2e5, 100),
                            np.r_[np.full(5, 262143.0),
                                  np.linspace(10, 1e4, 95)]])
    s = _fcs(tmp_path, data, ['FSC-A', 'FL1-A'], datatype='F')
    before = saturation.sample_saturation('synthetic_1', s)
    s.data['FL1-A'] = s.data['FL1-A'] * 0.5 - 100.0   # as compensation does
    after = saturation.sample_saturation('synthetic_1', s)
    assert after['FL1-A'] == before['FL1-A'] == pytest.approx(5.0)


def test_a_channel_with_no_known_range_reads_na():
    from openflo.pipeline import FlowSample
    df = pd.DataFrame({'FSC-A': np.linspace(1, 1e5, 50),
                       'FL1-A': np.full(50, 262143.0)})
    s = FlowSample.from_dataframe(df, name='synthetic_1')
    r = saturation.sample_saturation('synthetic_1', s)
    assert np.isnan(r['FL1-A']) and np.isnan(r['Any channel'])
    assert r['__unknown__'] == ['FSC-A', 'FL1-A']
    assert r['__flagged__'] == []


def test_ceiling_reads_the_file_keywords():
    md = {'datatype': 'F', 'par': '2', 'p1n': 'FSC-A', 'p1r': '262144',
          'p2n': 'FL1-A', 'p2r': '1024'}
    assert saturation.channel_ceiling(md, 1) == 262143.0
    assert saturation.channel_ceiling(md, 2) == 1023.0
    assert saturation.channel_ceiling(md, 3) is None
    assert saturation.channel_ceiling({'p1r': 'x'}, 1) is None
    assert saturation._parameter_index(md, ['FL1-A', 'FSC-A'], 'FL1-A') == 2


# ── The windows ─────────────────────────────────────────────────────────────

def _inline(widget, work, on_done=None, on_error=None, on_finally=None):
    try:
        r = work()
    except Exception as exc:                      # noqa: BLE001
        if on_error:
            on_error(exc)
    else:
        if on_done:
            on_done(r)
    finally:
        if on_finally:
            on_finally()


def _register(ed, s, gates):
    from test_statistics_export import _register as reg
    reg(ed, s, gates, list(gates))
    ed._trial_order[:] = list(dict.fromkeys(ed._trial_order))  # one trial


@pytest.fixture
def editor(monkeypatch):
    from test_gui_smoke import _editor_or_skip

    import openflo.async_task as _at
    monkeypatch.setattr(_at, 'run_async', _inline)
    root, ed, _gui = _editor_or_skip()
    yield ed
    try:
        ed.destroy()
        root.destroy()
    except Exception:                             # noqa: BLE001
        pass


def _sample(name, vals):
    from openflo.pipeline import FlowSample
    return FlowSample.from_dataframe(
        pd.DataFrame({'FSC-A': np.linspace(1.0, 10.0, len(vals)),
                      'FL1-A': np.asarray(vals, dtype=float)}), name=name)


def test_statistics_window_offers_the_new_columns_and_copies(editor):
    from openflo.ui_statistics import StatisticsWindow
    gates = {'g1': {'kind': 'threshold', 'channel': 'FSC-A', 'value': 0.0,
                    'parent_id': None, 'name': 'All', 'enabled': True},
             'g2': {'kind': 'threshold', 'channel': 'FSC-A', 'value': 5.0,
                    'parent_id': 'g1', 'name': 'Upper', 'enabled': True}}
    ctrl = np.linspace(90.0, 110.0, 21)
    _register(editor, _sample('control_1', ctrl), gates)
    _register(editor, _sample('stained_1', ctrl * 4), gates)
    w = StatisticsWindow(editor)
    w.withdraw()
    for st in ('%Grandparent', 'SD', 'rSD', 'MAD'):
        w._stat_vars[st].set(True)
    w._pct_on.set(True)
    w._pct_text.set('10, 90')
    w._ratio_on.set(True)
    w._sync_ratio_choices()
    assert w._ratio_ref.get() == 'control_1'           # first loaded
    w._refresh()
    for col in ('%Grandparent', 'SD FL1-A', 'rSD FL1-A', 'MAD FL1-A',
                'P10 FL1-A', 'P90 FL1-A', 'MFI ratio FL1-A'):
        assert col in w._cols, col
    assert w._heading('P10 FL1-A') == 'P10 FL1-A (linear)'
    assert w._heading('MFI ratio FL1-A') == 'MFI ratio FL1-A (vs control_1)'
    by = {(r['Sample'], r['Population']): r for r in w._rows}
    assert by[('stained_1', 'All')]['MFI ratio FL1-A'] == pytest.approx(4.0)
    assert by[('stained_1', 'All')]['P90 FL1-A'] == pytest.approx(
        np.percentile(ctrl * 4, 90))
    assert 'MFI ratio = median' in w.status_var.get()

    w._copy_table()
    lines = w.clipboard_get().splitlines()
    assert lines[0].split('\t')[:2] == ['Sample', 'Population']
    assert 'MFI ratio FL1-A (vs control_1)' in lines[0].split('\t')
    assert len(lines) == 1 + len(w._rows)

    # A curated table without the reference still gets its ratio.
    w._set_targets([('stained_1', 'g1')], 'editor')
    assert [r['Sample'] for r in w._rows] == ['stained_1']
    assert w._rows[0]['MFI ratio FL1-A'] == pytest.approx(4.0)

    # A typo in the percentiles is said, and adds no column.
    w._pct_text.set('10, ninety')
    w._refresh()
    assert not any(c.startswith('P10 ') for c in w._cols)
    assert 'Percentiles' in w.status_var.get()
    w.destroy()


def test_copy_plot_menu_item_copies_the_main_figure(editor, monkeypatch):
    seen = []
    monkeypatch.setattr(clipboard_export, 'copy_figure',
                        lambda fig, dpi=None: (seen.append(fig)
                                               or (True, 'Copied.')))
    editor._copy_plot_image()
    assert seen == []                       # nothing loaded: nothing to copy
    _register(editor, _sample('synthetic_1', np.arange(10.0)), {})
    editor._copy_plot_image()
    assert seen == [editor.fig]
    assert editor.status_var.get() == 'Copied.'


def test_figure_windows_offer_copy_on_right_click(editor, monkeypatch):
    import tkinter as tk

    from openflo.tool_window import add_copy_menu, figure_panel
    top = tk.Toplevel(editor)
    _frame, fig, canvas = figure_panel(top, dark=False)
    seen = []
    monkeypatch.setattr(clipboard_export, 'copy_figure',
                        lambda f, dpi=None: (seen.append(f) or (True, '')))
    canvas.get_tk_widget().copy_figure()       # the panel wired it up
    assert seen == [fig]
    add_copy_menu(top, fig)()
    assert seen == [fig, fig]
    top.destroy()


def test_saturation_window_lists_each_sample(editor, tmp_path, monkeypatch):
    import csv

    import openflo.tool_window as tw
    from openflo.ui_saturation import SaturationReportWindow
    data = np.column_stack([np.linspace(1e3, 2e5, 200),
                            np.r_[np.full(4, 262143.0),
                                  np.linspace(10, 1e4, 196)]])
    s = _fcs(tmp_path, data, ['FSC-A', 'FL1-A'], datatype='F')
    _register(editor, s, {})
    _register(editor, _sample('synthetic_2', np.arange(10.0)), {})
    w = SaturationReportWindow(editor)
    w.withdraw()
    rows = {r['Sample']: r for r in w._rows}
    assert rows['synthetic_1']['FL1-A'] == pytest.approx(2.0)
    assert rows['synthetic_1']['__flagged__'] == ['FL1-A']
    assert np.isnan(rows['synthetic_2']['FL1-A'])        # no range known
    status = w.status_var.get()
    assert 'synthetic_1 (FL1-A)' in status and 'NA' in status
    items = w.tv.get_children()
    assert w.tv.item(items[0], 'tags') == ('flag',)
    assert w.tv.item(items[1], 'tags') in ('', ())
    assert 'NA' in w.tv.item(items[1], 'values')

    out = tmp_path / 'sat.csv'
    monkeypatch.setattr(tw.filedialog, 'asksaveasfilename',
                        lambda **k: str(out))
    w._export_csv()
    got = list(csv.reader(open(out, encoding='utf-8')))
    assert got[0][:4] == ['Sample', 'Events', 'Any channel %',
                          'Removed by QC']
    assert got[1][0] == 'synthetic_1' and float(got[1][5]) == 2.0
    assert got[2][5] == 'NA'
    w.destroy()


def test_saturation_report_is_on_the_analyze_menu(editor):
    opened = []
    editor._show_or_raise = lambda key, make: opened.append(key)
    editor._open_saturation_report()
    assert opened == []                    # no samples: says so instead
    assert 'Load samples' in editor.status_var.get()
    _register(editor, _sample('synthetic_1', np.arange(10.0)), {})
    editor._open_saturation_report()
    assert opened == ['saturation']
