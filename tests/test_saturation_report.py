"""The per-sample saturation report (complaint 22): events at the top of each
channel's range, from the file's keywords, unchanged by compensation and
transforms; Analyze > Saturation report lists each sample.

Every number is checked against generated data (synthetic sample names,
generic FL channels), with negative partners: an event one unit under the
ceiling is not counted, a channel with no known range reads NA, not 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import tests._inline_editor as shared
from openflo import saturation
from tests._inline_editor import _register, _sample

editor = shared.editor


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
