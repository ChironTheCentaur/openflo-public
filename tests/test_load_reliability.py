"""Damaged or unusual input files read sensibly, problems reach the user, Undo
says what it can't reverse, and exported FCS files keep their keywords.

* Load notes: a sample's load warnings (a renamed channel, a wrong $TOT, a
  compensation matrix that could not be used) went only to the log or a
  status line the next sample overwrote. They now show once, in one dialog,
  and are kept in the history.
* FCS files: a missing $PnN escaped as "KeyError: 3", a missing $PnR was
  refused, two channels with one name broke the transform, and damaged files
  gave raw parser errors.
* $SPILLOVER naming markers ($PnS) instead of detectors left the sample
  uncompensated.
* CSV input: no header row, semicolons with decimal commas, tabs and '#'
  comment lines were misread or refused with a raw error.
* Undo after a compensation or transform change silently undid the gate
  change before it.
* Exported FCS files carried six keywords and nothing saying the values were
  compensated.

Generated data throughout (synthetic FCS and CSV files written here), with a
negative case beside each behaviour.
"""
from __future__ import annotations

import os

import flowio
import numpy as np
import pandas as pd
import pytest

from tests._fcs_writer import write_fcs
from tests.test_transform_records import (
    _new_editor,
    capture_warnings,
    isolate_home,
    settle,
)

NAMES = ['FL1-A', 'FL2-A', 'FL3-A']


def _events(n=500, seed=0):
    return np.random.default_rng(seed).integers(0, 1000, (n, 3))


def _fcs(tmp_path, stem='synthetic', names=NAMES, data=None, **kw):
    return write_fcs(str(tmp_path / f'{stem}.fcs'),
                     _events() if data is None else data, names, **kw)


# ── item 11: damaged or unusual FCS files ────────────────────────────────


def test_a_clean_file_loads_without_notes(tmp_path):
    from openflo.pipeline import FlowSample
    s = FlowSample(_fcs(tmp_path))
    assert s.channel_names == NAMES
    assert s.load_warnings == []


def test_a_parameter_without_a_name_is_called_by_its_number(tmp_path):
    from openflo.pipeline import FlowSample
    s = FlowSample(_fcs(tmp_path, drop=('$P2N',)))
    assert s.channel_names == ['FL1-A', 'Ch2', 'FL3-A']
    np.testing.assert_array_equal(s.raw['Ch2'].to_numpy(), _events()[:, 1])


def test_a_missing_range_is_taken_from_the_bit_width(tmp_path):
    from openflo.pipeline import FlowSample
    s = FlowSample(_fcs(tmp_path, drop=('$P2R',)))
    np.testing.assert_array_equal(s.raw['FL2-A'].to_numpy(), _events()[:, 1])
    assert any('$P2R' in w and '65,536' in w for w in s.load_warnings)


def test_duplicate_channel_names_are_kept_apart(tmp_path):
    from openflo.pipeline import FlowSample
    s = FlowSample(_fcs(tmp_path, names=['FL1-A', 'FL1-A', 'FL2-A']))
    assert s.channel_names == ['FL1-A', 'FL1-A (2)', 'FL2-A']
    ev = _events()
    np.testing.assert_array_equal(s.raw['FL1-A'].to_numpy(), ev[:, 0])
    np.testing.assert_array_equal(s.raw['FL1-A (2)'].to_numpy(), ev[:, 1])
    assert any("'FL1-A (2)'" in w for w in s.load_warnings)


def test_an_event_count_that_disagrees_with_the_data_is_noted(tmp_path):
    from openflo.pipeline import FlowSample
    s = FlowSample(_fcs(tmp_path, extra={'$TOT': '600'}))
    assert len(s.raw) == 500
    assert s.load_warnings == ['$TOT says 600 events; the data holds 500']


@pytest.mark.parametrize('damage, words', [
    ('cut', 'cut short'),
    ('text', 'not an FCS file'),
])
def test_a_damaged_file_says_why_in_plain_words(tmp_path, damage, words):
    from openflo.pipeline import FcsParseError, FlowSample
    p = _fcs(tmp_path)
    if damage == 'cut':
        with open(p, 'rb') as f:
            b = f.read()
        with open(p, 'wb') as f:
            f.write(b[:len(b) - 700])
    else:
        with open(p, 'w', encoding='utf-8') as f:
            f.write('time,value\n1,2\n')
    with pytest.raises(FcsParseError) as e:
        FlowSample(p)
    assert words in str(e.value)
    assert os.path.basename(p) in str(e.value)


def test_plain_problem_names_a_missing_keyword():
    from openflo.pipeline import plain_fcs_problem
    assert plain_fcs_problem(KeyError('p3n')) == (
        'required keyword $P3N (channel name) is missing')
    assert plain_fcs_problem(KeyError('par')) == (
        'required keyword $PAR is missing')


# ── item 12: spillover named by marker ───────────────────────────────────


def _spill_file(tmp_path, markers, spill_names, extra_values=''):
    rng = np.random.default_rng(3)
    data = rng.normal(5000, 300, (400, 3)).clip(0)
    kw = {f'$P{i}S': m for i, m in enumerate(markers, 1)}
    n = len(spill_names)
    matrix = np.eye(n)
    matrix[0, 1] = 0.1
    kw['$SPILLOVER'] = ','.join(
        [str(n), *spill_names, *(f'{v:g}' for v in matrix.ravel())]
    ) + extra_values
    return write_fcs(str(tmp_path / 'spill.fcs'), data, NAMES,
                     datatype='F', extra=kw)


def test_a_spillover_naming_markers_is_matched_to_detectors(tmp_path):
    from openflo.pipeline import FlowSample
    s = FlowSample(_spill_file(tmp_path, ['M1', 'M2', 'M3'],
                               ['M1', 'M2', 'M3']))
    s.auto_compensate()
    assert list(s.comp_channels) == NAMES
    assert not any(w.startswith('compensation') for w in s.load_warnings)


def test_a_spillover_naming_detectors_still_works(tmp_path):
    from openflo.pipeline import FlowSample
    s = FlowSample(_spill_file(tmp_path, ['M1', 'M2', 'M3'], NAMES))
    s.auto_compensate()
    assert list(s.comp_channels) == NAMES


def test_an_ambiguous_marker_is_not_guessed_and_the_user_is_told(tmp_path):
    from openflo.pipeline import FlowSample
    # M1 is the marker of two detectors: which one the matrix means is a
    # guess, so the file stays uncompensated and the user is told.
    s = FlowSample(_spill_file(tmp_path, ['M1', 'M1', 'M3'], ['M1', 'M3']))
    s.auto_compensate()
    assert not s.comp_channels
    assert any(w.startswith('compensation:') for w in s.load_warnings)


def test_values_past_the_matrix_are_reported(tmp_path):
    from openflo.pipeline import FlowSample
    s = FlowSample(_spill_file(tmp_path, ['M1', 'M2', 'M3'], NAMES,
                               extra_values=',9,9'))
    s.auto_compensate()
    assert list(s.comp_channels) == NAMES
    assert any('past its 3x3 matrix' in w for w in s.load_warnings)


# ── item 13: CSV layouts ─────────────────────────────────────────────────


def _frame():
    rng = np.random.default_rng(5)
    return pd.DataFrame({'FL1-A': rng.normal(100, 10, 50).round(3),
                         'FL2-A': rng.normal(50, 5, 50).round(3)})


def test_a_plain_csv_reads_as_before_without_notes(tmp_path):
    from openflo.pipeline import read_events_csv
    p = tmp_path / 'plain.csv'
    _frame().to_csv(p, index=False)
    df, notes = read_events_csv(str(p))
    pd.testing.assert_frame_equal(df, pd.read_csv(p))
    assert notes == []


@pytest.mark.parametrize('layout', ['semicolon', 'tab', 'comments',
                                    'no_header', 'quoted_commas'])
def test_common_csv_layouts_read_every_event(tmp_path, layout):
    from openflo.pipeline import read_events_csv
    want = _frame()
    p = tmp_path / f'{layout}.csv'
    if layout == 'semicolon':
        want.to_csv(p, index=False, sep=';', decimal=',')
    elif layout == 'tab':
        want.to_csv(p, index=False, sep='\t')
    elif layout == 'comments':
        p.write_text('# exported by a synthetic tool\n# run 1\n'
                     + want.to_csv(index=False), encoding='utf-8')
    elif layout == 'no_header':
        want.to_csv(p, index=False, header=False)
    else:
        rows = ['"FL1-A","FL2-A"'] + [
            f'"{a:.3f}","{b:.3f}"'.replace('.', ',')
            for a, b in want.to_numpy()]
        p.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    df, notes = read_events_csv(str(p))
    assert len(df) == len(want)
    np.testing.assert_allclose(df.to_numpy(dtype=float), want.to_numpy(),
                               rtol=1e-9)
    assert notes
    if layout == 'no_header':
        assert list(df.columns) == ['Ch1', 'Ch2']
    else:
        assert list(df.columns) == ['FL1-A', 'FL2-A']


def test_a_csv_without_numbers_is_refused_in_plain_words(tmp_path):
    from openflo.pipeline import read_events_csv
    p = tmp_path / 'names.csv'
    p.write_text('sample,tube\nsynthetic_A,one\nsynthetic_B,two\n',
                 encoding='utf-8')
    with pytest.raises(ValueError, match='no numeric column'):
        read_events_csv(str(p))


# ── items 10 and 14: the editor ──────────────────────────────────────────


@pytest.fixture
def env(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    shown = capture_warnings(monkeypatch)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"Tk cannot initialise: {e}")
    root.withdraw()
    yield root, tmp_path, shown
    root.destroy()


def _loaded(path):
    from openflo.pipeline import FlowSample
    s = FlowSample(path)
    s.run_qc()
    s.auto_compensate()
    s.apply_transform()
    return s


def _notes(shown):
    return [m for t, m in shown if t == 'Notes from loading']


def test_load_problems_are_shown_once_and_kept(env):
    root, tmp, shown = env
    ed = _new_editor(root)
    kept = []
    real = ed._audit
    ed._audit = lambda action, **k: (
        kept.append(k) if action == 'load.note' else None, real(action, **k))
    ed._on_loaded('synthetic_A', _loaded(_fcs(
        tmp, 'a', names=['FL1-A', 'FL1-A', 'FL2-A'])))
    ed._on_loaded('synthetic_B', _loaded(_fcs(
        tmp, 'b', extra={'$TOT': '600'})))
    settle(root, 1.5)
    notes = _notes(shown)
    assert len(notes) == 1
    assert "synthetic_A: duplicate channel name 'FL1-A'" in notes[0]
    assert 'synthetic_B: $TOT says 600 events' in notes[0]
    assert len(kept) == 2                         # kept in the history


def test_clean_samples_raise_no_load_notes(env):
    root, tmp, shown = env
    ed = _new_editor(root)
    ed._on_loaded('synthetic_A', _loaded(_fcs(tmp, 'a')))
    settle(root, 1.5)
    assert _notes(shown) == []


def _two_gates(ed, root):
    """Two gates, each its own undo step (one Tk event coalesces)."""
    ed._set_active_sample('synthetic_A')
    ids = []
    for name, v in (('first', 0.3), ('second', 0.6)):
        root.update()
        ids.append(ed._add_gate(
            {'kind': 'threshold', 'channel': 'FL1-A', 'value': v,
             'parent_id': None, 'name': name, 'enabled': True},
            replace_1d=False))
    root.update()
    return ids


def _gate_names(ed):
    return sorted(g.get('name') for g in ed._sample_gates['synthetic_A']
                  .values())


def test_undo_after_a_transform_change_says_so_and_keeps_the_gate(env):
    root, tmp, _ = env
    ed = _new_editor(root)
    ed._on_loaded('synthetic_A', _loaded(_fcs(tmp, 'a')))
    _two_gates(ed, root)
    ed._apply_channel_transforms({'FL1-A': 'asinh'})
    root.update()
    ed._undo()
    assert 'cannot reverse the transform change of FL1-A' in \
        ed.status_var.get()
    assert _gate_names(ed) == ['first', 'second']
    ed._undo()
    assert _gate_names(ed) == ['first']


def test_undo_without_such_a_change_undoes_the_gate_at_once(env):
    root, tmp, _ = env
    ed = _new_editor(root)
    ed._on_loaded('synthetic_A', _loaded(_fcs(tmp, 'a')))
    _two_gates(ed, root)
    ed._undo()
    assert _gate_names(ed) == ['first']
    assert 'cannot reverse' not in ed.status_var.get()


def test_undo_never_brings_back_a_removed_samples_gates(env):
    root, tmp, _ = env
    ed = _new_editor(root)
    ed._on_loaded('synthetic_A', _loaded(_fcs(tmp, 'a')))
    ed._on_loaded('synthetic_B', _loaded(_fcs(tmp, 'b')))
    _two_gates(ed, root)
    ed._remove_samples(['synthetic_B'])
    ed._undo()
    assert 'synthetic_B' not in ed._sample_gates
    assert _gate_names(ed) == ['first']


# ── item 15: exported FCS keywords ───────────────────────────────────────


def _source_sample(tmp, compensated):
    from openflo.pipeline import FlowSample
    kw = {'$SRC': 'synthetic source', '$OP': 'operator one',
          '$FIL': 'synthetic_A.fcs', 'TUBE NAME': 'tube one',
          'WELL ID': 'A01', '$CYT': 'synthetic cytometer',
          'APPLY COMPENSATION': 'TRUE',
          '$P1S': 'M1', '$P2S': 'M2', '$P3S': 'M3'}
    if compensated:
        kw['$SPILLOVER'] = '3,' + ','.join(NAMES) + ',1,0.1,0,0,1,0,0,0,1'
    s = FlowSample(write_fcs(str(tmp / 'kw.fcs'),
                             np.random.default_rng(1).normal(
                                 5000, 300, (300, 3)).clip(0),
                             NAMES, datatype='F', extra=kw))
    s.auto_compensate()
    return s


@pytest.mark.parametrize('compensated', [True, False])
def test_exported_fcs_keeps_the_source_keywords(tmp_path, compensated):
    from openflo.fcs_export import source_keywords
    from openflo.fcs_export import write_fcs as export
    s = _source_sample(tmp_path, compensated)
    keywords, _ = source_keywords(s)
    out = str(tmp_path / 'out.fcs')
    export(s.data[NAMES], out, channel_labels={'FL1-A': 'M1',
                                               'FL2-A': 'FL2-A'},
           keywords=keywords)
    text = flowio.FlowData(out).text
    assert text['src'] == 'synthetic source'
    assert text['op'] == 'operator one'
    assert text['tube name'] == 'tube one'
    assert text['well id'] == 'A01'
    assert text['openflo_source_fil'] == 'synthetic_A.fcs'
    assert text.get('p1s') == 'M1'
    assert not text.get('p2s')                  # a label that is its name
    # Nothing another reader could take for a matrix still to apply.
    assert not [k for k in text if 'spill' in k or 'comp' in k]
    if compensated:
        assert text['openflo_applied_matrix'].startswith('3,FL1-A,FL2-A,FL3-A,')
        assert 'openflo_matrix_source' in text
    else:
        assert 'openflo_applied_matrix' not in text
        assert 'openflo_matrix_source' not in text
