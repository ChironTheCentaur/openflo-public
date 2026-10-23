"""File → Compare with saved session: each population counted with this
session's gates and with a saved session's gates, on the same events.

Generated events: FSC-A = SSC-A = 0..999 and FL1-A = 0..999 (linear), so a
rectangle [x0, x1) on FSC-A holds exactly x1 - x0 events and a threshold at
v holds 1000 - ceil(v)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from openflo.session_compare import COLUMNS, compare_sessions

FRAME = pd.DataFrame({c: np.arange(1000, dtype=float)
                      for c in ('FSC-A', 'SSC-A', 'FL1-A')})


def _rect(x1, name='Cells'):
    return {'kind': 'rect', 'name': name, 'parent_id': None,
            'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
            'x0': 100.0, 'x1': x1, 'y0': 0.0, 'y1': 1e6}


def _pos(value, parent='g1'):
    return {'kind': 'threshold', 'name': 'Pos', 'parent_id': parent,
            'channel': 'FL1-A', 'value': value}


def _saved(gates, transforms=None):
    entry = {'name': 's1'}
    if transforms:
        entry['transforms'] = transforms
    return {'samples': [entry],
            'sample_gates': {'s1': [dict(g, id=gid)
                                    for gid, g in gates.items()]}}


def _rows(now, saved, rec=None):
    rows = compare_sessions({'s1': (now, FRAME, rec)}, saved)
    assert all(set(COLUMNS) <= set(r) for r in rows)
    return {r['Population']: r for r in rows}


def test_identical_sessions_match():
    gates = {'g1': _rect(600.0), 'g2': _pos(800.0)}
    rows = _rows(gates, _saved(gates))
    assert {p: (r['Count now'], r['Count saved'], r['moved'])
            for p, r in rows.items()} == {
        'Cells': (500, 500, False), 'Cells/Pos': (0, 0, False)}


def test_changed_gate_and_moved_counts():
    now = {'g1': _rect(600.0), 'g2': _pos(300.0)}
    old = {'a': _rect(500.0), 'b': _pos(300.0, parent='a')}
    rows = _rows(now, _saved(old))
    cells, pos = rows['Cells'], rows['Cells/Pos']
    assert (cells['Count now'], cells['Count saved'], cells['Change']) == \
        (500, 400, 100)
    assert cells['% change'] == 25.0
    assert cells['Note'] == 'gate changed'
    # Pos (FL1-A >= 300) inside Cells: 300..599 now, 300..499 saved.
    assert (pos['Count now'], pos['Count saved']) == (300, 200)
    assert pos['%Parent now'] == 60.0 and pos['%Parent saved'] == 50.0
    # Its own cut is the same, but its parent's is not.
    assert pos['moved'] and pos['Note'] == 'gate changed'


def test_population_on_one_side_only():
    rows = _rows({'g1': _rect(600.0)},
                 _saved({'a': _rect(600.0), 'b': _pos(10.0, parent='a')}))
    assert rows['Cells/Pos']['Note'] == 'only in the saved session'
    assert rows['Cells/Pos']['Count now'] is None
    assert rows['Cells/Pos']['Count saved'] == 500     # all of 100..599
    assert rows['Cells/Pos']['moved']
    assert not rows['Cells']['moved']


def test_saved_gate_on_another_scale_is_converted():
    # Saved with FL1-A on asinh (cofactor 150): its stored 1.0 is the linear
    # 150·sinh(1) = 176.28, which on this session's linear data keeps
    # FL1-A >= 176.28 → 177..599 inside Cells, 423 events.
    old = {'a': _rect(600.0), 'b': _pos(1.0, parent='a')}
    rows = _rows({'g1': _rect(600.0), 'g2': _pos(150 * math.sinh(1.0))},
                 _saved(old, {'FL1-A': {'method': 'asinh',
                                        'cofactor': 150.0}}),
                 rec={})
    assert rows['Cells/Pos']['Count saved'] == 423
    assert rows['Cells/Pos']['Count now'] == 423


def test_unknown_saved_scale_reads_na():
    old = {'a': _rect(600.0), 'b': _pos(1.0, parent='a')}
    rows = _rows({'g1': _rect(600.0), 'g2': _pos(500.0)},
                 _saved(old, {'FL1-A': {'method': 'unknown'}}), rec={})
    assert rows['Cells/Pos']['Count saved'] is None
    assert 'unknown scale' in rows['Cells/Pos']['Note']


def test_sample_missing_from_saved_session_is_skipped():
    assert compare_sessions({'other': ({'g1': _rect(600.0)}, FRAME, None)},
                            _saved({'a': _rect(600.0)})) == []


def test_file_menu_compares_with_a_saved_file(tmp_path, monkeypatch):
    import json

    from tests.test_layout_editor import _editor_or_skip, _load
    root, ed = _editor_or_skip()
    try:
        import openflo.async_task as at
        import openflo.editor_session as es
        from openflo.ui_table import TableWindow

        def inline(_w, work, on_done=None, on_error=None, on_finally=None):
            try:
                res = work()
            except Exception as exc:                  # noqa: BLE001
                on_error(exc)
            else:
                on_done(res)
        monkeypatch.setattr(at, 'run_async', inline)
        _load(ed, 's1', 1, 500, 300.0)
        path = tmp_path / 'before.flowsession'
        path.write_text(json.dumps(ed._session_state()), encoding='utf-8')
        # Re-gate: Pos moves from 300 to 600 after the save.
        for g in ed._sample_gates['s1'].values():
            if g.get('name') == 'Pos':
                g['value'] = 600.0
        monkeypatch.setattr(es.filedialog, 'askopenfilename',
                            lambda *a, **k: str(path))
        ed._compare_saved_session()
        (win,) = [w for w in ed.winfo_children()
                  if isinstance(w, TableWindow)]
        rows = {r['Population']: r for r in win._rows}
        assert not rows['Cells']['moved']
        pos = rows['Cells/Pos']
        data = ed._samples['s1'].data
        cells = data[(data['FSC-A'] >= 100) & (data['FSC-A'] < 900)
                     & (data['SSC-A'] >= 100) & (data['SSC-A'] < 900)]
        assert pos['Count now'] == int((cells['FL1-A'] >= 600).sum())
        assert pos['Count saved'] == int((cells['FL1-A'] >= 300).sum())
        assert pos['moved'] and pos['Note'] == 'gate changed'
        assert '1 differ' in ed.status_var.get()
    finally:
        root.destroy()
