"""Typed gate coordinates: a gate's bounds or vertices typed in axis units
(intensity), stored in the channel's own coordinates, with undo.

Generated data on generic channels. FL2-A is stored as logicle (as the
loader stores fluorescence), so a cut typed at an intensity of 1,000 must be
stored as logicle(1,000) and must select exactly the events whose linear
value is at or above 1,000.
"""
import math
import os
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openflo import gate_coords as gc
from openflo.pipeline import gate_to_mask, transform_spec, transform_values
from tests.conftest import gui_unavailable

LOG = transform_spec('logicle')
TR = {'FL2-A': LOG, 'FL3-A': LOG}


def _logicle(v):
    return float(np.asarray(transform_values(np.array([float(v)]),
                                             'logicle')).ravel()[0])


def test_parse_number():
    assert gc.parse_number('1500') == 1500
    assert gc.parse_number(' 1,500 ') == 1500
    assert gc.parse_number('1.5K') == 1500
    assert gc.parse_number('2m') == 2e6
    assert gc.parse_number('-200') == -200
    assert gc.parse_number('−200') == -200
    assert gc.parse_number('2e3') == 2000
    assert gc.parse_number('.5') == 0.5
    assert gc.parse_number('') is None
    for bad in ('abc', '1..2', '1K5', 'nan', 'inf', '1e999'):
        with pytest.raises(ValueError):
            gc.parse_number(bad)


GATES = {
    'threshold': {'kind': 'threshold', 'channel': 'FL2-A',
                  'value': _logicle(987.6)},
    'interval': {'kind': 'interval', 'channel': 'FL2-A',
                 'lo': _logicle(100.0), 'hi': 1e12},
    'rect': {'kind': 'rect', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
             'x0': 1234.5, 'x1': 1e12, 'y0': -1e12, 'y1': _logicle(5000.0)},
    'polygon': {'kind': 'polygon', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
                'vertices': [[10.123456789, 0.2], [500.5, 0.31], [250.0, 0.9]]},
    'ellipsoid': {'kind': 'ellipsoid', 'x_channel': 'FL1-A',
                  'y_channel': 'FL2-A', 'mean': [100.0, 0.5],
                  'cov': [[400.0, 1.0], [1.0, 0.01]], 'distance_sq': 4.0},
    'quadrant': {'kind': 'rect', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
                 'x0': 300.0, 'x1': 1e12, 'y0': 0.4, 'y1': 1e12,
                 'label': 'Q++ (x>, y>)', 'quad_set': 'qs1',
                 'quad_origin_x': 300.0, 'quad_origin_y': 0.4},
}


@pytest.mark.parametrize('kind', list(GATES))
def test_ok_without_typing_changes_nothing(kind):
    g = GATES[kind]
    typed = {k: t for k, _lab, t in gc.fields(g, TR)}
    assert gc.apply(g, typed, TR) == g


def test_fields_say_intensity_and_open_sides():
    f = {k: t for k, _l, t in gc.fields(GATES['threshold'], TR)}
    assert float(f['value']) == pytest.approx(987.6, rel=1e-5)
    f = {k: t for k, _l, t in gc.fields(GATES['rect'], TR)}
    assert f == {'x0': '1234.5', 'x1': '', 'y0': '', 'y1': '5000'}
    f = {k: t for k, _l, t in gc.fields(GATES['quadrant'], TR)}
    assert f['quad_origin_x'] == '300'
    f = {k: t for k, _l, t in gc.fields(GATES['polygon'], TR)}
    assert f['vertices'].splitlines()[0].startswith('10.1235, ')


def test_typed_intensity_selects_exactly_those_events():
    rng = np.random.default_rng(3)
    lin = np.concatenate([rng.uniform(-200, 5000, 3000), [1000.0] * 7])
    df = pd.DataFrame({'FL2-A': np.asarray(transform_values(lin, 'logicle'))})
    g = gc.apply(GATES['threshold'], {'value': '1K'}, TR)
    assert g['value'] == pytest.approx(_logicle(1000))
    m = gate_to_mask(g, df)
    assert int(m.sum()) == int((lin >= 1000 - 1e-6).sum())
    # A linear channel stores what is typed.
    g = gc.apply({'kind': 'threshold', 'channel': 'FL1-A', 'value': 1.0},
                 {'value': '250'}, TR)
    assert g['value'] == 250.0


def test_rect_interval_and_validation():
    g = gc.apply(GATES['rect'], {'x0': '100', 'x1': '200', 'y0': '',
                                 'y1': '1e3'}, TR)
    assert (g['x0'], g['x1'], g['y0']) == (100.0, 200.0, -1e12)
    assert g['y1'] == pytest.approx(_logicle(1000))
    g = gc.apply(GATES['interval'], {'lo': '', 'hi': '2.5K'}, TR)
    assert g['lo'] == -1e12 and g['hi'] == pytest.approx(_logicle(2500))
    with pytest.raises(ValueError, match="below"):
        gc.apply(GATES['rect'], {'x0': '300', 'x1': '200'}, TR)
    with pytest.raises(ValueError, match="below"):
        gc.apply(GATES['interval'], {'lo': '3K', 'hi': '2K'}, TR)
    with pytest.raises(ValueError, match='needs a value'):
        gc.apply(GATES['threshold'], {'value': ' '}, TR)
    with pytest.raises(ValueError, match='not a number'):
        gc.apply(GATES['threshold'], {'value': 'high'}, TR)


def test_polygon_typed_vertices():
    g = gc.apply(GATES['polygon'], {'vertices': '0, 0\n100 1000\n\n50;5K\n'},
                 TR)
    assert g['vertices'][0] == [0.0, pytest.approx(_logicle(0))]
    assert g['vertices'][1] == [100.0, pytest.approx(_logicle(1000))]
    assert g['vertices'][2] == [50.0, pytest.approx(_logicle(5000))]
    with pytest.raises(ValueError, match='three vertices'):
        gc.apply(GATES['polygon'], {'vertices': '0, 0\n1, 1'}, TR)
    with pytest.raises(ValueError, match="line 2"):
        gc.apply(GATES['polygon'], {'vertices': '0, 0\n1\n2, 2'}, TR)


def test_ellipse_centre_and_shape():
    e = GATES['ellipsoid']
    cx, cy, a, b, ang = gc.ellipse_params(e['mean'], e['cov'], 4.0)
    cov = gc.ellipse_cov(a, b, ang, 4.0)
    assert np.allclose(cov, e['cov'])
    g = gc.apply(e, {'cx': '150'}, TR)
    assert g['mean'][0] == 150.0 and g['cov'] == e['cov']
    g = gc.apply(e, {'a': '10', 'b': '0.05', 'angle': '0'}, TR)
    assert np.allclose(g['cov'], [[25.0, 0], [0, 0.000625]])
    with pytest.raises(ValueError, match='above 0'):
        gc.apply(e, {'a': '0'}, TR)
    assert math.isclose(gc.ellipse_params(g['mean'], g['cov'], 4.0)[2], 10.0)


def test_unknown_scale_types_stored_values():
    tr = {'FL2-A': dict(LOG, method='unknown')}
    g = {'kind': 'threshold', 'channel': 'FL2-A', 'value': 0.45}
    assert gc.unknown_channels(g, tr) == ['FL2-A']
    assert gc.fields(g, tr)[0][2] == '0.45'
    assert gc.apply(g, {'value': '0.5'}, tr)['value'] == 0.5
    assert not gc.editable(dict(g, scale_unknown=True))
    assert not gc.editable({'kind': 'boolean'})


def test_set_quadrant_origin_moves_every_member():
    gates = {}
    for lab, x0, x1, y0, y1 in (('Q++', 3, 1e12, 4, 1e12), ('Q+-', 3, 1e12, -1e12, 4),
                                ('Q-+', -1e12, 3, 4, 1e12), ('Q--', -1e12, 3, -1e12, 4)):
        gates[lab] = {'kind': 'rect', 'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1,
                      'label': lab, 'quad_set': 'q', 'quad_origin_x': 3,
                      'quad_origin_y': 4}
    assert gc.set_quadrant_origin(gates, 'q', 5, 6) == 4
    assert (gates['Q++']['x0'], gates['Q++']['y0']) == (5, 6)
    assert (gates['Q+-']['x0'], gates['Q+-']['y1']) == (5, 6)
    assert (gates['Q-+']['x1'], gates['Q-+']['y0']) == (5, 6)
    assert (gates['Q--']['x1'], gates['Q--']['y1']) == (5, 6)
    assert gates['Q--']['x0'] == -1e12


# ── In the editor ────────────────────────────────────────────────────────────

def _editor_or_skip():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        gui_unavailable(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    import importlib
    gui = importlib.import_module('openflo.gui')
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    return root, ed


def _load(ed, name, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'FL1-A': rng.uniform(0, 1000, 500),
                       'FL2-A': np.asarray(transform_values(
                           rng.uniform(-100, 5000, 500), 'logicle'))})
    ed._samples[name] = SimpleNamespace(
        name=name, path=f'{name}.fcs', data=df, data_transforms={'FL2-A': LOG},
        fluor_channels=['FL2-A'], channel_labels={c: c for c in df.columns})
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    if 'T' not in ed._trial_order:
        ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = True
    ed._channels = list(df.columns)
    ed._channel_labels = {c: c for c in df.columns}
    ed._populate_channel_combos()


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def test_editor_sets_coordinates_on_every_sample_with_undo():
    root, ed = _editor_or_skip()
    try:
        _load(ed, 's1', 1)
        _load(ed, 's2', 2)
        gid = {}
        for n in ('s2', 's1'):
            ed._set_active_sample(n)
            gid[n] = ed._add_gate({'kind': 'threshold', 'channel': 'FL2-A',
                                   'value': _logicle(400 if n == 's1' else 600),
                                   'name': 'Pos', 'parent_id': None})
        ed._clear_undo_pending()
        n = ed._apply_gate_coordinates('s1', gid['s1'], {'value': '2K'},
                                       every_sample=True)
        assert n == 2
        for s in ('s1', 's2'):
            assert ed._sample_gates[s][gid[s]]['value'] == pytest.approx(_logicle(2000))
            assert ed._sample_gates[s][gid[s]]['name'] == 'Pos'
        ed._undo()
        assert ed._sample_gates['s1'][gid['s1']]['value'] == pytest.approx(_logicle(400))
        assert ed._sample_gates['s2'][gid['s2']]['value'] == pytest.approx(_logicle(600))
        # Only this sample; bad input changes nothing.
        ed._clear_undo_pending()
        assert ed._apply_gate_coordinates('s1', gid['s1'], {'value': '3K'}) == 1
        assert ed._sample_gates['s2'][gid['s2']]['value'] == pytest.approx(_logicle(600))
        with pytest.raises(ValueError):
            ed._apply_gate_coordinates('s1', gid['s1'], {'value': 'x'})
        assert ed._sample_gates['s1'][gid['s1']]['value'] == pytest.approx(_logicle(3000))
    finally:
        root.destroy()


def test_editor_dialog_and_quadrant():
    root, ed = _editor_or_skip()
    try:
        _load(ed, 's1', 3)
        ed._set_active_sample('s1')
        ids = []
        for lab, x0, x1, y0, y1 in (('Q++ (x>, y>)', 300, 1e12, 0.4, 1e12),
                                    ('Q+- (x>, y<)', 300, 1e12, -1e12, 0.4),
                                    ('Q-+ (x<, y>)', -1e12, 300, 0.4, 1e12),
                                    ('Q-- (x<, y<)', -1e12, 300, -1e12, 0.4)):
            ids.append(ed._add_gate({
                'kind': 'rect', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
                'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1, 'label': lab,
                'quad_set': 'qs1', 'quad_origin_x': 300.0,
                'quad_origin_y': 0.4, 'parent_id': None}))
        ed._clear_undo_pending()
        dlg = ed._edit_gate_coordinates('s1', ids[3])
        assert dlg.title().startswith('Edit coordinates')
        ents = [w for w in _walk(dlg) if isinstance(w, ttk.Entry)]
        assert len(ents) == 2
        ents[0].delete(0, 'end')
        ents[0].insert(0, '500')
        (ok,) = [w for w in _walk(dlg) if isinstance(w, ttk.Button)
                 and w.cget('text') == 'OK']
        ok.invoke()
        g = ed._sample_gates['s1']
        assert g[ids[0]]['x0'] == 500.0 and g[ids[3]]['x1'] == 500.0
        assert g[ids[0]]['y0'] == pytest.approx(0.4)
        assert 'set on 1 sample' in ed.status_var.get()
        # A boolean gate has no coordinates: no dialog.
        bid = ed._add_gate({'kind': 'boolean', 'op': 'not',
                            'operands': [ids[0]], 'parent_id': None})
        assert ed._edit_gate_coordinates('s1', bid) is None
    finally:
        root.destroy()


def test_gate_menu_offers_the_new_tools(monkeypatch):
    """A gate's right-click menu offers Edit coordinates… (gates with
    coordinates), Re-place on each sample… (gates that can re-fit) and
    Combination gates…; the Edit menu offers all three."""
    from openflo.tree_ids import gate_iid
    root, ed = _editor_or_skip()
    try:
        _load(ed, 's1', 4)
        ed._set_active_sample('s1')
        thr = ed._add_gate({'kind': 'threshold', 'channel': 'FL2-A',
                            'value': 0.5, 'parent_id': None})
        rect = ed._add_gate({'kind': 'rect', 'x_channel': 'FL1-A',
                             'y_channel': 'FL2-A', 'x0': 0, 'x1': 1, 'y0': 0,
                             'y1': 1, 'parent_id': None})
        boo = ed._add_gate({'kind': 'boolean', 'op': 'not',
                            'operands': [thr], 'parent_id': None})
        ed._refresh_gate_list()
        shown = {}

        def popup(menu, *_a):
            shown['labels'] = [menu.entrycget(i, 'label')
                               for i in range(menu.index('end') + 1)
                               if menu.type(i) == 'command']
        monkeypatch.setattr(tk.Menu, 'tk_popup', popup)
        ev = SimpleNamespace(y=0, x_root=0, y_root=0)
        for gid, want in ((thr, {'Edit coordinates…', 'Re-place on each sample…'}),
                          (rect, {'Edit coordinates…'}), (boo, set())):
            monkeypatch.setattr(ed.gate_tv, 'identify_row',
                                lambda _y, _g=gid: gate_iid('s1', _g))
            ed._on_right_click(ev)
            labels = set(shown['labels'])
            assert 'Combination gates…' in labels
            assert labels & {'Edit coordinates…', 'Re-place on each sample…'} == want
        edit = [w for w in _walk(ed) if isinstance(w, tk.Menu)
                and any(w.type(i) == 'command'
                        and w.entrycget(i, 'label') == 'Combination gates…'
                        for i in range((w.index('end') or 0) + 1))]
        labels = {edit[0].entrycget(i, 'label')
                  for i in range(edit[0].index('end') + 1)
                  if edit[0].type(i) == 'command'}
        assert {'Combination gates…', 'Edit gate coordinates…',
                'Re-place gate on each sample…'} <= labels
    finally:
        root.destroy()
