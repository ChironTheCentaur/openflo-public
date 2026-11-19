"""A FlowJo .wsp or Gating-ML template retargets its gates by marker.

A JSON template stamps each gate with the antibody its channels stood for, so
applied to a sample whose panel puts that marker on another detector the gate
reads the marker, not the detector. .wsp and Gating-ML templates skipped
that: their gates stayed on the source's detectors and gated the wrong marker
(scope audit, G1 review N1). Worse, the 1-D retargeting read a gate's
``label``, which for a .wsp or Gating-ML gate is the POPULATION's name: a
population named like a marker moved to that marker's detector.

Every sample here is generated: two panels with the same two markers swapped
between FL1-A and FL2-A, and FL3-A unlabelled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import openflo.pipeline as fp
from openflo import gate_templates
from openflo.label_alignment import relabel_gate_for_sample
from tests._fcs_writer import write_float_fcs

CH = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A']
SOURCE_PANEL = ['', '', 'CD901b', 'CD902', '']      # $PnS of the source
TARGET_PANEL = ['', '', 'CD902', 'CD901b', '']      # markers swapped


def _events(n=6000, seed=0):
    """Linear events: CD901b bright on 25% of events, CD902 on 75%, FL3-A
    bright on 50%, so a gate reading the wrong channel counts another number."""
    rng = np.random.default_rng(seed)

    def bright(frac):
        hi = rng.random(n) < frac
        return np.where(hi, rng.lognormal(np.log(2e4), 0.3, n),
                        rng.normal(0, 200, n))
    return {'FSC-A': rng.uniform(2e4, 2e5, n), 'SSC-A': rng.uniform(1e4, 1e5, n),
            'CD901b': bright(0.25), 'CD902': bright(0.75), 'FL3-A': bright(0.5)}


def _write_fcs(path, panel, seed):
    ev = _events(seed=seed)
    cols = []
    for ch, marker in zip(CH, panel, strict=True):
        cols.append(ev[marker] if marker else ev[ch])
    write_float_fcs(path, np.column_stack(cols), CH,
                    opt_channel_names=[m or c for m, c in zip(panel, CH, strict=True)])
    return str(path), pd.DataFrame(np.column_stack(cols), columns=CH)


# Source gates in linear units (a .wsp holds FlowJo's), with population names.
SOURCE_GATES = [
    {'id': 'pos', 'parent_id': None, 'kind': 'threshold', 'label': 'CD901b+',
     'channel': 'FL1-A', 'value': 3000.0},
    # Named like a marker, on an unlabelled channel: must stay on FL3-A.
    {'id': 'pop', 'parent_id': None, 'kind': 'threshold', 'label': 'CD902',
     'channel': 'FL3-A', 'value': 3000.0},
    {'id': 'box', 'parent_id': None, 'kind': 'rect', 'label': 'Box',
     'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
     'x0': 3000.0, 'x1': 1e12, 'y0': -1e12, 'y1': 3000.0},
]


def _apply(tmp_path, monkeypatch, template, target_fcs, fl2_linear=False):
    """Load the target FCS into an editor, apply `template` (overwrite) and
    return (editor, root, sample name, {population: gate}, warnings).
    ``fl2_linear`` keeps the target's FL2-A stored linear (FL1-A and FL3-A on
    logicle), so a gate put on the scale of the channel it came from, not
    the one it now reads, counts wrong."""
    from tests.test_gui_smoke import _editor_or_skip
    root, ed, _gui = _editor_or_skip()
    s = fp.FlowSample(target_fcs)
    if fl2_linear:
        s.apply_transform(channels=['FL1-A', 'FL3-A'])
    else:
        s.apply_transform()
    name = ed._sample_name_for(target_fcs)
    ed._on_loaded(name, s)
    ed._set_active_sample(name)
    monkeypatch.setattr(ed, '_ask_template_apply',
                        lambda **_kw: ([name], True))
    warned = []
    import openflo.editor_template as et
    monkeypatch.setattr(et.messagebox, 'showwarning',
                        lambda title, msg, **_k: warned.append((title, msg)))
    ed._apply_template_path(template)
    gates = {(g.get('label') or g.get('name')): g
             for g in ed._sample_gates[name].values()}
    return ed, root, name, gates, warned


def _count(g, sample):
    return int(fp.gate_to_mask(g, sample.data).sum())


@pytest.fixture
def panels(tmp_path):
    src, _src_df = _write_fcs(tmp_path / 'src.fcs', SOURCE_PANEL, seed=1)
    tgt, tgt_df = _write_fcs(tmp_path / 'tgt.fcs', TARGET_PANEL, seed=2)
    return src, tgt, tgt_df


def test_wsp_template_reads_its_markers_from_the_source_keywords(tmp_path, panels):
    """From the sample whose gates the template takes, not the first one
    (here a gateless sample of the other panel)."""
    src, tgt, _df = panels
    w = fp.WspWriter()
    w.add_sample('other', tgt, CH, [])
    w.add_sample('src', src, CH, [dict(g) for g in SOURCE_GATES])
    path = str(tmp_path / 't.wsp')
    w.write(path)
    assert gate_templates.wsp_template_labels(path) == {
        'FL1-A': 'CD901b', 'FL2-A': 'CD902'}


def _logicle_axis(ch):
    return ('<transforms:logicle transforms:length="256" transforms:T="262144" '
            'transforms:W="1.0" transforms:M="4.4185" transforms:A="0">'
            f'<data-type:parameter data-type:name="{ch}"/></transforms:logicle>')


# Logicle axes FlowJo declares for the source's FL1-A (CD901b) and FL3-A
# (unlabelled): GeoMeans are taken on them (wsp_gate_coords.wsp_stat_axes).
LOGICLE_FL1_FL3 = (
    '<Transformations xmlns:transforms="http://www.isac-net.org/std/'
    'Gating-ML/v2.0/transformations">' + _logicle_axis('FL1-A')
    + _logicle_axis('FL3-A') + '</Transformations>')


def test_wsp_template_gates_follow_the_marker(tmp_path, monkeypatch, panels):
    src, tgt, tgt_df = panels
    w = fp.WspWriter()
    w.add_sample('src', src, CH, [dict(g) for g in SOURCE_GATES])
    path = tmp_path / 't.wsp'
    w.write(str(path))
    text = path.read_text(encoding='utf-8')
    assert text.count('<Transformations />') == 1
    path.write_text(text.replace('<Transformations />', LOGICLE_FL1_FL3),
                    encoding='utf-8')
    path = str(path)
    assert set(gate_templates.wsp_template_stat_axes(path)) == {'FL1-A', 'FL3-A'}
    # The target's FL2-A (CD901b) is stored linear, its FL1-A logicle: the
    # gate must go onto FL2-A's scale.
    ed, root, name, gates, warned = _apply(tmp_path, monkeypatch, path, tgt,
                                           fl2_linear=True)
    try:
        # CD901b sits on FL2-A in the target: the gate reads it there.
        assert gates['CD901b+']['channel'] == 'FL2-A'
        assert (gates['Box']['x_channel'], gates['Box']['y_channel']) == (
            'FL2-A', 'FL1-A')
        # A population NAMED like a marker is not a marker stamp.
        assert gates['CD902']['channel'] == 'FL3-A'
        smp = ed._samples[name]
        want_pos = int((tgt_df['FL2-A'] >= 3000.0).sum())
        assert abs(_count(gates['CD901b+'], smp) - want_pos) <= 0.01 * len(tgt_df)
        want_box = int(((tgt_df['FL2-A'] >= 3000.0)
                        & (tgt_df['FL1-A'] <= 3000.0)).sum())
        assert abs(_count(gates['Box'], smp) - want_box) <= 0.01 * len(tgt_df)
        assert not [t for t, _m in warned if 'missing' in t]
        # CD901b's axis goes with it, onto FL2-A; FL3-A's stays.
        assert set(smp.stat_axes) == {'FL2-A', 'FL3-A'}
    finally:
        root.destroy()


def test_gatingml_dimension_named_by_marker_reads_that_marker(
        tmp_path, monkeypatch, panels):
    """Gating-ML names dimensions only. One named by a marker the target
    labels (no such detector there) reads the detector carrying it, on that
    channel's scale, and is not reported missing."""
    from openflo.gatingml import write_gatingml
    _src, tgt, tgt_df = panels
    tree = [{'id': 'pos', 'parent_id': None, 'kind': 'threshold',
             'name': 'Bright', 'channel': 'CD901b', 'value': 3000.0},
            {'id': 'pop', 'parent_id': None, 'kind': 'threshold',
             'name': 'CD902', 'channel': 'FL3-A', 'value': 3000.0}]
    path = str(tmp_path / 't.xml')
    write_gatingml(path, tree, {})          # linear values in the file
    ed, root, name, gates, warned = _apply(tmp_path, monkeypatch, path, tgt)
    try:
        assert gates['Bright']['channel'] == 'FL2-A'
        assert gates['CD902']['channel'] == 'FL3-A'
        smp = ed._samples[name]
        want = int((tgt_df['FL2-A'] >= 3000.0).sum())
        assert abs(_count(gates['Bright'], smp) - want) <= 0.01 * len(tgt_df)
        assert not [t for t, _m in warned if 'missing' in t]
    finally:
        root.destroy()


def test_saved_template_stamps_a_1d_gate_without_renaming_it(
        tmp_path, monkeypatch, panels):
    """A JSON template stamps a 1-D gate's marker in 'channel_label', so a
    population's 'label' (its name) survives, and the stamp retargets."""
    import json
    import tkinter.filedialog as fdlg
    src, tgt, _df = panels
    from tests.test_gui_smoke import _editor_or_skip
    root, ed, _gui = _editor_or_skip()
    try:
        s = fp.FlowSample(src)
        s.apply_transform()
        name = ed._sample_name_for(src)
        ed._on_loaded(name, s)
        ed._set_active_sample(name)
        # A stamp left from an earlier template gives way to the channel's
        # label now.
        ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A', 'value': 0.5,
                      'label': 'Bright cells', 'channel_label': 'CD999',
                      'parent_id': None})
        # Unlabelled channel, population named like a marker (as a .wsp
        # import names it).
        ed._add_gate({'kind': 'threshold', 'channel': 'FL3-A', 'value': 0.5,
                      'label': 'CD902', 'parent_id': None})
        out = tmp_path / 't.json'
        monkeypatch.setattr(fdlg, 'asksaveasfilename', lambda *a, **k: str(out))
        ed._save_template()
    finally:
        root.destroy()
    saved = {g['label']: g for g in
             json.loads(out.read_text(encoding='utf-8'))['gates']}
    assert saved['Bright cells']['channel_label'] == 'CD901b'
    assert 'channel_label' in saved['CD902']
    assert saved['CD902']['channel_label'] is None
    ed, root, name, gates, _w = _apply(tmp_path, monkeypatch, str(out), tgt)
    try:
        assert gates['Bright cells']['channel'] == 'FL2-A'
        assert gates['CD902']['channel'] == 'FL3-A'
    finally:
        root.destroy()


def test_an_older_json_template_still_retargets_by_its_label(
        tmp_path, monkeypatch, panels):
    """Templates saved before 'channel_label' stamped a 1-D gate's marker in
    'label': it still retargets them."""
    import json
    _src, tgt, _df = panels
    out = tmp_path / 'old.json'
    out.write_text(json.dumps({'version': 2, 'gates': [
        {'id': 'g1', 'parent_id': None, 'kind': 'threshold', 'name': 'Pos',
         'channel': 'FL1-A', 'label': 'CD901b', 'value': 0.5}]}),
        encoding='utf-8')
    _ed, root, _name, gates, _w = _apply(tmp_path, monkeypatch, str(out), tgt)
    try:
        assert gates['CD901b']['channel'] == 'FL2-A'
    finally:
        root.destroy()


def test_installing_wsp_gates_directly_reads_label_as_a_name(panels):
    """_apply_template_to_sample called without retargeted gates retargets
    them itself, with a .wsp gate's 'label' as the population's name."""
    _src, tgt, _df = panels
    from tests.test_gui_smoke import _editor_or_skip
    root, ed, _gui = _editor_or_skip()
    try:
        s = fp.FlowSample(tgt)
        s.apply_transform()
        name = ed._sample_name_for(tgt)
        ed._on_loaded(name, s)
        ed._apply_template_to_sample(name, [dict(SOURCE_GATES[1])], True,
                                     from_wsp=True)
        (g,) = ed._sample_gates[name].values()
        assert g['channel'] == 'FL3-A'
    finally:
        root.destroy()


# ── the helpers, directly ─────────────────────────────────────────────────────

def test_relabel_moves_every_channel_reference():
    g = {'kind': 'rect', 'x_channel': 'FL1-A', 'y_channel': 'FL3-A',
         'x_label': 'CD901b',
         '_gml_axes': {'FL1-A': {'a': 1}, 'FL3-A': {'b': 2}},
         'wsp_comp_channels': ['FL1-A', 'FL3-A']}
    out = relabel_gate_for_sample(g, {'CD901b': 'FL2-A', 'FL3-A': 'FL3-A'})
    assert out['x_channel'] == 'FL2-A' and out['y_channel'] == 'FL3-A'
    assert out['_gml_axes'] == {'FL2-A': {'a': 1}, 'FL3-A': {'b': 2}}
    assert out['wsp_comp_channels'] == ['FL2-A', 'FL3-A']
    assert g['x_channel'] == 'FL1-A'              # the template is untouched


def test_relabel_legacy_label_only_when_asked():
    g = {'kind': 'threshold', 'channel': 'FL3-A', 'label': 'CD902'}
    to_det = {'CD902': 'FL1-A', 'FL3-A': 'FL3-A'}
    assert relabel_gate_for_sample(g, to_det)['channel'] == 'FL1-A'
    assert relabel_gate_for_sample(
        g, to_det, legacy_label=False)['channel'] == 'FL3-A'
    stamped = dict(g, channel_label='CD901b')
    assert relabel_gate_for_sample(
        stamped, {'CD901b': 'FL2-A', **to_det})['channel'] == 'FL2-A'


def test_a_detector_name_is_never_read_as_a_marker():
    """A channel the sample has as a detector stays, even when another
    detector carries a label spelled like it."""
    g = {'kind': 'threshold', 'channel': 'FL1-A'}
    assert relabel_gate_for_sample(
        g, {'FL1-A': 'FL2-A', 'CD5': 'FL1-A'})['channel'] == 'FL1-A'


def test_stat_axes_follow_the_moved_channels():
    from openflo.editor_template import _channel_moves, _merge_stat_axes

    class S:
        stat_axes = {}
        data = pd.DataFrame(columns=CH)
    gates = [{'channel': 'FL1-A'}, {'x_channel': 'FL2-A', 'y_channel': 'FL3-A'}]
    moved = [{'channel': 'FL2-A'}, {'x_channel': 'FL1-A', 'y_channel': 'FL3-A'}]
    moves = _channel_moves(gates, moved)
    assert moves == {'FL1-A': 'FL2-A', 'FL2-A': 'FL1-A'}
    smp = S()
    _merge_stat_axes(smp, {'FL1-A': {'k': 1}, 'FL2-A': {'k': 2},
                           'FL3-A': {'k': 3}}, moves)
    assert smp.stat_axes == {'FL2-A': {'k': 1}, 'FL1-A': {'k': 2},
                             'FL3-A': {'k': 3}}


def test_gatingml_notes_name_the_channel_the_gate_now_reads():
    """The 'drawn on uncompensated values' note and the missing-channel count
    look at the gate as the sample reads it (retargeted), not as the file
    names it."""
    from types import SimpleNamespace

    from openflo import editor_template as et
    smp = SimpleNamespace(fluor_channels=['FL1-A', 'FL2-A'],
                          channel_labels={'FL1-A': 'CD902', 'FL2-A': 'CD901b'},
                          comp_channels=['FL1-A', 'FL2-A'],
                          data=pd.DataFrame(columns=CH))

    class Ed:
        _samples = {'t': smp}
        _template_gates_for = et.TemplateMixin._template_gates_for
        _gate_channels = staticmethod(lambda g: {
            g[k] for k in ('channel', 'x_channel', 'y_channel') if g.get(k)})
        _count_channel_mismatches = et.TemplateMixin._count_channel_mismatches

        def _apply_template_to_sample(self, *a, **k):
            return [], []
    gml = [{'kind': 'threshold', 'label': 'Bright', 'channel': 'CD901b',
            'value': 1.0, 'wsp_comp_channels': []}]     # uncompensated
    notes = []
    mism, _lost, _built = et._apply_to_targets(
        Ed(), ['t'], gml, True, 'x.xml', notes, {})
    assert mism == {}
    assert len(notes) == 1 and 'uncompensated FL2-A' in notes[0]


def test_a_moved_axis_wins_over_the_one_already_on_that_channel():
    """FL1-A's gates moved to FL2-A, FL2-A's did not move: FL2-A takes the
    axis of the marker its gates now read (FL1-A's in the workspace)."""
    from openflo.editor_template import _merge_stat_axes

    class S:
        stat_axes = {}
        data = pd.DataFrame(columns=CH)
    smp = S()
    _merge_stat_axes(smp, {'FL1-A': {'k': 1}, 'FL2-A': {'k': 2}},
                     {'FL1-A': 'FL2-A'})
    assert smp.stat_axes == {'FL2-A': {'k': 1}}
