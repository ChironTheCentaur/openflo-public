"""openflo.document.Document: open, gate, count, plot and save without a
window, as the phone app drives it.

Generated FCS files, workspaces and sessions only; channel names FSC-A,
SSC-A, FL1-A ... and sample names s1, a, b. No Tk.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from openflo import gating
from openflo.document import Document, DocumentError
from openflo.sample_prep import prepare_fcs_sample
from tests._fcs_writer import write_fcs
from tests._open_fixtures import make_fcs, write_wsp

FL = ['FL1-A', 'FL2-A', 'FL3-A']
CHANNELS = ['FSC-A', 'SSC-A', *FL]
SPILL = np.array([[1.0, 0.10, 0.02],
                  [0.05, 1.0, 0.10],
                  [0.00, 0.03, 1.0]])


def spill_fcs(path, seed=1, n=3000, spill=True):
    """A float FCS with three spilled fluorescence channels and, with
    `spill`, $SPILLOVER (no FlowIO needed to write it)."""
    rng = np.random.default_rng(seed)
    true = np.column_stack([rng.lognormal(7, 1, n), rng.lognormal(6, 1, n),
                            rng.lognormal(5, 1, n)])
    ev = np.column_stack([rng.normal(5e4, 1e4, n), rng.normal(3e4, 8e3, n),
                          true @ SPILL])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    text = '3,' + ','.join(FL) + ',' + ','.join(map(str, SPILL.ravel()))
    return write_fcs(str(path), ev, CHANNELS, datatype='F',
                     extra={'$SPILLOVER': text} if spill else None)


CELLS = {'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
         'x0': 3e4, 'x1': 7e4, 'y0': 1e4, 'y1': 5e4, 'name': 'cells'}


def gated_doc(tmp_path):
    """A document with sample s1 (compensated by its $SPILLOVER) gated
    cells > FL1+ (threshold) and cells > FL2 mid (interval)."""
    doc = Document()
    name = doc.open_fcs(spill_fcs(tmp_path / 'raw' / 's1.fcs'))
    cells = doc.add_gate(name, CELLS)['id']
    pos = doc.add_gate(name, {'kind': 'threshold', 'channel': 'FL1-A',
                              'value': 0.5, 'name': 'FL1+'}, cells)['id']
    mid = doc.add_gate(name, {'kind': 'interval', 'channel': 'FL2-A',
                              'lo': 0.3, 'hi': 0.6, 'name': 'FL2 mid'},
                       cells)['id']
    return doc, name, cells, pos, mid


def _same(a, b):
    """Two stats tables equal, value for value (None for NA)."""
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ── opening and listing ─────────────────────────────────────────────────────

def test_a_file_opens_as_the_editor_prepares_it(tmp_path):
    p = spill_fcs(tmp_path / 's1.fcs')
    doc = Document()
    assert doc.open_fcs(p) == 's1'
    ref = prepare_fcs_sample(p)
    s = doc._samples['s1'].data
    np.testing.assert_array_equal(s.data.to_numpy(), ref.data.to_numpy())
    assert s.data_transforms == ref.data_transforms
    (row,) = doc.samples()
    assert row == {'name': 's1', 'loaded': True, 'path': p, 'events': 3000,
                   'channels': 5, 'gates': 0, 'compensated': True}
    kinds = {c['name']: (c['kind'], c['transform'])
             for c in doc.channels('s1')}
    assert kinds == {'FSC-A': ('scatter', 'linear'),
                     'SSC-A': ('scatter', 'linear'),
                     'FL1-A': ('fluor', 'logicle'),
                     'FL2-A': ('fluor', 'logicle'),
                     'FL3-A': ('fluor', 'logicle')}


def test_a_second_file_of_the_same_name_gets_its_folder(tmp_path):
    doc = Document()
    a = doc.open_fcs(spill_fcs(tmp_path / 'Day 1' / 's1.fcs'))
    b = doc.open_fcs(spill_fcs(tmp_path / 'Day 2' / 's1.fcs'))
    assert (a, b) == ('s1', 's1 [Day 2]')


def test_data_that_feeds_gates_and_stats_stays_float64(tmp_path):
    """Only the density grid drawn on screen is float32: the sample's
    values keep the loader's types (compensated and transformed channels
    float64; an untouched channel as the file stores it, exactly)."""
    doc, name, cells, *_ = gated_doc(tmp_path)
    ref = prepare_fcs_sample(str(tmp_path / 'raw' / 's1.fcs'))
    doc.density(name, 'FL1-A', 'FL2-A', gate=cells)
    doc.stats(name)
    df = doc._samples[name].data.data
    assert df.dtypes.to_dict() == ref.data.dtypes.to_dict()
    assert {str(df[c].dtype) for c in FL} == {'float64'}


# ── the gate tree ───────────────────────────────────────────────────────────

def test_gates_form_a_tree(tmp_path):
    doc, name, cells, pos, mid = gated_doc(tmp_path)
    (root,) = doc.gate_tree(name)
    assert (root['id'], root['name'], root['kind']) == (cells, 'cells',
                                                         'rect')
    assert [(c['id'], c['name'], c['parent_id']) for c in root['children']] \
        == [(pos, 'FL1+', cells), (mid, 'FL2 mid', cells)]


def test_a_second_threshold_on_a_channel_replaces_the_first(tmp_path):
    doc, name, cells, pos, _mid = gated_doc(tmp_path)
    doc._samples[name].gates[pos]['color'] = '#123456'
    got = doc.add_gate(name, {'kind': 'threshold', 'channel': 'FL1-A',
                              'value': 0.7}, cells)
    assert got == {'id': pos, 'replaced': True}
    g = doc.gate(name, pos)
    assert (g['value'], g['color'], g['name']) == (0.7, '#123456', 'FL1+')


def test_an_update_keeps_the_place_and_look_of_the_gate(tmp_path):
    doc, name, cells, _pos, mid = gated_doc(tmp_path)
    doc._samples[name].gates[mid]['color'] = '#abcdef'
    g = doc.update_gate(name, mid, {'kind': 'interval', 'channel': 'FL2-A',
                                    'lo': 0.2, 'hi': 0.7})
    assert (g['lo'], g['parent_id'], g['color'], g['name']) == (
        0.2, cells, '#abcdef', 'FL2 mid')
    assert doc._samples[name].order[-1] == mid


def test_a_gate_cannot_go_under_itself_or_a_missing_parent(tmp_path):
    doc, name, cells, pos, _mid = gated_doc(tmp_path)
    with pytest.raises(DocumentError, match='rest on itself'):
        doc.update_gate(name, cells, dict(CELLS, parent_id=pos))
    with pytest.raises(DocumentError, match='no gate'):
        doc.add_gate(name, CELLS, parent_id='g99')
    with pytest.raises(DocumentError, match="'kind'"):
        doc.add_gate(name, {'channel': 'FL1-A'})


def test_deleting_a_gate_deletes_the_gates_under_it(tmp_path):
    doc, name, cells, pos, mid = gated_doc(tmp_path)
    assert doc.delete_gate(name, cells) == [cells, pos, mid]
    assert doc.gate_tree(name) == []
    # Ids are never handed out twice in a sample.
    assert doc.add_gate(name, CELLS)['id'] == 'g4'


def test_deleting_what_a_boolean_gate_combines_is_named_in_the_desktop_words(
        tmp_path):
    from openflo.gate_text import lost_operand_note
    doc, name, cells, pos, _mid = gated_doc(tmp_path)
    assert doc.lost_operands(name) == {'lost': [], 'built_on': [], 'note': ''}
    other = doc.add_gate(name, {'kind': 'threshold', 'channel': 'SSC-A',
                                'value': 1e4, 'name': 'Big'})['id']
    both = doc.add_gate(name, {'kind': 'boolean', 'op': 'and',
                               'operands': [pos, other], 'name': 'Both'})['id']
    doc.add_gate(name, {'kind': 'boolean', 'op': 'not', 'operands': [both],
                        'name': 'Not both'})
    # An imported one is named by its label, as the gate list names it.
    doc.add_gate(name, {'kind': 'boolean', 'op': 'or', 'operands': [pos],
                        'label': 'Either'})
    assert doc.lost_operands(name)['note'] == ''
    doc.delete_gate(name, cells)
    got = doc.lost_operands(name)
    assert got == {'lost': ['Both', 'Either'], 'built_on': ['Not both'],
                   'note': lost_operand_note(['Both', 'Either'], ['Not both'])}
    assert got['note'] == ("; boolean gate 'Both', 'Either' lost an operand "
                           "and now selects nothing; boolean gate 'Not both', "
                           "built on them, selects nothing too")


def test_gates_put_back_follow_each_other_in_any_order(tmp_path):
    """Undo's restore: a boolean listed before a gate it combines, and a
    child before its parent, follow them to their new ids; an operand still
    in the sample stays; one in neither stays as it was (still lost)."""
    doc, name, cells, pos, mid = gated_doc(tmp_path)
    saved = {i: doc.gate(name, i) for i in (cells, pos, mid)}
    both = doc.add_gate(name, {'kind': 'boolean', 'op': 'and',
                               'operands': [pos, mid], 'name': 'Both'})['id']
    lone = doc.add_gate(name, {'kind': 'boolean', 'op': 'not',
                               'operands': [pos], 'name': 'Lone'})
    saved[both] = doc.gate(name, both)
    saved[lone['id']] = doc.gate(name, lone['id'])
    doc.delete_gate(name, both)
    doc.delete_gate(name, lone['id'])
    doc.delete_gate(name, cells)          # pos and mid go with it
    # The boolean first, then pos's child-of-cells order reversed.
    ids = doc.restore_gates(name, [saved[both], saved[pos], saved[cells]])
    rec = doc._samples[name]
    assert set(ids) == {both, pos, cells}
    assert rec.gates[ids[pos]]['parent_id'] == ids[cells]
    # mid is not back: its operand stays as it was, and Both is lost.
    assert rec.gates[ids[both]]['operands'] == [ids[pos], mid]
    assert doc.lost_operands(name)['lost'] == ['Both']
    # Put back later, Lone combines the pos that is there now.
    got = doc.restore_gates(name, [dict(saved[lone['id']],
                                        operands=[ids[pos]])])
    assert rec.gates[got[lone['id']]]['operands'] == [ids[pos]]
    # Two thresholds on one channel under one parent are two populations:
    # put back together, neither replaces the other.
    pos_now = doc.gate(name, ids[pos])
    pair = [doc.gate(name, ids[cells]), pos_now,
            dict(pos_now, id='twin', value=0.8, name='FL1 hi')]
    doc.delete_gate(name, ids[cells])
    again = doc.restore_gates(name, pair)
    assert len({again[g['id']] for g in pair}) == 3
    assert rec.gates[again['twin']]['value'] == 0.8
    assert rec.gates[again[ids[pos]]]['value'] == pos_now['value']
    # A parent in neither is refused, and nothing changes.
    before = list(rec.order)
    with pytest.raises(DocumentError, match='no gate'):
        doc.restore_gates(name, [dict(saved[mid], id='x')])
    assert rec.order == before


def test_a_boolean_that_lost_an_operand_can_still_be_changed(tmp_path):
    """A boolean keeps naming an operand that was deleted: renaming it, and
    Undo of two deletes in turn (put one back, point the boolean at it while
    the other is still gone), name only what it named already. A gate it
    did not name, missing from the sample, is still refused."""
    doc, name, _cells, pos, mid = gated_doc(tmp_path)
    both = doc.add_gate(name, {'kind': 'boolean', 'op': 'and',
                               'operands': [pos, mid], 'name': 'Both'})['id']
    saved_mid, dep_mid = doc.gate(name, mid), doc.gate(name, both)
    doc.delete_gate(name, mid)
    saved_pos, dep_pos = doc.gate(name, pos), doc.gate(name, both)
    doc.delete_gate(name, pos)
    # A rename of the lost boolean is kept, as on the desktop.
    doc.update_gate(name, both, dict(dep_pos, name='Renamed'))
    assert doc.gate(name, both)['name'] == 'Renamed'
    # Undo the second delete: pos back, Both points at it, mid still gone.
    ids = doc.restore_gates(name, [saved_pos])
    doc.update_gate(name, both, dict(
        dep_pos, operands=[ids.get(o, o) for o in dep_pos['operands']]))
    assert doc.gate(name, both)['operands'] == [ids[pos], mid]
    # Undo the first: mid back, and Both selects again.
    ids2 = doc.restore_gates(name, [saved_mid])
    doc.update_gate(name, both, dict(dep_mid, operands=[
        ids.get(o, ids2.get(o, o)) for o in dep_mid['operands']]))
    assert doc.gate(name, both)['operands'] == [ids[pos], ids2[mid]]
    assert doc.lost_operands(name)['lost'] == []
    # A gate Both never named, missing from the sample, is refused.
    with pytest.raises(DocumentError, match='no gate'):
        doc.update_gate(name, both, dict(dep_mid, operands=[ids[pos], 'gone']))
    # One a template left with no operands at all can be renamed too.
    doc._samples[name].gates[both]['operands'] = []
    doc.update_gate(name, both, dict(dep_mid, operands=[], name='None left'))
    assert doc.gate(name, both)['name'] == 'None left'
    # A gate that was not a boolean named nothing: made one, it needs some.
    with pytest.raises(DocumentError, match='needs'):
        doc.update_gate(name, ids2[mid], {'kind': 'boolean', 'op': 'and',
                                          'operands': [], 'name': 'Empty'})
    # So is a lost operand's id put on a gate that never named it.
    doc.delete_gate(name, ids[pos])
    with pytest.raises(DocumentError, match='no gate'):
        doc.add_gate(name, {'kind': 'boolean', 'op': 'not',
                            'operands': [ids[pos]], 'name': 'Other'})


def test_the_lost_operand_note_lists_three_and_counts_the_rest():
    from openflo.gate_text import lost_operand_note
    assert lost_operand_note([]) == ''
    assert lost_operand_note(['a', 'b', 'c', 'd', 'a']) == (
        "; boolean gate 'a', 'b', 'c' (+1 more) lost an operand and now "
        "selects nothing")
    assert lost_operand_note([], ['x']) == (
        "; boolean gate 'x', built on one that cannot be evaluated, selects "
        "nothing too")
    assert lost_operand_note(['a', 'b'], ['a', 'y']) == (
        "; boolean gate 'a', 'b' lost an operand and now selects nothing; "
        "boolean gate 'y', built on them, selects nothing too")


# ── counts and statistics ───────────────────────────────────────────────────

def test_counts_and_percentages_match_the_events(tmp_path):
    doc, name, cells, pos, _mid = gated_doc(tmp_path)
    df = doc._samples[name].data.data
    in_cells = ((df['FSC-A'] >= 3e4) & (df['FSC-A'] < 7e4)
                & (df['SSC-A'] >= 1e4) & (df['SSC-A'] < 5e4))
    n_pos = int((in_cells & (df['FL1-A'] >= 0.5)).sum())
    rows = {r['id']: r for r in doc.stats(name, channels=[])}
    assert rows[None]['count'] == 3000 and rows[None]['pct_total'] == 100.0
    assert rows[cells]['count'] == int(in_cells.sum())
    assert rows[pos]['count'] == n_pos
    assert rows[pos]['pct_parent'] == pytest.approx(
        100 * n_pos / in_cells.sum(), rel=1e-12)
    assert rows[pos]['pct_total'] == pytest.approx(100 * n_pos / 3000,
                                                   rel=1e-12)


def test_channel_stats_are_the_editors_on_linear_values(tmp_path):
    """The statistics are gating.population_stats', as the editor's
    Statistics window takes them: on the compensated linear values."""
    from openflo.channel_transforms import linear_values
    from openflo.gate_eval import cumulative_gate_mask
    doc, name, _cells, pos, _mid = gated_doc(tmp_path)
    rec = doc._samples[name]
    rows = {r['id']: r for r in doc.stats(name)}
    ref = gating.population_stats(
        name, rec.data.data, rec.gates, rec.order, {c: c for c in FL}, FL,
        {'Count', '%Parent', '%Total', 'Median', 'Mean', 'GeoMean', 'CV'},
        gating.CHANNEL_STAT_ORDER, transforms=rec.data.data_transforms)
    for r in ref:
        for ch in FL:
            got = rows[r['__gid__']]['channels'][ch]
            assert got == {'median': r[f'Median {ch}'],
                           'mean': r[f'Mean {ch}'],
                           'geomean': r[f'GeoMean {ch}'],
                           'cv': r[f'CV {ch}']}
    m = np.asarray(cumulative_gate_mask(rec.gates, pos, rec.data.data))
    lin = linear_values(rec.data, 'FL2-A')[m]
    assert rows[pos]['channels']['FL2-A']['median'] == pytest.approx(
        float(np.median(lin)), rel=1e-9)


def test_an_empty_population_reads_none_not_nan(tmp_path):
    doc, name, cells, *_ = gated_doc(tmp_path)
    gid = doc.add_gate(name, {'kind': 'threshold', 'channel': 'FL3-A',
                              'value': 99.0}, cells)['id']
    (row,) = doc.stats(name, gates=[gid], all_events=False)
    assert row['count'] == 0 and row['pct_parent'] == 0.0
    assert row['channels']['FL3-A'] == {'median': None, 'mean': None,
                                        'geomean': None, 'cv': None}
    json.dumps(row, allow_nan=False)          # plain data for the bridge


def test_unknown_names_are_refused(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    with pytest.raises(DocumentError, match='no channel'):
        doc.stats(name, channels=['FL9-A'])
    with pytest.raises(DocumentError, match='unknown statistic'):
        doc.stats(name, stats=['Mode'])
    with pytest.raises(DocumentError, match='no loaded sample'):
        doc.stats('nobody')


# ── the plot ────────────────────────────────────────────────────────────────

def test_density_counts_the_population_on_a_grid(tmp_path):
    doc, name, _cells, pos, _mid = gated_doc(tmp_path)
    df = doc._samples[name].data.data
    d = doc.density(name, 'FL1-A', 'FL2-A', gate=pos, bins=(40, 20))
    assert d['counts'].dtype == np.float32
    assert d['counts'].shape == (20, 40)          # rows are y bins
    rows = {r['id']: r for r in doc.stats(name, channels=[])}
    assert d['events'] == d['binned'] == rows[pos]['count']
    assert d['counts'].sum() == rows[pos]['count']
    # The axes span the whole sample, not just the population.
    assert d['x_range'] == [float(df['FL1-A'].min()),
                            float(df['FL1-A'].max())]
    # Row 0 is the low end of y: the population's lowest FL2-A event.
    m = np.asarray(df['FL1-A'] >= 0.5) & np.asarray(
        (df['FSC-A'] >= 3e4) & (df['FSC-A'] < 7e4)
        & (df['SSC-A'] >= 1e4) & (df['SSC-A'] < 5e4))
    lo = float(df['FL2-A'][m].min())
    iy = int(np.searchsorted(d['y_edges'], lo, side='right')) - 1
    assert d['counts'][:iy].sum() == 0 and d['counts'][iy].sum() > 0


def test_a_display_scale_is_a_view_of_the_linear_values(tmp_path):
    from openflo.channel_transforms import linear_values
    doc, name, *_ = gated_doc(tmp_path)
    s = doc._samples[name].data
    lin = linear_values(s, 'FL1-A')
    stored = s.data['FL1-A'].values
    log = doc.to_view(name, 'FL1-A', stored, 'log')
    np.testing.assert_allclose(log, np.log10(np.clip(lin, 1e-6, None)),
                               rtol=1e-9, atol=1e-9)
    back = doc.to_view(name, 'FL1-A', log[lin > 1], 'log', inverse=True)
    np.testing.assert_allclose(back, stored[lin > 1], rtol=0, atol=1e-9)
    asinh = doc.to_view(name, 'FL1-A', stored, 'asinh:c=40')
    np.testing.assert_allclose(asinh, np.arcsinh(lin / 40), atol=1e-9)
    np.testing.assert_array_equal(doc.to_view(name, 'FL1-A', stored),
                                  stored)
    d = doc.density(name, 'FL1-A', 'FSC-A', x_scale='log', bins=8)
    assert d['x_range'][1] == pytest.approx(float(np.log10(lin.max())))


# ── sessions ────────────────────────────────────────────────────────────────

def test_a_saved_session_reopens_with_the_same_gates_and_numbers(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    out = tmp_path / 'proj' / 'run.flowsession'
    out.parent.mkdir()
    doc.save_session(str(out))
    data = json.loads(out.read_text(encoding='utf-8'))
    (entry,) = data['samples']
    assert entry['rel_path'] == '../raw/s1.fcs'
    assert entry['basename'] == 's1.fcs'
    assert entry['comp']['source'] == '$SPILL'
    doc2 = Document()
    assert doc2.open_session(str(out))['samples'] == [name]
    _same(doc2.stats(name), doc.stats(name))
    assert doc2.gate_tree(name) == doc.gate_tree(name)


def test_a_label_set_in_the_session_names_the_channel(tmp_path):
    """The editor puts a session's own channel labels over the file's; the
    Document lists them the same way and saves them back."""
    doc, name, *_ = gated_doc(tmp_path)
    out = tmp_path / 'run.flowsession'
    doc.save_session(str(out))
    data = json.loads(out.read_text(encoding='utf-8'))
    data['channel_labels'] = {'FL2-A': 'Label two'}
    out.write_text(json.dumps(data), 'utf-8')
    doc2 = Document()
    doc2.open_session(str(out))
    got = {c['name']: c['label'] for c in doc2.channels(name)}
    assert got['FL2-A'] == 'Label two'
    assert got['FL1-A'] == {c['name']: c['label']
                            for c in doc.channels(name)}['FL1-A']
    doc2.save_session(str(tmp_path / 'again.flowsession'))
    again = json.loads((tmp_path / 'again.flowsession').read_text('utf-8'))
    assert again['channel_labels'] == {'FL2-A': 'Label two'}


def test_a_sample_reopens_on_the_transforms_it_was_saved_with(tmp_path):
    """Its gates were drawn on those: a raw reload arrives on the loader's
    logicle and is put back on the saved arcsinh first."""
    doc = Document()
    name = doc.open_fcs(spill_fcs(tmp_path / 's1.fcs'))
    doc._samples[name].data.retransform(['FL1-A'], method='asinh')
    gid = doc.add_gate(name, {'kind': 'threshold', 'channel': 'FL1-A',
                              'value': float(np.arcsinh(2000 / 150))})['id']
    (row,) = doc.stats(name, gates=[gid], channels=[], all_events=False)
    assert 300 < row['count'] < 2700
    doc.save_session(str(tmp_path / 'a.flowsession'))
    doc2 = Document()
    doc2.open_session(str(tmp_path / 'a.flowsession'))
    s2 = doc2._samples[name].data
    assert s2.data_transforms['FL1-A']['method'] == 'asinh'
    _same(doc2.stats(name), doc.stats(name))


def test_a_session_moved_with_its_files_still_opens(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    doc.save_session(str(tmp_path / 'run.flowsession'))
    moved = tmp_path.parent / (tmp_path.name + '_moved')
    os.rename(tmp_path, moved)
    try:
        doc2 = Document()
        assert doc2.open_session(str(moved / 'run.flowsession'))[
            'samples'] == [name]
        assert doc2.samples()[0]['path'] == str(moved / 'raw' / 's1.fcs')
    finally:
        os.rename(moved, tmp_path)


def test_a_missing_sample_is_kept_by_a_save(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    doc.save_session(str(tmp_path / 'one.flowsession'))
    os.remove(tmp_path / 'raw' / 's1.fcs')
    doc2 = Document()
    got = doc2.open_session(str(tmp_path / 'one.flowsession'))
    assert got['samples'] == [] and got['missing'] == [name]
    assert doc2.samples()[0]['loaded'] is False
    doc2.save_session(str(tmp_path / 'two.flowsession'))
    two = json.loads((tmp_path / 'two.flowsession').read_text('utf-8'))
    one = json.loads((tmp_path / 'one.flowsession').read_text('utf-8'))
    assert two['sample_gates'] == one['sample_gates']
    assert two['samples'][0]['rel_path'] == 'raw/s1.fcs'


def test_a_sidecar_sample_is_saved_with_a_copy_of_its_sidecar(tmp_path):
    """A session sample that loads from its processed-data sidecar (a
    column the raw FCS lacks) keeps it: the save copies it beside the new
    session, and the copy is what reopens."""
    from openflo.session_sidecar import write_frame
    from openflo.transform_sidecar import write_transforms_sidecar
    doc, name, *_ = gated_doc(tmp_path)
    s = doc._samples[name].data
    s.data['cluster'] = np.arange(len(s.data)) % 3
    # Saved first: a save of s1 from its raw FCS removes an s1 sidecar
    # already in old_data (it would be read for it by name).
    doc.save_session(str(tmp_path / 'old.flowsession'))
    side = tmp_path / 'old_data' / 's1.npz'
    side.parent.mkdir()
    assert write_frame(s.data, str(side))
    write_transforms_sidecar(str(side), s.data_transforms, s.data.columns,
                             len(s.data))
    data = json.loads((tmp_path / 'old.flowsession').read_text('utf-8'))
    data['samples'][0]['processed_data'] = 'old_data/s1.npz'
    (tmp_path / 'old.flowsession').write_text(json.dumps(data), 'utf-8')
    doc2 = Document()
    doc2.open_session(str(tmp_path / 'old.flowsession'))
    assert 'cluster' in doc2._samples[name].data.data.columns
    got = doc2.save_session(str(tmp_path / 'new' / 'x.flowsession'))
    assert got['sidecars'] == [str(tmp_path / 'new' / 'x_data' / 's1.npz')]
    assert os.path.isfile(tmp_path / 'new' / 'x_data' / 's1_transforms.json')
    os.remove(side)                   # the copy, not the original, reopens
    doc3 = Document()
    doc3.open_session(str(tmp_path / 'new' / 'x.flowsession'))
    np.testing.assert_array_equal(doc3._samples[name].data.data['cluster'],
                                  np.arange(3000) % 3)
    _same(doc3.stats(name), doc.stats(name))


def test_nothing_it_opened_is_ever_written(tmp_path):
    doc, *_ = gated_doc(tmp_path)
    sess = tmp_path / 'run.flowsession'
    doc.save_session(str(sess))
    doc2 = Document()
    before = sorted(os.listdir(tmp_path))
    doc2.open_session(str(sess))
    assert sorted(os.listdir(tmp_path)) == before         # no lock file
    stamp = sess.stat().st_mtime_ns
    for path in (sess, tmp_path / 'raw' / 's1.fcs',
                 tmp_path / 'raw' / '..' / 'run.flowsession'):
        with pytest.raises(DocumentError, match='opened from'):
            doc2.save_session(str(path))
    assert sess.stat().st_mtime_ns == stamp
    assert sorted(os.listdir(tmp_path)) == before


def test_a_save_whose_sidecar_copy_would_land_on_a_source_is_refused(
        tmp_path):
    """``run.json`` beside ``run.flowsession`` shares its ``run_data``
    folder: copying the sidecar there would write over the file read."""
    from openflo.session_sidecar import write_frame
    doc, name, *_ = gated_doc(tmp_path)
    s = doc._samples[name].data
    doc.save_session(str(tmp_path / 'run.flowsession'))
    side = tmp_path / 'run_data' / 's1.npz'
    side.parent.mkdir()
    write_frame(s.data, str(side))
    data = json.loads((tmp_path / 'run.flowsession').read_text('utf-8'))
    data['samples'][0]['processed_data'] = 'run_data/s1.npz'
    (tmp_path / 'run.flowsession').write_text(json.dumps(data), 'utf-8')
    doc2 = Document()
    doc2.open_session(str(tmp_path / 'run.flowsession'))
    stamp = side.stat().st_mtime_ns
    with pytest.raises(DocumentError, match='s1.npz'):
        doc2.save_session(str(tmp_path / 'run.json'))
    assert not (tmp_path / 'run.json').exists()
    assert side.stat().st_mtime_ns == stamp


def test_a_file_that_is_not_a_session_is_refused(tmp_path):
    p = tmp_path / 'x.flowsession'
    p.write_text('{"samples": []}', encoding='utf-8')
    with pytest.raises(DocumentError, match='format marker'):
        Document().open_session(str(p))
    p.write_text('{"format": "openflo-session", "version": 99}', 'utf-8')
    with pytest.raises(DocumentError, match='newer OpenFlo'):
        Document().open_session(str(p))


# ── a FlowJo workspace ──────────────────────────────────────────────────────

def test_a_workspace_brings_its_samples_gates_and_groups(tmp_path):
    from openflo.wsp_gate_coords import gate_scale_fns, map_gate_coords
    a = make_fcs(tmp_path / 'run' / 'a.fcs', seed=1)
    b = make_fcs(tmp_path / 'run' / 'b.fcs', seed=2)
    wsp = write_wsp(tmp_path / 'run' / 'x.wsp',
                    [('a', a, 'Stim', True), ('b', b, None, False)],
                    uncompensated=True)
    doc = Document()
    got = doc.import_wsp(wsp)
    assert got['samples'] == ['a', 'b'] and got['unresolved'] == []
    assert ['FL2 mid', 'FL2-A'] in got['uncompensated']
    assert doc._samples['a'].trial == 'Stim'
    assert doc._samples['a'].data.compensation_source.startswith('wsp:')
    assert not doc._samples['b'].data.comp_channels
    (root,) = doc.gate_tree('a')
    assert root['name'] == 'Cells'
    assert [c['name'] for c in root['children']] == ['FL1+', 'FL2 mid',
                                                      'both']
    # FL1+ was written at 2000 (linear); it is on the stored logicle now.
    s = doc._samples['a'].data
    fns = gate_scale_fns(
        {c: t for c, t in s.data_transforms.items()
         if t.get('method') != 'linear'}, 'from_linear')
    want = map_gate_coords({'kind': 'threshold', 'channel': 'FL1-A',
                            'value': 2000.0}, fns)['value']
    fl1 = next(c for c in root['children'] if c['name'] == 'FL1+')
    assert doc.gate('a', fl1['id'])['value'] == pytest.approx(want)
    assert want < 1.5


def test_a_workspace_import_says_what_did_not_open_as_saved(tmp_path):
    # Found with another event count than recorded, file missing, a gate on
    # FlowJo's uncompensated FL2-A (on both samples found): each is a note,
    # in the desktop's words, and each text once.
    run = tmp_path / 'run'
    a = make_fcs(run / 'a.fcs', seed=1)
    b = make_fcs(run / 'b.fcs', seed=2)
    c = make_fcs(run / 'c.fcs', seed=3)
    wsp = write_wsp(run / 'x.wsp', [('a', a, None, True), ('b', b, None, True),
                                    ('c', c, None, False)],
                    uncompensated=True)
    os.remove(c)
    text = open(wsp, encoding='utf-8').read()
    assert text.count('<SampleNode name="b" ') == 1
    text = text.replace('<SampleNode name="b" ', '<SampleNode name="b" x="1" ')
    text = text.replace('x="1" sampleID="2" owningGroup="" expanded="1" '
                        'sortPriority="10" count="600"',
                        'sampleID="2" owningGroup="" expanded="1" '
                        'sortPriority="10" count="700"')
    assert 'count="700"' in text
    with open(wsp, 'w', encoding='utf-8') as fh:
        fh.write(text)
    got = Document().import_wsp(wsp)
    assert got['samples'] == ['a', 'b']
    assert got['notes'] == [
        'x.wsp: FCS file not found for c',
        "x.wsp: 'b': the workspace records 700 events for this sample but "
        "the FCS file has 600: it may be a different file of the same name",
        "imported gate 'FL2 mid' is on FlowJo's uncompensated FL2-A, but "
        "this sample holds it compensated: its count will differ from "
        "FlowJo's"]


def _dup_channel_fcs(path):
    """An FCS file at `path` that names FL1-A twice: reading it renames the
    second and says so in the sample's load warnings."""
    from tests._fcs_writer import write_float_fcs
    ev = np.random.default_rng(1).uniform(100, 1e5, (600, 4))
    return write_float_fcs(str(path), ev, ['FSC-A', 'SSC-A', 'FL1-A', 'FL1-A'])


def test_a_workspace_import_notes_come_in_the_desktop_order(tmp_path):
    # d's file is gone from beside the workspace but two of that name sit
    # in its subfolders (none is picked), e's file names a channel twice,
    # a has a gate on FlowJo's uncompensated FL2-A: the workspace's notes,
    # then each sample's load warnings, then the uncompensated gates.
    import shutil
    run = tmp_path / 'run'
    a = make_fcs(run / 'a.fcs', seed=1)
    d = make_fcs(run / 'd.fcs', seed=4)
    e = _dup_channel_fcs(run / 'e.fcs')
    wsp = write_wsp(run / 'x.wsp', [('d', d, None, False), ('e', e, None, False),
                                    ('a', a, None, True)], uncompensated=True)
    for sub in ('one', 'two'):
        (run / sub).mkdir()
        shutil.copy(d, run / sub / 'd.fcs')
    os.remove(d)
    got = Document().import_wsp(wsp)
    assert got['samples'] == ['e', 'a']
    assert got['notes'] == [
        'x.wsp: FCS file not found for d',
        'x.wsp: 2 files named d.fcs in its folders; none was picked. Point '
        'the FCS folder at the right one.',
        "e: duplicate channel name 'FL1-A': the second is 'FL1-A (2)'",
        "imported gate 'FL2 mid' is on FlowJo's uncompensated FL2-A, but "
        "this sample holds it compensated: its count will differ from "
        "FlowJo's"]


def test_a_session_notes_each_sample_load_warning(tmp_path):
    e = _dup_channel_fcs(tmp_path / 'e.fcs')
    doc = Document()
    name = doc.open_fcs(e)
    out = tmp_path / 'run.flowsession'
    doc.save_session(str(out))
    got = Document().open_session(str(out))
    assert got['samples'] == [name]
    assert ("e: duplicate channel name 'FL1-A': the second is 'FL1-A (2)'"
            in got['notes'])


def test_a_session_entry_naming_no_sample_notes_its_load_warning(tmp_path):
    # An entry with no name is opened from its raw file, nothing restored
    # onto it; its file's load warnings are noted all the same.
    e = _dup_channel_fcs(tmp_path / 'e.fcs')
    doc = Document()
    doc.open_fcs(e)
    out = tmp_path / 'run.flowsession'
    doc.save_session(str(out))
    data = json.loads(out.read_text(encoding='utf-8'))
    (entry,) = data['samples']
    entry['name'] = ''
    out.write_text(json.dumps(data), encoding='utf-8')
    got = Document().open_session(str(out))
    assert got['samples'] == ['e']
    assert got['notes'] == [
        "e: duplicate channel name 'FL1-A': the second is 'FL1-A (2)'"]


def test_a_workspace_import_notes_populations_left_out(tmp_path):
    a = make_fcs(tmp_path / 'a.fcs', seed=1)
    wsp = write_wsp(tmp_path / 'x.wsp', [('a', a, None, True)])
    text = open(wsp, encoding='utf-8').read()
    # Every gate becomes a type OpenFlo cannot read: the top population
    # is left out with the two populations under it.
    with open(wsp, 'w', encoding='utf-8') as fh:
        fh.write(text.replace('gating:RectangleGate', 'gating:CurlyQuad'))
    got = Document().import_wsp(wsp)
    assert got['samples'] == ['a']
    assert got['notes'] == [
        "x.wsp: population 'Cells' (CurlyQuad) not imported, nor the 2 "
        "under it"]


def test_a_workspace_import_notes_unconverted_ellipses_and_what_rests_on_them(
        tmp_path, monkeypatch):
    # FlowJo writes ellipses OpenFlo cannot always convert; and a sample in
    # two groups goes in the first. Planted on the plan, as a workspace
    # holding them would give it.
    from openflo import wsp_open
    a = make_fcs(tmp_path / 'a.fcs', seed=1)
    gates = [
        {'_import_id': 'e1', 'parent_id': None, 'kind': 'flowjo_ellipse',
         'x_channel': 'FL1-A', 'y_channel': 'FL2-A', 'label': 'Blob',
         'name': 'Blob', 'flowjo': {}},
        {'_import_id': 't1', 'parent_id': None, 'kind': 'threshold',
         'channel': 'FL1-A', 'value': 1.0, 'name': 'Pos'},
        {'_import_id': 'b1', 'parent_id': None, 'kind': 'boolean',
         'op': 'and', 'operands': ['e1', 't1'], 'name': 'Both'},
        {'_import_id': 'b2', 'parent_id': None, 'kind': 'boolean',
         'op': 'not', 'operands': ['t1'], 'name': 'Neg'},
    ]
    plan = wsp_open.WspPlan(samples=[wsp_open.WspSample(
        node_name='a node', fcs_path=a, gates=gates, stat_axes={},
        groups=['G1', 'G2'], comp=None)])
    monkeypatch.setattr(wsp_open, 'plan_wsp', lambda path, locator: plan)
    got = Document().import_wsp(str(tmp_path / 'x.wsp'))
    assert got['notes'] == [
        "x.wsp: 'a node' is in the groups G1, G2; it was put in 'G1'",
        "x.wsp: ellipse 'Blob' could not be converted and reads NA",
        "x.wsp: boolean gate 'Both' is built on an ellipse that was not "
        "converted and reads NA"]


# ── without FlowUtils ───────────────────────────────────────────────────────

_NO_FLOWUTILS = textwrap.dedent('''
    import json, sys
    for m in ('flowutils', 'flowutils.transforms', 'flowutils.gating'):
        sys.modules[m] = None
    from openflo.document import Document
    import openflo.channel_transforms as ct, openflo.biexp_np as bn
    assert ct.transforms is bn
    doc = Document()
    n = doc.open_fcs(sys.argv[1])
    c = doc.add_gate(n, {'kind': 'polygon', 'x_channel': 'FL1-A',
                         'y_channel': 'FL2-A', 'vertices':
                         [[0.3, 0.2], [0.9, 0.2], [0.9, 0.8], [0.3, 0.8]]})
    doc.add_gate(n, {'kind': 'threshold', 'channel': 'FL3-A',
                     'value': 0.4}, c['id'])
    print(json.dumps(doc.stats(n), sort_keys=True))
''')


def test_it_gives_the_same_numbers_without_flowutils(tmp_path):
    """Run in a fresh interpreter with FlowUtils unimportable (as in a
    browser runtime): the logicle and polygon test are the numpy ports."""
    p = spill_fcs(tmp_path / 's1.fcs')
    env = dict(os.environ, OMP_NUM_THREADS='1')
    out = subprocess.run([sys.executable, '-c', _NO_FLOWUTILS, p],
                         capture_output=True, text=True, env=env,
                         timeout=300, check=False)
    assert out.returncode == 0, out.stderr[-2000:]
    doc = Document()
    n = doc.open_fcs(p)
    c = doc.add_gate(n, {'kind': 'polygon', 'x_channel': 'FL1-A',
                         'y_channel': 'FL2-A', 'vertices':
                         [[0.3, 0.2], [0.9, 0.2], [0.9, 0.8], [0.3, 0.8]]})
    doc.add_gate(n, {'kind': 'threshold', 'channel': 'FL3-A',
                     'value': 0.4}, c['id'])
    got = json.loads(out.stdout.strip().splitlines()[-1])
    want = doc.stats(n)
    assert [r['count'] for r in got] == [r['count'] for r in want]
    assert want[1]['count'] > 100 and want[2]['count'] > 10
    for g, w in zip(got, want, strict=True):
        for ch in FL:
            for k, v in w['channels'][ch].items():
                assert g['channels'][ch][k] == pytest.approx(v, rel=1e-9)


# ── Gating-ML on the phone: the desktop's Apply template ────────────────


def _gml_of(doc, name, path, operand=None):
    """Write sample `name`'s gates of `doc` as Gating-ML at `path` (as the
    desktop exports them); with `operand`, a boolean's first gate
    reference is pointed at that id instead (one the file does not hold)."""
    from openflo.gatingml import write_gatingml
    from openflo.sample_prep import stored_transforms
    rec = doc._samples[name]
    gates = [dict(rec.gates[i], id=i) for i in rec.order]
    s = rec.data
    write_gatingml(str(path), gates, stored_transforms(s),
                   compensation=(s.comp_channels, s.comp_matrix))
    if operand:
        text = path.read_text(encoding='utf-8')
        at = text.index('gateReference')
        ref = text.index('ref="', at) + len('ref="')
        end = text.index('"', ref)
        path.write_text(text[:ref] + operand + text[end:], encoding='utf-8')
    return str(path)


def _named(doc, name, ids):
    """{name the gate list shows: id} for the gates `ids` of sample `name`."""
    from openflo.gating import population_names
    gates = doc._samples[name].gates
    return {population_names(gates, i)[-1]: i for i in ids}


def _counts(doc, name):
    return {r['population']: r['count'] for r in doc.stats(name)}


def test_a_gatingml_file_puts_the_desktops_gates_on_a_phone_sample(tmp_path):
    doc, s1, cells, _pos, _mid = gated_doc(tmp_path)
    doc.add_gate(s1, {'kind': 'boolean', 'op': 'not', 'operands': [cells],
                      'name': 'not cells'})
    gml = _gml_of(doc, s1, tmp_path / 'panel.xml')
    s2 = doc.open_fcs(spill_fcs(tmp_path / 'raw' / 's2.fcs', seed=2))
    doc.add_gate(s2, {'kind': 'threshold', 'channel': 'FL3-A', 'value': 0.4,
                      'name': 'own'})
    out = doc.apply_gatingml(gml, [s2])
    assert out['status'] == ("Applied 4 gate(s) [boolean, interval, rect, "
                             "threshold] from Gating-ML (panel.xml): "
                             "overwrote 1 sample(s).")
    assert out['notes'] == ["the file's gates use its compensation matrix "
                            "Spill; OpenFlo gates each sample with its own "
                            "compensation"]
    # The same gates as a copy of s1's: the same populations counting the
    # same events of s2, its own gate gone.
    twin = Document()
    t1 = twin.open_fcs(str(tmp_path / 'raw' / 's1.fcs'))
    for i in doc._samples[s1].order:
        twin._samples[t1].gates[i] = dict(doc._samples[s1].gates[i])
        twin._samples[t1].order.append(i)
    t2 = twin.open_fcs(str(tmp_path / 'raw' / 's2.fcs'))
    twin.copy_gates(t1, [t2])
    assert _counts(doc, s2) == _counts(twin, t2)
    assert 'own' not in str(_counts(doc, s2))
    assert out['ids'][s2] == doc._samples[s2].order
    # The boolean gate combines the file's cells gate as put on s2.
    tree = _named(doc, s2, out['ids'][s2])
    assert doc.gate(s2, tree['not cells'])['operands'] == [tree['cells']]


def test_a_gatingml_file_added_to_a_samples_own_gates(tmp_path):
    doc, s1, cells, _pos, _mid = gated_doc(tmp_path)
    doc.add_gate(s1, {'kind': 'boolean', 'op': 'not', 'operands': [cells],
                      'name': 'not cells'})
    gml = _gml_of(doc, s1, tmp_path / 'panel.xml')
    # s1 keeps its gates; the file's come after them, and its boolean
    # combines the file's cells, not s1's own (ids g1.. are s1's).
    out = doc.apply_gatingml(gml, [s1], overwrite=False)
    assert out['status'].endswith(': added to 1 sample(s).')
    rec = doc._samples[s1]
    assert len(rec.order) == 8 and rec.order[:4] == [cells, *rec.order[1:4]]
    new = _named(doc, s1, out['ids'][s1])
    assert new['cells'] != cells
    assert doc.gate(s1, new['not cells'])['operands'] == [new['cells']]
    # A sibling 1-D gate of the file never replaces another of the file;
    # FL1+ under the new cells is a new gate, not s1's own moved.
    assert doc.gate(s1, new['FL1+'])['parent_id'] == new['cells']


def test_a_gatingml_file_says_what_will_not_count(tmp_path):
    doc, s1, cells, _pos, _mid = gated_doc(tmp_path)
    doc.add_gate(s1, {'kind': 'boolean', 'op': 'and', 'operands': [cells],
                      'name': 'both'})
    gml = _gml_of(doc, s1, tmp_path / 'lost.xml', operand='nowhere')
    # A sample without FL2-A or FL3-A: the gate on FL2-A sits inert there.
    rng = np.random.default_rng(4)
    narrow = write_fcs(str(tmp_path / 'raw' / 'n.fcs'),
                       rng.normal(4e4, 1e4, (500, 3)),
                       ['FSC-A', 'SSC-A', 'FL1-A'], datatype='F')
    n = doc.open_fcs(narrow)
    out = doc.apply_gatingml(gml, [s1, n])
    assert out['status'] == (
        "Applied 4 gate(s) [boolean, interval, rect, threshold] from "
        "Gating-ML (lost.xml): overwrote 2 sample(s); boolean gate 'both' "
        "lost an operand and now selects nothing.")
    assert out['notes'] == [
        "the file's gates use its compensation matrix Spill; OpenFlo gates "
        "each sample with its own compensation",
        "'both': an operand was left out, so it reads NA",
        "Some gates reference channels that aren't present in these "
        "samples — they'll be inactive (no-op) there until the channels "
        "exist:",
        f"{n}: 1 gate(s)"]


def test_a_gatingml_files_one_d_gates_replace_the_samples_own_never_each_other(
        tmp_path):
    from openflo.gatingml import write_gatingml
    doc = Document()
    s1 = doc.open_fcs(spill_fcs(tmp_path / 'raw' / 's1.fcs'))
    own = doc.add_gate(s1, {'kind': 'interval', 'channel': 'SSC-A',
                            'lo': 0.0, 'hi': 1e5, 'name': 'SSC own'})['id']
    path = tmp_path / 'bands.xml'
    write_gatingml(str(path), [
        {'id': 'b1', 'parent_id': None, 'kind': 'interval', 'channel': 'FSC-A',
         'lo': 2e4, 'hi': 5e4, 'name': 'FSC lo'},
        {'id': 'b2', 'parent_id': None, 'kind': 'interval', 'channel': 'FSC-A',
         'lo': 5e4, 'hi': 8e4, 'name': 'FSC hi'},
        {'id': 'b3', 'parent_id': None, 'kind': 'interval', 'channel': 'SSC-A',
         'lo': 2e4, 'hi': 4e4, 'name': 'SSC band'}])
    out = doc.apply_gatingml(str(path), [s1], overwrite=False)
    rec = doc._samples[s1]
    # Both FSC bands, and s1's own SSC gate holding the file's band.
    assert len(rec.order) == 3 and out['ids'][s1][2] == own
    assert {'FSC lo', 'FSC hi'} <= set(_named(doc, s1, rec.order))
    assert (rec.gates[own]['lo'], rec.gates[own]['hi']) == (2e4, 4e4)


def test_a_gatingml_file_names_only_its_own_lost_booleans(tmp_path):
    doc, s1, _cells, pos, _mid = gated_doc(tmp_path)
    gml = _gml_of(doc, s1, tmp_path / 'plain.xml')
    # s1's own boolean that lost an operand is not the file's to name.
    doc.add_gate(s1, {'kind': 'boolean', 'op': 'and', 'operands': [pos],
                      'name': 'mine'})
    doc.delete_gate(s1, pos)
    out = doc.apply_gatingml(gml, [s1], overwrite=False)
    assert out['status'] == ("Applied 3 gate(s) [interval, rect, threshold] "
                             "from Gating-ML (plain.xml): added to 1 "
                             "sample(s).")
    assert doc.lost_operands(s1)['lost'] == ['mine']


def test_a_gatingml_file_on_linear_values_goes_onto_the_samples_scales(
        tmp_path):
    from openflo.channel_transforms import linear_values
    from openflo.gatingml import write_gatingml
    doc = Document()
    s1 = doc.open_fcs(spill_fcs(tmp_path / 'raw' / 's1.fcs'))
    s = doc._samples[s1].data
    assert s.data_transforms['FL1-A']['method'] == 'logicle'
    path = tmp_path / 'linear.xml'
    write_gatingml(str(path), [
        {'id': 'p1', 'parent_id': None, 'kind': 'threshold',
         'channel': 'FL1-A', 'value': 1000.0, 'name': 'FL1 over 1000'}],
        compensation=(s.comp_channels, s.comp_matrix))
    doc.apply_gatingml(str(path), [s1])
    lin = np.asarray(linear_values(s, 'FL1-A'))
    got = {r['population']: r['count'] for r in doc.stats(s1)}
    assert got['FL1 over 1000'] == int((lin >= 1000.0).sum())


def test_a_gatingml_gate_on_uncompensated_values_is_noted(tmp_path):
    from openflo.gatingml import write_gatingml
    from openflo.sample_prep import stored_transforms
    doc, s1, *_ = gated_doc(tmp_path)
    rec = doc._samples[s1]
    path = tmp_path / 'raw.xml'
    # Written without the matrix: its fluorescence gates read uncompensated
    # values, which s1 holds compensated.
    gates = [dict(rec.gates[i], id=i) for i in rec.order]
    # A second gate of the same name on the same channel: one note.
    gates.append(dict(gates[1], id='again', value=0.7))
    write_gatingml(str(path), gates, stored_transforms(rec.data))
    out = doc.apply_gatingml(str(path), [s1])
    assert out['notes'] == [
        f"'FL1+' is drawn on uncompensated FL1-A; {s1} is compensated there, "
        "so it counts compensated values",
        f"'FL2 mid' is drawn on uncompensated FL2-A; {s1} is compensated "
        "there, so it counts compensated values"]


def test_a_gatingml_dimension_named_by_marker_reads_where_the_marker_sits(
        tmp_path):
    """A file that names a dimension by its marker (CD901b) puts the gate on
    the detector each sample labels it on; a sample without the label gets
    it inert and is listed. A population named like a marker is not moved
    (the file's names are populations, not markers). A sample named twice
    is applied once."""
    from openflo.gatingml import write_gatingml
    from openflo.sample_prep import stored_transforms
    doc, s1, cells, _pos, _mid = gated_doc(tmp_path)
    # A population named like the marker, on another channel.
    doc.add_gate(s1, {'kind': 'threshold', 'channel': 'FL3-A', 'value': 0.4,
                      'name': 'CD901b'}, cells)
    rec = doc._samples[s1]
    path = tmp_path / 'marker.xml'
    write_gatingml(str(path), [dict(rec.gates[i], id=i) for i in rec.order],
                   stored_transforms(rec.data))
    text = path.read_text(encoding='utf-8')
    assert text.count('name="FL1-A"')
    path.write_text(text.replace('name="FL1-A"', 'name="CD901b"'),
                    encoding='utf-8')
    rng = np.random.default_rng(5)
    n = 3000
    true = np.column_stack([rng.lognormal(7, 1, n), rng.lognormal(6, 1, n),
                            rng.lognormal(5, 1, n)])
    ev = np.column_stack([rng.normal(5e4, 1e4, n), rng.normal(3e4, 8e3, n),
                          true @ SPILL])
    spill = '3,' + ','.join(FL) + ',' + ','.join(map(str, SPILL.ravel()))
    m = doc.open_fcs(write_fcs(str(tmp_path / 'raw' / 'm.fcs'), ev, CHANNELS,
                               datatype='F', extra={'$SPILLOVER': spill,
                                                    '$P4S': 'CD901b'}))
    plain = doc.open_fcs(spill_fcs(tmp_path / 'raw' / 'p.fcs', seed=6))
    out = doc.apply_gatingml(str(path), [m, plain, m], overwrite=False)
    assert out['status'].endswith(': added to 2 sample(s).')
    assert list(out['ids']) == [m, plain]
    on_m = _named(doc, m, out['ids'][m])
    assert len(doc._samples[m].order) == len(rec.order)
    assert doc.gate(m, on_m['FL1+'])['channel'] == 'FL2-A'
    assert doc.gate(m, on_m['CD901b'])['channel'] == 'FL3-A'
    counts = _counts(doc, m)
    pos_row = next(k for k in counts if k.endswith('FL1+'))
    assert counts[pos_row] is not None and counts[pos_row] > 0
    # Not labelled on `plain`: its gate stays on the marker and is listed.
    on_plain = _named(doc, plain, out['ids'][plain])
    assert doc.gate(plain, on_plain['FL1+'])['channel'] == 'CD901b'
    assert out['notes'][-2:] == [
        "Some gates reference channels that aren't present in these samples "
        "— they'll be inactive (no-op) there until the channels exist:",
        f'{plain}: 1 gate(s)']
    # Drawn on uncompensated values: noted on the detector it reads on m.
    assert (f"'FL1+' is drawn on uncompensated FL2-A; {m} is compensated "
            "there, so it counts compensated values") in out['notes']


def test_a_file_that_is_not_gatingml_changes_nothing(tmp_path):
    doc, s1, *_ = gated_doc(tmp_path)
    before = list(doc._samples[s1].order)
    bad = tmp_path / 'notes.xml'
    bad.write_text('<html>not gates</html>', encoding='utf-8')
    with pytest.raises(DocumentError, match='notes.xml could not be read as Gating-ML'):
        doc.apply_gatingml(str(bad), [s1])
    assert doc._samples[s1].order == before
    with pytest.raises(DocumentError, match='no sample'):
        doc.apply_gatingml(str(bad), [])
    with pytest.raises(DocumentError, match='gone.xml could not be read'):
        doc.apply_gatingml(str(tmp_path / 'gone.xml'), [s1])
    assert doc._samples[s1].order == before
