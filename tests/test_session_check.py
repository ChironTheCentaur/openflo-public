"""A damaged session gives a plain-words error, not a raw Python one.

Of 1,500 randomly damaged copies of the sample session, about a third
failed with a raw TypeError / AttributeError / KeyError from deep inside
the open or the statistics (a gate list that is text, a sample path that
is a number, a gate name that is a list). ``session_check`` finds those at
open (``migrate_session``), so the desktop and the phone's document refuse
the file with a message that names the sample or gate. A gate whose
REGION cannot be read (unknown kind, missing bound) still opens, as an
unknown population (``gate_record``), so one bad gate does not lose the
session. Generated files and generic channel names only.
"""
from __future__ import annotations

import copy
import json
import math

import pytest

from openflo.document import Document, DocumentError
from openflo.session_check import session_structure_problem
from openflo.session_format import (
    SessionDamagedError,
    SessionError,
    SessionVersionError,
    migrate_session,
)
from tests._open_fixtures import session_dict, session_gates
from tests.test_document import gated_doc


def _good():
    entries = [{'name': 'A', 'path': 'raw/A.fcs', 'rel_path': 'raw/A.fcs',
                'basename': 'A.fcs', 'color': '#123456', 'trial': 'Day 1',
                'derived': {'r': 'FL1-A/FL2-A'},
                'transforms': {'FL1-A': {'method': 'logicle'}}},
               {'name': 'B', 'path': 'raw/B.fcs'}]
    data = session_dict(entries, {'A': session_gates('A'),
                                  'B': session_gates('B')})
    data.update(channel_labels={'FL1-A': 'one'}, channel_scale={},
                channel_range=None, cluster_labels={}, audit=[],
                fixes_applied=[], fixes_declined=None)
    return data


def _set(data, path, value):
    """`data` with the field at `path` (keys and indexes) set to `value`."""
    o = data
    for k in path[:-1]:
        o = o[k]
    o[path[-1]] = value
    return data


DAMAGE = [
    # (where, value, words the message must hold)
    (('samples',), 'x', 'the sample list is text'),
    (('samples',), None, 'the sample list is empty'),
    (('samples', 1), 'B.fcs', 'sample 2 is text, not a sample record'),
    (('samples', 0, 'name'), ['A'], "sample 1: 'name' is a list"),
    (('samples', 0, 'path'), 3, "sample 'A': 'path' is a number"),
    (('samples', 0, 'rel_path'), 3, "sample 'A': 'rel_path' is a number"),
    (('samples', 0, 'basename'), {'a': 1}, "'basename' is a record"),
    (('samples', 0, 'color'), 7, "'color' is a number"),
    (('samples', 0, 'trial'), ['x'], "'trial' is a list"),
    (('samples', 0, 'derived'), ['x'], "'derived' is a list, not a record"),
    (('samples', 0, 'processed_data'), 1, "'processed_data' is a number"),
    (('samples', 0, 'processed_csv'), [], "'processed_csv' is a list"),
    (('samples', 0, 'container_member'), 2, "'container_member' is a"),
    (('sample_gates',), ['x'], 'the gate list is a list'),
    (('sample_gates', 'A'), 'g2', "the gates of sample 'A' are text"),
    (('sample_gates', 'A'), None, "the gates of sample 'A' are empty"),
    (('sample_gates', 'A', 1), 'g5', "gate 2 of sample 'A' is text"),
    (('sample_gates', 'A', 0, 'name'), ['x'], "'name' is a list, not text"),
    (('sample_gates', 'A', 0, 'label'), 3, "'label' is a number, not text"),
    (('sample_gates', 'A', 1, 'parent_id'), ['g2'],
     "'parent_id' is a list, not a gate id"),
    (('sample_gates', 'A', 0, 'id'), True, "'id' is true/false"),
    (('sample_gates', 'A', 0, 'id'), 1.5, "'id' is a number"),
    (('sample_gates', 'A', 0, '_import_id'), {}, "'_import_id' is a record"),
    (('sample_gates', 'A', 0, 'quad_set'), [1], "'quad_set' is a list"),
    (('sample_gates', 'A', 2, 'operands'), 'g2', 'boolean operands'),
    (('sample_gates', 'A', 2, 'operands'), [{'a': 1}], 'boolean operands'),
    (('channel_labels',), ['x'], "'channel_labels' is a list"),
    (('channel_scale',), 'x', "'channel_scale' is text"),
    (('channel_range',), 3, "'channel_range' is a number"),
    (('cluster_labels',), True, "'cluster_labels' is true/false"),
    (('audit',), math.nan, "'audit' is a number, not a list"),
    (('fixes_applied',), 'x', "'fixes_applied' is text"),
    (('fixes_declined',), {'a': 1}, "'fixes_declined' is a record"),
]


def test_a_whole_session_has_no_problem():
    assert session_structure_problem(_good()) is None
    assert migrate_session(_good())[0]['samples'][0]['name'] == 'A'


@pytest.mark.parametrize('where,value,words', DAMAGE,
                         ids=lambda v: str(v)[:30])
def test_each_damage_is_named(where, value, words):
    data = _set(_good(), where, value)
    assert words in (session_structure_problem(data) or '')
    with pytest.raises(SessionDamagedError, match='damaged') as got:
        migrate_session(copy.deepcopy(data))
    assert words in str(got.value)


@pytest.mark.parametrize('where,value', [
    (('samples', 0, 'app_version'), ['x']),       # informational only
    (('samples', 0, 'transforms'), 'not a record'),   # read leniently
    (('samples', 0, 'comp'), 3),                  # falls back to $SPILL
    (('sample_gates', 'A', 0, 'id'), 4),          # a hand-written id
    (('sample_gates', 'A', 2, 'operands'), ['g5', 2]),
    (('sample_gates', 'A', 0, 'kind'), 'spline'),  # an unknown population
    (('sample_gates', 'A', 0, 'value'), 'x'),     # an unknown population
    (('channel_scale',), None),
    (('sample_gates',), []),
    (('sample_gates',), None),
    (('audit',), 0),
])
def test_what_opened_before_and_is_read_leniently_still_opens(where, value):
    assert session_structure_problem(_set(_good(), where, value)) is None


def _with_tab(gates, name='Tab 2'):
    data = _good()
    data['view_tabs'] = {'current': 0, 'tabs': [
        {'name': 'All'}, {'name': name, 'sample_gates': {'A': gates}}]}
    return data


@pytest.mark.parametrize('gate,words', [
    (dict(session_gates('A')[1], parent_id=['g2']),
     "tab 'Tab 2': gate 1 of sample 'A': 'parent_id' is a list"),
    (dict(session_gates('A')[0], id=['g2']),
     "tab 'Tab 2': gate 1 of sample 'A': 'id' is a list"),
    (dict(session_gates('A')[0], name=5),
     "tab 'Tab 2': gate 1 of sample 'A': 'name' is a number"),
    (dict(session_gates('A')[2], operands='g2'),
     "tab 'Tab 2': gate 1 of sample 'A': its boolean operands"),
])
def test_a_tabs_own_gates_are_checked(gate, words):
    """A tab keeps gates of its own; a list as one's parent raised
    TypeError in the restore after the session had opened (#446 review)."""
    data = _with_tab([gate])
    assert words in (session_structure_problem(data) or '')
    with pytest.raises(SessionDamagedError, match='damaged'):
        migrate_session(data)


@pytest.mark.parametrize('view_tabs', [
    None, [], 'x', {'tabs': 'x'}, {'tabs': ['x', {'name': 'T'}]},
    {'tabs': [{'name': 'T', 'sample_gates': 'x'}]},
    {'tabs': [{'name': 'T', 'sample_gates': {'A': 'x'}}]},
    {'tabs': [{'name': 'T', 'sample_gates': {'A': ['x', 3]}}]},
])
def test_what_the_tab_restore_drops_is_not_an_error(view_tabs):
    """view_tabs.ViewTab.from_dict already leaves out a tab, a gate list
    or a gate that is not a record or list: those open as before."""
    data = _good()
    data['view_tabs'] = view_tabs
    assert session_structure_problem(data) is None


def test_whole_tab_gates_pass():
    assert session_structure_problem(_with_tab(session_gates('A'))) is None


def test_a_missing_sample_list_is_no_samples():
    data = _good()
    del data['samples'], data['sample_gates']
    assert session_structure_problem(data) is None


def test_the_damage_error_is_a_session_error():
    """Every caller that opens a session catches SessionError: the editor's
    Open session and Resume, Upgrade saved session, and the document."""
    assert issubclass(SessionDamagedError, SessionError)
    assert issubclass(SessionVersionError, SessionError)
    with pytest.raises(SessionVersionError, match='Not an OpenFlo session'):
        migrate_session(['not', 'a', 'session'])


# ── through the document the phone opens sessions with ─────────────────────

def _saved(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    path = tmp_path / 'run.flowsession'
    doc.save_session(str(path))
    return path, name, json.loads(path.read_text('utf-8'))


def test_the_document_refuses_a_damaged_session_in_plain_words(tmp_path):
    """Before: TypeError from the gate restore."""
    path, name, data = _saved(tmp_path)
    data['sample_gates'][name][1]['parent_id'] = ['x']
    path.write_text(json.dumps(data), 'utf-8')
    with pytest.raises(DocumentError) as got:
        Document().open_session(str(path))
    assert "damaged and can't be opened" in str(got.value)
    assert f"gate 2 of sample '{name}'" in str(got.value)


@pytest.mark.parametrize('damage', [
    {'kind': 'spline'},             # a kind from a newer OpenFlo
    {'kind': None},                 # no kind
    {'x0': None},                   # a missing bound
])
def test_an_unknown_gate_reads_na_on_the_phone_not_the_parents_numbers(
        tmp_path, damage):
    """Before: an unknown kind (or none) reported its parent's count, 100 %
    of parent and its parent's medians under its own name; a missing x0
    stopped the statistics of the whole sample with a TypeError."""
    path, name, data = _saved(tmp_path)
    gates = data['sample_gates'][name]
    # The cells gate (a rect) goes under nothing; FL1+ and FL2 mid under it.
    cells = gates[0]
    for k, v in damage.items():
        if v is None:
            cells.pop(k, None)
        else:
            cells[k] = v
    path.write_text(json.dumps(data), 'utf-8')
    doc = Document()
    assert doc.open_session(str(path))['samples'] == [name]
    rows = doc.stats(name)
    by = {r['name']: r for r in rows}
    for pop in ('cells', 'FL1+', 'FL2 mid'):
        r = by[pop]
        assert r['unknown'], pop
        assert r['count'] is None and r['pct_parent'] is None, pop
    # Nothing to draw for it either.
    assert all(o['name'] != 'cells' for o in
               doc.overlays(name, 'FSC-A', 'SSC-A'))


def test_a_threshold_whose_cut_is_text_is_not_drawn_on_the_phone(tmp_path):
    """Before: the plot's gate outlines raised ValueError (float('x')), so
    the phone could not draw that plot at all."""
    path, name, data = _saved(tmp_path)
    fl1 = next(g for g in data['sample_gates'][name] if g['name'] == 'FL1+')
    fl1['value'] = 'x'
    path.write_text(json.dumps(data), 'utf-8')
    doc = Document()
    doc.open_session(str(path))
    (cells,) = doc.gate_tree(name)
    drawn = [o['name'] for o in doc.overlays(name, 'FL1-A', 'FL2-A',
                                             parent=cells['id'])]
    assert drawn == ['FL2 mid']
    by = {r['name']: r for r in doc.stats(name)}
    assert "'value'" in by['FL1+']['unknown']
    assert by['FL2 mid']['unknown'] is None
