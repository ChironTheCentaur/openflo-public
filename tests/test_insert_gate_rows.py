"""The gate and auto-clean method rows _insert_sample_subtree puts in the tree.

Pins the gate rows (parent-first order, skipped and repeated ids, a parent
the sample does not have, on/off tags and ticks, colours, open state, the
auto-clean "drops N" readout, tray tabs that show only some gates) and the
auto-clean group's method rows (label, effective debris cut in every mode,
per-method drop readout, the reason for an enabled method that drops
nothing, ticks, tags, iids, order). The tree is editor_tree._TreeRecording,
so each test sees every insert and tag_configure in order; the editor around
the call is a stub with the state and helpers the build reads. Data is made
up: sample S1, gates G1.., channels FL1-A / FL2-A.
"""
from __future__ import annotations

from openflo import plotmath, tree_ids
from openflo.editor_dnd import DnDMixin
from openflo.editor_tree import _TreeRecording
from openflo.theme import current_palette

ON, OFF = '☑', '☐'


class _Host(DnDMixin):
    """The editor state and helpers the subtree build reads."""
    _sample_iid = staticmethod(tree_ids.sample_iid)
    _gate_iid = staticmethod(tree_ids.gate_iid)
    _method_iid = staticmethod(tree_ids.method_iid)
    _drop_suffix = staticmethod(plotmath.drop_suffix)

    def __init__(self, gates, order=(), shown=None, ac=None, counts=None):
        self._sample_gates = {'S1': gates}
        self._sample_gate_order = {'S1': list(order)}
        self._sample_plot_enabled = {'S1': True}
        self._sample_set_aside = {}
        self._sample_notes = {}
        self.shown = shown
        self.ac = ac or {}
        self.counts = counts if counts is not None else {}
        self.ac_calls = []

    def _display_transforms(self, name):
        return 'TR'

    def _color_for(self, name):
        return '#010203'

    def _sample_display_count(self, name):
        return (10, 10)

    def _plotcache_peek_counts(self, name, gids):
        return self.counts

    def _tree_gate_text(self, name, gid, transforms, counts):
        return f'{name}/{gid}|{transforms}|{counts.get(gid)}'

    def _autoclean_counts(self, name, gid):
        self.ac_calls.append((name, gid))
        return self.ac.get(gid)

    def _tab_shown_gates(self, name, gates):
        return self.shown


def _ops(host):
    tv = _TreeRecording()
    host._insert_sample_subtree('S1', 'T:t1', tv)
    return tv.ops


def _rows(ops):
    """Inserted rows after the sample row, as dicts (parent, iid, text,
    values, tags, open)."""
    out = [dict(op[3], parent=op[1]) for op in ops if op[0] == 'insert']
    assert out[0]['iid'] == 'S:S1'
    for r in out:
        assert set(r) == {'parent', 'iid', 'text', 'values', 'tags', 'open'}
    return out[1:]


def _method_rows(ops):
    return [r for r in _rows(ops) if r['iid'].startswith('M:')]


def _gate(**kw):
    g = {'kind': 'rect', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A'}
    g.update(kw)
    return g


def _ac(*methods, **kw):
    g = {'kind': 'autoclean', 'name': 'G9'}
    g.update(kw)
    if methods != (None,):
        g['methods'] = list(methods)
    return g


# ── gate rows ────────────────────────────────────────────────────────────

def test_whole_build_exact_ops():
    host = _Host({'g1': _gate(name='G1', color='#aa0000', open=False),
                  'g2': _gate(name='G2', parent_id='g1', enabled=False)},
                 order=['g2'], counts={'g1': 7})
    fg = current_palette()['fg']
    assert _ops(host) == [
        ('tag', 'sample_col_S1', (('foreground', '#010203'),)),
        ('insert', 'T:t1', 'end', (('iid', 'S:S1'), ('open', True),
                                   ('tags', ('sample_col_S1',)),
                                   ('text', '■ S1  (10)'), ('values', (ON,)))),
        ('tag', 'col_S1_g1', (('foreground', '#aa0000'),)),
        ('insert', 'S:S1', 'end', (('iid', 'G:S1/g1'), ('open', False),
                                   ('tags', ('col_S1_g1',)),
                                   ('text', 'S1/g1|TR|7'), ('values', (ON,)))),
        ('tag', 'col_S1_g2', (('foreground', fg),)),
        ('insert', 'G:S1/g1', 'end', (('iid', 'G:S1/g2'), ('open', True),
                                      ('tags', ('off',)),
                                      ('text', 'S1/g2|TR|None'), ('values', (OFF,)))),
    ]


def test_parent_inserted_before_child_listed_first():
    host = _Host({'g1': _gate(parent_id='g2'), 'g2': _gate(parent_id='g3'),
                  'g3': _gate()}, order=['g1'])
    rows = _rows(_ops(host))
    assert [(r['parent'], r['iid']) for r in rows] == [
        ('S:S1', 'G:S1/g3'), ('G:S1/g3', 'G:S1/g2'), ('G:S1/g2', 'G:S1/g1')]


def test_each_gate_once_order_then_dict_order():
    host = _Host({'g1': _gate(), 'g2': _gate(), 'g3': _gate()},
                 order=['g3', 'g1', 'g3', 'nope'])
    assert [r['iid'] for r in _rows(_ops(host))] == [
        'G:S1/g3', 'G:S1/g1', 'G:S1/g2']


def test_parent_the_sample_lacks_goes_under_sample():
    host = _Host({'g1': _gate(parent_id='gone'), 'g2': _gate(parent_id='')})
    assert [(r['parent'], r['iid']) for r in _rows(_ops(host))] == [
        ('S:S1', 'G:S1/g1'), ('S:S1', 'G:S1/g2')]


def test_on_off_ticks_tags_and_colour_tags():
    host = _Host({'g1': _gate(enabled=True, color='#00ff00'),
                  'g2': _gate(enabled=False, color='#0000ff'),
                  'g3': _gate(color='')})
    ops = _ops(host)
    tags = [op for op in ops if op[0] == 'tag'][1:]
    assert tags == [('tag', 'col_S1_g1', (('foreground', '#00ff00'),)),
                    ('tag', 'col_S1_g2', (('foreground', '#0000ff'),)),
                    ('tag', 'col_S1_g3', (('foreground', current_palette()['fg']),))]
    rows = _rows(ops)
    assert [(r['values'], r['tags'], r['open']) for r in rows] == [
        ((ON,), ('col_S1_g1',), True), ((OFF,), ('off',), True),
        ((ON,), ('col_S1_g3',), True)]


def test_tray_tab_shows_some_gates_plus_their_ancestors():
    host = _Host({'g1': _gate(), 'g2': _gate(parent_id='g1'), 'g3': _gate()},
                 order=['g3', 'g2'], shown={'g2'})
    assert [r['iid'] for r in _rows(_ops(host))] == ['G:S1/g1', 'G:S1/g2']


def test_tab_showing_none_and_sample_without_gates():
    assert _rows(_ops(_Host({'g1': _gate()}, order=['g1'], shown=set()))) == []
    assert _rows(_ops(_Host({}))) == []


# ── auto-clean group and method rows ─────────────────────────────────────

def test_autoclean_gate_drop_readout_and_counts_per_gate():
    host = _Host({'g1': _ac(None), 'g2': _gate()},
                 ac={'g1': (1000, 30, {}, {})}, counts={'g1': 5})
    rows = _rows(_ops(host))
    assert rows[0]['text'] == 'S1/g1|TR|5  —  drops 30 (3.0%)'
    assert rows[1]['text'] == 'S1/g2|TR|None'
    assert host.ac_calls == [('S1', 'g1')]


def test_autoclean_gate_without_counts_has_no_readout():
    host = _Host({'g1': _ac({'key': 'time'})})
    rows = _rows(_ops(host))
    assert rows == [
        {'parent': 'S:S1', 'iid': 'G:S1/g1', 'text': 'S1/g1|TR|None',
         'values': (ON,), 'tags': ('col_S1_g1',), 'open': True},
        {'parent': 'G:S1/g1', 'iid': 'M:S1/g1/time', 'text': '   time',
         'values': (ON,), 'tags': (), 'open': True}]


def test_method_rows_label_key_ticks_tags_order():
    host = _Host({'g1': _ac({'key': 'time', 'label': 'Time jumps'},
                            {'key': 'doublets', 'label': '', 'enabled': False},
                            {'key': 'viability', 'label': None, 'enabled': True},
                            {'label': 'Nameless'},
                            open=False, enabled=False)})
    rows = _rows(_ops(host))
    assert (rows[0]['open'], rows[0]['tags'], rows[0]['values']) == (False, ('off',), (OFF,))
    assert [(r['parent'], r['iid'], r['text'], r['values'], r['tags'], r['open'])
            for r in rows[1:]] == [
        ('G:S1/g1', 'M:S1/g1/time', '   Time jumps', (ON,), (), True),
        ('G:S1/g1', 'M:S1/g1/doublets', '   doublets', (OFF,), ('off',), True),
        ('G:S1/g1', 'M:S1/g1/viability', '   viability', (ON,), (), True),
        ('G:S1/g1', 'M:S1/g1/', '   Nameless', (ON,), (), True)]


def test_autoclean_without_methods_has_no_method_rows():
    assert _method_rows(_ops(_Host({'g1': _ac(None)}))) == []
    assert _method_rows(_ops(_Host({'g1': _ac()}))) == []


def test_debris_effective_cut_in_every_mode():
    modes = [
        ({'mode': 'bead', 'bead_fsc': 812.0}, '  [beads]'),
        ({'mode': 'bead'}, '  [beads: no bead file, nothing cut]'),
        ({'mode': 'bead', 'bead_fsc': 0}, '  [beads: no bead file, nothing cut]'),
        (None, '  [beads: no bead file, nothing cut]'),
        ({'bead_fsc': 5.0}, '  [beads]'),
        ({'mode': ' Valley '}, '  [valley]'),
        ({'mode': 'valley', 'bead_fsc': 5.0}, '  [valley]'),
        ({'mode': 'fsc-only'}, '  [unknown mode, nothing cut]'),
    ]
    for params, suffix in modes:
        m = {'key': 'debris', 'label': 'Debris'}
        if params is not None:
            m['params'] = params
        rows = _method_rows(_ops(_Host({'g1': _ac(m)})))
        assert [r['text'] for r in rows] == ['   Debris' + suffix], params
    # params only matter for the debris method
    rows = _method_rows(_ops(_Host({'g1': _ac(
        {'key': 'time', 'params': {'mode': 'valley'}})})))
    assert rows[0]['text'] == '   time'


def test_per_method_drop_readout_and_zero_drop_reason():
    methods = [
        {'key': 'debris', 'label': 'Debris', 'params': {'mode': 'valley'}},
        {'key': 'time', 'label': 'Time'},
        {'key': 'doublets', 'label': 'Doublets'},
        {'key': 'viability', 'label': 'Viability', 'enabled': False},
        {'key': 'm5', 'label': 'Five'},
        {'key': 'm6', 'label': 'Six'},
        {'key': 'm7', 'label': 'Seven'},
    ]
    per = {'debris': 50, 'time': 0, 'doublets': None, 'viability': 0,
           'm5': 0, 'm6': 0}
    reasons = {'debris': 'r-debris', 'time': 'no FL1-A jump', 'viability':
               'no FL2-A dye', 'm5': '', 'm7': 'r-seven'}
    host = _Host({'g1': _ac(*methods)},
                 ac={'g1': (2000, 70, per, reasons)})
    assert [r['text'] for r in _method_rows(_ops(host))] == [
        '   Debris  [valley]  —  drops 50 (2.5%)',
        '   Time  ·  no FL1-A jump',
        '   Doublets',
        '   Viability',
        '   Five',
        '   Six',
        '   Seven',
    ]


def test_counts_without_reasons_and_without_total():
    m = [{'key': 'time', 'label': 'Time'}, {'key': 'doublets', 'label': 'Dbl'}]
    host = _Host({'g1': _ac(*m)}, ac={'g1': (400, 3, {'time': 0, 'doublets': 4})})
    assert [r['text'] for r in _method_rows(_ops(host))] == [
        '   Time', '   Dbl  —  drops 4 (1.0%)']
    host = _Host({'g1': _ac(*m)},
                 ac={'g1': (None, 3, {'time': 0, 'doublets': 4}, {'time': 'why'})})
    assert [r['text'] for r in _method_rows(_ops(host))] == [
        '   Time  ·  why', '   Dbl']


def test_two_autoclean_groups_each_get_their_own_rows():
    host = _Host({'g1': _ac({'key': 'time', 'label': 'T1'}),
                  'g2': _ac({'key': 'time', 'label': 'T2'}, parent_id='g1')},
                 ac={'g1': (100, 1, {'time': 1}, {}),
                     'g2': (100, 2, {'time': 2}, {})})
    rows = _rows(_ops(host))
    assert [(r['parent'], r['iid'], r['text']) for r in rows] == [
        ('S:S1', 'G:S1/g1', 'S1/g1|TR|None  —  drops 1 (1.0%)'),
        ('G:S1/g1', 'M:S1/g1/time', '   T1  —  drops 1 (1.0%)'),
        ('G:S1/g1', 'G:S1/g2', 'S1/g2|TR|None  —  drops 2 (2.0%)'),
        ('G:S1/g2', 'M:S1/g2/time', '   T2  —  drops 2 (2.0%)')]


def test_autoclean_group_exact_ops():
    host = _Host({'g1': _ac({'key': 'debris', 'label': 'Debris',
                             'params': {'mode': 'valley'}},
                            {'key': 'time', 'enabled': False},
                            color='#334455')},
                 ac={'g1': (500, 5, {'debris': 5, 'time': 0}, {'time': 'r1'})})
    assert _ops(host)[2:] == [
        ('tag', 'col_S1_g1', (('foreground', '#334455'),)),
        ('insert', 'S:S1', 'end', (('iid', 'G:S1/g1'), ('open', True),
                                   ('tags', ('col_S1_g1',)),
                                   ('text', 'S1/g1|TR|None  —  drops 5 (1.0%)'),
                                   ('values', (ON,)))),
        ('insert', 'G:S1/g1', 'end', (('iid', 'M:S1/g1/debris'), ('open', True),
                                      ('tags', ()),
                                      ('text', '   Debris  [valley]  —  drops 5 (1.0%)'),
                                      ('values', (ON,)))),
        ('insert', 'G:S1/g1', 'end', (('iid', 'M:S1/g1/time'), ('open', True),
                                      ('tags', ('off',)), ('text', '   time'),
                                      ('values', (OFF,)))),
    ]
