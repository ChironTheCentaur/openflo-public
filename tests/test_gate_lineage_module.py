"""R7: one shared walk up a gate's parents (openflo.gate_lineage).

Modules used to hand-write the same ``while cur in gates and cur not in
seen`` loop; they all call gate_lineage now, and the static check keeps
it that way."""
import ast
import pathlib

import openflo
from openflo import gate_combos, gate_lineage
from openflo.gate_lineage import ancestors, lineage, root_first

SRC = pathlib.Path(openflo.__file__).parent

# Walks up parent_id still written by hand, per module: none left. Never
# add one; call gate_lineage instead.
HAND_WALKS_LEFT: dict[str, int] = {}


def _gates(**parents):
    return {gid: {'id': gid, 'kind': 'rect', 'parent_id': pid}
            for gid, pid in parents.items()}


def test_lineage_runs_from_the_gate_to_its_root():
    gates = _gates(a=None, b='a', c='b', d='a')
    assert lineage(gates, 'c') == ['c', 'b', 'a']
    assert ancestors(gates, 'c') == ['b', 'a']
    assert root_first(gates, 'c') == ['a', 'b', 'c']
    assert lineage(gates, 'a') == ['a']
    assert ancestors(gates, 'a') == []


def test_lineage_stops_at_a_missing_parent_or_an_unknown_gate():
    gates = _gates(b='gone', c='b')
    assert lineage(gates, 'c') == ['c', 'b']
    assert lineage(gates, 'gone') == []
    assert lineage(gates, None) == []
    assert ancestors(gates, 'nope') == []


def test_lineage_ends_on_a_cycle():
    gates = _gates(a='c', b='a', c='b')
    assert lineage(gates, 'a') == ['a', 'c', 'b']
    assert ancestors(gates, 'a') == ['c', 'b']
    assert lineage(_gates(s='s'), 's') == ['s']


def test_gate_combos_ancestors_is_the_shared_walk():
    assert gate_combos.ancestors is gate_lineage.ancestors


def _hand_walks(path):
    """Loops that step a name up to its parent: a ``while`` whose body
    assigns a name from an expression reading ``'parent_id'``."""
    n = 0
    for loop in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if not isinstance(loop, ast.While):
            continue
        n += any(isinstance(node, ast.Assign)
                 and not isinstance(node.value, (ast.BoolOp, ast.Compare))
                 and all(isinstance(t, ast.Name) for t in node.targets)
                 and any(isinstance(c, ast.Constant) and c.value == 'parent_id'
                         for c in ast.walk(node.value))
                 for stmt in loop.body for node in ast.walk(stmt))
    return n


def test_no_new_hand_written_parent_walks():
    found = {p.name: k for p in sorted(SRC.glob('*.py'))
             if p.name != 'gate_lineage.py' and (k := _hand_walks(p))}
    assert found == HAND_WALKS_LEFT


def test_wsp_export_reroots_onto_the_nearest_kept_ancestor():
    from openflo.editor_export import _exported_parent
    gates = _gates(a=None, b='a', c='b', d='c', e='gone', f='e')
    assert _exported_parent(gates, 'd', {'b', 'c'}) == 'a'
    assert _exported_parent(gates, 'd', {'c'}) == 'b'
    # Every ancestor skipped: the last one's own parent, root or missing.
    assert _exported_parent(gates, 'd', {'a', 'b', 'c'}) is None
    assert _exported_parent(gates, 'f', {'e'}) == 'gone'
    assert _exported_parent(gates, 'b', set()) == 'a'


def test_gate_tools_descendant_check_looks_up_from_the_candidate():
    from types import SimpleNamespace

    from openflo.editor_gatetools import GateToolsMixin
    ed = SimpleNamespace(_sample_gates={'S1': _gates(a=None, b='a', c='b')})
    below = GateToolsMixin._is_descendant_of
    assert below(ed, 'S1', 'a', 'c')
    assert below(ed, 'S1', 'b', 'b')
    assert not below(ed, 'S1', 'c', 'a')
    assert not below(ed, 'S1', 'a', None)


def test_plot_cache_chain_holds_the_gate_and_every_ancestor():
    from openflo.editor_plotcache import PlotCacheMixin
    gates = _gates(a=None, b='a', c='b')
    assert PlotCacheMixin._plotcache_chain(gates, 'c') == ['c', 'b', 'a']
    assert PlotCacheMixin._plotcache_chain(gates, 'x') == []
