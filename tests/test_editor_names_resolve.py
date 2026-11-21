"""Every name the gate editor reads on itself exists somewhere in it.

``ViewGateEditorWindow`` is ``tk.Toplevel`` plus about forty ``editor_*``
mixins. Under type checking the mixins inherit ``EditorMixin.__getattr__ ->
Any``, so pyright cannot see a call to a method that a refactor renamed or
dropped: the window opens, and the call fails only when that menu item or
button is used. These tests read the source (AST) and check that each
``self.<name>`` read inside the window class and its mixins, and each
``<editor>.<name>`` read from other modules, names something the composed
editor defines: a method or class attribute (``dir()`` of the class, Tk's
names included), an attribute some editor code assigns (``self.x = ...``,
``setattr(self, 'x', ...)``, ``self.__dict__.setdefault('x', ...)``), or one
code elsewhere assigns on an editor reference.

Reads count whether written as attributes or as ``getattr``/``hasattr``
with a literal name: a ``getattr(self, 'x', None)`` fallback turns a rename
into a call that silently never happens.

Blind spots: editor references held under other names are not scanned
(``e.`` in ui_comp_spread.py and workspace.py, ``parent.`` in
ui_spectral_qc.py, ``model.`` in fixes.py), nor names built at run time.

What this proves: the name exists. It does not prove the signature is
unchanged, or that an attribute is set before it is read. It is a safety net
for refactors, not the proof for one.

The ``swaps`` list in tests/test_editor_load_on_loaded.py and ``SPIED`` in
tests/test_editor_gating_dispatch.py name functions by string and tolerate
a missing name (the stand-in is installed anyway and the real function is
never replaced), so a rename would let those tests pass while testing the
wrong thing. They are checked here too.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from tests.conftest import gui_unavailable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src' / 'openflo'
TESTS = ROOT / 'tests'

# Variable names that hold the gate editor in modules outside the editor
# classes, and `self.<attr>` spellings of it in helper windows.
EDITOR_REFS = ('editor', '_editor', '_ed', 'ed')
# Names that hold the editor only in one module ('owner' is a dict elsewhere).
EDITOR_REFS_IN = {'editor_plotcache.py': ('owner',)}
SELF_EDITOR_ATTRS = ('editor', '_editor', 'ed', '_ed')

# Names read on the editor that no editor code defines, each with its reason.
ALLOWED_UNRESOLVED = {
    # tkinterdnd2 adds these to Tk's widget classes when it is imported; it
    # is optional, and the editor calls them only when it loaded (HAS_DND).
    'dnd_bind': 'tkinterdnd2 DnDWrapper method',
    'drop_target_register': 'tkinterdnd2 DnDWrapper method',
    # Hooks the editor reads with a getattr default and never sets itself:
    # tests/test_saved_stats.py turns warming off with the first; the second
    # is a debug log a session may attach.
    '_plotcache_warm_enabled': 'set only by tests',
    '_plotcache_warm_log': 'optional debug hook, never set in src',
}


def _editor_class():
    try:
        from openflo.gui import ViewGateEditorWindow
    except ImportError as exc:
        gui_unavailable(f"the GUI cannot be imported: {exc}")
    return ViewGateEditorWindow


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def _editor_class_nodes(cls):
    """(path, ClassDef) for the window class and every openflo class in its
    MRO (a class defined in both branches of an `if TYPE_CHECKING` gives two)."""
    out = []
    for k in cls.__mro__:
        if not k.__module__.startswith('openflo.'):
            continue
        path = SRC / (k.__module__.split('.', 1)[1].replace('.', '/') + '.py')
        nodes = [n for n in ast.walk(_parse(path))
                 if isinstance(n, ast.ClassDef) and n.name == k.__name__]
        assert nodes, (k, path)
        out.extend((path, n) for n in nodes)
    return out


def _literal(node) -> str | None:
    return node.value if isinstance(node, ast.Constant) and \
        isinstance(node.value, str) else None


def _is_ref(node, names, self_attrs=()) -> bool:
    """True when `node` is one of `names`, or `self.<one of self_attrs>`."""
    if isinstance(node, ast.Name):
        return node.id in names
    return (isinstance(node, ast.Attribute) and node.attr in self_attrs
            and isinstance(node.value, ast.Name) and node.value.id == 'self')


class _Uses:
    """Names read and defined on an editor reference, with one place each."""

    def __init__(self):
        self.reads: dict[str, str] = {}
        self.defines: set[str] = set()

    def scan(self, node, where, names, self_attrs=()):
        """Walk `node`. A nested class has its own `self`, and a nested def
        or lambda that takes `self` belongs to some other object: neither
        body is scanned. (A def that takes `editor` is where an editor
        reference is bound, so it is scanned.)"""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef) and 'self' in names:
                continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.Lambda)) and 'self' in names:
                a = child.args
                if 'self' in {p.arg for p in (*a.posonlyargs, *a.args,
                                              *a.kwonlyargs)}:
                    continue
            self._note(child, where, names, self_attrs)
            self.scan(child, where, names, self_attrs)

    def _note(self, node, where, names, self_attrs):
        if isinstance(node, ast.Attribute) and _is_ref(node.value, names,
                                                       self_attrs):
            if isinstance(node.ctx, ast.Load):
                self.reads.setdefault(node.attr, f'{where}:{node.lineno}')
            else:
                self.defines.add(node.attr)
        elif isinstance(node, ast.Call) and node.args:
            fn = node.func
            first = _is_ref(node.args[0], names, self_attrs)
            name = _literal(node.args[1]) if len(node.args) > 1 else None
            if first and isinstance(fn, ast.Name) and name:
                if fn.id == 'setattr':
                    self.defines.add(name)
                elif fn.id in ('getattr', 'hasattr'):
                    # getattr(self, 'x', None) with a fallback still means
                    # "call x if it exists": a rename makes it silently skip.
                    self.reads.setdefault(name, f'{where}:{node.lineno}')
            # self.__dict__.setdefault('x', ...) / self.__dict__['x'] = ...
            if (isinstance(fn, ast.Attribute) and fn.attr == 'setdefault'
                    and isinstance(fn.value, ast.Attribute)
                    and fn.value.attr == '__dict__'
                    and _is_ref(fn.value.value, names, self_attrs)):
                lit = _literal(node.args[0])
                if lit:
                    self.defines.add(lit)
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store) \
                and isinstance(node.value, ast.Attribute) \
                and node.value.attr == '__dict__' \
                and _is_ref(node.value.value, names, self_attrs):
            lit = _literal(node.slice)
            if lit:
                self.defines.add(lit)


def _scan_methods(uses: _Uses, node: ast.ClassDef, where: str) -> None:
    """Add the `self.<name>` uses in each method of class `node`."""
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            uses.scan(item, where, ('self',))


def _inside_uses(cls) -> _Uses:
    """`self.<name>` uses in the window class and its mixins."""
    uses = _Uses()
    for path, node in _editor_class_nodes(cls):
        _scan_methods(uses, node, path.name)
    return uses


def _outside_uses() -> _Uses:
    """`editor.<name>`-style uses in every src module (helper windows,
    module-level helpers in the editor modules)."""
    uses = _Uses()
    for path in sorted(SRC.rglob('*.py')):
        if '__pycache__' in path.parts:
            continue
        tree = _parse(path)
        uses.scan(tree, path.name,
                  EDITOR_REFS + EDITOR_REFS_IN.get(path.name, ()),
                  SELF_EDITOR_ATTRS)
    return uses


def _tk_instance_attrs() -> set[str]:
    """Attributes tkinter's own classes set on a widget instance
    (``self.tk``, ``self.master``, ``self._w``...): not in ``dir()`` of the
    class, so read from tkinter's source."""
    import inspect
    import tkinter as tk
    uses = _Uses()
    for node in ast.walk(ast.parse(inspect.getsource(tk))):
        if isinstance(node, ast.ClassDef) and node.name in {
                k.__name__ for k in tk.Toplevel.__mro__}:
            _scan_methods(uses, node, 'tkinter')
    return uses.defines


def _defined(cls, *uses: _Uses) -> set[str]:
    out = set(dir(cls)) | _tk_instance_attrs()
    for u in uses:
        out |= u.defines
    return out


def test_every_self_read_in_the_editor_resolves():
    """A renamed or dropped mixin method (or an attribute nobody sets any
    more) fails here instead of when the user reaches that code."""
    cls = _editor_class()
    inside, outside = _inside_uses(cls), _outside_uses()
    defined = _defined(cls, inside, outside)
    missing = sorted(f'{name} (read at {where})'
                     for name, where in inside.reads.items()
                     if name not in defined and name not in ALLOWED_UNRESOLVED)
    assert missing == [], \
        'self.<name> read in the editor but defined nowhere:\n  ' + \
        '\n  '.join(missing)


def test_every_editor_read_from_other_modules_resolves():
    """Helper windows and modules call editor methods through `editor.`,
    `_editor.`, `_ed.` and `self._editor.`: a rename in a mixin must not
    leave them calling a name that is gone."""
    cls = _editor_class()
    inside, outside = _inside_uses(cls), _outside_uses()
    defined = _defined(cls, inside, outside)
    missing = sorted(f'{name} (read at {where})'
                     for name, where in outside.reads.items()
                     if name not in defined and name not in ALLOWED_UNRESOLVED)
    assert missing == [], \
        'editor.<name> read outside the editor but defined nowhere:\n  ' + \
        '\n  '.join(missing)


def test_allowed_unresolved_names_are_still_read():
    """The allowlist may only shrink: an entry nothing reads any more goes."""
    cls = _editor_class()
    inside, outside = _inside_uses(cls), _outside_uses()
    read = set(inside.reads) | set(outside.reads)
    assert sorted(set(ALLOWED_UNRESOLVED) - read) == []


def _swaps_entries():
    """(module name, attribute) for each literal entry of the `swaps` list in
    tests/test_editor_load_on_loaded.py."""
    tree = _parse(TESTS / 'test_editor_load_on_loaded.py')
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == 'swaps'):
            value = node.value
            while isinstance(value, ast.BinOp):     # [...] + [...] + [...]
                value = value.left
            assert isinstance(value, ast.List)
            for elt in value.elts:
                assert isinstance(elt, ast.Tuple)
                mod, attr = elt.elts[0], elt.elts[1]
                assert isinstance(mod, ast.Attribute) and \
                    isinstance(mod.value, ast.Name) and mod.value.id == 'openflo'
                yield f'openflo.{mod.attr}', _literal(attr)
            return
    raise AssertionError('no `swaps = [...]` in test_editor_load_on_loaded.py')


def test_swapped_names_in_the_load_test_exist():
    """The load test's stand-ins only replace names that exist; a missing
    one is silently added and the real function keeps running."""
    entries = list(_swaps_entries())
    assert len(entries) >= 10
    missing = [f'{m}.{a}' for m, a in entries
               if a is None or not hasattr(importlib.import_module(m), a)]
    assert missing == []


def test_spied_editor_methods_exist():
    """tests/test_editor_gating_dispatch.py wraps each SPIED name with
    getattr/setattr on a live editor; a renamed method would be spied under
    its old name and drop out of the golden record."""
    from tests.test_editor_gating_dispatch import SPIED
    cls = _editor_class()
    assert [n for n in SPIED if not callable(getattr(cls, n, None))] == []


@pytest.mark.parametrize('kind', ['inside', 'outside'])
def test_the_scan_finds_reads(kind):
    """The scan is not vacuous: it sees hundreds of names on each side."""
    cls = _editor_class()
    uses = _inside_uses(cls) if kind == 'inside' else _outside_uses()
    assert len(uses.reads) > (500 if kind == 'inside' else 50)
