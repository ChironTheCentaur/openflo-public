"""Editor mixins read the attributes the editor's ``__init__`` always sets
directly, not through ``getattr(self, '_name', default)`` (best-practices
survey N8).

``ViewGateEditorWindow.__init__`` assigns its state (``self._samples = {}``
and so on) before ``self._build()``; the only mixin method it calls earlier,
``_restore_geometry``, reads none of that state, and the update check it
schedules runs seconds later. A ``getattr`` default for one of those
attributes can never be used by the app; it only hid a missing attribute on
a test stub. Tests
build their stubs with ``tests._editor_stub.editor_state()`` instead.

Attributes first set in ``_build`` or later, or only on demand, keep their
guards: a mixin method can run before they exist.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests._editor_stub import editor_state, init_state_names

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'src' / 'openflo'


def _guards(text, names):
    """``(line, name)`` for each ``getattr(self, '<name>', default)`` in a
    mixin class of the module source `text`, for the names in `names`."""
    tree = ast.parse(text)
    for cls in tree.body:
        if not (isinstance(cls, ast.ClassDef) and any(
                getattr(b, 'id', '').endswith('Mixin') for b in cls.bases)):
            continue
        for node in ast.walk(cls):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, 'id', '') == 'getattr'
                    and len(node.args) == 3
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == 'self'
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value in names):
                yield node.lineno, node.args[1].value


def test_no_mixin_guards_an_attribute_init_always_sets():
    names = init_state_names()
    found = [f'{p.name}:{line}: {name}'
             for p in sorted(SRC.glob('editor_*.py'))
             for line, name in _guards(p.read_text(encoding='utf-8'), names)]
    assert found == []


def test_init_sets_its_state_before_build():
    """The names the check relies on are found, and include the ones the
    removed guards read (so a pass above is not an empty scan)."""
    names = init_state_names()
    assert {'_samples', '_sample_set_aside', '_pending_sample_prep',
            '_shape_artists', '_audit_window', '_error_count',
            '_legend_pos_cache', '_quad_set_seq'} <= names
    # Set in _build, after mixin methods have run: not in the list.
    assert '_left_popped' not in names and 'gate_tv' not in names


@pytest.mark.parametrize('src, expected', [
    ("class AMixin(EditorMixin):\n"
     "    def f(self):\n"
     "        return getattr(self, '_samples', {})\n",
     [(3, '_samples')]),
    # A guard for an attribute set later stays allowed.
    ("class AMixin(EditorMixin):\n"
     "    def f(self):\n"
     "        return getattr(self, '_plotcache_worker', None)\n", []),
    # Another object, or a class that is not a mixin, is not checked.
    ("class AMixin(EditorMixin):\n"
     "    def f(self, s):\n"
     "        return getattr(s, '_samples', {})\n", []),
    ("class Recording:\n"
     "    def f(self):\n"
     "        return getattr(self, '_samples', {})\n", []),
])
def test_the_check_sees_a_guard_coming_back(src, expected):
    assert list(_guards(src, {'_samples'})) == expected


def test_editor_state_matches_init():
    """The shared stub state has every attribute __init__ sets before
    _build, with the same empty values for the plain containers."""
    state = editor_state()
    assert init_state_names() <= set(state)
    assert state['_samples'] == {} and state['_loading'] == set()
    assert state['_error_count'] == 0 and state['_primary'] is False
    # Each call gets its own containers.
    assert editor_state()['_samples'] is not state['_samples']
