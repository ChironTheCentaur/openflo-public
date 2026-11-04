"""The state ``ViewGateEditorWindow.__init__`` sets before ``_build``, for
tests that run editor mixin methods on a stub instead of a real window.

Mixins read that state directly (best-practices survey N8), so a stub must
carry it: ``SimpleNamespace(**editor_state(), _samples={...})`` or
``stub.__dict__.update(editor_state())``. The names and values come from
the ``__init__`` source, so the stub follows the editor when it changes.
Tk variables (which need a root window) and other objects become None;
pass a real value as an override where a test needs one.
"""
from __future__ import annotations

import ast
import functools
import queue
import threading
from pathlib import Path

GUI = Path(__file__).resolve().parent.parent / 'src' / 'openflo' / 'gui.py'

# Calls in __init__ that build an empty container a stub can hold as is.
_EMPTY_CALLS = {
    'set': set,
    'dict': dict,
    'list': list,
    'queue.PriorityQueue': queue.PriorityQueue,
    'threading.Event': threading.Event,
}


@functools.lru_cache(maxsize=1)
def _init_assignments():
    """``(name, value node, parameter defaults)`` for each ``self.<name> =``
    statement of ``__init__`` before ``self._build()``, in source order."""
    tree = ast.parse(GUI.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'ViewGateEditorWindow')
    init = next(f for f in cls.body if isinstance(f, ast.FunctionDef)
                and f.name == '__init__')
    args = init.args.args[-len(init.args.defaults):]
    params = {a.arg: ast.literal_eval(d)
              for a, d in zip(args, init.args.defaults, strict=True)}
    out = []
    for st in init.body:
        if (isinstance(st, ast.Expr) and isinstance(st.value, ast.Call)
                and getattr(st.value.func, 'attr', '') == '_build'):
            break
        if isinstance(st, ast.Assign):
            targets, value = st.targets, st.value
        elif isinstance(st, ast.AnnAssign) and st.value is not None:
            targets, value = [st.target], st.value
        else:
            continue
        for t in targets:
            if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                    and t.value.id == 'self'):
                out.append((t.attr, value, params))
    return tuple(out)


def init_state_names():
    """The attribute names ``__init__`` always sets before ``_build``."""
    return {name for name, _, _ in _init_assignments()}


def _value(node, params):
    try:
        return ast.literal_eval(node)
    except ValueError:
        pass
    if isinstance(node, ast.Name):
        return params.get(node.id)
    if isinstance(node, ast.Call) and not node.args and not node.keywords:
        make = _EMPTY_CALLS.get(ast.unparse(node.func))
        if make is not None:
            return make()
    return None


def editor_state(**overrides):
    """A fresh dict of the editor's ``__init__`` state, with `overrides`."""
    state = {name: _value(node, params)
             for name, node, params in _init_assignments()}
    state.update(overrides)
    return state
