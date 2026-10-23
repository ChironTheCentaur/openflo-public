"""Integrity tests for the editor mixin decomposition.

`ViewGateEditorWindow` is composed from ~24 ``editor_*`` mixins. These tests
guard the *decomposition itself* — the failure modes that unit tests of the
individual behaviours don't catch, and that bit us repeatedly during the split:

  * a mixin module that references a name it forgot to import (only blows up at
    runtime when that method is first called — the F821 class of bug),
  * two mixins defining the same method name (the MRO silently shadows one),
  * a mixin class that never got added to the class bases,
  * a re-export the test-suite / back-compat depends on getting pruned by a
    future ``ruff --fix`` (which once caused a 60+ test failure spike).
"""
import importlib
import pkgutil

import pytest

import openflo


def _editor_modules():
    return sorted(f'openflo.{m.name}'
                  for m in pkgutil.iter_modules(openflo.__path__)
                  if m.name.startswith('editor_'))


@pytest.mark.parametrize('modname', _editor_modules())
def test_editor_module_imports_standalone(modname):
    """Every editor_* module imports on its own — catches a mixin that
    references a name it never imported."""
    importlib.import_module(modname)


def _mixin_classes():
    """{class name -> class} for every *Mixin defined in an editor_* module,
    excluding the shared EditorMixin base."""
    out = {}
    for modname in _editor_modules():
        mod = importlib.import_module(modname)
        for obj in vars(mod).values():
            if (isinstance(obj, type) and obj.__name__.endswith('Mixin')
                    and obj.__module__ == modname
                    and obj.__name__ != 'EditorMixin'):
                out[obj.__name__] = obj
    return out


def test_all_mixins_are_mixed_into_the_editor_class():
    """Every *Mixin that exists is actually in ViewGateEditorWindow's MRO —
    catches defining a mixin but forgetting to add it to the class bases."""
    from openflo.gui import ViewGateEditorWindow
    mro = set(ViewGateEditorWindow.__mro__)
    missing = sorted(name for name, cls in _mixin_classes().items()
                     if cls not in mro)
    assert not missing, \
        f"Mixins defined but not mixed into ViewGateEditorWindow: {missing}"


def test_no_method_name_collisions_between_mixins():
    """No two mixins define the same method name — a collision means the MRO
    silently shadows one mixin's version with another's. Dunders and names
    provided by the shared EditorMixin base are exempt."""
    from openflo.editor_base import EditorMixin
    seen = {}
    collisions = []
    for name, cls in sorted(_mixin_classes().items()):
        for meth in cls.__dict__:
            if meth.startswith('__') or hasattr(EditorMixin, meth):
                continue
            if meth in seen:
                collisions.append(f"{meth}: {seen[meth]} vs {name}")
            else:
                seen[meth] = name
    assert not collisions, \
        "Method-name collisions across mixins (MRO shadows one):\n  " \
        + "\n  ".join(collisions)


def test_gui_back_compat_reexports_present():
    """ruff --fix prunes 'unused' imports; these names are re-exported from
    openflo.gui on purpose (test monkeypatch points + back-compat). If a prune
    drops one, fail HERE with a clear message rather than as a spray of
    AttributeErrors across the GUI suite."""
    import openflo.gui as gui
    for name in ('messagebox', 'filedialog', 'savefig_background',
                 '_LOAD_POOL_SIZE'):
        assert hasattr(gui, name), \
            f"openflo.gui.{name} re-export is missing — did ruff --fix prune it?"
