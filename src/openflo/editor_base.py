"""Shared type-only base for the ``ViewGateEditorWindow`` mixins.

The editor's behaviour is split across ``editor_*.py`` mixins that all operate
on the same ``ViewGateEditorWindow`` instance — every method reads/writes
editor state and calls Tk methods on ``self``. To satisfy the type checker
without a huge per-mixin attribute-declaration block, mixins inherit
``EditorMixin``:

* under ``TYPE_CHECKING`` it's a ``tk.Misc`` (so ``self`` is usable as a dialog
  parent/master) with a permissive ``__getattr__`` returning ``Any`` (so any
  editor attribute or helper method resolves without a "missing attribute"
  error);
* at runtime it's a plain empty class (``object``-like) — the real attributes
  and methods come from the composed ``ViewGateEditorWindow``.

This keeps the mixins thin glue while leaving pyright at zero errors.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import tkinter as tk

    class EditorMixin(tk.Misc):
        # Editor state / helper methods are provided by the composed window;
        # resolve any of them to Any for the type checker.
        def __getattr__(self, name: str) -> Any: ...
else:
    class EditorMixin:
        pass
