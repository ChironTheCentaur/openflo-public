"""Editor methods that only forwarded to another function are gone, and
nothing calls them (best-practices survey N2).

* ``_sync_hist_y_combo`` only called ``_update_mode_options``; its name
  described what it did before the mode options grew past Hist-Y.
* ``_resolve_channel`` only returned ``ui_logic.resolve_channel(display)``.
* ``_offline_session_parts`` was a second name for
  ``editor_session._offline_parts``.

The behaviour stays with the functions they forwarded to; this checks the
source, so a forwarding copy cannot come back unnoticed.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GONE = ('_sync_hist_y_combo', '_resolve_channel', '_offline_session_parts')
# voltage.VoltageTitration._resolve_channel(sample, name) is a different,
# real lookup (sample columns by detector or label), not the editor wrapper.
OWN_RESOLVER = {('src/openflo/voltage.py', '_resolve_channel'),
                ('tests/test_voltage.py', '_resolve_channel')}


def _python_files():
    for top in ('src', 'tests', 'scripts'):
        yield from sorted((ROOT / top).rglob('*.py'))


def _uses(path):
    """``(line, name, how)`` for each definition of, or attribute access
    to, a name in GONE in the file at `path`."""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name, how = node.name, 'defined'
        elif isinstance(node, ast.Attribute):
            name, how = node.attr, 'used'
        else:
            continue
        if name in GONE:
            yield node.lineno, name, how


def _offenders():
    found = []
    for path in _python_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel == 'tests/test_no_pass_through_wrappers.py':
            continue
        for line, name, how in _uses(path):
            if (rel, name) not in OWN_RESOLVER:
                found.append(f'{rel}:{line}: {name} {how}')
    return found


def test_no_forwarding_wrapper_is_defined_or_called():
    assert _offenders() == []


@pytest.mark.parametrize('src, expected', [
    ('class A:\n    def _resolve_channel(self, d):\n        return d\n',
     [(2, '_resolve_channel', 'defined')]),
    ('self.editor._sync_hist_y_combo()\n',
     [(1, '_sync_hist_y_combo', 'used')]),
    ('_, g = self._offline_session_parts()\n',
     [(1, '_offline_session_parts', 'used')]),
    ('self._update_mode_options()\nresolve_channel(x)\n', []),
])
def test_the_check_sees_a_wrapper_coming_back(tmp_path, src, expected):
    """The scan itself finds each kind of use, so a pass above is not an
    empty scan."""
    f = tmp_path / 'mod.py'
    f.write_text(src, encoding='utf-8')
    assert list(_uses(f)) == expected


def test_the_kept_functions_still_do_the_work():
    """What the wrappers forwarded to is still there and still behaves the
    same: the display-name resolver and the offline-sample split."""
    import types

    from openflo.editor_mode import ModeMixin
    from openflo.editor_session import _offline_parts
    from openflo.ui_logic import resolve_channel

    assert resolve_channel('CD901 (FL1-A)') == 'FL1-A'
    assert resolve_channel('FSC-A') == 'FSC-A'
    assert resolve_channel('') is None
    assert callable(ModeMixin._update_mode_options)

    ed = types.SimpleNamespace(
        _offline_samples={
            'synthetic_A': {'entry': {'name': 'synthetic_A'},
                            'gates': [{'name': 'g1'}]},
            'synthetic_B': {'entry': {'name': 'synthetic_B'}, 'gates': []}},
        _samples={'synthetic_B': object()})
    entries, gates = _offline_parts(ed)
    assert entries == [{'name': 'synthetic_A'}]
    assert gates == {'synthetic_A': [{'name': 'g1'}]}
