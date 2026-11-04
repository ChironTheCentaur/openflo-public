"""Names live in one module, and callers use that module (finding N7).

``gui`` used to re-export ``_LOAD_POOL_SIZE`` (from editor_loadpool) and
``savefig_background`` (from theme), and ``theme`` re-exported
``EXPORT_DPI_CHOICES`` (from layout_page), only so tests could reach them
there. The tests now import each from its home. The two public names still
work at their old address, with a DeprecationWarning.
"""
import ast
import pathlib
import warnings

import pytest

from openflo import editor_loadpool, gui, layout_page, theme

ROOT = pathlib.Path(__file__).resolve().parents[1]

# (module the old re-export lived on, attribute) -> where it lives now
_MOVED = {
    ('gui', '_LOAD_POOL_SIZE'): 'editor_loadpool',
    ('gui', 'savefig_background'): 'theme',
    ('theme', 'EXPORT_DPI_CHOICES'): 'layout_page',
    ('gui', 'apply_theme'): 'ui_style',
    ('gui', '_apply_sleek'): 'ui_style',
    ('gui', '_install_flat_indicators'): 'ui_style',
    ('gui', '_FLAT_IND'): 'ui_style',
}


def test_the_private_pool_size_lives_only_in_the_loader():
    assert '_LOAD_POOL_SIZE' not in vars(gui)
    assert not hasattr(gui, '_LOAD_POOL_SIZE')
    assert isinstance(editor_loadpool._LOAD_POOL_SIZE, int)


def test_old_public_names_still_work_with_a_warning():
    assert 'savefig_background' not in vars(gui)
    assert 'EXPORT_DPI_CHOICES' not in vars(theme)
    with pytest.warns(DeprecationWarning, match='theme.savefig_background'):
        assert gui.savefig_background is theme.savefig_background
    with pytest.warns(DeprecationWarning,
                      match='layout_page.EXPORT_DPI_CHOICES'):
        assert theme.EXPORT_DPI_CHOICES is layout_page.EXPORT_DPI_CHOICES


def test_the_new_homes_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert callable(theme.savefig_background)
        assert layout_page.EXPORT_DPI_CHOICES == (150, 300, 600, 1200)
        assert theme.EXPORT_DPI is layout_page.EXPORT_DPI


def _reaches(tree, aliases):
    """Every ``<module>.<name>`` and ``from openflo.<module> import <name>``
    in ``tree`` that names an old re-export. ``aliases`` maps a local name
    to the openflo module it is bound to."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            mod = aliases.get(node.value.id)
            if (mod, node.attr) in _MOVED:
                found.append(f'{mod}.{node.attr} (line {node.lineno})')
        elif isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module.split('.')[-1]
            if node.module.startswith(('openflo', '.')) or node.level:
                for a in node.names:
                    if (mod, a.name) in _MOVED:
                        found.append(f'from {node.module} import {a.name} '
                                     f'(line {node.lineno})')
    return found


def _module_aliases(tree):
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in ('openflo', None):
            for a in node.names:
                aliases[a.asname or a.name] = a.name
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith('openflo.'):
                    aliases[a.asname or a.name] = a.name.split('.')[-1]
    return aliases


def test_no_code_reaches_a_name_through_its_old_re_export():
    hits = []
    for folder in ('src/openflo', 'tests', 'scripts'):
        for path in sorted((ROOT / folder).rglob('*.py')):
            if path.name == pathlib.Path(__file__).name:
                continue
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for hit in _reaches(tree, _module_aliases(tree)):
                hits.append(f'{path.relative_to(ROOT).as_posix()}: {hit}')
    assert hits == []


def test_the_scan_finds_each_old_route():
    src = ('from openflo import gui, theme\n'
           'from openflo.gui import savefig_background\n'
           'from .theme import EXPORT_DPI_CHOICES\n'
           'x = gui._LOAD_POOL_SIZE\n'
           'y = theme.EXPORT_DPI_CHOICES\n'
           'z = theme.EXPORT_DPI\n')
    tree = ast.parse(src)
    assert len(_reaches(tree, _module_aliases(tree))) == 4
