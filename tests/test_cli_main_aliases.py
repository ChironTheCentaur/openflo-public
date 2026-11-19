"""openflo-run's argparse entry point lives in openflo.cli_main and its
channel-label helpers in openflo.cli_panel. openflo.cli.main and the public
helper names are still those objects on cli; the old private names resolve
to the same objects, with a DeprecationWarning that names the new home."""
from __future__ import annotations

import pytest

import openflo.cli as cli
from openflo import cli_main, cli_panel

OLD_TO_NEW = {
    '_norm_token': (cli_panel, 'norm_token'),
    '_write_tables_workbook': (cli_main, 'write_tables_workbook'),
}


@pytest.mark.parametrize('old', sorted(OLD_TO_NEW))
def test_old_cli_name_is_the_new_object_with_a_warning(old):
    mod, new = OLD_TO_NEW[old]
    short = mod.__name__.rsplit('.', 1)[1]
    with pytest.warns(DeprecationWarning, match=f'use {short}.{new}$'):
        got = getattr(cli, old)
    assert got is getattr(mod, new)
    assert old not in vars(cli)


def test_main_stays_the_public_cli_entry_point():
    assert vars(cli)['main'] is cli_main.main


@pytest.mark.parametrize('name', ['parse_labels', 'read_staining_panel',
                                  'find_panel_xlsx'])
def test_panel_helpers_stay_on_cli(name):
    assert vars(cli)[name] is getattr(cli_panel, name)


def test_norm_token_folds_case_and_punctuation():
    assert cli_panel.norm_token('Dye/X7') == cli_panel.norm_token('dye-x7') \
        == 'dyex7'


def test_cli_main_never_imports_cli():
    # cli caps the BLAS threads and picks the backend before importing
    # cli_main; an import of cli from cli_main would make a second copy of
    # cli under `python -m openflo.cli`.
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(cli_main))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module != 'cli', ast.unparse(node)
            assert not (node.level and node.module is None
                        and any(a.name == 'cli' for a in node.names)), \
                ast.unparse(node)
