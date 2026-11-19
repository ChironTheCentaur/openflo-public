"""openflo-run's group naming and group checks live in openflo.cli_groups;
their old names on openflo.cli still resolve to the same objects:
output_name as it was, the private names with a DeprecationWarning that
names the new one."""
from __future__ import annotations

import subprocess
import sys

import pytest

import openflo.cli as cli
from openflo import cli_compare, cli_groups, cli_samples

OLD_TO_NEW = {
    '_compare_base': 'compare_base',
    '_group_spec_problems': 'group_spec_problems',
    '_sample_file_problems': 'sample_file_problems',
    '_discover_day_folders': 'discover_day_folders',
    '_auto_groups_by_day': 'auto_groups_by_day',
}


@pytest.mark.parametrize('old', sorted(OLD_TO_NEW))
def test_old_cli_name_is_the_new_object_with_a_warning(old):
    new = OLD_TO_NEW[old]
    with pytest.warns(DeprecationWarning, match=f'use cli_groups.{new}'):
        got = getattr(cli, old)
    assert got is getattr(cli_groups, new)
    assert old not in vars(cli)


def test_cli_uses_the_moved_objects():
    # run() and main() check groups and name files with these objects.
    for name in ('output_name', 'group_spec_problems',
                 'sample_file_problems', 'auto_groups_by_day'):
        assert vars(cli)[name] is getattr(cli_groups, name)
    # The group comparisons name their files with the moved object.
    assert vars(cli_compare)['compare_base'] is cli_groups.compare_base
    # The duplicate-file check resolves samples as the run does.
    assert vars(cli_groups)['fcs_path'] is cli_samples.fcs_path


def test_module_does_not_load_pyplot_on_import():
    code = ('import sys, openflo.cli_groups\n'
            'print("matplotlib.pyplot" in sys.modules)\n')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == 'False'
