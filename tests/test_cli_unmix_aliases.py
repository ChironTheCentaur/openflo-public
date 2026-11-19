"""openflo-run's batch unmixing lives in openflo.cli_unmix; its old names on
openflo.cli still resolve to the same objects: run_batch_unmix as it was,
the private helpers with a DeprecationWarning that names the new one."""
from __future__ import annotations

import subprocess
import sys

import pytest

import openflo.cli as cli
from openflo import cli_unmix

OLD_TO_NEW = {
    '_load_unmix_controls': 'load_unmix_controls',
    '_write_unmix_qc': 'write_unmix_qc',
}


@pytest.mark.parametrize('old,new', sorted(OLD_TO_NEW.items()))
def test_old_cli_name_is_the_new_object_with_a_warning(old, new):
    with pytest.warns(DeprecationWarning, match=f'use cli_unmix.{new}'):
        got = getattr(cli, old)
    assert got is getattr(cli_unmix, new)
    assert old not in vars(cli)


def test_cli_runs_the_moved_unmix():
    # main() hands --unmix to this object, not a copy.
    assert vars(cli)['run_batch_unmix'] is cli_unmix.run_batch_unmix


def test_cli_unmix_does_not_load_pyplot_on_import():
    code = ('import sys, openflo.cli_unmix\n'
            'print("matplotlib.pyplot" in sys.modules)\n')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == 'False'
