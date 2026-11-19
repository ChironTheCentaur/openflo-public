"""openflo-run's run manifest lives in openflo.cli_manifest; its old names
on openflo.cli still resolve to the same objects: MANIFEST_NAME as it was,
the private names with a DeprecationWarning that names the new one."""
from __future__ import annotations

import subprocess
import sys

import pytest

import openflo.cli as cli
from openflo import cli_manifest, cli_samples

OLD_TO_NEW = {
    '_MANIFEST_PACKAGES': 'MANIFEST_PACKAGES',
    '_sha256': 'sha256',
    '_file_entry': 'file_entry',
    '_run_input_files': 'run_input_files',
    '_write_run_manifest': 'write_run_manifest',
}


@pytest.mark.parametrize('old', sorted(OLD_TO_NEW))
def test_old_cli_name_is_the_new_object_with_a_warning(old):
    new = OLD_TO_NEW[old]
    with pytest.warns(DeprecationWarning, match=f'use cli_manifest.{new}'):
        got = getattr(cli, old)
    assert got is getattr(cli_manifest, new)
    assert old not in vars(cli)


def test_cli_uses_the_moved_objects():
    # run() writes the manifest with this object, not a copy.
    assert vars(cli)['write_run_manifest'] is cli_manifest.write_run_manifest
    assert cli.MANIFEST_NAME == cli_manifest.MANIFEST_NAME == 'run_manifest.json'
    # The manifest lists inputs with the same lookup the run uses.
    assert vars(cli_manifest)['fcs_path'] is cli_samples.fcs_path
    assert vars(cli_manifest)['normalise_groups'] is cli_samples.normalise_groups


def test_module_does_not_load_pyplot_on_import():
    code = ('import sys, openflo.cli_manifest\n'
            'print("matplotlib.pyplot" in sys.modules)\n')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == 'False'
