"""openflo-run's sample lookup lives in openflo.cli_samples and its .wsp
export in openflo.cli_wsp; their old names on openflo.cli still resolve to
the same objects: fcs_path as it was, the private names with a
DeprecationWarning that names the new one."""
from __future__ import annotations

import subprocess
import sys

import pytest

import openflo.cli as cli
from openflo import cli_samples, cli_trials, cli_wsp

OLD_TO_NEW = {
    '_normalise_groups': (cli_samples, 'normalise_groups'),
    '_scan_for_token': (cli_samples, 'scan_for_token'),
    '_find_unstained': (cli_samples, 'find_unstained'),
    '_build_export_gate_list': (cli_wsp, 'build_export_gate_list'),
    '_own_spillover': (cli_wsp, 'own_spillover'),
    '_export_pipeline_workspace': (cli_wsp, 'export_pipeline_workspace'),
}


@pytest.mark.parametrize('old', sorted(OLD_TO_NEW))
def test_old_cli_name_is_the_new_object_with_a_warning(old):
    module, new = OLD_TO_NEW[old]
    short = module.__name__.rsplit('.', 1)[1]
    with pytest.warns(DeprecationWarning, match=f'use {short}.{new}'):
        got = getattr(cli, old)
    assert got is getattr(module, new)
    assert old not in vars(cli)


def test_cli_uses_the_moved_objects():
    # run() and main() call these objects, not copies.
    assert vars(cli)['fcs_path'] is cli_samples.fcs_path
    assert vars(cli)['normalise_groups'] is cli_samples.normalise_groups
    assert vars(cli)['export_pipeline_workspace'] is cli_wsp.export_pipeline_workspace
    # The FMO thresholds fall back to the same unstained-control lookup.
    assert vars(cli_trials)['find_unstained'] is cli_samples.find_unstained
    # The export resolves samples with the same lookup the run uses.
    assert vars(cli_wsp)['fcs_path'] is cli_samples.fcs_path


@pytest.mark.parametrize('module', ['cli_samples', 'cli_wsp'])
def test_module_does_not_load_pyplot_on_import(module):
    code = (f'import sys, openflo.{module}\n'
            'print("matplotlib.pyplot" in sys.modules)\n')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == 'False'
