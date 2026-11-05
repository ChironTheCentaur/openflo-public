"""openflo-run's gate reading lives in openflo.cli_gates; its old names on
openflo.cli still resolve to the same objects, each with a
DeprecationWarning that names the new one."""
from __future__ import annotations

import pytest

import openflo.cli as cli
from openflo import cli_gates, cli_wsp

OLD_TO_NEW = {
    '_parse_gates_arg': 'parse_gates_arg',
    '_data_units_to_stored': 'data_units_to_stored',
    '_pipeline_transform_specs': 'pipeline_transform_specs',
    '_gate_name': 'gate_name',
    '_session_gates': 'session_gates',
    '_read_gate_file': 'read_gate_file',
    '_population_gates': 'population_gates',
    '_pipeline_gate_transforms': 'pipeline_gate_transforms',
}


@pytest.mark.parametrize('old,new', sorted(OLD_TO_NEW.items()))
def test_old_cli_name_is_the_new_object_with_a_warning(old, new):
    with pytest.warns(DeprecationWarning, match=f'use cli_gates.{new}'):
        got = getattr(cli, old)
    assert got is getattr(cli_gates, new)


def test_cli_runs_the_moved_functions():
    # main() calls these by their cli_gates objects, not copies.
    for name in ('parse_gates_arg', 'read_gate_file', 'population_gates',
                 'gate_name'):
        assert vars(cli)[name] is getattr(cli_gates, name)
    # --export-wsp (now in cli_wsp) uses the same transforms helper.
    assert vars(cli_wsp)['pipeline_gate_transforms'] is cli_gates.pipeline_gate_transforms
    assert not any(n in vars(cli) for n in OLD_TO_NEW)
