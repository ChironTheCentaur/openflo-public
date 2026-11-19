"""The openflo-run parser is built by one helper per option group.

build_parser() makes the parser, calls the eight cli_args._add_*_args helpers
in a fixed order on that one parser, and adds --show-plots between the
unmixing and gate groups. These tests pin which options each helper adds
(with their defaults, choices, nargs and types), the order of the calls, and
that the whole parser is exactly the helpers' options in that order.
"""
from __future__ import annotations

import argparse
import random

import pytest

from openflo import cli_args
from openflo.cli_args import DEFAULT_DPI, build_parser
from openflo.som_kernel import FLOWSOM_DEFAULT_METACLUSTERS

STORE = '_StoreAction'
FLAG = '_StoreTrueAction'

# helper -> the options it adds, in order:
# (option strings, dest, default, choices, nargs, const, type name, action class)
HELPERS = {
    '_add_input_args': [
        (('--trials',), 'trials', '', None, None, None, None, STORE),
        (('--fcs',), 'fcs', '', None, None, None, None, STORE),
        (('--out',), 'out', 'outputs', None, None, None, None, STORE),
        (('--groups',), 'groups', '', None, None, None, None, STORE),
        (('--fmo-sets',), 'fmo_sets', '', None, None, None, None, STORE),
        (('--samples',), 'samples', '', None, None, None, None, STORE),
        (('--batch-mode',), 'batch_mode', 'independent',
         ('independent', 'concatenate'), None, None, None, STORE),
        (('--workers',), 'workers', 1, None, None, None, 'int', STORE),
    ],
    '_add_cluster_args': [
        (('--cluster-method',), 'cluster_method', 'phenograph',
         ('phenograph', 'leiden', 'flowsom'), None, None, None, STORE),
        (('--reproducible',), 'reproducible', False, None, 0, True, None, FLAG),
        (('--no-reproducible',), 'no_reproducible', False, None, 0, True, None, FLAG),
        (('--resolution',), 'resolution', 1.0, None, None, None, '_positive_float', STORE),
        (('--fast-graph',), 'fast_graph', False, None, 0, True, None, FLAG),
        (('--n-metaclusters',), 'n_metaclusters', FLOWSOM_DEFAULT_METACLUSTERS,
         None, None, None, 'int', STORE),
        (('--max-events',), 'max_events', 0, None, None, None, 'int', STORE),
        (('--admission-ram',), 'admission_gb', 4.0, None, None, None, 'float', STORE),
        (('--admission-vram',), 'vram_admission_gb', 1.0, None, None, None, 'float', STORE),
    ],
    '_add_cleanup_args': [
        (('--filter-debris-um',), 'filter_debris_um', 0.0, None, None, None, 'float', STORE),
        (('--bead-um',), 'bead_um', 8.0, None, None, None, 'float', STORE),
        (('--doublet-tol',), 'doublet_tol', 0.0, None, None, None, 'float', STORE),
        (('--qc',), 'qc', False, None, 0, True, None, FLAG),
    ],
    '_add_label_args': [
        (('--k',), 'k', 30, None, None, None, 'int', STORE),
        (('--fmo-percentile',), 'fmo_percentile', 99.0, None, None, None, 'float', STORE),
        (('--labels',), 'labels', '', None, None, None, None, STORE),
        (('--panel',), 'panel', '', None, None, None, None, STORE),
        (('--pairs',), 'pairs', '', None, None, None, None, STORE),
        (('--xlsx',), 'xlsx', False, None, 0, True, None, FLAG),
    ],
    '_add_cytonorm_args': [
        (('--batch-correct',), 'batch_correct', False, None, 0, True, None, FLAG),
        (('--cytonorm-mode',), 'cytonorm_mode', 'goal', ('goal', 'controls'),
         None, None, None, STORE),
        (('--cytonorm-control',), 'cytonorm_control', '', None, None, None, None, STORE),
        (('--cytonorm-metaclusters',), 'cytonorm_metaclusters', 10,
         None, None, None, 'int', STORE),
    ],
    '_add_unmix_args': [
        (('--unmix',), 'unmix', False, None, 0, True, None, FLAG),
        (('--unmix-controls',), 'unmix_controls', '', None, None, None, None, STORE),
        (('--unmix-input',), 'unmix_input', '', None, None, None, None, STORE),
        (('--unmix-detectors',), 'unmix_detectors', 'auto', None, None, None, None, STORE),
        (('--unmix-nonneg',), 'unmix_nonneg', False, None, 0, True, None, FLAG),
        (('--unmix-spectra',), 'unmix_spectra', 'auto', ('auto', 'total', 'matched'),
         None, None, None, STORE),
        (('--unmix-beads',), 'unmix_beads', False, None, 0, True, None, FLAG),
    ],
    '_add_gate_args': [
        (('--gates',), 'gates', '', None, None, None, None, STORE),
        (('--gate-units',), 'gate_units', 'stored', ('stored', 'data'),
         None, None, None, STORE),
        (('--gates-from',), 'gates_from', '', None, None, None, None, STORE),
        (('--gate-population',), 'gate_population', '', None, None, None, None, STORE),
    ],
    '_add_run_args': [
        (('--skip-existing',), 'skip_existing', False, None, 0, True, None, FLAG),
        (('--export-wsp',), 'export_wsp', '', None, None, None, None, STORE),
        (('--seed',), 'seed', 42, None, None, None, 'int', STORE),
        (('--dpi',), 'dpi', DEFAULT_DPI, None, None, None, '_dpi', STORE),
        (('--palette',), 'palette', 'auto', None, None, None, None, STORE),
        (('-v', '--verbose'), 'verbose', 0, None, 0, None, None, '_CountAction'),
        (('-q', '--quiet'), 'quiet', False, None, 0, True, None, FLAG),
    ],
}
ORDER = list(HELPERS)
SHOW_PLOTS = (('--show-plots',), 'show_plots', False, None, 0, True, None, FLAG)


def _spec(action):
    return (tuple(action.option_strings), action.dest, action.default,
            tuple(action.choices) if action.choices is not None else None,
            action.nargs, action.const, getattr(action.type, '__name__', None),
            type(action).__name__)


def _specs(ap):
    return [_spec(a) for a in ap._actions if a.dest != 'help']


def _expected_parser_specs():
    specs = []
    for name in ORDER:
        specs += HELPERS[name]
        if name == '_add_unmix_args':
            specs.append(SHOW_PLOTS)
    return specs


@pytest.mark.parametrize('name', ORDER)
def test_each_helper_adds_its_options(name):
    ap = argparse.ArgumentParser(add_help=False)
    assert getattr(cli_args, name)(ap) is None
    assert _specs(ap) == HELPERS[name]


@pytest.mark.parametrize('name', ORDER)
def test_each_helper_adds_only_to_the_parser_it_is_given(name):
    ap = argparse.ArgumentParser(add_help=False)
    other = argparse.ArgumentParser(add_help=False)
    getattr(cli_args, name)(ap)
    assert other._actions == []
    assert len(ap._actions) == len(HELPERS[name])


def test_build_parser_calls_the_helpers_in_order_on_one_parser(monkeypatch):
    calls = []
    for name in ORDER:
        real = getattr(cli_args, name)

        def record(ap, _name=name, _real=real):
            calls.append((_name, ap, len(ap._actions)))
            _real(ap)
        monkeypatch.setattr(cli_args, name, record)
    ap = cli_args.build_parser()
    assert [c[0] for c in calls] == ORDER
    assert all(c[1] is ap for c in calls)
    # what was already on the parser when each helper ran: --help, then each
    # earlier group, and --show-plots before the gate group
    before = 1
    for name, _ap, seen in calls:
        assert seen == before, name
        before += len(HELPERS[name]) + (name == '_add_unmix_args')


def test_build_parser_is_the_helpers_options_in_order():
    ap = build_parser()
    assert ap._actions[0].dest == 'help'
    assert _specs(ap) == _expected_parser_specs()
    dests = [s[1] for s in _specs(ap)]
    assert len(dests) == len(set(dests))


def test_show_plots_sits_between_the_unmix_and_gate_options():
    dests = [a.dest for a in build_parser()._actions]
    i = dests.index('show_plots')
    assert dests[i - 1] == HELPERS['_add_unmix_args'][-1][1]
    assert dests[i + 1] == HELPERS['_add_gate_args'][0][1]


def test_help_lists_every_option_in_builder_order():
    text = build_parser().format_help()
    pos = 0
    for spec in _expected_parser_specs():
        flag = spec[0][-1]
        found = text.find(flag, pos)
        assert found >= 0, flag
        pos = found + len(flag)


def test_defaults_of_an_empty_command_line():
    ns = vars(build_parser().parse_args([]))
    assert ns == {s[1]: s[2] for s in _expected_parser_specs()}


def _values(spec, rng):
    """Generated valid values for one option, as strings, with what each parses to."""
    choices, type_name = spec[3], spec[6]
    if choices:
        return [(c, c) for c in choices]
    if type_name == 'int':
        n = rng.randint(2, 900)
        return [('0', 0), (str(n), n)]
    if type_name == 'float':
        x = round(rng.uniform(0.1, 90), 3)
        return [('0', 0.0), (str(x), x)]
    if type_name == '_positive_float':
        return [('0.25', 0.25), ('3', 3.0)]
    if type_name == '_dpi':
        n = rng.randint(50, 2400)
        return [('50', 50), ('2400', 2400), (str(n), n)]
    tok = ''.join(rng.choice('abcdef0123') for _ in range(6))
    return [(f'gen_{tok}', f'gen_{tok}'), ('', '')]


def test_every_option_parses_generated_values():
    rng = random.Random(7)
    for spec in _expected_parser_specs():
        flag, dest = spec[0][-1], spec[1]
        if spec[4] == 0:
            ns = build_parser().parse_args([flag])
            assert getattr(ns, dest) == (1 if spec[7] == '_CountAction' else True), flag
            continue
        for text, want in _values(spec, rng):
            ns = build_parser().parse_args([flag, text])
            assert getattr(ns, dest) == want, flag
            ns = build_parser().parse_args([f'{flag}={text}'])
            assert getattr(ns, dest) == want, flag


@pytest.mark.parametrize('spec', [s for s in _expected_parser_specs() if s[4] is None],
                         ids=lambda s: s[0][-1])
def test_a_valued_option_takes_exactly_one_value(spec, capsys):
    flag = spec[0][-1]
    value = spec[3][0] if spec[3] else '7'
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([flag])
    assert exc.value.code == 2
    assert 'expected one argument' in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([flag, value, value])
    assert exc.value.code == 2


@pytest.mark.parametrize('spec', [s for s in _expected_parser_specs() if s[3]],
                         ids=lambda s: s[0][-1])
def test_a_choice_option_refuses_other_values(spec, capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([spec[0][-1], 'gen_not_a_choice'])
    assert exc.value.code == 2
    assert 'invalid choice' in capsys.readouterr().err


def test_a_flag_takes_no_value(capsys):
    for spec in _expected_parser_specs():
        if spec[4] != 0 or not spec[0][-1].startswith('--'):
            continue
        with pytest.raises(SystemExit) as exc:
            build_parser().parse_args([f'{spec[0][-1]}=1'])
        assert exc.value.code == 2, spec[0][-1]
    capsys.readouterr()
