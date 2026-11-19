"""pytest's warning filters hide no whole category (survey T3).

The suite used to ignore every FutureWarning and DeprecationWarning, which is
how pandas, numpy and scikit-learn announce their next breaking change, so an
upcoming break in openflo's own calls could only show up once it broke. Now
pyproject.toml turns every one into a test failure, wherever it is
attributed (openflo's code, a test, or a library's own code), and ignores
only named third-party noise.

The filter reader is checked on synthetic filter lists first; the rest run
under the filters this very test runs with, raising warnings from generated
modules the way a library does (attributed to its caller).
"""
import tomllib
import warnings
from pathlib import Path

import pytest
from _pytest.config import parse_warning_filter

ROOT = Path(__file__).resolve().parents[1]


def wholesale_ignores(filters):
    """The entries of a pytest ``filterwarnings`` list that ignore a whole
    category: no message and no module to narrow them."""
    out = []
    for line in filters:
        action, message, _category, module, _lineno = parse_warning_filter(
            line, escape=False)
        if action == 'ignore' and not message and not module:
            out.append(line)
    return out


def _ini_filters():
    with open(ROOT / 'pyproject.toml', 'rb') as f:
        return tomllib.load(f)['tool']['pytest']['ini_options']['filterwarnings']


def _caller_in(module_name, category, message):
    """Raise `message` the way pandas and numpy do: from library code, with
    ``stacklevel=2`` so it is attributed to the caller, a function defined
    in a module named `module_name`."""
    lib = {'__name__': 'fakelib', 'warnings': warnings}
    exec('def deprecated_call(message, category):\n'
         '    warnings.warn(message, category, stacklevel=2)\n', lib)
    caller = {'__name__': module_name, 'lib': lib}
    exec('def run(message, category):\n'
         '    lib["deprecated_call"](message, category)\n', caller)
    caller['run'](message, category)


def test_wholesale_ignores_reads_filter_lines():
    assert wholesale_ignores([
        'ignore::FutureWarning',
        'ignore::DeprecationWarning',
        'ignore:Please import:DeprecationWarning:phenograph',
        'ignore:some message:FutureWarning',
        'ignore::FutureWarning:sklearn',
        'error::FutureWarning:openflo',
        'error::FutureWarning',
    ]) == ['ignore::FutureWarning', 'ignore::DeprecationWarning']


def test_no_warning_category_is_ignored_wholesale():
    assert wholesale_ignores(_ini_filters()) == []


DEPRECATIONS = [FutureWarning, DeprecationWarning, PendingDeprecationWarning]


@pytest.mark.parametrize('category', DEPRECATIONS)
def test_every_deprecation_category_is_an_error_everywhere(category):
    """Each category has an ``error`` line with no module, so it fails the
    test wherever it is attributed."""
    errors = set()
    for line in _ini_filters():
        action, message, cat, module, _lineno = parse_warning_filter(
            line, escape=False)
        if action == 'error' and not message and not module:
            errors.add(cat)
    assert category in errors


@pytest.mark.parametrize('category', DEPRECATIONS)
def test_a_library_warning_about_openflo_code_fails_the_test(category):
    with pytest.raises(category, match='planted T3 break'):
        _caller_in('openflo._t3_probe', category, 'planted T3 break')


@pytest.mark.parametrize('category', DEPRECATIONS)
def test_a_library_warning_about_other_code_fails_the_test(category):
    with pytest.raises(category, match='planted third-party break'):
        _caller_in('otherlib._t3_probe', category, 'planted third-party break')


def test_an_old_openflo_name_fails_the_test():
    """A test still calling a deprecated alias fails rather than passing
    with a line in the warnings summary."""
    from openflo import pipeline
    with pytest.raises(DeprecationWarning, match='openflo.flowsom.som_train'):
        pipeline._som_train  # noqa: B018


def test_named_third_party_noise_is_ignored():
    with warnings.catch_warnings(record=True) as seen:
        warnings.warn_explicit(
            'Please import `spmatrix` from the `scipy.sparse` namespace; the '
            '`scipy.sparse.base` namespace is deprecated',
            DeprecationWarning, 'cluster.py', 13, module='phenograph.cluster')
    assert seen == []


def test_no_filter_ignores_the_fork_warning():
    """Python attributes its fork warning to multiprocessing.popen_fork
    whoever opened the pool, so an ignore for it would cover OpenFlo's own
    pools too. tests/_fork_guard.py fails any fork outside phenograph
    instead (tests/test_fork_guard.py)."""
    for line in _ini_filters():
        _action, message, _cat, module, _lineno = parse_warning_filter(
            line, escape=False)
        assert 'fork' not in message.lower(), line
        assert not module.startswith('multiprocessing'), line
