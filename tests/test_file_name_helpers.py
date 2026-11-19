"""The three name-to-file-name helpers, each named for the files it names.

They differ on purpose (fallback word, allowed characters, length cap), so
each keeps its own rules; the old shared name ``safe_filename`` (and the
CLI's ``_safe_filename``) still works but warns."""
import pytest

from openflo import cli, fcs_export, layout_page

# Generated names: whitespace, both path separators, characters Windows
# refuses, dots, and one long name.
NAMES = ['Group A', 'ctrl day 1', 'a/b\\c:d*?', '  pop  B  ', 'x.y-z_1',
         'trailing.', '...', '   ', '<>|"?*', 'n' * 120]


def test_each_helper_keeps_its_own_rules():
    assert [cli.output_name(n) for n in NAMES] == [
        'Group_A', 'ctrl_day_1', 'a_b_c_d', 'pop_B', 'x.y-z_1',
        'trailing.', '...', 'unnamed', 'unnamed', 'n' * 120]
    assert [fcs_export.population_file_stem(n) for n in NAMES] == [
        'Group_A', 'ctrl_day_1', 'a_b_cd', 'pop__B', 'x.y-z_1',
        'trailing', 'population', 'population', 'population', 'n' * 120]
    assert [layout_page.layout_file_part(n) for n in NAMES] == [
        'Group_A', 'ctrl_day_1', 'a_b_c_d', 'pop__B', 'x.y-z_1',
        'trailing', 'sample', 'sample', 'sample', 'n' * 80]
    assert layout_page.layout_file_part('n' * 120, maxlen=10) == 'n' * 10


@pytest.mark.parametrize('module, old, new', [
    (cli, '_safe_filename', 'output_name'),
    (fcs_export, 'safe_filename', 'population_file_stem'),
    (layout_page, 'safe_filename', 'layout_file_part'),
])
def test_old_name_still_works_and_warns(module, old, new):
    with pytest.warns(DeprecationWarning, match=new):
        alias = getattr(module, old)
    assert alias is getattr(module, new)
    assert [alias(n) for n in NAMES] == [getattr(module, new)(n) for n in NAMES]


def test_old_name_is_not_exported():
    assert 'safe_filename' not in fcs_export.__all__
    assert 'population_file_stem' in fcs_export.__all__
    with pytest.raises(AttributeError):
        _ = fcs_export.no_such_name
