"""scripts/best_practices.py: each rule flags the pattern the code survey
found and leaves its near miss alone, and the baseline is a ratchet -- a
(rule, file) count above the baseline fails, equal or below passes.

Every snippet here is synthetic; paths only decide GUI or core and the
per-module exemptions (cli.py may print, table_export.py may write CSV...).
"""
import importlib.util
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    'best_practices', ROOT / 'scripts' / 'best_practices.py')
assert _spec and _spec.loader
bp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bp)


def rules(path, source, rule=None):
    """Rule names `check_source` reports for `source` saved as `path`."""
    found = [v.rule for v in bp.check_source(path, textwrap.dedent(source))]
    return [r for r in found if rule is None or r == rule]


def cross(sources, rule):
    found = bp.check_sources({p: textwrap.dedent(s) for p, s in sources.items()})
    return [(v.path, v.rule) for v in found if v.rule == rule]


# ── BP1 ─────────────────────────────────────────────────────────────────────

_BODY = """
def {name}(text):
    '''{doc}'''
    try:
        v = float(text)
    except ValueError:
        return None
    return v if v > 0 else None
"""


def test_bp1_a_body_copied_into_another_module_is_flagged_in_both():
    found = cross({'src/openflo/foo.py': _BODY.format(name='parse', doc='One.'),
                   'src/openflo/bar.py': _BODY.format(name='_parse', doc='Two.')},
                  'duplicate-function')
    assert sorted(found) == [('src/openflo/bar.py', 'duplicate-function'),
                             ('src/openflo/foo.py', 'duplicate-function')]


def test_bp1_a_short_wrapper_repeated_across_modules_is_not_a_duplicate():
    wrapper = "def {}(self):\n    self.foo.set(1)\n    self.redraw()\n"
    found = cross({'src/openflo/foo.py': wrapper.format('reset'),
                   'src/openflo/bar.py': wrapper.format('reset')},
                  'duplicate-function')
    assert found == []


def test_bp1_a_public_name_defined_in_two_modules_is_flagged():
    found = cross({'src/openflo/foo.py': 'def write_foo(path):\n    pass\n',
                   'src/openflo/bar.py': 'def write_foo(path, n):\n    return n\n'},
                  'duplicate-public-name')
    assert len(found) == 2


def test_bp1_private_names_and_main_may_repeat():
    src = 'def main():\n    pass\n\ndef _helper():\n    pass\n'
    found = cross({'src/openflo/foo.py': src, 'src/openflo/bar.py': src},
                  'duplicate-public-name')
    assert found == []


# ── BP2 ─────────────────────────────────────────────────────────────────────

def test_bp2_a_core_module_importing_tk_or_theme_at_import_time_is_flagged():
    assert rules('src/openflo/foo.py', 'import tkinter as tk\n', 'core-imports-tk')
    assert rules('src/openflo/foo.py', 'from . import theme\n', 'core-imports-tk')
    assert rules('src/openflo/foo.py', 'from .editor_plot import bar\n',
                 'core-imports-tk')


def test_bp2_tk_inside_a_function_under_type_checking_or_in_gui_is_allowed():
    lazy = """
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        import tkinter as tk

    def show():
        from .theme import foo
        import tkinter
    """
    assert rules('src/openflo/foo.py', lazy, 'core-imports-tk') == []
    assert rules('src/openflo/ui_foo.py', 'import tkinter\n', 'core-imports-tk') == []
    assert rules('src/openflo/gui.py', 'from . import theme\n', 'core-imports-tk') == []


def test_bp2_a_core_module_reading_prefs_is_flagged_even_lazily():
    src = """
    def stats():
        from .prefs import load_foo
        return load_foo()
    """
    assert rules('src/openflo/foo.py', src, 'core-reads-prefs')


def test_bp2_gui_modules_may_read_prefs():
    src = 'from .prefs import read_prefs\n'
    assert rules('src/openflo/editor_foo.py', src, 'core-reads-prefs') == []
    assert rules('src/openflo/theme.py', src, 'core-reads-prefs') == []


# ── BP3 ─────────────────────────────────────────────────────────────────────

def test_bp3_warning_filters_and_exits_at_import_time_are_flagged():
    src = """
    import sys
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    if len(sys.argv) < 2:
        sys.exit('usage: foo')
    class Foo:
        raise SystemExit(1)
    """
    assert len(rules('src/openflo/foo.py', src, 'import-side-effect')) == 3


def test_bp3_the_same_calls_in_a_function_main_guard_or_scoped_block_pass():
    src = """
    import sys
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        import foo

    def main():
        warnings.filterwarnings('ignore')
        sys.exit(0)

    if __name__ == '__main__':
        warnings.simplefilter('ignore')
        raise SystemExit(main())
    """
    assert rules('src/openflo/foo.py', src, 'import-side-effect') == []


# ── BP5 ─────────────────────────────────────────────────────────────────────

def test_bp5_writing_csv_outside_the_shared_writer_is_flagged():
    src = """
    import csv
    def save(df, path, rows):
        df.to_csv(path, index=False)
        with open(path, 'w') as f:
            csv.writer(f).writerows(rows)
    """
    assert len(rules('src/openflo/foo.py', src, 'csv-bypass')) == 2


def test_bp5_the_shared_csv_writer_may_use_csv_and_to_csv():
    src = 'import csv\ndef w(df, f):\n    df.to_csv(f)\n    csv.writer(f)\n'
    assert rules('src/openflo/table_export.py', src, 'csv-bypass') == []


def test_bp5_reading_the_version_outside_the_version_module_is_flagged():
    assert rules('src/openflo/foo.py', 'from . import __version__\n', 'version-read')
    assert rules('src/openflo/foo.py',
                 'import importlib.metadata\nV = importlib.metadata.version("openflo")\n',
                 'version-read')
    assert rules('src/openflo/foo.py', 'import pkg_resources\n', 'version-read')


def test_bp5_the_version_module_and_other_packages_versions_pass():
    src = 'import importlib.metadata as m\nV = m.version("openflo")\nfrom . import __version__\n'
    assert rules('src/openflo/update.py', src, 'version-read') == []
    assert rules('src/openflo/foo.py', 'import importlib.metadata as m\n'
                 'V = m.version("numpy")\n', 'version-read') == []


# ── BP7 ─────────────────────────────────────────────────────────────────────

def test_bp7_print_in_a_library_function_is_flagged():
    src = 'def gate(x):\n    print("FL1-A", x)\n    return x\n'
    assert rules('src/openflo/foo.py', src, 'print-in-library')


def test_bp7_print_in_cli_gui_main_and_main_guard_passes():
    lib = 'def gate(x):\n    print("FL1-A", x)\n'
    assert rules('src/openflo/cli.py', lib, 'print-in-library') == []
    assert rules('src/openflo/ui_foo.py', lib, 'print-in-library') == []
    src = """
    def main():
        print('done')

    if __name__ == '__main__':
        print('done')
    """
    assert rules('src/openflo/foo.py', src, 'print-in-library') == []



def test_bp7_print_in_a_top_level_entry_point_passes_but_not_in_a_helper():
    src = """
    def subprocess_main():
        print('progress')

    def domain(x):
        print(x)

    class Job:
        def run_main(self):
            print('x')
    """
    found = bp.check_source('src/openflo/foo.py', textwrap.dedent(src))
    assert [v.line for v in found if v.rule == 'print-in-library'] == [6, 10]

# ── BP9 ─────────────────────────────────────────────────────────────────────

def test_bp9_a_bare_open_bound_is_flagged_in_any_spelling():
    src = 'lo, hi = -1e12, 1000000000000\nbig = 1.0E+12\n'
    assert len(rules('src/openflo/foo.py', src, 'magic-open-bound')) == 3


def test_bp9_the_named_constant_definition_and_other_numbers_pass():
    src = '# An open gate side.\nOPEN = 1e12\nLIMIT = 1e11\n'
    assert rules('src/openflo/scales.py', src, 'magic-open-bound') == []


def test_bp9_the_tree_has_no_bare_open_bound_left():
    baseline = json.loads((ROOT / 'scripts' / 'best_practices_baseline.json').read_text())
    assert 'magic-open-bound' not in baseline


# Parameter defaults that are scales.LOGICLE_DEFAULTS or an asinh cofactor.
_SCALE_DEFAULTS = {('t', 262144), ('m', 4.5), ('w', 0.5), ('cofactor', 150.0),
                   ('cofactor', 5.0)}


def test_bp9_no_signature_retypes_a_logicle_default_or_asinh_cofactor():
    import ast
    found = []
    for path in sorted((ROOT / 'src' / 'openflo').rglob('*.py')):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args.posonlyargs + node.args.args
            pairs = list(zip(args[len(args) - len(node.args.defaults):],
                             node.args.defaults, strict=True))
            pairs += [(a, d) for a, d in zip(node.args.kwonlyargs,
                                             node.args.kw_defaults, strict=True) if d]
            found += [f'{path.name}:{node.name}({a.arg})' for a, d in pairs
                      if isinstance(d, ast.Constant)
                      and (a.arg, d.value) in _SCALE_DEFAULTS]
    assert found == []


# ── BP10 ────────────────────────────────────────────────────────────────────

def test_bp10_private_names_used_from_another_module_are_flagged():
    src = """
    from . import foo
    from .bar import _parse
    from openflo.baz import _qux

    def run():
        return foo._helper()
    """
    assert len(rules('src/openflo/foo2.py', src, 'private-cross-import')) == 3


def test_bp10_dunders_private_modules_and_own_attributes_pass():
    src = """
    from . import _console
    from . import __version__
    from .bar import parse

    class Foo:
        def run(self):
            return self._helper(parse)
    """
    assert rules('src/openflo/foo.py', src, 'private-cross-import') == []


# ── BP11 ────────────────────────────────────────────────────────────────────

def test_bp11_tracker_tags_in_comments_and_docstrings_are_flagged():
    src = '''
    """Foo export (Complaints 12)."""

    def foo():
        """Fixed in PR #78."""
        return 1  # see issue #5
    '''
    assert len(rules('src/openflo/foo.py', src, 'tracker-tag')) == 3


def test_bp11_ordinary_text_and_non_docstring_strings_pass():
    src = '''
    """Each item (one of 3) gets a PR review."""
    URL = "see issue #5"  # PR notes go to the changelog
    '''
    assert rules('src/openflo/foo.py', src, 'tracker-tag') == []


def test_bp11_bare_tracker_numbers_and_wrapped_tags_are_flagged():
    src = '''
    """Axis lookup (#48), see #47 and PR #50; until #78 is merged.

    One sample per batch (Complaints 26; the command line
    did this first). Drawn in the worker (Speed 9), menu (UI 35).
    """
    X = 1  # columns (Gap 3)
    '''
    assert len(rules('src/openflo/foo.py', src, 'tracker-tag')) == 8


def test_bp11_sweep_finding_ids_are_flagged_and_channel_names_pass():
    src = '''
    """Zoom in log space (MAIN-7)."""
    X = 1  # ── TOOLS-6 / STATS-12: Dark exports dark
    Y = 2  # FL1-A against FSC-A, Comp-FL2-A
    '''
    # one per docstring tag, one per tagged comment; Y's channels pass
    found = bp.check_source('src/openflo/foo.py', textwrap.dedent(src))
    assert sorted(v.line for v in found if v.rule == 'tracker-tag') == [2, 3]


def test_bp11_colours_anchors_and_tree_column_zero_pass():
    src = '''
    """Header colour #333; link to #a1-b; the #0 column stretches.

    CSS {color:#222} and a step #1-2 of the wizard.
    """
    '''
    assert rules('src/openflo/foo.py', src, 'tracker-tag') == []


def test_bp11_vague_export_and_parser_names_are_flagged():
    src = '''
    def _num(v):
        return float(v)

    def _number(v):
        return float(v)

    class W:
        def _export_png(self):
            pass

        def _export_fig(self):
            pass

        def _export(self):
            pass
    '''
    assert len(rules('src/openflo/ui_foo.py', src, 'vague-name')) == 5


def test_bp11_descriptive_export_and_parser_names_pass():
    src = '''
    def _finite_float(v):
        return float(v)

    class W:
        def _export(self, kind):
            pass

        def _export_figure(self):
            pass

        def _export_csv(self):
            pass
    '''
    assert rules('src/openflo/ui_foo.py', src, 'vague-name') == []


def test_bp11_cryptic_mixin_prefixes_are_flagged():
    src = '''
    class W:
        def _pc_eval(self):
            self._ac_cache = {}
            return getattr(self, '_tb_menu', None)
    '''
    assert len(rules('src/openflo/ui_foo.py', src, 'vague-name')) == 3


def test_bp11_spelled_out_mixin_prefixes_and_lookalikes_pass():
    src = '''
    class W:
        def _plotcache_store(self):
            self._autoclean_cache = {}
            self._toolbar_button = None
            return getattr(self, '_pcx', 'pc_ok')
    '''
    assert rules('src/openflo/ui_foo.py', src, 'vague-name') == []


# ── BP13 ────────────────────────────────────────────────────────────────────

def _undocumented(source, path='src/openflo/foo.py'):
    """Names `missing-docstring` reports, in the order it reports them."""
    found = bp.check_source(path, textwrap.dedent(source))
    return [v.message.split()[2] for v in found if v.rule == 'missing-docstring']


def test_bp13_public_functions_classes_and_methods_without_docstrings_are_flagged():
    src = """
    import sys
    def load(path):
        return path
    async def fetch(url):
        return url
    class Reader:
        def read(self):
            return 1
        class Options:
            def apply(self):
                pass
    if sys.version_info >= (3, 12):
        def fast():
            pass
    try:
        import numpy
    except ImportError:
        def fallback():
            pass
    """
    assert _undocumented(src) == ['load', 'fetch', 'Reader', 'Reader.read',
                                  'Reader.Options', 'Reader.Options.apply',
                                  'fast', 'fallback']


def test_bp13_documented_private_nested_and_accessor_names_pass():
    src = """
    import typing
    from typing import TYPE_CHECKING, overload
    def load(path):
        '''Read the file at `path`.'''
        def helper():
            return 1
        return helper()
    def _private():
        pass
    class Reader:
        '''Reads samples.'''
        def __init__(self):
            pass
        def _step(self):
            pass
        @property
        def size(self):
            '''Bytes read so far.'''
            return 0
        @size.setter
        def size(self, value):
            pass
        @size.deleter
        def size(self):
            pass
    class _Hidden:
        def run(self):
            pass
        class Inner:
            def go(self):
                pass
    @overload
    def pick(x: int) -> int: ...
    @typing.overload
    def pick(x: str) -> str: ...
    def pick(x):
        '''Return `x`.'''
        return x
    if TYPE_CHECKING:
        def stub(): ...
    if __name__ == '__main__':
        def run():
            pass
    """
    assert _undocumented(src) == []


def test_bp13_every_public_name_in_the_tree_has_a_docstring():
    baseline = json.loads((ROOT / 'scripts' / 'best_practices_baseline.json').read_text())
    assert 'missing-docstring' not in baseline


# ── BP14────────────────────────────────────────────────────────────────────

def test_bp14_a_raw_thread_in_a_gui_module_is_flagged():
    src = 'import threading\ndef go(fn):\n    threading.Thread(target=fn).start()\n'
    assert rules('src/openflo/ui_foo.py', src, 'raw-thread')


def test_bp14_threads_in_async_task_and_core_modules_pass():
    src = 'from threading import Thread\ndef go(fn):\n    Thread(target=fn).start()\n'
    assert rules('src/openflo/async_task.py', src, 'raw-thread') == []
    assert rules('src/openflo/foo.py', src, 'raw-thread') == []


# ── BP15 ────────────────────────────────────────────────────────────────────

def _loose_lines(source, path='tests/test_foo.py'):
    return [v.line for v in bp.check_source(path, textwrap.dedent(source))
            if v.rule == 'loose-tolerance']


def test_bp15_loose_tolerances_without_a_comment_are_flagged():
    src = """
    import math, unittest
    import numpy as np
    import pytest
    def test_a(x, got, want):
        assert x == pytest.approx(3.0, rel=0.05)
        assert np.allclose(got, want, atol=2.0)
        np.testing.assert_allclose(got, want,
                                   rtol=0.02)
        assert math.isclose(x, 3.0, abs_tol=0.5, rel_tol=1e-9) or x == pytest.approx(3, abs=1)
        assert x == pytest.approx(3.0, rel=0.01)
        assert x == pytest.approx(3.0, abs=0.05)
    class T(unittest.TestCase):
        def test_b(self):
            self.assertAlmostEqual(1.0, 1.2, delta=0.5)
    """
    assert _loose_lines(src) == [6, 7, 8, 10, 11, 12, 15]


def test_bp15_commented_tight_and_non_constant_tolerances_pass():
    src = """
    import numpy as np
    import pytest
    TOL = 0.5
    def test_a(x, got, want):
        assert x == pytest.approx(3.0, rel=1e-6)
        assert np.allclose(got, want, rtol=0.009, atol=0.049)
        assert x == pytest.approx(3.0, rel=0.05)  # 2,000 random events
        # a fit on noisy data: its SE is ~0.3
        assert x == pytest.approx(3.0, abs=1)
        np.testing.assert_allclose(
            got, want,
            rtol=0.05,  # 1,000 events per bin
        )
        assert x == pytest.approx(3.0, abs=TOL)
        assert x == pytest.approx(3.0, abs=True)
    """
    assert _loose_lines(src) == []


def test_bp15_a_comment_two_lines_up_or_after_does_not_count():
    src = """
    import pytest
    def test_a(x):
        # unrelated note

        assert x == pytest.approx(3.0, rel=0.05)
        # explanation placed after
    """
    assert _loose_lines(src) == [6]


def test_bp15_only_test_files_are_checked_and_they_get_only_bp15():
    loose = 'import pytest\ndef check(x):\n    print(x)\n    return x == pytest.approx(3, rel=0.5)\n'
    assert rules('src/openflo/foo.py', loose, 'loose-tolerance') == []
    assert rules('tests/test_foo.py', loose) == ['loose-tolerance']
    # a body copied between a test and the package is not a BP1 duplicate
    assert cross({'src/openflo/a.py': _BODY.format(name='parse', doc='Parse.'),
                  'tests/test_a.py': _BODY.format(name='parse', doc='Parse.')},
                 'duplicate-function') == []


def test_bp15_read_tree_covers_the_tests(tmp_path):
    (tmp_path / 'src' / 'openflo').mkdir(parents=True)
    (tmp_path / 'tests' / 'sub').mkdir(parents=True)
    (tmp_path / 'src' / 'openflo' / 'a.py').write_text('', encoding='utf-8')
    (tmp_path / 'tests' / 'sub' / 'test_b.py').write_text('', encoding='utf-8')
    assert sorted(bp.read_tree(tmp_path)) == ['src/openflo/a.py', 'tests/sub/test_b.py']


def test_bp15_every_loose_tolerance_in_the_tests_says_why():
    baseline = json.loads((ROOT / 'scripts' / 'best_practices_baseline.json').read_text())
    assert 'loose-tolerance' not in baseline
    found = [str(v) for v in bp.check_sources(bp.read_tree(ROOT))
             if v.rule == 'loose-tolerance']
    assert found == []


# ── the ratchet ─────────────────────────────────────────────────────────────

# private, so the only rule it breaks is print-in-library
_PRINTS = 'def _foo():\n    print(1)\n    print(2)\n'


def _ratchet(baseline_count):
    found = bp.check_source('src/openflo/foo.py', _PRINTS)
    baseline = {'print-in-library': {'src/openflo/foo.py': baseline_count}}
    return bp.compare(found, baseline)


def test_a_count_above_the_baseline_fails():
    new, _ = _ratchet(1)
    assert [v.rule for v in new] == ['print-in-library'] * 2


def test_a_count_equal_to_the_baseline_passes_silently():
    assert _ratchet(2) == ([], [])


def test_a_count_below_the_baseline_passes_with_a_note():
    new, notes = _ratchet(3)
    assert new == []
    assert len(notes) == 1 and 'fixed 1' in notes[0] and '--write-baseline' in notes[0]


def test_write_baseline_round_trip(tmp_path, capsys):
    pkg = tmp_path / 'src' / 'openflo'
    pkg.mkdir(parents=True)
    (pkg / 'foo.py').write_text(_PRINTS, encoding='utf-8')
    (tmp_path / 'scripts').mkdir()
    base = tmp_path / 'scripts' / 'best_practices_baseline.json'

    assert bp.main(['--root', str(tmp_path)]) == 1          # no baseline yet
    assert bp.main(['--root', str(tmp_path), '--write-baseline']) == 0
    assert json.loads(base.read_text(encoding='utf-8')) == {
        'print-in-library': {'src/openflo/foo.py': 2}}
    assert bp.main(['--root', str(tmp_path)]) == 0

    (pkg / 'foo.py').write_text(_PRINTS + '    print(3)\n', encoding='utf-8')
    capsys.readouterr()
    assert bp.main(['--root', str(tmp_path)]) == 1
    assert 'src/openflo/foo.py:4: BP7 print-in-library' in capsys.readouterr().out


def test_the_tree_passes_against_the_committed_baseline(capsys):
    assert bp.main([]) == 0, capsys.readouterr().out


def test_ui_logic_is_core_despite_its_prefix():
    assert rules('src/openflo/ui_logic.py', 'import tkinter\n', 'core-imports-tk')
    assert rules('src/openflo/ui_foo.py', 'import tkinter\n', 'core-imports-tk') == []


def test_the_shared_csv_writer_may_read_prefs_for_its_default_style():
    src = 'from .prefs import read_prefs\n'
    assert rules('src/openflo/table_export.py', src, 'core-reads-prefs') == []
    assert rules('src/openflo/foo.py', src, 'core-reads-prefs')


def test_write_baseline_refuses_to_raise_a_count_unless_allowed(tmp_path, capsys):
    pkg = tmp_path / 'src' / 'openflo'
    pkg.mkdir(parents=True)
    (pkg / 'foo.py').write_text(_PRINTS, encoding='utf-8')
    (tmp_path / 'scripts').mkdir()
    base = tmp_path / 'scripts' / 'best_practices_baseline.json'
    assert bp.main(['--root', str(tmp_path), '--write-baseline']) == 0

    (pkg / 'foo.py').write_text(_PRINTS + '    print(3)\n', encoding='utf-8')
    capsys.readouterr()
    assert bp.main(['--root', str(tmp_path), '--write-baseline']) == 1
    assert '2 -> 3' in capsys.readouterr().out
    assert json.loads(base.read_text(encoding='utf-8')) == {
        'print-in-library': {'src/openflo/foo.py': 2}}

    assert bp.main(['--root', str(tmp_path), '--write-baseline',
                    '--allow-increase']) == 0
    assert json.loads(base.read_text(encoding='utf-8')) == {
        'print-in-library': {'src/openflo/foo.py': 3}}


def test_a_file_that_does_not_parse_is_reported_by_name(tmp_path, capsys):
    pkg = tmp_path / 'src' / 'openflo'
    pkg.mkdir(parents=True)
    (pkg / 'foo.py').write_text('def foo(:\n', encoding='utf-8')
    assert bp.main(['--root', str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert 'src/openflo/foo.py:1: cannot parse' in out


def test_the_baseline_ships_in_the_sdist():
    manifest = (ROOT / 'MANIFEST.in').read_text(encoding='utf-8').splitlines()
    assert 'include scripts/best_practices_baseline.json' in manifest
