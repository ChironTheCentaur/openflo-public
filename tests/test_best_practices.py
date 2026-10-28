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


# ── BP9 ─────────────────────────────────────────────────────────────────────

def test_bp9_a_bare_open_bound_is_flagged_in_any_spelling():
    src = 'lo, hi = -1e12, 1000000000000\nbig = 1.0E+12\n'
    assert len(rules('src/openflo/foo.py', src, 'magic-open-bound')) == 3


def test_bp9_the_named_constant_definition_and_other_numbers_pass():
    src = '# An open gate side.\nOPEN = 1e12\nLIMIT = 1e11\n'
    assert rules('src/openflo/scales.py', src, 'magic-open-bound') == []


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


# ── BP14 ────────────────────────────────────────────────────────────────────

def test_bp14_a_raw_thread_in_a_gui_module_is_flagged():
    src = 'import threading\ndef go(fn):\n    threading.Thread(target=fn).start()\n'
    assert rules('src/openflo/ui_foo.py', src, 'raw-thread')


def test_bp14_threads_in_async_task_and_core_modules_pass():
    src = 'from threading import Thread\ndef go(fn):\n    Thread(target=fn).start()\n'
    assert rules('src/openflo/async_task.py', src, 'raw-thread') == []
    assert rules('src/openflo/foo.py', src, 'raw-thread') == []


# ── the ratchet ─────────────────────────────────────────────────────────────

_PRINTS = 'def foo():\n    print(1)\n    print(2)\n'


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
