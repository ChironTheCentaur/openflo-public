"""CONTRIBUTING's "Module naming" section and scripts/best_practices.py name
the same GUI / core exceptions, and every module the section names exists.

The checker's lists decide what may import Tk; the section is what a
contributor reads. A module added to one list but not the other, or renamed
without updating the section, fails here.
"""
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    'best_practices', ROOT / 'scripts' / 'best_practices.py')
assert _spec and _spec.loader
bp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bp)

PACKAGE = ROOT / 'src' / 'openflo'


def naming_section(text: str) -> str:
    match = re.search(r'^## Module naming\n(.*?)(?=^## )', text, re.M | re.S)
    assert match, 'CONTRIBUTING.md has no "## Module naming" section'
    return match.group(1)


def named_in(section: str) -> set[str]:
    return set(re.findall(r'`([A-Za-z_][\w.]*\*?)`', section))


def section() -> str:
    return naming_section((ROOT / 'CONTRIBUTING.md').read_text(encoding='utf-8'))


def test_every_checker_exception_is_documented():
    named = named_in(section())
    missing = sorted((bp.GUI_MODULES | bp.CORE_MODULES) - named)
    assert missing == [], f'add to CONTRIBUTING "Module naming": {missing}'


def test_every_checker_prefix_is_documented():
    named = named_in(section())
    missing = [p for p in bp.GUI_MODULE_PREFIXES if f'{p}*' not in named]
    assert missing == [], f'add to CONTRIBUTING "Module naming": {missing}'


def module_names(section: str) -> set[str]:
    """Backticked plain lowercase names: the modules. Paths, rule ids,
    constants, globs and dotted attributes have other shapes."""
    return {n for n in named_in(section) if re.fullmatch(r'[a-z][a-z0-9_]*', n)}


def test_every_module_the_section_names_exists():
    gone = sorted(n for n in module_names(section())
                  if not (PACKAGE / f'{n}.py').exists())
    assert gone == [], f'CONTRIBUTING "Module naming" names no such module: {gone}'


def test_the_section_documents_the_documented_classifications():
    # What the section says each exception is must match the checker.
    for name in ('ui_logic', 'view_tabs', 'layout_page', 'layout_render',
                 'gating', 'gating_helpers'):
        assert not bp.is_gui(f'{name}.py'), name
    for name in ('gui', 'theme', 'workspace', 'clipboard_export'):
        assert bp.is_gui(f'{name}.py'), name


def test_the_helpers_catch_a_missing_or_stale_name():
    doc = ('## Module naming\n\n- `ui_logic` is core.\n- `ghost_module` too.\n'
           '\n## Next\n\n`theme`\n')
    named = module_names(naming_section(doc))
    assert named == {'ui_logic', 'ghost_module'}   # `theme` is outside it
    assert not (PACKAGE / 'ghost_module.py').exists()
