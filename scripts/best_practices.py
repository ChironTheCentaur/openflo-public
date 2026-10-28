#!/usr/bin/env python
"""Stop new code from bringing back what the code survey found.

    python scripts/best_practices.py                  # check against the baseline
    python scripts/best_practices.py --list           # every violation, baselined too
    python scripts/best_practices.py --write-baseline # lock in today's counts

Each rule below has an id (BPn) matching the best-practices checklist, and
each message names the shared thing to use instead. The rules are cheap
static checks over src/openflo/**/*.py (stdlib ast and tokenize, a few
seconds), not proofs: they catch the patterns the survey kept finding.

The tree had violations of most rules when this check was written. Fixing
them all first would have been one giant change, so they are frozen in
scripts/best_practices_baseline.json as {rule: {file: count}} -- a ratchet:

  * a (rule, file) count ABOVE its baseline fails: the change added one;
  * a count BELOW it passes, with a note to run --write-baseline. Several
    branches fix findings at once, and a stale entry must not turn another
    branch's run red. Lock the lower count in with the fix.

A count cannot say WHICH occurrence is new, so a failure lists every
occurrence of that rule in that file; the new one is among them.

The GUI / core split the BP2, BP7 and BP14 rules use is GUI_MODULE_PREFIXES
and GUI_MODULES below: one place, read before adding a module.

Exit 0 = no (rule, file) count above the baseline; 1 = a new violation.
"""
from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import io
import json
import re
import sys
import tokenize
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'scripts' / 'best_practices_baseline.json'
PACKAGE = 'src/openflo'

RULE_IDS = {
    'duplicate-function': 'BP1',
    'duplicate-public-name': 'BP1',
    'core-imports-tk': 'BP2',
    'core-reads-prefs': 'BP2',
    'import-side-effect': 'BP3',
    'csv-bypass': 'BP5',
    'version-read': 'BP5',
    'print-in-library': 'BP7',
    'magic-open-bound': 'BP9',
    'private-cross-import': 'BP10',
    'tracker-tag': 'BP11',
    'raw-thread': 'BP14',
}

# ── GUI / core ───────────────────────────────────────────────────────────────
# A GUI module may import Tk and read preferences; a core module may not, so
# the library, the CLI and the tests can use it without a display. A module is
# GUI when its name starts with one of these prefixes or is in GUI_MODULES.
# Everything else is core -- including layout_page, layout_render and
# view_tabs, Tk-free model code, which is what keeps them testable.
# CORE_MODULES overrides a prefix for a module that is Tk-free by design.
GUI_MODULE_PREFIXES = ('editor_', 'ui_')
CORE_MODULES = frozenset({
    'ui_logic',          # pure UI decision logic, no Tk (its docstring says so)
})
GUI_MODULES = frozenset({
    'gui',               # the main window
    'theme',             # Tk styles and colours
    'workspace',         # the Pipeline Workspace window
    'tool_window',       # base class of the tool windows
    'dialog_inputs',     # Tk entry validation
    'async_task',        # run_async and the busy strip
    'group_test_choice', # the "Test" box shared by comparison windows
    'tk_threads',        # Tk finalisers kept on the main thread
    'clipboard_export',  # copies through Tk's clipboard
})
# prefs.py is Tk-free but is the GUI's preference store (~/.openflo/prefs.json):
# core code takes settings as parameters instead of reading it (BP2).
PREFS_MODULE = 'prefs'
# The shared CSV writer applies the user's CSV style when a caller passes none:
# that is its job under BP5, so it alone among core modules may read prefs.
PREFS_READERS = frozenset({'table_export'})


def module_name(path: str) -> str:
    return PurePosixPath(path).stem


def is_gui(path: str) -> bool:
    name = module_name(path)
    if name in CORE_MODULES:
        return False
    return name in GUI_MODULES or name.startswith(GUI_MODULE_PREFIXES)


def _is_gui_module_ref(dotted: str) -> bool:
    """`theme`, `editor_plot`, `openflo.gui`... names a GUI module."""
    parts = dotted.split('.')
    if parts[0] == 'openflo':
        parts = parts[1:]
    return bool(parts) and is_gui(parts[0] + '.py')


# Command-line entry modules: printing IS their interface (BP7). Core modules
# that also have a console script (compare, voltage, synthetic) may print in
# their top-level `main()` and under `if __name__ == '__main__':`, not in the
# library functions a caller imports.
CLI_MODULES = frozenset({
    'cli', 'cli_args', '_console', '__main__', 'inspect_fcs',
    'selftest', 'selftest_controls', 'diagnostics',
    'errors',            # errors.note IS the stderr report the others should use
})

# The module that writes CSV in the user's chosen style (separator, decimal
# mark, NA): table_export.write_rows_csv / write_frame_csv (BP5).
CSV_MODULE = 'table_export'
# The one place the running version is read: update.current_version (a
# source checkout's pyproject first, then the installed metadata, then
# openflo.__version__); result_stamp.app_version caches it. capabilities,
# provenance and export_stamp each read it their own way (BP5).
VERSION_MODULES = frozenset({'update', '__init__'})
# The named open-bound sentinel (BP9).
OPEN_MODULE, OPEN_NAME = 'scales', 'OPEN'
# Where worker threads are started for the GUI (BP14).
THREAD_MODULE = 'async_task'

# BP1 duplicate-function: a body must have at least this many statements,
# counted at every nesting level, docstring excluded. A one-statement body is
# a wrapper (`return self._x`, a callback forwarding to one call) and two or
# three are the `self.var.set(..); self._redraw()` Tk callbacks every window
# has: identical by nature, nothing to share. Four is the smallest real
# logic: try / parse / except-return / return. Measured when this was
# written: thresholds 2, 3 and 4 find the same four copied pairs (a
# Benjamini-Hochberg correction, a file stamp, a number parser, a population
# lister); 5 already misses two of them.
MIN_DUPLICATE_STATEMENTS = 4

# BP1 duplicate-public-name: names that legitimately exist once per module.
# Private (_name) and dunder names (__getattr__, __dir__) are never counted.
DUPLICATE_NAME_EXEMPT = frozenset({
    'main',              # one console-script entry point per CLI module
})

TRACKER_RES = (
    re.compile(r'\((?:Complaints?|Batch|Item|Fix)\s*#?\d+[^)\n]{0,40}\)'),
    re.compile(r'\bPR\s*#\d+', re.IGNORECASE),
    re.compile(r'\bissue\s*#\d+', re.IGNORECASE),
)


class Violation(NamedTuple):
    """One finding; sorts by path, then line."""
    path: str
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f'{self.path}:{self.line}: {RULE_IDS[self.rule]} {self.rule}: {self.message}'


# ── helpers ─────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=8)
def _all_nodes(tree: ast.Module) -> tuple[ast.AST, ...]:
    """ast.walk(tree), walked once and shared by the rules (keeps a run fast)."""
    return tuple(ast.walk(tree))


def _dotted(node: ast.AST) -> str:
    """`a.b.c` for a Name/Attribute chain, '' otherwise."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return '.'.join(reversed(parts))
    return ''


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    t = node.test
    return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
            and t.left.id == '__name__' and len(t.comparators) == 1
            and isinstance(t.comparators[0], ast.Constant)
            and t.comparators[0].value == '__main__')


def _is_type_checking(node: ast.AST) -> bool:
    return isinstance(node, ast.If) and _dotted(node.test) in (
        'TYPE_CHECKING', 'typing.TYPE_CHECKING')


def _is_catch_warnings(node: ast.AST) -> bool:
    return isinstance(node, ast.With) and any(
        isinstance(i.context_expr, ast.Call)
        and _dotted(i.context_expr.func).endswith('catch_warnings')
        for i in node.items)


_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _import_time_nodes(tree: ast.Module, *, skip=lambda n: False):
    """Every node that runs when the module is imported: module and class
    bodies, not function bodies or `if __name__ == '__main__':`; `skip(node)`
    prunes more (TYPE_CHECKING blocks, catch_warnings blocks)."""
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if _is_main_guard(node) or skip(node):
            continue
        if isinstance(node, _FUNCS):
            # decorators and defaults do run at import
            stack.extend(getattr(node, 'decorator_list', []))
            args = node.args
            stack.extend(d for d in args.defaults + args.kw_defaults if d)
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _docstrings(tree: ast.Module):
    """(first line, text) of the module, class and function docstrings."""
    for node in _all_nodes(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                yield first.lineno, first.value.value


def _imported_names(tree: ast.Module, module: str) -> set[str]:
    """Local names bound to `module.x` by `from module import x [as y]`."""
    return {a.asname or a.name for n in _all_nodes(tree)
            if isinstance(n, ast.ImportFrom) and n.module == module and not n.level
            for a in n.names}


# ── per-file rules ──────────────────────────────────────────────────────────

def _core_imports_tk(path, tree, out):
    if is_gui(path):
        return
    for node in _import_time_nodes(tree, skip=_is_type_checking):
        bad = ''
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split('.')[0]
                if top in ('tkinter', '_tkinter', 'tkinterdnd2') or (
                        top == 'openflo' and _is_gui_module_ref(a.name)):
                    bad = a.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ''
            if not node.level and mod.split('.')[0] in ('tkinter', 'tkinterdnd2'):
                bad = mod
            elif (node.level or mod.split('.')[0] == 'openflo') and mod not in ('', 'openflo'):
                if _is_gui_module_ref(mod):
                    bad = mod
            elif node.level or mod == 'openflo':
                gui = [a.name for a in node.names if _is_gui_module_ref(a.name)]
                if gui:
                    bad = gui[0]
        if bad:
            out.append(Violation(path, node.lineno, 'core-imports-tk',
                                 f'core module imports {bad} at import time; import it '
                                 'inside the GUI function that needs it, or move the code '
                                 'to a GUI module (see GUI_MODULES)'))


def _core_reads_prefs(path, tree, out):
    if is_gui(path) or module_name(path) in (PREFS_MODULE, *PREFS_READERS):
        return
    for node in _all_nodes(tree):
        hit = False
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ''
            if mod in ('prefs', 'openflo.prefs') and (node.level or mod.startswith('openflo')):
                hit = True
            elif (node.level and not mod) or mod == 'openflo':
                hit = any(a.name == 'prefs' for a in node.names)
        elif isinstance(node, ast.Import):
            hit = any(a.name == 'openflo.prefs' for a in node.names)
        if hit:
            out.append(Violation(path, node.lineno, 'core-reads-prefs',
                                 'core module reads the GUI preferences; take the setting as '
                                 'a parameter and let the GUI caller pass prefs.read_prefs() '
                                 'values in'))


def _import_side_effect(path, tree, out):
    skip = _is_catch_warnings
    warn_funcs = _imported_names(tree, 'warnings') & {'filterwarnings', 'simplefilter'}
    exit_funcs = {'sys.exit', 'exit', 'quit', 'os._exit'}   # bare exit: builtin or from sys
    for node in _import_time_nodes(tree, skip=skip):
        what = ''
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name in ('warnings.filterwarnings', 'warnings.simplefilter') or name in warn_funcs:
                what = f'{name}() changes warning filters for every importer'
            elif name in exit_funcs:
                what = f'{name}() ends the importing process'
        elif isinstance(node, ast.Raise) and node.exc is not None:
            exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if _dotted(exc) == 'SystemExit':
                what = 'raise SystemExit ends the importing process'
        if what:
            out.append(Violation(path, node.lineno, 'import-side-effect',
                                 f'{what} at import time; do it inside a function (main(), or '
                                 "under `if __name__ == '__main__':`), or scope it with "
                                 'warnings.catch_warnings()'))


def _csv_bypass(path, tree, out):
    if module_name(path) == CSV_MODULE:
        return
    writers = {'csv.writer', 'csv.DictWriter'} | {
        n for n in _imported_names(tree, 'csv') if n in ('writer', 'DictWriter')}
    for node in _all_nodes(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == 'to_csv') or name in writers:
            what = '.to_csv()' if name not in writers else f'{name}()'
            out.append(Violation(path, node.lineno, 'csv-bypass',
                                 f'{what} ignores the CSV style preference; write through '
                                 'table_export.write_frame_csv / write_rows_csv'))


def _version_read(path, tree, out):
    if module_name(path) in VERSION_MODULES:
        return
    msg = ('reads the OpenFlo version its own way; call update.current_version() '
           '(or result_stamp.app_version())')
    for node in _all_nodes(tree):
        hit = False
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ''
            if mod.split('.')[0] == 'pkg_resources':
                hit = True
            elif (node.level and not mod) or mod == 'openflo':
                hit = any(a.name == '__version__' for a in node.names)
        elif isinstance(node, ast.Import):
            hit = any(a.name.split('.')[0] == 'pkg_resources' for a in node.names)
        elif isinstance(node, ast.Attribute):
            hit = node.attr == '__version__' and _dotted(node.value) == 'openflo'
        elif isinstance(node, ast.Call) and node.args:
            # metadata.version('openflo'), or a local wrapper _version('openflo')
            name = _dotted(node.func).rsplit('.', 1)[-1]
            arg = node.args[0]
            hit = ('version' in name.lower() and isinstance(arg, ast.Constant)
                   and arg.value == 'openflo')
        if hit:
            out.append(Violation(path, node.lineno, 'version-read', msg))


def _print_in_library(path, tree, out):
    name = module_name(path)
    if is_gui(path) or name in CLI_MODULES:
        return
    mains = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name == 'main']
    allowed = {id(n) for m in mains for n in ast.walk(m)}
    allowed |= {id(n) for g in tree.body if _is_main_guard(g) for n in ast.walk(g)}
    for node in _all_nodes(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'print' and id(node) not in allowed):
            out.append(Violation(path, node.lineno, 'print-in-library',
                                 'print() in a library module writes to whatever stdout the '
                                 'caller has; return the text, log it, or report it through '
                                 'errors.note'))


def _magic_open_bound(path, tree, out):
    exempt_lines: set[int] = set()
    if module_name(path) == OPEN_MODULE:
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Name) and t.id == OPEN_NAME for t in targets):
                    exempt_lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    for node in _all_nodes(tree):
        if (isinstance(node, ast.Constant) and type(node.value) in (int, float)
                and node.value == 1e12 and node.lineno not in exempt_lines):
            out.append(Violation(path, node.lineno, 'magic-open-bound',
                                 f'bare 1e12 open-gate bound; use {OPEN_MODULE}.{OPEN_NAME} '
                                 f'(+-{OPEN_MODULE}.{OPEN_NAME} for an open side)'))


def _private_cross_import(path, tree, out, package_modules=None):
    def private(name):
        return name.startswith('_') and not (name.startswith('__') and name.endswith('__'))

    def openflo_mod(node):
        return node.level > 0 or (node.module or '').split('.')[0] == 'openflo'

    module_aliases: set[str] = set()
    for node in _all_nodes(tree):
        if isinstance(node, ast.ImportFrom) and openflo_mod(node):
            mod = node.module or ''
            for a in node.names:
                is_module = (mod in ('', 'openflo') and (
                    package_modules is None or a.name in package_modules))
                if is_module and mod in ('', 'openflo'):
                    module_aliases.add(a.asname or a.name)
                elif private(a.name):
                    out.append(Violation(path, node.lineno, 'private-cross-import',
                                         f'imports private {a.name} from {"." * node.level}{mod}; '
                                         'make it public (drop the underscore) in its module, '
                                         'or move it to a shared helper'))
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith('openflo.') and a.asname:
                    module_aliases.add(a.asname)
    for node in _all_nodes(tree):
        if (isinstance(node, ast.Attribute) and private(node.attr)
                and isinstance(node.value, ast.Name) and node.value.id in module_aliases):
            out.append(Violation(path, node.lineno, 'private-cross-import',
                                 f'uses private {node.value.id}.{node.attr} of another module; '
                                 'make it public in its module, or move it to a shared helper'))


def _tracker_tag(path, source, tree, out):
    msg = ('work-tracker tag in a comment or docstring; say what the code does and '
           'why, the history lives in git and the tracker')
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT and any(r.search(tok.string) for r in TRACKER_RES):
                out.append(Violation(path, tok.start[0], 'tracker-tag', msg))
    except (tokenize.TokenError, SyntaxError):
        pass
    for first, text in _docstrings(tree):
        for offset, line in enumerate(text.splitlines()):
            for r in TRACKER_RES:
                out.extend(Violation(path, first + offset, 'tracker-tag', msg)
                           for _ in r.finditer(line))


def _raw_thread(path, tree, out):
    if not is_gui(path) or module_name(path) == THREAD_MODULE:
        return
    threads = {'threading.Thread'} | {
        n for n in _imported_names(tree, 'threading') if n == 'Thread'}
    for node in _all_nodes(tree):
        if isinstance(node, ast.Call) and _dotted(node.func) in threads:
            out.append(Violation(path, node.lineno, 'raw-thread',
                                 'starts a raw thread in a GUI module; run the work with '
                                 'async_task.run_async, which reports errors and returns '
                                 'to the Tk thread'))


def check_source(path: str, source: str, package_modules=None) -> list[Violation]:
    """Per-file rules on one module. `path` is its repo-relative posix path
    (only its file name matters: it decides GUI or core and the exemptions)."""
    return _check_tree(path, source, ast.parse(source, filename=path), package_modules)


def _check_tree(path, source, tree, package_modules) -> list[Violation]:
    out: list[Violation] = []
    _core_imports_tk(path, tree, out)
    _core_reads_prefs(path, tree, out)
    _import_side_effect(path, tree, out)
    _csv_bypass(path, tree, out)
    _version_read(path, tree, out)
    _print_in_library(path, tree, out)
    _magic_open_bound(path, tree, out)
    _private_cross_import(path, tree, out, package_modules)
    _tracker_tag(path, source, tree, out)
    _raw_thread(path, tree, out)
    return out


# ── cross-file rules ────────────────────────────────────────────────────────

def _functions(tree: ast.Module):
    """Top-level functions and the methods of top-level classes."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node, node.name
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield sub, f'{node.name}.{sub.name}'


def _body_key(fn) -> str | None:
    body = fn.body
    if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    n_stmts = sum(isinstance(n, ast.stmt) for b in body for n in ast.walk(b))
    if n_stmts < MIN_DUPLICATE_STATEMENTS:
        return None
    dump = ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)
    # the function's own name (a recursive call, a self.<name> re-entry) is
    # not part of what it does
    dump = dump.replace(f"id='{fn.name}'", "id='_SELF_'").replace(
        f"attr='{fn.name}'", "attr='_SELF_'")
    return hashlib.sha1(dump.encode()).hexdigest()


def check_cross(trees: dict[str, ast.Module]) -> list[Violation]:
    """Rules that compare modules: {repo-relative path: parsed module}."""
    out: list[Violation] = []
    bodies = defaultdict(list)
    names = defaultdict(list)
    for path, tree in trees.items():
        for fn, qual in _functions(tree):
            key = _body_key(fn)
            if key:
                bodies[key].append((path, fn.lineno, qual))
        for node in tree.body:
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and not node.name.startswith('_')
                    and node.name not in DUPLICATE_NAME_EXEMPT):
                names[node.name].append((path, node.lineno))
    for group in bodies.values():
        if len({p for p, _, _ in group}) < 2:
            continue
        for path, line, qual in group:
            other = next(f'{p}:{ln} {q}' for p, ln, q in group if p != path)
            out.append(Violation(path, line, 'duplicate-function',
                                 f'{qual} has the same body as {other}; keep one shared '
                                 'function and import it'))
    for name, where in names.items():
        if len({p for p, _ in where}) < 2:
            continue
        for path, line in where:
            others = ', '.join(sorted({module_name(p) for p, _ in where if p != path}))
            out.append(Violation(path, line, 'duplicate-public-name',
                                 f'public {name}() is also defined in {others}; keep one '
                                 'and import it, or rename the one that does something else'))
    return out


class CannotParse(Exception):
    """A scanned file is not valid Python."""


def check_sources(sources: dict[str, str]) -> list[Violation]:
    """Every rule over {repo-relative posix path: source text}."""
    modules = {module_name(p) for p in sources}
    out: list[Violation] = []
    trees = {}
    for path in sorted(sources):
        try:
            trees[path] = ast.parse(sources[path], filename=path)
        except SyntaxError as exc:
            raise CannotParse(f'{path}:{exc.lineno}: cannot parse ({exc.msg}); '
                              'fix the syntax error first') from None
        out.extend(_check_tree(path, sources[path], trees[path], modules))
    out.extend(check_cross(trees))
    return sorted(out)


def read_tree(root: Path = ROOT) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): p.read_text(encoding='utf-8')
            for p in sorted((root / PACKAGE).rglob('*.py'))
            if '__pycache__' not in p.parts}


# ── the ratchet ─────────────────────────────────────────────────────────────

def counts(violations) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for v in violations:
        per = out.setdefault(v.rule, {})
        per[v.path] = per.get(v.path, 0) + 1
    return {r: dict(sorted(out[r].items())) for r in sorted(out)}


def compare(violations, baseline) -> tuple[list[Violation], list[str]]:
    """(violations in a (rule, file) above its baseline, notes on fixed counts)."""
    now = counts(violations)
    new = [v for v in violations
           if now[v.rule][v.path] > baseline.get(v.rule, {}).get(v.path, 0)]
    notes = []
    for rule, per in sorted(baseline.items()):
        for path, allowed in sorted(per.items()):
            have = now.get(rule, {}).get(path, 0)
            if have < allowed:
                notes.append(f'{path}: {RULE_IDS.get(rule, "?")} {rule}: fixed '
                             f'{allowed - have}, run --write-baseline to lock it in')
    return new, notes


def dump_baseline(violations) -> str:
    return json.dumps(counts(violations), indent=1, sort_keys=True) + '\n'


def _read_baseline(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--write-baseline', action='store_true',
                    help='rewrite the baseline from the current tree')
    ap.add_argument('--allow-increase', action='store_true',
                    help='with --write-baseline: also raise counts (say why in the PR)')
    ap.add_argument('--list', action='store_true',
                    help='print every violation, baselined ones too')
    ap.add_argument('--root', type=Path, default=ROOT, help=argparse.SUPPRESS)
    ap.add_argument('--baseline', type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    baseline_path = args.baseline or args.root / 'scripts' / 'best_practices_baseline.json'

    try:
        violations = check_sources(read_tree(args.root))
    except CannotParse as exc:
        print(exc)
        return 1
    if args.list:
        for v in violations:
            print(v)
        for rule, per in counts(violations).items():
            print(f'{RULE_IDS[rule]} {rule}: {sum(per.values())} in {len(per)} files')
        return 0
    if args.write_baseline:
        old = _read_baseline(baseline_path)
        raised = [(r, f, n, old.get(r, {}).get(f, 0))
                  for r, per in counts(violations).items() for f, n in per.items()
                  if n > old.get(r, {}).get(f, 0)]
        if raised and baseline_path.exists() and not args.allow_increase:
            for rule, path, n, was in raised:
                print(f'{path}: {RULE_IDS[rule]} {rule}: {was} -> {n}')
            print('These counts would go UP. Fix the new violations, or rerun with '
                  '--allow-increase and say why in the PR (a renamed file shows up '
                  'here too: its old entry drops and the new name is raised).')
            return 1
        baseline_path.write_text(dump_baseline(violations), encoding='utf-8')
        print(f'wrote {baseline_path.name}: {len(violations)} baselined violations')
        return 0

    baseline = _read_baseline(baseline_path)
    new, notes = compare(violations, baseline)
    for note in notes:
        print(note)
    if new:
        now = counts(violations)
        print(f'\n{len({(v.rule, v.path) for v in new})} (rule, file) counts went up. '
              'Each line below is one occurrence in such a file; the new one is '
              'among them:')
        for v in new:
            print(f'{v}  [{now[v.rule][v.path]} here, baseline allows '
                  f'{baseline.get(v.rule, {}).get(v.path, 0)}]')
        return 1
    print(f'best practices: OK ({len(violations)} baselined violations, none new)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
