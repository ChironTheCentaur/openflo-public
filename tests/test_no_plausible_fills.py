"""Guard against re-introducing the "plausible number for an unknown" bug.

This shape has produced six real defects in this codebase, every one of which
shipped and none of which any test noticed:

  calibration.fit_mesf_calibration   r2 = 1.0 when r2 is UNDEFINED
                                     -> a perfect fit for a useless calibration
  trajectory._geodesic_pseudotime    unreachable cells filled with the MAX
                                     -> a corrupt event at the trajectory end
  trajectory.pseudotime_trends       NaN pseudotime clipped into the LAST bin
                                     -> 1.0 became 600.4 in a published curve
  trajectory.robust_root             index 0 when the score is unusable
                                     -> a trajectory rooted on an arbitrary cell
  stats.compare_groups               sd = 0.0 for n=1
                                     -> "this group had zero variability"
  pipeline.filter_doublets           median 0.0 -> window [0,0]
                                     -> the ENTIRE sample deleted, silently

The common failure is NOT missing error handling — every one of these had a
guard. The guard noticed the degeneracy and then chose a value that reads as a
successful measurement instead of one that means "unknown". `NaN`, `None` and
raising are all honest. A finite number is a claim.

One entry in ALLOWED was itself an example of the bug. `preview.py`'s -999
was excused as "an internal sentinel, never reported to the user"; it was in
fact passed straight into the quadrant annotation, so a plot with only one FMO
threshold still displayed four quadrant percentages, and on a compensated
channel whose dim population sits near -2000 the sentinel fell inside the data
and split it 67.9 / 7.0 / 23.2 / 2.0. The lesson is not that allow-lists are
bad but that an entry asserting "this never reaches the user" has to be
TRACED, not assumed — the same standard as the code it excuses.

So this test scans for the shape and fails on anything new. When it fires,
prefer NaN / None / raise. If a finite fallback really is right, add the site
to ALLOWED with a reason that says why the number is *measured or conventional*
rather than *assumed* — and note that "it keeps the pipeline running" is the
reasoning that produced all six bugs above.
"""
from __future__ import annotations

import ast
import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'openflo'

# Modules that compute values a user reads, exports or publishes. UI/geometry
# modules are excluded: a default width or colour is not a measurement.
ANALYSIS = {
    'pipeline.py', 'stats.py', 'diffexp.py', 'compare.py', 'calibration.py',
    'spectral.py', 'density.py', 'trajectory.py', 'comp_qc.py', 'voltage.py',
    'interop.py', 'gating.py', 'scales.py', 'annotate.py', 'dr_compare.py',
    'gating_helpers.py', 'plotmath.py', 'compliance.py', 'provenance.py',
    'fcs_export.py', 'preview.py',
}

# LIMITS, stated so this guard is not mistaken for a proof:
# the test only matches a degeneracy check that uses a COMPARISON operator or a
# recognised predicate, so a bare truthiness test (`if arr.size else 1.0`) slips
# past, as does any fallback computed rather than written as a literal. It is a
# net for the common shape, not a proof of absence — the reviewed inventory in
# ALLOWED and the per-bug tests carry the real weight.
DEGENERATE = re.compile(
    r'\b(size|len|shape|count|sum|std|var|ptp|max|min)\b\s*[<>=!]|'
    r'==\s*0|<\s*[12]\b|>\s*0\b|isfinite|isnan|is None|not \w+|\.empty|'
    r'\.any\(\)|\.all\(\)')

# (module, fallback-value, reason it is NOT a fabricated measurement).
# Reviewed 2026-09-07. Keep the reasons specific; a vague entry here is the
# same failure this file exists to catch.
ALLOWED: dict[tuple[str, str, str], str] = {
    # Reviewed 2026-09-07, each one TRACED to what a user can actually see.
    # Note Python bools are ints, so a `return False` is scanned as 0.
    ('annotate.py', '_iqr', '0'):
        'IQR of a marker with no finite values. Traced: the MEM score for '
        'such a marker comes back NaN, not a number — measured on a cluster '
        'whose marker was entirely NaN, which scored NaN while the same '
        "cluster's real marker scored normally",
    ('annotate.py', 'mem_scores', '1'):
        'the NEUTRAL element of the MEM spread term, not a measurement: the '
        'score adds `ratio - 1`, so a ratio of 1.0 contributes exactly '
        'nothing. It is used when the population has no measurable spread, '
        'which is precisely the case where no claim about spread can be '
        'made — the alternative (letting `+ eps` stand in for a zero IQR) is '
        'the bug this replaced',
    ('diffexp.py', 'differential_abundance', '0'):
        'z when the standard error is 0. Traced: the p on that same branch is '
        'NaN, and z has no consumer outside this module, so nothing a user '
        'acts on is fabricated',
    ('dr_compare.py', '_backend_importable', '0'):
        'returns False for "backend not importable" — a predicate, not a '
        'measurement',
    ('interop.py', 'mds_embed', '1'):
        'MDS fill distance when every pairwise distance is degenerate. A '
        'layout unit for a relative embedding, not a measured distance, and '
        'deliberate: 0.0 collapsed every sample onto the origin, which was '
        'the previous bug',
    ('pipeline.py', 'parse_ellipsoid', '4'):
        "FlowJo's own default distanceSquare (2 SD) for an ellipse gate with "
        'no explicit value — a format convention, not a computed result',
    ('pipeline.py', '_extract_matrix', '0'):
        'a loop counter (`seen = 0`) for parsed spillover coefficients',
    ('pipeline.py', '_quantisation_step', '0'):
        '0.0 means "this channel has no range, so no scale can be taken from '
        'it". The one caller tests `not scale > 0` and returns an all-False '
        'mask, so the value is a handled sentinel and never reaches a result',
    ('pipeline.py', 'cluster', '1'):
        'GPU admission control — whether a GPU run is allowed. Control flow, '
        'not a reported value',
    ('pipeline.py', 'cluster', '0'):
        'a boolean mask assignment (`rest_mask[sub_idx] = False`) marking '
        'which events still need a label',
    ('plotmath.py', 'unplottable_count', '0'):
        'a COUNT of events the scale cannot represent. Zero is MEASURED, not '
        'assumed: with no finite values there are no events to hide, and on '
        'any scale other than log every finite value is displayable, so none '
        'is hidden. The alternative reading — "unknown" — does not exist here, '
        'because the question is answerable in both cases',
    ('plotmath.py', 'in_box', '0'):
        'returns False for a missing point or box — a predicate, not a '
        'measurement',
    ('voltage.py', 'main', '1'):
        'a CLI exit code, not a value',
}


def _finite_number(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        v = float(node.value)
        return v if v == v and abs(v) != float('inf') else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _finite_number(node.operand)
        return None if inner is None else -inner
    return None


def _scan_tree(tree: ast.AST, module: str) -> list[tuple[tuple[str, str, str],
                                                        int, str]]:
    """Sites in one parsed module, keyed by (module, function, value)."""
    found: list[tuple[tuple[str, str, str], int, str]] = []

    def visit(node, where):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            where = node.name
        if isinstance(node, ast.IfExp):
            val = _finite_number(node.orelse)
            test = ast.unparse(node.test)
            if val is not None and DEGENERATE.search(test):
                found.append(((module, where, f'{val:g}'), node.lineno, test))
        elif isinstance(node, ast.If) and DEGENERATE.search(
                ast.unparse(node.test)):
            for stmt in node.body:
                tgt = (stmt.value
                       if isinstance(stmt, (ast.Return, ast.Assign))
                       else None)
                val = _finite_number(tgt) if tgt is not None else None
                if val is not None:
                    found.append(((module, where, f'{val:g}'), stmt.lineno,
                                  ast.unparse(node.test)))
        for child in ast.iter_child_nodes(node):
            visit(child, where)

    visit(tree, '<module>')
    return found


def _scan() -> list[tuple[tuple[str, str, str], int, str]]:
    """Every degenerate-case-to-finite-number site in the analysis modules.

    The enclosing FUNCTION is part of the key on purpose. Keying only on
    (module, value) let a new fallback inherit an unrelated entry's excuse: a
    `0.0` added to pipeline.py was waved through by an entry written about
    compensation passthrough and loop counters, and this guard said nothing.
    An allow-list whose keys are coarser than its reasons is somewhere for
    this bug to hide.
    """
    found = []
    for path in sorted(SRC.glob('*.py')):
        if path.name in ANALYSIS:
            found.extend(_scan_tree(
                ast.parse(path.read_text(encoding='utf-8')), path.name))
    return found


def test_no_new_plausible_value_stands_in_for_unknown():
    unreviewed = [(key, ln, test) for key, ln, test in _scan()
                  if key not in ALLOWED]
    assert not unreviewed, (
        'A degenerate case returns a finite NUMBER where "unknown" may be the '
        'truth. This exact shape has produced six shipped bugs here — a '
        'calibration reporting a perfect fit, a corrupt event placed at the '
        'end of a trajectory, an entire sample silently deleted.\n\n'
        'Prefer NaN / None / raising. If a finite value really is correct, add '
        '(module, function, value) to ALLOWED in this file with a reason '
        'saying why the number is measured or conventional rather than '
        'assumed — and TRACE that reason rather than assuming it. One entry '
        'here already claimed a sentinel never reached the user; it was '
        'rendered onto a figure.\n\n'
        + '\n'.join(f'  {mod}:{ln}  in {fn}()  -> {val}   when  {test[:60]}'
                    for (mod, fn, val), ln, test in unreviewed))


def test_the_scanner_still_finds_the_shape_it_is_looking_for():
    """Guard the guard. If the scan silently stopped matching, the test above
    would pass vacuously — which is the very failure mode being policed."""
    sample = ast.parse(
        'def f(a):\n'
        '    return 1.0 - x if a.size > 1 else 1.0\n')
    hits = []
    for node in ast.walk(sample):
        if isinstance(node, ast.IfExp):
            val = _finite_number(node.orelse)
            if val is not None and DEGENERATE.search(ast.unparse(node.test)):
                hits.append(val)
    assert hits == [1.0], (
        'the scanner no longer recognises the calibration-r2 shape it was '
        'written to catch')


def test_every_allowed_entry_is_still_reachable():
    """An ALLOWED entry whose site no longer exists is stale — it would keep
    silently excusing a value that has moved or changed."""
    live = {key for key, _ln, _t in _scan()}
    stale = sorted(k for k in ALLOWED if k not in live)
    assert not stale, (
        'these ALLOWED entries no longer match any site and should be removed '
        f'so the list keeps meaning what it says: {stale}')


def test_a_new_fallback_is_not_excused_by_another_function_s_entry():
    """The hole this file had. `('pipeline.py', '0')` was written about
    compensation passthrough and loop counters, and it silently excused an
    unrelated 0.0 added later in a different function. Keys carry the function
    so a new site has to be reviewed on its own terms."""
    src = (
        'def already_reviewed(m):\n'
        '    if m is None:\n'
        '        return 0\n'
        'def added_later(a):\n'
        '    if a.size == 0:\n'
        '        return 0.0\n')
    keys = {k for k, _ln, _t in _scan_tree(ast.parse(src), 'mod.py')}
    assert ('mod.py', 'already_reviewed', '0') in keys
    assert ('mod.py', 'added_later', '0') in keys, (
        'the scanner no longer separates two functions in the same module')

    allowed = {('mod.py', 'already_reviewed', '0'): 'reviewed'}
    unreviewed = [k for k in keys if k not in allowed]
    assert unreviewed == [('mod.py', 'added_later', '0')], (
        'a fallback in a new function inherited an unrelated excuse')
