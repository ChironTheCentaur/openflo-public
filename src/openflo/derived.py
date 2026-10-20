"""Derived channels: a new channel computed per event from existing ones.

The standard way to read ratio dyes (a calcium or pH indicator whose
emission shifts, read as the ratio of two detectors, which does not depend
on how much dye a cell took up), FRET, and one marker normalised to another
per cell. A derived channel is written ``D:<name>`` beside the measured
ones, plotted, gated and summarised like them, kept in the session with its
formula, and recomputed from that formula when a sample is reloaded from
its FCS file.

A formula names channels in square brackets and may use numbers,
``+ - * / **``, parentheses and log, log10, log2, sqrt, exp, abs, asinh,
min and max: ``[FL1-A] / [FL2-A]``, ``log10([FL1-A] / [FL2-A])``. It is
evaluated on each channel's compensated LINEAR values (pipeline.
linear_values), never the display coordinates, and its result is linear.
An event where the result is not a finite number (0 / 0, the log of a
negative) gets NaN there: unknown, not a value.

Tk-free, so it is tested without a display.
"""
from __future__ import annotations

import ast
import re

import numpy as np

PREFIX = 'D:'
_REF = re.compile(r'\[([^\[\]]+)\]')
_FUNCS = {'log': np.log, 'log10': np.log10, 'log2': np.log2,
          'sqrt': np.sqrt, 'exp': np.exp, 'abs': np.abs,
          'asinh': np.arcsinh, 'min': np.minimum, 'max': np.maximum}
_BINOPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
           ast.Div: np.divide, ast.Pow: np.power}
_UNOPS = {ast.USub: np.negative, ast.UAdd: np.positive}


class FormulaError(ValueError):
    """A formula that can't be read, or names a channel that isn't there."""


def column_name(name):
    """The data column a derived channel called `name` is stored as."""
    name = str(name).strip()
    return name if name.startswith(PREFIX) else PREFIX + name


def is_derived(column):
    return str(column).startswith(PREFIX)


class Formula:
    """A parsed formula: ``refs`` the channel names it uses, in order of
    first use, and ``__call__(values)`` evaluating it on ``{channel:
    array}``."""

    def __init__(self, text):
        self.text = str(text).strip()
        if not self.text:
            raise FormulaError("The formula is empty.")
        self.refs: list[str] = []
        var = {}

        def sub(m):
            ref = m.group(1).strip()
            if ref not in var:
                var[ref] = f'_c{len(var)}'
                self.refs.append(ref)
            return var[ref]
        code = _REF.sub(sub, self.text)
        if '[' in code or ']' in code:
            raise FormulaError("Unbalanced [ ] around a channel name.")
        try:
            tree = ast.parse(code, mode='eval')
        except SyntaxError as exc:
            raise FormulaError(f"Can't read the formula: {exc.msg}.") from None
        self._names = {v: k for k, v in var.items()}
        self._check(tree.body)
        self._tree = tree.body
        if not self.refs:
            raise FormulaError("The formula uses no channel; name one in "
                               "square brackets, e.g. [FL1-A].")

    def _check(self, node):
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            self._check(node.left)
            self._check(node.right)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
            self._check(node.operand)
        elif isinstance(node, ast.Constant) and isinstance(
                node.value, (int, float)) and not isinstance(node.value,
                                                             bool):
            pass
        elif isinstance(node, ast.Name):
            if node.id not in self._names:
                raise FormulaError(
                    f"'{node.id}' isn't a channel or a function. Put channel "
                    "names in square brackets, e.g. [FL1-A].")
        elif isinstance(node, ast.Call):
            fn = node.func
            if not (isinstance(fn, ast.Name) and fn.id in _FUNCS):
                raise FormulaError(
                    "Only these functions can be used: "
                    + ', '.join(sorted(_FUNCS)) + '.')
            if node.keywords:
                raise FormulaError("Functions take no named arguments.")
            want = 2 if fn.id in ('min', 'max') else 1
            if len(node.args) != want:
                raise FormulaError(f"{fn.id}() takes {want} argument"
                                   f"{'s' if want > 1 else ''}.")
            for a in node.args:
                self._check(a)
        else:
            raise FormulaError("Only numbers, channels in [ ], + - * / **, "
                               "parentheses and "
                               + ', '.join(sorted(_FUNCS))
                               + " can be used.")

    def __call__(self, values):
        missing = [r for r in self.refs if r not in values]
        if missing:
            raise FormulaError("Not in this sample: " + ', '.join(missing))

        def ev(node):
            if isinstance(node, ast.BinOp):
                return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
            if isinstance(node, ast.UnaryOp):
                return _UNOPS[type(node.op)](ev(node.operand))
            if isinstance(node, ast.Constant):
                v = node.value          # _check let only int / float through
                assert isinstance(v, (int, float))
                return float(v)
            if isinstance(node, ast.Name):
                return np.asarray(values[self._names[node.id]], dtype=float)
            assert isinstance(node, ast.Call) and isinstance(node.func,
                                                             ast.Name)
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        n = len(np.asarray(values[self.refs[0]]))
        with np.errstate(all='ignore'):
            out = np.broadcast_to(np.asarray(ev(self._tree), dtype=float),
                                  (n,)).copy()
        out[~np.isfinite(out)] = np.nan
        return out


def resolve_refs(formula, channels, labels=None):
    """`formula` text with each bracketed name that is a marker LABEL
    (``labels``: column -> label) replaced by its column, so ``[CD4]`` and
    ``[FL1-A]`` both work. A name that is neither stays as typed, and
    evaluation then says it isn't in the sample."""
    labels = labels or {}
    by_label = {}
    for col, lab in labels.items():
        by_label.setdefault(str(lab), col)
    chans = set(map(str, channels))

    def sub(m):
        ref = m.group(1).strip()
        if ref in chans:
            return f'[{ref}]'
        return f'[{by_label.get(ref, ref)}]'
    return _REF.sub(sub, str(formula))


def compute(sample, formula):
    """`formula` (a Formula or its text) evaluated on `sample`'s linear
    values. Raises FormulaError for a missing channel and
    pipeline.UnknownScaleError for a channel whose scale is unknown."""
    from .pipeline import linear_values
    f = formula if isinstance(formula, Formula) else Formula(formula)
    cols = sample.data.columns
    missing = [r for r in f.refs if r not in cols]
    if missing:
        raise FormulaError("Not in this sample: " + ', '.join(missing))
    return f({r: linear_values(sample, r) for r in f.refs})


def add_to_sample(sample, name, formula):
    """Compute derived channel `name` from `formula` and store it in
    `sample` (column ``D:<name>``, linear, its formula in
    ``sample.derived``). Returns the column name."""
    col = column_name(name)
    if col == PREFIX:
        raise FormulaError("Give the channel a name.")
    values = compute(sample, formula)
    sample.data[col] = values
    rec = getattr(sample, 'derived', None)
    if not isinstance(rec, dict):
        rec = {}
    rec[col] = str(formula if not isinstance(formula, Formula)
                   else formula.text)
    sample.derived = rec
    # Linear values, like MESF: and U: columns: no transform to undo.
    tr = getattr(sample, 'data_transforms', None)
    if isinstance(tr, dict):
        tr.pop(col, None)
    return col


def restore(sample, definitions):
    """Put a saved sample's derived channels back: attach their formulas
    and compute any column the reloaded data lacks (a reload from the raw
    FCS). Returns ``[(column, error)]`` for those that could not be."""
    from .pipeline import UnknownScaleError
    failed = []
    for col, text in (definitions or {}).items():
        try:
            if col in sample.data.columns:
                rec = getattr(sample, 'derived', None)
                rec = rec if isinstance(rec, dict) else {}
                rec[col] = str(text)
                sample.derived = rec
            else:
                add_to_sample(sample, col, text)
        except (FormulaError, UnknownScaleError) as exc:
            failed.append((col, str(exc)))
    return failed


def apply_to_samples(samples, name, formula, labels=None):
    """Add derived channel `name` to every sample in ``{name: sample}``.
    Marker labels in the formula are resolved per sample (its own
    ``channel_labels`` over `labels`). Returns ``(column, applied names,
    [(sample, reason)] not applied)``; the formula is checked once first,
    so a typo raises FormulaError before any sample is touched."""
    from .pipeline import UnknownScaleError
    Formula(formula)
    col = column_name(name)
    if col == PREFIX:
        raise FormulaError("Give the channel a name.")
    applied, failed = [], []
    for nm, s in samples.items():
        lab = dict(labels or {})
        lab.update(getattr(s, 'channel_labels', None) or {})
        text = resolve_refs(formula, s.data.columns, lab)
        try:
            add_to_sample(s, col, text)
            applied.append(nm)
        except (FormulaError, UnknownScaleError) as exc:
            failed.append((nm, str(exc)))
    return col, applied, failed


def remove_from_sample(sample, column):
    """Drop derived `column` and its formula from `sample`."""
    if column in sample.data.columns:
        sample.data.drop(columns=[column], inplace=True)
    rec = getattr(sample, 'derived', None)
    if isinstance(rec, dict):
        rec.pop(column, None)


def gates_using(gates_by_sample, column):
    """Sample names whose gates are drawn on `column`."""
    out = []
    for nm, gates in (gates_by_sample or {}).items():
        for g in (gates or {}).values():
            if column in (g.get('x_channel'), g.get('y_channel'),
                          g.get('channel')):
                out.append(nm)
                break
    return out
