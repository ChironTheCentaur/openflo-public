#!/usr/bin/env python
"""US spelling in what users read, with one word table for a later relabel.

    python scripts/us_spelling.py           # list British spellings; exit 1 if any
    python scripts/us_spelling.py --list    # every US word that has a British form

The app's text is written in US English (Sky, 2026-11-18). WORDS below is
the one table of words whose spelling differs between US and British
English. It drives both directions:

* the check (also run by tests/test_us_spelling.py) fails on a British form
  in user-visible text;
* ``--list`` prints every place a US form of a table word appears in that
  text, file:line and word, so a later British/European edition can be
  relabeled from one list instead of a hunt through the code.

User-visible text means: string literals in src/openflo (docstrings and
comments are left out, and so are bare lowercase tokens such as state keys
and matplotlib color names, which users never read), the in-app guide
(src/openflo/guide/*.md), and the phone app's string literals and page text
(mobile/src/**/*.ts, mobile/index.html) when mobile/ is present.

Run from the repository root.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (British stem, US stem). A word matches when it starts with the stem and
# goes on with one of the stem's endings (SUFFIXES, or the listed ones).
_IZE = ('organis', 'reorganis', 'normalis', 'optimis', 'initialis', 'recognis',
        'summaris', 'visualis', 'minimis', 'maximis', 'customis', 'prioritis',
        'standardis', 'synchronis', 'serialis', 'finalis', 'utilis',
        'specialis', 'authoris', 'categoris', 'realis', 'randomis',
        'characteris', 'parameteris', 'discretis', 'binaris', 'linearis',
        'regularis', 'stabilis', 'emphasis', 'apologis', 'memoris',
        'personalis', 'localis', 'materialis', 'tokenis', 'vectoris',
        'sanitis', 'centralis', 'generalis', 'harmonis', 'symbolis',
        'quantis', 'digitis', 'polymeris', 'immunis', 'ionis', 'hybridis',
        'crystallis', 'neutralis', 'capitalis', 'criticis', 'familiaris')
_ISE_ENDINGS = ('e', 'es', 'ed', 'er', 'ers', 'ing', 'ation', 'ations',
                'able', 'ability')
WORDS: list[tuple[str, str, tuple[str, ...]]] = [
    *[(s, s[:-1] + 'z', _ISE_ENDINGS) for s in _IZE],
    ('analys', 'analyz', ('e', 'ed', 'er', 'ers', 'ing', 'able')),
    ('colour', 'color', ('', 's', 'ed', 'ing', 'ful', 'less', 'blind',
                         'way', 'ways', 'map', 'maps', 'bar', 'bars')),
    ('behaviour', 'behavior', ('', 's', 'al', 'ally')),
    ('favour', 'favor', ('', 's', 'ed', 'ing', 'able', 'ite', 'ites')),
    ('honour', 'honor', ('', 's', 'ed', 'ing')),
    ('neighbour', 'neighbor', ('', 's', 'ing', 'hood', 'hoods')),
    ('labour', 'labor', ('', 's', 'ed', 'ing')),
    ('humour', 'humor', ('',)),
    ('tumour', 'tumor', ('', 's', 'al')),
    ('odour', 'odor', ('', 's', 'less')),
    ('rigour', 'rigor', ('',)),
    ('vigour', 'vigor', ('',)),
    ('armour', 'armor', ('', 's', 'ed')),
    ('endeavour', 'endeavor', ('', 's', 'ed', 'ing')),
    ('flavour', 'flavor', ('', 's')),
    ('vapour', 'vapor', ('', 's')),
    ('centre', 'center', ('', 's')),
    ('centred', 'centered', ('',)),
    ('centring', 'centering', ('',)),
    ('fibre', 'fiber', ('', 's')),
    ('litre', 'liter', ('', 's')),
    ('millimetre', 'millimeter', ('', 's')),
    ('micrometre', 'micrometer', ('', 's')),
    ('nanometre', 'nanometer', ('', 's')),
    ('centimetre', 'centimeter', ('', 's')),
    ('metre', 'meter', ('', 's')),
    ('licence', 'license', ('', 's')),
    ('defence', 'defense', ('', 's')),
    ('offence', 'offense', ('', 's')),
    ('grey', 'gray', ('', 's', 'ed', 'ing', 'ish', 'scale')),
    ('cancell', 'cancel', ('ed', 'ing')),
    ('modell', 'model', ('ed', 'ing', 'er', 'ers')),
    ('labell', 'label', ('ed', 'ing', 'er', 'ers')),
    ('travell', 'travel', ('ed', 'ing', 'er', 'ers')),
    ('signall', 'signal', ('ed', 'ing')),
    ('channell', 'channel', ('ed', 'ing')),
    ('levell', 'level', ('ed', 'ing')),
    ('totall', 'total', ('ed', 'ing')),
    ('fuell', 'fuel', ('ed', 'ing')),
    ('tunnell', 'tunnel', ('ed', 'ing')),
    ('catalogue', 'catalog', ('', 's')),
    ('catalogued', 'cataloged', ('',)),
    ('cataloguing', 'cataloging', ('',)),
    ('dialogue', 'dialog', ('', 's')),
    ('judgement', 'judgment', ('', 's')),
    ('acknowledgement', 'acknowledgment', ('', 's')),
    ('artefact', 'artifact', ('', 's')),
    ('ageing', 'aging', ('',)),
    ('learnt', 'learned', ('',)),
    ('spelt', 'spelled', ('',)),
    ('whilst', 'while', ('',)),
    ('amongst', 'among', ('',)),
    ('aluminium', 'aluminum', ('',)),
    ('sulphur', 'sulfur', ('', 's', 'ic', 'ous')),
    ('sulphate', 'sulfate', ('', 's')),
    ('sulphide', 'sulfide', ('', 's')),
    ('haemoglobin', 'hemoglobin', ('',)),
    ('haematocrit', 'hematocrit', ('',)),
    ('oestrogen', 'estrogen', ('', 's')),
    ('mould', 'mold', ('', 's', 'ed', 'ing')),
    ('programme', 'program', ('', 's')),
    ('enrol', 'enroll', ('', 's')),
    ('enrolment', 'enrollment', ('', 's')),
    ('fulfil', 'fulfill', ('', 's')),
    ('fulfilment', 'fulfillment', ('',)),
]

# A table word may follow one of these prefixes in the same word
# ("Relabelled", "microlitre", "unrecognised"); a prefix after a hyphen
# ("re-labelled") is a separate word and matches anyway.
PREFIXES = ('re', 'mis', 'un', 'pre', 'de', 'non', 'sub', 'over', 'under',
            'multi', 'micro', 'milli', 'nano', 'centi', 'kilo', 'deci')


def _alternation(pairs: list[tuple[str, tuple[str, ...]]]) -> re.Pattern[str]:
    words = sorted({stem + end for stem, ends in pairs for end in ends},
                   key=len, reverse=True)
    prefix = '(?:' + '|'.join(sorted(PREFIXES, key=len, reverse=True)) + ')?'
    return re.compile(r'(?<!\w)(' + prefix + '(?:'
                      + '|'.join(map(re.escape, words)) + r'))(?![\w])', re.I)


BRITISH = _alternation([(uk, ends) for uk, _us, ends in WORDS])
AMERICAN = _alternation([(us, ends) for _uk, us, ends in WORDS])
_TO_US = {uk + end: us + end for uk, us, ends in WORDS for end in ends}


def us_form(word: str) -> str:
    """The US spelling of a British table word, with its capitals kept."""
    low = word.lower()
    if low in _TO_US:
        us = _TO_US[low]
    else:
        pre = next(p for p in sorted(PREFIXES, key=len, reverse=True)
                   if low.startswith(p) and low[len(p):] in _TO_US)
        us = pre + _TO_US[low[len(pre):]]
    if word.isupper():
        return us.upper()
    if word[:1].isupper():
        return us[:1].upper() + us[1:]
    return us


def british_words(text: str) -> list[str]:
    """British table words in text, in order."""
    return [m.group(1) for m in BRITISH.finditer(text)]


def american_words(text: str) -> list[str]:
    """US table words in text: the relabel inventory."""
    return [m.group(1) for m in AMERICAN.finditer(text)]


# --- where users read text --------------------------------------------------

# Literals that look like words but name code: matplotlib colormaps.
NOT_TEXT = {'Greys', 'Greys_r'}


def is_user_text(s: str) -> bool:
    """A string literal users may read: it has a space, a capital, or ends a
    sentence. Bare lowercase tokens ('cancelled' as a state key, 'grey' as a
    matplotlib color) are code, not text."""
    return (' ' in s or s[:1].isupper() or s.rstrip().endswith(('.', '…', ':'))) \
        and not s.isupper() and s not in NOT_TEXT


def python_texts(path: Path):
    """(line, text) for each user-readable string literal, docstrings left out."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Module)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docs.add(id(first.value))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docs and is_user_text(node.value)):
            yield node.lineno, node.value


_LITERAL = re.compile(r"""(['"`])((?:\\.|(?!\1).)*)\1""")
_TS_COMMENT = re.compile(r'^\s*(//|\*|/\*)')


def script_texts(path: Path):
    """(line, text) for each user-readable string literal of a TS file."""
    for i, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if _TS_COMMENT.match(line):
            continue
        for _q, body in _LITERAL.findall(line):
            if is_user_text(body):
                yield i, body


def page_texts(path: Path):
    """(line, text) for each line of a guide or page."""
    yield from enumerate(path.read_text(encoding='utf-8').splitlines(), 1)


PHONE_DIR = 'mobile'   # the phone app's folder in a checkout (not in the sdist)


def sources(root: Path | None = None):
    """(path, reader) for every file whose text users read."""
    root = root or ROOT
    out = [(p, python_texts) for p in sorted((root / 'src/openflo').rglob('*.py'))]
    out += [(p, page_texts) for p in sorted((root / 'src/openflo/guide').glob('*.md'))]
    mobile = root / PHONE_DIR
    if (mobile / 'src').is_dir():
        out += [(p, script_texts) for p in sorted((mobile / 'src').rglob('*.ts'))]
    if (mobile / 'index.html').is_file():
        out.append((mobile / 'index.html', page_texts))
    return out


def find(words_in, root: Path | None = None):
    """(relative path, line, word) for each table word words_in finds."""
    root = root or ROOT
    hits = []
    for path, reader in sources(root):
        for line, text in reader(path):
            hits += [(path.relative_to(root).as_posix(), line, w)
                     for w in words_in(text)]
    return hits


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ['--list']:
        for rel, line, word in find(american_words):
            print(f'{rel}:{line}: {word}')
        return 0
    hits = find(british_words)
    for rel, line, word in hits:
        print(f'{rel}:{line}: {word} -> {us_form(word)}')
    if hits:
        print(f'us spelling: {len(hits)} British spelling(s) in user text.')
        return 1
    print('us spelling: clean')
    return 0


if __name__ == '__main__':
    sys.exit(main())
