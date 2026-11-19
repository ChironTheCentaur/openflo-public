"""Read an antibody vial label: its marker and fluor, and the detector that
reads that fluor on the user's instrument.

The text is a vial's printed label as a user pastes it or OCR reads it, in
any of the usual shapes ('<fluor> anti-human <marker> Antibody', '<marker>
<fluor>', '<fluor> Mouse Anti-Human <marker>'), often with catalog, lot and
clone lines and OCR noise (stray spaces, registered and trademark signs,
mixed case). ``parse_vial_label`` pulls out the marker, the fluor (a key of
fluor_catalog.FLUOR_SPECTRA), the target species, clone, catalog number and
lot. ``primary_detector`` picks the detector that sees the fluor best from
panel_check's spectral model; ``detectors_from_names`` turns an FCS file's
detector names (and its $PnL / $PnF keywords, through ``fcs_optics``) into
the detectors that model needs; ``assign`` maps a panel of vials to
``{detector: marker}`` and says which vials need the user.

The commercial fluor spellings live in fluor_catalog (FLUOR_LONG_FORMS,
FLUOR_ALIASES, FLUOR_FAMILIES) and the non-CD marker names in marker_catalog;
none is written here. Pure Python and numpy, no Tk.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .fcs_text import keyword_value
from .fluor_catalog import (
    FLUOR_ALIASES,
    FLUOR_FAMILIES,
    FLUOR_LONG_FORMS,
    FLUOR_SPECTRA,
)
from .marker_catalog import MARKER_FAMILIES, MARKER_SYNONYMS, NAMED_MARKERS
from .panel_check import DETECTOR_SETS, Detector, model_signature
from .shared_clustering import fold_label

# The longest run of label words one fluor or marker name spans (a vendor's
# long form of a fluor is up to 3 words, as is a marker synonym).
MAX_NAME_WORDS = 5

# ── Label text ───────────────────────────────────────────────────────────────

# Registered / trademark / copyright signs, as printed or as OCR spells them.
_SIGNS = re.compile(r'[®™©℠]|\((?:R|TM|C)\)', re.I)

# A label word: anything between whitespace and the separators vendors put
# inside names ('/', '-', en dash) or between fields.
_WORD = re.compile(r'[^\s/\-–,;:()\[\]]+')


@dataclass(frozen=True)
class _Word:
    """One label word: its text and where it sits in the cleaned label."""
    text: str
    start: int
    end: int


def clean_label_text(text: str) -> str:
    """`text` with the registered, trademark and copyright signs dropped and
    compatibility characters folded (NFKC), so that every offset the parser
    reports refers to this string."""
    return unicodedata.normalize('NFKC', _SIGNS.sub('', str(text)))


def _words(text: str) -> list[_Word]:
    return [_Word(m.group(0), m.start(), m.end()) for m in _WORD.finditer(text)]


def _runs(words: Sequence[_Word], max_words: int = MAX_NAME_WORDS
          ) -> Iterator[tuple[int, int, str]]:
    """``(i, j, folded)`` for every run of 1..`max_words` consecutive words
    ``words[i:j]``, folded as one name."""
    for i in range(len(words)):
        folded = ''
        for j in range(i, min(len(words), i + max_words)):
            folded += fold_label(words[j].text)
            if folded:
                yield i, j + 1, folded


def _overlaps(start: int, end: int, spans: Iterable[tuple[int, int]]) -> bool:
    return any(start < b and a < end for a, b in spans)


# ── Fluor ────────────────────────────────────────────────────────────────────

def canonical_fluor_fold(folded: str) -> str:
    """A folded fluor name with every vendor long form folded to its short
    one (fluor_catalog.FLUOR_LONG_FORMS), the form fluor names compare in."""
    for long_form, short_form in FLUOR_LONG_FORMS:
        folded = folded.replace(fold_label(long_form), fold_label(short_form))
    return folded


@lru_cache(maxsize=1)
def _fluor_index() -> dict[str, str]:
    """``{canonical fold: FLUOR_SPECTRA key}`` for every catalogue key and
    alias."""
    index = {canonical_fluor_fold(fold_label(k)): k for k in FLUOR_SPECTRA}
    for alias, key in FLUOR_ALIASES.items():
        index.setdefault(canonical_fluor_fold(fold_label(alias)), key)
    return index


_FAMILY = re.compile(r'(?:' + '|'.join(map(re.escape, FLUOR_FAMILIES))
                     + r')\d{3}')

# Words that may follow a fluor without making it a tandem
# ('<fluor>-conjugated', '<fluor>-anti-human', '<fluor> Antibody, 100 tests'),
# folded. Target species and markers never start an acceptor either.
_NOT_ACCEPTOR = frozenset(fold_label(w) for w in (
    'anti', 'conjugated', 'conjugate', 'labeled', 'labelled', 'antibody',
    'antibodies', 'mab', 'monoclonal', 'polyclonal', 'recombinant',
    'purified', 'clone', 'cat', 'catalog', 'catalogue', 'ref', 'lot',
    'isotype', 'size', 'test', 'tests', 'kit', 'reagent', 'vol', 'volume',
    'ug', 'µg', 'mg', 'ml', 'µl', 'ul'))


@dataclass(frozen=True)
class _FluorHit:
    key: str | None          # FLUOR_SPECTRA key, None when not catalogued
    first: int               # word index range [first, last)
    last: int


def _known_fluor(words: Sequence[_Word], skip: list[tuple[int, int]]
                 ) -> _FluorHit | None:
    """The longest run of words naming a catalogued fluor (a tandem is
    longer than its donor alone); the earliest of equal length."""
    index = _fluor_index()
    best: tuple[int, int, int] | None = None     # (length, i, j)
    for i, j, folded in _runs(words):
        if canonical_fluor_fold(folded) not in index:
            continue
        if _overlaps(words[i].start, words[j - 1].end, skip):
            continue
        if best is None or len(folded) > best[0]:
            best = (len(folded), i, j)
    if best is None:
        return None
    _n, i, j = best
    folded = ''.join(fold_label(w.text) for w in words[i:j])
    return _FluorHit(index[canonical_fluor_fold(folded)], i, j)


def _family_fluor(words: Sequence[_Word], skip: list[tuple[int, int]]
                  ) -> _FluorHit | None:
    """The first run of words naming a fluor of a known family that the
    catalogue does not list (a family word and a three-digit number)."""
    for i, j, folded in _runs(words, 3):
        if (_FAMILY.fullmatch(canonical_fluor_fold(folded))
                and not _overlaps(words[i].start, words[j - 1].end, skip)):
            return _FluorHit(None, i, j)
    return None


@lru_cache(maxsize=1)
def _tandem_parts() -> tuple[frozenset[str], re.Pattern[str]]:
    """``(donors, acceptor family pattern)`` from the catalogue: the
    FLUOR_SPECTRA keys another key extends with '-' (the tandem donors), and
    a pattern for one folded acceptor word or run: a FLUOR_FAMILIES word or
    the leading letters of a catalogued tandem's acceptor, then a number (a
    name of three letters or more may stand without one)."""
    donors = frozenset(d for d in FLUOR_SPECTRA
                       if any(k.startswith(d + '-') for k in FLUOR_SPECTRA))
    families: set[str] = set(FLUOR_FAMILIES)
    for k in FLUOR_SPECTRA:
        for d in donors:
            if k.startswith(d + '-'):
                acceptor = canonical_fluor_fold(fold_label(k[len(d) + 1:]))
                lead = re.match(r'[a-z]+', acceptor)
                if lead:
                    families.add(lead.group(0))
    alternatives = sorted(
        (re.escape(f) + (r'\d*' if len(f) >= 3 else r'\d+') for f in families),
        key=len, reverse=True)
    return donors, re.compile('(?:' + '|'.join(alternatives) + ')')


@lru_cache(maxsize=1)
def _species_words() -> frozenset[str]:
    return frozenset(fold_label(s) for s in (*SPECIES, *_SPECIES_ABBREVIATIONS))


def _gap(text: str, words: Sequence[_Word], k: int) -> str:
    """The text between word ``k - 1`` and word `k`."""
    return text[words[k - 1].end:words[k].start]


def _joined(gap: str) -> bool:
    """Whether a gap joins two words of one fluor name: spaces, '/', '-' or
    an en dash, on one line."""
    return bool(gap) and '\n' not in gap and gap.strip() in ('', '/', '-', '–')


def _marker_starts(text: str, words: Sequence[_Word], k: int) -> bool:
    """Whether a marker name begins at word `k` ('PD' of '<fluor>-PD-1')."""
    if _CD.match(text, words[k].start):
        return True
    folded = ''
    for j in range(k, min(len(words), k + MAX_NAME_WORDS)):
        if j > k and not _joined(_gap(text, words, j)):
            break
        folded += fold_label(words[j].text)
        if _named_marker(folded) is not None:
            return True
    return False


def _may_be_acceptor(text: str, words: Sequence[_Word], k: int) -> bool:
    """Whether word `k` may be (the start of) a tandem's acceptor: it starts
    with a letter and is no species, marker, field key ('Lot', 'Clone') or
    filler word."""
    w = words[k]
    folded = fold_label(w.text)
    return (w.text[:1].isalpha() and folded not in _NOT_ACCEPTOR
            and folded not in _species_words()
            and not _marker_starts(text, words, k))


def _fluor_run_end(text: str, words: Sequence[_Word], k: int) -> int | None:
    """Index past the longest run of up to 3 joined words from `k` that
    names a catalogued fluor or an acceptor family (``_tandem_parts``),
    else None."""
    index = _fluor_index()
    family = _tandem_parts()[1]
    best = None
    folded = ''
    for j in range(k, min(len(words), k + 3)):
        if j > k and not _joined(_gap(text, words, j)):
            break
        folded += fold_label(words[j].text)
        canon = canonical_fluor_fold(folded)
        if canon in index or family.fullmatch(canon) or _FAMILY.fullmatch(canon):
            best = j + 1
    return best


def _spaced_acceptor_end(text: str, words: Sequence[_Word], k: int,
                         donor: bool) -> int | None:
    """Index past an acceptor that follows a fluor after a space, starting
    at word `k`, else None. It is a run naming a fluor or an acceptor family
    ('<donor> <family>999', two catalogued fluors side by side) or, after a
    donor, up to 3 acceptor words followed by a number ('<donor> <new name>
    640'; the number is taken by ``_acceptor_end``)."""
    end = _fluor_run_end(text, words, k)
    if end is not None:
        return end
    if not donor:
        return None
    j = k
    while (j < len(words) and j - k < 3
           and (j == k or _gap(text, words, j).strip() == ''
                and _joined(_gap(text, words, j)))
           and words[j].text.isalpha()
           and _may_be_acceptor(text, words, j)):
        j += 1
    if (k < j < len(words) and words[j].text.isdigit()
            and _gap(text, words, j).strip() == ''
            and _joined(_gap(text, words, j))):
        return j
    return None


def _acceptor_end(text: str, words: Sequence[_Word], hit: _FluorHit) -> int:
    """Index past the last word of an uncatalogued acceptor joined to `hit`
    ('<donor>/<new acceptor> 640', '<donor> <new acceptor> 640',
    '<donor>-<family>999'), or ``hit.last`` when none follows. A '/', '-' or
    en dash joins any acceptor word; a space joins only what looks like an
    acceptor (``_spaced_acceptor_end``). A catalogued fluor read alone would
    hide that the vial is a tandem on another detector."""
    donor = hit.key in _tandem_parts()[0]
    k = hit.last
    while (k < len(words) and _joined(_gap(text, words, k))
           and _may_be_acceptor(text, words, k)):
        if _gap(text, words, k).strip():
            k += 1
            continue
        end = _spaced_acceptor_end(text, words, k, donor or k > hit.last)
        if end is None:
            break
        k = end
    if (k > hit.last and k < len(words) and words[k].text.isdigit()
            and _gap(text, words, k).strip() == ''
            and _joined(_gap(text, words, k))):
        k += 1
    return k


def _fluor_before(text: str, words: Sequence[_Word], hit: _FluorHit,
                  skip: list[tuple[int, int]]) -> int:
    """Index of the first word of a catalogued or family fluor joined right
    before `hit` ('<donor> <catalogued acceptor>': a tandem the catalogue
    does not list, never its acceptor alone), or ``hit.first`` when none
    is."""
    i = hit.first
    if i == 0 or not _joined(_gap(text, words, i)):
        return i
    index = _fluor_index()
    best = i
    folded = ''
    for a in range(i - 1, max(-1, i - 1 - MAX_NAME_WORDS), -1):
        if a < i - 1 and not _joined(_gap(text, words, a + 1)):
            break
        if _overlaps(words[a].start, words[a].end, skip):
            break
        folded = fold_label(words[a].text) + folded
        canon = canonical_fluor_fold(folded)
        if canon in index or _FAMILY.fullmatch(canon):
            best = a
    return best


def _find_fluor(text: str, words: Sequence[_Word],
                skip: list[tuple[int, int]]) -> tuple[str | None, str | None,
                                                      tuple[int, int] | None]:
    """``(key, fluor_text, span)`` of the label's fluor. A fluor joined to
    another fluor or acceptor-looking words before or after it is a tandem
    the catalogue does not list: key None, its whole run as fluor_text."""
    hit = _known_fluor(words, skip) or _family_fluor(words, skip)
    if hit is None:
        return None, None, None
    first = _fluor_before(text, words, hit, skip)
    end = _acceptor_end(text, words, hit)
    key = hit.key if (first, end) == (hit.first, hit.last) else None
    span = (words[first].start, words[end - 1].end)
    return key, text[span[0]:span[1]], span


# ── Marker ───────────────────────────────────────────────────────────────────

# A CD marker: CD, a number, then an optional sub-number (.1), a letter or
# two or Greek letters (8a, 62L, 45RA, 8β) and a second number (16/32).
_CD = re.compile(
    r'(?<![A-Za-z0-9])CD\s?\d{1,3}(?:\.\d{1,2})?'
    r'(?:[A-Za-z]{1,2}|[α-ωΑ-Ω]{1,2})?'
    r'(?:/(?:CD)?\d{1,3}[A-Za-z]?)?(?![A-Za-z0-9])', re.I)

# The non-CD markers (marker_catalog): {fold_label(printed): marker} for
# each listed name and synonym, and a pattern for each named family.
_MARKER_FAMILY_PATTERNS = tuple(
    (re.compile(re.escape(folded) + r'(\d{1,2})'
                + (f'([{suffix}]?)' if suffix else '()')), spelled)
    for folded, spelled, suffix in MARKER_FAMILIES)


@lru_cache(maxsize=1)
def _marker_index() -> dict[str, str]:
    """``{fold_label(printed): marker}`` for marker_catalog.NAMED_MARKERS
    and MARKER_SYNONYMS."""
    index = {fold_label(m): m for m in NAMED_MARKERS}
    for printed, marker in MARKER_SYNONYMS.items():
        index.setdefault(fold_label(printed), marker)
    return index


def _named_marker(folded: str) -> str | None:
    """The marker a folded run of words names, else None."""
    hit = _marker_index().get(folded)
    if hit is not None:
        return hit
    for pattern, spelled in _MARKER_FAMILY_PATTERNS:
        m = pattern.fullmatch(folded)
        if m:
            return f'{spelled}{m.group(1)}{m.group(2).upper()}'
    return None


def _cd_spelling(printed: str) -> str:
    """A CD marker as printed, with 'CD' upper case and no spaces."""
    return 'CD' + re.sub(r'\s+', '', printed[2:])


def _find_marker(text: str, words: Sequence[_Word],
                 skip: list[tuple[int, int]]) -> str | None:
    """The earliest marker in `text` outside the `skip` spans. No listed
    name's words begin another's, so one start holds one marker."""
    found: list[tuple[int, str]] = []                 # (start, marker)
    for m in _CD.finditer(text):
        if not _overlaps(m.start(), m.end(), skip):
            found.append((m.start(), _cd_spelling(m.group(0))))
    for i, j, folded in _runs(words):
        marker = _named_marker(folded)
        start, end = words[i].start, words[j - 1].end
        if marker is not None and not _overlaps(start, end, skip):
            found.append((start, marker))
    return min(found)[1] if found else None


# ── Species, clone, catalog number, lot ─────────────────────────────────────

# Target species as printed after 'anti-', and their abbreviations.
SPECIES = ('human', 'mouse', 'rat', 'hamster', 'rabbit', 'goat', 'monkey',
           'rhesus', 'cynomolgus', 'pig', 'dog', 'canine', 'bovine',
           'chicken', 'horse', 'sheep', 'zebrafish')
_SPECIES_ABBREVIATIONS = {'hu': 'human', 'ms': 'mouse', 'mu': 'mouse',
                          'rt': 'rat'}
_SPECIES_WORD = '|'.join(sorted((*SPECIES, *_SPECIES_ABBREVIATIONS),
                                key=len, reverse=True))
_SPECIES = re.compile(
    rf'\banti[\s\-–]*((?:{_SPECIES_WORD})\b(?:\s*/\s*(?:{_SPECIES_WORD})\b)*)',
    re.I)

# An isotype ('Mouse IgG1, κ'): the host's antibody class, not the marker.
# 'anti-<species> Ig<class>' names a target and is left alone.
_ISOTYPE = re.compile(
    rf'\bisotype\b[^\n]*|(?<!anti[\-–\s])\b'
    rf'(?:{_SPECIES_WORD})\s+Ig[A-Z]\d?[a-c]?\b(?:\s*,\s*[κλ])?', re.I)

# '<key> [no. | number | #] [:#.] <value>': the value is one word.
_FIELD_TAIL = r'\b\.?\s*(?:no\b\.?|number\b|#)?\s*[:#.]?\s*'
_CLONE = re.compile(r'\bclone' + _FIELD_TAIL + r'([A-Za-z0-9][\w.\-/]*)',
                    re.I)
# Catalog numbers and lots always carry a digit ('Lot: see vial' is none).
_CATALOG = re.compile(r'\b(?:cat(?:alog(?:ue)?)?|ref)' + _FIELD_TAIL
                      + r'((?=[\w.\-]*\d)[A-Za-z0-9][\w.\-]*)', re.I)
_LOT = re.compile(r'\blot' + _FIELD_TAIL
                  + r'((?=[\w.\-]*\d)[A-Za-z0-9][\w.\-]*)', re.I)


def _field(pattern: re.Pattern[str], text: str
           ) -> tuple[str | None, tuple[int, int] | None]:
    """``(value, span of key and value)`` of the first `pattern` match."""
    m = pattern.search(text)
    if m is None:
        return None, None
    return m.group(1).rstrip('.-/'), m.span()


def _species(text: str) -> tuple[str | None, tuple[int, int] | None]:
    """``(species, span)`` of the first 'anti-<species>' ('human/mouse'
    when it names several), lower case and spelled out."""
    m = _SPECIES.search(text)
    if m is None:
        return None, None
    names = [_SPECIES_ABBREVIATIONS.get(s, s)
             for s in re.split(r'\s*/\s*', m.group(1).casefold())]
    return '/'.join(dict.fromkeys(names)), m.span()


# ── The parsed label ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VialLabel:
    """What one vial label says. ``fluor`` is a key of
    fluor_catalog.FLUOR_SPECTRA, or None when the label names none the
    catalogue lists; ``fluor_text`` is the fluor's words as printed even
    then (None when no fluor word was found). ``raw`` is the text given."""
    marker: str | None
    fluor: str | None
    fluor_text: str | None
    species: str | None
    clone: str | None
    catalog: str | None
    lot: str | None
    raw: str


def parse_vial_label(text: str) -> VialLabel:
    """Read the marker, fluor, target species, clone, catalog number and lot
    from the printed text of an antibody vial label (see the module
    docstring for the shapes it reads). Fields the label does not show are
    None."""
    clean = clean_label_text(text)
    words = _words(clean)
    clone, clone_span = _field(_CLONE, clean)
    catalog, catalog_span = _field(_CATALOG, clean)
    lot, lot_span = _field(_LOT, clean)
    species, species_span = _species(clean)
    fields = [s for s in (clone_span, catalog_span, lot_span) if s]
    fluor, fluor_text, fluor_span = _find_fluor(clean, words, fields)
    skip = fields + [m.span() for m in _ISOTYPE.finditer(clean)]
    skip += [s for s in (species_span, fluor_span) if s]
    return VialLabel(marker=_find_marker(clean, words, skip), fluor=fluor,
                     fluor_text=fluor_text, species=species, clone=clone,
                     catalog=catalog, lot=lot, raw=text)


# ── Detectors ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DetectorPick:
    """The detector that sees a fluor best (``primary``) and the next best
    one with other optics (``runner_up``, offered when two vials want the
    same detector). Both None when no detector sees the fluor."""
    primary: str | None
    runner_up: str | None


def primary_detector(fluor: str, detectors: Sequence[Detector]
                     ) -> DetectorPick:
    """Where catalogued `fluor` is read among `detectors`: the largest
    value of its modelled signature (panel_check.model_signature), the first
    detector of a tie. Detectors with the primary's laser and filter (its
    -A / -H / -W twins) are never the runner-up. Raises KeyError for a
    fluor the catalogue does not list."""
    sig = model_signature(fluor, detectors)
    if not np.any(sig > 0):
        return DetectorPick(None, None)
    order = np.argsort(-sig, kind='stable')
    best = detectors[int(order[0])]
    runner_up = next((detectors[int(k)][0] for k in order[1:]
                      if sig[k] > 0 and detectors[int(k)][1:] != best[1:]),
                     None)
    return DetectorPick(best[0], runner_up)


def _letter_lasers() -> dict[str, float]:
    """``{laser letters: nm}`` from the generic instruments
    (panel_check.DETECTOR_SETS: 'B530/30' is on 488 nm), plus the other
    letterings of the same lasers ('BL1', 'VL1', 'YL1', 'RL1')."""
    lasers: dict[str, float] = {}
    for spec in DETECTOR_SETS.values():
        for name, laser, _c, _w in spec['detectors']:
            m = re.match(r'[A-Za-z]+', name)
            if m:
                lasers.setdefault(m.group(0).upper(), float(laser))
    for other, letters in _LASER_LETTER_SYNONYMS.items():
        if letters in lasers:
            lasers.setdefault(other, lasers[letters])
    return lasers


_LASER_LETTER_SYNONYMS = {'BL': 'B', 'VL': 'V', 'YL': 'YG', 'Y': 'YG',
                          'RL': 'R', 'UVL': 'UV'}

# A filter in a detector name: optional laser letters or wavelength, then
# centre/width ('B530/30-A', 'YG 586/15', '488-530/30', 'Comp-V450/50-A').
# A wavelength counts only standing alone (no letter or digit right before
# it): in 'XX421 450/50-A' the 421 is part of a renamed channel's fluor, not
# a laser.
_NAME_FILTER = re.compile(
    r'(?:(?<![A-Za-z])(?P<letters>[A-Za-z]{1,3})'
    r'|(?<![A-Za-z\d])(?P<nm>\d{3})[\s_-]+)?'
    r'[\s_-]*(?P<centre>\d{3})\s*/\s*(?P<width>\d{1,3})(?!\d)')
_FILTER = re.compile(r'(?<!\d)(\d{3})\s*/\s*(\d{1,3})(?!\d)')
# A wavelength in $PnL: the first number of 300-999 nm.
_WAVELENGTH = re.compile(r'(?<!\d)([3-9]\d{2})(?:\.\d+)?(?!\d)')


@dataclass(frozen=True)
class DetectorNames:
    """``detectors_from_names``'s answer: the ``Detector`` tuples it could
    build, and the names it could not (no laser or no filter found)."""
    detectors: tuple[Detector, ...]
    unmatched: tuple[str, ...]


def fcs_optics(metadata: Mapping[str, object], names: Sequence[str]
               ) -> tuple[dict[str, float], dict[str, str]]:
    """``(lasers, filters)`` the FCS file itself records for its detector
    `names` (the $PnN order): ``{name: nm}`` from $PnL (the first wavelength
    when it lists several) and ``{name: 'centre/width'}`` from $PnF when it
    holds one. Detectors without the keyword are left out. When two
    parameters share one $PnN the first one's keywords are kept: the loader
    keeps that name for the first and renames the second ('FL1-A (2)',
    flow_sample), so the first is the one the name reaches."""
    lasers: dict[str, float] = {}
    filters: dict[str, str] = {}
    for n, name in enumerate(names, 1):
        wl = _WAVELENGTH.search(keyword_value(metadata, f'$P{n}L') or '')
        if wl:
            lasers.setdefault(name, float(wl.group(1)))
        filt = _FILTER.search(keyword_value(metadata, f'$P{n}F') or '')
        if filt:
            filters.setdefault(name, f'{filt.group(1)}/{filt.group(2)}')
    return lasers, filters


# Laser lines a wavelength in a detector name may name: those of the generic
# instruments (panel_check.DETECTOR_SETS), plus lines common cytometers carry
# that those sets leave out (375 and 445 nm diodes, 532 and 594 nm lasers,
# 785 nm on some spectral instruments), so a name such as '532-575/25' still
# reads. A number within _LASER_TOLERANCE_NM of one is a laser; that span
# also takes in the 633 nm HeNe and 637/638 nm diodes sold as 'red'.
_EXTRA_LASER_LINES = (375.0, 445.0, 532.0, 594.0, 785.0)
_LASER_TOLERANCE_NM = 8.0


@lru_cache(maxsize=1)
def _laser_lines() -> tuple[float, ...]:
    """The laser lines (nm) a wavelength in a detector name may name."""
    lines = {float(d[1]) for spec in DETECTOR_SETS.values()
             for d in spec['detectors']}
    return tuple(sorted(lines | set(_EXTRA_LASER_LINES)))


def _is_laser_line(nm: float) -> bool:
    return any(abs(nm - line) <= _LASER_TOLERANCE_NM
               for line in _laser_lines())


def _laser_from_name(name: str) -> float | None:
    """The laser (nm) a detector name spells: a wavelength before its
    filter ('488-530/30') when it is near a known laser line (else the name
    spells none), or laser letters before its filter or channel number
    ('B530/30', 'BL1'); None when it spells none."""
    m = _NAME_FILTER.search(name)
    if m is not None and m.group('nm'):
        nm = float(m.group('nm'))
        return nm if _is_laser_line(nm) else None
    letters = m.group('letters') if m is not None else None
    if letters is None:
        lead = re.match(r'([A-Za-z]{1,3})\d', name.strip())
        letters = lead.group(1) if lead else None
    return _letter_lasers().get(letters.upper()) if letters else None


def _filter_from_name(name: str) -> tuple[float, float] | None:
    """``(centre, width)`` of the filter a detector name spells, else
    None."""
    m = _NAME_FILTER.search(name)
    return (float(m.group('centre')), float(m.group('width'))) if m else None


def detectors_from_names(names: Iterable[str],
                         lasers: Mapping[str, float] | None = None,
                         filters: Mapping[str, str] | None = None
                         ) -> DetectorNames:
    """``Detector`` tuples for the FCS detector `names` that say which laser
    and filter they read. The file's own $PnL (`lasers`, ``{name: nm}``) and
    $PnF (`filters`, ``{name: 'centre/width'}``), from ``fcs_optics``, win
    over what the name spells; otherwise the name's laser letters ('UV',
    'V', 'B' / 'BL', 'YG' / 'Y' / 'YL', 'R' / 'RL') or wavelength and its
    'centre/width' are used. A name with no filter on either side ('FL1-A',
    'BL1-A' without $PnF) or no laser is returned in ``unmatched``, for the
    user to place. Pass fluorescence detectors only: scatter and time
    channels are unmatched too."""
    lasers = lasers or {}
    filters = filters or {}
    found: list[Detector] = []
    unmatched: list[str] = []
    for name in names:
        filt = _FILTER.search(filters.get(name, ''))
        band = ((float(filt.group(1)), float(filt.group(2))) if filt
                else _filter_from_name(name))
        laser = lasers.get(name)
        if laser is None:
            laser = _laser_from_name(name)
        if band is None or laser is None:
            unmatched.append(name)
            continue
        found.append((name, float(laser), band[0], band[1]))
    return DetectorNames(tuple(found), tuple(unmatched))


# ── A panel of vials ─────────────────────────────────────────────────────────

# What each vial needs from the user, VialPlacement.status.
PLACED = 'placed'                   # its detector is labelled
CONFLICT = 'conflict'               # another vial wants the same detector
UNREADABLE = 'unreadable'           # no detector here sees its fluor
NO_MARKER = 'no marker'             # the label names no marker
NO_FLUOR = 'no fluor'               # the label names no fluor
UNKNOWN_FLUOR = 'fluor not cataloged'


@dataclass(frozen=True)
class VialPlacement:
    """One vial of ``assign``: its label, its detector pick (None when its
    fluor is missing or not catalogued) and its status (PLACED, CONFLICT,
    UNREADABLE, NO_MARKER, NO_FLUOR or UNKNOWN_FLUOR)."""
    label: VialLabel
    pick: DetectorPick | None
    status: str


@dataclass(frozen=True)
class Assignment:
    """``assign``'s answer. ``labels`` is ``{detector: marker}`` for the
    vials it could place (each primary detector and its -A / -H / -W
    twins); ``vials`` holds every vial, in order, with its status;
    ``conflicts`` is ``{detector: indexes of the vials that want it}``."""
    labels: dict[str, str]
    vials: tuple[VialPlacement, ...]
    conflicts: dict[str, tuple[int, ...]]


def _first_status(label: VialLabel, pick: DetectorPick | None) -> str:
    """A vial's status before conflicts are counted."""
    if label.fluor is None:
        return UNKNOWN_FLUOR if label.fluor_text else NO_FLUOR
    if pick is None or pick.primary is None:
        return UNREADABLE
    return PLACED if label.marker else NO_MARKER


def _twins(primary: str, detectors: Sequence[Detector]) -> list[str]:
    """`primary` and the detectors with its laser and filter."""
    optics = next(d[1:] for d in detectors if d[0] == primary)
    return [d[0] for d in detectors if d[1:] == optics]


def assign(labels: Sequence[VialLabel], detectors: Sequence[Detector]
           ) -> Assignment:
    """Map parsed vial `labels` onto `detectors`: each vial with a marker and
    a catalogued fluor labels its fluor's primary detector. Two or more such
    vials on one detector are all left out and marked CONFLICT (each pick
    carries its runner-up to offer); the other statuses say why a vial was
    not placed."""
    picks = [primary_detector(v.fluor, detectors) if v.fluor else None
             for v in labels]
    status = [_first_status(v, p) for v, p in zip(labels, picks, strict=True)]
    wanted: dict[str, list[int]] = {}
    for i, (s, p) in enumerate(zip(status, picks, strict=True)):
        if s == PLACED and p is not None and p.primary is not None:
            wanted.setdefault(p.primary, []).append(i)
    conflicts = {d: tuple(ix) for d, ix in wanted.items() if len(ix) > 1}
    out: dict[str, str] = {}
    for det, ix in wanted.items():
        if det in conflicts:
            for i in ix:
                status[i] = CONFLICT
            continue
        marker = labels[ix[0]].marker
        for twin in _twins(det, detectors):
            out[twin] = str(marker)
    vials = tuple(VialPlacement(v, p, s)
                  for v, p, s in zip(labels, picks, status, strict=True))
    return Assignment(out, vials, conflicts)
