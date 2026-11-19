"""Vial label reader (openflo.vial_label): marker, fluor, species, clone,
catalog number and lot from a vial's printed text, and the detector that
reads the fluor.

Every label is generated here: the fluor names come from
fluor_catalog at run time (FLUOR_SPECTRA keys, FLUOR_LONG_FORMS,
FLUOR_ALIASES), the markers are generic CD9xx numbers, invented family
members (an interleukin numbered 99) or names drawn at run time from
marker_catalog, and clones, catalog numbers and lots are invented. The textbook
detector checks pick each fluor by its catalogued peak wavelengths and name
the expected laser and filter centre as numbers.
"""
from __future__ import annotations

import hashlib
import re

import pytest

from openflo import batch_normalize, fcs_text
from openflo import vial_label as vl
from openflo.fluor_catalog import (
    FLUOR_ALIASES,
    FLUOR_FAMILIES,
    FLUOR_LONG_FORMS,
    FLUOR_SPECTRA,
)
from openflo.marker_catalog import (
    MARKER_FAMILIES,
    MARKER_SYNONYMS,
    NAMED_MARKERS,
)
from openflo.panel_check import DETECTOR_SETS, model_signature
from openflo.shared_clustering import fold_label

FLUORS = list(FLUOR_SPECTRA)
FOUR_LASER = DETECTOR_SETS['4-laser conventional (405/488/561/640 nm)'][
    'detectors']
SPECTRAL = DETECTOR_SETS[
    '5-laser spectral, 64 detectors (355/405/488/561/640 nm)']['detectors']

# Label shapes as vendors print them: (template, the species it names).
SHAPES = (
    ('{fluor} anti-human {marker} Antibody', 'human'),
    ('{marker} {fluor}', None),
    ('Anti-Mouse {marker} {fluor}', 'mouse'),
    ('{fluor} Mouse Anti-Human {marker}', 'human'),
    ('{fluor}  anti-rat   {marker}\nClone: XY-12\nCat. No. 900-123\n'
     'Lot: B900123', 'rat'),
)


def _label(shape, fluor, marker='CD901b'):
    return shape.format(fluor=fluor, marker=marker)


# ── fluor ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('fluor', FLUORS)
@pytest.mark.parametrize('shape, species', SHAPES)
def test_every_catalogue_fluor_is_read_in_every_label_shape(fluor, shape,
                                                            species):
    got = vl.parse_vial_label(_label(shape, fluor))
    assert (got.fluor, got.fluor_text, got.marker, got.species) == (
        fluor, fluor, 'CD901b', species)


def _short_named_in(key):
    """The catalogue keys some other key starts with, joined by '-'
    (donor and tandem)."""
    return [d for d in FLUORS if key != d and key.startswith(d + '-')]


TANDEMS = [k for k in FLUORS if _short_named_in(k)]


def test_the_catalogue_has_tandems_to_test():
    assert len(TANDEMS) >= 10, TANDEMS


@pytest.mark.parametrize('tandem', TANDEMS)
@pytest.mark.parametrize('sep', ['-', '/', ' ', ' / '])
def test_a_tandem_is_never_read_as_its_donor(tandem, sep):
    donor = _short_named_in(tandem)[0]
    printed = donor + sep + tandem[len(donor) + 1:]
    for shape, _sp in SHAPES[:4]:
        got = vl.parse_vial_label(_label(shape, printed))
        assert got.fluor == tandem, (printed, got)


def _long_form_variants(key):
    """`key` with one vendor long form swapped in or out (the first pair
    of FLUOR_LONG_FORMS that applies), with the pair used."""
    out = []
    for long_form, short_form in FLUOR_LONG_FORMS:
        if long_form.strip() in key:
            out.append((key.replace(long_form, short_form, 1)
                        .replace(long_form.strip(), short_form, 1),
                        (long_form, short_form)))
            break
        pat = r'(?<![A-Za-z])' + re.escape(short_form) + r'(?=\d)'
        if re.search(pat, key):
            out.append((re.sub(pat, long_form, key, count=1),
                        (long_form, short_form)))
            break
    return out


LONG_FORM_CASES = [(k, printed, pair) for k in FLUORS
                   for printed, pair in _long_form_variants(k)]


def test_every_long_form_pair_is_exercised():
    """Each pair is used by some catalogue key above, or is a shorter
    spelling of an earlier pair's long form (a long form with a word left
    out): that one is tested by swapping it in for the longer form."""
    used = {pair for _k, _p, pair in LONG_FORM_CASES}
    assert len(LONG_FORM_CASES) >= 25
    shorter = [(p, q) for p in FLUOR_LONG_FORMS if p not in used
               for q in FLUOR_LONG_FORMS[:FLUOR_LONG_FORMS.index(p)]
               if p[0].strip() in q[0] and p[1] == q[1]]
    assert {p for p, _q in shorter} == set(FLUOR_LONG_FORMS) - used
    for (long_form, _s), (longer, _s2) in shorter:
        keys = [k for k in FLUORS if longer in k]
        assert keys, longer
        for k in keys:
            printed = k.replace(longer, long_form)
            got = vl.parse_vial_label(f'{printed} anti-human CD902')
            assert (got.fluor, got.fluor_text) == (k, printed)


@pytest.mark.parametrize('key, printed, pair', LONG_FORM_CASES)
def test_vendor_long_and_short_forms_are_one_fluor(key, printed, pair):
    for shape, _sp in SHAPES[:4]:
        got = vl.parse_vial_label(_label(shape, printed))
        assert got.fluor == key, (printed, pair, got)
        assert got.fluor_text == printed


def test_the_cyanine_long_form_with_a_slash_is_the_cy_tandem():
    cases = [k for k in FLUORS if re.search(r'-Cy\d', k)]
    assert len(cases) >= 4
    for k in cases:
        printed = re.sub(r'-Cy(?=\d)', '/Cyanine', k)
        assert vl.parse_vial_label(f'{printed} anti-human CD902').fluor == k


def test_the_785_polymer_is_also_read_as_786():
    keys = [k for k in FLUORS if re.fullmatch(r'[A-Z]+785', k)]
    assert keys
    for k in keys:
        printed = k.replace('785', '786')
        assert vl.parse_vial_label(f'{printed} anti-human CD902').fluor == k


@pytest.mark.parametrize('alias', sorted(FLUOR_ALIASES))
def test_aliases_name_their_catalogue_fluor(alias):
    for shape, _sp in SHAPES[:4]:
        got = vl.parse_vial_label(_label(shape, alias))
        assert (got.fluor, got.fluor_text) == (FLUOR_ALIASES[alias], alias)


# A pin on FLUOR_ALIASES, independent of the table: the test above is
# parametrised over the table itself, so renaming an alias would rename its
# own test case. Update the digest when the table changes on purpose.
ALIAS_COUNT = 2
ALIAS_DIGEST = 'b698f62c0478a79317900e3981a74bb353a0b96fa9b956d99f1218e48728b039'


def _alias_digest(aliases):
    pairs = sorted(f'{fold_label(a)}={fold_label(k)}'
                   for a, k in aliases.items())
    return hashlib.sha256('\n'.join(pairs).encode()).hexdigest()


def test_the_alias_table_is_pinned():
    assert len(FLUOR_ALIASES) == ALIAS_COUNT
    assert _alias_digest(FLUOR_ALIASES) == ALIAS_DIGEST
    renamed = dict(FLUOR_ALIASES)
    first = sorted(renamed)[0]
    renamed[first[:-1] + chr(ord(first[-1]) + 1)] = renamed.pop(first)
    assert _alias_digest(renamed) != ALIAS_DIGEST


@pytest.mark.parametrize('fluor', FLUORS)
def test_signs_case_and_spacing_noise_do_not_hide_the_fluor(fluor):
    words = fluor.split(' ')
    marked = words[0] + '®' + (' ' + ' '.join(words[1:]) if words[1:]
                               else '') + '™'
    for printed in (marked, fluor.upper(), fluor.lower(),
                    fluor.replace(' ', '(R) ', 1) + '(TM)',
                    fluor.replace('-', '/'), fluor.replace(' ', '')):
        got = vl.parse_vial_label(f'{printed}   ANTI-HUMAN   cd902 antibody')
        assert (got.fluor, got.marker, got.species) == (fluor, 'CD902',
                                                        'human'), printed


def test_every_catalogue_name_folds_to_its_own_entry():
    index = {vl.canonical_fluor_fold(fold_label(k)): k for k in FLUORS}
    assert len(index) == len(FLUORS)
    for alias, key in FLUOR_ALIASES.items():
        assert vl.canonical_fluor_fold(fold_label(alias)) not in index
        assert key in FLUOR_SPECTRA


@pytest.mark.parametrize('family', FLUOR_FAMILIES)
def test_an_uncatalogued_family_fluor_is_named_but_not_guessed(family):
    printed = family.upper() + '999'
    got = vl.parse_vial_label(f'{printed} anti-human CD903 Antibody')
    assert (got.fluor, got.fluor_text, got.marker) == (None, printed,
                                                       'CD903')


DONORS = sorted({d for k in TANDEMS for d in _short_named_in(k)})


JOINS = ('/', '-', '–', ' ', ' / ', ' - ')


def _uncatalogued_acceptors(donor):
    """Acceptors printed after `donor` that make no catalogued tandem: an
    invented name and number, each family with an invented number, and each
    catalogued fluor that is not one of `donor`'s tandems."""
    out = ['Zzfire 640', 'Zz Fire 640']
    out += [f.upper() + '999' for f in FLUOR_FAMILIES]
    out += [k for k in FLUORS if k not in DONORS
            and not k.startswith(donor + '-')]
    return out


@pytest.mark.parametrize('sep', JOINS)
@pytest.mark.parametrize('donor', DONORS)
def test_an_uncatalogued_tandem_is_not_read_as_its_donor(donor, sep):
    """Neither the donor alone nor the acceptor alone: either would place
    the vial on a wrong detector."""
    for acceptor in _uncatalogued_acceptors(donor):
        printed = f'{donor}{sep}{acceptor}'
        for shape, _sp in SHAPES[:4]:
            got = vl.parse_vial_label(_label(shape, printed, 'CD903'))
            assert (got.fluor, got.fluor_text, got.marker) == (
                None, printed, 'CD903'), (shape, printed, got)


def test_two_catalogued_fluors_side_by_side_are_one_uncatalogued_tandem():
    a, b = FLUORS[0], FLUORS[-1]
    for sep in JOINS:
        for printed in (f'{a}{sep}{b}', f'{b}{sep}{a}'):
            got = vl.parse_vial_label(f'{printed} anti-human CD903')
            assert (got.fluor, got.fluor_text) == (None, printed), got


@pytest.mark.parametrize('donor', DONORS)
def test_a_donor_followed_by_label_words_is_still_the_donor(donor):
    named = [m for m in NAMED_MARKERS if ' ' not in m][:8]
    for after in (' anti-human CD903', ' Mouse Anti-Human CD903',
                  ' Anti-Mouse CD903', ' Rat anti-mouse CD903',
                  '-conjugated anti-human CD903', '-Mouse Anti-Human CD903',
                  '/Rat anti-mouse CD903',
                  ' Conjugated Anti-Human CD903', '-anti-human CD903',
                  ' CD903', '-CD903', '/CD903', ' CD 903',
                  ' anti-human CD903 Antibody, 100 tests',
                  ' anti-human CD903\nClone: AB-12\nLot 9001',
                  *(f' {m}' for m in named), *(f'-{m}' for m in named)):
        got = vl.parse_vial_label(f'{donor}{after}')
        assert (got.fluor, got.fluor_text) == (donor, donor), (after, got)
        assert got.marker is not None, after
    got = vl.parse_vial_label(f'CD903-{donor}')
    assert (got.fluor, got.fluor_text, got.marker) == (donor, donor, 'CD903')


def _hyphenated_markers():
    """Markers whose name holds a hyphen: from marker_catalog, and each
    family's invented member 99 when it spells one."""
    out = [(m, m) for m in NAMED_MARKERS if '-' in m]
    for _folded, spelled, suffix in MARKER_FAMILIES:
        name = f'{spelled}99{suffix[:1].upper()}'
        if '-' in name:
            out.append((name, name))
    return out


def test_the_catalogue_has_hyphenated_markers_to_test():
    assert len(_hyphenated_markers()) >= 10


@pytest.mark.parametrize('printed, marker', _hyphenated_markers())
def test_a_fluor_hyphenated_to_a_hyphenated_marker_reads_both(printed,
                                                               marker):
    for fl in FLUORS:
        for label in (f'{fl}-{printed}', f'{fl}-{printed} Antibody'):
            got = vl.parse_vial_label(label)
            assert (got.fluor, got.fluor_text, got.marker) == (
                fl, fl, marker), (label, got)


def test_a_fluor_name_inside_a_lot_or_catalog_number_is_not_the_fluor():
    words = [k for k in FLUORS if re.fullmatch(r'[A-Za-z]+\d+', k)]
    assert words
    for k in words:
        got = vl.parse_vial_label(f'anti-human CD902\nLot: {k}\nCat #: {k}')
        assert (got.fluor, got.fluor_text, got.lot, got.catalog) == (
            None, None, k, k)
        # A fluor printed right after the field is not joined to it.
        other = next(f for f in FLUORS if f != k)
        got = vl.parse_vial_label(f'Lot: {k} {other} anti-human CD902')
        assert (got.fluor, got.fluor_text, got.lot) == (other, other, k)


def test_a_label_without_a_fluor_has_none():
    got = vl.parse_vial_label('Purified anti-mouse CD904 Antibody')
    assert (got.fluor, got.fluor_text, got.marker) == (None, None, 'CD904')


# ── marker ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('printed, marker', [
    ('CD901b', 'CD901b'), ('cd902', 'CD902'), ('CD 905', 'CD905'),
    ('CD903.1', 'CD903.1'), ('CD916/32', 'CD916/32'),
    ('CD916/CD932', 'CD916/CD932'), ('CD962L', 'CD962L'),
    ('CD945RA', 'CD945RA'), ('CD908α', 'CD908α'), ('CD908β', 'CD908β'),
])
def test_cd_markers_are_read_by_pattern(printed, marker):
    fl = FLUORS[0]
    for shape, _sp in SHAPES:
        got = vl.parse_vial_label(_label(shape, fl, printed))
        assert got.marker == marker, (shape, got)


def test_greek_letters_keep_markers_apart():
    """Greek letters must survive: two markers that differ only by a
    Greek letter (such as 'CD908α' and 'CD908β') are different markers on
    different detectors."""
    fl = FLUORS[0]
    greek = [m for m in NAMED_MARKERS if re.search('[α-ωΑ-Ω]', m)]
    assert len(greek) >= 8
    seen = {}
    for printed in (*greek, 'CD908α', 'CD908β'):
        seen[printed] = vl.parse_vial_label(
            f'{fl} anti-mouse {printed} Antibody').marker
    assert seen == {k: k for k in seen}
    # Names differing only in a Greek letter stay apart.
    stems = {}
    for m in greek:
        stems.setdefault(re.sub('[α-ωΑ-Ω]', '', m), []).append(m)
    assert any(len(v) > 1 for v in stems.values())
    pair = next(m for m in greek if re.search('[α-ω]{2}$', m))
    spaced = f'{pair[:-2]} {pair[-2]}/{pair[-1]}'
    assert vl.parse_vial_label(f'{spaced} {fl}').marker == pair


@pytest.mark.parametrize('marker', NAMED_MARKERS)
def test_every_named_marker_is_read_as_printed_and_in_other_spellings(
        marker):
    fl = FLUORS[-1]
    for printed in (marker, marker.upper(), marker.replace('-', ''),
                    marker.replace('-', ' ')):
        for shape, _sp in SHAPES:
            got = vl.parse_vial_label(_label(shape, fl, printed))
            assert (got.marker, got.fluor) == (marker, fl), (printed, shape)


def _family_cases():
    """(printed, marker) for each marker family's invented members 9 and
    99, as printed with and without the hyphen and in lower case, and with
    the first suffix letter it allows."""
    out = []
    for folded, spelled, suffix in MARKER_FAMILIES:
        for number in ('9', '99'):
            tail = suffix[:1]
            marker = f'{spelled}{number}{tail.upper()}'
            out += [(marker, marker), (marker.replace('-', ''), marker),
                    (f'{folded}{number}{tail}', marker)]
    return out


@pytest.mark.parametrize('printed, marker',
                         sorted(MARKER_SYNONYMS.items()) + _family_cases())
def test_marker_synonyms_and_families(printed, marker):
    got = vl.parse_vial_label(f'{FLUORS[0]} anti-mouse {printed}')
    assert got.marker == marker


def test_a_marker_family_spells_its_hyphen_and_upper_case_suffix():
    il = next(f for f in MARKER_FAMILIES if f[1].endswith('-') and f[2])
    got = vl.parse_vial_label(f'{FLUORS[0]} anti-mouse {il[0]}17{il[2][0]}')
    assert got.marker == f'{il[1]}17{il[2][0].upper()}'
    assert got.marker.count('-') == 1


# Antibody classes from marker_catalog (an isotype names one of them).
IG_CLASSES = [m for m in NAMED_MARKERS if re.fullmatch('Ig[A-Z]', m)]


def test_an_isotype_is_not_the_marker():
    """The isotype line comes first on some labels; the earliest marker
    would be the host's antibody class."""
    assert len(IG_CLASSES) >= 3
    fl = FLUORS[0]
    for ig in IG_CLASSES:
        got = vl.parse_vial_label(f'Mouse {ig}, κ\n{fl} Mouse Anti-Human '
                                  'CD906')
        assert got.marker == 'CD906'
        got = vl.parse_vial_label(f'Isotype: {ig}\n{fl} anti-human CD906')
        assert got.marker == 'CD906'
        assert vl.parse_vial_label(f'{fl} anti-mouse {ig}').marker == ig
        assert vl.parse_vial_label(f'{fl} Rat Anti-Mouse {ig}').marker == ig


@pytest.mark.parametrize('text, marker', [
    ('{fl} anti-human CD902\nCross-reacts with CD903', 'CD902'),
    ('CD902 {fl}\nAlso binds {ig}', 'CD902'),
    ('{ig} {fl}\nNot CD902', '{ig}'),
])
def test_the_first_marker_printed_is_the_vials(text, marker):
    ig = IG_CLASSES[0]
    got = vl.parse_vial_label(text.format(fl=FLUORS[0], ig=ig))
    assert got.marker == marker.format(ig=ig)


def test_no_named_marker_begins_with_another_ones_words():
    """The parser takes the earliest marker and has no rule for two names
    starting at one word: the list must never need one."""
    split = [[fold_label(w) for w in re.split(r'[\s/\-]+', m)]
             for m in NAMED_MARKERS]
    clashes = [(a, b) for a, wa in zip(NAMED_MARKERS, split, strict=True)
               for b, wb in zip(NAMED_MARKERS, split, strict=True)
               if a != b and wb[:len(wa)] == wa]
    assert clashes == []
    fl = FLUORS[0]
    for printed, marker in MARKER_SYNONYMS.items():
        assert vl.parse_vial_label(f'{printed} {fl}').marker == marker


# A pin on marker_catalog, independent of the tables: the marker tests are
# parametrised over the tables, so an edited entry (a Greek letter dropped,
# a synonym misspelled) would edit its own test case. Update the counts and
# digest when the catalogue changes on purpose.
MARKER_COUNTS = (77, 6, 5)
MARKER_DIGEST = (
    '85ed29934a835610f8e38ab355c5075c6fd59a5e9666a0271b4f2b9c5782840b')


def test_the_marker_catalogue_is_pinned():
    assert (len(NAMED_MARKERS), len(MARKER_SYNONYMS),
            len(MARKER_FAMILIES)) == MARKER_COUNTS
    text = '\n'.join([
        *NAMED_MARKERS,
        *(f'{a}={b}' for a, b in sorted(MARKER_SYNONYMS.items())),
        *('|'.join(f) for f in MARKER_FAMILIES)])
    assert hashlib.sha256(text.encode()).hexdigest() == MARKER_DIGEST


def test_vial_label_spells_no_marker_name():
    """The non-CD names live in marker_catalog; the reader takes them from
    there."""
    assert vl.NAMED_MARKERS is NAMED_MARKERS
    source = open(vl.__file__, encoding='utf-8').read()
    for m in (*NAMED_MARKERS, *MARKER_SYNONYMS):
        assert f"'{m}'" not in source, m


def test_the_marker_is_not_taken_from_the_clone_catalog_or_lot():
    fl = FLUORS[0]
    got = vl.parse_vial_label(
        f'Clone: CD907\nCat #: CD907-12\nLot CD907\n{fl} anti-human CD908')
    assert (got.marker, got.clone, got.catalog, got.lot) == (
        'CD908', 'CD907', 'CD907-12', 'CD907')


# ── species, clone, catalog number, lot ─────────────────────────────────────

@pytest.mark.parametrize('text, species', [
    ('anti-human CD902', 'human'), ('Anti-Mouse CD902', 'mouse'),
    ('ANTI-RAT CD902', 'rat'), ('anti human CD902', 'human'),
    ('anti-human/mouse CD902', 'human/mouse'),
    ('Mouse Anti-Human CD902', 'human'), ('Rat anti-Mouse CD902', 'mouse'),
    ('anti-Hu CD902', 'human'), ('CD902 antibody', None),
])
def test_species_is_the_target_after_anti(text, species):
    assert vl.parse_vial_label(f'{FLUORS[0]} {text}').species == species


@pytest.mark.parametrize('lines, want', [
    ('Clone: XY-12\nCat #: 900123\nLot: B900123',
     ('XY-12', '900123', 'B900123')),
    ('clone ab12  Cat. No. 900-123  LOT 9001234',
     ('ab12', '900-123', '9001234')),
    ('Catalog: 90-0123-42\nLot No. Z9012\nClone 9C9.1',
     ('9C9.1', '90-0123-42', 'Z9012')),
    ('REF 900123\nLot# 77A12', (None, '900123', '77A12')),
    ('Catalogue number: 900123.', (None, '900123', None)),
    ('Lot: see vial\nCat: none', (None, None, None)),
])
def test_clone_catalog_and_lot_are_read(lines, want):
    got = vl.parse_vial_label(f'{FLUORS[0]} anti-human CD902\n{lines}')
    assert (got.clone, got.catalog, got.lot) == want
    assert (got.marker, got.fluor) == ('CD902', FLUORS[0])
    assert got.raw.endswith(lines)


# ── primary detector ─────────────────────────────────────────────────────────

def _detector(name, detectors):
    return next(d for d in detectors if d[0] == name)


def _main_excitation(entry):
    return max(entry[0], key=lambda p: p[1])[0]


# (excitation peak range nm, emission peak range nm, laser nm, filter
# centre nm): where a textbook reads a fluor on the 4-laser set.
TEXTBOOK = [
    ((488, 500), (515, 525), 488.0, 530.0),     # blue-excited green
    ((560, 570), (570, 585), 561.0, 586.0),     # yellow-green-excited orange
    ((400, 410), (415, 455), 405.0, 450.0),     # violet-excited blue
    ((645, 655), (655, 670), 640.0, 670.0),     # red-excited far red
    ((560, 570), (775, 790), 561.0, 780.0),     # its far-red tandem
    ((645, 655), (775, 790), 640.0, 780.0),     # red-excited tandem
    ((480, 486), (690, 700), 488.0, 695.0),     # blue-excited far red
    ((400, 410), (600, 610), 405.0, 610.0),     # violet-excited orange
]


@pytest.mark.parametrize('ex, em, laser, centre', TEXTBOOK)
def test_textbook_fluors_land_on_their_textbook_filter(ex, em, laser,
                                                       centre):
    picked = [k for k, e in FLUOR_SPECTRA.items()
              if ex[0] <= _main_excitation(e) <= ex[1]
              and em[0] <= e[2] <= em[1]]
    assert picked, (ex, em)
    for fluor in picked:
        pick = vl.primary_detector(fluor, FOUR_LASER)
        assert pick.primary is not None, fluor
        d = _detector(pick.primary, FOUR_LASER)
        assert (d[1], d[2]) == (laser, centre), (fluor, d)


def test_uv_only_fluors_are_unreadable_without_a_uv_laser():
    uv = [k for k, e in FLUOR_SPECTRA.items()
          if all(pk < 360 for pk, _h in e[0])]
    assert len(uv) >= 5
    for fluor in uv:
        assert vl.primary_detector(fluor, FOUR_LASER) == vl.DetectorPick(
            None, None)
        pick = vl.primary_detector(fluor, SPECTRAL)
        assert _detector(pick.primary, SPECTRAL)[1] == 355.0


@pytest.mark.parametrize('detectors', [FOUR_LASER, SPECTRAL])
def test_primary_and_runner_up_are_the_two_largest_signals(detectors):
    for fluor in FLUORS:
        sig = model_signature(fluor, detectors)
        pick = vl.primary_detector(fluor, detectors)
        if not sig.any():
            assert pick == vl.DetectorPick(None, None)
            continue
        ranked = sorted(range(len(detectors)), key=lambda k: (-sig[k], k))
        assert pick.primary == detectors[ranked[0]][0]
        second = ranked[1]
        assert pick.runner_up == (detectors[second][0] if sig[second] > 0
                                  else None)


def test_the_runner_up_is_never_a_twin_of_the_primary():
    green = next(k for k, e in FLUOR_SPECTRA.items() if 515 <= e[2] <= 525)
    dets = [('X530-A', 488.0, 530.0, 30.0), ('X530-H', 488.0, 530.0, 30.0),
            ('X525-A', 405.0, 525.0, 50.0)]
    assert vl.primary_detector(green, dets) == vl.DetectorPick('X530-A',
                                                               'X525-A')


def test_an_uncatalogued_fluor_has_no_model():
    with pytest.raises(KeyError):
        vl.primary_detector('Zz999', FOUR_LASER)


# ── detector names ───────────────────────────────────────────────────────────

def test_detector_names_that_spell_a_filter_become_detectors():
    names = ['B530/30-A', 'V450/50-A', 'YG586/15-A', 'R670/30-A',
             'UV395/25-A', 'BL695/40-H', 'Y610/20-A', 'RL780/60-A',
             'YL710/50-A', '488-530/30', 'Comp-V450/50-A', 'YG 586/15']
    got = vl.detectors_from_names(names)
    assert got.unmatched == ()
    assert got.detectors == (
        ('B530/30-A', 488.0, 530.0, 30.0), ('V450/50-A', 405.0, 450.0, 50.0),
        ('YG586/15-A', 561.0, 586.0, 15.0), ('R670/30-A', 640.0, 670.0, 30.0),
        ('UV395/25-A', 355.0, 395.0, 25.0), ('BL695/40-H', 488.0, 695.0, 40.0),
        ('Y610/20-A', 561.0, 610.0, 20.0), ('RL780/60-A', 640.0, 780.0, 60.0),
        ('YL710/50-A', 561.0, 710.0, 50.0), ('488-530/30', 488.0, 530.0, 30.0),
        ('Comp-V450/50-A', 405.0, 450.0, 50.0),
        ('YG 586/15', 561.0, 586.0, 15.0))


def test_a_number_glued_to_letters_before_the_filter_is_not_a_laser():
    """A renamed channel '<letters><number> <filter>' spells no laser: the
    number belongs to the name, so the detector is the user's to place."""
    names = ['XX421 450/50-A', 'XX510 525/50-A', 'XX605 610/20-A',
             'Q9488 530/30-A', 'XX488-530/30-A', 'B530/30-A']
    got = vl.detectors_from_names(names)
    assert got.unmatched == tuple(names[:-1])
    assert got.detectors == (('B530/30-A', 488.0, 530.0, 30.0),)


def test_a_wavelength_before_the_filter_must_be_near_a_laser_line():
    lines = sorted({d[1] for spec in DETECTOR_SETS.values()
                    for d in spec['detectors']})
    near = [f'{int(nm)}-530/30' for nm in lines]
    near += [f'{int(lines[0]) + 3}_530/30', '633 660/20', '532-575/25',
             '375-450/50', '594-620/15']
    got = vl.detectors_from_names(near)
    assert got.unmatched == ()
    assert [d[1] for d in got.detectors] == [
        *lines, lines[0] + 3, 633.0, 532.0, 375.0, 594.0]
    far = ['999-530/30', '700-530/30', '300-530/30', '520-530/30',
           '470 530/30']
    got = vl.detectors_from_names(far)
    assert (got.detectors, got.unmatched) == ((), tuple(far))


def test_names_without_a_filter_or_laser_are_unmatched():
    names = ['FL1-A', 'BL1-A', 'FSC-A', 'Time', 'Q530/30-A', '530/30-A']
    got = vl.detectors_from_names(names)
    assert (got.detectors, got.unmatched) == ((), tuple(names))


def test_a_keyword_is_found_with_or_without_its_dollar_in_any_case():
    meta = {'p3f': ' 530/30 ', '$P4L': '488', 'P5S': '', '$p6n': None}
    kv = fcs_text.keyword_value
    assert kv(meta, '$P3F') == '530/30'
    assert kv(meta, 'p4l') == '488'
    assert kv(meta, '$P5S') is None       # blank
    assert kv(meta, '$P6N') is None       # no value
    assert kv(meta, '$P7N') is None
    assert kv(None, '$P3F') is None


def test_one_keyword_reader_is_shared():
    """vial_label and batch_normalize read keywords by one rule (fcs_text),
    and batch_normalize still offers it by its public name."""
    assert vl.keyword_value is fcs_text.keyword_value
    assert batch_normalize.keyword_value is fcs_text.keyword_value
    assert 'keyword_value' in batch_normalize.__all__
    assert not hasattr(vl, 'fcs_keyword')


def test_a_repeated_detector_name_keeps_the_first_parameters_optics():
    """The loader keeps a repeated $PnN for the first parameter and renames
    the second, so the name's optics are the first one's."""
    names = ['FL1-A', 'FL2-A', 'FL1-A']
    meta = {'$P1L': '488', '$P1F': '530/30', '$P2L': '561',
            '$P3L': '640', '$P3F': '670/30'}
    lasers, filters = vl.fcs_optics(meta, names)
    assert lasers == {'FL1-A': 488.0, 'FL2-A': 561.0}
    assert filters == {'FL1-A': '530/30'}


def test_the_files_own_laser_and_filter_win():
    names = ['BL1-A', 'FL2-A', 'B530/30-A', '530/30-A', 'FL9-A']
    meta = {'$P1F': '530/30 BP', 'p2l': '561,640', '$P2F': '586/15',
            '$P3L': '405', '$P4L': ' 488 ', '$P5L': '488'}
    lasers, filters = vl.fcs_optics(meta, names)
    assert lasers == {'FL2-A': 561.0, 'B530/30-A': 405.0, '530/30-A': 488.0,
                      'FL9-A': 488.0}
    assert filters == {'BL1-A': '530/30', 'FL2-A': '586/15'}
    got = vl.detectors_from_names(names, lasers, filters)
    assert got.detectors == (
        ('BL1-A', 488.0, 530.0, 30.0), ('FL2-A', 561.0, 586.0, 15.0),
        ('B530/30-A', 405.0, 530.0, 30.0), ('530/30-A', 488.0, 530.0, 30.0))
    assert got.unmatched == ('FL9-A',)


# ── a panel ──────────────────────────────────────────────────────────────────

def _fluor_on(laser, centre, detectors=FOUR_LASER, skip=()):
    """A catalogued fluor whose primary detector is (laser, centre)."""
    for k in FLUORS:
        p = vl.primary_detector(k, detectors).primary
        if (p is not None and k not in skip
                and _detector(p, detectors)[1:3] == (laser, centre)):
            return k
    raise AssertionError((laser, centre))


def test_a_panel_labels_each_vials_primary_detector():
    a, b = _fluor_on(488.0, 530.0), _fluor_on(640.0, 670.0)
    vials = [vl.parse_vial_label(f'{a} anti-human CD901b'),
             vl.parse_vial_label(f'CD902 {b}')]
    got = vl.assign(vials, FOUR_LASER)
    assert got.labels == {'B530/30': 'CD901b', 'R670/30': 'CD902'}
    assert [v.status for v in got.vials] == [vl.PLACED, vl.PLACED]
    assert got.conflicts == {}


def test_two_vials_on_one_detector_are_a_conflict_with_a_runner_up():
    a = _fluor_on(488.0, 530.0)
    b = _fluor_on(488.0, 530.0, skip=(a,))
    c = _fluor_on(640.0, 670.0)
    vials = [vl.parse_vial_label(f'{a} anti-human CD901b'),
             vl.parse_vial_label(f'{c} anti-human CD903'),
             vl.parse_vial_label(f'{b} anti-human CD902')]
    got = vl.assign(vials, FOUR_LASER)
    assert got.conflicts == {'B530/30': (0, 2)}
    assert got.labels == {'R670/30': 'CD903'}
    assert [v.status for v in got.vials] == [vl.CONFLICT, vl.PLACED,
                                             vl.CONFLICT]
    for i in (0, 2):
        pick = got.vials[i].pick
        assert pick is not None and pick.primary == 'B530/30'
        assert pick.runner_up not in (None, 'B530/30')


def test_vials_that_cannot_be_placed_say_why():
    uv = next(k for k, e in FLUOR_SPECTRA.items()
              if all(pk < 360 for pk, _h in e[0]))
    ok = _fluor_on(561.0, 586.0)
    family = FLUOR_FAMILIES[0].upper() + '999'
    vials = [vl.parse_vial_label(f'{uv} anti-human CD901b'),
             vl.parse_vial_label(f'{ok} Antibody'),
             vl.parse_vial_label('anti-human CD902'),
             vl.parse_vial_label(f'{family} anti-human CD903'),
             vl.parse_vial_label(f'{ok} anti-human CD904')]
    got = vl.assign(vials, FOUR_LASER)
    assert [v.status for v in got.vials] == [
        vl.UNREADABLE, vl.NO_MARKER, vl.NO_FLUOR, vl.UNKNOWN_FLUOR,
        vl.PLACED]
    assert got.labels == {'YG586/15': 'CD904'}
    assert got.conflicts == {}
    assert got.vials[2].pick is None and got.vials[3].pick is None


def test_a_label_reaches_every_twin_of_its_detector():
    a = _fluor_on(488.0, 530.0)
    names = ['B530/30-A', 'B530/30-H', 'B530/30-W', 'R670/30-A']
    dets = vl.detectors_from_names(names).detectors
    got = vl.assign([vl.parse_vial_label(f'{a} anti-human CD901b')], dets)
    assert got.labels == {'B530/30-A': 'CD901b', 'B530/30-H': 'CD901b',
                          'B530/30-W': 'CD901b'}
