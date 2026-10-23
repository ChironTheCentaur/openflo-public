"""Common commercial fluorophore names, for the channel matcher and the
panel check.

A public catalogue of fluorophores sold for flow cytometry, as their vendors
name them. openflo-run's shared channel matching (cli._label_key) needs it to
recognise a fluor word in an antibody label, and Tools > Panel check
(panel_check.py) models each one's spectrum from its peak wavelengths
(FLUOR_SPECTRA). It holds no lab-specific content: no panel, marker or
sample. It is the one module that spells out commercial fluor names.
"""
from __future__ import annotations

# Folded as cli._fold_label folds a label (case, spacing and punctuation
# dropped), so 'Alexa Fluor 488' is 'alexafluor488'. A word from this list
# standing apart at either end of a label is dropped from it ('CD3 <fluor>'
# is CD3 on any detector), with each panel's own detector fluors added
# (cli._fluor_tokens). No marker spelling is one of them
# (test_a_marker_word_is_never_taken_for_a_fluor checks 31 real markers
# against every word here). Viability / DNA dyes are left out on purpose:
# they are often the marker itself.
COMMON_FLUORS = frozenset((
    'fitc', 'pe', 'apc', 'percp', 'percpcy55', 'percpef710', 'percpefluor710',
    'pecy5', 'pecy55', 'pecy7', 'pecf594', 'pedazzle594', 'petexasred',
    'apccy7', 'apch7', 'apcr700', 'apcfire750', 'apcef780', 'apcefluor780',
    'af488', 'af532', 'af594', 'af647', 'af700', 'alexafluor488',
    'alexafluor532', 'alexafluor594', 'alexafluor647', 'alexafluor700',
    'bv421', 'bv480', 'bv510', 'bv570', 'bv605', 'bv650', 'bv711', 'bv750',
    'bv785', 'bv786', 'buv395', 'buv496', 'buv563', 'buv615', 'buv661',
    'buv737', 'buv805', 'bb515', 'bb700', 'pacificblue', 'pacificorange',
    'pb', 'po', 'ef450', 'efluor450', 'ef506', 'efluor506', 'ef660',
    'efluor660', 'sb436', 'sb600', 'sb645', 'sb702', 'sb780', 'v450', 'v500',
    'qdot605', 'qd605'))


# Approximate spectral shapes, for Tools > Panel check (panel_check.py): the
# predicted similarity of a panel before its reagents are bought. Each entry
# is a handful of public peak numbers, not a copied spectrum table:
#
#   (excitation peaks ((nm, relative height), ...), excitation FWHM nm,
#    emission peak nm, emission FWHM nm,
#    extra emission ((peak nm, height relative to the main peak, FWHM), ...))
#
# The extra emission is a tandem's donor leak or a second band. Peaks are
# rounded to the nanometre and the widths to 5 nm; the model built from them
# (panel_check.model_signature) is a rough guide, and measured single-stain
# controls always beat it.
FLUOR_SPECTRA = {
    # UV-laser polymer dyes and their tandems.
    'BUV395': (((348, 1.0),), 50, 395, 30, ()),
    'BUV496': (((348, 1.0),), 50, 496, 30, ((395, 0.05, 30),)),
    'BUV563': (((348, 1.0),), 50, 563, 30, ((395, 0.05, 30),)),
    'BUV615': (((348, 1.0),), 50, 615, 35, ((395, 0.05, 30),)),
    'BUV661': (((348, 1.0),), 50, 661, 30, ((395, 0.05, 30),)),
    'BUV737': (((348, 1.0),), 50, 737, 40, ((395, 0.05, 30),)),
    'BUV805': (((348, 1.0),), 50, 805, 45, ((395, 0.05, 30),)),
    # Violet-laser dyes.
    'BV421': (((407, 1.0),), 60, 421, 30, ()),
    'Super Bright 436': (((414, 1.0),), 60, 436, 35, ()),
    'Pacific Blue': (((401, 1.0),), 40, 452, 50, ()),
    'eFluor 450': (((405, 1.0),), 40, 450, 45, ()),
    'V450': (((404, 1.0),), 40, 448, 45, ()),
    'BV480': (((436, 1.0),), 60, 478, 45, ()),
    'V500': (((415, 1.0),), 40, 500, 60, ()),
    'eFluor 506': (((405, 1.0),), 40, 506, 50, ()),
    'BV510': (((405, 1.0),), 60, 510, 50, ((421, 0.05, 30),)),
    'Pacific Orange': (((400, 1.0),), 40, 551, 80, ()),
    'BV570': (((405, 1.0),), 60, 570, 35, ((421, 0.05, 30),)),
    'Super Bright 600': (((414, 1.0),), 60, 600, 35, ((436, 0.05, 35),)),
    'BV605': (((405, 1.0),), 60, 603, 30, ((421, 0.05, 30),)),
    'Qdot 605': (((350, 1.0), (405, 0.7)), 120, 605, 25, ()),
    'Super Bright 645': (((414, 1.0),), 60, 645, 35, ((436, 0.05, 35),)),
    'BV650': (((405, 1.0),), 60, 645, 30, ((421, 0.05, 30),)),
    'Super Bright 702': (((414, 1.0),), 60, 702, 40, ((436, 0.05, 35),)),
    'BV711': (((405, 1.0),), 60, 711, 40, ((421, 0.05, 30),)),
    'BV750': (((405, 1.0),), 60, 750, 45, ((421, 0.05, 30),)),
    'Super Bright 780': (((414, 1.0),), 60, 780, 45, ((436, 0.05, 35),)),
    'BV785': (((405, 1.0),), 60, 785, 45, ((421, 0.05, 30),)),
    # Blue-laser dyes.
    'FITC': (((495, 1.0),), 40, 519, 40, ()),
    'Alexa Fluor 488': (((495, 1.0),), 40, 519, 40, ()),
    'BB515': (((490, 1.0),), 40, 515, 35, ()),
    'PerCP': (((482, 1.0),), 50, 678, 30, ()),
    'PerCP-Cy5.5': (((482, 1.0),), 50, 695, 45, ((678, 0.1, 30),)),
    'PerCP-eFluor 710': (((482, 1.0),), 50, 710, 40, ((678, 0.1, 30),)),
    'BB700': (((485, 1.0),), 50, 693, 40, ()),
    # PE and its tandems (blue and yellow-green lasers).
    'Alexa Fluor 532': (((532, 1.0),), 35, 554, 35, ()),
    'PE': (((496, 0.75), (565, 1.0)), 40, 578, 35, ()),
    'PE-CF594': (((496, 0.75), (565, 1.0)), 40, 612, 40, ((578, 0.05, 35),)),
    'PE-Dazzle 594': (((496, 0.75), (565, 1.0)), 40, 610, 40,
                      ((578, 0.05, 35),)),
    'PE-Texas Red': (((496, 0.75), (565, 1.0)), 40, 615, 40,
                     ((578, 0.05, 35),)),
    'Alexa Fluor 594': (((590, 1.0),), 40, 617, 40, ()),
    'PE-Cy5': (((496, 0.75), (565, 1.0)), 40, 667, 35, ((578, 0.05, 35),)),
    'PE-Cy5.5': (((496, 0.75), (565, 1.0)), 40, 695, 40, ((578, 0.05, 35),)),
    'PE-Cy7': (((496, 0.75), (565, 1.0)), 40, 780, 50, ((578, 0.05, 35),)),
    # Red-laser dyes.
    'APC': (((594, 0.5), (650, 1.0)), 40, 660, 35, ()),
    'Alexa Fluor 647': (((650, 1.0),), 40, 668, 35, ()),
    'eFluor 660': (((633, 1.0),), 40, 659, 35, ()),
    'APC-R700': (((594, 0.5), (650, 1.0)), 40, 704, 40, ((660, 0.05, 35),)),
    'Alexa Fluor 700': (((696, 1.0),), 70, 719, 35, ()),
    'APC-Cy7': (((594, 0.5), (650, 1.0)), 40, 785, 50, ((660, 0.05, 35),)),
    'APC-H7': (((594, 0.5), (650, 1.0)), 40, 785, 50, ((660, 0.05, 35),)),
    'APC-Fire 750': (((594, 0.5), (650, 1.0)), 40, 787, 45,
                     ((660, 0.05, 35),)),
    'APC-eFluor 780': (((594, 0.5), (650, 1.0)), 40, 780, 50,
                       ((660, 0.05, 35),)),
}
