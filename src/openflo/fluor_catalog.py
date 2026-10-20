"""Common commercial fluorophore names, for the channel matcher.

A public catalogue of fluorophores sold for flow cytometry, as their vendors
name them. openflo-run's shared channel matching (cli._label_key) needs it to
recognise a fluor word in an antibody label. It holds no lab-specific content:
no panel, marker or sample.
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
