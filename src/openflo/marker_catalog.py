"""Common antibody targets that are not CD numbers, for the vial label reader.

A public, generic list of targets that antibody vendors commonly sell for
flow cytometry, as they print them. The vial label reader (vial_label.py)
needs it to find the marker on a label that names no CD number. It holds no
panel, sample or lab-specific content, and lists far more targets than any
one panel uses. It is the one module that spells out non-CD marker names;
code and tests elsewhere take them from here at run time.

CD markers are read by pattern (vial_label's ``_CD``) and are not listed.
Pure Python, no imports.
"""
from __future__ import annotations

# Targets as printed here; a label matches one by
# shared_clustering.fold_label (case, spacing and punctuation dropped), so a
# hyphen, a space or none is the same name. Greek letters fold to distinct
# Latin ones and so keep names apart. No entry's words begin another entry's
# (tests/test_vial_label.py checks), so one start on a label holds one name.
NAMED_MARKERS = (
    'HLA-DR', 'HLA-DQ', 'HLA-DP', 'HLA-ABC', 'HLA-A2', 'MHC I', 'MHC II',
    'I-A/I-E', 'H-2Kb', 'H-2Kd', 'H-2Db',
    'TCRβ', 'TCRγδ', 'TCRαβ', 'TCR Vα7.2', 'TCR Vδ2', 'TCR Vγ9',
    'FoxP3', 'Ki-67', 'T-bet', 'GATA3', 'RORγt', 'Eomes', 'Helios',
    'Bcl-2', 'Bcl-6', 'IRF4', 'Blimp-1', 'TCF1', 'TOX',
    'PD-1', 'PD-L1', 'PD-L2', 'CTLA-4', 'LAG-3', 'TIM-3', 'TIGIT', 'ICOS',
    'KLRG1', 'NKG2A', 'NKG2C', 'NKG2D', 'NKp30', 'NKp44', 'NKp46', 'NK1.1',
    'IgD', 'IgM', 'IgG', 'IgA', 'IgE', 'Igκ', 'Igλ',
    'B220', 'F4/80', 'Ly-6G', 'Ly-6C', 'Ly-6A/E', 'Gr-1', 'Sca-1', 'c-Kit',
    'TER-119', 'Siglec-F', 'Siglec-H', 'FcεRIα', 'EpCAM', 'MERTK',
    'IFN-γ', 'IFN-α', 'TNF-α', 'IL-1α', 'IL-1β', 'Granzyme A', 'Granzyme B',
    'Perforin', 'GM-CSF', 'TGF-β',
)

# Other printings of a listed target: {printed: NAMED_MARKERS entry}.
MARKER_SYNONYMS = {
    'MHC Class I': 'MHC I', 'MHC Class II': 'MHC II',
    'IFN gamma': 'IFN-γ', 'TNF alpha': 'TNF-α',
    'TCR beta': 'TCRβ', 'TCR gamma/delta': 'TCRγδ',
}

# Families of targets named by a prefix and a number of 1 or 2 digits
# (interleukins, chemokine receptors): (folded prefix, spelled prefix,
# suffix letters allowed after the number). A label word folding to prefix,
# number and suffix is spelled prefix + number + upper-case suffix
# ('il17a' is 'IL-17A').
MARKER_FAMILIES = (
    ('il', 'IL-', 'abcdef'),
    ('ccr', 'CCR', ''),
    ('cxcr', 'CXCR', ''),
    ('cx3cr', 'CX3CR', ''),
    ('xcr', 'XCR', ''),
)
