"""Tests for openflo.annotate — MEM scoring + reference-table annotation.
Uses the synthetic PBMC generator so the expected biology is known."""
from __future__ import annotations

import numpy as np
import pandas as pd

from openflo.annotate import (
    annotate_by_reference,
    mem_label,
    mem_scores,
    parse_signature_table,
    population_states,
    scale_markers,
)
from openflo.synthetic import (
    PBMC_MARKER_DET,
    PBMC_MARKERS,
    immunophenotyping_sample,
)


def _pbmc_clustered(n=8000, seed=0):
    """A PBMC sample with a ground-truth population label per cell."""
    df = immunophenotyping_sample(n=n, seed=seed)
    dets = [PBMC_MARKER_DET[m] for m in PBMC_MARKERS]
    # Derive a coarse truth label from the dominant markers (live cells only).
    live = df['LiveDead-A'].to_numpy() < 1000
    cd3 = df['BV510-A'].to_numpy()
    cd4 = df['FITC-A'].to_numpy()
    cd8 = df['APC-A'].to_numpy()
    cd19 = df['PE-A'].to_numpy()
    cd56 = df['BV605-A'].to_numpy()
    cd14 = df['APC-Fire-A'].to_numpy()
    label = np.full(len(df), 'other', dtype=object)
    label[live & (cd3 > 2000) & (cd4 > 2000)] = 'CD4T'
    label[live & (cd3 > 2000) & (cd8 > 2000)] = 'CD8T'
    label[live & (cd3 < 1000) & (cd19 > 2000)] = 'B'
    label[live & (cd3 < 1000) & (cd56 > 2000)] = 'NK'
    label[live & (cd3 < 1000) & (cd14 > 2000)] = 'Mono'
    return df, label, dets


# ── scaling ───────────────────────────────────────────────────────────────────

def test_scale_markers_range():
    X = np.array([[0.0, 100.0], [50.0, 200.0], [100.0, 300.0]])
    S = scale_markers(X, lo_pct=0, hi_pct=100, out_max=10)
    assert S.min() >= 0 and S.max() <= 10
    assert np.allclose(S[:, 0], [0, 5, 10])


# ── MEM ───────────────────────────────────────────────────────────────────────

def test_mem_enriches_lineage_markers():
    df, label, dets = _pbmc_clustered()
    mem = mem_scores(df, label, dets)
    assert set(['CD4T', 'B', 'NK']).issubset(set(mem.index))
    cd3, cd4 = PBMC_MARKER_DET['CD3'], PBMC_MARKER_DET['CD4']
    cd19, cd56 = PBMC_MARKER_DET['CD19'], PBMC_MARKER_DET['CD56']
    # CD4 T cells are CD3+ CD4+ (strongly positive MEM).
    assert mem.loc['CD4T', cd3] > 3 and mem.loc['CD4T', cd4] > 3
    # B cells are CD19+ and CD3- (negative).
    assert mem.loc['B', cd19] > 3
    assert mem.loc['B', cd3] < 0
    # NK cells are CD56+.
    assert mem.loc['NK', cd56] > 3


def test_mem_scores_bounded_to_ten():
    df, label, dets = _pbmc_clustered()
    mem = mem_scores(df, label, dets)
    assert mem.to_numpy().max() <= 10 and mem.to_numpy().min() >= -10


def test_mem_label_formats_sorted_signed():
    row = pd.Series({'CD3': 8.0, 'CD4': 6.0, 'CD8': -4.0, 'CD19': -1.0})
    lbl = mem_label(row, threshold=2.0)
    # CD19 (|1| < 2) dropped; sorted by magnitude; signed integers.
    assert lbl == 'CD3+8 CD4+6 CD8-4'


def test_mem_label_empty_when_nothing_enriched():
    row = pd.Series({'CD3': 1.0, 'CD4': -1.0})
    assert mem_label(row, threshold=2.0) == ''


# ── reference-table parsing + annotation ──────────────────────────────────────

def test_parse_signature_table_forms():
    text = """
    # comment
    CD4 T: CD3+ CD4+ CD8-
    B cell: CD3-, CD19+
    NK = CD3lo CD56hi
    """
    table = parse_signature_table(text)
    assert table['CD4 T'] == {'CD3': 1, 'CD4': 1, 'CD8': -1}
    assert table['B cell'] == {'CD3': -1, 'CD19': 1}
    assert table['NK'] == {'CD3': -1, 'CD56': 1}


def test_population_states_and_annotation_round_trip():
    df, label, dets = _pbmc_clustered()
    mem = mem_scores(df, label, dets)
    # Re-key the MEM columns from detector → marker for a readable table.
    det_to_marker = {d: m for m, d in PBMC_MARKER_DET.items()}
    mem = mem.rename(columns=det_to_marker)
    states = population_states(mem, threshold=3.0)
    table = parse_signature_table(
        "CD4 T: CD3+ CD4+ CD8-\n"
        "CD8 T: CD3+ CD8+ CD4-\n"
        "B cell: CD3- CD19+\n"
        "NK cell: CD3- CD56+\n"
        "Monocyte: CD14+")
    ann = annotate_by_reference(states, table)
    # The ground-truth clusters get the right names.
    assert ann['CD4T']['name'] == 'CD4 T'
    assert ann['B']['name'] == 'B cell'
    assert ann['NK']['name'] == 'NK cell'
    assert ann['Mono']['name'] == 'Monocyte'


def test_annotate_unknown_when_no_match():
    states = {'c0': {'CD3': 1, 'CD4': 1}}
    table = {'B cell': {'CD19': 1, 'CD3': -1}}     # opposite of c0's CD3
    ann = annotate_by_reference(states, table)
    assert ann['c0']['name'] == 'unknown'
