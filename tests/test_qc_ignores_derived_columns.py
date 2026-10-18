"""Acquisition QC must not read our own analysis output as a detector.

`AcquisitionQC` selects channels as "every numeric column minus EXCLUDE_CLUSTER".
That denylist had drifted out of step with the columns the app writes back onto
`sample.data`: three of the five embeddings were listed and TSNE/PHATE were
not, `leiden` and `pseudotime` were missing entirely, and the FMO step's
`<channel>_pos` booleans were never considered.

The margin detector flags events piled up at a channel's ceiling. A cluster
label IS piled up at its ceiling — every event in the highest-numbered cluster
sits at the column's maximum — and numpy treats a boolean as 0/1, so every
marker-positive event does too. Measured on a clean 50,000-event sample:

    a 'leiden' column      ->  12.4% of events deleted (all of cluster 7)
    a 'CD3-A_pos' column   ->  49.8% deleted (every positive event)

This is reachable: an auto-clean gate re-runs `AcquisitionQC` over the live
DataFrame (`autoclean_keep_mask`), which after clustering or FMO positivity
carries exactly these columns. The load-time `run_qc` happens to be safe only
because it runs before any analysis.

The first test below is the durable one: it RUNS the analyses and asserts that
every column they add is ignored, so the denylist cannot silently fall behind
the code again.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from openflo.pipeline import EXCLUDE_CLUSTER, AcquisitionQC, FlowSample

N = 6_000


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'Time': np.linspace(0.0, 300.0, N),
        'FSC-A': rng.normal(60_000.0, 8_000.0, N),
        'CD3-A': np.concatenate([rng.normal(300.0, 60.0, N // 2),
                                 rng.normal(3_000.0, 400.0, N // 2)]),
        'CD19-A': rng.normal(800.0, 150.0, N)})
    s = FlowSample.from_dataframe(df, name='derived')
    s.fluor_channels = ['CD3-A', 'CD19-A']
    s.scatter_channels = ['FSC-A']
    return s


def _qc_channels(df):
    """The channels QC would judge, by the same rule `run()` uses when no
    explicit channel list is passed."""
    import pandas.api.types as pdt
    qc = AcquisitionQC(df.reset_index(drop=True))
    return [c for c in qc.data.columns
            if c != qc.time_channel
            and not any(k in c for k in EXCLUDE_CLUSTER)
            and not pdt.is_bool_dtype(qc.data[c])
            and pdt.is_numeric_dtype(qc.data[c])]


def test_columns_the_app_writes_are_never_judged_as_detectors(sample):
    """The coupling test. Run the analyses, see what columns appear, and
    require QC to ignore every one of them — so adding a new analysis output
    without listing it fails here rather than deleting a user's events."""
    detectors = set(sample.data.columns)

    _write_every_analysis_column(sample)

    added = set(sample.data.columns) - detectors
    assert 'leiden' in added, 'clustering no longer writes a leiden column'

    judged = set(_qc_channels(sample.data))
    leaked = sorted(added & judged)
    assert not leaked, (
        'acquisition QC would treat these analysis outputs as measurement '
        f'channels, and can delete every event at their maximum: {leaked}')


def _write_every_analysis_column(sample):
    """Put every column the app writes back onto `sample.data`.

    This used to run run_leiden and hand-write the embeddings, which left
    `cluster`, `flowsom`, `flowsom_meta` and `cell_cycle` never produced — so
    those denylist entries were unprotected. Measured by deleting each from
    EXCLUDE_CLUSTER: removing 'leiden' or 'UMAP1' turned this file red;
    removing 'cluster', 'flowsom' or 'cell_cycle' left it green, meaning QC
    could have started judging PhenoGraph's own output as a detector — the
    exact defect that once deleted 12.4% of a clean sample — with nothing to
    say so.

    The integer-valued labels matter: QC deletes events at a channel's
    ceiling, so a label column is dangerous precisely because its maximum is
    a real, populated value.
    """
    n = len(sample.data)
    rng = np.random.default_rng(1)

    sample.run_leiden(resolution=0.5)

    # Clustering backends other than Leiden. Written directly rather than run:
    # cluster() shells out to PhenoGraph (spawns a pool, seconds per call) and
    # run_flowsom trains a SOM, neither of which this test needs — what it
    # needs is the COLUMN, shaped as the app writes it.
    sample.data['cluster'] = np.arange(n) % 7
    sample.data['flowsom'] = np.arange(n) % 25
    sample.data['flowsom_meta'] = np.arange(n) % 10
    sample.data['cell_cycle'] = np.arange(n) % 3

    sample.data['pseudotime'] = np.linspace(0.0, 1.0, n)
    sample.data['CD3-A_pos'] = sample.data['CD3-A'] > 1_000.0
    # The workspace's per-event source tags, as prepare_unit writes them
    # (and every run's _events.csv carries them back into the editor).
    from openflo.workspace import _source_tags
    sample.data['__group__'] = _source_tags(['(ungrouped)'], [n])
    sample.data['__sample__'] = _source_tags(['derived'], [n])
    for prefix in ('UMAP', 'TSNE', 'TRIMAP', 'PACMAP', 'PHATE'):
        sample.data[f'{prefix}1'] = rng.normal(0.0, 5.0, n)
        sample.data[f'{prefix}2'] = rng.normal(0.0, 5.0, n)


# EXCLUDE_CLUSTER entries that are INPUT channels rather than analysis
# outputs, so no amount of running the pipeline will produce them.
_INPUT_ONLY = {'Time', 'time'}

# Two entries are REDUNDANT rather than unprotected, and it is worth writing
# down so the next person who deletes entries one at a time to see what breaks
# does not chase a phantom. `_qc_channels` matches by SUBSTRING, so:
#
#   'pseudotime'   is already caught by 'time'
#   'flowsom_meta' is already caught by 'flowsom'
#
# Removing either from EXCLUDE_CLUSTER changes no behaviour, so no test can
# fail. They are kept because naming every written column documents the
# coupling, and because the cover disappears the moment 'flowsom' is renamed.


def test_every_denylist_entry_is_actually_exercised(sample):
    """The ratchet. An entry nothing produces is an entry no test protects.

    Without this, adding a name to EXCLUDE_CLUSTER looks like protection while
    being untested — which is how `cluster`, `flowsom` and `cell_cycle` came
    to sit in the list with nothing checking them.
    """
    before = set(sample.data.columns)
    _write_every_analysis_column(sample)
    added = set(sample.data.columns) - before

    unexercised = sorted(
        entry for entry in EXCLUDE_CLUSTER
        if entry not in _INPUT_ONLY
        and not any(entry in col for col in added))

    assert not unexercised, (
        f'{unexercised} are on the QC denylist but no column produced here '
        f'matches them, so deleting them from EXCLUDE_CLUSTER would not fail '
        f'a single test. Add them to _write_every_analysis_column, or to '
        f'_INPUT_ONLY if they are input channels.')


@pytest.mark.parametrize('column, values', [
    pytest.param('leiden', np.arange(N) % 8, id='cluster-label'),
    pytest.param('CD3-A_pos', np.arange(N) % 2 == 0, id='fmo-positive-flag'),
    pytest.param('pseudotime', np.linspace(0.0, 1.0, N), id='pseudotime'),
    pytest.param('TSNE1', np.linspace(-10.0, 10.0, N), id='tsne'),
    pytest.param('PHATE2', np.linspace(-1.0, 1.0, N), id='phate'),
])
def test_a_derived_column_does_not_change_the_qc_verdict(sample, column,
                                                         values):
    """Whatever QC decides about a sample, adding one of our own analysis
    columns to it must not change that decision."""
    df = sample.data
    logging.disable(logging.CRITICAL)
    try:
        before = len(AcquisitionQC(df.copy()).run(n_bins=200, threshold=5))
        marked = df.copy()
        marked[column] = values
        after = len(AcquisitionQC(marked).run(n_bins=200, threshold=5))
    finally:
        logging.disable(logging.NOTSET)
    assert after == before, (
        f'adding a {column!r} column changed the QC verdict from {before} to '
        f'{after} kept — QC judged our own output as a detector')


def test_real_channels_are_still_judged(sample):
    """The guard must not exclude the detectors QC exists to check."""
    judged = set(_qc_channels(sample.data))
    assert {'FSC-A', 'CD3-A', 'CD19-A'} <= judged, (
        f'real measurement channels were excluded from QC: {judged}')


def test_every_embedding_prefix_is_listed():
    """`_store_embedding` writes `<PREFIX>1` / `<PREFIX>2`. All five prefixes
    the app uses must be excluded — two of them were not."""
    for prefix in ('UMAP', 'TSNE', 'TRIMAP', 'PACMAP', 'PHATE'):
        for axis in ('1', '2'):
            name = f'{prefix}{axis}'
            assert any(k in name for k in EXCLUDE_CLUSTER), (
                f'{name} is written by _store_embedding but is not excluded')
