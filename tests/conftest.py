"""Shared pytest fixtures.

`synthetic_fcs` writes a tiny in-memory FCS via FlowIO so the suite can
exercise FlowSample / compensation / gating without checking real patient
data into the repo.

Tests that need a real FlowJo `.wsp` or the full clinical dataset opt in
via the `real_wsp_path` and `real_fcs_dir` fixtures, which auto-skip when
those paths aren't present.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

# ── Synthetic data ────────────────────────────────────────────────────────────

@pytest.fixture(scope='session')
def synthetic_channels():
    return ['FSC-A', 'SSC-A', 'BV421-A', 'APC-A', 'PE-Cy7-A']


@pytest.fixture(scope='session')
def synthetic_fcs(tmp_path_factory, synthetic_channels):
    """Create a tiny on-disk FCS with 5 channels and 1000 events.

    Channels:
      FSC-A, SSC-A : log-normal scatter (positive scalar)
      BV421-A      : bimodal (negative + positive populations)
      APC-A        : negative-skewed
      PE-Cy7-A     : positive-skewed

    Returns the path as a string.
    """
    import flowio
    rng = np.random.default_rng(seed=0)
    n_events = 1000

    fsc = rng.lognormal(mean=10, sigma=0.3, size=n_events)
    ssc = rng.lognormal(mean=9, sigma=0.4, size=n_events)
    bv421 = np.concatenate([
        rng.normal(loc=100, scale=20, size=n_events // 2),
        rng.normal(loc=5000, scale=500, size=n_events - n_events // 2),
    ])
    apc = rng.exponential(scale=200, size=n_events) + 50
    pecy7 = rng.exponential(scale=500, size=n_events) + 100

    events = np.column_stack([fsc, ssc, bv421, apc, pecy7]).astype(np.float32)
    flat = events.flatten().tolist()

    out = tmp_path_factory.mktemp('fcs') / 'synthetic.fcs'
    with open(out, 'wb') as f:
        flowio.create_fcs(
            f, flat, synthetic_channels,
            opt_channel_names=['', '', 'CD11b', 'CD34', 'CD45'])
    return str(out)


# ── Optional real-data fixtures ───────────────────────────────────────────────

@pytest.fixture(scope='session')
def real_wsp_path():
    """Skip the test unless a real FlowJo workspace is provided via the
    OPENFLO_TEST_WSP environment variable (points at a local .wsp file)."""
    path = os.environ.get('OPENFLO_TEST_WSP')
    if path and os.path.isfile(path):
        return path
    pytest.skip("real .wsp not available — set OPENFLO_TEST_WSP env var")


@pytest.fixture(scope='session')
def real_fcs_dir():
    """Skip the test unless a real FCS dataset directory is provided via the
    OPENFLO_TEST_FCS_DIR environment variable."""
    path = os.environ.get('OPENFLO_TEST_FCS_DIR')
    if path and os.path.isdir(path):
        return path
    pytest.skip(
        "real FCS dataset not available — set OPENFLO_TEST_FCS_DIR env var")
