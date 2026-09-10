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
import re

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


# ── GUI availability: skip politely, but FAIL where a GUI was promised ──────

def gui_unavailable(reason):
    """Tk could not start. Skip on a headless dev box; FAIL in CI.

    Every GUI test guards itself with "skip if Tk is unavailable", which is
    right on a headless machine — the GUI genuinely cannot be tested there and
    a red suite would be noise. But that guard is FAIL-OPEN: if Tk stopped
    initialising everywhere, every GUI test would skip and the suite would go
    green having tested no GUI at all. That is the same "check that cannot
    fail" shape this project keeps finding in its own code, sitting in the test
    harness.

    It is not hypothetical. On windows-latest / py3.11 a test skipped with
    "Can't find a usable init.tcl", and nothing in the run said so — the leg
    was green with that file's coverage silently missing.

    So CI, which provides Xvfb on Linux and a desktop session on Windows, sets
    ``OPENFLO_REQUIRE_GUI=1`` and a Tk failure becomes an error there. Local
    runs are unaffected.
    """
    if os.environ.get('OPENFLO_REQUIRE_GUI') != '1':
        pytest.skip(reason)

    # Before failing, establish whether Tk is BROKEN or merely hiccuped.
    # Creating a Tk root fails transiently when several xdist workers do it at
    # once — observed locally within an hour of turning this guard on, and on
    # windows/py3.11 in CI ("Can't find a usable init.tcl"), while the same
    # test passes alone and under xdist within its own file. Failing on that
    # would trade a silent coverage hole for flaky red CI, which is a bad
    # trade: nobody trusts a suite that cries wolf.
    #
    # So try once more. If Tk comes up, the environment is fine and this run
    # simply lost a race — skip, and say so. If it does not, Tk is genuinely
    # unavailable where it was promised, which is what this guard is for.
    try:
        import tkinter as tk
        probe = tk.Tk()
        probe.destroy()
    except Exception as exc:                                 # noqa: BLE001
        pytest.fail(
            f"Tk was required in this environment but is unavailable: {reason}"
            f"\nA second attempt also failed ({type(exc).__name__}: {exc}), so "
            "this is not a transient race."
            "\nOPENFLO_REQUIRE_GUI=1 is set, so this is an error rather than a "
            "skip — otherwise the GUI tests would pass by not running.")
    pytest.skip(f"{reason} — TRANSIENT: Tk came up on retry, so the "
                f"environment is fine and this run lost a race")


# Matches the ways Tk announces it cannot start. Deliberately narrow: this
# only decides whether to PROBE, and the probe decides the outcome.
_TK_FAILURE = re.compile(
    r"tclerror|tkinter|init\.tcl|no display|\$DISPLAY|"
    r"couldn't connect to display|can't find a usable|"
    r"tk could not start|no usable tk",
    re.I)


def _tk_really_broken():
    """One fresh root. True only if Tk genuinely will not start."""
    try:
        import tkinter as tk
        probe = tk.Tk()
        probe.destroy()
    except Exception:                                        # noqa: BLE001
        return True
    return False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Turn a Tk-caused skip into a failure when a GUI was promised.

    ``gui_unavailable`` does this for tests that call it. Most do not: 7 files
    create a root and skip on their own, and 10 more route only ImportError
    through the guard. Measured under a deliberately broken Tk, 55 GUI tests
    skipped silently while the suite still reported success for them.

    Doing it here catches every skip however it was written, so a new test
    cannot reintroduce the hole by forgetting to use the helper.
    """
    outcome = yield
    report = outcome.get_result()

    # 'setup' matters as much as 'call': a skip raised inside a fixture
    # (which is where several of these live) reports at setup, and
    # handling only 'call' would leave exactly those bypassed.
    if report.when not in ('setup', 'call') or not report.skipped:
        return
    if os.environ.get('OPENFLO_REQUIRE_GUI') != '1':
        return

    # For a skip, longrepr is (path, lineno, "Skipped: <reason>").
    reason = ''
    lr = getattr(report, 'longrepr', None)
    if isinstance(lr, tuple) and len(lr) == 3:
        reason = str(lr[2])
    elif lr is not None:
        reason = str(lr)
    if not _TK_FAILURE.search(reason):
        return

    # Only convert if Tk is genuinely down. Roots fail transiently when xdist
    # workers race, and a suite that reds on a race stops being trusted.
    if not _tk_really_broken():
        return

    report.outcome = 'failed'
    report.longrepr = (
        f"{reason}\n\n"
        "This skip was converted to a failure: OPENFLO_REQUIRE_GUI=1 promises "
        "a working GUI in this environment, and a second Tk root also failed, "
        "so this is not a transient xdist race.\n"
        "Skipping here would let the GUI tests pass by not running — the "
        "failure mode tests/conftest.py:gui_unavailable exists to prevent, "
        "which only protected tests that remembered to call it.")
