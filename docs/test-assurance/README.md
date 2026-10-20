# Test assurance: every test proven to pass and to fail

Every test in the suite should do two things: pass on the real code, and fail
when the behaviour it guards is broken. A test that cannot fail checks nothing.
This folder holds the evidence that each test does both, test by test.

## What is here

- `ledger.csv`: one row per collected test. The columns are the data it runs on, whether it passes on
  the real code, whether it fails when the code is broken, and the exact break
  that made it fail (file, line, change, original text).
- `aimed.json`: the breaks chosen by hand for tests the automatic search
  cannot reach (see below). Each one is applied, the tests are run, and the
  break is undone.

## How it was proven

`scripts/prove_tests.py` does this mechanically:

1. **Data.** Every test runs on generated data: `openflo.synthetic`, an FCS
   file or workspace written by the test itself, or values built in the test.
   No test reads a real sample any more; the four that used to (and so never
   ran) now write their own sample or workspace with a known answer.
2. **Positive half.** Every test runs once on an unmodified copy of the
   checkout and must pass.
3. **Negative half.** The code each test guards is then broken one change at
   a time (`<` becomes `<=`, a constant moves by one, `and` becomes `or`, a
   condition is negated, a return gives `None`, a statement, raise or import
   is deleted, a line of a config file is removed...). A test that passed in
   step 2 and fails on a break is proven, and that break is recorded as its
   evidence.
4. **Hand-aimed breaks.** Some tests guard things the automatic search does
   not change: a default value, a list of names, an exception's base class,
   the suite's own checks (the GUI guard in `conftest.py`, a fixture writer,
   a deliberately wrong reference rule a negative test must catch). For
   these, `aimed.json` names the break, and the same pass-then-fail check is
   run.

A failure from an assertion is the proof we want: the test checked something
and caught the break. A failure only by an error (an exception escaping the
code) is recorded separately. For a test whose claim is "this module imports"
or "this raises the right error", the error is the check.

## Results

Measured on master as of #69, 3,420 collected tests:

| | tests |
|---|---:|
| Pass on the real code and fail on a break, caught by an assertion | 3,230 |
| Pass on the real code and fail on a break, by an error | 176 |
| Skipped on this machine (GPU, Windows-only, one optional extra) | 13 |
| Cannot be failed by changing OpenFlo (a numba assumption) | 1 |

Of the 3,406 proven, 92 needed a hand-aimed break from `aimed.json`; the
rest were found by the automatic search. Every test that skips here because
an optional package (anndata, trimap, pacmap, phate) is missing was proven in
a second environment with those packages installed, except the one that also
needs h5py.

What the work changed besides adding the evidence:

- Four tests that read a real sample or FlowJo workspace from the
  developer's disk (and so never ran) now generate their own:
  `test_autoclean_counts.py`, `test_apply_region_gates.py`,
  `test_wsp_extract.py` and `test_wsp_writer.py`. The `real_fcs_dir` and
  `real_wsp_path` fixtures are gone.
- Tests that could never fail were strengthened: the batch-correction score
  test only exercised numpy (both callers now share
  `CytoNorm.qc_means`, and the test checks it both ways); the popup-menu
  colour test passed on Linux whatever the code did; the workspace queue
  test was satisfied by a similarly named method; the DPI test on non-Windows
  passed even if the platform check was removed.
- Negative halves were added where a test only checked the good case, e.g.
  polygon gating (a rule that double-counts a shared edge, depends on
  winding, or forgets the closing edge must be caught), gating helpers
  (missing-channel messages), ellipse gates (singular, indefinite, zero
  radius) and the theme switch.
- Twelve dialogs the tooltip test used to skip are now opened on generated
  data; doing so found controls with no hover help in the auto-gate,
  embedding and axis-settings windows, which now have it.

## What cannot be proven on this machine

- **GPU tests** (`tests/test_gpu_accel.py`) skip without a GPU: they compare
  the CuPy and PyTorch paths with the CPU path, and need a CUDA, ROCm, Intel
  or Apple GPU to run at all.
- **Windows-only tests**: `test_a_really_open_compare_file_is_reported_not_a_crash`
  (only Windows blocks a rename of an open file) and
  `test_copy_figure_reaches_the_windows_clipboard`.
- **Optional extras** (anndata, h5py, trimap, pacmap, phate): these tests skip
  in the main environment. They were proven in a second environment with the
  extras installed, on the code of 2026-10-14.
- **Known-defect pins**: a test marked strict xfail documents a confirmed
  defect that is not fixed yet. It fails on the real code on purpose and
  turns red the day the defect is fixed, so both halves hold by
  construction. There are none at present: the four there were are fixed.
- **`test_numba_thread_masking_is_thread_local`** checks an assumption about
  the numba library (its thread mask is per thread), not OpenFlo code. No
  change to OpenFlo can make it fail; it fails only if numba changes.

## Re-running it

```
# per-test coverage (GUI tests need a display)
xvfb-run -a python -m pytest -n auto --cov=openflo --cov-context=test \
    --cov-report= -p no:randomly
# the sweep
xvfb-run -a python scripts/prove_tests.py --coverage .coverage \
    --include-uncovered --budget 40 --out sweep
# the hand-aimed breaks
xvfb-run -a python scripts/prove_tests.py aimed docs/test-assurance/aimed.json aimed
# after the code moves on: re-run each test's recorded break, and list
# the tests whose break is gone (sweep those with --tests-from)
xvfb-run -a python scripts/prove_tests.py recheck sweep/proofs.json recheck
# the ledger, from every run
python scripts/prove_tests.py report docs/test-assurance sweep/proofs.json aimed/proofs.json
```

A full sweep takes several hours on 4 cores; it checkpoints and resumes from
`mutants.jsonl` if stopped.
