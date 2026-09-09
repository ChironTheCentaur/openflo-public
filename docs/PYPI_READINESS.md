# PyPI readiness checklist

Publishing is **deliberately deferred** (decided 2026-09-08 at v2.4.2): "let's
make sure it's fully functional first." This tracks what would have to be true
before `pip install openflo` is a path we support, so it can be chipped away at
alongside normal work rather than done in a rush at release time.

Everything below was **verified by doing it**, not read off the config —
`python -m build`, then installing the wheel into a fresh venv built from stock
Python 3.12 and running the entry points. Where a box is ticked, the evidence is
in the line.

---

## Already true

- [x] **The name `openflo` is available on PyPI.** Checked against the JSON API
      (404). For contrast, `flowkit` is taken — the neighbouring package in this
      space.
- [x] **The wheel installs into a clean environment and works.** A fresh venv +
      `pip install openflo-2.4.2-py3-none-any.whl` resolved all 35 packages and
      `openflo-doctor` reported **HEALTHY — 8/8 golden checks** in that
      environment. This is the check that matters most, and it passes today.
- [x] **Metadata is complete.** description, `readme` (with content-type in the
      built METADATA), `requires-python = ">=3.11"`, authors, keywords,
      classifiers, and `[project.urls]` — the URLs point at the **public
      mirror**, never the private repo.
- [x] **Entry points install and run.** Six console scripts plus one
      `gui_scripts` entry, all present in `Scripts/` after install;
      `openflo-doctor` executes end to end.
- [x] **Package data ships.** `_golden.json` and the five
      `template_library/*.json` files are in the wheel — so the behavioural
      self-test works from a pip install, which is what makes `openflo-doctor`
      meaningful for someone who never cloned the repo.
- [x] **The sdist carries the test suite** (`MANIFEST.in`), so an sdist consumer
      can reproduce the golden baseline.
- [x] **The build is already validated on every tag.** `release.yml` runs
      `python -m build` + `twine check dist/*` on every `v*` push; green on
      v2.4.0, v2.4.1 and v2.4.2. Only the upload step is gated, behind
      `PYPI_PUBLISH`.
- [x] **`openflo-doctor` exists.** For a pip audience this matters more than for
      a repo audience: it is how a user answers "did this install correctly"
      without a checkout.
- [x] **Optional dependencies are factored sensibly** — `gui`, `embed`,
      `interop`, `gpu`, `gpu-torch`, `gpu-dml`, `dev`. Heavy and
      platform-specific things are already opt-in rather than core.

## Blocking — would cause real problems for installers

- [x] **~~Unpin the core dependencies.~~** DONE (2026-09-08). All **8 shared**
      deps are now ranges; the 6 remaining exact pins are exactly the leaf deps
      (FlowIO, FlowUtils, PhenoGraph, umap-learn, igraph, leidenalg), where a
      pin costs nothing in coexistence because almost nothing else depends on
      them. Every floor was **tested at its edge**, and only two of the eight
      came from this repo's own code — the rest were set by a dependency's
      requirement or by the numpy-2 ABI. End-to-end check: a brand-new venv
      installing the resulting wheel reports `pip check` clean and
      `openflo-doctor` HEALTHY.
      *Original finding:* All 14 were exact (`==`): numpy, pandas,
      scipy, scikit-learn, matplotlib and the rest. Exact pins are right for a
      reproducible application checkout and wrong for a published package —
      they make coexistence a coin flip. Measured: `pip install numpy==2.3.0`
      into the openflo venv reports "Would install numpy-2.3.0", i.e. pip
      downgrades numpy out from under the declared constraint and openflo's
      stated requirement is simply violated. It happens not to bite scanpy
      today, which is luck rather than design.
      *Shape of the fix:* compatible-release ranges for the scientific core
      (`numpy>=2.0,<3`, etc.), keeping exact pins only where a version is known
      to change results — the golden baseline and response controls are what
      make that safe to attempt, since a behaviour change fails loudly.
      *Do this incrementally, one dependency at a time, with the full suite +
      `openflo-selftest --all` after each.*

- [x] **~~Install-test in CI.~~** DONE (2026-09-08). New `install` job in
      `.github/workflows/ci.yml`: 4 legs — {ubuntu, windows} x {floor deps on
      py3.11, newest resolvable on py3.12}. Each builds the wheel, installs it
      into a clean environment **for real** (no `-e`, no `--no-deps`), then
      asserts `import openflo` lands in site-packages, the package data
      shipped, `pip check` is clean, `openflo-doctor` reports HEALTHY (which
      runs the golden self-test), the response controls pass, and the console
      scripts run. Verified locally at both edges before merging, and both new
      assertions were checked to actually FAIL when violated.

      Two things it does that the existing `test` job structurally cannot: it
      tests the **built artifact** (an editable install imports out of `src/`,
      so missing package data is invisible), and it tests **both edges of the
      declared ranges** (requirements.txt pins one exact point, so every floor
      in pyproject.toml was an unfalsifiable assertion).

      It also confirmed the `highest` leg is not redundant: pip resolves
      numpy 2.5.3 / pandas 3.0.5 / scipy 1.18.1 / scikit-learn 1.9.0 /
      matplotlib 3.11.1, all **newer** than the requirements.txt pins CI
      otherwise runs. Golden 8/8 and controls 12/12 on that stack too.

      The floor pins come from `scripts/lowest_direct_requirements.py`, which
      generates the constraints file from `pyproject.toml` rather than
      duplicating the floors somewhere a widened bound could go stale.

      macOS is deliberately not in the matrix: nothing in the dependency set is
      macOS-specific, all wheels are universal or provided for both arm64 and
      x86_64, and a third OS triples cost for the least likely failure. Add it
      if a macOS-only report ever arrives.

      **It has found three real bugs so far.** The matplotlib floor (below),
      plus two more on its first CI run:

      - **`openflo-selftest` crashed on a legacy Windows console.** Its results
        table prints check marks, which cp1252 cannot encode, so the command
        that exists to tell a new user whether their install reproduces
        reference behaviour instead gave them a `UnicodeEncodeError`. Probing
        the other entry points found `openflo-doctor --help` failing the same
        way *despite* having a guard — the guard sat after `parse_args`, and
        argparse prints help and exits from inside it. Fixed with a shared
        `openflo._console.force_utf8_streams()` called first in every `main()`,
        and pinned by `tests/test_console_encoding.py`.

        Worth noting how it hid: every local verification run had
        `PYTHONUTF8=1` exported in the shell, which makes the crash impossible.
        The environment that finds this bug is the plain one.

      - **The psutil floor was not installable on Linux.** See its row in the
        evidence table. CI structurally could not catch this one, because an
        ubuntu runner has a C compiler and just built psutil from source; the
        floor leg went green. `python:3.11-slim` has no toolchain and failed,
        which is what an ordinary user's environment looks like. The job now
        asserts every floor pin ships a wheel rather than trusting the runner
        to lack a compiler.

- [x] **~~Correct the matplotlib floor.~~** DONE (2026-09-08). The floor job's
      first run failed with `ResolutionImpossible`: **matplotlib 3.8.0-3.8.3
      declare `numpy<2`**, so `matplotlib>=3.8` with `numpy>=2.0` promised pip
      a combination it can never satisfy. 3.8.4 dropped the cap; the floor is
      now `>=3.8.4`.

      Why bisection missed it: the floor was tested with `pip install --no-deps
      matplotlib==3.8.0`, chosen (correctly) to avoid the venv contamination
      recorded below. But `--no-deps` also skips the *declared* bound, so the
      test proved 3.8.0 **runs** with numpy 2.4.6 — which is true — while the
      thing that was actually broken is that it cannot be **installed**
      alongside it. Running and resolving are different claims, and only a real
      resolve tests the second one.

## Should have before publishing

- [x] **~~Fix the deprecated license declaration.~~** DONE (2026-09-08). Moved
      to PEP 639 — `license = "MIT"` + `license-files = ["LICENSE.txt"]`, trove
      classifier dropped. The build emitted three deprecation warnings
      (`project.license` as a TOML table was slated for removal 2027-Feb-18);
      it now emits none, and the wheel carries `License-Expression: MIT` with
      the licence text under `dist-info/licenses/`.

- [x] **~~Say what to do when Tk is missing.~~** DONE (2026-09-08).
      `openflo-doctor` now has a "Graphical interface" section: it reports the
      Tk version when present, and when absent prints the platform-specific
      command (`apt install python3-tk` / `dnf install python3-tkinter` /
      `brew install python-tk` / reinstall from python.org) plus the fact that
      the CLI and library API do not need Tk. A Tk-less install is explicitly
      NOT reported as unhealthy — a headless server running `openflo-run` is a
      perfectly good install. Tested by simulating the ImportError on each
      platform. Still TODO: the same note in the README.
      *Original finding:* `tkinter` is not installable from
      PyPI — it ships with CPython on Windows and macOS, but on Debian/Ubuntu it
      needs `apt install python3-tk`. A `pip install openflo` on a bare Linux
      box therefore installs fine and then fails at `openflo-gui` with an
      ImportError that does not name the fix. `openflo-doctor` should check for
      it and print the platform-specific command, and the README should say it
      before someone hits it. (Verified present on stock Python 3.12/Windows:
      Tk 8.6, and `openflo.gui` imports cleanly.)

- [x] **~~Ship `py.typed`.~~** DONE (2026-09-08). Added
      `src/openflo/py.typed` and the `package-data` entry; verified present in
      the built wheel. The type-checking CI already does (pyright, 0 errors on
      both platforms) now reaches consumers instead of stopping at the repo
      boundary.

- [ ] **Check the README renders on PyPI.** `twine check` validates the metadata
      but not how the Markdown looks. Relative links and repo-relative image
      paths break on PyPI, which serves the description standalone.

## Worth deciding, not obviously required

- [ ] **Decide what `pip install openflo` promises.** It is a Tk desktop
      application whose primary distribution is the repo plus a launcher.
      Publishing invites `pip install` as a supported path, which means owning
      dependency resolution in arbitrary environments. A defensible middle
      ground: publish, but have the README lead with the repo/launcher install
      and present pip as the route for the CLI and the library API
      (`openflo-run`, `openflo-compare`, importing `openflo.pipeline`).

- [x] **~~Test against the lowest supported Python.~~** DONE (2026-09-08) as a
      side effect of the install job: its floor legs run on **3.11**, so the
      oldest dependency set and the oldest supported Python are now resolved
      and exercised together. Deliberately paired rather than tested
      separately — the floors were chosen as "the first release built for
      numpy 2", and numpy 2.0 ships no wheels for 3.13, so the old stack can
      only run on an old Python anyway.

- [ ] **Consider a `Development Status` bump.** Currently `3 - Alpha`. Honest
      today; revisit when the audit sweeps stop finding event-deleting defects,
      which is the same signal being used to decide when to publish at all.

---

## If unpinning fails — the contingency

Not hypothetical for numpy: the golden baseline's own comment says its 1e-9
tolerances *"rely on the numpy==2.4.6 pin holding ... a numpy bump requires
re-baselining"*. Expect at least one dependency to "fail". That is planned for,
not a blocker.

### "Failed" is three different things, with three different answers

1. **The golden drifts, but nothing is actually wrong.** The 1e-9 metrics
   measure bit-reproducibility of a SEEDED RNG stream, not correctness. A numpy
   bump reshuffles the stream, the metric moves, and no behaviour has changed.
2. **A response control fails.** *That* is a real behaviour change. The controls
   are relational with no pinned numbers, so they survive an RNG reshuffle and
   break only when something genuinely stops tracking its input.
3. **It will not install or import.** Loud, and cheap to bisect.

**The decision rule:**

> golden drifts + controls pass  ->  re-baseline, keep the wider range
> a control fails               ->  keep the pin; the result depends on that
>                                   library version, which is worth knowing

This is the first use of the response controls outside bug-hunting, and it is
the reason they were built without pinned numbers.

### It was never all-or-nothing

The 14 core deps split by how much a pin actually costs:

- **Shared (8)** — numpy, pandas, scipy, scikit-learn, matplotlib, seaborn,
  openpyxl, psutil. These are what a peer package fights over. Widening these
  IS the goal.
- **Leaf (6)** — FlowIO, FlowUtils, PhenoGraph, umap-learn, igraph, leidenalg.
  Almost nothing else in a user's environment depends on them, so a pin here
  costs close to nothing in coexistence terms.

Total failure on the leaves changes very little. The realistic target is the
shared eight, and mostly numpy / pandas / scipy.

### Fallbacks, in order

1. **Exclude the bad release, keep the range** — `numpy>=2.0,<3,!=2.2.0` — when
   one version misbehaves rather than the whole span.
2. **Ranges in `pyproject`, exact set in `constraints.txt`.** Coexistence for
   installers, reproducibility for us: `pip install openflo -c constraints.txt`
   reproduces the tested stack, and CI + the dev venv use it. `requirements.txt`
   already does half of this.
3. **Keep the pin and state why, in the file.** "Our numbers assume this stack"
   is defensible for a scientific tool — but as a stated claim, not an accident.
4. **If most of them resist, change the distribution model, not the project.**
   Exact pins only hurt when sharing an environment. `pipx install openflo` /
   `uv tool install openflo` builds an isolated venv where pins are harmless —
   and for a desktop app with CLI entry points that is arguably the better
   instruction anyway. The README would lead with pipx and offer `pip install`
   for library use. **There is no branch where this dead-ends; worst case the
   install instruction changes.**

### How not to fake a success

Test the EDGES of any range declared, not the middle. Claiming `numpy>=2.0,<3`
while only ever testing 2.4.6 makes the range an unverified assertion — the
same "check that cannot fail" shape this codebase keeps finding. CI should
install the floor and the ceiling and run the golden AND the controls at each.

**This is now the `install` job** in `.github/workflows/ci.yml`, and it paid
for itself on the first run by rejecting the `matplotlib>=3.8` floor. Note what
that failure required to surface: a real dependency resolve. Verifying a floor
by installing it with `--no-deps` and running the suite tests whether the
versions *work together*, which is a weaker claim than whether pip can *put
them together* — and it is the second claim a published range makes.

### Order of attack

Least to most likely to move the numbers, one at a time, with the full suite +
`openflo-selftest --all` + `scripts/mutate_controls.py` after each:

**psutil -> openpyxl -> seaborn -> matplotlib -> scikit-learn -> scipy ->
pandas -> numpy**

numpy last, because it is the one already known to need a deliberate
re-baseline.

| dep | widened to | golden | controls | evidence |
|---|---|---|---|---|
| psutil | `>=5.9.4` (no cap) | 8/8 | 22/22 | Only `virtual_memory()` is used, and both call sites already degrade gracefully without psutil. Floor **tested, not assumed**: installed 5.9.0 into the clean venv — golden 8/8, controls 22/22, `_default_pool_size()` still 2, and `pip check` reports no broken requirements (it reported a conflict before). Re-checked at 7.2.2. **Corrected from `>=5.9` after the Linux container run**: that floor was verified on Windows, where psutil ships per-version wheels. On Linux, 5.9.0-5.9.3 publish wheels only to cp310, so Python 3.11+ must COMPILE psutil. ubuntu-latest has a toolchain and installed it silently; `python:3.11-slim` could not. 5.9.4 ships a `cp36-abi3` wheel covering every supported Python. |
| openpyxl | `>=3.1.5` | 8/8 | 22/22 | Never imported directly — pandas' engine for `read_excel`/`to_excel`. **The floor test earned its keep here:** `>=3.1` looked obviously safe, but with openpyxl 3.1.0 installed pandas raises *"Pandas requires version '3.1.5' or newer of 'openpyxl'"* on the first `read_excel`. The range would have been a false claim that only failed at runtime, in a user's environment. Floor corrected to pandas' own minimum and re-verified with a real `to_excel`/`read_excel` round trip. |
| seaborn | `>=0.12` | 8/8 | 22/22 | One lazy-imported call, `sns.heatmap(cmap=, center=, ax=)`. Floor tested by running the REAL `FlowSample.cluster_heatmap()` under seaborn 0.12.0 in the clean venv, not a reconstruction of the call. |
| matplotlib | `>=3.8.4` | 8/8 | 22/22 | Floor found by **bisection, not judgement**: 3.8.0 runs the full suite clean; **3.6.0 SEGFAULTS**. Cause is a cross-dependency boundary — matplotlib 3.6 predates numpy 2.0, so its compiled extensions hit a numpy-2 ABI mismatch. **Corrected from `>=3.8` after the CI floor job**: 3.8.0-3.8.3 declare `numpy<2`, so that range was unresolvable even though the versions run together — `--no-deps` bisection cannot see a declared bound. |
| scikit-learn | `>=1.6` | 8/8 | 22/22 | Three different limits, only the last of which binds. **Our own** calls — `KMeans(n_init='auto')`, `MDS(normalized_stress='auto')` — work from 1.4. **numpy-2 ABI**: 1.3.2 will not import (`numpy.dtype size changed`), as predicted by matplotlib. **umap-learn** is what actually sets it: 0.5.12 requires `scikit-learn>=1.6` and calls `check_array(ensure_all_finite=...)`, which does not exist earlier. At 1.4.2 UMAP failed **silently** — no exception in the pipeline, the embedding columns were just absent from the events export, and only `test_compute_run_writes_importable_events_csv` caught it. |
| scipy | `>=1.15` | 8/8 | 22/22 | Our own use is all long-stable API and would allow far older. 1.11.4 will not import (numpy-2 ABI). **1.13 and 1.14 import and pass the ENTIRE suite — while warning that numpy 2.4.6 is outside their supported range (<2.3.0).** A green suite is not the same as a supported combination, and shipping that pairing would declare something SciPy disowns. 1.15.0 is the first that supports the numpy we require. |
| pandas | `>=2.2.2` | 8/8 | 22/22 | **Spans the 2/3 major boundary deliberately, on evidence.** Full suite, golden and controls all pass on 2.2.2 AND 3.0.3; `compare_conditions` returns identical numbers on both; no pandas warning is raised. The 2-vs-3 risk is copy-on-write, so it was probed directly — with CoW off (the 2.x default) the aliasing behaves the same, so nothing here depends on it. Floor is the numpy-2 ABI once more: 2.1.4 fails with "numpy.dtype size changed". **This one fixes a live breakage rather than merely permitting one:** `anndata` requires `pandas<3`, so the 3.0.3 pin shipped an `interop` extra that could not be installed alongside its own dependency — `pip check` reported that conflict before and does not now. |
| numpy | `>=2.0,<3` | 8/8 | 22/22 | **The range is not ours to choose:** FlowUtils declares `numpy>=2.0,<3.0`, and that is the same boundary every other compiled dep bottoms out at. The golden warns its 1e-9 tolerances depend on the exact pin — tested rather than believed, and at 2.0.2 the golden is **8/8 with those tolerances intact**, because numpy guarantees the `Generator` stream across versions and the ops are IEEE-deterministic float64. The predicted re-baseline never happened. |

### Two things learned doing matplotlib

**A floor can come from three places, and the loosest guess is usually wrong.**
Across these five: our own API usage (seaborn), a dependency's requirement on
the same package (openpyxl via pandas, scikit-learn via umap-learn), and the
numpy-2 ABI boundary (matplotlib, scikit-learn). Two of the five floors were
set by something other than the code in this repo, so reading our own call
sites is not enough to pick one.

**A passing test suite does not mean a supported combination.** scipy 1.13
and 1.14 ran the whole suite green and reported the golden 8/8 — while
printing a UserWarning that the installed numpy was outside the range they
support. Declaring that pairing in metadata would be asserting something the
library itself disowns. Read the warnings, not just the exit code.

**Silent failure is the reason to test rather than reason.** sklearn 1.4.2
raised nothing — UMAP simply produced no embedding, and the export was missing
two columns. Anyone eyeballing a run would have seen a successful run.

**The compiled floors are really "the first release built for numpy 2".**
matplotlib, scipy and scikit-learn all ship C extensions compiled against a
numpy ABI. Their floors are not independent judgements about their own APIs —
they are all downstream of the numpy major version. Expect the same boundary
for scipy and scikit-learn, and expect these floors to move together if numpy
is ever widened across a major.

**Do not test floors by mutating the dev venv.** Installing matplotlib 3.6.0
silently pulled **numpy from 2.4.6 down to 1.26.4** (3.6 requires numpy<2), and
reinstalling matplotlib did not put it back. A verification run immediately
afterwards reported "full suite green, golden 8/8" — against the wrong numpy.
`pip check` is what caught it. Test floors in the **isolated clean venv**, or
install with `--no-deps`, and check the versions you think you are testing
before believing a result.


Note `requirements.txt` keeps the exact pin. That divergence is the
ranges-vs-constraints split above, and is deliberate: pyproject says what
OpenFlo is COMPATIBLE with, requirements.txt reproduces what it is TESTED
against.

---

## How to re-check any of this

```bash
python -m build                                    # sdist + wheel
python -m twine check dist/*                       # metadata validity

# the real test — a clean environment, not your dev venv:
python -m venv /tmp/fresh
/tmp/fresh/bin/python -m pip install dist/openflo-*.whl
/tmp/fresh/bin/openflo-doctor                      # must print HEALTHY
```

`gh variable list` shows whether `PYPI_PUBLISH` is set — empty means publishing
is still off, which is the current and intended state.
