# Best practices checklist

The rules new code is reviewed against. Each rule says what to do, why, and
whether `scripts/best_practices.py` checks it mechanically. Rules marked
*review* need a person (or the review agent) to read the diff.

`scripts/best_practices.py` is a ratchet: violations that existed when a rule
was added are counted per file in `scripts/best_practices_baseline.json`, and
the check fails only when a change makes a file's count go up. When a change
removes violations, run `python scripts/best_practices.py --write-baseline`
so they cannot come back. A diff that *raises* a baseline count needs a stated
reason in the PR; a reviewer treats it as a finding. Preflight and the CI lint
job run the check.

## BP1. One home for each calculation

Do not copy a helper, statistic, regex or gate rule into a second module.
Import the existing one, or move it to a shared module and import it from both
places. Two copies drift: two "is this process alive" checks already disagreed
on a permission error, two singlet gates used opposite ratios, and the robust
SD existed in five versions.

Before adding a function, search for one that already does it (by what it
computes, not only by name). Two public functions with the same name in
different modules (two `write_fcs`) are a copy in waiting, and confuse callers.

Checked: `duplicate-function` (identical bodies across modules),
`duplicate-public-name`.

## BP2. The library does not depend on the GUI

Core modules (everything that is not an `editor_*` mixin, a `ui_*` window or
the other GUI modules listed in `scripts/best_practices.py`) must import
without Tk and must not read the GUI preferences file. A GUI setting reaches
library code as an argument the GUI passes in; the library's own default is
the documented one (FlowJo forms for statistics). Otherwise the same library
call gives different numbers on two machines.

Checked: `core-imports-tk`, `core-reads-prefs`.

## BP3. Importing a module does nothing else

No process-wide warning filters, no `sys.exit`, no module globals that a
function call changes (pass the value as a parameter instead). Importing
`openflo.pipeline` once silenced every FutureWarning in the process.

Checked: `import-side-effect`. *Review:* module globals set by `run()`.

## BP4. `TYPE_CHECKING` imports are for annotations only

A name imported under `if TYPE_CHECKING:` does not exist at runtime. Using it
in an `except`, `isinstance` or call turns a handled error into a NameError
crash. Import it normally if code uses it.

Checked: ruff's TC004 rule (in `ruff check`).

## BP5. Output goes through the shared writers

CSV files are written with the shared writer (`table_export.write_frame_csv`
/ `write_rows_csv`) so the CSV-style preference (separator, decimal mark, NA
text) applies to every export. The app version is read through
`update.current_version()` so Diagnostics and export stamps agree.

Checked: `csv-bypass`, `version-read`.

## BP6. The library, CLI and GUI give the same answer

When the CLI and the library (or the GUI) do the same job (channel matching,
FMO thresholds, export columns), they call the same code. A difference in
behaviour is a documented option, not a second implementation.

*Review.*

## BP7. Errors are reported, never lost

- A broad `except Exception` says why it is broad (`# noqa: BLE001 - reason`)
  and logs or records what it caught, unless it is Tk teardown.
- A counted failure keeps its message ("3 failed" must say why).
- Library code logs; it does not `print`. Printing is for the CLI.
- Prefer the narrow exception (`tk.TclError`, `OSError`) over `Exception`.

Checked: `print-in-library`; silent handlers are policed by the existing
silent-handler test. *Review:* the rest.

## BP8. Runtime checks are real checks

`assert` disappears under `python -O`. A condition the program relies on
(thread safety, valid input) raises a real error. `assert` is fine for type
narrowing and in tests.

*Review.*

## BP9. Name the numbers

A number with a meaning (the open-gate bound, logicle defaults, asinh
cofactors, the MAD constant) is a named constant defined once and imported.

Checked: `magic-open-bound` (the 1e12 bound is `scales.OPEN`). *Review:* other constants.

## BP10. Small public surface, no dead code

- No private name (`_x`) imported or reached into from another module. If
  other modules need it, make it public under a clear name.
- No pass-through functions that only forward to another, and no second name
  for the same function.
- No unused functions, constants, locals or commented-out code. A public name
  being removed gets a deprecation first.
- Re-exports exist for users, not for tests: point tests at the real home.

Checked: `private-cross-import`. *Review:* the rest.

## BP11. Names and comments say what the code does now

- A name describes current behaviour; rename it when the behaviour changes.
- One name for one idea across windows (`_export_png` everywhere, not three
  spellings); no cryptic prefixes.
- Comments and docstrings state the rule the code keeps. History ("used to",
  "PR #78") and work-tracker tags belong in git and the CHANGELOG.

Checked: `tracker-tag`. *Review:* naming.

## BP12. Keep modules and functions in their lane

- Module naming: Tk-free core, `editor_*` mixins of the gate editor, `ui_*`
  windows. A new module follows it.
- Big files grow by new modules, not by more sections. A new function over
  about 100 lines or with deeply nested branching gets split, or a per-kind
  dispatch table replaces the if-chain.
- Walk a gate's parents with the shared helper, build scrollable panels with
  the shared helper.

*Review.*

## BP13. Public functions are consistent, typed and documented

Public functions take arguments in the order their siblings use (data, then
path, then options, or whatever the module already does), carry type hints,
and have a docstring. Renaming a public library name keeps the old name as an
alias for at least one release.

*Review.*

## BP14. Background work uses the shared helper

GUI code starts worker threads through `async_task.run_async` (closed-window
guard and error routing), not a raw `threading.Thread` plus `after`.

Checked: `raw-thread`.

## BP15. Tests prove something

- Every new or changed test is shown to fail on broken code and pass on the
  fix (`scripts/prove_tests.py` or a mutation check).
- Shared fixtures and helpers live in `conftest.py`; do not copy them.
- Assert the expected value, not just `is not None`. Never sleep and then
  assert that something did not happen; wait on the event instead.
- Do not blanket-ignore warnings; a loose numeric tolerance has a comment
  saying why.
- Test files are named for the feature they test.
- Synthetic data and names only (FL1-A, sample_1): no real reagents, panels,
  sample names or local paths, in tests or anywhere else.

*Review.*

## BP16. Docs and packaging stay true

When a change makes CONTRIBUTING, the README, MANIFEST.in, environment.yml or
a comment wrong, fix them in the same PR. Changes to numeric output need a
regression test or a CHANGELOG behaviour-change note.

*Review.*
