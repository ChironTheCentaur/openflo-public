# Contributing

Thanks for opening a PR — flow cytometry tooling is a small community and
every contribution helps.

## Dev setup

```bash
python -m venv .venv
.venv\Scripts\activate         # or source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -r requirements-typecheck.txt   # optional backends, for pyright
pip install -e ".[dev]"
pip install pre-commit          # optional, not in the dev extra
pre-commit install              # runs scripts/preflight.py before every push
```

The hooks are optional: without them, run `python scripts/preflight.py`
yourself before you push (below).

On Linux the GUI tests also need Tk and, without a desktop, Xvfb
(Debian/Ubuntu: `apt-get install python3-tk xvfb`).

Working in a git worktree? Don't make another venv: the type check finds the
main checkout's `.venv` by itself.

## Before you push

```bash
python scripts/preflight.py           # what CI runs; the pre-push hook
python scripts/preflight.py --quick   # same without the tests, while iterating
```

It checks that master is merged in (the tree you merge is the tree that must
pass in full), then runs ruff, `scripts/typecheck.py`, the privacy gate and the full test
suite with the GUI required. Don't rely on a bare `pytest`: without a display
or tkinter, about 500 GUI tests skip and the run still says green, while CI
runs them under Xvfb.

Use `scripts/typecheck.py`, not bare `pyright` or the editor's squiggles.
pyright's answer depends on which interpreter's packages it reads, and bare
`pyright` quietly falls back to whatever `python` is on PATH when it cannot
find `.venv`; Pylance in VS Code also bundles its own pyright version. The
script uses the pinned pyright, checks that your venv has exactly the package
versions CI installs, and confirms from pyright's own output which
site-packages it read. If it exits 2, it tells you what differs from CI.

New code is reviewed against [docs/best-practices.md](docs/best-practices.md).
`python scripts/best_practices.py` (part of preflight and CI's lint job)
checks the mechanical rules there and fails when a change adds a violation.

## CI

The full matrix (tests with the GUI on Linux, Windows and macOS, the type
check, the install checks) runs on the public mirror, `openflo-public`, where
runners are free. CI on this repository runs only lint and the privacy gate,
on every PR push and every push to master.

To test a change in full, push a scrubbed export of your committed `HEAD` to a
`ci/<name>` branch on the mirror, then check the result:

```bash
python scripts/publish_public.py --ci-branch <name>          # dry run
python scripts/publish_public.py --ci-branch <name> --push   # mirror CI runs the full matrix
python scripts/publish_public.py --ci-status                 # 0 green, 1 failed, 2 none yet / running
```

Uncommitted changes are not exported. A newer push to the same branch cancels
the run it supersedes. Merge a PR once master is merged into it and the mirror
run for that final tree is green: a release is built only after a green
mirror run for the same tree (`release.yml`), so it does not run the matrix
again. "Run workflow" on the Actions tab still runs the full matrix here, as a
paid escape hatch.

## Issue templates

Bug reports are far more useful with:

- OS and Python version
- Output of `pip freeze | grep -iE "flow|phenograph|umap|numpy|scipy"`
- A minimal FCS that reproduces the issue (anonymise if needed)
- The CLI invocation or GUI steps
- Full traceback (not just the last line)

## Code style

- Lint with `ruff check` (the version pinned in the dev extra and CI). Code is
  not auto-formatted: `ruff format` would rewrite almost every file, and
  nothing checks formatting, so it was removed on purpose (the reasons are in
  `.pre-commit-config.yaml`). Match the style of the file you are editing and
  don't reformat lines you are not changing.
- Type hints encouraged on public functions; gradual typing is fine
- New IO formats: add a round-trip test under `tests/`

### Exceptions

An error OpenFlo raises on purpose, so a caller can catch it by type, is one
of these:

- **A failed pipeline step** (an FCS file, a compensation matrix, a FlowJo
  workspace, a gate or a clustering run that can't be read or applied):
  raise or subclass `OpenFloError` from `src/openflo/exceptions.py`
  (`FcsParseError`, `CompensationError`, `WspParseError`, `GateError`,
  `ClusteringError`). Top-level callers catch `OpenFloError` to show a plain
  message.
- **Bad input a user typed or a file they picked**, rejected by a parser:
  subclass `ValueError`, so callers that already catch `ValueError` keep
  working. These are `ProjectArchiveError`, `FormulaError`, `GatingMLError`,
  `SpectraCSVError`, `SessionVersionError`, `BarcodeKeyError` and
  `UnknownScaleError`.
- **A cancelled background job**: `JobCancelled` and `Cancelled` subclass
  `Exception` directly. They are control flow, not errors, so
  `except OpenFloError` must not swallow them.

A new exception class goes in one of these groups, and a new one outside the
`OpenFloError` tree is named in this list; `tests/test_contributing_guide.py`
fails when the two disagree.

## Tests

Every test must be proved both ways, on generated data (`openflo.synthetic`,
or a file the test writes itself), never a real sample: it passes on the real
code, and it fails when the code it guards is broken. Before you push a new or
changed test, break the code it guards by hand (flip a comparison, change a
default, delete a line), watch the test fail, then undo the break.

`scripts/prove_tests.py` does this mechanically and keeps the record in
`docs/test-assurance/` (`ledger.csv` has one row per test and the break that
made it fail; its README explains the method). A full sweep takes hours, so
for a few tests write the breaks by hand, as a JSON list in the shape of
`docs/test-assurance/aimed.json` (the tests, the file, the text to find and
its replacement, and why), and run only those:

```bash
python scripts/prove_tests.py aimed my-breaks.json aimed-out
```

A test the automatic search can't reach keeps its break: add the entry to
`docs/test-assurance/aimed.json`.

## Module naming

A module's name says which side of the GUI / core split it is on, so you can
tell from the file list what may import Tk:

- **Core** (no prefix): Tk-free library code. It imports without a display
  and takes GUI settings as arguments, not from the preferences file (only
  `table_export`, the shared CSV writer, may read them). The library, the CLI
  and the tests use it directly.
- **`editor_*`**: mixins of the gate editor window (`gui.ViewGateEditorWindow`
  is built from them). Each holds one area of the editor, such as
  `editor_plot` or `editor_session`.
- **`ui_*`**: the secondary windows and dialogs, one per tool, such as
  `ui_comp` or `ui_preferences`.

A new module follows this. The split is checked by
`scripts/best_practices.py` (rules `core-imports-tk` and `core-reads-prefs`),
and its `GUI_MODULE_PREFIXES`, `GUI_MODULES` and `CORE_MODULES` are the one
list of what counts as GUI. Some older modules break the naming; they are
documented here rather than renamed:

- `ui_logic` is core despite its prefix: Tk-free decision logic behind the
  windows, so it can be tested headless.
- `view_tabs` and `layout_page` are core: Tk-free models of the Samples &
  Gates tabs and of layout pages, used by GUI windows. `layout_render`, which
  draws a layout page through the editor's plotting code, is core too and
  Tk-free, but still reads the layout colours from the preferences file (a
  known violation kept in the best-practices baseline).
- `gui`, `theme`, `workspace`, `tool_window`, `dialog_inputs`, `async_task`,
  `group_test_choice`, `tk_threads` and `clipboard_export` are GUI without a
  prefix: the main window, Tk styles, the Pipeline Workspace window, the
  base class and shared widgets of the tool windows, and Tk plumbing.
- `gating` and `gating_helpers` are both core and read alike, but do
  different jobs. `gating` works on the gates a session already has (event
  counts and % of parent, labels, population paths, remapping a combination
  gate's operands, the per-population statistics table). `gating_helpers` builds new gate dicts from data
  (singlet and FMO threshold gates) for scripts, the pipeline and the
  editor's gate tools.

When a module needs an exception, add it to the lists in
`scripts/best_practices.py` and here; `tests/test_module_naming_documented.py`
fails when the two disagree.

## Scientific correctness

Cytometry is reproducibility-sensitive. Changes that affect numeric output
(gates, compensation, clustering) need either:

1. a fixture-based regression test under `tests/`, **or**
2. an explicit note in `CHANGELOG.md` flagged as a behaviour change.

## Commit messages

One-line summary, imperative mood, ≤72 chars. Body explains *why*, not
*what*. Keep each commit focused and its message clear; tidy up fixup
commits before review. PRs are merged with a merge commit, so your commits
land as you wrote them.
