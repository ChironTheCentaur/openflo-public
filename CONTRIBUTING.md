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
pre-commit install              # runs scripts/preflight.py before every push
```

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

- PEP 8 via `ruff format`
- Type hints encouraged on public functions; gradual typing is fine
- New exceptions inherit from `OpenFloError` (see `src/openflo/pipeline.py`)
- New IO formats: add a round-trip test under `tests/`

## Scientific correctness

Cytometry is reproducibility-sensitive. Changes that affect numeric output
(gates, compensation, clustering) need either:

1. a fixture-based regression test under `tests/`, **or**
2. an explicit note in `CHANGELOG.md` flagged as a behaviour change.

## Commit messages

One-line summary, imperative mood, ≤72 chars. Body explains *why*, not
*what*. Squash before merge.
