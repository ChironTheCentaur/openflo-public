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
pre-commit install              # runs the type check before every push
```

Working in a git worktree? Don't make another venv: the type check finds the
main checkout's `.venv` by itself.

## Before you push

```bash
pytest                          # tests/ must stay green
python scripts/typecheck.py     # type check, exactly as CI does (no errors, no warnings)
python -m ruff check .          # lint
```

Use `scripts/typecheck.py`, not bare `pyright` or the editor's squiggles.
pyright's answer depends on which interpreter's packages it reads, and bare
`pyright` quietly falls back to whatever `python` is on PATH when it cannot
find `.venv`; Pylance in VS Code also bundles its own pyright version. The
script uses the pinned pyright, checks that your venv has exactly the package
versions CI installs, and confirms from pyright's own output which
site-packages it read. If it exits 2, it tells you what differs from CI.

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
