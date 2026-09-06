# Contributing

Thanks for opening a PR — flow cytometry tooling is a small community and
every contribution helps.

## Dev setup

```bash
python -m venv .venv
.venv\Scripts\activate         # or source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Before you push

```bash
pytest                          # tests/ must stay green
python -m pyright               # type check (warnings ok, errors are not)
python -m ruff check .          # lint
```

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
- New exceptions inherit from `OpenFloError` (see `flow_pipeline.py`)
- New IO formats: add a round-trip test under `tests/`

## Scientific correctness

Cytometry is reproducibility-sensitive. Changes that affect numeric output
(gates, compensation, clustering) need either:

1. a fixture-based regression test under `tests/`, **or**
2. an explicit note in `CHANGELOG.md` flagged as a behaviour change.

## Commit messages

One-line summary, imperative mood, ≤72 chars. Body explains *why*, not
*what*. Squash before merge.
