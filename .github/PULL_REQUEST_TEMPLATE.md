<!--
Thanks for the PR.  A few quick prompts so reviewers can land it faster:
-->

## What this changes

<!-- One or two sentences. The "why" matters more than the "what". -->

## Scientific impact

<!--
Does this change any numeric output (gates, compensation, clustering,
stats)?  If yes:
  - which output(s),
  - by how much (rough magnitude is fine),
  - whether the change is intentional (e.g. fixing a bug) or
    incidental (e.g. picked up via a numpy bump).

Reproducibility-sensitive changes need either a fixture-based
regression test under `tests/` or an explicit CHANGELOG entry
flagged as a behaviour change.
-->

## Test plan

- [ ] `pytest` passes locally
- [ ] `ruff check .` clean
- [ ] `pyright` clean
- [ ] Tested in the GUI (if relevant)

## Checklist

- [ ] CHANGELOG.md updated (under `[Unreleased]`)
- [ ] New public API has a docstring
- [ ] Public type signatures are accurate (or `# type: ignore` with reason)
