"""The macOS CI job runs only on the public mirror, where GitHub's macOS
runners are free; on the private repository they bill at ten times Linux."""
import re
from pathlib import Path

CI = Path(__file__).resolve().parents[1] / '.github' / 'workflows' / 'ci.yml'


def _jobs(text):
    """{job id: its block} for the top-level jobs in a workflow file."""
    body = text.split('\njobs:\n', 1)[1]
    parts = re.split(r'^  ([\w-]+):\n', body, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def _macos_jobs(text):
    return {k: v for k, v in _jobs(text).items()
            if re.search(r'^    runs-on:.*macos', v, re.M)}


def _gated_to_mirror(block):
    m = re.search(r'^    if:\s*(.+)$', block, re.M)
    return bool(m) and \
        "github.repository == 'ChironTheCentaur/openflo-public'" in m.group(1)


def test_there_is_a_macos_job_and_it_runs_only_on_the_public_mirror():
    jobs = _macos_jobs(CI.read_text(encoding='utf-8'))
    assert jobs, 'ci.yml has no macOS job'
    for name, block in jobs.items():
        assert _gated_to_mirror(block), name
        assert 'pytest -n auto' in block and "OPENFLO_REQUIRE_GUI: '1'" in block


def test_the_gate_check_catches_an_ungated_macos_job():
    # Negative half: a macOS job without the mirror condition (it would bill
    # the private repository on every push) is found and flagged.
    wf = ('on: push\njobs:\n  mac:\n    name: m\n    runs-on: macos-latest\n'
          '    steps:\n      - run: pytest -n auto\n')
    jobs = _macos_jobs(wf)
    assert list(jobs) == ['mac']
    assert not _gated_to_mirror(jobs['mac'])
    other = wf.replace("    runs-on:", "    if: github.repository == 'someone/else'\n    runs-on:")
    assert not _gated_to_mirror(_macos_jobs(other)['mac'])
