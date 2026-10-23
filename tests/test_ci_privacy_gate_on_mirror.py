"""The privacy gate runs on the private repository, never on the mirror.

scripts/publish_public.py holds the private token list, so it is kept out of
the public mirror (export-ignore). The mirror's CI ran it anyway and failed
with "No such file" on every snapshot, and the release build's gate step
would have failed the same way. Both skip on the mirror; on the private
repository they still run on every push and before every PyPI upload."""
from pathlib import Path

from tests.test_ci_macos_job import _jobs

WF = Path(__file__).resolve().parents[1] / '.github' / 'workflows'
SKIP_ON_MIRROR = "github.repository != 'ChironTheCentaur/openflo-public'"


def _gate_ifs(text):
    """The ``if:`` of every job or step that runs publish_public.py, by its
    job id; None where it has none."""
    out = {}
    for name, block in _jobs(text).items():
        if 'publish_public.py' not in block:
            continue
        job_if = next((ln.split('if:', 1)[1].strip()
                       for ln in block.splitlines()
                       if ln.startswith('    if:')), None)
        lines = block.splitlines()
        for i, ln in enumerate(lines):
            if 'publish_public.py' in ln and 'run:' in ln:
                step = lines[max(0, i - 3):i]
                step_if = next((s.split('if:', 1)[1].strip() for s in step
                                if s.strip().startswith('if:')), None)
                out.setdefault(name, []).append(step_if or job_if)
    return out


def test_every_gate_run_skips_on_the_mirror_only():
    for wf in ('ci.yml', 'release.yml'):
        found = _gate_ifs((WF / wf).read_text(encoding='utf-8'))
        assert found, f"{wf} no longer runs the privacy gate"
        for job, conds in found.items():
            for c in conds:
                assert c == SKIP_ON_MIRROR, (wf, job, c)


def test_the_check_catches_an_unguarded_gate():
    # Negative half: a gate step with no condition, or one that would skip
    # the gate on the private repository, is flagged.
    wf = ('on: push\njobs:\n  privacy:\n    runs-on: ubuntu-latest\n'
          '    steps:\n      - name: gate\n'
          '        run: python scripts/publish_public.py --check\n')
    assert _gate_ifs(wf) == {'privacy': [None]}
    wrong = wf.replace('    runs-on:', "    if: github.repository == "
                       "'ChironTheCentaur/openflo-public'\n    runs-on:")
    assert _gate_ifs(wrong)['privacy'] != [SKIP_ON_MIRROR]
