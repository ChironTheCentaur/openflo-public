"""When ci.yml runs its expensive jobs (tests, type check, install checks).

They cost about 300 Linux-equivalent runner minutes a run on the private
repository, so they run at the major steps only: a ready PR, a release, a
manual run, and the public mirror (free runners). A push to master, which is
every merge, gets only the fast checks: the PR's own full run already tested
the tree master gets. release.yml calls ci.yml on a tag push and a called
workflow sees the caller's event, so the release case is a push to a tag."""
import re
from pathlib import Path

CI = Path(__file__).resolve().parents[1] / '.github' / 'workflows' / 'ci.yml'
HEAVY = ('test', 'typecheck', 'install')
LIGHT = ('lint', 'privacy')

PRIVATE = 'ChironTheCentaur/openflo'
MIRROR = 'ChironTheCentaur/openflo-public'


def _jobs(text):
    """{job id: its block} for the top-level jobs in a workflow file."""
    body = text.split('\njobs:\n', 1)[1]
    parts = re.split(r'^  ([\w-]+):\n', body, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def _condition(block):
    """The job's ``if:``, folded onto one line; None when it has none."""
    m = re.search(r'^    if: >-\n((?:      .*\n)+)', block, re.M)
    if m:
        return ' '.join(line.strip() for line in m.group(1).splitlines())
    m = re.search(r'^    if:\s*(.+)$', block, re.M)
    return m.group(1).strip() if m else None


def _runs(condition, event, ref='refs/heads/master', draft=False, repo=PRIVATE):
    """Evaluate a GitHub expression made of the pieces ci.yml uses."""
    if condition is None:
        return True
    context = {
        'github.repository': repo,
        'github.event_name': event,
        'github.ref': ref,
        'github.event.pull_request.draft': draft,
    }
    py = condition
    py = re.sub(r'startsWith\(([^,]+),\s*([^)]+)\)', r'(\1).startswith(\2)', py)
    py = re.sub(r'github(\.[\w]+)+', lambda m: repr(context[m.group(0)]), py)
    py = py.replace('||', ' or ').replace('&&', ' and ')
    py = re.sub(r'!(?!=)', ' not ', py)
    py = re.sub(r'\btrue\b', 'True', re.sub(r'\bfalse\b', 'False', py))
    return bool(eval(py, {'__builtins__': {}}))  # noqa: S307 - our own workflow text


CASES = [
    # (what happens,                          kwargs,                                           heavy?)
    ('draft PR push',                          dict(event='pull_request', draft=True),           False),
    ('ready PR push or marked ready',          dict(event='pull_request'),                       True),
    ('merge (push to master)',                 dict(event='push'),                               False),
    ('release (tag push calling ci.yml)',      dict(event='push', ref='refs/tags/v2.7.0'),       True),
    ('manual run',                             dict(event='workflow_dispatch'),                  True),
    ('public mirror master push',              dict(event='push', repo=MIRROR),                  True),
]


def _verdicts(text):
    jobs = _jobs(text)
    return {(name, what): _runs(_condition(jobs[name]), **kwargs)
            for name in (*HEAVY, *LIGHT) for what, kwargs, _ in CASES}


def _wrong(text):
    verdicts = _verdicts(text)
    wrong = []
    for what, _, heavy in CASES:
        for name in HEAVY:
            if verdicts[(name, what)] != heavy:
                wrong.append(f'{name} on {what}: runs={verdicts[(name, what)]}')
        for name in LIGHT:
            if not verdicts[(name, what)]:
                wrong.append(f'{name} on {what}: skipped')
    return wrong


def test_the_full_matrix_runs_only_at_major_steps():
    assert _wrong(CI.read_text(encoding='utf-8')) == []


def test_the_check_catches_a_full_run_on_every_merge_and_a_release_without_tests():
    # Negative half. The old condition ran the matrix on every master push.
    text = CI.read_text(encoding='utf-8')
    jobs = _jobs(text)
    old = "    if: github.event_name != 'pull_request' || !github.event.pull_request.draft\n"
    before = text
    for name in HEAVY:
        cond_block = re.search(r'^    if: >-\n(?:      .*\n)+', jobs[name], re.M).group(0)
        before = before.replace(jobs[name], jobs[name].replace(cond_block, old))
    assert any('on merge (push to master): runs=True' in w for w in _wrong(before))
    # A cut that also dropped tag pushes would let a release build untested.
    no_tags = text.replace(
        "(github.event_name == 'push' && !startsWith(github.ref, 'refs/heads/'))", 'false')
    assert no_tags != text
    assert any('on release (tag push calling ci.yml): runs=False' in w
               for w in _wrong(no_tags))
