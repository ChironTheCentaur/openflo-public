"""When ci.yml runs its expensive jobs (tests, type check, install checks).

They cost about 300 Linux-equivalent runner minutes a run on the private
repository, so they run there only by hand ("Run workflow", a paid escape
hatch). The full matrix runs on the public mirror instead (free runners): on
pushes to its ci/** branches, where developers push a scrubbed export of their
tree, and to its main. Every push to a private PR or to master gets only the
fast checks (lint, privacy gate). A private tag push no longer runs ci.yml's
matrix: release.yml checks the mirror's result for the same tree instead
(tests/test_release_mirror_gate.py)."""
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
    ('ready PR push or marked ready',          dict(event='pull_request'),                       False),
    ('merge (push to master)',                 dict(event='push'),                               False),
    ('private tag push',                       dict(event='push', ref='refs/tags/v2.7.0'),       False),
    ('manual run',                             dict(event='workflow_dispatch'),                  True),
    ('public mirror main push',                dict(event='push', ref='refs/heads/main',
                                                    repo=MIRROR),                                True),
    ('public mirror ci/** branch push',        dict(event='push', ref='refs/heads/ci/topic',
                                                    repo=MIRROR),                                True),
]


def _verdicts(text):
    jobs = _jobs(text)
    return {(name, what): _runs(_condition(jobs[name]), **kwargs)
            for name in (*HEAVY, *LIGHT) for what, kwargs, _ in CASES}


def _wrong(text):
    verdicts = _verdicts(text)
    wrong = []
    for what, kwargs, heavy in CASES:
        for name in HEAVY:
            if verdicts[(name, what)] != heavy:
                wrong.append(f'{name} on {what}: runs={verdicts[(name, what)]}')
        for name in LIGHT:
            # The mirror has no publish_public.py (it holds the private token
            # list), so the privacy gate runs everywhere but there.
            want = not (name == 'privacy' and kwargs.get('repo') == MIRROR)
            if verdicts[(name, what)] != want:
                wrong.append(f'{name} on {what}: runs={verdicts[(name, what)]}')
    return wrong


def test_the_full_matrix_runs_only_on_the_mirror_or_by_hand():
    assert _wrong(CI.read_text(encoding='utf-8')) == []


def _with_heavy_condition(text, condition):
    """ci.yml with every heavy job's ``if:`` replaced by *condition*."""
    jobs = _jobs(text)
    out = text
    for name in HEAVY:
        cond_block = re.search(r'^    if: >-\n(?:      .*\n)+', jobs[name], re.M).group(0)
        out = out.replace(jobs[name], jobs[name].replace(cond_block, f'    if: {condition}\n'))
    assert out != text
    return out


def test_the_check_catches_a_private_matrix_on_ready_prs_and_a_dropped_mirror():
    # Negative half. The previous policy also ran the matrix on every push to
    # a ready PR and on a tag push, billed to the private repository.
    text = CI.read_text(encoding='utf-8')
    previous = _with_heavy_condition(text, (
        "github.repository == 'ChironTheCentaur/openflo-public' || "
        "github.event_name == 'workflow_dispatch' || "
        "(github.event_name == 'pull_request' && !github.event.pull_request.draft) || "
        "(github.event_name == 'push' && !startsWith(github.ref, 'refs/heads/'))"))
    wrong = _wrong(previous)
    assert any('on ready PR push or marked ready: runs=True' in w for w in wrong)
    assert any('on private tag push: runs=True' in w for w in wrong)
    # A cut that dropped the mirror would leave no full run anywhere but by hand.
    no_mirror = _with_heavy_condition(text, "github.event_name == 'workflow_dispatch'")
    assert any('on public mirror ci/** branch push: runs=False' in w
               for w in _wrong(no_mirror))


def _push_branches(text):
    """The branch patterns of ``on.push.branches`` (flow-style list)."""
    on = text.split('\njobs:\n', 1)[0]
    m = re.search(r'^  push:\n(?:    #.*\n)*    branches:\s*\[([^\]]*)\]', on, re.M)
    assert m, 'ci.yml has no on.push.branches list'
    return [b.strip().strip('\'"') for b in m.group(1).split(',')]


def test_ci_runs_on_pushes_to_mirror_ci_branches():
    # publish_public.py --ci-branch pushes to ci/<name> on the mirror; without
    # this pattern that push starts no run at all.
    assert 'ci/**' in _push_branches(CI.read_text(encoding='utf-8'))


def test_the_branch_check_catches_a_missing_ci_pattern():
    text = CI.read_text(encoding='utf-8')
    cut = re.sub(r",\s*'ci/\*\*'", '', text, count=1)
    assert cut != text
    assert 'ci/**' not in _push_branches(cut)


def test_superseded_ci_branch_runs_are_cancelled():
    text = CI.read_text(encoding='utf-8')
    m = re.search(r'^  cancel-in-progress:\s*\$\{\{(.+)\}\}\s*$', text, re.M)
    assert m, 'ci.yml has no expression-valued cancel-in-progress'
    cond = m.group(1).strip()
    assert _runs(cond, 'pull_request')
    assert _runs(cond, 'push', ref='refs/heads/ci/topic', repo=MIRROR)
    assert not _runs(cond, 'push', ref='refs/heads/master')
    assert not _runs(cond, 'push', ref='refs/heads/main', repo=MIRROR)
