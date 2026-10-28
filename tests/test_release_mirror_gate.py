"""release.yml builds a tag only after the full CI passed on the tagged tree.

On the public mirror that is the whole of ci.yml (free runners). On the
private repository the matrix is not run again (metered minutes): the
mirror-ci job asks the mirror, with `publish_public.py --ci-status`, for a
green ci.yml run on a commit with the same export tree. Exactly one of the two
runs, so build needs both and must tolerate the skipped one, while never
running when the one that ran failed or was cancelled. publish must still run
after a good build, although one of build's own needs is always skipped."""
import re
from pathlib import Path

from tests.test_ci_macos_job import _jobs

RELEASE = Path(__file__).resolve().parents[1] / '.github' / 'workflows' / 'release.yml'
MIRROR = 'ChironTheCentaur/openflo-public'
PRIVATE = 'ChironTheCentaur/openflo'
STATUS_FUNCS = re.compile(r'\b(always|cancelled|success|failure)\(\)')


def _if(block):
    m = re.search(r'^    if:\s*(.+)$', block, re.M)
    if not m:
        return None
    cond = m.group(1).strip()
    m = re.fullmatch(r'\$\{\{(.*)\}\}', cond)
    return m.group(1).strip() if m else cond


def _needs(block):
    m = re.search(r'^    needs:\s*(.+)$', block, re.M)
    if not m:
        return []
    return [n.strip() for n in m.group(1).strip().strip('[]').split(',') if n.strip()]


def _eval(cond, results, repo=PRIVATE, publish_var='true'):
    """Evaluate a job ``if:`` built from the pieces release.yml uses.

    *results* maps each direct need to its result. A condition with no status
    function gets GitHub's implicit ``success()``: every need succeeded."""
    if cond is None or not STATUS_FUNCS.search(cond):
        implicit = all(r == 'success' for r in results.values())
        if cond is None or not implicit:
            return implicit
    py = cond
    py = re.sub(r"contains\(needs\.\*\.result,\s*('[\w]+')\)",
                lambda m: repr(eval(m.group(1)) in results.values()), py)  # noqa: S307
    py = re.sub(r'needs\.([\w-]+)\.result', lambda m: repr(results[m.group(1)]), py)
    py = py.replace('github.repository', repr(repo))
    py = py.replace('vars.PYPI_PUBLISH', repr(publish_var))
    py = py.replace('always()', 'True').replace('cancelled()', 'False')
    py = py.replace('success()', repr(all(r == 'success' for r in results.values())))
    py = py.replace('failure()', repr('failure' in results.values()))
    py = py.replace('||', ' or ').replace('&&', ' and ')
    py = re.sub(r'!(?!=)', ' not ', py)
    return bool(eval(py, {'__builtins__': {}}))  # noqa: S307 - our own workflow text


def _gate_jobs(text):
    """(the job that calls ci.yml, the job that runs --ci-status)."""
    jobs = _jobs(text)
    ci = next((n for n, b in jobs.items()
               if re.search(r'^    uses: \./\.github/workflows/ci\.yml\s*$', b, re.M)), None)
    mirror = next((n for n, b in jobs.items()
                   if re.search(r'run: python scripts/publish_public\.py --ci-status\s*$',
                                b, re.M)), None)
    return ci, mirror


BUILD_CASES = [
    # (ci result, mirror-ci result, build runs?)
    ('skipped', 'success', True),     # private release, mirror run was green
    ('success', 'skipped', True),     # mirror release, ci.yml passed
    ('skipped', 'failure', False),    # mirror run red, missing or still running
    ('skipped', 'cancelled', False),
    ('failure', 'skipped', False),
    ('cancelled', 'skipped', False),
    ('skipped', 'skipped', False),    # neither gate ran: nothing tested
    # Both ran (a gate's repository condition broke): one red gate is enough
    # to stop the build.
    ('success', 'failure', False),
    ('failure', 'success', False),
    ('cancelled', 'success', False),
    ('success', 'success', True),
]


def _build_wrong(text):
    jobs = _jobs(text)
    ci, mirror = _gate_jobs(text)
    assert ci and mirror, (ci, mirror)
    cond = _if(jobs['build'])
    wrong = []
    for ci_r, mirror_r, want in BUILD_CASES:
        got = _eval(cond, {ci: ci_r, mirror: mirror_r})
        if got != want:
            wrong.append(f'ci={ci_r} mirror-ci={mirror_r}: build runs={got}')
    return wrong


def test_exactly_one_gate_runs_per_repository():
    jobs = _jobs(RELEASE.read_text(encoding='utf-8'))
    ci, mirror = _gate_jobs(RELEASE.read_text(encoding='utf-8'))
    assert ci and mirror
    for repo in (PRIVATE, MIRROR):
        ran = [n for n in (ci, mirror) if _eval(_if(jobs[n]), {}, repo=repo)]
        assert ran == [ci if repo == MIRROR else mirror], repo


def test_the_mirror_gate_asks_for_the_mirror_result_with_a_token():
    jobs = _jobs(RELEASE.read_text(encoding='utf-8'))
    _, mirror = _gate_jobs(RELEASE.read_text(encoding='utf-8'))
    block = jobs[mirror]
    assert 'run: python scripts/publish_public.py --check\n' in block
    step = block.split('--ci-status', 1)[0].rsplit('- name:', 1)[1]
    assert 'GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}' in step


def test_build_runs_only_after_a_passing_gate():
    text = RELEASE.read_text(encoding='utf-8')
    assert set(_needs(_jobs(text)['build'])) == set(_gate_jobs(text))
    assert _build_wrong(text) == []


def test_the_build_check_catches_a_build_that_ignores_a_failed_mirror_run():
    # Negative half: `always()` alone would build after a red mirror run, and
    # the default (no if:) would never build, since one need is always skipped.
    text = RELEASE.read_text(encoding='utf-8')
    build = _jobs(text)['build']
    line = re.search(r'^    if:.*\n', build, re.M).group(0)
    lax = text.replace(build, build.replace(line, '    if: ${{ always() }}\n'))
    assert 'ci=skipped mirror-ci=failure: build runs=True' in _build_wrong(lax)
    bare = text.replace(build, build.replace(line, ''))
    assert 'ci=skipped mirror-ci=success: build runs=False' in _build_wrong(bare)


def test_publish_runs_after_a_good_build_and_only_then():
    jobs = _jobs(RELEASE.read_text(encoding='utf-8'))
    assert _needs(jobs['publish']) == ['build']
    cond = _if(jobs['publish'])
    # GitHub's implicit success() also looks past build to its skipped need,
    # so a condition without a status function would never publish here.
    assert STATUS_FUNCS.search(cond or ''), cond
    assert _eval(cond, {'build': 'success'})
    assert not _eval(cond, {'build': 'success'}, publish_var='false')
    for bad in ('skipped', 'failure', 'cancelled'):
        assert not _eval(cond, {'build': bad}), bad
