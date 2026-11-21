"""Update on a source checkout (openflo.update._run_git_update), run against
real throw-away git repos: a bare "GitHub" repo and a clone of it.

The case that started this: the checkout sat on a pull request's branch that
was merged and then deleted on GitHub, and Update ran `git pull`, which
failed with "no such ref was fetched" and showed git's raw output.
"""
from __future__ import annotations

import subprocess

import pytest

from openflo import update


@pytest.fixture(autouse=True)
def _git_identity(monkeypatch):
    # An empty email is allowed when set this way, and keeps the scrub
    # gate's email check quiet.
    for k, v in (('GIT_AUTHOR_NAME', 't'), ('GIT_AUTHOR_EMAIL', ''),
                 ('GIT_COMMITTER_NAME', 't'), ('GIT_COMMITTER_EMAIL', '')):
        monkeypatch.setenv(k, v)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')


def git(cwd, *args):
    out = subprocess.run(['git', '-C', str(cwd), *args], capture_output=True,
                         text=True, check=True)
    return out.stdout.strip()


def commit(cwd, name, text):
    (cwd / name).write_text(text, encoding='utf-8')
    git(cwd, 'add', name)
    git(cwd, 'commit', '-q', '-m', f'{name}: {text}')
    return git(cwd, 'rev-parse', 'HEAD')


class Hub:
    """A bare 'GitHub' repo, a maintainer clone that pushes to it, and the
    user's checkout (``co``)."""

    def __init__(self, tmp_path, main='master'):
        self.main = main
        self.bare = tmp_path / 'hub.git'
        subprocess.run(['git', 'init', '-q', '--bare', '-b', main,
                        str(self.bare)], check=True)
        self.dev = tmp_path / 'dev'
        subprocess.run(['git', 'clone', '-q', str(self.bare), str(self.dev)],
                       check=True, capture_output=True)
        git(self.dev, 'checkout', '-q', '-b', main)
        commit(self.dev, 'a.txt', 'one')
        git(self.dev, 'push', '-q', 'origin', main)
        self.co = tmp_path / 'co'
        subprocess.run(['git', 'clone', '-q', str(self.bare), str(self.co)],
                       check=True, capture_output=True)

    def push_branch(self, name, text):
        git(self.dev, 'checkout', '-q', '-b', name, self.main)
        sha = commit(self.dev, 'b.txt', text)
        git(self.dev, 'push', '-q', 'origin', name)
        git(self.dev, 'checkout', '-q', self.main)
        return sha

    def merge_and_delete(self, name, delete=True):
        git(self.dev, 'merge', '-q', '--no-ff', '-m', f'merge {name}', name)
        tip = commit(self.dev, 'c.txt', 'after merge')
        git(self.dev, 'push', '-q', 'origin', self.main)
        if delete:
            git(self.dev, 'push', '-q', 'origin', '--delete', name)
        return tip

    def checkout_branch(self, name):
        git(self.co, 'fetch', '-q', 'origin')
        git(self.co, 'checkout', '-q', '-b', name, '--track', f'origin/{name}')


def head(co):
    return git(co, 'rev-parse', 'HEAD')


def branch(co):
    return git(co, 'rev-parse', '--abbrev-ref', 'HEAD')


@pytest.fixture
def hub(tmp_path):
    return Hub(tmp_path)


def test_merged_branch_deleted_on_github_switches_to_master(hub):
    hub.push_branch('pr-1', 'feature')
    hub.checkout_branch('pr-1')
    (hub.co / 'my-launcher.bat').write_text('local', encoding='utf-8')
    tip = hub.merge_and_delete('pr-1')

    ok, msg = update._run_git_update(str(hub.co))

    assert ok, msg
    assert branch(hub.co) == 'master' and head(hub.co) == tip
    assert "Switched from branch 'pr-1'" in msg
    assert 'no such ref' not in msg
    # an untracked local file is kept
    assert (hub.co / 'my-launcher.bat').read_text(encoding='utf-8') == 'local'


def test_merged_branch_still_on_github_switches_to_master(hub):
    hub.push_branch('pr-2', 'feature')
    hub.checkout_branch('pr-2')
    tip = hub.merge_and_delete('pr-2', delete=False)

    ok, msg = update._run_git_update(str(hub.co))

    assert ok, msg
    assert branch(hub.co) == 'master' and head(hub.co) == tip


def test_live_branch_with_new_work_is_pulled_not_switched(hub):
    hub.push_branch('dev-x', 'v1')
    hub.checkout_branch('dev-x')
    git(hub.dev, 'checkout', '-q', 'dev-x')
    new = commit(hub.dev, 'b.txt', 'v2')
    git(hub.dev, 'push', '-q', 'origin', 'dev-x')

    ok, msg = update._run_git_update(str(hub.co))

    assert ok, msg
    assert branch(hub.co) == 'dev-x' and head(hub.co) == new


def test_merged_branch_with_local_edits_is_left_alone(hub):
    hub.push_branch('pr-3', 'feature')
    hub.checkout_branch('pr-3')
    hub.merge_and_delete('pr-3')
    before = head(hub.co)
    (hub.co / 'a.txt').write_text('my edit', encoding='utf-8')

    ok, msg = update._run_git_update(str(hub.co))

    assert not ok
    assert 'git stash' in msg and 'Update did not switch' in msg
    assert 'git stash pop' in msg
    assert branch(hub.co) == 'pr-3' and head(hub.co) == before
    assert (hub.co / 'a.txt').read_text(encoding='utf-8') == 'my edit'


def test_deleted_branch_with_unmerged_work_is_left_alone(hub):
    hub.push_branch('pr-4', 'feature')
    hub.checkout_branch('pr-4')
    mine = commit(hub.co, 'd.txt', 'not pushed')
    git(hub.dev, 'push', '-q', 'origin', '--delete', 'pr-4')

    ok, msg = update._run_git_update(str(hub.co))

    assert not ok
    assert "branch 'pr-4', which was deleted on GitHub" in msg
    assert 'git checkout master' in msg
    assert branch(hub.co) == 'pr-4' and head(hub.co) == mine


def test_branch_without_upstream_unmerged_is_left_alone(hub):
    git(hub.co, 'checkout', '-q', '-b', 'local-only')
    mine = commit(hub.co, 'd.txt', 'local')

    ok, msg = update._run_git_update(str(hub.co))

    assert not ok
    assert 'deleted on GitHub' not in msg and 'git checkout master' in msg
    assert branch(hub.co) == 'local-only' and head(hub.co) == mine


def test_merged_branch_without_upstream_switches(hub):
    git(hub.co, 'checkout', '-q', '-b', 'local-only')
    tip = commit(hub.dev, 'e.txt', 'new on master')
    git(hub.dev, 'push', '-q', 'origin', 'master')

    ok, msg = update._run_git_update(str(hub.co))

    assert ok, msg
    assert branch(hub.co) == 'master' and head(hub.co) == tip


def test_detached_head_switches_to_master(hub):
    git(hub.co, 'checkout', '-q', '--detach')
    tip = commit(hub.dev, 'e.txt', 'new on master')
    git(hub.dev, 'push', '-q', 'origin', 'master')

    ok, msg = update._run_git_update(str(hub.co))

    assert ok, msg
    assert 'Switched from a detached commit' in msg
    assert branch(hub.co) == 'master' and head(hub.co) == tip


def test_switch_creates_master_when_the_checkout_has_none(hub):
    hub.push_branch('pr-5', 'feature')
    hub.checkout_branch('pr-5')
    git(hub.co, 'branch', '-q', '-D', 'master')
    # Without these git's own guesses would make a plain checkout work.
    git(hub.co, 'config', 'checkout.guess', 'false')
    git(hub.co, 'config', 'branch.autoSetupMerge', 'false')
    tip = hub.merge_and_delete('pr-5')

    ok, msg = update._run_git_update(str(hub.co))

    assert ok, msg
    assert branch(hub.co) == 'master' and head(hub.co) == tip
    assert git(hub.co, 'rev-parse', '--abbrev-ref', 'master@{u}') == 'origin/master'


def test_master_is_fast_forwarded(hub):
    tip = commit(hub.dev, 'e.txt', 'new on master')
    git(hub.dev, 'push', '-q', 'origin', 'master')

    ok, msg = update._run_git_update(str(hub.co))

    assert ok, msg
    assert 'Switched' not in msg
    assert branch(hub.co) == 'master' and head(hub.co) == tip


def test_master_with_local_commits_is_not_forced(hub):
    commit(hub.dev, 'e.txt', 'new on master')
    git(hub.dev, 'push', '-q', 'origin', 'master')
    mine = commit(hub.co, 'f.txt', 'local commit')

    ok, msg = update._run_git_update(str(hub.co))

    assert not ok
    assert 'commits that are not on GitHub' in msg
    assert head(hub.co) == mine


def test_unreachable_remote_reports_plainly_and_changes_nothing(hub):
    hub.push_branch('pr-6', 'feature')
    hub.checkout_branch('pr-6')
    git(hub.co, 'remote', 'set-url', 'origin', str(hub.bare) + '-missing')
    before = head(hub.co)

    ok, msg = update._run_git_update(str(hub.co))

    assert not ok
    assert msg.startswith('Could not download the update from GitHub')
    assert branch(hub.co) == 'pr-6' and head(hub.co) == before


def test_run_update_uses_the_git_flow_for_a_checkout(hub, monkeypatch):
    hub.push_branch('pr-7', 'feature')
    hub.checkout_branch('pr-7')
    tip = hub.merge_and_delete('pr-7')
    monkeypatch.setattr(update, '_package_git_root', lambda: str(hub.co))

    ok, msg = update.run_update(kind='git')

    assert ok, msg
    assert branch(hub.co) == 'master' and head(hub.co) == tip


def test_merged_branch_with_new_work_on_github_is_pulled(hub):
    hub.push_branch('pr-8', 'feature')
    hub.checkout_branch('pr-8')
    hub.merge_and_delete('pr-8', delete=False)
    git(hub.dev, 'checkout', '-q', 'pr-8')
    new = commit(hub.dev, 'b.txt', 'more work after the merge')
    git(hub.dev, 'push', '-q', 'origin', 'pr-8')

    ok, msg = update._run_git_update(str(hub.co))

    assert ok, msg
    assert branch(hub.co) == 'pr-8' and head(hub.co) == new


def test_switch_blocked_by_an_untracked_file_reports_and_keeps_it(hub):
    hub.push_branch('pr-9', 'feature')
    hub.checkout_branch('pr-9')
    hub.merge_and_delete('pr-9')       # master now tracks c.txt
    (hub.co / 'c.txt').write_text('mine', encoding='utf-8')
    before = head(hub.co)

    ok, msg = update._run_git_update(str(hub.co))

    assert not ok
    assert msg.startswith('Could not switch to master')
    assert branch(hub.co) == 'pr-9' and head(hub.co) == before
    assert (hub.co / 'c.txt').read_text(encoding='utf-8') == 'mine'


def test_merged_branch_with_local_master_commits_is_left_alone(hub):
    hub.push_branch('pr-10', 'feature')
    hub.checkout_branch('pr-10')
    git(hub.co, 'checkout', '-q', 'master')
    mine = commit(hub.co, 'f.txt', 'local commit on master')
    git(hub.co, 'checkout', '-q', 'pr-10')
    hub.merge_and_delete('pr-10')
    before = head(hub.co)

    ok, msg = update._run_git_update(str(hub.co))

    assert not ok
    assert 'its master has commits that are not on GitHub' in msg
    assert branch(hub.co) == 'pr-10' and head(hub.co) == before
    assert git(hub.co, 'rev-parse', 'master') == mine


def tail12(msg):
    """What the Update failed box shows: the message's last 12 lines."""
    return '\n'.join(msg.splitlines()[-12:])


def test_diverged_live_branch_is_refused_and_fits_the_dialog(hub):
    hub.push_branch('dev-y', 'v1')
    hub.checkout_branch('dev-y')
    mine = commit(hub.co, 'd.txt', 'local work')
    git(hub.dev, 'checkout', '-q', 'dev-y')
    commit(hub.dev, 'b.txt', 'remote work')
    git(hub.dev, 'push', '-q', 'origin', 'dev-y')

    ok, msg = update._run_git_update(str(hub.co))

    assert not ok
    assert branch(hub.co) == 'dev-y' and head(hub.co) == mine
    assert git(hub.co, 'status', '--porcelain') == ''
    assert "Could not update branch 'dev-y'" in tail12(msg)


def test_long_git_output_keeps_the_plain_sentence_in_the_dialog(hub,
                                                                monkeypatch):
    real = update._git

    def chatty(root, *args, timeout=60):
        out = real(root, *args, timeout=timeout)
        if args[:1] == ('fetch',):
            out = subprocess.CompletedProcess(
                out.args, 1, '', '\n'.join(f'hint: line {i}' for i in range(40)))
        return out
    monkeypatch.setattr(update, '_git', chatty)

    ok, msg = update._run_git_update(str(hub.co))

    assert not ok
    assert 'Could not download the update from GitHub' in tail12(msg)


def test_master_open_in_another_worktree_is_not_moved(hub, tmp_path):
    hub.push_branch('pr-11', 'feature')
    hub.checkout_branch('pr-11')
    wt = tmp_path / 'other'
    git(hub.co, 'worktree', 'add', '-q', str(wt), 'master')
    wt_before = head(wt)
    hub.merge_and_delete('pr-11')
    before = head(hub.co)

    ok, msg = update._run_git_update(str(hub.co))

    assert not ok
    assert 'open in another folder' in msg
    assert head(wt) == wt_before and branch(wt) == 'master'
    assert git(wt, 'status', '--porcelain') == ''
    assert branch(hub.co) == 'pr-11' and head(hub.co) == before


def test_detached_unmerged_work_is_told_to_save_it_first(hub):
    git(hub.co, 'checkout', '-q', '--detach')
    mine = commit(hub.co, 'd.txt', 'detached work')

    ok, msg = update._run_git_update(str(hub.co))

    assert not ok
    assert msg.index('git switch -c') < msg.index('git checkout master')
    assert head(hub.co) == mine


def test_unfinished_merge_is_left_alone(hub):
    hub.push_branch('pr-12', 'feature')
    hub.checkout_branch('pr-12')
    hub.merge_and_delete('pr-12')
    git(hub.co, 'fetch', '-q', 'origin')
    git(hub.co, 'merge', '-q', '--no-ff', '--no-commit', 'origin/master')
    before = head(hub.co)

    ok, msg = update._run_git_update(str(hub.co))

    assert not ok
    assert msg.startswith('A git merge is not finished')
    assert branch(hub.co) == 'pr-12' and head(hub.co) == before


def test_rebase_stopped_on_a_merged_branch_is_left_alone(hub, monkeypatch):
    hub.push_branch('pr-13', 'feature')
    hub.checkout_branch('pr-13')
    hub.merge_and_delete('pr-13')
    monkeypatch.setenv('GIT_SEQUENCE_EDITOR', "sed -i '1i break'")
    git(hub.co, 'rebase', '-q', '-i', 'HEAD~1')

    ok, msg = update._run_git_update(str(hub.co))

    assert not ok
    assert msg.startswith('A git rebase is not finished')


def test_status_failure_counts_as_changes(hub, monkeypatch):
    hub.push_branch('pr-14', 'feature')
    hub.checkout_branch('pr-14')
    hub.merge_and_delete('pr-14')
    real = update._git

    def no_status(root, *args, timeout=60):
        if args[:1] == ('status',):
            return subprocess.CompletedProcess(args, 128, '', 'fatal')
        return real(root, *args, timeout=timeout)
    monkeypatch.setattr(update, '_git', no_status)

    ok, msg = update._run_git_update(str(hub.co))

    assert not ok and 'Update did not switch' in msg
    assert branch(hub.co) == 'pr-14'


def test_main_branch_named_main_without_origin_head(tmp_path):
    hub = Hub(tmp_path, main='main')
    hub.push_branch('pr-15', 'feature')
    hub.checkout_branch('pr-15')
    git(hub.co, 'remote', 'set-head', 'origin', '-d')
    tip = hub.merge_and_delete('pr-15')

    ok, msg = update._run_git_update(str(hub.co))

    assert ok, msg
    assert branch(hub.co) == 'main' and head(hub.co) == tip


def test_origin_head_names_the_main_branch(tmp_path):
    # GitHub's default is 'main' although an old 'master' still exists.
    hub = Hub(tmp_path, main='main')
    git(hub.dev, 'push', '-q', 'origin', 'main:master')
    hub.push_branch('pr-16', 'feature')
    hub.checkout_branch('pr-16')
    tip = hub.merge_and_delete('pr-16')
    git(hub.co, 'remote', 'set-head', 'origin', 'main')

    ok, msg = update._run_git_update(str(hub.co))

    assert ok, msg
    assert branch(hub.co) == 'main' and head(hub.co) == tip
