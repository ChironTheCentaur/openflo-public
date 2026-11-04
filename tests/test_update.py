"""Tests for the update-check core (openflo.update) — pure logic only;
the network fetch and git/pip subprocess calls are monkeypatched."""
from __future__ import annotations

import sys

from openflo import update


def test_parse_release_tag():
    assert update.parse_release_tag({'tag_name': 'v1.2.0'}) == '1.2.0'
    assert update.parse_release_tag({'tag_name': '2.0'}) == '2.0'
    assert update.parse_release_tag({}) is None
    assert update.parse_release_tag(None) is None


def test_is_newer():
    assert update.is_newer('1.2.0', '1.1.0')
    assert update.is_newer('1.1.1', '1.1.0')
    assert not update.is_newer('1.1.0', '1.1.0')
    assert not update.is_newer('1.0.0', '1.1.0')
    assert not update.is_newer(None, '1.1.0')      # garbage never nags
    assert not update.is_newer('1.1.0', None)


def test_a_beta_is_offered_only_to_someone_on_a_pre_release():
    # A final release is never told to update to a beta ...
    assert not update.is_newer('2.7.0b1', '2.6.2')
    # ... but a beta tester is told about the next beta and the final.
    assert update.is_newer('2.7.0b2', '2.7.0b1')
    assert update.is_newer('2.7.0', '2.7.0b1')
    assert not update.is_newer('2.7.0b1', '2.7.0')


def test_fallback_order_puts_a_beta_below_its_release(monkeypatch):
    # Without packaging. Reading every digit made 2.7.0b1 newer than 2.7.0.
    import builtins
    real_import = builtins.__import__

    def no_packaging(name, *a, **k):
        if name.startswith('packaging'):
            raise ImportError(name)
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, '__import__', no_packaging)
    assert update._fallback_key('2.7.0b1') < update._fallback_key('2.7.0')
    assert update.is_newer('2.7.0', '2.7.0b1')
    assert not update.is_newer('2.7.0b1', '2.7.0')
    assert not update.is_newer('2.7.0b1', '2.6.2')
    assert update.is_newer('1.10.0', '1.9.0')
    assert update.latest_tag([{'name': 'v2.7.0b1'}, {'name': 'v2.7.0'},
                              {'name': 'junk'}]) == '2.7.0'


def test_check_for_update_available(monkeypatch):
    monkeypatch.setattr(update, 'fetch_latest_release',
                        lambda **k: {'tag_name': 'v9.9.9',
                                     'html_url': 'https://example/rel'})
    monkeypatch.setattr(update, 'current_version', lambda: '1.1.0')
    res = update.check_for_update()
    assert res['available'] is True
    assert res['latest'] == '9.9.9' and res['current'] == '1.1.0'
    assert res['url'] == 'https://example/rel'


def test_check_for_update_up_to_date(monkeypatch):
    monkeypatch.setattr(update, 'fetch_latest_release',
                        lambda **k: {'tag_name': 'v1.1.0'})
    monkeypatch.setattr(update, 'current_version', lambda: '1.1.0')
    assert update.check_for_update()['available'] is False


def test_check_for_update_offline_returns_none(monkeypatch):
    monkeypatch.setattr(update, 'fetch_latest_release', lambda **k: None)
    monkeypatch.setattr(update, 'fetch_latest_tag', lambda **k: None)
    assert update.check_for_update() is None      # None = couldn't check


def test_latest_tag_picks_highest():
    tags = [{'name': 'v1.2.0'}, {'name': '1.10.0'}, {'name': 'v1.9.0'},
            {'name': 'not-a-version'}]
    assert update.latest_tag(tags) == '1.10.0'    # semver, not lexical
    assert update.latest_tag([]) is None
    assert update.latest_tag(None) is None


def test_check_for_update_falls_back_to_tags(monkeypatch):
    """No formal Release (force-pushed mirror) → use the newest tag instead."""
    monkeypatch.setattr(update, 'fetch_latest_release', lambda **k: None)
    monkeypatch.setattr(update, 'fetch_latest_tag', lambda **k: '1.5.0')
    monkeypatch.setattr(update, 'current_version', lambda: '1.4.2')
    res = update.check_for_update()
    assert res['available'] is True and res['latest'] == '1.5.0'
    assert res['url'].endswith('/tags')


def test_update_command_pip(monkeypatch):
    cmd = update.update_command(kind='pip')
    assert cmd[:4] == [sys.executable, '-m', 'pip', 'install']
    assert cmd[-1].startswith('git+https://github.com/') and cmd[-1].endswith('.git')
    assert '--upgrade' in cmd


def test_update_command_git(monkeypatch):
    monkeypatch.setattr(update, '_package_git_root', lambda: '/some/repo')
    cmd = update.update_command(kind='git')
    assert cmd[0] == 'git' and 'pull' in cmd and '/some/repo' in cmd


def test_detect_install_kind(monkeypatch):
    monkeypatch.setattr(update, '_package_git_root', lambda: '/repo')
    assert update.detect_install_kind() == 'git'
    monkeypatch.setattr(update, '_package_git_root', lambda: None)
    assert update.detect_install_kind() == 'pip'
