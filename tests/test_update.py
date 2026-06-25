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
    assert update.check_for_update() is None      # None = couldn't check


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
