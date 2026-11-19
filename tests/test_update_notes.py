"""Update notice for every install kind, its release notes, and What's new.

* An install made by the Windows installer (its marker file in sys.prefix)
  is told to run the new installer; pip is never run there.
* The update offer shows the release's own notes as short plain text, or,
  for a version tagged without a Release, says so and links the changelog.
* After an upgrade, the first start shows What's new from the CHANGELOG
  shipped in the package; a fresh install records its version silently.

Every input is generated here: fake release JSON, a fake CHANGELOG, a
temporary sys.prefix. Nothing touches the network.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import types
from pathlib import Path

import pytest

from openflo import editor_update, prefs, update, whats_new
from openflo.editor_update import UpdateMixin
from tests.conftest import gui_unavailable

ROOT = Path(__file__).resolve().parents[1]

FAKE_CHANGELOG = """# Changelog

Intro text.

## [Unreleased]

### Added
- Something not released yet.

## [9.2.0] - 2030-02-01

### Added
- **Feature two.** It does a second thing.

### Fixed
- A fix in two.

## [9.1.0] - 2030-01-01

### Added
- Feature one.

[9.1.0]: https://example.invalid/compare/9.0...9.1
"""


def _marker(prefix, data):
    path = Path(prefix) / update.INSTALL_MARKER
    path.write_text(data if isinstance(data, str) else json.dumps(data),
                    encoding='utf-8')
    return path


# ── install kind ────────────────────────────────────────────────────────────

def test_installer_marker_is_checked_first(tmp_path, monkeypatch):
    monkeypatch.setattr(update.sys, 'prefix', str(tmp_path))
    monkeypatch.setattr(update, '_package_git_root', lambda: '/some/checkout')
    assert update.detect_install_kind() == 'git'          # no marker yet
    _marker(tmp_path, {'kind': 'installer', 'version': '9.1.0'})
    assert update.detect_install_kind() == 'installer'
    monkeypatch.setattr(update, '_package_git_root', lambda: None)
    assert update.detect_install_kind() == 'installer'


@pytest.mark.parametrize('content', [
    '{"kind": "portable", "version": "9.1.0"}', 'not json', '["installer"]'])
def test_a_marker_that_is_not_the_installers_is_ignored(tmp_path, monkeypatch,
                                                        content):
    monkeypatch.setattr(update.sys, 'prefix', str(tmp_path))
    monkeypatch.setattr(update, '_package_git_root', lambda: None)
    _marker(tmp_path, content)
    assert update.installer_marker() is None
    assert update.detect_install_kind() == 'pip'


def test_installer_marker_with_a_byte_order_mark(tmp_path):
    path = Path(tmp_path) / update.INSTALL_MARKER
    path.write_text(json.dumps({'kind': 'installer', 'version': '9.2.0'}),
                    encoding='utf-8-sig')
    assert path.read_bytes().startswith(b'\xef\xbb\xbf')
    assert update.installer_marker(str(tmp_path)) == {
        'kind': 'installer', 'version': '9.2.0'}


def test_installer_marker_reads_the_given_prefix(tmp_path):
    _marker(tmp_path, {'kind': 'installer', 'version': '9.1.0'})
    assert update.installer_marker(str(tmp_path))['version'] == '9.1.0'
    assert update.installer_marker(str(tmp_path / 'nowhere')) is None


def test_an_installer_install_never_runs_pip(monkeypatch):
    def no_subprocess(*_a, **_k):
        raise AssertionError('no command may run for an installer install')
    monkeypatch.setattr(update.subprocess, 'run', no_subprocess)
    ok, msg = update.run_update(kind='installer')
    assert ok is False and msg == update.INSTALLER_ADVICE
    assert 'installer' in msg and 'pip' not in msg
    with pytest.raises(ValueError):
        update.update_command(kind='installer')


def test_run_update_detects_an_installer_install(tmp_path, monkeypatch):
    monkeypatch.setattr(update.sys, 'prefix', str(tmp_path))
    _marker(tmp_path, {'kind': 'installer', 'version': '9.1.0'})
    monkeypatch.setattr(update.subprocess, 'run', lambda *a, **k: (
        _ for _ in ()).throw(AssertionError('ran a command')))
    assert update.run_update() == (False, update.INSTALLER_ADVICE)


# ── release notes ───────────────────────────────────────────────────────────

def test_plain_notes_drops_markdown():
    body = ("## What's changed\r\n\r\n\r\n* **Bold** item with "
            "[a link](https://example.invalid/x) and `code`\r\n"
            "<!-- hidden -->\r\n+ second\r\n\r\n\r\n### Fixed\r\n- a fix")
    assert update.plain_notes(body) == (
        "What's changed\n\n- Bold item with a link and code\n- second\n\n"
        "Fixed\n- a fix")
    assert update.plain_notes(None) == ''
    assert update.plain_notes('') == ''


def test_plain_notes_keeps_less_than_and_greater_than_in_text():
    body = ("FL1-A < 1000 and FL2-A > 50 are kept.\n"
            "Ratio 1<2 and 3>2, arrows -> and <- too.\n"
            "Tags go: a<br>b <details><summary>More</summary></details> "
            "<img src=x.png alt='y'/> <!-- note -->end")
    assert update.plain_notes(body) == (
        "FL1-A < 1000 and FL2-A > 50 are kept.\n"
        "Ratio 1<2 and 3>2, arrows -> and <- too.\n"
        "Tags go: ab More  end")


def test_plain_notes_keeps_an_autolinks_address():
    body = ("See <https://example.invalid/notes?a=1&b=2> or "
            "<ftp://example.invalid/x>.\n- Also <HTTP://example.invalid>")
    assert update.plain_notes(body) == (
        "See https://example.invalid/notes?a=1&b=2 or "
        "ftp://example.invalid/x.\n- Also HTTP://example.invalid")


def test_plain_notes_without_a_limit_is_never_cut():
    body = '\n'.join(f'- change number {i:03d}' for i in range(200))
    out = update.plain_notes(body, limit=None)
    assert out == body
    assert len(out) > update.NOTES_LIMIT


def test_plain_notes_cuts_long_notes_at_a_line_end():
    body = '\n'.join(f'- change number {i:03d}' for i in range(200))
    # 310 falls mid-line (each line is 19 characters and a newline).
    out = update.plain_notes(body, limit=310)
    assert out.endswith('\n…')
    kept = out[:-2]
    assert len(kept) <= 310
    assert kept.splitlines()[-1].startswith('- change number ')
    assert len(kept.splitlines()[-1]) == len('- change number 000')
    short = update.plain_notes('- one\n- two', limit=300)
    assert short == '- one\n- two'


def test_check_for_update_carries_the_release_notes(monkeypatch):
    release = {'tag_name': 'v9.2.0', 'html_url': 'https://example.invalid/r',
               'body': '## Added\n- **Feature two.**'}
    monkeypatch.setattr(update, 'fetch_latest_release',
                        lambda repo=None, timeout=None: release)
    monkeypatch.setattr(update, 'fetch_latest_tag', lambda **k: (
        _ for _ in ()).throw(AssertionError('a Release needs no tag lookup')))
    monkeypatch.setattr(update, 'current_version', lambda: '9.1.0')
    res = update.check_for_update()
    assert res['available'] is True and res['has_release'] is True
    assert res['notes'] == 'Added\n- Feature two.'
    assert res['url'] == 'https://example.invalid/r'


def test_a_tag_only_version_has_no_release_notes(monkeypatch):
    monkeypatch.setattr(update, 'fetch_latest_release',
                        lambda repo=None, timeout=None: None)
    monkeypatch.setattr(update, 'fetch_latest_tag',
                        lambda repo=None, timeout=None: '9.2.0')
    monkeypatch.setattr(update, 'current_version', lambda: '9.1.0')
    res = update.check_for_update()
    assert res['has_release'] is False and res['notes'] == ''
    assert res['url'].endswith('/tags')


# ── the offer ───────────────────────────────────────────────────────────────

_RES = {'current': '9.1.0', 'latest': '9.2.0', 'available': True,
        'url': 'https://example.invalid/release', 'notes': 'Added\n- Two.',
        'has_release': True}


@pytest.mark.parametrize('kind', ['pip', 'git'])
def test_offer_for_pip_and_git_updates_and_shows_the_notes(kind):
    p = editor_update.offer_parts(_RES, kind)
    assert p['action'] == 'Update' and p['action_tip'] is None
    assert p['notes'] == 'Added\n- Two.'
    assert p['link'] == ('Full release notes', _RES['url'])
    assert ('git pull' in p['text']) is (kind == 'git')
    assert ('pip install' in p['text']) is (kind == 'pip')


def test_offer_for_an_installer_install_opens_the_release_page():
    p = editor_update.offer_parts(_RES, 'installer')
    assert p['action'] == 'Open release page'
    assert 'run it' in p['text'] and 'in place' in p['text']
    assert 'pip' not in p['text'] and 'git' not in p['text']
    assert 'installer' in p['action_tip']


def test_offer_without_a_release_says_so_and_links_the_changelog():
    res = dict(_RES, has_release=False, notes='',
               url='https://example.invalid/tags')
    p = editor_update.offer_parts(res, 'pip')
    assert 'without a GitHub release' in p['notes']
    assert p['link'] == ('Changelog', update.CHANGELOG_URL)
    empty = editor_update.offer_parts(dict(_RES, notes=''), 'pip')
    assert empty['notes'] == 'This release has no notes.'


class _Ed:
    def __init__(self):
        self.status = []
        self.status_var = types.SimpleNamespace(set=self.status.append)
        self.scheduled = []
        self.whats_new = 0

    def after(self, ms, fn):
        self.scheduled.append((ms, fn))

    def _show_whats_new(self):
        self.whats_new += 1
        return 'window'

    def _after_resume_step(self, fn):
        return fn()                     # the resume step is over

    _offer_update = UpdateMixin._offer_update
    _set_update_status = UpdateMixin._set_update_status
    _schedule_startup_checks = UpdateMixin._schedule_startup_checks
    _whats_new_due = UpdateMixin._whats_new_due


@pytest.fixture
def offer(monkeypatch):
    seen = types.SimpleNamespace(asked=[], answers=[], opened=[], ran=[])

    def ask(*a, **k):
        seen.asked.append((a, k))
        return seen.answers.pop(0)
    monkeypatch.setattr(editor_update, 'ask_update_or_cancel', ask)
    monkeypatch.setattr(editor_update, '_open_url', seen.opened.append)
    monkeypatch.setattr(update, 'run_update',
                        lambda kind=None: (seen.ran.append(kind), (True, ''))[1])
    return seen


def test_installer_offer_opens_the_page_and_never_runs_an_update(
        monkeypatch, offer):
    monkeypatch.setattr(update, 'detect_install_kind', lambda: 'installer')
    ed = _Ed()
    ed._run_update = lambda kind: offer.ran.append(kind)
    offer.answers.append(False)
    ed._offer_update(_RES)
    assert offer.opened == [] and offer.ran == []
    offer.answers.append(True)
    ed._offer_update(_RES)
    assert offer.opened == [_RES['url']] and offer.ran == []
    assert ed.status[-1] == ('Opened the release page: run the new '
                             'installer to update.')
    _a, kw = offer.asked[-1]
    assert kw['action'] == 'Open release page' and kw['notes'] == _RES['notes']


def test_pip_offer_passes_the_notes_and_still_runs_pip(monkeypatch, offer):
    monkeypatch.setattr(update, 'detect_install_kind', lambda: 'pip')
    ed = _Ed()
    ed._run_update = lambda kind: offer.ran.append(kind)
    offer.answers.append(True)
    ed._offer_update(_RES)
    assert offer.ran == ['pip'] and offer.opened == []
    _a, kw = offer.asked[-1]
    assert kw['notes'] == _RES['notes']
    assert kw['link'] == ('Full release notes', _RES['url'])


# ── startup ─────────────────────────────────────────────────────────────────

def test_no_startup_window_under_pytest():
    assert editor_update.startup_popups_allowed() is False
    assert editor_update.startup_popups_allowed({}) is True
    assert editor_update.startup_popups_allowed(
        {'PYTEST_CURRENT_TEST': 'x'}) is False
    ed = _Ed()
    ed._schedule_startup_checks()
    assert ed.scheduled == []


@pytest.mark.parametrize('stored, checks', [
    ({}, True), ({'check_updates_on_start': True}, True),
    ({'check_updates_on_start': False}, False)])
@pytest.mark.parametrize('due', [True, False])
def test_startup_check_can_be_turned_off(monkeypatch, stored, checks, due):
    monkeypatch.setattr(editor_update, 'startup_popups_allowed', lambda: True)
    monkeypatch.setattr(prefs, 'read_prefs', lambda: dict(stored))
    ed = _Ed()
    ed._whats_new_due = lambda: due
    ed._show_whats_new = lambda: 'whats-new'
    ed._check_for_updates = lambda silent=False: ('check', silent)
    ed._schedule_startup_checks()
    got = [fn() for _ms, fn in ed.scheduled]
    want = (['whats-new'] if due else []) + (
        [('check', True)] if checks else [])
    assert got == want
    # Both wait for the window: neither runs during startup itself.
    assert all(ms > 0 for ms, _fn in ed.scheduled)


@pytest.fixture
def own_prefs(monkeypatch, tmp_path):
    """A preferences file of this test's own (the test home is shared)."""
    path = str(tmp_path / 'prefs.json')
    monkeypatch.setattr(prefs, '_prefs_path', lambda: path)
    return path


def test_whats_new_is_decided_before_the_welcome_saves_its_flag(
        monkeypatch, own_prefs):
    """A fresh install: the decision is taken when the editor starts, so the
    welcome's flag (or any other preference saved later in this start) never
    makes the next check think an earlier OpenFlo ran."""
    monkeypatch.setattr(editor_update, 'startup_popups_allowed', lambda: True)
    monkeypatch.setattr(update, 'current_version', lambda: '9.2.0')
    ed = _Ed()
    ed._check_for_updates = lambda silent=False: None
    ed._schedule_startup_checks()
    assert prefs.read_prefs()[whats_new.LAST_SEEN_KEY] == '9.2.0'
    prefs.write_pref('geometry', '900x600+10+10')   # saved later this start
    assert all(fn != ed._show_whats_new for _ms, fn in ed.scheduled)
    assert ed._whats_new_due() is False


def test_prefs_default_checks_at_start(monkeypatch):
    monkeypatch.setattr(prefs, 'read_prefs', dict)
    assert prefs.load_check_updates_on_start() is True
    monkeypatch.setattr(prefs, 'read_prefs',
                        lambda: {'check_updates_on_start': False})
    assert prefs.load_check_updates_on_start() is False


class _FakeRegistry:
    """A generated stand-in for ``winreg``: ``values`` maps
    (key path, value name) to (value, type)."""
    HKEY_CURRENT_USER = 'HKCU'
    REG_DWORD, REG_SZ = 4, 1

    def __init__(self, values):
        self.values = values

    def OpenKey(self, hive, path):
        if hive != 'HKCU' or not any(k == path for k, _n in self.values):
            raise FileNotFoundError(path)
        return contextlib.nullcontext(path)

    def QueryValueEx(self, key, name):
        try:
            return self.values[(key, name)]
        except KeyError:
            raise FileNotFoundError(name) from None


def _installer_answer(value, kind=_FakeRegistry.REG_DWORD):
    return _FakeRegistry({('Software\\OpenFlo', 'CheckUpdatesOnStart'): (value, kind)})


def test_installer_answer_is_read_from_its_registry_value():
    assert prefs.installer_check_updates_default(_installer_answer(0)) is False
    assert prefs.installer_check_updates_default(_installer_answer(1)) is True
    # no key, another value only, or the wrong type: no answer
    assert prefs.installer_check_updates_default(_FakeRegistry({})) is None
    assert prefs.installer_check_updates_default(_FakeRegistry(
        {('Software\\OpenFlo', 'InstallDir'): ('C:\\x', 1)})) is None
    assert prefs.installer_check_updates_default(
        _installer_answer('0', _FakeRegistry.REG_SZ)) is None


def test_installer_answer_is_none_off_windows(monkeypatch):
    monkeypatch.setattr(prefs.sys, 'platform', 'linux')
    assert prefs.installer_check_updates_default() is None


def test_installer_answer_is_the_default_until_preferences_set_it(monkeypatch):
    monkeypatch.setattr(prefs, 'read_prefs', dict)
    assert prefs.load_check_updates_on_start(_installer_answer(0)) is False
    assert prefs.load_check_updates_on_start(_installer_answer(1)) is True
    assert prefs.load_check_updates_on_start(_FakeRegistry({})) is True
    # a choice saved in Preferences wins over the installer's, both ways
    monkeypatch.setattr(prefs, 'read_prefs', lambda: {'check_updates_on_start': True})
    assert prefs.load_check_updates_on_start(_installer_answer(0)) is True
    monkeypatch.setattr(prefs, 'read_prefs', lambda: {'check_updates_on_start': False})
    assert prefs.load_check_updates_on_start(_installer_answer(1)) is False


def test_installer_and_app_name_the_same_registry_value():
    nsi = (Path(__file__).resolve().parents[1] / 'packaging' / 'windows'
           / 'openflo.nsi').read_text(encoding='utf-8')
    assert f'!define APP_KEY "{prefs.INSTALLER_KEY}"' in nsi
    assert f'!define UPDATES_VALUE "{prefs.INSTALLER_UPDATES_VALUE}"' in nsi


def _store():
    data = {}
    return data, (lambda: dict(data)), data.__setitem__


_OLD = {'geometry': '900x600+10+10', 'theme': 'dark'}   # 3.0.0's own prefs


@pytest.mark.parametrize('stored, current, show, recorded', [
    ({}, '9.2.0', False, '9.2.0'),                          # fresh install
    (None, '9.2.0', False, '9.2.0'),                        # no file at all
    ({whats_new.LAST_SEEN_KEY: ''}, '9.2.0', False, '9.2.0'),
    ({'welcome_done': True}, '9.2.0', False, '9.2.0'),      # this start's own
    ({'welcome_done': True, whats_new.LAST_SEEN_KEY: '  '}, '9.2.0', False,
     '9.2.0'),
    (['damaged'], '9.2.0', False, '9.2.0'),                 # not a dict
    ('damaged', '9.2.0', False, '9.2.0'),
    (_OLD, '9.2.0', True, '9.2.0'),                         # 3.0.0 upgrade
    ({'geometry': '900x600'}, '9.2.0', True, '9.2.0'),
    ({**_OLD, 'welcome_done': True}, '9.2.0', True, '9.2.0'),
    ({**_OLD, whats_new.LAST_SEEN_KEY: ''}, '9.2.0', True, '9.2.0'),
    ({**_OLD, whats_new.LAST_SEEN_KEY: '  '}, '9.2.0', True, '9.2.0'),
    ({whats_new.LAST_SEEN_KEY: '9.1.0'}, '9.2.0', True, '9.2.0'),   # older
    ({**_OLD, whats_new.LAST_SEEN_KEY: '9.1.0'}, '9.2.0b1', True,
     '9.2.0b1'),                                            # a beta counts
    ({**_OLD, whats_new.LAST_SEEN_KEY: '9.2.0'}, '9.2.0', False,
     '9.2.0'),                                              # same version
    ({whats_new.LAST_SEEN_KEY: 'v9.2.0'}, '9.2.0', False, 'v9.2.0'),
    ({**_OLD, whats_new.LAST_SEEN_KEY: '9.3.0'}, '9.2.0', False,
     '9.2.0'),                                              # downgrade
])
def test_first_start_of_new_version(stored, current, show, recorded):
    state = {'file': stored}

    def read():
        # read_prefs returns whatever JSON the file holds ({} when missing).
        return {} if state['file'] is None else state['file']

    def write(key, value):
        # write_pref starts a damaged (non-dict) file afresh.
        base = state['file'] if isinstance(state['file'], dict) else {}
        state['file'] = {**base, key: value}
    assert whats_new.first_start_of_new_version(current, read, write) is show
    assert state['file'][whats_new.LAST_SEEN_KEY] == recorded
    # Only once: the next start of the same version shows nothing.
    assert whats_new.first_start_of_new_version(current, read, write) is False


@pytest.mark.parametrize('stored, before', [
    (None, False), ({}, False), (['x'], False), ({'welcome_done': True}, False),
    ({whats_new.LAST_SEEN_KEY: '9.1.0'}, False),
    ({'welcome_done': True, whats_new.LAST_SEEN_KEY: ''}, False),
    ({'geometry': '900x600'}, True), ({'recent_sessions': []}, True)])
def test_ran_before(stored, before):
    assert whats_new.ran_before(stored) is before


def test_same_version_does_not_write():
    data, read, _write = _store()
    data[whats_new.LAST_SEEN_KEY] = '9.2.0'
    wrote = []
    assert whats_new.first_start_of_new_version(
        '9.2.0', read, lambda k, v: wrote.append((k, v))) is False
    assert wrote == []


@pytest.mark.parametrize('stored', [
    {whats_new.LAST_SEEN_KEY: '9.1.0'},                 # recorded older
    {'geometry': '900x600+10+10'},                      # 3.0.0: no key
])
def test_startup_shows_whats_new_once_after_an_upgrade(monkeypatch, stored,
                                                      own_prefs):
    monkeypatch.setattr(update, 'current_version', lambda: '9.2.0')
    for k, v in stored.items():
        prefs.write_pref(k, v)
    ed = _Ed()
    assert ed._whats_new_due() is True
    assert ed._whats_new_due() is False
    assert prefs.read_prefs()[whats_new.LAST_SEEN_KEY] == '9.2.0'


def test_a_damaged_prefs_file_starts_silently(monkeypatch):
    monkeypatch.setattr(update, 'current_version', lambda: '9.2.0')
    monkeypatch.setattr(prefs, 'read_prefs', lambda: ['not', 'a', 'dict'])
    wrote = []
    monkeypatch.setattr(prefs, 'write_pref', lambda k, v: wrote.append((k, v)))
    assert _Ed()._whats_new_due() is False
    assert wrote == [(whats_new.LAST_SEEN_KEY, '9.2.0')]


def test_a_prefs_failure_never_stops_the_start(monkeypatch):
    monkeypatch.setattr(update, 'current_version', lambda: '9.2.0')

    def broken(*_a):
        raise OSError('prefs unreadable')
    monkeypatch.setattr(whats_new, 'first_start_of_new_version', broken)
    ed = _Ed()
    assert ed._whats_new_due() is False and ed.whats_new == 0


# ── the changelog ───────────────────────────────────────────────────────────

def test_changelog_section_is_this_versions_notes():
    two = whats_new.changelog_section(FAKE_CHANGELOG, '9.2.0')
    assert two.startswith('### Added\n- **Feature two.**')
    assert two.endswith('- A fix in two.')
    assert 'Feature one' not in two and 'not released' not in two
    one = whats_new.changelog_section(FAKE_CHANGELOG, 'v9.1.0')
    assert one == '### Added\n- Feature one.'          # link refs dropped
    assert whats_new.changelog_section(FAKE_CHANGELOG, '9.0.0') is None
    assert whats_new.changelog_section(None, '9.1.0') is None


def test_a_v_prefix_matches_without_packaging(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_packaging(name, *a, **k):
        if name.startswith('packaging'):
            raise ImportError(name)
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, '__import__', no_packaging)
    assert whats_new.changelog_section(FAKE_CHANGELOG, 'v9.1.0') == (
        '### Added\n- Feature one.')
    assert whats_new.changelog_section(FAKE_CHANGELOG, '9.1') is None


def test_whats_new_text_falls_back_to_a_sentence():
    assert whats_new.whats_new_text('9.1.0', FAKE_CHANGELOG) == (
        'Added\n- Feature one.')
    msg = whats_new.whats_new_text('9.9.9', FAKE_CHANGELOG)
    assert msg.startswith('The changelog shipped with OpenFlo 9.9.9 has no '
                          'notes')


def test_whats_new_text_is_plain_text():
    assert whats_new.whats_new_text('9.2.0', FAKE_CHANGELOG) == (
        'Added\n- Feature two. It does a second thing.\n\n'
        'Fixed\n- A fix in two.')


def test_whats_new_text_is_never_cut():
    items = '\n'.join(f'- `FL{i % 9 + 1}-A` change {i:03d}' for i in range(150))
    log = f'## [9.3.0] - 2030-03-01\n\n### Changed\n{items}\n'
    out = whats_new.whats_new_text('9.3.0', log)
    assert len(out) > update.NOTES_LIMIT
    assert out == 'Changed\n' + items.replace('`', '')


def test_load_changelog_prefers_the_packaged_copy(tmp_path, monkeypatch):
    packaged, checkout = tmp_path / 'pkg.md', tmp_path / 'repo.md'
    monkeypatch.setattr(whats_new, '_changelog_paths',
                        lambda: [str(packaged), str(checkout)])
    assert whats_new.load_changelog() is None
    checkout.write_text('checkout copy', encoding='utf-8')
    assert whats_new.load_changelog() == 'checkout copy'
    packaged.write_text('packaged copy', encoding='utf-8')
    assert whats_new.load_changelog() == 'packaged copy'


def test_the_changelog_paths_are_the_package_then_the_checkout():
    pkg = os.path.dirname(os.path.abspath(whats_new.__file__))
    first, second = (os.path.normpath(p) for p in whats_new._changelog_paths())
    assert first == os.path.join(pkg, 'CHANGELOG.md')
    assert second == os.path.normpath(
        os.path.join(pkg, '..', '..', 'CHANGELOG.md'))


def _setup_module():
    spec = importlib.util.spec_from_file_location('openflo_setup_py',
                                                  ROOT / 'setup.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_build_copies_the_changelog_into_the_package(tmp_path):
    setup_py = _setup_module()
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'CHANGELOG.md').write_text(FAKE_CHANGELOG, encoding='utf-8')
    dest = setup_py.copy_changelog(str(tmp_path / 'build'), root=str(root))
    assert Path(dest) == tmp_path / 'build' / 'openflo' / 'CHANGELOG.md'
    assert Path(dest).read_text(encoding='utf-8') == FAKE_CHANGELOG
    with pytest.raises(FileNotFoundError, match="What's new needs it"):
        setup_py.copy_changelog(str(tmp_path / 'b2'), root=str(tmp_path))


def test_build_py_runs_the_copy_after_the_normal_build(monkeypatch, tmp_path):
    setup_py = _setup_module()
    calls = []
    monkeypatch.setattr(setup_py.build_py, 'run',
                        lambda self: calls.append('build_py'))
    monkeypatch.setattr(setup_py, 'copy_changelog',
                        lambda lib: calls.append(('copy', lib)))
    cmd = object.__new__(setup_py.BuildPyWithChangelog)
    cmd.build_lib = str(tmp_path)
    cmd.run()
    assert calls == ['build_py', ('copy', str(tmp_path))]


# ── the windows ─────────────────────────────────────────────────────────────

def _tk_root():
    import tkinter as tk
    try:
        root = tk.Tk()
    except Exception as e:                            # noqa: BLE001 - any Tk start failure means no GUI
        gui_unavailable(str(e))
    root.withdraw()
    return root


def _all(w):
    for c in w.winfo_children():
        yield c
        yield from _all(c)


def _toplevel(root):
    import tkinter as tk
    return next(w for w in root.winfo_children() if isinstance(w, tk.Toplevel))


def test_the_offer_shows_notes_and_a_link_that_leaves_it_open(monkeypatch):
    import tkinter as tk
    from tkinter import ttk
    opened = []
    monkeypatch.setattr(editor_update, '_open_url', opened.append)
    root = _tk_root()
    seen = {}
    try:
        def look():
            dlg = _toplevel(root)
            try:
                inspect(dlg)
            except Exception as e:                    # noqa: BLE001 - a broken dialog is closed so the test fails, not hangs
                seen['error'] = repr(e)
                if dlg.winfo_exists():
                    dlg.destroy()

        def inspect(dlg):
            texts = [w for w in _all(dlg) if isinstance(w, tk.Text)]
            seen['notes'] = texts[0].get('1.0', 'end-1c')
            seen['state'] = str(texts[0].cget('state'))
            by = {str(b.cget('text')): b for b in _all(dlg)
                  if isinstance(b, ttk.Button)}
            seen['buttons'] = sorted(by)
            by['Full release notes'].invoke()
            seen['still_open'] = dlg.winfo_exists()
            by['Open release page'].invoke()
        root.after(200, look)
        parts = editor_update.offer_parts(_RES, 'installer')
        answer = editor_update.ask_update_or_cancel(root, 't', **parts)
        assert 'error' not in seen, seen.get('error')
        assert answer is True
        assert seen['notes'] == _RES['notes'] and seen['state'] == 'disabled'
        assert seen['buttons'] == ['Cancel', 'Full release notes',
                                   'Open release page']
        assert seen['still_open'] and opened == [_RES['url']]
    finally:
        root.destroy()


def test_the_whats_new_window(monkeypatch):
    import tkinter as tk
    from tkinter import ttk
    opened = []
    monkeypatch.setattr(editor_update, '_open_url', opened.append)
    root = _tk_root()
    try:
        dlg = editor_update.show_whats_new(root, '9.2.0', 'notes body')
        assert dlg.title() == "What's new in OpenFlo 9.2.0"
        text = next(w for w in _all(dlg) if isinstance(w, tk.Text))
        assert text.get('1.0', 'end-1c') == 'notes body'
        by = {str(b.cget('text')): b for b in _all(dlg)
              if isinstance(b, ttk.Button)}
        assert sorted(by) == ['Close', 'Full changelog']
        by['Full changelog'].invoke()
        assert opened == [update.CHANGELOG_URL]
        by['Close'].invoke()
        assert not dlg.winfo_exists()
    finally:
        root.destroy()


def test_help_whats_new_shows_this_versions_section(monkeypatch):
    monkeypatch.setattr(update, 'current_version', lambda: '9.2.0')
    monkeypatch.setattr(whats_new, 'load_changelog', lambda: FAKE_CHANGELOG)
    shown = []
    monkeypatch.setattr(editor_update, 'show_whats_new',
                        lambda parent, v, text: shown.append((v, text)))
    UpdateMixin._show_whats_new(object())
    assert shown == [('9.2.0', 'Added\n- Feature two. It does a second '
                      'thing.\n\nFixed\n- A fix in two.')]


def test_the_changelog_url_is_the_public_mirrors():
    assert update.CHANGELOG_URL.startswith(update.PUBLIC_URL + '/')
    assert update.CHANGELOG_URL.endswith('/CHANGELOG.md')


@pytest.mark.parametrize('primary', [True, False])
def test_only_the_main_window_schedules_the_startup_checks(monkeypatch,
                                                           primary):
    import importlib
    gui = importlib.import_module('openflo.gui')
    calls = []
    monkeypatch.setattr(UpdateMixin, '_schedule_startup_checks',
                        lambda self: calls.append(self))
    monkeypatch.setattr(gui.messagebox, 'askyesno', lambda *a, **k: True)
    root = _tk_root()
    try:
        ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                      on_save=None, primary=primary)
        ed.withdraw()
        assert calls == ([ed] if primary else [])
    finally:
        root.destroy()


def test_changelog_link_is_the_one_the_package_metadata_names():
    """The app's changelog link and pyproject's [project.urls] Changelog
    name the same page, so they cannot drift to different branches."""
    import pathlib
    import tomllib
    root = pathlib.Path(__file__).resolve().parents[1]
    urls = tomllib.loads((root / 'pyproject.toml').read_text())['project']['urls']
    assert update.CHANGELOG_URL == urls['Changelog']
