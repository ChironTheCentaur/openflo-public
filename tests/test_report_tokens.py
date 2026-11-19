"""The error report's tokeniser (editor_lifecycle._tokenise_for_report),
which Help > Report a problem can now send: folder and file names are
replaced, spaces and all, whether or not the sample was loaded. Generated
names only; the home folder and user are stood in for.
"""
from __future__ import annotations

import json
import time

import pytest

from openflo import editor_lifecycle as lifecycle


@pytest.fixture
def tok(tmp_path, monkeypatch):
    keys = tmp_path / 'keys.json'
    monkeypatch.setattr(lifecycle, '_error_keymap_path', lambda: str(keys))
    home = tmp_path / 'Alex Example'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('USERPROFILE', str(home))
    import getpass
    import socket
    monkeypatch.setattr(getpass, 'getuser', lambda: 'zz-no-user')
    monkeypatch.setattr(socket, 'gethostname', lambda: 'zz-no-host')

    def run(text, extra=()):
        out = lifecycle._tokenise_for_report(text, extra)
        return out, json.loads(keys.read_text(encoding='utf-8'))
    run.home = str(home)
    return run


@pytest.mark.parametrize('text, secrets', [
    ("FileNotFoundError: [Errno 2] No such file or directory: "
     "'D:\\\\Lab Data\\\\Study Seven\\\\tube 3.fcs'", ['Lab Data', 'Study Seven', 'tube 3']),
    ('could not open D:\\Lab Data\\Study Seven\\tube 3.fcs', ['Lab Data', 'Study Seven', 'tube']),
    ("PermissionError: [Errno 13] Permission denied: '/srv/Lab Data/Study Seven/tube 3.fcs'",
     ['Lab Data', 'Study Seven', 'tube 3']),
    ('cannot read /srv/Lab Data/Study Seven/tube 3.fcs now', ['Lab Data', 'Study Seven', 'tube']),
    ('ValueError: tubeQ7_donor12.fcs: not an OpenFlo session sidecar', ['tubeQ7_donor12']),
    ("FileNotFoundError: 'tubeQ8.fcs'", ['tubeQ8']),
    ("FileNotFoundError: 'donor twelve.fcs'", ['donor', 'twelve']),
    ('workspace Study_Seven.wsp and table run4_counts.csv', ['Study_Seven', 'run4_counts']),
    ('share \\\\labserver\\Flow Group\\run 4\\tubeA.fcs gone', ['labserver', 'Flow Group', 'run 4', 'tubeA']),
    # Round 2: an unquoted name with spaces (as OpenFlo's own messages put
    # a file's basename), a path's last part with spaces, a relative path,
    # and figure / report files.
    ('ValueError: Donor Twelve day 4.flowsession: not an OpenFlo session sidecar', ['Donor', 'Twelve', 'day 4']),
    ('could not open D:\\LabQ\\Donor Twelve day 4.fcs now', ['LabQ', 'Donor', 'Twelve', 'day 4']),
    ('cannot read /srv/labq/Donor Twelve day 4.fcs', ['labq', 'Donor', 'Twelve', 'day 4']),
    ('could not read data/donorQ5/tubeQ10.fcs', ['donorQ5', 'tubeQ10']),
    ("OSError: 'notes_donorQ7.pdf'", ['donorQ7']),
    ('wrote fig_donorQ8.png and report_donorQ9.html', ['donorQ8', 'donorQ9']),
    # Each of these is caught by one rule alone: a quoted folder path with a
    # space and no file at its end; an unquoted folder path with a space; a
    # network share with no file; a quoted name no file-name rule reads.
    ("PermissionError: '/srv/Lab Data/Study Seven'", ['Lab Data', 'Seven']),
    ('cannot list /srv/Lab Data/donorQ3 now', ['Lab Data', 'donorQ3']),
    ('share \\\\labserver\\Flow Group\\run 4 is offline', ['labserver', 'Flow Group']),
    ("OSError: 'donor (twelve).fcs'", ['donor', 'twelve']),
])
def test_folder_and_file_names_are_replaced(tok, text, secrets):
    out, keys = tok(text)
    assert not [s for s in secrets if s in out], out
    for s in secrets:
        assert any(s in v for v in keys.values()), (s, keys)


def test_a_path_under_a_home_folder_with_a_space(tok):
    out, _keys = tok(f"OSError: '{tok.home}/Study Seven/tube 3.fcs'")
    assert 'Alex' not in out and 'Study Seven' not in out and 'tube 3' not in out


def test_the_message_around_a_path_is_kept_and_tokens_are_stable(tok):
    secret = 'C:\\Lab Data\\tubeA.fcs'
    out, _ = tok(f'load {secret} failed; retry {secret}')
    assert out == 'load <path:1> failed; retry <path:1>'
    assert tok(out)[0] == out          # tokenising again changes nothing


def test_code_files_and_line_numbers_stay_readable(tok):
    text = '  File "openflo/gui.py", line 12, in draw\nKeyError: \'FL2-A\''
    assert tok(text)[0] == text


@pytest.mark.parametrize('text, want', [
    # One token for a whole path, its last part spaces and all, and the
    # message after a file name kept.
    ('could not open D:\\LabQ\\Donor Twelve day 4.fcs now', 'could not open <path:1> now'),
    ('cannot read /srv/labq/Donor Twelve day 4.fcs', 'cannot read <path:1>'),
    ('ValueError: Donor Twelve day 4.flowsession: not an OpenFlo session sidecar',
     'ValueError: <file:1>: not an OpenFlo session sidecar'),
    # A name at the very start of the report: its first letter too.
    ('tubeQ1.fcs could not be read', '<file:1> could not be read'),
])
def test_one_token_per_name_and_the_message_kept(tok, text, want):
    assert tok(text)[0] == want


def test_a_very_long_line_is_left_out_and_still_quick(tok):
    head = "OSError: 'D:\\\\Lab Data\\\\tubeQ.fcs' "
    tail = ' donorQ99.fcs'
    line = head + 'x' * (64 * 1024) + tail
    t0 = time.monotonic()
    out, _ = tok(line + '\nnext line')
    assert time.monotonic() - t0 < 5.0
    assert 'Lab Data' not in out and 'tubeQ' not in out
    assert 'donorQ99' not in out                     # left out, never sent
    assert out == f'…(a line of {len(line)} characters left out)\nnext line'


def _long_line_with_cap_inside(name_text, keep):
    """A line over the cap whose first ``keep`` characters of ``name_text``
    fall before the cap."""
    pre = 'y' * 50 + ' ' + 'z' * (lifecycle._LINE_CAP - 52 - keep) + ' '
    return pre + name_text + ' ' + 'w' * 3000


@pytest.mark.parametrize('name_text, secret, extra', [
    ('Donor Qx 77 marrow', 'Qx', ['Donor Qx 77 marrow']),   # a loaded sample
    ('Donor Qx 77 tube.fcs', 'Qx', []),                      # a data file name
    ('/data/Lab Q/Donor Qx 77 tube.fcs', 'Qx', []),          # a POSIX path
    ('qperson@example.com', 'qperson', []),               # an email
])
def test_no_part_of_a_name_in_a_long_line_is_sent(tok, name_text, secret, extra):
    start = name_text.index(secret)
    for keep in range(start + len(secret), len(name_text)):
        out, _ = tok(_long_line_with_cap_inside(name_text, keep), extra)
        assert secret not in out, keep


def test_many_data_file_names_stay_quick(tok):
    # each name still runs back only to its own line's start
    out, _ = tok('x tubeQ1.fcs\nx tubeQ2.fcs')
    assert out.count('\n') == 1 and 'tubeQ' not in out
    t0 = time.monotonic()
    out, _ = tok('x tubeQ.fcs\n' * 50000)
    assert time.monotonic() - t0 < 3.0       # was ~12 s: each name copied the text before it
    assert 'tubeQ' not in out
