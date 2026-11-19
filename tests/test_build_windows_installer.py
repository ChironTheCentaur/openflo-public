"""scripts/build_windows_installer.py: the pure parts of the Windows
installer build (requirement list to pip arguments, the .nsi rendering, the
install marker, the hash checks), on generated inputs.

Nothing here downloads, runs pip or runs makensis.
"""
import base64
import hashlib
import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    'build_windows_installer', ROOT / 'scripts' / 'build_windows_installer.py')
assert _spec and _spec.loader
bwi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bwi)

TEMPLATE = (ROOT / 'packaging' / 'windows' / 'openflo.nsi').read_text(encoding='utf-8')


def _values(**over):
    v = {'VERSION': '9.8.7', 'VI_VERSION': '9.8.7.0', 'GUI_MODULE': 'pkg.window',
         'PAYLOAD_DIR': '/work/payload', 'OUTFILE': '/work/Setup-9.8.7.exe',
         'MARKER_NAME': 'marker.json', 'LICENSE_FILE': '/work/LICENSE.txt',
         'ABOUT_URL': 'https://example.org/app', 'ESTIMATED_SIZE_KB': 1234,
         'PRIVACY_URL': 'https://example.org/privacy',
         'PUBLISHER': 'Example Maker LLC',
         'COPYRIGHT': 'Copyright (c) 2001 Example Maker LLC'}
    v.update(over)
    return v


# ---- requirements -> pip arguments ----

def test_read_requirements_drops_comments_blanks_and_notes():
    text = '# heading\n\nAaa==1.0\nbbb==2.0      # a note\n   \n# ccc==3\nddd>=4\n'
    assert bwi.read_requirements(text) == ['Aaa==1.0', 'bbb==2.0', 'ddd>=4']


def test_runtime_requirements_add_the_gui_extra_once():
    pyproject = {'project': {'optional-dependencies': {
        'gui': ['dnd_thing==0.1', 'Pinned.Already>=1'], 'other': ['zzz==1']}}}
    reqs = bwi.runtime_requirements('aaa==1\npinned-already==2\n', pyproject)
    # the pin wins for a project both name; extras other than gui stay out
    assert reqs == ['aaa==1', 'pinned-already==2', 'dnd_thing==0.1']


def test_runtime_requirements_of_this_repo_include_the_gui_extra():
    import tomllib
    pyproject = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    reqs = bwi.runtime_requirements(
        (ROOT / 'requirements.txt').read_text(encoding='utf-8'), pyproject)
    names = {bwi.requirement_name(r) for r in reqs}
    gui = pyproject['project']['optional-dependencies']['gui']
    assert {bwi.requirement_name(r) for r in gui} <= names


def test_gui_module_is_the_gui_scripts_entry():
    assert bwi.gui_module({'project': {'gui-scripts': {
        'openflo-gui': 'some.window:main'}}}) == 'some.window'
    with pytest.raises(bwi.BuildError, match='no \\[project.gui-scripts\\]'):
        bwi.gui_module({'project': {}})
    # `python -m module` runs main(); another function would not be reached
    with pytest.raises(bwi.BuildError, match='calls start'):
        bwi.gui_module({'project': {'gui-scripts': {'openflo-gui': 'm:start'}}})


def test_split_sdist_only_routes_the_listed_pure_packages():
    reqs = ['FlowIO==1.4.0', 'numpy==2.0', 'flow-io-not==1']
    assert bwi.split_sdist_only(reqs) == (['numpy==2.0', 'flow-io-not==1'],
                                          ['FlowIO==1.4.0'])


def test_pip_download_targets_windows_wheels_and_resolves_the_closure():
    args = bwi.pip_download_args('req.txt', 'wh')
    assert args[0] == 'download'
    joined = ' '.join(args)
    for need in ('--platform win_amd64', '--python-version 3.12', '--abi cp312',
                 '--implementation cp', '--only-binary=:all:', '-r req.txt',
                 '--dest wh', '--find-links wh'):
        assert need in joined
    # dependencies the pins leave out are fetched, so the closure is complete
    assert '--no-deps' not in args


def test_pip_install_uses_only_the_downloaded_wheels():
    args = bwi.pip_install_args('req.txt', 'wh', 'site')
    joined = ' '.join(args)
    for need in ('install', '--target site', '--no-index', '--find-links wh',
                 '--platform win_amd64', '--only-binary=:all:', '--no-compile',
                 '-r req.txt'):
        assert need in joined


def test_own_wheel_installs_without_dependencies():
    args = bwi.pip_install_own_wheel_args('own.whl', 'site')
    assert args[0] == 'install' and args[-1] == 'own.whl'
    for need in ('--no-deps', '--no-compile', '--target'):
        assert need in args
    # the Windows target, or pip refuses the wheel on a Linux build machine
    n = len(bwi.PLATFORM_ARGS)
    assert any(args[i:i + n] == list(bwi.PLATFORM_ARGS) for i in range(len(args)))


def test_sdist_wheel_builds_only_that_project_from_source():
    args = bwi.pip_sdist_wheel_args('FlowIO==1.4.0', 'wh')
    assert args[:2] == ['wheel', '--no-deps']
    assert args[args.index('--no-binary') + 1] == 'flowio'
    assert args[-1] == 'FlowIO==1.4.0'


@pytest.mark.parametrize('name, pure', [
    ('pkg-1.0-py3-none-any.whl', True),
    ('pkg-1.0-py2.py3-none-any.whl', True),
    ('pkg-1.0-cp312-cp312-win_amd64.whl', False),
    ('pkg-1.0-py3-none-win_amd64.whl', False),
    ('pkg-1.0-py2-none-any.whl', False),
])
def test_is_pure_wheel(name, pure):
    assert bwi.is_pure_wheel(name) is pure


# ---- marker, versions, names ----

def test_marker_content_is_the_update_notice_contract():
    assert json.loads(bwi.marker_content('4.5.6')) == {'kind': 'installer',
                                                      'version': '4.5.6'}
    assert bwi.MARKER_NAME == 'openflo-install.json'


@pytest.mark.parametrize('version, vi', [
    ('3.0.0', '3.0.0.0'), ('3.1', '3.1.0.0'), ('3.1.0b2', '3.1.0.0'),
    ('1.2.3.4.5', '1.2.3.4'), ('10.20.30.post1', '10.20.30.0'),
])
def test_vi_version(version, vi):
    assert bwi.vi_version(version) == vi


def test_vi_version_needs_a_number():
    with pytest.raises(bwi.BuildError):
        bwi.vi_version('dev')


def test_installer_name():
    assert bwi.installer_name('3.0.0') == 'OpenFlo-Setup-3.0.0.exe'


# ---- .nsi rendering ----

def test_render_replaces_every_placeholder():
    out = bwi.render_nsi('A @@X@@ b @@Y@@ @@X@@', {'X': 1, 'Y': 'two'})
    assert out == 'A 1 b two 1'


def test_render_refuses_a_missing_or_unused_value():
    with pytest.raises(bwi.BuildError, match='No value for Y'):
        bwi.render_nsi('@@X@@ @@Y@@', {'X': 1})
    with pytest.raises(bwi.BuildError, match='no Z placeholder'):
        bwi.render_nsi('@@X@@', {'X': 1, 'Z': 2})


@pytest.mark.parametrize('bad', ['a"b', 'C:\\$x', 'a`b', 'a\nb'])
def test_render_refuses_text_nsis_would_misread(bad):
    with pytest.raises(bwi.BuildError, match='NSIS cannot take'):
        bwi.render_nsi('"@@X@@"', {'X': bad})


def test_real_template_renders_completely():
    out = bwi.render_nsi(TEMPLATE, _values())
    assert '@@' not in out
    assert '!define APP_VERSION "9.8.7"' in out
    assert 'VIProductVersion "9.8.7.0"' in out
    assert 'OutFile "/work/Setup-9.8.7.exe"' in out
    assert '!define PAYLOAD "/work/payload"' in out
    assert '!define MARKER "marker.json"' in out
    assert 'MUI_PAGE_LICENSE "/work/LICENSE.txt"' in out
    assert '"EstimatedSize" 1234' in out
    assert '"-P -m ${GUI_MODULE}"' in out and '!define GUI_MODULE "pkg.window"' in out


def test_real_template_names_the_publisher_and_copyright():
    out = bwi.render_nsi(TEMPLATE, _values())
    assert '"Publisher" "Example Maker LLC"' in out
    assert 'VIAddVersionKey "CompanyName" "Example Maker LLC"' in out
    assert ('VIAddVersionKey "LegalCopyright" '
            '"Copyright (c) 2001 Example Maker LLC. MIT License."') in out


def test_publisher_values_come_from_openflo_publisher():
    from openflo.publisher import COPYRIGHT, PUBLISHER
    assert bwi.publisher_values() == {'PUBLISHER': PUBLISHER,
                                      'COPYRIGHT': COPYRIGHT}
    # Values NSIS can take in a quoted string (render_nsi refuses others).
    bwi.render_nsi('"@@PUBLISHER@@" "@@COPYRIGHT@@"', bwi.publisher_values())


def test_nsi_values_fill_the_real_template():
    values = bwi.nsi_values('9.8.7', 'pkg.window', '/work/payload', '/work/S.exe', 99)
    out = bwi.render_nsi(TEMPLATE, values)
    assert '!define PRIVACY_URL "https://www.getopenflo.com/privacy"' in out
    assert '!define APP_VERSION "9.8.7"' in out and '"EstimatedSize" 99' in out


def _privacy_page():
    return _code_lines(_block('Function PrivacyPage', 'FunctionEnd'))


def test_privacy_page_comes_before_anything_is_installed():
    pages = [ln for ln in _code_lines(TEMPLATE)
             if ln.startswith(('!insertmacro MUI_PAGE_', 'Page '))]
    privacy = pages.index('Page custom PrivacyPage PrivacyPageLeave')
    assert pages[privacy - 1].startswith('!insertmacro MUI_PAGE_LICENSE')
    assert privacy < pages.index('!insertmacro MUI_PAGE_INSTFILES')
    assert '!include "nsDialogs.nsh"' in TEMPLATE


def _policy_text():
    m = re.search(r'^!define POLICY_TEXT `([^`]*)`$', TEMPLATE, re.M)
    assert m, 'no POLICY_TEXT define'
    return m.group(1).replace('$\\r$\\n', '\n')


def test_privacy_page_shows_the_policy_desktop_text_read_only():
    text = _policy_text()
    # The policy's "Desktop application" paragraphs, word for word.
    for sentence in (
            'sends no analytics or usage data to us.',
            'Unless you turn it off in Preferences, it makes a quiet request to '
            'GitHub when it starts to see whether a newer release exists.',
            "GitHub receives your IP address and the request, under GitHub's own "
            'privacy statement. Nothing about your files is sent.',
            'Receiving is off until you turn it on.',
            'Reports you choose to send from the app work as described in Section 6'):
        assert sentence in text, sentence
    assert text.count('\n\n') == 3          # four paragraphs
    page = _privacy_page()
    box = [ln for ln in page if ln.endswith('`${POLICY_TEXT}`')]
    assert len(box) == 1
    assert box[0].startswith('nsDialogs::CreateControl EDIT ')
    for style in ('${ES_MULTILINE}', '${ES_READONLY}', '${WS_VSCROLL}'):
        assert style in box[0], style


def test_privacy_page_says_how_to_change_it_later():
    labels = ' '.join(ln for ln in _privacy_page() if ln.startswith('${NSD_CreateLabel}'))
    assert 'You can change this later in Edit > Preferences > Updates.' in labels
    assert ('Problem reports and Nearby send are used only when you choose them.'
            in labels)
    # The menu names the page gives are the app's own.
    menu = (ROOT / 'src' / 'openflo' / 'editor_menu.py').read_text(encoding='utf-8')
    for item in ('Preferences', 'Report a problem', 'Nearby send'):
        assert f'label="{item}\u2026"' in menu, item
    prefs = (ROOT / 'src' / 'openflo' / 'ui_preferences.py').read_text(encoding='utf-8')
    assert 'ttk.Label(frm, text="Updates"' in prefs
    assert "write_pref('check_updates_on_start'" in prefs


def test_privacy_page_has_the_update_check_option_ticked_by_default():
    page = _privacy_page()
    box = page.index('${NSD_CreateCheckbox} 0 79u 100% 12u '
                     '"Check for updates when OpenFlo starts"')
    assert page[box + 1:box + 4] == ['Pop $CheckUpdatesBox', '${If} $CheckUpdates == 1',
                                     '${NSD_Check} $CheckUpdatesBox']
    init = _code_lines(_block('Function .onInit', 'FunctionEnd'))
    assert init.index('StrCpy $CheckUpdates 1') < init.index('${GetParameters} $R0')
    # /NOUPDATECHECK unticks it for a silent install
    opt = init.index('${GetOptions} $R0 "/NOUPDATECHECK" $R1')
    assert init[opt - 1] == 'ClearErrors'
    assert init[opt + 1:opt + 3] == ['${IfNot} ${Errors}', 'StrCpy $CheckUpdates 0']


def test_privacy_page_answer_is_stored_for_the_app():
    leave = _code_lines(_block('Function PrivacyPageLeave', 'FunctionEnd'))
    assert leave == ['${NSD_GetState} $CheckUpdatesBox $0',
                     '${If} $0 == ${BST_CHECKED}', 'StrCpy $CheckUpdates 1',
                     '${Else}', 'StrCpy $CheckUpdates 0', '${EndIf}']
    out = bwi.render_nsi(TEMPLATE, _values())
    main = _code_lines(out.split('Section "OpenFlo (required)"', 1)[1]
                       .split('\nSectionEnd', 1)[0])
    assert 'WriteRegDWORD HKCU "${APP_KEY}" "${UPDATES_VALUE}" $CheckUpdates' in main
    assert '!define UPDATES_VALUE "CheckUpdatesOnStart"' in out
    # never in the user's settings folder: nothing writes a file there
    writes = [ln for ln in _code_lines(out)
              if ln.startswith(('FileOpen', 'WriteINIStr', 'CopyFiles', 'SetOutPath'))]
    assert not [ln for ln in writes if '.openflo' in ln]


def test_privacy_page_links_to_the_privacy_policy():
    page = _privacy_page()
    link = [i for i, ln in enumerate(page) if ln.startswith('${NSD_CreateLink}')]
    assert len(link) == 1 and page[link[0]].endswith('"${PRIVACY_URL}"')
    assert page[link[0] + 2] == '${NSD_OnClick} $0 OpenPrivacyPolicy'
    assert page[-1] == 'nsDialogs::Show'
    opener = _code_lines(_block('Function OpenPrivacyPolicy', 'FunctionEnd'))
    assert opener == ['Pop $0', 'ExecShell "open" "${PRIVACY_URL}"']
    out = bwi.render_nsi(TEMPLATE, _values())
    assert '!define PRIVACY_URL "https://example.org/privacy"' in out
    assert bwi.PRIVACY_URL == 'https://www.getopenflo.com/privacy'


def test_real_template_shortcuts_start_the_gui_module_windowless():
    starts = [ln for ln in TEMPLATE.splitlines()
              if ln.strip().startswith(('CreateShortcut', 'Exec '))]
    assert len(starts) == 3      # Start menu, Desktop, the finish page's Start now
    for line in starts:
        assert 'python\\pythonw.exe' in line and '-P -m ${GUI_MODULE}' in line, line


def test_real_template_starts_in_the_home_folder():
    # A start's working folder is the last SetOutPath before it: the user's
    # home, so the first file dialog does not open in the install folder.
    out_path = None
    starts = 0
    for line in TEMPLATE.splitlines():
        line = line.strip()
        if line.startswith('SetOutPath'):
            out_path = line
        elif line.startswith(('Function ', 'Section ')):
            out_path = None
        elif line.startswith(('CreateShortcut', 'Exec ')):
            starts += 1
            assert out_path == 'SetOutPath "$PROFILE"', line
    assert starts == 3


def test_real_template_installs_per_user():
    out = bwi.render_nsi(TEMPLATE, _values())
    assert 'RequestExecutionLevel user' in out
    assert 'InstallDir "$LOCALAPPDATA\\Programs\\OpenFlo"' in out
    # Add/Remove Programs entry in the user's hive, never the machine's
    assert 'HKLM' not in out
    assert re.search(r'WriteRegStr HKCU "\$\{UNINST_KEY\}" "UninstallString"', out)
    assert 'Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\OpenFlo' in out
    # file types go in the user's own Classes (test_open_session_files)
    assert 'WriteRegStr HKCU "${CLASSES}' in out


def test_real_template_upgrade_removes_only_its_own_python_folder():
    body = _block('Function RemoveOldRuntime', 'FunctionEnd')
    marker_check = body.index('${IfNot} ${FileExists} "$INSTDIR\\python\\${MARKER}"')
    assert marker_check < body.index('Call MoveRuntimeAside')
    section = _block('Section "OpenFlo (required)"', 'SectionEnd')
    assert section.index('Call RemoveOldRuntime') < section.index('File /r')
    uninstall = _block('Section "Uninstall"', 'SectionEnd')
    assert uninstall.index('${If} ${FileExists} "$INSTDIR\\python\\${MARKER}"') \
        < uninstall.index('Call un.MoveRuntimeAside')
    # nothing else deletes the python folder itself, only the moved-aside copy
    assert re.findall(r'RMDir /r "\$INSTDIR\\python[^"]*"', TEMPLATE) == \
        ['RMDir /r "$INSTDIR\\python.old"'] * 3
    # the user's settings folder is never deleted
    assert '.openflo' not in ''.join(
        line for line in TEMPLATE.splitlines() if 'RMDir' in line or 'Delete ' in line)


def _block(start, end):
    """The template's text from ``start`` to the next ``end`` line."""
    return TEMPLATE.split(start, 1)[1].split('\n' + end, 1)[0]


def _code_lines(text):
    """Lines that NSIS runs: stripped, without comments and blanks."""
    return [ln.strip() for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith((';', '#'))]


def test_running_openflo_stops_upgrade_and_uninstall_before_deleting():
    body = _code_lines(_block('!macro MOVE_RUNTIME_ASIDE UN', '!macroend'))
    rename = body.index('Rename "$INSTDIR\\python" "$INSTDIR\\python.old"')
    # Only the leftover python.old goes before the rename; a failed rename asks
    # the user and then stops.
    assert [ln for ln in body[:rename] if ln.startswith(('RMDir', 'Delete'))] == \
        ['RMDir /r "$INSTDIR\\python.old"']
    after = body[rename:]
    assert after[1] == '${If} ${Errors}'
    assert after[2].startswith('MessageBox MB_RETRYCANCEL') and after[2].endswith('IDRETRY retry')
    assert after[3].startswith('Abort ')
    assert '!insertmacro MOVE_RUNTIME_ASIDE ""' in TEMPLATE
    assert '!insertmacro MOVE_RUNTIME_ASIDE "un."' in TEMPLATE
    # The uninstaller checks first, before it deletes anything.
    uninstall = _code_lines(_block('Section "Uninstall"', 'SectionEnd'))
    first_delete = min(i for i, ln in enumerate(uninstall)
                       if ln.startswith(('RMDir', 'Delete', 'DeleteRegKey')))
    assert uninstall.index('Call un.MoveRuntimeAside') < first_delete


def test_every_message_box_has_a_silent_answer():
    # Without /SD a silent (/S) upgrade or uninstall waits on a hidden dialog.
    boxes = [ln for ln in _code_lines(TEMPLATE) if ln.startswith('MessageBox')]
    assert len(boxes) == 2
    for box in boxes:
        assert re.search(r' /SD ID(CANCEL|OK)\b', box), box
    assert any('/SD IDCANCEL' in box for box in boxes)


def test_uninstaller_removes_its_registry_entries():
    uninstall = _code_lines(_block('Section "Uninstall"', 'SectionEnd'))
    assert 'DeleteRegKey HKCU "${UNINST_KEY}"' in uninstall   # Settings > Apps
    assert 'DeleteRegKey HKCU "${APP_KEY}"' in uninstall      # the install folder


def test_upgrade_drops_a_desktop_shortcut_that_is_unticked():
    main = _code_lines(_block('Section "OpenFlo (required)"', 'SectionEnd'))
    assert 'Delete "$DESKTOP\\OpenFlo.lnk"' in main
    desktop = _code_lines(_block('Section "Desktop shortcut"', 'SectionEnd'))
    assert any(ln.startswith('CreateShortcut "$DESKTOP\\OpenFlo.lnk"') for ln in desktop)


def test_install_compiles_the_python_files():
    main = _code_lines(_block('Section "OpenFlo (required)"', 'SectionEnd'))
    compiles = [i for i, ln in enumerate(main)
                if ln.startswith('nsExec::Exec') and ' -m compileall ' in ln]
    assert len(compiles) == 1
    assert '$INSTDIR\\python\\python.exe' in main[compiles[0]]
    assert main.index('File /r "${PAYLOAD}\\python"') < compiles[0]


# ---- the staged Python ----

def _tcltk_tree(root):
    """A generated stand-in for the unpacked tcltk.msi."""
    for rel in ('DLLs/_tkinter.pyd', 'DLLs/tcl86t.dll', 'Lib/tkinter/__init__.py',
                'Lib/idlelib/idle.py', 'tcl/tcl8.6/init.tcl', 'tcl/tcl8/8.5/msgcat-1.6.1.tm',
                'tcl/tk8.6/tk.tcl', 'tcl/tix8.4.3/tix.tcl', 'tcl/include/tcl.h'):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text('x', encoding='utf-8')


def test_copy_tcltk_keeps_what_tk_needs_and_nothing_else(tmp_path):
    tk_dir, py = tmp_path / 'tcltk', tmp_path / 'python'
    _tcltk_tree(tk_dir)
    for exe in ('python.exe', 'pythonw.exe'):
        (py / exe).parent.mkdir(parents=True, exist_ok=True)
        (py / exe).write_text('x', encoding='utf-8')
    bwi.copy_tcltk(tk_dir, py)
    assert (py / 'tcl/tcl8/8.5/msgcat-1.6.1.tm').is_file()
    assert not (py / 'Lib/idlelib').exists()
    assert not (py / 'tcl/tix8.4.3').exists() and not (py / 'tcl/include').exists()
    bwi.check_staged_python(py)


@pytest.mark.parametrize('missing', ['tcl/tcl8/8.5/msgcat-1.6.1.tm', 'tcl/tcl8.6/init.tcl',
                                     'pythonw.exe', 'tcl/tk8.6/tk.tcl'])
def test_check_staged_python_names_a_missing_file(tmp_path, missing):
    py = tmp_path / 'python'
    _tcltk_tree(py)
    for exe in ('python.exe', 'pythonw.exe'):
        (py / exe).write_text('x', encoding='utf-8')
    bwi.check_staged_python(py)
    (py / missing).unlink()
    with pytest.raises(bwi.BuildError, match='has no'):
        bwi.check_staged_python(py)


def test_stage_python_adds_tk_to_the_nuget_python(tmp_path, monkeypatch):
    # Generated stand-ins for the two downloads; nothing is fetched or run.
    import zipfile
    cache, py = tmp_path / 'cache', tmp_path / 'payload' / 'python'
    cache.mkdir()
    nupkg = cache / 'python.nupkg'
    with zipfile.ZipFile(nupkg, 'w') as z:
        for rel in ('tools/python.exe', 'tools/pythonw.exe', 'tools/Lib/os.py'):
            z.writestr(rel, 'x')
    downloads = iter([nupkg, cache / 'tcltk.msi'])
    monkeypatch.setattr(bwi, 'fetch', lambda *a, **k: next(downloads))
    drop = {}

    def fake_run(cmd, what):                 # msiextract -C <dir> <msi>
        assert cmd[0] == 'msiextract'
        _tcltk_tree(Path(cmd[2]))
        for rel in drop:
            (Path(cmd[2]) / rel).unlink()
    monkeypatch.setattr(bwi, 'run', fake_run)
    bwi.stage_python(cache, py, '3.12.10', ('a', 'b'))
    assert (py / 'Lib/os.py').is_file() and (py / 'tcl/tcl8/8.5/msgcat-1.6.1.tm').is_file()
    assert not (py / 'Lib/idlelib').exists()
    drop['tcl/tcl8/8.5/msgcat-1.6.1.tm'] = True
    downloads = iter([nupkg, cache / 'tcltk.msi'])
    with pytest.raises(bwi.BuildError, match='msgcat'):
        bwi.stage_python(cache, tmp_path / 'other' / 'python', '3.12.10', ('a', 'b'))


# ---- hashes ----

def test_pinned_hash_refuses_an_unpinned_version():
    with pytest.raises(bwi.BuildError, match='No pinned hash for thing 9.9.9'):
        bwi.pinned_hash({'1.0': 'x'}, '9.9.9', 'thing')
    assert bwi.pinned_hash({'1.0': 'x'}, '1.0', 'thing') == 'x'


def test_python_hashes_need_both_pins(monkeypatch):
    monkeypatch.setattr(bwi, 'NUGET_SHA512', {'9.0.1': 'n'})
    monkeypatch.setattr(bwi, 'TCLTK_SHA256', {})
    with pytest.raises(bwi.BuildError, match='tcltk.msi 9.0.1'):
        bwi.python_hashes('9.0.1')
    monkeypatch.setattr(bwi, 'TCLTK_SHA256', {'9.0.1': 't'})
    assert bwi.python_hashes('9.0.1') == ('n', 't')


def test_default_python_is_pinned():
    assert bwi.python_hashes(bwi.PYTHON_VERSION)


def test_check_digest_hex_and_base64(tmp_path):
    f = tmp_path / 'blob.bin'
    f.write_bytes(bytes(range(256)) * 10)
    sha256 = hashlib.sha256(f.read_bytes()).hexdigest()
    sha512 = base64.b64encode(hashlib.sha512(f.read_bytes()).digest()).decode()
    bwi.check_digest(f, 'sha256', sha256, 'hex')
    bwi.check_digest(f, 'sha512', sha512, 'base64')
    with pytest.raises(bwi.BuildError, match='not the pinned file'):
        bwi.check_digest(f, 'sha256', '0' * 64, 'hex')
    with pytest.raises(bwi.BuildError, match='not the pinned file'):
        bwi.check_digest(f, 'sha512', sha512[:-4] + 'AAA=', 'base64')


def test_fetch_deletes_a_cached_file_that_fails_the_hash(tmp_path, monkeypatch):
    def no_network(*a, **k):
        raise AssertionError('fetch went to the network')
    monkeypatch.setattr(bwi.urllib.request, 'urlopen', no_network)
    cached = tmp_path / 'pkg.zip'
    cached.write_bytes(b'generated bytes')
    good = hashlib.sha256(b'generated bytes').hexdigest()
    assert bwi.fetch('https://example.invalid/x', cached, 'sha256', good, 'hex') == cached
    with pytest.raises(bwi.BuildError):
        bwi.fetch('https://example.invalid/x', cached, 'sha256', '1' * 64, 'hex')
    assert not cached.exists()


def test_main_refuses_an_unpinned_python_before_downloading(tmp_path, monkeypatch, capsys):
    def no_network(*a, **k):
        raise AssertionError('main went to the network')
    monkeypatch.setattr(bwi.urllib.request, 'urlopen', no_network)
    assert bwi.main(['--python-version', '3.99.0', '--build-dir', str(tmp_path)]) == 1
    assert 'No pinned hash for the NuGet python package 3.99.0' in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


# ---- the installed bundle ----

def _dist(site, name, version, requires=(), files=()):
    info = site / f'{name}-{version}.dist-info'
    info.mkdir(parents=True)
    lines = ['Metadata-Version: 2.1', f'Name: {name}', f'Version: {version}']
    lines += [f'Requires-Dist: {r}' for r in requires]
    (info / 'METADATA').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    for rel in files:
        (info / rel).write_text('x', encoding='utf-8')
    return info


def test_unmet_requirements_use_windows_markers(tmp_path):
    site = tmp_path / 'site'
    _dist(site, 'Bar_Tool', '1.0', ['colorlib; platform_system == "Windows"',
                                    'posixlib; sys_platform == "linux"',
                                    'testlib; extra == "test"',
                                    'Base-Lib>=2'])
    _dist(site, 'base.lib', '1.5')
    dists = bwi.installed_dists(site)
    assert set(dists) == {'bar-tool', 'base-lib'}
    assert bwi.unmet_requirements(dists) == [
        'bar-tool: colorlib; platform_system == "Windows"', 'bar-tool: Base-Lib>=2']
    _dist(site, 'colorlib', '0.1')
    (site / 'base.lib-1.5.dist-info' / 'METADATA').write_text(
        'Metadata-Version: 2.1\nName: base.lib\nVersion: 2.0\n', encoding='utf-8')
    assert bwi.unmet_requirements(bwi.installed_dists(site)) == []


def test_unmet_requirements_accept_a_prerelease_that_meets_it():
    dists = {'app': ('1.0', ['lib>=2']), 'lib': ('2.1rc1', [])}
    assert bwi.unmet_requirements(dists) == []


def test_exact_pins_take_only_single_version_pins():
    reqs = ['Aa_A==1.0', 'bbb>=2', 'ccc==3.*', 'ddd===4.0x', 'eee==5.1,<6', 'fff']
    assert bwi.exact_pins(reqs) == {'aa-a': '1.0', 'ddd': '4.0x', 'eee': '5.1'}


def test_unpinned_dists_name_each_staged_dist_without_a_pin():
    dists = {'openflo': ('9.0', []), 'pinned-lib': ('1.0', []),
             'loose-lib': ('2.0', []), 'wild-lib': ('3.1', []), 'deep-dep': ('0.4', [])}
    reqs = ['Pinned_Lib==1.0', 'loose-lib>=2', 'wild-lib==3.*']
    assert bwi.unpinned_dists(dists, reqs) == ['deep-dep 0.4', 'loose-lib 2.0',
                                               'wild-lib 3.1']
    assert bwi.unpinned_dists(dists, [*reqs, 'loose-lib==2.0', 'wild-lib==3.1',
                                      'deep-dep==0.4']) == []


def test_unpinned_dists_name_a_staged_version_that_is_not_the_pin():
    dists = {'lib-a': ('2.0', []), 'lib-b': ('1.0', []), 'lib-c': ('3.0', [])}
    reqs = ['lib-a==1.0', 'lib-b==1.0.0', 'lib-c==3.0']
    assert bwi.unpinned_dists(dists, reqs) == ['lib-a 2.0 (pinned 1.0)']
    assert bwi.same_version('1.0', '1.0.0') and not bwi.same_version('1.0', '1.0.1')
    assert bwi.same_version('local-x', 'local-x') and not bwi.same_version('x', 'y')


def test_exact_pins_skip_a_pin_whose_marker_does_not_hold_on_windows():
    reqs = ['lin-lib==1.0; sys_platform == "linux"', 'win-lib==2.0; os_name == "nt"']
    assert bwi.exact_pins(reqs) == {'win-lib': '2.0'}
    assert bwi.unpinned_dists({'lin-lib': ('1.0', [])}, reqs) == ['lin-lib 1.0']


def test_unpinned_dists_exempt_what_came_with_python_at_that_version():
    dists = {'tool-pkg': ('25.0', []), 'deep-dep': ('0.4', [])}
    assert bwi.unpinned_dists(dists, [], {'tool-pkg': '25.0'}) == ['deep-dep 0.4']
    # replaced by another version during staging: no longer the hashed one
    assert bwi.unpinned_dists(dists, [], {'tool-pkg': '24.0'}) == ['deep-dep 0.4',
                                                                   'tool-pkg 25.0']


def test_check_bundle_stops_on_an_unpinned_or_unmet_dist():
    dists = {'app-lib': ('1.0', ['dep-lib>=1']), 'dep-lib': ('1.2', [])}
    bwi.check_bundle(dists, ['app-lib==1.0', 'dep-lib==1.2'])
    with pytest.raises(bwi.BuildError, match=r'than pinned:\n  dep-lib 1.2 \(pinned 1.1\)\n'):
        bwi.check_bundle(dists, ['app-lib==1.0', 'dep-lib==1.1'])
    with pytest.raises(bwi.BuildError, match=r'than pinned:\n  dep-lib 1.2\n'):
        bwi.check_bundle(dists, ['app-lib==1.0'])
    with pytest.raises(bwi.BuildError, match=r'meets on Windows:\n  app-lib: dep-lib>=1'):
        bwi.check_bundle({'app-lib': ('1.0', ['dep-lib>=1'])}, ['app-lib==1.0'])


def test_stage_packages_stops_on_an_unpinned_dist(tmp_path, monkeypatch):
    # pip is replaced: "installing" writes a pinned and an unpinned dist, and
    # building the own wheel writes an empty wheel file.
    site = tmp_path / 'site'

    def fake_pip(*args, what):
        if args[0] == 'install' and '--no-deps' not in args:
            _dist(site, 'pinned_lib', '1.0', ['deep_dep'])
            _dist(site, 'deep_dep', '0.4')
        elif args[0] == 'wheel':
            out = Path(args[args.index('--wheel-dir') + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / 'openflo-9.0-py3-none-any.whl').write_bytes(b'')
        elif args[0] == 'install':
            _dist(site, 'openflo', '9.0', ['pinned_lib'])

    monkeypatch.setattr(bwi, 'pip', fake_pip)
    _dist(site, 'tool_pkg', '25.0')          # what the Python package brings
    with pytest.raises(bwi.BuildError, match='no == pin.*\n  deep-dep 0.4\n'):
        bwi.stage_packages(tmp_path / 'build', site, ['pinned-lib==1.0'], '9.0')
    for info in site.glob('*.dist-info'):
        if not info.name.startswith('tool_pkg-'):
            shutil.rmtree(info)
    bwi.stage_packages(tmp_path / 'build', site, ['pinned-lib==1.0', 'deep-dep==0.4'], '9.0')


def test_requirements_txt_pins_every_dependency_of_its_pins_on_windows():
    # Offline stand-in for resolving the installer's wheel closure: every
    # requirement (under Windows markers) of a pinned package installed in
    # this environment names a pinned project. A new transitive dependency
    # shows up here before the installer build stops on it.
    import tomllib
    from importlib import metadata

    from packaging.requirements import Requirement
    pyproject = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    pins = bwi.exact_pins(bwi.runtime_requirements(
        (ROOT / 'requirements.txt').read_text(encoding='utf-8'), pyproject))
    checked, missing = [], []
    for name in sorted(pins):
        try:
            requires = metadata.distribution(name).requires or []
        except metadata.PackageNotFoundError:
            continue
        checked.append(name)
        for spec in requires:
            req = Requirement(spec)
            if req.marker is not None and not req.marker.evaluate(bwi.WINDOWS_MARKER_ENV):
                continue
            if bwi.requirement_name(req.name) not in pins:
                missing.append(f'{name}: {spec}')
    assert {'cryptography', 'cffi', 'openpyxl', 'phenograph'} <= set(checked)
    assert missing == []


def test_drop_host_traces(tmp_path):
    site = tmp_path / 'site'
    (site / 'bin').mkdir(parents=True)
    (site / 'bin' / 'tool').write_text('#!/build/machine/python\n', encoding='utf-8')
    info = _dist(site, 'own', '1.0', files=('direct_url.json',))
    (info / 'RECORD').write_text(
        'own/__init__.py,sha256=a,1\n'
        'own-1.0.dist-info/direct_url.json,sha256=b,2\n'
        'own-1.0.dist-info/METADATA,sha256=c,3\n', encoding='utf-8')
    other = _dist(site, 'dep', '2.0')
    (other / 'RECORD').write_text('dep/x.py,,\n', encoding='utf-8')
    bwi.drop_host_traces(site)
    assert not (site / 'bin').exists()
    assert not (info / 'direct_url.json').exists()
    assert (info / 'RECORD').read_text(encoding='utf-8') == (
        'own/__init__.py,sha256=a,1\nown-1.0.dist-info/METADATA,sha256=c,3\n')
    assert (other / 'RECORD').read_text(encoding='utf-8') == 'dep/x.py,,\n'
