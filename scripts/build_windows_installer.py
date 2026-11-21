#!/usr/bin/env python
"""Build the one-click Windows installer, OpenFlo-Setup-<version>.exe, on Linux.

Lab users install OpenFlo without setting up Python: the installer carries
its own Windows Python with tkinter, every pinned dependency and OpenFlo
itself, and puts it in %LOCALAPPDATA%\\Programs\\OpenFlo for the current user
(no administrator rights). packaging/windows/openflo.nsi says what it does on
the PC.

    python scripts/build_windows_installer.py            # needs makensis, msiextract (+ packaging, numpy, pillow)
    python scripts/build_windows_installer.py --build-dir /tmp/ofb

Debian/Ubuntu tools: ``apt-get install nsis msitools``; the Python running
this script also needs ``packaging``, ``numpy`` and ``pillow`` (pinned in
requirements.txt). Nothing here needs
Windows or GitHub Actions. Downloads are cached in the build folder
(default build/windows-installer/cache) and each is checked against the hash
pinned below before use; a mismatch stops the build.

Where the Windows Python comes from:

* the core interpreter and standard library: the official ``python`` NuGet
  package (python.exe, pythonw.exe, Lib, DLLs). Its SHA-512 is pinned below
  and is the one NuGet publishes for the package (``packageHash`` in the
  catalog entry for that version).
* tkinter and Tcl/Tk: the NuGet package leaves them out, so they come from
  ``tcltk.msi`` of the same python.org release (the file the official
  installer installs for "tcl/tk and IDLE"), unpacked with msiextract. Its
  SHA-256 is pinned below; the file was checked against python.org's GPG
  signature (``tcltk.msi.asc``) when the hash was pinned.

Dependencies: requirements.txt's pins plus the ``gui`` extra, as win_amd64 /
cp312 binary wheels only. Every distribution that lands in site-packages,
dependencies of dependencies included, must be pinned there with ``==``; the
build stops on one that is not, so what ships never changes on its own.
(The pip that comes inside the NuGet package is pinned by that package's
hash instead.) A pinned package with no wheel at all is built from
its sdist only when it is listed in PURE_SDIST_PACKAGES and the result is a
pure-Python wheel. Then every installed distribution's requirements are
checked under a Windows environment, because pip evaluates markers such as
``platform_system == "Windows"`` for the machine it runs on, not the target.

The bundled Python gets ``openflo-install.json`` in its prefix
(``{"kind": "installer", "version": ...}``): how OpenFlo knows it was
installed by this installer.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tomllib
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NSI_TEMPLATE = ROOT / 'packaging' / 'windows' / 'openflo.nsi'
DEFAULT_BUILD_DIR = ROOT / 'build' / 'windows-installer'

PYTHON_VERSION = '3.12.10'
# The NuGet "python" package: {version: base64 SHA-512}, as NuGet publishes
# it (catalog entry field packageHash, packageHashAlgorithm SHA512).
NUGET_SHA512 = {
    '3.12.10': 'u9pNz2iKlCEbYtUJaKkbOPMF0LjR7NkCafdKhvigpPzrt8oWKgdTpHaR6z3w'
               'yWQAm9PYGUxv0Zr66NX9AeHMDw==',
}
# python.org's amd64/tcltk.msi for the same release: {version: hex SHA-256}.
TCLTK_SHA256 = {
    '3.12.10': '55c96ffad69b1c834aa52e11b9ce41637a178ba6ad6607e83956044834276e2a',
}
NUGET_URL = ('https://api.nuget.org/v3-flatcontainer/python/{v}/'
             'python.{v}.nupkg')
TCLTK_URL = 'https://www.python.org/ftp/python/{v}/amd64/tcltk.msi'

# Pinned packages PyPI has only as an sdist, accepted because they are pure
# Python (the built wheel's tag is checked to be py3-none-any).
PURE_SDIST_PACKAGES = frozenset({'flowio'})
# What the wheels must be built for.
PLATFORM_ARGS = ('--platform', 'win_amd64', '--python-version', '3.12',
                 '--implementation', 'cp', '--abi', 'cp312',
                 '--only-binary=:all:')
# The environment markers of a 64-bit Windows CPython 3.12.
WINDOWS_MARKER_ENV = {
    'os_name': 'nt', 'sys_platform': 'win32', 'platform_system': 'Windows',
    'platform_machine': 'AMD64', 'platform_python_implementation': 'CPython',
    'implementation_name': 'cpython', 'python_version': '3.12',
    'python_full_version': PYTHON_VERSION,
    'implementation_version': PYTHON_VERSION, 'extra': '',
}
# From tcltk.msi, only what tkinter needs at run time (no IDLE, turtle demo,
# Tix, or the Tcl build files).
TCLTK_KEEP = ('DLLs/', 'Lib/tkinter/', 'tcl/tcl8/', 'tcl/tcl8.6/',
              'tcl/tk8.6/', 'tcl/dde1.4/', 'tcl/reg1.3/')
# Files the staged Python must have. Tk's init.tcl loads msgcat from tcl8/.
STAGED_PYTHON_NEEDS = ('python.exe', 'pythonw.exe', 'DLLs/_tkinter.pyd',
                       'DLLs/tcl86t.dll', 'Lib/tkinter/__init__.py',
                       'tcl/tcl8.6/init.tcl', 'tcl/tcl8/8.5/msgcat-*.tm',
                       'tcl/tk8.6/tk.tcl')
MARKER_NAME = 'openflo-install.json'
GUI_SCRIPT = 'openflo-gui'
ABOUT_URL = 'https://github.com/ChironTheCentaur/openflo-public'
# The policy the installer's privacy page links to (covers the desktop app).
PRIVACY_URL = 'https://www.getopenflo.com/privacy'
# OpenFlo's own wheel: staged from this checkout, so requirements.txt does
# not pin it.
OWN_DISTRIBUTION = 'openflo'
_PLACEHOLDER = re.compile(r'@@([A-Z_]+)@@')


class BuildError(Exception):
    """A build step that cannot go on; the message says what to do."""


# ---- pure parts (unit-tested in tests/test_build_windows_installer.py) ----

def read_requirements(text):
    """The requirement strings in a requirements.txt body: comments, blank
    lines and trailing ``# ...`` notes dropped, order kept."""
    out = []
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            out.append(line)
    return out


def requirement_name(req):
    """The normalised project name of a requirement string."""
    m = re.match(r'\s*([A-Za-z0-9][A-Za-z0-9._-]*)', req)
    if not m:
        raise BuildError(f'Cannot read a project name from {req!r}.')
    return re.sub(r'[-_.]+', '-', m.group(1)).lower()


def runtime_requirements(requirements_text, pyproject):
    """requirements.txt's pins plus the ``gui`` extra's entries that the pins
    do not already name (the pins win where both name a project)."""
    reqs = read_requirements(requirements_text)
    named = {requirement_name(r) for r in reqs}
    gui = pyproject.get('project', {}).get('optional-dependencies', {}).get('gui', [])
    for req in gui:
        if requirement_name(req) not in named:
            reqs.append(req)
            named.add(requirement_name(req))
    return reqs


def gui_module(pyproject):
    """The module the ``openflo-gui`` gui-script runs (``pythonw -m`` target)."""
    target = pyproject.get('project', {}).get('gui-scripts', {}).get(GUI_SCRIPT)
    if not target or ':' not in target:
        raise BuildError(f'pyproject.toml has no [project.gui-scripts] {GUI_SCRIPT} entry.')
    module, func = target.split(':', 1)
    if func != 'main':
        raise BuildError(f'{GUI_SCRIPT} calls {func}, but `python -m {module}` runs main().')
    return module


def split_sdist_only(reqs):
    """``(binary, sdist_only)``: the requirements PyPI has as wheels, and
    the pure-Python ones that have only an sdist (PURE_SDIST_PACKAGES)."""
    binary = [r for r in reqs if requirement_name(r) not in PURE_SDIST_PACKAGES]
    sdist = [r for r in reqs if requirement_name(r) in PURE_SDIST_PACKAGES]
    return binary, sdist


def pip_download_args(requirements_file, wheel_dir):
    """pip arguments that fetch the pinned closure as Windows wheels."""
    return ['download', '--dest', str(wheel_dir), '--find-links', str(wheel_dir),
            *PLATFORM_ARGS, '-r', str(requirements_file)]


def pip_sdist_wheel_args(req, wheel_dir):
    """pip arguments that build a pure-Python wheel from ``req``'s sdist
    (only that project from source; its build tools may be wheels)."""
    return ['wheel', '--no-deps', '--no-binary', requirement_name(req),
            '--wheel-dir', str(wheel_dir), req]


def pip_install_args(requirements_file, wheel_dir, target):
    """pip arguments that install the pinned closure into ``target`` from
    the downloaded wheels only (no index): what was checked is what ships.
    No .pyc files: they would record this machine's paths."""
    return ['install', '--target', str(target), '--no-index',
            '--find-links', str(wheel_dir), *PLATFORM_ARGS, '--no-compile',
            '--no-warn-conflicts', '--no-warn-script-location',
            '-r', str(requirements_file)]


def pip_install_own_wheel_args(wheel, target):
    """pip arguments that add OpenFlo's own wheel, without its dependencies
    (they are installed already, at the pinned versions)."""
    return ['install', '--target', str(target), '--no-deps', '--no-compile',
            '--no-warn-script-location', *PLATFORM_ARGS, str(wheel)]


def is_pure_wheel(filename):
    """True for a wheel whose tag runs on any Python 3 platform."""
    parts = filename[:-len('.whl')].split('-') if filename.endswith('.whl') else []
    if len(parts) < 5:
        return False
    py, abi, plat = parts[-3:]
    return abi == 'none' and plat == 'any' and 'py3' in py.split('.')


def marker_content(version):
    """The install marker the bundled Python's prefix carries."""
    return json.dumps({'kind': 'installer', 'version': version},
                      indent=2) + '\n'


def vi_version(version):
    """A version as Windows file-version numbers, ``a.b.c.d``
    (``3.1.0b2`` -> ``3.1.0.0``)."""
    m = re.match(r'(\d+(?:\.\d+)*)', version)
    if not m:
        raise BuildError(f'Version {version!r} does not start with a number.')
    parts = [int(p) for p in m.group(1).split('.')][:4]
    return '.'.join(str(p) for p in parts + [0] * (4 - len(parts)))


def installer_name(version):
    return f'OpenFlo-Setup-{version}.exe'


def publisher_values():
    """The installer's publisher and copyright fields, from
    openflo.publisher (the name the About window and LICENSE.txt carry)."""
    sys.path.insert(0, str(ROOT / 'src'))
    from openflo.publisher import COPYRIGHT, PUBLISHER
    return {'PUBLISHER': PUBLISHER, 'COPYRIGHT': COPYRIGHT}


def render_nsi(template, values):
    """``template`` with every ``@@NAME@@`` replaced by ``values[NAME]``.
    A placeholder without a value, a value without a placeholder, or a value
    NSIS would misread inside a quoted string is a BuildError."""
    used = set(_PLACEHOLDER.findall(template))
    missing = sorted(used - set(values))
    if missing:
        raise BuildError(f'No value for {", ".join(missing)} in the .nsi template.')
    unused = sorted(set(values) - used)
    if unused:
        raise BuildError(f'The .nsi template has no {", ".join(unused)} placeholder.')
    for key, value in values.items():
        if re.search(r'["$`\r\n]', str(value)):
            raise BuildError(f'{key} = {value!r} has a character NSIS cannot take '
                             'in a quoted string (", $, ` or a line break).')
    return _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), template)


def nsi_values(version, module, payload, outfile, size_kb):
    """The value of every placeholder in packaging/windows/openflo.nsi."""
    return {
        'VERSION': version, 'VI_VERSION': vi_version(version), 'GUI_MODULE': module,
        'PAYLOAD_DIR': payload, 'OUTFILE': outfile, 'MARKER_NAME': MARKER_NAME,
        'LICENSE_FILE': ROOT / 'LICENSE.txt', 'ABOUT_URL': ABOUT_URL,
        'PRIVACY_URL': PRIVACY_URL, 'ESTIMATED_SIZE_KB': size_kb,
        **publisher_values(),
    }


def check_digest(data_path, algorithm, expected, encoding):
    """Raise BuildError unless ``data_path``'s digest equals ``expected``
    (``encoding`` 'hex' or 'base64')."""
    h = hashlib.new(algorithm)
    with open(data_path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    got = (h.hexdigest() if encoding == 'hex'
           else base64.b64encode(h.digest()).decode('ascii'))
    if got != expected:
        raise BuildError(f'{Path(data_path).name}: {algorithm} is {got}, '
                         f'expected {expected}. The download is not the pinned file.')


def pinned_hash(table, version, what):
    """The pinned hash of ``what`` for ``version``; refuses an unpinned one."""
    try:
        return table[version]
    except KeyError:
        raise BuildError(
            f'No pinned hash for {what} {version}. Add the published hash to '
            f'scripts/build_windows_installer.py before building with it.') from None


def python_hashes(version):
    """``(NuGet SHA-512, tcltk.msi SHA-256)`` pinned for ``version``; a
    BuildError, before anything is downloaded, when either is not pinned."""
    return (pinned_hash(NUGET_SHA512, version, 'the NuGet python package'),
            pinned_hash(TCLTK_SHA256, version, 'tcltk.msi'))


def unmet_requirements(dists, env=None):
    """Requirements of installed distributions that nothing installed meets,
    under the Windows marker environment.

    ``dists`` maps a normalised name to ``(version, [Requires-Dist, ...])``.
    Returns ``["needer: requirement", ...]``."""
    from packaging.requirements import Requirement
    env = dict(WINDOWS_MARKER_ENV if env is None else env)
    out = []
    for name, (_version, requires) in sorted(dists.items()):
        for spec in requires:
            req = Requirement(spec)
            if req.marker is not None and not req.marker.evaluate(env):
                continue
            have = dists.get(requirement_name(req.name))
            if have is None or not req.specifier.contains(have[0], prereleases=True):
                out.append(f'{name}: {spec}')
    return out


def exact_pins(reqs, env=None):
    """``{name: version}`` for each requirement pinned with ``==`` (or
    ``===``) to one version; a wildcard such as ``==2.*`` is not a pin, and
    a requirement whose marker does not hold on Windows (``env``, default
    WINDOWS_MARKER_ENV) pins nothing."""
    from packaging.requirements import Requirement
    env = dict(WINDOWS_MARKER_ENV if env is None else env)
    pins = {}
    for req in reqs:
        parsed = Requirement(req)
        if parsed.marker is not None and not parsed.marker.evaluate(env):
            continue
        for spec in parsed.specifier:
            if spec.operator in ('==', '===') and '*' not in spec.version:
                pins[requirement_name(req)] = spec.version
    return pins


def same_version(a, b):
    """True when two version strings name the same version (``1.0`` and
    ``1.0.0`` do); compared as text when either is not a valid version."""
    from packaging.version import InvalidVersion, Version
    try:
        return Version(a) == Version(b)
    except InvalidVersion:
        return a == b


def unpinned_dists(dists, reqs, with_python=None):
    """Staged distributions that ``reqs`` does not pin with ``==``, as
    ``"name version"``, or pins at another version, as ``"name version
    (pinned X)"``. ``dists`` is as from installed_dists. Exempt: OpenFlo's
    own wheel, and what ``with_python`` (``{name: version}``) says came
    inside the hash-pinned Python package (its pip), at that version."""
    pins = exact_pins(reqs)
    with_python = with_python or {}
    out = []
    for name, (version, _requires) in sorted(dists.items()):
        if name == OWN_DISTRIBUTION or with_python.get(name) == version:
            continue
        pin = pins.get(name)
        if pin is None:
            out.append(f'{name} {version}')
        elif not same_version(pin, version):
            out.append(f'{name} {version} (pinned {pin})')
    return out


def check_bundle(dists, reqs, with_python=None):
    """Raise BuildError when a staged distribution's requirement is unmet on
    Windows, or when a staged distribution has no ``==`` pin in ``reqs`` or
    another version than its pin (``with_python`` as in unpinned_dists)."""
    unmet = unmet_requirements(dists)
    if unmet:
        raise BuildError('Requirements nothing in the bundle meets on Windows:\n  '
                         + '\n  '.join(unmet) + '\nPin them in requirements.txt.')
    unpinned = unpinned_dists(dists, reqs, with_python)
    if unpinned:
        raise BuildError('Distributions in the bundle with no == pin in requirements.txt, '
                         'or at another version than pinned:'
                         '\n  ' + '\n  '.join(unpinned) + '\nPin each one with == in '
                         'requirements.txt, at a version with a win_amd64 cp312 wheel '
                         'or a pure-Python one.')


# ---- I/O steps ----

def fetch(url, dest, algorithm, expected, encoding):
    """``url`` saved at ``dest`` (kept between builds) and hash-checked."""
    dest = Path(dest)
    if not dest.exists():
        print(f'Downloading {url}')
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + '.part')
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(part, 'wb') as f:
                shutil.copyfileobj(r, f)
        except OSError as exc:
            raise BuildError(f'Could not download {url}: {exc}') from exc
        part.replace(dest)
    try:
        check_digest(dest, algorithm, expected, encoding)
    except BuildError:
        dest.unlink()
        raise
    return dest


def run(cmd, what):
    print('+', ' '.join(str(c) for c in cmd), flush=True)
    try:
        subprocess.run([str(c) for c in cmd], check=True)
    except FileNotFoundError as exc:
        raise BuildError(f'{what}: {cmd[0]} is not installed.') from exc
    except subprocess.CalledProcessError as exc:
        raise BuildError(f'{what} failed (exit {exc.returncode}); see the output above.') from exc


def pip(*args, what):
    run([sys.executable, '-m', 'pip', *args], what)


def stage_python(cache, python_dir, version, hashes):
    """The Windows Python, with tkinter, unpacked at ``python_dir``.
    ``hashes`` is ``(NuGet SHA-512, tcltk.msi SHA-256)`` (see python_hashes)."""
    nupkg = fetch(NUGET_URL.format(v=version), cache / f'python.{version}.nupkg',
                  'sha512', hashes[0], 'base64')
    msi = fetch(TCLTK_URL.format(v=version), cache / f'tcltk-{version}.msi', 'sha256',
                hashes[1], 'hex')
    with zipfile.ZipFile(nupkg) as z:
        for info in z.infolist():
            if not info.filename.startswith('tools/') or info.is_dir():
                continue
            out = python_dir / info.filename[len('tools/'):]
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(out, 'wb') as dst:
                shutil.copyfileobj(src, dst)
    tk_dir = cache.parent / 'tcltk'
    shutil.rmtree(tk_dir, ignore_errors=True)
    tk_dir.mkdir(parents=True)
    run(['msiextract', '-C', tk_dir, msi], 'Unpacking tcltk.msi')
    copy_tcltk(tk_dir, python_dir)
    check_staged_python(python_dir)


def copy_tcltk(tk_dir, python_dir):
    """Copy the parts of the unpacked tcltk.msi that TCLTK_KEEP names."""
    for path in sorted(Path(tk_dir).rglob('*')):
        rel = path.relative_to(tk_dir).as_posix()
        if path.is_file() and rel.startswith(TCLTK_KEEP):
            out = Path(python_dir) / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)


def check_staged_python(python_dir):
    """Raise BuildError when the staged Python lacks a file Tk needs to start.
    A pattern with ``*`` needs at least one match."""
    for need in STAGED_PYTHON_NEEDS:
        if not any(p.is_file() for p in Path(python_dir).glob(need)):
            raise BuildError(f'The staged Python has no {need}.')


def installed_dists(site_packages):
    """{name: (version, requires)} read from the .dist-info folders."""
    from email.parser import Parser
    dists = {}
    for meta in Path(site_packages).glob('*.dist-info/METADATA'):
        msg = Parser().parsestr(meta.read_text(encoding='utf-8'))
        dists[requirement_name(msg['Name'])] = (msg['Version'],
                                               msg.get_all('Requires-Dist') or [])
    return dists


def drop_host_traces(site_packages):
    """Remove what pip wrote about THIS machine: console-script wrappers
    with Linux shebangs (useless on Windows) and a local wheel's
    direct_url.json (its file:// path), with its RECORD line."""
    site_packages = Path(site_packages)
    shutil.rmtree(site_packages / 'bin', ignore_errors=True)
    for direct in site_packages.glob('*.dist-info/direct_url.json'):
        record = direct.with_name('RECORD')
        if record.exists():
            lines = record.read_text(encoding='utf-8').splitlines(keepends=True)
            record.write_text(''.join(ln for ln in lines if not ln.startswith(
                f'{direct.parent.name}/direct_url.json,')), encoding='utf-8')
        direct.unlink()


def stage_packages(build, site_packages, reqs, version):
    """Pinned dependencies and OpenFlo's own wheel in ``site_packages``."""
    # What the Python package itself put there (pip); its hash pins it.
    with_python = {name: version for name, (version, _r)
                   in installed_dists(site_packages).items()}
    wheels = build / 'cache' / 'wheels'
    wheels.mkdir(parents=True, exist_ok=True)
    req_file = build / 'requirements-windows.txt'
    req_file.write_text('\n'.join(reqs) + '\n', encoding='utf-8')
    _binary, sdist_only = split_sdist_only(reqs)
    for req in sdist_only:
        pip(*pip_sdist_wheel_args(req, wheels), what=f'Building a wheel for {req}')
        name = requirement_name(req).replace('-', '_')
        built = [p.name for p in wheels.glob('*.whl')
                 if p.name.lower().startswith(name + '-')]
        if not built or not all(is_pure_wheel(b) for b in built):
            raise BuildError(f'{req} did not build as a pure-Python wheel ({built}).')
    pip(*pip_download_args(req_file, wheels), what='Downloading Windows wheels')
    pip(*pip_install_args(req_file, wheels, site_packages),
        what='Installing the dependencies')
    drop_host_traces(site_packages)
    own = build / 'openflo-wheel'
    shutil.rmtree(own, ignore_errors=True)
    pip('wheel', '--no-deps', '--wheel-dir', own, ROOT, what='Building the OpenFlo wheel')
    (wheel,) = own.glob(f'openflo-{version}-*.whl')
    pip(*pip_install_own_wheel_args(wheel, site_packages), what='Installing OpenFlo')
    drop_host_traces(site_packages)
    check_bundle(installed_dists(site_packages), reqs, with_python)


def write_icon(path):
    """OpenFlo's mark (openflo.app_icon) as a multi-size .ico."""
    sys.path.insert(0, str(ROOT / 'src'))
    from openflo.app_icon import icon_image
    sizes = (256, 128, 64, 48, 32, 24, 16)
    images = [icon_image(s) for s in sizes]
    images[0].save(path, format='ICO', sizes=[(s, s) for s in sizes],
                   append_images=images[1:])


def folder_kb(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file()) // 1024


def build(build_dir, python_version):
    pyproject = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    version = pyproject['project']['version']
    module = gui_module(pyproject)
    hashes = python_hashes(python_version)
    reqs = runtime_requirements((ROOT / 'requirements.txt').read_text(encoding='utf-8'),
                                pyproject)
    for tool in ('makensis', 'msiextract'):
        if shutil.which(tool) is None:
            raise BuildError(f'{tool} is not installed (Debian/Ubuntu: apt-get install '
                             'nsis msitools).')
    payload = build_dir / 'payload'
    shutil.rmtree(payload, ignore_errors=True)
    python_dir = payload / 'python'
    python_dir.mkdir(parents=True)
    stage_python(build_dir / 'cache', python_dir, python_version, hashes)
    stage_packages(build_dir, python_dir / 'Lib' / 'site-packages', reqs, version)
    (python_dir / MARKER_NAME).write_text(marker_content(version), encoding='utf-8')
    write_icon(payload / 'OpenFlo.ico')
    out = build_dir / installer_name(version)
    nsi = build_dir / 'openflo.nsi'
    nsi.write_text(render_nsi(NSI_TEMPLATE.read_text(encoding='utf-8'), nsi_values(
        version, module, payload, out, folder_kb(payload))), encoding='utf-8')
    run(['makensis', '-V2', '-INPUTCHARSET', 'UTF8', nsi], 'makensis')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--build-dir', type=Path, default=DEFAULT_BUILD_DIR,
                    help='work and download-cache folder (default build/windows-installer)')
    ap.add_argument('--python-version', default=PYTHON_VERSION,
                    help=f'Windows Python to bundle (default {PYTHON_VERSION}; '
                         'needs pinned hashes)')
    args = ap.parse_args(argv)
    try:
        out = build(args.build_dir.resolve(), args.python_version)
    except BuildError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f'Built {out} ({out.stat().st_size / 1e6:.1f} MB, sha256 {digest})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
