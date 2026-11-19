"""Every place that names OpenFlo's publisher names the same company:
openflo.publisher is the one source and LICENSE.txt must carry the same
copyright line. The About window and the Windows installer read it from
openflo.publisher (test_about_window.py, test_build_windows_installer.py);
the phone app's Settings and store notes are checked in
mobile/tests/test_publisher.py."""
from __future__ import annotations

from pathlib import Path

from openflo.publisher import COPYRIGHT, COPYRIGHT_YEAR, PUBLISHER

ROOT = Path(__file__).resolve().parents[1]


def test_copyright_line_is_year_and_publisher():
    assert COPYRIGHT == f"Copyright (c) {COPYRIGHT_YEAR} {PUBLISHER}"
    assert PUBLISHER.endswith(' LLC')


def test_licence_holds_the_publishers_copyright_line():
    text = (ROOT / 'LICENSE.txt').read_text(encoding='utf-8')
    assert text.startswith('MIT License')
    lines = [ln for ln in text.splitlines() if ln.startswith('Copyright')]
    assert lines == [COPYRIGHT]


MIT_BODY = """\
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def test_license_is_the_plain_mit_text():
    """Exactly the MIT text (words compared, line breaks ignored) and nothing
    after it, so GitHub and license scanners see MIT (SignPath Foundation
    needs an OSI license it can recognize); the citation request lives in
    README.md and CITATION.cff."""
    text = (ROOT / 'LICENSE.txt').read_text(encoding='utf-8')
    expected = f'MIT License\n\n{COPYRIGHT}\n\n{MIT_BODY}'
    assert text.split() == expected.split()
    assert text.endswith('SOFTWARE.\n')
