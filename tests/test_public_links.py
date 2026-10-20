"""Every link a user can follow must name the public mirror.

The development repository is private, so a link to it is a 404 for anyone
else; Help > Report a problem once sent beta testers there.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_PRIVATE = re.compile(r'github\.com/ChironTheCentaur/openflo(?![-\w])', re.I)


def _private_links(text):
    return _PRIVATE.findall(text)


def test_the_detector_tells_private_from_public():
    # Proves the scan below can fail: it flags the private repository in
    # every form a link takes, and passes the public mirror.
    for bad in ('https://github.com/ChironTheCentaur/openflo/issues',
                'https://github.com/ChironTheCentaur/openflo#readme',
                '"https://github.com/ChironTheCentaur/openflo"',
                'git clone https://github.com/ChironTheCentaur/openflo.git'):
        assert _private_links(bad), bad
    for good in ('https://github.com/ChironTheCentaur/openflo-public/issues',
                 'https://github.com/ChironTheCentaur/openflo-public#readme'):
        assert not _private_links(good), good


def test_nothing_that_ships_links_to_the_private_repository():
    shipped = [*sorted((REPO / 'src').rglob('*.py')),
               *sorted((REPO / 'src').rglob('*.md')),
               REPO / 'pyproject.toml', REPO / 'CITATION.cff',
               REPO / 'README.md']
    found = {str(p.relative_to(REPO)): _private_links(p.read_text('utf-8'))
             for p in shipped if p.is_file()}
    assert not {k: v for k, v in found.items() if v}


def test_report_a_problem_opens_the_public_tracker(monkeypatch):
    import webbrowser
    from tkinter import messagebox

    from openflo import editor_lifecycle
    from openflo.update import DOCS_URL, ISSUES_URL
    opened, shown = [], []
    monkeypatch.setattr(webbrowser, 'open', opened.append)
    monkeypatch.setattr(messagebox, 'showinfo',
                        lambda title, msg, **kw: shown.append(msg))
    monkeypatch.setattr(editor_lifecycle, '_error_report_path',
                        lambda: str(REPO / 'no-such-report.txt'))

    class Editor(editor_lifecycle.LifecycleMixin):
        pass
    editor_lifecycle.LifecycleMixin._report_a_problem(Editor())
    assert opened == [ISSUES_URL]
    assert ISSUES_URL in shown[0]
    assert ISSUES_URL.startswith('https://github.com/ChironTheCentaur/openflo-public/')
    assert DOCS_URL.startswith('https://github.com/ChironTheCentaur/openflo-public')
