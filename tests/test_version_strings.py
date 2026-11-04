"""Every place that names the release names the same one.

The version lives in pyproject.toml. The README's beta note, CITATION.cff
and the newest dated CHANGELOG heading repeat it by hand, and CITATION.cff
once drifted months behind the release unnoticed. These checks
fail the moment one of them is left behind on a bump.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _project_version() -> str:
    data = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    return data['project']['version']


def newest_changelog_release(text: str) -> str:
    """The version of the first ``## [x.y.z] - date`` heading (not Unreleased)."""
    m = re.search(r'^## \[(\d+\.\d+\.\d+[^\]]*)\] - \d{4}-\d{2}-\d{2}\s*$',
                  text, re.MULTILINE)
    assert m, 'no dated release heading in CHANGELOG.md'
    return m.group(1)


def readme_versions(text: str) -> list[str]:
    """Versions the README's beta note names."""
    found = re.findall(r'\*\*Public beta \(([^)]+)\)\.\*\*', text)
    found += re.findall(r'both give (\d+\.\d+\.\d+\S*?)\.(?:\s|$)', text)
    return found


def citation_version(text: str) -> str:
    m = re.search(r'^version:\s*"?([^"\n]+)"?\s*$', text, re.MULTILINE)
    assert m, 'no version: line in CITATION.cff'
    return m.group(1).strip()


def test_newest_changelog_release_is_the_project_version():
    text = (ROOT / 'CHANGELOG.md').read_text(encoding='utf-8')
    assert newest_changelog_release(text) == _project_version()


def test_readme_beta_note_names_the_project_version():
    found = readme_versions((ROOT / 'README.md').read_text(encoding='utf-8'))
    assert len(found) == 2, (
        f'README beta note changed shape ({found}); update readme_versions')
    assert found == [_project_version()] * 2


def test_citation_names_the_project_version():
    text = (ROOT / 'CITATION.cff').read_text(encoding='utf-8')
    assert citation_version(text) == _project_version()


def test_parsers_read_the_version_the_text_states():
    # The helpers must report what the text says, or the checks above would
    # pass on any file.
    assert newest_changelog_release(
        '## [Unreleased]\n\n## [1.2.3] - 2026-01-01\n\n## [1.2.2] - 2025-12-01\n'
    ) == '1.2.3'
    assert readme_versions(
        '**Public beta (1.2.3).** ... both give 1.2.3. Report') == ['1.2.3', '1.2.3']
    assert citation_version('cff-version: 1.2.0\nversion: "0.9.1"\n') == '0.9.1'
