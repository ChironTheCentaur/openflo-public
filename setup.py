"""Build hook: ship CHANGELOG.md inside the package.

Everything else about the build is in pyproject.toml. Help > What's new reads
``openflo/CHANGELOG.md``; package-data cannot name a file outside the package,
so the build copies the repository's CHANGELOG into the built package (the
sdist carries it too, through MANIFEST.in, so a wheel built from the sdist
gets it as well).
"""
from __future__ import annotations

import os
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py

_HERE = os.path.dirname(os.path.abspath(__file__))


def copy_changelog(build_lib, root=_HERE):
    """Copy ``root``/CHANGELOG.md to ``build_lib``/openflo/CHANGELOG.md and
    return the copy's path. A missing CHANGELOG fails the build."""
    src = os.path.join(root, "CHANGELOG.md")
    if not os.path.isfile(src):
        raise FileNotFoundError(
            "CHANGELOG.md is missing; Help > What's new needs it in the "
            "package")
    dest = os.path.join(build_lib, "openflo", "CHANGELOG.md")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copyfile(src, dest)
    return dest


class BuildPyWithChangelog(build_py):
    """build_py, then copy CHANGELOG.md into the built openflo package."""

    def run(self):
        super().run()
        copy_changelog(self.build_lib)


if __name__ == "__main__":   # setuptools runs this file as __main__
    setup(cmdclass={"build_py": BuildPyWithChangelog})
