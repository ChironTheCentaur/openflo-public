"""A beta tag becomes a GitHub pre-release, so the update check keeps
pointing final-version users at the last final release."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    'publish_public', ROOT / 'scripts' / 'publish_public.py')
assert _spec and _spec.loader
publish_public = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(publish_public)


def test_pre_releases_are_recognised():
    for v in ('2.7.0b1', '2.7.0a3', '3.0.0rc2', '2.7.0.dev4'):
        assert publish_public._is_prerelease(v), v


def test_final_releases_are_not_pre_releases():
    for v in ('2.6.2', '2.7.0', '10.0.1', '2.7.0.post1'):
        assert not publish_public._is_prerelease(v), v
