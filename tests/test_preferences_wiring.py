"""Every preference must be both writable and readable.

`gpu_backend` was READ at startup and in the Preferences dialog but written by
nothing, so the portable PyTorch backend (AMD / Intel / Apple / DirectML) was
reachable only through an undocumented environment variable — a feature that
existed, was wired at one end, and could not be turned on.

This is a meta-test: it scans the source rather than exercising one setting, so
the whole class stays closed. A genuinely one-directional key must be added to
the allow-list below with a reason, which makes it a deliberate decision.
"""
import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'openflo'

# Keys that are legitimately one-directional, each with the reason.
WRITE_ONLY_OK: dict[str, str] = {}
READ_ONLY_OK: dict[str, str] = {}


def _scan():
    written, read = {}, {}
    for path in SRC.glob('*.py'):
        text = path.read_text(encoding='utf-8')
        for m in re.finditer(r"write_pref\(\s*['\"]([\w.]+)['\"]", text):
            written.setdefault(m.group(1), set()).add(path.name)
        for pat in (r"read_prefs\(\)\.get\(\s*['\"]([\w.]+)['\"]",
                    r"\bprefs\.get\(\s*['\"]([\w.]+)['\"]"):
            for m in re.finditer(pat, text):
                read.setdefault(m.group(1), set()).add(path.name)
    return written, read


def test_no_preference_is_read_but_never_written():
    """A read-only key is a setting the user can never change."""
    written, read = _scan()
    orphans = {k: sorted(v) for k, v in read.items()
               if k not in written and k not in READ_ONLY_OK}
    assert not orphans, (
        'these preferences are read but nothing ever writes them, so the user '
        f'cannot change them: {orphans}. Wire a control, or add the key to '
        'READ_ONLY_OK with a reason.')


def test_no_preference_is_written_but_never_read():
    """A write-only key is a control that silently does nothing."""
    written, read = _scan()
    orphans = {k: sorted(v) for k, v in written.items()
               if k not in read and k not in WRITE_ONLY_OK}
    assert not orphans, (
        'these preferences are written but nothing ever reads them, so the '
        f'control has no effect: {orphans}. Consume the value, or add the key '
        'to WRITE_ONLY_OK with a reason.')


def test_gpu_backend_is_reachable_from_preferences():
    """Regression for the specific gap: the backend picker must persist."""
    written, read = _scan()
    assert 'gpu_backend' in written, (
        'gpu_backend is no longer writable — the portable PyTorch backend is '
        'unreachable except through OPENFLO_GPU_BACKEND again')
    assert 'gpu_backend' in read
    assert 'ui_preferences.py' in written['gpu_backend']


def test_backend_choices_are_all_accepted_by_gpu_accel():
    """The picker must not offer a value set_backend silently ignores.

    Checked by DRIVING set_backend with each offered value and asserting the
    module actually adopted it — comparing two hardcoded lists would just be
    the same constant written twice.
    """
    from openflo import gpu_accel

    text = (SRC / 'ui_preferences.py').read_text(encoding='utf-8')
    block = text.split('_backend_labels = {', 1)[1].split('}', 1)[0]
    offered = set(re.findall(r"'(\w+)':", block))
    assert offered, 'could not parse the backend picker choices'

    original = gpu_accel._backend_pref
    try:
        for choice in offered:
            gpu_accel.set_backend(choice)
            assert gpu_accel._backend_pref == choice, (
                f'the picker offers {choice!r} but gpu_accel did not adopt it')
    finally:
        gpu_accel.set_backend(original)
