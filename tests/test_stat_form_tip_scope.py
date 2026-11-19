"""The GeoMean / rCV form preference says it is global (scope audit W3).

The forms are one global choice, not saved with a session (Sky's call), and
changing one changes numbers already shown: an open Statistics window,
layout stats boxes and exports on their next redraw, and a session reopened
later. The tip used to say "Applies to statistics computed from now on",
which reads as if earlier numbers stayed as they were.
"""
from tests.conftest import editor_or_skip


def _stat_tips(monkeypatch):
    from openflo.ui_preferences import PreferencesDialog
    root, ed = editor_or_skip()
    tips = {}
    monkeypatch.setattr(ed, '_tip', lambda w, t: tips.__setitem__(w, t),
                        raising=False)
    dlg = PreferencesDialog(ed)
    root.update()
    found = {key: t for t in tips.values() if isinstance(t, str)
             for key, head in (('geomean', 'How the GeoMean column'),
                               ('rcv', 'How the robust spread'))
             if t.startswith(head)}
    return root, dlg, found


def test_both_stat_form_tips_say_the_choice_is_global(monkeypatch):
    root, dlg, found = _stat_tips(monkeypatch)
    try:
        assert set(found) == {'geomean', 'rcv'}
        for key, text in found.items():
            assert 'global and not saved with a session' in text, key
            assert 'already on screen' in text, key
            assert 'layout stats boxes and exports' in text, key
            assert 'from now on' not in text, key
    finally:
        dlg.destroy()
        root.destroy()


def test_the_scope_sentence_is_one_shared_text():
    """Both tips end with the same sentence, kept in one place."""
    from openflo import ui_preferences
    text = ui_preferences.STAT_FORM_SCOPE
    assert text.startswith(' This choice is global')
    assert 'redraw' in text and 'opened later' in text
