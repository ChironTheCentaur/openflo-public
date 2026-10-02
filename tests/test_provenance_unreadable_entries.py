"""An unreadable audit entry must not vanish from the Methods paragraph.

`methods_paragraph` output is meant to be pasted into a manuscript. It builds
that prose by walking the audit trail, and `_iter_entries` used to `continue`
past any entry that was not a dict — and to replace a non-dict `details` with
`{}`. Both are silent. The consequence is a step the user actually performed
being absent from their published description of what they did, with nothing
anywhere saying so.

The drop still happens (there is nothing to recover from a corrupt record), but
it is now reported inside the paragraph itself, where it cannot be published
without being noticed.
"""
from openflo.provenance import _iter_entries, methods_paragraph

GOOD = [
    {'action': 'sample.load', 'details': {'n_events': 1000}},
    {'action': 'transform', 'details': {'changes': {'FITC-A': 'logicle'}}},
    {'action': 'cluster', 'details': {'method': 'leiden', 'embedding': 'umap'}},
]


def test_a_clean_trail_produces_no_warning():
    text = methods_paragraph(GOOD)
    assert '[!]' not in text, 'a clean audit trail must produce clean prose'
    assert 'Leiden' in text and 'UMAP' in text


def test_a_stray_non_record_entry_is_reported():
    text = methods_paragraph([*GOOD, 'this is not a record'])
    assert '[!]' in text, (
        'an unreadable audit entry was dropped silently — the Methods text is '
        'incomplete and says nothing about it')
    assert 'may be incomplete' in text
    assert 'str' in text, 'the reason should name what was found'


def test_malformed_details_are_reported():
    """`details` as a list still yields the action, but its parameters are
    gone — so the prose built from it is silently thinner."""
    bad = [*GOOD, {'action': 'cluster', 'details': ['not', 'a', 'record']}]
    text = methods_paragraph(bad)
    assert '[!]' in text and 'details is list' in text


def test_the_count_and_the_first_reason_are_given():
    text = methods_paragraph([*GOOD, 1, 2, 3])
    assert '3 audit entries were unreadable' in text
    assert '+2 more' in text, 'only the first reason should be spelled out'


def test_singular_phrasing_for_one_bad_entry():
    text = methods_paragraph([*GOOD, None])
    assert '1 audit entry was unreadable' in text
    assert 'is NOT described' in text


def test_the_good_entries_still_produce_their_prose():
    """Reporting the problem must not cost the content that WAS readable."""
    text = methods_paragraph([*GOOD, 'junk'])
    assert 'Leiden' in text and 'UMAP' in text and 'logicle' in text.lower()


def test_iter_entries_collects_reasons_without_a_collector():
    """The `skipped` argument is optional — existing callers are unaffected."""
    out = list(_iter_entries([*GOOD, 'junk']))
    assert len(out) == len(GOOD), 'the bad entry should still be skipped'


def test_a_non_iterable_trail_is_reported_not_raised():
    skipped = []
    assert list(_iter_entries(12345, skipped=skipped)) == []
    assert skipped and 'not iterable' in skipped[0]


def test_none_trail_is_silent():
    assert list(_iter_entries(None, skipped=[])) == []
    assert '[!]' not in methods_paragraph(None)
