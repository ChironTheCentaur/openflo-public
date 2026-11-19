"""scripts/us_spelling.py: users read US spelling, and one word table lists
every place a later British/European edition would relabel.

The real-tree test checks the app as it is; the rest run on synthetic text.
"""
import importlib.util
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    'us_spelling', ROOT / 'scripts' / 'us_spelling.py')
assert _spec and _spec.loader
us = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(us)


def test_the_app_text_has_no_british_spelling():
    hits = us.find(us.british_words)
    assert hits == [], '\n'.join(f'{f}:{n}: {w} -> {us.us_form(w)}' for f, n, w in hits)


def test_no_prism_ready_wording_in_app_text():
    hits = [(p.name, n) for p, reader in us.sources() for n, t in reader(p)
            if 'prism-ready' in t.lower()]
    assert hits == []


def test_each_kind_of_british_spelling_is_found():
    text = ('The colours, a behavioural note, the centre, centred plots, '
            'cancelled and labelling, cancelling, a modelled fit, normalisation and '
            'organising, analysed data, greyed out, the catalogue, a licence, '
            'learnt whilst amongst the fibres in micrometres; Self-organising. '
            'Relabelled per microlitre, under tumour cells, re-labelled.')
    assert us.british_words(text) == [
        'colours', 'behavioural', 'centre', 'centred', 'cancelled',
        'labelling', 'cancelling', 'modelled', 'normalisation', 'organising', 'analysed',
        'greyed', 'catalogue', 'licence', 'learnt', 'whilst', 'amongst',
        'fibres', 'micrometres', 'organising', 'Relabelled', 'microlitre',
        'tumour', 'labelled']


def test_us_words_and_words_spelled_the_same_are_not_british():
    text = ('colors, behavior, center, canceled, labeling, analyses of the '
            'analysis, emphasis, a model, the cancel button, gray, catalog, '
            'license, programs, colorful parameters, relabeled, microliter, '
            'tumor, preset, undercover, decent, center, '
            # a table word inside a longer token is not that word
            'xcolour')
    assert us.british_words(text) == []


def test_us_form_keeps_capitals_and_maps_inflections():
    assert us.us_form('Cancelled') == 'Canceled'
    assert us.us_form('CENTRE') == 'CENTER'
    assert us.us_form('centred') == 'centered'
    assert us.us_form('catalogued') == 'cataloged'
    assert us.us_form('normalisation') == 'normalization'
    assert us.us_form('Neighbourhoods') == 'Neighborhoods'
    assert us.us_form('Relabelled') == 'Relabeled'
    assert us.us_form('microlitres') == 'microliters'
    assert us.us_form('UNRECOGNISED') == 'UNRECOGNIZED'


def test_the_table_has_no_word_that_is_both_british_and_american():
    uk = {uk + e for uk, _us, ends in us.WORDS for e in ends}
    am = {a + e for _uk, a, ends in us.WORDS for e in ends}
    assert uk & am == set()


def test_only_text_users_read_counts():
    assert us.is_user_text('Cancelled.')
    assert us.is_user_text('cancelled.')        # a sentence's end
    assert us.is_user_text('the centre')
    assert us.is_user_text('Labelled')
    assert not us.is_user_text('cancelled')       # a state key
    assert not us.is_user_text('grey')            # a matplotlib color
    assert not us.is_user_text('Greys')           # a matplotlib colormap
    assert not us.is_user_text('CENTRE')          # a constant's value


def _tree(tmp_path, files):
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text), encoding='utf-8')
    return tmp_path


def test_python_docstrings_comments_and_keys_are_left_out(tmp_path):
    root = _tree(tmp_path, {'src/openflo/m.py': '''\
        """The colour module."""
        STATE = {'cancelled': "Export cancelled."}   # colour comment
        def f():
            """A centre docstring."""
            return f"Centre {1}"
        '''})
    assert us.find(us.british_words, root) == [
        ('src/openflo/m.py', 2, 'cancelled'), ('src/openflo/m.py', 5, 'Centre')]


# The phone folder's name comes from the script: the sdist ships this test
# but not the phone app, and a literal name would read as a file it needs.
PHONE = us.PHONE_DIR


def test_guide_and_phone_text_are_checked_and_listed(tmp_path):
    root = _tree(tmp_path, {
        'src/openflo/guide/01-a.md': '# Colour\nSet the gate color.\n',
        f'{PHONE}/src/a.ts': ("// a colour comment\n"
                              "const s = cond ? 'Save cancelled.' : `the color`;\n"),
        f'{PHONE}/index.html': '<p>Each in its own colour.</p>\n'})
    assert us.find(us.british_words, root) == [
        ('src/openflo/guide/01-a.md', 1, 'Colour'),
        (f'{PHONE}/src/a.ts', 2, 'cancelled'),
        (f'{PHONE}/index.html', 1, 'colour')]
    assert us.find(us.american_words, root) == [
        ('src/openflo/guide/01-a.md', 2, 'color'),
        (f'{PHONE}/src/a.ts', 2, 'color')]


def test_a_tree_without_the_phone_app_is_still_checked(tmp_path):
    root = _tree(tmp_path, {'src/openflo/m.py': 'X = "the centre"\n'})
    assert us.find(us.british_words, root) == [('src/openflo/m.py', 1, 'centre')]


def test_main_exit_codes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(us, 'ROOT', _tree(tmp_path, {'src/openflo/m.py': 'X = "the color"\n'}))
    assert us.main([]) == 0
    assert us.main(['--list']) == 0
    assert 'src/openflo/m.py:1: color' in capsys.readouterr().out
    (tmp_path / 'src/openflo/n.py').write_text('Y = "the colour"\n', encoding='utf-8')
    assert us.main([]) == 1
    assert 'src/openflo/n.py:1: colour -> color' in capsys.readouterr().out
