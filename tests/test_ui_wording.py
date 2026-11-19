"""Labels use one wording: no arrows used as words, sentence case, American
spelling, the methods' published names, no "FlowJo-style", display
names instead of settings keys, and side panel buttons that say what
they do.

Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by a
mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import re

import tests._sample_editor as shared
import tests.test_windows_nothing_hidden as sweep
from tests._editor_ui import _open_windows, _texts

editor = shared.editor
virtual_screen = shared.virtual_screen
no_dialogs = shared.no_dialogs
engines_known = shared.engines_known


# Words that keep their capital inside a sentence-case label: names.
NAMES = {'OpenFlo', 'FlowJo', 'Gating', 'PhenoGraph', 'Leiden', 'FlowSOM',
         'CytoNorm', 'AnnData', 'Markdown', 'Gaussian', 'Prism', 'Column',
         'Grouped', 'TriMap', 'PaCMAP', 'Python', 'Louvain', 'Excel',
         'Okabe', 'Ito'}


# Spellings and names the labels must not use (UI 26, 27, 29).
BANNED = (
    (re.compile(r'[Cc]olour'), "British spelling 'colour'"),
    (re.compile(r'[Oo]ptimis'), "British spelling 'optimise'"),
    (re.compile(r'FlowJo-style'), "describes a feature by another product"),
    (re.compile(r'\bPhenograph\b'), "PhenoGraph's name misspelled"),
    (re.compile(r'\b(TSNE|TRIMAP|PACMAP)\b'), "a method's name in capitals"),
    (re.compile(r'^(phenograph|leiden|flowsom|light|dark|midnight)$'),
     "a settings key shown as a choice"),
    (re.compile(r'^(k|res|max ev|seed)$'), "a settings key as a label"),
    (re.compile(r'\.\.\.'), "three dots instead of an ellipsis"),
)


def problems_in_texts(texts):
    """Arrows used as words, banned spellings and names, and Title Case in
    menus, titles and buttons; one line each."""
    out = []
    for where, kind, t in texts:
        one = t.replace('\n', ' ')
        if '→' in t:
            out.append(f"{where} {kind} {one!r}: arrow used as a word")
        for pat, why in BANNED:
            if pat.search(t):
                out.append(f"{where} {kind} {one!r}: {why}")
        if kind in ('menu', 'title', 'TButton', 'TCheckbutton',
                    'TRadiobutton'):
            head = one.split(' — ')[0]
            words = [x for x in re.split(r"[\s/(),:&+-]+", head) if x]
            words = [x for x in words if any(ch.isalpha() for ch in x)]
            for x in words[1:]:
                core = x.strip('…."\'')
                if (core[:1].isupper() and not core.isupper()
                        and not any(ch.isdigit() for ch in core)
                        and core not in NAMES):
                    out.append(f"{where} {kind} {one!r}: {core!r} in Title "
                               "Case")
    return out


def test_labels_use_one_wording(editor):
    root, ed = editor
    texts = [('editor', 'title', ed.title())]
    _texts(ed, texts, 'editor')
    for label, win in _open_windows(root, ed):
        texts.append((label, 'title', win.title()))
        _texts(win, texts, label)
    assert len(texts) > 500
    found = problems_in_texts(texts)
    assert not found, "\n".join(sorted(set(found)))


def test_the_wording_check_finds_each_old_label():
    old = [('editor', 'menu', 'Export → FlowJo .wsp…'),
           ('editor', 'menu', 'Open Recent'),
           ('editor', 'title', 'Compensation Matrix'),
           ('x', 'TRadiobutton', 'Phenograph'),
           ('x', 'TCheckbutton', 'TSNE'),
           ('x', 'choice', 'phenograph'),
           ('x', 'TLabel', 'max ev'),
           ('x', 'menu', 'Colours…'),
           ('x', 'TButton', 'Run optimisation'),
           ('x', 'TButton', 'Browse...'),
           ('x', 'TLabel', 'FlowJo-style stats table')]
    found = problems_in_texts(old)
    for _w, _k, t in old:
        assert any(repr(t.replace('\n', ' ')) in f for f in found), t
    assert not problems_in_texts([
        ('editor', 'menu', 'Export FlowJo workspace…'),
        ('x', 'TButton', 'Prism Column…'),
        ('x', 'title', 'Compensation matrix'),
        ('x', 'choice', 'PhenoGraph')])


def test_side_panel_buttons_say_what_they_do(editor):
    _root, ed = editor
    labels = {str(w.cget('text')): w for w in sweep._walk(ed)
              if w.winfo_class() == 'TButton'}
    for want in ('Delete gate', 'Delete all', 'Copy gates…', 'Populations…',
                 'Undo', 'Redo', 'Fit', 'Pipeline', 'Layout…'):
        assert want in labels, want
    for old in ('Clear', 'Clear all', 'Pops', 'Copy', '↶', '↷', '⌂'):
        assert old not in labels, old
    tips = ed.__dict__.get('_tip_texts', {})
    for w in sweep._walk(ed):            # a symbol alone needs its tip
        if w.winfo_class() != 'TButton':
            continue
        text = str(w.cget('text'))
        if text and not any(ch.isalpha() for ch in text):
            assert str(w) in tips, f"{text!r} ({w}) has no tooltip"
