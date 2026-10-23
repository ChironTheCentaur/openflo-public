"""The bundled user guide (Complaints 27): topics read from the Markdown
files in ``openflo/guide``, which ship with the package, so Help works
offline.

Each topic file starts with ``# Title``. The renderer understands what the
guide uses: ``#`` and ``##`` headings, ``-`` bullets, numbered steps,
``**bold**`` and `` `code` ``. Menu paths are written as
``**Menu > Item**``; a test checks every one names a real menu item.

Pure Python.
"""
from __future__ import annotations

import re
from importlib import resources

_INLINE = re.compile(r'(\*\*[^*]+\*\*|`[^`]+`)')
MENU_PATH = re.compile(r'\*\*((?:File|Edit|View|Analyze|Tools|Help)'
                       r'(?: > [^*>]+)+)\*\*')


def topics():
    """``[(slug, title, text)]`` of every topic, in file-name order."""
    folder = resources.files('openflo') / 'guide'
    out = []
    for f in sorted((p for p in folder.iterdir() if p.name.endswith('.md')),
                    key=lambda p: p.name):
        text = f.read_text(encoding='utf-8')
        first = text.splitlines()[0] if text else ''
        title = first[2:].strip() if first.startswith('# ') else f.name
        out.append((f.name[:-3], title, text))
    return out


def inline(text):
    """``[(kind, text)]`` runs of one line: 'text', 'bold' or 'code'."""
    out = []
    for part in _INLINE.split(text):
        if not part:
            continue
        if part.startswith('**') and part.endswith('**') and len(part) > 4:
            out.append(('bold', part[2:-2]))
        elif part.startswith('`') and part.endswith('`') and len(part) > 2:
            out.append(('code', part[1:-1]))
        else:
            out.append(('text', part))
    return out


def blocks(text):
    """``[(kind, runs)]`` of a topic: kind 'h1', 'h2', 'bullet', 'step'
    (a numbered step, its number kept in the text) or 'para'; runs as
    inline() gives them. Lines of one paragraph are joined."""
    out = []
    para = []

    def flush():
        if para:
            out.append(('para', inline(' '.join(para))))
            para.clear()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush()
        elif line.startswith('## '):
            flush()
            out.append(('h2', inline(line[3:])))
        elif line.startswith('# '):
            flush()
            out.append(('h1', inline(line[2:])))
        elif line.startswith('- '):
            flush()
            out.append(('bullet', inline(line[2:])))
        elif re.match(r'\d+\. ', line):
            flush()
            out.append(('step', inline(line)))
        else:
            para.append(line)
    flush()
    return out


def plain(text):
    """The topic's text without Markdown marks, for searching."""
    return ' '.join(''.join(t for _k, t in runs) for _kind, runs in
                    blocks(text))


def search(query, items=None):
    """``[(slug, title, snippet)]`` of the topics holding every word of
    ``query`` (case ignored), title matches first. The snippet is the
    sentence around the first word found."""
    words = [w for w in re.findall(r'\w+', query.lower()) if w]
    if not words:
        return []
    hits = []
    for slug, title, text in (topics() if items is None else items):
        body = plain(text)
        low = body.lower()
        if not all(w in low for w in words):
            continue
        i = low.find(words[0])
        start = max(body.rfind('. ', 0, i) + 2, 0) if i > 0 else 0
        end = body.find('. ', i)
        snippet = body[start:end + 1 if end >= 0 else len(body)].strip()
        in_title = sum(w in title.lower() for w in words)
        hits.append((-in_title, slug, title, snippet))
    hits.sort(key=lambda h: h[0])
    return [(slug, title, snippet) for _s, slug, title, snippet in hits]


def menu_paths(text):
    """Every ``Menu > Item`` path the topic names, as lists of labels."""
    return [m.split(' > ') for m in MENU_PATH.findall(text)]
