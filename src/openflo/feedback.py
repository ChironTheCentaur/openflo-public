"""Send a bug report or an idea to the OpenFlo developers from the app.

The report goes to the website's feedback form (``FEEDBACK_URL``), which
files it as an issue in a private repository: the person needs no GitHub
account, and nothing they send is public. Each report is signed with this
app's key and the time (``signature``), as the website's path for the apps
asks. The phone app posts to the same address (mobile/src/feedback.ts),
with the same fields and limits and its own key.

The rules here mirror the website's (openflo-website lib/feedback.ts), so a
report the app accepts is one the server accepts, and the person is told
what to fix before anything leaves the computer:

  kind     'bug' or 'feature'
  app      'desktop' (the website's list: desktop, android, iphone, ...)
  title    3 to 120 characters, one line
  details  10 to 5000 characters
  version  at most 40 characters (the running OpenFlo's version)
  contact  optional reply address, at most 200 characters

Lengths are counted as the server counts them (JavaScript's string length:
UTF-16 code units), after line endings are made \\n and the ends trimmed.

Only what the person typed is sent, plus the app's version, unless they tick
"Attach the error report": then the tokenised report (paths, sample names,
user and host names and emails already replaced by tokens when it was
written, editor_lifecycle._tokenise_for_report) is added to the details.
The token key file is never read here.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

# The www host is the one the site is served from (the bare domain answers
# with a redirect); a POST is never sent through a redirect automatically.
# The apps' own path: each report is signed (below), and the website holds
# the apps to a tighter daily ceiling than its form.
FEEDBACK_URL = "https://www.getopenflo.com/api/feedback/app"
FORM_URL = "https://www.getopenflo.com/feedback"
_HOSTS = ("www.getopenflo.com", "getopenflo.com")

# Who signs: this app's name on the website's list, and its key
# (openflo-website lib/guard.ts, FEEDBACK_APP_KEYS). The key is NOT a
# secret: anyone can read it out of the app. It keeps drive-by robots off
# the website's form, and the website can shut off this one app's reports
# by dropping it. A new key goes in with a new release, the old one stays
# on the website's list until that release is out.
CLIENT = "openflo-desktop"
CLIENT_KEY = "QDzahOY4x2ruQzCairb0y7mWgBEALJEw"
# The fields a signature covers, in the website's order (a missing one is
# empty), each percent-encoded as JavaScript's encodeURIComponent does.
SIGNED_FIELDS = ('kind', 'app', 'product', 'version', 'title', 'details', 'contact')
_URI_SAFE = "-_.!~*'()"

KINDS = {'bug': "Something is wrong (bug)",
         'feature': "An idea or a missing feature"}
LIMITS = {'title': 120, 'details': 5000, 'version': 40, 'contact': 200}
MIN = {'title': 3, 'details': 10}

# What the person is told for each error code: the server's codes (it only
# ever answers with these) and 'offline' when it could not be reached.
ERRORS = {
    'kind': "Choose a bug or an idea.",
    'app': "This version of OpenFlo cannot send reports; please use the "
           "form on the website.",
    'title': "Give it a short title (3 to 120 characters).",
    'details': "Add a few words of detail (10 to 5000 characters, the "
               "attached error report included).",
    'version': "The version is too long to send.",
    'contact': "The reply address should be an email address (at most 200 "
               "characters), or leave it empty.",
    'origin': "The website refused the report. Please use the form on the "
              "website.",
    'form': "The report did not arrive. Please try again.",
    'busy': "Too many reports have been sent recently. Please try "
            "again in a few minutes.",
    'full': "Many reports have arrived today, so new ones are paused until "
            "tomorrow. Please try again then.",
    'clock': "This computer's date and time are wrong, so the website refused "
             "the report. Set the date and time, then try again.",
    'unavailable': "Reports can't be sent right now. Please try again later.",
    'offline': "The website could not be reached. Check the internet "
               "connection and try again.",
}

# One address, no spaces, a dot in the domain (the website's rule).
_EMAIL = re.compile(r'^[^\s@<>()\[\]\\,;:"`]+@[^\s@<>()\[\]\\,;:"`]+'
                    r'\.[^\s@<>()\[\]\\,;:"`]+$')

ATTACH_HEADER = "\n\n--- Error report (paths, names and emails replaced by tokens) ---\n"
CUT_NOTE = "(earlier entries left out)\n"


def js_len(text):
    """Length as JavaScript counts it (UTF-16 code units): the server's
    limits are in those."""
    return len(text.encode('utf-16-le')) // 2


def block_text(text):
    """``text`` as the server reads a many-line field: line endings made
    \\n and the ends trimmed."""
    return str(text or '').replace('\r\n', '\n').replace('\r', '\n').strip()


def _line(text):
    return re.sub(r'\s+', ' ', block_text(text))


def clean(fields):
    """The fields as the server will read them: line endings made \\n, the
    ends trimmed, and every field but the details made one line."""
    out = {k: _line(v) for k, v in fields.items() if k != 'details'}
    out['details'] = block_text(fields.get('details', ''))
    return out


def check(fields):
    """The first error code the server would give for these (cleaned)
    fields, or None when it would accept them."""
    f = clean(fields)
    if f.get('kind') not in KINDS:
        return 'kind'
    if f.get('app') != 'desktop':
        return 'app'
    if js_len(f.get('title', '')) < MIN['title']:
        return 'title'
    if js_len(f.get('details', '')) < MIN['details']:
        return 'details'
    for k in ('title', 'details', 'version', 'contact'):
        if js_len(f.get(k, '')) > LIMITS[k]:
            return k
    if f.get('contact') and not _EMAIL.match(f['contact']):
        return 'contact'
    return None


def _tail(text, room):
    """The end of ``text`` that fits in ``room`` UTF-16 units."""
    if js_len(text) <= room:
        return text
    s = text[-room:] if room > 0 else ''
    while s and js_len(s) > room:
        s = s[1:]
    return s


def with_report(details, report):
    """The details with the error report added after them, cut from the
    front (the newest errors are at the end of the file) so the whole fits
    the server's limit. Returns (text, how): how is 'none' (no report text,
    or no room left after what was typed), 'whole' or 'cut'."""
    typed = block_text(details)
    rep = block_text(report)
    if not rep:
        return typed, 'none'
    room = LIMITS['details'] - js_len(typed) - js_len(ATTACH_HEADER)
    if js_len(rep) <= room:
        return typed + ATTACH_HEADER + rep, 'whole'
    room -= js_len(CUT_NOTE)
    if room < 40:
        return typed, 'none'
    part = _tail(rep, room)
    # Start at a line, so the first one shown is not half a line.
    nl = part.find('\n')
    if 0 <= nl < len(part) - 1:
        part = part[nl + 1:]
    return typed + ATTACH_HEADER + CUT_NOTE + part, 'cut'


def signed_text(client, when, fields):
    """What the signature covers: the scheme, the app, the time and every
    signed field, one per line (openflo-website lib/guard.ts signedText)."""
    parts = ['feedback-v1', client, str(when)]
    parts += [urllib.parse.quote(str(fields.get(k, '')), safe=_URI_SAFE) for k in SIGNED_FIELDS]
    return '\n'.join(parts)


def signature(key, client, when, fields):
    """HMAC-SHA256 of ``signed_text`` with the app's key, as hex."""
    return hmac.new(key.encode('utf-8'), signed_text(client, when, fields).encode('utf-8'),
                    hashlib.sha256).hexdigest()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Hand every redirect back as an error: a POST is followed only to the
    website's own hosts, by ``send``."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _opener():
    return urllib.request.build_opener(_NoRedirect()).open


def _answer(raw):
    try:
        data = json.loads(raw.decode('utf-8'))
    except (ValueError, UnicodeDecodeError, AttributeError):
        return None
    return data if isinstance(data, dict) else None


def send(fields, open_url=None, timeout=20.0, user_agent="OpenFlo", now=time.time):
    """Post the report, signed with this app's key and the time. Returns
    None when the website took it, else an error code from ``ERRORS``.
    Never raises for a network or server failure. ``open_url(request,
    timeout=...)`` and ``now()`` stand in for urllib's and the clock in
    tests."""
    f = clean(fields)
    bad = check(f)
    if bad:
        return bad
    open_url = open_url or _opener()
    body = urllib.parse.urlencode(f).encode('utf-8')
    when = int(now())
    sig = signature(CLIENT_KEY, CLIENT, when, f)
    url = FEEDBACK_URL
    for _ in range(2):           # the address, and at most one redirect
        req = urllib.request.Request(url, data=body, method='POST', headers={
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'User-Agent': user_agent,
            'X-Feedback-Client': CLIENT,
            'X-Feedback-Time': str(when),
            'X-Feedback-Signature': sig,
        })
        try:
            with open_url(req, timeout=timeout) as resp:
                data = _answer(resp.read())
            return None if data and data.get('ok') is True else 'unavailable'
        except urllib.error.HTTPError as exc:
            if exc.code in (307, 308):
                nxt = urllib.parse.urljoin(url, exc.headers.get('Location') or '')
                parts = urllib.parse.urlsplit(nxt)
                if parts.scheme == 'https' and parts.hostname in _HOSTS and nxt != url:
                    url = nxt
                    continue
                return 'unavailable'
            try:
                data = _answer(exc.read())
            except OSError:
                data = None
            code = data.get('error') if data else None
            return code if code in ERRORS else 'unavailable'
        except (urllib.error.URLError, OSError, ValueError):
            return 'offline'
    return 'unavailable'
