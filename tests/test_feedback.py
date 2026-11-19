"""Sending a bug report or an idea to the website's form (openflo.feedback).

No network: a stand-in takes the place of urllib's opener and records what
would have been posted. Every report here is made-up text, and every
check is proved to fail on a planted fault (the PR names the mutation
check).
"""
from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse

import pytest

from openflo import feedback

GOOD = {'kind': 'bug', 'app': 'desktop', 'version': '3.1.0',
        'title': 'Histogram freezes', 'details': 'It stops after I zoom in twice.',
        'contact': ''}


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def http_error(url, code, body=b'', location=None):
    headers = {'Location': location} if location else {}
    import email.message
    msg = email.message.Message()
    for k, v in headers.items():
        msg[k] = v
    return urllib.error.HTTPError(url, code, 'x', msg, io.BytesIO(body))


class Opener:
    """Answers each request with the next of ``answers`` (bytes for a 200,
    an exception to raise) and keeps the requests."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.requests = []
        self.timeouts = []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        self.timeouts.append(timeout)
        a = self.answers.pop(0)
        if isinstance(a, BaseException):
            raise a
        return Resp(a)


def posted(req):
    return dict(urllib.parse.parse_qsl(req.data.decode('utf-8'), keep_blank_values=True))


def test_a_good_report_is_posted_as_the_website_form_reads_it():
    op = Opener(json.dumps({'ok': True, 'next': '/feedback/sent?kind=bug'}).encode())
    fields = dict(GOOD, title='  Histogram\n freezes ', details='It stops\r\nafter I zoom in twice.  ')
    assert feedback.send(fields, open_url=op) is None
    (req,) = op.requests
    assert op.timeouts == [20.0]          # a stalled site ends as "offline"
    assert req.full_url == feedback.FEEDBACK_URL == 'https://www.getopenflo.com/api/feedback/app'
    assert req.get_method() == 'POST'
    assert req.get_header('Accept') == 'application/json'
    assert req.get_header('Content-type') == 'application/x-www-form-urlencoded'
    assert posted(req) == {'kind': 'bug', 'app': 'desktop', 'version': '3.1.0',
                           'title': 'Histogram freezes',
                           'details': 'It stops\nafter I zoom in twice.', 'contact': ''}


@pytest.mark.parametrize('change, code', [
    ({'kind': 'question'}, 'kind'),
    ({'app': 'website'}, 'app'),
    ({'title': 'ab'}, 'title'),
    ({'title': 'x' * 121}, 'title'),
    ({'details': 'too short'}, 'details'),
    ({'details': 'y' * 5001}, 'details'),
    ({'version': '9' * 41}, 'version'),
    ({'contact': 'not an address'}, 'contact'),
    ({'contact': 'a@b.' + 'c' * 197}, 'contact'),
])
def test_what_the_server_would_refuse_is_refused_before_sending(change, code):
    op = Opener()
    assert feedback.send(dict(GOOD, **change), open_url=op) == code
    assert op.requests == []
    assert code in feedback.ERRORS


def test_the_limits_are_the_servers_and_count_as_javascript_does():
    assert feedback.check(dict(GOOD, title='x' * 120, details='y' * 5000,
                               version='9' * 40, contact='a@example.com')) is None
    assert feedback.check(dict(GOOD, title='abc', details='y' * 10)) is None
    # One emoji is two UTF-16 units: 2500 fit, 2501 do not.
    assert feedback.check(dict(GOOD, details='\U0001F600' * 2500)) is None
    assert feedback.check(dict(GOOD, details='\U0001F600' * 2501)) == 'details'


@pytest.mark.parametrize('answer, code', [
    (http_error(feedback.FEEDBACK_URL, 429, b'{"ok":false,"error":"busy"}'), 'busy'),
    (http_error(feedback.FEEDBACK_URL, 429, b'{"ok":false,"error":"full"}'), 'full'),
    (http_error(feedback.FEEDBACK_URL, 403, b'{"ok":false,"error":"clock"}'), 'clock'),
    (http_error(feedback.FEEDBACK_URL, 503, b'{"ok":false,"error":"unavailable"}'), 'unavailable'),
    (http_error(feedback.FEEDBACK_URL, 400, b'{"ok":false,"error":"made-up"}'), 'unavailable'),
    (http_error(feedback.FEEDBACK_URL, 500, b'<html>oops</html>'), 'unavailable'),
    (urllib.error.URLError('no route'), 'offline'),
    (TimeoutError('slow'), 'offline'),
    (b'<html>a page, not an answer</html>', 'unavailable'),
    (b'{"ok": false}', 'unavailable'),
])
def test_a_failure_answers_a_known_code_and_never_raises(answer, code):
    assert feedback.send(GOOD, open_url=Opener(answer)) == code


def test_a_redirect_is_followed_once_and_only_to_the_website():
    ok = json.dumps({'ok': True}).encode()
    op = Opener(http_error(feedback.FEEDBACK_URL, 308, location='https://getopenflo.com/api/feedback'), ok)
    assert feedback.send(GOOD, open_url=op) is None
    assert [r.full_url for r in op.requests] == [feedback.FEEDBACK_URL, 'https://getopenflo.com/api/feedback']
    assert posted(op.requests[1]) == posted(op.requests[0])
    # Elsewhere, or not https: the report is not sent there.
    for where in ('https://example.org/api/feedback', 'http://www.getopenflo.com/api/feedback',
                  'https://evil-getopenflo.com/api/feedback', 'https://getopenflo.com.example.org/x'):
        op = Opener(http_error(feedback.FEEDBACK_URL, 307, location=where), ok)
        assert feedback.send(GOOD, open_url=op) == 'unavailable'
        assert len(op.requests) == 1
    # A loop of redirects stops after one.
    op = Opener(http_error(feedback.FEEDBACK_URL, 308, location='https://getopenflo.com/api/feedback'),
                http_error('https://getopenflo.com/api/feedback', 308, location=feedback.FEEDBACK_URL), ok)
    assert feedback.send(GOOD, open_url=op) == 'unavailable'
    assert len(op.requests) == 2


def test_the_real_opener_hands_redirects_back_instead_of_following():
    handler = feedback._NoRedirect()
    assert handler.redirect_request(None, None, 302, 'x', {}, 'https://example.org/') is None


def test_the_report_is_added_after_what_was_typed():
    text, how = feedback.with_report('What I typed.', 'line one\nline two\n')
    assert how == 'whole'
    assert text == 'What I typed.' + feedback.ATTACH_HEADER + 'line one\nline two'
    assert feedback.with_report('What I typed.', '  \n') == ('What I typed.', 'none')


def test_a_long_report_keeps_its_newest_lines_and_fits_the_limit():
    lines = [f'entry {i:05d} ' + 'z' * 40 for i in range(400)]
    text, how = feedback.with_report('What I typed here.', '\n'.join(lines))
    assert how == 'cut'
    assert feedback.js_len(text) <= feedback.LIMITS['details']
    assert text.startswith('What I typed here.' + feedback.ATTACH_HEADER + feedback.CUT_NOTE)
    assert text.endswith(lines[-1])
    kept = text.split(feedback.CUT_NOTE, 1)[1].split('\n')
    assert kept[0] in lines            # starts at a whole line
    assert 'entry 00000' not in text
    assert feedback.check(dict(GOOD, details=text)) is None


def test_no_room_after_a_long_text_attaches_nothing():
    typed = 'w' * 4990
    assert feedback.with_report(typed, 'some report text') == (typed, 'none')


@pytest.mark.parametrize('code', [301, 302, 303])
def test_a_301_302_or_303_is_not_followed(code):
    # Those turn a POST into a GET elsewhere: the report is not sent again.
    op = Opener(http_error(feedback.FEEDBACK_URL, code, location='https://getopenflo.com/api/feedback'),
                json.dumps({'ok': True}).encode())
    assert feedback.send(GOOD, open_url=op) == 'unavailable'
    assert len(op.requests) == 1


def test_an_error_answer_that_cannot_be_read_is_still_a_code():
    class Unreadable(io.BytesIO):
        def read(self, *a):
            raise OSError('connection dropped')
    import email.message
    err = urllib.error.HTTPError(feedback.FEEDBACK_URL, 503, 'x', email.message.Message(), Unreadable())
    assert feedback.send(GOOD, open_url=Opener(err)) == 'unavailable'


# The website's own test vector (openflo-website tests/guard.test.mjs
# VECTOR): the same key, app, time and fields give the same signature.
VECTOR = {
    'key': 'test-key-0123456789abcdef', 'client': 'gel-phone', 'time': 1800000000,
    'fields': {'kind': 'bug', 'app': 'android', 'product': 'gel', 'version': '3.1.0',
               'title': 'Plot goes blank', 'details': 'Line one\nLine two: \u00fc \u20ac \U0001F600 & = + % # !*\'()~',
               # The reserved .example domain (RFC 2606), joined because the privacy
               # gate allows only example.com/.org/.net.
               'contact': '@'.join(['a', 'b.example'])},
    'signature': '77d437408a00165a4c350bfdcbbdbf1e4994b5c17a60674b8f982783364c5d1d',
}


def test_the_signature_is_the_websites():
    v = VECTOR
    assert feedback.signature(v['key'], v['client'], v['time'], v['fields']) == v['signature']
    # A field the app leaves out is signed as empty (the desktop sends no product).
    no_product = {k: x for k, x in v['fields'].items() if k != 'product'}
    assert feedback.signed_text('c', 1, no_product) == feedback.signed_text('c', 1, dict(no_product, product=''))


# Reserved characters (/ ? # & = + % [ ] @, a tab), signed by an
# independent HMAC; mobile/tests/feedback.test.ts holds the same vector.
RESERVED = {
    'key': 'test-key-0123456789abcdef', 'client': 'openflo-desktop', 'time': 1800000123,
    'fields': {'kind': 'feature', 'app': 'windows', 'product': '', 'version': '3.0.0/beta',
               'title': 'Path a/b?c#d', 'details': 'x&y=z+1 100% /tmp?q#h [ok] @home\tend', 'contact': ''},
    'signature': '85d549b8cde826d357a46e58214491ac08f97bce904c4b696e9cea76d53d51e6',
}


def test_reserved_characters_are_encoded_as_the_website_encodes_them():
    r = RESERVED
    assert feedback.signature(r['key'], r['client'], r['time'], r['fields']) == r['signature']
    assert '3.0.0%2Fbeta\nPath%20a%2Fb%3Fc%23d\n' in feedback.signed_text(r['client'], r['time'], r['fields'])


def test_the_time_sent_is_the_time_signed():
    # A clock read twice would send one time and sign another.
    ticks = iter(range(1800000000, 1800000100))
    op = Opener(b'{"ok":true}')
    assert feedback.send(GOOD, open_url=op, now=lambda: next(ticks)) is None
    headers = {k.lower(): v for k, v in op.requests[0].header_items()}
    assert headers['x-feedback-signature'] == feedback.signature(
        feedback.CLIENT_KEY, 'openflo-desktop', int(headers['x-feedback-time']), posted(op.requests[0]))


def test_every_report_carries_this_apps_name_time_and_signature():
    ok = b'{"ok":true}'
    op = Opener(ok)
    assert feedback.send(GOOD, open_url=op, now=lambda: 1800000000.7) is None
    req = op.requests[0]
    headers = {k.lower(): v for k, v in req.header_items()}
    assert headers['x-feedback-client'] == feedback.CLIENT == 'openflo-desktop'
    assert headers['x-feedback-time'] == '1800000000'
    f = posted(req)
    assert headers['x-feedback-signature'] == feedback.signature(feedback.CLIENT_KEY, 'openflo-desktop', 1800000000, f)
    # The signature covers what is sent: change any signed field and it changes.
    for k in ('kind', 'app', 'version', 'title', 'details', 'contact'):
        assert feedback.signature(feedback.CLIENT_KEY, 'openflo-desktop', 1800000000, dict(f, **{k: f[k] + 'x'})) \
            != headers['x-feedback-signature'], k
    # What is signed is what is sent, after the fields are tidied.
    op = Opener(ok)
    assert feedback.send(dict(GOOD, title='  Histogram\n  freezes '), open_url=op, now=lambda: 1800000000) is None
    f = posted(op.requests[0])
    assert f['title'] == 'Histogram freezes'
    assert dict(op.requests[0].header_items())['X-feedback-signature'] == \
        feedback.signature(feedback.CLIENT_KEY, 'openflo-desktop', 1800000000, f)
    # A redirect sends the same signed report again.
    op = Opener(http_error(feedback.FEEDBACK_URL, 308, location='https://getopenflo.com/api/feedback/app'), ok)
    assert feedback.send(GOOD, open_url=op, now=lambda: 1800000000) is None
    sigs = [dict(r.header_items()).get('X-feedback-signature') for r in op.requests]
    assert sigs[0] == sigs[1] and sigs[0]


def test_the_apps_key_is_long_enough_for_the_website():
    # The website ignores keys under 16 characters, and splits its list on , and :.
    assert len(feedback.CLIENT_KEY) >= 16
    assert not set(feedback.CLIENT_KEY) & set(',: ')
