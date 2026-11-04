"""Turning a seed setting (a number, or a word or phrase) into the int every
random step uses: ``resolve_seed``.

Core module: numpy and the standard library. No Tk, and it does not import
``openflo.pipeline``.
"""

import hashlib

import numpy as np


def resolve_seed(value, default=42):
    """Turn a seed setting into an int, accepting a word or phrase as well.

    ``42`` and ``"42"`` both mean 42. Anything else — ``"pilot run 3"``,
    ``"donor A rerun"`` — is hashed to an int, so a lab can name a run
    instead of remembering a number and still get the identical answer back
    on any machine, any OS, any Python build, forever.

    That last part is why this uses blake2b rather than the obvious
    ``hash(value)``. Python randomises string hashing per process (PEP 456,
    on by default): ``hash("pilot")`` differs between two runs of
    the same script on the same machine. A seed derived that way would look
    reproducible in one session and silently stop being reproducible in the
    next — the failure would appear as unexplained drift in results, not as
    an error. A cryptographic digest has no such freedom.

    Returns an int in [0, 2**32) — the range numpy's legacy ``RandomState``
    and every downstream library accept.
    """
    if value is None:
        return int(default) % (2 ** 32)
    if isinstance(value, bytes):
        value = value.decode('utf-8', 'replace')
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value) % (2 ** 32)
    # A whole-numbered float means the number, not a phrase: JSON has no int
    # type, so a recipe or session round-tripped through it can hand us 42.0
    # where 42 was written. Hashing that would silently give a different run
    # the same settings were supposed to reproduce.
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value) % (2 ** 32)
    text = str(value).strip()
    if not text:
        return int(default) % (2 ** 32)
    try:
        return int(text) % (2 ** 32)
    except ValueError:
        pass
    try:
        as_float = float(text)
    except ValueError:
        pass
    else:
        if as_float.is_integer():
            return int(as_float) % (2 ** 32)
    digest = hashlib.blake2b(text.encode('utf-8'), digest_size=4).digest()
    return int.from_bytes(digest, 'big')


def signed32_seed(seed):
    """``seed`` folded into [0, 2**31), for a library that takes a C ``int``.

    ``resolve_seed`` returns [0, 2**32): numpy and FlowSOM take all of it,
    but leidenalg (and PhenoGraph's seeded Leiden route, which calls it)
    raises ``OverflowError`` for anything at or above 2**31. Half of all
    phrase seeds, every negative number and 2**31 itself land there. A seed
    already in range is returned unchanged, so every run that worked before
    gives the identical answer; only the boundary to leidenalg is folded,
    never the seed the other engines see.
    """
    return int(seed) % (2 ** 31)
