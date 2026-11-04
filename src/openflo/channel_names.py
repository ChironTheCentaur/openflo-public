"""What a channel name says about the parameter it holds.

One answer for "is this the acquisition-time channel?", so every module
agrees on names like ``Time (s)`` and ``TIMESTAMP``.
"""

from __future__ import annotations

import re

__all__ = ['TIME_PATTERN', 'is_time_channel']

# TIME with no letter or digit right before it: Time, TIME, Time (s),
# TIMESTAMP, HDR-T Time; not LIFETIME or ACQTIME.
TIME_PATTERN = re.compile(r'(?<![A-Z0-9])TIME')


def is_time_channel(name: object) -> bool:
    """True for an acquisition-time parameter, in any case."""
    return bool(TIME_PATTERN.search(str(name).upper()))
