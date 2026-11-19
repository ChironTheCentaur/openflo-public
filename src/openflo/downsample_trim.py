"""Replaying a saved propagate down-sample trim -- Tk-free.

The gate editor's 'Display + data' down-sampling cuts every ticked sample to
the smallest one's event count. A session records each cut (its starting
length, the floors it was cut to and the seed), so a reopened sample is cut
to the same events whatever is loaded and ticked then.

Core module: no Tk.
"""
from __future__ import annotations

from typing import Any

# The seed every propagate trim samples with. Recorded with each trim in the
# session (``trim_record_from_entry``), so a reopen replays the trim it was
# saved with even if this ever changes.
TRIM_SEED = 42


def replay_trim(frame: Any, steps: list[int], seed: int = TRIM_SEED) -> Any:
    """``frame`` (a DataFrame) cut down as the propagate trim cut it: one
    seeded subsample per step (a later, smaller floor trims the
    already-trimmed frame again). The rows picked depend only on the frame's
    length, so the same length and steps give the same events."""
    for k in steps:
        if len(frame) > k:
            frame = frame.sample(k, random_state=seed).reset_index(drop=True)
    return frame


def trim_record_from_entry(rec: object) -> dict[str, Any] | None:
    """A session entry's ``downsample_trim`` as ``{'from', 'steps', 'seed'}``,
    or None when it is missing or malformed (the sample then loads as it is
    stored, untrimmed)."""
    if not isinstance(rec, dict):
        return None
    try:
        n_from = int(rec['from'])
        steps = [int(k) for k in rec['steps']]
        seed = int(rec.get('seed', TRIM_SEED))
    except (KeyError, TypeError, ValueError):
        return None
    # Each step cuts the frame left by the one before: strictly smaller.
    if n_from <= 0 or not steps or any(
            k <= 0 or k >= prev
            for prev, k in zip([n_from] + steps[:-1], steps, strict=True)):
        return None
    return {'from': n_from, 'steps': steps, 'seed': seed}
