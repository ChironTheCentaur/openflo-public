"""Who publishes OpenFlo: one place for the publisher's legal name.

OpenFlo stays the product name; the company below publishes it and holds the
copyright. The desktop About window, the Windows installer's publisher and
copyright fields, LICENSE.txt and the phone app's Settings all carry this
name; tests/test_publisher_name.py and mobile/tests/test_publisher.py
check they agree.
"""
from __future__ import annotations

PUBLISHER = "Pocket Lab Essentials LLC"
# The year OpenFlo was first published under this copyright line.
COPYRIGHT_YEAR = 2026
COPYRIGHT = f"Copyright (c) {COPYRIGHT_YEAR} {PUBLISHER}"
