#!/usr/bin/env python
"""Write the desktop first-run sample that ships inside the package
(``src/openflo/sample_data/openflo-sample.flowproject``): the phone
tutorial's two synthetic blood-like samples, a control and a treated one,
with their gate tree and $SPILLOVER, the control ticked for the plot so
a gated plot shows as it opens, at fewer events so the wheel stays
small. The generator is openflo.tutorial_sample, which the phone's
mobile/scripts/make_sample.py uses too.

    python scripts/make_desktop_sample.py [OUT_FILE]
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Allow running from a source checkout without installing.
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

from openflo.first_run import (  # noqa: E402
    SAMPLE_EVENTS,
    SAMPLE_FILE,
    SAMPLE_PLOTTED,
)
from openflo.tutorial_sample import write_sample  # noqa: E402

OUT = os.path.join(HERE, '..', 'src', 'openflo', 'sample_data', SAMPLE_FILE)


def main(out=OUT):
    return write_sample(out, SAMPLE_EVENTS, plotted=SAMPLE_PLOTTED)


if __name__ == '__main__':
    print(main(*sys.argv[1:2]))
