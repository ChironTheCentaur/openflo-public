#!/usr/bin/env python
"""Generate the OpenFlo synthetic example dataset.

A realistic cell-differentiation time-course (CD34 → CD11b across days,
Stim vs Ctrl) plus spectral-unmixing controls and a staining panel — enough to
exercise every feature and double-check the install. Thin CLI over
``openflo.synthetic.make_dataset``.

    python scripts/make_synthetic_dataset.py --out synthetic_data

Then in the GUI drop the ``synthetic_data/diff`` folder, or run the CLI, e.g.:

    openflo-run --unmix \
        --unmix-controls synthetic_data/spectral/controls.json \
        --unmix-input    synthetic_data/spectral/mixed_sample.fcs \
        --out spectral_out
"""
from __future__ import annotations

import os
import sys

# Allow running from a source checkout without installing.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from openflo.synthetic import main  # noqa: E402

# The CLI now lives in the installed package as the `openflo-synth` console
# entry point; this script is a thin shim for source checkouts. Run either:
#   python scripts/make_synthetic_dataset.py --out synthetic_data
#   openflo-synth --out synthetic_data        (after `pip install`)

if __name__ == '__main__':
    raise SystemExit(main())
