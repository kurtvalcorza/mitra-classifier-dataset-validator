#!/usr/bin/env python3
"""DIMER portal entrypoint for the Mitra tabular-classification validator.

The DIMER Pipeline Builder builds this repository from its root and launches the
validation container by the portal naming convention (``validate.py``). The
tested implementation lives in ``validator.py``; this thin shim only delegates to
it so that ``validator.py``, its unit tests, and the cross-repo shared block stay
byte-identical. Keep this file free of logic.
"""
from __future__ import annotations

import sys

from validator import main

if __name__ == "__main__":
    sys.exit(main())
