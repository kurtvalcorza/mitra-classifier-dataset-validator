#!/usr/bin/env python3
"""DIMER portal entrypoint for the Mitra tabular-classification validator.

The tested task implementation remains in ``validator.py``. The repo-owned contract hardening
layer is installed here before delegation so DIMER sees the versioned/coded result envelope,
canonical task identity, and validator/finetuner-shared dataset identity without duplicating ML
validation semantics.
"""
from __future__ import annotations

import sys

import contract_hardening
import validator

contract_hardening.install_validator(validator, "tabular_classification")

if __name__ == "__main__":
    sys.exit(validator.main())
