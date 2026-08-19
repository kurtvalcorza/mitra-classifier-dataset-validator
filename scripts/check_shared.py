#!/usr/bin/env python3
"""Enforce the canonical dataset-resolution + archive-safety block.

Two invariants, both CI-gated (exit non-zero on any drift):
  1. Within a repo, every file that carries the block is byte-identical (the block is duplicated
     because DIMER builds one image per repo).
  2. Across repos, the block matches EXPECTED_SHARED_BLOCK_SHA256 — the SAME constant in every
     mitra-*-{dataset-validator,finetuner,pipeline} repo. This is what makes the separate
     deployment repos provably in sync rather than merely in sync today: if any repo's block
     drifts, that repo's own CI fails here.

This one script serves both the umbrella layout (validator/validator.py + finetuner/train.py)
and the standalone container layout (validator.py or train.py) — only the files that exist
are checked.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

START = "# CANONICAL DATASET RESOLUTION + ARCHIVE SAFETY"
END = "# END shared block"

# Cross-repo pin. When the block legitimately changes, re-propagate it to every repo AND update
# this constant in every repo's scripts/check_shared.py together (they must stay identical).
EXPECTED_SHARED_BLOCK_SHA256 = "33b41e870c410457933f73bc8fbc603c00bbdc1a8e17d612164e302507dba3a7"

CANDIDATES = [
    Path("validator/validator.py"), Path("finetuner/train.py"),  # umbrella layout
    Path("validator.py"), Path("train.py"),                      # standalone layout
]


def extract_block(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"{path}: shared-block markers not found")
    return text.split(START, 1)[1].split(END, 1)[0]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    files = [f for f in CANDIDATES if (root / f).exists()]
    if not files:
        print("no shared-block file found in this repo")
        return 1
    blocks = {f: extract_block(root / f) for f in files}
    reference = next(iter(blocks.values()))
    drift = [str(f) for f, b in blocks.items() if b != reference]
    if drift:
        print(f"Shared block DRIFT within repo: {', '.join(drift)}")
        print("The canonical dataset-resolution block must be identical across containers.")
        return 1
    digest = hashlib.sha256(reference.encode("utf-8")).hexdigest()
    if digest != EXPECTED_SHARED_BLOCK_SHA256:
        print(f"Shared block SHA {digest}")
        print(f"        != pinned  {EXPECTED_SHARED_BLOCK_SHA256}")
        print("The block drifted from the other deployment repos. Re-propagate the block, then "
              "update the pinned SHA in every repo's scripts/check_shared.py together.")
        return 1
    print(f"Shared block OK in {len(files)} file(s): {len(reference.splitlines())} lines, "
          f"sha256 {digest[:12]} matches the cross-repo pin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
