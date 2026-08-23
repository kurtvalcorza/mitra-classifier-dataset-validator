#!/usr/bin/env python3
"""CI gate for the repo-owned Mitra/DIMER contract hardening layer."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

EXPECTED_HARDENING_SHA256 = "d694532cc23366c119e975cf67ba51f3d85f722ae1c0d1711b4071a1cca09393"
EXPECTED_CONTRACT_SHA256 = "d2dc62fe4f0437941a29140c35e083f39e0eba9db87561d3bda6a36e208d8bb9"
REQUIRED_CODES = {
    "VALID", "INVALID_DATASET", "VALIDATION_FAILED", "SUCCEEDED",
    "INVALID_CONFIGURATION", "RESOURCE_LIMIT", "TRAINING_FAILED", "ARTIFACT_FAILED",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    contract = root / "dimer-runtime-contract.json"
    if not contract.exists():
        print("missing dimer-runtime-contract.json")
        return 1
    if sha(contract) != EXPECTED_CONTRACT_SHA256:
        print(f"runtime contract drifted: {sha(contract)} != {EXPECTED_CONTRACT_SHA256}")
        return 1
    data = json.loads(contract.read_text(encoding="utf-8"))
    if data.get("schemaVersion") != 1 or data.get("sharedCodeStrategy") != "KEEP_PARITY_COPIES":
        print("runtime contract version/shared-code strategy mismatch")
        return 1
    codes = set(data.get("resultContract", {}).get("stableCodes", []))
    if not REQUIRED_CODES.issubset(codes):
        print(f"missing stable result codes: {sorted(REQUIRED_CODES - codes)}")
        return 1
    model_dir = next((x for x in data.get("runtimeInputs", []) if x.get("name") == "DIMER_MODEL_DIR"), None)
    if not model_dir or model_dir.get("requirement") != "unsupported":
        print("DIMER_MODEL_DIR must remain explicitly unsupported pending the separate #9 contract proof")
        return 1

    candidates = [
        root / "contract_hardening.py",
        root / "validator" / "contract_hardening.py",
        root / "finetuner" / "contract_hardening.py",
    ]
    modules = [p for p in candidates if p.exists()]
    if not modules:
        print("missing contract_hardening.py")
        return 1
    for path in modules:
        digest = sha(path)
        if digest != EXPECTED_HARDENING_SHA256:
            print(f"{path.relative_to(root)} drifted: {digest} != {EXPECTED_HARDENING_SHA256}")
            return 1

    manifests = list(root.glob("dimer-pipeline.json")) + list(root.glob("finetuner/dimer-pipeline.json"))
    expected_task = "tabular_regression" if "regressor" in root.name else "tabular_classification"
    for manifest in manifests:
        doc = json.loads(manifest.read_text(encoding="utf-8"))
        if doc.get("taskType") != expected_task:
            print(f"{manifest.relative_to(root)} taskType={doc.get('taskType')!r}, expected {expected_task!r}")
            return 1

    print(
        f"Contract OK: {len(modules)} hardening module(s), task={expected_task}, "
        f"result schema v1, pinned runtime contract."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
