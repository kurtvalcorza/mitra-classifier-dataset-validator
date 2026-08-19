# Issue #1 — acceptance record (classification dataset validator)

Traceability for *Harden tabular-classification dataset validation for DIMER*. Each acceptance
criterion maps to the code and test that satisfy it on `main`. Verified 2026-08-19 (validator
suite green).

| Acceptance criterion | Status | Evidence |
|---|---|---|
| Validator and finetuner share the same deterministic split-selection rules | ✅ | Shared `DatasetSource` block byte-identical across `validator.py` + the finetuner's `train.py`, enforced by `scripts/check_shared.py` against a cross-repo pinned SHA |
| Duplicate/ambiguous datasets are rejected | ✅ | `DatasetSource.resolve_single` raises; `no_duplicate_tables` + `no_nested_zip` checks |
| Minimum row checks use usable target rows | ✅ | `_usable_target_mask` (non-null) drives `minimum_rows` |
| Class-count constraints are validated before finetuning | ✅ | ≥2 classes and per-class sufficiency checked; Mitra 10-class ceiling enforced |
| Mitra feature limits are enforced | ✅ | `feature_limit` check (≤500) with `MITRA_FEATURE_LIMIT` |
| Archive/resource limits produce structured DIMER failures | ✅ | `_assert_zip_safe` (compression ratio, per-member + total uncompressed bytes) plus per-file byte ceiling and chunked-read row ceiling **before pandas** (also covers directory inputs); `test_member_byte_cap_zip`, `test_row_ceiling_rejected_not_truncated`, `test_directory_mode_byte_cap` |
| Malformed runtime config still yields `result.json` + callback attempt | ✅ | `load_config()` parses inside `main()`'s protected path; failure writes `result.json` and attempts the callback |
| CI covers the validator contract | ✅ | `.github/workflows/ci.yml`: compile + `scripts/check_shared.py` + `pytest` |

**Result contract** (`metadata` block): resolved columns, train row count, usable target
counts, feature count, class names/distribution, warnings vs blocking failures, and archive/
source metadata.

No open items: every criterion is satisfied in-repo. `Closes #1`.
