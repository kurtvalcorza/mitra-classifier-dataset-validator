# Mitra ↔ DIMER Workbench contract

This repository owns the **Mitra worker side** of the integration contract. The ML implementations
remain task-specific; the shared hardening layer only standardizes the DIMER-facing wire contract.

## Contract layers

1. **Repository manifest** — `dimer-pipeline.json` is authoritative for user-facing preprocessing
   and finetuning parameters.
2. **Platform runtime** — `DIMER_*` environment inputs are documented in
   `dimer-runtime-contract.json` with consumer, requirement and evidence status.
3. **Worker output** — `result.json`, artifact descriptors and secondary JSON outputs use stable
   versioned semantics.

The exact task identities are:

- classifier: `tabular_classification`
- regressor: `tabular_regression`

Worker results always report the repo-owned task identity even if a platform-provided task label is
wrong; a conflicting platform value is retained only as diagnostic metadata.

## Dataset identity

Validator and finetuner use the same dataset-identity convention:

- when a root-level ZIP exists, hash the exact lexicographically selected archive bytes, matching
  the current `DatasetSource` selection rule;
- otherwise hash a path-framed directory tree (`DIMER-DATASET-TREE-v1`) including relative paths,
  sizes and file contents in sorted order.

Validator output exposes `metadata.datasetIdentity` / `metadata.datasetSha256`; finetuner output
exposes the same identity in provenance. The intended invariant is that the bytes validated are the
bytes trained.

Nested-ZIP behavior is **not** changed here; it remains tracked separately in the validator #8
issues and must be propagated with the existing shared-block SHA discipline.

## Result envelope

`result.json` is additive-compatible with the existing DIMER keys and now carries:

```json
{
  "schemaVersion": 1,
  "successful": true,
  "code": "...",
  "message": "human-readable diagnostics",
  "metadata": {
    "taskType": "tabular_classification|tabular_regression"
  }
}
```

Stable machine-facing codes are documented in `dimer-runtime-contract.json`. Consumers must not
parse `message` to determine state.

For validator results, `metadata.classNames` remains mandatory on every path: actual class names
for classification when known, and `[]` for regression or when unknown.

## Finetuner artifact contract

The logical model is an **AutoGluon `TabularPredictor` directory**. The DIMER compatibility package
remains:

```text
artifacts/best.pt
```

but that file is a ZIP package, **not a native PyTorch checkpoint**. The model artifact descriptor
records:

- `logicalFormat = autogluon-tabular-predictor`
- `packageFormat = zip`
- SHA-256 of the exact packaged `best.pt` bytes
- `reloadVerified = true` only after the exact package is unpacked, loaded with AutoGluon, and used
  to predict a probe row.

Secondary JSON outputs are stable schema v1:

- `evaluation/report.json`
- `logs/run-summary.json`
- `progress/epoch_*.json`

The package/export compatibility of zipped `best.pt` with DIMER itself remains the separate
finetuner #11 platform-acceptance item.

## Provenance

Successful finetuner output records the existing exact base-model revision/checksums plus:

- canonical dataset identity/digest;
- resolved preprocessing values;
- resolved finetuning values;
- requested/effective device and effective mode;
- DIMER run/session/pipeline identity when supplied;
- worker/source revision when available from runtime metadata.

## Shared-code decision

Decision: **`KEEP_PARITY_COPIES`**.

The six repositories remain independently buildable. `contract_hardening.py` and
`dimer-runtime-contract.json` are byte-pinned by `scripts/check_contract.py`; task-specific ML
logic is not moved into the shared layer.

## DIMER-side requirements — documentation only

**No DIMER/backend code is changed by this work.** The platform team still needs to address or
confirm the following external dependencies:

- first-class `tabular_classification` and `tabular_regression` task typing; unknown tasks must not
  silently become object detection;
- task-neutral generic registry defaults instead of implicit YOLO dataset/runner/artifact defaults;
- typed logical artifact support for packaged/directory models instead of assuming native `.pt`;
- backend registry derivation from or validation against repository `dimer-pipeline.json`;
- validation of known result/artifact schema versions rather than treating result fields as opaque;
- preservation/checking of validated dataset identity when starting finetuning;
- non-vision model registration/unpack/load/serving for AutoGluon `TabularPredictor` artifacts;
- confirmation of the Base Model upload/model-id transport contract (tracked separately in
  finetuner #9; `DIMER_MODEL_DIR` remains explicitly unsupported/unproven in the canonical matrix).

These are external platform requirements only and are not implementation acceptance criteria for
this repository-side change.
