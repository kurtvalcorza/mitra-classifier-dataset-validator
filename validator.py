"""
DIMER Dataset Validator — Mitra tabular classification (Custom / Other task type)
================================================================================
Validates a tabular-classification dataset zip before a Mitra fine-tuning run.

Dataset contract (format: custom):
    dataset.zip
    ├── train.csv          (required)
    ├── val.csv            (optional; else the finetuner splits off validation_split)
    └── test.csv           (optional; scored but not required)

Each CSV is one row per training example. One column is the classification target
(default "target", overridable via the pipeline's `target_column` preprocessing
field); its distinct values are the class labels. Every other non-dropped column is
a feature.

Contract with DIMER:
  - reads the dataset from DIMER_DATASET_DIR (raw zip or unzipped dir)
  - writes result.json to DIMER_RESULT_PATH
  - POSTs DIMER_DONE_CALLBACK when finished
  - exit 0 on pass, 1 on fail
"""
from __future__ import annotations

import io
import json
import os
import sys
import traceback
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

TEMPLATE_NAME = "mitra-classifier-dataset-validator"
DATASET_DIR = Path(os.getenv("DIMER_DATASET_DIR", "/data/dataset"))
RESULT_PATH = Path(os.getenv("DIMER_RESULT_PATH", "/data/dataset-validations/result.json"))
DONE_CALLBACK = os.getenv("DIMER_DONE_CALLBACK", "").strip()
CALLBACK_TIMEOUT_SECONDS = float(os.getenv("DIMER_CALLBACK_TIMEOUT_SECONDS", "10"))
MAX_SAMPLE_FILES = int(os.getenv("DIMER_MAX_SAMPLE_FILES", "25"))
PIPELINE_METADATA = json.loads(os.getenv("DIMER_PIPELINE_METADATA_JSON", "{}") or "{}")
# Preprocessing args may or may not reach the validator depending on run stage; read
# defensively and fall back to a convention.
PREPROCESSING = json.loads(os.getenv("DIMER_PREPROCESSING_ARGS_JSON", "{}") or "{}")

TARGET_COLUMN = str(PREPROCESSING.get("target_column") or "target").strip()
DROP_COLUMNS = [
    c.strip() for c in str(PREPROCESSING.get("drop_columns") or "").split(",") if c.strip()
]
MIN_TRAIN_ROWS = 50
MITRA_ROW_LIMIT = 10_000  # upstream training-row ceiling for Mitra
MITRA_CLASS_LIMIT = 10    # upstream class ceiling for Mitra classification
MIN_ROWS_PER_CLASS = 2    # each class needs at least this many rows for a stratifiable split
CSV_SUFFIXES = {".csv"}


def log(message: str) -> None:
    print(f"[{TEMPLATE_NAME}] {message}", flush=True)


def write_result(payload: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    RESULT_PATH.write_text(content, encoding="utf-8")
    _upload_result_to_s3(content)


def _upload_result_to_s3(content: str) -> None:
    endpoint = os.getenv("S3_ENDPOINT_URL", "").strip()
    bucket = os.getenv("S3_BUCKET", "").strip()
    s3_key = os.getenv("S3_RESULT_KEY", "").strip()
    access_key = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    if not all([endpoint, bucket, s3_key, access_key, secret_key]):
        return
    try:
        import boto3

        s3 = boto3.client(
            "s3", endpoint_url=endpoint,
            aws_access_key_id=access_key, aws_secret_access_key=secret_key,
        )
        s3.put_object(Bucket=bucket, Key=s3_key, Body=content.encode("utf-8"))
        log(f"Result uploaded to s3://{bucket}/{s3_key}")
    except Exception as exc:  # noqa: BLE001 - best effort
        log(f"Failed to upload result to S3: {exc}")


def notify_done_callback() -> dict[str, Any]:
    if not DONE_CALLBACK:
        return {"attempted": False, "message": "DIMER_DONE_CALLBACK not set; skipping."}
    parsed = urlparse(DONE_CALLBACK)
    if parsed.scheme not in {"http", "https"}:
        return {"attempted": False, "message": f"Unsupported scheme: {parsed.scheme}"}
    try:
        response = requests.post(DONE_CALLBACK, timeout=CALLBACK_TIMEOUT_SECONDS)
        return {"attempted": True, "ok": response.ok, "statusCode": response.status_code}
    except requests.RequestException as exc:
        return {"attempted": True, "ok": False, "error": str(exc)}


# --- Dataset source (zip or directory), CSV-aware ---

def _normalize_zip_member(name: str) -> str | None:
    if not name or name.endswith("/"):
        return None
    normalized = name.lstrip("./")
    parts = Path(normalized).parts
    if len(parts) > 1 and parts[0].lower() in {"dataset", "datasets"}:
        normalized = str(Path(*parts[1:]))
    return normalized


class DatasetSource:
    def __init__(self) -> None:
        self.archive_name: str | None = None
        self.source_type = "directory"
        self._archive: zipfile.ZipFile | None = None
        self._file_map: dict[str, Any] = {}

        zip_files = sorted(DATASET_DIR.glob("*.zip"))
        if zip_files:
            archive_path = zip_files[0]
            self.archive_name = archive_path.name
            self.source_type = "zip"
            self._archive = zipfile.ZipFile(archive_path)
            for member in self._archive.namelist():
                normalized = _normalize_zip_member(member)
                if normalized:
                    self._file_map[normalized] = member
            return
        for path in sorted(DATASET_DIR.rglob("*")):
            if path.is_file():
                self._file_map[str(path.relative_to(DATASET_DIR))] = path

    @property
    def files(self) -> list[str]:
        return sorted(self._file_map.keys())

    def read_bytes(self, relative_path: str) -> bytes:
        target = self._file_map[relative_path]
        if self._archive is not None:
            with self._archive.open(target) as handle:
                return handle.read()
        return Path(target).read_bytes()

    def has_nested_zip(self) -> bool:
        return any(Path(f).suffix.lower() == ".zip" for f in self.files)

    def find_csv(self, stem: str) -> str | None:
        """Return the path of <stem>.csv anywhere in the archive, if present."""
        for f in self.files:
            p = Path(f)
            if p.suffix.lower() == ".csv" and p.stem.lower() == stem:
                return f
        return None

    def read_csv(self, relative_path: str, nrows: int | None = None) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(self.read_bytes(relative_path)), nrows=nrows)

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()


def _build_checks(source: DatasetSource) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = source.files
    checks: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"targetColumn": TARGET_COLUMN, "dropColumns": DROP_COLUMNS}

    checks.append({
        "name": "no_nested_zip",
        "successful": not source.has_nested_zip(),
        "message": (
            "No nested zip found."
            if not source.has_nested_zip()
            else "A .zip inside the dataset zip was found — extract it and re-upload the "
                 "CSVs at the archive root."
        ),
    })

    train_path = source.find_csv("train")
    checks.append({
        "name": "train_csv_present",
        "successful": train_path is not None,
        "message": (
            f"Found training table at {train_path}."
            if train_path else "No train.csv in the archive."
        ),
    })
    if train_path is None:
        return checks, meta

    # Parse it.
    try:
        train = source.read_csv(train_path)
    except Exception as exc:  # noqa: BLE001
        checks.append({
            "name": "train_csv_parses",
            "successful": False,
            "message": f"train.csv could not be parsed as CSV: {exc}",
        })
        return checks, meta
    checks.append({
        "name": "train_csv_parses",
        "successful": True,
        "message": f"Parsed train.csv: {len(train)} rows x {train.shape[1]} columns.",
    })

    columns = list(train.columns)
    meta["columns"] = columns
    meta["rowCount"] = int(len(train))

    has_target = TARGET_COLUMN in columns
    checks.append({
        "name": "target_column_present",
        "successful": has_target,
        "message": (
            f"Target column '{TARGET_COLUMN}' found."
            if has_target
            else f"Target column '{TARGET_COLUMN}' is not in {columns}. Set the "
                 f"target_column preprocessing field to one of them, or rename the column."
        ),
    })

    if has_target:
        target = train[TARGET_COLUMN]
        non_null = int(target.notna().sum())
        checks.append({
            "name": "target_has_values",
            "successful": non_null > 0,
            "message": f"Target has {non_null} non-null values of {len(target)} rows.",
        })

        # The distinct target values are the class labels. Mitra classifies 2–10 classes.
        class_counts = target.dropna().value_counts()
        n_classes = int(class_counts.size)
        meta["classCount"] = n_classes
        meta["classes"] = [str(c) for c in class_counts.index.tolist()[:MITRA_CLASS_LIMIT + 1]]
        in_range = 2 <= n_classes <= MITRA_CLASS_LIMIT
        checks.append({
            "name": "target_class_count",
            "successful": in_range,
            "message": (
                f"Target '{TARGET_COLUMN}' has {n_classes} classes (2–{MITRA_CLASS_LIMIT} allowed)."
                if in_range
                else (
                    f"Target '{TARGET_COLUMN}' has only {n_classes} distinct value(s); "
                    f"classification needs at least 2 classes."
                    if n_classes < 2
                    else f"Target '{TARGET_COLUMN}' has {n_classes} classes; Mitra supports at "
                         f"most {MITRA_CLASS_LIMIT}. Merge rare labels or bin the target."
                )
            ),
        })

        smallest = int(class_counts.min()) if n_classes else 0
        checks.append({
            "name": "min_rows_per_class",
            "successful": n_classes >= 2 and smallest >= MIN_ROWS_PER_CLASS,
            "message": (
                f"Smallest class has {smallest} row(s) (need at least {MIN_ROWS_PER_CLASS})."
                if n_classes >= 2
                else "Cannot assess rows per class until the target has at least 2 classes."
            ),
        })

    feature_cols = [c for c in columns if c != TARGET_COLUMN and c not in DROP_COLUMNS]
    meta["featureColumnCount"] = len(feature_cols)
    checks.append({
        "name": "feature_columns_present",
        "successful": len(feature_cols) >= 1,
        "message": (
            f"{len(feature_cols)} feature column(s) after target and drop_columns removed."
            if feature_cols
            else "No feature columns remain after removing target and drop_columns."
        ),
    })

    checks.append({
        "name": "minimum_rows",
        "successful": len(train) >= MIN_TRAIN_ROWS,
        "message": f"{len(train)} rows (need at least {MIN_TRAIN_ROWS}).",
    })

    # Advisory only — must NOT fail. The finetuner seeded-samples to the ceiling.
    over_limit = len(train) > MITRA_ROW_LIMIT
    checks.append({
        "name": "row_limit_advisory",
        "successful": True,
        "message": (
            f"{len(train)} rows exceeds Mitra's {MITRA_ROW_LIMIT:,}-row ceiling; the "
            f"finetuner will seed-sample down to it."
            if over_limit
            else f"{len(train)} rows is within Mitra's {MITRA_ROW_LIMIT:,}-row ceiling."
        ),
    })

    # If a val/test split is present, its columns must match train.
    for stem in ("val", "test"):
        path = source.find_csv(stem)
        if path is None:
            continue
        try:
            other_cols = list(source.read_csv(path, nrows=5).columns)
        except Exception as exc:  # noqa: BLE001
            checks.append({
                "name": f"{stem}_csv_parses",
                "successful": False,
                "message": f"{stem}.csv could not be parsed: {exc}",
            })
            continue
        same = set(other_cols) == set(columns)
        checks.append({
            "name": f"{stem}_schema_matches_train",
            "successful": same,
            "message": (
                f"{stem}.csv columns match train.csv."
                if same
                else f"{stem}.csv columns {other_cols} differ from train {columns}."
            ),
        })

    return checks, meta


def _summarize(source: DatasetSource) -> dict[str, Any]:
    files = source.files
    extensions = Counter(Path(p).suffix.lower() or "<no_extension>" for p in files)
    return {
        "fileCount": len(files),
        "extensions": dict(sorted(extensions.items())),
        "sampleFiles": files[:MAX_SAMPLE_FILES],
        "source": source.source_type,
        "archive": source.archive_name,
    }


def run() -> int:
    source = DatasetSource()
    try:
        checks, check_meta = _build_checks(source)
        successful = all(c["successful"] for c in checks)
        payload = {
            "successful": successful,
            "message": (
                "Tabular dataset validation succeeded." if successful
                else "Tabular dataset validation failed — see checks."
            ),
            "datasetSummary": _summarize(source),
            "checks": checks,
            "metadata": {
                "template": TEMPLATE_NAME,
                "datasetDir": str(DATASET_DIR),
                "resultPath": str(RESULT_PATH),
                "taskType": PIPELINE_METADATA.get("taskType", "unknown"),
                "supportedDatasetFormat": PIPELINE_METADATA.get(
                    "supportedDatasetFormat", "custom"
                ),
                **check_meta,
            },
        }
        write_result(payload)
        log(f"Callback: {json.dumps(notify_done_callback(), sort_keys=True)}")
        return 0 if successful else 1
    finally:
        source.close()


def main() -> int:
    try:
        return run()
    except Exception as exc:  # noqa: BLE001
        payload = {
            "successful": False,
            "message": "Dataset validator crashed.",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "metadata": {"template": TEMPLATE_NAME, "datasetDir": str(DATASET_DIR)},
        }
        try:
            write_result(payload)
            notify_done_callback()
        except Exception as write_exc:  # noqa: BLE001
            log(f"Failed to persist crash result: {write_exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
