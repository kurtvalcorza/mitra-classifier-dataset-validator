"""
DIMER Dataset Validator — Mitra tabular classification (Custom / Other task type)
================================================================================
Validates a tabular-classification dataset zip before a Mitra fine-tuning run.

Dataset contract (format: custom):
    dataset.zip
    ├── train.csv          (required)
    ├── val.csv            (optional; else the finetuner splits off validation_split)
    └── test.csv           (optional; scored after fitting)

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

import json
import os
import sys
import traceback
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

TEMPLATE_NAME = "mitra-classifier-dataset-validator"

MIN_TRAIN_ROWS = 50        # minimum USABLE rows (non-null target), not raw rows
MITRA_ROW_LIMIT = 10_000   # upstream training-row ceiling for Mitra
MITRA_FEATURE_LIMIT = 500  # upstream feature ceiling for Mitra
MITRA_CLASS_LIMIT = 10     # upstream class ceiling for Mitra classification
MIN_ROWS_PER_CLASS = 2     # each class needs at least this many rows for a stratifiable split

# ============================================================================
# CANONICAL DATASET RESOLUTION + ARCHIVE SAFETY
# Keep this block byte-identical across the validator and finetuner containers.
# The CI parity check (scripts/check_shared.py) enforces it.
# ============================================================================
def _safe_int(value: str | None, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _safe_float(value: str | None, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


MAX_TOTAL_UNCOMPRESSED_BYTES = _safe_int(os.getenv("DIMER_MAX_UNCOMPRESSED_BYTES"), 4 * 1024**3)
MAX_COMPRESSION_RATIO = _safe_float(os.getenv("DIMER_MAX_COMPRESSION_RATIO"), 200.0)
# Per-file uncompressed-byte ceiling and a full-read row ceiling. These bound memory BEFORE
# pandas materializes a table, and — unlike the zip-bomb guard — also apply to directory-mode
# inputs. An over-limit table is rejected, never silently truncated (truncation would corrupt
# the validator's row/usable counts and the finetuner's class-preservation guarantees).
MAX_MEMBER_UNCOMPRESSED_BYTES = _safe_int(os.getenv("DIMER_MAX_MEMBER_BYTES"), 1 * 1024**3)
MAX_CSV_ROWS = _safe_int(os.getenv("DIMER_MAX_CSV_ROWS"), 5_000_000)
CSV_READ_CHUNK_ROWS = _safe_int(os.getenv("DIMER_CSV_CHUNK_ROWS"), 200_000)
_DATASET_DIR_ALIASES = {"dataset", "datasets"}


def _normalize_member(name: str) -> str | None:
    """Normalize an archive member or relative path to a canonical form, or None
    for directories/junk. Strips a single leading dataset/ or datasets/ wrapper."""
    if not name or name.endswith("/"):
        return None
    cleaned = name.replace("\\", "/").lstrip("./")
    parts = [p for p in cleaned.split("/") if p not in ("", ".")]
    if not parts:
        return None
    if len(parts) > 1 and parts[0].lower() in _DATASET_DIR_ALIASES:
        parts = parts[1:]
    return "/".join(parts)


def _assert_zip_safe(zf: zipfile.ZipFile) -> None:
    """Reject pathological archives (zip bombs, oversized expansion) before any read."""
    total = 0
    for info in zf.infolist():
        if info.is_dir():
            continue
        total += info.file_size
        if info.file_size > MAX_MEMBER_UNCOMPRESSED_BYTES:
            raise ValueError(
                f"archive member {info.filename!r} is {info.file_size:,} uncompressed bytes "
                f"(> {MAX_MEMBER_UNCOMPRESSED_BYTES:,}); refusing (per-file guard)"
            )
        if info.compress_size > 0:
            ratio = info.file_size / info.compress_size
            if ratio > MAX_COMPRESSION_RATIO:
                raise ValueError(
                    f"archive member {info.filename!r} expands {ratio:.0f}x "
                    f"(> {MAX_COMPRESSION_RATIO:.0f}); refusing (zip-bomb guard)"
                )
    if total > MAX_TOTAL_UNCOMPRESSED_BYTES:
        raise ValueError(
            f"archive expands to {total:,} bytes (> {MAX_TOTAL_UNCOMPRESSED_BYTES:,}); "
            f"refusing to load"
        )


class DatasetSource:
    """Canonical dataset reader: a zip (preferred) or an unzipped directory.

    Resolves train/val/test.csv deterministically and rejects ambiguous archives
    (two members that resolve to the same table). Streams CSV members straight into
    pandas rather than materializing whole members in memory.
    """

    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        self.archive_name: str | None = None
        self.source_type = "directory"
        self._zip: zipfile.ZipFile | None = None
        self._members: dict[str, list[str]] = {}  # normalized -> [raw member/path, ...]

        zips = sorted(dataset_dir.glob("*.zip"))
        if zips:
            self.archive_name = zips[0].name
            self.source_type = "zip"
            self._zip = zipfile.ZipFile(zips[0])
            _assert_zip_safe(self._zip)
            for raw in self._zip.namelist():
                nm = _normalize_member(raw)
                if nm:
                    self._members.setdefault(nm, []).append(raw)
        else:
            for p in sorted(dataset_dir.rglob("*")):
                if p.is_file():
                    key = str(p.relative_to(dataset_dir)).replace("\\", "/")
                    self._members.setdefault(key, []).append(str(p))

    @property
    def files(self) -> list[str]:
        return sorted(self._members)

    def has_nested_zip(self) -> bool:
        return any(f.lower().endswith(".zip") for f in self._members)

    def candidates(self, stem: str) -> list[str]:
        return [
            f for f in self.files
            if f.lower().endswith(".csv") and Path(f).stem.lower() == stem
        ]

    def duplicate_raw_count(self, stem: str) -> int:
        """Total raw members that resolve to <stem>.csv (across normalized names)."""
        return sum(len(self._members[nm]) for nm in self.candidates(stem))

    def resolve_single(self, stem: str) -> str | None:
        """Return the one normalized <stem>.csv, or None. Raise if ambiguous."""
        cands = self.candidates(stem)
        raw_total = self.duplicate_raw_count(stem)
        if len(cands) > 1 or raw_total > 1:
            raise ValueError(
                f"ambiguous dataset: multiple {stem}.csv candidates "
                f"({cands or raw_total} raw members). Put exactly one {stem}.csv at the "
                f"archive root."
            )
        return cands[0] if cands else None

    def open(self, normalized: str):
        raw = self._members[normalized][0]
        if self._zip is not None:
            return self._zip.open(raw)  # ZipExtFile — streams, no full-member read
        return open(raw, "rb")

    def _member_bytes(self, normalized: str) -> int:
        raw = self._members[normalized][0]
        if self._zip is not None:
            return self._zip.getinfo(raw).file_size
        return os.path.getsize(raw)

    def _guard_size(self, normalized: str) -> None:
        """Reject an oversized member/file before pandas materializes it. Covers directory-mode
        inputs too (the zip-bomb guard only sees archives)."""
        size = self._member_bytes(normalized)
        if size > MAX_MEMBER_UNCOMPRESSED_BYTES:
            raise ValueError(
                f"{normalized}: {size:,} uncompressed bytes exceeds the per-file limit "
                f"{MAX_MEMBER_UNCOMPRESSED_BYTES:,} (DIMER_MAX_MEMBER_BYTES); refusing to load."
            )

    def read_csv(self, normalized: str, nrows: int | None = None) -> pd.DataFrame:
        """Read a CSV member with memory bounded before materialization: a raw-byte ceiling per
        file, and — for a full read — a chunked parse that refuses to build a frame past
        MAX_CSV_ROWS rather than OOM on a hostile or accidental giant table. Rows are never
        silently dropped: an over-limit table is rejected, not truncated."""
        self._guard_size(normalized)
        if nrows is not None:
            with self.open(normalized) as handle:
                return pd.read_csv(handle, nrows=nrows)
        with self.open(normalized) as handle:
            chunks: list[pd.DataFrame] = []
            rows = 0
            for chunk in pd.read_csv(handle, chunksize=CSV_READ_CHUNK_ROWS):
                rows += len(chunk)
                if rows > MAX_CSV_ROWS:
                    raise ValueError(
                        f"{normalized}: exceeds the {MAX_CSV_ROWS:,}-row read ceiling "
                        f"(DIMER_MAX_CSV_ROWS); refusing to load the whole table into memory."
                    )
                chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
# ============================================================================
# END shared block
# ============================================================================


@dataclass
class Config:
    dataset_dir: Path
    result_path: Path
    done_callback: str
    callback_timeout: float
    max_sample_files: int
    pipeline_metadata: dict[str, Any]
    target_column: str
    drop_columns: list[str]


def load_config() -> Config:
    """Parse the DIMER_* environment. Called inside main()'s try so malformed input
    produces a structured failure result rather than an import-time crash."""
    preprocessing = json.loads(os.getenv("DIMER_PREPROCESSING_ARGS_JSON", "{}") or "{}")
    return Config(
        dataset_dir=Path(os.getenv("DIMER_DATASET_DIR", "/data/dataset")),
        result_path=Path(os.getenv("DIMER_RESULT_PATH", "/data/dataset-validations/result.json")),
        done_callback=os.getenv("DIMER_DONE_CALLBACK", "").strip(),
        callback_timeout=float(os.getenv("DIMER_CALLBACK_TIMEOUT_SECONDS", "10")),
        max_sample_files=int(os.getenv("DIMER_MAX_SAMPLE_FILES", "25")),
        pipeline_metadata=json.loads(os.getenv("DIMER_PIPELINE_METADATA_JSON", "{}") or "{}"),
        target_column=str(preprocessing.get("target_column") or "target").strip(),
        drop_columns=[
            c.strip() for c in str(preprocessing.get("drop_columns") or "").split(",") if c.strip()
        ],
    )


def log(message: str) -> None:
    print(f"[{TEMPLATE_NAME}] {message}", flush=True)


def write_result(cfg: Config, payload: dict[str, Any]) -> None:
    cfg.result_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    cfg.result_path.write_text(content, encoding="utf-8")
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


def notify_done_callback(cfg: Config) -> dict[str, Any]:
    if not cfg.done_callback:
        return {"attempted": False, "message": "DIMER_DONE_CALLBACK not set; skipping."}
    parsed = urlparse(cfg.done_callback)
    if parsed.scheme not in {"http", "https"}:
        return {"attempted": False, "message": f"Unsupported scheme: {parsed.scheme}"}
    try:
        response = requests.post(cfg.done_callback, timeout=cfg.callback_timeout)
        return {"attempted": True, "ok": response.ok, "statusCode": response.status_code}
    except requests.RequestException as exc:
        return {"attempted": True, "ok": False, "error": str(exc)}


def _build_checks(cfg: Config, source: DatasetSource) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    # `classNames` is a DIMER-mandatory metadata key (empty default; set to the real
    # labels once the target's classes are known). Present on every return path.
    meta: dict[str, Any] = {"targetColumn": cfg.target_column, "dropColumns": cfg.drop_columns, "classNames": []}

    checks.append({
        "name": "no_nested_zip",
        "successful": not source.has_nested_zip(),
        "message": (
            "No nested zip found." if not source.has_nested_zip()
            else "A .zip inside the dataset zip was found — extract it and re-upload the "
                 "CSVs at the archive root."
        ),
    })

    # Reject ambiguous archives up front (two members resolving to the same table).
    try:
        train_path = source.resolve_single("train")
        for stem in ("val", "test"):
            source.resolve_single(stem)
        ambiguous_ok, ambiguous_msg = True, "No duplicate train/val/test tables."
    except ValueError as exc:
        ambiguous_ok, ambiguous_msg, train_path = False, str(exc), None
    checks.append({"name": "no_duplicate_tables", "successful": ambiguous_ok, "message": ambiguous_msg})
    if not ambiguous_ok:
        return checks, meta

    checks.append({
        "name": "train_csv_present",
        "successful": train_path is not None,
        "message": (
            f"Found training table at {train_path}." if train_path
            else "No train.csv in the archive."
        ),
    })
    if train_path is None:
        return checks, meta

    try:
        train = source.read_csv(train_path)
    except Exception as exc:  # noqa: BLE001
        checks.append({
            "name": "train_csv_parses", "successful": False,
            "message": f"train.csv could not be parsed as CSV: {exc}",
        })
        return checks, meta
    checks.append({
        "name": "train_csv_parses", "successful": True,
        "message": f"Parsed train.csv: {len(train)} rows x {train.shape[1]} columns.",
    })

    columns = list(train.columns)
    meta["columns"] = columns
    meta["rowCount"] = int(len(train))

    has_target = cfg.target_column in columns
    checks.append({
        "name": "target_column_present", "successful": has_target,
        "message": (
            f"Target column '{cfg.target_column}' found." if has_target
            else f"Target column '{cfg.target_column}' is not in {columns}. Set the "
                 f"target_column preprocessing field to one of them, or rename the column."
        ),
    })
    if not has_target:
        return checks, meta

    target_dropped = cfg.target_column in cfg.drop_columns
    checks.append({
        "name": "target_not_dropped", "successful": not target_dropped,
        "message": (
            "Target column is not listed in drop_columns." if not target_dropped
            else f"Target column '{cfg.target_column}' also appears in drop_columns; remove it "
                 f"from drop_columns."
        ),
    })
    if target_dropped:
        return checks, meta

    # Usable rows = non-null target rows, the exact population the finetuner keeps.
    target = train[cfg.target_column]
    usable_target = target.dropna()
    usable = int(len(usable_target))
    meta["usableRowCount"] = usable
    checks.append({
        "name": "target_has_values", "successful": usable > 0,
        "message": f"Target has {usable} non-null values of {len(target)} rows.",
    })

    class_counts = usable_target.value_counts()
    train_classes = set(str(c) for c in class_counts.index)
    n_classes = int(class_counts.size)
    meta["classCount"] = n_classes
    meta["classes"] = sorted(train_classes)[:MITRA_CLASS_LIMIT + 1]
    meta["classNames"] = meta["classes"]  # DIMER-mandatory key, now the real labels
    in_range = 2 <= n_classes <= MITRA_CLASS_LIMIT
    checks.append({
        "name": "target_class_count", "successful": in_range,
        "message": (
            f"Target '{cfg.target_column}' has {n_classes} classes (2–{MITRA_CLASS_LIMIT} allowed)."
            if in_range
            else (
                f"Target '{cfg.target_column}' has only {n_classes} distinct value(s); "
                f"classification needs at least 2 classes."
                if n_classes < 2
                else f"Target '{cfg.target_column}' has {n_classes} classes; Mitra supports at "
                     f"most {MITRA_CLASS_LIMIT}. Merge rare labels or bin the target."
            )
        ),
    })

    smallest = int(class_counts.min()) if n_classes else 0
    checks.append({
        "name": "min_rows_per_class",
        "successful": n_classes >= 2 and smallest >= MIN_ROWS_PER_CLASS,
        "message": (
            f"Smallest class has {smallest} row(s) (need at least {MIN_ROWS_PER_CLASS} for a "
            f"stratified split)."
            if n_classes >= 2
            else "Cannot assess rows per class until the target has at least 2 classes."
        ),
    })

    feature_cols = [c for c in columns if c != cfg.target_column and c not in cfg.drop_columns]
    meta["featureColumnCount"] = len(feature_cols)
    checks.append({
        "name": "feature_columns_present", "successful": len(feature_cols) >= 1,
        "message": (
            f"{len(feature_cols)} feature column(s) after target and drop_columns removed."
            if feature_cols
            else "No feature columns remain after removing target and drop_columns."
        ),
    })
    within = len(feature_cols) <= MITRA_FEATURE_LIMIT
    checks.append({
        "name": "feature_limit", "successful": within,
        "message": (
            f"{len(feature_cols)} features is within Mitra's {MITRA_FEATURE_LIMIT}-feature limit."
            if within
            else f"{len(feature_cols)} features exceeds Mitra's {MITRA_FEATURE_LIMIT}-feature "
                 f"limit; drop columns or reduce the feature set."
        ),
    })

    checks.append({
        "name": "minimum_rows", "successful": usable >= MIN_TRAIN_ROWS,
        "message": f"{usable} usable rows (need at least {MIN_TRAIN_ROWS}).",
    })

    over_limit = usable > MITRA_ROW_LIMIT
    checks.append({
        "name": "row_limit_advisory", "successful": True,
        "message": (
            f"{usable} usable rows exceeds Mitra's {MITRA_ROW_LIMIT:,}-row ceiling; the "
            f"finetuner will class-preservingly sample down to it."
            if over_limit
            else f"{usable} usable rows is within Mitra's {MITRA_ROW_LIMIT:,}-row ceiling."
        ),
    })

    # val/test: columns must match train, and their labels must be a subset of train's classes.
    for stem in ("val", "test"):
        path = source.resolve_single(stem)
        if path is None:
            continue
        try:
            other = source.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            checks.append({
                "name": f"{stem}_csv_parses", "successful": False,
                "message": f"{stem}.csv could not be parsed: {exc}",
            })
            continue
        same = set(other.columns) == set(columns)
        checks.append({
            "name": f"{stem}_schema_matches_train", "successful": same,
            "message": (
                f"{stem}.csv columns match train.csv." if same
                else f"{stem}.csv columns {list(other.columns)} differ from train {columns}."
            ),
        })
        if same and cfg.target_column in other.columns:
            other_labels = other[cfg.target_column].dropna()
            usable_other = int(len(other_labels))
            checks.append({
                "name": f"{stem}_has_usable_targets", "successful": usable_other > 0,
                "message": (
                    f"{stem}.csv has {usable_other} non-null label rows."
                    if usable_other > 0
                    else f"{stem}.csv has no non-null labels; it would be scored on zero rows."
                ),
            })
            other_classes = set(str(c) for c in other_labels.unique())
            unseen = sorted(other_classes - train_classes)
            # Only meaningful when the holdout actually has labels; an empty label set is caught
            # by the usable-targets check above rather than passing vacuously here.
            checks.append({
                "name": f"{stem}_labels_subset_train",
                "successful": bool(other_classes) and not unseen,
                "message": (
                    f"{stem}.csv target labels are all present in train."
                    if other_classes and not unseen
                    else f"{stem}.csv has labels not in train: {unseen[:10]}." if unseen
                    else f"{stem}.csv has no labels to check against train."
                ),
            })

    return checks, meta


def _summarize(cfg: Config, source: DatasetSource) -> dict[str, Any]:
    files = source.files
    extensions = Counter(Path(p).suffix.lower() or "<no_extension>" for p in files)
    return {
        "fileCount": len(files),
        "extensions": dict(sorted(extensions.items())),
        "sampleFiles": files[:cfg.max_sample_files],
        "source": source.source_type,
        "archive": source.archive_name,
    }


def run(cfg: Config) -> int:
    source = DatasetSource(cfg.dataset_dir)
    try:
        checks, check_meta = _build_checks(cfg, source)
        successful = all(c["successful"] for c in checks)
        payload = {
            "successful": successful,
            "message": (
                "Tabular dataset validation succeeded." if successful
                else "Tabular dataset validation failed — see checks."
            ),
            "datasetSummary": _summarize(cfg, source),
            "checks": checks,
            "metadata": {
                "template": TEMPLATE_NAME,
                "datasetDir": str(cfg.dataset_dir),
                "resultPath": str(cfg.result_path),
                "taskType": cfg.pipeline_metadata.get("taskType", "unknown"),
                "supportedDatasetFormat": cfg.pipeline_metadata.get("supportedDatasetFormat", "custom"),
                **check_meta,
            },
        }
        write_result(cfg, payload)
        log(f"Callback: {json.dumps(notify_done_callback(cfg), sort_keys=True)}")
        return 0 if successful else 1
    finally:
        source.close()


def main() -> int:
    try:
        cfg = load_config()
    except Exception as exc:  # noqa: BLE001 - config parse failure, no cfg to persist with
        print(f"[{TEMPLATE_NAME}] configuration error: {exc}", flush=True)
        fallback = Path(os.getenv("DIMER_RESULT_PATH", "/data/dataset-validations/result.json"))
        try:
            fallback.parent.mkdir(parents=True, exist_ok=True)
            fallback.write_text(json.dumps({
                "successful": False,
                "message": "Dataset validator configuration error.",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "metadata": {"template": TEMPLATE_NAME},
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception as write_exc:  # noqa: BLE001
            print(f"[{TEMPLATE_NAME}] failed to persist config error: {write_exc}", flush=True)
        return 1
    try:
        return run(cfg)
    except Exception as exc:  # noqa: BLE001
        payload = {
            "successful": False,
            "message": "Dataset validator crashed.",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "metadata": {"template": TEMPLATE_NAME, "datasetDir": str(cfg.dataset_dir)},
        }
        try:
            write_result(cfg, payload)
            notify_done_callback(cfg)
        except Exception as write_exc:  # noqa: BLE001
            log(f"Failed to persist crash result: {write_exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
