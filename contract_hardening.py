"""Repository-owned hardening layer for the DIMER worker contract.

The ML/task implementations stay in ``validator.py`` and ``train.py``. This module owns only
DIMER-facing wire concerns that must remain consistent across Mitra classifier/regressor workers:
result envelope/versioning, exact task identity, deterministic dataset identity, resolved-run
provenance, and packaged-artifact verification.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable

RESULT_SCHEMA_VERSION = 1
SECONDARY_SCHEMA_VERSION = 1
LOGICAL_MODEL_FORMAT = "autogluon-tabular-predictor"
PACKAGE_FORMAT = "zip"

VALIDATOR_SUCCESS = "VALID"
VALIDATOR_INVALID_DATASET = "INVALID_DATASET"
VALIDATOR_RUNTIME_FAILURE = "VALIDATION_FAILED"
FINETUNER_SUCCESS = "SUCCEEDED"
FINETUNER_RUNTIME_FAILURE = "TRAINING_FAILED"
INVALID_CONFIGURATION = "INVALID_CONFIGURATION"
RESOURCE_LIMIT = "RESOURCE_LIMIT"
ARTIFACT_FAILURE = "ARTIFACT_FAILED"


class ArtifactContractError(RuntimeError):
    """The produced model package cannot satisfy the repo-owned artifact contract."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_identity(dataset_dir: Path) -> dict[str, Any] | None:
    """Return the canonical identity of the dataset bytes the workers resolve.

    DatasetSource prefers the lexicographically first root-level ZIP when one exists, so archive
    identity follows that same rule and hashes the exact archive bytes. Directory mode hashes a
    framed, path-aware tree so traversal order and filename/content concatenation cannot collide.
    """
    root = Path(dataset_dir)
    zips = sorted(root.glob("*.zip"))
    if zips:
        selected = zips[0]
        return {
            "algorithm": "sha256",
            "kind": "archive",
            "file": selected.name,
            "sha256": _sha256_file(selected),
        }

    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        return None
    h = hashlib.sha256()
    h.update(b"DIMER-DATASET-TREE-v1\0")
    rels: list[str] = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        rel_bytes = rel.encode("utf-8")
        size = path.stat().st_size
        h.update(len(rel_bytes).to_bytes(8, "big"))
        h.update(rel_bytes)
        h.update(size.to_bytes(8, "big"))
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        rels.append(rel)
    return {
        "algorithm": "sha256",
        "kind": "tree",
        "files": rels,
        "sha256": h.hexdigest(),
    }


def _resource_limit_error(exc: Exception) -> bool:
    if isinstance(exc, MemoryError):
        return True
    text = str(exc).lower()
    markers = (
        "zip-bomb guard",
        "per-file guard",
        "uncompressed bytes exceeds",
        "archive expands to",
        "row read ceiling",
        "row ceiling",
        "refusing to load the whole table",
    )
    return any(marker in text for marker in markers)


def _failure_code(role: str, cfg: Any | None, exc: Exception) -> str:
    if cfg is None:
        return INVALID_CONFIGURATION
    if isinstance(exc, ArtifactContractError):
        return ARTIFACT_FAILURE
    if _resource_limit_error(exc):
        return RESOURCE_LIMIT
    return VALIDATOR_RUNTIME_FAILURE if role == "validator" else FINETUNER_RUNTIME_FAILURE


def _result_path(worker: Any, cfg: Any | None, role: str) -> Path:
    if cfg is not None:
        return Path(cfg.result_path)
    default = "/data/dataset-validations/result.json" if role == "validator" else "/data/results/result.json"
    return Path(os.getenv("DIMER_RESULT_PATH", default))


def _task_metadata(payload: dict[str, Any], task_type: str, role: str) -> dict[str, Any]:
    metadata = payload.setdefault("metadata", {})
    platform_task = os.getenv("DIMER_TASK_TYPE", "").strip()
    if platform_task and platform_task != task_type:
        metadata.setdefault("platformTaskType", platform_task)
    metadata["taskType"] = task_type
    pipeline_id = os.getenv("DIMER_PIPELINE_ID", "").strip()
    if pipeline_id:
        metadata.setdefault("pipelineId", pipeline_id)
    if role == "validator":
        metadata.setdefault("classNames", [])
    return metadata


def normalize_payload(
    payload: dict[str, Any],
    *,
    task_type: str,
    role: str,
    cfg: Any | None = None,
    forced_code: str | None = None,
) -> dict[str, Any]:
    """Apply the stable additive result envelope without removing existing DIMER-consumed keys."""
    payload["schemaVersion"] = RESULT_SCHEMA_VERSION
    successful = bool(payload.get("successful"))
    if forced_code is not None:
        payload["code"] = forced_code
    elif successful:
        payload["code"] = VALIDATOR_SUCCESS if role == "validator" else FINETUNER_SUCCESS
    elif "error" in payload:
        payload["code"] = VALIDATOR_RUNTIME_FAILURE if role == "validator" else FINETUNER_RUNTIME_FAILURE
    else:
        payload["code"] = VALIDATOR_INVALID_DATASET if role == "validator" else FINETUNER_RUNTIME_FAILURE

    metadata = _task_metadata(payload, task_type, role)
    dataset_dir = Path(cfg.dataset_dir) if cfg is not None else Path(os.getenv("DIMER_DATASET_DIR", "/data/dataset"))
    try:
        identity = dataset_identity(dataset_dir)
    except Exception:  # identity is diagnostic; never mask the original result
        identity = None
    if identity:
        metadata.setdefault("datasetIdentity", identity)
        metadata.setdefault("datasetSha256", identity["sha256"])
        if role == "finetuner":
            provenance = payload.setdefault("provenance", {})
            provenance.setdefault("dataset", identity)
            provenance.setdefault("datasetSha256", identity["sha256"])
    return payload


def _rewrite_result_file(worker: Any, cfg: Any | None, task_type: str, role: str, code: str) -> None:
    path = _result_path(worker, cfg, role)
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        normalize_payload(payload, task_type=task_type, role=role, cfg=cfg, forced_code=code)
        content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        path.write_text(content, encoding="utf-8")
        if role == "validator" and hasattr(worker, "_upload_result_to_s3"):
            worker._upload_result_to_s3(content)
    except Exception:
        return


def _augment_provenance(cfg: Any, metrics: dict[str, Any], provenance: dict[str, Any], task_type: str) -> None:
    identity = dataset_identity(Path(cfg.dataset_dir))
    if identity:
        provenance["dataset"] = identity
        provenance["datasetSha256"] = identity["sha256"]
    provenance["taskType"] = task_type
    provenance["resolvedConfiguration"] = {
        "preprocessing": {
            "target_column": cfg.target_column,
            "drop_columns": list(cfg.drop_columns),
            "max_train_rows": cfg.max_train_rows,
            "validation_split": cfg.validation_split,
        },
        "finetuning": {
            "time_limit_seconds": cfg.time_limit,
            "seed": cfg.seed,
            "eval_metric": cfg.eval_metric,
            "fine_tune": cfg.fine_tune,
            "fine_tune_steps": cfg.fine_tune_steps,
        },
    }
    provenance["execution"] = {
        "runId": getattr(cfg, "run_id", "") or None,
        "sessionId": getattr(cfg, "session_id", "") or None,
        "pipelineId": os.getenv("DIMER_PIPELINE_ID", "").strip() or None,
        "requestedDevice": getattr(cfg, "train_device", None),
        "effectiveDevice": metrics.get("resolvedDevice") or metrics.get("device"),
        "effectiveMode": metrics.get("mode"),
        "sourceRevision": (
            os.getenv("DIMER_WORKER_SOURCE_REVISION", "").strip()
            or os.getenv("GIT_COMMIT_SHA", "").strip()
            or os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION", "").strip()
            or None
        ),
    }


def _probe_frame(worker: Any, cfg: Any):
    source = worker.DatasetSource(Path(cfg.dataset_dir))
    try:
        train_path = source.resolve_single("train")
        if train_path is None:
            raise ArtifactContractError("cannot verify packaged predictor: train.csv is unavailable")
        frame = source.read_csv(train_path, nrows=1)
    finally:
        source.close()
    if frame.empty:
        raise ArtifactContractError("cannot verify packaged predictor: train.csv has no rows")
    drop = [c for c in getattr(cfg, "drop_columns", []) if c != cfg.target_column]
    return frame.drop(columns=drop + [cfg.target_column], errors="ignore")


def verify_packaged_predictor(worker: Any, cfg: Any, package: Path) -> None:
    """Unpack, reload and predict from the exact DIMER-facing package bytes."""
    if not package.exists() or not zipfile.is_zipfile(package):
        raise ArtifactContractError(f"model artifact {package} is not a readable ZIP package")
    probe = _probe_frame(worker, cfg)
    try:
        from autogluon.tabular import TabularPredictor
    except Exception as exc:  # pragma: no cover - exercised by real-stack integration
        raise ArtifactContractError(f"AutoGluon unavailable for artifact reload verification: {exc}") from exc

    with tempfile.TemporaryDirectory(prefix="dimer-artifact-verify-") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(package) as zf:
            zf.extractall(root)
        predictor_dir = root / "mitra_predictor"
        if not predictor_dir.is_dir():
            raise ArtifactContractError("model package does not contain mitra_predictor/")
        try:
            predictor = TabularPredictor.load(str(predictor_dir))
            predictions = predictor.predict(probe)
        except Exception as exc:  # pragma: no cover - exercised by real-stack integration
            raise ArtifactContractError(f"packaged predictor reload/predict failed: {exc}") from exc
        if len(predictions) != len(probe):
            raise ArtifactContractError(
                f"packaged predictor returned {len(predictions)} prediction(s) for {len(probe)} row(s)"
            )


def _version_json(path: Path, additions: dict[str, Any] | None = None) -> None:
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schemaVersion"] = SECONDARY_SCHEMA_VERSION
    if additions:
        for key, value in additions.items():
            payload.setdefault(key, value)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _harden_artifacts(worker: Any, cfg: Any, metrics: dict[str, Any], provenance: dict[str, Any], artifacts: dict[str, Any], task_type: str) -> dict[str, Any]:
    best = Path(cfg.output_dir) / "artifacts" / "best.pt"
    model = artifacts.setdefault("modelArtifact", {})
    model.update({
        "logicalFormat": LOGICAL_MODEL_FORMAT,
        "packageFormat": PACKAGE_FORMAT,
        "sha256": _sha256_file(best),
        "isNativePyTorchCheckpoint": False,
    })
    verify_packaged_predictor(worker, cfg, best)
    model["reloadVerified"] = True
    model["verification"] = "unpack-reload-predict"

    summary_additions = {
        "taskType": task_type,
        "pipelineId": os.getenv("DIMER_PIPELINE_ID", "").strip() or None,
        "datasetSha256": provenance.get("datasetSha256"),
        "resolvedConfiguration": provenance.get("resolvedConfiguration"),
        "artifact": model,
    }
    _version_json(Path(cfg.output_dir) / "evaluation" / "report.json", {"modelArtifact": model})
    _version_json(Path(cfg.output_dir) / "logs" / "run-summary.json", summary_additions)
    _version_json(Path(cfg.output_dir) / "progress" / "epoch_0001.json")

    for key, rel in (
        ("evaluationReport", Path("evaluation") / "report.json"),
        ("logArtifact", Path("logs") / "run-summary.json"),
    ):
        path = Path(cfg.output_dir) / rel
        if key in artifacts and path.exists():
            artifacts[key]["sizeBytes"] = path.stat().st_size
            artifacts[key]["sha256"] = _sha256_file(path)
    return artifacts


def install_validator(worker: Any, task_type: str) -> None:
    if getattr(worker, "_DIMER_CONTRACT_HARDENING", False):
        return
    worker._DIMER_CONTRACT_HARDENING = True

    original_write: Callable[..., Any] = worker.write_result

    def write_result(cfg: Any, payload: dict[str, Any]) -> Any:
        normalize_payload(payload, task_type=task_type, role="validator", cfg=cfg)
        return original_write(cfg, payload)

    worker.write_result = write_result

    original_persist: Callable[..., Any] = worker._persist_failure

    def persist_failure(cfg: Any | None, exc: Exception) -> Any:
        result = original_persist(cfg, exc)
        _rewrite_result_file(worker, cfg, task_type, "validator", _failure_code("validator", cfg, exc))
        return result

    worker._persist_failure = persist_failure


def install_finetuner(worker: Any, task_type: str) -> None:
    if getattr(worker, "_DIMER_CONTRACT_HARDENING", False):
        return
    worker._DIMER_CONTRACT_HARDENING = True

    # Preserve the existing public provenance key while replacing its directory-mode digest
    # with the same path-framed algorithm used by the validator.
    worker._dataset_sha256 = lambda cfg: dataset_identity(Path(cfg.dataset_dir))

    original_package: Callable[..., Any] = worker._package_artifacts

    def package_artifacts(cfg: Any, metrics: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
        _augment_provenance(cfg, metrics, provenance, task_type)
        artifacts = original_package(cfg, metrics, provenance)
        return _harden_artifacts(worker, cfg, metrics, provenance, artifacts, task_type)

    worker._package_artifacts = package_artifacts

    original_write: Callable[..., Any] = worker.write_result

    def write_result(cfg: Any, payload: dict[str, Any]) -> Any:
        normalize_payload(payload, task_type=task_type, role="finetuner", cfg=cfg)
        return original_write(cfg, payload)

    worker.write_result = write_result

    original_persist: Callable[..., Any] = worker._persist_failure

    def persist_failure(cfg: Any | None, exc: Exception) -> Any:
        result = original_persist(cfg, exc)
        _rewrite_result_file(worker, cfg, task_type, "finetuner", _failure_code("finetuner", cfg, exc))
        return result

    worker._persist_failure = persist_failure
