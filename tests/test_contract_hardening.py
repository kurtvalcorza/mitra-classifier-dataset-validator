from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contract_hardening as C  # noqa: E402


def test_directory_dataset_identity_is_path_aware_and_order_stable(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "b" / "train.csv").write_bytes(b"x,y\n1,2\n")
    (tmp_path / "a" / "val.csv").write_bytes(b"x,y\n3,4\n")
    first = C.dataset_identity(tmp_path)
    assert first and first["kind"] == "tree"
    assert first["files"] == ["a/val.csv", "b/train.csv"]
    second = C.dataset_identity(tmp_path)
    assert second == first

    (tmp_path / "a" / "val.csv").rename(tmp_path / "a" / "other.csv")
    changed = C.dataset_identity(tmp_path)
    assert changed and changed["sha256"] != first["sha256"]


def test_archive_identity_hashes_exact_selected_zip(tmp_path):
    package = tmp_path / "dataset.zip"
    with zipfile.ZipFile(package, "w") as zf:
        zf.writestr("train.csv", "x,target\n1,a\n")
    identity = C.dataset_identity(tmp_path)
    assert identity == {
        "algorithm": "sha256",
        "kind": "archive",
        "file": "dataset.zip",
        "sha256": C._sha256_file(package),
    }


def test_validator_payload_is_versioned_and_task_locked(tmp_path, monkeypatch):
    monkeypatch.setenv("DIMER_TASK_TYPE", "object_detection")
    cfg = SimpleNamespace(dataset_dir=tmp_path)
    payload = {"successful": False, "message": "bad dataset", "metadata": {}}
    C.normalize_payload(payload, task_type="tabular_regression", role="validator", cfg=cfg)
    assert payload["schemaVersion"] == 1
    assert payload["code"] == "INVALID_DATASET"
    assert payload["metadata"]["taskType"] == "tabular_regression"
    assert payload["metadata"]["platformTaskType"] == "object_detection"
    assert payload["metadata"]["classNames"] == []


def test_finetuner_payload_preserves_dataset_identity(tmp_path):
    (tmp_path / "train.csv").write_text("x,target\n1,2\n", encoding="utf-8")
    cfg = SimpleNamespace(dataset_dir=tmp_path)
    payload = {"successful": True, "metadata": {}, "provenance": {}}
    C.normalize_payload(payload, task_type="tabular_classification", role="finetuner", cfg=cfg)
    assert payload["code"] == "SUCCEEDED"
    assert payload["metadata"]["datasetSha256"] == payload["provenance"]["datasetSha256"]
    assert payload["provenance"]["dataset"]["kind"] == "tree"


def test_artifact_metadata_versions_secondary_json(tmp_path, monkeypatch):
    out = tmp_path / "out"
    (out / "artifacts").mkdir(parents=True)
    (out / "evaluation").mkdir()
    (out / "logs").mkdir()
    (out / "progress").mkdir()
    best = out / "artifacts" / "best.pt"
    with zipfile.ZipFile(best, "w") as zf:
        zf.writestr("mitra_predictor/predictor.pkl", b"fake")
    (out / "evaluation" / "report.json").write_text("{}", encoding="utf-8")
    (out / "logs" / "run-summary.json").write_text("{}", encoding="utf-8")
    (out / "progress" / "epoch_0001.json").write_text("{}", encoding="utf-8")

    cfg = SimpleNamespace(
        output_dir=out, dataset_dir=tmp_path, target_column="target", drop_columns=[],
        max_train_rows=10000, validation_split=0.2, time_limit=60, seed=0,
        eval_metric="accuracy", fine_tune=False, fine_tune_steps=0,
        run_id="r1", session_id="s1", train_device="cpu",
    )
    artifacts = {
        "modelArtifact": {"path": "fine-tuning/r1/artifacts/best.pt", "name": "best.pt", "sizeBytes": best.stat().st_size},
        "evaluationReport": {"path": "fine-tuning/r1/evaluation/report.json", "name": "report.json", "sizeBytes": 2},
        "logArtifact": {"path": "fine-tuning/r1/logs/run-summary.json", "name": "run-summary.json", "sizeBytes": 2},
    }
    monkeypatch.setattr(C, "verify_packaged_predictor", lambda worker, cfg, package: None)
    hardened = C._harden_artifacts(SimpleNamespace(), cfg, {"mode": "zero-shot"}, {"datasetSha256": "abc"}, artifacts, "tabular_classification")
    model = hardened["modelArtifact"]
    assert model["logicalFormat"] == "autogluon-tabular-predictor"
    assert model["packageFormat"] == "zip"
    assert model["sha256"] == C._sha256_file(best)
    assert model["reloadVerified"] is True
    assert json.loads((out / "evaluation" / "report.json").read_text())["schemaVersion"] == 1
    assert json.loads((out / "logs" / "run-summary.json").read_text())["schemaVersion"] == 1
    assert json.loads((out / "progress" / "epoch_0001.json").read_text())["schemaVersion"] == 1
