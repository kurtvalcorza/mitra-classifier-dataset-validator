"""Unit tests for the Mitra classifier dataset validator."""
from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validator as V  # noqa: E402


def _zip(tmp: Path, members: dict[str, pd.DataFrame], name: str = "d.zip") -> Path:
    p = tmp / name
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        for arc, df in members.items():
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            zf.writestr(arc, buf.getvalue())
    return p


def _train(n=60, classes=("a", "b")):
    rows = []
    for i in range(n):
        rows.append({"f1": i, "f2": i * 2, "target": classes[i % len(classes)]})
    return pd.DataFrame(rows)


def _run(tmp: Path, members: dict[str, pd.DataFrame], target="target", drop="") -> dict:
    _zip(tmp, members)
    cfg = V.Config(
        dataset_dir=tmp, result_path=tmp / "out.json", done_callback="", callback_timeout=1.0,
        max_sample_files=25, pipeline_metadata={}, target_column=target,
        drop_columns=[c.strip() for c in drop.split(",") if c.strip()],
    )
    src = V.DatasetSource(tmp)
    try:
        checks, meta = V._build_checks(cfg, src)
    finally:
        src.close()
    return {c["name"]: c["successful"] for c in checks}, meta


def test_happy_path(tmp_path):
    checks, meta = _run(tmp_path, {"train.csv": _train(60, ("a", "b", "c"))})
    assert all(checks.values()), checks
    assert meta["classCount"] == 3
    assert meta["classNames"] == ["a", "b", "c"]  # DIMER-mandatory metadata key


def test_write_result_injects_class_names(tmp_path):
    """Any payload routed through write_result (incl. the crash fallback) gets classNames."""
    import json
    out = tmp_path / "r.json"
    cfg = V.Config(
        dataset_dir=tmp_path, result_path=out, done_callback="", callback_timeout=1.0,
        max_sample_files=25, pipeline_metadata={}, target_column="target", drop_columns=[],
    )
    V.write_result(cfg, {"successful": False, "message": "crash", "metadata": {"template": "x"}})
    payload = json.loads(out.read_text())
    assert payload["metadata"]["classNames"] == []


def test_config_error_result_has_class_names(tmp_path, monkeypatch):
    """The config-error fallback path (malformed env, no cfg) still emits classNames."""
    import json
    out = tmp_path / "result.json"
    monkeypatch.setenv("DIMER_RESULT_PATH", str(out))
    monkeypatch.setenv("DIMER_PREPROCESSING_ARGS_JSON", "{not valid json")
    rc = V.main()
    assert rc == 1
    payload = json.loads(out.read_text())
    assert payload["successful"] is False
    assert payload["metadata"]["classNames"] == []


def test_duplicate_train_rejected(tmp_path):
    # two members that both resolve to train.csv (one wrapped in dataset/)
    checks, _ = _run(tmp_path, {"train.csv": _train(), "dataset/train.csv": _train()})
    assert checks["no_duplicate_tables"] is False


def test_single_class_fails(tmp_path):
    checks, _ = _run(tmp_path, {"train.csv": _train(60, ("only",))})
    assert checks["target_class_count"] is False


def test_over_ten_classes_fails(tmp_path):
    checks, _ = _run(tmp_path, {"train.csv": _train(66, tuple(f"c{i}" for i in range(11)))})
    assert checks["target_class_count"] is False


def test_null_targets_reduce_usable_rows(tmp_path):
    df = _train(60, ("a", "b"))
    df.loc[df.index[:20], "target"] = None  # 40 usable
    checks, meta = _run(tmp_path, {"train.csv": df})
    assert meta["usableRowCount"] == 40
    # 40 usable < MIN_TRAIN_ROWS(50) -> minimum_rows fails even though raw rows == 60
    assert checks["minimum_rows"] is False


def test_feature_limit(tmp_path):
    wide = pd.DataFrame({f"f{i}": range(60) for i in range(510)})
    wide["target"] = ["a" if i % 2 else "b" for i in range(60)]
    checks, _ = _run(tmp_path, {"train.csv": wide})
    assert checks["feature_limit"] is False


def test_unseen_val_label_flagged(tmp_path):
    train = _train(60, ("a", "b"))
    val = _train(20, ("a", "b")).copy()
    val.loc[val.index[0], "target"] = "z"  # unseen label
    checks, _ = _run(tmp_path, {"train.csv": train, "val.csv": val})
    assert checks["val_labels_subset_train"] is False


def test_min_rows_per_class(tmp_path):
    df = pd.DataFrame({"f1": range(60), "target": ["a"] * 59 + ["b"]})  # class b has 1 row
    checks, _ = _run(tmp_path, {"train.csv": df})
    assert checks["min_rows_per_class"] is False


def test_zip_bomb_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "MAX_COMPRESSION_RATIO", 2.0)
    big = pd.DataFrame({"f1": [0] * 5000, "target": ["a", "b"] * 2500})  # very compressible
    _zip(tmp_path, {"train.csv": big})
    with pytest.raises(ValueError):
        V.DatasetSource(tmp_path)  # highly compressible CSV -> ratio guard trips


def test_target_in_drop_columns_fails(tmp_path):
    checks, _ = _run(tmp_path, {"train.csv": _train(60, ("a", "b"))}, drop="target")
    assert checks["target_not_dropped"] is False


def test_val_with_no_usable_labels_fails(tmp_path):
    train = _train(60, ("a", "b"))
    val = _train(20, ("a", "b")).copy()
    val["target"] = None  # all labels null
    checks, _ = _run(tmp_path, {"train.csv": train, "val.csv": val})
    assert checks["val_has_usable_targets"] is False


def test_member_byte_cap_zip(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "MAX_MEMBER_UNCOMPRESSED_BYTES", 100)  # smaller than the CSV
    _zip(tmp_path, {"train.csv": _train(60, ("a", "b"))})
    with pytest.raises(ValueError):
        V.DatasetSource(tmp_path)


def test_row_ceiling_rejected_not_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "MAX_CSV_ROWS", 10)
    monkeypatch.setattr(V, "CSV_READ_CHUNK_ROWS", 4)
    _zip(tmp_path, {"train.csv": _train(60, ("a", "b"))})  # 60 rows > ceiling of 10
    src = V.DatasetSource(tmp_path)
    try:
        with pytest.raises(ValueError):
            src.read_csv("train.csv")
    finally:
        src.close()


def test_directory_mode_byte_cap(tmp_path, monkeypatch):
    # A CSV dropped in a directory (no zip) must still be size-guarded before pandas reads it.
    monkeypatch.setattr(V, "MAX_MEMBER_UNCOMPRESSED_BYTES", 50)
    _train(60, ("a", "b")).to_csv(tmp_path / "train.csv", index=False)
    src = V.DatasetSource(tmp_path)
    try:
        with pytest.raises(ValueError):
            src.read_csv("train.csv")
    finally:
        src.close()


def test_safe_parse_bad_env(monkeypatch):
    monkeypatch.setenv("DIMER_MAX_COMPRESSION_RATIO", "not-a-number")
    assert V._safe_float(os.environ.get("DIMER_MAX_COMPRESSION_RATIO"), 200.0) == 200.0


class _FakePost:
    """Records callback POST URLs and returns a minimal ok response (stands in for requests.post)."""

    def __init__(self):
        self.urls: list[str] = []

    def __call__(self, url, timeout=None):
        self.urls.append(url)
        return type("_Resp", (), {"ok": True, "status_code": 200})()


def test_config_error_still_notifies_callback(tmp_path, monkeypatch):
    """A config-parse failure must still POST the done callback, or the UI hangs at 'Validating...'."""
    fake = _FakePost()
    monkeypatch.setattr(V.requests, "post", fake)
    monkeypatch.setenv("DIMER_RESULT_PATH", str(tmp_path / "result.json"))
    monkeypatch.setenv("DIMER_DONE_CALLBACK", "http://backend/done")
    monkeypatch.setenv("DIMER_PREPROCESSING_ARGS_JSON", "{not valid json")
    assert V.main() == 1
    assert fake.urls == ["http://backend/done"]


def test_write_failure_still_notifies_callback(tmp_path, monkeypatch):
    """If result-writing raises, the done callback must still fire — it is decoupled from the write."""
    fake = _FakePost()
    monkeypatch.setattr(V.requests, "post", fake)

    def _boom(cfg, payload):
        raise OSError("disk full")

    monkeypatch.setattr(V, "write_result", _boom)
    monkeypatch.setenv("DIMER_DATASET_DIR", str(tmp_path))     # empty dir -> normal run, no train.csv
    monkeypatch.setenv("DIMER_RESULT_PATH", str(tmp_path / "result.json"))
    monkeypatch.setenv("DIMER_DONE_CALLBACK", "http://backend/done")
    assert V.main() == 1
    assert fake.urls == ["http://backend/done"]
