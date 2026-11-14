"""Nested (zip-in-zip) uploads: one level is unwrapped, anything deeper or mixed is rejected."""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validator as V  # noqa: E402


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for arc, data in members.items():
            zf.writestr(arc, data)
    return buf.getvalue()


def _upload(tmp: Path, members: dict[str, bytes]) -> Path:
    p = tmp / "upload.zip"
    p.write_bytes(_zip_bytes(members))
    return p


def _train(n: int = 60) -> pd.DataFrame:
    return pd.DataFrame({"f1": range(n), "f2": [i * 2 for i in range(n)],
                         "target": [("a", "b")[i % 2] if V.TEMPLATE_NAME.startswith("mitra-classifier")
                                    else float(i) for i in range(n)]})


def _checks(tmp: Path, target: str = "target") -> tuple[dict[str, bool], dict[str, str], V.DatasetSource]:
    cfg = V.Config(
        dataset_dir=tmp, result_path=tmp / "out.json", done_callback="", callback_timeout=1.0,
        max_sample_files=25, pipeline_metadata={}, target_column=target, drop_columns=[],
    )
    src = V.DatasetSource(tmp)
    try:
        checks, _ = V._build_checks(cfg, src)
    finally:
        src.close()
    return ({c["name"]: c["successful"] for c in checks},
            {c["name"]: c["message"] for c in checks}, src)


def test_single_inner_zip_is_unwrapped_and_validated(tmp_path):
    inner = _zip_bytes({"train.csv": _csv_bytes(_train()), "val.csv": _csv_bytes(_train(20))})
    _upload(tmp_path, {"dataset.zip": inner})
    ok, _, src = _checks(tmp_path)
    assert all(ok.values()), ok
    assert src.unwrapped_from == "dataset.zip"
    assert src.files == ["train.csv", "val.csv"]


def test_single_inner_zip_under_wrapper_folder_is_unwrapped(tmp_path):
    inner = _zip_bytes({"train.csv": _csv_bytes(_train())})
    _upload(tmp_path, {"dataset/inner.zip": inner})
    ok, _, src = _checks(tmp_path)
    assert all(ok.values()), ok
    assert src.unwrapped_from == "dataset/inner.zip"


def test_inner_zip_with_invalid_data_reports_inner_error(tmp_path):
    inner = _zip_bytes({"train.csv": _csv_bytes(_train())})
    _upload(tmp_path, {"inner.zip": inner})
    ok, msg, _ = _checks(tmp_path, target="no_such_column")
    assert ok["no_nested_zip"]  # the wrapper itself is fine ...
    assert not ok["target_column_present"]  # ... the inner dataset is what fails
    assert "no_such_column" in msg["target_column_present"]


def test_inner_zip_without_train_is_rejected(tmp_path):
    _upload(tmp_path, {"inner.zip": _zip_bytes({"val.csv": _csv_bytes(_train(20))})})
    ok, _, _ = _checks(tmp_path)
    assert ok["no_nested_zip"]
    assert not ok["train_csv_present"]


def test_two_levels_of_nesting_still_rejected(tmp_path):
    innermost = _zip_bytes({"train.csv": _csv_bytes(_train())})
    _upload(tmp_path, {"outer.zip": _zip_bytes({"deeper.zip": innermost})})
    ok, msg, src = _checks(tmp_path)
    assert src.unwrapped_from == "outer.zip"  # exactly one level, never recursive
    assert not ok["no_nested_zip"]
    assert ".zip" in msg["no_nested_zip"]


def test_inner_zip_beside_other_files_still_rejected(tmp_path):
    inner = _zip_bytes({"train.csv": _csv_bytes(_train())})
    _upload(tmp_path, {"inner.zip": inner, "README.txt": b"hello"})
    ok, _, src = _checks(tmp_path)
    assert src.unwrapped_from is None
    assert not ok["no_nested_zip"]


def test_two_inner_zips_still_rejected(tmp_path):
    inner = _zip_bytes({"train.csv": _csv_bytes(_train())})
    _upload(tmp_path, {"a.zip": inner, "b.zip": inner})
    ok, _, src = _checks(tmp_path)
    assert src.unwrapped_from is None
    assert not ok["no_nested_zip"]


def test_zip_bomb_inside_inner_zip_rejected(tmp_path, monkeypatch):
    big = pd.DataFrame({"f1": [0] * 5000, "target": ["a", "b"] * 2500})  # very compressible
    inner = _zip_bytes({"train.csv": _csv_bytes(big)})
    # Stored (not deflated) outer wrapper, so only the INNER member trips the ratio guard.
    p = tmp_path / "upload.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("inner.zip", inner)
    monkeypatch.setattr(V, "MAX_COMPRESSION_RATIO", 2.0)
    with pytest.raises(ValueError, match=r"nested archive 'inner.zip'.*zip-bomb guard"):
        V.DatasetSource(tmp_path)


def test_oversized_member_inside_inner_zip_rejected(tmp_path, monkeypatch):
    inner = _zip_bytes({"train.csv": _csv_bytes(_train())})
    p = tmp_path / "upload.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("inner.zip", inner)
    # Outer member (the inner .zip) fits; the inner CSV does not.
    monkeypatch.setattr(V, "MAX_MEMBER_UNCOMPRESSED_BYTES", len(inner) + 1)
    monkeypatch.setattr(V, "MAX_COMPRESSION_RATIO", 1e9)
    assert len(_csv_bytes(_train())) > len(inner) + 1
    with pytest.raises(ValueError, match=r"nested archive 'inner.zip'.*per-file guard"):
        V.DatasetSource(tmp_path)


def test_unreadable_inner_zip_rejected_clearly(tmp_path):
    _upload(tmp_path, {"inner.zip": b"not a zip at all"})
    with pytest.raises(ValueError, match=r"nested archive 'inner.zip' is not a readable zip"):
        V.DatasetSource(tmp_path)
