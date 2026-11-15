"""Read errors on the upload become a clean "is not a readable zip" rejection, not a crash.

An OSError from reading the OUTER file, or a struct.error from zipfile's record parsing, used to
escape DatasetSource as a raw exception (issue #12). Both are now the same ValueError the inner
archive already raises, and they keep the result code the top-level failure path produced before:
VALIDATION_FAILED, with no RESOURCE_LIMIT marker in the message.
"""
from __future__ import annotations

import builtins
import io
import struct
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contract_hardening as H  # noqa: E402
import validator as V  # noqa: E402


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for arc, data in members.items():
            zf.writestr(arc, data)
    return buf.getvalue()


def _valid_upload(tmp: Path) -> Path:
    p = tmp / "upload.zip"
    p.write_bytes(_zip_bytes({"train.csv": b"f1,target\n1,a\n2,b\n"}))
    return p


class _FailingReads(io.RawIOBase):
    """A readable, seekable file object whose every read fails like a dying disk."""

    def __init__(self, real) -> None:
        self._real = real

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._real.seek(offset, whence)

    def tell(self) -> int:
        return self._real.tell()

    def read(self, size: int = -1) -> bytes:
        raise OSError(5, "Input/output error")

    def readinto(self, b) -> int:
        raise OSError(5, "Input/output error")

    def close(self) -> None:
        self._real.close()
        super().close()


def _code(exc: Exception) -> str:
    return H._failure_code("validator", SimpleNamespace(), exc)


def test_outer_file_read_oserror_is_rejected_cleanly(tmp_path, monkeypatch):
    upload = _valid_upload(tmp_path)
    real_open = builtins.open

    def failing_open(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        if Path(file) == upload and "b" in mode:
            return _FailingReads(handle)
        return handle

    monkeypatch.setattr(V, "open", failing_open, raising=False)
    with pytest.raises(ValueError, match=r"archive 'upload.zip' is not a readable zip: OSError") as info:
        V.DatasetSource(tmp_path)
    assert isinstance(info.value.__cause__, OSError)
    assert _code(info.value) == H.VALIDATOR_RUNTIME_FAILURE


def test_outer_zipfile_oserror_is_rejected_cleanly(tmp_path, monkeypatch):
    _valid_upload(tmp_path)

    def failing_zipfile(*args, **kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(V.zipfile, "ZipFile", failing_zipfile)
    with pytest.raises(ValueError, match=r"archive 'upload.zip' is not a readable zip: OSError"):
        V.DatasetSource(tmp_path)


@pytest.mark.parametrize("nested", [False, True], ids=["outer", "inner"])
def test_struct_error_is_rejected_cleanly(tmp_path, monkeypatch, nested):
    # No archive content reached struct.error in a 20,000-mutation fuzz of the outer upload, so the
    # error is injected at zipfile's own end-of-central-directory reader, which the pre-check calls.
    if nested:
        (tmp_path / "upload.zip").write_bytes(_zip_bytes({"inner.zip": _zip_bytes({"train.csv": b"a\n1\n"})}))
    else:
        _valid_upload(tmp_path)
    real = zipfile._EndRecData

    def flaky_endrec(fp):
        is_upload = str(getattr(fp, "name", "")).endswith("upload.zip")
        if is_upload != nested:  # outer: fail on the upload; inner: fail on the spooled member
            raise struct.error("unpack requires a buffer of 22 bytes")
        return real(fp)

    monkeypatch.setattr(zipfile, "_EndRecData", flaky_endrec)
    name = "nested archive 'inner.zip'" if nested else "archive 'upload.zip'"
    with pytest.raises(ValueError, match=rf"{name} is not a readable zip: error") as info:
        V.DatasetSource(tmp_path)
    assert isinstance(info.value.__cause__, struct.error)
    assert _code(info.value) == H.VALIDATOR_RUNTIME_FAILURE


def test_unreadable_outer_and_inner_share_one_result_code(tmp_path):
    (tmp_path / "upload.zip").write_bytes(_zip_bytes({"inner.zip": b"not a zip at all"}))
    with pytest.raises(ValueError, match=r"is not a readable zip") as info:
        V.DatasetSource(tmp_path)
    outer = ValueError("archive 'upload.zip' is not a readable zip: OSError: [Errno 5] Input/output error")
    assert _code(info.value) == _code(outer) == H.VALIDATOR_RUNTIME_FAILURE


def test_resource_limit_rejections_are_not_swallowed(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "MAX_ARCHIVE_MEMBERS", 0)
    _valid_upload(tmp_path)
    with pytest.raises(ValueError, match=r"entry-count zip-bomb guard") as info:
        V.DatasetSource(tmp_path)
    assert "not a readable zip" not in str(info.value)
    assert _code(info.value) == H.RESOURCE_LIMIT
