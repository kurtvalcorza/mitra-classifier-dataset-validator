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


def _flood(n: int) -> bytes:
    """A zip of n zero-byte entries: invisible to the size and ratio guards."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for i in range(n):
            zf.writestr(f"e{i}.txt", b"")
    return buf.getvalue()


_FLOOD = rf"entries \(> {V.MAX_ARCHIVE_MEMBERS:,} entries\); refusing \(entry-count zip-bomb guard\)"


def test_entry_flood_in_outer_zip_rejected(tmp_path):
    (tmp_path / "upload.zip").write_bytes(_flood(V.MAX_ARCHIVE_MEMBERS + 1))
    with pytest.raises(ValueError, match=_FLOOD):
        V.DatasetSource(tmp_path)


def test_entry_flood_in_inner_zip_rejected(tmp_path):
    _upload(tmp_path, {"inner.zip": _flood(V.MAX_ARCHIVE_MEMBERS + 1)})
    with pytest.raises(ValueError, match=r"^nested archive 'inner.zip': .*" + _FLOOD):
        V.DatasetSource(tmp_path)


def test_directory_entries_count_toward_the_cap(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("train.csv", _csv_bytes(_train()))
        for i in range(V.MAX_ARCHIVE_MEMBERS):  # one file + MAX directory entries = MAX + 1
            zf.writestr(f"d{i}/", b"")
    _upload(tmp_path, {"inner.zip": buf.getvalue()})
    with pytest.raises(ValueError, match=_FLOOD):
        V.DatasetSource(tmp_path)


def test_entry_count_at_the_cap_is_accepted(tmp_path):
    members = {f"e{i}.txt": b"" for i in range(V.MAX_ARCHIVE_MEMBERS - 1)}
    members["train.csv"] = _csv_bytes(_train())
    _upload(tmp_path, {"inner.zip": _zip_bytes(members)})
    src = V.DatasetSource(tmp_path)
    try:
        assert src.unwrapped_from == "inner.zip"
        assert src.resolve_single("train") == "train.csv"
    finally:
        src.close()


def _set_encrypted_flag(data: bytes) -> bytes:
    """The stdlib cannot write encrypted zips, so mark the entries encrypted by setting bit 0 of
    the general-purpose flag in each local header (offset 6) and central directory entry (offset 8)."""
    out = bytearray(data)
    for sig, off in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        pos = out.find(sig)
        while pos != -1:
            out[pos + off] |= 0x01
            pos = out.find(sig, pos + 4)
    return bytes(out)


def test_encrypted_nested_member_rejected_clearly(tmp_path):
    inner = _zip_bytes({"train.csv": _csv_bytes(_train())})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("inner.zip", inner)
    (tmp_path / "upload.zip").write_bytes(_set_encrypted_flag(buf.getvalue()))
    with pytest.raises(ValueError, match=r"nested archive 'inner.zip' is not a readable zip: RuntimeError"):
        V.DatasetSource(tmp_path)


# --- the cap is enforced from the end-of-central-directory record, before zipfile parses it ---

def _peak_bytes_constructing(tmp: Path) -> tuple[int, Exception | None]:
    import tracemalloc
    tracemalloc.start()
    err = None
    try:
        V.DatasetSource(tmp).close()
    except (ValueError, zipfile.BadZipFile) as exc:  # BadZipFile: outer archive zipfile refuses
        err = exc
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak, err


@pytest.mark.parametrize("nested", [False, True], ids=["outer", "inner"])
def test_20k_entry_flood_rejected_before_directory_parse(tmp_path, nested, monkeypatch):
    parsed: list[int] = []  # entry counts of every central directory zipfile actually parsed
    real = zipfile.ZipFile._RealGetContents

    def recording(self):
        real(self)
        parsed.append(len(self.filelist))

    monkeypatch.setattr(zipfile.ZipFile, "_RealGetContents", recording)
    flood = _flood(2 * V.MAX_ARCHIVE_MEMBERS)
    parsed.clear()  # building the flood parses nothing, but start clean
    if nested:
        _upload(tmp_path, {"inner.zip": flood})
    else:
        (tmp_path / "upload.zip").write_bytes(flood)
    peak, err = _peak_bytes_constructing(tmp_path)
    assert err is not None and "entry-count zip-bomb guard" in str(err)
    assert f"{2 * V.MAX_ARCHIVE_MEMBERS:,} entries" in str(err)  # from the EOCD record
    assert peak < 20 * 1024 * 1024, f"peak {peak:,} bytes"
    assert all(n <= V.MAX_ARCHIVE_MEMBERS for n in parsed), parsed  # flood never parsed


def test_oversized_central_directory_rejected_before_parse(tmp_path):
    # Few entries, but names long enough to blow the central-directory byte budget.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for i in range(V.MAX_CENTRAL_DIRECTORY_BYTES // 60_000 + 2):
            zf.writestr(f"{i:05d}" + "n" * 60_000, b"")
    (tmp_path / "upload.zip").write_bytes(buf.getvalue())
    with pytest.raises(ValueError, match=r"central directory .*entry-count zip-bomb guard"):
        V.DatasetSource(tmp_path)


def _zip64(members: dict[str, bytes], monkeypatch) -> bytes:
    """Force zipfile to write ZIP64 end-of-central-directory records for a small archive."""
    monkeypatch.setattr(zipfile, "ZIP_FILECOUNT_LIMIT", 1)
    data = _zip_bytes(members)
    monkeypatch.undo()
    assert b"PK\x06\x06" in data and b"PK\x06\x07" in data
    return data


def test_zip64_archive_is_read_and_its_count_used(tmp_path, monkeypatch):
    data = bytearray(_zip64({"train.csv": _csv_bytes(_train()), "val.csv": _csv_bytes(_train(20))},
                            monkeypatch))
    # Put the 0xFFFF sentinel in the classic EOCD: only the ZIP64 record has the true count (2).
    eocd = data.rfind(b"PK\x05\x06")
    data[eocd + 8:eocd + 12] = b"\xff\xff\xff\xff"
    (tmp_path / "upload.zip").write_bytes(bytes(data))
    src = V.DatasetSource(tmp_path)
    try:
        assert src.resolve_single("train") == "train.csv"
    finally:
        src.close()


def test_zip64_entry_flood_rejected(tmp_path, monkeypatch):
    members = {f"e{i}.txt": b"" for i in range(V.MAX_ARCHIVE_MEMBERS + 1)}
    (tmp_path / "upload.zip").write_bytes(_zip64(members, monkeypatch))
    with pytest.raises(ValueError, match=_FLOOD):
        V.DatasetSource(tmp_path)


# --- end to end through validate.py: the flood maps to the RESOURCE_LIMIT result code ---

@pytest.mark.parametrize("nested", [False, True], ids=["outer", "inner"])
def test_entry_flood_result_code_is_resource_limit(tmp_path, nested):
    import json
    import os
    import subprocess

    data_dir = tmp_path / "ds"
    data_dir.mkdir()
    flood = _flood(V.MAX_ARCHIVE_MEMBERS + 1)
    if nested:
        _upload(data_dir, {"inner.zip": flood})
    else:
        (data_dir / "upload.zip").write_bytes(flood)
    result = tmp_path / "result.json"
    env = {k: v for k, v in os.environ.items() if not k.startswith("DIMER_")}
    env.update(DIMER_DATASET_DIR=str(data_dir), DIMER_RESULT_PATH=str(result))
    repo = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "validate.py"], cwd=repo, env=env, capture_output=True,
                   text=True, timeout=120)
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["code"] == "RESOURCE_LIMIT", payload


# --- the pre-check reads the same EOCD record zipfile will parse (zipfile._EndRecData) ---

def test_zipfile_private_names_used_by_the_precheck_exist():
    # A Python upgrade that removes these must fail here (and at import), not skip the check.
    for name in ("_EndRecData", "_ECD_ENTRIES_TOTAL", "_ECD_SIZE"):
        assert hasattr(zipfile, name), name


def _short_name_flood(n: int) -> bytes:
    """n zero-byte entries with 1-3 character names: the most entries per directory byte."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for i in range(n):
            zf.writestr(f"{i:x}", b"")
    return buf.getvalue()


def _eocd(data: bytes) -> int:
    return data.rfind(b"PK\x05\x06")


def _bypass_archives(monkeypatch) -> dict[str, bytes]:
    """Round-3 review bypass shapes, rebuilt deterministically at 2x the entry cap."""
    import struct

    n = 2 * V.MAX_ARCHIVE_MEMBERS
    out: dict[str, bytes] = {}
    classic = bytearray(_flood(n))
    e = _eocd(classic)
    classic[e + 4:e + 8] = b"PK\x05\x06"  # EOCD signature planted in the disk-number fields
    out["classic_sig_in_disk_fields"] = bytes(classic)
    monkeypatch.setattr(zipfile, "ZIP_FILECOUNT_LIMIT", 1)  # force ZIP64 records
    z64 = _flood(n)
    monkeypatch.undo()
    e = _eocd(z64)
    assert z64[e - 20:e - 16] == b"PK\x06\x07" and z64[e - 76:e - 72] == b"PK\x06\x06"
    far = bytearray(z64)  # locator points nowhere; classic EOCD understates to 1 entry / 46 B
    far[e + 8:e + 12] = struct.pack("<2H", 1, 1)
    far[e + 12:e + 16] = struct.pack("<I", 46)
    far[e - 12:e - 4] = struct.pack("<Q", 2 ** 62)
    out["zip64_locator_far_away"] = bytes(far)
    decoy = bytearray(z64)  # locator points at a decoy ZIP64 record hidden in the comment
    record = struct.pack("<4sQ2H2I4Q", b"PK\x06\x06", 44, 45, 45, 0, 0, 1, 1, 46, 0)
    decoy[e + 20:e + 22] = struct.pack("<H", len(record))
    decoy += record
    decoy[e - 12:e - 4] = struct.pack("<Q", e + 22)
    out["zip64_decoy_record_in_comment"] = bytes(decoy)
    return out


def _what_zipfile_parses(data: bytes) -> tuple[int, int] | None:
    try:  # interpreters with the CVE-2025-8291 fix reject a corrupt ZIP64 locator outright
        rec = zipfile._EndRecData(io.BytesIO(data))
    except zipfile.BadZipFile:
        return None
    return None if rec is None else (rec[zipfile._ECD_ENTRIES_TOTAL], rec[zipfile._ECD_SIZE])


@pytest.mark.parametrize("nested", [False, True], ids=["outer", "inner"])
def test_round3_bypass_archives_are_rejected_before_parse(tmp_path, monkeypatch, nested):
    for name, data in _bypass_archives(monkeypatch).items():
        try:
            V._assert_directory_bounded(io.BytesIO(data))
            prechecked = False
        except (ValueError, zipfile.BadZipFile):
            prechecked = True
        parsed = _what_zipfile_parses(data)
        # Differential rule: the pre-check rejects, or zipfile would parse a bounded directory.
        assert prechecked or parsed is None or (parsed[0] <= V.MAX_ARCHIVE_MEMBERS
                                                and parsed[1] <= V.MAX_CENTRAL_DIRECTORY_BYTES), (name, parsed)
        case = tmp_path / name
        case.mkdir()
        if nested:
            _upload(case, {"inner.zip": data})
        else:
            (case / "upload.zip").write_bytes(data)
        peak, err = _peak_bytes_constructing(case)
        assert err is not None, name
        assert peak < 20 * 1024 * 1024, (name, f"peak {peak:,} bytes")


def test_understated_entry_total_is_bounded_by_directory_size(tmp_path):
    """The declared total can lie low; zipfile then parses entries until the declared directory
    size is used up. The byte budget bounds that worst case, and the post-open count rejects it."""
    import struct

    n = V.MAX_CENTRAL_DIRECTORY_BYTES // 50  # ~50-byte entries (46 + hex name): densest that fits
    data = bytearray(_short_name_flood(n))
    e = _eocd(data)
    cd_bytes = struct.unpack("<I", data[e + 12:e + 16])[0]
    assert cd_bytes <= V.MAX_CENTRAL_DIRECTORY_BYTES
    data[e + 8:e + 12] = struct.pack("<2H", 1, 1)  # claim 1 entry
    (tmp_path / "upload.zip").write_bytes(bytes(data))
    peak, err = _peak_bytes_constructing(tmp_path)
    assert err is not None and "entry-count zip-bomb guard" in str(err)
    print(f"understated-total worst case: {n:,} entries parsed, peak {peak / 2**20:.2f} MiB")
    assert peak < 64 * 1024 * 1024, f"peak {peak:,} bytes for {n:,} parsed entries"


def test_archive_without_eocd_fails_closed(tmp_path):
    with pytest.raises(zipfile.BadZipFile):
        V._assert_directory_bounded(io.BytesIO(b"PK\x03\x04 not really a zip"))
    fp = io.BytesIO(_zip_bytes({"train.csv": b"a\n1\n"}))
    fp.seek(5)
    V._assert_directory_bounded(fp)
    assert fp.tell() == 5  # file position restored
