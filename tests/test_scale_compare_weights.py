"""Pure CPU/file-format tests; no MLX import and no model allocation."""
import hashlib
import json
import struct

import numpy as np
import pytest

from winward_scale import compare_weights as comparison


def write_checkpoint(folder, tensors, step):
    folder.mkdir(parents=True)
    header, pieces, offset, count = {}, [], 0, 0
    for name, dtype, value in tensors:
        if dtype == "BF16":
            bits = (np.asarray(value, dtype="<f4").view("<u4") >> 16).astype("<u2")
        else:
            bits = np.asarray(value, dtype="<f4")
        data = bits.tobytes()
        header[name] = {"dtype": dtype, "shape": list(bits.shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        count += bits.size
        pieces.append(data)
    encoded = json.dumps(header).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    path = folder / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(pieces))
    info = {"track": "structured", "parameters": count, "step": step, "total_optimizer_updates": step,
            "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (folder / "checkpoint.json").write_text(json.dumps(info))
    return folder


def pair(tmp_path):
    a = write_checkpoint(tmp_path / "before", [("matrix.weight", "BF16", [[1., 0.], [-0., 2.]]),
                                               ("norm.weight", "F32", [1., 1.])], 1)
    b = write_checkpoint(tmp_path / "after", [("matrix.weight", "BF16", [[1., -0.], [0., 3.]]),
                                              ("norm.weight", "F32", [1., 1.25])], 2)
    return a, b


def test_exact_counts_signed_zero_and_bounded_reads(tmp_path, monkeypatch):
    a, b = pair(tmp_path)
    reads, original = [], comparison.os.pread
    def bounded(fd, count, offset):
        reads.append(count)
        return original(fd, count, offset)
    monkeypatch.setattr(comparison.os, "pread", bounded)
    result = comparison.compare_checkpoints(a, b, chunk_elements=1)
    assert result["parameters"] == 6 and result["changed_parameters"] == 2
    assert result["changed_parameter_tensors"] == 2 and result["changed_matrix_tensors"] == 1
    assert result["changed_matrix_parameters"] == 1
    assert max(reads) <= 4
    assert str(tmp_path) not in json.dumps(result)
    assert result["checksums_verified"]


def test_unchanged_and_non_multiple_chunks(tmp_path):
    a, _ = pair(tmp_path)
    result = comparison.compare_checkpoints(a, a, chunk_elements=3)
    assert result["changed_parameters"] == result["changed_parameter_tensors"] == 0
    assert result["unchanged_tensors"] == ["matrix.weight", "norm.weight"]


def test_bad_checksum_rejected_before_comparing(tmp_path):
    a, b = pair(tmp_path)
    path = b / "model.safetensors"
    path.write_bytes(path.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="checksum"):
        comparison.compare_checkpoints(a, b)


@pytest.mark.parametrize("change", ["shape", "dtype", "nonfinite", "count", "track"])
def test_malformed_or_incompatible_checkpoint_rejected(tmp_path, change):
    a = write_checkpoint(tmp_path / "before", [("matrix.weight", "BF16", [[1., 2.], [3., 4.]])], 1)
    value = [[1., 2.], [3., float("nan")]] if change == "nonfinite" else [1., 2., 3., 4.] if change == "shape" else [[1., 2.], [3., 4.]]
    b = write_checkpoint(tmp_path / "after", [("matrix.weight", "F32" if change == "dtype" else "BF16", value)], 2)
    if change in ("count", "track"):
        info = json.loads((b / "checkpoint.json").read_text())
        info["parameters" if change == "count" else "track"] = 999 if change == "count" else "byte"
        (b / "checkpoint.json").write_text(json.dumps(info))
    with pytest.raises(ValueError):
        comparison.compare_checkpoints(a, b)


@pytest.mark.parametrize("chunk", [0, -1, True, 16_777_217])
def test_invalid_chunk_rejected_without_opening_models(chunk):
    with pytest.raises(ValueError, match="chunk_elements"):
        comparison.compare_checkpoints("missing", "missing", chunk_elements=chunk)
