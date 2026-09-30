"""Read-only, bounded-memory comparison of local structured safetensors files.

No model construction, MLX import, GPU work, mmap, or full-weight allocation.
Both checkpoint checksums are verified before comparing every stored scalar in
small CPU chunks. BF16/F16/F32 values are compared exactly; positive and negative
zero are treated as equal. Nonfinite values are rejected. Changed values prove
stored training updates, not that decision quality improved.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct

import numpy as np


FORMATS = {
    "BF16": (np.dtype("<u2"), 0x7F80, 0x7FFF),
    "F16": (np.dtype("<u2"), 0x7C00, 0x7FFF),
    "F32": (np.dtype("<u4"), 0x7F800000, 0x7FFFFFFF),
}


def _sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(16 * 1024**2), b""):
            h.update(piece)
    return h.hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Safetensors header contains a duplicate key")
        result[key] = value
    return result


def _header(stream):
    prefix = stream.read(8)
    if len(prefix) != 8:
        raise ValueError("Incomplete safetensors length prefix")
    length = struct.unpack("<Q", prefix)[0]
    if not 2 <= length <= 16 * 1024**2:
        raise ValueError("Unexpected safetensors header length")
    content = stream.read(length)
    if len(content) != length:
        raise ValueError("Incomplete safetensors header")
    header = json.loads(content, object_pairs_hook=_object)
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be an object")
    start = 8 + length
    data_size = os.fstat(stream.fileno()).st_size - start
    entries, intervals = {}, []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict) or entry.get("dtype") not in FORMATS:
            raise ValueError(f"Unsupported safetensors tensor: {name}")
        shape, offsets = entry.get("shape"), entry.get("data_offsets")
        if not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape):
            raise ValueError(f"Invalid tensor shape: {name}")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets):
            raise ValueError(f"Invalid tensor offsets: {name}")
        count = math.prod(shape)
        size = FORMATS[entry["dtype"]][0].itemsize
        low, high = offsets
        if not 0 <= low <= high <= data_size or high - low != count * size:
            raise ValueError(f"Tensor data size does not match shape: {name}")
        entries[name] = dict(dtype=entry["dtype"], shape=shape, count=count,
                             offset=start + low, size=size)
        intervals.append((low, high))
    end = 0
    for low, high in sorted(intervals):
        if low != end:
            raise ValueError("Safetensors tensor data has a gap or overlap")
        end = high
    if end != data_size:
        raise ValueError("Unaccounted safetensors data")
    return entries


def _checkpoint(directory):
    directory = Path(directory)
    info = json.loads((directory / "checkpoint.json").read_text())
    if info.get("track") != "structured":
        raise ValueError("Comparison requires our structured checkpoints")
    for key in ("parameters", "step", "total_optimizer_updates"):
        if type(info.get(key)) is not int or info[key] < 0:
            raise ValueError(f"Invalid checkpoint metadata: {key}")
    path = directory / "model.safetensors"
    stat = path.stat()
    if _sha256(path) != info.get("model_sha256"):
        raise ValueError("Checkpoint checksum does not match its recorded weights")
    identity = {"run": directory.parent.parent.name if directory.parent.name == "checkpoints" else directory.name,
                "checkpoint": directory.name, "model_sha256": info["model_sha256"],
                "parameters": info["parameters"], "step": info["step"],
                "total_optimizer_updates": info["total_optimizer_updates"]}
    return path, identity, (stat.st_size, stat.st_mtime_ns)


def compare_checkpoints(before_directory, after_directory, *, chunk_elements=1_048_576):
    """Compare all parameters, returning public metadata without private paths."""
    if type(chunk_elements) is not int or chunk_elements <= 0 or chunk_elements > 16_777_216:
        raise ValueError("chunk_elements must be an integer from 1 to 16777216")
    before_path, before, before_stat = _checkpoint(before_directory)
    after_path, after, after_stat = _checkpoint(after_directory)
    records = []
    with before_path.open("rb") as a_file, after_path.open("rb") as b_file:
        a_header, b_header = _header(a_file), _header(b_file)
        if a_header.keys() != b_header.keys():
            raise ValueError("Checkpoint parameter names differ")
        for name, a in a_header.items():
            b = b_header[name]
            if (a["shape"], a["dtype"]) != (b["shape"], b["dtype"]):
                raise ValueError(f"Checkpoint shape or dtype differs: {name}")
            dtype, exponent_mask, magnitude_mask = FORMATS[a["dtype"]]
            changed = 0
            for offset in range(0, a["count"], chunk_elements):
                count = min(chunk_elements, a["count"] - offset)
                bytes_count = count * a["size"]
                a_bytes = os.pread(a_file.fileno(), bytes_count, a["offset"] + offset * a["size"])
                b_bytes = os.pread(b_file.fileno(), bytes_count, b["offset"] + offset * b["size"])
                if len(a_bytes) != bytes_count or len(b_bytes) != bytes_count:
                    raise ValueError("Checkpoint became incomplete during comparison")
                av, bv = np.frombuffer(a_bytes, dtype=dtype), np.frombuffer(b_bytes, dtype=dtype)
                if np.any((av & exponent_mask) == exponent_mask) or np.any((bv & exponent_mask) == exponent_mask):
                    raise ValueError(f"Nonfinite stored weights: {name}")
                unequal = av != bv
                both_zero = ((av & magnitude_mask) == 0) & ((bv & magnitude_mask) == 0)
                changed += int(np.count_nonzero(unequal & ~both_zero))
            records.append({"name": name, "shape": a["shape"], "dtype": a["dtype"],
                            "parameters": a["count"], "changed_parameters": changed,
                            "matrix": len(a["shape"]) == 2})
        for stream, expected in ((a_file, before_stat), (b_file, after_stat)):
            stat = os.fstat(stream.fileno())
            if (stat.st_size, stat.st_mtime_ns) != expected:
                raise ValueError("Checkpoint file changed during comparison")
    total = sum(x["parameters"] for x in records)
    if total != before["parameters"] or total != after["parameters"]:
        raise ValueError("Recorded checkpoint parameter count differs from its weights")
    changed = sum(x["changed_parameters"] for x in records)
    matrices = [x for x in records if x["matrix"]]
    return {"version": "winward-stored-weight-change-proof-1", "before": before, "after": after,
            "checksums_verified": True, "comparison": "Every stored numeric value; signed zeros are equal; nonfinite values rejected.",
            "chunk_elements": chunk_elements, "device": "CPU, bounded file reads; no model or GPU allocation",
            "parameters": total, "changed_parameters": changed,
            "changed_fraction": changed / total if total else 0.,
            "parameter_tensors": len(records), "changed_parameter_tensors": sum(x["changed_parameters"] > 0 for x in records),
            "matrix_tensors": len(matrices), "changed_matrix_tensors": sum(x["changed_parameters"] > 0 for x in matrices),
            "matrix_parameters": sum(x["parameters"] for x in matrices),
            "changed_matrix_parameters": sum(x["changed_parameters"] for x in matrices),
            "unchanged_tensors": [x["name"] for x in records if x["changed_parameters"] == 0],
            "tensors": records,
            "limitation": "Stored changes prove weights were updated; they do not show every scalar changed, useful new skills, or improved decision quality."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-elements", type=int, default=1_048_576)
    args = parser.parse_args()
    result = compare_checkpoints(args.before, args.after, chunk_elements=args.chunk_elements)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("parameters", "changed_parameters", "changed_fraction",
                                                  "changed_parameter_tensors", "changed_matrix_tensors")}))


if __name__ == "__main__":
    main()
