"""Read-only scaling reports. This module never imports MLX or loads weights."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time


RUNS = Path(__file__).resolve().parents[1] / "runs"
AUDIT_SUMMARY = Path(__file__).resolve().parents[1] / "reports" / "scale" / "4b-final-audit-summary.json"
# Declared architecture, not a claim that any model has been materialized.
# (hidden, intermediate, layers, attention heads, key/value heads)
ARCHITECTURES = {
    "32m": (512, 1536, 10, 8, 4), "135m": (1024, 2816, 12, 16, 4),
    "500m": (1536, 4096, 20, 24, 8), "1b": (2048, 5632, 22, 32, 8),
    "4b": (3072, 8192, 40, 24, 8),
}
STRUCTURED_WIDTHS = {"4.86m": 256, "19m": 512, "171m": 1536, "4.25b": 7680}
TRACK_NAMES = {"byte": "Scratch workflow reader", "structured": "Grow the trained decision policy"}
MAX_REPORT_BYTES = 4 * 1024 * 1024
LIMITATIONS = [
    "Two separate tracks use our own weights: a scratch byte-workflow reader and width growth from our trained v4 structured policy. Neither uses Qwen weights.",
    "The byte track reads a compact workflow language. The structured track consumes 505 numeric features per candidate. Neither establishes understanding of arbitrary natural-language situations.",
    "Expanding a trained policy aims to preserve its existing choices. More parameters do not add new skills, remove prior failures, or increase its planning depth by themselves.",
    "Teachers use supplied facts, permissions, effects, or declared uncertainty. Those controlled mechanics are not a learned general model of the world.",
    "Validation is used during development. A larger model, lower training loss, or saved checkpoint is not a quality promotion or a final independent audit.",
    "Reports show recorded progress, not a process-health check. Memory is MLX allocation, not total Mac memory; throughput includes padded training tokens.",
]


def parameter_counts():
    result = {}
    for name, (h, f, layers, heads, kv_heads) in ARCHITECTURES.items():
        per_layer = 2 * h * h + 2 * h * kv_heads * (h // heads) + 3 * h * f + 2 * h
        result[name] = 260 * h + layers * per_layer + h
    return result


def structured_parameter_count(config):
    """Pure count for the six-layer policy family; does not construct a model."""
    values = {}
    for name in ("input_dim", "width", "layers", "heads", "expansion"):
        value = _number(_dict(config).get(name), integer=True, minimum=1, maximum=1_000_000)
        if value is None:
            raise ValueError("invalid structured architecture")
        values[name] = value
    d, hidden = values["width"], values["width"] * values["expansion"]
    if d % values["heads"]:
        raise ValueError("structured width must divide evenly into heads")
    block = 4 * d * d + 2 * d * hidden + hidden + 5 * d
    return values["layers"] * block + (values["input_dim"] + 4) * d + 1


def structured_parameter_counts():
    return {name: structured_parameter_count({"input_dim": 505, "width": width, "layers": 6,
                                              "heads": width // 32, "expansion": 4})
            for name, width in STRUCTURED_WIDTHS.items()}


def _number(value, *, integer=False, minimum=0, maximum=None):
    if type(value) not in ((int,) if integer else (int, float)):
        return None
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < minimum or (maximum is not None and value > maximum):
        return None
    return value


def _dict(value):
    return value if isinstance(value, dict) else {}


def _read(path):
    """Ignore missing/atomic-in-flight files; never follow a report symlink."""
    try:
        if path.is_symlink():
            return None, None, "A linked report was ignored."
        stat = path.stat()
        if not path.is_file() or stat.st_size > MAX_REPORT_BYTES:
            return None, None, "A report was too large or not a regular file."
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if not isinstance(value, dict):
            return None, None, "A report had an unexpected format."
        return value, stat.st_mtime, None
    except FileNotFoundError:
        return None, None, None
    except (OSError, UnicodeError, ValueError):
        return None, None, "A report could not be read; the next refresh will try again."


def _validation(value, track="byte"):
    value = _dict(value)
    accuracy = _number(value.get("next_action_accuracy"), maximum=1)
    if accuracy is None:
        return None
    return {"accuracy": accuracy, "examples": _number(value.get("examples"), integer=True),
            "completion_loss": _number(value.get("completion_loss")),
            "scope": ("Development validation: exact first action byte; all outputs compete." if track == "byte"
                      else "Development validation: agreement with the supplied next-action teacher.")
                     + " Not a final audit or complete task success."}


def _audit_status(path):
    summary, _, warning = _read(Path(path))
    pending = {"status": "pending", "promotion": False,
               "note": "The final audit summary has not been published yet. Training and development results remain separate."}
    if summary is None:
        if warning:
            return {**pending, "status": "invalid", "note": "The final audit summary could not be read. No audit result is shown."}
        return pending
    invalid = {**pending, "status": "invalid", "note": "The final audit summary failed validation. No audit result is shown."}

    def checksum(value):
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

    if (summary.get("status") != "completed" or summary.get("promotion") is not False
            or type(summary.get("parameters")) is not int or summary["parameters"] != 4_251_056_641
            or not checksum(summary.get("model_sha256")) or not checksum(summary.get("protocol_sha256"))):
        return invalid
    cases = _number(summary.get("cases"), integer=True, minimum=1, maximum=1_000_000)
    rollout_cases = _number(summary.get("rollout_cases"), integer=True, minimum=1, maximum=1_000_000)
    next_action, retention, rollouts, timing = (_dict(summary.get(key)) for key in ("next_action", "retention", "rollouts", "timing"))
    if cases is None or rollout_cases is None:
        return invalid
    next_result = {}
    for model in ("candidate", "baseline"):
        accuracy = _number(next_action.get(f"{model}_accuracy"), maximum=1)
        correct = _number(next_action.get(f"{model}_correct"), integer=True, maximum=cases)
        if accuracy is None or correct is None or not math.isclose(accuracy, correct / cases, rel_tol=0, abs_tol=1e-6):
            return invalid
        next_result.update({f"{model}_accuracy": accuracy, f"{model}_correct": correct})
    total = _number(retention.get("total_checks"), integer=True, minimum=1, maximum=10000)
    passed = _number(retention.get("passed_checks"), integer=True, maximum=total)
    all_passed = retention.get("all_measured_retention_checks_passed")
    if total is None or passed is None or type(all_passed) is not bool or all_passed != (passed == total):
        return invalid
    denominator = _number(rollouts.get("denominator"), integer=True, minimum=1, maximum=rollout_cases)
    expected = {f"{model}_success": _number(rollouts.get(f"{model}_success"), maximum=1)
                for model in ("candidate", "baseline", "reference")}
    if denominator is None or any(value is None for value in expected.values()):
        return invalid
    times = {f"{model}_ms": _number(timing.get(f"{model}_ms")) for model in ("candidate", "baseline")}
    scope, time_scope = summary.get("scope"), timing.get("scope")
    if (any(value is None for value in times.values()) or not isinstance(scope, str) or not 1 <= len(scope) <= 1000
            or not isinstance(time_scope, str) or not 1 <= len(time_scope) <= 500):
        return invalid
    return {"status": "completed", "parameters": summary["parameters"], "cases": cases, "rollout_cases": rollout_cases,
            "next_action": next_result,
            "retention": {"all_measured_retention_checks_passed": all_passed, "passed_checks": passed, "total_checks": total},
            "rollouts": {**expected, "denominator": denominator,
                         "scope": "Mean expected verified completion over initially unfinished, reference-reachable cases; branch probabilities are integrated. Not observed real-tool success counts."},
            "timing": {**times, "scope": time_scope}, "model_sha256": summary["model_sha256"],
            "protocol_sha256": summary["protocol_sha256"], "scope": scope, "promotion": False,
            "note": "Final held-out results for this frozen checkpoint. These measurements do not automatically promote the model."}


def _run_status(path, now, counts):
    records, timestamps, warnings = {}, {}, []
    for name in ("protocol", "progress", "report", "latest"):
        records[name], timestamps[name], warning = _read(path / f"{name}.json")
        if warning:
            warnings.append(f"{name}: {warning}")
    protocol = _dict(records["protocol"])
    args = _dict(protocol.get("arguments"))
    config = _dict(protocol.get("config"))
    sources = [name for name in ("progress", "report") if records[name] is not None]
    source = max(sources, key=lambda key: (timestamps[key], key == "report")) if sources else None
    progress = _dict(records[source]) if source else {}
    latest = _dict(records["latest"])
    declared_track = protocol.get("track", progress.get("track"))
    track = declared_track if declared_track in ("byte", "structured") else (
        "structured" if protocol.get("version") == "winward-structured-scale-1" or path.name.startswith("scale-structured-") else "byte")
    counts = structured_parameter_counts() if track == "structured" else counts
    parameters = _number(progress.get("parameters"), integer=True, minimum=1)
    if parameters is None:
        parameters = _number(protocol.get("parameters"), integer=True, minimum=1)
    preset = args.get("preset", config.get("name"))
    if not isinstance(preset, str) or preset not in counts:
        preset = next((name for name, count in counts.items() if count == parameters), None)
    architecture_count = None
    if track == "structured":
        try:
            architecture_count = structured_parameter_count(config)
        except ValueError:
            warnings.append("The structured architecture could not be verified from its report.")
    purpose = protocol.get("purpose", progress.get("purpose", args.get("purpose")))
    purpose = purpose if purpose in ("capacity_probe", "pilot", "training") else "unspecified"
    step = max(_number(progress.get("step"), integer=True) or 0,
               _number(latest.get("step"), integer=True) or 0)
    updates = _number(progress.get("total_optimizer_updates"), integer=True)
    if updates is None and step:
        updates = (_number(progress.get("parent_updates"), integer=True) or 0) + step
    initialized = bool(updates or step or progress.get("status") == "initialized" or (
        _number(progress.get("parameters"), integer=True, minimum=1) is not None and
        isinstance(progress.get("initial_validation"), dict)))
    # Inherited optimizer updates do not prove that a newly widened model has
    # received even one update at its new size.
    state = "weights_updated" if step else "initialized" if initialized else "preparing" if protocol else "unavailable"
    reported_status = progress.get("status")
    if reported_status not in ("training", "completed", "stopped", "failed", "preparing", "initialized", "initializing"):
        reported_status = None
    if reported_status == "failed":
        warnings.append("The run reported a failure. Its saved updates are not a successful completion.")
    validation = _validation(progress.get("validation"), track)
    if validation is None:
        history = progress.get("validation_history")
        if isinstance(history, list):
            for item in reversed(history):
                validation = _validation(_dict(item).get("validation"), track)
                if validation is not None:
                    break
    initial = _validation(progress.get("initial_validation"), track)
    best_validation = None
    history = progress.get("validation_history")
    for item in (history if isinstance(history, list) else []) + [progress]:
        candidate = _validation(_dict(item).get("validation"), track)
        if candidate and (best_validation is None or candidate["accuracy"] > best_validation["accuracy"]):
            best_validation = {**candidate, "step": _number(_dict(item).get("step"), integer=True)}
    baseline = _number(_dict(progress.get("majority_label_baseline")).get("validation_accuracy"), maximum=1)
    updated = max((stamp for stamp in timestamps.values() if stamp is not None), default=None)
    age = max(0, now - updated) if updated is not None else None
    checkpoint = isinstance(latest.get("checkpoint"), str) and _number(latest.get("step"), integer=True) is not None
    return {
        "name": path.name, "track": track, "track_name": TRACK_NAMES[track], "preset": preset, "parameters": parameters,
        "parameter_count_matches_plan": (parameters == counts[preset] and
                                         (track != "structured" or architecture_count == parameters)) if preset and parameters else None,
        "config": {key: value for key in ("width", "layers", "heads", "expansion", "input_dim")
                   if (value := _number(config.get(key), integer=True, minimum=1)) is not None} if track == "structured" else {},
        "state": state, "reported_status": reported_status, "purpose": purpose,
        "optimizer_steps": step, "total_optimizer_updates": updates,
        "planned_steps": _number(args.get("steps"), integer=True, minimum=1),
        "checkpoint_reported": checkpoint,
        "checkpoint_step": _number(latest.get("step"), integer=True) if checkpoint else None,
        "validation": validation, "initial_validation": initial, "best_validation": best_validation, "majority_baseline": baseline,
        "loss": _number(progress.get("loss")),
        "peak_mlx_gib": _number(progress.get("peak_mlx_gib")),
        "active_mlx_gib": _number(progress.get("active_mlx_gib")),
        "step_seconds": _number(progress.get("step_seconds")),
        "padded_tokens_per_second": _number(progress.get("padded_tokens_per_second")),
        "training_seconds": _number(progress.get("training_seconds")),
        "wall_seconds": _number(progress.get("wall_seconds")),
        "updated_at": datetime.fromtimestamp(updated, timezone.utc).isoformat() if updated is not None else None,
        "report_age_seconds": age, "report_stale": age is not None and age > 120,
        "curriculum": args.get("curriculum") if args.get("curriculum") in ("primitives", "tiny", "graph") else None,
        "warnings": warnings, "promoted": False,
    }


def read_status(runs_dir=RUNS, *, now=None, audit_summary=AUDIT_SUMMARY):
    """A fixed-directory snapshot; no user-selected paths, models, or mutations."""
    now = time.time() if now is None else now
    counts = parameter_counts()
    runs, warnings = [], []
    try:
        paths = sorted((p for p in Path(runs_dir).glob("scale-*") if not p.is_symlink() and p.is_dir()), key=lambda p: p.name)
        if len(paths) > 200:
            warnings.append("Only the first 200 scaling run directories are shown.")
        for path in paths[:200]:
            runs.append(_run_status(path, now, counts))
    except OSError:
        warnings.append("The scaling report directory is temporarily unavailable.")
    runs.sort(key=lambda row: row["updated_at"] or "", reverse=True)
    rank = {"unavailable": 0, "preparing": 1, "initialized": 2, "weights_updated": 3}
    tracks = []
    for track, track_counts in (("structured", structured_parameter_counts()), ("byte", counts)):
        stages = []
        for name, count in track_counts.items():
            relevant = [run for run in runs if run["track"] == track and run["preset"] == name and run["parameter_count_matches_plan"] is not False]
            state = max((r["state"] for r in relevant), key=lambda value: rank[value], default="planned")
            stages.append({"preset": name, "parameters": count, "state": state,
                           "source_reference": track == "structured" and name == "4.86m",
                           "runs": len(relevant), "latest_run": relevant[0]["name"] if relevant else None,
                           "promoted": False})
        tracks.append({"id": track, "name": TRACK_NAMES[track], "target_parameters": list(track_counts.values())[-1], "stages": stages})
    stages = tracks[1]["stages"]  # Backwards-compatible byte-track fields.
    return {"target_parameters": counts["4b"], "stages": stages, "runs": runs,
            "tracks": tracks,
            "final_audit": _audit_status(audit_summary),
            "latest_run": runs[0]["name"] if runs else None, "warnings": warnings,
            "limitations": LIMITATIONS, "refresh_seconds": 15,
            "scope": "Own-weight byte and structured-policy scaling experiments", "promoted": False}
