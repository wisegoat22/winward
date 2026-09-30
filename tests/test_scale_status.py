import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

from fastapi.testclient import TestClient
import pytest

from winward_scale.status import (ARCHITECTURES, MAX_REPORT_BYTES, parameter_counts, read_status,
                                  structured_parameter_count, structured_parameter_counts)


def record(root, run, name, data, timestamp=1000):
    path = root / run / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    os.utime(path, (timestamp, timestamp))
    return path


def protocol(preset="32m", purpose="pilot", steps=100):
    return {"parameters": parameter_counts()[preset], "config": {"name": preset},
            "arguments": {"preset": preset, "steps": steps, "curriculum": "tiny", "output": "/private/path/never/expose"},
            "purpose": purpose, "source_sha256": {"private/path": "hidden"}}


def test_empty_status_is_only_a_plan_and_does_not_create_files(tmp_path):
    missing = tmp_path / "not-created"
    status = read_status(missing)
    assert not missing.exists() and status["runs"] == []
    assert all(stage["state"] == "planned" and not stage["promoted"] for stage in status["stages"])
    assert status["target_parameters"] == 4_027_579_392


def test_declared_counts_match_model_architectures_without_importing_mlx():
    source = Path(__file__).resolve().parents[1] / "winward_scale/model.py"
    tree = ast.parse(source.read_text())
    config = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelConfig")
    assignment = next(node for node in config.body if isinstance(node, ast.AnnAssign) and node.target.id == "PRESETS")
    assert ast.literal_eval(assignment.value) == ARCHITECTURES
    assert parameter_counts() == {"32m": 31_601_152, "135m": 135_558_144, "500m": 503_778_816,
                                  "1b": 992_577_536, "4b": 4_027_579_392}
    command = "import sys; import winward_scale.status; assert not any(k == 'mlx' or k.startswith('mlx.') for k in sys.modules)"
    subprocess.run([sys.executable, "-c", command], check=True, capture_output=True)


def test_protocol_is_preparing_not_initialized_or_trained_and_drops_private_fields(tmp_path):
    record(tmp_path, "scale-32m-first", "protocol", protocol())
    result = read_status(tmp_path, now=1050)
    run = result["runs"][0]
    assert run["state"] == "preparing" and run["optimizer_steps"] == 0
    assert run["purpose"] == "pilot" and run["planned_steps"] == 100
    assert run["report_age_seconds"] == 50 and not run["report_stale"]
    assert not run["checkpoint_reported"] and not run["promoted"]
    assert "/private/path" not in json.dumps(result) and "source_sha256" not in json.dumps(result)
    assert result["stages"][0]["state"] == "preparing"


def test_initialized_record_and_updates_are_different_milestones(tmp_path):
    record(tmp_path, "scale-initial", "protocol", protocol())
    initial = {"parameters": parameter_counts()["32m"], "initialization_seconds": 2.4,
               "initial_validation": {"next_action_accuracy": .02, "examples": 100}, "status": "training"}
    record(tmp_path, "scale-initial", "progress", initial, timestamp=1010)
    assert read_status(tmp_path, now=1020)["runs"][0]["state"] == "initialized"
    record(tmp_path, "scale-initial", "progress", {**initial, "step": 2, "total_optimizer_updates": 2}, timestamp=1020)
    result = read_status(tmp_path, now=1020)
    assert result["runs"][0]["state"] == "weights_updated" and not result["promoted"]
    assert result["stages"][0]["state"] == "weights_updated"


def test_probe_checkpoint_does_not_promote_model_and_metrics_are_labeled(tmp_path):
    record(tmp_path, "scale-4b-probe", "protocol", protocol("4b", "capacity_probe", 1))
    report = {"parameters": parameter_counts()["4b"], "status": "completed", "step": 1, "total_optimizer_updates": 1,
              "validation": {"next_action_accuracy": .3, "examples": 20, "completion_loss": 1.2},
              "majority_label_baseline": {"validation_accuracy": .2}, "peak_mlx_gib": 21.7,
              "active_mlx_gib": 11.3, "step_seconds": 5.2, "padded_tokens_per_second": 420.5}
    record(tmp_path, "scale-4b-probe", "report", report, timestamp=1020)
    record(tmp_path, "scale-4b-probe", "latest", {"checkpoint": "checkpoints/step-00000001", "step": 1}, timestamp=1019)
    result = read_status(tmp_path, now=1025)
    run = result["runs"][0]
    assert run["purpose"] == "capacity_probe" and run["reported_status"] == "completed"
    assert run["checkpoint_reported"] and run["checkpoint_step"] == 1
    assert run["validation"]["accuracy"] == .3 and run["majority_baseline"] == .2
    assert run["peak_mlx_gib"] == 21.7 and run["padded_tokens_per_second"] == 420.5
    assert not run["promoted"] and not result["stages"][-1]["promoted"]


def test_newer_progress_wins_over_old_final_report_and_marks_staleness(tmp_path):
    record(tmp_path, "scale-resumed", "protocol", protocol())
    record(tmp_path, "scale-resumed", "report", {"status": "completed", "step": 10}, timestamp=1010)
    record(tmp_path, "scale-resumed", "progress", {"status": "training", "step": 11, "total_optimizer_updates": 21}, timestamp=1020)
    record(tmp_path, "scale-resumed", "latest", {"checkpoint": "checkpoints/step-10", "step": 10}, timestamp=1015)
    run = read_status(tmp_path, now=1200)["runs"][0]
    assert run["reported_status"] == "training" and run["optimizer_steps"] == 11
    assert run["total_optimizer_updates"] == 21 and run["report_stale"]


def test_partial_and_bad_reports_do_not_break_the_snapshot(tmp_path):
    record(tmp_path, "scale-bad", "protocol", protocol())
    path = record(tmp_path, "scale-bad", "progress", {})
    path.write_text('{"status":')
    record(tmp_path, "scale-bad", "latest", [])
    result = read_status(tmp_path)
    assert result["runs"][0]["state"] == "preparing"
    assert len(result["runs"][0]["warnings"]) == 2
    assert "JSONDecodeError" not in json.dumps(result)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, "0.9", 10**400])
def test_invalid_numeric_values_are_not_rendered_or_returned_as_nonfinite_json(tmp_path, value):
    record(tmp_path, "scale-numbers", "protocol", protocol())
    record(tmp_path, "scale-numbers", "progress", {"peak_mlx_gib": value, "validation": {"next_action_accuracy": value}})
    result = read_status(tmp_path)
    assert result["runs"][0]["peak_mlx_gib"] is None
    assert result["runs"][0]["validation"] is None
    json.dumps(result, allow_nan=False)


def test_symlinks_unrelated_runs_and_oversized_reports_are_ignored(tmp_path):
    external = tmp_path / "outside"
    record(external, "scale-secret", "protocol", protocol())
    (tmp_path / "scale-link").symlink_to(external / "scale-secret", target_is_directory=True)
    record(tmp_path, "goalpolicy-v4", "protocol", protocol())
    record(tmp_path, "scale-real", "protocol", protocol())
    (tmp_path / "scale-real" / "progress.json").symlink_to(external / "scale-secret" / "protocol.json")
    (tmp_path / "scale-real" / "report.json").write_text(" " * (MAX_REPORT_BYTES + 1))
    result = read_status(tmp_path)
    assert [run["name"] for run in result["runs"]] == ["scale-real"]
    assert len(result["runs"][0]["warnings"]) == 2


def test_parameter_count_mismatch_does_not_mark_ladder_reached(tmp_path):
    record(tmp_path, "scale-wrong", "protocol", {**protocol("4b"), "parameters": 50})
    record(tmp_path, "scale-wrong", "progress", {"step": 5, "parameters": 50})
    result = read_status(tmp_path)
    assert result["runs"][0]["parameter_count_matches_plan"] is False
    assert result["stages"][-1]["state"] == "planned"


def test_nonscalar_configuration_values_do_not_crash_status(tmp_path):
    record(tmp_path, "scale-weird", "protocol", {"arguments": {"preset": [], "curriculum": {}}, "purpose": {}})
    run = read_status(tmp_path)["runs"][0]
    assert run["preset"] is None and run["purpose"] == "unspecified"


def test_read_only_api_and_page_do_not_call_inference(tmp_path, monkeypatch):
    import jev_local.app as app_module
    record(tmp_path, "scale-web", "protocol", protocol())
    monkeypatch.setattr(app_module, "scaling_status", lambda: read_status(tmp_path, now=1020))
    no_model = lambda: object()
    app = app_module.create_app(no_model, no_model, no_model, no_model, no_model)
    with TestClient(app) as client:
        response = client.get("/api/scale/status")
        assert response.status_code == 200 and response.json()["runs"][0]["state"] == "preparing"
        assert response.headers["cache-control"] == "no-store"
        page = client.get("/scale")
        assert page.status_code == 200 and "/static/scale.js" in page.text
        assert "<form" not in page.text and "<button" not in page.text
        assert client.post("/api/scale/status").status_code == 405
        assert client.get("/static/scale.js").status_code == 200
        assert '/scale' in client.get("/try").text and '/scale' in client.get("/v4").text


def structured_protocol(width=512, purpose="training", steps=50):
    config = {"width": width, "layers": 6, "heads": width // 32, "expansion": 4, "input_dim": 505}
    return {"version": "winward-structured-scale-1", "track": "structured", "config": config,
            "parameters": structured_parameter_count(config), "purpose": purpose,
            "arguments": {"steps": steps}, "initialization": "/private/local/parent/checkpoint"}


def test_structured_counts_and_track_ladders_are_separate_without_claiming_materialization(tmp_path):
    assert structured_parameter_counts() == {"4.86m": 4_862_721, "19m": 19_162_625,
                                             "171m": 170_734_081, "4.25b": 4_251_056_641}
    result = read_status(tmp_path)
    tracks = {t["id"]: t for t in result["tracks"]}
    assert tracks["structured"]["target_parameters"] == 4_251_056_641
    assert tracks["byte"]["target_parameters"] == 4_027_579_392
    assert all(s["state"] == "planned" for track in tracks.values() for s in track["stages"])
    assert tracks["structured"]["stages"][0]["source_reference"]


def test_structured_preparation_initialization_and_local_updates_are_distinct(tmp_path):
    proto = structured_protocol()
    record(tmp_path, "scale-structured-19m", "protocol", proto)
    assert read_status(tmp_path)["runs"][0]["state"] == "preparing"
    initial = {"status": "training", "track": "structured", "parameters": proto["parameters"],
               "step": 0, "total_optimizer_updates": 10000,
               "initial_validation": {"next_action_accuracy": .91, "examples": 100}}
    record(tmp_path, "scale-structured-19m", "progress", initial, timestamp=1010)
    result = read_status(tmp_path)
    run = result["runs"][0]
    assert run["state"] == "initialized" and run["track"] == "structured"
    assert run["total_optimizer_updates"] == 10000 and run["optimizer_steps"] == 0
    assert result["tracks"][0]["stages"][1]["state"] == "initialized"
    assert all(stage["state"] == "planned" for stage in result["tracks"][1]["stages"])
    record(tmp_path, "scale-structured-19m", "progress", {**initial, "step": 1, "total_optimizer_updates": 10001}, timestamp=1020)
    result = read_status(tmp_path)
    assert result["tracks"][0]["stages"][1]["state"] == "weights_updated"
    assert "/private/local" not in json.dumps(result)
    assert "first action byte" not in result["runs"][0]["initial_validation"]["scope"]


def test_structured_config_must_match_claimed_parameters_before_ladder_stage_counts(tmp_path):
    proto = structured_protocol()
    proto["config"]["layers"] = 5
    record(tmp_path, "scale-structured-wrong", "protocol", proto)
    record(tmp_path, "scale-structured-wrong", "progress", {"step": 2, "parameters": proto["parameters"]})
    result = read_status(tmp_path)
    assert result["runs"][0]["parameter_count_matches_plan"] is False
    assert all(s["state"] == "planned" for s in result["tracks"][0]["stages"])


@pytest.mark.parametrize("patch", [{"width": True}, {"heads": 7}, {"expansion": []}, {"input_dim": -1}])
def test_invalid_structured_configs_do_not_crash_status(tmp_path, patch):
    proto = structured_protocol()
    proto["config"].update(patch)
    record(tmp_path, "scale-structured-invalid", "protocol", proto)
    result = read_status(tmp_path)
    assert result["runs"][0]["warnings"]
    assert result["runs"][0]["parameter_count_matches_plan"] is False


def test_best_validation_is_a_development_measurement_and_failure_stays_visible(tmp_path):
    record(tmp_path, "scale-structured-failed", "protocol", structured_protocol())
    record(tmp_path, "scale-structured-failed", "progress", {
        "parameters": 19_162_625, "status": "failed", "step": 3,
        "validation": {"next_action_accuracy": .6, "examples": 100},
        "validation_history": [{"step": 2, "validation": {"next_action_accuracy": .8, "examples": 100}}],
        "failure": "Private traceback /private/path"})
    run = read_status(tmp_path)["runs"][0]
    assert run["reported_status"] == "failed" and run["warnings"]
    assert run["best_validation"]["accuracy"] == .8 and run["best_validation"]["step"] == 2
    assert run["validation"]["accuracy"] == .6 and not run["promoted"]
    assert "/private/path" not in json.dumps(run)


def audit_summary():
    return {"status": "completed", "parameters": 4_251_056_641, "model_sha256": "a" * 64,
            "protocol_sha256": "b" * 64, "cases": 512, "rollout_cases": 40,
            "next_action": {"candidate_accuracy": 480 / 512, "baseline_accuracy": 470 / 512,
                            "candidate_correct": 480, "baseline_correct": 470},
            "retention": {"all_measured_retention_checks_passed": False, "passed_checks": 7, "total_checks": 8},
            "rollouts": {"candidate_success": .87, "baseline_success": .85, "reference_success": .97, "denominator": 28},
            "timing": {"candidate_ms": 6.4, "baseline_ms": .12, "scope": "Warmed batch8, amortized ms per decision"},
            "promotion": False, "scope": "Controlled synthetic structured decisions, not arbitrary situations or actual tools."}


def write_audit(tmp_path, data):
    path = tmp_path / "reports" / "scale" / "4b-final-audit-summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def test_final_audit_is_pending_until_valid_summary_appears(tmp_path):
    summary = tmp_path / "reports" / "scale" / "4b-final-audit-summary.json"
    result = read_status(tmp_path / "runs", audit_summary=summary)
    assert result["final_audit"]["status"] == "pending" and not summary.exists()
    write_audit(tmp_path, audit_summary())
    result = read_status(tmp_path / "runs", audit_summary=summary)
    audit = result["final_audit"]
    assert audit["status"] == "completed"
    assert audit["next_action"]["candidate_correct"] == 480
    assert audit["rollouts"]["candidate_success"] == .87 and audit["rollouts"]["denominator"] == 28
    assert "probabilities are integrated" in audit["rollouts"]["scope"]
    assert audit["timing"]["candidate_ms"] == 6.4
    assert "Warmed batch8" in audit["timing"]["scope"]
    assert not audit["promotion"] and not result["promoted"]
    assert all(stage["state"] == "planned" for track in result["tracks"] for stage in track["stages"])


@pytest.mark.parametrize("section,key,value", [
    (None, "parameters", 4_027_579_392), (None, "status", "training"), (None, "promotion", True),
    (None, "model_sha256", "<script>bad</script>"), (None, "cases", True),
    ("next_action", "candidate_accuracy", .5), ("next_action", "baseline_correct", 600),
    ("retention", "all_measured_retention_checks_passed", True), ("retention", "passed_checks", 9),
    ("rollouts", "candidate_success", 27), ("rollouts", "denominator", 41),
    ("timing", "candidate_ms", -1), ("timing", "scope", None),
])
def test_invalid_audit_cannot_publish_misleading_scores(tmp_path, section, key, value):
    summary = copy.deepcopy(audit_summary())
    (summary[section] if section else summary)[key] = value
    path = write_audit(tmp_path, summary)
    audit = read_status(tmp_path / "runs", audit_summary=path)["final_audit"]
    assert audit["status"] == "invalid" and not audit["promotion"]
    assert "next_action" not in audit and "rollouts" not in audit


def test_all_retention_checks_passing_does_not_promote_a_model(tmp_path):
    summary = audit_summary()
    summary["retention"] = {"all_measured_retention_checks_passed": True, "passed_checks": 8, "total_checks": 8}
    path = write_audit(tmp_path, summary)
    audit = read_status(tmp_path / "runs", audit_summary=path)["final_audit"]
    assert audit["status"] == "completed" and audit["retention"]["all_measured_retention_checks_passed"]
    assert not audit["promotion"]


def test_symlink_and_partial_audit_reports_are_not_followed_or_displayed(tmp_path):
    external = tmp_path / "external.json"
    external.write_text(json.dumps(audit_summary()))
    path = tmp_path / "summary.json"
    path.symlink_to(external)
    assert read_status(tmp_path / "runs", audit_summary=path)["final_audit"]["status"] == "invalid"
    path.unlink(); path.write_text('{"status":')
    assert read_status(tmp_path / "runs", audit_summary=path)["final_audit"]["status"] == "invalid"


def test_audit_dom_reports_expected_probabilities_and_hides_invalid_replacements():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed for the lightweight DOM test")
    result = subprocess.run([node, str(Path(__file__).with_name("scale_audit_ui.cjs"))], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
