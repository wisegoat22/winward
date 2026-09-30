"""Metadata/fake-model audit tests. No generated final-audit cases or large models."""
from contextlib import contextmanager
from copy import deepcopy
import json

import pytest

mx = pytest.importorskip("mlx.core")

from agent_training.simulator import Action, Scenario
from winward_scale.structured import StructuredConfig
from winward_scale.structured_data import row_from_graph
from winward_scale import structured_audit as audit


def setup(tmp_path, monkeypatch):
    source = tmp_path / "dependency.py"
    source.write_text("fixture = True\n")
    manifest = {"candidate": {"path": "fake-candidate", "model_sha256": "candidate-sha", "parameters": 1},
                "v4_parent": {"path": "fake-v4", "model_sha256": "v4-sha", "parameters": 1}}
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "artifact_manifest", lambda _: deepcopy(manifest))
    monkeypatch.setattr(audit, "source_hashes", lambda: {"dependency.py": audit.digest(source)})
    @contextmanager
    def locked():
        yield
    monkeypatch.setattr(audit, "local_compute", locked)
    monkeypatch.setattr(audit, "load_parent", lambda *_: pytest.fail("must not load models during metadata checks"))
    monkeypatch.setattr(audit, "corpus", lambda *_a, **_k: pytest.fail("must not generate final cases"))
    args = audit.parser().parse_args(["--run", "fake-run", "--output", str(tmp_path / "audit.json"),
                                     "--cases", "2", "--rollouts", "2", "--seed", "1001"])
    return args, manifest


def seal(args):
    args.declare = True
    result = audit.prepare(args)
    args.declare, args.run_audit = False, True
    return result[0]


def test_default_audit_size_and_seed_are_explicit():
    args = audit.parser().parse_args(["--run", "none", "--output", "none", "--declare"])
    assert (args.cases, args.rollouts, args.seed, args.batch_size) == (512, 40, 810000001, 8)


def test_declaration_snapshots_sources_without_loading_or_generating(tmp_path, monkeypatch):
    args, _ = setup(tmp_path, monkeypatch)
    args.declare = True
    sealed = audit.main(args)
    assert sealed["protocol"]["split"] == "final_audit"
    assert sealed["protocol"]["promotion"] is False
    assert (tmp_path / "audit.json.source" / "dependency.py").read_text() == "fixture = True\n"
    assert not (tmp_path / "audit.json").exists()
    assert not (tmp_path / "audit.json.started.json").exists()


def test_run_requires_seal_and_is_consumed_only_once(tmp_path, monkeypatch):
    args, _ = setup(tmp_path, monkeypatch)
    args.run_audit = True
    with pytest.raises(ValueError, match="Seal"):
        audit.prepare(args)
    args.run_audit = False
    seal(args)
    audit.prepare(args)
    assert (tmp_path / "audit.json.started.json").exists()
    with pytest.raises(FileExistsError):
        audit.prepare(args)


@pytest.mark.parametrize("changed", ["source", "candidate", "baseline", "cases", "rollouts", "budget"])
def test_changed_sealed_inputs_fail_before_final_generation(tmp_path, monkeypatch, changed):
    args, manifest = setup(tmp_path, monkeypatch)
    seal(args)
    if changed == "source":
        (tmp_path / "dependency.py").write_text("fixture = False\n")
    elif changed in ("candidate", "baseline"):
        manifest["candidate" if changed == "candidate" else "v4_parent"]["model_sha256"] = "changed"
    elif changed == "budget":
        args.max_seconds += 1
    elif changed == "cases":
        args.cases += 1
    else:
        args.rollouts = 1
    with pytest.raises(ValueError, match="changed"):
        audit.prepare(args)
    assert not (tmp_path / "audit.json.started.json").exists()


def test_new_output_or_case_count_cannot_reuse_a_reserved_final_seed(tmp_path, monkeypatch):
    args, _ = setup(tmp_path, monkeypatch)
    seal(args)
    args.run_audit, args.declare = False, True
    args.output = str(tmp_path / "different.json")
    args.cases += 1
    with pytest.raises(FileExistsError):
        audit.prepare(args)


def test_modified_saved_source_snapshot_fails_closed(tmp_path, monkeypatch):
    args, _ = setup(tmp_path, monkeypatch)
    seal(args)
    (tmp_path / "audit.json.source" / "dependency.py").write_text("tampered")
    with pytest.raises(ValueError, match="snapshot"):
        audit.prepare(args)


def test_cli_holds_gpu_lock_even_for_declaration(tmp_path, monkeypatch):
    args, _ = setup(tmp_path, monkeypatch)
    args.declare = True
    held = []
    @contextmanager
    def lock():
        held.append(True)
        try:
            yield
        finally:
            held.pop()
    original = audit.prepare
    def checked(value):
        assert held == [True]
        return original(value)
    monkeypatch.setattr(audit, "local_compute", lock)
    monkeypatch.setattr(audit, "prepare", checked)
    audit.main(args)
    assert held == []


class FakeModel:
    training = False
    def parameters(self): return {"weight": mx.array([1.], dtype=mx.float32)}
    def eval(self): self.training = False
    def train(self): self.training = True
    def __call__(self, x, valid, eligible):
        # Every supplied manual case has its sole action at index 0. Eligibility
        # masks are part of the public model interface; no targets are read.
        ranks = mx.arange(12, dtype=mx.float32)[None, :]
        return mx.where(eligible, -ranks, mx.array(-1e9))


def fake_execution(tmp_path, monkeypatch):
    args, manifest = setup(tmp_path, monkeypatch)
    case = Scenario("manual", "manual", "validation", ("done",), 0, 1,
                    (Action("A", "finish", sets=1, tokens=2, latency_ms=10),))
    rows = [row_from_graph(case, row_id=f"manual-{i}") for i in range(args.cases)]
    generated = []
    def mock_corpus(count, split, seed, *, allow_final=False):
        assert (count, split, seed, allow_final) == (args.cases, "final_audit", args.seed, True)
        assert (tmp_path / "audit.json.started.json").exists()
        generated.append(True)
        return rows  # Prebuilt manual fixtures; no actual final generator is invoked.
    def mock_load(path):
        item = manifest["candidate"] if path == "fake-candidate" else manifest["v4_parent"]
        return FakeModel(), path, {}, item["model_sha256"]
    monkeypatch.setattr(audit, "corpus", mock_corpus)
    monkeypatch.setattr(audit, "load_parent", mock_load)
    monkeypatch.setattr(audit.mx, "set_memory_limit", lambda *_: None)
    monkeypatch.setattr(audit.mx, "set_cache_limit", lambda *_: None)
    monkeypatch.setattr(audit.mx, "reset_peak_memory", lambda: None)
    monkeypatch.setattr(audit.mx, "get_peak_memory", lambda: 0)
    return args, generated


def test_fake_audit_captures_actual_inventory_and_never_claims_promotion(tmp_path, monkeypatch):
    args, generated = fake_execution(tmp_path, monkeypatch)
    seal(args)
    result = audit.main(args)
    assert generated == [True]
    assert result["status"] == "completed" and not result["promotion"]
    assert result["inventory"]["candidate"]["parameters"] == 1
    assert result["inventory"]["candidate"]["by_dtype"]["float32"]["parameters"] == 1
    assert result["next_action"]["candidate"]["next_action_accuracy"] == 1
    assert result["next_action"]["candidate"]["by_domain"]["graph"]["error_breakdown"]["wrong_eligible_action"] == 0
    assert result["next_action"]["v4_parent"]["next_action_accuracy"] == 1
    assert "final-audit" in result["next_action"]["candidate"]["scope"]
    assert result["next_action"]["candidate"]["audit_context"]["seed"] == args.seed
    assert result["timing_protocol"] == result["protocol"]["timing_protocol"]
    assert result["timing_protocol"]["method_order"] == ["candidate", "v4_parent"]
    assert all(item["examples"] == 2 and item["forward_calls"] == 1 for item in result["warmup"].values())
    assert result["paired_next_action"]["both_correct"] == 2
    assert result["rollouts"]["methods"]["candidate"]["overall"]["expected_verified_success_all"] == 1
    assert result["retention"]["promotion"] is False
    assert "next_action_belief" in result["retention"]["unassessed_checks"]
    assert result["frozen_inputs_verified_after_audit"] is True
    with pytest.raises(FileExistsError, match="existing"):
        audit.main(args)


def test_budget_failure_is_recorded_and_the_final_audit_remains_consumed(tmp_path, monkeypatch):
    args, generated = fake_execution(tmp_path, monkeypatch)
    args.max_forward_calls = 1
    seal(args)
    with pytest.raises(RuntimeError, match="forward-call"):
        audit.main(args)
    result = json.loads((tmp_path / "audit.json").read_text())
    assert result["status"] == "failed" and result["audit_consumed"]
    assert not result["partial_quality_result"] and not result["promotion"]
    assert generated == [True]


def test_runtime_budget_is_checked_before_forward_calls(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(audit.time, "perf_counter", lambda: clock[0])
    budget = audit.Budget(10, 3)
    clock[0] = 11
    with pytest.raises(RuntimeError, match="runtime"):
        budget.before_forward()
    assert budget.calls == 0


def test_paired_agreement_counts_differences_and_rejects_case_mismatch():
    def report(values):
        return {"decisions": [{"row_id": str(i), "correct": value} for i, value in enumerate(values)]}
    candidate, parent = report([True, False, True, False]), report([False, True, True, False])
    result = audit.paired_agreement(candidate, parent)
    assert result["cases"] == 4
    for name in ("candidate_only_correct", "v4_only_correct", "both_correct", "both_incorrect"):
        assert result[name] == 1
    parent["decisions"][0]["row_id"] = "different"
    with pytest.raises(ValueError, match="identical ordered"):
        audit.paired_agreement(candidate, parent)


def test_model_inventory_rejects_declared_size_inflation():
    with pytest.raises(ValueError, match="parameter count"):
        audit.model_inventory(FakeModel(), 4_000_000_000)


def test_overall_improvement_cannot_hide_a_domain_retention_failure():
    def decisions(overall, graph, belief):
        return {"next_action_accuracy": overall, "invalid_or_ineligible_count": 0,
                "by_domain": {"graph": {"next_action_accuracy": graph}, "belief": {"next_action_accuracy": belief}}}
    def rollout_metrics(value):
        return {"expected_verified_success_unfinished_reachable": value}
    candidate = decisions(.9, .99, .7)
    baseline = decisions(.85, .9, .8)
    rollouts = {"methods": {"candidate": {"overall": rollout_metrics(.9), "episodes": [],
        "by_domain": {"graph": rollout_metrics(.99), "belief": rollout_metrics(.7)}},
        "v4_parent": {"overall": rollout_metrics(.85), "episodes": [],
        "by_domain": {"graph": rollout_metrics(.9), "belief": rollout_metrics(.8)}}}}
    result = audit.comparison_checks(candidate, baseline, rollouts)
    assert result["checks"]["next_action_overall"]["passed"]
    assert not result["checks"]["next_action_belief"]["passed"]
    assert not result["checks"]["rollout_success_belief"]["passed"]
    assert not result["all_measured_retention_checks_passed"]
    assert not result["promotion"]


def test_actual_manifest_checks_metadata_without_loading_model(tmp_path, monkeypatch):
    candidate = tmp_path / "checkpoint"
    baseline = tmp_path / "runs" / "goalpolicy-v4"
    candidate.mkdir()
    baseline.mkdir(parents=True)
    (candidate / "model.safetensors").write_bytes(b"metadata-only-candidate")
    (baseline / "model.safetensors").write_bytes(b"metadata-only-v4")
    baseline_hash = audit.digest(baseline / "model.safetensors")
    config = StructuredConfig(width=8, heads=2, layers=1)
    (candidate / "checkpoint.json").write_text(json.dumps({"track": "structured", "step": 1,
        "total_optimizer_updates": 1, "config": config.to_dict(), "parameters": config.parameter_count(),
        "model_sha256": audit.digest(candidate / "model.safetensors")}))
    (baseline / "config.json").write_text(json.dumps({"input_dim": 505}))
    (baseline / "report.json").write_text(json.dumps({"schema_version": "winward-v4-public-observation-1",
        "checkpoint_sha256": baseline_hash, "parameter_count": 1}))
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "BASELINE_SHA256", baseline_hash)
    monkeypatch.setattr(audit, "load_parent", lambda *_: pytest.fail("metadata check must not load"))
    result = audit.artifact_manifest(candidate)
    assert result["candidate"]["parameters"] == config.parameter_count()
    (baseline / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="baseline changed"):
        audit.artifact_manifest(candidate)
