"""Strict v4 gate and freeze checks; no reserved generated examples."""
from dataclasses import replace
import hashlib
import json

import pytest

from agent_lab.belief import BeliefAction, BeliefProblem, Hypothesis, Outcome, STOP
from agent_training.evaluate_v4 import (PROTOCOL, cheap_action, expected_episode,
                                        graph_cost_retention, matched_success_efficiency, promotion_gate,
                                        sample_episode, seal_protocol, protocol_sha256, validate_frozen_run,
                                        source_manifest, frozen_artifacts, assert_frozen_inputs, audit_beliefs)


def diagnosis():
    inspect = BeliefAction("inspect", "Inspect", tokens=2, outcomes_by_world={
        "a": (Outcome(1, "a"),), "b": (Outcome(1, "b"),)})
    repair_a = BeliefAction("repair_a", "Repair A", tokens=5, forbids=2, outcomes_by_world={
        "a": (Outcome(1, "passed", sets=1),), "b": (Outcome(1, "broken", sets=2),)})
    repair_b = BeliefAction("repair_b", "Repair B", tokens=5, forbids=2, outcomes_by_world={
        "a": (Outcome(1, "broken", sets=2),), "b": (Outcome(1, "passed", sets=1),)})
    return BeliefProblem("fixture", ("verified", "broken"), 1,
                         (Hypothesis("a", 0, .5), Hypothesis("b", 0, .5)),
                         (inspect, repair_a, repair_b))


def informed_policy(problem, remaining):
    if len(problem.belief) > 1:
        return "inspect"
    return "repair_" + problem.belief[0].world_id


def metrics():
    graphs = {suite: {"methods": {"v4": {"success_rate": .95}, "v1": {"success_rate": .95}}}
              for suite in ("legacy", "curriculum")}
    graphs["legacy"]["paired_success_action_cost"] = {"paired_successes": 20,
        "candidate_mean_declared_cost": 5, "baseline_mean_declared_cost": 5}
    graphs["goal_changes"] = {"methods": {name: {"neural_accuracy": .95,
        "neural_both_members_correct": 180, "pairs": 200, "cases": 400} for name in ("v4", "v2.1")}}
    tools = {"methods": {"v4": {"verified_and_stopped_rate": 1}, "v2.1": {"verified_and_stopped_rate": 1}}}
    belief = {"expected": {name: {"verified_success_rate": 1} for name in ("v4", "myopic_progress", "information_first")},
              "sampled": {name: {"success_rate": 1} for name in ("v4", "myopic_progress", "information_first")},
              "paired_efficiency": {"baseline": "information_first", "paired_successes": 20,
                                    "candidate_mean_cost": 5, "baseline_mean_cost": 6}}
    return graphs, tools, belief, {"validation_retention_passed": True}


def test_gate_requires_all_retention_and_usefulness_checks():
    result = promotion_gate(*metrics())
    assert result["promoted"]
    assert len(result["checks"]) == 12


@pytest.mark.parametrize("flag", [False, None])
def test_failed_or_missing_frozen_validation_gate_blocks_promotion(flag):
    graphs, tools, belief, frozen_report = metrics()
    frozen_report["validation_retention_passed"] = flag
    result = promotion_gate(graphs, tools, belief, frozen_report)
    assert not result["promoted"]
    assert "validation_retention" in result["failed_gates"]


@pytest.mark.parametrize("suite", ["legacy", "curriculum"])
def test_even_small_graph_regression_prevents_promotion(suite):
    graphs, tools, belief, frozen_report = metrics()
    graphs[suite]["methods"]["v4"]["success_rate"] = .949999
    result = promotion_gate(graphs, tools, belief, frozen_report)
    assert not result["promoted"]
    assert f"{suite}_retention" in result["failed_gates"]


def test_real_tool_verification_regression_prevents_promotion():
    graphs, tools, belief, frozen_report = metrics()
    tools["methods"]["v4"]["verified_and_stopped_rate"] = .975
    assert not promotion_gate(graphs, tools, belief, frozen_report)["promoted"]


@pytest.mark.parametrize("field,gate", [("neural_accuracy", "changed_goal_action_retention"),
                                      ("neural_both_members_correct", "changed_goal_pair_retention")])
def test_changed_goal_capability_must_be_retained(field, gate):
    graphs, tools, belief, frozen_report = metrics()
    graphs["goal_changes"]["methods"]["v4"][field] -= .01
    assert gate in promotion_gate(graphs, tools, belief, frozen_report)["failed_gates"]


def test_legacy_action_cost_regression_prevents_promotion():
    graphs, tools, belief, frozen_report = metrics()
    graphs["legacy"]["paired_success_action_cost"]["candidate_mean_declared_cost"] = 5.01
    assert "legacy_action_cost_retention" in promotion_gate(graphs, tools, belief, frozen_report)["failed_gates"]


def test_graph_cost_comparison_uses_only_matched_successful_tasks():
    candidate = {"records": [{"scenario_id": "a", "success": True, "declared_weighted_action_cost": 5},
                              {"scenario_id": "b", "success": False, "declared_weighted_action_cost": 0}]}
    baseline = {"records": [{"scenario_id": "a", "success": True, "declared_weighted_action_cost": 4},
                             {"scenario_id": "b", "success": True, "declared_weighted_action_cost": 100}]}
    result = graph_cost_retention(candidate, baseline)
    assert result["paired_successes"] == 1
    assert result["candidate_mean_declared_cost"] == 5
    assert result["baseline_mean_declared_cost"] == 4


@pytest.mark.parametrize("kind,field", [("expected", "verified_success_rate"), ("sampled", "success_rate")])
@pytest.mark.parametrize("baseline", ["myopic_progress", "information_first"])
def test_must_match_each_cheap_baseline(kind, field, baseline):
    graphs, tools, belief, frozen_report = metrics()
    belief[kind]["v4"][field] = .99
    other = "information_first" if baseline == "myopic_progress" else "myopic_progress"
    belief[kind][other][field] = .98
    assert f"uncertainty_{kind}_vs_{baseline}" in promotion_gate(graphs, tools, belief, frozen_report)["failed_gates"]


@pytest.mark.parametrize("candidate_cost", [6, 7, None])
def test_tied_worse_or_missing_efficiency_cannot_promote(candidate_cost):
    graphs, tools, belief, frozen_report = metrics()
    belief["paired_efficiency"]["candidate_mean_cost"] = candidate_cost
    assert "useful_efficiency" in promotion_gate(graphs, tools, belief, frozen_report)["failed_gates"]


def test_efficiency_excludes_failed_outcomes_from_both_sides():
    a = [{"problem_id": "a", "seed": 1, "success": True, "total_weighted_cost": 8},
         {"problem_id": "b", "seed": 2, "success": False, "total_weighted_cost": 0}]
    b = [{"problem_id": "a", "seed": 1, "success": True, "total_weighted_cost": 7},
         {"problem_id": "b", "seed": 2, "success": True, "total_weighted_cost": 100}]
    result = matched_success_efficiency(a, b)
    assert result["paired_successes"] == 1
    assert result["mean_cost_delta"] == 1
    assert not result["strictly_better"]


def test_efficiency_rejects_unmatched_or_duplicate_episodes():
    a = [{"problem_id": "a", "seed": 1, "success": True, "total_weighted_cost": 8}]
    with pytest.raises(ValueError, match="identical"):
        matched_success_efficiency(a, [])
    with pytest.raises(ValueError, match="unique"):
        matched_success_efficiency(a+a, a+a)




def test_v3_diagnostic_cannot_change_any_gate():
    graphs, tools, belief, report = metrics()
    for suite in ("legacy", "curriculum", "goal_changes"):
        graphs[suite]["methods"]["v3"] = {"success_rate": 1, "neural_accuracy": 1,
                                              "neural_both_members_correct": 200}
    tools["methods"]["v3"] = {"verified_and_stopped_rate": 1}
    belief["expected"]["v3"] = {"verified_success_rate": 1}
    belief["sampled"]["v3"] = {"success_rate": 1}
    result = promotion_gate(graphs, tools, belief, report)
    assert result["promoted"]
    assert result["protocol_version"] == "winward-v4-gate-1"
    assert result["protocol_sha256"] == protocol_sha256()
    graphs["legacy"]["methods"]["v4"]["success_rate"] = .94
    graphs["legacy"]["methods"]["v3"]["success_rate"] = 0
    assert not promotion_gate(graphs, tools, belief, report)["promoted"]


def test_protocol_is_fresh_and_immutable_relative_to_v3():
    from agent_training.evaluate_v3 import PROTOCOL as old
    assert PROTOCOL["seeds"] == {"belief": 430000000, "legacy": 440000000,
        "curriculum": 441000000, "goal_changes": 442000000, "tools": 450000000}
    assert old["seeds"]["belief"] == 330000000
    assert PROTOCOL["gates"] == old["gates"]
    assert PROTOCOL["counts"]["belief_problems"] == 240
    assert "not 960 independent" in PROTOCOL["sample_dependence"]


def frozen_fixture(path):
    path.mkdir(exist_ok=True)
    seal_protocol(path)
    (path / "model.safetensors").write_bytes(b"frozen model fixture")
    (path / "config.json").write_text("{}")
    report = {
        "checkpoint_frozen_before_test_generation": True,
        "checkpoint_sha256": hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest(),
        "sealed_audit_protocol_sha256": hashlib.sha256((path / "audit-protocol.json").read_bytes()).hexdigest(),
        "validation_retention_passed": False,
    }
    (path / "report.json").write_text(json.dumps(report))
    return report


def test_sealing_creates_no_final_data_and_cannot_overwrite(tmp_path):
    seal_protocol(tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["audit-protocol.json"]
    with pytest.raises(FileExistsError):
        seal_protocol(tmp_path)


def test_frozen_validation_failure_is_auditable_but_cannot_promote(tmp_path):
    report = frozen_fixture(tmp_path)
    assert validate_frozen_run(tmp_path) == report
    assert "validation_retention" in promotion_gate(*metrics()[:3], report)["failed_gates"]


@pytest.mark.parametrize("marker", ["evaluation.json", "audit-started.json", "belief-test.jsonl", "audit-source-manifest.json"])
def test_audit_cannot_restart_after_any_output_marker(tmp_path, marker):
    frozen_fixture(tmp_path)
    (tmp_path / marker).write_text("{}")
    with pytest.raises(ValueError, match="already started"):
        validate_frozen_run(tmp_path)


@pytest.mark.parametrize("mutation", ["weights", "protocol", "report_protocol", "already_tested", "unfrozen"])
def test_audit_rejects_changed_or_unfrozen_artifacts(tmp_path, mutation):
    report = frozen_fixture(tmp_path)
    if mutation == "weights":
        (tmp_path / "model.safetensors").write_bytes(b"changed")
    elif mutation == "protocol":
        sealed = json.loads((tmp_path / "audit-protocol.json").read_text())
        sealed["protocol"]["counts"]["belief_problems"] = 1
        (tmp_path / "audit-protocol.json").write_text(json.dumps(sealed))
    elif mutation == "report_protocol":
        report["sealed_audit_protocol_sha256"] = "wrong"
    elif mutation == "already_tested":
        report["test"] = {}
    else:
        report["checkpoint_frozen_before_test_generation"] = False
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        validate_frozen_run(tmp_path)


def test_dependency_and_artifact_changes_are_detected(tmp_path):
    repo = tmp_path / "repo"
    (repo / "agent_training").mkdir(parents=True)
    dependency = repo / "agent_training" / "generator.py"
    dependency.write_text("version = 1")
    run = tmp_path / "run"
    frozen_fixture(run)
    manifest = {"source_sha256": source_manifest(repo), "artifacts": {"v4": frozen_artifacts(run)}}
    assert_frozen_inputs(repo, {"v4": run}, manifest)
    dependency.write_text("version = 2")
    with pytest.raises(RuntimeError, match="source changed"):
        assert_frozen_inputs(repo, {"v4": run}, manifest)
    dependency.write_text("version = 1")
    (run / "config.json").write_text('{"changed": true}')
    with pytest.raises(RuntimeError, match="artifacts changed"):
        assert_frozen_inputs(repo, {"v4": run}, manifest)


class InformedPolicy:
    def predict_belief(self, problem, remaining_horizon=5):
        return informed_policy(problem, remaining_horizon)


def test_belief_audit_exposes_family_mix_and_dependent_replicates():
    first = diagnosis()
    completed = replace(first, id="completed", belief=(Hypothesis("a", 1, 1),))
    impossible = replace(first, id="unreachable", actions=())
    result, episodes = audit_beliefs(InformedPolicy(), InformedPolicy(),
                                    [first, completed, impossible], ["diagnosis", "completed", "unreachable"], 2)
    assert result["initially_verified_problems"] == 1
    assert result["zero_success_within_horizon_problems"] == 1
    assert result["by_family"]["completed"]["initially_verified"] == 1
    assert result["by_family"]["unreachable"]["zero_success_within_horizon"] == 1
    assert result["expected"]["v4"]["verified_success_rate"] == pytest.approx(2/3)
    assert result["v3_diagnostic"]["expected_completion_delta"] == 0
    assert len(episodes["v4"]) == 6
    assert len(result["per_problem_expected"]["v4"]) == 3
    assert "not 960 independent" in result["sample_dependence"]


def test_belief_audit_rejects_unaligned_families_before_policy_calls():
    with pytest.raises(ValueError, match="aligned"):
        audit_beliefs(None, None, [diagnosis()], [])
