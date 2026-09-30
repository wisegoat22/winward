"""Gate and observation-boundary checks use hand-constructed fixtures only."""
from dataclasses import replace

import pytest

from agent_lab.belief import BeliefAction, BeliefProblem, Hypothesis, Outcome, STOP
from agent_training.evaluate_v3 import (PROTOCOL, cheap_action, expected_episode,
                                        graph_cost_retention, matched_success_efficiency, promotion_gate,
                                        sample_episode, seal_protocol)


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
    graphs = {suite: {"methods": {"v3": {"success_rate": .95}, "v1": {"success_rate": .95}}}
              for suite in ("legacy", "curriculum")}
    graphs["legacy"]["paired_success_action_cost"] = {"paired_successes": 20,
        "candidate_mean_declared_cost": 5, "baseline_mean_declared_cost": 5}
    graphs["goal_changes"] = {"methods": {name: {"neural_accuracy": .95,
        "neural_both_members_correct": 180, "pairs": 200, "cases": 400} for name in ("v3", "v2.1")}}
    tools = {"methods": {"v3": {"verified_and_stopped_rate": 1}, "v2.1": {"verified_and_stopped_rate": 1}}}
    belief = {"expected": {name: {"verified_success_rate": 1} for name in ("v3", "myopic_progress", "information_first")},
              "sampled": {name: {"success_rate": 1} for name in ("v3", "myopic_progress", "information_first")},
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
    graphs[suite]["methods"]["v3"]["success_rate"] = .949999
    result = promotion_gate(graphs, tools, belief, frozen_report)
    assert not result["promoted"]
    assert f"{suite}_retention" in result["failed_gates"]


def test_real_tool_verification_regression_prevents_promotion():
    graphs, tools, belief, frozen_report = metrics()
    tools["methods"]["v3"]["verified_and_stopped_rate"] = .975
    assert not promotion_gate(graphs, tools, belief, frozen_report)["promoted"]


@pytest.mark.parametrize("field,gate", [("neural_accuracy", "changed_goal_action_retention"),
                                      ("neural_both_members_correct", "changed_goal_pair_retention")])
def test_changed_goal_capability_must_be_retained(field, gate):
    graphs, tools, belief, frozen_report = metrics()
    graphs["goal_changes"]["methods"]["v3"][field] -= .01
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
    belief[kind]["v3"][field] = .99
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


def test_policy_only_observes_belief_then_observation_posterior():
    seen = []

    def observer(problem, remaining):
        seen.append(problem.belief)
        return informed_policy(problem, remaining)

    result = sample_episode(diagnosis(), observer, 987)
    assert result["success"] and result["actual_goal_met"] and result["belief_verified"]
    assert len(seen[0]) == 2 and len(seen[1]) == 1
    assert result["steps"] == 2
    assert [row["action_id"] for row in result["trace"]][0] == "inspect"
    assert result["declared_tokens"] == 7
    assert result["decision_ms"] >= 0


def test_sampled_world_and_outcome_streams_match_between_methods():
    a = sample_episode(diagnosis(), informed_policy, 123)
    b = sample_episode(diagnosis(), lambda p, d: informed_policy(p, d), 123)
    assert [(e["action_id"], e["observation"]) for e in a["trace"]] == [(e["action_id"], e["observation"]) for e in b["trace"]]
    assert a["success"] == b["success"]
    assert a["declared_tokens"] == b["declared_tokens"]


def test_actual_success_without_observable_verification_is_failure():
    action = BeliefAction("try", "Try", outcomes=(Outcome(.5, "same", sets=1), Outcome(.5, "same")))
    problem = BeliefProblem("hidden", ("goal",), 1, (Hypothesis("world", 0, 1),), (action,))
    episodes = [sample_episode(problem, lambda p, d: "try", seed, horizon=1) for seed in range(10)]
    assert any(e["actual_goal_met"] for e in episodes)
    assert all(not e["success"] and not e["belief_verified"] for e in episodes)
    assert expected_episode(problem, lambda p, d: "try", horizon=1)["expected_verified_success"] == 0


def test_expected_tree_accounts_for_both_observations_without_hidden_state():
    result = expected_episode(diagnosis(), informed_policy)
    assert result["expected_verified_success"] == 1
    assert result["expected_declared_tokens"] == 7
    assert result["expected_steps"] == 2
    assert expected_episode(diagnosis(), lambda p, d: "repair_a")["expected_verified_success"] == .5


def test_information_baseline_skips_inspection_when_evidence_is_known():
    problem = diagnosis()
    assert cheap_action(problem, information_first=True) == "inspect"
    known = replace(problem, belief=(Hypothesis("a", 0, 1),))
    assert cheap_action(known, information_first=True) == "repair_a"


@pytest.mark.parametrize("information_first", [False, True])
def test_known_cause_baselines_skip_irrelevant_evidence_bit(information_first):
    problem = BeliefProblem("known", ("fixed", "verified", "evidence"), 2,
        (Hypothesis("known", 0, 1),), (
            BeliefAction("inspect", "Read known evidence", tokens=1, outcomes=(Outcome(1, "known", sets=4),)),
            BeliefAction("repair", "Repair known cause", tokens=5, outcomes=(Outcome(1, "fixed", sets=1),)),
            BeliefAction("test", "Verify", requires=1, tokens=2, outcomes=(Outcome(1, "passed", sets=2),))))
    assert cheap_action(problem, information_first=information_first) == "repair"


def test_unverified_stop_is_never_counted_as_success():
    assert not sample_episode(diagnosis(), lambda p, d: STOP, 4)["success"]
    assert expected_episode(diagnosis(), lambda p, d: STOP)["expected_verified_success"] == 0


def test_protocol_sealing_is_exclusive_and_does_not_generate_data(tmp_path):
    result = seal_protocol(tmp_path)
    assert result["protocol"] == PROTOCOL
    assert [p.name for p in tmp_path.iterdir()] == ["audit-protocol.json"]
    with pytest.raises(FileExistsError):
        seal_protocol(tmp_path)
