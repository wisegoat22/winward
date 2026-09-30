"""Fresh development/manual data and mock models only; no final instances."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from agent_lab.belief import BeliefAction, BeliefProblem, Hypothesis, Outcome
from agent_training.simulator import Action, Scenario
from winward_scale import structured_data as data


def graph():
    return Scenario("unit", "unit", "validation", ("ready", "done"), 0, 2,
                    (Action("A", "prepare", sets=1, forbids=1, tokens=2, latency_ms=100),
                     Action("B", "verify", requires=1, sets=2, forbids=2, tokens=3, latency_ms=200)))


def uncertain():
    return BeliefProblem("manual", ("done", "blocked"), 1,
                         (Hypothesis("left", 0, .5), Hypothesis("right", 0, .5)), (
        BeliefAction("inspect", "inspect", tokens=1, outcomes_by_world={
            "left": (Outcome(1., "left"),), "right": (Outcome(1., "right"),)}),
        BeliefAction("repair_left", "repair left", forbids=3, tokens=2, outcomes_by_world={
            "left": (Outcome(1., "done", sets=1),), "right": (Outcome(1., "failed", sets=2),)}),
        BeliefAction("repair_right", "repair right", forbids=3, tokens=2, outcomes_by_world={
            "left": (Outcome(1., "failed", sets=2),), "right": (Outcome(1., "done", sets=1),)}),
    ))


class Scripted:
    """Predetermined slot choices; never receives target IDs or public objects."""
    def __init__(self, choices):
        self.choices = iter(choices)
        self.calls = 0
        self.training = True
    def eval(self): self.training = False
    def train(self): self.training = True
    def __call__(self, x, valid, eligible):
        assert x.shape[1:] == (12, 505)
        assert valid.shape == eligible.shape == x.shape[:2]
        result = np.full(valid.shape, -10., np.float32)
        for i in range(x.shape[0]):
            result[i, next(self.choices)] = 10
            self.calls += 1
        return mx.array(result)


def test_fresh_mixed_corpus_is_reproducible_and_encoder_targets_are_aligned():
    first = data.corpus(20, "development", 7811)
    second = data.corpus(20, "development", 7811)
    arrays = data.batch_arrays(first)
    assert [x.shape for x in arrays] == [(20, 12, 505)] + [(20, 12)] * 3
    assert [x.dtype for x in arrays] == [np.float32, np.bool_, np.bool_, np.float32]
    assert sum(row["domain"] == "graph" for row in first) == 14
    assert sum(row["domain"] == "belief" for row in first) == 6
    assert len({row["row_id"] for row in first}) == 20
    np.testing.assert_allclose(arrays[3].sum(axis=1), 1)
    assert np.all(arrays[2][arrays[3] > 0])
    for a, b in zip(arrays, data.batch_arrays(second)):
        np.testing.assert_array_equal(a, b)
    assert any(row.get("provenance", {}).get("source") == "paired_goal" for row in first)
    assert any(row.get("provenance", {}).get("source") == "downstream_cost" for row in first)


def test_corpus_prefix_stays_identical_when_training_count_grows():
    short, long = data.corpus(8, "train", 8001), data.corpus(15, "train", 8001)
    for a, b in zip(data.batch_arrays(short), data.batch_arrays(long[:8])):
        np.testing.assert_array_equal(a, b)


def test_seed_domains_are_disjoint_and_far_from_historical_audits():
    values = {data.domain_seed(123, split, kind) for split in data.SPLITS for kind in ("graph", "belief")}
    assert len(values) == 6
    assert min(values) > 1_000_000_000_000
    assert min(abs(a - b) for a in values for b in values if a != b) > 1_000_000_000
    assert data.domain_seed(123, "train", "belief") != data.domain_seed(124, "train", "belief")
    with pytest.raises(ValueError, match="sealed"):
        data.corpus(1, "final_audit", 123)


@pytest.mark.parametrize("args", [(0, "train", 1), (True, "train", 1), (1, "other", 1),
                                  (1, "train", -1), (1, "train", True)])
def test_bad_corpus_controls_rejected(args):
    with pytest.raises(ValueError):
        data.corpus(*args)


def test_teacher_labels_and_provenance_have_no_path_into_public_features():
    original = data.row_from_graph(graph())
    contaminated = {**original, "teacher": {"secret": "B"}, "target_ids": ["B"],
                    "family": "fake", "row_id": "hidden", "corpus_seed": 999,
                    "provenance": {"answer": "B"}}
    for a, b in zip(data.encode_observation(original)[:3], data.encode_observation(contaminated)[:3]):
        np.testing.assert_array_equal(a, b)
    belief = data.row_from_belief(uncertain(), depth=2)
    corrupted = {**belief, "teacher": {"actual_world": "left"}, "target_ids": ["repair_left"]}
    for a, b in zip(data.encode_observation(belief)[:3], data.encode_observation(corrupted)[:3]):
        np.testing.assert_array_equal(a, b)


def test_uniform_target_distribution_preserves_all_exact_optimal_ties():
    scenario = Scenario("tie", "tie", "validation", ("done",), 0, 1,
                        (Action("A", "one", sets=1, tokens=5), Action("B", "two", sets=1, tokens=5)))
    row = data.row_from_graph(scenario)
    np.testing.assert_array_equal(data.batch_arrays([row])[3][0, :2], [.5, .5])
    model = Scripted([1])
    result = data.evaluate(model, [row])
    assert result["next_action_accuracy"] == 1
    assert result["prediction_counts"] == {"B": 1}
    assert result["expected_counts"] == {"A": .5, "B": .5}
    assert model.training


def test_fast_evaluation_never_calls_a_planner_or_corrects_padding(monkeypatch):
    row = data.row_from_graph(graph())
    monkeypatch.setattr(data, "search", lambda *_a, **_k: pytest.fail("No evaluation solver calls"))
    monkeypatch.setattr(data, "plan", lambda *_a, **_k: pytest.fail("No evaluation solver calls"))
    result = data.evaluate(Scripted([11]), [row])
    assert result["next_action_accuracy"] == 0
    assert result["invalid_or_ineligible_count"] == 1
    assert result["prediction_counts"] == {"INVALID_PADDING_11": 1}


def test_fast_evaluation_reports_each_domain_and_accepts_only_public_arrays():
    rows = [data.row_from_graph(graph()), data.row_from_belief(uncertain(), depth=2)]
    result = data.evaluate(Scripted([0, 0]), rows, batch_size=2)
    assert result["next_action_accuracy"] == 1
    assert set(result["by_domain"]) == {"graph", "belief"}
    assert result["by_domain"]["belief"]["next_action_accuracy"] == 1
    assert result["batch_size"] == 2 and result["inference_ms"] >= 0
    assert result["promotion"] is False


def test_graph_rollout_applies_own_actions_and_costs_without_oracle_tail():
    row = data.row_from_graph(graph(), depth=2)
    # First A succeeds, then the model defers. The oracle could finish with B.
    result = data.rollout_evaluate(Scripted([0, 3]), [row], count=1)
    episode = result["methods"]["candidate"]["episodes"][0]
    assert episode["trajectory"] == ["A", "NEEDS_CLARIFICATION"]
    assert episode["expected_verified_success"] == 0
    assert episode["expected_declared_cost"] == 3
    assert result["references"][0]["expected_verified_success"] == 1


def test_parent_comparison_uses_same_cases_and_never_substitutes_parent_choices():
    row = data.row_from_graph(graph(), depth=2)
    result = data.rollout_evaluate(Scripted([3]), [row], parent_model=Scripted([0, 1]), count=1)
    assert result["methods"]["candidate"]["overall"]["expected_verified_success_all"] == 0
    assert result["methods"]["v4_parent"]["overall"]["expected_verified_success_all"] == 1
    assert result["methods"]["v4_parent"]["overall"]["mean_cost_regret_at_matched_success"] == 0


def test_belief_rollout_integrates_observations_with_own_contingent_choices(monkeypatch):
    row = data.row_from_belief(uncertain(), depth=2)
    model = Scripted([0, 1, 2])  # inspect, then repair left/right after observation
    original_plan = data.plan
    def reference_after_predictions(*args, **kwargs):
        assert model.calls == 3
        return original_plan(*args, **kwargs)
    monkeypatch.setattr(data, "plan", reference_after_predictions)
    result = data.rollout_evaluate(model, [row], count=1)
    episode = result["methods"]["candidate"]["episodes"][0]
    assert episode["expected_verified_success"] == pytest.approx(1)
    assert episode["expected_declared_tokens"] == pytest.approx(3)
    assert episode["unique_policy_decisions"] == 3


def test_uncertain_guess_gets_only_its_actual_expected_success():
    row = data.row_from_belief(uncertain(), depth=2)
    result = data.rollout_evaluate(Scripted([1, 4]), [row], count=1)
    episode = result["methods"]["candidate"]["episodes"][0]
    assert episode["expected_verified_success"] == pytest.approx(.5)
    assert result["references"][0]["expected_verified_success"] == 1
    assert result["methods"]["candidate"]["overall"]["matched_reference_success_cases"] == 0


def test_belief_node_budget_never_reports_a_partial_result_as_exact():
    row = data.row_from_belief(uncertain(), depth=2)
    with pytest.raises(RuntimeError, match="no partial score"):
        data.rollout_evaluate(Scripted([0]), [row], count=1, max_belief_nodes=1)


def test_completion_denominator_separates_done_and_unreachable_cases():
    base = graph()
    rows = [data.row_from_graph(base, depth=2),
            data.row_from_graph(replace(base, state=2), depth=2),
            data.row_from_graph(replace(base, actions=()), depth=2)]
    result = data.rollout_evaluate(Scripted([3, 1]), rows, count=3)
    overall = result["methods"]["candidate"]["overall"]
    assert overall["initially_verified"] == 1
    assert overall["unfinished_positive_reference_success"] == 1
    assert overall["unfinished_zero_reference_success"] == 1
    assert overall["expected_verified_success_unfinished_reachable"] == 0
    assert overall["expected_verified_success_all"] == pytest.approx(1 / 3)


def test_nonfinite_logits_fail_and_original_training_mode_is_restored():
    class Broken(Scripted):
        def __call__(self, x, valid, eligible):
            return mx.full(valid.shape, float("nan"))
    model = Broken([])
    with pytest.raises(FloatingPointError, match="no fallback"):
        data.evaluate(model, [data.row_from_graph(graph())])
    assert model.training is True
