"""Observe real training fixtures; never consult reserved evaluation artifacts."""
from dataclasses import replace

import numpy as np

from agent_lab.runner import evidence_first
from agent_lab.sandbox import SandboxTask
from agent_training.features import encode
from agent_training.simulator import Scenario
from agent_training.train_tools import (augment_rows, collect_episodes, fixture_specs,
                                        observed_targets, permute_observation)


def test_training_fixture_scope_and_seed_separation():
    train = fixture_specs(16, 210000000)
    valid = fixture_specs(16, 220000000)
    assert {s["kind"] for s in train} == {"boundary", "rounding"}
    assert {(s["changed_goal"], s["uncertain"]) for s in train} == {(False,False), (False,True), (True,False), (True,True)}
    assert not {s["seed"] for s in train} & {s["seed"] for s in valid}


def test_actual_failure_precedes_recovery_labels_and_no_hidden_answer():
    rows, summaries = collect_episodes(2, 210900000, "train")
    assert all(s["success"] and s["done"] for s in summaries)
    failed = [r for r in rows if r["teacher_action_id"] == "run_tests" and r["observed_result"]["passed"] is False]
    assert failed and all(r["forecast_matched"] is False for r in failed)
    for row in rows:
        scenario = Scenario.from_dict(row["scenario"])
        assert row["teacher_action_id"] == evidence_first(scenario)
        assert row["teacher_action_id"] in row["target_ids"]
        assert not any(key in row["provenance"] for key in ("correct_patch", "hidden_answer", "oracle_action"))


def test_unverified_patches_are_tied_without_candidate_code_labels():
    with SandboxTask(210900004) as task:
        task.step("run_tests")
        task.step("inspect")
        observation = task.observe()
        targets = observed_targets(observation, evidence_first(observation))
        assert set(targets) == {"apply_candidate_a", "apply_candidate_b"}
        changed_text = replace(observation, context={"supplied_candidates": {"apply_candidate_a": "wrong", "apply_candidate_b": "right"}})
        assert observed_targets(changed_text, evidence_first(changed_text)) == targets


def test_permutation_preserves_eligibility_goal_and_effect_relations():
    rows, _ = collect_episodes(1, 210900008, "train")
    for row in rows:
        moved = permute_observation(row, 83000001)
        first, second = Scenario.from_dict(row["scenario"]), Scenario.from_dict(moved["scenario"])
        mapping = moved["provenance"]["action_id_mapping"]
        permutation = moved["provenance"]["fact_permutation"]
        remap = lambda mask: sum(1 << permutation[i] for i in range(len(permutation)) if mask & (1 << i))
        assert second.state == remap(first.state) and second.goal == remap(first.goal)
        assert first.goal_met() == second.goal_met()
        lookup = {a.id: a for a in second.actions}
        for action in first.actions:
            other = lookup[mapping[action.id]]
            assert other.eligible(second.state) == action.eligible(first.state)
            for field in ("requires", "forbids", "sets", "clears"):
                assert getattr(other, field) == remap(getattr(action, field))
        assert moved["target_ids"] == [mapping.get(action, action) for action in row["target_ids"]]
        assert moved["teacher_action_id"] in moved["target_ids"]
        assert second.context == {}


def test_augmentation_is_reproducible_and_labels_are_eligible():
    rows, _ = collect_episodes(1, 210900016, "train")
    first = augment_rows(rows, 30, 83000000)
    assert first == augment_rows(rows, 30, 83000000)
    for row in first:
        _, _, eligible, candidates = encode(row["scenario"], max_depth=row["max_depth"])
        allowed = {candidates[i]["id"] for i in np.flatnonzero(eligible)}
        assert set(row["target_ids"]) <= allowed
