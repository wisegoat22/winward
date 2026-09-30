"""Semantic checks on v2 data; no dependency on a learned checkpoint."""
from dataclasses import replace

import numpy as np
import pytest

from agent_training.curriculum import (SPLIT_SHAPES, competing_goal_pair, downstream_cost_pair,
                                       encode_rows, generate_rows, make_world, provenance_summary)
from agent_training.features import encode
from agent_training.simulator import Scenario, search


def test_splits_reserve_distinct_graph_shapes():
    shapes = [set(group) for group in SPLIT_SHAPES.values()]
    assert not shapes[0] & shapes[1]
    assert not shapes[0] & shapes[2]
    assert not shapes[1] & shapes[2]
    with pytest.raises(ValueError, match="reserved"):
        make_world(42, "test", shape="independent_routes")


def test_goal_pair_changes_only_goal_and_display_identifier():
    first, second = competing_goal_pair(731, "train")
    assert first.goal != second.goal
    assert replace(first, id=second.id, goal=second.goal) == second
    # Equal state/actions does not imply an equal optimal action when the goal changes.
    actionable_pairs = 0
    for seed in range(100):
        first, second = competing_goal_pair(seed, "train", progress=False)
        a, b = search(first), search(second)
        if a["outcome"] == b["outcome"] == "plan" and not set(a["optimal_action_ids"]) & set(b["optimal_action_ids"]):
            actionable_pairs += 1
    assert actionable_pairs > 10


def test_cost_pair_does_not_change_goal_state_or_transitions():
    first, second = downstream_cost_pair(44, "train")
    assert first.goal == second.goal and first.state == second.state
    assert [(a.id, a.requires, a.forbids, a.sets, a.clears, a.allowed) for a in first.actions] == [
        (a.id, a.requires, a.forbids, a.sets, a.clears, a.allowed) for a in second.actions]
    assert [(a.tokens, a.latency_ms) for a in first.actions] != [(a.tokens, a.latency_ms) for a in second.actions]


def test_mid_progress_goal_revisions_are_present_and_provenance_is_not_input():
    rows = generate_rows(400, "train", 20261002)
    assert provenance_summary(rows)["mid_progress_goal_changes"] > 0
    row = next(r for r in rows if r["provenance"]["source"] == "paired_goal")
    scenario = Scenario.from_dict(row["scenario"])
    before = encode(scenario)
    after = encode(replace(scenario, context={"target": "bogus", "previous_goal": 999}, family="bogus"))
    assert all(np.array_equal(before[i], after[i]) for i in range(3))


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_reproducible_valid_labels_and_exact_plan_execution(split):
    rows = generate_rows(64, split, 975100)
    assert rows == generate_rows(64, split, 975100)
    arrays = encode_rows(rows)
    assert arrays[0].shape == (64, 12, 103)
    assert np.all(arrays[2][arrays[3] > 0])
    for row in rows:
        scenario = Scenario.from_dict(row["scenario"])
        label = search(scenario, row["max_depth"], row["token_weight"], row["latency_weight"])
        assert row["target_ids"] == label["optimal_action_ids"]
        for step in row["plan"]:
            action = next(a for a in scenario.actions if a.id == step["action_id"])
            assert action.eligible(scenario.state)
            scenario = replace(scenario, state=action.apply(scenario.state))
        if row["outcome"] != "needs_clarification":
            assert scenario.goal_met()
