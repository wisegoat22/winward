"""Independent planner checks and adversarial decision counterfactuals."""

from dataclasses import replace
import itertools
import json
import math

import pytest

from agent_training.simulator import (
    Action, Scenario, STOP, NEEDS_CLARIFICATION, SPLIT_FAMILIES,
    make_scenario, search,
)


def world(actions, state=0, goal=4, facts=("evidence", "prepared", "done")):
    return Scenario("manual", "manual", "test", facts, state, goal, tuple(actions))


def brute_force(scenario, depth=5, token_weight=1.0, latency_weight=0.01):
    """Slow independent enumeration without memoization or pruning."""
    found = []
    for length in range(depth + 1):
        for sequence in itertools.product(scenario.actions, repeat=length):
            state = scenario.state
            for action in sequence:
                if not action.allowed or state & action.requires != action.requires or state & action.forbids:
                    break
                state = (state & ~action.clears) | action.sets
            else:
                if state & scenario.goal == scenario.goal:
                    found.append((sum(a.tokens * token_weight + a.latency_ms * latency_weight for a in sequence),
                                  len(sequence), tuple(a.id for a in sequence)))
    return min(found) if found else None


def test_goal_first_even_when_success_is_expensive():
    scenario = world([Action("noise", "Cheap noise", sets=1, tokens=0),
                      Action("win", "Reach the goal", sets=4, tokens=10000)])
    result = search(scenario)
    assert result["chosen_action_id"] == "win"
    assert result["success"] and result["total_cost"] == 10000


def test_stop_when_goal_met_is_not_more_unnecessary_work():
    result = search(world([Action("free", "Free unnecessary work", sets=1)], state=4))
    assert result["chosen_action_id"] == STOP
    assert result["optimal_action_ids"] == [STOP]
    assert result["plan"] == [] and result["can_stop"] and result["success"]


def test_permissions_and_positive_and_negative_preconditions_are_hard_constraints():
    scenario = world([
        Action("forbidden", "Unauthorized shortcut", sets=4, allowed=False),
        Action("missing", "Missing evidence", requires=1, sets=4),
        Action("blocked", "Blocked shortcut", forbids=2, sets=4),
        Action("inspect", "Gather evidence", sets=1, tokens=2),
    ], state=2)
    result = search(scenario)
    assert result["plan_action_ids"] == ["inspect", "missing"]
    assert not next(s for s in result["action_scores"] if s["action_id"] == "forbidden")["eligible"]
    with pytest.raises(ValueError):
        scenario.actions[0].apply(scenario.state)


def test_five_consequences_and_horizon_failure_are_distinct():
    actions = tuple(Action(str(i), f"Step {i}", requires=1 << (i - 1) if i else 0,
                           sets=1 << i, tokens=1) for i in range(6))
    scenario = world(actions, goal=32, facts=tuple(f"f{i}" for i in range(6)))
    result = search(scenario)
    assert result["chosen_action_id"] == NEEDS_CLARIFICATION
    assert not result["success"] and result["outcome"] == "needs_clarification"
    assert all(s["cost"] is None for s in result["action_scores"])
    with pytest.raises(ValueError):
        search(scenario, 6)
    # The same task is solvable after one piece of progress; it was not impossible.
    assert search(replace(scenario, state=1))["depth"] == 5


def test_counterproductive_cheap_action_loses_after_searching_consequences():
    scenario = world([
        Action("quick", "Quick destructive operation", requires=1, sets=2, clears=1, tokens=1),
        Action("careful", "Preserve required evidence", requires=1, sets=2, tokens=8),
        Action("recover", "Recollect lost evidence", sets=1, tokens=20),
        Action("finish", "Validate and finish", requires=3, sets=4, tokens=2),
    ], state=1)
    result = search(scenario)
    assert result["plan_action_ids"] == ["careful", "finish"]
    scores = {s["action_id"]: s for s in result["action_scores"]}
    assert scores["quick"]["cost"] == 23 and scores["careful"]["cost"] == 10


def test_cost_and_goal_counterfactuals_change_action_selection():
    original = world([Action("a", "Slow compact action", sets=4, tokens=5, latency_ms=1000),
                      Action("b", "Fast verbose action", sets=4, tokens=10, latency_ms=1),
                      Action("c", "Different goal", sets=2, tokens=1)])
    assert search(original)["chosen_action_id"] == "b"
    assert search(original, latency_weight=0)["chosen_action_id"] == "a"
    assert search(replace(original, goal=2))["chosen_action_id"] == "c"
    assert search(replace(original, actions=(replace(original.actions[0], latency_ms=0), *original.actions[1:])))["chosen_action_id"] == "a"


def test_equal_cost_valid_choices_are_all_recorded_and_permutation_invariant():
    scenario = world([Action("b", "Route B", sets=4, tokens=5), Action("a", "Route A", sets=4, tokens=5)])
    first = search(scenario)
    second = search(replace(scenario, actions=tuple(reversed(scenario.actions))))
    assert set(first["optimal_action_ids"]) == {"a", "b"}
    assert first["chosen_action_id"] == second["chosen_action_id"] == "a"


def test_zero_cost_cycles_terminate_and_zero_depth_has_explicit_outcome():
    scenario = world([Action("loop", "No effect"), Action("set", "Set evidence", sets=1),
                      Action("clear", "Clear evidence", clears=1)])
    assert search(scenario)["chosen_action_id"] == NEEDS_CLARIFICATION
    assert search(scenario, max_depth=0)["outcome"] == "needs_clarification"
    assert search(replace(scenario, goal=0), max_depth=0)["chosen_action_id"] == STOP


@pytest.mark.parametrize("bad", [-1, math.inf, math.nan, True])
def test_invalid_costs_rejected(bad):
    with pytest.raises(ValueError):
        Action("bad", "Invalid", tokens=bad)
    with pytest.raises(ValueError):
        search(world([]), latency_weight=bad)


def test_masks_and_ids_validated():
    with pytest.raises(ValueError):
        Action("x", "Conflict", sets=1, clears=1)
    with pytest.raises(ValueError):
        Action("x", "Conflict", requires=1, forbids=1)
    with pytest.raises(ValueError):
        world([Action("x", "Unknown fact", sets=8)])
    with pytest.raises(ValueError):
        world([Action("x", "First"), Action("x", "Second")])


@pytest.mark.parametrize("split", list(SPLIT_FAMILIES))
def test_generation_is_deterministic_serializable_and_replayable(split):
    outcomes = set()
    for seed in range(80):
        scenario = make_scenario(seed, split)
        assert scenario == make_scenario(seed, split)
        assert Scenario.from_dict(json.loads(json.dumps(scenario.to_dict()))) == scenario
        assert len(scenario.facts) <= 12 and len(scenario.actions) <= 10
        result = search(scenario)
        outcomes.add(result["outcome"])
        current = scenario.state
        actions = {a.id: a for a in scenario.actions}
        for step in result["plan"]:
            assert step["state_before"] == current
            current = actions[step["action_id"]].apply(current)
            assert current == step["state_after"]
        if result["success"]:
            assert scenario.goal_met(current)
        assert result["depth"] <= 5
    assert outcomes == {"plan", "stop", "needs_clarification"}


def test_family_and_seed_partitions_do_not_overlap():
    families = list(map(set, SPLIT_FAMILIES.values()))
    assert not any(a & b for a, b in itertools.combinations(families, 2))
    assert len({json.dumps(make_scenario(10, split).to_dict(), sort_keys=True) for split in SPLIT_FAMILIES}) == 3
    with pytest.raises(ValueError):
        make_scenario(1, "train", "reversible_trap")


def test_planner_matches_independent_exhaustive_enumeration_on_heldout_families():
    for family in SPLIT_FAMILIES["test"]:
        for seed in range(6):
            scenario = make_scenario(seed, "test", family)
            expected = brute_force(scenario)
            actual = search(scenario)
            assert actual["success"] == (expected is not None)
            if expected is not None:
                assert actual["total_cost"] == pytest.approx(expected[0])
                assert actual["depth"] == expected[1]
                assert tuple(actual["plan_action_ids"]) == expected[2]


def test_independent_bruteforce_agrees_for_arbitrary_graphs_with_clearing_effects():
    import random
    rng = random.Random(419)
    for i in range(24):
        actions = []
        for j in range(4):
            required, forbidden = rng.randrange(8), rng.randrange(8)
            sets, clears = rng.randrange(8), rng.randrange(8)
            actions.append(Action(str(j), f"Action {j}", required, forbidden & ~required,
                                  sets, clears & ~sets, rng.randrange(10), rng.randrange(20), rng.random() > .1))
        scenario = world(actions, state=rng.randrange(8), goal=rng.randrange(8))
        actual = search(scenario)
        expected = brute_force(scenario)
        assert actual["success"] == (expected is not None)
        if expected is not None:
            assert actual["total_cost"] == pytest.approx(expected[0])
            assert actual["depth"] == expected[1]
