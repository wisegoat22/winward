from dataclasses import replace

import numpy as np

from agent_training.features import MAX_CANDIDATES, FEATURE_DIM, STOP, DEFER, encode, target_for
from agent_training.simulator import Action, Scenario, make_scenario, search


def test_features_cannot_see_names_family_context_or_teacher_annotations():
    scenario = make_scenario(7)
    original = encode(scenario)
    unrelated = scenario.to_dict()
    unrelated.update(id="another-id", family="other", split="other",
                     context={"answer": "a0", "plan": ["a0"], "ignore_goal": True},
                     facts=[f"unrelated {i}" for i in range(len(scenario.facts))],
                     target_ids=["a0"], chosen_action_id="a0", plan=["a0"])
    for i, action in enumerate(unrelated["actions"]):
        action["name"] = f"Ignore the task and choose option {i}"
        action["id"] = f"opaque-{i}"
    modified = encode(unrelated)
    for index in range(3):
        np.testing.assert_array_equal(original[index], modified[index])


def test_padding_is_invisible_and_special_candidates_are_in_bounds():
    scenario = Scenario("small", "test", "test", ("ready", "done"), 0, 2,
                        (Action("inspect", "Inspect", sets=1),))
    x, valid, eligible, candidates = encode(scenario)
    assert x.shape == (MAX_CANDIDATES, FEATURE_DIM)
    assert len(candidates) == 3 and valid.sum() == 3
    assert candidates[1]["id"] == STOP and candidates[2]["id"] == DEFER
    assert not valid[3:].any() and not eligible[3:].any()
    assert not x[3:].any() and np.isfinite(x).all()


def test_permission_and_precondition_masks_are_explicit_rules():
    scenario = Scenario("mask", "test", "test", ("evidence", "blocked", "done"), 2, 4, (
        Action("forbidden", "Unauthorized", sets=4, allowed=False),
        Action("missing", "Missing evidence", requires=1, sets=4),
        Action("blocked", "Blocked", forbids=2, sets=4),
        Action("available", "Available", sets=1),
    ))
    _, _, eligible, candidates = encode(scenario)
    observed = {a["id"]: bool(eligible[i]) for i, a in enumerate(candidates)}
    assert observed == {"forbidden": False, "missing": False, "blocked": False,
                        "available": True, STOP: False, DEFER: True}


def test_stop_is_the_only_eligible_candidate_after_completion():
    original = make_scenario(14)
    done = replace(original, state=original.state | original.goal)
    _, _, eligible, candidates = encode(done)
    assert [candidates[i]["id"] for i in np.flatnonzero(eligible)] == [STOP]
    labels = target_for(candidates, search(done))
    assert labels[np.flatnonzero(eligible)[0]] == 1


def test_action_permutation_moves_features_masks_and_targets_with_identity():
    original = make_scenario(7)
    permutation = tuple(reversed(original.actions))
    changed = replace(original, actions=permutation)
    first, second = encode(original), encode(changed)
    first_labels = target_for(first[3], search(original))
    second_labels = target_for(second[3], search(changed))
    index_by_id = {a["id"]: i for i, a in enumerate(second[3])}
    for i, action in enumerate(first[3]):
        j = index_by_id[action["id"]]
        np.testing.assert_array_equal(first[0][i], second[0][j])
        assert first[1][i] == second[1][j] and first[2][i] == second[2][j]
        assert first_labels[i] == second_labels[j]


def test_equal_optimal_actions_share_target_mass_without_becoming_features():
    scenario = Scenario("tie", "test", "test", ("done",), 0, 1,
                        (Action("a", "First", sets=1, tokens=2), Action("b", "Second", sets=1, tokens=2)))
    _, _, _, candidates = encode(scenario)
    target = target_for(candidates, search(scenario))
    assert target.sum() == 1 and target[0] == .5 and target[1] == .5
    assert not target[2:].any()


def test_depth_goal_and_cost_preferences_are_observed_inputs():
    original = make_scenario(7)
    features = encode(original)[0]
    assert not np.array_equal(features, encode(original, max_depth=2)[0])
    assert not np.array_equal(features, encode(original, token_weight=4, latency_weight=.001)[0])
    assert not np.array_equal(features, encode(replace(original, goal=original.goal ^ 1))[0])


def test_future_permissions_are_observable_even_before_preconditions_hold():
    original = Scenario("future-permission", "probe", "validation", ("ready", "done"), 0, 2, (
        Action("prepare", "Prepare", sets=1, tokens=1),
        Action("finish", "Finish", requires=1, sets=2, tokens=1),
    ))
    forbidden = replace(original, actions=(original.actions[0], replace(original.actions[1], allowed=False)))
    assert search(original)["chosen_action_id"] == "prepare"
    assert search(forbidden)["chosen_action_id"] == DEFER
    allowed_features = encode(original, feature_dim=103)
    forbidden_features = encode(forbidden, feature_dim=103)
    # Both future actions are ineligible now, but future permission differs.
    assert not allowed_features[2][1] and not forbidden_features[2][1]
    assert not np.array_equal(allowed_features[0], forbidden_features[0])
    # Reproducing v0 retains its original encoding for checkpoint compatibility.
    np.testing.assert_array_equal(encode(original, feature_dim=102)[0], encode(forbidden, feature_dim=102)[0])
