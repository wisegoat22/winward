from dataclasses import replace

import numpy as np
import pytest

from agent_lab.belief import (BeliefAction, BeliefProblem, Hypothesis, Outcome,
                              NEEDS_INFORMATION, STOP, branches, plan, update_belief)
from agent_training import belief_data
from agent_training.belief_data import (FEATURE_DIM, FAMILIES, encode_belief,
                                        generate_rows, make_problem, target_ids)


def uncertain():
    return BeliefProblem("example", ("verified", "blocked"), 1,
                        (Hypothesis("a", 0, .7), Hypothesis("b", 0, .3)),
                        (BeliefAction("inspect", "Inspect", tokens=1, outcomes_by_world={
                            "a": (Outcome(.8, "red"), Outcome(.2, "blue")),
                            "b": (Outcome(.2, "red"), Outcome(.8, "blue"))}),
                         BeliefAction("repair", "Repair a", forbids=2, tokens=3, outcomes_by_world={
                             "a": (Outcome(1., "done", sets=1),),
                             "b": (Outcome(1., "broken", sets=2),)}),
                         BeliefAction("safe", "Reliable repair", tokens=20,
                                      outcomes=(Outcome(1., "done", sets=1),))))


def test_encoder_shape_masks_and_search_separation(monkeypatch):
    monkeypatch.setattr(belief_data, "plan", lambda *a, **k: (_ for _ in ()).throw(AssertionError("planner called")))
    x, valid, eligible, candidates = encode_belief(uncertain())
    assert x.shape == (12, FEATURE_DIM)
    assert x.dtype == np.float32
    assert np.isfinite(x).all()
    assert valid.sum() == 5
    assert [c["id"] for c in candidates][-2:] == [STOP, NEEDS_INFORMATION]
    assert not eligible[3] and eligible[4]
    assert not valid[5:].any()
    assert not eligible[5:].any()


def test_observation_renaming_and_order_are_irrelevant():
    problem = uncertain()
    names = {"red": "zz", "blue": "aa", "done": "value3", "broken": "value4"}
    def convert(outcomes):
        return tuple(replace(o, observation=names.get(o.observation, "other")) for o in reversed(outcomes))
    changed = replace(problem, actions=tuple(replace(a, outcomes=convert(a.outcomes),
                    outcomes_by_world={w: convert(outcomes) for w, outcomes in a.outcomes_by_world.items()})
                    for a in problem.actions))
    for before, after in zip(encode_belief(problem)[:3], encode_belief(changed)[:3]):
        np.testing.assert_array_equal(before, after)


def test_ids_text_and_hidden_metadata_cannot_change_features():
    problem = uncertain()
    renamed = replace(problem, id="some other id", facts=("emotion", "unrelated text"), context="PANIC or CELEBRATE",
                      belief=tuple(replace(h, world_id={"a": "zz", "b": "aa"}[h.world_id]) for h in reversed(problem.belief)),
                      actions=tuple(replace(a, id=f"other_{i}", name="NOISE", outcomes_by_world={
                          {"a": "zz", "b": "aa"}[w]: rows for w, rows in a.outcomes_by_world.items()})
                          for i, a in enumerate(problem.actions)))
    serialized = renamed.to_dict()
    serialized["actual_world"] = "secret hidden state MUST NOT be an observation"
    serialized["teacher_action"] = "inspect"
    serialized["family"] = "leaked answer"
    for before, after in zip(encode_belief(problem)[:3], encode_belief(serialized)[:3]):
        np.testing.assert_array_equal(before, after)


def test_action_permutation_equivariance():
    problem = uncertain()
    permutation = [2, 0, 1]
    changed = replace(problem, actions=tuple(problem.actions[i] for i in permutation))
    original = encode_belief(problem)
    permuted = encode_belief(changed)
    permutation += list(range(3, 12))
    for before, after in zip(original[:3], permuted[:3]):
        np.testing.assert_array_equal(before[permutation], after)


def test_symmetric_worlds_do_not_use_id_or_action_order_tiebreak():
    problem = replace(uncertain(), belief=(Hypothesis("a", 0, .5), Hypothesis("b", 0, .5)))
    changed = replace(problem, belief=(Hypothesis("a", 0, .5), Hypothesis("z", 0, .5)),
                      actions=tuple(replace(a, outcomes_by_world={
                          {"a": "z", "b": "a"}[w]: rows for w, rows in a.outcomes_by_world.items()})
                          for a in reversed(problem.actions)))
    first = encode_belief(problem)[0]
    second = encode_belief(changed)[0]
    np.testing.assert_array_equal(first[[2, 1, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11]], second)


def test_tiny_probability_uncertainty_prevents_false_stop():
    problem = replace(uncertain(), belief=(Hypothesis("a", 1, 1 - 1e-12), Hypothesis("b", 0, 1e-12)))
    x, _, eligible, candidates = encode_belief(problem)
    stop = next(i for i, c in enumerate(candidates) if c["id"] == STOP)
    assert not eligible[stop]
    assert np.any(x[0] == np.float32(1e-12))


def test_finish_and_permissions_are_hard_rules():
    problem = uncertain()
    problem = replace(problem, actions=(replace(problem.actions[0], allowed=False), *problem.actions[1:]))
    assert not encode_belief(problem)[2][0]
    complete = replace(problem, goal=0)
    _, _, eligible, candidates = encode_belief(complete)
    assert eligible.sum() == 1
    assert candidates[int(eligible.argmax())]["id"] == STOP


def test_capacity_errors_never_truncate():
    problem = uncertain()
    too_many = replace(problem, belief=tuple(Hypothesis(str(i), 0, .2) for i in range(5)))
    with pytest.raises(ValueError, match="4 live hypotheses"):
        encode_belief(too_many)
    many_outcomes = BeliefAction("three", "Three", outcomes=(Outcome(.2), Outcome(.3), Outcome(.5)))
    with pytest.raises(ValueError, match="2 outcomes"):
        encode_belief(replace(problem, actions=(many_outcomes,)))


@pytest.mark.parametrize("kwargs", [{"depth": 0}, {"depth": 6}, {"depth": True},
                                    {"token_weight": float("nan")}, {"latency_weight": -1},
                                    {"token_weight": 0, "latency_weight": 0}])
def test_encoder_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        encode_belief(uncertain(), **kwargs)


def test_noisy_observation_uses_bayes_and_changes_features():
    problem = uncertain()
    posterior = update_belief(problem, "inspect", "red")
    assert posterior.belief[0].probability == pytest.approx(.56 / .62)
    assert len(posterior.belief) == 2
    assert not np.array_equal(encode_belief(problem)[0], encode_belief(posterior)[0])


def test_generation_reproducible_complete_and_bounded_without_test_instances():
    assert set(FAMILIES["train"]).isdisjoint(FAMILIES["validation"])
    assert set(FAMILIES["test"]).isdisjoint(FAMILIES["train"] + FAMILIES["validation"])
    for split, seed in (("train", 310_000_000), ("validation", 320_000_000)):
        rows, arrays = generate_rows(30, split, seed)
        again, arrays_again = generate_rows(30, split, seed)
        assert rows == again
        for a, b in zip(arrays, arrays_again):
            np.testing.assert_array_equal(a, b)
        assert all(row["teacher"]["search_complete"] for row in rows)
        assert any(row["public_history"] for row in rows)
        assert len({row["family"] for row in rows}) == len(FAMILIES[split])
        for row in rows:
            p = BeliefProblem.from_dict(row["problem"])
            for a in p.actions:
                if a.eligible(p.belief):
                    for _, _, posterior in branches(p.belief, a):
                        encode_belief(replace(p, belief=posterior))


def test_teacher_includes_ties_without_action_id_supervision():
    problem = BeliefProblem("ties", ("done",), 1, (Hypothesis("w", 0, 1.),),
                           (BeliefAction("a", "first", outcomes=(Outcome(1., sets=1),), tokens=2),
                            BeliefAction("b", "second", outcomes=(Outcome(1., sets=1),), tokens=2)))
    assert target_ids(plan(problem)) == ["a", "b"]


def test_changed_goal_changes_observation():
    problem = uncertain()
    assert not np.array_equal(encode_belief(problem)[0], encode_belief(replace(problem, goal=2))[0])
