from dataclasses import replace
from collections import Counter

import numpy as np
import pytest

from agent_lab.belief import (BeliefProblem, Hypothesis, NEEDS_INFORMATION, STOP,
                              branches, plan, verified)
from agent_training import belief_data_v4 as data


def anchor_seed(anchor, split="train"):
    base = data.SEED_DOMAINS[split]
    return next(s for s in range(base, base + 12) if data.family_for_seed(s, split) == anchor)


def test_fresh_seed_protocol_and_shared_composition_contract():
    assert data.SEED_DOMAINS == {"train": 410_000_000, "validation": 420_000_000, "test": 430_000_000}
    assert data.FAMILIES["train"] == data.FAMILIES["validation"] == data.FAMILIES["test"]
    # Final test problems are deliberately never instantiated by this suite.
    assert len(data.ANCHORS) == 12


@pytest.mark.parametrize("split", ["train", "validation"])
def test_generation_reproducibility_exact_labels_and_auxiliary_alignment(split):
    seed = data.SEED_DOMAINS[split]
    rows, arrays = data.generate_rows(36, split, seed)
    again, repeated = data.generate_rows(36, split, seed)
    assert rows == again
    for first, second in zip(arrays, repeated):
        np.testing.assert_array_equal(first, second)
    assert arrays[0].shape == (36, 12, 400)
    assert set(r["family"] for r in rows) == set(data.ANCHORS)
    assert any(r["public_history"] for r in rows)
    for i, row in enumerate(rows):
        problem = BeliefProblem.from_dict(row["problem"])
        solution = plan(problem, row["max_depth"], row["token_weight"], row["latency_weight"], max_nodes=100_000)
        assert solution["search_complete"]
        assert row["target_ids"] == data.target_ids(solution)
        x, valid, eligible, candidates = data.encode_belief(problem, row["max_depth"], row["token_weight"], row["latency_weight"])
        np.testing.assert_array_equal(arrays[0][i], x)
        assert arrays[3][i].sum() == pytest.approx(1.)
        teacher = row["teacher"]
        assert teacher["candidate_ids"][:len(candidates)] == [c["id"] for c in candidates]
        mask = np.array(teacher["candidate_value_mask"])
        values = np.array(teacher["candidate_expected_success"])
        regret = np.array(teacher["candidate_success_regret"])
        assert mask.shape == values.shape == regret.shape == (12,)
        assert np.all((0 <= values) & (values <= 1))
        assert not mask[~eligible].any()
        assert mask[arrays[3][i] > 0].all()
        assert np.max(regret[arrays[3][i] > 0]) < 1e-7
        for score in solution["action_scores"]:
            slot = teacher["candidate_ids"].index(score["action_id"])
            if score.get("eligible") and not score.get("pruned"):
                assert values[slot] == pytest.approx(score["expected_verified_success"])
            else:
                assert not mask[slot]


@pytest.mark.parametrize("split", ["train", "validation"])
def test_every_five_step_reachable_posterior_fits_encoder(split):
    # Traverse every eligible action and every observable branch, not only the
    # teacher path, to check no ambiguous effect can split a world into >4 rows.
    for seed in range(data.SEED_DOMAINS[split], data.SEED_DOMAINS[split] + 12):
        problem = data.make_problem(seed, split)
        frontier = {problem.belief}
        seen = set()
        for _ in range(6):
            next_frontier = set()
            for belief in frontier - seen:
                seen.add(belief)
                assert len(belief) <= 4
                assert all(h.probability > 0 for h in belief)
                current = replace(problem, belief=belief)
                data.encode_belief(current)
                for action in problem.actions:
                    if action.eligible(belief):
                        next_frontier.update(posterior for _, _, posterior in branches(belief, action))
            frontier = next_frontier


@pytest.mark.parametrize("split", ["train", "validation"])
def test_preparation_is_a_successful_multistep_training_root(split):
    seed = anchor_seed("preparation_chain", split)
    for current_seed in (seed, seed + 12, seed + 24):
        problem = data.make_problem(current_seed, split)
        solution = plan(problem, 5, max_nodes=100_000)
        assert solution["search_complete"]
        assert solution["expected_verified_success"] == 1
        assert solution["expected_steps"] == 5
        action = next(a for a in problem.actions if a.id == solution["chosen_action_id"])
        assert "access" in action.name.lower()
        assert all(not verified(posterior, problem.goal) for _, _, posterior in branches(problem.belief, action))
        assert not any(o.sets & problem.goal for h in problem.belief for o in action.effects(h.world_id))


@pytest.mark.parametrize("split", ["train", "validation"])
def test_terminal_anchors_are_explicit_and_do_not_dominate(split):
    rows, _ = data.generate_rows(120, split, data.SEED_DOMAINS[split])
    totals = Counter("finished" if r["target_ids"] == [STOP] else
                     "defer" if r["target_ids"] == [NEEDS_INFORMATION] else "action" for r in rows)
    assert totals["finished"] >= 10
    assert totals["defer"] >= 10
    assert totals["action"] >= 80
    for row in rows:
        if row["family"] == "completed":
            assert row["target_ids"] == [STOP]
        elif row["family"] == "unreachable":
            assert row["target_ids"] == [NEEDS_INFORMATION]


def test_teacher_metadata_never_enters_features(monkeypatch):
    problem = data.make_problem(anchor_seed("prepared_diagnosis"))
    baseline = data.encode_belief(problem)
    payload = problem.to_dict()
    payload.update(actual_world="secret", teacher={"expected_verified_success": 123},
                   family="answer leak", candidate_expected_success=[999] * 12)
    monkeypatch.setattr(data, "plan", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("teacher invoked")))
    changed = data.encode_belief(payload)
    for before, after in zip(baseline[:3], changed[:3]):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("count,split,seed", [(0, "train", 410_000_000), (True, "train", 410_000_000),
                                             (1, "other", 410_000_000), (1, "train", True)])
def test_invalid_generation_arguments(count, split, seed):
    with pytest.raises(ValueError):
        data.generate_rows(count, split, seed)
