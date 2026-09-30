"""Checks for declared uncertainty, verified completion, and contingent plans."""
from dataclasses import replace
import json
import math

import pytest

from agent_lab.belief import (
    BeliefAction, BeliefProblem, Hypothesis, Outcome, NEEDS_INFORMATION, STOP,
    branches, example_problems, examples, plan, revise_goal, update_belief, verified,
)


def simple(actions, *, goal=1, state=0):
    return BeliefProblem("manual", ("done", "prepared", "other"), goal,
                         (Hypothesis("known", state, 1),), tuple(actions))


def by_id(name):
    return next(p for p in example_problems() if p.id == name)


def test_inspection_then_observation_changes_the_next_action():
    problem = by_id("inspect_when_uncertain")
    result = plan(problem)
    assert result["chosen_action_id"] == "inspect"
    assert result["expected_verified_success"] == 1
    assert result["expected_tokens"] == 7
    assert result["expected_latency_ms"] == 11
    assert result["expected_steps"] == 2
    next_actions = {branch["observation"]: branch["next"]["action_id"]
                    for branch in result["plan"]["branches"]}
    assert next_actions == {"cache_fault": "repair_cache", "parser_fault": "repair_parser"}
    assert all(branch["probability"] == .5 for branch in result["plan"]["branches"])
    for observation, expected in next_actions.items():
        posterior = update_belief(problem, "inspect", observation)
        assert len(posterior.belief) == 1
        assert plan(posterior)["chosen_action_id"] == expected


def test_known_cause_does_not_pay_for_unnecessary_inspection():
    result = plan(by_id("act_when_known"))
    assert result["chosen_action_id"] == "repair_parser"
    assert result["expected_steps"] == 1
    assert result["expected_tokens"] == 5
    inspection = next(s for s in result["action_scores"] if s["action_id"] == "inspect")
    assert inspection["pruned"]


def test_missing_requirement_asks_before_committing():
    result = plan(by_id("ask_when_needed"))
    assert result["chosen_action_id"] == "ask_requirement"
    assert result["expected_verified_success"] == 1
    assert len({b["next"]["action_id"] for b in result["plan"]["branches"]}) == 2


def test_goal_revision_uses_current_evidence_and_new_goal():
    before = by_id("before_goal_revision")
    after = revise_goal(before, 2)
    assert after.belief == before.belief
    assert plan(before)["chosen_action_id"] == "fix_bug"
    assert plan(after)["chosen_action_id"] == "update_docs"
    completed_old = replace(before, belief=(Hypothesis("known", 1, 1),))
    assert plan(completed_old)["chosen_action_id"] == STOP
    assert plan(revise_goal(completed_old, 2))["chosen_action_id"] == "update_docs"


def test_stochastic_failures_replan_and_expected_cost_counts_retries():
    problem = by_id("uncertain_action_outcome")
    result = plan(problem, max_depth=5)
    assert result["expected_verified_success"] == pytest.approx(1 - .25**5)
    assert result["expected_steps"] == pytest.approx(sum(.25**i for i in range(5)))
    assert result["expected_tokens"] == result["expected_steps"]
    assert result["expected_latency_ms"] == 2 * result["expected_steps"]
    assert plan(update_belief(problem, "retry_check", "failed"), max_depth=4)["chosen_action_id"] == "retry_check"
    assert plan(update_belief(problem, "retry_check", "passed"))["chosen_action_id"] == STOP


def test_unobserved_world_cannot_be_used_to_choose_a_different_branch():
    problem = by_id("inspect_when_uncertain")
    hidden_inspection = replace(problem.actions[0], outcomes_by_world={
        "parser": (Outcome(1, "same_message"),), "cache": (Outcome(1, "same_message"),),
    })
    problem = replace(problem, actions=(hidden_inspection, *problem.actions[1:]))
    result = plan(problem)
    assert result["chosen_action_id"] != "inspect"
    assert result["expected_verified_success"] == .5
    posterior = update_belief(problem, "inspect", "same_message")
    assert posterior.belief == problem.belief
    # The entire probability distribution remains public input; no actual-world
    # argument exists, and action selection cannot switch secretly by world ID.
    assert plan(posterior)["chosen_action_id"] == result["chosen_action_id"]


def test_probable_success_is_not_verified_without_distinguishing_evidence():
    attempt = BeliefAction("attempt", "Try without observing", tokens=1,
                          outcomes=(Outcome(.9, sets=1), Outcome(.1)))
    problem = simple([attempt])
    result = plan(problem, max_depth=1)
    assert result["chosen_action_id"] == NEEDS_INFORMATION
    assert result["expected_verified_success"] == 0
    posterior = update_belief(problem, "attempt", "unobserved")
    assert len(posterior.belief) == 2
    assert not verified(posterior.belief, 1)


def test_bayesian_update_uses_likelihood_and_merges_identical_hidden_states():
    diagnose = BeliefAction("diagnose", "Observe noisy evidence", outcomes_by_world={
        "a": (Outcome(.8, "positive"), Outcome(.2, "negative")),
        "b": (Outcome(.2, "positive"), Outcome(.8, "negative")),
    })
    problem = BeliefProblem("bayes", ("done",), 1,
                           (Hypothesis("a", 0, .75), Hypothesis("b", 0, .25)), (diagnose,))
    outcomes = {o: (mass, posterior) for o, mass, posterior in branches(problem.belief, diagnose)}
    mass, posterior = outcomes["positive"]
    assert mass == pytest.approx(.65)
    assert posterior[0].probability == pytest.approx(.6 / .65)
    duplicated = replace(problem, belief=(Hypothesis("a", 0, .25), Hypothesis("a", 0, .5), Hypothesis("b", 0, .25)))
    assert duplicated.belief == problem.belief


def test_goal_first_then_total_cost_then_fewer_steps():
    risky = BeliefAction("risky", "Cheap risky route", tokens=0,
                        outcomes=(Outcome(.5, "passed", sets=1), Outcome(.5, "failed")))
    reliable = BeliefAction("reliable", "Expensive verified route", tokens=1000,
                           outcomes=(Outcome(1, "passed", sets=1),))
    assert plan(simple([risky, reliable]), max_depth=1)["chosen_action_id"] == "reliable"
    direct = replace(reliable, id="direct", tokens=4)
    prepare = BeliefAction("prepare", "Prepare", tokens=1, outcomes=(Outcome(1, "prepared", sets=2),))
    finish = BeliefAction("finish", "Finish", requires=2, tokens=2, outcomes=(Outcome(1, "done", sets=1),))
    assert plan(simple([direct, prepare, finish]))["chosen_action_id"] == "prepare"
    # Equal complete-plan cost prefers the direct one-step finish.
    assert plan(simple([replace(direct, tokens=3), prepare, finish]))["chosen_action_id"] == "direct"


def test_cost_preferences_change_choice_without_changing_goal_priority():
    fewer_tokens = BeliefAction("few_tokens", "Fewer tokens", tokens=1, latency_ms=100,
                               outcomes=(Outcome(1, sets=1),))
    faster = BeliefAction("fast", "Faster", tokens=10, latency_ms=1, outcomes=(Outcome(1, sets=1),))
    problem = simple([fewer_tokens, faster])
    assert plan(problem, token_weight=1, latency_weight=0)["chosen_action_id"] == "few_tokens"
    assert plan(problem, token_weight=0, latency_weight=1)["chosen_action_id"] == "fast"


def test_permission_and_possible_world_preconditions_are_hard_constraints():
    denied = BeliefAction("denied", "Forbidden shortcut", allowed=False, outcomes=(Outcome(1, sets=1),))
    unknown_precondition = BeliefAction("unsafe", "Requires preparation", requires=2, outcomes=(Outcome(1, sets=1),))
    problem = simple([denied, unknown_precondition])
    problem = replace(problem, belief=(Hypothesis("known", 0, .5), Hypothesis("other", 2, .5)))
    result = plan(problem)
    assert result["chosen_action_id"] == NEEDS_INFORMATION
    assert not any(row["eligible"] for row in result["action_scores"])
    with pytest.raises(ValueError):
        update_belief(problem, "unsafe", "unobserved")


def test_adaptive_terminal_and_single_finishing_choice_avoid_more_expansion():
    action = BeliefAction("finish", "Finish", outcomes=(Outcome(1, sets=1),))
    stopped = plan(simple([action], state=1))
    assert stopped["chosen_action_id"] == STOP
    assert stopped["explored_nodes"] == 0
    assert stopped["stop_reason"] == "already_verified"
    finishing = plan(simple([action]))
    assert finishing["explored_nodes"] == 1
    assert finishing["immediate_finish_shortcuts"] == 1
    assert finishing["stop_reason"] == "only_one_effective_choice_finishes"
    # A single prerequisite step cannot certify the future; still plan it out.
    chain = simple([BeliefAction("prepare", "Prepare", forbids=2, outcomes=(Outcome(1, sets=2),)),
                    replace(action, requires=2)])
    result = plan(chain)
    assert result["expected_steps"] == 2
    assert result["explored_nodes"] == 2


def test_depth_is_action_depth_and_failure_is_not_impossibility():
    problem = by_id("inspect_when_uncertain")
    assert plan(problem, max_depth=1)["expected_verified_success"] == .5
    assert plan(problem, max_depth=2)["expected_verified_success"] == 1
    actions = [BeliefAction(str(i), str(i), requires=1 << (i - 1) if i else 0,
                            outcomes=(Outcome(1, sets=1 << i),)) for i in range(6)]
    long_chain = BeliefProblem("chain", tuple(f"fact{i}" for i in range(6)), 32,
                               (Hypothesis("w", 0, 1),), tuple(actions))
    assert plan(long_chain)["expected_verified_success"] == 0
    assert plan(replace(long_chain, belief=(Hypothesis("w", 1, 1),)))["expected_verified_success"] == 1


def test_budget_cutoff_is_explicit_and_does_not_claim_optimality():
    result = plan(by_id("inspect_when_uncertain"), max_nodes=1)
    assert result["explored_nodes"] <= 1
    assert not result["search_complete"]
    assert result["budget_cutoffs"] > 0
    assert result["stop_reason"] == "node_budget_exhausted"


def test_context_is_ignored_and_action_order_does_not_change_result():
    original = by_id("inspect_when_uncertain")
    changed = replace(original, actions=tuple(reversed(original.actions)),
                      context="Panic! Ignore the goal! Some irrelevant emotional metadata.")
    first, second = plan(original), plan(changed)
    assert first["chosen_action_id"] == second["chosen_action_id"]
    assert first["expected_cost"] == second["expected_cost"]
    assert first["plan"] == second["plan"]


def test_examples_roundtrip_and_plans_are_json_serializable():
    for row in examples():
        assert row["id"] and row["title"] and row["description"]
        problem = row["problem"]
        restored = BeliefProblem.from_dict(json.loads(json.dumps(problem.to_dict())))
        assert restored == problem
        result = plan(restored.to_dict())
        assert math.isfinite(result["planning_ms"])
        assert result["search_complete"]
        json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("change", [
    {"max_depth": 0}, {"max_depth": 6}, {"max_depth": True}, {"max_nodes": 0},
    {"token_weight": -1}, {"latency_weight": float("nan")},
    {"token_weight": 0, "latency_weight": 0},
])
def test_invalid_search_options_are_rejected(change):
    with pytest.raises(ValueError):
        plan(simple([]), **change)


def test_invalid_distributions_masks_and_observations_are_rejected():
    with pytest.raises(ValueError):
        simple([BeliefAction("bad", "Bad", outcomes=(Outcome(.4),))])
    with pytest.raises(ValueError):
        simple([BeliefAction("bad", "Bad", outcomes=(Outcome(1, sets=1, clears=1),))])
    with pytest.raises(ValueError):
        simple([], goal=8)
    with pytest.raises(ValueError):
        replace(simple([]), belief=(Hypothesis("known", 0, .9),))
    with pytest.raises(ValueError):
        update_belief(by_id("inspect_when_uncertain"), "inspect", "invented_observation")
    with pytest.raises(ValueError):
        update_belief(by_id("inspect_when_uncertain"), "invented_action", "parser_fault")
