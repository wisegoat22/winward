"""Compositional public-belief curriculum for the fourth policy experiment.

V3's final failures are development evidence for this curriculum. All splits
sample the same primitives/compositions with independent seed domains; the
final evaluation is a fresh-seed test, not an unseen-family claim. Defining the
test split does not generate it. Final seeds are reserved until model freeze.

The encoder is unchanged: only declared beliefs, effects and observations enter
its 400 features. Exact teacher values live solely in training-label metadata.
"""
from dataclasses import replace
import math
import random

import numpy as np

from agent_lab.belief import (BeliefAction, BeliefProblem, Hypothesis, Outcome,
                              NEEDS_INFORMATION, STOP, branches, plan, verified)
from .belief_data import (FEATURE_DIM, MAX_CANDIDATES, _scramble, encode_belief,
                          target_ids)


SEED_DOMAINS = {"train": 410_000_000, "validation": 420_000_000, "test": 430_000_000}
ANCHORS = ("prepared_diagnosis", "missing_requirements", "noisy_diagnosis",
           "stochastic_verification", "reversible_repair", "goal_revision",
           "known_cause", "completed", "unreachable", "composed",
           "preparation_chain", "efficient_choice")
FAMILIES = {split: ANCHORS for split in SEED_DOMAINS}
FACTS = ("workspace_ready", "access_ready", "repair_applied", "acceptance_verified",
         "repair_blocked", "documentation_verified", "evidence_obtained")


def _split_name(split):
    split = "validation" if split == "valid" else split
    if split not in FAMILIES:
        raise ValueError("Split must be train, validation, or test.")
    return split


def family_for_seed(seed, split):
    _split_name(split)
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer.")
    return ANCHORS[seed % len(ANCHORS)]


def _cost(rng, low=1, high=30):
    return {"tokens": rng.randint(low, high), "latency_ms": rng.randint(1, 180)}


def make_problem(seed, split="train"):
    """Build a bounded software decision using randomized primitive composition.

Anchors ensure coverage of important states while optional primitives overlap
between anchors. Preparation and prerequisite chains occur in training roots,
not only in a held-out family. No persistent actual world is sampled.
"""
    split = _split_name(split)
    family = family_for_seed(seed, split)
    rng = random.Random(seed)
    n = rng.choices((1, 2, 3, 4), (2, 5, 2, 1))[0]
    prep = rng.choices((0, 1, 2), (5, 4, 1))[0]
    noisy = rng.random() < .25
    flaky = rng.random() < .3
    reversible = rng.random() < .25
    revision = rng.random() < .2
    integrated = rng.random() < .25
    if family == "prepared_diagnosis":
        prep, n = 1, rng.choice((2, 3, 4))
    elif family == "missing_requirements":
        n, prep = rng.choice((2, 3)), rng.choice((0, 1))
    elif family == "noisy_diagnosis":
        n, noisy, prep = rng.choice((2, 3)), True, rng.choice((0, 1))
    elif family == "stochastic_verification":
        flaky, integrated = True, False
    elif family == "reversible_repair":
        n, reversible = rng.choice((2, 3)), True
    elif family == "goal_revision":
        revision = True
    elif family == "known_cause":
        n = 1
    elif family == "preparation_chain":
        prep, n, integrated, revision, flaky = 2, rng.choice((2, 3)), False, False, False
    elif family == "efficient_choice":
        n, prep, reversible = rng.choice((1, 2)), 0, False

    # Two preparations, four repairs, diagnosis, verification and documentation
    # use nine actions; optional primitives fill at most one remaining slot.
    ready, access, fixed, tested, blocked, docs, evidence = (1 << i for i in range(7))
    initial = (ready if prep == 0 else 0) | (access if prep < 2 else 0)
    worlds = [f"cause_{i}" for i in range(n)]
    masses = [rng.uniform(.05, 1.) for _ in worlds]
    mass_total = math.fsum(masses)
    belief = tuple(Hypothesis(w, initial, m / mass_total) for w, m in zip(worlds, masses))
    actions = []
    if prep == 2:
        actions.append(BeliefAction("access", "Obtain access to the isolated workspace", forbids=access,
                                   outcomes=(Outcome(1., "access_ready", sets=access),), **_cost(rng, 1, 8)))
    if prep:
        actions.append(BeliefAction("prepare", "Prepare the isolated workspace", requires=access, forbids=ready,
                                   outcomes=(Outcome(1., "workspace_ready", sets=ready),), **_cost(rng, 1, 12)))

    reliable_available = not (family == "noisy_diagnosis" and rng.random() < .5)
    inspect_cost = _cost(rng, 2, 36)
    actions.append(BeliefAction("inspect", "Clarify the missing requirement" if family == "missing_requirements"
                               else "Inspect the uncertain cause", requires=ready,
                               outcomes_by_world={w: (Outcome(1., f"observed_{i}", sets=evidence),)
                                                  for i, w in enumerate(worlds)},
                               allowed=reliable_available, **inspect_cost))

    for i, w in enumerate(worlds):
        effects = {possible: (Outcome(1., "repair_applied", sets=fixed | (tested if integrated else 0)),)
                   if possible == w else (Outcome(1., "repair_failed", sets=blocked, clears=fixed | tested),)
                   for possible in worlds}
        actions.append(BeliefAction(f"repair_{i}", f"Apply the repair for possible cause {i + 1}",
                                   requires=ready, forbids=blocked | fixed,
                                   outcomes_by_world=effects, **_cost(rng, 2, 28)))
    success = rng.choice((.45, .65, .8, .95)) if flaky else 1.
    # A failed verification sometimes invalidates the repair, teaching the
    # difference between retrying a check and needing to repair again.
    invalidates = flaky and rng.random() < .4
    outcomes = (Outcome(success, "checks_passed", sets=tested),)
    if flaky:
        outcomes += (Outcome(1. - success, "checks_failed", clears=fixed if invalidates else 0),)
    actions.append(BeliefAction("verify", "Run the acceptance checks", requires=fixed,
                               forbids=blocked | tested, outcomes=outcomes, **_cost(rng, 1, 15)))
    actions.append(BeliefAction("docs", "Update the currently requested documentation", requires=ready,
                               forbids=docs, outcomes=(Outcome(1., "documentation_verified", sets=docs),),
                               **_cost(rng, 1, 20)))

    optional = []
    if reversible:
        optional.append(BeliefAction("recover", "Revert the failed repair and restore the workspace", requires=blocked,
                                    outcomes=(Outcome(1., "restored", clears=blocked | fixed | tested),),
                                    **_cost(rng, 2, 20)))
    if noisy and n > 1:
        accuracy = rng.choice((.6, .75, .9))
        optional.append(BeliefAction("noisy", "Read a noisy diagnostic", requires=ready,
                                    outcomes_by_world={w: (Outcome(accuracy, f"signal_{i}", sets=evidence),
                                                          Outcome(1. - accuracy, f"signal_{(i + 1) % n}", sets=evidence))
                                                       for i, w in enumerate(worlds)},
                                    tokens=max(1., inspect_cost["tokens"] / rng.choice((3, 5, 8))),
                                    latency_ms=rng.randint(1, 12)))
    if family == "noisy_diagnosis":
        optional.sort(key=lambda a: a.id != "noisy")
    if family == "reversible_repair":
        optional.sort(key=lambda a: a.id != "recover")
    actions.extend(optional[:10 - len(actions)])
    # The all-in-one option makes efficient choice nontrivial, but it is not
    # always available and cannot erase the dedicated prerequisite curriculum.
    comprehensive = family == "efficient_choice" or rng.random() < .35
    if comprehensive and family not in ("prepared_diagnosis", "preparation_chain") and len(actions) < 10:
        actions.append(BeliefAction("comprehensive", "Apply and verify a comprehensive repair", requires=ready,
                                   forbids=blocked | tested, outcomes=(Outcome(1., "all_passed", sets=fixed | tested),),
                                   tokens=rng.randint(25, 130), latency_ms=rng.randint(80, 700),
                                   allowed=rng.random() > .15))
    if len(actions) < 10:
        actions.append(BeliefAction("repeat", "Repeat an unchanged status check", **_cost(rng, 0, 5)))

    goal = tested
    if revision:
        goal = rng.choice((docs, docs | tested))
        if rng.random() < .5:
            belief = tuple(replace(h, state=h.state | fixed | tested) for h in belief)
    if family == "completed":
        belief = tuple(replace(h, state=h.state | goal) for h in belief)
    elif family == "unreachable":
        # Missing a declared recovery path is a reason to defer, not evidence
        # that the real-world goal is impossible.
        goal = tested
        belief = tuple(replace(h, state=(h.state | blocked) & ~(fixed | tested)) for h in belief)
        actions = [replace(a, allowed=False) if a.id == "recover" else a for a in actions]
    elif family == "reversible_repair" and rng.random() < .45:
        belief = tuple(replace(h, state=(h.state | blocked) & ~(fixed | tested)) for h in belief)
    elif family == "known_cause" and rng.random() < .4:
        belief = tuple(replace(h, state=h.state | fixed) for h in belief)

    problem = BeliefProblem(f"belief_v4_{split}_{seed}", FACTS, goal, belief, tuple(actions),
                            f"Synthetic {family}; achieve only the currently declared acceptance goal.")
    return _scramble(problem, rng)


def teacher_values(solution, candidates, eligible):
    """Return padded, candidate-aligned supervision, never inference features.

All vectors have length 12 and follow encode_belief's candidate order, then
padding. Values for ineligible/padded/pruned no-effect actions are masked out;
the exact planner does not expand the latter. Terminal candidates are labeled
when eligible. Success regret is optimal success minus candidate success.
"""
    by_id = {r["action_id"]: r for r in solution["action_scores"]}
    values = np.zeros(MAX_CANDIDATES, np.float32)
    regrets = np.zeros(MAX_CANDIDATES, np.float32)
    costs = np.zeros(MAX_CANDIDATES, np.float32)
    mask = np.zeros(MAX_CANDIDATES, bool)
    for i, candidate in enumerate(candidates):
        if not eligible[i]:
            continue
        action_id = candidate["id"]
        if action_id in (STOP, NEEDS_INFORMATION):
            values[i] = float(action_id == STOP)
            mask[i] = True
        else:
            row = by_id.get(action_id)
            if row is None or row.get("pruned") or not row.get("eligible"):
                continue
            values[i] = row["expected_verified_success"]
            costs[i] = row["expected_cost"]
            mask[i] = True
        regrets[i] = max(0., solution["expected_verified_success"] - float(values[i]))
    return {"candidate_ids": [c["id"] for c in candidates] + [None] * (MAX_CANDIDATES - len(candidates)),
            "candidate_expected_success": values.tolist(), "candidate_success_regret": regrets.tolist(),
            "candidate_expected_cost": costs.tolist(), "candidate_value_mask": mask.tolist()}


def _solve(problem, depth, token_weight, latency_weight, seed):
    solution = plan(problem, depth, token_weight, latency_weight, max_nodes=100_000)
    if not solution["search_complete"]:
        raise RuntimeError(f"Incomplete teacher search for {seed}; label discarded.")
    return solution


def generate_rows(count, split, seed):
    """Generate roots and public posterior states with exact lexicographic labels.

Return the existing (rows, (features, valid, eligible, target)) contract. The
target preserves exact success/cost/step ties. Roughly two thirds of examples
remain roots; the rest cover actions after teacher/exploratory observations.
Finished and unreachable anchors each receive one twelfth of root cases, and
teacher prefixes stop before completion so easy terminal labels do not dominate.
"""
    split = _split_name(split)
    if type(count) is not int or count < 1:
        raise ValueError("Count must be a positive integer.")
    family_for_seed(seed, split)
    rows, xx, vv, ee, yy = [], [], [], [], []
    for i in range(count):
        case_seed = seed + i
        rng = random.Random(case_seed ^ 0x7453)
        family = family_for_seed(case_seed, split)
        problem = make_problem(case_seed, split)
        token_weight = rng.choice((.25, 1., 2., 4.))
        latency_weight = rng.choice((.001, .01, .05, .2))
        depth = rng.randint(2, 5) if rng.random() < 1 / 6 else 5
        # Prerequisite chains are deliberately represented with enough budget
        # to show their successful completion, not only short-horizon defer.
        if family in ("prepared_diagnosis", "preparation_chain"):
            depth = 5
        history = []
        solution = _solve(problem, depth, token_weight, latency_weight, case_seed)
        prefix_draw = rng.random()
        prefix_probability = .25 if family in ("prepared_diagnosis", "preparation_chain") else 1 / 3
        if family not in ("completed", "unreachable") and prefix_draw < prefix_probability:
            for _ in range(rng.randint(1, 2)):
                node = solution["plan"]
                if node["kind"] != "action" or depth <= 1:
                    break
                unfinished = [b for b in node["branches"]
                              if not verified(tuple(Hypothesis(**h) for h in b["posterior"]), problem.goal)]
                if not unfinished:
                    break
                observed = rng.choices(unfinished, weights=[b["probability"] for b in unfinished])[0]
                problem = replace(problem, belief=tuple(Hypothesis(**h) for h in observed["posterior"]))
                history.append({"action_id": node["action_id"], "observation": observed["observation"],
                                "source": "teacher_plan_public_observation"})
                depth -= 1
                solution = _solve(problem, depth, token_weight, latency_weight, case_seed)
        elif family not in ("completed", "unreachable") and prefix_draw < prefix_probability + 1 / 12:
            available = [a for a in problem.actions if a.eligible(problem.belief)]
            exploratory = [a for a in available if any(len(a.effects(h.world_id)) > 1 for h in problem.belief)]
            if exploratory and depth > 1:
                action = rng.choice(exploratory)
                outcomes = branches(problem.belief, action)
                label, _, posterior = rng.choices(outcomes, weights=[b[1] for b in outcomes])[0]
                problem = replace(problem, belief=posterior)
                history.append({"action_id": action.id, "observation": label,
                                "source": "exploratory_public_observation"})
                depth -= 1
                solution = _solve(problem, depth, token_weight, latency_weight, case_seed)
        x, valid, eligible, candidates = encode_belief(problem, depth, token_weight, latency_weight)
        winners = target_ids(solution)
        target = np.array([float(c["id"] in winners) for c in candidates] +
                          [0.] * (MAX_CANDIDATES - len(candidates)), np.float32)
        if not target.sum() or not eligible[target > 0].all():
            raise RuntimeError("Teacher target is absent or violates a hard rule.")
        target /= target.sum()
        teacher = {k: solution[k] for k in ("search_complete", "expected_verified_success", "expected_cost",
                                          "expected_steps", "explored_nodes")}
        teacher.update(teacher_values(solution, candidates, eligible))
        rows.append({"problem": problem.to_dict(), "family": family, "split": split, "seed": case_seed,
                     "source": "synthetic_compositional_declared_world_exact_contingent_search_v4",
                     "token_weight": token_weight, "latency_weight": latency_weight, "max_depth": depth,
                     "target_ids": winners, "public_history": history, "teacher": teacher})
        xx.append(x); vv.append(valid); ee.append(eligible); yy.append(target)
        if (i + 1) % 2000 == 0:
            print(f"Generated v4 belief {split}: {i + 1}/{count}", flush=True)
    return rows, (np.stack(xx), np.stack(vv), np.stack(ee), np.stack(yy))
