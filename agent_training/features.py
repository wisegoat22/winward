"""Observation-only features. Never include planner labels, plans, or family IDs."""
import math

import numpy as np

MAX_FACTS = 12
MAX_ACTIONS = 10
MAX_CANDIDATES = MAX_ACTIONS + 2
FEATURE_DIM = 103
STOP = "STOP"
DEFER = "NEEDS_CLARIFICATION"


def bits(mask):
    return [float(bool(mask & (1 << i))) for i in range(MAX_FACTS)]


def as_dict(scenario):
    return scenario.to_dict() if hasattr(scenario, "to_dict") else scenario


def encode(scenario, token_weight=1.0, latency_weight=0.01, max_depth=5, *, feature_dim=FEATURE_DIM):
    scenario = as_dict(scenario)
    state, goal = scenario["state"], scenario["goal"]
    actions = scenario["actions"]
    fact_count = len(scenario["facts"])
    if not 1 <= fact_count <= MAX_FACTS or len(actions) > MAX_ACTIONS:
        raise ValueError("A scenario supports 1–12 facts and at most 10 actions.")
    done = state & goal == goal
    fact_mask = (1 << fact_count) - 1
    goal_count = max(goal.bit_count(), 1)
    costs = [a["tokens"] * token_weight + a["latency_ms"] * latency_weight for a in actions]
    cost_scale = max([1.0] + costs)
    weight_scale = max(token_weight + 100 * latency_weight, 1e-9)
    candidates = actions + [
        {"id": STOP, "name": "Finish — the goal is already satisfied", "requires": 0,
         "forbids": 0, "sets": 0, "clears": 0, "tokens": 0, "latency_ms": 0, "allowed": True},
        {"id": DEFER, "name": "Ask for information or revise the plan", "requires": 0,
         "forbids": 0, "sets": 0, "clears": 0, "tokens": 0, "latency_ms": 0, "allowed": True},
    ]
    if feature_dim not in (102, 103):
        raise ValueError("Unsupported observation feature version.")
    x = np.zeros((MAX_CANDIDATES, feature_dim), dtype=np.float32)
    valid = np.zeros(MAX_CANDIDATES, dtype=np.bool_)
    eligible = np.zeros(MAX_CANDIDATES, dtype=np.bool_)
    for i, a in enumerate(candidates):
        regular, stop, defer = a["id"] not in (STOP, DEFER), a["id"] == STOP, a["id"] == DEFER
        preconditions = (state & a["requires"] == a["requires"] and not state & a["forbids"])
        can_run = (bool(a["allowed"]) and preconditions and not done) if regular else (done if stop else not done)
        after = (state & ~a["clears"]) | a["sets"]
        weighted = a["tokens"] * token_weight + a["latency_ms"] * latency_weight
        scalar = [
            math.log1p(a["tokens"]) / math.log(10001),
            math.log1p(a["latency_ms"]) / math.log(60001),
            weighted / cost_scale,
            (a["requires"] & ~state).bit_count() / MAX_FACTS,
            (a["forbids"] & state).bit_count() / MAX_FACTS,
            (a["sets"] & goal & ~state).bit_count() / goal_count,
            (a["clears"] & goal & state).bit_count() / goal_count,
            (after & goal).bit_count() / goal_count,
            float(can_run), float(regular), float(stop), float(defer),
            max_depth / 5, token_weight / weight_scale, 100 * latency_weight / weight_scale,
            state.bit_count() / MAX_FACTS, goal.bit_count() / MAX_FACTS, fact_count / MAX_FACTS,
        ]
        # Future permission differs from current eligibility: an action can be
        # allowed but not ready yet. Version 2 preserves that distinction.
        values = bits(state) + bits(goal) + bits(a["requires"]) + bits(a["forbids"]) + bits(a["sets"]) + bits(a["clears"]) + bits(fact_mask) + scalar + [float(a["allowed"])]
        x[i] = values[:feature_dim]
        valid[i] = True
        eligible[i] = can_run
    return x, valid, eligible, candidates


def target_for(candidates, solution):
    targets = np.zeros(MAX_CANDIDATES, dtype=np.float32)
    winners = solution.get("optimal_action_ids") or [solution["chosen_action_id"]]
    for i, action in enumerate(candidates):
        if action["id"] in winners:
            targets[i] = 1
    if targets.sum() == 0:
        raise ValueError("Planner answer is absent from candidate list.")
    return targets / targets.sum()
