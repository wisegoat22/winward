"""Deterministic, bounded agent planning world used to create scratch-training labels.

Consequences here are declared state transitions, not language-model predictions.
The solver searches every applicable branch through at most five actions, optimizing
goal achievement before cost. Failure means *not found within the horizon*.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
import math
import random
from typing import Any


MAX_FACTS = 12
MAX_ACTIONS = 10
MAX_DEPTH = 5
STOP = "STOP"
NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
DEFAULT_TOKEN_WEIGHT = 1.0
DEFAULT_LATENCY_WEIGHT = 0.01

# Families, as well as random streams, are separated between splits. The test
# compositions are deliberately absent from training, not just fresh examples.
SPLIT_FAMILIES = {
    "train": ("simple_chain", "fork", "shortcut", "shared_prerequisite"),
    "validation": ("two_stage_fork", "blocked_shortcut"),
    "test": ("nested_dependencies", "changed_goal", "reversible_trap"),
}
FACTS = (
    "requirements_known", "evidence_collected", "workspace_ready",
    "change_ready", "checks_passed", "review_complete", "delivered",
    "outcome_recorded", "backup_available", "permission_granted",
    "irrelevant_work_done", "blocker_present",
)


def _mask(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer bit mask")
    return value


@dataclass(frozen=True)
class Action:
    id: str
    name: str
    requires: int = 0
    forbids: int = 0
    sets: int = 0
    clears: int = 0
    tokens: float = 0.0
    latency_ms: float = 0.0
    allowed: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id or self.id in {STOP, NEEDS_CLARIFICATION}:
            raise ValueError("Action id must be nonempty and not a reserved outcome")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Action name must be nonempty")
        for key in ("requires", "forbids", "sets", "clears"):
            _mask(getattr(self, key), key)
        if self.requires & self.forbids:
            raise ValueError("Action cannot require and forbid the same fact")
        if self.sets & self.clears:
            raise ValueError("Action cannot set and clear the same fact")
        for key in ("tokens", "latency_ms"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if not isinstance(self.allowed, bool):
            raise ValueError("allowed must be a boolean")

    def eligible(self, state: int) -> bool:
        return self.allowed and state & self.requires == self.requires and not state & self.forbids

    def apply(self, state: int) -> int:
        if not self.eligible(state):
            raise ValueError(f"Action {self.id!r} is not permitted or its preconditions are unmet")
        return (state & ~self.clears) | self.sets

    def cost(self, token_weight: float = DEFAULT_TOKEN_WEIGHT, latency_weight: float = DEFAULT_LATENCY_WEIGHT) -> float:
        return self.tokens * token_weight + self.latency_ms * latency_weight

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Action:
        return cls(**value)


@dataclass(frozen=True)
class Scenario:
    id: str
    family: str
    split: str
    facts: tuple[str, ...]
    state: int
    goal: int
    actions: tuple[Action, ...]
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "facts", tuple(self.facts))
        object.__setattr__(self, "actions", tuple(self.actions))
        if not 1 <= len(self.facts) <= MAX_FACTS or len(set(self.facts)) != len(self.facts):
            raise ValueError("Scenario requires 1–12 unique facts")
        if any(not isinstance(name, str) or not name for name in self.facts):
            raise ValueError("Fact names must be nonempty strings")
        if len(self.actions) > MAX_ACTIONS or len({a.id for a in self.actions}) != len(self.actions):
            raise ValueError("Scenario requires at most 10 actions with unique ids")
        maximum = (1 << len(self.facts)) - 1
        for key in ("state", "goal"):
            if _mask(getattr(self, key), key) & ~maximum:
                raise ValueError(f"{key} refers to a fact outside this scenario")
        if any((a.requires | a.forbids | a.sets | a.clears) & ~maximum for a in self.actions):
            raise ValueError("Action refers to a fact outside this scenario")

    def goal_met(self, state: int | None = None) -> bool:
        return (self.state if state is None else state) & self.goal == self.goal

    def fact_names(self, mask: int) -> list[str]:
        return [name for i, name in enumerate(self.facts) if mask & (1 << i)]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["facts"] = list(self.facts)
        value["actions"] = [a.to_dict() for a in self.actions]
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Scenario:
        data = dict(value)
        data["facts"] = tuple(data["facts"])
        data["actions"] = tuple(Action.from_dict(a) for a in data["actions"])
        return cls(**data)


def search(
    scenario: Scenario | dict[str, Any],
    max_depth: int = MAX_DEPTH,
    token_weight: float = DEFAULT_TOKEN_WEIGHT,
    latency_weight: float = DEFAULT_LATENCY_WEIGHT,
) -> dict[str, Any]:
    """Return the cheapest goal-achieving plan of <= ``max_depth`` actions.

    Weights are cost per token and cost per millisecond. A successful plan always
    beats any incomplete one regardless of cost. Equal-cost solutions prefer fewer
    actions; remaining ties use action ids for reproducibility. Costs are estimates
    supplied by the caller, not measured future runtime or calibrated confidence.
    """
    if isinstance(scenario, dict):
        scenario = Scenario.from_dict(scenario)
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or not 0 <= max_depth <= MAX_DEPTH:
        raise ValueError("max_depth must be an integer from 0 through 5")
    for value in (token_weight, latency_weight):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Cost weights must be finite and nonnegative")
    actions = tuple(sorted(scenario.actions, key=lambda a: a.id))
    by_id = {a.id: a for a in actions}
    costs = {a.id: a.cost(token_weight, latency_weight) for a in actions}
    transitions_examined = 0

    @lru_cache(maxsize=None)
    def best(state: int, remaining: int) -> tuple[float, tuple[str, ...]] | None:
        nonlocal transitions_examined
        if scenario.goal_met(state):
            return 0.0, ()
        if remaining == 0:
            return None
        winner = None
        for action in actions:
            if not action.eligible(state):
                continue
            after = action.apply(state)
            transitions_examined += 1
            # A nonnegative-cost no-op can never improve a shortest optimal plan.
            if after == state:
                continue
            tail = best(after, remaining - 1)
            if tail is None:
                continue
            candidate = (costs[action.id] + tail[0], (action.id,) + tail[1])
            if winner is None or (candidate[0], len(candidate[1]), candidate[1]) < (winner[0], len(winner[1]), winner[1]):
                winner = candidate
        return winner

    solution = best(scenario.state, max_depth)
    can_stop = scenario.goal_met()
    scores = []
    for action in scenario.actions:
        eligible = action.eligible(scenario.state)
        tail = None
        if eligible and max_depth > 0:
            tail = best(action.apply(scenario.state), max_depth - 1)
        succeeds = tail is not None
        reason = "goal_reachable_within_horizon" if succeeds else "no_goal_reaching_continuation_within_horizon"
        if not action.allowed:
            reason = "not_permitted"
        elif not eligible:
            reason = "preconditions_unmet"
        elif max_depth == 0:
            reason = "depth_limit"
        scores.append({
            "action_id": action.id, "action_name": action.name,
            "eligible": eligible, "success": succeeds,
            "cost": costs[action.id] + tail[0] if succeeds else None,
            "steps": 1 + len(tail[1]) if succeeds else None,
            "reason": reason,
        })
    if solution is None:
        chosen_id, chosen_name, outcome = NEEDS_CLARIFICATION, "Need more information or a revised plan", "needs_clarification"
        ids: tuple[str, ...] = ()
        optimal_ids = [NEEDS_CLARIFICATION]
    elif can_stop:
        chosen_id, chosen_name, outcome = STOP, "Stop: the goal is already satisfied", "stop"
        ids = ()
        optimal_ids = [STOP]
    else:
        ids = solution[1]
        chosen_id, chosen_name, outcome = ids[0], by_id[ids[0]].name, "plan"
        optimal_ids = [s["action_id"] for s in scores if s["success"] and math.isclose(s["cost"], solution[0], rel_tol=1e-12, abs_tol=1e-9) and s["steps"] == len(ids)]
    plan = []
    current = scenario.state
    for action_id in ids:
        action = by_id[action_id]
        after = action.apply(current)
        plan.append({"action_id": action.id, "action_name": action.name,
                     "state_before": current, "state_after": after,
                     "tokens": action.tokens, "latency_ms": action.latency_ms,
                     "cost": costs[action.id]})
        current = after
    cache = best.cache_info()
    return {
        "chosen_action_id": chosen_id, "chosen_action_name": chosen_name,
        "optimal_action_ids": optimal_ids, "plan": plan,
        "plan_action_ids": list(ids), "success": solution is not None,
        "outcome": outcome, "can_stop": can_stop, "depth": len(ids),
        "max_depth": max_depth, "final_state": current,
        "total_cost": solution[0] if solution is not None else None,
        "total_tokens": sum(p["tokens"] for p in plan),
        "total_latency_ms": sum(p["latency_ms"] for p in plan),
        "cost_weights": {"tokens": token_weight, "latency_ms": latency_weight},
        "explored_states": cache.misses, "transitions_examined": transitions_examined,
        "action_scores": scores,
        "limitation": "Exact only for supplied deterministic mechanics, costs, permissions, and this search horizon.",
    }


def optimal_solution(scenario: Scenario | dict[str, Any], max_depth: int = MAX_DEPTH, **weights: float) -> dict[str, Any]:
    return search(scenario, max_depth=max_depth, **weights)


def make_scenario(seed: int, split: str = "train", family: str | None = None) -> Scenario:
    """Produce a reproducible synthetic task with alternatives, noise, and traps.

    Seed controls only this example. Family sets and random streams are disjoint
    by split. Every scenario independently permutes its bit encoding, candidate
    order, and opaque action ids, so slot position does not reveal the label.
    """
    if split == "valid":
        split = "validation"
    if split not in SPLIT_FAMILIES:
        raise ValueError(f"split must be one of {tuple(SPLIT_FAMILIES)}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = random.Random(f"agent-simulator-v1:{split}:{seed}")
    family = family or rng.choice(SPLIT_FAMILIES[split])
    if family not in SPLIT_FAMILIES[split]:
        raise ValueError(f"Family {family!r} is not assigned to split {split!r}")
    actions: list[Action] = []
    state = 0
    progress: list[int] = []
    bit = lambda index: 1 << index

    def add(name: str, requires: int = 0, sets: int = 0, *, forbids: int = 0,
            clears: int = 0, tokens: float | None = None,
            latency: float | None = None, allowed: bool = True) -> int:
        actions.append(Action(str(len(actions)), name, requires, forbids, sets, clears,
                              rng.randint(15, 350) if tokens is None else tokens,
                              rng.randint(15, 4000) if latency is None else latency, allowed))
        return len(actions) - 1

    if family in {"simple_chain", "shortcut"}:
        length = rng.randint(2, 5)
        for i in range(length):
            progress.append(add(f"Establish {FACTS[i]}", bit(i - 1) if i else 0, bit(i), forbids=bit(i)))
        goal = bit(length - 1)
        if rng.random() < 0.5:
            goal |= bit(rng.randrange(length))
        start = 0 if family == "simple_chain" else 1
        affected = sum(bit(i) for i in range(start, length))
        add("Complete remaining work in one larger operation", bit(0) if start else 0, affected,
            tokens=rng.randint(100, 1400), latency=rng.randint(100, 14000))
    elif family in {"fork", "shared_prerequisite"}:
        progress.append(add("Inspect the request and gather evidence", sets=bit(0), forbids=bit(0)))
        for i in (1, 2, 3):
            progress.append(add(f"Prepare independent requirement {i}", bit(0), bit(i), forbids=bit(i)))
        if family == "fork":
            progress.append(add("Combine the prepared requirements", bit(1) | bit(2), bit(4), forbids=bit(4)))
            goal = bit(4) | (bit(3) if rng.random() < 0.35 else 0)
        else:
            goal = bit(1) | bit(2) | (bit(3) if rng.random() < 0.6 else 0)
        add("Prepare two requirements together", bit(0), bit(1) | bit(2),
            tokens=rng.randint(20, 900), latency=rng.randint(20, 9000))
    elif family == "two_stage_fork":
        progress.append(add("Gather missing requirements", sets=bit(0), forbids=bit(0)))
        progress.append(add("Prepare first branch", bit(0), bit(1), forbids=bit(1)))
        progress.append(add("Validate first branch", bit(1), bit(2), forbids=bit(2)))
        progress.append(add("Prepare second branch", bit(0), bit(3), forbids=bit(3)))
        progress.append(add("Combine both validated branches", bit(2) | bit(3), bit(4), forbids=bit(4)))
        add("Prepare and validate first branch together", bit(0), bit(1) | bit(2), tokens=rng.randint(30, 1000))
        goal = bit(4) | (bit(1) if rng.random() < 0.5 else 0)
    elif family == "blocked_shortcut":
        state = bit(11)
        progress.append(add("Resolve current blocker", clears=bit(11), tokens=rng.randint(20, 500)))
        progress.append(add("Gather evidence", sets=bit(0), forbids=bit(0)))
        progress.append(add("Prepare change after blocker is resolved", bit(0), bit(1), forbids=bit(11)))
        progress.append(add("Validate the change", bit(1), bit(2), forbids=bit(2)))
        progress.append(add("Deliver the result", bit(2), bit(3), forbids=bit(3)))
        add("Resolve blocker and prepare change together", bit(0), bit(1), clears=bit(11), tokens=rng.randint(50, 1100))
        goal = bit(2) | bit(3)
    elif family == "nested_dependencies":
        progress.append(add("Collect evidence", sets=bit(0), forbids=bit(0)))
        progress.append(add("Prepare work from evidence", bit(0), bit(1), forbids=bit(1)))
        progress.append(add("Inspect independent dependency", sets=bit(2), forbids=bit(2)))
        progress.append(add("Verify work and dependency together", bit(1) | bit(2), bit(3), forbids=bit(3)))
        progress.append(add("Deliver verified work", bit(3), bit(4), forbids=bit(4)))
        add("Prepare work and inspect dependency in one operation", bit(0), bit(1) | bit(2), tokens=rng.randint(30, 1100))
        goal = bit(4) | (bit(2) if rng.random() < 0.5 else 0)
    elif family == "changed_goal":
        state = bit(2)  # An old deliverable exists, but is no longer the whole goal.
        progress.append(add("Read the revised request", sets=bit(0), forbids=bit(0)))
        progress.append(add("Collect newly relevant evidence", bit(0), bit(1), forbids=bit(1)))
        progress.append(add("Produce updated deliverable", bit(1) | bit(2), bit(3), forbids=bit(3)))
        progress.append(add("Validate updated deliverable", bit(3), bit(4), forbids=bit(4)))
        add("Inspect revision and evidence together", sets=bit(0) | bit(1), tokens=rng.randint(20, 1100))
        add("Improve the superseded deliverable", bit(2), bit(5), tokens=2, latency=1)
        goal = bit(3) | bit(4)
    else:  # reversible_trap: taking the cheaper immediate step can add rework.
        progress.append(add("Inspect current work", sets=bit(0), forbids=bit(0)))
        progress.append(add("Prepare the required evidence", bit(0), bit(1), forbids=bit(1)))
        progress.append(add("Save a backup of the evidence", bit(1), bit(8), forbids=bit(8)))
        add("Apply quick change that invalidates evidence", bit(1), bit(3), clears=bit(1), forbids=bit(8), tokens=3, latency=2)
        progress.append(add("Apply change with evidence retained", bit(1) | bit(8), bit(3), forbids=bit(3)))
        progress.append(add("Check the result against evidence", bit(1) | bit(3), bit(4), forbids=bit(4)))
        goal = bit(3) | bit(4)

    # Progress snapshots teach acting on the current situation, including stop.
    prefix = rng.choices([0, 1, 2, 3, len(progress)], weights=[42, 20, 15, 12, 11])[0]
    for index in progress[:prefix]:
        if actions[index].eligible(state):
            state = actions[index].apply(state)

    # Some requests ask for an intermediate or additional outcome instead.
    if rng.random() < 0.16 and family != "changed_goal":
        goal = bit(rng.choice([0, 1, 2]))
    if rng.random() < 0.08:
        state |= goal  # A completed goal makes all additional work unnecessary.

    add("Polish unrelated formatting", sets=bit(10), tokens=1, latency=1)
    add("Restart work from the beginning", clears=bit(0) | bit(1) | bit(2), tokens=2, latency=1)
    # An attractive but unauthorized route must never be selected.
    if len(actions) < MAX_ACTIONS:
        add("Bypass required authorization", sets=goal, tokens=0, latency=0, allowed=False)

    # Withhold permissions occasionally. Labels then require an alternate route
    # or explicitly report that no plan was found in the bounded world.
    if rng.random() < 0.18:
        selected = rng.choice(progress)
        actions[selected] = replace(actions[selected], allowed=False)
    if rng.random() < 0.06:
        goal |= bit(9)  # Missing capability; no supplied action grants permission.

    permutation = list(range(MAX_FACTS))
    rng.shuffle(permutation)

    def remap(mask: int) -> int:
        return sum(1 << permutation[i] for i in range(MAX_FACTS) if mask & (1 << i))

    facts = [""] * MAX_FACTS
    for i, name in enumerate(FACTS):
        facts[permutation[i]] = name
    rng.shuffle(actions)
    actions = [replace(a, id=f"a{i}", requires=remap(a.requires), forbids=remap(a.forbids),
                       sets=remap(a.sets), clears=remap(a.clears)) for i, a in enumerate(actions)]
    return Scenario(
        id=f"{split}-{seed}", family=family, split=split, facts=tuple(facts),
        state=remap(state), goal=remap(goal), actions=tuple(actions),
        context={"domain": "synthetic deterministic agent workflow",
                 "goal_policy": "Satisfy the requested goal and permissions, then minimize estimated token and time cost.",
                 "dynamics": "Only the declared effects occur; outside-world uncertainty is not modeled."},
    )
