"""Reproducible v2 curriculum for goal-sensitive, bounded agent decisions.

All labels come from declared deterministic transitions, never a language model.
A pair shares its world, costs and current state: only the requested goal changes.
Train/validation/test have separate seeds and reserved graph compositions. Text
and provenance are never encoded as model inputs. Inspection means an explicit
information prerequisite here, not a learned belief over hidden real outcomes.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import random

import numpy as np

from .features import encode, target_for
from .simulator import Action, Scenario, make_scenario, search

CURRICULUM_VERSION = "winward-v2-2026-09-30"
SPLIT_SHAPES = {
    "train": ("independent_routes", "shared_inspection", "revision_rework"),
    "validation": ("gated_alternative", "parallel_inspection"),
    "test": ("cross_checked_revision", "split_evidence_revision"),
}


def _split(value):
    value = "validation" if value == "valid" else value
    if value not in SPLIT_SHAPES:
        raise ValueError("split must be train, validation, or test")
    return value


def _remap(scenario, rng):
    permutation = list(range(12))
    rng.shuffle(permutation)

    def remap(mask):
        return sum(1 << permutation[i] for i in range(12) if mask & (1 << i))

    facts = [""] * 12
    for i, name in enumerate(scenario.facts):
        facts[permutation[i]] = name
    actions = list(scenario.actions)
    rng.shuffle(actions)
    actions = tuple(replace(a, id=f"option_{i}", requires=remap(a.requires), forbids=remap(a.forbids),
                            sets=remap(a.sets), clears=remap(a.clears)) for i, a in enumerate(actions))
    context = dict(scenario.context)
    for key in ("goal_a", "goal_b", "previous_goal"):
        context[key] = remap(context[key])
    return replace(scenario, facts=tuple(facts), state=remap(scenario.state), goal=remap(scenario.goal),
                   actions=actions, context=context)


def make_world(seed, split="train", *, shape=None, common_goal=False, progress=True):
    """Construct two independently costed paths, with split-reserved dependencies.

    Common-goal worlds provide alternate ways to the same win; competing-goal
    worlds make choosing the correct branch dependent on the requested outcome.
    Progress is performed toward the *old* objective before selecting the new
    goal. Old work may therefore be reusable, irrelevant, or require rework.
    """
    split = _split(split)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = random.Random(f"{CURRICULUM_VERSION}:{split}:{seed}")
    shape = shape or rng.choice(SPLIT_SHAPES[split])
    if shape not in SPLIT_SHAPES[split]:
        raise ValueError("Graph shape is reserved for another split")
    bit = lambda i: 1 << i
    actions = []
    state = 0
    length_a, length_b = rng.choice(((2, 2), (2, 3), (3, 2), (3, 3)))
    shared = shape != "independent_routes"
    prerequisite = bit(8) if shared else 0
    if shared:
        actions.append(Action("inspect", "Inspect current requirements", sets=bit(8), forbids=bit(8),
                              tokens=rng.randint(1, 160), latency_ms=rng.randint(5, 1500)))
    # Reserve structural combinations, not just different prose or seeds.
    if shape in ("parallel_inspection", "split_evidence_revision"):
        actions.append(Action("inspect_dependency", "Inspect independent dependency", sets=bit(9), forbids=bit(9),
                              tokens=rng.randint(1, 180), latency_ms=rng.randint(5, 1800)))
    goal_a, goal_b = bit(length_a - 1), bit(4 + length_b - 1)
    for route, length, offset in (("A", length_a, 0), ("B", length_b, 4)):
        for index in range(length):
            requires = bit(offset + index - 1) if index else prerequisite
            if index == length - 1 and shape in ("parallel_inspection", "split_evidence_revision"):
                requires |= bit(9)
            if index == length - 1 and shape == "cross_checked_revision":
                # Test-only composition: terminal work requires evidence on the
                # other branch, while each branch has a distinct requested win.
                requires |= bit(4 if route == "A" else 0)
            if index == 0 and route == "B" and shape == "gated_alternative":
                requires |= bit(0)
            sets = goal_a if common_goal and route == "B" and index == length - 1 else bit(offset + index)
            clears = 0
            if index == 0 and route == "B" and shape in ("revision_rework", "split_evidence_revision"):
                clears = goal_a
            actions.append(Action(f"{route}_{index}", f"{'Check' if index == length-1 else 'Prepare'} route {route}, stage {index+1}",
                                  requires=requires, forbids=sets, sets=sets, clears=clears & ~sets,
                                  tokens=rng.randint(1, 1000), latency_ms=rng.randint(1, 8000)))
    # At most ten actions; distraction and prohibited shortcut have no teaching
    # effect on permission enforcement because masks provide that rule.
    if len(actions) < 9:
        actions.append(Action("noise", "Polish unrelated output", sets=bit(10), forbids=bit(10), tokens=1, latency_ms=1))
    if len(actions) < 10:
        actions.append(Action("prohibited", "Use unavailable capability", sets=goal_a | goal_b, allowed=False))
    # For common-goal tasks, randomize which complete route is cheapest. Neither
    # first-step cost, action order, nor route length reliably reveals the label.
    if common_goal:
        goal_b = goal_a
    previous_goal = rng.choice((goal_a, goal_b))
    old = Scenario(f"v2-{split}-{seed}", shape, split, tuple(f"condition_{i}" for i in range(12)),
                   state, previous_goal, tuple(actions))
    old_plan = search(old)
    progress_steps = 0
    if progress and old_plan["plan"] and rng.random() < 0.7:
        progress_steps = rng.randint(0, min(3, len(old_plan["plan"])))
        if progress_steps:
            state = old_plan["plan"][progress_steps - 1]["state_after"]
    context = {"curriculum_version": CURRICULUM_VERSION, "shape": shape, "goal_a": goal_a,
               "goal_b": goal_b, "previous_goal": previous_goal, "old_goal_progress_steps": progress_steps,
               "common_goal": common_goal, "provenance": "synthetic deterministic graph; exact bounded-search labels",
               "information_scope": "Information prerequisites are explicit facts; hidden outcomes are not modeled."}
    return _remap(replace(old, state=state, context=context), rng)


def competing_goal_pair(seed, split="train", *, progress=True):
    world = make_world(seed, split, progress=progress)
    return (replace(world, id=world.id + "-goal-a", goal=world.context["goal_a"]),
            replace(world, id=world.id + "-goal-b", goal=world.context["goal_b"]))


def downstream_cost_pair(seed, split="train"):
    """Swap terminal-route costs without moving either goal or current state."""
    world = make_world(seed, split, common_goal=True, progress=False)
    terminals = [a for a in world.actions if a.allowed and a.sets & world.goal]
    if len(terminals) != 2:
        raise AssertionError("A common-goal world must have exactly two terminal actions")
    first, second = terminals
    base = tuple(replace(a, tokens=5, latency_ms=5) for a in world.actions)
    def with_costs(cheap, expensive, suffix):
        actions = tuple(replace(a, tokens=1, latency_ms=1) if a.id == cheap.id else
                        replace(a, tokens=2000, latency_ms=12000) if a.id == expensive.id else a for a in base)
        return replace(world, id=world.id + suffix, actions=actions)
    return with_costs(first, second, "-cost-a"), with_costs(second, first, "-cost-b")


def label_row(scenario, token_weight=1.0, latency_weight=0.01, depth=5, provenance=None):
    label = search(scenario, max_depth=depth, token_weight=token_weight, latency_weight=latency_weight)
    return {"scenario": scenario.to_dict(), "token_weight": float(token_weight),
            "latency_weight": float(latency_weight), "max_depth": int(depth),
            "target_ids": label["optimal_action_ids"], "outcome": label["outcome"],
            "plan": label["plan"], "provenance": provenance or {"source": "curriculum"}}


def generate_rows(count, split, seed, *, progress_log=False):
    """Mix 25% legacy graphs, 50% paired goals, and 25% alternate-route costs.

    Adjacent paired examples expose the same state under two goals. Horizons and
    costs vary independently. Legacy family separation remains intact. Generation
    never calls a learned model, and rejected or failed examples are retained.
    """
    split = _split(split)
    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    rows = []
    for group in range((count + 7) // 8):
        group_seed = seed + group
        worlds = []
        for index in range(2):
            legacy = make_scenario(seed + 1_000_000 + 2 * group + index, split)
            if index == 1:
                plan = search(legacy)
                if plan["plan"]:
                    prefix = int(rng.integers(0, len(plan["plan"]) + 1))
                    if prefix:
                        legacy = replace(legacy, state=plan["plan"][prefix-1]["state_after"])
            worlds.append((legacy, "legacy", None))
        # Half of paired goal tasks start after old-goal progress. The pair still
        # shares exactly the same state, preventing sunk work from fixing a goal.
        for progress in (False, True):
            pair = competing_goal_pair(group_seed + (2_000_000 if progress else 0), split, progress=progress)
            worlds.extend((s, "paired_goal", f"goal-pair-{group_seed}-{progress}") for s in pair)
        worlds.extend((s, "downstream_cost", f"cost-pair-{group_seed}") for s in downstream_cost_pair(group_seed + 4_000_000, split))
        # Pair weights/horizon are shared, so changing the goal or downstream
        # costs is the causal intervention within each pair.
        weights = (float(rng.choice([0.25, 1, 2, 4])), float(rng.choice([0.001, 0.01, 0.05, 0.2])))
        depth = 5 if group % 4 else int(rng.integers(1, 6))
        for scenario, source, pair_id in worlds:
            rows.append(label_row(scenario, *weights, depth, provenance={"source": source, "pair_id": pair_id,
                                  "group_seed": group_seed, "split": split}))
            if len(rows) == count:
                return rows
        if progress_log and len(rows) % 8000 == 0:
            print(f"Generated {split}: {len(rows)}/{count}", flush=True)
    return rows


def encode_rows(rows):
    arrays, valid, eligible, targets = [], [], [], []
    for row in rows:
        x, v, e, candidates = encode(row["scenario"], row["token_weight"], row["latency_weight"], row["max_depth"])
        y = target_for(candidates, {"optimal_action_ids": row["target_ids"]})
        if not np.all(e[y > 0]):
            raise AssertionError("Search teacher chose an ineligible action")
        arrays.append(x); valid.append(v); eligible.append(e); targets.append(y)
    return tuple(np.stack(a) for a in (arrays, valid, eligible, targets))


def provenance_summary(rows):
    return {"version": CURRICULUM_VERSION, "cases": len(rows),
            "sources": dict(Counter(r["provenance"]["source"] for r in rows)),
            "shapes": dict(Counter(r["scenario"]["family"] for r in rows)),
            "outcomes": dict(Counter(r["outcome"] for r in rows)),
            "horizons": dict(Counter(str(r["max_depth"]) for r in rows)),
            "mid_progress_goal_changes": sum(r["scenario"].get("context", {}).get("old_goal_progress_steps", 0) > 0 and
                                             r["scenario"]["goal"] != r["scenario"].get("context", {}).get("previous_goal") for r in rows),
            "labels": "Exact search on declared transitions; no pretrained weights or language-model outputs.",
            "shape_split": SPLIT_SHAPES}
