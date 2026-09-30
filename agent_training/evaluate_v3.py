"""Predeclared, one-time audit of Winward v3; never trains or selects weights.

The protocol is sealed before any audit instances are generated. Frozen weights
are checked before and after the audit, and every comparison uses matched tasks.
This module's small baseline and rollout helpers need no MLX installation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from agent_lab.belief import (BeliefProblem, NEEDS_INFORMATION, STOP, branches,
                              plan, update_belief, verified)


# Change this only for a NEW experiment, before its fresh data is generated.
PROTOCOL = {
    "version": "winward-v3-gate-2",
    "seeds": {"belief": 330_000_000, "legacy": 340_000_000,
              "curriculum": 341_000_000, "goal_changes": 342_000_000, "tools": 350_000_000},
    "counts": {"belief_problems": 120, "episodes_per_problem": 4,
               "graph_rows_per_suite": 2400, "graph_rollouts_per_suite": 300,
               "actual_tool_fixtures": 40, "goal_change_pairs": 200},
    "horizon": 5,
    "weights": {"tokens": 1.0, "latency_ms": 0.01},
    "tool_kinds": ["interval", "collection"],
    "gates": {
        "validation_retention": "Frozen candidate must already pass the trainer's predeclared strict validation-retention checks",
        "legacy_retention": "Matched fresh legacy goal completion >= frozen v1",
        "curriculum_retention": "Matched fresh v2-composition goal completion >= frozen v1",
        "legacy_action_cost_retention": "Mean declared weighted action cost <= frozen v1 on exactly the same legacy tasks both policies complete",
        "changed_goal_action_retention": "Fresh paired changed-goal next-action accuracy >= frozen v2.1",
        "changed_goal_pair_retention": "Fresh changed-goal pairs with both members correct >= frozen v2.1",
        "actual_tool_retention": "Matched real fixture verified-and-stopped completion >= frozen v2.1",
        "uncertainty_expected": "Exact expected verified completion >= each of two cheap baselines",
        "uncertainty_sampled": "Matched sampled verified completion >= each of two cheap baselines",
        "useful_efficiency": "Strictly lower mean weighted declared action cost plus measured decision overhead on the same episodes both methods complete, against the strongest cheap baseline",
    },
    "strong_baseline_selection": "Highest exact expected verified completion, then sampled completion, then lowest sampled total cost; ties broken by name",
    "promotion": "All gates must pass. No allowed regression tolerance. Failure retains existing defaults.",
    "timing": "Tokens and action latency are declared estimates for belief tasks. Decision latency is measured. Weighted cost = tokens + .01*(declared action ms + measured decision ms). No real tool time is claimed for belief simulation.",
    "provenance": "Fresh instances after freeze; generator families and fixture templates are developer-known. This is not an unseen-repository or natural-language benchmark.",
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def protocol_sha256():
    return hashlib.sha256(_canonical(PROTOCOL)).hexdigest()


def seal_protocol(run):
    """Exclusive creation is allowed before training; creates no test data."""
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    path = run / "audit-protocol.json"
    value = {"protocol": PROTOCOL, "protocol_sha256": protocol_sha256()}
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    return value


def _mean(values):
    return statistics.mean(values) if values else None


def _entropy(belief):
    return -math.fsum(h.probability * math.log2(h.probability) for h in belief if h.probability)


def cheap_action(problem, remaining=5, *, information_first=False,
                 token_weight=1.0, latency_weight=0.01):
    """One-step rules using supplied outcomes; no recursive planner or hidden state.

    Both skip actions with no effect or information and avoid re-inspecting known
    information. The information rule prefers a guaranteed immediate finish,
    otherwise the greatest information gain, then immediate expected goal gain.
    Both recognize one-step prerequisites of direct goal-producing actions so a
    known cause does not invite a redundant inspection just for its evidence bit.
    """
    if verified(problem.belief, problem.goal):
        return STOP
    if remaining <= 0:
        return NEEDS_INFORMATION
    before = math.fsum(h.probability * (h.state & problem.goal).bit_count() for h in problem.belief)
    useful = problem.goal
    for future in problem.actions:
        if future.allowed and any(o.sets & problem.goal for h in problem.belief for o in future.effects(h.world_id)):
            useful |= future.requires
    before_useful = math.fsum(h.probability * (h.state & useful).bit_count() for h in problem.belief)
    options = []
    for action in problem.actions:
        if not action.eligible(problem.belief):
            continue
        children = branches(problem.belief, action)
        if all(posterior == problem.belief for _, _, posterior in children):
            continue
        immediate = math.fsum(p * verified(b, problem.goal) for _, p, b in children)
        gain = math.fsum(p * math.fsum(h.probability * (h.state & problem.goal).bit_count() for h in b)
                         for _, p, b in children) - before
        prerequisite_gain = math.fsum(p * math.fsum(h.probability * (h.state & useful).bit_count() for h in b)
                                     for _, p, b in children) - before_useful
        information = max(0.0, _entropy(problem.belief) - math.fsum(p * _entropy(b) for _, p, b in children))
        cost = action.tokens * token_weight + action.latency_ms * latency_weight
        if information_first:
            score = (-int(immediate >= 1.0), -information, cost if information > 0 else 0,
                     -immediate, -gain, -prerequisite_gain, cost, action.id)
        else:
            score = (-immediate, -gain, -prerequisite_gain, cost, -information, action.id)
        options.append((score, action.id))
    return min(options)[1] if options else NEEDS_INFORMATION


def _choice(policy, problem, remaining):
    result = policy(problem, remaining)
    return result["action_id"] if isinstance(result, dict) else result


def expected_episode(problem, policy, horizon=5, token_weight=1.0, latency_weight=0.01):
    """Integrate the policy's entire observation tree, without correcting choices."""
    if not 1 <= horizon <= 5:
        raise ValueError("Horizon must be 1 to 5")
    policy_decisions = 0

    @lru_cache(None)
    def visit(belief, remaining):
        nonlocal policy_decisions
        if verified(belief, problem.goal):
            return (1.0, 0.0, 0.0, 0.0, 0.0)
        if remaining == 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        current = replace(problem, belief=belief)
        policy_decisions += 1
        started = time.perf_counter()
        action_id = _choice(policy, current, remaining)
        decision_ms = (time.perf_counter() - started) * 1000
        action = next((a for a in problem.actions if a.id == action_id), None)
        if action is None or not action.eligible(belief):
            return (0.0, 0.0, 0.0, decision_ms, 0.0)
        children = [(p, visit(posterior, remaining - 1)) for _, p, posterior in branches(belief, action)]
        expectation = lambda i: math.fsum(p * value[i] for p, value in children)
        return (min(1.0, expectation(0)), action.tokens + expectation(1),
                action.latency_ms + expectation(2), decision_ms + expectation(3),
                1.0 + expectation(4))

    success, tokens, latency_ms, decision_ms, steps = visit(problem.belief, horizon)
    return {"expected_verified_success": success, "expected_declared_tokens": tokens,
            "expected_declared_action_ms": latency_ms, "expected_measured_decision_ms": decision_ms,
            "expected_steps": steps, "expected_total_weighted_cost":
                token_weight * tokens + latency_weight * (latency_ms + decision_ms),
            "unique_policy_decisions": policy_decisions,
            "unique_belief_nodes": visit.cache_info().misses}


def _uniform(seed, *keys):
    # Stable, independently keyed action/step streams preserve common randomness
    # without letting a method's extra random draws alter another method's world.
    digest = hashlib.sha256(_canonical([seed, *keys])).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _draw(items, probabilities, value):
    total = math.fsum(probabilities)
    cumulative = 0.0
    for item, probability in zip(items, probabilities):
        cumulative += probability / total
        if value < cumulative:
            return item
    return items[-1]


def sample_episode(problem, policy_callable, seed, horizon=5,
                   token_weight=1.0, latency_weight=0.01):
    """Sample actual hidden outcomes; policies see only the posterior belief.

    policy_callable(problem, remaining)->action ID or {action_id: ...}. A single
    local demonstration may call this too. Costs are declared simulation costs.
    Success requires both a verified belief and the actual world's goal.
    """
    if not 1 <= horizon <= 5:
        raise ValueError("Horizon must be 1 to 5")
    hidden = _draw(problem.belief, [h.probability for h in problem.belief], _uniform(seed, "initial-world"))
    actual_state, world_id = hidden.state, hidden.world_id
    current = problem
    tokens = latency_ms = decision_ms = 0.0
    trace = []
    reason = "horizon_exhausted"
    for step in range(horizon):
        if verified(current.belief, current.goal):
            reason = "verified"
            break
        started = time.perf_counter()
        action_id = _choice(policy_callable, current, horizon - step)
        elapsed = (time.perf_counter() - started) * 1000
        decision_ms += elapsed
        action = next((a for a in current.actions if a.id == action_id), None)
        if action is None or not action.eligible(current.belief):
            reason = "deferred" if action_id == NEEDS_INFORMATION else "premature_stop" if action_id == STOP else "invalid_action"
            trace.append({"action_id": action_id, "observation": None, "decision_ms": elapsed,
                          "verified": False, "reason": reason})
            break
        outcomes = action.effects(world_id)
        outcome = _draw(outcomes, [o.probability for o in outcomes], _uniform(seed, "outcome", step, action.id))
        actual_state = (actual_state & ~outcome.clears) | outcome.sets
        before = current.belief
        # Only an observation enters the posterior. The sampled world/state is
        # never passed to the policy or used to select its next action.
        current = update_belief(current, action.id, outcome.observation)
        if not any(h.world_id == world_id and h.state == actual_state for h in current.belief):
            raise AssertionError("The actual outcome vanished from the observation posterior")
        tokens += action.tokens
        latency_ms += action.latency_ms
        trace.append({"action_id": action.id, "action_name": action.name,
                      "observation": outcome.observation, "decision_ms": elapsed,
                      "declared_tokens": action.tokens, "declared_action_ms": action.latency_ms,
                      "belief_before": [asdict(h) for h in before],
                      "belief_after": [asdict(h) for h in current.belief],
                      "verified": verified(current.belief, current.goal)})
    belief_verified = verified(current.belief, current.goal)
    actual_goal = actual_state & current.goal == current.goal
    success = belief_verified and actual_goal
    return {"problem_id": problem.id, "seed": seed, "success": success,
            "belief_verified": belief_verified, "actual_goal_met": actual_goal,
            "outcome": "verified" if success else reason, "steps": sum(e.get("observation") is not None for e in trace),
            "decision_ms": decision_ms, "declared_tokens": tokens,
            "declared_action_ms": latency_ms,
            "total_weighted_cost": token_weight * tokens + latency_weight * (latency_ms + decision_ms),
            "estimated_end_to_end_ms": latency_ms + decision_ms, "trace": trace,
            "cost_note": "Action tokens and action time are declared estimates; only decision time is measured. No real tools executed."}


def summarize_samples(episodes):
    winners = [e for e in episodes if e["success"]]
    return {"cases": len(episodes), "successes": len(winners),
            "success_rate": len(winners) / len(episodes) if episodes else 0.0,
            "mean_actions": _mean([e["steps"] for e in episodes]),
            "mean_measured_decision_ms": _mean([e["decision_ms"] for e in episodes]),
            "mean_declared_tokens": _mean([e["declared_tokens"] for e in episodes]),
            "mean_declared_action_ms": _mean([e["declared_action_ms"] for e in episodes]),
            "mean_total_weighted_cost": _mean([e["total_weighted_cost"] for e in episodes]),
            "mean_total_weighted_cost_when_successful": _mean([e["total_weighted_cost"] for e in winners]),
            "invalid_actions": sum(e["outcome"] == "invalid_action" for e in episodes),
            "premature_stops": sum(e["outcome"] == "premature_stop" for e in episodes)}


def matched_success_efficiency(candidate, baseline):
    left = {(e["problem_id"], e["seed"]): e for e in candidate}
    right = {(e["problem_id"], e["seed"]): e for e in baseline}
    if len(left) != len(candidate) or len(right) != len(baseline) or left.keys() != right.keys():
        raise ValueError("Efficiency requires unique, identical problem/episode keys")
    pairs = [(left[k], right[k]) for k in left if left[k]["success"] and right[k]["success"]]
    a, b = ([x["total_weighted_cost"] for x, _ in pairs], [y["total_weighted_cost"] for _, y in pairs])
    return {"paired_successes": len(pairs), "candidate_mean_cost": _mean(a), "baseline_mean_cost": _mean(b),
            "mean_cost_delta": _mean([x-y for x, y in zip(a, b)]),
            "strictly_better": bool(pairs) and statistics.mean(a) < statistics.mean(b),
            "selection": "Exactly the same problem and stochastic episode, successful under both methods. Failed episodes cannot make costs appear lower."}


def graph_cost_retention(candidate, baseline):
    left = {e["scenario_id"]: e for e in candidate["records"]}
    right = {e["scenario_id"]: e for e in baseline["records"]}
    if left.keys() != right.keys() or len(left) != len(candidate["records"]) or len(right) != len(baseline["records"]):
        raise ValueError("Legacy cost retention requires identical unique scenario IDs")
    pairs = [(left[k], right[k]) for k in left if left[k]["success"] and right[k]["success"]]
    return {"paired_successes": len(pairs),
            "candidate_mean_declared_cost": _mean([a["declared_weighted_action_cost"] for a, _ in pairs]),
            "baseline_mean_declared_cost": _mean([b["declared_weighted_action_cost"] for _, b in pairs]),
            "selection": "Same legacy scenario IDs completed by both policies. Declared weighted action costs only; neural decision time reported separately."}


def promotion_gate(graphs, tools, belief, frozen_report):
    """Pure, testable gate; strict comparisons were specified before outcomes."""
    checks = []

    def add(name, candidate, baseline, strict=False, lower=False, evidence=None):
        passed = (candidate is not None and baseline is not None and
                  (candidate < baseline if strict else candidate <= baseline if lower else candidate >= baseline))
        checks.append({"name": name, "passed": bool(passed), "candidate": candidate,
                       "baseline": baseline, "delta": candidate-baseline if candidate is not None and baseline is not None else None,
                       "requirement": "strictly lower" if strict else "no greater than baseline" if lower else "at least baseline", "evidence": evidence})

    add("validation_retention", int(frozen_report.get("validation_retention_passed") is True), 1,
        evidence=frozen_report.get("best_validation", {}).get("retention_gates"))
    for suite in ("legacy", "curriculum"):
        methods = graphs[suite]["methods"]
        add(f"{suite}_retention", methods["v3"]["success_rate"], methods["v1"]["success_rate"])
    cost = graphs["legacy"]["paired_success_action_cost"]
    add("legacy_action_cost_retention", cost["candidate_mean_declared_cost"], cost["baseline_mean_declared_cost"],
        lower=True, evidence={"paired_successes": cost["paired_successes"]})
    changed = graphs["goal_changes"]["methods"]
    for field, gate_name in (("neural_accuracy", "changed_goal_action_retention"),
                             ("neural_both_members_correct", "changed_goal_pair_retention")):
        add(gate_name, changed["v3"][field], changed["v2.1"][field],
            evidence={"pairs": changed["v3"]["pairs"], "cases": changed["v3"]["cases"]})
    add("actual_tool_retention", tools["methods"]["v3"]["verified_and_stopped_rate"],
        tools["methods"]["v2.1"]["verified_and_stopped_rate"])
    for method in ("myopic_progress", "information_first"):
        add(f"uncertainty_expected_vs_{method}", belief["expected"]["v3"]["verified_success_rate"],
            belief["expected"][method]["verified_success_rate"])
        add(f"uncertainty_sampled_vs_{method}", belief["sampled"]["v3"]["success_rate"],
            belief["sampled"][method]["success_rate"])
    efficiency = belief["paired_efficiency"]
    add("useful_efficiency", efficiency["candidate_mean_cost"], efficiency["baseline_mean_cost"], strict=True,
        evidence={"baseline": efficiency["baseline"], "paired_successes": efficiency["paired_successes"]})
    passed = all(check["passed"] for check in checks)
    return {"protocol_version": PROTOCOL["version"], "protocol_sha256": protocol_sha256(),
            "promoted": passed, "status": "eligible_for_promotion" if passed else "experimental",
            "checks": checks, "failed_gates": [c["name"] for c in checks if not c["passed"]],
            "default_policy_action": "Candidate meets all predeclared gates" if passed else "Retain prior default policies; do not silently route old tasks to v3"}


def graph_rollouts(model, rows, count=300, batch_size=128):
    from .evaluate import predict
    from .simulator import Scenario, search
    selected = []
    for row in rows:
        scenario = Scenario.from_dict(row["scenario"])
        if not scenario.goal_met() and search(scenario, max_depth=5, token_weight=row["token_weight"], latency_weight=row["latency_weight"])["success"]:
            selected.append((scenario, row))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("Insufficient initially unfinished reachable graph cases")
    current = [s for s, _ in selected]
    costs, tokens, latency, decisions, steps = ([0.0] * count for _ in range(5))
    ended = [False] * count
    for step in range(5):
        active = [i for i in range(count) if not ended[i] and not current[i].goal_met()]
        if not active:
            break
        observations = [(current[i], selected[i][1]["token_weight"], selected[i][1]["latency_weight"], 5-step) for i in active]
        started = time.perf_counter()
        choices, _, _ = predict(model, observations, batch_size)
        per_case_ms = (time.perf_counter()-started) * 1000 / len(active)
        for i, choice in zip(active, choices):
            decisions[i] += per_case_ms
            action = next((a for a in current[i].actions if a.id == choice), None)
            if action is None or not action.eligible(current[i].state):
                ended[i] = True
                continue
            current[i] = replace(current[i], state=action.apply(current[i].state))
            costs[i] += action.cost(selected[i][1]["token_weight"], selected[i][1]["latency_weight"])
            tokens[i] += action.tokens
            latency[i] += action.latency_ms
            steps[i] += 1
    records = [{"scenario_id": s.id, "success": s.goal_met(), "steps": steps[i],
                "declared_weighted_action_cost": costs[i], "declared_tokens": tokens[i],
                "declared_action_ms": latency[i], "batch_amortized_decision_ms": decisions[i]}
               for i, s in enumerate(current)]
    return {"cases": count, "successes": sum(s.goal_met() for s in current),
            "success_rate": sum(s.goal_met() for s in current)/count,
            "mean_actions": statistics.mean(steps), "records": records,
            "timing_note": "Measured batch prediction time divided among active tasks, not single-request latency.",
            "selection": "First initially unfinished cases with an exact five-step solution, selected independently of policy outcomes."}


def audit_beliefs(policy, problems, episodes_per_problem=4):
    methods = {"v3": lambda p, d: policy.predict_belief(p, remaining_horizon=d),
               "myopic_progress": lambda p, d: cheap_action(p, d),
               "information_first": lambda p, d: cheap_action(p, d, information_first=True)}
    expected = {name: [] for name in methods}
    episodes = {name: [] for name in methods}
    ceilings = []
    names = list(methods)
    for index, problem in enumerate(problems):
        reference_started = time.perf_counter()
        reference = plan(problem)
        ceilings.append({"problem_id": problem.id, "expected_verified_success": reference["expected_verified_success"],
                         "expected_declared_cost": reference["expected_cost"], "search_complete": reference["search_complete"],
                         "measured_planning_ms": (time.perf_counter()-reference_started)*1000})
        # Rotate to reduce one-sided cache or thermal timing effects.
        for name in names[index % len(names):] + names[:index % len(names)]:
            expected[name].append(expected_episode(problem, methods[name]))
            for repetition in range(episodes_per_problem):
                seed = PROTOCOL["seeds"]["belief"] + index * episodes_per_problem + repetition
                episodes[name].append(sample_episode(problem, methods[name], seed))
        if (index+1) % 20 == 0:
            print(f"Belief audit: {index+1}/{len(problems)} matched problems", flush=True)
    expected_summary = {name: {"problems": len(items), "verified_success_rate": statistics.mean(e["expected_verified_success"] for e in items),
                              "mean_declared_tokens": statistics.mean(e["expected_declared_tokens"] for e in items),
                              "mean_declared_action_ms": statistics.mean(e["expected_declared_action_ms"] for e in items),
                              "mean_measured_decision_ms": statistics.mean(e["expected_measured_decision_ms"] for e in items),
                              "mean_total_weighted_cost": statistics.mean(e["expected_total_weighted_cost"] for e in items)}
                        for name, items in expected.items()}
    sampled_summary = {name: summarize_samples(items) for name, items in episodes.items()}
    strongest = min(("myopic_progress", "information_first"), key=lambda name: (
        -expected_summary[name]["verified_success_rate"], -sampled_summary[name]["success_rate"],
        sampled_summary[name]["mean_total_weighted_cost"], name))
    efficiency = {"baseline": strongest, **matched_success_efficiency(episodes["v3"], episodes[strongest])}
    return {"expected": expected_summary, "sampled": sampled_summary, "paired_efficiency": efficiency,
            "initially_verified_problems": sum(verified(p.belief, p.goal) for p in problems),
            "initially_unverified_problems": sum(not verified(p.belief, p.goal) for p in problems),
            "exact_planner_reference": {"expected_verified_success": statistics.mean(c["expected_verified_success"] for c in ceilings),
                "all_searches_complete": all(c["search_complete"] for c in ceilings),
                "mean_measured_planning_ms": statistics.mean(c["measured_planning_ms"] for c in ceilings)},
            "per_problem_expected": expected, "cost_note": PROTOCOL["timing"]}, episodes


def audit_tools(policy, baseline):
    from agent_lab.benchmark import summarize
    from agent_lab.runner import run_episode
    specs = [{"seed": PROTOCOL["seeds"]["tools"] + index*100 + repeat,
              "kind": kind, "changed_goal": changed, "uncertain": uncertain}
             for index, (kind, changed, uncertain) in enumerate((k, c, u) for k in PROTOCOL["tool_kinds"] for c in (False, True) for u in (False, True))
             for repeat in range(5)]
    episodes = {name: [] for name in ("v3", "v2.1", "evidence_first")}
    names = list(episodes)
    for index, spec in enumerate(specs):
        for name in names[index % 3:] + names[:index % 3]:
            episodes[name].append(run_episode(**spec, policy_name="evidence_first" if name == "evidence_first" else "neural",
                                             neural_policy=policy if name == "v3" else baseline))
        if (index+1) % 10 == 0:
            print(f"Actual fixture audit: {index+1}/{len(specs)} matched tasks", flush=True)
    measures = {name: {**summarize(items), "verified_and_stopped_rate": sum(e["success"] and e["done"] for e in items)/len(items)}
                for name, items in episodes.items()}
    return {"methods": measures, "specification": specs,
            "scope": "Real files and subprocess checks on controlled generated interval/collection fixtures, using supplied patches. No code generation or arbitrary repository work.",
            "cost_note": "Wall time, tool execution and policy decision times measured; token costs remain declared estimates."}, episodes


def _write_rows(path, rows):
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/goalpolicy-v3")
    parser.add_argument("--v1-run", default="runs/goalpolicy-v1")
    parser.add_argument("--tools-run", default="runs/goalpolicy-v2-tools")
    parser.add_argument("--declare-protocol", action="store_true")
    args = parser.parse_args()
    run = Path(args.run)
    if args.declare_protocol:
        print(json.dumps(seal_protocol(run), indent=2))
        return
    sealed = json.loads((run / "audit-protocol.json").read_text())
    if sealed != {"protocol": PROTOCOL, "protocol_sha256": protocol_sha256()}:
        raise SystemExit("The predeclared protocol differs; do not alter gates after outcomes")
    if any((run / name).exists() for name in ("evaluation.json", "audit-started.json", "belief-test.jsonl")):
        raise SystemExit("This audit has already started. Preserve results; do not rerun to select outcomes.")
    from .model_v3 import GraphView, V3Policy
    from .evaluate_v2 import fresh_goal_pairs, load_model, paired_goal_metrics
    from .evaluate import fresh_rows
    from .curriculum import generate_rows
    from .belief_data import make_problem
    from agent_lab.predictor import NeuralPolicy
    started = time.perf_counter()
    policy = V3Policy(run)
    if hasattr(policy, "load"):
        policy.load()
    report = json.loads((run / "report.json").read_text())
    if not report.get("checkpoint_frozen_before_test_generation") or report.get("test") is not None:
        raise SystemExit("Expected frozen weights without test evaluation")
    protocol_file_sha = hashlib.sha256((run / "audit-protocol.json").read_bytes()).hexdigest()
    if report.get("sealed_audit_protocol_sha256") != protocol_file_sha:
        raise SystemExit("The sealed protocol does not match the protocol recorded during training")
    digest = hashlib.sha256((run / "model.safetensors").read_bytes()).hexdigest()
    if digest != report["checkpoint_sha256"]:
        raise SystemExit("Checkpoint hash mismatch")
    v1, v1_report, v1_digest = load_model(Path(args.v1_run))
    tool_baseline = NeuralPolicy(args.tools_run)
    tool_baseline.load()
    initial = {"checkpoint_sha256": digest, "protocol_sha256": protocol_sha256(),
               "sealed_audit_protocol_file_sha256": protocol_file_sha,
               "checkpoint_frozen_before_test_generation": True,
               "baseline_sha256": {"v1": v1_digest, "v2.1": tool_baseline.report["checkpoint_sha256"]}}
    with (run / "audit-started.json").open("x") as stream:
        json.dump(initial, stream, indent=2)
    # Every fresh generator call is below the checkpoint/protocol verification.
    problems = [make_problem(PROTOCOL["seeds"]["belief"] + i, "test") for i in range(PROTOCOL["counts"]["belief_problems"])]
    _write_rows(run / "belief-test.jsonl", [p.to_dict() for p in problems])
    graph_rows = {"legacy": fresh_rows(PROTOCOL["seeds"]["legacy"], PROTOCOL["counts"]["graph_rows_per_suite"], "test"),
                  "curriculum": generate_rows(PROTOCOL["counts"]["graph_rows_per_suite"], "test", PROTOCOL["seeds"]["curriculum"])}
    graphs = {}
    for name, rows in graph_rows.items():
        _write_rows(run / f"{name}-test.jsonl", rows)
        graphs[name] = {"methods": {"v3": graph_rollouts(GraphView(policy.model), rows, PROTOCOL["counts"]["graph_rollouts_per_suite"]),
                                    "v1": graph_rollouts(v1, rows, PROTOCOL["counts"]["graph_rollouts_per_suite"])}}
        print(f"{name} graph audit complete", flush=True)
    graphs["legacy"]["paired_success_action_cost"] = graph_cost_retention(
        graphs["legacy"]["methods"]["v3"], graphs["legacy"]["methods"]["v1"])
    pairs, scanned = fresh_goal_pairs(PROTOCOL["seeds"]["goal_changes"], PROTOCOL["counts"]["goal_change_pairs"])
    _write_rows(run / "goal-change-test.jsonl", [{"first": a.to_dict(), "second": b.to_dict(),
                "optimal_ids": [labels[0]["optimal_action_ids"], labels[1]["optimal_action_ids"]]}
                for a, b, labels in pairs])
    graphs["goal_changes"] = {"methods": {"v3": paired_goal_metrics(GraphView(policy.model), pairs),
                                          "v2.1": paired_goal_metrics(tool_baseline.model, pairs)},
                              "candidates_examined": scanned,
                              "selection": "Model-independent: both goals unfinished, reachable, with disjoint optimal next-action sets. Same graph, state, and costs per pair."}
    print("Changed-goal retention audit complete", flush=True)
    belief, belief_traces = audit_beliefs(policy, problems, PROTOCOL["counts"]["episodes_per_problem"])
    _write_rows(run / "belief-audit-traces.jsonl", [{"method": name, **e} for name, items in belief_traces.items() for e in items])
    actual_tools, tool_traces = audit_tools(policy, tool_baseline)
    _write_rows(run / "actual-tool-audit-traces.jsonl", [{"method": name, **e} for name, items in tool_traces.items() for e in items])
    if hashlib.sha256((run / "model.safetensors").read_bytes()).hexdigest() != digest:
        raise RuntimeError("Checkpoint changed during audit")
    result = {**initial, "graph_retention": graphs, "belief": belief, "actual_tools": actual_tools,
              "promotion": promotion_gate(graphs, actual_tools, belief, report),
              "protocol": PROTOCOL, "elapsed_seconds": time.perf_counter()-started,
              "limitations": "Numeric declared-world decisions, known generator families and controlled supplied-patch fixtures. Fresh seeds are reserved from training but not proof of general real-world competence. Goal/permission masks are supplied rules. A single shared-Mac timing session is not a precision performance benchmark.",
              "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob("*-test.jsonl")},
              "trace_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob("*-audit-traces.jsonl")}}
    with (run / "evaluation.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"promotion": result["promotion"], "belief_expected": belief["expected"],
                      "belief_sampled": belief["sampled"], "actual_tools": actual_tools["methods"]}, indent=2))


if __name__ == "__main__":
    main()
