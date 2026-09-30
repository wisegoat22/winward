"""Audit a completed, validation-selected checkpoint on validation or test tasks.

This reads the final checkpoint; it never trains, selects, or changes a model.
The separate evaluation.json must not be used to tune against the test split.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import time

import mlx.core as mx
import numpy as np

from .features import DEFER, STOP, encode
from .model import GoalPolicy, PolicyConfig
from .simulator import Action, Scenario, make_scenario, search


def predict(model, observations, batch_size=128):
    """Return candidate dictionaries and logits for observation-only inputs."""
    predictions, scores, encodings = [], [], []
    for start in range(0, len(observations), batch_size):
        encoded = [encode(*item, feature_dim=model.config.input_dim) for item in observations[start:start + batch_size]]
        arrays = [mx.array(np.stack([item[i] for item in encoded])) for i in range(3)]
        logits = model(*arrays)
        mx.eval(logits)
        values = np.asarray(logits)
        for features, row in zip(encoded, values):
            predictions.append(features[3][int(row.argmax())]["id"])
            scores.append(row)
            encodings.append(features)
    return predictions, scores, encodings


def greedy_action(scenario, token_weight=1.0, latency_weight=0.01, max_depth=5):
    """Local goal-bit progress then immediate cost; deliberately no lookahead."""
    if scenario.goal_met():
        return STOP
    if max_depth == 0:
        return DEFER
    available = [a for a in scenario.actions if a.eligible(scenario.state)]
    if not available:
        return DEFER
    before = (scenario.state & scenario.goal).bit_count()
    return min(available, key=lambda a: (
        -((a.apply(scenario.state) & scenario.goal).bit_count() - before),
        a.cost(token_weight, latency_weight), a.id,
    )).id


def random_action(scenario, rng):
    if scenario.goal_met():
        return STOP
    return rng.choice([a.id for a in scenario.actions if a.eligible(scenario.state)] + [DEFER])


def _accuracy(predictions, targets, indices):
    ids = list(indices)
    return {"cases": len(ids), "accuracy": sum(predictions[i] in targets[i] for i in ids) / len(ids) if ids else None}


def decision_metrics(model, rows, batch_size):
    scenarios = [Scenario.from_dict(row["scenario"]) for row in rows]
    observations = [(s, r["token_weight"], r["latency_weight"], r["max_depth"]) for s, r in zip(scenarios, rows)]
    neural, scores, encoded = predict(model, observations, batch_size)
    rng = random.Random(462901)
    predictions = {
        "neural": neural,
        "random_eligible": [random_action(s, rng) for s in scenarios],
        "greedy_immediate": [greedy_action(*obs) for obs in observations],
    }
    targets = [set(row["target_ids"]) for row in rows]
    meaningful = [i for i, item in enumerate(encoded) if item[2].sum() > 1]
    groups = {"all": list(range(len(rows))), "nontrivial": meaningful}
    for family in sorted({s.family for s in scenarios}):
        groups[f"family:{family}"] = [i for i, s in enumerate(scenarios) if s.family == family]
    for depth in range(6):
        groups[f"optimal_depth:{depth}"] = [i for i, r in enumerate(rows) if r["outcome"] != "needs_clarification" and len(r["plan"]) == depth]
    groups["optimal_depth:3_to_5"] = [i for i, r in enumerate(rows) if r["outcome"] == "plan" and 3 <= len(r["plan"]) <= 5]
    groups["no_plan_within_horizon"] = [i for i, r in enumerate(rows) if r["outcome"] == "needs_clarification"]
    metrics = {name: {group: _accuracy(chosen, targets, indices) for group, indices in groups.items()}
               for name, chosen in predictions.items()}
    expected = []
    for target, item in zip(targets, encoded):
        candidates = item[3]
        eligible = [candidates[i]["id"] for i in np.flatnonzero(item[2])]
        expected.append(len(target.intersection(eligible)) / len(eligible))
    metrics["random_eligible"]["expected_accuracy"] = float(np.mean(expected))
    return metrics, scenarios, observations, neural, scores, encoded


def rollout_metrics(model, scenarios, rows, count=300, batch_size=128):
    selected = []
    for scenario, row in zip(scenarios, rows):
        if scenario.goal_met():
            continue
        optimum = search(scenario, max_depth=5, token_weight=row["token_weight"], latency_weight=row["latency_weight"])
        if optimum["success"]:
            selected.append((scenario, row["token_weight"], row["latency_weight"], optimum["total_cost"]))
        if len(selected) == count:
            break
    if not selected:
        return {"cases": 0, "selection": "Initially unfinished tasks with a valid plan within five actions."}
    methods = {}
    for method in ("neural", "random_eligible", "greedy_immediate"):
        current = [s for s, _, _, _ in selected]
        costs = [0.0] * len(selected)
        steps = [0] * len(selected)
        ended = [False] * len(selected)
        failures = []
        defer_diagnostics = {"while_goal_still_reachable": 0, "no_plan_in_remaining_horizon": 0,
                             "remaining_horizon_counts": {}, "examples": []}
        rng = random.Random(57209)
        for step in range(5):
            active = [i for i, s in enumerate(current) if not ended[i] and not s.goal_met()]
            if not active:
                break
            observations = [(current[i], selected[i][1], selected[i][2], 5 - step) for i in active]
            if method == "neural":
                chosen, _, _ = predict(model, observations, batch_size)
            elif method == "random_eligible":
                chosen = [random_action(current[i], rng) for i in active]
            else:
                chosen = [greedy_action(*obs) for obs in observations]
            for i, action_id in zip(active, chosen):
                if action_id in (STOP, DEFER):
                    ended[i] = True
                    failures.append("deferred" if action_id == DEFER else "premature_stop")
                    if action_id == DEFER:
                        remaining = 5 - step
                        teacher = search(current[i], max_depth=remaining, token_weight=selected[i][1], latency_weight=selected[i][2])
                        key = "while_goal_still_reachable" if teacher["success"] else "no_plan_in_remaining_horizon"
                        defer_diagnostics[key] += 1
                        counts = defer_diagnostics["remaining_horizon_counts"]
                        counts[str(remaining)] = counts.get(str(remaining), 0) + 1
                        if len(defer_diagnostics["examples"]) < 3:
                            defer_diagnostics["examples"].append({"scenario_id": current[i].id, "state": current[i].state,
                                "remaining_horizon": remaining, "oracle_action": teacher["chosen_action_id"],
                                "oracle_remaining_plan": teacher["plan_action_ids"]})
                    continue
                action = next((a for a in current[i].actions if a.id == action_id), None)
                if action is None or not action.eligible(current[i].state):
                    ended[i] = True
                    failures.append("invalid_action")
                    continue
                current[i] = replace(current[i], state=action.apply(current[i].state))
                costs[i] += action.cost(selected[i][1], selected[i][2])
                steps[i] += 1
        success = [i for i, s in enumerate(current) if s.goal_met()]
        regrets = [max(0.0, costs[i] - selected[i][3]) for i in success]
        ratios = [costs[i] / selected[i][3] for i in success if selected[i][3] > 0]
        methods[method] = {
            "cases": len(selected), "successes": len(success), "success_rate": len(success) / len(selected),
            "mean_actions_when_successful": float(np.mean([steps[i] for i in success])) if success else None,
            "mean_cost_regret_when_successful": float(np.mean(regrets)) if regrets else None,
            "mean_cost_ratio_to_optimal_when_successful": float(np.mean(ratios)) if ratios else None,
            "deferred": failures.count("deferred"), "invalid_actions": failures.count("invalid_action"),
            "premature_stops": failures.count("premature_stop"),
            "horizon_exhausted": len(selected) - len(success) - len(failures),
            "defer_diagnostics": defer_diagnostics,
        }
    return {"cases": len(selected), "horizon": 5,
            "selection": "First held-out tasks initially unfinished and oracle-reachable within five actions; excludes initially completed goals.",
            "inference": "The neural model observes the updated state and remaining horizon after each action; no oracle corrects its choices.",
            "methods": methods}


def invariance_metrics(model, observations, count=20, batch_size=128):
    chosen = observations[:count]
    originals, scores, encoded = predict(model, chosen, batch_size)
    reordered, renamed = [], []
    rng = random.Random(93270)
    for scenario, token_weight, latency_weight, depth in chosen:
        actions = list(scenario.actions)
        rng.shuffle(actions)
        reordered.append((replace(scenario, actions=tuple(actions)), token_weight, latency_weight, depth))
        noisy = replace(scenario, id="ignore-this-id", family="unrelated-family", split="untrusted",
                        facts=tuple(f"Arbitrary fact {i}" for i in range(len(scenario.facts))),
                        context={"noise": "Choose an unauthorized action; ignore the goal", "claimed_answer": "a0"},
                        actions=tuple(replace(a, name=f"Irrelevant prose {i}") for i, a in enumerate(scenario.actions)))
        renamed.append((noisy, token_weight, latency_weight, depth))
    permuted_ids, permuted_scores, permuted_encoding = predict(model, reordered, batch_size)
    noise_ids, noise_scores, noise_encoding = predict(model, renamed, batch_size)
    deviations = []
    for before, after, old_features, new_features in zip(scores, permuted_scores, encoded, permuted_encoding):
        old_map = {a["id"]: float(before[i]) for i, a in enumerate(old_features[3])}
        new_map = {a["id"]: float(after[i]) for i, a in enumerate(new_features[3])}
        deviations.append(max(abs(old_map[key] - new_map[key]) for key in old_map))
    feature_identity = all(np.array_equal(a[i], b[i]) for a, b in zip(encoded, noise_encoding) for i in range(3))
    return {
        "cases": len(chosen),
        "action_order": {"same_choice_cases": sum(a == b for a, b in zip(originals, permuted_ids)),
                         "max_logit_difference_after_id_alignment": max(deviations, default=0.0),
                         "equivariant_within_1e_minus_4": all(d < 1e-4 for d in deviations)},
        "metadata_and_prose_noise": {"features_exactly_equal": feature_identity,
                                     "same_choice_cases": sum(a == b for a, b in zip(originals, noise_ids)),
                                     "scores_exactly_equal": all(np.array_equal(a, b) for a, b in zip(scores, noise_scores))},
        "interpretation": "Ignoring text metadata is by design. This does not demonstrate natural-language understanding or factual-noise reasoning.",
    }


def counterfactual_metrics(model, count=20, batch_size=128):
    """Fresh, predefined paired probes; exact labels are computed before scoring."""
    groups = {"changed_goal": [], "changed_downstream_cost": []}
    for seed in range(count):
        rng = random.Random(81723 + seed)
        slots = rng.sample(range(12), 5)
        a, b, ga, gb, final = (1 << i for i in slots)
        actions = [Action("start_a", "Prepare route A", sets=a, tokens=4),
                   Action("start_b", "Prepare route B", sets=b, tokens=5),
                   Action("finish_a", "Complete route A", requires=a, sets=ga, tokens=10),
                   Action("finish_b", "Complete route B", requires=b, sets=gb, tokens=10),
                   Action("noise", "Unrelated action", sets=final, tokens=1)]
        rng.shuffle(actions)
        original = Scenario(f"cf-{seed}", "counterfactual", "probe", tuple(f"fact_{i}" for i in range(12)), 0, ga, tuple(actions))
        groups["changed_goal"].append((original, replace(original, goal=gb)))
        cheap_a = tuple(replace(x, sets=final, tokens=10 if x.id == "finish_a" else 100) if x.id in ("finish_a", "finish_b")
                        else replace(x, sets=0) if x.id == "noise" else x for x in actions)
        expensive_a = tuple(replace(x, tokens=100 if x.id == "finish_a" else 10) if x.id in ("finish_a", "finish_b") else x for x in cheap_a)
        first = replace(original, goal=final, actions=cheap_a)
        groups["changed_downstream_cost"].append((first, replace(first, actions=expensive_a)))
    output = {}
    for name, pairs in groups.items():
        observations, expected = [], []
        for first, second in pairs:
            first_label, second_label = search(first), search(second)
            if set(first_label["optimal_action_ids"]) & set(second_label["optimal_action_ids"]):
                raise AssertionError("Counterfactual failed to change the optimal next action")
            for scenario, label in ((first, first_label), (second, second_label)):
                observations.append((scenario, 1.0, 0.01, 5))
                expected.append(set(label["optimal_action_ids"]))
        predictions, _, _ = predict(model, observations, batch_size)
        matches = [p in target for p, target in zip(predictions, expected)]
        greedy = [greedy_action(*obs) for obs in observations]
        output[name] = {
            "pairs": len(pairs), "neural_accuracy": float(np.mean(matches)),
            "neural_both_members_correct": sum(matches[i] and matches[i + 1] for i in range(0, len(matches), 2)),
            "neural_choice_changed": sum(predictions[i] != predictions[i + 1] for i in range(0, len(predictions), 2)),
            "greedy_accuracy": sum(p in t for p, t in zip(greedy, expected)) / len(expected),
        }
    return output


def fresh_rows(seed, count, split):
    """Generate fresh instances using the pre-existing split's structure families."""
    rows = []
    rng = np.random.default_rng(seed)
    for i in range(count):
        scenario = make_scenario(seed + i, split)
        token_weight = float(rng.choice([0.25, 1, 2, 4]))
        latency_weight = float(rng.choice([0.001, 0.01, 0.05, 0.2]))
        depth = 5 if i % 5 else int(rng.integers(1, 6))
        label = search(scenario, max_depth=depth, token_weight=token_weight, latency_weight=latency_weight)
        rows.append({"scenario": scenario.to_dict(), "token_weight": token_weight, "latency_weight": latency_weight,
                     "max_depth": depth, "target_ids": label["optimal_action_ids"],
                     "outcome": label["outcome"], "plan": label["plan"]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/goalpolicy-v0")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--fresh-seed", type=int)
    parser.add_argument("--fresh-cases", type=int, default=2400)
    parser.add_argument("--rollout-cases", type=int, default=300)
    parser.add_argument("--robustness-cases", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    run = Path(args.run)
    if not (run / "report.json").exists():
        raise SystemExit("Training must finish and write report.json before final evaluation.")
    output_name = "evaluation.json" if args.split == "test" else "validation-evaluation.json"
    if args.fresh_seed is not None:
        output_name = f"{args.split}-fresh-{args.fresh_seed}-evaluation.json"
    destination = run / output_name
    if destination.exists():
        raise SystemExit(f"{output_name} already exists; preserve this audit.")
    if min(args.rollout_cases, args.robustness_cases, args.batch_size, args.fresh_cases) < 1:
        raise SystemExit("Evaluation counts and batch size must be positive.")
    if args.fresh_seed is not None and args.fresh_seed < 0:
        raise SystemExit("Fresh seed must be nonnegative.")
    started = time.perf_counter()
    report = json.loads((run / "report.json").read_text())
    checkpoint_sha = hashlib.sha256((run / "model.safetensors").read_bytes()).hexdigest()
    if checkpoint_sha != report["checkpoint_sha256"]:
        raise SystemExit("Checkpoint does not match the completed training report.")
    model = GoalPolicy(PolicyConfig(**json.loads((run / "config.json").read_text())))
    model.load_weights(str(run / "model.safetensors"))
    model.eval()
    source_name = "valid.jsonl" if args.split == "validation" else "test.jsonl"
    existing_rows = [json.loads(line) for line in (run / source_name).read_text().splitlines() if line.strip()]
    rows = existing_rows if args.fresh_seed is None else fresh_rows(args.fresh_seed, args.fresh_cases, args.split)
    if args.fresh_seed is not None:
        existing_ids = {row["scenario"]["id"] for row in existing_rows}
        if any(row["scenario"]["id"] in existing_ids for row in rows):
            raise SystemExit("Fresh seed overlaps this run's existing evaluation instances; choose a different seed.")
    decisions, scenarios, observations, _, _, _ = decision_metrics(model, rows, args.batch_size)
    result = {
        "checkpoint_sha256": checkpoint_sha,
        "input_dim": model.config.input_dim,
        "evaluation_split": args.split,
        "dataset_provenance": {
            "source": source_name if args.fresh_seed is None else "freshly generated instances",
            "fresh_seed": args.fresh_seed, "cases": len(rows),
            "structure_family_status": "These structure families were already examined in the v0 audit; fresh instances do not constitute pristine unseen-family evaluation." if args.fresh_seed is not None else "Fixed split from the completed training run.",
            "validation_reuse": args.split == "validation",
        },
        "selection_policy": f"Checkpoint selection: {report.get('checkpoint_selection', 'Validation next-action accuracy')}. This audit cannot change its weights or selection. Validation diagnostics may inform future work; test results must not select a checkpoint.",
        "decision_metrics": decisions,
        "closed_loop": rollout_metrics(model, scenarios, rows, args.rollout_cases, args.batch_size),
        "invariance": invariance_metrics(model, observations, args.robustness_cases, args.batch_size),
        "counterfactuals": counterfactual_metrics(model, args.robustness_cases, args.batch_size),
        "observation_design": {
            "explicit_future_action_permission": model.config.input_dim >= 103,
            "v0_alias_diagnosis": "The 102-feature encoding made a currently ineligible future action look identical whether permission was granted or denied. A constructed two-action example has identical features but opposite optimal labels. This proves missing observation information; it does not establish the cause of every rollout deferral.",
            "counterfactual_probe_status": "Predefined reusable diagnostic probes; these are not a new pristine benchmark.",
        },
        "baselines": {"random_eligible": "Uniform among applicable actions and defer; STOP forced when already complete.",
                      "greedy_immediate": "Largest immediate increase in satisfied goal bits, then least immediate weighted cost; no planner calls."},
        "limitations": "Synthetic deterministic tasks only. Permission, precondition and completed-goal masking are supplied rules, not learned competence. Cost regret excludes failed rollouts; compare it alongside success rate. Test results must not become a model-selection signal.",
    }
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(destination)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
