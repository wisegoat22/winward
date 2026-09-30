"""One-time audit of a frozen v2 checkpoint, with a matched v1 comparison.

Fresh rows are generated only after verifying the frozen checkpoint. This command
never trains or selects weights, and refuses to overwrite a completed audit.
Reusable v1 probes are labeled diagnostics, not blind evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .curriculum import competing_goal_pair, generate_rows, provenance_summary
from .evaluate import (counterfactual_metrics, decision_metrics, fresh_rows, greedy_action,
                       invariance_metrics, predict, rollout_metrics)
from .model import GoalPolicy, PolicyConfig
from .simulator import search
from .train import write_json


def fresh_goal_pairs(seed, count):
    """Model-independent selection of causal, reachable, unfinished goal pairs.

    Filtering uses exact mechanics only: both goals must have a solution and
    their optimal first-action sets must be disjoint. This is a targeted probe,
    not a representative estimate of all software-agent decisions.
    """
    pairs, scanned = [], 0
    for offset in range(count * 200):
        first, second = competing_goal_pair(seed + offset, "test", progress=True)
        labels = (search(first), search(second))
        scanned += 1
        if all(label["outcome"] == "plan" for label in labels) and not (
                set(labels[0]["optimal_action_ids"]) & set(labels[1]["optimal_action_ids"])):
            pairs.append((first, second, labels))
        if len(pairs) == count:
            return pairs, scanned
    raise RuntimeError("Insufficient causal pairs in reserved graph compositions")


def paired_goal_metrics(model, pairs, batch_size=128):
    observations, targets = [], []
    for first, second, labels in pairs:
        observations.extend(((first, 1.0, 0.01, 5), (second, 1.0, 0.01, 5)))
        targets.extend(set(label["optimal_action_ids"]) for label in labels)
    predictions, _, _ = predict(model, observations, batch_size)
    matches = [p in target for p, target in zip(predictions, targets)]
    greedy = [greedy_action(*observation) for observation in observations]
    failures = []
    for index, (first, second, labels) in enumerate(pairs):
        if len(failures) < 3 and not (matches[2*index] and matches[2*index+1]):
            failures.append({"scenario_id": first.id, "shape": first.family, "state": first.state,
                             "goals": [first.goal, second.goal], "chosen": predictions[2*index:2*index+2],
                             "optimal": [labels[0]["optimal_action_ids"], labels[1]["optimal_action_ids"]]})
    return {"pairs": len(pairs), "cases": len(observations), "neural_accuracy": float(np.mean(matches)),
            "neural_both_members_correct": sum(matches[i] and matches[i+1] for i in range(0, len(matches), 2)),
            "neural_choice_changed": sum(predictions[i] != predictions[i+1] for i in range(0, len(matches), 2)),
            "greedy_accuracy": float(np.mean([p in target for p, target in zip(greedy, targets)])),
            "failure_examples": failures}


def load_model(run):
    report = json.loads((run / "report.json").read_text())
    digest = hashlib.sha256((run / "model.safetensors").read_bytes()).hexdigest()
    if digest != report["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch in {run}")
    model = GoalPolicy(PolicyConfig(**json.loads((run / "config.json").read_text())))
    model.load_weights(str(run / "model.safetensors"))
    model.eval()
    return model, report, digest


def audit(model, rows, pairs, legacy_rows, batch_size, rollout_cases):
    decisions, scenarios, observations, _, _, _ = decision_metrics(model, rows, batch_size)
    legacy, legacy_scenarios, _, _, _, _ = decision_metrics(model, legacy_rows, batch_size)
    return {"decision_metrics": decisions,
            "closed_loop": rollout_metrics(model, scenarios, rows, rollout_cases, batch_size),
            "fresh_goal_changes": paired_goal_metrics(model, pairs, batch_size),
            "invariance": invariance_metrics(model, observations, 30, batch_size),
            "known_diagnostics": counterfactual_metrics(model, count=20, batch_size=batch_size),
            "legacy_fresh": {"decision_metrics": legacy,
                             "closed_loop": rollout_metrics(model, legacy_scenarios, legacy_rows, rollout_cases, batch_size)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/goalpolicy-v2")
    parser.add_argument("--baseline-run", default="runs/goalpolicy-v1")
    parser.add_argument("--fresh-seed", type=int, default=110000000)
    parser.add_argument("--cases", type=int, default=2400)
    parser.add_argument("--goal-pairs", type=int, default=200)
    parser.add_argument("--rollout-cases", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if min(args.cases, args.goal_pairs, args.rollout_cases, args.batch_size) < 1 or args.fresh_seed < 0:
        raise SystemExit("Counts must be positive and the seed nonnegative")
    run = Path(args.run)
    destination = run / "evaluation.json"
    if destination.exists() or (run / "test.jsonl").exists():
        raise SystemExit("The blind audit already exists; preserve it rather than rerunning or tuning against it.")
    model, report, digest = load_model(run)
    if not report.get("checkpoint_frozen_before_test_generation") or report.get("test") is not None:
        raise SystemExit("Expected a frozen v2 checkpoint that has not seen its test audit")
    if args.fresh_seed != report["test_seed_reserved"]:
        raise SystemExit("Use the test seed reserved before training")
    started = time.perf_counter()
    rows = generate_rows(args.cases, "test", args.fresh_seed)
    pairs, scanned = fresh_goal_pairs(args.fresh_seed + 10_000_000, args.goal_pairs)
    legacy_rows = fresh_rows(args.fresh_seed + 20_000_000, args.cases, "test")
    for name, data in (("test", rows), ("legacy-test", legacy_rows)):
        with (run / f"{name}.jsonl").open("w") as stream:
            for row in data:
                stream.write(json.dumps(row) + "\n")
    with (run / "goal-change-test.jsonl").open("w") as stream:
        for first, second, labels in pairs:
            stream.write(json.dumps({"first": first.to_dict(), "second": second.to_dict(),
                                     "optimal_ids": [label["optimal_action_ids"] for label in labels]}) + "\n")
    measured = audit(model, rows, pairs, legacy_rows, args.batch_size, args.rollout_cases)
    print("Frozen v2 audit complete; scoring v1 on the identical data", flush=True)
    baseline, baseline_report, baseline_digest = load_model(Path(args.baseline_run))
    baseline_measured = audit(baseline, rows, pairs, legacy_rows, args.batch_size, args.rollout_cases)
    result = {"checkpoint_sha256": digest, "input_dim": model.config.input_dim, "evaluation_split": "test",
              "dataset_provenance": {"source": "freshly generated after checkpoint freeze", "fresh_seed": args.fresh_seed,
                                     "goal_pair_seed": args.fresh_seed + 10_000_000, "goal_pair_candidates_examined": scanned,
                                     "legacy_seed": args.fresh_seed + 20_000_000, **provenance_summary(rows),
                                     "structure_family_status": "Curriculum test compositions reserved from v2 training and validation. Legacy families and reusable 20-pair diagnostics were previously examined during v1 work.",
                                     "validation_reuse": False,
                                     "goal_pair_selection": "Model-independent: both goals unfinished, reachable, and disjoint optimal next-action sets. Same graph, state and costs within a pair."},
              "selection_policy": report["checkpoint_selection"] + " Test data generated only after checkpoint hash was frozen; no post-test weight changes.",
              **measured,
              "counterfactuals": {"changed_goal": measured["fresh_goal_changes"],
                                  "changed_downstream_cost": measured["known_diagnostics"]["changed_downstream_cost"]},
              "matched_baseline": {"model_name": baseline_report["model_name"], "checkpoint_sha256": baseline_digest,
                                   "selection": "Existing v1 checkpoint, unchanged; scored on exactly the same rows and pairs.", **baseline_measured},
              "baselines": {"random_eligible": "Uniform applicable action or defer; STOP is forced for completed goals.",
                            "greedy_immediate": "Immediate goal-bit gain, then immediate cost; no lookahead.",
                            "v1": "Prior frozen checkpoint; matched cases and fixed seeds."},
              "limitations": "Synthetic deterministic task graphs, not general software work. Explicit inspection prerequisites do not represent real-world uncertainty. Masks enforce authorization, prerequisites and goal completion. Cost comparisons among successes exclude failures. Targeted goal pairs are filtered for causal action changes and are not representative of all tasks.",
              "elapsed_seconds": round(time.perf_counter() - started, 3)}
    result["data_sha256"] = {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                             for name in ("test.jsonl", "legacy-test.jsonl", "goal-change-test.jsonl")}
    write_json(destination, result)
    accuracy = measured["decision_metrics"]["neural"]
    report["test_cases"] = len(rows)
    report["test"] = {"accuracy": accuracy["all"]["accuracy"], "nontrivial_accuracy": accuracy["nontrivial"]["accuracy"],
                      "cases": len(rows), "nontrivial_cases": accuracy["nontrivial"]["cases"]}
    report["blind_audit"] = {"seed": args.fresh_seed, "evaluation_file": "evaluation.json",
                             "closed_loop_success_rate": measured["closed_loop"]["methods"]["neural"]["success_rate"],
                             "fresh_goal_change_accuracy": measured["fresh_goal_changes"]["neural_accuracy"]}
    write_json(run / "report.json", report)
    write_json(run / "status.json", {"status": "ready", "parameter_count": report["parameter_count"], "test": report["test"]})
    summary = {"checkpoint_sha256": digest, "v2": {"action_accuracy": report["test"]["accuracy"],
               "rollout": measured["closed_loop"]["methods"]["neural"], "fresh_goal_changes": measured["fresh_goal_changes"],
               "known_diagnostics": measured["known_diagnostics"], "legacy_rollout": measured["legacy_fresh"]["closed_loop"]["methods"]["neural"]},
               "v1": {"action_accuracy": baseline_measured["decision_metrics"]["neural"]["all"]["accuracy"],
                      "rollout": baseline_measured["closed_loop"]["methods"]["neural"],
                      "fresh_goal_changes": baseline_measured["fresh_goal_changes"],
                      "legacy_rollout": baseline_measured["legacy_fresh"]["closed_loop"]["methods"]["neural"]}}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
