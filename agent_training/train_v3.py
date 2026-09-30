"""Train one shared Winward v3 candidate using training/validation data only.

Our own v1 weights initialize the shared network; no third-party weights, text
teacher, cloud service, final-audit instance, or hidden realized world is read.
Strict retention gates are fixed before training. A failed candidate remains a
failed candidate; the trainer cannot silently relax gates or promote it.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from agent_lab.belief import BeliefProblem, NEEDS_INFORMATION, STOP as BELIEF_STOP, branches, verified
from agent_lab.runner import run_episode
from .curriculum import encode_rows
from .evaluate import rollout_metrics
from .model import GoalPolicy, PolicyConfig, loss_fn
from .model_v3 import (FEATURE_DIM, SCHEMA_VERSION, GraphView, UnifiedPolicy, V3Policy,
                       expand_belief, expand_graph, initialize_from_v1)
from .simulator import Scenario
from .train import evaluate, write_json
from .train_tools import fixture_specs
from .train_v2 import goal_sensitive_metrics

AUDIT_SEEDS = {"belief": 330000000, "graph": 340000000, "changed_goal": 342000000, "tools": 350000000}
ALLOWED_SOURCE_FILES = {"train.jsonl", "valid.jsonl", "valid-goals.jsonl", "valid-tools.jsonl"}


def read_source_rows(path):
    path = Path(path)
    if path.name not in ALLOWED_SOURCE_FILES:
        raise ValueError("Only explicitly permitted training/validation source files can be read")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_legacy(path):
    model = GoalPolicy(PolicyConfig(**json.loads((path / "config.json").read_text())))
    model.load_weights(str(path / "model.safetensors"))
    model.eval()
    return model


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_rows(path, rows):
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def add_correct_teacher_distillation(model, dataset, batch_size=128, mixing=0.15):
    """Retain train-only soft preferences when their top action is a true label.

    The 85% exact-label component remains primary. Incorrect v1 predictions get
    zero distillation weight, so old mistakes are not mandatory targets.
    """
    x, valid, eligible, targets = dataset
    result = targets.copy()
    accepted = 0
    for start in range(0, len(x), batch_size):
        stop = start + batch_size
        logits = model(*[mx.array(array[start:stop]) for array in (x, valid, eligible)])
        mx.eval(logits)
        values = np.asarray(logits)
        values = values - values.max(axis=1, keepdims=True)
        probability = np.exp(values)
        probability /= probability.sum(axis=1, keepdims=True)
        correct = targets[start:stop][np.arange(len(values)), values.argmax(axis=1)] > 0
        result[start:stop][correct] = (1 - mixing) * targets[start:stop][correct] + mixing * probability[correct]
        accepted += int(correct.sum())
    return (x, valid, eligible, result), {"rows": len(x), "teacher_correct_rows": accepted,
                                        "mixing_when_correct": mixing, "temperature": 1.0}


def expected_belief_completion(policy, rows, count=48):
    """Roll the learned decisions over observable branches, without correction."""
    selected = rows[:count]
    totals = []
    for row in selected:
        problem = BeliefProblem.from_dict(row["problem"])
        tw, lw = row["token_weight"], row["latency_weight"]

        def visit(current, remaining):
            if verified(current.belief, current.goal):
                return 1.0
            if remaining == 0:
                return 0.0
            chosen = policy.predict_belief(current, remaining, tw, lw)
            if chosen in (BELIEF_STOP, NEEDS_INFORMATION):
                return 0.0
            action = next((a for a in current.actions if a.id == chosen), None)
            if action is None or not action.eligible(current.belief):
                return 0.0
            return sum(probability * visit(replace(current, belief=posterior), remaining - 1)
                       for _, probability, posterior in branches(current.belief, action))

        totals.append(visit(problem, row["max_depth"]))
    return {"cases": len(totals), "expected_verified_completion": float(np.mean(totals)) if totals else 0.0,
            "scope": "Validation declared probability trees; neural choices only, including unfinished/unreachable cases."}


def retention_selection(metrics, baseline):
    """Strict per-capability gates, followed by uncertainty quality.

    A positive score never hides a regression: gate flags and raw deltas remain
    in the report, and promotion is reserved for the independent final audit.
    """
    pairs = {
        "legacy_completion": (metrics["legacy_completion"], baseline["legacy_completion"]),
        "goal_change_accuracy": (metrics["goal_change_accuracy"], baseline["goal_change_accuracy"]),
        "actual_tool_completion": (metrics["actual_tool_completion"], baseline["actual_tool_completion"]),
    }
    gates = {name: {"candidate": candidate, "baseline": reference, "delta": candidate - reference,
                    "passed": candidate + 1e-12 >= reference} for name, (candidate, reference) in pairs.items()}
    worst_delta = min(item["delta"] for item in gates.values())
    passed = all(item["passed"] for item in gates.values())
    # Until all pass, select the smallest worst regression. Once all pass, select
    # expected uncertainty completion, then teacher agreement, then retention.
    selection = (int(passed), 0.0 if passed else worst_delta,
                 metrics["belief_expected_completion"], metrics["belief_accuracy"],
                 metrics["legacy_completion"] + metrics["goal_change_accuracy"])
    return selection, gates


def main():
    from .belief_data import generate_rows as generate_beliefs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/goalpolicy-v3")
    parser.add_argument("--v1-run", default="runs/goalpolicy-v1")
    parser.add_argument("--tools-run", default="runs/goalpolicy-v2-tools")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--belief-cases", type=int, default=24000)
    parser.add_argument("--belief-valid-cases", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=300000000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    args = parser.parse_args()
    if not 1 <= args.steps <= 6000:
        raise SystemExit("The preset v3 training budget supports 1 to 6000 steps")
    if min(args.belief_cases, args.belief_valid_cases, args.batch_size) < 1 or not 0 < args.learning_rate <= 0.001:
        raise SystemExit("Counts must be positive and learning rate must be in (0, 0.001]")
    out, v1_dir, tools_dir = Path(args.output), Path(args.v1_run), Path(args.tools_run)
    if out.exists() and any(path.name != "audit-protocol.json" for path in out.iterdir()):
        raise SystemExit("Choose an empty output directory (or one containing only the sealed audit protocol); v3 runs are immutable")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    audit_protocol = out / "audit-protocol.json"
    protocol = {"schema_version": SCHEMA_VERSION, "input_dim": FEATURE_DIM, "training_seed": args.seed,
                "sealed_audit_protocol_sha256": _digest(audit_protocol) if audit_protocol.exists() else None,
                "belief_training_seed": 310000000, "belief_validation_seed": 320000000,
                "reserved_final_audit_seeds": AUDIT_SEEDS,
                "maximum_training_steps": 6000, "steps_requested": args.steps,
                "validation_selection_interval": 500, "regression_tolerance": 0.0,
                "training_mix_probabilities": {"legacy": 0.375, "goals": 0.1875, "tools": 0.1875, "belief": 0.25},
                "selection": "All strict retention gates first; otherwise minimize worst regression; then expected belief completion and belief teacher agreement.",
                "no_test_read_or_generation": True, "promotion": "Independent final audit required; trainer never promotes."}
    write_json(out / "protocol.json", protocol)
    write_json(out / "status.json", {"status": "preparing_training_data"})
    legacy_rows = read_source_rows(v1_dir / "train.jsonl")
    legacy_valid_rows = read_source_rows(v1_dir / "valid.jsonl")
    source_tool_rows = read_source_rows(tools_dir / "train.jsonl")
    goal_rows = [r for r in source_tool_rows if r.get("provenance", {}).get("source") != "actual_tool_observation"]
    tool_rows = [r for r in source_tool_rows if r.get("provenance", {}).get("source") == "actual_tool_observation"]
    goal_valid_rows = read_source_rows(tools_dir / "valid-goals.jsonl")
    tool_valid_rows = read_source_rows(tools_dir / "valid-tools.jsonl")
    if not goal_rows or not tool_rows:
        raise SystemExit("The v2.1 training source must contain both synthetic goals and actual tool observations")
    rng = np.random.default_rng(args.seed)
    # Subsample only training material; validation data is never augmented into it.
    goal_rows = [goal_rows[i] for i in rng.choice(len(goal_rows), min(16000, len(goal_rows)), replace=False)]
    tool_rows = [tool_rows[i] for i in rng.choice(len(tool_rows), min(16000, len(tool_rows)), replace=False)]
    belief_rows, belief_data = generate_beliefs(args.belief_cases, "train", 310000000)
    belief_valid_rows, belief_valid_data = generate_beliefs(args.belief_valid_cases, "validation", 320000000)
    for name, rows in (("train-beliefs", belief_rows), ("valid-beliefs", belief_valid_rows),
                       ("train-goals", goal_rows), ("train-tools", tool_rows)):
        _write_rows(out / f"{name}.jsonl", rows)
    train = {"legacy": encode_rows(legacy_rows), "goals": encode_rows(goal_rows),
             "tools": encode_rows(tool_rows), "belief": belief_data}
    valid = {"legacy": encode_rows(legacy_valid_rows), "goals": encode_rows(goal_valid_rows),
             "tools": encode_rows(tool_valid_rows), "belief": belief_valid_data}
    v1, previous_tools = load_legacy(v1_dir), load_legacy(tools_dir)
    train["legacy"], distillation = add_correct_teacher_distillation(v1, train["legacy"], args.batch_size)
    mx.random.seed(args.seed)
    config = PolicyConfig(**{**v1.config.to_dict(), "input_dim": FEATURE_DIM})
    model = UnifiedPolicy(config)
    initialize_from_v1(model, v1)
    model.train()
    mx.eval(model.parameters())
    write_json(out / "config.json", config.to_dict())
    legacy_scenarios = [Scenario.from_dict(row["scenario"]) for row in legacy_valid_rows]
    baseline_graph = rollout_metrics(v1, legacy_scenarios, legacy_valid_rows, count=200, batch_size=args.batch_size)
    baseline_goals = goal_sensitive_metrics(previous_tools, goal_valid_rows, args.batch_size)

    class LegacyTools:
        def choose(self, observation, remaining=5):
            from .features import encode
            x, valid_mask, eligible, candidates = encode(observation, max_depth=remaining)
            logits = previous_tools(mx.array(x[None]), mx.array(valid_mask[None]), mx.array(eligible[None]))
            mx.eval(logits)
            return {"action_id": candidates[int(np.asarray(logits)[0].argmax())]["id"]}

    validation_specs = fixture_specs(16, 220100000)
    baseline_tool_runs = [run_episode(**spec, policy_name="neural", neural_policy=LegacyTools()) for spec in validation_specs]
    baseline = {"legacy_completion": baseline_graph["methods"]["neural"]["success_rate"],
                "goal_change_accuracy": baseline_goals["paired_goal_active_accuracy"],
                "actual_tool_completion": float(np.mean([r["success"] and r["done"] for r in baseline_tool_runs])),
                "v1_checkpoint_sha256": _digest(v1_dir / "model.safetensors"),
                "v2_1_checkpoint_sha256": _digest(tools_dir / "model.safetensors"),
                "tool_validation_seed": 220100000, "legacy_validation_cases": baseline_graph["cases"],
                "goal_validation_cases": len(goal_valid_rows), "tool_validation_cases": len(validation_specs)}
    write_json(out / "validation_baselines.json", baseline)
    print("Strict validation baselines: " + json.dumps(baseline), flush=True)
    policy = V3Policy(model=model)
    graph_view = GraphView(model)
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.01)
    gradient = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def update(x, valid_mask, eligible, targets):
        loss, grads = gradient(model, x, valid_mask, eligible, targets)
        grads, norm = optim.clip_grad_norm(grads, max_norm=1.0)
        optimizer.update(model, grads)
        return loss, norm

    names = list(protocol["training_mix_probabilities"])
    probabilities = [protocol["training_mix_probabilities"][name] for name in names]
    best, history = None, []
    for iteration in range(1, args.steps + 1):
        source_ids = rng.choice(len(names), args.batch_size, p=probabilities)
        chunks = [[] for _ in range(4)]
        for source_id, name in enumerate(names):
            count = int((source_ids == source_id).sum())
            if not count:
                continue
            data = train[name]
            indices = rng.integers(0, len(data[0]), count)
            chunks[0].append((expand_belief if name == "belief" else expand_graph)(data[0][indices]))
            for field in range(1, 4):
                chunks[field].append(data[field][indices])
        batch = [mx.array(np.concatenate(parts)) for parts in chunks]
        warmup = min(1.0, iteration / 100)
        decay = 0.15 + 0.85 * 0.5 * (1 + math.cos(math.pi * iteration / args.steps))
        optimizer.learning_rate = args.learning_rate * warmup * decay
        loss, norm = update(*batch)
        mx.eval(state, loss, norm)
        if iteration == 1 or iteration % 100 == 0:
            progress = {"status": "training", "step": iteration, "steps": args.steps,
                        "loss": float(loss.item()), "parameter_count": model.count_parameters(),
                        "elapsed_seconds": round(time.perf_counter() - started, 2)}
            print(json.dumps(progress), flush=True)
            write_json(out / "status.json", progress)
        if iteration % 500 == 0 or iteration == args.steps:
            graph = rollout_metrics(graph_view, legacy_scenarios, legacy_valid_rows, count=200, batch_size=args.batch_size)
            goals = goal_sensitive_metrics(graph_view, goal_valid_rows, args.batch_size)
            tool_decisions = evaluate(graph_view, valid["tools"], args.batch_size)
            actual = [run_episode(**spec, policy_name="neural", neural_policy=policy) for spec in validation_specs]
            belief_arrays = (expand_belief(valid["belief"][0]), *valid["belief"][1:])
            belief_decisions = evaluate(model, belief_arrays, args.batch_size)
            belief_completion = expected_belief_completion(policy, belief_valid_rows)
            metrics = {"step": iteration, "legacy_completion": graph["methods"]["neural"]["success_rate"],
                       "goal_change_accuracy": goals["paired_goal_active_accuracy"],
                       "actual_tool_completion": float(np.mean([r["success"] and r["done"] for r in actual])),
                       "belief_expected_completion": belief_completion["expected_verified_completion"],
                       "belief_accuracy": belief_decisions["accuracy"],
                       "goal_decisions": goals, "tool_decisions": tool_decisions, "belief_decisions": belief_decisions,
                       "belief_rollout": belief_completion}
            selection, gates = retention_selection(metrics, baseline)
            metrics.update(selection=list(selection), retention_gates=gates, all_retention_gates_passed=bool(selection[0]))
            history.append(metrics)
            print("Validation: " + json.dumps(metrics), flush=True)
            if best is None or selection > best:
                best = selection
                model.save_weights(str(out / "model.safetensors"))
                write_json(out / "best_validation.json", metrics)
    source_files = [v1_dir / "train.jsonl", v1_dir / "valid.jsonl", tools_dir / "train.jsonl",
                    tools_dir / "valid-goals.jsonl", tools_dir / "valid-tools.jsonl"]
    data_files = [out / f"{name}.jsonl" for name in ("train-beliefs", "valid-beliefs", "train-goals", "train-tools")]
    best_validation = json.loads((out / "best_validation.json").read_text())
    report = {"model_name": "Winward v3 candidate", "schema_version": SCHEMA_VERSION,
              "initialization": "Our immutable v1 checkpoint; legacy weights copied, additional input columns zero initialized",
              "sealed_audit_protocol_sha256": protocol["sealed_audit_protocol_sha256"],
              "training_origin": "Transfer and continued training from our own random-initialized v1; new belief/domain columns initialized to zero. No third-party pretrained weights or Qwen outputs.",
              "architecture": "One shared set transformer and one output head; explicit observation-format channels, no ensemble, no exact-planner correction at inference.",
              "parameter_count": model.count_parameters(), "config": config.to_dict(), "seed": args.seed,
              "training_cases": sum(len(data[0]) for data in train.values()),
              "validation_cases": sum(len(data[0]) for data in valid.values()), "test_cases": 0, "test": None,
              "training_mix_cases": {name: len(data[0]) for name, data in train.items()},
              "sampling_probabilities": protocol["training_mix_probabilities"], "distillation": distillation,
              "steps": args.steps, "selected_step": best_validation["step"], "batch_size": args.batch_size,
              "learning_rate": args.learning_rate, "validation_baselines": baseline,
              "best_validation": best_validation, "validation_history": history,
              "validation_retention_passed": best_validation["all_retention_gates_passed"],
              "checkpoint_selection": protocol["selection"], "regression_tolerance": 0.0,
              "checkpoint_sha256": _digest(out / "model.safetensors"),
              "checkpoint_frozen_before_test_generation": True, "reserved_final_audit_seeds": AUDIT_SEEDS,
              "elapsed_seconds": round(time.perf_counter() - started, 2),
              "source_file_sha256": {str(path): _digest(path) for path in source_files},
              "data_sha256": {path.name: _digest(path) for path in data_files},
              "limitations": "Structured public models only, supplied transition probabilities and candidate actions. No language understanding or autonomous patch generation. Permissions, applicability and verified-stop masks are hard rules. No claim of superiority or promotion until the frozen independent audit."}
    write_json(out / "report.json", report)
    write_json(out / "status.json", {"status": "awaiting_blind_audit", "parameter_count": model.count_parameters(),
                                     "validation_retention_passed": report["validation_retention_passed"],
                                     "checkpoint_sha256": report["checkpoint_sha256"]})
    print(json.dumps({"checkpoint_frozen": report["checkpoint_sha256"], "selected_step": report["selected_step"],
                      "validation_retention_passed": report["validation_retention_passed"],
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
