"""A fixed two-candidate validation-only reliability experiment on Apple MLX.

Declare both protocols before training. The final audit is separate and must
never feed checkpoint selection. Existing runs and completed trials are not
overwritten. There is one learned shared policy at inference, without teachers.
"""
from __future__ import annotations

import argparse
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

from agent_lab.runner import run_episode
from .curriculum import encode_rows
from .evaluate import rollout_metrics
from .model import PolicyConfig
from .model_v4 import (FEATURE_DIM, SCHEMA_VERSION, GraphView, UnifiedPolicy,
                       V4Policy, expand_belief, expand_graph, initialize_from_v1)
from .simulator import Scenario
from .train import evaluate, write_json
from .train_tools import fixture_specs
from .train_v2 import goal_sensitive_metrics
from .train_v3 import (add_correct_teacher_distillation, expected_belief_completion,
                       load_legacy, read_source_rows)


TRAINING_PLAN = {
    "version": "winward-v4-training-1",
    "schema_version": SCHEMA_VERSION,
    "architecture": "Unchanged shared505-input, six-layer256-width set transformer, one policy head",
    "candidates": [
        {"name": "replay60", "distillation_mixing": 0.60, "learning_rate": 0.00005, "seed": 400000001},
        {"name": "replay80", "distillation_mixing": 0.80, "learning_rate": 0.00003, "seed": 400000002},
    ],
    "steps_per_candidate": 10000, "total_step_budget": 20000,
    "batch_size": 128, "validation_interval": 500,
    "training_mix_probabilities": {"legacy": 0.4, "goals": 0.2, "tools": 0.1, "belief": 0.3},
    "belief_training_cases": 48000, "belief_validation_cases": 2400,
    "belief_training_seed": 410000000, "belief_validation_seed": 420000000,
    "belief_validation_rollouts": 120, "legacy_validation_rollouts": 200,
    "actual_tool_validation_fixtures": 16, "actual_tool_validation_seed": 220100000,
    "initialization": "Our frozen v1, unchanged graph logits before training; no third-party weights",
    "replay": "All original v1 training rows and all v2.1 goal/tool training rows, without subsampling",
    "distillation": "Training rows only: v1 on legacy and v2.1 on goals/tools, only when teacher top action is an exact accepted target; temperature1",
    "success_regret_coefficient": 4.0,
    "success_regret": "Belief only: conditional auxiliary success ranking loss over exact-value-labeled candidates, not full-policy expected regret. Unknown/pruned values are excluded from this auxiliary softmax. Full eligible-action cross-entropy remains active.",
    "selection": "Across both candidates and all predeclared validation checkpoints: all strict retention gates first; otherwise minimize worst regression; then expected belief completion, belief next-action accuracy, and legacy completion plus goal-change action accuracy. Exact ties keep the earlier checkpoint.",
    "retention_gates": ["legacy_completion_vs_v1", "changed_goal_accuracy_vs_v2.1", "changed_goal_pair_correctness_vs_v2.1", "actual_tool_completion_vs_v2.1"],
    "regression_tolerance": 0.0, "test_data_read_or_generated": False,
    "reserved_final_audit_seeds": {"belief": 430000000, "legacy": 440000000, "curriculum": 441000000, "changed_goal": 442000000, "tools": 450000000},
    "promotion": "Independent sealed final audit only. Validation or training never promotes a policy.",
}

SOURCE_FILES = tuple(f"agent_training/{name}.py" for name in (
    "belief_data", "belief_data_v4", "curriculum", "evaluate", "features", "model",
    "model_v3", "model_v4", "simulator", "train", "train_tools", "train_v2", "train_v3", "train_v4"
)) + tuple(f"agent_lab/{name}.py" for name in ("belief", "runner", "sandbox"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {name: digest(root / name) for name in SOURCE_FILES}


def checkpoint_hashes(v1_dir, tools_dir):
    return {"v1": digest(Path(v1_dir) / "model.safetensors"),
            "v2.1": digest(Path(tools_dir) / "model.safetensors"),
            "v1_config": digest(Path(v1_dir) / "config.json"),
            "v2.1_config": digest(Path(tools_dir) / "config.json")}


def seal_training_plan(output, v1_dir="runs/goalpolicy-v1", tools_dir="runs/goalpolicy-v2-tools"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    value = {"plan": TRAINING_PLAN, "training_plan_sha256": canonical_digest(TRAINING_PLAN),
             "initialization_checkpoint_sha256": checkpoint_hashes(v1_dir, tools_dir),
             "source_sha256": source_hashes()}
    with (output / "training-protocol.json").open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    return value


def validate_sealed_protocols(output, v1_dir="runs/goalpolicy-v1", tools_dir="runs/goalpolicy-v2-tools"):
    output = Path(output)
    allowed = {"audit-protocol.json", "training-protocol.json"}
    if not all((output / name).is_file() for name in allowed):
        raise ValueError("Seal audit-protocol.json and training-protocol.json before training")
    if any(path.name not in allowed for path in output.iterdir()):
        raise ValueError("Training output must contain only the two sealed protocols; runs are immutable")
    value = json.loads((output / "training-protocol.json").read_text())
    if value.get("plan") != TRAINING_PLAN or value.get("training_plan_sha256") != canonical_digest(TRAINING_PLAN):
        raise ValueError("Sealed training plan does not match this implementation")
    if value.get("initialization_checkpoint_sha256") != checkpoint_hashes(v1_dir, tools_dir):
        raise ValueError("An initialization checkpoint changed after sealing")
    if value.get("source_sha256") != source_hashes():
        raise ValueError("Training source changed after sealing")
    audit = json.loads((output / "audit-protocol.json").read_text())
    if audit.get("protocol_sha256") != canonical_digest(audit.get("protocol")):
        raise ValueError("Sealed audit protocol content does not match its declared hash")
    if not audit["protocol"].get("version", "").startswith("winward-v4-"):
        raise ValueError("Training requires a v4 audit protocol")
    return {"sealed_audit_protocol_sha256": digest(output / "audit-protocol.json"),
            "sealed_training_protocol_sha256": digest(output / "training-protocol.json")}


def belief_value_arrays(rows, eligible):
    regrets = np.asarray([row["teacher"]["candidate_success_regret"] for row in rows], dtype=np.float32)
    masks = np.asarray([row["teacher"]["candidate_value_mask"] for row in rows], dtype=bool)
    if regrets.shape != eligible.shape or masks.shape != eligible.shape:
        raise ValueError("Exact value labels must align with every candidate slot")
    if not np.isfinite(regrets).all() or (regrets < -1e-7).any() or (regrets > 1.0000001).any():
        raise ValueError("Success regrets must be finite probabilities in [0, 1]")
    if np.any(masks & ~eligible):
        raise ValueError("Only eligible candidates may have value labels")
    if np.any(~masks.any(axis=1)):
        raise ValueError("Every belief training row needs at least one exact candidate value")
    return np.maximum(regrets, 0.0), masks


def success_regret(logits, regrets, value_mask):
    """Teacher values are loss targets, never supplied to the policy forward pass."""
    known_logits = mx.where(value_mask, logits, mx.array(-1e9))
    probability = mx.softmax(known_logits, axis=-1) * value_mask
    return mx.mean(mx.sum(probability * regrets, axis=-1))


def reliability_loss(model, x, valid, eligible, targets, regrets, value_mask):
    logits = model(x, valid, eligible)
    cross_entropy = -mx.mean(mx.sum(targets * (logits - mx.logsumexp(logits, axis=-1, keepdims=True)), axis=-1))
    return cross_entropy + TRAINING_PLAN["success_regret_coefficient"] * success_regret(logits, regrets, value_mask)


def retention_selection(metrics, baseline):
    names = ("legacy_completion", "goal_change_accuracy", "goal_change_pair_correctness", "actual_tool_completion")
    gates = {name: {"candidate": metrics[name], "baseline": baseline[name],
                    "delta": metrics[name] - baseline[name],
                    "passed": metrics[name] >= baseline[name]} for name in names}
    passed = all(row["passed"] for row in gates.values())
    worst = min(row["delta"] for row in gates.values())
    selection = (int(passed), 0.0 if passed else worst, metrics["belief_expected_completion"],
                 metrics["belief_accuracy"], metrics["legacy_completion"] + metrics["goal_change_accuracy"])
    return selection, gates


def write_rows(path, rows):
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def _validation(model, valid, rows, specs, baseline, *, step, candidate):
    policy, graph_view = V4Policy(model=model), GraphView(model)
    graph = rollout_metrics(graph_view, [Scenario.from_dict(r["scenario"]) for r in rows["legacy"]],
                            rows["legacy"], count=TRAINING_PLAN["legacy_validation_rollouts"])
    goals = goal_sensitive_metrics(graph_view, rows["goals"])
    goals["causal_choice_pairs_both_correct_count"] = round(goals["causal_choice_pair_both_correct"] * goals["causal_choice_pairs"])
    tool_decisions = evaluate(graph_view, valid["tools"])
    actual = [run_episode(**spec, policy_name="neural", neural_policy=policy) for spec in specs]
    belief_decisions = evaluate(model, (expand_belief(valid["belief"][0]), *valid["belief"][1:]))
    belief_completion = expected_belief_completion(policy, rows["belief"], count=TRAINING_PLAN["belief_validation_rollouts"])
    metrics = {"candidate": candidate, "step": step,
               "legacy_completion": graph["methods"]["neural"]["success_rate"],
               "goal_change_accuracy": goals["paired_goal_active_accuracy"],
               "goal_change_pair_correctness": goals["causal_choice_pair_both_correct"],
               "actual_tool_completion": float(np.mean([r["success"] and r["done"] for r in actual])),
               "belief_expected_completion": belief_completion["expected_verified_completion"],
               "belief_accuracy": belief_decisions["accuracy"], "goal_decisions": goals,
               "tool_decisions": tool_decisions, "belief_decisions": belief_decisions, "belief_rollout": belief_completion}
    selection, gates = retention_selection(metrics, baseline)
    return {**metrics, "selection": list(selection), "retention_gates": gates,
            "all_retention_gates_passed": bool(selection[0])}


def train(output, v1_dir, tools_dir):
    from .belief_data_v4 import generate_rows
    output, v1_dir, tools_dir = Path(output), Path(v1_dir), Path(tools_dir)
    sealed = validate_sealed_protocols(output, v1_dir, tools_dir)
    frozen_sources = source_hashes()
    started = time.perf_counter()
    write_json(output / "status.json", {"status": "preparing_training_data"})
    legacy = read_source_rows(v1_dir / "train.jsonl")
    tool_source = read_source_rows(tools_dir / "train.jsonl")
    goal_rows = [r for r in tool_source if r.get("provenance", {}).get("source") != "actual_tool_observation"]
    tool_rows = [r for r in tool_source if r.get("provenance", {}).get("source") == "actual_tool_observation"]
    if not goal_rows or not tool_rows:
        raise ValueError("Both v2.1 replay domains are required")
    belief_rows, belief_data = generate_rows(TRAINING_PLAN["belief_training_cases"], "train", TRAINING_PLAN["belief_training_seed"])
    belief_valid_rows, belief_valid = generate_rows(TRAINING_PLAN["belief_validation_cases"], "validation", TRAINING_PLAN["belief_validation_seed"])
    rows = {"legacy": read_source_rows(v1_dir / "valid.jsonl"), "goals": read_source_rows(tools_dir / "valid-goals.jsonl"),
            "tools": read_source_rows(tools_dir / "valid-tools.jsonl"), "belief": belief_valid_rows}
    for filename, source in (("train-beliefs", belief_rows), ("valid-beliefs", belief_valid_rows)):
        write_rows(output / f"{filename}.jsonl", source)
    datasets = {"legacy": encode_rows(legacy), "goals": encode_rows(goal_rows), "tools": encode_rows(tool_rows), "belief": belief_data}
    valid = {name: encode_rows(rows[name]) for name in ("legacy", "goals", "tools")}
    valid["belief"] = belief_valid
    regrets, value_masks = belief_value_arrays(belief_rows, belief_data[2])
    v1, previous_tools = load_legacy(v1_dir), load_legacy(tools_dir)
    specs = fixture_specs(TRAINING_PLAN["actual_tool_validation_fixtures"], TRAINING_PLAN["actual_tool_validation_seed"])
    baseline_graph = rollout_metrics(v1, [Scenario.from_dict(r["scenario"]) for r in rows["legacy"]],
                                    rows["legacy"], count=TRAINING_PLAN["legacy_validation_rollouts"])
    baseline_goals = goal_sensitive_metrics(previous_tools, rows["goals"])
    # V4's observation-only adapter can wrap a copied v2.1 model for real tools.
    tool_adapter = UnifiedPolicy(PolicyConfig(**{**previous_tools.config.to_dict(), "input_dim": FEATURE_DIM}))
    initialize_from_v1(tool_adapter, previous_tools)
    baseline_tools = [run_episode(**spec, policy_name="neural", neural_policy=V4Policy(model=tool_adapter)) for spec in specs]
    baseline = {"legacy_completion": baseline_graph["methods"]["neural"]["success_rate"],
                "goal_change_accuracy": baseline_goals["paired_goal_active_accuracy"],
                "goal_change_pair_correctness": baseline_goals["causal_choice_pair_both_correct"],
                "actual_tool_completion": float(np.mean([r["success"] and r["done"] for r in baseline_tools])),
                "goal_change_pair_cases": baseline_goals["causal_choice_pairs"],
                "goal_change_pairs_both_correct": round(baseline_goals["causal_choice_pairs"] * baseline_goals["causal_choice_pair_both_correct"]),
                "v1_checkpoint_sha256": digest(v1_dir / "model.safetensors"),
                "v2_1_checkpoint_sha256": digest(tools_dir / "model.safetensors")}
    write_json(output / "validation_baselines.json", baseline)
    print("Strict validation baselines: " + json.dumps(baseline), flush=True)
    config = PolicyConfig(**{**v1.config.to_dict(), "input_dim": FEATURE_DIM})
    write_json(output / "config.json", config.to_dict())
    names = list(TRAINING_PLAN["training_mix_probabilities"])
    probabilities = [TRAINING_PLAN["training_mix_probabilities"][name] for name in names]
    best_global, histories, trial_reports = None, [], []
    for candidate in TRAINING_PLAN["candidates"]:
        trial = output / candidate["name"]
        trial.mkdir(exist_ok=False)
        write_json(trial / "candidate.json", candidate)
        training, distillation = {"belief": datasets["belief"]}, {}
        for name in ("legacy", "goals", "tools"):
            training[name], distillation[name] = add_correct_teacher_distillation(
                v1 if name == "legacy" else previous_tools, datasets[name], mixing=candidate["distillation_mixing"])
        rng = np.random.default_rng(candidate["seed"])
        mx.random.seed(candidate["seed"])
        model = UnifiedPolicy(config)
        initialize_from_v1(model, v1)
        model.train()
        mx.eval(model.parameters())
        optimizer = optim.AdamW(learning_rate=candidate["learning_rate"], weight_decay=0.01)
        gradient = nn.value_and_grad(model, reliability_loss)
        state = [model.state, optimizer.state, mx.random.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def update(x, valid_mask, eligible, target, regret, value_mask):
            loss, grads = gradient(model, x, valid_mask, eligible, target, regret, value_mask)
            grads, norm = optim.clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(model, grads)
            return loss, norm

        best_trial, history = None, []
        for iteration in range(1, TRAINING_PLAN["steps_per_candidate"] + 1):
            source_ids = rng.choice(len(names), TRAINING_PLAN["batch_size"], p=probabilities)
            chunks = [[] for _ in range(6)]
            for source_id, name in enumerate(names):
                count = int((source_ids == source_id).sum())
                if not count:
                    continue
                data = training[name]
                ids = rng.integers(0, len(data[0]), count)
                chunks[0].append((expand_belief if name == "belief" else expand_graph)(data[0][ids]))
                for field in range(1, 4):
                    chunks[field].append(data[field][ids])
                chunks[4].append(regrets[ids] if name == "belief" else np.zeros((count, 12), np.float32))
                chunks[5].append(value_masks[ids] if name == "belief" else np.zeros((count, 12), bool))
            batch = [mx.array(np.concatenate(parts)) for parts in chunks]
            warmup = min(1.0, iteration / 100)
            decay = 0.15 + 0.85 * 0.5 * (1 + math.cos(math.pi * iteration / TRAINING_PLAN["steps_per_candidate"]))
            optimizer.learning_rate = candidate["learning_rate"] * warmup * decay
            loss, norm = update(*batch)
            mx.eval(state, loss, norm)
            if iteration == 1 or iteration % 100 == 0:
                progress = {"status": "training", "candidate": candidate["name"], "step": iteration,
                            "steps": TRAINING_PLAN["steps_per_candidate"], "total_step_budget": TRAINING_PLAN["total_step_budget"],
                            "loss": float(loss.item()), "parameter_count": model.count_parameters(),
                            "elapsed_seconds": round(time.perf_counter() - started, 2)}
                write_json(output / "status.json", progress)
                print(json.dumps(progress), flush=True)
            if iteration % TRAINING_PLAN["validation_interval"] == 0:
                metrics = _validation(model, valid, rows, specs, baseline, step=iteration, candidate=candidate["name"])
                selection = tuple(metrics["selection"])
                history.append(metrics)
                histories.append(metrics)
                write_json(trial / "validation_history.json", history)
                print("Validation: " + json.dumps(metrics), flush=True)
                if best_trial is None or selection > best_trial:
                    best_trial = selection
                    model.save_weights(str(trial / "model.safetensors"))
                    write_json(trial / "best_validation.json", metrics)
                if best_global is None or selection > best_global:
                    best_global = selection
                    # Selection changes only inside the validation-only training run.
                    model.save_weights(str(output / "model.safetensors"))
                    write_json(output / "best_validation.json", metrics)
        trial_report = {**candidate, "distillation": distillation, "steps": TRAINING_PLAN["steps_per_candidate"],
                        "best_validation": json.loads((trial / "best_validation.json").read_text()),
                        "checkpoint_sha256": digest(trial / "model.safetensors"), "validation_history": history}
        write_json(trial / "report.json", trial_report)
        trial_reports.append(trial_report)
    selected = json.loads((output / "best_validation.json").read_text())
    source_files = [v1_dir / "train.jsonl", v1_dir / "valid.jsonl", tools_dir / "train.jsonl",
                    tools_dir / "valid-goals.jsonl", tools_dir / "valid-tools.jsonl"]
    report = {"model_name": "Winward v4 candidate", "schema_version": SCHEMA_VERSION, **sealed,
              "training_origin": "Continued training from our immutable v1, with training-only distillation from our v1 and v2.1. No third-party pretrained weights or Qwen outputs.",
              "architecture": TRAINING_PLAN["architecture"], "parameter_count": model.count_parameters(), "config": config.to_dict(),
              "training_plan": TRAINING_PLAN, "training_cases": sum(len(data[0]) for data in datasets.values()),
              "validation_cases": sum(len(data[0]) for data in valid.values()), "test_cases": 0, "test": None,
              "training_mix_cases": {name: len(data[0]) for name, data in datasets.items()},
              "sampling_probabilities": TRAINING_PLAN["training_mix_probabilities"], "candidate_reports": trial_reports,
              "steps_per_candidate": TRAINING_PLAN["steps_per_candidate"], "total_training_steps": TRAINING_PLAN["total_step_budget"],
              "selected_candidate": selected["candidate"], "selected_step": selected["step"],
              "batch_size": TRAINING_PLAN["batch_size"], "best_validation": selected, "validation_baselines": baseline,
              "validation_history": histories, "validation_retention_passed": selected["all_retention_gates_passed"],
              "checkpoint_selection": TRAINING_PLAN["selection"], "regression_tolerance": 0.0,
              "checkpoint_sha256": digest(output / "model.safetensors"), "checkpoint_frozen_before_test_generation": True,
              "reserved_final_audit_seeds": TRAINING_PLAN["reserved_final_audit_seeds"],
              "elapsed_seconds": round(time.perf_counter() - started, 2),
              "source_file_sha256": {str(path): digest(path) for path in source_files},
              "source_sha256": frozen_sources,
              "data_sha256": {name: digest(output / name) for name in ("train-beliefs.jsonl", "valid-beliefs.jsonl")},
              "limitations": "Structured supplied action/world models, at most four hypotheses and five actions of planning horizon. No language understanding, generated code, world-model learning, runtime search or teacher fallback. Permissions, applicability and verified stopping are supplied hard rules. Final audit required for promotion."}
    # Fail if a sealed protocol changed while training; do not freeze a false report.
    for key, filename in (("sealed_audit_protocol_sha256", "audit-protocol.json"),
                          ("sealed_training_protocol_sha256", "training-protocol.json")):
        if digest(output / filename) != sealed[key]:
            raise RuntimeError("A sealed protocol changed during training")
    if source_hashes() != frozen_sources:
        raise RuntimeError("Training source changed during training")
    write_json(output / "report.json", report)
    write_json(output / "status.json", {"status": "awaiting_blind_audit", "parameter_count": report["parameter_count"],
                                        "validation_retention_passed": report["validation_retention_passed"],
                                        "checkpoint_sha256": report["checkpoint_sha256"]})
    print(json.dumps({"checkpoint_frozen": report["checkpoint_sha256"], "selected_candidate": report["selected_candidate"],
                      "selected_step": report["selected_step"], "validation_retention_passed": report["validation_retention_passed"],
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/goalpolicy-v4")
    parser.add_argument("--v1-run", default="runs/goalpolicy-v1")
    parser.add_argument("--tools-run", default="runs/goalpolicy-v2-tools")
    parser.add_argument("--declare-plan", action="store_true")
    args = parser.parse_args()
    if args.declare_plan:
        print(json.dumps(seal_training_plan(args.output, args.v1_run, args.tools_run), indent=2))
    else:
        train(args.output, args.v1_run, args.tools_run)


if __name__ == "__main__":
    main()
