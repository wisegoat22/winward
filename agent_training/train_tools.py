"""Train a scratch tool-recovery candidate from actual observed fixture traces.

This experiment mixes v2 synthetic goal decisions with training-only Python
fixture episodes. The teacher selects tools from observed state; it never reads
hidden candidate correctness. Validation uses independent fixture seeds. This
entry point never creates or evaluates a test split, and never changes v1/v2.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from agent_lab.runner import evidence_first
from agent_lab.sandbox import SandboxTask
from .curriculum import encode_rows, generate_rows, provenance_summary
from .features import DEFER, STOP, encode
from .simulator import Scenario


TRAIN_FIXTURE_KINDS = ("boundary", "rounding")


def observed_targets(scenario, teacher_action):
    """Accept equivalent unverified patches, without consulting their code.

    The evidence-first teacher breaks patch ties by id. Identical structural
    alternatives cannot support a learnable id preference after permutation, so
    both minimum-cost untried patches receive equal target mass.
    """
    if not teacher_action.startswith("apply_candidate_"):
        return [teacher_action]
    candidates = [a for a in scenario.actions if a.id.startswith("apply_candidate_") and a.eligible(scenario.state)]
    best = min(a.cost() for a in candidates)
    return [a.id for a in candidates if a.cost() == best]


def fixture_specs(count, seed):
    if count < 1 or seed < 0:
        raise ValueError("Count must be positive and seed nonnegative")
    return [{"seed": seed + i, "kind": TRAIN_FIXTURE_KINDS[i % 2],
             "changed_goal": bool((i // 2) % 2), "uncertain": bool((i // 4) % 2)} for i in range(count)]


def collect_episodes(count, seed, split):
    """Record the observation before each real action and its actual result."""
    rows, summaries = [], []
    for spec in fixture_specs(count, seed):
        with SandboxTask(**spec) as task:
            for step in range(16):
                observation = task.observe()
                action_id = evidence_first(observation)
                targets = observed_targets(observation, action_id)
                # Teacher selection is complete before the action is executed.
                event = task.step(action_id)
                rows.append({"scenario": observation.to_dict(), "token_weight": 1.0, "latency_weight": 0.01,
                             "max_depth": min(5, 16-step), "target_ids": targets,
                             "outcome": "stop" if action_id == STOP else "needs_clarification" if action_id == DEFER else "plan",
                             "plan": [], "teacher_action_id": action_id,
                             "observed_result": event["result"], "observed_state_after": event["state_after"],
                             "forecast_matched": event["forecast_matched"],
                             "provenance": {"source": "actual_tool_observation", "split": split,
                                            "episode_seed": spec["seed"], "fixture_kind": spec["kind"], "step": step,
                                            "goal_changed_after_action": event["goal_changed"],
                                            "label_scope": "Observed-state evidence-first scheduling; candidate correctness is not provided."}})
                if task.done:
                    break
            summaries.append(task.summary())
    return rows, summaries


def permute_observation(row, seed):
    """Rename/reorder fact bits and action ids, preserving exact target relations.

    No text, fixture kind, answer, outcome, or provenance is encoded. The source
    observation remains in the recorded episode; augmented rows are separate.
    """
    rng = random.Random(seed)
    original = Scenario.from_dict(row["scenario"])
    permutation = list(range(len(original.facts)))
    rng.shuffle(permutation)
    def remap(mask):
        return sum(1 << permutation[i] for i in range(len(permutation)) if mask & (1 << i))
    facts = [""] * len(original.facts)
    for index, name in enumerate(original.facts):
        facts[permutation[index]] = name
    actions = list(original.actions)
    rng.shuffle(actions)
    names = {a.id: f"option_{i}" for i, a in enumerate(actions)}
    actions = tuple(replace(a, id=names[a.id], requires=remap(a.requires), forbids=remap(a.forbids),
                            sets=remap(a.sets), clears=remap(a.clears)) for a in actions)
    scenario = replace(original, id=f"{original.id}-permutation-{seed}", facts=tuple(facts),
                       state=remap(original.state), goal=remap(original.goal), actions=actions, context={})
    return {"scenario": scenario.to_dict(), "token_weight": row["token_weight"],
            "latency_weight": row["latency_weight"], "max_depth": row["max_depth"],
            "target_ids": [names.get(action_id, action_id) for action_id in row["target_ids"]],
            "outcome": row["outcome"], "plan": [],
            "teacher_action_id": names.get(row["teacher_action_id"], row["teacher_action_id"]),
            "provenance": {**row["provenance"], "augmentation_seed": seed,
                           "source_observation_id": original.id, "fact_permutation": permutation,
                           "action_id_mapping": names}}


def augment_rows(rows, count, seed):
    if not rows or count < 1:
        raise ValueError("Augmentation needs observed rows and a positive count")
    rng = np.random.default_rng(seed)
    # Recovering after failed checks and reacting to goal revisions are retained
    # as normal observed states; no simulator oracle supplies their targets.
    return [permute_observation(rows[int(rng.integers(0, len(rows)))], seed + index) for index in range(count)]


def _write_rows(path, rows):
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def main():
    # Keep dataset utilities importable without MLX for semantic/CI tests.
    import math
    from functools import partial
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from agent_lab.runner import run_episode
    from .model import GoalPolicy, PolicyConfig, loss_fn
    from .train import evaluate, write_json
    from .train_v2 import goal_sensitive_metrics

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/goalpolicy-v2-tools")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--train-cases", type=int, default=64000)
    parser.add_argument("--valid-cases", type=int, default=2400)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=210000000)
    parser.add_argument("--validation-seed", type=int, default=220000000)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if min(args.episodes, args.validation_episodes, args.train_cases, args.valid_cases, args.steps, args.batch_size) < 1:
        raise SystemExit("Counts must be positive")
    if args.seed == args.validation_seed:
        raise SystemExit("Training and validation seeds must differ")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("config.json", "model.safetensors", "report.json")):
        raise SystemExit("Choose a new output directory; existing experiments are never overwritten")
    started = time.perf_counter()
    write_json(out / "status.json", {"status": "collecting_real_tool_observations"})
    raw_train, train_episodes = collect_episodes(args.episodes, args.seed, "train")
    raw_valid, valid_episodes = collect_episodes(args.validation_episodes, args.validation_seed, "validation")
    for name, rows in (("tool-train-observations", raw_train), ("tool-validation-observations", raw_valid)):
        _write_rows(out / f"{name}.jsonl", rows)
    write_json(out / "tool-episode-summaries.json", {"train": train_episodes, "validation": valid_episodes})
    print(f"Recorded {len(raw_train)} actual training tool decisions from {args.episodes} episodes; {len(raw_valid)} independent validation decisions", flush=True)
    if not all(item["success"] and item["done"] for item in train_episodes + valid_episodes):
        raise RuntimeError("Observed-state teacher did not complete a training/validation episode; inspect collection before training")
    curriculum_rows = generate_rows(args.train_cases // 2, "train", args.seed + 1_000_000, progress_log=True)
    tool_rows = augment_rows(raw_train, args.train_cases - len(curriculum_rows), args.seed + 2_000_000)
    train_rows = curriculum_rows + tool_rows
    valid_goal_rows = generate_rows(args.valid_cases, "validation", args.validation_seed + 1_000_000)
    valid_tool_rows = augment_rows(raw_valid, args.valid_cases, args.validation_seed + 2_000_000)
    # Add original observed encodings only to validation: permutation-only
    # training must learn relations well enough to transfer to fixture positions.
    valid_tool_rows.extend(raw_valid)
    _write_rows(out / "train.jsonl", train_rows)
    _write_rows(out / "valid-goals.jsonl", valid_goal_rows)
    _write_rows(out / "valid-tools.jsonl", valid_tool_rows)
    datasets = {"train": encode_rows(train_rows), "goals": encode_rows(valid_goal_rows), "tools": encode_rows(valid_tool_rows)}
    config = PolicyConfig()
    write_json(out / "config.json", config.to_dict())
    mx.random.seed(args.seed)
    model = GoalPolicy(config)
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=0.0003, weight_decay=0.01)
    grad_fn = nn.value_and_grad(model, loss_fn)
    compiled_state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=compiled_state, outputs=compiled_state)
    def update(x, valid, eligible, targets):
        loss, grads = grad_fn(model, x, valid, eligible, targets)
        grads, norm = optim.clip_grad_norm(grads, max_norm=1.0)
        optimizer.update(model, grads)
        return loss, norm

    class Candidate:
        def choose(self, observation, remaining=5):
            x, valid, eligible, candidates = encode(observation, max_depth=remaining)
            logits = model(mx.array(x[None]), mx.array(valid[None]), mx.array(eligible[None]))
            mx.eval(logits)
            return {"action_id": candidates[int(np.array(logits)[0].argmax())]["id"]}

    before = {name: evaluate(model, dataset, args.batch_size) for name, dataset in datasets.items() if name != "train"}
    rng = np.random.default_rng(args.seed)
    train = datasets["train"]
    best_selection, history = (-1.0, -1.0, -1.0), []
    for iteration in range(1, args.steps + 1):
        ids = rng.integers(0, len(train_rows), size=args.batch_size)
        warmup = min(1.0, iteration / 100)
        decay = 0.15 + 0.85 * 0.5 * (1 + math.cos(math.pi * iteration / args.steps))
        optimizer.learning_rate = 0.0003 * warmup * decay
        loss, norm = update(*[mx.array(array[ids]) for array in train])
        mx.eval(compiled_state, loss, norm)
        if iteration == 1 or iteration % 100 == 0:
            progress = {"status": "training", "step": iteration, "steps": args.steps,
                        "loss": float(loss.item()), "parameter_count": model.count_parameters(),
                        "elapsed_seconds": round(time.perf_counter() - started, 2)}
            write_json(out / "status.json", progress)
            print(json.dumps(progress), flush=True)
        if iteration % 500 == 0 or iteration == args.steps:
            goal_metrics = evaluate(model, datasets["goals"], args.batch_size)
            goal_metrics.update(goal_sensitive_metrics(model, valid_goal_rows, args.batch_size))
            tool_metrics = evaluate(model, datasets["tools"], args.batch_size)
            actual_runs = [run_episode(**spec, policy_name="neural", neural_policy=Candidate())
                           for spec in fixture_specs(16, args.validation_seed + 100_000)]
            tool_success = sum(r["success"] and r["done"] for r in actual_runs) / len(actual_runs)
            # Fixed before training: actual completion first; among equal
            # completion rates protect the weaker of goal and tool competence.
            balance = min(goal_metrics["paired_goal_active_accuracy"], tool_metrics["nontrivial_accuracy"])
            selection = (tool_success, balance, goal_metrics["accuracy"])
            metrics = {"step": iteration, "goal_decisions": goal_metrics, "observed_tool_decisions": tool_metrics,
                       "actual_validation_episodes": len(actual_runs), "actual_validation_success_rate": tool_success,
                       "actual_validation_mean_actions": float(np.mean([r["steps"] for r in actual_runs])),
                       "selection": list(selection)}
            history.append(metrics)
            print(f"Validation: {json.dumps(metrics)}", flush=True)
            if selection > best_selection:
                best_selection = selection
                model.save_weights(str(out / "model.safetensors"))
                write_json(out / "best_validation.json", metrics)
                write_json(out / "best-validation-tool-summaries.json", [{k: v for k, v in r.items() if k != "trace"} for r in actual_runs])
    digest = hashlib.sha256((out / "model.safetensors").read_bytes()).hexdigest()
    data_files = ("tool-train-observations.jsonl", "tool-validation-observations.jsonl", "train.jsonl", "valid-goals.jsonl", "valid-tools.jsonl")
    report = {"model_name": "Winward v2.1 tools candidate", "training_origin": "Random initialization; no pretrained, v1, or v2 weights loaded",
              "parameter_count": model.count_parameters(), "config": config.to_dict(), "seed": args.seed,
              "training_cases": len(train_rows), "validation_cases": len(valid_goal_rows) + len(valid_tool_rows),
              "test_cases": 0, "test": None, "steps": args.steps, "batch_size": args.batch_size,
              "training_fixture_kinds": list(TRAIN_FIXTURE_KINDS), "training_fixture_episodes": args.episodes,
              "training_observed_tool_decisions": len(raw_train), "validation_fixture_seed": args.validation_seed,
              "validation_fixture_episodes": args.validation_episodes, "validation_rollout_seed": args.validation_seed + 100_000,
              "before_training": before, "best_validation": json.loads((out / "best_validation.json").read_text()),
              "validation_history": history,
              "checkpoint_selection": "Validation-only lexicographic: actual16-episode completion, minimum of paired-goal accuracy and nontrivial tool-decision accuracy, then overall goal accuracy.",
              "checkpoint_sha256": digest, "checkpoint_frozen_before_test_generation": True,
              "test_seed_reserved": 240000000, "real_task_test_seed_reserved": 230000000,
              "elapsed_seconds": round(time.perf_counter()-started, 2),
              "teacher": "Tool samples: observed-state evidence-first rule, with tied unverified patches equally acceptable; actual tool outcomes recorded after selection. Synthetic samples: bounded-search goal curriculum.",
              "curriculum_mix": {"goal_curriculum_cases": len(curriculum_rows), "augmented_tool_cases": len(tool_rows)},
              "data_sha256": {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in data_files},
              "limitations": "Only boundary/rounding training fixtures; validation shares these families with independent seeds. Test is not generated here. Model schedules supplied patches from numeric state, without reading code. Success/permissions are enforced by supplied environment rules. Learned confidence is not calibrated. No claim that the neural policy beats evidence-first rules."}
    write_json(out / "report.json", report)
    write_json(out / "status.json", {"status": "awaiting_blind_audit", "parameter_count": model.count_parameters(), "checkpoint_sha256": digest})
    print(json.dumps({"frozen_checkpoint": digest, "best_validation": report["best_validation"],
                      "training_observations": len(raw_train), "elapsed_seconds": report["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
