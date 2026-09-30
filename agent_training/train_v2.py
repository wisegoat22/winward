"""Train Winward v2 from scratch; freeze before any test data is generated.

The separate evaluate_v2 command performs the one-time blind audit. No existing
v0/v1 source, data, report, or weights are modified by this entry point.
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

from .curriculum import CURRICULUM_VERSION, encode_rows, generate_rows, provenance_summary
from .evaluate import predict, rollout_metrics
from .model import GoalPolicy, PolicyConfig, loss_fn
from .simulator import Scenario
from .train import evaluate, write_json


def goal_sensitive_metrics(model, rows, batch_size=128):
    observations = [(Scenario.from_dict(r["scenario"]), r["token_weight"], r["latency_weight"], r["max_depth"]) for r in rows]
    predicted, _, _ = predict(model, observations, batch_size)
    correct = [p in r["target_ids"] for p, r in zip(predicted, rows)]
    paired_active = [i for i, r in enumerate(rows) if r["provenance"]["source"] == "paired_goal" and r["outcome"] == "plan"]
    by_pair = {}
    for i, row in enumerate(rows):
        if row["provenance"].get("pair_id"):
            by_pair.setdefault(row["provenance"]["pair_id"], []).append(i)
    changed = [ids for ids in by_pair.values() if len(ids) == 2 and all(rows[i]["outcome"] == "plan" for i in ids)
               and not set(rows[ids[0]]["target_ids"]) & set(rows[ids[1]]["target_ids"])]
    return {"paired_goal_active_cases": len(paired_active),
            "paired_goal_active_accuracy": float(np.mean([correct[i] for i in paired_active])) if paired_active else 0.0,
            "causal_choice_pairs": len(changed),
            "causal_choice_pair_both_correct": float(np.mean([all(correct[i] for i in ids) for ids in changed])) if changed else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/goalpolicy-v2")
    parser.add_argument("--train-cases", type=int, default=64000)
    parser.add_argument("--valid-cases", type=int, default=2400)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    args = parser.parse_args()
    if min(args.train_cases, args.valid_cases, args.steps, args.batch_size) < 1 or not 0 < args.learning_rate < 1:
        raise SystemExit("Counts and learning rate must be positive.")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / filename).exists() for filename in ("model.safetensors", "report.json", "config.json")):
        raise SystemExit("Choose a new output directory; completed or interrupted runs are never overwritten.")
    started = time.perf_counter()
    write_json(out / "status.json", {"status": "generating_data", "seed": args.seed})
    datasets, all_rows, manifests = {}, {}, {}
    for split, count, seed in (("train", args.train_cases, args.seed), ("validation", args.valid_cases, args.seed + 20_000_000)):
        rows = generate_rows(count, split, seed, progress_log=True)
        dataset = encode_rows(rows)
        filename = "train" if split == "train" else "valid"
        with (out / f"{filename}.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        np.savez_compressed(out / f"{filename}.npz", **dict(zip(("x", "valid", "eligible", "targets"), dataset)))
        manifests[split] = {"seed": seed, **provenance_summary(rows),
                            "jsonl_sha256": hashlib.sha256((out / f"{filename}.jsonl").read_bytes()).hexdigest()}
        datasets[split], all_rows[split] = dataset, rows
        print(f"{split}: {count} examples ready", flush=True)
    write_json(out / "data_provenance.json", manifests)
    config = PolicyConfig()
    write_json(out / "config.json", config.to_dict())
    mx.random.seed(args.seed)
    model = GoalPolicy(config)
    mx.eval(model.parameters())
    count = model.count_parameters()
    before = evaluate(model, datasets["validation"], args.batch_size)
    print(f"Randomly initialized {count:,} parameters; initial validation {before}", flush=True)
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.01)
    value_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, valid, eligible, targets):
        loss, grads = value_and_grad(model, x, valid, eligible, targets)
        grads, norm = optim.clip_grad_norm(grads, max_norm=1.0)
        optimizer.update(model, grads)
        return loss, norm

    rng = np.random.default_rng(args.seed)
    train = datasets["train"]
    validation_rows = all_rows["validation"]
    validation_scenarios = [Scenario.from_dict(r["scenario"]) for r in validation_rows]
    best, history = (-1.0, -1.0), []
    training_started = time.perf_counter()
    for iteration in range(1, args.steps + 1):
        ids = rng.integers(0, len(train[0]), size=args.batch_size)
        warmup = min(1.0, iteration / 100)
        decay = 0.15 + 0.85 * 0.5 * (1 + math.cos(math.pi * iteration / args.steps))
        optimizer.learning_rate = args.learning_rate * warmup * decay
        loss, norm = step(*[mx.array(array[ids]) for array in train])
        mx.eval(state, loss, norm)
        if iteration == 1 or iteration % 100 == 0:
            progress = {"status": "training", "step": iteration, "steps": args.steps, "loss": float(loss.item()),
                        "parameter_count": count, "elapsed_seconds": round(time.perf_counter() - training_started, 2)}
            write_json(out / "status.json", progress)
            print(json.dumps(progress), flush=True)
        if iteration % 250 == 0 or iteration == args.steps:
            metrics = evaluate(model, datasets["validation"], args.batch_size)
            metrics.update(goal_sensitive_metrics(model, validation_rows, args.batch_size))
            rollout = rollout_metrics(model, validation_scenarios, validation_rows, count=200, batch_size=args.batch_size)
            metrics["validation_goal_success_rate"] = rollout["methods"]["neural"]["success_rate"]
            # Fixed before training: balance goal-sensitive active decisions,
            # actual bounded completion, and next-action accuracy.
            score = (0.45 * metrics["paired_goal_active_accuracy"] +
                     0.35 * metrics["validation_goal_success_rate"] + 0.20 * metrics["accuracy"])
            metrics["selection_score"] = score
            history.append({"step": iteration, **metrics})
            print(f"Validation {iteration}: {metrics}", flush=True)
            selection = (score, metrics["causal_choice_pair_both_correct"])
            if selection > best:
                best = selection
                model.save_weights(str(out / "model.safetensors"))
                write_json(out / "best_validation.json", {"step": iteration, **metrics})
    checkpoint_sha = hashlib.sha256((out / "model.safetensors").read_bytes()).hexdigest()
    report = {"model_name": "Winward v2", "training_origin": "Random initialization; no pretrained weights or Qwen outputs",
              "curriculum_version": CURRICULUM_VERSION, "parameter_count": count, "config": config.to_dict(), "seed": args.seed,
              "training_cases": args.train_cases, "validation_cases": args.valid_cases, "test_cases": 0,
              "steps": args.steps, "batch_size": args.batch_size, "before_training": before,
              "checkpoint_selection": "Validation-only: 45% paired-goal active accuracy + 35% 200-task goal completion + 20% next-action accuracy; paired causal correctness breaks ties.",
              "validation_history": history, "best_validation": json.loads((out / "best_validation.json").read_text()),
              "test": None, "elapsed_seconds": round(time.perf_counter() - started, 2), "checkpoint_sha256": checkpoint_sha,
              "checkpoint_frozen_before_test_generation": True, "test_seed_reserved": 110000000,
              "teacher": "Exact bounded search: satisfy the stated goal first, then weighted token/latency cost, then fewer actions; maximum five actions.",
              "limitations": "Structured deterministic simulation only. Information prerequisites are explicit facts, not a learned uncertainty model. No language understanding. Goal, permission, precondition, and stop rules remain supplied constraints. Real execution results must be measured separately."}
    write_json(out / "report.json", report)
    write_json(out / "status.json", {"status": "awaiting_blind_audit", "parameter_count": count, "checkpoint_sha256": checkpoint_sha})
    print(json.dumps({"checkpoint_frozen": checkpoint_sha, "best_validation": report["best_validation"],
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
