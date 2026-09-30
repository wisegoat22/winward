"""Train a fresh GoalPolicy checkpoint from synthetic planner demonstrations."""
import argparse
import hashlib
import json
import math
import time
from functools import partial
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .features import encode, target_for
from .model import GoalPolicy, PolicyConfig, loss_fn
from .simulator import make_scenario, search


def generate_split(count, split, seed, trajectories=False):
    rows, arrays, targets, valid, eligible, labels = [], [], [], [], [], []
    rng = np.random.default_rng(seed)
    for i in range(count):
        scenario = make_scenario(seed + i, split=split)
        token_weight = float(rng.choice([0.25, 1, 2, 4]))
        latency_weight = float(rng.choice([0.001, 0.01, 0.05, 0.2]))
        depth = 5 if i % 5 else int(rng.integers(1, 6))
        if trajectories and split == "train":
            # Diverse cost scales and goal changes prevent cheap/no-op tokens or
            # one familiar goal position from becoming a shortcut to the label.
            if i % 4 == 0:
                actions = tuple(replace(a, tokens=float(rng.integers(0, 1001)),
                                        latency_ms=float(rng.integers(0, 8001)))
                                for a in scenario.actions)
                scenario = replace(scenario, actions=actions)
            if i % 7 == 0:
                achieved_by_actions = [a.sets for a in scenario.actions if a.allowed and a.sets]
                if achieved_by_actions:
                    scenario = replace(scenario, goal=int(rng.choice(achieved_by_actions)))
            # Train on the same updated states and shrinking horizon encountered
            # during execution, including states created by combined actions.
            full = search(scenario, max_depth=5, token_weight=token_weight, latency_weight=latency_weight)
            if full["plan"] and i % 3 != 0:
                prefix = int(rng.integers(0, len(full["plan"])))
                scenario = replace(scenario, state=full["plan"][prefix]["state_before"])
                depth = 5 - prefix
            elif i % 3 == 0:
                for _ in range(int(rng.integers(0, 4))):
                    available = [a for a in scenario.actions if a.eligible(scenario.state)]
                    if not available:
                        break
                    action = available[int(rng.integers(0, len(available)))]
                    scenario = replace(scenario, state=action.apply(scenario.state))
        solution = search(scenario, max_depth=depth, token_weight=token_weight, latency_weight=latency_weight)
        x, v, e, candidates = encode(scenario, token_weight, latency_weight, depth)
        y = target_for(candidates, solution)
        assert np.all(e[y > 0]), "Teacher selected an ineligible action"
        rows.append({"scenario": scenario.to_dict(), "token_weight": token_weight,
                     "latency_weight": latency_weight, "max_depth": depth,
                     "target_ids": solution.get("optimal_action_ids") or [solution["chosen_action_id"]],
                     "outcome": solution["outcome"], "plan": solution["plan"]})
        arrays.append(x); targets.append(y); valid.append(v); eligible.append(e)
        labels.append(solution["chosen_action_id"])
        if (i + 1) % 5000 == 0:
            print(f"Generated {split}: {i+1}/{count}", flush=True)
    return rows, (np.stack(arrays), np.stack(valid), np.stack(eligible), np.stack(targets))


def evaluate(model, dataset, batch_size=128):
    x, valid, eligible, targets = dataset
    correct, meaningful_correct, meaningful_count, losses = 0, 0, 0, []
    for start in range(0, len(x), batch_size):
        batch = [mx.array(a[start:start + batch_size]) for a in dataset]
        logits = model(*batch[:3])
        mx.eval(logits)
        scores = np.array(logits)
        chosen = scores.argmax(axis=1)
        match = targets[start:start + batch_size][np.arange(len(chosen)), chosen] > 0
        meaningful = eligible[start:start + batch_size].sum(axis=1) > 1
        correct += int(match.sum())
        meaningful_correct += int(match[meaningful].sum())
        meaningful_count += int(meaningful.sum())
        log_probs = scores - np.logaddexp.reduce(scores, axis=-1, keepdims=True)
        losses.extend((-(targets[start:start + batch_size] * log_probs).sum(axis=1)).tolist())
    return {"accuracy": correct / len(x), "nontrivial_accuracy": meaningful_correct / max(meaningful_count, 1),
            "cases": len(x), "nontrivial_cases": meaningful_count, "loss": float(np.mean(losses))}


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/goalpolicy-v0")
    parser.add_argument("--train-cases", type=int, default=24000)
    parser.add_argument("--valid-cases", type=int, default=2400)
    parser.add_argument("--test-cases", type=int, default=2400)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--trajectories", action="store_true")
    parser.add_argument("--select-rollout", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "model.safetensors").exists():
        raise SystemExit("This run already has a checkpoint. Choose a new --output to preserve it.")
    started = time.time()
    write_json(out / "status.json", {"status": "generating_data", "seed": args.seed})
    datasets, dataset_rows = {}, {}
    for split, count, seed in [("train", args.train_cases, args.seed),
                                ("valid", args.valid_cases, args.seed + 1000000),
                                ("test", args.test_cases, args.seed + 2000000)]:
        rows, dataset = generate_split(count, split, seed, trajectories=args.trajectories)
        dataset_rows[split] = rows
        datasets[split] = dataset
        with (out / f"{split}.jsonl").open("w") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")
        np.savez_compressed(out / f"{split}.npz", **dict(zip(["x", "valid", "eligible", "targets"], dataset)))
        print(f"{split}: {count} examples ready", flush=True)
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    config = PolicyConfig(width=args.width, layers=args.layers)
    model = GoalPolicy(config)
    mx.eval(model.parameters())
    count = model.count_parameters()
    write_json(out / "config.json", config.to_dict())
    print(f"Randomly initialized {count:,} parameters. No pretrained weights.", flush=True)
    before = evaluate(model, datasets["valid"])
    print(f"Before training validation: {before}", flush=True)
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.01)
    value_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, valid, eligible, targets):
        loss, grads = value_and_grad(model, x, valid, eligible, targets)
        grads, grad_norm = optim.clip_grad_norm(grads, max_norm=1.0)
        optimizer.update(model, grads)
        return loss, grad_norm

    train = datasets["train"]
    best_selection, history = (-1, -1), []
    training_started = time.time()
    for i in range(1, args.steps + 1):
        ids = rng.integers(0, len(train[0]), size=args.batch_size)
        warmup = min(1.0, i / 100)
        decay = 0.15 + 0.85 * 0.5 * (1 + math.cos(math.pi * i / args.steps))
        optimizer.learning_rate = args.learning_rate * warmup * decay
        loss, grad_norm = step(*[mx.array(a[ids]) for a in train])
        mx.eval(state, loss, grad_norm)
        if i % 100 == 0 or i == 1:
            progress = {"status": "training", "step": i, "steps": args.steps,
                        "loss": float(loss.item()), "parameter_count": count,
                        "elapsed_seconds": round(time.time() - training_started, 2)}
            print(json.dumps(progress), flush=True)
            write_json(out / "status.json", progress)
        if i % 250 == 0 or i == args.steps:
            metrics = evaluate(model, datasets["valid"])
            if args.select_rollout:
                from .evaluate import rollout_metrics
                from .simulator import Scenario
                valid_rows = dataset_rows["valid"]
                valid_scenarios = [Scenario.from_dict(r["scenario"]) for r in valid_rows]
                rollout = rollout_metrics(model, valid_scenarios, valid_rows, count=150)
                metrics["validation_goal_success_rate"] = rollout["methods"]["neural"]["success_rate"]
            history.append({"step": i, **metrics})
            print(f"Validation {i}: {metrics}", flush=True)
            selection = (metrics.get("validation_goal_success_rate", metrics["accuracy"]), metrics["accuracy"])
            if selection > best_selection:
                best_selection = selection
                model.save_weights(str(out / "model.safetensors"))
                write_json(out / "best_validation.json", {"step": i, **metrics})
    model.load_weights(str(out / "model.safetensors"))
    test = evaluate(model, datasets["test"])
    checkpoint_sha = hashlib.sha256((out / "model.safetensors").read_bytes()).hexdigest()
    report = {
        "model_name": "GoalPolicy v1" if args.trajectories else "GoalPolicy v0", "training_origin": "Random initialization; no pretrained weights or Qwen outputs",
        "parameter_count": count, "config": config.to_dict(), "seed": args.seed,
        "training_cases": args.train_cases, "validation_cases": args.valid_cases, "test_cases": args.test_cases,
        "steps": args.steps, "batch_size": args.batch_size, "before_training": before,
        "trajectory_augmentation": args.trajectories,
        "checkpoint_selection": "Validation goal completion, then validation next-action accuracy" if args.select_rollout else "Validation next-action accuracy",
        "validation_history": history, "best_validation": json.loads((out / "best_validation.json").read_text()),
        "test": test, "elapsed_seconds": round(time.time() - started, 2),
        "checkpoint_sha256": checkpoint_sha,
        "teacher": "Deterministic search through declared simulator transitions up to 5 actions; goal success first, then weighted token and latency cost, then steps.",
        "limitations": "Structured simulated software tasks only. No natural-language understanding, real tool execution, or claim of general-world planning. Permission/precondition/stop masks are deterministic rules, not learned capabilities. Synthetic evaluation is not real-agent validation.",
    }
    write_json(out / "report.json", report)
    write_json(out / "status.json", {"status": "ready", "parameter_count": count, "test": test})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
