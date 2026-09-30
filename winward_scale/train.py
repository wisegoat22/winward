"""Measured, resumable full-parameter training of our own byte decision decoder.

No pretrained parameters, adapters, external inference, or silent truncation.
Capacity probes establish feasibility, not useful decision-making capability.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from .data import ByteTokenizer, make_example
from .model import Decoder, ModelConfig

GIB = 1024 ** 3
ROOT = Path(__file__).resolve().parents[1]


class FP32Adafactor(optim.Adafactor):
    """Factored FP32 moments and update math; BF16 stored trainable weights.

No full-size momentum or FP32 master-weight copy. Storage rounding remains a
limitation; this optimizer is benchmarked rather than assumed to match FP32.
"""
    def init_single(self, parameter, state):
        if parameter.ndim >= 2:
            state["exp_avg_sq_row"] = mx.zeros(parameter.shape[:-1], dtype=mx.float32)
            state["exp_avg_sq_col"] = mx.zeros(parameter.shape[:-2] + parameter.shape[-1:], dtype=mx.float32)
        else:
            state["exp_avg_sq"] = mx.zeros(parameter.shape, dtype=mx.float32)

    def apply_single(self, gradient, parameter, state):
        return super().apply_single(gradient.astype(mx.float32), parameter.astype(mx.float32), state).astype(parameter.dtype)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(16 * 1024 ** 2):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def source_hashes():
    names = ["winward_scale/data.py", "winward_scale/model.py", "winward_scale/train.py", "agent_training/simulator.py"]
    return {name: digest(ROOT / name) for name in names}


def resolve_checkpoint(path):
    path = Path(path)
    if (path / "latest.json").is_file():
        return path / json.loads((path / "latest.json").read_text())["checkpoint"]
    if (path / "model.safetensors").is_file():
        return path
    raise ValueError("No complete local checkpoint found at the supplied path.")


def encode_corpus(count, split, seed, curriculum, limit):
    tokenizer = ByteTokenizer()
    rows = []
    for index in range(count):
        example = make_example(index, split=split, seed=seed, curriculum=curriculum)
        ids, mask = tokenizer.encode_example(example)
        if len(ids) - 1 > limit:
            raise ValueError(f"{split} example {index} needs {len(ids) - 1} tokens, beyond {limit}; nothing was truncated.")
        rows.append((np.asarray(ids, dtype=np.int32), np.asarray(mask, dtype=np.float32), example))
    return rows


def batch_arrays(rows, maximum):
    length = min(maximum, ((max(len(row[0]) - 1 for row in rows) + 63) // 64) * 64)
    tokens = np.full((len(rows), length + 1), ByteTokenizer.pad_id, dtype=np.int32)
    masks = np.zeros((len(rows), length + 1), dtype=np.float32)
    for i, (ids, mask, _) in enumerate(rows):
        tokens[i, :len(ids)] = ids
        masks[i, :len(mask)] = mask
    return mx.array(tokens[:, :-1]), mx.array(tokens[:, 1:]), mx.array(masks[:, 1:])


def decision_loss(model, inputs, targets, mask):
    logits = model(inputs).astype(mx.float32)
    losses = nn.losses.cross_entropy(logits, targets, reduction="none")
    return (losses * mask).sum() / mx.maximum(mask.sum(), 1)


def validate(model, rows, maximum, batch_size=4):
    model.eval()
    correct, count, total_loss = 0, 0, 0.0
    predictions = {}
    started = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        inputs, targets, mask = batch_arrays(batch, maximum)
        logits = model(inputs).astype(mx.float32)
        losses = nn.losses.cross_entropy(logits, targets, reduction="none")
        loss = (losses * mask).sum() / mx.maximum(mask.sum(), 1)
        # First completion byte: all 260 outputs compete. No answer/action mask.
        positions = mx.argmax(mask > 0, axis=1)
        predicted = mx.argmax(logits[mx.arange(len(batch)), positions], axis=-1)
        expected = targets[mx.arange(len(batch)), positions]
        matched = (predicted == expected).sum()
        mx.eval(loss, predicted, matched)
        if not np.isfinite(float(loss.item())):
            raise FloatingPointError("Nonfinite validation loss; the checkpoint is not valid.")
        correct += int(matched.item())
        total_loss += float(loss.item()) * len(batch)
        count += len(batch)
        for value in predicted.tolist():
            predictions[str(value)] = predictions.get(str(value), 0) + 1
        del logits, inputs, targets, mask, losses
    model.train()
    return {"examples": count, "next_action_correct": correct,
            "next_action_accuracy": correct / count, "completion_loss": total_loss / count,
            "prediction_token_counts": predictions, "wall_seconds": time.perf_counter() - started,
            "scope": "Validation, used for development; exact canonical first-byte agreement, unconstrained output. Not a final audit or rollout score."}


def parameter_samples(model):
    samples = {}
    for name, value in tree_flatten(model.trainable_parameters()):
        indexes = mx.array(np.linspace(0, value.size - 1, min(value.size, 32), dtype=np.int32))
        samples[name] = value.reshape(-1)[indexes].astype(mx.float32)
    mx.eval(samples)
    return samples


def training_fit(model, rows, maximum):
    result = validate(model, rows[:32], maximum)
    result["scope"] = "Fit on the first32 training examples, not held-out generalization or an audit."
    return result


def assert_finite_tree(tree, description):
    checks = [mx.all(mx.isfinite(value)) for _, value in tree_flatten(tree)]
    mx.eval(checks)
    if not all(bool(value.item()) for value in checks):
        raise FloatingPointError(f"Nonfinite {description}; no invalid checkpoint will be saved.")


def save_checkpoint(run, model, optimizer, step, report):
    assert_finite_tree(model.parameters(), "model weights")
    assert_finite_tree(optimizer.state, "optimizer state")
    relative = f"checkpoints/step-{step:08d}"
    destination = run / relative
    destination.mkdir(parents=True, exist_ok=False)
    model.save_weights(str(destination / "model.safetensors"))
    mx.save_safetensors(str(destination / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
    metadata = {"step": step, "total_optimizer_updates": int(optimizer.step.item()),
                "config": asdict(model.config), "model_sha256": digest(destination / "model.safetensors"),
                "optimizer_sha256": digest(destination / "optimizer.safetensors"),
                "parameters": model.config.parameter_count(), "report": report}
    write_json(destination / "checkpoint.json", metadata)
    write_json(run / "latest.json", {"checkpoint": relative, "step": step})
    return metadata


def train(args):
    run = Path(args.output)
    if args.steps < 1 or args.batch_size < 1 or args.train_cases < 1 or args.valid_cases < 1:
        raise ValueError("Steps, batch size, and dataset counts must be positive.")
    if min(args.eval_every, args.save_every, args.log_every, args.max_seq_len) < 1:
        raise ValueError("Intervals and sequence limits must be positive.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive.")
    if not 1 <= args.memory_gib <= 28:
        raise ValueError("This local experiment limits its requested GPU allocation guideline to 1–28 GiB.")
    # MLX describes this as a guideline, not a hard operating-system RAM cap.
    mx.set_memory_limit(int(args.memory_gib * GIB))
    mx.set_cache_limit(256 * 1024 ** 2)
    config = ModelConfig.preset(args.preset)
    estimated_base_gib = 3 * config.parameter_count() * 2 / GIB
    if estimated_base_gib + 2 > args.memory_gib:
        raise ValueError(f"Conservative weights/gradients/update estimate is {estimated_base_gib:.2f} GiB plus activation overhead; choose a sufficient declared memory guideline before allocating.")
    mx.reset_peak_memory()
    parent = resolve_checkpoint(args.init_from) if args.init_from else None
    if args.resume:
        protocol = json.loads((run / "protocol.json").read_text())
        if protocol["source_sha256"] != source_hashes():
            raise ValueError("Training source changed. Start a new declared run with --init-from instead of rewriting provenance.")
        if protocol["arguments"] != {k: v for k, v in vars(args).items() if k != "resume"}:
            raise ValueError("Resume arguments must match the declared run.")
        parent = resolve_checkpoint(run)
    else:
        run.mkdir(parents=True, exist_ok=False)
        protocol = {"version": "winward-byte-scale-1", "config": asdict(config), "parameters": config.parameter_count(),
                    "arguments": {k: v for k, v in vars(args).items() if k != "resume"},
                    "source_sha256": source_hashes(), "initialization": "own local checkpoint" if parent else "random weights; no pretrained parameters",
                    "optimizer": "Adafactor; FP32 factored moments/update math, BF16 stored parameters, no master weights or first moment",
                    "parent_sha256": digest(parent / "model.safetensors") if parent else None,
                    "hardware": mx.device_info(), "mlx_version": mx.__version__,
                    "purpose": args.purpose, "promotion": "No automatic quality promotion from size, loss reduction, or validation."}
        write_json(run / "protocol.json", protocol)
        for name, expected in protocol["source_sha256"].items():
            if digest(ROOT / name) != expected:
                raise ValueError("A training dependency changed while the protocol was being written.")
            destination = run / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
    train_rows = encode_corpus(args.train_cases, "train", args.seed, args.curriculum, args.max_seq_len)
    valid_rows = encode_corpus(args.valid_cases, "validation", args.seed, args.curriculum, args.max_seq_len)
    label_counts = {}
    for _, _, example in train_rows:
        label_counts[example.completion] = label_counts.get(example.completion, 0) + 1
    most_common = max(label_counts, key=label_counts.get)
    majority_accuracy = sum(row[2].completion == most_common for row in valid_rows) / len(valid_rows)
    mx.random.seed(args.seed)
    started = time.perf_counter()
    model = Decoder(config)
    actual = sum(value.size for _, value in tree_flatten(model.trainable_parameters()))
    if actual != config.parameter_count():
        raise ValueError("Materialized parameter count differs from the declared architecture.")
    optimizer = FP32Adafactor(learning_rate=args.learning_rate, relative_step=False, scale_parameter=False,
                             beta_1=None, weight_decay=0.0)
    local_start = 0
    parent_updates = 0
    previous_report = {}
    if parent:
        info = json.loads((parent / "checkpoint.json").read_text())
        if info["config"] != asdict(config) or digest(parent / "model.safetensors") != info["model_sha256"]:
            raise ValueError("Parent checkpoint has the wrong architecture or checksum.")
        if digest(parent / "optimizer.safetensors") != info["optimizer_sha256"]:
            raise ValueError("Parent optimizer checkpoint checksum failed.")
        model.load_weights(str(parent / "model.safetensors"))
        optimizer.state = tree_unflatten(list(mx.load(str(parent / "optimizer.safetensors")).items()))
        # A newly declared continuation can change learning rate while retaining
        # moments. Do not silently restore the parent's old rate over this plan.
        optimizer.learning_rate = args.learning_rate
        if args.resume:
            local_start = info["step"]
            previous_report = info["report"]
            if local_start >= args.steps:
                raise ValueError("This run is already complete. Continue with a new declared run using --init-from.")
        parent_updates = info["total_optimizer_updates"] - local_start
    mx.eval(model.parameters(), optimizer.state)
    model.train()
    samples_before = parameter_samples(model)
    loss_grad = nn.value_and_grad(model, decision_loss)
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    history = list(previous_report.get("validation_history", []))
    training_seconds = previous_report.get("training_seconds", 0.0)
    total_tokens = previous_report.get("padded_training_tokens", 0)
    previous_wall = previous_report.get("wall_seconds", 0.0)
    last_saved = local_start if args.resume else -1
    initial_validation = validate(model, valid_rows, args.max_seq_len)
    report = {**previous_report, "status": "training", "parameters": actual, "purpose": args.purpose,
              "majority_label_baseline": {"label": most_common, "validation_accuracy": majority_accuracy},
              "initialization_seconds": time.perf_counter() - started,
              "initial_validation": previous_report.get("initial_validation", initial_validation),
              "initial_training_fit": previous_report.get("initial_training_fit") or training_fit(model, train_rows, args.max_seq_len),
              "parent_updates": parent_updates, "validation_history": history,
              "limitations": ["Synthetic software workflow text, not general language pretraining.",
                              "Parameter count is not a capability score.",
                              "Stored BF16 weights round optimizer updates; no FP32 master weights.",
                              "MLX memory limit is a guideline, not a hard RAM cap."]}
    if args.resume:
        report["resume_validation"] = initial_validation
    write_json(run / "progress.json", report)
    print(json.dumps({"event": "initialized", "parameters": actual, "initial_validation": report["initial_validation"]}), flush=True)
    if mx.get_peak_memory() / GIB > args.memory_gib:
        report.update(status="stopped", stop_reason="Initial model/validation exceeded the declared memory guideline.",
                      peak_mlx_gib=mx.get_peak_memory() / GIB)
        write_json(run / "report.json", report)
        write_json(run / "progress.json", report)
        return
    step = local_start
    for step in range(local_start + 1, args.steps + 1):
        tick = time.perf_counter()
        rng = np.random.default_rng(args.seed + parent_updates + step)
        indexes = rng.integers(0, len(train_rows), size=args.batch_size)
        inputs, targets, masks = batch_arrays([train_rows[i] for i in indexes], args.max_seq_len)
        loss, grads = loss_grad(model, inputs, targets, masks)
        mx.eval(loss, grads)
        loss_value = float(loss.item())
        if not np.isfinite(loss_value):
            raise FloatingPointError("Nonfinite loss; no invalid checkpoint will be saved.")
        gradient_checks = {name: mx.all(mx.isfinite(value)) for name, value in tree_flatten(grads)}
        mx.eval(gradient_checks)
        if not all(bool(value.item()) for value in gradient_checks.values()):
            raise FloatingPointError("Nonfinite gradient; optimizer update was not applied.")
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        seconds = time.perf_counter() - tick
        training_seconds += seconds
        total_tokens += int(inputs.size)
        del grads, inputs, targets, masks, gradient_checks
        if step == local_start + 1:
            samples_after = parameter_samples(model)
            changed = {name: bool(mx.any(samples_after[name] != value).item()) for name, value in samples_before.items()}
            sample_key = "resume_update_sample_check" if args.resume else "first_update_sample_check"
            report[sample_key] = {"tensors": len(changed), "tensors_with_changed_sample": sum(changed.values()),
                "unchanged_sample_tensors": [name for name, value in changed.items() if not value],
                "scope": "Up to32 fixed positions per trainable tensor; not proof every weight changed."}
            del samples_before, samples_after
        measured = {"step": step, "total_optimizer_updates": parent_updates + step,
                    "loss": loss_value, "step_seconds": seconds,
                    "training_seconds": training_seconds, "padded_training_tokens": total_tokens,
                    "wall_seconds": previous_wall + time.perf_counter() - started,
                    "padded_tokens_per_second": total_tokens / training_seconds,
                    "peak_mlx_gib": mx.get_peak_memory() / GIB,
                    "active_mlx_gib": mx.get_active_memory() / GIB}
        if step % args.eval_every == 0 or step == args.steps or stopping:
            measured["validation"] = validate(model, valid_rows, args.max_seq_len)
            measured["training_fit"] = training_fit(model, train_rows, args.max_seq_len)
            history.append(measured)
        measured["peak_mlx_gib"] = mx.get_peak_memory() / GIB
        if measured["peak_mlx_gib"] > args.memory_gib:
            stopping = True
            report["stop_reason"] = "Measured peak exceeded the declared memory guideline; stop and inspect before scaling."
        report.update(measured)
        report["status"] = "stopped" if stopping else "training"
        write_json(run / "progress.json", report)
        if step == local_start + 1 or step % args.log_every == 0 or "validation" in measured:
            print(json.dumps({"event": "progress", **measured}), flush=True)
        if step % args.save_every == 0 or step == args.steps or stopping:
            save_checkpoint(run, model, optimizer, step, report)
            last_saved = step
        mx.clear_cache()
        if stopping:
            break
    if step != last_saved:
        save_checkpoint(run, model, optimizer, step, report)
    report["status"] = "stopped" if stopping else "completed"
    report["wall_seconds"] = previous_wall + time.perf_counter() - started
    report["checkpoint"] = json.loads((run / "latest.json").read_text())["checkpoint"]
    write_json(run / "report.json", report)
    write_json(run / "progress.json", report)
    print(json.dumps({"event": "finished", "status": report["status"], "step": step,
                      "parameters": actual, "wall_seconds": report["wall_seconds"]}), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--preset", choices=("32m", "135m", "500m", "1b", "4b"), default="32m")
    p.add_argument("--purpose", choices=("capacity_probe", "pilot", "training"), default="pilot")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=768)
    p.add_argument("--train-cases", type=int, default=4096)
    p.add_argument("--valid-cases", type=int, default=128)
    p.add_argument("--curriculum", choices=("primitives", "tiny", "graph"), default="tiny")
    p.add_argument("--seed", type=int, default=510000001)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--memory-gib", type=float, default=24)
    p.add_argument("--init-from")
    p.add_argument("--resume", action="store_true")
    return p


if __name__ == "__main__":
    train(parser().parse_args())
