"""Train progressively wider copies of our own structured decision policy.

Growth preserves a starting skill, not a quality improvement. Only fresh measured
decisions assess quality. Every matrix is trainable; no outside model is loaded.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import signal
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from agent_training.model import PolicyConfig
from agent_training.model_v3 import UnifiedPolicy
from .structured import StructuredConfig, StructuredPolicy, grow_structured
from .structured_data import corpus, batch_arrays, evaluate
from .gpu_lock import local_compute
from .optimizer import SequentialAdafactor
from .train import (FP32Adafactor, GIB, ROOT, assert_finite_tree, digest,
                    parameter_samples, resolve_checkpoint, write_json)


def load_parent(path):
    path = Path(path)
    if (path / "latest.json").is_file():
        path = resolve_checkpoint(path)
    if (path / "checkpoint.json").is_file():
        info = json.loads((path / "checkpoint.json").read_text())
        if info.get("track") != "structured":
            raise ValueError("Expected our structured checkpoint, not a byte model.")
        config = StructuredConfig(**info["config"])
        expected = info["model_sha256"]
        model = StructuredPolicy(config)
    else:
        info = json.loads((path / "report.json").read_text())
        if info.get("schema_version") != "winward-v4-public-observation-1":
            raise ValueError("Initial growth requires our own frozen v4 checkpoint.")
        expected = info["checkpoint_sha256"]
        config = PolicyConfig(**json.loads((path / "config.json").read_text()))
        model = UnifiedPolicy(config)
    if digest(path / "model.safetensors") != expected:
        raise ValueError("Parent checkpoint checksum failed.")
    model.load_weights(str(path / "model.safetensors"), strict=True)
    mx.eval(model.parameters())
    model.eval()
    return model, path, info, expected


def source_hashes():
    # Capture the complete frozen encoder/teacher dependencies too.
    paths = list((ROOT / "agent_training").glob("*.py"))
    paths += [ROOT / "agent_lab" / "belief.py"]
    paths += [ROOT / "winward_scale" / name for name in (
        "structured.py", "structured_data.py", "structured_train.py", "grow.py", "train.py", "gpu_lock.py", "optimizer.py")]
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def decision_loss(model, x, valid, eligible, targets):
    logits = model(x, valid, eligible).astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return -mx.mean(mx.sum(targets * logprobs, axis=-1))


def validation_summary(value):
    """Keep progress/terminal output small; full decisions are saved separately."""
    fields = ("examples", "next_action_correct", "next_action_accuracy", "invalid_or_ineligible_count",
              "wall_seconds", "inference_ms", "batch_size", "amortized_inference_ms_per_example", "scope")
    result = {key: value[key] for key in fields if key in value}
    if "by_domain" in value:
        result["by_domain"] = {key: validation_summary(item) for key, item in value["by_domain"].items()}
    return result


def validation_record(run, name, model, rows, batch_size):
    value = evaluate(model, rows, batch_size=batch_size)
    write_json(run / f"validation-{name}.json", value)
    return validation_summary(value)


def save_checkpoint(run, model, optimizer, step, report):
    assert_finite_tree(model.parameters(), "model weights")
    assert_finite_tree(optimizer.state, "optimizer state")
    relative = f"checkpoints/step-{step:08d}"
    destination = run / relative
    destination.mkdir(parents=True, exist_ok=False)
    model.save_weights(str(destination / "model.safetensors"))
    mx.save_safetensors(str(destination / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
    info = {"track": "structured", "step": step,
            "total_optimizer_updates": report["total_optimizer_updates"],
            "config": model.config.to_dict(), "parameters": model.count_parameters(),
            "model_sha256": digest(destination / "model.safetensors"),
            "optimizer_sha256": digest(destination / "optimizer.safetensors"),
            "report": report}
    write_json(destination / "checkpoint.json", info)
    write_json(run / "latest.json", {"checkpoint": relative, "step": step})
    return info


def train(args):
    if any(value < 1 for value in (args.factor, args.steps, args.batch_size,
            args.train_cases, args.valid_cases, args.eval_every, args.save_every, args.log_every,
            args.validation_batch_size)):
        raise ValueError("Counts, growth factor, and intervals must be positive.")
    if not 1 <= args.memory_gib <= 28 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Use a finite positive learning rate and memory guideline of 1–28 GiB.")
    if not math.isfinite(args.noise) or not 0 <= args.noise <= .1:
        raise ValueError("Symmetry-breaking noise must be between 0 and .1.")
    run = Path(args.output)
    run.mkdir(parents=True, exist_ok=False)
    mx.set_memory_limit(int(args.memory_gib * GIB))
    mx.set_cache_limit(128 * 1024 ** 2)
    mx.random.seed(args.seed)
    mx.reset_peak_memory()
    started = time.perf_counter()
    parent, parent_path, parent_info, parent_hash = load_parent(args.init_from)
    config = StructuredConfig(input_dim=parent.config.input_dim,
        width=parent.config.width * args.factor, layers=parent.config.layers,
        heads=parent.config.heads * args.factor, expansion=parent.config.expansion,
        norm_eps=parent.norm.eps)
    if 3 * config.parameter_count() * 2 / GIB + 2 > args.memory_gib:
        raise ValueError("Conservative weights/gradients/update estimate plus overhead exceeds the declared guideline.")
    protocol = {"version": "winward-structured-scale-1", "track": "structured",
        "config": config.to_dict(), "parameters": config.parameter_count(),
        "arguments": vars(args), "purpose": args.purpose,
        "initialization": "Function-preserving expansion of our own trained policy, followed by full-parameter training",
        "parent_sha256": parent_hash, "source_sha256": source_hashes(),
        "optimizer": "Sequential Adafactor, FP32 factored moments and update math; BF16 stored matrix parameters, FP32 norms; evaluated gradients consumed one leaf at a time",
        "optimizer_reset": args.factor != 1 or not args.continue_optimizer,
        "hardware": mx.device_info(), "mlx_version": mx.__version__,
        "promotion": "Size and copying are not quality improvements. Development validation cannot promote a policy."}
    write_json(run / "protocol.json", protocol)
    for name, expected in protocol["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Source changed during protocol capture.")
        destination = run / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    write_json(run / "progress.json", {"status": "preparing", "track": "structured", "parameters": config.parameter_count()})
    train_rows = corpus(args.train_cases, "train", args.seed)
    valid_rows = corpus(args.valid_cases, "development", args.seed)
    parent_validation = validation_record(run, "parent", parent, valid_rows, args.validation_batch_size)
    model = grow_structured(parent, args.factor, noise=args.noise, seed=args.seed)
    del parent
    mx.clear_cache()
    if model.count_parameters() != config.parameter_count():
        raise ValueError("Materialized parameter count disagrees with protocol.")
    assert_finite_tree(model.parameters(), "expanded weights")
    optimizer = SequentialAdafactor(learning_rate=args.learning_rate, relative_step=False,
        scale_parameter=False, beta_1=None, weight_decay=0.0)
    if args.continue_optimizer:
        if args.factor != 1 or not (parent_path / "optimizer.safetensors").is_file():
            raise ValueError("Optimizer continuation requires factor 1 and an optimizer checkpoint.")
        if digest(parent_path / "optimizer.safetensors") != parent_info["optimizer_sha256"]:
            raise ValueError("Optimizer checkpoint checksum failed.")
        optimizer.state = tree_unflatten(list(mx.load(str(parent_path / "optimizer.safetensors")).items()))
        optimizer.learning_rate = args.learning_rate
    prior_updates = int(parent_info.get("total_optimizer_updates", 0))
    initial = validation_record(run, "initial", model, valid_rows, args.validation_batch_size)
    report = {"status": "training", "track": "structured", "purpose": args.purpose,
        "parameters": model.count_parameters(), "step": 0, "total_optimizer_updates": prior_updates,
        "parent_validation": parent_validation, "initial_validation": initial,
        "validation_history": [], "training_seconds": 0., "training_examples_seen": 0,
        "limitations": ["Structured public observations with fixed 505-feature encoding, not a language model.",
            "Five-step teacher labels do not prove the network searches five steps at inference.",
            "Width expansion initially retains skills; it does not add independent learned knowledge.",
            "Stored BF16 parameters round small updates; no FP32 master weights.",
            "Memory guideline is advisory, not a hard operating-system limit."]}
    write_json(run / "progress.json", report)
    print(json.dumps({"event": "initialized", "parameters": report["parameters"],
        "parent_validation": parent_validation, "initial_validation": initial}), flush=True)
    if mx.get_peak_memory() / GIB > args.memory_gib:
        report.update(status="stopped", stop_reason="Expansion/initial validation exceeded memory guideline.")
        write_json(run / "report.json", report)
        write_json(run / "progress.json", report)
        return
    samples_before = parameter_samples(model)
    gradient = nn.value_and_grad(model, decision_loss)
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    model.train()
    for step in range(1, args.steps + 1):
        tick = time.perf_counter()
        rng = np.random.default_rng(args.seed + prior_updates + step)
        indices = rng.integers(0, len(train_rows), args.batch_size)
        batch = [mx.array(value) for value in batch_arrays([train_rows[i] for i in indices])]
        # Gentle warmup/decay protects the retained skill while training all weights.
        schedule = min(1., step / min(20, args.steps)) * (.25 + .75 * .5 * (1 + math.cos(math.pi * step / args.steps)))
        optimizer.learning_rate = args.learning_rate * schedule
        loss, grads = gradient(model, *batch)
        mx.eval(loss, grads)
        value = float(loss.item())
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite loss; update was not applied.")
        assert_finite_tree(grads, "gradients")
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        del grads, batch
        elapsed = time.perf_counter() - tick
        report["training_seconds"] += elapsed
        report["training_examples_seen"] += args.batch_size
        if step == 1:
            after = parameter_samples(model)
            changed = {name: bool(mx.any(after[name] != before).item()) for name, before in samples_before.items()}
            report["first_update_sample_check"] = {"tensors": len(changed),
                "tensors_with_changed_sample": sum(changed.values()),
                "unchanged_sample_tensors": [name for name in changed if not changed[name]],
                "scope": "At most32 fixed positions per tensor; not proof every scalar changes. Common logit-shift biases have no decision effect."}
            del after, samples_before
        measured = {"step": step, "total_optimizer_updates": prior_updates + step,
            "loss": value, "step_seconds": elapsed, "peak_mlx_gib": mx.get_peak_memory() / GIB,
            "active_mlx_gib": mx.get_active_memory() / GIB,
            "wall_seconds": time.perf_counter() - started}
        if step % args.eval_every == 0 or step == args.steps or stopping:
            measured["validation"] = validation_record(run, f"step-{step:08d}", model, valid_rows, args.validation_batch_size)
            report["validation_history"].append(dict(measured))
            model.train()
        measured["peak_mlx_gib"] = mx.get_peak_memory() / GIB
        if measured["peak_mlx_gib"] > args.memory_gib:
            stopping = True
            report["stop_reason"] = "Measured peak exceeded memory guideline; inspect before continuing."
        report.update(measured)
        report["status"] = "stopped" if stopping else "training"
        write_json(run / "progress.json", report)
        if step == 1 or step % args.log_every == 0 or "validation" in measured:
            print(json.dumps({"event": "progress", **measured}), flush=True)
        if step % args.save_every == 0 or step == args.steps or stopping:
            save_checkpoint(run, model, optimizer, step, report)
        mx.clear_cache()
        if stopping:
            break
    report.update(status="stopped" if stopping else "completed",
        wall_seconds=time.perf_counter() - started,
        checkpoint=json.loads((run / "latest.json").read_text())["checkpoint"])
    write_json(run / "report.json", report)
    write_json(run / "progress.json", report)
    print(json.dumps({"event": "finished", "status": report["status"], "step": step,
        "parameters": report["parameters"], "wall_seconds": report["wall_seconds"]}), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--init-from", required=True)
    p.add_argument("--factor", type=int, default=1)
    p.add_argument("--noise", type=float, default=.01)
    p.add_argument("--continue-optimizer", action="store_true")
    p.add_argument("--purpose", choices=("capacity_probe", "pilot", "training"), default="training")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--train-cases", type=int, default=4096)
    p.add_argument("--valid-cases", type=int, default=256)
    p.add_argument("--validation-batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=710000001)
    p.add_argument("--learning-rate", type=float, default=.00003)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--memory-gib", type=float, default=28)
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    output_existed = Path(arguments.output).exists()
    try:
        with local_compute():
            train(arguments)
    except Exception as error:
        output = Path(arguments.output)
        if not output_existed and (output / "protocol.json").is_file() and not (output / "report.json").exists():
            progress_path = output / "progress.json"
            progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
            progress.update(status="failed", failure=f"{type(error).__name__}: {error}")
            write_json(output / "report.json", progress)
            write_json(progress_path, progress)
        raise
