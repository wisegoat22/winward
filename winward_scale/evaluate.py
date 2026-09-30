"""Bounded development evaluation and separately sealed, one-use final audits.

The decoder chooses an unrestricted vocabulary token. Its rollouts never receive
teacher choices, eligibility masks, corrections, or oracle continuations. The
reference solver scores those completed predictions afterwards. No tools run.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import time

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from agent_training.simulator import NEEDS_CLARIFICATION, STOP, search
from .data import (ByteTokenizer, DATA_VERSION, DEFER_LABEL, LIMITATION,
                   PRIMITIVES_VERSION, PRIMITIVE_PROMPT_HEADER,
                   STOP_LABEL, make_example, parse_prompt, serialize_prompt)
from .model import Decoder, ModelConfig
from .train import ROOT, digest, resolve_checkpoint, write_json

VERSION = "winward-byte-evaluation-1"
DEVELOPMENT_SPLITS = ("validation", "structural_validation")
FINAL_SPLITS = ("test", "structural_test")


@dataclass(frozen=True)
class Prediction:
    token_id: int
    decision_ms: float
    prompt_tokens: int = 0

    def __post_init__(self):
        if type(self.token_id) is not int or not 0 <= self.token_id < 260:
            raise ValueError("Prediction must be an unrestricted vocabulary token ID")
        if not np.isfinite(self.decision_ms) or self.decision_ms < 0:
            raise ValueError("Decision time must be finite and nonnegative")

    @property
    def label(self):
        return chr(self.token_id) if 32 <= self.token_id < 127 else None


class ModelPolicy:
    def __init__(self, model, max_input_tokens: int):
        self.model = model
        self.max_input_tokens = max_input_tokens
        self.tokenizer = ByteTokenizer()
        model.eval()

    def __call__(self, prompt: str) -> Prediction:
        started = time.perf_counter()
        tokens = self.tokenizer.encode_prompt(prompt)
        if len(tokens) > self.max_input_tokens:
            raise ValueError("Evaluation prompt exceeds the declared context; no truncation or fallback")
        logits = self.model(mx.array([tokens], dtype=mx.int32))
        last_logits = logits[0, -1]
        finite = mx.all(mx.isfinite(last_logits))
        choice = mx.argmax(last_logits, axis=-1)
        mx.eval(choice, finite)
        if not bool(finite.item()):
            raise FloatingPointError("Nonfinite prediction logits; no action or fallback is produced")
        token = int(choice.item())
        return Prediction(token, (time.perf_counter() - started) * 1000, len(tokens))


def _label(action_id):
    return STOP_LABEL if action_id == STOP else DEFER_LABEL if action_id == NEEDS_CLARIFICATION else action_id


def greedy_policy(prompt: str) -> Prediction:
    """Cheapest eligible state-changing action; no lookahead or solver."""
    started = time.perf_counter()
    scenario, _ = parse_prompt(prompt)
    if scenario.goal_met():
        chosen = STOP_LABEL
    else:
        eligible = [a for a in scenario.actions if a.eligible(scenario.state)
                    and a.apply(scenario.state) != scenario.state]
        chosen = min(eligible, key=lambda a: (a.cost(), a.id)).id if eligible else DEFER_LABEL
    return Prediction(ord(chosen), (time.perf_counter() - started) * 1000)


def exact_policy(prompt: str) -> Prediction:
    started = time.perf_counter()
    scenario, depth = parse_prompt(prompt)
    chosen = _label(search(scenario, max_depth=depth)["chosen_action_id"])
    return Prediction(ord(chosen), (time.perf_counter() - started) * 1000)


def rollout(policy, scenario, depth: int, first: Prediction | None = None,
            *, prompt_style: str = "workflow"):
    """Apply only the policy's own eligible actions, for at most supplied D <= 5.

    Goal completion is the observed simulator state, not a model claim. Once the
    goal becomes true execution ends; STOP quality is measured on first actions
    and on premature STOPs encountered before completion.
    """
    if type(depth) is not int or not 1 <= depth <= 5:
        raise ValueError("Rollout depth must be between one and five")
    if prompt_style not in ("workflow", "primitives"):
        raise ValueError("Unknown rollout prompt style")
    current = scenario
    trajectory = []
    result = {"initially_done": current.goal_met(), "completed": current.goal_met(),
              "actions_taken": 0, "estimated_action_cost": 0.0,
              "estimated_action_tokens": 0.0, "estimated_action_latency_ms": 0.0,
              "measured_decision_ms": 0.0, "premature_stop": False,
              "invalid_label": False, "ineligible_action": False,
              "unnecessary_after_goal": False, "trajectory": trajectory}
    for step in range(depth):
        prediction = first if step == 0 and first is not None else policy(
            serialize_prompt(current, depth - step, prompt_style=prompt_style))
        label = prediction.label
        entry = {"token_id": prediction.token_id, "label": label,
                 "state_before": current.state, "decision_ms": prediction.decision_ms}
        trajectory.append(entry)
        result["measured_decision_ms"] += prediction.decision_ms
        if current.goal_met():
            result["unnecessary_after_goal"] = label != STOP_LABEL
            result["termination"] = "already_done"
            break
        if label == STOP_LABEL:
            result.update(premature_stop=True, termination="premature_stop")
            break
        if label == DEFER_LABEL:
            result["termination"] = "deferred"
            break
        action = next((a for a in current.actions if a.id == label), None)
        if action is None:
            result.update(invalid_label=True, termination="invalid_label")
            break
        if not action.eligible(current.state):
            result.update(ineligible_action=True, termination="ineligible_action")
            break
        current = replace(current, state=action.apply(current.state))
        entry["state_after"] = current.state
        result["actions_taken"] += 1
        result["estimated_action_cost"] += action.cost()
        result["estimated_action_tokens"] += action.tokens
        result["estimated_action_latency_ms"] += action.latency_ms
        if current.goal_met():
            result.update(completed=True, termination="goal_reached")
            break
    else:
        result["termination"] = "horizon_exhausted"
    result["final_state"] = current.state
    return result


def first_score(prediction, scenario, reference):
    label = prediction.label
    action = next((a for a in scenario.actions if a.id == label), None)
    return {"canonical_match": label == _label(reference["chosen_action_id"]),
            "optimal_match": label in {_label(a) for a in reference["optimal_action_ids"]},
            "premature_stop": label == STOP_LABEL and not scenario.goal_met(),
            "invalid_label": label not in {a.id for a in scenario.actions} | {STOP_LABEL, DEFER_LABEL},
            "ineligible_action": action is not None and not action.eligible(scenario.state),
            "incorrect_defer": label == DEFER_LABEL and reference["success"],
            "already_done_correct_stop": scenario.goal_met() and label == STOP_LABEL}


def _time_summary(values):
    return {"count": len(values), "total_ms": sum(values),
            "mean_ms": float(np.mean(values)) if values else None,
            "median_ms": float(np.median(values)) if values else None,
            "p95_ms": float(np.percentile(values, 95)) if values else None}


def evaluate_cases(policy, examples, *, rollout_count: int):
    """Development/test-agnostic pure evaluator; caller owns split authorization.

    `make_example` computes synthetic labels when generating cases. Those labels
    and all teacher metadata are deliberately discarded here. Independent solver
    scoring and baseline evaluation begin only after ALL model predictions and
    model trajectories have completed.
    """
    examples = list(examples)
    if not examples or type(rollout_count) is not int or not 0 <= rollout_count <= len(examples):
        raise ValueError("Need cases and a rollout count between zero and case count")
    cases = []
    for index, example in enumerate(examples):
        scenario, depth = parse_prompt(example.prompt)
        prompt_style = "primitives" if example.prompt.startswith(PRIMITIVE_PROMPT_HEADER) else "workflow"
        prediction = policy(example.prompt)
        model_rollout = (rollout(policy, scenario, depth, prediction, prompt_style=prompt_style)
                         if index < rollout_count else None)
        cases.append({"id": example.id, "family": example.family, "scenario": scenario,
                      "depth": depth, "prompt": example.prompt, "model": prediction,
                      "prompt_style": prompt_style, "model_rollout": model_rollout})
    # Oracle information has no path back to the completed model predictions.
    for case in cases:
        case["reference"] = search(case["scenario"], max_depth=case["depth"])
        for name, baseline in (("greedy", greedy_policy), ("exact_search", exact_policy)):
            prediction = baseline(case["prompt"])
            case[name] = prediction
            case[name + "_rollout"] = (rollout(baseline, case["scenario"], case["depth"], prediction,
                                               prompt_style=case["prompt_style"])
                                        if case["model_rollout"] is not None else None)
    summaries = {}
    details = []
    for name in ("model", "greedy", "exact_search"):
        scores = [first_score(c[name], c["scenario"], c["reference"]) for c in cases]
        first = {key + "_count": sum(s[key] for s in scores) for key in scores[0]}
        first.update(cases=len(cases), canonical_accuracy=first["canonical_match_count"] / len(cases),
                     optimal_accuracy=first["optimal_match_count"] / len(cases),
                     measured_decision_time=_time_summary([c[name].decision_ms for c in cases]))
        selected = [c for c in cases if c[name + "_rollout"] is not None]
        done = [c for c in selected if c["scenario"].goal_met()]
        solvable = [c for c in selected if not c["scenario"].goal_met() and c["reference"]["success"]]
        unsolvable = [c for c in selected if not c["reference"]["success"]]
        successes = [c for c in solvable if c[name + "_rollout"]["completed"]]
        trajectories = [c[name + "_rollout"] for c in selected]
        rollups = {"cases": len(selected), "initially_done": len(done),
                   "initially_unsatisfied_solvable": len(solvable),
                   "unsolvable_within_supplied_horizon": len(unsolvable),
                   "solvable_completed": len(successes),
                   "solvable_completion_rate": len(successes) / len(solvable) if solvable else None,
                   "already_done_correct_stop": sum(not c[name + "_rollout"]["unnecessary_after_goal"] for c in done),
                   "unsolvable_correct_defer": sum(c[name].label == DEFER_LABEL for c in unsolvable),
                   "action_cost_total_all_attempts": sum(t["estimated_action_cost"] for t in trajectories),
                   "action_tokens_total_all_attempts": sum(t["estimated_action_tokens"] for t in trajectories),
                   "action_latency_ms_total_all_attempts": sum(t["estimated_action_latency_ms"] for t in trajectories),
                   "completed_solvable_mean_action_cost": (float(np.mean([c[name + "_rollout"]["estimated_action_cost"] for c in successes])) if successes else None),
                   "measured_decision_time": _time_summary([t["measured_decision_ms"] for t in trajectories]),
                   "premature_stop_count": sum(t["premature_stop"] for t in trajectories),
                   "invalid_label_count": sum(t["invalid_label"] for t in trajectories),
                   "ineligible_action_count": sum(t["ineligible_action"] for t in trajectories)}
        summaries[name] = {"first_action": first, "rollouts": rollups}
    for c in cases:
        details.append({"id": c["id"], "family": c["family"], "horizon": c["depth"],
                        "initially_done": c["scenario"].goal_met(), "reference_solvable": c["reference"]["success"],
                        "reference_action_cost": c["reference"]["total_cost"],
                        "policies": {name: {"prediction": asdict(c[name]),
                                           "score": first_score(c[name], c["scenario"], c["reference"]),
                                           "rollout": c[name + "_rollout"]}
                                     for name in summaries}})
    return {"version": VERSION, "promotion": False, "policies": summaries, "case_results": details,
            "limitations": [LIMITATION, "No actual software tools run; action costs/latencies are supplied estimates.",
                            "Measured decision time includes input encoding/parsing and inference; model loading excluded.",
                            "Model completion uses only its own sequential choices. Exact search is a separately labeled reference.",
                            "Completion excludes initially satisfied and horizon-unsolvable cases; those counts are separate.",
                            "Completed-only mean cost compares different successful subsets; use completion and all-attempt totals together.",
                            "No automatic promotion: no acceptance thresholds were set by this evaluator."]}


def checkpoint_info(run):
    checkpoint = resolve_checkpoint(run)
    info = json.loads((checkpoint / "checkpoint.json").read_text())
    if digest(checkpoint / "model.safetensors") != info["model_sha256"]:
        raise ValueError("Model checkpoint checksum mismatch")
    config = ModelConfig(**info["config"])
    if config.parameter_count() != info["parameters"]:
        raise ValueError("Checkpoint parameter count disagrees with architecture")
    return checkpoint, info, config


def load_policy(checkpoint, info, config, max_input_tokens):
    model = Decoder(config)
    model.load_weights(str(checkpoint / "model.safetensors"))
    mx.eval(model.parameters())
    if sum(v.size for _, v in tree_flatten(model.parameters())) != info["parameters"]:
        raise ValueError("Loaded parameter count disagrees with checkpoint")
    return ModelPolicy(model, max_input_tokens)


def evaluator_source_hashes():
    names = ("winward_scale/evaluate.py", "winward_scale/model.py", "winward_scale/data.py",
             "winward_scale/train.py", "agent_training/simulator.py")
    return {name: digest(ROOT / name) for name in names}


def _canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _exclusive_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def prepare_evaluation(args):
    """Validate/checksum/seal before model allocation or any dataset generation."""
    if args.cases < 1 or args.cases > 4096 or not 0 <= args.rollouts <= min(args.cases, 512):
        raise ValueError("Use 1–4096 cases and 0–min(cases,512) rollouts")
    if args.seed is None or args.seed < 0 or args.max_input_tokens < 1:
        raise ValueError("Declare a nonnegative seed and positive input-token limit")
    if not np.isfinite(args.memory_gib) or not 1 <= args.memory_gib <= 28:
        raise ValueError("Memory guideline must be between 1 and 28 GiB")
    checkpoint, info, config = checkpoint_info(args.run)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Evaluation output already exists; results are not overwritten")
    payload = {"version": VERSION, "checkpoint": str(checkpoint.resolve()),
               "model_sha256": info["model_sha256"], "parameters": info["parameters"],
               "config": asdict(config), "split": args.split, "cases": args.cases,
               "rollouts": args.rollouts, "seed": args.seed, "curriculum": args.curriculum,
               "max_input_tokens": args.max_input_tokens, "memory_gib": args.memory_gib,
               "data_version": PRIMITIVES_VERSION if args.curriculum == "primitives" else DATA_VERSION,
               "source_sha256": evaluator_source_hashes(),
               "output": str(output), "promotion": False}
    if args.split in DEVELOPMENT_SPLITS:
        if args.declare_audit or args.run_audit:
            raise ValueError("Audit flags apply only to final test splits")
        return checkpoint, info, config, payload, False
    if not args.declare_audit and not args.run_audit:
        raise ValueError("Final splits require --declare-audit, then a separate --run-audit invocation")
    if args.declare_audit and args.run_audit:
        raise ValueError("Seal and execute final audits in separate invocations")
    # Deliberately excludes checkpoint/count/output: even changing those must not
    # silently re-use the same final seed domain against another candidate.
    domain = {"split": args.split, "seed": args.seed, "curriculum": args.curriculum,
              "data_version": payload["data_version"]}
    reservation = ROOT / "runs" / "scale-audit-reservations" / (_canonical_digest(domain) + ".json")
    protocol_path = Path(str(output) + ".audit.json")
    sealed = {"protocol": payload, "protocol_sha256": _canonical_digest(payload)}
    if args.declare_audit:
        if protocol_path.exists():
            raise FileExistsError("Audit protocol already sealed")
        _exclusive_json(reservation, {"domain": domain, "protocol_sha256": sealed["protocol_sha256"]})
        _exclusive_json(protocol_path, sealed)
        return checkpoint, info, config, payload, True
    if not protocol_path.is_file() or not reservation.is_file():
        raise ValueError("No sealed/reserved audit protocol; declare it before running final cases")
    stored = json.loads(protocol_path.read_text())
    reserved = json.loads(reservation.read_text())
    if stored != sealed or reserved.get("protocol_sha256") != sealed["protocol_sha256"]:
        raise ValueError("Audit settings, sources, or checkpoint changed after sealing")
    _exclusive_json(Path(str(output) + ".audit-started.json"),
                    {"protocol_sha256": sealed["protocol_sha256"], "status": "consumed_before_generation"})
    return checkpoint, info, config, payload, False


def main(args):
    checkpoint, info, config, protocol, sealed_only = prepare_evaluation(args)
    if sealed_only:
        print(json.dumps({"status": "audit_sealed", "cases_generated": 0,
                          "next": "Repeat identical arguments with --run-audit instead of --declare-audit"}), flush=True)
        return
    mx.set_memory_limit(int(args.memory_gib * 1024 ** 3))
    mx.set_cache_limit(256 * 1024 ** 2)
    mx.reset_peak_memory()
    started = time.perf_counter()
    policy = load_policy(checkpoint, info, config, args.max_input_tokens)
    load_seconds = time.perf_counter() - started
    examples = [make_example(i, args.split, seed=args.seed, curriculum=args.curriculum) for i in range(args.cases)]
    result = evaluate_cases(policy, examples, rollout_count=args.rollouts)
    result.update(protocol=protocol, scope="final_audit" if args.split in FINAL_SPLITS else "development",
                  model_load_seconds=load_seconds, wall_seconds=time.perf_counter() - started,
                  peak_mlx_gib=mx.get_peak_memory() / 1024 ** 3)
    # Atomic complete result; the started audit marker survives crashes and
    # prevents a partial final audit from being silently retried or tuned on.
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps({"status": "evaluated", "scope": result["scope"], "promotion": False,
                      "model": result["policies"]["model"], "output": args.output}), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=DEVELOPMENT_SPLITS + FINAL_SPLITS, default="validation")
    p.add_argument("--cases", type=int, default=128)
    p.add_argument("--rollouts", type=int, default=32)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--curriculum", choices=("tiny", "graph", "primitives"), default="tiny")
    p.add_argument("--max-input-tokens", type=int, default=768)
    p.add_argument("--memory-gib", type=float, default=24)
    p.add_argument("--declare-audit", action="store_true")
    p.add_argument("--run-audit", action="store_true")
    return p


if __name__ == "__main__":
    main(parser().parse_args())
