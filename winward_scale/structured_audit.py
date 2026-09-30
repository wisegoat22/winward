"""Seal and execute a fresh, one-use audit of our trained structured policy.

Declaration hashes artifacts and sources but loads no model and generates no
cases. Execution consumes the seal before generation. The fixed comparison is
our original v4 checkpoint; no third-party model or oracle correction is used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np

from .gpu_lock import local_compute
from .structured import StructuredConfig
from .structured_data import VERSION as DATA_VERSION, batch_arrays, corpus, evaluate, rollout_evaluate
from .structured_train import load_parent, source_hashes as training_source_hashes
from .train import GIB, ROOT, digest, resolve_checkpoint, write_json

VERSION = "winward-structured-scale-audit-1"
BASELINE_SHA256 = "de989d050a45cc2399a31a32081b8f2961a67a2d70d545a8888f8de882ecbede"
RETENTION_RULES = {
    "next_action": "Candidate optimal-action agreement >= original v4, overall and separately graph/belief.",
    "invalid_choices": "Candidate selects zero invalid/ineligible slots; eligibility is a supplied hard rule.",
    "rollout_success": "Candidate mean expected success on initially unfinished, reference-reachable cases >= original v4, overall and separately by domain.",
    "cost": "Compare declared costs only on identical cases with equal positive expected success; descriptive, not a promotion gate.",
    "timing": "Measured model decision times are reported separately from supplied action costs; descriptive shared-Mac measurements.",
    "promotion": "Never automatically promote. This is not the historical twelve-check v4 audit, a real-tool audit, or proof of broad competence.",
}
TIMING_PROTOCOL = {
    "method_order": ["candidate", "v4_parent"],
    "warmup": "One forward batch per model on the same first final cases, with no updates or label scoring; included in global time/call budgets and excluded from measured decision time.",
    "comparison": "Fixed-order descriptive timings on one Mac; model loading and warmup reported separately. No precision latency or speedup claim.",
}


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_hashes():
    result = training_source_hashes()
    result["winward_scale/structured_audit.py"] = digest(ROOT / "winward_scale/structured_audit.py")
    return result


def artifact_manifest(run):
    """Read/checksum only. No constructor, MLX evaluation or dataset call."""
    checkpoint = resolve_checkpoint(run).resolve()
    info = json.loads((checkpoint / "checkpoint.json").read_text())
    if info.get("track") != "structured":
        raise ValueError("Final audit requires our structured checkpoint")
    if (type(info.get("step")) is not int or info["step"] < 1
            or type(info.get("total_optimizer_updates")) is not int or info["total_optimizer_updates"] < 1):
        raise ValueError("The candidate must have completed real optimizer updates")
    config = StructuredConfig(**info["config"])
    if config.input_dim != 505 or config.parameter_count() != info["parameters"]:
        raise ValueError("Candidate config does not match its declared parameter count/schema")
    if digest(checkpoint / "model.safetensors") != info["model_sha256"]:
        raise ValueError("Candidate weight checksum failed")
    baseline = (ROOT / "runs" / "goalpolicy-v4").resolve()
    base_info = json.loads((baseline / "report.json").read_text())
    base_config = json.loads((baseline / "config.json").read_text())
    if base_info.get("schema_version") != "winward-v4-public-observation-1":
        raise ValueError("Expected the original frozen v4 observation schema")
    if base_info.get("checkpoint_sha256") != BASELINE_SHA256 or digest(baseline / "model.safetensors") != BASELINE_SHA256:
        raise ValueError("The original frozen v4 baseline changed")
    if base_config.get("input_dim") != 505:
        raise ValueError("Baseline public feature schema differs")
    return {
        "candidate": {"path": str(checkpoint), "config": info["config"],
                      "parameters": info["parameters"], "step": info["step"],
                      "total_optimizer_updates": info["total_optimizer_updates"],
                      "model_sha256": info["model_sha256"],
                      "checkpoint_metadata_sha256": digest(checkpoint / "checkpoint.json"),
                      "config_sha256": canonical_digest(info["config"])},
        "v4_parent": {"path": str(baseline), "config": base_config,
                      "parameters": base_info["parameter_count"], "model_sha256": BASELINE_SHA256,
                      "report_sha256": digest(baseline / "report.json"),
                      "config_sha256": digest(baseline / "config.json")},
    }


def exclusive_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _paths(output):
    return {"protocol": Path(str(output) + ".protocol.json"),
            "started": Path(str(output) + ".started.json"),
            "source": Path(str(output) + ".source")}


def prepare(args):
    """Seal or verify/consume an audit, without generating final instances."""
    for name, low, high in (("cases", 1, 4096), ("rollouts", 1, 128), ("batch_size", 1, 64),
                            ("max_belief_nodes", 1, 4096), ("max_forward_calls", 1, 100_000)):
        value = getattr(args, name)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer between {low} and {high}")
    if args.rollouts > args.cases or type(args.seed) is not int or args.seed < 0:
        raise ValueError("Rollouts must not exceed cases; seed must be a nonnegative integer")
    if not math.isfinite(args.max_seconds) or not 1 <= args.max_seconds <= 7200:
        raise ValueError("Declare a finite runtime budget between 1 and 7200 seconds")
    if not math.isfinite(args.memory_gib) or not 1 <= args.memory_gib <= 28:
        raise ValueError("Memory guideline must be between 1 and 28 GiB")
    if args.declare == args.run_audit:
        raise ValueError("Use exactly one of --declare or --run-audit in separate invocations")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Preserve the existing audit result; no overwrite or rerun")
    artifacts = artifact_manifest(args.run)
    protocol = {
        "version": VERSION, "data_version": DATA_VERSION, "split": "final_audit",
        "artifacts": artifacts, "source_sha256": source_hashes(),
        "settings": {key: getattr(args, key) for key in ("cases", "rollouts", "batch_size", "seed",
            "max_belief_nodes", "max_forward_calls", "max_seconds", "memory_gib")},
        "output": str(output), "retention_rules": RETENTION_RULES, "timing_protocol": TIMING_PROTOCOL,
        "generation": "Fresh disjoint final seed domains; no saved v1–v4 final instances read.",
        "selection": "A single frozen checkpoint; cases/thresholds may not change after observation.",
        "promotion": False,
    }
    sealed = {"protocol": protocol, "protocol_sha256": canonical_digest(protocol)}
    paths = _paths(output)
    # A changed output, checkpoint or count still overlaps the same prefix and
    # cannot make an already used final seed fresh again.
    domain = {"data_version": DATA_VERSION, "split": "final_audit", "seed": args.seed}
    reservation = ROOT / "runs" / "structured-audit-reservations" / (canonical_digest(domain) + ".json")
    if args.declare:
        if paths["protocol"].exists():
            raise FileExistsError("This audit protocol is already sealed")
        exclusive_json(reservation, {"domain": domain, "protocol_sha256": sealed["protocol_sha256"]})
        exclusive_json(paths["protocol"], sealed)
        for name, expected in protocol["source_sha256"].items():
            destination = paths["source"] / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
            if digest(destination) != expected:
                raise ValueError("Source changed while the audit snapshot was being sealed")
        return sealed, True
    if not paths["protocol"].is_file() or not reservation.is_file():
        raise ValueError("Seal this final audit with --declare before execution")
    if json.loads(paths["protocol"].read_text()) != sealed:
        raise ValueError("Candidate, baseline, config, sources or settings changed after audit sealing")
    if json.loads(reservation.read_text()).get("protocol_sha256") != sealed["protocol_sha256"]:
        raise ValueError("This final seed is reserved by a different sealed audit")
    for name, expected in protocol["source_sha256"].items():
        if digest(paths["source"] / name) != expected:
            raise ValueError("Sealed source snapshot checksum changed")
    exclusive_json(paths["started"], {"protocol_sha256": sealed["protocol_sha256"],
                                      "status": "consumed_before_model_loading_or_final_generation"})
    return sealed, False


class Budget:
    def __init__(self, seconds, forward_calls):
        self.started = time.perf_counter()
        self.seconds, self.limit, self.calls = seconds, forward_calls, 0

    def check(self):
        if time.perf_counter() - self.started > self.seconds:
            raise RuntimeError("Declared audit runtime budget exceeded; no partial quality result")

    def before_forward(self):
        self.check()
        if self.calls >= self.limit:
            raise RuntimeError("Declared audit forward-call budget exceeded; no partial quality result")
        self.calls += 1


class BudgetedModel:
    def __init__(self, model, budget):
        self.model, self.budget = model, budget

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args):
        self.budget.before_forward()
        result = self.model(*args)
        # Synchronize before checking time so queued GPU work cannot evade the
        # deadline. A running kernel is not forcibly interrupted mid-operation.
        mx.eval(result)
        self.budget.check()
        return result


def model_inventory(model, expected):
    dtype_counts = defaultdict(lambda: {"parameters": 0, "bytes": 0, "tensors": 0})
    count = 0
    for _, value in tree_flatten(model.parameters()):
        count += int(value.size)
        name = str(value.dtype).rsplit(".", 1)[-1]
        item = dtype_counts[name]
        item["parameters"] += int(value.size)
        item["bytes"] += int(value.nbytes)
        item["tensors"] += 1
    if count != expected:
        raise ValueError("Materialized model parameter count differs from sealed metadata")
    return {"parameters": count, "by_dtype": dict(dtype_counts),
            "stored_parameter_gib": sum(v["bytes"] for v in dtype_counts.values()) / GIB}


def comparison_checks(candidate, parent, rollouts):
    """Predeclared narrow retention checks; never the original v4 promotion gate."""
    checks = {}
    def retain(name, candidate_value, parent_value):
        checks[name] = {"candidate": candidate_value, "v4_parent": parent_value,
                        "passed": candidate_value >= parent_value if candidate_value is not None and parent_value is not None else None}
    retain("next_action_overall", candidate["next_action_accuracy"], parent["next_action_accuracy"])
    for kind in ("graph", "belief"):
        retain("next_action_" + kind, candidate["by_domain"].get(kind, {}).get("next_action_accuracy"),
               parent["by_domain"].get(kind, {}).get("next_action_accuracy"))
    checks["no_invalid_or_ineligible_choices"] = {"candidate": candidate["invalid_or_ineligible_count"],
        "v4_parent": parent["invalid_or_ineligible_count"], "passed": candidate["invalid_or_ineligible_count"] == 0}
    methods = rollouts["methods"]
    for domain in ("overall", "graph", "belief"):
        a = methods["candidate"]["overall"] if domain == "overall" else methods["candidate"]["by_domain"].get(domain, {})
        b = methods["v4_parent"]["overall"] if domain == "overall" else methods["v4_parent"]["by_domain"].get(domain, {})
        retain("rollout_success_" + domain, a.get("expected_verified_success_unfinished_reachable"),
               b.get("expected_verified_success_unfinished_reachable"))
    matched_cost_deltas = []
    matched = []
    for index, (a, b) in enumerate(zip(methods["candidate"]["episodes"], methods["v4_parent"]["episodes"])):
        if not a["initially_verified"] and a["expected_verified_success"] > 0 and math.isclose(
                a["expected_verified_success"], b["expected_verified_success"], abs_tol=1e-9, rel_tol=0):
            matched.append(index)
            matched_cost_deltas.append(a["expected_declared_cost"] - b["expected_declared_cost"])
    return {"checks": checks, "all_measured_retention_checks_passed": all(c["passed"] is True for c in checks.values()),
            "unassessed_checks": [name for name, check in checks.items() if check["passed"] is None],
            "matched_success_cost_comparison": {"cases": len(matched), "row_indices": matched,
                "mean_candidate_minus_v4_declared_cost": float(np.mean(matched_cost_deltas)) if matched else None},
            "promotion": False,
            "scope": "Only these fresh structured-simulation checks. No actual tools, historical twelve-check gate, broad language capability or default-policy promotion."}


def _error_breakdown(rows, report):
    counts = {"premature_stop": 0, "deferred_reachable": 0, "invalid_or_ineligible": 0, "wrong_eligible_action": 0}
    for row, decision in zip(rows, report["decisions"]):
        action = decision["action_id"]
        if not decision["eligible"]:
            counts["invalid_or_ineligible"] += 1
        if action == "STOP" and "STOP" not in row["target_ids"]:
            counts["premature_stop"] += 1
        elif action in ("NEEDS_CLARIFICATION", "NEEDS_INFORMATION") and action not in row["target_ids"]:
            counts["deferred_reachable"] += 1
        elif not decision["correct"] and decision["eligible"]:
            counts["wrong_eligible_action"] += 1
    return counts


def paired_agreement(candidate, parent):
    """Descriptive paired counts on identical cases, not an uncertainty bound."""
    counts = {"candidate_only_correct": 0, "v4_only_correct": 0, "both_correct": 0, "both_incorrect": 0}
    a, b = candidate["decisions"], parent["decisions"]
    if len(a) != len(b):
        raise ValueError("Paired comparison requires identical case counts")
    for left, right in zip(a, b):
        if left["row_id"] != right["row_id"]:
            raise ValueError("Paired comparison requires identical ordered cases")
        key = ("both_correct" if left["correct"] and right["correct"] else
               "candidate_only_correct" if left["correct"] else
               "v4_only_correct" if right["correct"] else "both_incorrect")
        counts[key] += 1
    return {"cases": len(a), **counts, "scope": "Paired observed correctness; no confidence interval or significance claim."}


def execute(args, sealed):
    output = Path(args.output).resolve()
    protocol = sealed["protocol"]
    budget = Budget(args.max_seconds, args.max_forward_calls)
    mx.set_memory_limit(int(args.memory_gib * GIB))
    mx.set_cache_limit(128 * 1024 ** 2)
    mx.reset_peak_memory()
    started = time.perf_counter()
    try:
        models, inventory = {}, {}
        for name, item in protocol["artifacts"].items():
            budget.check()
            model, _, _, actual_sha = load_parent(item["path"])
            if actual_sha != item["model_sha256"]:
                raise ValueError("Loaded model hash differs from the sealed artifact")
            inventory[name] = model_inventory(model, item["parameters"])
            models[name] = BudgetedModel(model, budget)
        load_seconds = time.perf_counter() - started
        budget.check()
        if mx.get_peak_memory() / GIB > args.memory_gib:
            raise RuntimeError("Model loading exceeded the declared memory guideline")
        rows = corpus(args.cases, "final_audit", args.seed, allow_final=True)
        budget.check()
        corpus_hash = canonical_digest([{k: v for k, v in row.items() if k != "_encoded"} for row in rows])
        warmup = {}
        warmup_examples = min(args.batch_size, len(rows))
        public_inputs = tuple(mx.array(value) for value in batch_arrays(rows[:warmup_examples])[:3])
        mx.eval(*public_inputs)
        for name in TIMING_PROTOCOL["method_order"]:
            policy = models[name]
            policy.eval()
            warmup_started = time.perf_counter()
            # Exactly the three public inputs: targets never enter a forward call.
            result = policy(*public_inputs)
            warmup[name] = {"examples": warmup_examples, "forward_calls": 1,
                            "wall_seconds": time.perf_counter() - warmup_started}
            del result
        del public_inputs
        candidate = evaluate(models["candidate"], rows, batch_size=args.batch_size)
        parent = evaluate(models["v4_parent"], rows, batch_size=args.batch_size)
        rollouts = rollout_evaluate(models["candidate"], rows, parent_model=models["v4_parent"],
                                    count=args.rollouts, max_belief_nodes=args.max_belief_nodes)
        budget.check()
        if mx.get_peak_memory() / GIB > args.memory_gib:
            raise RuntimeError("Audit exceeded the declared memory guideline; no partial quality result")
        if artifact_manifest(args.run) != protocol["artifacts"] or source_hashes() != protocol["source_sha256"]:
            raise RuntimeError("A frozen artifact or dependency changed during this final audit")
        for report in (candidate, parent):
            report["scope"] = "Sealed fresh final-audit next-action agreement on the predeclared structured simulations."
            report["audit_context"] = {"split": "final_audit", "seed": args.seed,
                                       "protocol_sha256": sealed["protocol_sha256"], "corpus_sha256": corpus_hash}
            report["error_breakdown"] = _error_breakdown(rows, report)
            for domain, metrics in report["by_domain"].items():
                domain_rows = [row for row in rows if row["domain"] == domain]
                domain_decisions = [decision for decision in report["decisions"] if decision["domain"] == domain]
                metrics["error_breakdown"] = _error_breakdown(domain_rows, {"decisions": domain_decisions})
        result = {"version": VERSION, "status": "completed", "scope": "fresh_final_structured_audit",
            "protocol_sha256": sealed["protocol_sha256"], "protocol": protocol,
            "corpus_sha256": corpus_hash, "inventory": inventory,
            "next_action": {"candidate": candidate, "v4_parent": parent},
            "paired_next_action": paired_agreement(candidate, parent),
            "rollouts": rollouts, "retention": comparison_checks(candidate, parent, rollouts),
            "wall_seconds": time.perf_counter() - started, "model_load_seconds": load_seconds,
            "warmup": warmup, "timing_protocol": TIMING_PROTOCOL,
            "forward_calls": budget.calls, "peak_mlx_gib": mx.get_peak_memory() / GIB,
            "frozen_inputs_verified_after_audit": True, "promotion": False,
            "limitations": [
                "These are controlled structured simulations, not arbitrary situations or actual software tools.",
                "Generated exact labels/values are withheld from model forward inputs; eligibility remains a supplied rule.",
                "Policy rollouts use only model choices at each observed state; an oracle never repairs their actions.",
                "Initially verified and zero-reference-success cases are counted separately from unfinished reachable completion.",
                "Declared action costs and measured decision time are separate; one Mac run does not establish precise timing superiority.",
                "Budget checks occur between model calls/stages; a running kernel is not forcibly interrupted.",
                "The original v4 already failed some historical gates. Matching it here does not erase those failures.",
                "No confidence interval or broad reliability claim follows from this bounded synthetic case set.",
            ]}
        write_json(output, result)
        print(json.dumps({"status": "audit_completed", "promotion": False,
                          "candidate_parameters": inventory["candidate"]["parameters"],
                          "candidate_next_action_accuracy": candidate["next_action_accuracy"],
                          "v4_next_action_accuracy": parent["next_action_accuracy"],
                          "retention": result["retention"], "output": str(output)}), flush=True)
        return result
    except Exception as error:
        write_json(output, {"version": VERSION, "status": "failed", "promotion": False,
                           "protocol_sha256": sealed["protocol_sha256"],
                           "error": f"{type(error).__name__}: {error}",
                           "forward_calls": budget.calls, "wall_seconds": time.perf_counter() - started,
                           "audit_consumed": True, "partial_quality_result": False})
        raise


def main(args):
    with local_compute():
        sealed, declaration_only = prepare(args)
        if declaration_only:
            print(json.dumps({"status": "audit_sealed", "cases_generated": 0, "models_loaded": 0,
                              "protocol_sha256": sealed["protocol_sha256"],
                              "next": "Repeat identical arguments with --run-audit instead of --declare"}), flush=True)
            return sealed
        return execute(args, sealed)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--declare", action="store_true")
    p.add_argument("--run-audit", action="store_true")
    p.add_argument("--cases", type=int, default=512)
    p.add_argument("--rollouts", type=int, default=40)
    p.add_argument("--seed", type=int, default=810000001)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-belief-nodes", type=int, default=256)
    p.add_argument("--max-forward-calls", type=int, default=4096)
    p.add_argument("--max-seconds", type=float, default=1800)
    p.add_argument("--memory-gib", type=float, default=28)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
