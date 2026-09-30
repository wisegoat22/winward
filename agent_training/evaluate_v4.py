"""One-time v4 reliability audit: strict retention, fresh seeds, diagnostic v3.

No instance of the reserved final generator is constructed by protocol sealing.
The gate logic remains the v3 twelve-check policy, under the new candidate name.
All v3 outcomes are development evidence, never a fresh v4 promotion baseline.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import time

from agent_lab.belief import plan, verified
from .evaluate_v3 import (PROTOCOL as V3_PROTOCOL, _canonical, _write_rows,
    cheap_action, expected_episode, sample_episode, summarize_samples,
    matched_success_efficiency, graph_cost_retention, graph_rollouts,
    promotion_gate as v3_promotion_gate)

PROTOCOL = deepcopy(V3_PROTOCOL)
PROTOCOL.update({
    "version": "winward-v4-gate-1",
    "seeds": {"belief": 430_000_000, "legacy": 440_000_000,
              "curriculum": 441_000_000, "goal_changes": 442_000_000, "tools": 450_000_000},
    "counts": {"belief_problems": 240, "episodes_per_problem": 4,
               "graph_rows_per_suite": 2400, "graph_rollouts_per_suite": 300,
               "actual_tool_fixtures": 40, "goal_change_pairs": 200},
    "candidate": "v4",
    "diagnostic_comparison": "Frozen v3 is evaluated on the same fresh v4 cases. It is diagnostic only; beating v3 does not satisfy any gate.",
    "development_evidence": "V3 audit cases and outcomes are known development evidence for v4. They must not be described as blind v4 evaluation.",
    "belief_distribution": {
        "generator": "agent_training.belief_data_v4.make_problem(seed, 'test')",
        "anchors": ["prepared_diagnosis", "missing_requirements", "noisy_diagnosis",
                    "stochastic_verification", "reversible_repair", "goal_revision",
                    "known_cause", "completed", "unreachable", "composed",
                    "preparation_chain", "efficient_choice"],
        "holdout": "Fresh seeds and instances, with the same developer-known compositional anchors across splits. Not an unseen-family benchmark.",
        "diagnostics": "Per-anchor expected and sampled results, initially verified counts, and exact-planner zero-success-within-five counts reveal mixture easiness. All problems remain in all gates.",
    },
    "sample_dependence": "Four matched stochastic episodes share each problem. The 960 episodes are not 960 independent task draws. Exact expectation integrates declared outcomes; it does not estimate reliability outside the 240 generator instances. No independence-based confidence interval is reported.",
    "artifact_freeze": "Training setup and audit protocol precede training. Frozen selected weights, report and config plus generator, predictor, trainer, evaluator and mechanics source hashes are recorded before final generation; unchanged sources, candidate and baseline artifacts are verified after the audit. One exclusive started marker prevents reruns.",
    "timing": V3_PROTOCOL["timing"] + " Model loading and initial warm-up are reported separately and excluded from steady-state decision costs. Method order rotates per problem; one shared-Mac session is not a precision performance benchmark.",
})


def protocol_sha256():
    return hashlib.sha256(_canonical(PROTOCOL)).hexdigest()


def seal_protocol(run):
    """Create the protocol exclusively, without importing a final-data generator."""
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    value = {"protocol": PROTOCOL, "protocol_sha256": protocol_sha256()}
    with (run / "audit-protocol.json").open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    return value


def promotion_gate(graphs, tools, belief, frozen_report):
    """Use the unchanged v3 strict gate; v3 comparison cannot affect promotion."""
    def candidate_alias(value):
        if isinstance(value, dict):
            result = {key: candidate_alias(item) for key, item in value.items() if key != "v3"}
            if "v4" in result:
                result["v3"] = result.pop("v4")
            return result
        if isinstance(value, list):
            return [candidate_alias(item) for item in value]
        return value
    result = v3_promotion_gate(candidate_alias(graphs), candidate_alias(tools),
                               candidate_alias(belief), frozen_report)
    result.update(protocol_version=PROTOCOL["version"], protocol_sha256=protocol_sha256())
    if not result["promoted"]:
        result["default_policy_action"] = "Retain prior default policies; do not silently route old tasks to v4"
    return result


def source_manifest(repo):
    """Freeze model/generator/evaluator code and execution mechanics, not UI files."""
    repo = Path(repo)
    paths = list((repo / "agent_training").glob("*.py"))
    paths += [repo / "agent_lab" / name for name in
              ("__init__.py", "belief.py", "predictor.py", "runner.py", "benchmark.py", "sandbox.py")]
    return {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths) if path.is_file()}


def frozen_artifacts(run):
    run = Path(run)
    names = ("model.safetensors", "report.json", "config.json")
    return {name: hashlib.sha256((run / name).read_bytes()).hexdigest() for name in names}


def validate_frozen_run(run):
    """Fail before loading any final-data generator if provenance is incomplete."""
    run = Path(run)
    sealed_path = run / "audit-protocol.json"
    sealed = json.loads(sealed_path.read_text())
    if sealed != {"protocol": PROTOCOL, "protocol_sha256": protocol_sha256()}:
        raise ValueError("The predeclared protocol differs; do not alter gates after outcomes")
    if any((run / name).exists() for name in ("evaluation.json", "audit-started.json", "belief-test.jsonl", "audit-source-manifest.json")):
        raise ValueError("This audit has already started. Preserve results; do not rerun to select outcomes.")
    report = json.loads((run / "report.json").read_text())
    if not report.get("checkpoint_frozen_before_test_generation") or report.get("test") is not None:
        raise ValueError("Expected frozen weights without test evaluation")
    if report.get("sealed_audit_protocol_sha256") != hashlib.sha256(sealed_path.read_bytes()).hexdigest():
        raise ValueError("The sealed protocol differs from the protocol recorded during training")
    if hashlib.sha256((run / "model.safetensors").read_bytes()).hexdigest() != report["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    return report


def assert_frozen_inputs(repo, runs, manifest):
    if source_manifest(repo) != manifest["source_sha256"]:
        raise RuntimeError("Audit dependency source changed during the final audit")
    for name, run in runs.items():
        if frozen_artifacts(run) != manifest["artifacts"][name]:
            raise RuntimeError(f"Frozen artifacts changed during the final audit: {name}")


def expected_summary(items):
    return {"problems": len(items),
        "verified_success_rate": statistics.mean(e["expected_verified_success"] for e in items),
        "mean_declared_tokens": statistics.mean(e["expected_declared_tokens"] for e in items),
        "mean_declared_action_ms": statistics.mean(e["expected_declared_action_ms"] for e in items),
        "mean_measured_decision_ms": statistics.mean(e["expected_measured_decision_ms"] for e in items),
        "mean_total_weighted_cost": statistics.mean(e["expected_total_weighted_cost"] for e in items)}


def audit_beliefs(policy, v3, problems, families, episodes_per_problem=4):
    if not problems or len(problems) != len(families) or len({p.id for p in problems}) != len(problems):
        raise ValueError("Belief audit requires unique nonempty problems with aligned families")
    methods = {"v4": lambda p, d: policy.predict_belief(p, remaining_horizon=d),
               "v3": lambda p, d: v3.predict_belief(p, remaining_horizon=d),
               "myopic_progress": lambda p, d: cheap_action(p, d),
               "information_first": lambda p, d: cheap_action(p, d, information_first=True)}
    expected = {name: [] for name in methods}
    episodes = {name: [] for name in methods}
    ceilings = []
    names = list(methods)
    for index, problem in enumerate(problems):
        reference_started = time.perf_counter()
        reference = plan(problem)
        ceilings.append({"problem_id": problem.id, "expected_verified_success": reference["expected_verified_success"],
                         "expected_declared_cost": reference["expected_cost"], "search_complete": reference["search_complete"],
                         "measured_planning_ms": (time.perf_counter()-reference_started)*1000})
        # Rotate to reduce one-sided cache or thermal timing effects.
        for name in names[index % len(names):] + names[:index % len(names)]:
            expected[name].append(expected_episode(problem, methods[name]))
            for repetition in range(episodes_per_problem):
                seed = PROTOCOL["seeds"]["belief"] + index * episodes_per_problem + repetition
                episodes[name].append(sample_episode(problem, methods[name], seed))
        if (index+1) % 20 == 0:
            print(f"Belief audit: {index+1}/{len(problems)} matched problems", flush=True)
    expected_measures = {name: expected_summary(items) for name, items in expected.items()}
    sampled_summary = {name: summarize_samples(items) for name, items in episodes.items()}
    strongest = min(("myopic_progress", "information_first"), key=lambda name: (
        -expected_measures[name]["verified_success_rate"], -sampled_summary[name]["success_rate"],
        sampled_summary[name]["mean_total_weighted_cost"], name))
    efficiency = {"baseline": strongest, **matched_success_efficiency(episodes["v4"], episodes[strongest])}
    by_family = {}
    for family in sorted(set(families)):
        indices = [i for i, value in enumerate(families) if value == family]
        ids = {problems[i].id for i in indices}
        by_family[family] = {
            "problems": len(indices),
            "initially_verified": sum(verified(problems[i].belief, problems[i].goal) for i in indices),
            "zero_success_within_horizon": sum(ceilings[i]["search_complete"] and ceilings[i]["expected_verified_success"] == 0 for i in indices),
            "expected": {name: expected_summary([items[i] for i in indices]) for name, items in expected.items()},
            "sampled": {name: summarize_samples([e for e in items if e["problem_id"] in ids]) for name, items in episodes.items()},
        }
    diagnostic = {
        "role": "Descriptive matched v3 comparison only; no gate uses v3 outcomes",
        "expected_completion_delta": expected_measures["v4"]["verified_success_rate"] - expected_measures["v3"]["verified_success_rate"],
        "sampled_completion_delta": sampled_summary["v4"]["success_rate"] - sampled_summary["v3"]["success_rate"],
        "paired_success_efficiency": matched_success_efficiency(episodes["v4"], episodes["v3"]),
    }
    return {"expected": expected_measures, "sampled": sampled_summary, "paired_efficiency": efficiency,
            "initially_verified_problems": sum(verified(p.belief, p.goal) for p in problems),
            "initially_unverified_problems": sum(not verified(p.belief, p.goal) for p in problems),
            "exact_planner_reference": {"expected_verified_success": statistics.mean(c["expected_verified_success"] for c in ceilings),
                "all_searches_complete": all(c["search_complete"] for c in ceilings),
                "mean_measured_planning_ms": statistics.mean(c["measured_planning_ms"] for c in ceilings)},
            "per_problem_expected": {name: [{"problem_id": problems[i].id, "family": families[i], **e} for i, e in enumerate(items)] for name, items in expected.items()},
            "by_family": by_family, "v3_diagnostic": diagnostic,
            "zero_success_within_horizon_problems": sum(c["search_complete"] and c["expected_verified_success"] == 0 for c in ceilings),
            "sample_dependence": PROTOCOL["sample_dependence"], "cost_note": PROTOCOL["timing"]}, episodes


def audit_tools(policy, baseline, v3):
    from agent_lab.benchmark import summarize
    from agent_lab.runner import run_episode
    specs = [{"seed": PROTOCOL["seeds"]["tools"] + index*100 + repeat,
              "kind": kind, "changed_goal": changed, "uncertain": uncertain}
             for index, (kind, changed, uncertain) in enumerate((k, c, u) for k in PROTOCOL["tool_kinds"] for c in (False, True) for u in (False, True))
             for repeat in range(5)]
    episodes = {name: [] for name in ("v4", "v3", "v2.1", "evidence_first")}
    names = list(episodes)
    for index, spec in enumerate(specs):
        for name in names[index % len(names):] + names[:index % len(names)]:
            episodes[name].append(run_episode(**spec, policy_name="evidence_first" if name == "evidence_first" else "neural",
                                             neural_policy=policy if name == "v4" else v3 if name == "v3" else baseline))
        if (index+1) % 10 == 0:
            print(f"Actual fixture audit: {index+1}/{len(specs)} matched tasks", flush=True)
    measures = {name: {**summarize(items), "verified_and_stopped_rate": sum(e["success"] and e["done"] for e in items)/len(items)}
                for name, items in episodes.items()}
    return {"methods": measures, "specification": specs,
            "scope": "Real files and subprocess checks on controlled generated interval/collection fixtures, using supplied patches. No code generation or arbitrary repository work.",
            "cost_note": "Wall time, tool execution and policy decision times measured; token costs remain declared estimates."}, episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/goalpolicy-v4")
    parser.add_argument("--v1-run", default="runs/goalpolicy-v1")
    parser.add_argument("--tools-run", default="runs/goalpolicy-v2-tools")
    parser.add_argument("--v3-run", default="runs/goalpolicy-v3")
    parser.add_argument("--declare-protocol", action="store_true")
    args = parser.parse_args()
    run = Path(args.run)
    if args.declare_protocol:
        print(json.dumps(seal_protocol(run), indent=2))
        return
    report = validate_frozen_run(run)
    from .model_v4 import GraphView, V4Policy
    from .model_v3 import GraphView as V3GraphView, V3Policy
    from .evaluate_v2 import fresh_goal_pairs, load_model, paired_goal_metrics
    from .evaluate import fresh_rows
    from .curriculum import generate_rows
    from .belief_data_v4 import make_problem, family_for_seed
    from agent_lab.predictor import NeuralPolicy
    from agent_lab.belief import example_problems
    started = time.perf_counter()
    repo = Path(__file__).resolve().parents[1]
    runs = {"v4": run, "v1": Path(args.v1_run), "v2.1": Path(args.tools_run), "v3": Path(args.v3_run)}
    manifest = {"source_sha256": source_manifest(repo),
                "artifacts": {name: frozen_artifacts(path) for name, path in runs.items()},
                "protocol_sha256": protocol_sha256(),
                "scope": "Source and frozen model/report/config artifacts recorded before any final generation."}
    policy = V4Policy(run)
    if hasattr(policy, "load"):
        policy.load()
    v3 = V3Policy(args.v3_run)
    v1, v1_report, v1_digest = load_model(Path(args.v1_run))
    tool_baseline = NeuralPolicy(args.tools_run)
    tool_baseline.load()
    # Existing demonstration only: no reserved final-instance construction.
    warmup_started = time.perf_counter()
    warmup = example_problems()[0]
    policy.predict_belief(warmup)
    v3.predict_belief(warmup)
    warmup_ms = (time.perf_counter()-warmup_started)*1000
    digest = report["checkpoint_sha256"]
    initial = {"checkpoint_sha256": digest, "protocol_sha256": protocol_sha256(),
               "sealed_audit_protocol_file_sha256": hashlib.sha256((run / "audit-protocol.json").read_bytes()).hexdigest(),
               "checkpoint_frozen_before_test_generation": True,
               "source_and_artifact_manifest": manifest,
               "model_setup_ms": (time.perf_counter()-started)*1000,
               "initial_belief_warmup_ms": warmup_ms,
               "baseline_sha256": {"v1": v1_digest, "v2.1": tool_baseline.report["checkpoint_sha256"],
                                   "v3": v3.report["checkpoint_sha256"]}}
    assert_frozen_inputs(repo, runs, manifest)
    with (run / "audit-started.json").open("x") as stream:
        json.dump(initial, stream, indent=2)
    with (run / "audit-source-manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
    # Every fresh generator call is below the checkpoint/protocol verification.
    problems = [make_problem(PROTOCOL["seeds"]["belief"] + i, "test") for i in range(PROTOCOL["counts"]["belief_problems"])]
    families = [family_for_seed(PROTOCOL["seeds"]["belief"] + i, "test") for i in range(len(problems))]
    _write_rows(run / "belief-test.jsonl", [{"family": family, **p.to_dict()} for p, family in zip(problems, families)])
    graph_rows = {"legacy": fresh_rows(PROTOCOL["seeds"]["legacy"], PROTOCOL["counts"]["graph_rows_per_suite"], "test"),
                  "curriculum": generate_rows(PROTOCOL["counts"]["graph_rows_per_suite"], "test", PROTOCOL["seeds"]["curriculum"])}
    graphs = {}
    for name, rows in graph_rows.items():
        _write_rows(run / f"{name}-test.jsonl", rows)
        graphs[name] = {"methods": {"v4": graph_rollouts(GraphView(policy.model), rows, PROTOCOL["counts"]["graph_rollouts_per_suite"]),
                                    "v1": graph_rollouts(v1, rows, PROTOCOL["counts"]["graph_rollouts_per_suite"]),
                                    "v3": graph_rollouts(V3GraphView(v3.model), rows, PROTOCOL["counts"]["graph_rollouts_per_suite"])}}
        print(f"{name} graph audit complete", flush=True)
    graphs["legacy"]["paired_success_action_cost"] = graph_cost_retention(
        graphs["legacy"]["methods"]["v4"], graphs["legacy"]["methods"]["v1"])
    pairs, scanned = fresh_goal_pairs(PROTOCOL["seeds"]["goal_changes"], PROTOCOL["counts"]["goal_change_pairs"])
    _write_rows(run / "goal-change-test.jsonl", [{"first": a.to_dict(), "second": b.to_dict(),
                "optimal_ids": [labels[0]["optimal_action_ids"], labels[1]["optimal_action_ids"]]}
                for a, b, labels in pairs])
    graphs["goal_changes"] = {"methods": {"v4": paired_goal_metrics(GraphView(policy.model), pairs),
                                          "v3": paired_goal_metrics(V3GraphView(v3.model), pairs),
                                          "v2.1": paired_goal_metrics(tool_baseline.model, pairs)},
                              "candidates_examined": scanned,
                              "selection": "Model-independent: both goals unfinished, reachable, with disjoint optimal next-action sets. Same graph, state, and costs per pair."}
    print("Changed-goal retention audit complete", flush=True)
    belief, belief_traces = audit_beliefs(policy, v3, problems, families, PROTOCOL["counts"]["episodes_per_problem"])
    _write_rows(run / "belief-audit-traces.jsonl", [{"method": name, **e} for name, items in belief_traces.items() for e in items])
    actual_tools, tool_traces = audit_tools(policy, tool_baseline, v3)
    _write_rows(run / "actual-tool-audit-traces.jsonl", [{"method": name, **e} for name, items in tool_traces.items() for e in items])
    assert_frozen_inputs(repo, runs, manifest)
    result = {**initial, "graph_retention": graphs, "belief": belief, "actual_tools": actual_tools,
              "promotion": promotion_gate(graphs, actual_tools, belief, report),
              "protocol": PROTOCOL, "elapsed_seconds": time.perf_counter()-started,
              "limitations": "Numeric declared-world decisions, shared generator anchors and controlled supplied-patch fixtures. V3 audit outcomes are development evidence. Fresh v4 seeds are not proof of general real-world competence. Goal/permission masks are supplied rules. Sampled episodes are clustered by problem; costs are only compared on matched successes and do not erase completion failures. One shared-Mac timing session is not a precision benchmark.",
              "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob("*-test.jsonl")},
              "trace_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob("*-audit-traces.jsonl")}}
    with (run / "evaluation.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"promotion": result["promotion"], "belief_expected": belief["expected"],
                      "belief_sampled": belief["sampled"], "actual_tools": actual_tools["methods"]}, indent=2))


if __name__ == "__main__":
    main()
