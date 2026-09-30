"""Local-only inference for the newly trained policy, separate from Qwen."""
import json
import time
from pathlib import Path

from .features import encode
from .simulator import Scenario, make_scenario, search

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "goalpolicy-v1"


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


class PolicyRuntime:
    def __init__(self, run_dir=RUN_DIR):
        self.run_dir = Path(run_dir)
        self.model = None
        self.loaded_sha = None

    def status(self):
        report = read_json(self.run_dir / "report.json")
        progress = read_json(self.run_dir / "status.json") or {"status": "not_trained"}
        # Prefer the fresh-instance audit. Original test seeds overlap an older
        # prototype's audit, and its task families informed development.
        fresh_audit = self.run_dir / "test-fresh-90000000-evaluation.json"
        audit_path = fresh_audit if fresh_audit.exists() else self.run_dir / "evaluation.json"
        evaluation = read_json(audit_path)
        if evaluation and report and evaluation.get("checkpoint_sha256") != report["checkpoint_sha256"]:
            evaluation = None
        neural_metrics = (evaluation or {}).get("decision_metrics", {}).get("neural", {})
        result = {
            "status": "ready" if report else progress["status"],
            "model_name": report["model_name"] if report else "GoalPolicy",
            "parameter_count": report["parameter_count"] if report else progress.get("parameter_count"),
            "training_origin": "Trained from random initialization on this Mac. Labels come from deterministic search. No Qwen weights, adapters, text embeddings, or model-generated labels.",
            "training": {"train_examples": report["training_cases"],
                         "validation_examples": report["validation_cases"],
                         "test_examples": report["test_cases"],
                         "steps": report["steps"], "selected_step": report["best_validation"]["step"],
                         "elapsed_seconds": report["elapsed_seconds"]} if report else progress,
            "metrics": {"test_accuracy": neural_metrics.get("all", {}).get("accuracy", report["test"]["accuracy"]),
                        "nontrivial_test_accuracy": neural_metrics.get("nontrivial", {}).get("accuracy", report["test"]["nontrivial_accuracy"])} if report else {},
            "limitation": "A small research prototype for structured synthetic software-agent tasks. It is not a 4B language model, cannot read arbitrary text, and does not execute tools. Model preferences are not calibrated win probabilities. Search consequences are exact only inside the supplied simulation.",
            "evaluation": evaluation,
            "evaluation_file": audit_path.name if evaluation else None,
            "evaluation_provenance": (
                "Fresh simulated test instances, separate from training and validation. These task families were examined during prototype development; this is not a test of entirely new task types."
                if evaluation and audit_path == fresh_audit else
                "Saved synthetic test split. Its task families and nearby instance seeds were examined during prototype development."
            ),
        }
        if report:
            result["checkpoint_sha256"] = report["checkpoint_sha256"]
            result["test_details"] = neural_metrics or report["test"]
        return result

    def examples(self):
        examples = []
        # These fixtures are chosen by reference-plan properties, never by the
        # learned model's answers. They may expose model errors.
        requests = [
            ("nested_dependencies", "Follow dependencies before delivering", lambda s, p: p["outcome"] == "plan" and p["depth"] >= 4),
            ("changed_goal", "Ignore work for an outdated goal", lambda s, p: p["outcome"] == "plan" and p["depth"] >= 3),
            ("reversible_trap", "Check the consequences of a cheap shortcut", lambda s, p: p["outcome"] == "plan" and p["depth"] >= 3),
            ("changed_goal", "Stop when the requested goal is met", lambda s, p: p["outcome"] == "stop"),
            ("nested_dependencies", "Recognize when the supplied plan is insufficient", lambda s, p: p["outcome"] == "needs_clarification"),
        ]
        for family, title, predicate in requests:
            for seed in range(9000000, 9000500):
                scenario = make_scenario(seed, "test", family)
                reference = search(scenario)
                if predicate(scenario, reference):
                    examples.append({"id": f"{family}-{seed}", "title": title,
                                     "summary": "A synthetic software-agent state. Change the facts or the goal and compare the learned choice with the five-step search.",
                                     "scenario": scenario.to_dict()})
                    break
        return {"examples": examples}

    def decide(self, scenario, token_weight=1.0, latency_weight=0.01, max_depth=5):
        import mlx.core as mx
        from .model import GoalPolicy, PolicyConfig

        scenario = Scenario.from_dict(scenario) if isinstance(scenario, dict) else scenario
        report = read_json(self.run_dir / "report.json")
        if report is None:
            raise ValueError("Training has not produced a completed checkpoint yet.")
        if self.loaded_sha != report["checkpoint_sha256"]:
            self.model = GoalPolicy(PolicyConfig(**report["config"]))
            self.model.load_weights(str(self.run_dir / "model.safetensors"))
            self.model.eval()
            mx.eval(self.model.parameters())
            self.loaded_sha = report["checkpoint_sha256"]
        started = time.perf_counter()
        x, valid, eligible, candidates = encode(scenario, token_weight, latency_weight, max_depth,
                                                feature_dim=self.model.config.input_dim)
        logits = self.model(mx.array(x[None]), mx.array(valid[None]), mx.array(eligible[None]))[0]
        probabilities = mx.softmax(logits)
        mx.eval(probabilities)
        scores = probabilities.tolist()
        winner = max(range(len(candidates)), key=lambda i: scores[i])
        latency = (time.perf_counter() - started) * 1000
        planner_started = time.perf_counter()
        planner = search(scenario, max_depth=max_depth, token_weight=token_weight, latency_weight=latency_weight)
        planner_latency = (time.perf_counter() - planner_started) * 1000
        choice = candidates[winner]
        return {
            "model_name": report["model_name"], "parameter_count": report["parameter_count"],
            "decision": {"id": choice["id"], "name": choice["name"], "probability": scores[winner]},
            "probabilities": [{"id": a["id"], "name": a["name"], "probability": scores[i],
                               "eligible": bool(eligible[i])} for i, a in enumerate(candidates)],
            "model_latency_ms": round(latency, 3), "planner_latency_ms": round(planner_latency, 3),
            "planner": planner,
            "agrees_with_planner": choice["id"] in planner["optimal_action_ids"],
            "hard_rules": "Permissions, current preconditions, and stop-when-complete are enforced in code. The model ranks the remaining candidates.",
        }
