"""Serve the v3 candidate separately from retained v1 and v2 checkpoints."""
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "goalpolicy-v3"


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


class V3Runtime:
    def __init__(self, run_dir=RUN_DIR):
        self.run_dir = Path(run_dir)
        self.policy = None

    def status(self):
        report = read_json(self.run_dir / "report.json")
        progress = read_json(self.run_dir / "status.json")
        audit = read_json(self.run_dir / "evaluation.json")
        if audit and (not report or audit.get("checkpoint_sha256") != report.get("checkpoint_sha256")):
            audit = None
        return {"status": "ready" if report else (progress or {}).get("status", "not_trained"), "training": report,
                "progress": progress, "evaluation": audit,
                "retained_versions": ["GoalPolicy v1", "Winward v2.1 tools candidate"],
                "limitation": "A small structured policy using supplied possible worlds and action effects. It does not infer arbitrary real-world consequences or read natural-language tasks. Promotion is separate from having a trained checkpoint; earlier versions remain available.",
                "repository": "https://github.com/wisegoat22/winward"}

    def examples(self):
        from .belief import examples
        return {"examples": [{k:v for k,v in item.items() if k != "problem"} for item in examples()]}

    def _example(self, example_id):
        from .belief import examples
        found = next((e for e in examples() if e["id"] == example_id), None)
        if found is None:
            raise ValueError("Unknown v3 example")
        return found

    def _policy(self):
        if self.policy is None:
            if not (self.run_dir / "report.json").exists():
                raise ValueError("V3 training has not produced a frozen checkpoint yet.")
            from agent_training.model_v3 import V3Policy
            self.policy = V3Policy(self.run_dir)
        return self.policy

    def decide(self, example_id, max_depth=5):
        from .belief import plan
        example = self._example(example_id)
        policy = self._policy()
        started = time.perf_counter()
        chosen = policy.predict_belief(example["problem"], remaining_horizon=max_depth)
        decision_ms = (time.perf_counter() - started) * 1000
        reference = plan(example["problem"], max_depth=max_depth)
        actions = {a.id:a.name for a in example["problem"].actions}
        agrees = chosen == reference["chosen_action_id"]
        score = next((s for s in reference["action_scores"] if s["action_id"] == chosen), None)
        if score and "expected_verified_success" in score:
            agrees = all(math.isclose(score[k], reference[k], abs_tol=1e-9, rel_tol=0)
                         for k in ("expected_verified_success", "expected_cost", "expected_steps"))
        return {"example": {k:v for k,v in example.items() if k != "problem"},
                "problem": example["problem"].to_dict(),
                "learned_choice": {"action_id":chosen, "action_name":actions.get(chosen, chosen),
                                   "decision_ms":decision_ms, "generated_tokens":0},
                "reference": reference, "agrees_with_reference":agrees,
                "method": "The neural model chooses first. Search is displayed afterward for comparison and never replaces its answer."}

    def episode(self, example_id, seed=43001, max_depth=5):
        from agent_training.evaluate_v3 import sample_episode
        example = self._example(example_id)
        policy = self._policy()
        return sample_episode(example["problem"],
                              lambda p, remaining: policy.predict_belief(p, remaining_horizon=remaining),
                              seed=seed, horizon=max_depth)
