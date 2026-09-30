"""Local v2 lab; all executable tasks are generated, bounded fixtures."""
import json
from pathlib import Path

from .predictor import NeuralPolicy

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "goalpolicy-v2-tools"


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


class LabRuntime:
    def __init__(self, run_dir=RUN_DIR):
        self.run_dir = Path(run_dir)
        self.policy = NeuralPolicy(self.run_dir)

    def status(self):
        report = read_json(self.run_dir / "report.json")
        evaluation = read_json(self.run_dir / "evaluation.json")
        sandbox_audit = read_json(self.run_dir / "sandbox-audit.json")
        # Never display another checkpoint's audit beside the active model.
        for name, audit in (("evaluation", evaluation), ("sandbox", sandbox_audit)):
            if audit and (not report or audit.get("checkpoint_sha256") != report["checkpoint_sha256"]):
                if name == "evaluation":
                    evaluation = None
                else:
                    sandbox_audit = None
        return {
            "status": "ready" if report else "not_trained",
            "training": report,
            "progress": read_json(self.run_dir / "status.json"),
            "evaluation": evaluation,
            "sandbox_audit": sandbox_audit,
            "previous_experiment": {
                "model": "Winward v2, before training on observed tool outcomes",
                "evaluation": read_json(ROOT / "reports" / "v2" / "evaluation.json"),
                "sandbox_audit": read_json(ROOT / "reports" / "v2" / "sandbox.json"),
            },
            "repository": "https://github.com/wisegoat22/winward",
            "limitations": "A structured policy with supplied patch candidates, not a model that reads arbitrary code or writes patches. Uncertainty planning is a separate algorithmic reference. Coding tasks use trusted generated files in temporary directories, not an operating-system security sandbox.",
        }

    def examples(self):
        from .belief import examples
        from .sandbox import SUPPORTED_KINDS
        return {"uncertainty": [{k:v for k,v in item.items() if k != "problem"} for item in examples()],
                "task_kinds": list(SUPPORTED_KINDS)}

    def uncertainty(self, example_id, max_depth=5):
        from .belief import examples, plan
        example = next((x for x in examples() if x["id"] == example_id), None)
        if example is None:
            raise ValueError("Unknown uncertainty example.")
        return {"example": {k:v for k,v in example.items() if k != "problem"},
                "reference": plan(example["problem"], max_depth=max_depth),
                "method": "Algorithmic search over possible worlds and observable outcomes; no neural model and no tool execution."}

    def sandbox(self, seed, kind, changed_goal=False, uncertain=False, policy="neural"):
        from .runner import run_episode
        return run_episode(seed=seed, kind=kind, changed_goal=changed_goal, uncertain=uncertain,
                           policy_name=policy, neural_policy=self.policy, max_actions=16)
