"""Neural choice only: never silently replace it with a planner's answer."""
import hashlib
import json
import time
from pathlib import Path


class NeuralPolicy:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.model = None
        self.report = None

    def load(self):
        if self.model is not None:
            return
        import mlx.core as mx
        from agent_training.model import GoalPolicy, PolicyConfig
        report_path = self.run_dir / "report.json"
        checkpoint = self.run_dir / "model.safetensors"
        if not report_path.exists() or not checkpoint.exists():
            raise ValueError("The v2 checkpoint is not trained yet. See the local training status.")
        report = json.loads(report_path.read_text())
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != report["checkpoint_sha256"]:
            raise ValueError("The checkpoint does not match its training report.")
        model = GoalPolicy(PolicyConfig(**report["config"]))
        model.load_weights(str(checkpoint))
        model.eval()
        mx.eval(model.parameters())
        self.report, self.model = report, model

    def choose(self, scenario, remaining=5):
        import mlx.core as mx
        from agent_training.features import encode
        self.load()
        start = time.perf_counter()
        x, valid, eligible, candidates = encode(scenario, max_depth=max(1, min(5, remaining)),
                                                feature_dim=self.model.config.input_dim)
        probabilities = mx.softmax(self.model(mx.array(x[None]), mx.array(valid[None]),
                                              mx.array(eligible[None]))[0])
        mx.eval(probabilities)
        values = probabilities.tolist()
        chosen = max(range(len(candidates)), key=lambda i: values[i])
        return {"action_id": candidates[chosen]["id"], "action_name": candidates[chosen]["name"],
                "preference": values[chosen], "decision_ms": (time.perf_counter()-start)*1000,
                "generated_tokens": 0}
