"""Expose the v4 reliability candidate without replacing earlier experiments."""
from pathlib import Path

from .runtime_v3 import V3Runtime

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "goalpolicy-v4"


def demonstrations():
    """Existing demos plus two hand-written preparation cases, outside the audit."""
    from .belief import BeliefAction, BeliefProblem, Hypothesis, Outcome, examples
    result = examples()
    ready, repaired, checked, blocked, access = 1, 2, 4, 8, 16
    for chain in (False, True):
        actions = [BeliefAction("prepare", "Prepare the workspace", requires=access, forbids=ready,
                               outcomes=(Outcome(1., "workspace_ready", sets=ready),), tokens=2, latency_ms=5),
                   BeliefAction("inspect", "Identify the cause", requires=ready,
                               outcomes_by_world={"parser": (Outcome(1., "parser_cause"),),
                                                  "boundary": (Outcome(1., "boundary_cause"),)},
                               tokens=3, latency_ms=8)]
        for cause in ("parser", "boundary"):
            actions.append(BeliefAction(f"repair_{cause}", f"Repair the {cause} fault", requires=ready,
                           forbids=blocked | repaired,
                           outcomes_by_world={w: (Outcome(1., "repair_applied", sets=repaired),)
                                              if w == cause else (Outcome(1., "repair_failed", sets=blocked),)
                                              for w in ("parser", "boundary")}, tokens=5, latency_ms=10))
        actions.append(BeliefAction("verify", "Verify the acceptance checks", requires=repaired, forbids=blocked | checked,
                                   outcomes=(Outcome(1., "checks_passed", sets=checked),), tokens=3, latency_ms=10))
        if chain:
            actions.insert(0, BeliefAction("access", "Obtain workspace access", forbids=access,
                           outcomes=(Outcome(1., "access_ready", sets=access),), tokens=1, latency_ms=3))
        identity = "preparation_chain" if chain else "prepare_then_diagnose"
        problem = BeliefProblem(identity, ("ready", "repaired", "checked", "blocked", "access"), checked,
                   tuple(Hypothesis(w, 0 if chain else access, .5) for w in ("parser", "boundary")), tuple(actions),
                   "A hand-written demonstration: preparation enables diagnosis and a verifiable repair.")
        result.append({"id": identity, "title": "Two necessary preparation steps" if chain else "Prepare before diagnosis",
                       "description": "Reach verified completion in five actions." if chain else "Preparation has no immediate goal gain, but unlocks the useful actions.",
                       "problem": problem})
    return result


class V4Runtime(V3Runtime):
    def __init__(self, run_dir=RUN_DIR):
        super().__init__(run_dir)

    def status(self):
        status = super().status()
        status["retained_versions"] = ["GoalPolicy v1", "Winward v2.1 tools candidate", "Frozen Winward v3"]
        status["limitation"] = (
            "A small structured policy using supplied possible worlds and action effects. "
            "The v4 experiment trains on broader combinations and rehearses earlier skills. "
            "A trained checkpoint is not a promotion: the fresh audit must pass every check. "
            "Earlier experiments remain available."
        )
        return status

    def examples(self):
        return {"examples": [{k: v for k, v in item.items() if k != "problem"} for item in demonstrations()]}

    def _example(self, example_id):
        found = next((item for item in demonstrations() if item["id"] == example_id), None)
        if found is None:
            raise ValueError("Unknown v4 example")
        return found

    def _policy(self):
        if self.policy is None:
            if not (self.run_dir / "report.json").exists():
                raise ValueError("V4 training has not produced a frozen checkpoint yet.")
            from agent_training.model_v4 import V4Policy
            self.policy = V4Policy(self.run_dir)
        return self.policy
