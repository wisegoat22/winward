"""Run a policy on real generated Python fixtures, with no oracle corrections."""
import random
import time

from agent_training.features import DEFER, STOP
from .sandbox import SandboxTask


def evidence_first(scenario):
    """Simple observed-state baseline; never inspect hidden candidate correctness."""
    if scenario.goal_met():
        return STOP
    eligible = {a.id: a for a in scenario.actions if a.eligible(scenario.state)}
    for action_id in ("clarify", "run_tests", "inspect"):
        if action_id in eligible:
            return action_id
    patches = [a for a in eligible.values() if a.id.startswith("apply_candidate_")]
    if patches:
        return min(patches, key=lambda a: (a.cost(), a.id)).id
    return DEFER


def run_episode(*, seed, kind, changed_goal=False, uncertain=False,
                policy_name="neural", neural_policy=None, max_actions=16):
    if policy_name not in ("neural", "evidence_first", "random"):
        raise ValueError("Unknown policy")
    if isinstance(max_actions, bool) or not 1 <= max_actions <= 16:
        raise ValueError("An episode supports at most 16 actions")
    if policy_name == "neural" and neural_policy is None:
        raise ValueError("A trained neural policy is required")
    rng = random.Random(seed + 5701)
    started = time.perf_counter()
    decision_total = 0.0
    with SandboxTask(seed, kind, changed_goal, uncertain) as task:
        for step in range(max_actions):
            observation = task.observe()
            deciding = time.perf_counter()
            if policy_name == "neural":
                choice = neural_policy.choose(observation, remaining=min(5, max_actions-step))
                action_id = choice["action_id"]
            elif policy_name == "evidence_first":
                action_id = evidence_first(observation)
                choice = {"action_id": action_id}
            else:
                action_id = STOP if observation.goal_met() else rng.choice(
                    [a.id for a in observation.actions if a.eligible(observation.state)] + [DEFER])
                choice = {"action_id": action_id}
            decision_ms = (time.perf_counter()-deciding)*1000
            decision_total += decision_ms
            event = task.step(action_id)
            event["decision_ms"] = decision_ms
            event["neural_preference"] = choice.get("preference")
            event["planning_horizon"] = min(5, max_actions-step)
            if task.done:
                break
        result = task.summary()
        result.update(policy=policy_name, trace=task.trace,
                      decision_ms=decision_total, generated_model_tokens=0,
                      max_actions=max_actions, replanning_horizon=5,
                      horizon_exhausted=not task.done,
                      outcome="verified" if task.success else "deferred" if result["deferred"] else "budget_exhausted",
                      policy_scope="The model chooses tools and supplied candidate patches using numeric state. It does not read or generate Python code. No planner corrects its choices.")
    result["wall_ms"] = (time.perf_counter()-started)*1000
    return result
