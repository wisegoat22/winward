from agent_lab.runner import run_episode


class AlwaysDefer:
    def choose(self, observation, remaining):
        return {"action_id":"NEEDS_CLARIFICATION", "preference":1.0}


def test_neural_failure_is_recorded_without_reference_correction():
    result=run_episode(seed=51,kind="boundary",policy_name="neural",neural_policy=AlwaysDefer())
    assert not result["success"] and result["deferred"]
    assert [e["action_id"] for e in result["trace"]]==["NEEDS_CLARIFICATION"]
    assert result["wall_ms"]>=result["decision_ms"]>=0
    assert result["generated_model_tokens"]==0


def test_baseline_uses_observations_and_actual_checks_across_goal_change():
    result=run_episode(seed=51,kind="boundary",changed_goal=True,uncertain=True,policy_name="evidence_first")
    assert result["success"] and result["done"] and not result["horizon_exhausted"]
    assert result["steps"]<=16 and result["trace"][-1]["action_id"]=="STOP"
    assert any(e["goal_changed"] and not e["success"] for e in result["trace"])
    assert result["checks"]>=2
    assert result["wall_ms"]>=result["tool_elapsed_ms"]+result["decision_ms"]
