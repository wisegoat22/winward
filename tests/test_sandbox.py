"""Contract and real-process tests for the controlled coding fixture harness."""

import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

from agent_lab.sandbox import (
    CANDIDATES, FAILED, INSPECTED, PASSED, REQUIREMENTS, SUPPORTED_KINDS,
    VERIFY, SandboxTask,
)
from agent_training.features import encode
from agent_training.simulator import STOP


def evidence_first(task):
    """Baseline uses only public observations; no fixture internals or answers."""
    observation = task.observe()
    eligible = {action.id for action in observation.actions if action.eligible(observation.state)}
    if observation.goal_met():
        return STOP
    if "clarify" in eligible:
        return "clarify"
    if "run_tests" in eligible:
        return "run_tests"
    if "inspect" in eligible:
        return "inspect"
    for candidate in ("apply_candidate_a", "apply_candidate_b"):
        if candidate in eligible:
            return candidate
    raise AssertionError("Fixture unexpectedly has no useful next action")


@pytest.mark.parametrize("kind", SUPPORTED_KINDS)
@pytest.mark.parametrize("changed,uncertain", [(False, False), (True, True)])
def test_public_evidence_baseline_completes_actual_checks(kind, changed, uncertain):
    with SandboxTask(4, kind, changed, uncertain, timeout_seconds=0.1) as task:
        for _ in range(16):
            task.step(evidence_first(task))
            if task.done:
                break
        assert task.success and task.done
        assert task.summary()["checks"] >= 1
        assert all(e["accepted"] for e in task.trace)
        assert task.observe().goal_met()
        json.dumps(task.trace)
        if changed:
            assert task.summary()["goal_revision"] == 1
            assert any(e["goal_changed"] for e in task.trace)


def test_hidden_bug_and_candidate_answers_do_not_enter_neural_features():
    tasks = [SandboxTask(1, "boundary"), SandboxTask(93, "rounding"),
             SandboxTask(5, "already_correct"), SandboxTask(0, "timeout")]
    first = encode(tasks[0].observe())
    for task in tasks[1:]:
        features = encode(task.observe())
        for expected, actual in zip(first[:3], features[:3]):
            np.testing.assert_array_equal(expected, actual)
        assert "observed_source" not in task.observe().context
        assert "supplied_candidates" not in task.observe().context
    # Reveal source legitimately, but do not add a hidden "correct candidate" bit.
    with SandboxTask(1, "boundary") as left, SandboxTask(2, "boundary") as right:
        left.step("inspect")
        right.step("inspect")
        np.testing.assert_array_equal(encode(left.observe())[0], encode(right.observe())[0])
        assert "observed_source" in left.observe().context


def test_failing_real_check_replaces_optimistic_forecast_and_cannot_finish():
    with SandboxTask(3, "boundary") as task:
        event = task.step("run_tests")
        assert event["result"]["returncode"] != 0
        assert "AssertionError" in event["result"]["stderr"]
        assert event["forecast_state"] & PASSED
        assert not event["forecast_matched"]
        assert task.observe().state & FAILED
        assert not task.observe().state & PASSED
        assert not task.observe().state & VERIFY
        assert not task.success
        assert not task.step(STOP)["accepted"]
        assert not task.done
        assert not task.step("run_tests")["accepted"]
        assert "<fixture>" in event["result"]["stderr"]
        assert str(task.directory) not in event["result"]["stderr"]


def test_goal_revision_invalidates_a_real_pass_and_requires_new_evidence():
    with SandboxTask(2, "already_correct", changed_goal=True) as task:
        event = task.step("run_tests")
        assert event["result"]["passed"]
        assert event["goal_changed"]
        assert event["result"]["stale_result"]
        assert not task.success
        assert not task.observe().goal_met()
        assert not task.step(STOP)["accepted"]
        assert not task.step("run_tests")["accepted"]
        task.step("clarify")
        assert not task.step("run_tests")["result"]["passed"]
        assert not task.success


def test_ambiguous_goal_requires_clarification_before_code_or_checks():
    with SandboxTask(0, uncertain=True) as task:
        assert not task.observe().state & REQUIREMENTS
        assert not task.step("inspect")["accepted"]
        assert not task.step("run_tests")["accepted"]
        event = task.step("clarify")
        assert event["accepted"]
        assert "at least" in event["result"]["requirement"]
        assert task.observe().state & REQUIREMENTS


def test_patch_availability_is_based_on_observed_evidence():
    with SandboxTask(0) as task:
        assert not task.step("apply_candidate_a")["accepted"]
        task.step("inspect")
        assert task.observe().state & INSPECTED
        assert task.observe().state & CANDIDATES
        event = task.step("apply_candidate_a")
        assert event["accepted"]
        assert event["result"]["file"] == "solution.py"
        assert "diff" in event["result"]
        assert not event["result"]["verified"]
        assert not task.success
        assert not task.step("apply_candidate_a")["accepted"]


def test_already_correct_fixture_needs_no_edit_or_inspection():
    with SandboxTask(11, "already_correct") as task:
        task.step("run_tests")
        task.step(STOP)
        summary = task.summary()
        assert summary["success"]
        assert summary["edits"] == 0
        assert summary["steps"] == 2
        assert summary["read_bytes"] == 0
        assert summary["output_bytes"] > 0


def test_real_timeout_is_bounded_and_does_not_verify():
    with SandboxTask(0, "timeout", timeout_seconds=0.05) as task:
        event = task.step("run_tests")
        assert event["result"]["timed_out"]
        assert not event["result"]["passed"]
        assert event["elapsed_ms"] < 2000
        assert not task.success


def test_interval_checks_outside_points_and_changed_endpoint_semantics():
    # This test may inspect the trusted template. The policy's input features
    # still receive no code parser, endpoint answer, or correct-candidate label.
    with SandboxTask(17, "interval", changed_goal=True) as task:
        lower, upper = map(int, re.findall(r"-?\d+", task.observe().context["requirement"]))
        initial = task.step("run_tests")
        assert not initial["result"]["passed"]
        assert f"value={lower - 1}" in initial["stderr"]
        task.step("inspect")
        candidates = task.observe().context["supplied_candidates"]
        inclusive_id = next(key for key, source in candidates.items() if "<=" in source)
        exclusive_id = next(key for key, source in candidates.items() if "<=" not in source)
        assert lower < upper
        task.step(inclusive_id)
        assert task.step("run_tests")["goal_changed"]
        task.step("clarify")
        stale = task.step("run_tests")
        assert not stale["result"]["passed"]
        assert f"value={lower}" in stale["stderr"]
        task.step("inspect")
        task.step(exclusive_id)
        assert task.step("run_tests")["result"]["passed"]
        assert task.step(STOP)["accepted"]
        assert task.success


def test_interval_has_no_hidden_correctness_features():
    interval = SandboxTask(17, "interval")
    boundary = SandboxTask(52, "boundary")
    np.testing.assert_array_equal(encode(interval.observe())[0], encode(boundary.observe())[0])


def test_fixed_command_and_environment(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 1, "", "expected failure")

    monkeypatch.setattr("agent_lab.sandbox.subprocess.run", fake_run)
    with SandboxTask(0) as task:
        task.step("run_tests")
        args, kwargs = calls[0]
        assert Path(args[0]).is_absolute()
        assert args[1:] == ["-I", "-B", "acceptance_check.py"]
        assert kwargs["shell"] is False
        assert kwargs["cwd"] == task.directory
        assert kwargs["timeout"] == 0.5
        assert "PATH" not in kwargs["env"]


def test_only_fixed_trusted_actions_and_paths_are_accepted(tmp_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("unchanged")
    with SandboxTask(1) as task:
        directory = task.directory
        assert directory.parent != tmp_path
        assert set(p.name for p in directory.iterdir()) == {"solution.py", "acceptance_check.py"}
        for command in ("../outside.py", "rm -rf /", "python -c 'print(1)'", "__import__"):
            with pytest.raises(ValueError):
                task.step(command)
        with pytest.raises(ValueError):
            task._write(str(sentinel), "bad")
        task.step("inspect")
        task.step("apply_candidate_a")
        task.step("run_tests")
        assert set(p.name for p in directory.iterdir()) == {"solution.py", "acceptance_check.py"}
        assert sentinel.read_text() == "unchanged"
    assert not directory.exists()


@pytest.mark.parametrize("kwargs", [
    {"seed": True}, {"seed": 0, "kind": "arbitrary.py"},
    {"seed": 0, "timeout_seconds": float("inf")},
    {"seed": 0, "timeout_seconds": 99}, {"seed": 0, "changed_goal": "yes"},
])
def test_rejects_invalid_fixture_configuration(kwargs):
    with pytest.raises(ValueError):
        SandboxTask(**kwargs)


def test_requires_active_context_and_cleans_up_on_exception():
    task = SandboxTask(0)
    with pytest.raises(RuntimeError):
        task.step("inspect")
    with pytest.raises(RuntimeError, match="fixture failure"):
        with task:
            directory = task.directory
            raise RuntimeError("fixture failure")
    assert not directory.exists()
    with pytest.raises(RuntimeError):
        task.step("inspect")
