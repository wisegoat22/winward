"""Real checks over tiny, trusted Python fixtures in disposable directories.

This is a controlled fixture harness, NOT an operating-system security sandbox.
Only this module's generated source and fixed subprocess arguments are executed.
The policy schedules tools and selects supplied patches; it does not write code.
Action effects are forecasts, reconciled against actual tool observations after
every step. In particular, a predicted test pass is never accepted as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from typing import Any

from agent_training.simulator import Action, NEEDS_CLARIFICATION, STOP, Scenario


SUPPORTED_KINDS = ("boundary", "rounding", "collection", "already_correct", "timeout", "interval")
FACTS = (
    "requirements_known", "file_inspected", "candidates_available",
    "change_applied", "verification_due", "checks_passed", "checks_failed",
    "goal_current", "notes_read", "candidate_a_tried", "candidate_b_tried",
)
REQUIREMENTS, INSPECTED, CANDIDATES, CHANGED, VERIFY, PASSED, FAILED, CURRENT, NOTES, TRIED_A, TRIED_B = (
    1 << i for i in range(len(FACTS))
)


@dataclass(frozen=True)
class _Fixture:
    initial: str
    candidates: tuple[str, str]
    requirements: tuple[str, str]
    checks: tuple[str, str]


def _fixture(seed: int, kind: str) -> _Fixture:
    """Make source only from trusted literals and bounded integer parameters."""
    rng = random.Random(seed)
    if kind in {"boundary", "already_correct", "timeout"}:
        threshold = rng.randint(12, 80)
        candidates = (
            f"def eligible(value):\n    return value >= {threshold}\n",
            f"def eligible(value):\n    return value > {threshold}\n",
        )
        initial = f"def eligible(value):\n    return value >= {threshold + 1}\n"
        if kind == "already_correct":
            initial = candidates[0]
        elif kind == "timeout":
            initial = "def eligible(value):\n    while True:\n        pass\n"
        requirements = (
            f"eligible(value) must be true exactly when value is at least {threshold}.",
            f"Revised goal: eligible(value) must be true exactly when value is greater than {threshold}.",
        )
        checks = tuple(
            "fn = namespace['eligible']\n"
            + f"for value, expected in {[(threshold - 1, False), (threshold, revision == 0), (threshold + 1, True)]!r}:\n"
            + "    actual = fn(value)\n"
            + "    assert actual is expected, f'value={value}: expected {expected!r}, got {actual!r}'\n"
            for revision in range(2)
        )
    elif kind == "rounding":
        candidates = (
            "def rounded_ratio(numerator, denominator):\n    return numerator // denominator\n",
            "def rounded_ratio(numerator, denominator):\n    return -(-numerator // denominator)\n",
        )
        initial = "def rounded_ratio(numerator, denominator):\n    return numerator / denominator\n"
        requirements = (
            "Return rounded_ratio(numerator, denominator) as an integer rounded down; inputs are nonnegative integers with a positive denominator.",
            "Revised goal: return rounded_ratio(numerator, denominator) as an integer rounded up; inputs are nonnegative integers with a positive denominator.",
        )
        samples = ((7, 3), (6, 3), (0, 2), (1, 7), (rng.randint(10, 30), 4))
        checks = tuple(
            "fn = namespace['rounded_ratio']\n"
            + f"for numerator, denominator, expected in {[(n, d, n // d if revision == 0 else -(-n // d)) for n, d in samples]!r}:\n"
            + "    actual = fn(numerator, denominator)\n"
            + "    assert type(actual) is int and actual == expected, f'{numerator}/{denominator}: expected integer {expected}, got {actual!r}'\n"
            for revision in range(2)
        )
    elif kind == "interval":
        lower = rng.randint(-20, 20)
        upper = lower + rng.randint(3, 12)
        candidates = (
            f"def contains(value):\n    return {lower} <= value <= {upper}\n",
            f"def contains(value):\n    return {lower} < value < {upper}\n",
        )
        initial = f"def contains(value):\n    return value >= {lower} or value <= {upper}\n"
        requirements = (
            f"contains(value) must return whether the integer is between {lower} and {upper}, including both endpoints.",
            f"Revised goal: contains(value) must return whether the integer is between {lower} and {upper}, excluding both endpoints.",
        )
        samples = (lower - 1, lower, lower + 1, upper - 1, upper, upper + 1)
        checks = tuple(
            "fn = namespace['contains']\n"
            + f"for value, expected in {[(v, lower <= v <= upper if revision == 0 else lower < v < upper) for v in samples]!r}:\n"
            + "    actual = fn(value)\n"
            + "    assert actual is expected, f'value={value}: expected {expected!r}, got {actual!r}'\n"
            for revision in range(2)
        )
    elif kind == "collection":
        candidates = (
            "def unique_items(values):\n    return list(dict.fromkeys(values))\n",
            "def unique_items(values):\n    return sorted(set(values))\n",
        )
        initial = "def unique_items(values):\n    return list(values)\n"
        requirements = (
            "unique_items(values) must remove repeated integers and keep their first-occurrence order.",
            "Revised goal: unique_items(values) must remove repeated integers and return ascending order.",
        )
        samples = ([3, 1, 3, 2, 1], [], [5, 5], [9, -2, 9, 0])
        checks = tuple(
            "fn = namespace['unique_items']\n"
            + f"for values, expected in {[(v, list(dict.fromkeys(v)) if revision == 0 else sorted(set(v))) for v in samples]!r}:\n"
            + "    actual = fn(values)\n"
            + "    assert actual == expected, f'{values!r}: expected {expected!r}, got {actual!r}'\n"
            for revision in range(2)
        )
    else:
        raise ValueError(f"Unsupported fixture kind: {kind!r}")
    # Correctness is not correlated with a candidate's id, forecast, or price.
    if rng.getrandbits(1):
        candidates = tuple(reversed(candidates))
    return _Fixture(initial, candidates, requirements, checks)


class SandboxTask:
    """Stateful, partially observed task with inspect/edit/check/clarify actions.

    ``changed_goal`` schedules a goal revision immediately after the first real
    successful check. ``uncertain`` starts without an unambiguous requirement.
    Calling ``observe`` never runs a test or consults its hidden answer. Both
    candidate edits have the same predicted effect until real checks distinguish
    them. The small policy therefore learns tool scheduling, not code synthesis.
    """

    def __init__(
        self, seed: int, kind: str = "boundary", changed_goal: bool = False,
        uncertain: bool = False, *, timeout_seconds: float = 0.5,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if kind not in SUPPORTED_KINDS:
            raise ValueError(f"kind must be one of {SUPPORTED_KINDS!r}")
        if not isinstance(changed_goal, bool) or not isinstance(uncertain, bool):
            raise ValueError("changed_goal and uncertain must be booleans")
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or not 0.01 <= timeout_seconds <= 5):
            raise ValueError("timeout_seconds must be between 0.01 and 5")
        self.seed = seed
        self.kind = kind
        self.changed_goal = changed_goal
        self.uncertain = uncertain
        self.timeout_seconds = float(timeout_seconds)
        self._fixture = _fixture(seed, kind)
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._directory: Path | None = None
        self._closed = False
        self._state = VERIFY | CURRENT | (0 if uncertain else REQUIREMENTS)
        self._revision = 0
        self._verified_revision: int | None = None
        self._source_revision = 0
        self._verified_source_revision: int | None = None
        self._last_result: dict[str, Any] | None = None
        self._observed_source: str | None = None
        self._done = False
        self._deferred = False
        self.trace: list[dict[str, Any]] = []

    def __enter__(self) -> SandboxTask:
        if self._temporary is not None or self._closed:
            raise RuntimeError("A task can be entered only once")
        self._temporary = tempfile.TemporaryDirectory(prefix="winward-fixture-")
        self._directory = Path(self._temporary.name)
        self._write("solution.py", self._fixture.initial)
        self._write_acceptance()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
        self._closed = True

    @property
    def directory(self) -> Path:
        if self._directory is None or self._closed:
            raise RuntimeError("Use SandboxTask inside its context manager")
        return self._directory

    @property
    def done(self) -> bool:
        return self._done

    @property
    def success(self) -> bool:
        return bool(
            self._state & PASSED and self._state & CURRENT
            and self._verified_revision == self._revision
            and self._verified_source_revision == self._source_revision
        )

    def _write(self, filename: str, content: str) -> None:
        if filename not in {"solution.py", "acceptance_check.py"}:
            raise ValueError("Only fixed fixture files may be written")
        (self.directory / filename).write_text(content, encoding="utf-8")

    def _write_acceptance(self) -> None:
        content = (
            "import runpy\n"
            "namespace = runpy.run_path('solution.py')\n"
            + self._fixture.checks[self._revision]
            + "print('All acceptance checks passed for the current goal.')\n"
        )
        self._write("acceptance_check.py", content)

    def _actions(self) -> tuple[Action, ...]:
        # These describe possible progress, never hidden future test outcomes.
        # Tests may fail and edits may be wrong. step() reconciles both cases.
        return (
            Action("inspect", "Inspect source and supplied candidate patches",
                   requires=REQUIREMENTS | CURRENT, forbids=INSPECTED,
                   sets=INSPECTED | CANDIDATES, tokens=60, latency_ms=2),
            Action("clarify", "Ask for the missing or revised requirement",
                   forbids=REQUIREMENTS, sets=REQUIREMENTS | CURRENT,
                   tokens=40, latency_ms=15),
            Action("apply_candidate_a", "Apply supplied candidate A",
                   requires=REQUIREMENTS | CURRENT | CANDIDATES,
                   forbids=TRIED_A, sets=CHANGED | VERIFY | TRIED_A,
                   clears=PASSED | FAILED, tokens=90, latency_ms=3),
            Action("apply_candidate_b", "Apply supplied candidate B",
                   requires=REQUIREMENTS | CURRENT | CANDIDATES,
                   forbids=TRIED_B, sets=CHANGED | VERIFY | TRIED_B,
                   clears=PASSED | FAILED, tokens=90, latency_ms=3),
            Action("run_tests", "Run current-goal acceptance checks",
                   requires=REQUIREMENTS | CURRENT | VERIFY,
                   sets=PASSED, clears=VERIFY | FAILED, tokens=30, latency_ms=25),
            Action("read_notes", "Read unrelated formatting notes",
                   forbids=NOTES, sets=NOTES, tokens=80, latency_ms=2),
        )

    def observe(self) -> Scenario:
        requirement = (
            self._fixture.requirements[self._revision]
            if self._state & REQUIREMENTS else
            "The requirement is missing or has changed. Clarification is needed."
        )
        context: dict[str, Any] = {
            "requirement": requirement,
            "goal_revision": self._revision,
            "source_revision": self._source_revision,
            "last_tool_result": self._last_result,
            "forecast_semantics": "Test success and candidate usefulness are provisional; actual tool results replace these forecasts.",
            "provisional_action_ids": ["run_tests", "apply_candidate_a", "apply_candidate_b"],
            "policy_scope": "Select tools and trusted supplied patches; do not generate or execute arbitrary code.",
        }
        if self._observed_source is not None and self._state & INSPECTED:
            context["observed_source"] = self._observed_source
            context["supplied_candidates"] = {
                "apply_candidate_a": self._fixture.candidates[0],
                "apply_candidate_b": self._fixture.candidates[1],
            }
        return Scenario(
            id=f"fixture-{self.seed}-{self.kind}", family="real_python_fixture",
            split="sandbox", facts=FACTS, state=self._state,
            goal=PASSED | CURRENT, actions=self._actions(), context=context,
        )

    def _check(self) -> dict[str, Any]:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-B", "acceptance_check.py"],
                cwd=self.directory,
                env={"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"},
                shell=False, capture_output=True, text=True,
                timeout=self.timeout_seconds, check=False,
            )
            return {
                "passed": completed.returncode == 0,
                "returncode": completed.returncode,
                "timed_out": False,
                "stdout": completed.stdout.replace(str(self.directory), "<fixture>")[-4000:],
                "stderr": completed.stderr.replace(str(self.directory), "<fixture>")[-4000:],
                "command": ["python", "-I", "-B", "acceptance_check.py"],
                "output_bytes": len(completed.stdout.encode()) + len(completed.stderr.encode()),
                "goal_revision_checked": self._revision,
            }
        except subprocess.TimeoutExpired:
            return {"passed": False, "returncode": None, "timed_out": True,
                    "stdout": "", "stderr": "Acceptance check exceeded the fixture time limit.",
                    "command": ["python", "-I", "-B", "acceptance_check.py"], "output_bytes": 0,
                    "goal_revision_checked": self._revision}

    def _revise_goal(self) -> None:
        self._revision = 1
        self._verified_revision = None
        self._verified_source_revision = None
        self._state = VERIFY | (self._state & (CHANGED | NOTES))
        self._observed_source = None
        self._write_acceptance()

    def step(self, action_id: str) -> dict[str, Any]:
        # Accessing the property ensures the task has an active controlled tempdir.
        _ = self.directory
        if self._done:
            raise RuntimeError("Task already finished")
        if not isinstance(action_id, str):
            raise ValueError("action_id must be a known string")
        before = self.observe()
        action_by_id = {a.id: a for a in before.actions}
        if action_id not in {*action_by_id, STOP, NEEDS_CLARIFICATION}:
            raise ValueError("Unknown action id; arbitrary commands and paths are not supported")
        start = time.perf_counter()
        goal_changed = False
        accepted = True
        predicted_state: int | None = None
        estimated_tokens = 0.0
        if action_id == STOP:
            accepted = self.success
            if accepted:
                self._done = True
                result = {"stopped": True, "reason": "Actual acceptance checks passed for the current source and goal."}
            else:
                result = {"stopped": False, "reason": "Finish rejected: current-goal acceptance checks have not passed."}
        elif action_id == NEEDS_CLARIFICATION:
            self._done = True
            self._deferred = True
            result = {"deferred": True, "reason": "Policy requested more information instead of choosing an available tool."}
        else:
            action = action_by_id[action_id]
            accepted = action.eligible(self._state)
            if not accepted:
                result = {"rejected": True, "reason": "Action prerequisites are not satisfied."}
            else:
                estimated_tokens = action.tokens
                predicted_state = action.apply(self._state)
                self._state = predicted_state
                if action_id == "inspect":
                    self._observed_source = (self.directory / "solution.py").read_text(encoding="utf-8")
                    result = {"source": self._observed_source,
                              "file": "solution.py", "read_bytes": len(self._observed_source.encode()),
                              "candidate_count": 2, "note": "Candidate correctness is unverified."}
                elif action_id == "clarify":
                    result = {"requirement": self._fixture.requirements[self._revision],
                              "goal_revision": self._revision}
                elif action_id in {"apply_candidate_a", "apply_candidate_b"}:
                    index = int(action_id == "apply_candidate_b")
                    old_source = (self.directory / "solution.py").read_text(encoding="utf-8")
                    self._write("solution.py", self._fixture.candidates[index])
                    self._source_revision += 1
                    self._verified_revision = None
                    self._verified_source_revision = None
                    self._observed_source = self._fixture.candidates[index]
                    result = {"candidate_applied": action_id, "verified": False,
                              "file": "solution.py", "written_bytes": len(self._observed_source.encode()),
                              "diff": "".join(difflib.unified_diff(old_source.splitlines(keepends=True), self._observed_source.splitlines(keepends=True), fromfile="a/solution.py", tofile="b/solution.py")),
                              "source_revision": self._source_revision}
                elif action_id == "run_tests":
                    result = self._check()
                    if result["passed"]:
                        self._verified_revision = self._revision
                        self._verified_source_revision = self._source_revision
                        if self.changed_goal and self._revision == 0:
                            self._revise_goal()
                            goal_changed = True
                            result["goal_changed"] = True
                            result["stale_result"] = True
                            result["note"] = "The requirement changed after this check; its pass does not verify the new goal."
                    else:
                        self._state = (self._state & ~PASSED) | FAILED
                        self._verified_revision = None
                        self._verified_source_revision = None
                else:
                    result = {"notes": "Formatting preferences do not change the acceptance goal."}
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._last_result = {"action_id": action_id, **result}
        description = result.get("reason")
        if description is None:
            if action_id == "run_tests":
                description = ("Checks passed for the previous goal; the new goal needs verification." if goal_changed else
                               "Current-goal checks passed." if result.get("passed") else
                               "Checks timed out; the goal is unverified." if result.get("timed_out") else
                               "Checks failed; the goal is unverified.")
            else:
                description = {
                    "inspect": "Read source and made two unverified supplied patches available.",
                    "clarify": "Obtained the current requirement.",
                    "apply_candidate_a": "Applied supplied patch A; acceptance checks are required.",
                    "apply_candidate_b": "Applied supplied patch B; acceptance checks are required.",
                    "read_notes": "Read unrelated notes without advancing the acceptance goal.",
                }.get(action_id, "Action completed.")
        event = {
            "turn": len(self.trace) + 1, "action_id": action_id, "tool": action_id,
            "action_name": action_by_id[action_id].name if action_id in action_by_id else action_id,
            "summary": description,
            "accepted": accepted, "result": result, "elapsed_ms": elapsed_ms,
            "estimated_tokens": estimated_tokens, "goal_changed": goal_changed,
            "state_before": before.state, "state_after": self._state,
            "forecast_state": predicted_state,
            "forecast_matched": predicted_state == self._state if predicted_state is not None else None,
            "success": self.success, "done": self.done,
            "observation": self.observe().to_dict(),
        }
        for key in ("stdout", "stderr", "command", "file", "diff", "read_bytes", "written_bytes", "output_bytes"):
            if key in result:
                event[key] = result[key]
        self.trace.append(event)
        return event

    def summary(self) -> dict[str, Any]:
        return {
            "seed": self.seed, "kind": self.kind, "changed_goal": self.changed_goal,
            "uncertain": self.uncertain, "goal_revision": self._revision,
            "success": self.success, "done": self.done, "deferred": self._deferred,
            "steps": len(self.trace),
            "tool_elapsed_ms": sum(event["elapsed_ms"] for event in self.trace),
            "estimated_tokens": sum(event["estimated_tokens"] for event in self.trace),
            "checks": sum(e["accepted"] and e["action_id"] == "run_tests" for e in self.trace),
            "edits": sum(e["accepted"] and e["action_id"].startswith("apply_candidate_") for e in self.trace),
            "failed_checks": sum(e["action_id"] == "run_tests" and e["result"].get("passed") is False for e in self.trace),
            "rejected_actions": sum(not event["accepted"] for event in self.trace),
            "read_bytes": sum(e["result"].get("read_bytes", 0) for e in self.trace),
            "written_bytes": sum(e["result"].get("written_bytes", 0) for e in self.trace),
            "output_bytes": sum(e["result"].get("output_bytes", 0) for e in self.trace),
            "isolation": "Temporary directory and fixed trusted fixture code; not an OS security sandbox.",
            "cost_note": "Token counts are fixed estimates for tool scheduling; wall time is measured locally.",
        }
