"""Compile a reviewed software description into a conditional workflow.

Qwen extracts evidence; trusted code supplies workflow transitions. Winward's
frozen neural policy chooses. No repository is inspected and no tool executes.
This adapter is separate from every frozen training and evaluation artifact.
"""
from dataclasses import dataclass
import time

from agent_training.simulator import Action, Scenario
from agent_training.features import STOP, DEFER
from jev_local.situation_schemas import SituationDraft

FACT_LABELS = {
    "context_available": "The relevant code, logs, or documents are available",
    "requirements_clear": "The required behavior and success criteria are clear",
    "reproduced": "The failure or current behavior has been reproduced",
    "cause_known": "The cause or implementation approach is understood",
    "change_applied": "The intended change is already applied",
    "focused_checks_passed": "The focused checks have passed after the latest change",
    "acceptance_passed": "The full stated win has been checked and satisfied",
    "review_complete": "The requested review is complete",
}
BITS = {name: 1 << i for i, name in enumerate(FACT_LABELS)}
LIMITATION = (
    "Experimental advice for software workflows. Qwen interprets your description; Winward ranks a fixed set of workflow actions. "
    "Neither has inspected your files or run checks. Action effects and costs are assumptions, not predictions or measured token bills. "
    "The v4 model still failed five retention checks; free-text interpretation is outside its frozen benchmark."
)


@dataclass(frozen=True)
class Workflow:
    scenario: Scenario
    actions: list[dict]
    facts: list[dict]


def compile_workflow(draft, confirmed_facts):
    if isinstance(draft, dict):
        draft = SituationDraft.model_validate(draft)
    if draft.domain != "software":
        raise ValueError("This experimental workflow supports software-agent situations only.")
    confirmed = set(confirmed_facts)
    if not confirmed <= BITS.keys():
        raise ValueError("Unknown workflow fact.")
    state = sum(BITS[name] for name in confirmed)
    evidence = {item.fact: item.evidence for item in draft.confirmed}
    restrictions = {item.kind for item in draft.restrictions}
    facts = [{"id": key, "label": label, "confirmed": key in confirmed,
              "evidence": evidence.get(key, "")} for key, label in FACT_LABELS.items()]
    actions, details = [], []

    def add(identity, name, description, requires=(), sets=(), clears=(), cost=5, time=10, allowed=True):
        mask = lambda names: sum(BITS[item] for item in names)
        action = Action(id=identity, name=name, requires=mask(requires), forbids=mask(sets),
                        sets=mask(sets), clears=mask(clears), tokens=cost, latency_ms=time, allowed=allowed)
        actions.append(action)
        details.append({"id": identity, "name": name, "description": description,
                        "requires": [FACT_LABELS[x] for x in requires],
                        "effects": [FACT_LABELS[x] for x in sets],
                        "cost_label": "Low relative effort" if cost <= 5 else "Medium relative effort" if cost <= 12 else "Higher relative effort",
                        "available": action.eligible(state), "allowed": allowed})

    add("clarify", "Clarify the success criteria", "Ask what observable result would count as the win before changing anything.",
        sets=("requirements_clear",), cost=2, time=5)
    add("gather", "Get the relevant code, logs, or documents", "Obtain the smallest set of material needed to understand this task.",
        sets=("context_available",), cost=4, time=10)
    common = ("context_available", "requirements_clear")
    kind = draft.task_kind
    if kind in ("bugfix", "investigate"):
        add("reproduce", "Reproduce the observed failure", "Use the smallest existing check or example that demonstrates the issue.",
            requires=common, sets=("reproduced",), cost=5, time=15, allowed="no_running_tests" not in restrictions)
    if kind not in ("review", "tests"):
        add("understand", "Identify the cause from the evidence" if kind in ("bugfix", "investigate") else "Identify the smallest necessary change",
            "Inspect the relevant material and establish a concrete cause or approach. Reproduce the issue if the evidence is insufficient.",
            requires=common, sets=("cause_known",), cost=8, time=20)
    if kind not in ("investigate", "review", "tests"):
        add("change", "Make the smallest necessary documentation change" if kind == "documentation" else "Make the smallest necessary code change",
            "Apply only the change needed for the stated win, using the understood cause or approach.",
            requires=(*common, "cause_known"), sets=("change_applied",),
            clears=("focused_checks_passed", "acceptance_passed", "review_complete"), cost=20, time=45,
            allowed="no_edits" not in restrictions)
    if kind == "review":
        add("review", "Review the relevant changes against the goal", "Inspect the material, identify actionable findings, and record the requested review.",
            requires=common, sets=("review_complete",), cost=8, time=20)
        goal = BITS["review_complete"]
    elif kind == "investigate":
        goal = BITS["cause_known"]
    elif kind == "documentation":
        add("verify_docs", "Check the documentation against the stated win", "Review accuracy and completeness against the requested outcome.",
            requires=(*common, "change_applied"), sets=("acceptance_passed",), cost=5, time=10)
        goal = BITS["acceptance_passed"]
    else:
        requirements = common if kind == "tests" else (*common, "change_applied")
        add("focused", "Run the focused checks", "Run the smallest relevant checks and inspect their actual results.",
            requires=requirements, sets=("focused_checks_passed",),
            cost=3, time=10, allowed="no_running_tests" not in restrictions)
        add("verify", "Check that the stated win is satisfied", "Run the relevant acceptance checks and compare the observed result with the original goal.",
            requires=requirements, sets=("acceptance_passed",), cost=8, time=25,
            allowed="no_running_tests" not in restrictions)
        goal = BITS["acceptance_passed"]
    scenario = Scenario(id="custom-software-workflow", family="custom", split="custom", facts=tuple(FACT_LABELS.values()),
                        state=state, goal=goal, actions=tuple(actions), context={})
    return Workflow(scenario, details, facts)


def interpretation_result(request, result):
    draft = SituationDraft.model_validate(result["draft"]).check_evidence(request.situation, request.goal)
    status = "ready" if draft.domain == "software" else "needs_clarification" if draft.domain == "unclear" else "unsupported"
    workflow = compile_workflow(draft, [item.fact for item in draft.confirmed]) if status == "ready" else None
    return {"status": status, "draft": draft.model_dump(), "facts": workflow.facts if workflow else [],
            "actions": workflow.actions if workflow else [], "goal_label": request.goal,
            "questions": draft.unknowns or ([] if status == "ready" else ["Describe a software task and an observable result that would count as success."]),
            "interpretation": result["interpretation"], "limitation": LIMITATION}


def choose_next_step(runtime, request):
    started = time.perf_counter()
    draft = request.draft
    if draft.domain != "software":
        return {"status": "unsupported", "questions": draft.unknowns or ["Describe a software-agent task to use this experimental policy."], "limitation": LIMITATION}
    workflow = compile_workflow(draft, request.confirmed_facts)
    # The same frozen v4 weights used in the lab rank the actual reviewed state.
    choice = runtime._policy().choose(workflow.scenario, remaining=request.max_depth)
    identity = choice["action_id"]
    detail = next((action for action in workflow.actions if action["id"] == identity), None)
    if identity == STOP:
        if not workflow.scenario.goal_met():
            raise ValueError("Winward suggested completion without a satisfied workflow goal. No completion claim was made.")
        selected = {"id": identity, "name": "Review and report completion", "description": "Your confirmed facts say the workflow goal is satisfied. Check that they still match the stated win, then report completion."}
        check = "Confirm that the evidence covers your original win. This app has not independently checked it."
    elif identity == DEFER:
        selected = {"id": identity, "name": "Ask for more information or revise the plan", "description": "Winward deferred rather than selecting a workflow action. This can reflect a limitation of the learned policy or the supplied workflow."}
        check = draft.unknowns[0] if draft.unknowns else "Review the facts, constraints, and success criteria. A deferral does not prove the task is impossible."
    elif detail is not None and detail["available"]:
        selected = {k: detail[k] for k in ("id", "name", "description")}
        check = "After this step, verify: " + "; ".join(detail["effects"]) + ". Update the situation using the result you actually observe."
    else:
        raise ValueError("Winward returned an unavailable action. No alternative was silently substituted.")
    return {"status": "recommended", "choice": selected,
            "reason": "Winward selected this action from the reviewed facts and the supplied workflow. This is a conditional recommendation, not a verified outcome.",
            "next_check": check, "actions": workflow.actions,
            "assumptions": ["Checked facts are your assertions; unchecked facts are not established.",
                            "Workflow transitions describe the intended result if an action succeeds. Failures and hidden causes are not modeled in this text adapter.",
                            "The goal is represented by a workflow milestone; verify that it captures your actual win.",
                            "Costs are relative defaults. This recommendation has not inspected a repository or executed any action."],
            "timing": {"decision_ms": (time.perf_counter() - started) * 1000,
                       "model_decision_ms": choice["decision_ms"]},
            "source": {"reader": "Qwen3 4B (local)", "decision": "Winward v4 · 4.86M parameters"}, "limitation": LIMITATION}
