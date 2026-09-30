import copy
import threading

from fastapi.testclient import TestClient
import pytest

from agent_lab.situation import BITS, compile_workflow, choose_next_step
from agent_training.simulator import search
from jev_local.app import create_app
from jev_local.situation_schemas import SituationDraft, SituationDecision

INPUT = {"situation": "The checkout test fails. I have the source code. Do not edit any files.",
         "goal": "Identify the cause of the checkout failure."}
DRAFT = {"domain": "software", "task_kind": "investigate", "summary": "Investigate a checkout failure.",
         "confirmed": [{"fact": "context_available", "evidence": "I have the source code."}],
         "restrictions": [{"kind": "no_edits", "evidence": "Do not edit any files."}], "unknowns": []}


class Reader:
    def __init__(self):
        self.thread = threading.get_ident()
        self.calls = []
        self.draft = copy.deepcopy(DRAFT)

    def read_situation(self, situation, goal):
        assert threading.get_ident() == self.thread
        self.calls.append((situation, goal))
        return {"draft": self.draft, "interpretation": {"model": "local-reader", "input_tokens": 30, "output_tokens": 20, "latency_ms": 4}}


class FrozenPolicy:
    def __init__(self):
        self.choice = "understand"
        self.calls = []

    def _policy(self):
        return self

    def choose(self, scenario, remaining):
        self.calls.append((scenario, remaining, threading.get_ident()))
        return {"action_id": self.choice, "decision_ms": 1.2}


@pytest.fixture
def api():
    readers = []
    policy = FrozenPolicy()

    def factory():
        reader = Reader(); readers.append(reader); return reader

    with TestClient(create_app(factory, v4_factory=lambda: policy)) as client:
        yield client, readers[0], policy


def test_free_text_is_read_once_then_reviewed_facts_reach_frozen_policy(api):
    client, reader, policy = api
    result = client.post("/api/situation/interpret", json=INPUT)
    assert result.status_code == 200
    draft = result.json()
    assert draft["status"] == "ready" and not policy.calls
    assert [f["id"] for f in draft["facts"] if f["confirmed"]] == ["context_available"]
    body = {**INPUT, "draft": draft["draft"], "confirmed_facts": ["context_available", "requirements_clear"]}
    result = client.post("/api/situation/decide", json=body)
    assert result.status_code == 200 and result.json()["choice"]["id"] == "understand"
    scenario, depth, worker = policy.calls[0]
    assert scenario.state == BITS["context_available"] | BITS["requirements_clear"]
    assert worker == reader.thread and depth == 5 and len(reader.calls) == 1
    assert client.get("/try").status_code == 200


@pytest.mark.parametrize("domain", ["unsupported", "unclear"])
def test_nonsoftware_never_uses_the_decision_model(api, domain):
    client, reader, policy = api
    reader.draft = {**DRAFT, "domain": domain, "confirmed": [], "unknowns": ["Describe a software task."]}
    parsed = client.post("/api/situation/interpret", json=INPUT).json()
    assert parsed["status"] != "ready"
    result = client.post("/api/situation/decide", json={**INPUT, "draft": parsed["draft"], "confirmed_facts": []})
    assert result.json()["status"] == "unsupported" and not policy.calls


def test_evidence_cannot_span_two_separate_input_fields():
    draft = copy.deepcopy(DRAFT)
    draft["confirmed"][0]["evidence"] = "files.\nIdentify"
    with pytest.raises(ValueError, match="exact quote"):
        SituationDraft.model_validate(draft).check_evidence(**INPUT)


def test_ungrounded_generated_evidence_is_rejected(api):
    client, reader, policy = api
    reader.draft["confirmed"][0]["evidence"] = "All checks passed after the fix."
    assert client.post("/api/situation/interpret", json=INPUT).status_code == 422
    assert not policy.calls


@pytest.mark.parametrize("extra", [{"actual_world": "secret"}, {"max_depth": True}, {"confirmed_facts": ["made_up"]},
                                  {"confirmed_facts": ["context_available"] * 2}, {"max_depth": 6}])
def test_unbounded_or_hidden_controls_are_rejected(api, extra):
    client, _, policy = api
    body = {**INPUT, "draft": DRAFT, "confirmed_facts": [], **extra}
    assert client.post("/api/situation/decide", json=body).status_code == 422
    assert not policy.calls


def test_cross_origin_requests_cannot_use_text_or_decision_models(api):
    client, reader, policy = api
    headers = {"Origin": "https://outside.example"}
    assert client.post("/api/situation/interpret", json=INPUT, headers=headers).status_code == 403
    assert client.post("/api/situation/decide", json={**INPUT, "draft": DRAFT, "confirmed_facts": []}, headers=headers).status_code == 403
    assert not reader.calls and not policy.calls


def test_scope_and_constraints_change_available_actions_without_model_override():
    draft = SituationDraft.model_validate({**DRAFT, "task_kind": "bugfix", "restrictions": [
        {"kind": "no_edits", "evidence": "Do not edit any files."},
        {"kind": "no_running_tests", "evidence": "Do not run tests."}]})
    workflow = compile_workflow(draft, ["context_available", "requirements_clear", "cause_known"])
    assert not next(a for a in workflow.scenario.actions if a.id == "change").allowed
    assert all(not a.allowed for a in workflow.scenario.actions if a.id in ("reproduce", "focused", "verify"))
    for kind in ("investigate", "review", "tests"):
        row = compile_workflow(draft.model_copy(update={"task_kind": kind}), [])
        assert all(a.id != "change" for a in row.scenario.actions)


@pytest.mark.parametrize("kind", ["bugfix", "feature", "refactor", "investigate", "review", "documentation", "tests"])
def test_unrestricted_workflows_fit_the_five_action_horizon(kind):
    draft = SituationDraft.model_validate({**DRAFT, "task_kind": kind, "restrictions": []})
    solution = search(compile_workflow(draft, []).scenario)
    assert solution["outcome"] == "plan"
    assert len(solution["plan"]) <= 5


def test_premature_finish_and_unavailable_choice_are_not_silently_replaced(api):
    client, _, policy = api
    for action in ("STOP", "change"):
        policy.choice = action
        result = client.post("/api/situation/decide", json={**INPUT, "draft": DRAFT, "confirmed_facts": []})
        assert result.status_code == 422


def test_completion_is_explicitly_user_reported(api):
    client, _, policy = api
    policy.choice = "STOP"
    response = client.post("/api/situation/decide", json={**INPUT, "draft": DRAFT, "confirmed_facts": ["cause_known"]})
    assert response.status_code == 200
    assert "Your confirmed facts" in response.json()["choice"]["description"]


def test_actual_neural_choice_is_displayed_even_when_it_defers(api):
    client, _, policy = api
    policy.choice = "NEEDS_CLARIFICATION"
    result = client.post("/api/situation/decide", json={**INPUT, "draft": DRAFT, "confirmed_facts": []})
    assert result.json()["choice"]["id"] == "NEEDS_CLARIFICATION"
    assert "limitation" in result.json()["choice"]["description"]
