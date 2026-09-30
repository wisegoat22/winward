from fastapi.testclient import TestClient
import pytest

from jev_local.app import create_app


class FakePolicy:
    def __init__(self):
        self.calls = []

    def status(self):
        return {"status": "ready", "training_origin": "random initialization"}

    def examples(self):
        return {"examples": []}

    def decide(self, scenario, token_weight, latency_weight, max_depth):
        self.calls.append((scenario, token_weight, latency_weight, max_depth))
        return {"decision": {"id": "STOP"}, "model_name": "test policy"}


@pytest.fixture
def policy_api():
    policy = FakePolicy()
    with TestClient(create_app(lambda: object(), lambda: policy)) as client:
        yield client, policy


def body():
    return {"scenario": {"facts": ["ready", "done"], "state": 1, "goal": 2,
                         "actions": [{"id": "finish", "name": "Finish", "requires": 1, "sets": 2}]}}


def test_policy_is_separate_from_existing_llm_engine(policy_api):
    client, policy = policy_api
    response = client.post("/api/policy/decide", json=body())
    assert response.status_code == 200
    assert response.json()["model_name"] == "test policy"
    assert policy.calls[0][1:] == (1.0, 0.01, 5)
    assert client.get("/api/policy/status").json()["training_origin"] == "random initialization"
    assert client.get("/policy").status_code == 200


@pytest.mark.parametrize("change", [
    {"max_depth": 6}, {"max_depth": 0}, {"max_depth": True},
    {"token_weight": -1}, {"token_weight": 0, "latency_weight": 0},
])
def test_invalid_policy_controls_never_reach_inference(policy_api, change):
    client, policy = policy_api
    assert client.post("/api/policy/decide", json=body() | change).status_code == 422
    assert policy.calls == []


def test_invalid_world_cannot_reach_inference(policy_api):
    client, policy = policy_api
    request = body()
    request["scenario"]["actions"][0]["sets"] = 8  # No such fact exists.
    assert client.post("/api/policy/decide", json=request).status_code == 422
    assert policy.calls == []


def test_reserved_model_outcomes_cannot_be_spoofed_as_user_actions(policy_api):
    client, policy = policy_api
    request = body()
    request["scenario"]["actions"][0]["id"] = "STOP"
    assert client.post("/api/policy/decide", json=request).status_code == 422
    assert policy.calls == []
