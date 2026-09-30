import pytest
from fastapi.testclient import TestClient

from jev_local.app import create_app
from jev_local.lazy_engine import LazyEngine


class FakeLab:
    def __init__(self):
        self.calls = []

    def status(self):
        return {"status": "ready"}

    def examples(self):
        return {"uncertainty": [], "task_kinds": ["boundary"]}

    def uncertainty(self, example_id, max_depth):
        return {"example_id": example_id, "max_depth": max_depth}

    def sandbox(self, *args):
        self.calls.append(args)
        return {"success": True}


@pytest.fixture
def lab_api():
    lab = FakeLab()
    def unavailable():
        raise FileNotFoundError("No optional pretrained model installed")
    with TestClient(create_app(lambda: LazyEngine(unavailable), lab_factory=lambda: lab)) as client:
        yield client, lab


def test_local_lab_starts_without_loading_optional_qwen(lab_api):
    client, _ = lab_api
    assert client.get("/api/health").json()["qwen_loaded"] is False
    assert client.get("/api/v2/status").status_code == 200
    assert client.get("/api/v2/examples").json()["task_kinds"] == ["boundary"]
    response = client.post("/api/decide", json={"state":"test", "question":"Which?",
                                               "choices":[{"name":"a"},{"name":"b"}]})
    assert response.status_code == 503
    assert "optional Qwen" in response.json()["detail"]
    assert client.get("/api/v2/status").status_code == 200


@pytest.mark.parametrize("extra", [{"path":"/tmp"},{"command":"echo unsafe"},{"seed":True},
                                   {"kind":"../source"},{"policy":"shell"},{"uncertain":"true"}])
def test_sandbox_endpoint_accepts_only_bounded_controls(lab_api, extra):
    client, lab = lab_api
    response = client.post("/api/v2/sandbox", json={"kind":"boundary"}|extra)
    assert response.status_code == 422
    assert not lab.calls


def test_sandbox_has_same_origin_protection_and_forwards_only_controls(lab_api):
    client, lab = lab_api
    assert client.post("/api/v2/sandbox",json={"kind":"boundary"},headers={"Origin":"https://example.org"}).status_code == 403
    assert not lab.calls
    assert client.post("/api/v2/sandbox",json={"kind":"boundary"}).json()["success"]
    assert lab.calls == [(700001,"boundary",False,False,"neural")]


def test_uncertainty_horizon_is_limited(lab_api):
    client, _ = lab_api
    assert client.post("/api/v2/uncertainty",json={"example_id":"inspect","max_depth":6}).status_code == 422
