"""V4 stays opt-in and shares the bounded, local belief API."""
import json

from fastapi.testclient import TestClient

from agent_lab.runtime_v4 import V4Runtime
from jev_local.app import create_app


class FakeV4:
    def __init__(self):
        self.calls = []

    def status(self):
        return {"status": "ready", "evaluation": {"promotion": {"promoted": False}}}

    def examples(self):
        return {"examples": []}

    def decide(self, example_id, depth):
        self.calls.append((example_id, depth))
        return {"learned_choice": {"action_id": "prepare"}}

    def episode(self, example_id, seed, depth):
        self.calls.append((example_id, seed, depth))
        return {"success": False, "trace": []}


def test_v4_is_separate_and_readiness_does_not_imply_promotion():
    runtime = FakeV4()
    with TestClient(create_app(lambda: object(), v4_factory=lambda: runtime)) as client:
        status = client.get("/api/v4/status").json()
        assert status["status"] == "ready" and status["evaluation"]["promotion"]["promoted"] is False
        assert client.get("/v4").status_code == 200
        assert client.get("/v3").status_code == 200
        assert client.post("/api/v4/decide", json={"example_id": "known", "max_depth": 3}).status_code == 200
        assert runtime.calls == [("known", 3)]


def test_v4_blocks_hidden_world_inputs_and_external_origins():
    runtime = FakeV4()
    with TestClient(create_app(lambda: object(), v4_factory=lambda: runtime)) as client:
        for extra in ({"actual_world": "secret"}, {"max_depth": 6}, {"seed": True}, {"code": "run arbitrary code"}):
            assert client.post("/api/v4/episode", json={"example_id": "known", **extra}).status_code == 422
        assert client.post("/api/v4/episode", json={"example_id": "known"},
                           headers={"Origin": "https://outside.example"}).status_code == 403
        assert runtime.calls == []


def test_v4_hides_audit_from_different_checkpoint_and_preserves_versions(tmp_path):
    (tmp_path / "report.json").write_text(json.dumps({"checkpoint_sha256": "v4"}))
    (tmp_path / "evaluation.json").write_text(json.dumps({"checkpoint_sha256": "v3", "promotion": {"promoted": True}}))
    status = V4Runtime(tmp_path).status()
    assert status["evaluation"] is None
    assert "Frozen Winward v3" in status["retained_versions"]
