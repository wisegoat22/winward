import json

import pytest
from fastapi.testclient import TestClient

from agent_lab.runtime_v3 import V3Runtime
from jev_local.app import create_app


class FakeV3:
    def __init__(self):
        self.calls = []

    def status(self):
        return {"status":"ready", "evaluation":{"promotion":{"promoted":False}}}

    def examples(self):
        return {"examples":[{"id":"known", "title":"Known cause"}]}

    def decide(self, example_id, depth):
        self.calls.append(("decide",example_id,depth))
        return {"learned_choice":{"action_id":"inspect"}}

    def episode(self, example_id, seed, depth):
        self.calls.append(("episode",example_id,seed,depth))
        return {"success":False, "trace":[]}


@pytest.fixture
def v3_api():
    runtime = FakeV3()
    with TestClient(create_app(lambda: object(), v3_factory=lambda:runtime)) as client:
        yield client, runtime


def test_v3_checkpoint_readiness_does_not_imply_promotion(v3_api):
    client,_=v3_api
    assert client.get("/api/v3/status").json()["status"]=="ready"
    assert not client.get("/api/v3/status").json()["evaluation"]["promotion"]["promoted"]
    assert client.get("/v3").status_code==200


@pytest.mark.parametrize("extra",[{"actual_world":"secret"},{"code":"print(1)"},
                                  {"max_depth":6},{"max_depth":True},{"seed":True}])
def test_episode_rejects_hidden_world_code_or_unbounded_controls(v3_api,extra):
    client,runtime=v3_api
    assert client.post("/api/v3/episode",json={"example_id":"known"}|extra).status_code==422
    assert not runtime.calls


def test_v3_requests_share_local_origin_protection(v3_api):
    client,runtime=v3_api
    assert client.post("/api/v3/decide",json={"example_id":"known"},headers={"Origin":"https://outside.example"}).status_code==403
    assert not runtime.calls
    assert client.post("/api/v3/episode",json={"example_id":"known","seed":12,"max_depth":3}).json()["success"] is False
    assert runtime.calls==[("episode","known",12,3)]


def test_v3_never_displays_another_checkpoints_promotion(tmp_path):
    (tmp_path/"report.json").write_text(json.dumps({"checkpoint_sha256":"actual"}))
    (tmp_path/"evaluation.json").write_text(json.dumps({"checkpoint_sha256":"other", "promotion":{"promoted":True}}))
    assert V3Runtime(tmp_path).status()["evaluation"] is None
