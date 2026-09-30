import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException

from jev_local.app import create_app
from jev_local.config import MODEL_ID, MODEL_REVISION
from jev_local.schemas import DecisionRequest


DECISION = {
    "state": "I was charged twice.",
    "question": "Which team?",
    "choices": [{"name": "billing"}, {"name": "account"}],
}


class FakeEngine:
    def __init__(self):
        self.created_thread = threading.get_ident()
        self.calls = []

    def record(self, name, *args):
        assert threading.get_ident() == self.created_thread
        self.calls.append((name, args))

    def decide(self, request):
        self.record("decide", request)
        return {"choice": "billing", "output_tokens": 0}

    def generate(self, request):
        self.record("generate", request)
        return {"choice": "billing", "text": "billing: Duplicate charge.", "output_tokens": 6}

    def tokenize(self, text, add_special_tokens):
        self.record("tokenize", text, add_special_tokens)
        return [32]

    def score(self, request):
        self.record("score", request)
        if max(request.label_token_ids) > 1000000:
            raise ValueError("A requested token ID is outside this model's vocabulary.")
        return {"scores": [[0.8, 0.2]], "output_tokens": 0}


@pytest.fixture
def api():
    engines = []

    def factory():
        engine = FakeEngine()
        engines.append(engine)
        return engine

    with TestClient(create_app(factory)) as client:
        yield client, engines[0]


def test_readiness_and_model_identity(api):
    client, engine = api
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["offline"] is True
    assert health.json()["revision"] == MODEL_REVISION
    assert client.get("/v1/models").json()["data"][0]["id"] == MODEL_ID
    assert client.get("/api/examples").json()["examples"]
    assert engine.calls == []
    assert health.headers["cache-control"] == "no-store"
    assert "frame-ancestors 'none'" in health.headers["content-security-policy"]


def test_decision_and_generation_routes_use_dedicated_inference_thread(api):
    client, engine = api
    decision = client.post("/api/decide", json=DECISION)
    generation = client.post("/api/generate", json=DECISION | {"max_tokens": 32})
    assert decision.status_code == generation.status_code == 200
    assert decision.json()["output_tokens"] == 0
    assert generation.json()["output_tokens"] > 0
    assert [name for name, _ in engine.calls] == ["decide", "generate"]
    assert engine.calls[1][1][0].max_tokens == 32


@pytest.mark.parametrize("model", [None, "jev-local", MODEL_ID])
def test_sglang_compatible_routes_accept_loaded_model_names(api, model):
    client, engine = api
    tokenize = client.post("/tokenize", json={"prompt": "A", "model": model})
    score = client.post("/v1/score", json={
        "query": "Label:", "items": [""], "label_token_ids": [32, 33],
        "apply_softmax": True, "model": model,
    })
    assert tokenize.status_code == score.status_code == 200
    assert tokenize.json()["tokens"] == [32]
    assert score.json()["scores"] == [[0.8, 0.2]]
    assert engine.calls[0][1] == ("A", False)


@pytest.mark.parametrize(
    "path,body",
    [("/tokenize", {"prompt": "A"}), ("/v1/score", {"query": "Label:", "label_token_ids": [32, 33]})],
)
def test_unloaded_model_is_rejected_before_inference(api, path, body):
    client, engine = api
    response = client.post(path, json=body | {"model": "some-other-model"})
    assert response.status_code == 422
    assert engine.calls == []


def test_engine_validation_error_becomes_readable_client_error(api):
    client, _ = api
    response = client.post("/v1/score", json={"query": "Label:", "label_token_ids": [32, 999999999]})
    assert response.status_code == 422
    assert "vocabulary" in response.json()["detail"]


@pytest.mark.parametrize("origin", ["https://evil.example", "null", "http://testserver:9999", "https://testserver", "http://["])
def test_cross_origin_or_malformed_browser_posts_are_rejected(api, origin):
    client, engine = api
    response = client.post("/api/decide", json=DECISION, headers={"Origin": origin})
    assert response.status_code == 403
    assert engine.calls == []


def test_same_origin_browser_post_is_accepted(api):
    client, _ = api
    assert client.post("/api/decide", json=DECISION, headers={"Origin": "http://testserver"}).status_code == 200


def test_untrusted_host_is_rejected(api):
    client, engine = api
    response = client.post("/api/decide", json=DECISION, headers={"Host": "evil.example"})
    assert response.status_code == 400
    assert engine.calls == []


def test_invalid_decision_does_not_reach_model(api):
    client, engine = api
    response = client.post("/api/decide", json=DECISION | {"choices": [{"name": "one"}]})
    assert response.status_code == 422
    assert engine.calls == []


def test_busy_request_is_rejected_and_model_recovers_after_completion():
    started, release = threading.Event(), threading.Event()

    class BlockingEngine(FakeEngine):
        def decide(self, request):
            self.record("decide", request)
            started.set()
            assert release.wait(timeout=5), "Test failed to release fake model"
            return {"choice": "billing", "output_tokens": 0}

    with TestClient(create_app(BlockingEngine)) as client, ThreadPoolExecutor(max_workers=1) as caller:
        running = caller.submit(client.post, "/api/decide", json=DECISION)
        try:
            assert started.wait(timeout=5)
            assert client.post("/api/decide", json=DECISION).status_code == 429
            # Informational routes must stay responsive during an inference.
            assert client.get("/api/health").status_code == 200
        finally:
            release.set()
        assert running.result(timeout=5).status_code == 200
        assert client.post("/api/decide", json=DECISION).status_code == 200


def test_cancelled_client_cannot_release_busy_model_early():
    started, release = threading.Event(), threading.Event()

    class BlockingEngine(FakeEngine):
        def decide(self, request):
            self.record("decide", request)
            started.set()
            assert release.wait(timeout=5), "Test failed to release fake model"
            return {"choice": "billing", "output_tokens": 0}

    async def scenario():
        app = create_app(BlockingEngine)
        async with app.router.lifespan_context(app):
            endpoint = next(route.endpoint for route in app.routes if route.path == "/api/decide")
            request = DecisionRequest(**DECISION)
            running = asyncio.create_task(endpoint(request))
            try:
                assert await asyncio.to_thread(started.wait, 5)
                running.cancel()
                # Let the cancelled request enter its cleanup path.
                await asyncio.sleep(0)
                assert app.state.lock.locked()
                with pytest.raises(HTTPException) as error:
                    await endpoint(request)
                assert error.value.status_code == 429
            finally:
                release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(running, timeout=5)
            assert not app.state.lock.locked()
            assert (await endpoint(request))["choice"] == "billing"

    asyncio.run(scenario())
