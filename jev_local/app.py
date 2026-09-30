import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from urllib.parse import urlsplit

# Set these before importing any inference library. Setup is a separate command.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import MAX_INPUT_TOKENS, MODEL_ID, MODEL_REVISION
from .lazy_engine import LazyEngine, ModelNotReadyError
from .examples import EXAMPLES
from .schemas import DecisionRequest, GenerationRequest, ScoreRequest, TokenizeRequest
from .policy_schemas import PolicyRequest
from agent_training.runtime import PolicyRuntime
from .lab_schemas import SandboxRequest, UncertaintyRequest, V3EpisodeRequest
from agent_lab.runtime import LabRuntime
from agent_lab.runtime_v3 import V3Runtime
from agent_lab.runtime_v4 import V4Runtime
from agent_lab.situation import interpretation_result, choose_next_step
from .situation_schemas import SituationInput, SituationDecision
from winward_scale.status import read_status as scaling_status
from winward_scale.gpu_lock import ComputeBusy, run_local
from winward_scale.runtime import ScaledRuntime

STATIC = Path(__file__).parent / "static"


def create_app(engine_factory=LazyEngine, policy_factory=PolicyRuntime, lab_factory=LabRuntime, v3_factory=V3Runtime, v4_factory=V4Runtime):
    @asynccontextmanager
    async def lifespan(app):
        app.state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jev-inference")
        app.state.lock = asyncio.Lock()
        try:
            loop = asyncio.get_running_loop()
            app.state.engine = await loop.run_in_executor(app.state.executor, engine_factory)
            app.state.policy = policy_factory()
            app.state.lab = lab_factory()
            app.state.v3 = v3_factory()
            app.state.v4 = v4_factory()
            app.state.scaled = ScaledRuntime()
            yield
        finally:
            app.state.executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="Winward", version="0.2.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "::1", "testserver"])

    @app.middleware("http")
    async def local_browser_only(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method == "POST" and origin:
            try:
                parsed = urlsplit(origin)
                allowed = (parsed.scheme == request.url.scheme
                           and parsed.netloc == request.headers.get("host")
                           and parsed.path == "" and not parsed.query and not parsed.fragment)
            except ValueError:
                allowed = False
            if not allowed:
                return JSONResponse({"detail": "Cross-origin requests are not allowed."}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
            "object-src 'none'; base-uri 'self'; form-action 'self'"
        )
        return response

    async def infer(method, *args):
        # Keep at most one running inference; avoid unbounded GPU work queues.
        if app.state.lock.locked():
            raise HTTPException(429, "The model is busy. Wait for the current request to finish.")
        async with app.state.lock:
            loop = asyncio.get_running_loop()
            work = loop.run_in_executor(app.state.executor, partial(run_local, method, *args))
            try:
                return await asyncio.shield(work)
            except asyncio.CancelledError:
                # A disconnected client cannot release the model to a second
                # request while the original inference still runs.
                await work
                raise
            except ComputeBusy as error:
                raise HTTPException(503, str(error)) from error
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
            except ModelNotReadyError as error:
                raise HTTPException(503, str(error)) from error

    def check_model(model):
        if model is not None and model not in (MODEL_ID, "jev-local"):
            raise HTTPException(422, f"Loaded model is {MODEL_ID}. Use that name or omit model.")

    @app.get("/api/health")
    def health():
        return {"status": "ready", "model": MODEL_ID, "revision": MODEL_REVISION,
                "runtime": "MLX / Apple silicon", "offline": True,
                "qwen_loaded": not isinstance(app.state.engine, LazyEngine) or app.state.engine.engine is not None,
                "max_input_tokens": MAX_INPUT_TOKENS}

    @app.get("/api/examples")
    def examples():
        return {"examples": EXAMPLES}

    @app.get("/api/scale/status")
    def scale_status():
        return scaling_status()

    @app.get("/api/scale/model")
    def scale_model():
        return app.state.scaled.status()

    @app.post("/api/situation/decide4b")
    async def decide_situation_4b(request: SituationDecision):
        if not app.state.scaled.status()["ready"]:
            raise HTTPException(503, "Our trained 4B checkpoint is not ready to try yet. Follow progress at /scale.")
        return await infer(app.state.scaled.decide_situation, request)

    @app.get("/scale")
    def scale_page():
        return FileResponse(STATIC / "scale.html")

    @app.get("/api/v2/status")
    def lab_status():
        return app.state.lab.status()

    @app.get("/api/v3/status")
    def v3_status():
        return app.state.v3.status()

    @app.get("/api/v4/status")
    def v4_status():
        return app.state.v4.status()

    @app.post("/api/situation/interpret")
    async def interpret_situation(request: SituationInput):
        def interpret():
            started = time.perf_counter()
            result = app.state.engine.read_situation(request.situation, request.goal)
            interpreted = interpretation_result(request, result)
            interpreted["interpretation"] = {**interpreted["interpretation"],
                "model_work_ms": result["interpretation"]["latency_ms"],
                "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
            return interpreted
        return await infer(interpret)

    @app.post("/api/situation/decide")
    async def decide_situation(request: SituationDecision):
        return await infer(choose_next_step, app.state.v4, request)

    @app.get("/try")
    def situation_page():
        return FileResponse(STATIC / "situation.html")

    @app.get("/api/v4/examples")
    def v4_examples():
        return app.state.v4.examples()

    @app.post("/api/v4/decide")
    async def v4_decide(request: UncertaintyRequest):
        return await infer(app.state.v4.decide, request.example_id, request.max_depth)

    @app.post("/api/v4/episode")
    async def v4_episode(request: V3EpisodeRequest):
        return await infer(app.state.v4.episode, request.example_id, request.seed, request.max_depth)

    @app.get("/v4")
    def v4_page():
        return FileResponse(STATIC / "v4.html")

    @app.get("/api/v3/examples")
    def v3_examples():
        return app.state.v3.examples()

    @app.post("/api/v3/decide")
    async def v3_decide(request: UncertaintyRequest):
        return await infer(app.state.v3.decide, request.example_id, request.max_depth)

    @app.post("/api/v3/episode")
    async def v3_episode(request: V3EpisodeRequest):
        return await infer(app.state.v3.episode, request.example_id, request.seed, request.max_depth)

    @app.get("/v3")
    def v3_page():
        return FileResponse(STATIC / "v3.html")

    @app.get("/api/v2/examples")
    def lab_examples():
        return app.state.lab.examples()

    @app.post("/api/v2/uncertainty")
    async def uncertainty(request: UncertaintyRequest):
        return await infer(app.state.lab.uncertainty, request.example_id, request.max_depth)

    @app.post("/api/v2/sandbox")
    async def sandbox(request: SandboxRequest):
        return await infer(app.state.lab.sandbox, request.seed, request.kind,
                           request.changed_goal, request.uncertain, request.policy)

    @app.get("/v2")
    def lab_page():
        return FileResponse(STATIC / "v2.html")

    @app.get("/api/policy/status")
    def policy_status():
        return app.state.policy.status()

    @app.get("/api/policy/examples")
    def policy_examples():
        return app.state.policy.examples()

    @app.post("/api/policy/decide")
    async def policy_decide(request: PolicyRequest):
        return await infer(app.state.policy.decide, request.scenario.model_dump(),
                           request.token_weight, request.latency_weight, request.max_depth)

    @app.get("/policy")
    def policy_page():
        return FileResponse(STATIC / "policy.html")

    @app.post("/api/decide")
    async def decide(request: DecisionRequest):
        return await infer(app.state.engine.decide, request)

    @app.post("/api/generate")
    async def generate(request: GenerationRequest):
        return await infer(app.state.engine.generate, request)

    @app.post("/tokenize")
    async def tokenize(request: TokenizeRequest):
        check_model(request.model)
        return {"tokens": await infer(app.state.engine.tokenize, request.prompt, request.add_special_tokens)}

    @app.post("/v1/score")
    async def score(request: ScoreRequest):
        check_model(request.model)
        return await infer(app.state.engine.score, request)

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
