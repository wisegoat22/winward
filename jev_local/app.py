import asyncio
import os
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
from .engine import Engine
from .examples import EXAMPLES
from .schemas import DecisionRequest, GenerationRequest, ScoreRequest, TokenizeRequest
from .policy_schemas import PolicyRequest
from agent_training.runtime import PolicyRuntime

STATIC = Path(__file__).parent / "static"


def create_app(engine_factory=Engine, policy_factory=PolicyRuntime):
    @asynccontextmanager
    async def lifespan(app):
        app.state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jev-inference")
        app.state.lock = asyncio.Lock()
        try:
            loop = asyncio.get_running_loop()
            app.state.engine = await loop.run_in_executor(app.state.executor, engine_factory)
            app.state.policy = policy_factory()
            yield
        finally:
            app.state.executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="JEV Local", version="0.1.0", lifespan=lifespan,
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
            work = loop.run_in_executor(app.state.executor, partial(method, *args))
            try:
                return await asyncio.shield(work)
            except asyncio.CancelledError:
                # A disconnected client cannot release the model to a second
                # request while the original inference still runs.
                await work
                raise
            except ValueError as error:
                raise HTTPException(422, str(error)) from error

    def check_model(model):
        if model is not None and model not in (MODEL_ID, "jev-local"):
            raise HTTPException(422, f"Loaded model is {MODEL_ID}. Use that name or omit model.")

    @app.get("/api/health")
    def health():
        return {"status": "ready", "model": MODEL_ID, "revision": MODEL_REVISION,
                "runtime": "MLX / Apple silicon", "offline": True,
                "max_input_tokens": MAX_INPUT_TOKENS}

    @app.get("/api/examples")
    def examples():
        return {"examples": EXAMPLES}

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
