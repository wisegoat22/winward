"""One shared decision policy for graph observations and public beliefs.

Input formats have explicit channels; no simulator family, answer, hidden world,
or teacher score is supplied. Existing graph weights can be transferred from our
own v1 checkpoint with exactly unchanged graph logits before further training.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import mlx.core as mx
import numpy as np

from .features import FEATURE_DIM as GRAPH_DIM, encode
from .model import GoalPolicy, PolicyConfig

BELIEF_DIM = 400
FEATURE_DIM = GRAPH_DIM + BELIEF_DIM + 2
GRAPH_DOMAIN_INDEX = FEATURE_DIM - 2
BELIEF_DOMAIN_INDEX = FEATURE_DIM - 1
SCHEMA_VERSION = "winward-v3-public-observation-1"


def expand_graph(x):
    """Pad graph arrays, preserving the original 103-feature interpretation."""
    if x.shape[-1] != GRAPH_DIM:
        raise ValueError(f"Expected {GRAPH_DIM} graph features")
    result = np.zeros((*x.shape[:-1], FEATURE_DIM), dtype=np.float32)
    result[..., :GRAPH_DIM] = x
    result[..., GRAPH_DOMAIN_INDEX] = 1.0
    return result


def expand_belief(x):
    if x.shape[-1] != BELIEF_DIM:
        raise ValueError(f"Expected {BELIEF_DIM} public belief features")
    result = np.zeros((*x.shape[:-1], FEATURE_DIM), dtype=np.float32)
    result[..., GRAPH_DIM:GRAPH_DIM + BELIEF_DIM] = x
    result[..., BELIEF_DOMAIN_INDEX] = 1.0
    return result


def encode_graph(scenario, token_weight=1.0, latency_weight=0.01, max_depth=5):
    x, valid, eligible, candidates = encode(scenario, token_weight, latency_weight, max_depth)
    return expand_graph(x), valid, eligible, candidates


def encode_uncertainty(problem, depth=5, token_weight=1.0, latency_weight=0.01):
    from .belief_data import encode_belief
    x, valid, eligible, candidates = encode_belief(problem, depth, token_weight, latency_weight)
    return expand_belief(x), valid, eligible, candidates


class UnifiedPolicy(GoalPolicy):
    def __init__(self, config=None):
        config = config or PolicyConfig(input_dim=FEATURE_DIM)
        if config.input_dim != FEATURE_DIM:
            raise ValueError(f"This observation schema requires {FEATURE_DIM} features")
        super().__init__(config)


def initialize_from_v1(model, source_model):
    """Transfer our own trained weights, zero-initializing all new input columns."""
    from mlx.utils import tree_flatten
    if source_model.config.input_dim != GRAPH_DIM:
        raise ValueError("Initialization requires the 103-feature v1 checkpoint")
    for field in ("width", "layers", "heads", "expansion"):
        if getattr(model.config, field) != getattr(source_model.config, field):
            raise ValueError("Source and destination transformer configurations must match")
    weights = dict(tree_flatten(source_model.parameters()))
    old = weights["input_projection.weight"]
    weights["input_projection.weight"] = mx.concatenate(
        [old, mx.zeros((old.shape[0], FEATURE_DIM - GRAPH_DIM), dtype=old.dtype)], axis=1)
    model.load_weights(list(weights.items()))


class GraphView:
    """Use existing graph audits without altering or misinterpreting their encoder."""
    def __init__(self, model):
        self.model = model
        self.config = PolicyConfig(**{**model.config.to_dict(), "input_dim": GRAPH_DIM})

    def __call__(self, x, valid, eligible):
        zeros = mx.zeros((*x.shape[:-1], BELIEF_DIM), dtype=x.dtype)
        domains = mx.broadcast_to(mx.array([1.0, 0.0], dtype=x.dtype), (*x.shape[:-1], 2))
        return self.model(mx.concatenate([x, zeros, domains], axis=-1), valid, eligible)


class V3Policy:
    def __init__(self, run_dir="runs/goalpolicy-v3", *, model=None, report=None):
        self.run_dir = Path(run_dir)
        if model is not None:
            self.model = model
            self.report = report or {}
        else:
            self.report = json.loads((self.run_dir / "report.json").read_text())
            checkpoint = self.run_dir / "model.safetensors"
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if digest != self.report["checkpoint_sha256"]:
                raise ValueError("Checkpoint does not match the frozen training report")
            if self.report.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("Checkpoint uses a different public-observation schema")
            self.model = UnifiedPolicy(PolicyConfig(**json.loads((self.run_dir / "config.json").read_text())))
            self.model.load_weights(str(checkpoint))
            self.model.eval()
        self.config = self.model.config

    def _predict(self, encoded, batch_size=128):
        result = []
        for start in range(0, len(encoded), batch_size):
            batch = encoded[start:start + batch_size]
            arrays = [mx.array(np.stack([item[i] for item in batch])) for i in range(3)]
            logits = self.model(*arrays)
            mx.eval(logits)
            scores = np.asarray(logits)
            for item, row in zip(batch, scores):
                index = int(row.argmax())
                shifted = row - row.max()
                probabilities = np.exp(shifted) / np.exp(shifted).sum()
                result.append({"action_id": item[3][index]["id"], "preference": float(probabilities[index])})
        return result

    def predict_graph(self, scenarios, depths=None, token_weights=None, latency_weights=None):
        n = len(scenarios)
        if any(values is not None and len(values) != n for values in (depths, token_weights, latency_weights)):
            raise ValueError("Every optional observation argument must match the scenario count")
        encoded = [encode_graph(s, tw, lw, depth) for s, tw, lw, depth in zip(
            scenarios, token_weights if token_weights is not None else [1.0] * n,
            latency_weights if latency_weights is not None else [0.01] * n,
            depths if depths is not None else [5] * n)]
        return [row["action_id"] for row in self._predict(encoded)]

    def predict_beliefs(self, problems, depths=None, token_weights=None, latency_weights=None):
        n = len(problems)
        if any(values is not None and len(values) != n for values in (depths, token_weights, latency_weights)):
            raise ValueError("Every optional observation argument must match the belief count")
        encoded = [encode_uncertainty(p, depth, tw, lw) for p, tw, lw, depth in zip(
            problems, token_weights if token_weights is not None else [1.0] * n,
            latency_weights if latency_weights is not None else [0.01] * n,
            depths if depths is not None else [5] * n)]
        return [row["action_id"] for row in self._predict(encoded)]

    def choose(self, scenario, remaining=5):
        started = time.perf_counter()
        result = self._predict([encode_graph(scenario, max_depth=remaining)])[0]
        result["decision_ms"] = (time.perf_counter() - started) * 1000
        return result

    def choose_belief(self, problem, remaining=5, token_weight=1.0, latency_weight=0.01):
        started = time.perf_counter()
        result = self._predict([encode_uncertainty(problem, remaining, token_weight, latency_weight)])[0]
        result["decision_ms"] = (time.perf_counter() - started) * 1000
        return result

    def predict_belief(self, problem, remaining_horizon=5, token_weight=1.0, latency_weight=0.01):
        return self.choose_belief(problem, remaining_horizon, token_weight, latency_weight)["action_id"]


UnifiedPredictor = V3Policy
