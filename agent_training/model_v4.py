"""Winward v4: the same shared policy, with a separately frozen schema/run.

The architecture and public input channels are unchanged from v3. Only the
training curriculum/objective changes. Inference never reads teacher values,
training provenance, model alternatives, or a realized hidden world.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .model import PolicyConfig
from .model_v3 import (BELIEF_DIM, FEATURE_DIM, GraphView, UnifiedPolicy,
                       V3Policy, encode_graph, encode_uncertainty, expand_belief,
                       expand_graph, initialize_from_v1)

SCHEMA_VERSION = "winward-v4-public-observation-1"


class V4Policy(V3Policy):
    """One learned checkpoint using the unchanged public-observation encoder."""
    def __init__(self, run_dir="runs/goalpolicy-v4", *, model=None, report=None):
        self.run_dir = Path(run_dir)
        if model is not None:
            self.model, self.report = model, report or {}
        else:
            self.report = json.loads((self.run_dir / "report.json").read_text())
            checkpoint = self.run_dir / "model.safetensors"
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != self.report["checkpoint_sha256"]:
                raise ValueError("Checkpoint does not match the frozen v4 training report")
            if self.report.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("Checkpoint uses a different public-observation schema")
            self.model = UnifiedPolicy(PolicyConfig(**json.loads((self.run_dir / "config.json").read_text())))
            self.model.load_weights(str(checkpoint))
            self.model.eval()
        self.config = self.model.config


UnifiedPredictor = V4Policy
