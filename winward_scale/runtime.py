"""Explicit experimental 4B selection; never silently replace the frozen v4."""
import hashlib
import json
import math
from pathlib import Path
import re

from agent_lab.situation import choose_next_step

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "runs" / "scale-selected-model.json"
TARGET_PARAMETERS = 4_251_056_641
TARGET_ARCHITECTURE = {"input_dim": 505, "width": 7680, "layers": 6, "heads": 240, "expansion": 4}
MODEL_LABEL = "Our Winward 4.25B · experimental"
MAX_METADATA_BYTES = 4 * 1024 * 1024


def _json_object(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("Selected model metadata is unavailable or invalid.")
    value = json.loads(path.read_text(encoding="utf-8"),
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Invalid JSON number")))
    if not isinstance(value, dict):
        raise ValueError("Selected model metadata must be an object.")
    return value


def _digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def _valid_config(config):
    if not isinstance(config, dict) or set(config) - (set(TARGET_ARCHITECTURE) | {
        "dtype", "norm_dtype", "gradient_checkpointing", "norm_eps"
    }):
        return False
    if any(type(config.get(key)) is not int or config[key] != value for key, value in TARGET_ARCHITECTURE.items()):
        return False
    eps = config.get("norm_eps", 1e-5)
    return (config.get("dtype", "bfloat16") == "bfloat16"
            and config.get("norm_dtype", "float32") == "float32"
            and type(config.get("gradient_checkpointing", True)) is bool
            and type(eps) in (int, float) and math.isfinite(eps) and 0 < eps <= 1)


class ScaledRuntime:
    def __init__(self, manifest=MANIFEST):
        self.manifest = Path(manifest)
        self.policy = None
        self.loaded_sha256 = None
        self.loaded_identity = None

    def _selection(self):
        try:
            value = _json_object(self.manifest)
            relative = value.get("checkpoint")
            if not isinstance(relative, str) or not relative:
                raise ValueError("The selected checkpoint path is invalid.")
            relative = Path(relative)
            if (relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 3
                    or relative.parts[0] != "runs" or not relative.parts[1].startswith("scale-")):
                raise ValueError("The selected checkpoint must be a local scaling run.")
            path = (ROOT / relative).resolve()
            if not path.is_relative_to((ROOT / "runs").resolve()):
                raise ValueError("The selected checkpoint must be a local scaling run.")
            info = _json_object(path / "checkpoint.json")
            weights = path / "model.safetensors"
            checksum = value.get("model_sha256")
            if (not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
                    or info.get("model_sha256") != checksum):
                raise ValueError("Selected 4B checkpoint checksum metadata does not match.")
            if (info.get("track") != "structured" or type(info.get("parameters")) is not int
                    or info["parameters"] != TARGET_PARAMETERS
                    or type(info.get("step")) is not int or info["step"] < 1
                    or not _valid_config(info.get("config"))):
                raise ValueError("No trained checkpoint of the supported 4.25B architecture has been selected.")
            if weights.is_symlink() or not weights.is_file():
                raise ValueError("Selected 4B checkpoint weights are unavailable.")
            return value, path, info
        except (OSError, UnicodeError, KeyError, TypeError, OverflowError) as error:
            raise ValueError("The selected 4B checkpoint is unavailable or invalid.") from error

    def _identity(self, path, info):
        stat = (path / "model.safetensors").stat()
        return (str(path), info["model_sha256"], json.dumps(info["config"], sort_keys=True),
                stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def status(self):
        try:
            value, path, info = self._selection()
            identity = self._identity(path, info)
        except (OSError, ValueError, KeyError, TypeError):
            return {"ready": False, "label": "Our 4B model is not ready to try yet."}
        loaded = self.policy is not None and self.loaded_identity == identity
        return {"ready": True, "parameters": info["parameters"],
                "label": MODEL_LABEL,
                "loaded": loaded, "checksum_verification": "verified_on_load" if loaded else "required_before_load",
                "quality_promoted": False, "model_sha256": value["model_sha256"],
                "limitation": "An expanded and further-trained version of our own structured policy. Larger does not mean better; compare the measured results."}

    def _policy(self):
        try:
            _, path, info = self._selection()
            identity = self._identity(path, info)
            if self.policy is not None and self.loaded_identity == identity:
                return self.policy
            self.policy, self.loaded_sha256, self.loaded_identity = None, None, None
            if _digest(path / "model.safetensors") != info["model_sha256"]:
                raise ValueError("Selected 4B checkpoint checksum failed.")
            policy = self._load_policy(path, info)
            # A checkpoint/manifest replacement during loading must not publish
            # a stale model under the new selection.
            _, final_path, final_info = self._selection()
            if self._identity(final_path, final_info) != identity:
                raise ValueError("The selected model changed while loading; try again.")
            self.policy = policy
            self.loaded_sha256 = info["model_sha256"]
            self.loaded_identity = identity
            return self.policy
        except (OSError, RuntimeError, TypeError) as error:
            self.policy, self.loaded_sha256, self.loaded_identity = None, None, None
            raise ValueError("The selected 4B checkpoint could not be loaded.") from error
        except ValueError:
            self.policy, self.loaded_sha256, self.loaded_identity = None, None, None
            raise

    @staticmethod
    def _load_policy(path, info):
        # Import the GPU-backed classes only after all filesystem and checksum
        # guards pass. Unit tests replace this boundary with a tiny stub.
        from .structured import StructuredConfig, StructuredPolicy
        from agent_training.model_v4 import V4Policy
        model = StructuredPolicy(StructuredConfig(**info["config"]))
        if model.count_parameters() != info["parameters"]:
            raise ValueError("Selected checkpoint parameter count is inconsistent.")
        model.load_weights(str(path / "model.safetensors"), strict=True)
        model.eval()
        return V4Policy(model=model, report=info)

    def decide_situation(self, request):
        result = choose_next_step(self, request)
        if result.get("status") == "recommended":
            result["source"] = {"reader": "Qwen3 4B (local text reader)",
                                "decision": f"Our Winward 4.25B · {TARGET_PARAMETERS:,} parameters"}
        result["limitation"] = ("Experimental structured software policy expanded and trained from our own weights. "
            "It is not a general language model. The text adapter and supplied action effects limit what it can decide. "
            "Parameter size does not establish better decisions; see /scale for measured results.")
        return result
