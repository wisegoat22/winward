"""The only command in this project that downloads model artifacts."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

from huggingface_hub import snapshot_download
from jev_local.config import MODEL_ID, MODEL_REVISION

print(f"Downloading {MODEL_ID} (approximately 2.3 GB).")
path = snapshot_download(
    MODEL_ID, revision=MODEL_REVISION,
    allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors", "README.md"],
)
print(f"Model ready: {path}")
