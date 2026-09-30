"""Selection, API and local-compute guards without loading Qwen or a 4B model."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

from fastapi.testclient import TestClient
import pytest

from jev_local.situation_schemas import SituationDecision
from winward_scale import gpu_lock
from winward_scale import runtime as scaled


CONFIG = {**scaled.TARGET_ARCHITECTURE, "dtype": "bfloat16", "norm_dtype": "float32",
          "gradient_checkpointing": True, "norm_eps": 1e-5}
BODY = {
    "situation": "I have the source code for a failing unit test.",
    "goal": "Identify the cause of the failing unit test.",
    "draft": {"domain": "software", "task_kind": "investigate", "summary": "Identify the cause.",
              "confirmed": [{"fact": "context_available", "evidence": "I have the source code"}],
              "restrictions": [], "unknowns": []},
    "confirmed_facts": ["context_available", "requirements_clear"], "max_depth": 5,
}


class FakePolicy:
    def __init__(self):
        self.calls = []

    def _policy(self):
        return self

    def choose(self, scenario, remaining):
        self.calls.append((scenario, remaining))
        return {"action_id": "understand", "decision_ms": 1.2}


@pytest.fixture
def selection(tmp_path, monkeypatch):
    monkeypatch.setattr(scaled, "ROOT", tmp_path)
    monkeypatch.setattr(gpu_lock, "__file__", str(tmp_path / "winward_scale" / "gpu_lock.py"))
    manifest = tmp_path / "runs" / "scale-selected-model.json"
    manifest.parent.mkdir()

    def checkpoint(name="first", payload=b"fake weights for guard tests", **overrides):
        relative = Path("runs") / f"scale-structured-{name}" / "checkpoints" / "step-00000001"
        path = tmp_path / relative
        path.mkdir(parents=True)
        (path / "model.safetensors").write_bytes(payload)
        checksum = hashlib.sha256(payload).hexdigest()
        info = {"track": "structured", "parameters": scaled.TARGET_PARAMETERS, "step": 1,
                "model_sha256": checksum, "config": dict(CONFIG), **overrides}
        (path / "checkpoint.json").write_text(json.dumps(info))
        chosen = {"checkpoint": str(relative), "model_sha256": checksum}
        manifest.write_text(json.dumps(chosen))
        return path, info, chosen

    return manifest, checkpoint, tmp_path


def test_missing_selection_and_missing_weights_are_unavailable_without_loading(selection, monkeypatch):
    manifest, make, _ = selection
    runtime = scaled.ScaledRuntime(manifest)
    monkeypatch.setattr(runtime, "_load_policy", lambda *args: pytest.fail("must not load a model"))
    assert not runtime.status()["ready"]
    with pytest.raises(ValueError):
        runtime._policy()
    path, _, _ = make()
    (path / "model.safetensors").unlink()
    assert not runtime.status()["ready"]
    with pytest.raises(ValueError):
        runtime._policy()


@pytest.mark.parametrize("overrides", [
    {"parameters": 4_100_000_000}, {"parameters": True}, {"step": 0}, {"step": True},
    {"track": "byte"}, {"config": {**CONFIG, "width": 512}},
    {"config": {**CONFIG, "heads": 8}}, {"config": {**CONFIG, "dtype": "float32"}},
    {"config": {**CONFIG, "unused": "unsupported"}}, {"model_sha256": "0" * 64},
])
def test_only_exact_trained_supported_architecture_is_advertised(selection, overrides, monkeypatch):
    manifest, make, _ = selection
    make(**overrides)
    runtime = scaled.ScaledRuntime(manifest)
    monkeypatch.setattr(runtime, "_load_policy", lambda *args: pytest.fail("must not load a model"))
    assert not runtime.status()["ready"]
    with pytest.raises(ValueError):
        runtime._policy()


@pytest.mark.parametrize("bad", [[], {}, {"checkpoint": []}, {"checkpoint": "runs/x", "model_sha256": "bad"}])
def test_malformed_manifest_fails_closed(selection, bad):
    manifest, _, _ = selection
    manifest.write_text(json.dumps(bad))
    runtime = scaled.ScaledRuntime(manifest)
    assert not runtime.status()["ready"]
    with pytest.raises(ValueError):
        runtime._policy()


@pytest.mark.parametrize("path_value", ["../outside", "runs/../outside", "/tmp/absolute", "runs/goalpolicy-v4/checkpoint"])
def test_manifest_path_cannot_escape_or_select_an_unrelated_run(selection, path_value):
    manifest, make, _ = selection
    _, _, chosen = make()
    manifest.write_text(json.dumps({**chosen, "checkpoint": path_value}))
    assert not scaled.ScaledRuntime(manifest).status()["ready"]


def test_symlinked_weights_or_metadata_cannot_cross_checkpoint_boundary(selection):
    manifest, make, root = selection
    path, _, _ = make()
    external = root / "external-weights"
    external.write_bytes(b"external")
    (path / "model.safetensors").unlink()
    (path / "model.safetensors").symlink_to(external)
    assert not scaled.ScaledRuntime(manifest).status()["ready"]
    path2, _, _ = make("second")
    copied = root / "outside-metadata.json"
    copied.write_bytes((path2 / "checkpoint.json").read_bytes())
    (path2 / "checkpoint.json").unlink()
    (path2 / "checkpoint.json").symlink_to(copied)
    assert not scaled.ScaledRuntime(manifest).status()["ready"]


def test_status_is_metadata_only_but_real_checksum_precedes_every_first_load(selection, monkeypatch):
    manifest, make, _ = selection
    path, _, _ = make()
    runtime = scaled.ScaledRuntime(manifest)
    monkeypatch.setattr(runtime, "_load_policy", lambda *args: pytest.fail("checksum must fail first"))
    (path / "model.safetensors").write_bytes(b"corrupt")
    status = runtime.status()
    assert status["ready"] and not status["loaded"]
    assert status["checksum_verification"] == "required_before_load"
    with pytest.raises(ValueError, match="checksum failed"):
        runtime._policy()
    assert runtime.policy is None and runtime.loaded_sha256 is None


def test_cache_reuses_verified_model_and_reloads_on_selection_or_file_change(selection, monkeypatch):
    manifest, make, _ = selection
    path, _, _ = make()
    runtime, loads = scaled.ScaledRuntime(manifest), []

    def load(path, info):
        policy = FakePolicy()
        loads.append((path, policy))
        return policy

    monkeypatch.setattr(runtime, "_load_policy", load)
    original = runtime._policy()
    assert runtime._policy() is original and len(loads) == 1
    assert runtime.status()["loaded"] and runtime.status()["checksum_verification"] == "verified_on_load"
    # Even identical weight hashes at a different selected path get a new identity.
    second, _, _ = make("second")
    assert not runtime.status()["loaded"]
    assert runtime._policy() is not original and len(loads) == 2
    weights = second / "model.safetensors"
    old = weights.stat()
    weights.write_bytes(b"changed")
    os.utime(weights, ns=(old.st_atime_ns, old.st_mtime_ns + 1))
    assert not runtime.status()["loaded"]
    with pytest.raises(ValueError, match="checksum failed"):
        runtime._policy()
    assert runtime.policy is None and len(loads) == 2


def test_selection_change_during_loading_is_not_published(selection, monkeypatch):
    manifest, make, _ = selection
    make()
    runtime = scaled.ScaledRuntime(manifest)

    def load(path, info):
        make("replacement", payload=b"other weights")
        return FakePolicy()

    monkeypatch.setattr(runtime, "_load_policy", load)
    with pytest.raises(ValueError, match="changed while loading"):
        runtime._policy()
    assert runtime.policy is None and not runtime.status()["loaded"]


def test_failed_new_selection_never_falls_back_to_cached_old_policy(selection, monkeypatch):
    manifest, make, _ = selection
    make()
    runtime = scaled.ScaledRuntime(manifest)
    monkeypatch.setattr(runtime, "_load_policy", lambda *args: FakePolicy())
    runtime._policy()
    manifest.unlink()
    with pytest.raises(ValueError):
        runtime._policy()
    assert runtime.policy is None


def test_recommendation_source_matches_model_and_unsupported_does_not_claim_a_decision(selection, monkeypatch):
    manifest, make, _ = selection
    make()
    runtime, policy = scaled.ScaledRuntime(manifest), FakePolicy()
    monkeypatch.setattr(runtime, "_load_policy", lambda *args: policy)
    result = runtime.decide_situation(SituationDecision.model_validate(BODY))
    assert len(policy.calls) == 1 and result["choice"]["id"] == "understand"
    assert "4,251,056,641" in result["source"]["decision"]
    unsupported = {**BODY, "draft": {**BODY["draft"], "domain": "unsupported"}}
    result = runtime.decide_situation(SituationDecision.model_validate(unsupported))
    assert result["status"] == "unsupported" and "source" not in result and len(policy.calls) == 1


def test_local_compute_lock_excludes_a_separate_process_and_releases_on_error(selection):
    _, _, root = selection
    child = """import fcntl,sys
with open(sys.argv[1], 'a') as handle:
    try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: sys.exit(7)
"""
    with gpu_lock.local_compute():
        result = subprocess.run([sys.executable, "-c", child, str(root / ".runtime" / "metal.lock")])
        assert result.returncode == 7
        with pytest.raises(gpu_lock.ComputeBusy):
            gpu_lock.run_local(lambda: pytest.fail("busy call must not execute"))
    with pytest.raises(RuntimeError):
        gpu_lock.run_local(lambda: (_ for _ in ()).throw(RuntimeError("test")))
    assert gpu_lock.run_local(lambda: "released") == "released"


def test_api_default_remains_v4_and_4b_requires_explicit_ready_selection(selection, monkeypatch):
    import jev_local.app as app_module
    manifest, make, _ = selection
    runtime, policy4b, policyv4 = scaled.ScaledRuntime(manifest), FakePolicy(), FakePolicy()
    monkeypatch.setattr(runtime, "_load_policy", lambda *args: policy4b)
    monkeypatch.setattr(app_module, "ScaledRuntime", lambda: runtime)
    no_model = lambda: object()
    app = app_module.create_app(no_model, no_model, no_model, no_model, lambda: policyv4)
    with TestClient(app) as client:
        assert not client.get("/api/scale/model").json()["ready"]
        assert client.post("/api/situation/decide4b", json=BODY).status_code == 503
        old = client.post("/api/situation/decide", json=BODY)
        assert old.status_code == 200 and "4.86M" in old.json()["source"]["decision"]
        assert len(policyv4.calls) == 1 and not policy4b.calls
        path, _, _ = make()
        assert client.get("/api/scale/model").json()["ready"] and not policy4b.calls
        new = client.post("/api/situation/decide4b", json=BODY)
        assert new.status_code == 200 and "4,251,056,641" in new.json()["source"]["decision"]
        assert len(policy4b.calls) == 1 and len(policyv4.calls) == 1
        blocked = client.post("/api/situation/decide4b", json=BODY, headers={"Origin": "https://outside.invalid"})
        assert blocked.status_code == 403 and len(policy4b.calls) == 1
        (path / "model.safetensors").write_bytes(b"bad checkpoint")
        assert client.post("/api/situation/decide4b", json=BODY).status_code == 422
        assert len(policy4b.calls) == 1 and len(policyv4.calls) == 1
        with gpu_lock.local_compute():
            busy = client.post("/api/situation/decide", json=BODY)
            assert busy.status_code == 503 and len(policyv4.calls) == 1


def test_runtime_guard_imports_do_not_load_mlx():
    command = "import sys; import winward_scale.runtime; assert not any(k == 'mlx' or k.startswith('mlx.') for k in sys.modules)"
    subprocess.run([sys.executable, "-c", command], check=True, capture_output=True)


def test_model_selector_dom_uses_explicit_endpoint_and_never_falls_back():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed for the lightweight DOM test")
    script = Path(__file__).with_name("scale_model_ui.cjs")
    result = subprocess.run([node, str(script)], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
