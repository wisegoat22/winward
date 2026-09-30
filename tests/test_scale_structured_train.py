"""Tiny trained-checkpoint/optimizer integrity checks; no generated corpus."""
import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from agent_training.model import PolicyConfig
from agent_training.model_v3 import UnifiedPolicy
from winward_scale.structured import grow_structured
from winward_scale import structured_train as training


def arrays(model):
    return {name: np.asarray(value.astype(mx.float32)).copy()
            for name, value in tree_flatten(model.parameters())}


def make_batch():
    x = np.random.default_rng(736).normal(size=(2, 4, 505)).astype(np.float32)
    valid = np.ones((2, 4), dtype=bool)
    targets = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    return tuple(mx.array(value) for value in (x, valid, valid, targets))


def optimizer():
    return training.FP32Adafactor(learning_rate=.01, relative_step=False,
                                  scale_parameter=False, beta_1=None)


def update(model, opt, batch):
    value, grads = nn.value_and_grad(model, training.decision_loss)(model, *batch)
    opt.update(model, grads)
    mx.eval(value, model.parameters(), opt.state)
    return float(value.item())


@pytest.fixture
def saved(tmp_path):
    mx.random.seed(920)
    source = UnifiedPolicy(PolicyConfig(input_dim=505, width=8, layers=1, heads=2))
    model = grow_structured(source, 2)
    opt, batch = optimizer(), make_batch()
    assert np.isfinite(update(model, opt, batch))
    run = tmp_path / "parent"
    run.mkdir()
    info = training.save_checkpoint(run, model, opt, 1, {"total_optimizer_updates": 1})
    return run, model, opt, batch, info


def test_trained_model_optimizer_reload_preserves_next_update(saved):
    run, model, opt, batch, info = saved
    restored, checkpoint, loaded_info, expected = training.load_parent(run)
    assert expected == info["model_sha256"] and loaded_info == info
    assert training.digest(checkpoint / "optimizer.safetensors") == info["optimizer_sha256"]
    resumed = optimizer()
    resumed.state = tree_unflatten(list(mx.load(str(checkpoint / "optimizer.safetensors")).items()))
    for name, value in arrays(model).items():
        np.testing.assert_array_equal(value, arrays(restored)[name])
    before_state = dict(tree_flatten(opt.state))
    for name, value in tree_flatten(resumed.state):
        np.testing.assert_array_equal(np.asarray(value), np.asarray(before_state[name]))
    assert update(model, opt, batch) == pytest.approx(update(restored, resumed, batch), abs=1e-6)
    final = arrays(restored)
    for name, value in arrays(model).items():
        np.testing.assert_array_equal(value, final[name])
    assert int(resumed.step.item()) == int(opt.step.item()) == 2


@pytest.mark.parametrize("field,value,match", [("track", "byte", "structured checkpoint"),
                                               ("model_sha256", "wrong", "checksum")])
def test_parent_rejects_wrong_track_or_modified_model(saved, field, value, match):
    run, _, _, _, _ = saved
    checkpoint = training.resolve_checkpoint(run)
    path = checkpoint / "checkpoint.json"
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        training.load_parent(run)


def test_initial_parent_is_exact_local_v4_weights_with_checked_schema(tmp_path):
    source = UnifiedPolicy(PolicyConfig(input_dim=505, width=8, layers=1, heads=2))
    source.save_weights(str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(source.config.to_dict()))
    report = {"schema_version": "winward-v4-public-observation-1",
              "checkpoint_sha256": training.digest(tmp_path / "model.safetensors")}
    (tmp_path / "report.json").write_text(json.dumps(report))
    restored, path, info, sha = training.load_parent(tmp_path)
    assert type(restored) is UnifiedPolicy and path == tmp_path
    assert sha == report["checkpoint_sha256"]
    for name, value in arrays(source).items():
        np.testing.assert_array_equal(value, arrays(restored)[name])
    report["schema_version"] = "unrecognized-external-model"
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="own frozen v4"):
        training.load_parent(tmp_path)


def args_for(tmp_path, source):
    return training.parser().parse_args(["--output", str(tmp_path / "child"),
        "--init-from", str(source), "--steps", "1", "--train-cases", "2",
        "--valid-cases", "2", "--batch-size", "2", "--eval-every", "1",
        "--save-every", "1", "--log-every", "1", "--memory-gib", "8"])


@pytest.mark.parametrize("name,value", [("factor", 0), ("steps", 0), ("batch_size", 0),
    ("validation_batch_size", 0), ("learning_rate", float("nan")),
    ("learning_rate", -1), ("memory_gib", 29), ("noise", .2)])
def test_invalid_training_arguments_fail_before_reading_parent(tmp_path, name, value):
    args = args_for(tmp_path, tmp_path / "does-not-exist")
    setattr(args, name, value)
    with pytest.raises(ValueError):
        training.train(args)
    assert not (tmp_path / "child").exists()


def fake_training_data(monkeypatch, batch):
    rows = [{"_encoded": tuple(np.asarray(value)[i] for value in batch)} for i in range(2)]
    monkeypatch.setattr(training, "corpus", lambda count, split, seed: rows)
    def evaluate(model, examples, batch_size):
        scores = np.asarray(model(*batch[:3]))
        return {"examples": 2, "next_action_accuracy": float(np.mean(scores.argmax(-1) == [0, 1]))}
    monkeypatch.setattr(training, "evaluate", evaluate)
    monkeypatch.setattr(training, "source_hashes", lambda: {})
    monkeypatch.setattr(training.signal, "signal", lambda *args: None)
    # These tests exercise checkpoint correctness, not global resource settings.
    monkeypatch.setattr(mx, "set_memory_limit", lambda *args: None)
    monkeypatch.setattr(mx, "set_cache_limit", lambda *args: None)


def test_continue_optimizer_checks_checksum_before_training(saved, tmp_path, monkeypatch):
    run, _, _, batch, _ = saved
    checkpoint = training.resolve_checkpoint(run)
    metadata = checkpoint / "checkpoint.json"
    info = json.loads(metadata.read_text())
    info["optimizer_sha256"] = "modified"
    metadata.write_text(json.dumps(info))
    fake_training_data(monkeypatch, batch)
    args = args_for(tmp_path, run)
    args.continue_optimizer = True
    with pytest.raises(ValueError, match="Optimizer checkpoint checksum"):
        training.train(args)
    assert not (tmp_path / "child" / "latest.json").exists()


def test_tiny_full_training_resumes_model_and_optimizer(saved, tmp_path, monkeypatch):
    run, _, _, batch, _ = saved
    fake_training_data(monkeypatch, batch)
    args = args_for(tmp_path, run)
    args.continue_optimizer = True
    training.train(args)
    output = tmp_path / "child"
    restored, checkpoint, info, _ = training.load_parent(output)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "completed"
    assert report["step"] == 1 and report["total_optimizer_updates"] == 2
    assert info["total_optimizer_updates"] == 2
    state = tree_unflatten(list(mx.load(str(checkpoint / "optimizer.safetensors")).items()))
    resumed = optimizer()
    resumed.state = state
    assert int(resumed.step.item()) == 2
    assert np.isfinite(np.asarray(restored(*batch[:3]))).all()
    assert report["first_update_sample_check"]["tensors_with_changed_sample"] > 0
