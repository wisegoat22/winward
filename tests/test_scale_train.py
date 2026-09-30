"""Training correctness using only tiny decoders and miniature in-memory data."""
import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from winward_scale.data import ByteTokenizer, TextExample
from winward_scale.model import Decoder, ModelConfig
from winward_scale import train as training


def tiny(dtype="bfloat16"):
    return ModelConfig(hidden_size=16, intermediate_size=32, num_layers=1,
                       num_attention_heads=2, num_key_value_heads=1,
                       vocab_size=260, dtype=dtype, name="unit-test-tiny")


def make_row(prompt="read evidence", answer="A", identifier="unit"):
    example = TextExample(prompt, answer, "unit", identifier, {})
    ids, mask = ByteTokenizer().encode_example(example)
    return np.array(ids, dtype=np.int32), np.array(mask, dtype=np.float32), example


def arrays(model):
    return {name: np.asarray(value.astype(mx.float32)).copy()
            for name, value in tree_flatten(model.parameters())}


def optimizer():
    return training.FP32Adafactor(learning_rate=.02, relative_step=False,
                                  scale_parameter=False, beta_1=None)


def update(model, opt, batch):
    loss, grads = nn.value_and_grad(model, training.decision_loss)(model, *batch)
    opt.update(model, grads)
    mx.eval(loss, model.parameters(), opt.state)
    return float(loss.item())


def test_completion_mask_predicts_first_answer_from_separator_and_excludes_padding():
    row = make_row("facts", "B")
    inputs, targets, mask = training.batch_arrays([row], 64)
    positions = np.flatnonzero(np.asarray(mask)[0])
    assert len(positions) == 2
    first = int(positions[0])
    assert int(inputs[0, first].item()) == ByteTokenizer.sep_id
    assert int(targets[0, first].item()) == ord("B")
    assert int(targets[0, first + 1].item()) == ByteTokenizer.eos_id
    assert np.count_nonzero(np.asarray(mask)[0, first + 2:]) == 0
    assert np.all(np.asarray(targets)[0, first + 2:] == ByteTokenizer.pad_id)


def test_prompt_predictions_cannot_change_completion_only_loss():
    inputs, targets, mask = training.batch_arrays([make_row()], 64)
    shape = (*inputs.shape, 260)
    good = np.zeros(shape, dtype=np.float32)
    for i in np.flatnonzero(np.asarray(mask)[0]):
        good[0, i, int(targets[0, i].item())] = 12
    distracting = good.copy()
    distracting[0, np.flatnonzero(np.asarray(mask)[0] == 0), 123] = 100

    class Fixed:
        def __init__(self, value): self.value = mx.array(value)
        def __call__(self, tokens): return self.value

    baseline = training.decision_loss(Fixed(good), inputs, targets, mask)
    changed = training.decision_loss(Fixed(distracting), inputs, targets, mask)
    assert float(baseline.item()) < .01
    assert float(changed.item()) == pytest.approx(float(baseline.item()), abs=1e-7)


def test_validation_scores_unconstrained_first_action_not_eos_or_prompt():
    rows = [make_row("known", "A", "a"), make_row("other evidence", "S", "b")]
    inputs, targets, mask = training.batch_arrays(rows, 64)
    scores = np.zeros((*inputs.shape, 260), dtype=np.float32)
    first = np.argmax(np.asarray(mask) > 0, axis=1)
    # PAD wins over the correct action: an unconstrained decoder must be wrong.
    scores[0, first[0], ByteTokenizer.pad_id] = 20
    scores[0, first[0], ord("A")] = 10
    scores[1, first[1], ord("S")] = 20
    # Both EOS predictions are deliberately wrong. They affect loss, not action accuracy.
    scores[:, :, ord("Z")] += 1

    class Fixed:
        training = True
        def eval(self): self.training = False
        def train(self): self.training = True
        def __call__(self, tokens):
            assert not self.training
            return mx.array(scores)

    model = Fixed()
    result = training.validate(model, rows, 64, batch_size=2)
    assert result["next_action_correct"] == 1
    assert result["next_action_accuracy"] == .5
    assert result["prediction_token_counts"] == {str(ByteTokenizer.pad_id): 1, str(ord("S")): 1}
    assert model.training is True


def test_adafactor_states_are_factored_fp32_and_all_bf16_tensors_really_update():
    mx.random.seed(889)
    model = Decoder(tiny())
    before = arrays(model)
    opt = optimizer()
    batch = training.batch_arrays([make_row("a", "A"), make_row("b", "S")], 64)
    values = [update(model, opt, batch) for _ in range(3)]
    assert np.isfinite(values).all()
    assert values[-1] < values[0]
    states = dict(tree_flatten(opt.state))
    total_parameter_elements = 0
    total_state_elements = 0
    for name, value in tree_flatten(model.parameters()):
        total_parameter_elements += value.size
        assert value.dtype == (mx.float32 if value.ndim == 1 else mx.bfloat16), name
        after = np.asarray(value.astype(mx.float32))
        assert np.isfinite(after).all(), name
        assert np.any(after != before[name]), name
        if value.ndim >= 2:
            expected = {"exp_avg_sq_row": value.shape[:-1],
                        "exp_avg_sq_col": value.shape[:-2] + value.shape[-1:]}
        else:
            expected = {"exp_avg_sq": value.shape}
        for suffix, shape in expected.items():
            moment = states[f"{name}.{suffix}"]
            assert moment.shape == shape
            assert moment.dtype == mx.float32
            assert bool(mx.all(mx.isfinite(moment)).item())
            total_state_elements += moment.size
        assert f"{name}.exp_avg" not in states
    assert total_state_elements < total_parameter_elements / 4


def test_fp32_norm_vectors_update_at_default_learning_rate_with_bf16_matrices():
    mx.random.seed(998)
    model = Decoder(tiny())
    before = arrays(model)
    opt = training.FP32Adafactor(learning_rate=.001, relative_step=False,
                                 scale_parameter=False, beta_1=None)
    batch = training.batch_arrays([make_row("xyz", "A")], 64)
    update(model, opt, batch)
    norms = [(name, value) for name, value in tree_flatten(model.parameters()) if value.ndim == 1]
    assert len(norms) == 3
    for name, value in norms:
        assert value.dtype == mx.float32
        assert np.any(np.asarray(value) != before[name]), name


def test_checkpoint_reload_preserves_optimizer_and_next_update_exactly(tmp_path):
    mx.random.seed(1001)
    config = tiny()
    model, opt = Decoder(config), optimizer()
    batch = training.batch_arrays([make_row("a", "A"), make_row("b", "S")], 64)
    update(model, opt, batch)
    training.save_checkpoint(tmp_path, model, opt, 1, {"scope": "tiny test"})
    checkpoint = training.resolve_checkpoint(tmp_path)
    info = json.loads((checkpoint / "checkpoint.json").read_text())
    assert info["model_sha256"] == training.digest(checkpoint / "model.safetensors")
    assert info["optimizer_sha256"] == training.digest(checkpoint / "optimizer.safetensors")
    assert info["total_optimizer_updates"] == 1
    restored, restored_opt = Decoder(config), optimizer()
    restored.load_weights(str(checkpoint / "model.safetensors"))
    restored_opt.state = tree_unflatten(list(mx.load(str(checkpoint / "optimizer.safetensors")).items()))
    expected_loss = update(model, opt, batch)
    actual_loss = update(restored, restored_opt, batch)
    assert actual_loss == expected_loss
    for name, value in arrays(model).items():
        np.testing.assert_array_equal(value, arrays(restored)[name], err_msg=name)
    expected_state, actual_state = dict(tree_flatten(opt.state)), dict(tree_flatten(restored_opt.state))
    assert expected_state.keys() == actual_state.keys()
    for name in expected_state:
        np.testing.assert_array_equal(np.asarray(expected_state[name]), np.asarray(actual_state[name]), err_msg=name)


@pytest.mark.parametrize("invalid", ["weights", "optimizer"])
def test_checkpoint_rejects_nonfinite_state_before_writing(tmp_path, invalid):
    model, opt = Decoder(tiny()), optimizer()
    opt.init(model.parameters())
    if invalid == "weights":
        model.embedding.weight = mx.full(model.embedding.weight.shape, float("inf"), dtype=mx.bfloat16)
    else:
        opt.state["learning_rate"] = mx.array(float("nan"))
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        training.save_checkpoint(tmp_path, model, opt, 1, {})
    assert not (tmp_path / "latest.json").exists()
    assert not (tmp_path / "checkpoints").exists()


def tiny_training_setup(monkeypatch):
    # No scaling preset is instantiated; no real curriculum is generated.
    monkeypatch.setattr(training.ModelConfig, "preset", lambda name: tiny())
    # Unique bytes in each input and batch size one avoid Metal's order-dependent
    # repeated-index embedding scatter reductions confounding this resume test.
    monkeypatch.setattr(training, "encode_corpus", lambda count, *args: [
        make_row(f"xyz{chr(97 + i)}", "A" if i % 2 else "S", str(i)) for i in range(count)])
    source_name = "winward_scale/data.py"
    source_digest = training.digest(training.ROOT / source_name)
    monkeypatch.setattr(training, "source_hashes", lambda: {source_name: source_digest})
    monkeypatch.setattr(training.signal, "signal", lambda *args: None)
    monkeypatch.setattr(training.mx, "set_memory_limit", lambda value: None)
    monkeypatch.setattr(training.mx, "set_cache_limit", lambda value: None)


def tiny_args(path):
    return training.parser().parse_args([
        "--output", str(path), "--steps", "3", "--batch-size", "1",
        "--train-cases", "4", "--valid-cases", "2", "--max-seq-len", "64",
        "--eval-every", "1", "--save-every", "1", "--log-every", "1",
        "--learning-rate", ".02", "--seed", "9876",
    ])


def test_interrupted_run_resumes_same_data_sequence_and_final_weights(tmp_path, monkeypatch):
    tiny_training_setup(monkeypatch)
    continuous = tiny_args(tmp_path / "continuous")
    interrupted = tiny_args(tmp_path / "interrupted")
    training.train(continuous)
    protocol = json.loads((tmp_path / "continuous" / "protocol.json").read_text())
    for name, expected in protocol["source_sha256"].items():
        assert training.digest(tmp_path / "continuous" / "source" / name) == expected
    original_save = training.save_checkpoint

    class SimulatedInterruption(Exception):
        pass

    def interrupt_after_saved_first_step(run, model, opt, step, report):
        metadata = original_save(run, model, opt, step, report)
        if step == 1:
            raise SimulatedInterruption()
        return metadata

    monkeypatch.setattr(training, "save_checkpoint", interrupt_after_saved_first_step)
    with pytest.raises(SimulatedInterruption):
        training.train(interrupted)
    monkeypatch.setattr(training, "save_checkpoint", original_save)
    interrupted.resume = True
    training.train(interrupted)
    a, b = training.resolve_checkpoint(continuous.output), training.resolve_checkpoint(interrupted.output)
    assert training.digest(a / "model.safetensors") == training.digest(b / "model.safetensors")
    # Safetensors serialization can differ in metadata/header ordering despite
    # identical array values. Compare actual optimizer state, not file bytes.
    state_a = mx.load(str(a / "optimizer.safetensors"))
    state_b = mx.load(str(b / "optimizer.safetensors"))
    assert state_a.keys() == state_b.keys()
    for name in state_a:
        np.testing.assert_array_equal(np.asarray(state_a[name]), np.asarray(state_b[name]), err_msg=name)
    assert json.loads((b / "checkpoint.json").read_text())["total_optimizer_updates"] == 3


def test_resume_refuses_changed_source_before_allocating_model(tmp_path, monkeypatch):
    tiny_training_setup(monkeypatch)
    args = tiny_args(tmp_path / "run")
    training.train(args)
    args.resume = True
    monkeypatch.setattr(training, "source_hashes", lambda: {"test-fixture": "changed"})
    monkeypatch.setattr(training, "Decoder", lambda _: pytest.fail("must reject before allocating model"))
    with pytest.raises(ValueError, match="source changed"):
        training.train(args)


def test_resume_refuses_changed_declared_arguments_before_allocating_model(tmp_path, monkeypatch):
    tiny_training_setup(monkeypatch)
    args = tiny_args(tmp_path / "run")
    training.train(args)
    args.resume = True
    args.seed += 1
    monkeypatch.setattr(training, "Decoder", lambda _: pytest.fail("must reject before allocating model"))
    with pytest.raises(ValueError, match="arguments"):
        training.train(args)


def test_encode_corpus_refuses_silent_truncation():
    with pytest.raises(ValueError, match="nothing was truncated"):
        training.encode_corpus(1, "train", 42, "tiny", 1)
