"""Tiny-model checks; never allocate a scaling preset in tests."""
from dataclasses import replace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from winward_scale.model import Decoder, ModelConfig


def tiny(**overrides):
    return replace(ModelConfig(hidden_size=32, intermediate_size=64, num_layers=2,
                   num_attention_heads=4, num_key_value_heads=2, vocab_size=260,
                   dtype="float32"), **overrides)


def loss(model, tokens):
    return nn.losses.cross_entropy(model(tokens[:, :-1]), tokens[:, 1:], reduction="mean")


def test_parameter_count_is_exact_and_embedding_is_tied():
    model = Decoder(tiny())
    params = dict(tree_flatten(model.parameters()))
    assert sum(p.size for p in params.values()) == model.config.parameter_count()
    assert len([k for k in params if "embedding" in k]) == 1
    assert not any("lm_head" in k or "output" in k for k in params)
    tokens = mx.array([[257, 65, 66, 258]], dtype=mx.int32)
    assert model(tokens).shape == (1, 4, 260)
    assert ModelConfig.preset("4b").parameter_count() == 4_027_579_392


def test_causal_attention_cannot_see_changed_future_tokens():
    model = Decoder(tiny())
    model.eval()
    a = model(mx.array([[257, 65, 66, 67, 258]], dtype=mx.int32))
    b = model(mx.array([[257, 65, 100, 101, 102]], dtype=mx.int32))
    np.testing.assert_allclose(np.asarray(a[:, :2]), np.asarray(b[:, :2]), atol=1e-6, rtol=1e-6)
    assert not np.allclose(np.asarray(a[:, 2:]), np.asarray(b[:, 2:]))


def test_checkpoint_gradients_match_for_every_trainable_parameter():
    mx.random.seed(451)
    ordinary = Decoder(tiny(gradient_checkpointing=False))
    checkpointed = Decoder(tiny(gradient_checkpointing=True))
    checkpointed.update(ordinary.parameters())
    tokens = mx.array([[257, 65, 66, 259, 67, 258], [257, 68, 69, 259, 70, 258]], dtype=mx.int32)
    plain_loss, plain_grad = nn.value_and_grad(ordinary, loss)(ordinary, tokens)
    cp_loss, cp_grad = nn.value_and_grad(checkpointed, loss)(checkpointed, tokens)
    assert float(cp_loss.item()) == pytest.approx(float(plain_loss.item()), abs=1e-6)
    plain, cp = dict(tree_flatten(plain_grad)), dict(tree_flatten(cp_grad))
    assert plain.keys() == cp.keys() == dict(tree_flatten(ordinary.parameters())).keys()
    for name in plain:
        a, b = np.asarray(plain[name]), np.asarray(cp[name])
        assert np.isfinite(b).all(), name
        assert np.any(b != 0), name
        np.testing.assert_allclose(a, b, rtol=2e-4, atol=1e-6, err_msg=name)


def test_all_parameter_tensors_change_and_loss_falls_after_real_updates():
    mx.random.seed(45)
    model = Decoder(tiny())
    tokens = mx.array([[257, 65, 66, 259, 67, 258], [257, 70, 71, 259, 72, 258]], dtype=mx.int32)
    before = {k: np.array(v) for k, v in tree_flatten(model.parameters())}
    initial = float(loss(model, tokens).item())
    for _ in range(5):
        value, grad = nn.value_and_grad(model, loss)(model, tokens)
        model.update(tree_map(lambda w, g: w - 0.02 * g, model.parameters(), grad))
        mx.eval(value, model.parameters())
    assert float(loss(model, tokens).item()) < initial
    for name, parameter in tree_flatten(model.parameters()):
        assert np.isfinite(np.asarray(parameter)).all(), name
        assert np.any(np.asarray(parameter) != before[name]), name


def test_bfloat16_initialized_directly_and_dtype_conversion_works():
    model = Decoder(tiny(dtype="bfloat16"))
    assert all(p.dtype == (mx.float32 if p.ndim == 1 else mx.bfloat16)
               for _, p in tree_flatten(model.parameters()))
    out = model(mx.array([[257, 65, 259, 66, 258]], dtype=mx.int32))
    assert out.dtype == mx.bfloat16
    assert bool(mx.all(mx.isfinite(out)).item())
    model.set_dtype(mx.float32)
    assert all(p.dtype == mx.float32 for _, p in tree_flatten(model.parameters()))


def test_save_load_roundtrip_preserves_predictions(tmp_path):
    model = Decoder(tiny())
    tokens = mx.array([[257, 65, 259, 66, 258]], dtype=mx.int32)
    expected = np.asarray(model(tokens))
    path = tmp_path / "model.safetensors"
    model.save_weights(str(path))
    restored = Decoder(tiny())
    restored.load_weights(str(path))
    np.testing.assert_array_equal(np.asarray(restored(tokens)), expected)


@pytest.mark.parametrize("overrides", [
    {"hidden_size": 31}, {"num_key_value_heads": 3}, {"num_layers": 0},
    {"num_layers": True}, {"dtype": "float16"}, {"norm_dtype": "float16"}, {"rms_norm_eps": 0.0},
    {"initializer_std": float("nan")}, {"gradient_checkpointing": 1},
])
def test_invalid_architectures_rejected_before_allocation(overrides):
    with pytest.raises(ValueError):
        tiny(**overrides)


def test_presets_have_monotonic_counts_without_allocating_them():
    counts = [ModelConfig.preset(name).parameter_count() for name in ModelConfig.PRESETS]
    assert counts == sorted(counts)
    assert 4_000_000_000 < counts[-1] < 4_100_000_000


@pytest.mark.parametrize("tokens", [mx.zeros((0, 3), dtype=mx.int32),
                                   mx.zeros((1, 0), dtype=mx.int32),
                                   mx.zeros((1, 3)), mx.zeros((3,), dtype=mx.int32)])
def test_invalid_input_shape_or_dtype_rejected(tokens):
    with pytest.raises(ValueError, match="tokens"):
        Decoder(tiny())(tokens)
