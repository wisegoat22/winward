"""Mixed-precision structured growth; allocate at most the local 19M proof."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from agent_training.model import PolicyConfig
from agent_training.model_v3 import UnifiedPolicy
from agent_training.model_v4 import V4Policy, encode_graph, encode_uncertainty
from agent_training.belief_data_v4 import make_problem
from agent_training.simulator import make_scenario
from winward_scale.grow import parameter_count
from winward_scale.structured import (StructuredConfig, StructuredPolicy, grow_structured,
                                      save_structured, load_structured)


def source():
    mx.random.seed(3199)
    return UnifiedPolicy(PolicyConfig(input_dim=505, width=16, heads=4, layers=2, expansion=3))


def inputs():
    x = mx.array(np.random.default_rng(7001).normal(size=(3, 6, 505)).astype(np.float32))
    return x, mx.ones((3, 6), dtype=mx.bool_), mx.ones((3, 6), dtype=mx.bool_)


def test_counts_configs_and_no_large_allocation():
    base = StructuredConfig()
    assert base.parameter_count() == 4_862_721
    config = base
    for factor, count in [(2, 19_162_625), (3, 170_734_081), (5, 4_251_056_641)]:
        config = replace(config, width=config.width * factor, heads=config.heads * factor)
        assert config.parameter_count() == count
        assert config.parameter_count() == parameter_count(config)
        assert StructuredConfig(**config.to_dict()) == config
    with pytest.raises(ValueError):
        StructuredConfig(norm_dtype="bfloat16")
    with pytest.raises(ValueError):
        StructuredConfig(width=15, heads=4)


def test_placeholder_cannot_be_used_as_a_trained_model():
    model = StructuredPolicy(StructuredConfig(width=16, heads=4, layers=1))
    with pytest.raises(RuntimeError, match="Load complete"):
        model(*inputs())
    with pytest.raises(ValueError, match="no loaded weights"):
        grow_structured(model, 2)


def test_full_float32_mapping_retains_source_on_cpu():
    with mx.stream(mx.cpu):
        original = source()
        grown = grow_structured(original, 2, dtype="float32", row_chunk_size=3)
        np.testing.assert_allclose(np.asarray(original(*inputs())), np.asarray(grown(*inputs())),
                                   atol=2e-5, rtol=2e-5)


def test_growth_keeps_bf16_activations_fp32_norms_and_masks():
    model = grow_structured(source(), 2, row_chunk_size=5)
    leaves = dict(tree_flatten(model.parameters()))
    assert leaves.keys() == dict(tree_flatten(model.trainable_parameters())).keys()
    for name, parameter in leaves.items():
        is_norm = name.startswith("norm.") or ".norm1." in name or ".norm2." in name
        assert parameter.dtype == (mx.float32 if is_norm else mx.bfloat16), name
    x, valid, eligible = inputs()
    h = model.input_projection(x)
    assert h.dtype == model.norm(h).dtype == mx.bfloat16
    mask = mx.zeros((3, 1, 1, 6), dtype=mx.bfloat16)
    for block in model.blocks:
        assert block.norm1(h).dtype == block.up(block.norm2(h)).dtype == mx.bfloat16
        h = block(h, mask)
        assert h.dtype == mx.bfloat16
    eligible = mx.array([[True, False, True, False, True, False]] * 3)
    logits = np.asarray(model(x, valid, eligible))
    assert np.isfinite(logits).all()
    assert np.all(logits[:, [1, 3, 5]] == -1e9)


def test_checkpointing_gradients_and_every_matrix_can_update():
    ordinary = grow_structured(source(), 2, gradient_checkpointing=False)
    config = replace(ordinary.config, gradient_checkpointing=True)
    checkpointed = StructuredPolicy(config)
    checkpointed.load_weights(list(tree_flatten(ordinary.parameters())))
    x = inputs()
    def loss(model):
        return nn.losses.cross_entropy(model(*x), mx.array([0, 1, 2]), reduction="mean")
    a, ga = nn.value_and_grad(ordinary, loss)(ordinary)
    b, gb = nn.value_and_grad(checkpointed, loss)(checkpointed)
    assert float(a.item()) == pytest.approx(float(b.item()), abs=1e-6)
    first, second = dict(tree_flatten(ga)), dict(tree_flatten(gb))
    params = dict(tree_flatten(checkpointed.parameters()))
    before = {name: np.asarray(value.astype(mx.float32)).copy() for name, value in params.items()}
    assert first.keys() == second.keys() == params.keys()
    for name in first:
        aa, bb = np.asarray(first[name].astype(mx.float32)), np.asarray(second[name].astype(mx.float32))
        assert np.isfinite(bb).all(), name
        np.testing.assert_allclose(aa, bb, atol=1e-6, rtol=1e-5, err_msg=name)
        if params[name].ndim == 2:
            assert np.any(bb != 0), name
    # This checks a real stored BF16 update, not only a nonzero FP32 gradient.
    checkpointed.update(tree_map(lambda p, g: (p.astype(mx.float32) - .2 * g.astype(mx.float32)).astype(p.dtype),
                                checkpointed.parameters(), gb))
    for name, value in tree_flatten(checkpointed.parameters()):
        if value.ndim == 2:
            assert np.any(np.asarray(value.astype(mx.float32)) != before[name]), name


def test_growth_from_own_mixed_policy_and_save_reload(tmp_path):
    first = grow_structured(source(), 2, noise=.01)
    before = {name: np.asarray(value.astype(mx.float32)).copy() for name, value in tree_flatten(first.parameters())}
    second = grow_structured(first, 2, noise=.01, row_chunk_size=7)
    assert second.count_parameters() == parameter_count(first.config, 2)
    assert np.isfinite(np.asarray(second(*inputs()))).all()
    for name, value in tree_flatten(first.parameters()):
        np.testing.assert_array_equal(before[name], np.asarray(value.astype(mx.float32)))
    folder = save_structured(second, tmp_path / "checkpoint")
    restored = load_structured(folder)
    assert restored.config == second.config
    np.testing.assert_array_equal(np.asarray(second(*inputs())), np.asarray(restored(*inputs())))
    assert json.loads((folder / "config.json").read_text())["dtype"] == "bfloat16"
    with pytest.raises(FileExistsError):
        save_structured(second, folder)


def test_matrix_mapping_evaluates_only_bounded_fp32_row_expansions(monkeypatch):
    original_tile = mx.tile
    shapes = []
    def traced_tile(value, repeats):
        if value.ndim == 2:
            shapes.append(value.shape)
        return original_tile(value, repeats)
    monkeypatch.setattr(mx, "tile", traced_tile)
    grown = grow_structured(source(), 2, row_chunk_size=3)
    assert grown.count_parameters() > 0
    assert shapes and max(shape[0] for shape in shapes) <= 3


@pytest.mark.parametrize("kwargs", [{"factor": True}, {"factor": 0}, {"noise": -1},
                                    {"noise": float("inf")}, {"row_chunk_size": 0},
                                    {"seed": -1}, {"dtype": "float16"}])
def test_bad_growth_options_rejected(kwargs):
    with pytest.raises(ValueError):
        grow_structured(source(), **{"factor": 2, **kwargs})


def test_bf16_own_v4_development_retention_and_source_hash():
    directory = Path(__file__).resolve().parents[1] / "runs" / "goalpolicy-v4"
    checkpoint = directory / "model.safetensors"
    if not checkpoint.exists():
        pytest.skip("Own local v4 weights are intentionally not tracked in git")
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    original = V4Policy(directory).model
    grown = grow_structured(original, 2, noise=.01)
    rows = [encode_graph(make_scenario(2_008_100 + i, "validation")) for i in range(24)]
    rows += [encode_uncertainty(make_problem(420_090_000 + i, "validation")) for i in range(24)]
    x = [mx.array(np.stack([row[j] for row in rows])) for j in range(3)]
    before, after = np.asarray(original(*x)), np.asarray(grown(*x))
    assert np.isfinite(after).all()
    # Decision retention on this bounded development set, not exact BF16 logits
    # or a claim of broad validation/audit performance.
    np.testing.assert_array_equal(before.argmax(-1), after.argmax(-1))
    assert grown.count_parameters() == 19_162_625
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == sha
