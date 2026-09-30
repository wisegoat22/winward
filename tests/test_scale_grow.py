"""Bounded own-weight expansion checks; never instantiate a billion parameters."""
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from agent_training.belief_data_v4 import make_problem
from agent_training.model import GoalPolicy, PolicyConfig
from agent_training.model_v4 import UnifiedPolicy, V4Policy, encode_graph, encode_uncertainty
from agent_training.simulator import make_scenario
from winward_scale.grow import grow_policy, parameter_count


@pytest.fixture(autouse=True)
def full_precision_proof_stream():
    # M5 defaults to reduced-precision float32 GPU matmul. Do not change a
    # process-wide setting after MLX may already have cached it. CPU gives a
    # strict reference in an ordinary suite; the documented launch flag also
    # lets this same suite prove the mapping on full-float32 GPU kernels.
    device = mx.gpu if os.environ.get("MLX_ENABLE_TF32") == "0" else mx.cpu
    with mx.stream(device):
        yield


def tiny():
    return UnifiedPolicy(PolicyConfig(input_dim=505, width=8, layers=2, heads=2, expansion=3))


def dense_inputs():
    rng = np.random.default_rng(9811)
    return (mx.array(rng.normal(size=(3, 6, 505)).astype(np.float32)),
            mx.ones((3, 6), dtype=mx.bool_), mx.ones((3, 6), dtype=mx.bool_))


def test_parameter_count_without_allocating_large_model():
    config = PolicyConfig(input_dim=505)
    assert parameter_count(config) == 4_862_721
    assert parameter_count(config, 2) == 19_162_625
    assert parameter_count(config, 29) == 3_972_515_585
    model = tiny()
    assert parameter_count(model.config) == model.count_parameters()
    grown = grow_policy(model, 2)
    assert parameter_count(model.config, 2) == grown.count_parameters()
    assert grown.config.width // grown.config.heads == model.config.width // model.config.heads


@pytest.mark.parametrize("factor", [True, 0, -1, 1.5])
def test_bad_factor_rejected_before_construction(factor):
    with pytest.raises(ValueError, match="factor"):
        parameter_count(PolicyConfig(), factor)
    with pytest.raises(ValueError, match="factor"):
        grow_policy(tiny(), factor)


@pytest.mark.parametrize("kwargs", [{"dtype": "int8"}, {"noise": float("nan")},
                                   {"noise": -1}, {"noise": True}])
def test_invalid_options_rejected(kwargs):
    with pytest.raises(ValueError):
        grow_policy(tiny(), 2, **kwargs)


@pytest.mark.parametrize("noise", [0, .01])
def test_tiny_growth_preserves_logits_masks_and_source(noise):
    source = tiny()
    source.eval()
    source.norm.eps = 1e-4
    before = {k: np.asarray(v).copy() for k, v in tree_flatten(source.parameters())}
    grown = grow_policy(source, 2, noise=noise)
    x, valid, _ = dense_inputs()
    eligible = mx.array([[True, False, True, False, True, False]] * 3)
    valid = mx.array([[True, True, True, True, True, False]] * 3)
    a, b = np.asarray(source(x, valid, eligible)), np.asarray(grown(x, valid, eligible))
    np.testing.assert_allclose(a, b, atol=2e-5, rtol=2e-5)
    assert not grown.training and grown.norm.eps == source.norm.eps
    assert dict(tree_flatten(grown.parameters())).keys() == dict(tree_flatten(grown.trainable_parameters())).keys()
    for name, value in tree_flatten(source.parameters()):
        np.testing.assert_array_equal(before[name], np.asarray(value))
    assert np.all(b[:, [1, 3, 5]] == -1e9)


def test_every_tensor_has_a_forward_gradient_path_and_replicas_can_diverge():
    source = tiny()
    grown = grow_policy(source, 2, noise=.01)
    inputs = dense_inputs()
    # A raw-logit objective verifies every tensor's forward path. Ordinary CE
    # is invariant to common logit shifts: output.bias/final-norm bias cannot be
    # certified by a nonzero CE gradient, even in the original unexpanded model.
    def probe(model):
        return mx.mean(mx.square(model(*inputs) - mx.arange(6)[None, :]))
    value, gradients = nn.value_and_grad(grown, probe)(grown)
    assert np.isfinite(float(value.item()))
    params = dict(tree_flatten(grown.parameters()))
    leaves = dict(tree_flatten(gradients))
    assert leaves.keys() == params.keys()
    for name, gradient in leaves.items():
        array = np.asarray(gradient)
        assert np.isfinite(array).all() and np.any(array != 0), name
    # Decision-loss gradients reach all feature/attention/FF weight matrices.
    def decision_loss(model):
        return nn.losses.cross_entropy(model(*inputs), mx.array([0, 1, 2]), reduction="mean")
    _, decision_gradients = nn.value_and_grad(grown, decision_loss)(grown)
    dg = dict(tree_flatten(decision_gradients))
    for name, gradient in dg.items():
        if name.endswith(".weight") and "norm" not in name:
            assert np.any(np.asarray(gradient) != 0), name
    before = np.asarray(grown.input_projection.weight).copy()
    np.testing.assert_array_equal(before[:8], before[8:])
    grown.update(tree_map(lambda parameter, gradient: parameter - .01 * gradient,
                          grown.parameters(), decision_gradients))
    after = np.asarray(grown.input_projection.weight)
    assert np.max(np.abs(after[:8] - after[8:])) > 1e-8
    assert source.config.width == 8


def test_noise_is_reproducible_and_zero_sum_over_replicated_inputs():
    source = tiny()
    plain = dict(tree_flatten(grow_policy(source, 2).parameters()))
    one = dict(tree_flatten(grow_policy(source, 2, noise=.01).parameters()))
    two = dict(tree_flatten(grow_policy(source, 2, noise=.01).parameters()))
    for name in one:
        np.testing.assert_array_equal(np.asarray(one[name]), np.asarray(two[name]))
        if one[name].ndim == 2 and name != "input_projection.weight":
            delta = np.asarray(one[name] - plain[name])
            assert np.any(delta != 0), name
            np.testing.assert_allclose(delta.reshape(delta.shape[0], 2, -1).sum(axis=1),
                                       0, atol=4e-8, rtol=0, err_msg=name)


def test_float32_identity_and_dtype_conversion():
    source = GoalPolicy(PolicyConfig(width=8, layers=1, heads=2))
    grown = grow_policy(source, 1)
    for (a_name, a), (b_name, b) in zip(tree_flatten(source.parameters()), tree_flatten(grown.parameters())):
        assert a_name == b_name
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    small = grow_policy(source, 2, dtype="bfloat16")
    assert all(v.dtype == mx.bfloat16 for _, v in tree_flatten(small.parameters()))


@pytest.mark.parametrize("noise", [0, .01])
def test_own_v4_checkpoint_retains_development_graph_and_belief_decisions(noise):
    run = Path(__file__).resolve().parents[1] / "runs" / "goalpolicy-v4"
    checkpoint = run / "model.safetensors"
    if not checkpoint.exists():
        pytest.skip("local own-v4 checkpoint is intentionally not tracked in git")
    before_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    source = V4Policy(run).model
    grown = grow_policy(source, 2, noise=noise)
    # Development/validation inputs only. No final-audit files/seeds/labels.
    encoded = [encode_graph(make_scenario(2_008_100 + i, "validation")) for i in range(24)]
    encoded += [encode_uncertainty(make_problem(420_090_000 + i, "validation")) for i in range(24)]
    inputs = [mx.array(np.stack([row[i] for row in encoded])) for i in range(3)]
    a, b = np.asarray(source(*inputs)), np.asarray(grown(*inputs))
    eligible = np.asarray(inputs[2])
    np.testing.assert_allclose(a[eligible], b[eligible], atol=2e-4, rtol=2e-5)
    np.testing.assert_array_equal(a.argmax(axis=1), b.argmax(axis=1))
    assert grown.count_parameters() == 19_162_625
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before_sha
