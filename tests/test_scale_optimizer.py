"""Exact sequential-update equivalence on tiny CPU models, not memory forecasts."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from agent_training.model import PolicyConfig
from agent_training.model_v3 import UnifiedPolicy
from winward_scale.structured import StructuredPolicy, grow_structured
from winward_scale.structured_train import decision_loss
from winward_scale.train import FP32Adafactor
from winward_scale.optimizer import SequentialAdafactor


@pytest.fixture(autouse=True)
def cpu_only():
    # Keep this suite independent of a concurrent large local training process.
    with mx.stream(mx.cpu):
        yield


def models():
    mx.random.seed(517)
    original = UnifiedPolicy(PolicyConfig(input_dim=505, width=8, layers=1, heads=2))
    a = grow_structured(original, 2)
    b = StructuredPolicy(a.config)
    b.load_weights(list(tree_flatten(a.parameters())))
    return a, b


def batch():
    rng = np.random.default_rng(10)
    x = mx.array(rng.normal(size=(2, 4, 505)).astype(np.float32))
    valid = mx.ones((2, 4), dtype=mx.bool_)
    target = mx.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
    return x, valid, valid, target


def grads(model):
    value, gradient = nn.value_and_grad(model, decision_loss)(model, *batch())
    # Deliberately leave these lazy: the sequential optimizer must evaluate the
    # entire backward pass before it mutates any checkpointed module parameters.
    return value, gradient


def assert_same_tree(a, b):
    a, b = dict(tree_flatten(a)), dict(tree_flatten(b))
    assert a.keys() == b.keys()
    for name in a:
        assert a[name].dtype == b[name].dtype
        np.testing.assert_array_equal(np.asarray(a[name].astype(mx.float32)),
                                       np.asarray(b[name].astype(mx.float32)), err_msg=name)


@pytest.mark.parametrize("scheduled", [False, True])
def test_exact_updates_schedule_and_state_match_original(scheduled):
    a, b = models()
    rate = (lambda step: .02 / (step + 1)) if scheduled else .02
    options = dict(learning_rate=rate, relative_step=False, scale_parameter=False, beta_1=None)
    ordinary, sequential = FP32Adafactor(**options), SequentialAdafactor(**options)
    for step in range(1, 4):
        la, ga = grads(a)
        lb, gb = grads(b)
        ordinary.update(a, ga)
        mx.eval(a.parameters(), ordinary.state, la)
        sequential.update(b, gb)
        mx.eval(b.parameters(), sequential.state, lb)
        assert float(la.item()) == float(lb.item())
        assert int(sequential.step.item()) == int(ordinary.step.item()) == step
        assert_same_tree(a.parameters(), b.parameters())
        assert_same_tree(ordinary.state, sequential.state)
        assert all(value is None for _, value in tree_flatten(gb))


def test_checkpoint_restoration_keeps_next_update_exact(tmp_path):
    a, _ = models()
    opt = SequentialAdafactor(learning_rate=.01, relative_step=False, scale_parameter=False, beta_1=None)
    _, gradient = grads(a)
    opt.update(a, gradient)
    a.save_weights(str(tmp_path / "model.safetensors"))
    mx.save_safetensors(str(tmp_path / "state.safetensors"), dict(tree_flatten(opt.state)))
    b = StructuredPolicy(a.config)
    b.load_weights(str(tmp_path / "model.safetensors"))
    restored = SequentialAdafactor(learning_rate=.01, relative_step=False, scale_parameter=False, beta_1=None)
    restored.state = tree_unflatten(list(mx.load(str(tmp_path / "state.safetensors")).items()))
    _, ga = grads(a)
    _, gb = grads(b)
    opt.update(a, ga)
    restored.update(b, gb)
    assert_same_tree(a.parameters(), b.parameters())
    assert_same_tree(opt.state, restored.state)
    assert int(restored.step.item()) == 2


def test_each_updated_leaf_is_installed_and_previous_gradient_released(monkeypatch):
    model, _ = models()
    opt = SequentialAdafactor(learning_rate=.01, relative_step=False, scale_parameter=False, beta_1=None)
    _, gradient = grads(model)
    count = len(tree_flatten(gradient))
    original_apply = opt.apply_single
    observed = []
    def inspect(g, p, state):
        released = sum(value is None for _, value in tree_flatten(gradient))
        observed.append(released)
        updated = original_apply(g, p, state)
        assert updated.dtype == p.dtype
        return updated
    monkeypatch.setattr(opt, "apply_single", inspect)
    opt.update(model, gradient)
    assert observed == list(range(count))
    assert sum(value is None for _, value in tree_flatten(gradient)) == count


def test_bad_gradient_shape_rejected_before_any_parameter_or_step_change():
    a, b = models()
    opt = SequentialAdafactor(learning_rate=.01, relative_step=False, scale_parameter=False, beta_1=None)
    _, gradient = grads(a)
    gradient["output"]["weight"] = mx.zeros((1, 2))
    with pytest.raises(ValueError, match="Gradient shape"):
        opt.update(a, gradient)
    assert int(opt.step.item()) == 0
    assert_same_tree(a.parameters(), b.parameters())
