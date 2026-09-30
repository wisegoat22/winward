"""Reliability objective and freeze semantics without creating final-audit data."""
import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from agent_training import train_v4
from agent_training.belief_data_v4 import generate_rows
from agent_training.model import GoalPolicy, PolicyConfig
from agent_training.model_v4 import (FEATURE_DIM, SCHEMA_VERSION, UnifiedPolicy, V4Policy,
                                     encode_graph, initialize_from_v1)
from agent_training.simulator import make_scenario


def test_architecture_and_initial_graph_preferences_are_retained():
    source = GoalPolicy(PolicyConfig(width=16, layers=1, heads=4))
    model = UnifiedPolicy(PolicyConfig(input_dim=FEATURE_DIM, width=16, layers=1, heads=4))
    initialize_from_v1(model, source)
    from agent_training.features import encode
    scenario = make_scenario(41, "train")
    before = encode(scenario)
    after = encode_graph(scenario)
    logits_a = source(*[mx.array(x[None]) for x in before[:3]])
    logits_b = model(*[mx.array(x[None]) for x in after[:3]])
    np.testing.assert_allclose(np.asarray(logits_a), np.asarray(logits_b), atol=1e-5, rtol=1e-5)
    assert SCHEMA_VERSION != "winward-v3-public-observation-1"
    assert V4Policy(model=model).predict_graph([scenario]) == [after[3][int(np.asarray(logits_b)[0].argmax())]["id"]]


def test_success_loss_prioritizes_completion_over_wrong_cheap_action():
    regret = mx.array([[0.0, 1.0]])
    known = mx.array([[True, True]])
    correct = train_v4.success_regret(mx.array([[5., -5.]]), regret, known)
    wrong = train_v4.success_regret(mx.array([[-5., 5.]]), regret, known)
    assert float(correct.item()) < .001
    assert float(wrong.item()) > .999


def test_auxiliary_empty_masks_and_only_terminal_are_finite_zero():
    value = train_v4.success_regret(mx.array([[10., 9.], [1., 7.]]), mx.zeros((2, 2)),
                                   mx.array([[False, False], [False, True]]))
    assert float(value.item()) == 0
    derivative = mx.grad(lambda x: train_v4.success_regret(x, mx.zeros((2, 2)),
                         mx.array([[False, False], [False, True]])))(mx.array([[10., 9.], [1., 7.]]))
    assert np.isfinite(np.asarray(derivative)).all()


def test_unvalued_noop_cannot_reduce_conditional_auxiliary_loss():
    regrets, mask = mx.array([[0., 1., 0.]]), mx.array([[True, True, False]])
    low = train_v4.success_regret(mx.array([[0., 0., -20.]]), regrets, mask)
    high = train_v4.success_regret(mx.array([[0., 0., 20.]]), regrets, mask)
    assert float(low.item()) == pytest.approx(.5)
    assert float(high.item()) == pytest.approx(.5)
    class Scores:
        def __init__(self, third): self.third = third
        def __call__(self, x, valid, eligible): return mx.array([[0., 0., self.third]])
    args = (mx.zeros((1, 3, 505)), mx.ones((1, 3), dtype=mx.bool_), mx.ones((1, 3), dtype=mx.bool_),
            mx.array([[1., 0., 0.]]), regrets, mask)
    assert float(train_v4.reliability_loss(Scores(20.), *args).item()) > float(train_v4.reliability_loss(Scores(-20.), *args).item())


def test_teacher_values_are_never_forward_inputs():
    class Capture:
        def __call__(self, x, valid, eligible):
            assert x.shape == (1, 2, 505)
            assert valid.shape == eligible.shape == (1, 2)
            return mx.array([[0., 0.]])
    loss = train_v4.reliability_loss(Capture(), mx.zeros((1, 2, 505)),
            mx.array([[True, True]]), mx.array([[True, True]]), mx.array([[1., 0.]]),
            mx.array([[0., 1.]]), mx.array([[True, True]]))
    assert np.isfinite(float(loss.item()))


def test_exact_value_arrays_align_and_reject_ineligible_targets():
    rows, arrays = generate_rows(12, "train", 410000000)
    regrets, mask = train_v4.belief_value_arrays(rows, arrays[2])
    assert regrets.shape == mask.shape == (12, 12)
    assert mask.any(axis=1).all()
    with pytest.raises(ValueError, match="eligible"):
        train_v4.belief_value_arrays(rows, np.zeros_like(arrays[2]))


def test_goal_pair_regression_is_separate_strict_gate():
    baseline = dict(legacy_completion=.97, goal_change_accuracy=.99,
                    goal_change_pair_correctness=.9, actual_tool_completion=1.)
    metrics = dict(**baseline, belief_expected_completion=.95, belief_accuracy=.99)
    selected, gates = train_v4.retention_selection(metrics, baseline)
    assert selected[0] == 1
    metrics["goal_change_pair_correctness"] = .89
    regressed, gates = train_v4.retention_selection(metrics, baseline)
    assert regressed[0] == 0 and regressed < selected
    assert gates["goal_change_pair_correctness"]["delta"] == pytest.approx(-.01)


def sealed(tmp_path, monkeypatch):
    monkeypatch.setattr(train_v4, "source_hashes", lambda: {"source.py": "source-sha"})
    monkeypatch.setattr(train_v4, "checkpoint_hashes", lambda *args: {"v1": "old-sha", "v2.1": "tools-sha"})
    protocol = {"version": "winward-v4-unit-test"}
    (tmp_path / "audit-protocol.json").write_text(json.dumps({"protocol": protocol, "protocol_sha256": train_v4.canonical_digest(protocol)}))
    return train_v4.seal_training_plan(tmp_path)


def test_training_cannot_start_without_both_protocols(tmp_path):
    with pytest.raises(ValueError, match="Seal"):
        train_v4.validate_sealed_protocols(tmp_path)


def test_sealing_is_exclusive_and_budget_cannot_be_changed(tmp_path, monkeypatch):
    value = sealed(tmp_path, monkeypatch)
    assert value["plan"]["total_step_budget"] == 20000
    assert len(value["plan"]["candidates"]) == 2
    assert train_v4.validate_sealed_protocols(tmp_path)["sealed_training_protocol_sha256"]
    with pytest.raises(FileExistsError):
        train_v4.seal_training_plan(tmp_path)
    value["plan"] = {**value["plan"], "steps_per_candidate": 3}
    (tmp_path / "training-protocol.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="plan"):
        train_v4.validate_sealed_protocols(tmp_path)


@pytest.mark.parametrize("changed", ["source", "checkpoint", "audit"])
def test_sealed_inputs_cannot_change_before_training(tmp_path, monkeypatch, changed):
    sealed(tmp_path, monkeypatch)
    if changed == "source":
        monkeypatch.setattr(train_v4, "source_hashes", lambda: {"source.py": "changed"})
    elif changed == "checkpoint":
        monkeypatch.setattr(train_v4, "checkpoint_hashes", lambda *args: {"v1": "changed"})
    else:
        value = json.loads((tmp_path / "audit-protocol.json").read_text())
        value["protocol"]["version"] = "changed"
        (tmp_path / "audit-protocol.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="changed|hash"):
        train_v4.validate_sealed_protocols(tmp_path)


def test_completed_or_interrupted_trial_cannot_be_overwritten(tmp_path, monkeypatch):
    sealed(tmp_path, monkeypatch)
    (tmp_path / "replay60").mkdir()
    with pytest.raises(ValueError, match="immutable"):
        train_v4.validate_sealed_protocols(tmp_path)
