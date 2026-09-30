"""Check schema compatibility and promotion semantics without a trained model."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from agent_training.features import encode
from agent_training.model import GoalPolicy, PolicyConfig
from agent_training.model_v3 import (BELIEF_DIM, BELIEF_DOMAIN_INDEX, FEATURE_DIM,
                                     GRAPH_DOMAIN_INDEX, GraphView, UnifiedPolicy, V3Policy,
                                     encode_graph, expand_belief, expand_graph, initialize_from_v1)
from agent_training.simulator import make_scenario
from agent_training.train_v3 import add_correct_teacher_distillation, read_source_rows, retention_selection


def tiny_models():
    source = GoalPolicy(PolicyConfig(width=16, layers=1, heads=4))
    destination = UnifiedPolicy(PolicyConfig(input_dim=FEATURE_DIM, width=16, layers=1, heads=4))
    return source, destination


def test_transfer_preserves_graph_logits_and_initializes_no_hidden_predictor():
    source, destination = tiny_models()
    initialize_from_v1(destination, source)
    x, valid, eligible, _ = encode(make_scenario(3107, "train"))
    old = source(mx.array(x[None]), mx.array(valid[None]), mx.array(eligible[None]))
    new = destination(mx.array(expand_graph(x)[None]), mx.array(valid[None]), mx.array(eligible[None]))
    via_view = GraphView(destination)(mx.array(x[None]), mx.array(valid[None]), mx.array(eligible[None]))
    mx.eval(old, new, via_view)
    np.testing.assert_allclose(np.asarray(new), np.asarray(old), atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(np.asarray(new), np.asarray(via_view))
    assert np.all(np.asarray(destination.input_projection.weight)[:, 103:] == 0)
    assert destination.count_parameters() - source.count_parameters() == (FEATURE_DIM - 103) * 16


def test_domains_are_explicit_disjoint_and_graph_semantics_unchanged():
    graph = np.arange(2 * 12 * 103, dtype=np.float32).reshape(2, 12, 103)
    belief = np.ones((2, 12, BELIEF_DIM), dtype=np.float32)
    gx, bx = expand_graph(graph), expand_belief(belief)
    np.testing.assert_array_equal(gx[..., :103], graph)
    assert np.all(gx[..., 103:103 + BELIEF_DIM] == 0)
    assert np.all(bx[..., :103] == 0)
    np.testing.assert_array_equal(bx[..., 103:103 + BELIEF_DIM], belief)
    assert np.all(gx[..., GRAPH_DOMAIN_INDEX] == 1) and np.all(gx[..., BELIEF_DOMAIN_INDEX] == 0)
    assert np.all(bx[..., GRAPH_DOMAIN_INDEX] == 0) and np.all(bx[..., BELIEF_DOMAIN_INDEX] == 1)
    with pytest.raises(ValueError):
        expand_graph(np.zeros((12, 102)))
    with pytest.raises(ValueError):
        expand_belief(np.zeros((12, BELIEF_DIM - 1)))


def test_gate_does_not_hide_one_regression_behind_other_improvements():
    baseline = dict(legacy_completion=0.95, goal_change_accuracy=0.9, actual_tool_completion=1.0)
    regressed = dict(legacy_completion=0.949, goal_change_accuracy=1.0, actual_tool_completion=1.0,
                     belief_expected_completion=1.0, belief_accuracy=1.0)
    selection, gates = retention_selection(regressed, baseline)
    assert selection[0] == 0 and not gates["legacy_completion"]["passed"]
    retained = {**regressed, "legacy_completion": 0.95, "belief_expected_completion": 0.5}
    accepted, _ = retention_selection(retained, baseline)
    assert accepted[0] == 1 and accepted > selection


def test_training_reader_rejects_final_audit_files_before_opening(tmp_path):
    with pytest.raises(ValueError, match="permitted"):
        read_source_rows(tmp_path / "test.jsonl")
    with pytest.raises(ValueError, match="permitted"):
        read_source_rows(tmp_path / "evaluation.json")


def test_predictor_rejects_silently_truncated_batch_arguments():
    _, model = tiny_models()
    policy = V3Policy(model=model)
    with pytest.raises(ValueError, match="scenario count"):
        policy.predict_graph([make_scenario(41, "train")], depths=[])
    with pytest.raises(ValueError, match="belief count"):
        policy.predict_beliefs([], depths=[5])
    assert policy.predict_graph([]) == []


def test_graph_eligibility_is_preserved_in_the_expanded_schema():
    scenario = make_scenario(83, "train")
    old = encode(scenario)
    new = encode_graph(scenario)
    np.testing.assert_array_equal(old[1], new[1])
    np.testing.assert_array_equal(old[2], new[2])
    assert old[3] == new[3]


def test_distillation_does_not_require_old_incorrect_choices():
    class WrongTeacher:
        def __call__(self, x, valid, eligible):
            return mx.array([[0.0, 10.0], [10.0, 0.0]])

    original = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    dataset = (np.zeros((2, 2, 103), dtype=np.float32), np.ones((2, 2), dtype=bool),
               np.ones((2, 2), dtype=bool), original)
    distilled, metadata = add_correct_teacher_distillation(WrongTeacher(), dataset)
    np.testing.assert_array_equal(distilled[3][0], original[0])
    assert 0 < distilled[3][1, 1] < 0.15
    np.testing.assert_allclose(distilled[3].sum(axis=1), 1.0)
    assert metadata["teacher_correct_rows"] == 1
