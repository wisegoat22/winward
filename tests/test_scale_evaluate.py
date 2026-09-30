"""Manual/validation-only fixtures; never generate final audit instances."""
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from agent_training.simulator import Action, Scenario
from winward_scale.data import (ByteTokenizer, DATA_VERSION, PRIMITIVES_VERSION,
                               PRIMITIVE_PROMPT_HEADER, TextExample,
                               parse_prompt, serialize_prompt)
from winward_scale.model import ModelConfig
from winward_scale import evaluate as evaluation


def example(scenario, depth=5, name="unit"):
    # Deliberately corrupted labels prove evaluation reconstructs public mechanics.
    return TextExample(serialize_prompt(scenario, depth), "Z", "manual", name,
                       {"accepted_labels": ["Z"], "success": False})


def chain():
    return Scenario("unit", "manual", "validation", ("ready", "done"), 0, 2,
                    (Action("A", "prepare", sets=1, forbids=1, tokens=2, latency_ms=100),
                     Action("B", "verify", requires=1, sets=2, forbids=2, tokens=3, latency_ms=200)))


def prediction(label, ms=7):
    return evaluation.Prediction(ord(label), ms)


def test_model_predicts_unrestricted_final_prompt_token_without_label_mask():
    class Fake:
        def eval(self): pass
        def __call__(self, tokens):
            assert int(tokens[0, -1].item()) == ByteTokenizer.sep_id
            values = np.zeros((*tokens.shape, 260), dtype=np.float32)
            values[0, 0, ord("A")] = 100
            values[0, -1, ByteTokenizer.pad_id] = 20
            values[0, -1, ord("B")] = 10
            return mx.array(values)
    policy = evaluation.ModelPolicy(Fake(), 64)
    result = policy("abc")
    assert result.token_id == ByteTokenizer.pad_id
    assert result.label is None
    assert result.prompt_tokens == 5
    with pytest.raises(ValueError, match="no truncation"):
        evaluation.ModelPolicy(Fake(), 2)("abc")


def test_model_nonfinite_logits_fail_without_guessing_an_action():
    class Broken:
        def eval(self): pass
        def __call__(self, tokens):
            return mx.full((*tokens.shape, 260), float("nan"))
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        evaluation.ModelPolicy(Broken(), 64)("abc")


def test_sequential_rollout_uses_actual_effects_remaining_depth_and_own_choices():
    observed = []
    def policy(prompt):
        scenario, depth = parse_prompt(prompt)
        observed.append((scenario.state, depth))
        return prediction("A" if scenario.state == 0 else "B", 11)
    result = evaluation.rollout(policy, chain(), 2)
    assert observed == [(0, 2), (1, 1)]
    assert result["completed"] and result["actions_taken"] == 2
    assert result["estimated_action_cost"] == 8
    assert result["estimated_action_tokens"] == 5
    assert result["estimated_action_latency_ms"] == 300
    assert result["measured_decision_ms"] == 22
    assert result["final_state"] == 3


def test_primitive_prompt_style_is_preserved_for_every_policy_rollout(monkeypatch):
    observed = []
    def policy(prompt):
        assert prompt.startswith(PRIMITIVE_PROMPT_HEADER)
        scenario, depth = parse_prompt(prompt)
        observed.append((scenario.state, depth))
        return prediction("A" if scenario.state == 0 else "B")
    for name in ("greedy_policy", "exact_policy"):
        original = getattr(evaluation, name)
        def checked(prompt, original=original):
            assert prompt.startswith(PRIMITIVE_PROMPT_HEADER)
            return original(prompt)
        monkeypatch.setattr(evaluation, name, checked)
    # A manually composed two-step primitive prompt catches header switching;
    # the current generated primitive curriculum itself has a one-step horizon.
    prompt = serialize_prompt(chain(), 2, prompt_style="primitives")
    result = evaluation.evaluate_cases(policy, [TextExample(prompt, "Z", "manual", "manual", {})], rollout_count=1)
    assert observed == [(0, 2), (1, 1)]
    assert all(row["rollouts"]["solvable_completion_rate"] == 1 for row in result["policies"].values())


@pytest.mark.parametrize("label,termination", [("S", "premature_stop"), ("Z", "invalid_label"),
                                                ("B", "ineligible_action"), ("?", "deferred")])
def test_rollout_does_not_correct_invalid_or_early_terminal_choices(label, termination):
    result = evaluation.rollout(lambda _: prediction(label), chain(), 5)
    assert not result["completed"]
    assert result["termination"] == termination
    assert result["actions_taken"] == 0
    assert result["estimated_action_cost"] == 0
    assert result["final_state"] == 0


def test_rollout_cannot_use_an_oracle_tail_to_claim_completion():
    calls = iter((prediction("A"), prediction("S")))
    result = evaluation.rollout(lambda _: next(calls), chain(), 2)
    assert result["actions_taken"] == 1 and result["premature_stop"]
    assert result["final_state"] == 1 and not result["completed"]


def test_rollout_horizon_is_enforced():
    result = evaluation.rollout(lambda _: prediction("A"), chain(), 1)
    assert result["termination"] == "horizon_exhausted"
    assert not result["completed"]


def test_solvable_completion_excludes_already_done_and_unsolvable_cases():
    base = chain()
    done, blocked = replace(base, state=2), replace(base, actions=())
    result = evaluation.evaluate_cases(lambda _: prediction("S"),
                 [example(base, 2, "solvable"), example(done, 2, "done"), example(blocked, 2, "blocked")], rollout_count=3)
    model = result["policies"]["model"]
    rollouts = model["rollouts"]
    assert rollouts["initially_done"] == 1
    assert rollouts["initially_unsatisfied_solvable"] == 1
    assert rollouts["unsolvable_within_supplied_horizon"] == 1
    assert rollouts["solvable_completion_rate"] == 0
    assert rollouts["already_done_correct_stop"] == 1
    assert model["first_action"]["premature_stop_count"] == 2
    assert result["policies"]["exact_search"]["rollouts"]["solvable_completion_rate"] == 1
    assert result["promotion"] is False


def test_all_model_predictions_finish_before_reference_search(monkeypatch):
    calls = []
    def policy(prompt):
        scenario, _ = parse_prompt(prompt)
        calls.append(scenario.state)
        return prediction("A" if scenario.state == 0 else "B")
    original_search = evaluation.search
    def checked_search(*args, **kwargs):
        assert calls == [0, 1]
        return original_search(*args, **kwargs)
    monkeypatch.setattr(evaluation, "search", checked_search)
    result = evaluation.evaluate_cases(policy, [example(chain(), 2)], rollout_count=1)
    assert result["policies"]["model"]["rollouts"]["solvable_completed"] == 1


def test_equal_optimal_choices_are_distinct_from_canonical_agreement():
    scenario = Scenario("x", "manual", "validation", ("done",), 0, 1,
                        (Action("A", "one", sets=1, tokens=5), Action("B", "two", sets=1, tokens=5)))
    result = evaluation.evaluate_cases(lambda _: prediction("B"), [example(scenario)], rollout_count=1)
    scores = result["policies"]["model"]["first_action"]
    assert scores["canonical_accuracy"] == 0 and scores["optimal_accuracy"] == 1


def test_cheap_greedy_reference_is_not_an_exact_solver():
    scenario = Scenario("x", "manual", "validation", ("done", "noise"), 0, 1,
                        (Action("A", "noise", sets=2, tokens=0), Action("B", "win", sets=1, tokens=5)))
    result = evaluation.evaluate_cases(lambda _: prediction("B"), [example(scenario, 1)], rollout_count=1)
    assert result["policies"]["greedy"]["rollouts"]["solvable_completion_rate"] == 0
    assert result["policies"]["exact_search"]["rollouts"]["solvable_completion_rate"] == 1


def test_no_solvable_cases_has_null_rate_not_perfect_completion():
    done = replace(chain(), state=2)
    result = evaluation.evaluate_cases(lambda _: prediction("S"), [example(done)], rollout_count=1)
    assert result["policies"]["model"]["rollouts"]["solvable_completion_rate"] is None


def audit_setup(tmp_path, monkeypatch, split="test"):
    # Metadata-only sealed-audit tests. No model loads and no final generation.
    config = ModelConfig(hidden_size=16, intermediate_size=32, num_layers=1,
                         num_attention_heads=2, num_key_value_heads=1)
    info = {"model_sha256": "fixed-weights", "parameters": config.parameter_count()}
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    monkeypatch.setattr(evaluation, "checkpoint_info", lambda _: (tmp_path / "checkpoint", info, config))
    monkeypatch.setattr(evaluation, "evaluator_source_hashes", lambda: {"fixture": "fixed-source"})
    monkeypatch.setattr(evaluation, "make_example", lambda *a, **k: pytest.fail("must not generate final cases"))
    monkeypatch.setattr(evaluation, "load_policy", lambda *a, **k: pytest.fail("must not load a model"))
    return evaluation.parser().parse_args(["--run", str(tmp_path / "run"), "--output", str(tmp_path / "audit.json"),
                                          "--split", split, "--seed", "93001", "--cases", "4", "--rollouts", "2"])


def test_checkpoint_metadata_validated_before_any_model_allocation(tmp_path, monkeypatch):
    config = ModelConfig(hidden_size=16, intermediate_size=32, num_layers=1,
                         num_attention_heads=2, num_key_value_heads=1)
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"metadata-test-do-not-load")
    info = {"config": asdict(config), "parameters": config.parameter_count(),
            "model_sha256": evaluation.digest(weights)}
    (tmp_path / "checkpoint.json").write_text(json.dumps(info))
    monkeypatch.setattr(evaluation, "Decoder", lambda *_: pytest.fail("metadata checks must not allocate a model"))
    assert evaluation.checkpoint_info(tmp_path)[2] == config
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        evaluation.checkpoint_info(tmp_path)


def test_final_audit_requires_separate_declaration_before_generation(tmp_path, monkeypatch):
    args = audit_setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="declare-audit"):
        evaluation.main(args)
    args.declare_audit = True
    evaluation.main(args)
    sealed = json.loads((tmp_path / "audit.json.audit.json").read_text())
    assert sealed["protocol"]["model_sha256"] == "fixed-weights"
    assert not (tmp_path / "audit.json").exists()
    assert not (tmp_path / "audit.json.audit-started.json").exists()


def test_audit_execution_is_consumed_once_before_case_generation(tmp_path, monkeypatch):
    args = audit_setup(tmp_path, monkeypatch)
    args.declare_audit = True
    evaluation.prepare_evaluation(args)
    args.declare_audit, args.run_audit = False, True
    evaluation.prepare_evaluation(args)
    assert (tmp_path / "audit.json.audit-started.json").exists()
    with pytest.raises(FileExistsError):
        evaluation.prepare_evaluation(args)


@pytest.mark.parametrize("change", ["source", "seed", "cases", "checkpoint"])
def test_audit_rejects_any_changed_sealed_inputs(tmp_path, monkeypatch, change):
    args = audit_setup(tmp_path, monkeypatch)
    args.declare_audit = True
    evaluation.prepare_evaluation(args)
    args.declare_audit, args.run_audit = False, True
    if change == "source":
        monkeypatch.setattr(evaluation, "evaluator_source_hashes", lambda: {"fixture": "changed"})
    elif change == "checkpoint":
        original = evaluation.checkpoint_info
        monkeypatch.setattr(evaluation, "checkpoint_info", lambda run: (original(run)[0],
                            {**original(run)[1], "model_sha256": "changed"}, original(run)[2]))
    else:
        setattr(args, change, getattr(args, change) + 1)
    with pytest.raises(ValueError, match="sealed|sealing"):
        evaluation.prepare_evaluation(args)
    assert not (tmp_path / "audit.json.audit-started.json").exists()


def test_cannot_reuse_final_seed_by_changing_output(tmp_path, monkeypatch):
    args = audit_setup(tmp_path, monkeypatch)
    args.declare_audit = True
    evaluation.prepare_evaluation(args)
    args.output = str(tmp_path / "another.json")
    with pytest.raises(FileExistsError):
        evaluation.prepare_evaluation(args)


def test_validation_does_not_need_or_accept_final_audit_flags(tmp_path, monkeypatch):
    args = audit_setup(tmp_path, monkeypatch, "structural_validation")
    assert evaluation.prepare_evaluation(args)[-1] is False
    args.declare_audit = True
    with pytest.raises(ValueError, match="final"):
        evaluation.prepare_evaluation(args)


@pytest.mark.parametrize("curriculum,version", [("tiny", DATA_VERSION), ("graph", DATA_VERSION),
                                               ("primitives", PRIMITIVES_VERSION)])
def test_protocol_records_curriculum_specific_data_version(tmp_path, monkeypatch, curriculum, version):
    args = audit_setup(tmp_path, monkeypatch, "validation")
    args.curriculum = curriculum
    protocol = evaluation.prepare_evaluation(args)[3]
    assert protocol["data_version"] == version
    parsed = evaluation.parser().parse_args(["--run", "unused", "--output", "unused", "--seed", "1",
                                            "--curriculum", curriculum])
    assert parsed.curriculum == curriculum


def test_primitive_final_seal_records_its_distinct_data_version_without_generating_cases(tmp_path, monkeypatch):
    args = audit_setup(tmp_path, monkeypatch)
    args.curriculum = "primitives"
    args.declare_audit = True
    evaluation.prepare_evaluation(args)
    reservation = next((tmp_path / "runs" / "scale-audit-reservations").glob("*.json"))
    assert json.loads(reservation.read_text())["domain"]["data_version"] == PRIMITIVES_VERSION


@pytest.mark.parametrize("tokens", [-1, 260, True])
def test_bad_prediction_tokens_cannot_be_coerced_to_actions(tokens):
    with pytest.raises(ValueError):
        evaluation.Prediction(tokens, 0)
