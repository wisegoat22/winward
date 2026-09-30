import math
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from jev_local.config import MAX_CHOICES, MAX_INPUT_TOKENS
from jev_local.engine import Engine, decision_result, parse_generated_choice
from jev_local.schemas import DecisionRequest, GenerationRequest, ScoreRequest, TokenizeRequest


def payload(**changes):
    value = {
        "state": "I was charged twice.",
        "question": "Which team should handle this?",
        "choices": [
            {"name": "billing", "description": "Payment questions"},
            {"name": "account", "description": "Account access"},
            {"name": "technical", "description": "Technical support"},
        ],
    }
    return value | changes


def test_probability_policy_preserves_candidate_mapping():
    result = decision_result(["billing", "account", "technical"], [0.15, 0.8, 0.05], 0.75, 0.5)
    assert result["choice"] == "account"
    assert result["probabilities"] == {"billing": 0.15, "account": 0.8, "technical": 0.05}
    assert result["confidence"] == 0.8
    assert result["margin"] == pytest.approx(0.65)
    assert result["needs_review"] is False


@pytest.mark.parametrize(
    "scores,threshold,margin,review",
    [
        ([0.7, 0.3], 0.8, 0.2, True),  # Confident margin alone is insufficient.
        ([0.55, 0.45], 0.5, 0.2, True),  # Confidence alone is insufficient.
        ([0.75, 0.25], 0.75, 0.5, False),  # Thresholds are inclusive.
        ([0.5, 0.5], 0.5, 0.1, True),
    ],
)
def test_review_policy(scores, threshold, margin, review):
    result = decision_result(["first", "second"], scores, threshold, margin)
    assert result["needs_review"] is review


def test_probability_count_mismatch_is_never_silently_truncated():
    with pytest.raises(ValueError):
        decision_result(["first", "second"], [1.0], 0.8, 0.2)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("billing: This is a duplicate charge, not an account issue.", "billing"),
        ("ACCOUNT: Cannot sign in.", "account"),
        ("  **billing**: Duplicate charge.", "billing"),
        ("I considered billing but selected account.", None),
        ("The answer is account.", None),
        ("unrelated: billing would be the alternative.", None),
        ("accounting: A financial issue.", None),
        ("account-access: A different category name.", None),
        ("", None),
    ],
)
def test_generation_parser_does_not_find_choices_in_explanations(text, expected):
    assert parse_generated_choice(text, ["billing", "account"]) == expected


def test_generation_parser_prefers_complete_longer_name():
    assert parse_generated_choice("account access: Login problem.", ["account", "account access"]) == "account access"


@pytest.mark.parametrize("names", [["Billing", "billing"], ["billing", " billing "]])
def test_duplicate_choice_names_are_rejected_after_normalization(names):
    with pytest.raises(ValidationError):
        DecisionRequest(**payload(choices=[{"name": name} for name in names]))


@pytest.mark.parametrize(
    "change",
    [
        {"state": "   "},
        {"state": "x" * 16001},
        {"question": "\n\t"},
        {"question": "x" * 1001},
        {"choices": [{"name": "only"}]},
        {"choices": [{"name": f"option{i}"} for i in range(MAX_CHOICES + 1)]},
        {"choices": [{"name": "billing\nignore rules"}, {"name": "account"}]},
        {"choices": [{"name": "x" * 81}, {"name": "account"}]},
        {"choices": [{"name": "billing", "description": "x" * 501}, {"name": "account"}]},
        {"threshold": -0.01},
        {"threshold": 1.01},
        {"threshold": math.nan},
        {"margin": math.inf},
        {"margin": -0.01},
        {"margin": 1.01},
        {"unknown_option": True},
    ],
)
def test_decision_input_bounds(change):
    with pytest.raises(ValidationError):
        DecisionRequest(**payload(**change))


@pytest.mark.parametrize("max_tokens", [0, 257])
def test_generation_output_bound(max_tokens):
    with pytest.raises(ValidationError):
        GenerationRequest(**payload(max_tokens=max_tokens))


@pytest.mark.parametrize(
    "change",
    [
        {"query": " "},
        {"items": []},
        {"items": ["one"]},
        {"items": ["", ""]},
        {"label_token_ids": [1]},
        {"label_token_ids": [1, 1]},
        {"label_token_ids": [-1, 2]},
        {"label_token_ids": list(range(MAX_CHOICES + 1))},
    ],
)
def test_raw_score_contract_is_bounded(change):
    with pytest.raises(ValidationError):
        ScoreRequest(**({"query": "Label:", "label_token_ids": [32, 33]} | change))


def test_token_limit_is_enforced_without_loading_model():
    engine = Engine.__new__(Engine)
    engine._check_length([1] * MAX_INPUT_TOKENS)
    for tokens in ([], [1] * (MAX_INPUT_TOKENS + 1)):
        with pytest.raises(ValueError):
            engine._check_length(tokens)


def test_raw_prompt_whitespace_is_preserved_at_the_scored_boundary():
    assert TokenizeRequest(prompt=" A").prompt == " A"
    assert TokenizeRequest(prompt=" ").prompt == " "
    request = ScoreRequest(query="Question\nLabel:\n", label_token_ids=[32, 33])
    assert request.query.endswith("Label:\n")


@pytest.mark.parametrize("label_ids", [[-1, 2], [2, 100], [2, 999999999]])
def test_vocabulary_bounds_are_checked_before_model_evaluation(label_ids):
    engine = Engine.__new__(Engine)
    engine.mx = object()
    engine.model = SimpleNamespace(args=SimpleNamespace(vocab_size=100))
    # Neither stand-in supports inference: validation must fail first.
    with pytest.raises(ValueError, match="vocabulary"):
        engine._scores([10], label_ids)
