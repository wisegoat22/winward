"""Independent correctness checks for the narrow scratch text-policy curriculum."""

from dataclasses import replace
from collections import Counter
import hashlib
import itertools

import pytest

from agent_training.simulator import Action, Scenario
from winward_scale.data import (
    ByteTokenizer, DEFER_LABEL, GRAPH_FAMILIES, PRIMITIVE_FAMILIES, PRIMITIVES_VERSION,
    SPLITS, STOP_LABEL, TINY_FAMILIES,
    TextExample, encode_example, iter_examples, make_example, parse_prompt,
    serialize_prompt, verify_prediction,
)


def brute_force(scenario, depth):
    """Exhaustive independent transition enumeration; no teacher helper calls."""
    wins = []
    for n in range(depth + 1):
        for sequence in itertools.product(scenario.actions, repeat=n):
            state = scenario.state
            for action in sequence:
                if not action.allowed or state & action.requires != action.requires or state & action.forbids:
                    break
                state = (state & ~action.clears) | action.sets
            else:
                if state & scenario.goal == scenario.goal:
                    wins.append((sum(a.tokens + a.latency_ms / 100 for a in sequence),
                                 len(sequence), tuple(a.id for a in sequence)))
    return min(wins) if wins else None


def as_example(scenario, depth=5):
    return TextExample(serialize_prompt(scenario, depth), "unused", "manual", "manual", {})


def test_byte_tokenizer_is_owned_fixed_utf8_and_control_tokens_cannot_be_injected():
    tokenizer = ByteTokenizer()
    value = "Goal: fix a café 🐐. <bos><sep><eos>"
    ids = tokenizer.encode(value)
    assert ids == list(value.encode("utf-8"))
    assert max(ids) < 256 and tokenizer.vocab_size == 260
    assert tokenizer.decode(ids) == value
    assert tokenizer.decode([tokenizer.bos_id, *ids, tokenizer.eos_id, tokenizer.pad_id]) == value
    assert len(set(tokenizer.specials.values())) == 4
    assert tokenizer.encode_prompt("abc") == [257, 97, 98, 99, 259]


@pytest.mark.parametrize("bad", [[-1], [260], [True], [1.0], ["a"]])
def test_tokenizer_rejects_invalid_ids(bad):
    with pytest.raises(ValueError):
        ByteTokenizer().decode(bad)


def test_tokenizer_does_not_hide_broken_utf8_or_special_tokens_when_requested():
    with pytest.raises(UnicodeDecodeError):
        ByteTokenizer().decode([255])
    with pytest.raises(ValueError):
        ByteTokenizer().decode([257], skip_special_tokens=False)


def test_completion_only_loss_mask_trains_one_label_then_eos_and_never_truncates():
    example = make_example(0)
    ids, mask = encode_example(example)
    assert len(ids) == len(mask)
    assert ids[-2:] == [ord(example.completion), ByteTokenizer.eos_id]
    assert mask[-2:] == [1.0, 1.0] and not any(mask[:-2])
    assert ids[-3] == ByteTokenizer.sep_id
    assert encode_example(example, max_length=len(ids)) == (ids, mask)
    with pytest.raises(ValueError, match="no truncation"):
        encode_example(example, max_length=len(ids) - 1)
    with pytest.raises(ValueError):
        encode_example(example, max_length=True)


@pytest.mark.parametrize("curriculum,limit", [("tiny", 512), ("graph", 768)])
def test_streaming_is_resumable_and_each_split_has_a_separate_rng_domain(curriculum, limit):
    complete = list(iter_examples("train", 24, seed=7, curriculum=curriculum))
    resumed = list(iter_examples("train", 14, start=10, seed=7, curriculum=curriculum))
    assert complete[10:] == resumed
    assert complete[0] == make_example(0, seed=7, curriculum=curriculum)
    assert complete[0] != make_example(0, seed=8, curriculum=curriculum)
    for split in SPLITS:
        for example in iter_examples(split, 36, seed=7, curriculum=curriculum):
            ids, _ = encode_example(example, max_length=limit)
            assert len(ids) <= limit
            assert len(example.completion) == 1
            assert example.completion in example.teacher["accepted_labels"]
            assert example.id not in example.prompt
            assert "generator_seed" not in example.prompt and "estimated_cost" not in example.prompt
    seeds = {make_example(2, split, seed=7, curriculum=curriculum).teacher["generator_seed"] for split in SPLITS}
    assert len(seeds) == len(SPLITS)


@pytest.mark.parametrize("families", [TINY_FAMILIES, GRAPH_FAMILIES])
def test_structural_holdouts_are_distinct_from_training_and_each_other(families):
    assert families["train"] == families["validation"] == families["test"]
    assert not set(families["train"]) & set(families["structural_validation"])
    assert not set(families["train"]) & set(families["structural_test"])
    assert not set(families["structural_validation"]) & set(families["structural_test"])


@pytest.mark.parametrize("kwargs", [dict(index=True), dict(index=-1), dict(index=0, seed=-1),
                                    dict(index=0, split="valid"), dict(index=0, curriculum="unknown")])
def test_generator_rejects_ambiguous_or_invalid_domains(kwargs):
    with pytest.raises(ValueError):
        make_example(**kwargs)


def test_empty_stream_still_checks_split_and_count():
    assert list(iter_examples("train", 0)) == []
    with pytest.raises(ValueError):
        list(iter_examples("unknown", 0))
    with pytest.raises(ValueError):
        list(iter_examples("train", True))


def test_public_prompt_round_trip_preserves_all_mechanics_and_no_generator_context():
    scenario = Scenario("secret-id", "private-family", "test", ("evidence", "change", "passed"), 1, 4,
                        (Action("B", "Name is irrelevant", 1, 4, 2, 1, 13, 81, False),
                         Action("A", "Verify", 2, 0, 4, 0, 3, 20, True)), {"hidden": "unused"})
    prompt = serialize_prompt(scenario, 3)
    recovered, depth = parse_prompt(prompt)
    assert depth == 3 and recovered.facts == scenario.facts
    assert recovered.state == scenario.state and recovered.goal == scenario.goal
    assert recovered.context == {}
    assert all(word not in prompt for word in ("secret-id", "private-family", "unused", "Name is irrelevant"))
    for original, parsed in zip(scenario.actions, recovered.actions):
        assert replace(original, name=parsed.name) == parsed
    assert serialize_prompt(recovered, depth) == prompt


@pytest.mark.parametrize("split", SPLITS)
def test_teacher_labels_match_independent_enumeration_on_all_tiny_split_types(split):
    for example in iter_examples(split, 16, seed=915):
        scenario, depth = parse_prompt(example.prompt)
        expected = brute_force(scenario, depth)
        chosen = DEFER_LABEL if expected is None else expected[2][0] if expected[2] else STOP_LABEL
        assert example.completion == chosen
        if expected is not None:
            assert example.teacher["estimated_cost"] == pytest.approx(expected[0])
        verified = verify_prediction(example, example.completion)
        assert verified["optimal"] and verified["canonical_match"] and verified["eligible"]


def test_label_assignment_and_candidate_row_order_are_not_shortcuts():
    positions, labels = set(), set()
    for example in iter_examples("train", 150):
        scenario, _ = parse_prompt(example.prompt)
        ids = [a.id for a in scenario.actions]
        if example.completion in ids:
            labels.add(example.completion)
            positions.add(ids.index(example.completion))
    assert len(labels) >= 5 and len(positions) >= 5


def test_goal_first_then_cost_and_available_labels_include_no_forbidden_actions():
    scenario = Scenario("x", "x", "x", ("done", "noise"), 0, 1,
                        (Action("A", "noise", sets=2, tokens=0),
                         Action("B", "costly win", sets=1, tokens=10000),
                         Action("C", "forbidden win", sets=1, allowed=False)))
    example = as_example(scenario)
    assert verify_prediction(example, "B")["optimal"]
    assert not verify_prediction(example, "A")["optimal"]
    assert verify_prediction(example, "A")["goal_reachable_with_oracle_continuation"]
    assert not verify_prediction(example, "C")["eligible"]
    assert not verify_prediction(example, "I choose B")["valid_label"]
    for generated in iter_examples("train", 32):
        problem, _ = parse_prompt(generated.prompt)
        for action in problem.actions:
            assert (action.id in generated.teacher["available_labels"]) == action.eligible(problem.state)
        for label in generated.teacher["candidate_labels"]:
            assert (label in generated.teacher["available_labels"]) == verify_prediction(generated, label)["eligible"]


def test_equal_optima_are_accepted_but_canonical_target_and_cost_regret_stay_distinct():
    scenario = Scenario("x", "x", "x", ("done",), 0, 1,
                        (Action("B", "same", sets=1, tokens=5),
                         Action("A", "same", sets=1, tokens=5),
                         Action("C", "costlier", sets=1, tokens=7)))
    example = as_example(scenario)
    assert verify_prediction(example, "A")["canonical_match"]
    assert verify_prediction(example, "B")["optimal"]
    assert not verify_prediction(example, "B")["canonical_match"]
    assert verify_prediction(example, "B")["estimated_cost_regret"] == 0
    assert verify_prediction(example, "C")["estimated_cost_regret"] == 2


def test_stop_and_defer_cannot_claim_unverified_completion():
    incomplete = Scenario("x", "x", "x", ("done",), 0, 1, ())
    assert verify_prediction(as_example(incomplete), DEFER_LABEL)["optimal"]
    assert not verify_prediction(as_example(incomplete), STOP_LABEL)["eligible"]
    complete = replace(incomplete, state=1)
    assert verify_prediction(as_example(complete), STOP_LABEL)["optimal"]
    assert not verify_prediction(as_example(complete), DEFER_LABEL)["optimal"]


def test_verification_ignores_teacher_metadata_and_target_and_reconstructs_public_problem():
    original = make_example(0)
    corrupted = replace(original, completion="not a label", teacher={"accepted_labels": ["Z"], "success": False})
    assert verify_prediction(corrupted, original.completion) == verify_prediction(original, original.completion)


def test_public_cost_and_goal_counterfactuals_change_the_objective_choice():
    original = Scenario("x", "x", "x", ("code_done", "docs_done"), 0, 1,
                        (Action("A", "slow", sets=1, tokens=5, latency_ms=1000),
                         Action("B", "fast", sets=1, tokens=10, latency_ms=1),
                         Action("C", "other goal", sets=2, tokens=1)))
    assert verify_prediction(as_example(original), "B")["optimal"]
    cheaper = replace(original, actions=(replace(original.actions[0], latency_ms=0), *original.actions[1:]))
    assert verify_prediction(as_example(cheaper), "A")["optimal"]
    assert verify_prediction(as_example(replace(original, goal=2)), "C")["optimal"]


def test_five_consequences_and_bounded_failure_are_distinct():
    actions = tuple(Action(chr(65 + i), "advance", requires=1 << (i - 1) if i else 0,
                           sets=1 << i, tokens=1) for i in range(5))
    scenario = Scenario("x", "x", "x", tuple(f"f{i}" for i in range(5)), 0, 16, actions)
    assert verify_prediction(as_example(scenario, 5), "A")["optimal"]
    assert verify_prediction(as_example(scenario, 4), DEFER_LABEL)["optimal"]
    assert not verify_prediction(as_example(scenario, 4), "A")["goal_reachable_with_oracle_continuation"]


@pytest.mark.parametrize("curriculum,split,seed_value,prompt_sha", [
    ("tiny", "train", 29176653177271281506960368074994083643, "57d094359cb5b141cd4a5b44665125ce2e5061c650c1dada25d557faef286ab3"),
    ("tiny", "validation", 261891555305219236436785766664787112148, "e1139924ff8495470f5e6faf0fa24c0b3b81b319bdb2fa4abcb08a467bbd76aa"),
    ("graph", "train", 155490547673984773611892233640229194798, "1bd54d51ec2a4f23523960ac0cff488a1fa3a781c42f8df4be6766189efd7707"),
    ("graph", "validation", 188188252036645187771346875393080994855, "9128903e9a527907233da079d6d955104d01b2acabf5539bfd3e58970cb909a5"),
])
def test_adding_primitives_preserves_existing_seed_mapping_and_exact_prompt_bytes(curriculum, split, seed_value, prompt_sha):
    example = make_example(17, split, seed=42, curriculum=curriculum)
    assert example.teacher["generator_seed"] == seed_value
    assert hashlib.sha256((example.prompt + example.completion).encode()).hexdigest() == prompt_sha


@pytest.mark.parametrize("split", ["train", "validation"])
def test_primitive_curriculum_is_balanced_short_deterministic_and_publicly_solvable(split):
    examples = list(iter_examples(split, 200, seed=512000001, curriculum="primitives"))
    assert examples == list(iter_examples(split, 200, seed=512000001, curriculum="primitives"))
    assert Counter(x.teacher["outcome"] for x in examples) == {"stop": 40, "needs_clarification": 40, "plan": 120}
    assert len({x.completion for x in examples}) >= 4
    for example in examples:
        assert example.teacher["version"] == PRIMITIVES_VERSION
        assert example.id.startswith(PRIMITIVES_VERSION)
        assert len(encode_example(example, max_length=256)[0]) <= 256
        problem, depth = parse_prompt(example.prompt)
        assert len(problem.facts) == 2 and 2 <= len(problem.actions) <= 4 and depth == 1
        expected = brute_force(problem, depth)
        label = DEFER_LABEL if expected is None else expected[2][0] if expected[2] else STOP_LABEL
        assert example.completion == label
        assert verify_prediction(example, label)["optimal"]
        assert serialize_prompt(problem, depth, prompt_style="primitives") == example.prompt
        assert example.family not in example.prompt and "generator_seed" not in example.prompt


def test_primitive_validation_uses_distinct_seeds_and_structural_combinations():
    # Final test and structural_test examples are intentionally not generated here.
    train = list(iter_examples("train", 40, seed=23, curriculum="primitives"))
    valid = list(iter_examples("validation", 40, seed=23, curriculum="primitives"))
    assert {x.teacher["generator_seed"] for x in train}.isdisjoint(x.teacher["generator_seed"] for x in valid)
    assert not set(PRIMITIVE_FAMILIES["train"]) & set(PRIMITIVE_FAMILIES["structural_validation"])
    assert not set(PRIMITIVE_FAMILIES["train"]) & set(PRIMITIVE_FAMILIES["structural_test"])
    assert not set(PRIMITIVE_FAMILIES["structural_validation"]) & set(PRIMITIVE_FAMILIES["structural_test"])
    outcomes = set()
    for example in iter_examples("structural_validation", 32, seed=23, curriculum="primitives"):
        problem, depth = parse_prompt(example.prompt)
        expected = brute_force(problem, depth)
        assert example.completion == (DEFER_LABEL if expected is None else expected[2][0] if expected[2] else STOP_LABEL)
        outcomes.add(example.teacher["outcome"])
    assert outcomes == {"plan", "needs_clarification"}


def test_primitive_permissions_requirements_and_time_cost_are_real_input_dependencies():
    examples = list(iter_examples("train", 200, seed=512000001, curriculum="primitives"))
    seen = set()
    for example in examples:
        problem, depth = parse_prompt(example.prompt)
        if example.family == "permission" and example.completion == DEFER_LABEL:
            unlocked = replace(problem, actions=tuple(replace(a, allowed=True) for a in problem.actions))
            assert not verify_prediction(as_example(unlocked, depth), DEFER_LABEL)["optimal"]
            seen.add("permission")
        if example.family == "prerequisite" and example.completion == DEFER_LABEL:
            goal_action = next(a for a in problem.actions if a.sets & problem.goal == problem.goal)
            prepared = replace(problem, state=(problem.state | goal_action.requires) & ~goal_action.forbids)
            assert not verify_prediction(as_example(prepared, depth), DEFER_LABEL)["optimal"]
            seen.add("blocked" if goal_action.forbids else "prerequisite")
        if example.family == "cost_choice":
            winners = [a for a in problem.actions if a.sets & problem.goal == problem.goal]
            shortest = min(winners, key=lambda a: (a.tokens, a.id))
            if shortest.id != example.completion:
                assert next(a for a in winners if a.id == example.completion).cost() <= shortest.cost()
                seen.add("time_cost")
    assert seen == {"permission", "prerequisite", "blocked", "time_cost"}


def test_four_action_primitive_schema_fits_256_without_generating_a_final_holdout():
    problem = Scenario("manual", "manual", "manual", ("win", "ready"), 2, 1,
                       tuple(Action(chr(65 + i), "action", requires=2, forbids=1,
                                    sets=1, clears=2, tokens=4, latency_ms=200) for i in range(4)))
    example = TextExample(serialize_prompt(problem, 1, prompt_style="primitives"), "A", "manual", "manual", {})
    assert len(encode_example(example, max_length=256)[0]) <= 256
