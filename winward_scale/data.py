"""Own byte tokenizer and exact synthetic text-policy supervision.

Version 1 is a narrow, fully observed workflow DSL, not natural-language
pretraining. Every fact, permission, transition and cost used by the teacher is
in the prompt. No Qwen labels, sampled hidden world, or external model is used.
Validation/test have separate deterministic seed domains. Structural splits
also hold out graph families; test splits are never generated implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import random
from typing import Iterator

from agent_training.simulator import (
    Action, Scenario, STOP, NEEDS_CLARIFICATION, make_scenario, search,
)


DATA_VERSION = "winward-scale-dsl-v1"
PRIMITIVES_VERSION = "winward-scale-primitives-v1"
STOP_LABEL = "S"
DEFER_LABEL = "?"
SPLITS = ("train", "validation", "test", "structural_validation", "structural_test")
TINY_FAMILIES = {
    "train": ("chain", "shortcut", "cost_choice"),
    "validation": ("chain", "shortcut", "cost_choice"),
    "test": ("chain", "shortcut", "cost_choice"),
    "structural_validation": ("fork",),
    "structural_test": ("clearing_trap", "goal_revision"),
}
GRAPH_FAMILIES = {
    "train": ("simple_chain", "fork", "shortcut", "shared_prerequisite"),
    "validation": ("simple_chain", "fork", "shortcut", "shared_prerequisite"),
    "test": ("simple_chain", "fork", "shortcut", "shared_prerequisite"),
    "structural_validation": ("two_stage_fork", "blocked_shortcut"),
    "structural_test": ("nested_dependencies", "changed_goal", "reversible_trap"),
}
PRIMITIVE_FAMILIES = {
    "train": ("goal_check", "action_match", "permission", "prerequisite", "cost_choice"),
    "validation": ("goal_check", "action_match", "permission", "prerequisite", "cost_choice"),
    "test": ("goal_check", "action_match", "permission", "prerequisite", "cost_choice"),
    "structural_validation": ("combined_gate", "revised_goal"),
    "structural_test": ("state_and_cost", "mixed_constraints"),
}
PROMPT_HEADER = (
    "Pick one ID. Win before cost(tokens+ms/100), steps, ID. Look ahead D steps. "
    "S=win true; ?=no plan. Exact effects; unlisted facts false. -=empty.\n"
)
PRIMITIVE_PROMPT_HEADER = (
    "Pick ID. Win, then tokens+ms/100, steps, ID. Exact effects; <=D steps. "
    "S=win met; ?=no plan; -=empty.\n"
)
ACTION_HEADER = "ID|need|forbid|set|clear|tokens|ms|allowed\n"
LIMITATION = (
    "Synthetic deterministic software-workflow DSL only. Exact supplied effects "
    "and estimated costs, at most five actions. This does not establish free-text "
    "understanding, hidden-cause reasoning, actual tool success, or broad competence."
)


class ByteTokenizer:
    """A fixed tokenizer created here: UTF-8 bytes and four control tokens.

    A byte is usually a character for this ASCII curriculum. This avoids any
    borrowed tokenizer/weights, but uses longer sequences than learned subwords.
    Special-looking strings remain ordinary bytes, never injected control IDs.
    """

    vocab_size = 260
    pad_id = 256
    bos_id = 257
    eos_id = 258
    sep_id = 259
    specials = {"pad": pad_id, "bos": bos_id, "eos": eos_id, "sep": sep_id}

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return ([self.bos_id] if add_bos else []) + list(text.encode("utf-8")) + (
            [self.eos_id] if add_eos else []
        )

    def decode(self, ids, *, skip_special_tokens: bool = True) -> str:
        values = []
        for value in ids:
            if type(value) is not int or not 0 <= value < self.vocab_size:
                raise ValueError("token IDs must be integers within the vocabulary")
            if value >= 256:
                if not skip_special_tokens:
                    raise ValueError("control tokens do not have a UTF-8 text representation")
                continue
            values.append(value)
        return bytes(values).decode("utf-8", errors="strict")

    def encode_prompt(self, prompt: str) -> list[int]:
        return self.encode(prompt, add_bos=True) + [self.sep_id]

    def encode_example(self, example: "TextExample", *, max_length: int | None = None):
        return encode_example(example, tokenizer=self, max_length=max_length)


@dataclass(frozen=True)
class TextExample:
    prompt: str
    completion: str
    family: str
    id: str
    teacher: dict

    def to_dict(self) -> dict:
        return {"prompt": self.prompt, "completion": self.completion,
                "family": self.family, "id": self.id, "teacher": self.teacher}


def encode_example(example: TextExample, *, tokenizer: ByteTokenizer | None = None,
                   max_length: int | None = None) -> tuple[list[int], list[float]]:
    """Return tokens and an aligned completion-only loss mask, without truncation.

    For next-token training use tokens[:-1], tokens[1:], and loss_mask[1:].
    Prompt and separator targets are masked out; completion and EOS are trained.
    """
    tokenizer = tokenizer or ByteTokenizer()
    prefix = tokenizer.encode_prompt(example.prompt)
    target = tokenizer.encode(example.completion, add_eos=True)
    ids = prefix + target
    if max_length is not None:
        _nonnegative_int(max_length, "max_length")
        if len(ids) > max_length:
            raise ValueError(f"Example {example.id} needs {len(ids)} tokens; limit is {max_length}; no truncation")
    return ids, [0.0] * len(prefix) + [1.0] * len(target)


def _nonnegative_int(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _domain_seed(index: int, split: str, seed: int, curriculum: str) -> int:
    _nonnegative_int(index, "index")
    _nonnegative_int(seed, "seed")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if curriculum not in ("tiny", "graph", "primitives"):
        raise ValueError("curriculum must be tiny, graph, or primitives")
    version = PRIMITIVES_VERSION if curriculum == "primitives" else DATA_VERSION
    key = f"{version}|{curriculum}|{split}|{seed}|{index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(key).digest()[:16], "big")


def _primitive_scenario(seed: int, family: str, variant: int) -> Scenario:
    """Single-step building blocks; held-out families combine known mechanics.

    Families never enter the prompt. Costs/effects are displayed, and the same
    public exact solver produces every target. The training cycle has 20% STOP,
    20% no-plan, and 60% action outcomes before any label randomization.
    """
    rng = random.Random(seed)
    facts = ("win", "ready")
    state, goal = 0, 1
    actions = []

    def add(*, requires=0, forbids=0, sets=0, clears=0, tokens=1, ms=0, allowed=True):
        actions.append(Action(str(len(actions)), "public action", requires, forbids, sets,
                              clears, tokens, ms, allowed))

    if family == "goal_check":
        state = 1 | (2 if variant % 2 else 0)
        add(sets=1, tokens=rng.randint(1, 4))
        add(sets=2, tokens=0)
    elif family == "action_match":
        state = 2 if variant % 2 else 0
        add(sets=1 if variant != 3 else 2)
        add(sets=2, tokens=0)
    elif family == "permission":
        add(sets=1, allowed=False, tokens=0)
        add(sets=1 if variant >= 2 else 2, tokens=2)
    elif family == "prerequisite":
        negative = bool(rng.getrandbits(1))
        state = (2 if variant == 0 else 0) if negative else (0 if variant == 0 else 2)
        add(requires=0 if negative else 2, forbids=2 if negative else 0, sets=1)
        # This no-effect alternative cannot supply the missing prerequisite.
        add(sets=0, tokens=0)
    elif family == "cost_choice":
        add(sets=1, tokens=rng.randint(1, 4), ms=rng.choice((0, 100, 200)))
        add(sets=1, tokens=rng.randint(1, 4), ms=rng.choice((0, 100, 200)))
        if variant % 2:
            add(sets=2, tokens=0)
    elif family == "combined_gate":
        state = 2 if variant in (1, 3) else 0
        add(requires=2, sets=1, allowed=variant >= 2, tokens=0)
        add(sets=2, tokens=1)
    elif family == "revised_goal":
        state, goal = 1, 2
        add(sets=1, tokens=0)
        add(requires=1, sets=2, allowed=variant != 0, tokens=2)
    elif family == "state_and_cost":
        state = 2 if variant % 2 else 0
        add(requires=2, sets=1, tokens=1)
        add(forbids=2, sets=1, tokens=2)
        add(sets=1, tokens=3, ms=rng.choice((0, 100)))
    elif family == "mixed_constraints":
        state = (1 if variant == 0 else 0) | (2 if variant % 2 else 0)
        add(requires=2, sets=1, tokens=1)
        add(forbids=2, sets=1, tokens=2)
        add(sets=1, tokens=0, allowed=False)
        add(sets=1, tokens=3, ms=rng.choice((0, 100)))
    else:
        raise ValueError(f"unknown primitive family {family}")
    return Scenario("primitive", family, "train", facts, state, goal, tuple(actions))


def _tiny_scenario(seed: int, family: str) -> Scenario:
    rng = random.Random(seed)
    facts = ("known", "prepared", "changed", "verified", "noise", "blocked")
    actions = []
    progress = []
    state = 0

    def add(name, *, requires=0, forbids=0, sets=0, clears=0, tokens=None, ms=None, allowed=True):
        action = Action(str(len(actions)), name, requires, forbids, sets, clears,
                        rng.randint(1, 30) if tokens is None else tokens,
                        rng.randint(1, 500) if ms is None else ms, allowed)
        actions.append(action)
        return len(actions) - 1

    if family in ("chain", "shortcut"):
        length = rng.randint(1 if family == "chain" else 2, 3)
        for i in range(length):
            progress.append(add("advance", requires=1 << (i - 1) if i else 0,
                                forbids=1 << i, sets=1 << i))
        goal = 1 << (length - 1)
        if family == "shortcut":
            add("integrated", sets=(1 << length) - 1, tokens=rng.randint(5, 95), ms=rng.randint(1, 1400))
    elif family == "cost_choice":
        goal = 8
        for _ in range(3):
            progress.append(add("verify", sets=goal, forbids=goal))
    elif family == "fork":
        progress.append(add("branch1", sets=1, forbids=1))
        progress.append(add("branch2", sets=2, forbids=2))
        progress.append(add("combine", requires=3, sets=8, forbids=8))
        goal = 8
    elif family == "clearing_trap":
        state, goal = 1, 8
        add("quick", requires=1, sets=4, clears=1, forbids=4, tokens=1, ms=1)
        progress.append(add("careful", requires=1, sets=4, forbids=4, tokens=rng.randint(3, 15), ms=1))
        add("recover", sets=1, forbids=1, tokens=rng.randint(20, 35), ms=1)
        progress.append(add("verify", requires=5, sets=8, forbids=8, tokens=2, ms=1))
    elif family == "goal_revision":
        state, goal = 4, 8
        progress.append(add("read_revision", sets=1, forbids=1))
        progress.append(add("prepare_revision", requires=1, sets=2, forbids=2))
        progress.append(add("verify_revision", requires=2, sets=8, forbids=8))
        add("old_goal", sets=4, tokens=0, ms=0)
    else:
        raise ValueError(f"unknown tiny family {family}")
    prefix = rng.choices(range(len(progress) + 1), weights=[6] + [1] * len(progress))[0]
    for index in progress[:prefix]:
        if actions[index].eligible(state):
            state = actions[index].apply(state)
    add("irrelevant", sets=16, forbids=16, tokens=1, ms=1)
    add("forbidden", sets=goal, tokens=0, ms=0, allowed=False)
    if rng.random() < .18:
        index = rng.choice(progress)
        actions[index] = replace(actions[index], allowed=False)
    if rng.random() < .08:
        goal |= 32
    if rng.random() < .07:
        state |= goal
    return Scenario("generated", family, "train", facts, state, goal, tuple(actions))


def _scramble(scenario: Scenario, rng: random.Random) -> Scenario:
    """Randomize fact encoding, row order, and label assignment independently."""
    permutation = list(range(len(scenario.facts)))
    rng.shuffle(permutation)
    facts = [""] * len(permutation)
    for old, new in enumerate(permutation):
        facts[new] = scenario.facts[old]

    def remap(mask):
        return sum(1 << permutation[i] for i in range(len(permutation)) if mask & (1 << i))

    actions = list(scenario.actions)
    rng.shuffle(actions)
    actions = [replace(a, id=chr(65 + i), requires=remap(a.requires), forbids=remap(a.forbids),
                       sets=remap(a.sets), clears=remap(a.clears)) for i, a in enumerate(actions)]
    rng.shuffle(actions)
    return replace(scenario, facts=tuple(facts), actions=tuple(actions), state=remap(scenario.state), goal=remap(scenario.goal))


def serialize_prompt(scenario: Scenario, max_depth: int = 5, *, prompt_style: str = "workflow") -> str:
    """Serialize only public mechanics, with no family, seed, plan or label."""
    if type(max_depth) is not int or not 1 <= max_depth <= 5:
        raise ValueError("max_depth must be an integer from 1 through 5")
    if prompt_style not in ("workflow", "primitives"):
        raise ValueError("unknown prompt style")
    if any(a.id not in "ABCDEFGHIJ" or len(a.id) != 1 for a in scenario.actions):
        raise ValueError("action IDs must be single letters A through J")

    def symbols(mask):
        return "".join(chr(97 + i) for i in range(len(scenario.facts)) if mask & (1 << i)) or "-"

    if any(any(c in f for c in "|=;\n") for f in scenario.facts):
        raise ValueError("fact names contain reserved DSL separators")
    header = PRIMITIVE_PROMPT_HEADER if prompt_style == "primitives" else PROMPT_HEADER
    lines = [header, "F:" + ";".join(f"{chr(97+i)}={f}" for i, f in enumerate(scenario.facts)) + "\n",
             f"H:{symbols(scenario.state)};G:{symbols(scenario.goal)};D:{max_depth}\n", ACTION_HEADER]
    for a in scenario.actions:
        lines.append("|".join((a.id, symbols(a.requires), symbols(a.forbids), symbols(a.sets),
                               symbols(a.clears), f"{a.tokens:g}", f"{a.latency_ms:g}", str(int(a.allowed)))) + "\n")
    return "".join(lines) + "Next:"


def parse_prompt(prompt: str) -> tuple[Scenario, int]:
    """Recover exactly the teacher-visible problem from a serialized prompt."""
    header = next((h for h in (PROMPT_HEADER, PRIMITIVE_PROMPT_HEADER) if isinstance(prompt, str) and prompt.startswith(h)), None)
    if header is None or not prompt.endswith("\nNext:"):
        raise ValueError("not a version 1 workflow prompt")
    lines = prompt[len(header):].splitlines()
    if len(lines) < 4 or not lines[0].startswith("F:") or lines[2] + "\n" != ACTION_HEADER:
        raise ValueError("invalid workflow prompt structure")
    pairs = [part.split("=", 1) for part in lines[0][2:].split(";")]
    if any(len(pair) != 2 or pair[0] != chr(97 + i) for i, pair in enumerate(pairs)):
        raise ValueError("invalid fact declarations")
    facts = tuple(pair[1] for pair in pairs)

    def mask(symbols):
        if symbols == "-":
            return 0
        if not symbols or len(set(symbols)) != len(symbols) or any(not "a" <= c <= chr(96 + len(facts)) for c in symbols):
            raise ValueError("invalid fact reference")
        return sum(1 << (ord(c) - 97) for c in symbols)

    fields = lines[1].split(";")
    if len(fields) != 3 or not all(v.startswith(k) for v, k in zip(fields, ("H:", "G:", "D:"))):
        raise ValueError("invalid state or horizon")
    state, goal, depth = mask(fields[0][2:]), mask(fields[1][2:]), int(fields[2][2:])
    if not 1 <= depth <= 5:
        raise ValueError("invalid horizon")
    actions = []
    for line in lines[3:-1]:
        row = line.split("|")
        if len(row) != 8 or len(row[0]) != 1 or row[0] not in "ABCDEFGHIJ" or row[7] not in ("0", "1"):
            raise ValueError("invalid action row")
        actions.append(Action(row[0], row[0], mask(row[1]), mask(row[2]), mask(row[3]), mask(row[4]),
                              float(row[5]), float(row[6]), row[7] == "1"))
    return Scenario("prompt", "public_dsl", "inference", facts, state, goal, tuple(actions)), depth


def _label(action_id: str) -> str:
    return STOP_LABEL if action_id == STOP else DEFER_LABEL if action_id == NEEDS_CLARIFICATION else action_id


def make_example(index: int, split: str = "train", *, seed: int = 0,
                 curriculum: str = "tiny") -> TextExample:
    generator_seed = _domain_seed(index, split, seed, curriculum)
    rng = random.Random(generator_seed)
    families = {"tiny": TINY_FAMILIES, "graph": GRAPH_FAMILIES, "primitives": PRIMITIVE_FAMILIES}[curriculum]
    family = families[split][index % len(families[split])]
    if curriculum == "primitives":
        scenario = _primitive_scenario(generator_seed, family, (index // len(families[split])) % 4)
    elif curriculum == "tiny":
        scenario = _tiny_scenario(generator_seed, family)
    else:
        source_split = "validation" if split == "structural_validation" else "test" if split == "structural_test" else "train"
        scenario = make_scenario(generator_seed, source_split, family)
    scenario = _scramble(scenario, rng)
    depth = 1 if curriculum == "primitives" else rng.choices((1, 2, 3, 4, 5), weights=(1, 1, 2, 2, 6))[0]
    prompt = serialize_prompt(scenario, depth, prompt_style="primitives" if curriculum == "primitives" else "workflow")
    # Solve the serialized public problem itself, preventing accidental reliance
    # on generator-only context or facts not transmitted to the model.
    public_problem, depth = parse_prompt(prompt)
    result = search(public_problem, max_depth=depth)
    version = PRIMITIVES_VERSION if curriculum == "primitives" else DATA_VERSION
    example_id = f"{version}/{curriculum}/{split}/{seed}/{index}"
    return TextExample(prompt, _label(result["chosen_action_id"]), family, example_id, {
        "version": version, "split": split, "curriculum": curriculum,
        "generator_seed": generator_seed, "max_depth": depth,
        "optimal_labels": [_label(a) for a in result["optimal_action_ids"]],
        "accepted_labels": [_label(a) for a in result["optimal_action_ids"]],
        "candidate_labels": [a.id for a in public_problem.actions] + [STOP_LABEL, DEFER_LABEL],
        "available_labels": [a.id for a in public_problem.actions if a.eligible(public_problem.state)]
                            + ([STOP_LABEL] if public_problem.goal_met() else [])
                            + ([DEFER_LABEL] if not result["success"] else []),
        "initial_state": public_problem.state, "goal": public_problem.goal,
        "plan": [_label(a) for a in result["plan_action_ids"]],
        "success": result["success"], "estimated_cost": result["total_cost"],
        "outcome": result["outcome"], "limitation": LIMITATION,
    })


def iter_examples(split: str, count: int, *, start: int = 0, seed: int = 0,
                  curriculum: str = "tiny") -> Iterator[TextExample]:
    """Stateless, resumable streaming; each index has its own independent RNG."""
    _nonnegative_int(count, "count")
    _nonnegative_int(start, "start")
    _domain_seed(start, split, seed, curriculum)
    for index in range(start, start + count):
        yield make_example(index, split, seed=seed, curriculum=curriculum)


def verify_prediction(example: TextExample, prediction: str) -> dict:
    """Objective first-action verification, conditional on an exact future solver.

    This is not a model rollout. Feasibility means an oracle continuation exists
    after this first action, under the public mechanics and remaining horizon.
    """
    scenario, depth = parse_prompt(example.prompt)
    result = search(scenario, max_depth=depth)
    prediction = prediction.strip() if isinstance(prediction, str) else ""
    optimal = {_label(a) for a in result["optimal_action_ids"]}
    row = {"valid_label": prediction in {a.id for a in scenario.actions} | {STOP_LABEL, DEFER_LABEL},
           "canonical_match": prediction == _label(result["chosen_action_id"]),
           "optimal": prediction in optimal, "eligible": False,
           "goal_reachable_with_oracle_continuation": False, "estimated_cost_regret": None,
           "teacher_outcome": result["outcome"]}
    if prediction == STOP_LABEL:
        row["eligible"] = scenario.goal_met()
        row["goal_reachable_with_oracle_continuation"] = scenario.goal_met()
        if scenario.goal_met():
            row["estimated_cost_regret"] = 0.0
    elif prediction == DEFER_LABEL:
        row["eligible"] = not result["success"]
    else:
        action = next((a for a in scenario.actions if a.id == prediction), None)
        if action is not None and action.eligible(scenario.state):
            row["eligible"] = True
            after = replace(scenario, state=action.apply(scenario.state))
            tail = search(after, max_depth=depth - 1)
            row["goal_reachable_with_oracle_continuation"] = tail["success"]
            if tail["success"] and result["success"]:
                row["estimated_cost_regret"] = action.cost() + tail["total_cost"] - result["total_cost"]
    return row
