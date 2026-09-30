"""A finite, declared-world reference planner; this is not a learned policy.

The planner receives a distribution over possible worlds, never the actual world.
It conditions only on observable labels after an action. A goal is *verified*
when it holds in every hypothesis still possible, given the supplied world model.
This is logical verification inside that model, not proof about a real system.

Plans maximize expected verified completion, then minimize expected weighted
token/time cost, then expected actions. Costs and transition probabilities are
supplied assumptions. No tool, shell, network, or language model is called here.
"""

from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
import math
import time
from typing import Mapping


STOP = "STOP"
NEEDS_INFORMATION = "NEEDS_INFORMATION"
_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Hypothesis:
    world_id: str
    state: int
    probability: float


@dataclass(frozen=True)
class Outcome:
    probability: float
    observation: str = "unobserved"
    sets: int = 0
    clears: int = 0


@dataclass(frozen=True)
class BeliefAction:
    id: str
    name: str
    requires: int = 0
    forbids: int = 0
    outcomes: tuple[Outcome, ...] = (Outcome(1.0),)
    outcomes_by_world: Mapping[str, tuple[Outcome, ...]] = field(default_factory=dict)
    tokens: float = 0.0
    latency_ms: float = 0.0
    allowed: bool = True

    def eligible(self, belief: tuple[Hypothesis, ...]) -> bool:
        # Permissions and applicability are hard constraints, not predictions.
        return self.allowed and all(
            h.state & self.requires == self.requires and not h.state & self.forbids
            for h in belief
        )

    def effects(self, world_id: str) -> tuple[Outcome, ...]:
        return self.outcomes_by_world.get(world_id, self.outcomes)


@dataclass(frozen=True)
class BeliefProblem:
    id: str
    facts: tuple[str, ...]
    goal: int
    belief: tuple[Hypothesis, ...]
    actions: tuple[BeliefAction, ...]
    context: str = ""

    def __post_init__(self):
        if not 1 <= len(self.facts) <= 12 or len(set(self.facts)) != len(self.facts):
            raise ValueError("Supply between 1 and 12 distinct fact names.")
        if not all(isinstance(f, str) and f for f in self.facts):
            raise ValueError("Fact names must be nonempty strings.")
        limit = (1 << len(self.facts)) - 1

        def check_mask(mask):
            if type(mask) is not int or mask < 0 or mask > limit:
                raise ValueError("Masks must be nonnegative integers using the declared facts.")

        check_mask(self.goal)
        if not 1 <= len(self.belief) <= 32:
            raise ValueError("Supply between 1 and 32 initial hypotheses.")
        for h in self.belief:
            check_mask(h.state)
            if not isinstance(h.world_id, str) or not h.world_id:
                raise ValueError("World IDs must be nonempty strings.")
            _finite_nonnegative(h.probability, "Hypothesis probability")
        if not math.isclose(math.fsum(h.probability for h in self.belief), 1, abs_tol=1e-9):
            raise ValueError("Initial hypothesis probabilities must sum to 1.")
        object.__setattr__(self, "belief", _normalize(self.belief))
        if len(self.actions) > 10:
            raise ValueError("At most 10 supplied actions are supported.")
        ids = [a.id for a in self.actions]
        if len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
            raise ValueError("Action IDs must be distinct nonempty strings.")
        if set(ids) & {STOP, NEEDS_INFORMATION}:
            raise ValueError("STOP and NEEDS_INFORMATION are reserved action IDs.")
        world_ids = {h.world_id for h in self.belief}
        for a in self.actions:
            check_mask(a.requires)
            check_mask(a.forbids)
            if not isinstance(a.name, str) or not a.name:
                raise ValueError("Action names must be nonempty strings.")
            if type(a.allowed) is not bool:
                raise ValueError("Action permission must be a Boolean.")
            _finite_nonnegative(a.tokens, "Token cost")
            _finite_nonnegative(a.latency_ms, "Time cost")
            # Mappings for a now-eliminated world are retained on belief updates.
            if not all(isinstance(w, str) and w for w in a.outcomes_by_world):
                raise ValueError("Outcome world IDs must be nonempty strings.")
            for outcomes in (a.outcomes, *a.outcomes_by_world.values()):
                if not 1 <= len(outcomes) <= 8:
                    raise ValueError("Each outcome distribution needs 1 to 8 outcomes.")
                for o in outcomes:
                    _finite_nonnegative(o.probability, "Outcome probability")
                    check_mask(o.sets)
                    check_mask(o.clears)
                    if o.sets & o.clears:
                        raise ValueError("An outcome cannot both set and clear the same fact.")
                    if not isinstance(o.observation, str) or not o.observation:
                        raise ValueError("Observations must be nonempty strings.")
                if not math.isclose(math.fsum(o.probability for o in outcomes), 1, abs_tol=1e-9):
                    raise ValueError("Each outcome distribution must sum to 1.")
            if any(not a.effects(w) for w in world_ids):
                raise ValueError("Every possible world needs an outcome distribution.")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "BeliefProblem":
        actions = []
        for row in value["actions"]:
            a = dict(row)
            a["outcomes"] = tuple(Outcome(**o) for o in a.get("outcomes", [{"probability": 1.0}]))
            a["outcomes_by_world"] = {
                w: tuple(Outcome(**o) for o in rows)
                for w, rows in a.get("outcomes_by_world", {}).items()
            }
            actions.append(BeliefAction(**a))
        return cls(id=value["id"], facts=tuple(value["facts"]), goal=value["goal"],
                   belief=tuple(Hypothesis(**h) for h in value["belief"]),
                   actions=tuple(actions), context=value.get("context", ""))


def _finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number.")


def _normalize(rows) -> tuple[Hypothesis, ...]:
    masses = {}
    for h in rows:
        if h.probability > 0:
            key = (h.world_id, h.state)
            masses.setdefault(key, []).append(h.probability)
    combined = {key: math.fsum(values) for key, values in masses.items()}
    total = math.fsum(combined.values())
    if not total:
        raise ValueError("A belief must have positive probability mass.")
    # Do not round or drop small positive hypotheses: even rare uncertainty can
    # matter when deciding whether a goal has actually been verified.
    return tuple(Hypothesis(w, state, probability / total)
                 for (w, state), probability in sorted(combined.items()))


def verified(belief: tuple[Hypothesis, ...], goal: int) -> bool:
    return bool(belief) and all(h.state & goal == goal for h in belief)


def branches(belief: tuple[Hypothesis, ...], action: BeliefAction) -> tuple[tuple[str, float, tuple[Hypothesis, ...]], ...]:
    """Observable-label probabilities and Bayesian posteriors after an action."""
    if not action.eligible(belief):
        raise ValueError("The action is forbidden or inapplicable in a possible world.")
    grouped = {}
    for h in belief:
        effects = action.effects(h.world_id)
        total = math.fsum(o.probability for o in effects)
        for o in effects:
            mass = h.probability * o.probability / total
            if mass > 0:
                next_state = (h.state & ~o.clears) | o.sets
                grouped.setdefault(o.observation, []).append(Hypothesis(h.world_id, next_state, mass))
    result = []
    for observation, rows in sorted(grouped.items()):
        result.append((observation, math.fsum(h.probability for h in rows), _normalize(rows)))
    return tuple(result)


def update_belief(problem: BeliefProblem, action_id: str, observation: str) -> BeliefProblem:
    """Condition on a supplied observed result; this function executes nothing."""
    action = next((a for a in problem.actions if a.id == action_id), None)
    if action is None:
        raise ValueError("Unknown action ID.")
    for label, _, posterior in branches(problem.belief, action):
        if label == observation:
            return replace(problem, belief=posterior)
    raise ValueError("The observation is impossible under the declared world model.")


def revise_goal(problem: BeliefProblem, goal: int) -> BeliefProblem:
    """Keep current evidence and replace the goal; call plan again afterward."""
    return replace(problem, goal=goal)


@dataclass
class _Value:
    success: float
    cost: float
    tokens: float
    latency: float
    steps: float
    tree: dict


def _better(a: _Value, b: _Value) -> bool:
    for x, y, maximize in ((a.success, b.success, True), (a.cost, b.cost, False),
                           (a.steps, b.steps, False)):
        if not math.isclose(x, y, abs_tol=_TOLERANCE, rel_tol=0):
            return x > y if maximize else x < y
    return a.tree["action_id"] < b.tree["action_id"]


def _summary(value: _Value) -> dict:
    return {"expected_verified_success": value.success, "expected_cost": value.cost,
            "expected_tokens": value.tokens, "expected_latency_ms": value.latency,
            "expected_steps": value.steps}


def _terminal(success: bool, reason: str) -> _Value:
    return _Value(float(success), 0.0, 0.0, 0.0, 0.0,
                  {"kind": "terminal", "action_id": STOP if success else NEEDS_INFORMATION,
                   "action_name": "Finish: goal verified" if success else "Stop and request more information",
                   "reason": reason, "branches": []})


def plan(problem: BeliefProblem | dict, max_depth: int = 5, token_weight: float = 1.0,
         latency_weight: float = 0.01, *, max_nodes: int = 20_000) -> dict:
    """Search a contingent plan up to five actions, under a bounded node budget.

    Results are exact within this finite horizon unless search_complete is false.
    A budget cutoff supplies a conservative zero-completion continuation; reported
    success then describes the returned partial plan, not an optimality claim.
    The objective compares floating-point values with a 1e-12 absolute tolerance.
    """
    started = time.perf_counter()
    if isinstance(problem, dict):
        problem = BeliefProblem.from_dict(problem)
    if type(max_depth) is not int or not 1 <= max_depth <= 5:
        raise ValueError("The action horizon must be an integer from 1 to 5.")
    if type(max_nodes) is not int or not 1 <= max_nodes <= 1_000_000:
        raise ValueError("The search budget must be between 1 and 1,000,000 nodes.")
    _finite_nonnegative(token_weight, "Token weight")
    _finite_nonnegative(latency_weight, "Time weight")
    if token_weight + latency_weight == 0:
        raise ValueError("At least one cost weight must be positive.")
    explored_nodes = 0
    cutoffs = 0
    shortcuts = 0
    root_scores = []
    root_effective_actions = 0

    @lru_cache(maxsize=None)
    def solve(belief, remaining):
        nonlocal explored_nodes, cutoffs, shortcuts, root_effective_actions
        # These terminal checks use no branching effort.
        if verified(belief, problem.goal):
            return _terminal(True, "Goal holds in every remaining hypothesis.")
        if remaining == 0:
            return _terminal(False, "No action horizon remains; the goal is unverified.")
        if explored_nodes >= max_nodes:
            cutoffs += 1
            return _terminal(False, "Planning node budget exhausted; the goal is unverified.")
        explored_nodes += 1
        is_root = belief == problem.belief and remaining == max_depth
        best = _terminal(False, "No verified completion found within the remaining horizon.")
        options = []
        for action in sorted(problem.actions, key=lambda a: a.id):
            if not action.eligible(belief):
                if is_root:
                    root_scores.append({"action_id": action.id, "action_name": action.name,
                                        "eligible": False, "reason": "Permission or prerequisites fail in a possible world."})
                continue
            children = branches(belief, action)
            if all(posterior == belief for _, _, posterior in children):
                # No effect and no new evidence. With nonnegative costs, this
                # cannot improve a plan and only consumes an action.
                if is_root:
                    root_scores.append({"action_id": action.id, "action_name": action.name,
                                        "eligible": True, "pruned": True,
                                        "reason": "No state change or information gain."})
                continue
            options.append((action, children))
        if is_root:
            root_effective_actions = len(options)
        for action, children in options:
            branch_rows = []
            values = []
            unique_finish = len(options) == 1 and all(verified(b, problem.goal) for _, _, b in children)
            if unique_finish:
                # With one effective option and a verified immediate finish,
                # there is no need to expand any further action level.
                shortcuts += 1
            for observation, probability, posterior in children:
                child = (_terminal(True, "Goal verified; further search is unnecessary.")
                         if unique_finish else solve(posterior, remaining - 1))
                values.append((probability, child))
                branch_rows.append({"observation": observation, "probability": probability,
                                    "posterior": [asdict(h) for h in posterior], "next": child.tree})
            def expectation(attribute):
                return math.fsum(p * getattr(value, attribute) for p, value in values)
            success = (1.0 if all(v.success == 1 for _, v in values)
                       else expectation("success"))
            candidate = _Value(success, action.tokens * token_weight + action.latency_ms * latency_weight + expectation("cost"),
                               action.tokens + expectation("tokens"), action.latency_ms + expectation("latency"),
                               1 + expectation("steps"),
                               {"kind": "action", "action_id": action.id, "action_name": action.name,
                                "reason": "Choose using the full belief, then condition the next decision on the observation.",
                                "branches": branch_rows})
            candidate.tree.update(_summary(candidate))
            if is_root:
                root_scores.append({"action_id": action.id, "action_name": action.name,
                                    "eligible": True, "pruned": False, **_summary(candidate)})
            if _better(candidate, best):
                best = candidate
        return best

    result = solve(problem.belief, max_depth)
    already_verified = verified(problem.belief, problem.goal)
    reason = ("already_verified" if already_verified else
              "node_budget_exhausted" if cutoffs else
              "only_one_effective_choice_finishes" if root_effective_actions == 1 and result.steps == 1 and result.success == 1 else
              "no_verified_path_within_horizon" if result.success == 0 else "best_contingent_plan")
    return {"problem_id": problem.id, "chosen_action_id": result.tree["action_id"],
            "chosen_action_name": result.tree["action_name"], **_summary(result),
            "plan": result.tree, "action_scores": sorted(root_scores, key=lambda a: a["action_id"]),
            "max_depth": max_depth, "explored_nodes": explored_nodes,
            "cache_hits": solve.cache_info().hits, "max_nodes": max_nodes,
            "search_complete": cutoffs == 0, "budget_cutoffs": cutoffs,
            "immediate_finish_shortcuts": shortcuts, "stop_reason": reason,
            "planning_ms": round((time.perf_counter() - started) * 1000, 3),
            "cost_weights": {"tokens": token_weight, "latency_ms": latency_weight},
            "comparison_tolerance": _TOLERANCE,
            "method": "Declared-world belief search; algorithmic reference, not the trained neural policy.",
            "limitation": "Probabilities, effects, and observations are supplied assumptions. Verified means every remaining modeled hypothesis meets the goal; no real tools or tests were executed. A horizon or budget failure does not prove impossibility."}


def example_problems() -> list[BeliefProblem]:
    """Small demonstrations selected by construction, not held-out benchmarks."""
    # A wrong repair destroys the opportunity to complete this toy task. Seeing
    # the cause first allows the policy to choose the appropriate safe repair.
    good = (Outcome(1, "tests_passed", sets=1),)
    bad = (Outcome(1, "wrong_repair", sets=2),)
    inspect = BeliefAction("inspect", "Inspect the failing component", tokens=2, latency_ms=3,
                          outcomes_by_world={"parser": (Outcome(1, "parser_fault"),),
                                             "cache": (Outcome(1, "cache_fault"),)})
    parser = BeliefAction("repair_parser", "Repair parser and verify", forbids=2, tokens=5, latency_ms=8,
                         outcomes_by_world={"parser": good, "cache": bad})
    cache = BeliefAction("repair_cache", "Repair cache and verify", forbids=2, tokens=5, latency_ms=8,
                        outcomes_by_world={"parser": bad, "cache": good})
    uncertain = BeliefProblem("inspect_when_uncertain", ("goal_verified", "irreversible_wrong_repair"), 1,
                              (Hypothesis("parser", 0, .5), Hypothesis("cache", 0, .5)), (inspect, parser, cache),
                              "The symptom has two possible causes. A wrong repair prevents success in this declared toy world.")
    known = replace(uncertain, id="act_when_known", belief=(Hypothesis("parser", 0, 1),),
                    context="Existing evidence has already identified the parser fault; another inspection adds no information.")
    ask = replace(inspect, id="ask_requirement", name="Ask which behavior the user needs", tokens=3, latency_ms=40,
                  outcomes_by_world={"parser": (Outcome(1, "preserve_format"),),
                                     "cache": (Outcome(1, "normalize_format"),)})
    ask_problem = replace(uncertain, id="ask_when_needed", actions=(ask, replace(parser, name="Preserve format and verify"),
                          replace(cache, name="Normalize format and verify")),
                          context="Two interpretations of the requirement remain possible. Ask before committing to one.")
    goal_a = BeliefProblem("before_goal_revision", ("bug_fixed", "documentation_updated"), 1,
                          (Hypothesis("known", 0, 1),),
                          (BeliefAction("fix_bug", "Fix and verify the bug", outcomes=(Outcome(1, "fixed", sets=1),), tokens=4),
                           BeliefAction("update_docs", "Update and verify documentation", outcomes=(Outcome(1, "updated", sets=2),), tokens=2)),
                          "The current requested win is a verified bug fix.")
    goal_b = replace(revise_goal(goal_a, 2), id="after_goal_revision",
                     context="The user now needs documentation only. Preserve evidence and replan toward the new goal.")
    retry = BeliefProblem("uncertain_action_outcome", ("goal_verified",), 1,
                         (Hypothesis("service", 0, 1),),
                         (BeliefAction("retry_check", "Run the flaky verification", tokens=1, latency_ms=2,
                                       outcomes=(Outcome(.75, "passed", sets=1), Outcome(.25, "failed"))),),
                         "A declared stochastic check succeeds with probability 0.75 per attempt. Failure is observed and permits retry.")
    return [uncertain, known, ask_problem, goal_a, goal_b, retry]


def examples() -> list[dict]:
    return [{"id": p.id, "title": p.id.replace("_", " ").capitalize(),
             "description": p.context, "problem": p} for p in example_problems()]
