"""Fresh public-observation curriculum for scaling our own structured policy.

The unchanged v4 encoders expose 12 candidate slots and 505 public features.
Teacher labels, values, family names and seeds are supervision/metadata only.
No previously saved training rows or final-audit instances are read. Hard
permission/eligibility masks are supplied rules, not learned safety guarantees.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from functools import lru_cache
import hashlib
import math
import time

import mlx.core as mx
import numpy as np

from agent_lab.belief import (BeliefProblem, NEEDS_INFORMATION, branches, plan,
                              verified)
from agent_training import curriculum
from agent_training.belief_data import target_ids
from agent_training.belief_data_v4 import generate_rows as generate_belief_rows
from agent_training.features import MAX_CANDIDATES, target_for
from agent_training.model_v4 import FEATURE_DIM, encode_graph, encode_uncertainty
from agent_training.simulator import NEEDS_CLARIFICATION, STOP, Scenario, search

VERSION = "winward-structured-scale-corpus-1"
SPLITS = ("train", "development", "final_audit")
MIX = ("graph", "belief", "graph", "graph", "belief", "graph", "graph", "belief", "graph", "graph")
SPLIT_MAP = {"train": "train", "development": "validation", "final_audit": "test"}
LIMITATIONS = [
    "Structured synthetic public graphs and beliefs; no free-text understanding or real tool execution.",
    "The 505 inputs describe supplied effects/probabilities, at most 10 actions, 12 facts and 4 live hypotheses.",
    "Eligibility and permission masks are hard supplied rules; enforcing them is not learned permission reasoning.",
    "Graph development uses reserved compositions; belief anchors are shared with independent fresh seeds.",
    "Exact labels optimize verified goal completion before declared token/time cost and necessary action count.",
    "No model promotion follows automatically from parameter count, next-action accuracy, or this evaluator.",
]


def domain_seed(seed: int, split: str, domain: str) -> int:
    if type(seed) is not int or seed < 0 or split not in SPLITS or domain not in ("graph", "belief"):
        raise ValueError("Use a nonnegative integer seed, declared split and graph/belief domain")
    hashed = hashlib.sha256(f"{VERSION}|{seed}".encode()).digest()
    # Disjoint numerical regions for every split/domain; all are far above the
    # historical v1–v4 seed ranges. Small offsets inside the frozen generators
    # cannot cross a region boundary.
    return ((SPLITS.index(split) + 1) << 61) + ((domain == "belief") << 59) + int.from_bytes(hashed[:6], "big")


def encode_observation(row):
    """Only public mechanics/weights/horizon are read; no target metadata."""
    tw, lw, depth = row["token_weight"], row["latency_weight"], row["max_depth"]
    if row["domain"] == "graph":
        return encode_graph(row["scenario"], tw, lw, depth)
    if row["domain"] == "belief":
        return encode_uncertainty(row["problem"], depth, tw, lw)
    raise ValueError("Unknown structured domain")


def _cache_row(row):
    x, valid, eligible, candidates = encode_observation(row)
    target = target_for(candidates, {"optimal_action_ids": row["target_ids"]})
    if x.shape != (12, 505) or not np.all(eligible[target > 0]):
        raise ValueError("Teacher targets must align with eligible public candidate slots")
    row = dict(row)
    row["candidate_ids"] = [c["id"] for c in candidates]
    row["_encoded"] = (x, valid, eligible, target)
    for value in row["_encoded"]:
        value.setflags(write=False)
    return row


def row_from_graph(scenario, *, depth=5, token_weight=1.0, latency_weight=.01, row_id="manual"):
    result = search(scenario, max_depth=depth, token_weight=token_weight, latency_weight=latency_weight)
    return _cache_row({"domain": "graph", "scenario": scenario.to_dict(), "max_depth": depth,
                      "token_weight": token_weight, "latency_weight": latency_weight,
                      "target_ids": result["optimal_action_ids"], "family": scenario.family,
                      "row_id": row_id, "split": "manual", "teacher": result})


def row_from_belief(problem, *, depth=5, token_weight=1.0, latency_weight=.01, row_id="manual"):
    result = plan(problem, depth, token_weight, latency_weight, max_nodes=100_000)
    if not result["search_complete"]:
        raise RuntimeError("Incomplete reference search cannot supply labels")
    return _cache_row({"domain": "belief", "problem": problem.to_dict(), "max_depth": depth,
                      "token_weight": token_weight, "latency_weight": latency_weight,
                      "target_ids": target_ids(result), "family": "manual",
                      "row_id": row_id, "split": "manual", "teacher": result})


def corpus(count: int, split: str, seed: int, *, allow_final: bool = False):
    """Return rows with runtime NumPy caches; strip `_encoded` to serialize.

    `final_audit` requires explicit authorization by an external sealed audit
    runner. This function does not seal models, select winners, or reuse audits.
    The interleaved prefix is stable when count grows: 70% graph, 30% belief.
    Graphs include paired changed goals and swapped downstream costs. Beliefs
    include diagnosis, known causes, preparation, noisy outcomes and recovery.
    """
    if type(count) is not int or not 1 <= count <= 100_000:
        raise ValueError("count must be a positive integer no greater than 100000")
    graph_seed, belief_seed = domain_seed(seed, split, "graph"), domain_seed(seed, split, "belief")
    if split == "final_audit" and not allow_final:
        raise ValueError("Final instances require an explicitly authorized sealed audit runner")
    kinds = [MIX[i % len(MIX)] for i in range(count)]
    counts = Counter(kinds)
    graph_rows = curriculum.generate_rows(counts["graph"], SPLIT_MAP[split], graph_seed) if counts["graph"] else []
    belief_rows = (generate_belief_rows(counts["belief"], SPLIT_MAP[split], belief_seed)[0]
                   if counts["belief"] else [])
    iterators = {"graph": iter(graph_rows), "belief": iter(belief_rows)}
    rows = []
    for index, kind in enumerate(kinds):
        row = dict(next(iterators[kind]))
        row.update(domain=kind, split=split, generator_split=SPLIT_MAP[split],
                   corpus_seed=seed, corpus_version=VERSION,
                   row_id=f"{VERSION}/{split}/{seed}/{index}")
        if kind == "graph":
            row["family"] = row["scenario"]["family"]
        rows.append(_cache_row(row))
    return rows


def batch_arrays(rows):
    if not rows:
        raise ValueError("A batch cannot be empty")
    encoded = [row["_encoded"] if "_encoded" in row else _cache_row(row)["_encoded"] for row in rows]
    return tuple(np.stack([item[i] for item in encoded]) for i in range(4))


def _model_scores(model, public_arrays):
    # Exactly these three arrays are available to the forward pass. Targets,
    # teacher values, row IDs and candidate names never enter it.
    started = time.perf_counter()
    scores = model(*[mx.array(value) for value in public_arrays])
    if isinstance(scores, mx.array):
        scores = scores.astype(mx.float32)
        mx.eval(scores)
    scores = np.asarray(scores)
    eligible = public_arrays[2]
    if scores.shape != eligible.shape:
        raise ValueError("Model must return one score for each of 12 candidate slots")
    if np.isnan(scores).any() or np.isposinf(scores).any() or not np.isfinite(scores[eligible]).all():
        raise FloatingPointError("Nonfinite candidate logits; no fallback action is supplied")
    return scores, (time.perf_counter() - started) * 1000


def _summarize_decisions(items):
    count = len(items)
    predictions, expected = Counter(), Counter()
    for item in items:
        predictions[item["action_id"]] += 1
        for action in item["target_ids"]:
            expected[action] += 1 / len(item["target_ids"])
    return {"examples": count, "next_action_correct": sum(i["correct"] for i in items),
            "next_action_accuracy": sum(i["correct"] for i in items) / count if count else None,
            "invalid_or_ineligible_count": sum(not i["eligible"] for i in items),
            "prediction_counts": dict(predictions), "expected_counts": dict(expected)}


def evaluate(model, rows, batch_size=4):
    """Fast next-action development measure; no planner calls or rollouts."""
    if not rows or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Evaluation requires rows and a positive batch size")
    was_training = getattr(model, "training", None)
    if hasattr(model, "eval"):
        model.eval()
    started, inference_ms, decisions = time.perf_counter(), 0.0, []
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            x, valid, eligible, targets = batch_arrays(batch)
            scores, elapsed = _model_scores(model, (x, valid, eligible))
            inference_ms += elapsed
            # No evaluator masking or correction: inspect the model's raw argmax.
            for row, index, correct_targets, allowed in zip(batch, scores.argmax(axis=1), targets, eligible):
                index = int(index)
                candidates = row.get("candidate_ids") or [c["id"] for c in encode_observation(row)[3]]
                action_id = candidates[index] if index < len(candidates) else f"INVALID_PADDING_{index}"
                target_ids = [candidates[i] for i in np.flatnonzero(correct_targets > 0)]
                decisions.append({"row_id": row["row_id"], "domain": row["domain"], "family": row["family"],
                                  "action_id": action_id, "candidate_index": index,
                                  "correct": bool(correct_targets[index] > 0), "eligible": bool(allowed[index]),
                                  "target_ids": target_ids})
    finally:
        if was_training is True and hasattr(model, "train"):
            model.train()
    result = _summarize_decisions(decisions)
    result.update(by_domain={kind: _summarize_decisions([d for d in decisions if d["domain"] == kind])
                            for kind in sorted({d["domain"] for d in decisions})},
                  by_family={family: _summarize_decisions([d for d in decisions if d["family"] == family])
                             for family in sorted({d["family"] for d in decisions})},
                  decisions=decisions, wall_seconds=time.perf_counter() - started,
                  inference_ms=inference_ms, batch_size=batch_size,
                  amortized_inference_ms_per_example=inference_ms / len(rows),
                  scope="Development next-action agreement with exact optimal ties; no rollout or promotion claim",
                  promotion=False, limitations=LIMITATIONS)
    return result


def _choose(model, kind, problem, depth, tw, lw):
    encoded = (encode_graph(problem, tw, lw, depth) if kind == "graph"
               else encode_uncertainty(problem, depth, tw, lw))
    scores, elapsed = _model_scores(model, tuple(value[None] for value in encoded[:3]))
    index = int(scores[0].argmax())
    candidates = encoded[3]
    action_id = candidates[index]["id"] if index < len(candidates) else f"INVALID_PADDING_{index}"
    return action_id, bool(encoded[2][index]), elapsed


def _graph_episode(model, row):
    current = Scenario.from_dict(row["scenario"])
    tw, lw, depth = row["token_weight"], row["latency_weight"], row["max_depth"]
    initially_done = current.goal_met()
    tokens = latency = decision_ms = 0.0
    trajectory = []
    failure = None
    for step in range(depth):
        if current.goal_met():
            break
        action_id, eligible, elapsed = _choose(model, "graph", current, depth - step, tw, lw)
        decision_ms += elapsed
        trajectory.append(action_id)
        if not eligible:
            failure = "invalid_or_ineligible"
            break
        if action_id in (STOP, NEEDS_CLARIFICATION):
            failure = "premature_stop" if action_id == STOP else "deferred"
            break
        action = next((a for a in current.actions if a.id == action_id), None)
        if action is None or not action.eligible(current.state):
            failure = "invalid_or_ineligible"
            break
        tokens += action.tokens
        latency += action.latency_ms
        current = replace(current, state=action.apply(current.state))
    return {"initially_verified": initially_done, "expected_verified_success": float(current.goal_met()),
            "expected_declared_tokens": tokens, "expected_declared_action_ms": latency,
            "expected_declared_cost": tw * tokens + lw * latency,
            "expected_measured_decision_ms": decision_ms, "unique_policy_decisions": len(trajectory),
            "trajectory": trajectory, "failure": failure}


def _belief_episode(model, row, max_nodes):
    problem = BeliefProblem.from_dict(row["problem"])
    tw, lw, depth = row["token_weight"], row["latency_weight"], row["max_depth"]
    decisions = []
    @lru_cache(None)
    def visit(belief, remaining):
        if verified(belief, problem.goal):
            return (1., 0., 0., 0., 0.)
        if remaining == 0:
            return (0., 0., 0., 0., 0.)
        if len(decisions) >= max_nodes:
            raise RuntimeError("Model-only belief rollout exceeded its declared node budget; no partial score")
        current = replace(problem, belief=belief)
        action_id, eligible, elapsed = _choose(model, "belief", current, remaining, tw, lw)
        decisions.append({"action_id": action_id, "eligible": eligible, "remaining": remaining})
        action = next((a for a in current.actions if a.id == action_id), None)
        if not eligible or action is None or not action.eligible(belief):
            return (0., 0., 0., elapsed, 0.)
        children = [(probability, visit(posterior, remaining - 1))
                    for _, probability, posterior in branches(belief, action)]
        expectation = lambda i: math.fsum(probability * values[i] for probability, values in children)
        return (expectation(0), action.tokens + expectation(1), action.latency_ms + expectation(2),
                elapsed + expectation(3), 1. + expectation(4))
    success, tokens, latency, decision_ms, steps = visit(problem.belief, depth)
    return {"initially_verified": verified(problem.belief, problem.goal),
            "expected_verified_success": success, "expected_declared_tokens": tokens,
            "expected_declared_action_ms": latency, "expected_declared_cost": tw * tokens + lw * latency,
            "expected_measured_decision_ms": decision_ms, "expected_steps": steps,
            "unique_policy_decisions": len(decisions), "decisions": decisions}


def rollout_evaluate(model, rows, *, parent_model=None, count=16, max_belief_nodes=256):
    """Separate bounded quality evaluation, optionally with an already loaded v4.

    Models supply all actions at every reached public state. Only after their
    trajectories are complete is exact search used to compute reference ceilings
    and costs. Belief outcomes are fully integrated, never corrected by an oracle
    or a sampled persistent hidden world. No checkpoints are loaded here.
    """
    if type(count) is not int or not 1 <= count <= min(len(rows), 128):
        raise ValueError("Choose 1–min(rows,128) rollout cases")
    if type(max_belief_nodes) is not int or not 1 <= max_belief_nodes <= 4096:
        raise ValueError("Belief node budget must be between 1 and 4096")
    selected = rows[:count]
    models = {"candidate": model}
    if parent_model is not None:
        models["v4_parent"] = parent_model
    episodes = {}
    for name, policy in models.items():
        was_training = getattr(policy, "training", None)
        if hasattr(policy, "eval"):
            policy.eval()
        try:
            episodes[name] = [_graph_episode(policy, row) if row["domain"] == "graph"
                              else _belief_episode(policy, row, max_belief_nodes) for row in selected]
        finally:
            if was_training is True and hasattr(policy, "train"):
                policy.train()
    references = []
    for row in selected:
        tw, lw, depth = row["token_weight"], row["latency_weight"], row["max_depth"]
        if row["domain"] == "graph":
            reference = search(Scenario.from_dict(row["scenario"]), depth, tw, lw)
            references.append({"expected_verified_success": float(reference["success"]),
                               "expected_declared_cost": reference["total_cost"]})
        else:
            reference = plan(BeliefProblem.from_dict(row["problem"]), depth, tw, lw, max_nodes=100_000)
            if not reference["search_complete"]:
                raise RuntimeError("Reference search incomplete; no exact comparison reported")
            references.append({"expected_verified_success": reference["expected_verified_success"],
                               "expected_declared_cost": reference["expected_cost"]})
    def summarize(indices, values):
        unfinished = [i for i in indices if not values[i]["initially_verified"]]
        reachable = [i for i in unfinished if references[i]["expected_verified_success"] > 0]
        mean = lambda field, ids: float(np.mean([values[i][field] for i in ids])) if ids else None
        matched = [i for i in reachable if math.isclose(values[i]["expected_verified_success"],
                                                       references[i]["expected_verified_success"], abs_tol=1e-9)]
        return {"cases": len(indices), "initially_verified": len(indices) - len(unfinished),
                "unfinished_positive_reference_success": len(reachable),
                "unfinished_zero_reference_success": len(unfinished) - len(reachable),
                "expected_verified_success_all": mean("expected_verified_success", indices),
                "expected_verified_success_unfinished_reachable": mean("expected_verified_success", reachable),
                "reference_success_unfinished_reachable": float(np.mean([references[i]["expected_verified_success"] for i in reachable])) if reachable else None,
                "mean_declared_cost_all_attempts": mean("expected_declared_cost", indices),
                "mean_measured_decision_ms": mean("expected_measured_decision_ms", indices),
                "matched_reference_success_cases": len(matched),
                "mean_cost_regret_at_matched_success": float(np.mean([values[i]["expected_declared_cost"] - references[i]["expected_declared_cost"] for i in matched])) if matched else None}
    results = {name: {"overall": summarize(list(range(count)), values),
                      "by_domain": {kind: summarize([i for i, r in enumerate(selected) if r["domain"] == kind], values)
                                    for kind in sorted({r["domain"] for r in selected})},
                      "episodes": values} for name, values in episodes.items()}
    return {"promotion": False, "methods": results, "references": references,
            "row_ids": [row["row_id"] for row in selected], "max_belief_nodes": max_belief_nodes,
            "scope": "Declared-state model-only rollouts; exact reference scored afterwards; no real tools",
            "limitations": LIMITATIONS}
