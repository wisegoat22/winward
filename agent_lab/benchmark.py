"""Matched evaluation on controlled Python tasks; never trains a checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from .predictor import NeuralPolicy
from .runner import run_episode
from .sandbox import SUPPORTED_KINDS


def summarize(episodes):
    winners = [e for e in episodes if e["success"]]
    return {"cases":len(episodes), "successes":len(winners), "success_rate":len(winners)/len(episodes),
            "verified_and_stopped":sum(e["success"] and e["done"] for e in episodes),
            "mean_actions":statistics.mean(e["steps"] for e in episodes),
            "mean_wall_ms":statistics.mean(e["wall_ms"] for e in episodes),
            "mean_decision_ms":statistics.mean(e["decision_ms"] for e in episodes),
            "mean_tool_ms":statistics.mean(e["tool_elapsed_ms"] for e in episodes),
            "mean_estimated_tool_tokens":statistics.mean(e["estimated_tokens"] for e in episodes),
            "mean_edits":statistics.mean(e["edits"] for e in episodes),
            "mean_failed_checks":statistics.mean(e["failed_checks"] for e in episodes),
            "unrelated_actions":sum(e["action_id"]=="read_notes" for episode in episodes for e in episode["trace"]),
            "rejected_actions":sum(e["rejected_actions"] for e in episodes),
            "deferred":sum(e["deferred"] for e in episodes),
            "horizon_exhausted":sum(e["horizon_exhausted"] for e in episodes)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default="runs/goalpolicy-v2")
    parser.add_argument("--seed",type=int,default=170000000)
    parser.add_argument("--seeds-per-condition",type=int,default=2)
    parser.add_argument("--kinds",nargs="+",choices=SUPPORTED_KINDS,default=list(SUPPORTED_KINDS))
    args=parser.parse_args()
    if not 1 <= args.seeds_per_condition <= 20 or args.seed<0:
        raise SystemExit("Choose 1–20 seeds per condition and a nonnegative seed")
    run=Path(args.run)
    output=run/"sandbox-audit.json"
    if output.exists():
        raise SystemExit("An audit already exists. Preserve it; use a different run for a new experiment.")
    started=time.perf_counter()
    policy=NeuralPolicy(run)
    policy.load()
    initialization_ms=(time.perf_counter()-started)*1000
    tool_trained = bool(policy.report.get("training_observed_tool_decisions"))
    if tool_trained and (args.seed != policy.report.get("real_task_test_seed_reserved") or args.kinds != ["interval"]):
        raise SystemExit("This candidate reserves the interval family and its recorded seed for the final tool audit.")
    spec=[{"seed":args.seed+index*100+repetition, "kind":kind, "changed_goal":changed,
           "uncertain":uncertain}
          for index,(kind,changed,uncertain) in enumerate((k,c,u) for k in args.kinds
            for c in (False,True) for u in (False,True))
          for repetition in range(args.seeds_per_condition)]
    all_episodes={name:[] for name in ("neural","evidence_first","random")}
    names=list(all_episodes)
    for index, task in enumerate(spec):
        # Rotate methods to spread initialization/cache/thermal timing effects.
        for name in names[index%3:]+names[:index%3]:
            all_episodes[name].append(run_episode(**task,policy_name=name,neural_policy=policy))
        if (index+1)%10==0:
            print(f"Completed {index+1}/{len(spec)} matched tasks",flush=True)
    measures={name:summarize(episodes) for name,episodes in all_episodes.items()}
    slices={}
    for name, episodes in all_episodes.items():
        slices[name]={}
        for kind in args.kinds:
            slices[name][kind]=summarize([e for e in episodes if e["kind"]==kind])
        for label,predicate in (("changed_goal",lambda e:e["changed_goal"]),
                                ("uncertain",lambda e:e["uncertain"])):
            slices[name][label]=summarize([e for e in episodes if predicate(e)])
    traces=run/"sandbox-traces.jsonl"
    with traces.open("w") as f:
        for name in names:
            for episode in all_episodes[name]: f.write(json.dumps(episode)+"\n")
    report={"checkpoint_sha256":policy.report["checkpoint_sha256"],"cases_per_method":len(spec),
            "seed":args.seed,"task_kinds":args.kinds,"specification":spec,"methods":measures,"slices":slices,
            "initialization_ms":initialization_ms,"elapsed_seconds":time.perf_counter()-started,
            "task_provenance":("Trusted generated Python fixtures, supplied patches, observed outcomes and real subprocess checks. "
                + ("This interval code family and task seeds were reserved from training/validation, which used boundary/rounding fixtures. The interval template was inspected during harness development. Numeric tool-state mechanics are shared across families. This is not an unseen-repository benchmark. No outcomes from these audited instances selected the checkpoint."
                   if tool_trained else "These fixture families were inspected during harness development; not an unseen-repository benchmark. No sandbox outcomes were used to train or select this checkpoint.")),
            "methodology":"Same seeds and conditions for all methods, fresh temporary directory per episode, rotated method order, at most16 actions and5-step neural horizon. Neural actions are never corrected by search. Evidence-first rules use eligible action IDs only, never hidden source correctness. Random includes defer. Timings include fixture creation/cleanup, decisions and tools; one-time checkpoint initialization is reported separately.",
            "cost_note":"Tool token costs are declared estimates, not measured LLM tokens. The policy generates zero language tokens. Actual wall time, decision time, subprocess time and byte counts are measured; inspect full traces for tool outcomes. Timings are indicative single-session measurements on a shared Mac, not an isolated throughput benchmark.",
            "limitations":"Small fixed code families and two supplied patches per task. Numeric observation adapter and rule-based permissions/finish verification are part of the system. This does not demonstrate code generation or arbitrary repository competence.",
            "traces_sha256":hashlib.sha256(traces.read_bytes()).hexdigest(),
            "example_traces":{name:next((e for e in episodes if not e["success"]),episodes[0])
                              for name,episodes in all_episodes.items()}}
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"methods":measures,"elapsed_seconds":report["elapsed_seconds"]},indent=2))


if __name__=="__main__":
    main()
