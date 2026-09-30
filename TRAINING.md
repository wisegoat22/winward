# Winward: our locally trained decision model

Open **http://127.0.0.1:8765/policy** after starting `Start JEV.command`.

The public source checkout excludes local `runs/` artifacts and checkpoint weights. Published v1 metrics are in [reports/v1](reports/v1/). Follow the root README to train the default serving checkpoint; paths under `runs/` below describe locally generated artifacts.

GoalPolicy v1 has **4,759,809 trainable parameters (4.76 million)**. Every weight starts randomly and is trained on this Mac using MLX. There are no pretrained weights, Qwen adapters, Qwen-generated labels, or text embeddings. The separate Qwen demo is still available at the home page.

This is the first small experiment toward the requested agent decision model. **A 4-billion-parameter model has not been trained.** GoalPolicy consumes structured facts and action descriptions expressed as numbers; it cannot understand arbitrary natural-language instructions or operate a real repository.

## Decision objective

1. Achieve the specified goal within the supplied action horizon.
2. Among successful plans, minimize `token_weight × tokens + latency_weight × milliseconds` across the entire plan.
3. Break equal-cost ties by taking fewer actions.
4. Stop immediately if the goal is already met. If no plan is found within the horizon, return `NEEDS_CLARIFICATION`.

The default cost weights are 1 per token and 0.01 per millisecond. These are adjustable preferences, not a universal conversion between time and tokens. Costs and effects are supplied by the simulation; they are not measured real-tool costs or learned predictions.

The teacher searches all applicable action sequences up to **five actions deep**, with memoization. It follows consequences and prerequisites, includes actions that undo progress, and rejects forbidden actions. It stops when the goal is reached. Failure to find a plan within five actions does not prove that the goal is impossible.

The neural policy learns to imitate the teacher's next choice. **It does not itself perform a guaranteed five-level search.** The interface displays a separate exact search reference and its consequences, so disagreements stay visible. During evaluation, the model chooses each next action from the updated state and remaining horizon, without the reference correcting it.

Names, prose, urgency, and emotional framing are excluded from model input by construction. This prevents that metadata from changing the answer; it does not teach the model to distinguish useful information from emotional wording in natural language. Relevant constraints must be encoded in the facts, goal, action permissions, prerequisites, and effects.

## Data and architecture

Each scenario supports up to 12 Boolean facts and 10 supplied actions, plus built-in finish and clarification candidates. Each action defines prerequisites, blockers, facts it sets or clears, permission, token cost, and time cost. GoalPolicy sees all actions, including future actions whose prerequisites are not yet met.

The six-layer set transformer has width 256, eight attention heads, and 103 features per candidate. It has no positional encoding, so action order should not determine its answer. Permission, current-precondition, and already-complete-goal masks are enforced by code. Perfect results on those constraints must not be credited as learned capabilities. Its softmax preferences are not calibrated probabilities of winning.

| Dataset | Cases | Construction |
| --- | ---: | --- |
| Training | 48,000 | Chains, forks, shortcuts, shared prerequisites; randomized facts, costs, goals, progress states, and remaining horizons |
| Validation | 2,400 | Two-stage forks and blocked shortcuts; used to select the checkpoint |
| Original test | 2,400 | Nested dependencies, changed goals, reversible traps |
| Fresh audit | 2,400 | New instances from the test families, seed 90000000 |

The run used 5,000 optimizer steps, batch size 128, AdamW, a warmup and decaying learning rate, and clipped gradients. The saved checkpoint is from **step 2,250**, selected by validation goal completion on 150 tasks, then validation next-action accuracy. The final step is not automatically the best model. Full training settings, selection history, data, and weights are saved under `runs/goalpolicy-v1/`.

The earlier v0 experiment is preserved under `runs/goalpolicy-v0/`. It achieved only 53/300 completed rollouts. That audit led to better coverage of intermediate states and remaining horizons, plus an explicit feature for future-action permission. No clean causal attribution to an individual change is possible from these two runs.

## Independent audit of the saved v1 checkpoint

Primary report: `runs/goalpolicy-v1/test-fresh-90000000-evaluation.json`.

| Check | Result |
| --- | ---: |
| Next-action agreement with the optimal reference, 2,400 cases | 87.83% |
| Agreement on 1,961 cases with more than one eligible candidate | 85.11% |
| Completed goals, 300 initially unfinished tasks solvable within five actions | **291/300 (97.0%)** |
| Same completion test, immediate-progress greedy baseline | 56/300 (18.67%) |
| Same completion test, random eligible baseline | 23/300 (7.67%) |
| Mean cost relative to optimal, successful model rollouts only | 1.0215× |
| Targeted changed-goal probes, 20 pairs / 40 decisions | **52.5%; both answers correct in 5/20 pairs** |
| Targeted downstream-cost probes, 20 pairs / 40 decisions | 100%; both answers correct in 20/20 pairs |

Single-action agreement and goal completion are different measurements on different subsets. A non-optimal action can still lead to success. The cost statistic excludes failures, so it must be read alongside completion rate. Nine rollouts deferred; eight still had a feasible plan. Zero invalid actions and premature finishes reflect the enforced masks.

All 20 action-reordering probes preserved the chosen action. Scores differed by up to 0.00185, so a strict 0.0001 numerical equality check failed. Metadata changes produced identical features and scores, as designed.

The fresh audit's task families were already examined during v0 development. The paired counterfactuals are reusable diagnostics, not a pristine benchmark. The original v1 test shares 2,329 scenario IDs with the earlier v0 audit and is retained as a secondary record; the interface uses the fresh audit. These numbers demonstrate performance in this deterministic simulator, not generalization to unfamiliar task families, real software repositories, or all decisions.

On the five live examples, learned inference took about 2.6–9.1 ms and reference search 0.02–0.20 ms. **Exact search is currently faster in this tiny simulator.** These are individual local measurements, not a throughput benchmark. The learned policy has not demonstrated a speed advantage; a useful larger system must count its own decision overhead as well as downstream action costs.

Build verification passed 102 automated tests, frontend syntax checks, live inference on all five examples, and an edited-state stop check. The example for an outdated goal visibly exposes a model/reference disagreement. Browser automation was unavailable due to administrator policy, so the rendered page has not been visually verified.

## Reproduce a run

From this folder, using the installed environment:

```sh
.venv/bin/python -m agent_training.train \
  --output runs/my-policy \
  --train-cases 48000 --valid-cases 2400 --test-cases 2400 \
  --steps 5000 --seed 20261001 --trajectories --select-rollout

.venv/bin/python -m agent_training.evaluate \
  --run runs/my-policy --fresh-seed 90000000 --fresh-cases 2400

.venv/bin/pytest -q
```

Choose an unused output directory; training refuses to overwrite an existing checkpoint. The saved seed reproduces data generation, but exact floating-point weights can depend on hardware and MLX version. Reusing the published audit is reproduction, not a new blind test. Use new instances and reserved task families for future model development.

The source snapshot and dependency lock for v1 are under `runs/goalpolicy-v1/source/`; `provenance.json` records settings, environment, and checksums. Active serving remains pinned to the audited v1 run in `agent_training/runtime.py`. A new training run does not silently replace it.

## Local API

- `GET /api/policy/status`: checkpoint origin, parameter count, training details, audit and limitations.
- `GET /api/policy/examples`: five fixtures selected using reference-plan properties, not the model's success.
- `POST /api/policy/decide`: `{ "scenario": ..., "token_weight": 1, "latency_weight": 0.01, "max_depth": 5 }`.

Use a scenario returned by the examples endpoint. The response separates the learned choice from the search reference. The API validates masks, costs, candidate IDs, and horizon, and never executes tools. The server binds to localhost. Training and inference require no network calls or paid APIs once dependencies are installed.

## What would make this a useful 4B model?

The current Mac has an M5 Max and 48 GiB of unified memory. A 4B model with float32 weights, gradients, and two Adam moment buffers alone needs approximately 64 GB, before activations, temporary buffers, and the operating system. The current trainer cannot safely be scaled to 4B on this machine just by changing its width. Memory-saving training would need a separately measured design and runtime budget.

More parameters also do not supply language understanding or reliable action effects. The useful next milestones are:

1. Fix goal-change failures and reserve genuinely new task families for blind evaluation.
2. Add uncertainty, failed actions, information-gathering choices, and measured costs. Let the planner spend more effort only when the likely benefit justifies it.
3. Collect real outcomes from sandboxed software tasks and verify wins with tests and task-specific acceptance checks.
4. Add a trained interface from actual agent observations into the policy's state. A from-scratch language encoder would need its own data and training; the existing Qwen demo is not that training.
5. Scale model size only when data and learning curves show a benefit. Establish memory and throughput at intermediate sizes before attempting approximately 4B parameters on Mac.

The present lab supplies the simulator, training loop, checkpoint, evaluation, and local comparison interface for that work. It makes no claim of having completed a general 4B agent model.
