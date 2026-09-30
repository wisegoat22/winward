# Winward

**Win first. Weigh consequences. Do only the necessary work.**

Winward is a local research project for choosing an AI agent's next action. Its decision objective is to reach a verifiable goal, then minimize token cost, elapsed time, and unnecessary actions. Search looks ahead up to five actions; execution should take one action, observe the result, and reconsider.

The name combines **win** and **onward**: keep moving toward the desired outcome while adapting to new evidence.

## Try your own situation

Open **http://127.0.0.1:8765/try** with the local server running. Enter a software situation and the desired win, select **Read my situation**, review the task type, facts, and constraints, then select **Get my next step**. After taking the suggested step yourself, update the situation with what actually happened and try again. Three built-in examples cover an unknown bug, a change awaiting tests, and completed work.

The optional local **Qwen 4B** model reads the description into a reviewable form. Our **Winward v4, with 4.86M parameters**, chooses from a fixed software workflow using the reviewed facts. The reader does not choose the action. This text interface requires both the downloaded Qwen model and a trained v4 checkpoint; the existing preset labs remain available separately.

Review matters: a quoted sentence can still be interpreted incorrectly. You can change the task type, correct facts, or uncheck a misread constraint. The app suggests an action without inspecting files or executing it. Requests stay on the Mac and are not saved by this interface. [How to test and interpret the result](docs/TRY_IT.md).

## Current state

- **V4 improves uncertainty decisions:** **856/960 episodes completed (89.17%)**, versus **406/960 (42.29%)** for frozen v3 and **779/960 (81.15%)** for information-first rules on the **same fresh v4 cases**. The v4 task distribution differs from the older v3 audit.
- One **4,862,721-parameter** model continued from our own v1 weights on **160,000 synthetic and observed-state examples**, entirely on an Apple silicon Mac. Two fixed training settings were compared using validation only. No external pretrained weights or language-model labels are used to train this policy. The separate text reader above uses pretrained Qwen.
- **V4 is not promoted:** five of twelve checks failed, covering retention of legacy completion/cost and changed-goal decisions. It passed **40/40 controlled coding fixtures**, matching rules that remained faster. Earlier versions stay available.

[V4 methods, complete results, and reproduction](docs/V4.md) · [Frozen v4 audit](reports/v4/evaluation.json) · [Training report](reports/v4/training.json)

**V3:** the earlier uncertainty experiment failed eight of twelve checks on its original audit, including 61.04% completion versus 92.5% for information-first rules. Its known failures informed v4 development; the new comparison evaluates both frozen models on fresh, matched v4 instances. [V3 methods and original results](docs/V3.md) remain preserved.

**V1:** the original 4.76-million-parameter policy was trained from random initialization on 48,000 examples. Its original fresh simulated audit reached **291/300 goals within five actions (97%)**; its original changed-goal probes reached **52.5%** next-action agreement. Later matched comparisons use different instances and should not be compared directly with those original percentages.

**V2 lab:** a separate uncertainty planner and a coding-task harness now run on this Mac. After the first v2 model failed most real-tool tasks, a scratch-trained **v2.1 tools candidate** learned from observed edits and check outcomes. It verified **40/40 reserved interval fixtures**; simple evidence-first rules also verified 40/40 and were faster. Fresh paired-goal decisions improved to **96.25% versus v1's 88% on matched cases**, but general graph-task completion regressed. V1 remains available unchanged; v2.1 is an experiment, not a general replacement.

[V2 methods, failures, results, and reproduction](docs/V2.md) · [V2.1 audit](reports/v2-tools/evaluation.json) · [Actual coding-task audit](reports/v2-tools/sandbox.json)

**This is not a 4B model, a general language model, or a production agent.** The learned policy consumes structured numeric facts and action definitions. Search labels use declared simulator effects. Permissions and finish checks are enforced rules. V4 does not discover its own world model or explicitly search five levels at inference. Exact search remains competitive on these tiny tasks. The separate original Qwen text-scoring demo is described in [its guide](docs/QWEN_DEMO.md).

[Training details and limitations](TRAINING.md) · [Saved training report](reports/v1/training.json) · [Fresh-instance audit](reports/v1/evaluation.json)

## Run locally on an Apple silicon Mac

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The dependency lock is included. The public repository contains source and evaluation reports; local environments, generated datasets, logs, and trained checkpoint files are excluded.

```sh
git clone https://github.com/wisegoat22/winward.git
cd winward
uv sync --frozen

# Train our policy locally from scratch.
.venv/bin/python -m agent_training.train \
  --output runs/goalpolicy-v1 --train-cases 48000 \
  --valid-cases 2400 --test-cases 2400 --steps 5000 \
  --seed 20261001 --trajectories --select-rollout

.venv/bin/python -m agent_training.evaluate \
  --run runs/goalpolicy-v1 --fresh-seed 90000000 --fresh-cases 2400

# Qwen is optional for preset labs; /try uses it to read descriptions.
.venv/bin/python scripts/server.py start
```

Open **http://127.0.0.1:8765/policy** for v1. Follow [the v2 guide](docs/V2.md#reproduce-on-mac) for the tools candidate and **http://127.0.0.1:8765/v2**. Follow [the v3 guide](docs/V3.md#reproduce-on-mac) for the earlier uncertainty candidate at **http://127.0.0.1:8765/v3**, then [the v4 guide](docs/V4.md#reproduce-on-mac) for the reliability experiment at **http://127.0.0.1:8765/v4**. Stop with `.venv/bin/python scripts/server.py stop`. Dependencies require internet to install; training and runtime run locally. Training refuses to overwrite existing checkpoints. No API key, paid service, or cloud GPU is required. The optional Qwen demo can be installed separately with `.venv/bin/python scripts/download_model.py` and loads only when requested.

## Development

```sh
uv run --frozen pytest -q
node --check jev_local/static/policy.js
node --check jev_local/static/v3.js
node --check jev_local/static/v4.js
node --check jev_local/static/situation.js
```

Tests cover planner correctness, unknown-world observations, goal revisions, failed and timed-out checks, checkpoint data boundaries, and local API restrictions. Live API and lightweight DOM checks also verify the interface logic. Rendered browser layout has not been visually verified. Current evidence supports continued work on task completion and efficiency before increasing model size.

The inference experiment was inspired by a [local fixed-choice scoring tutorial](https://blog.dailydoseofds.com/p/build-your-own-jev-100-local). Winward's scratch-trained policy and simulator are a separate experiment; this repository does not contain TypeSafe's proprietary Jev model.
