# Winward

**Win first. Weigh consequences. Do only the necessary work.**

Winward is a local research project for choosing an AI agent's next action. Its decision objective is to reach a verifiable goal, then minimize token cost, elapsed time, and unnecessary actions. Search looks ahead up to five actions; execution should take one action, observe the result, and reconsider.

The name combines **win** and **onward**: keep moving toward the desired outcome while adapting to new evidence.

## Current state

- A **4.76-million-parameter** policy trained from random initialization on an Apple silicon Mac. No pretrained weights, Qwen adapters, or model-generated labels.
- A deterministic simulator, synthetic training pipeline, independent audit, and local comparison interface.
- 48,000 training examples. Fresh simulated audit: **291/300 goals reached within five actions (97%)**. Changed-goal probes: **52.5%** next-action agreement, a known weakness.
- A separate original Qwen text-scoring demo, described in [its guide](docs/QWEN_DEMO.md).

**This is not a 4B model, a general language model, or a production agent.** The learned policy consumes structured numeric facts and action definitions. Search labels use declared simulator effects. Permissions and finish checks are enforced rules. Exact search is currently faster than the neural policy on these tiny tasks.

[Training details and limitations](TRAINING.md) · [Saved training report](reports/v1/training.json) · [Fresh-instance audit](reports/v1/evaluation.json)

## Run locally on an Apple silicon Mac

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The dependency lock is included. The public repository contains source and evaluation reports; local environments, generated datasets, logs, and trained checkpoint files are excluded.

```sh
git clone https://github.com/dagar1994/winward.git
cd winward
uv sync --frozen

# Train our policy locally from scratch.
.venv/bin/python -m agent_training.train \
  --output runs/goalpolicy-v1 --train-cases 48000 \
  --valid-cases 2400 --test-cases 2400 --steps 5000 \
  --seed 20261001 --trajectories --select-rollout

.venv/bin/python -m agent_training.evaluate \
  --run runs/goalpolicy-v1 --fresh-seed 90000000 --fresh-cases 2400

# The original combined server also loads the separate Qwen demo.
.venv/bin/python scripts/download_model.py
.venv/bin/python scripts/server.py start
```

Open **http://127.0.0.1:8765/policy**. Stop with `.venv/bin/python scripts/server.py stop`. Model downloads require internet once; training and runtime run locally. Training refuses to overwrite existing checkpoints. No API key, paid service, or cloud GPU is required.

## Development

```sh
uv run --frozen pytest -q
node --check jev_local/static/policy.js
```

Version 1 passed 102 automated tests and live API checks. Rendered browser layout has not been visually verified. The next work targets goal changes, uncertain observations, and real sandboxed coding tasks, measuring total decision and action costs before increasing model size.

The inference experiment was inspired by a [local fixed-choice scoring tutorial](https://blog.dailydoseofds.com/p/build-your-own-jev-100-local). Winward's scratch-trained policy and simulator are a separate experiment; this repository does not contain TypeSafe's proprietary Jev model.
