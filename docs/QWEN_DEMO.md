# JEV Local

Two local experiments run on your Apple silicon Mac:

- **[GoalPolicy training lab](http://127.0.0.1:8765/policy):** our own 4,759,809-parameter policy, trained from random initialization on 48,000 synthetic software-agent decisions. It ranks structured actions toward a stated goal. This is a 4.76-million-parameter research prototype, not a 4B language model. See [training, evaluation, and reproduction](../TRAINING.md).
- **[Original Qwen decision demo](http://127.0.0.1:8765/):** a pretrained Qwen3 4B model that scores answers to text questions. The remainder of this README describes that original demo.

GoalPolicy's fresh-instance audit completed **291 of 300 solvable simulated tasks within five actions**. Targeted changed-goal probes scored only **52.5%**. Both are shown in the lab; neither establishes real-agent reliability. The two experiments use separate weights, and Qwen did not supply GoalPolicy's training labels.

## Open it

1. Double-click **Start JEV.command** in this folder.
2. Wait for “JEV Local is ready”.
3. Open **http://127.0.0.1:8765**.

The server runs in the background; you can close the launcher window. Double-click **Stop JEV.command** to stop it. If the app is already running, just use the link. Startup loads approximately 2.3 GB of model weights and warms the GPU. Nothing is added to your login items; start it again after restarting the Mac.

For a fresh installation, run **setup.command** first. It uses `uv` to install Python 3.12 and the exact versions in `uv.lock`, then downloads a pinned model snapshot from Hugging Face. Internet is required for setup only. No API key, paid service, Docker, or Ollama is required.

## What's included

- An editable decision playground with 2–12 choices, relative probability bars, and review thresholds.
- A generation comparison using the same loaded model, plus a 12-ticket demonstration benchmark.
- A local JSON API and command-line example.
- An Apple GPU runtime using MLX and Qwen3-4B-Instruct-2507 quantized to 4 bits.
- Model revision `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b` and pinned Python dependencies.

Inference and the interface work offline. The server binds only to `127.0.0.1`, uses the downloaded snapshot with offline mode enabled, and does not save prompts or results. The browser loads all assets locally. API access logs are disabled. An explicitly run benchmark script saves its synthetic example results in this folder.

## API examples

Run the built-in support example:

```sh
.venv/bin/python decide.py
.venv/bin/python decide.py "I forgot my password and cannot sign in."
```

Or call the API:

```sh
curl http://127.0.0.1:8765/api/decide \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "I was charged twice for my subscription.",
    "question": "Which team should handle this ticket?",
    "choices": [
      {"name": "billing", "description": "Charges, refunds, invoices"},
      {"name": "technical", "description": "Bugs and product failures"},
      {"name": "account", "description": "Login and password problems"},
      {"name": "other", "description": "None of the listed categories"}
    ],
    "threshold": 0.8,
    "margin": 0.2
  }'
```

The response includes `choice`, `probabilities`, `confidence`, `margin`, `needs_review`, `latency_ms`, `input_tokens`, and `output_tokens` (always zero for scoring). Review is suggested when the winning probability is below `threshold` or its lead over the runner-up is below `margin`. The app makes no external routing decisions.

`POST /api/generate` accepts the same body and optional `max_tokens` (default 80; range 1–256). It returns generated text, a parsed choice or `null`, timings, token counts, and finish reason. A `length` finish reason means the output hit the cap.

`GET /api/health` reports the loaded model. `GET /api/examples` provides the demo inputs. `GET /openapi.json` provides the complete API schema. A second inference request during an active one receives HTTP 429; retry after the current request finishes. Inputs are limited to 4,096 tokens and are never silently truncated.

For the tutorial's raw scoring pattern, the app also implements:

- `POST /tokenize` with `{ "prompt": "A", "add_special_tokens": false }`.
- `POST /v1/score` with `{ "query": "Your complete prompt", "items": [""], "label_token_ids": [32,33,34], "apply_softmax": true }`.
- `GET /v1/models` lists the loaded model.

This is a small compatibility subset, not a full SGLang server. Resolve label IDs through `/tokenize` rather than assuming the example IDs. `/v1/score` uses the raw prompt as supplied, only supports a single empty item, and returns selected raw logits when `apply_softmax` is false. If you specify `model`, use `jev-local` or the full loaded model ID.

## How scoring works

Each semantic choice is mapped to one letter (A–L). The actual tokenizer verifies that these are distinct single tokens. The model's chat template formats the question and choices; the state is quoted as data. One model forward pass produces logits at the first answer position. We select only the label logits, convert to float32, apply softmax over those choices, and map them back to the original names. No answer token is sampled or generated. All inference runs on one dedicated worker so generation and scoring cannot overlap on the shared model.

Scores are relative probabilities across your supplied choices, **not calibrated accuracy**. A model can confidently choose a wrong answer, and choice ordering or wording can affect results. Include an `other` or `review` choice when the listed answers may not cover an input. Validate with your own labeled examples before connecting decisions to actions. This project recreates the tutorial's inference mechanism; it does not contain TypeSafe's proprietary Jev weights, training, or calibration.

## Validation

```sh
uv run --frozen pytest -q
.venv/bin/python scripts/benchmark.py
```

The benchmark warms both paths, uses 12 synthetic support tickets, alternates which path goes first, and compares sequential requests on the same GPU. Generation requests produce the choice plus a brief explanation, capped at 80 tokens. It reports actual measured latency and correctness and writes `benchmark-results.json`. This is a demonstration, not a representative accuracy evaluation or a reproduction of SGLang's concurrent batching benchmark. Server timing includes tokenization and inference but excludes request queue/network overhead.

Build verification on this M5 Max: both paths matched all 12 expected answers; scoring averaged 46.58 ms, generation 263.34 ms (5.65× latency ratio). Scoring generated zero tokens. The measured record is in `benchmark-results.json`. The model checksum, real API calls, offline restart, frontend syntax, and backend tests were checked. Browser automation was unavailable, so rendered layout and interactive browser flows have not been visually verified.

## Sources and adaptation

The requested [Medium URL](https://medium.com/@iamdgarcia/build-your-own-jev-100-local-56799bcf2909) was inaccessible during the build. The implementation follows the independently accessible same-title [Avi Chawla tutorial](https://blog.dailydoseofds.com/p/build-your-own-jev-100-local), which demonstrates SGLang fixed-answer scoring and compares it with generation. Its relationship to the Medium post could not be verified.

This version uses [Apple's MLX-LM](https://github.com/ml-explore/mlx-lm) directly for native Apple silicon inference. The [Qwen model snapshot](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit/tree/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b) is the quantized counterpart of the tutorial's Qwen3 4B benchmark model. SGLang's server, multi-model selector, and hosted demo are not installed.
