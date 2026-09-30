# Our trained 4.25B decision policy and the separate byte-model track

[Local scaling progress](http://127.0.0.1:8765/scale) separates two experiments: our completed **4,251,056,641-parameter structured policy** and the separate 32M byte-workflow pilot. Both use our own weights. Neither borrows Qwen weights, adapters, or language-model-generated labels. The preserved v1–v4 checkpoints remain unchanged. The Try page defaults to v4 and now offers the audited local 4.25B checkpoint at [Try 4B](http://127.0.0.1:8765/try?model=4b). The completed fresh audit did **not** establish a quality improvement; [the model card](MODEL_CARD_4B.md) describes its scope.

The dashboard reads `protocol.json`, `progress.json`, `report.json`, and `latest.json` from local `runs/scale-*` directories. It reports plans, initialization, optimizer updates, saved checkpoints, validation, memory, speed, and report age. It does not load a model or check whether a process is alive. A planned size is not a completed model, and a checkpoint is not a quality promotion.

## Track 1: grow the trained structured policy

This path starts from our own trained v4 policy, which has **4,862,721 parameters**. Width growth maps its learned weights into a larger, independently trainable dense policy. The mapping aims to preserve the original function before further training. Numerical precision can still change outputs, so retention must be measured after expansion.

| Role | Width | Layers | Heads | Features per candidate | Exact parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing v4 source | 256 | 6 | 8 | 505 | 4,862,721 |
| First growth stage | 512 | 6 | 16 | 505 | 19,162,625 |
| Intermediate stage | 1,536 | 6 | 48 | 505 | 170,734,081 |
| Large stage; training and fresh audit completed | 7,680 | 6 | 240 | 505 | 4,251,056,641 |

The feed-forward expansion is four at every stage. For width `d`, layer count `L`, input size `i`, and expansion `e`, the unique trainable parameter count is `L × (4d² + 2ed² + ed + 5d) + (i + 4)d + 1`. With the settings above this becomes `72d² + 563d + 1`. Counting the architecture does not initialize it: a new stage requires initialization and update evidence in its run reports.

Growing width adds capacity, not knowledge or planning depth. It does not repair v4’s known failed checks or turn numeric features into natural-language understanding. The input remains the existing structured candidate representation. A bigger candidate must retain earlier skills, complete tasks, and justify its additional decision cost. These protocols explicitly declare `track: "structured"` and their width, layers, heads, expansion, and input dimension.

### Completed 4.25B training and final audit

The local run `scale-structured-4b-probe01` completed **8 optimizer updates** and saved an actual **4,251,056,641-parameter** checkpoint. Measured peak MLX allocation was **21.1126 GiB**. Its development validation agreed with the supplied teacher on **244/256 examples (95.31%)**, compared with 243/256 immediately after expansion. This is a capacity and update check, not a final quality audit or proof that the larger model is better.

A separate run, `scale-structured-4b-train01`, completed **128 updates** with batch size 32 in **538.4 seconds**, with **21.113 GiB peak MLX allocation**. Across both large-stage runs, 136 updates were executed. Development validation selected main-run **step 96**, whose lineage contains 8 probe + 96 main = **104 updates at 4.25B size**. Its metadata's 304 cumulative updates also include the selected 100 updates at each smaller growth stage. Earlier ancestral v1–v4 training is recorded separately.

The selected checkpoint is `runs/scale-structured-4b-train01/checkpoints/step-00000096/model.safetensors`. Compared with the probe checkpoint used for identity initialization of the main run, **2,705,102,694 stored values changed (63.63%)**, across all 38 matrix tensors and 77 of 78 total tensors. This is evidence of real training; it does not mean every scalar changed or that the changes improved decisions. [Full stored-weight comparison](../reports/scale/4b-weight-change-proof.json).

After development selection, the model, original v4 baseline, dependencies, settings, and fresh seed were sealed before generating **512 final cases and 40 model-only rollouts**. The larger policy agreed with an optimal next action on **476/512 (92.97%)**, versus **479/512 (93.55%)** for v4. Mean expected completion was **96.748% for both** on the 31 initially unfinished, reference-reachable rollout cases; the exact reference achieved 100%. Five already-verified cases and four cases with zero reference success were counted separately. **Four of seven narrow retention checks passed; the model is not promoted.** Measured warmed batch-8 inference was **5.59 ms versus 0.127 ms per case**, amortized, on one Mac in fixed candidate-first order. This excludes model loading, warmup, and the separate Qwen reader. [Final summary](../reports/scale/4b-final-audit-summary.json) · [Development stages](../reports/scale/structured-stages.json).

The optional Try-page runtime checks the selected model’s exact architecture, training-step record, local paths, and checksum before loading its weights. V4 remains the default. Missing or inconsistent selections fail without substituting another model. Training and UI inference share a local compute lock; requests report that the Mac is busy when training holds it. The UI recommends a next action and executes no agent tools.

## Track 2: learn the byte-workflow language from scratch

This separate decoder starts from random weights or a recorded checkpoint trained here. Its planned ladder is **31,601,152 → 135,558,144 → 503,778,816 → 992,577,536 → 4,027,579,392 parameters**. Its approximately 4.03B target is a different architecture from the structured policy’s approximately 4.25B target.

The tokenizer is our fixed UTF-8 byte vocabulary with four control tokens. The decoder uses causal attention, shared input/output embeddings, and activation recomputation to save memory. All counted parameters participate in prediction; unused arrays are not added to reach a target size. Capacity probes, learning pilots, and longer training runs are reported separately.

### What the first byte curricula teach

The initial task is deliberately narrow: read a compact software-workflow description containing observed facts, a goal, allowed actions, prerequisites, effects, token costs, and time costs; choose one next action. An exact reference searches up to five steps through the **supplied** effects, prioritizing the win, then cost, then fewer actions. Training imitates this reference. No hidden state or reference answer is part of the input.

This is a small workflow language, not arbitrary English. It does not provide general language pretraining or a learned model of the real world. Byte tokenization also uses longer sequences than a learned subword vocabulary. The new model must not be advertised as replacing the current Qwen text reader on the strength of this curriculum.

The newer primitive curriculum teaches the foundations first: goal already met, immediate goal-action matching, required or blocked facts, permissions, and immediate token/time cost comparisons. These examples use a one-action horizon and approximately 214–234 byte tokens. Candidate labels and row order are randomized; the exact teacher uses only information serialized in the prompt.

Training and validation use distinct seed domains. Structural validation and structural test use different held-out task families. **Final independent holdouts for the byte track have not been run.** Validation is used during development and checkpoint selection; it is not a final audit. A future byte audit must freeze the selected checkpoint and reserve fresh cases before viewing them, include complete model-driven rollouts and changed goals, compare simple rules/search and smaller models, and count decision overhead. A lower training loss or larger parameter count is insufficient.

## Memory and saved evidence

The development Mac has 48 GiB unified memory. MLX reports a recommended GPU working set of approximately 37.44 GiB. The planned 4.03B byte decoder's BF16 matrices alone require about 7.5 GiB; gradients, updates, activations, and other applications also need memory. The training process uses factored FP32 optimizer statistics instead of two full-size Adam moment arrays. Its requested memory guideline is at most 28 GiB, and measured overages stop the run for inspection. MLX's memory setting is advisory, not a hard system RAM cap.

Runs record their protocol, dependency hashes, exact source snapshots, progress, validation measurements, and checkpoint evidence under an unused `runs/scale-*` directory. Byte-run resume checks the declared arguments and source hashes. Continuing after a code or configuration change starts a new declared run with explicit checkpoint lineage. Checkpoints and datasets stay local and are excluded from Git; source and concise reports can be published. The selected structured model stores about **8.5 GB (7.919 GiB)** of weights, mostly BF16 with 199,680 FP32 normalization parameters.

The first 32M pilot completed 400 updates in about 27 seconds, with a measured peak of 0.89 GiB. Its validation accuracy was **28.13%**, exactly the majority-answer baseline, so it did not demonstrate learned decision skill. The first-update sample check also found unchanged BF16 normalization weights. Those findings prompted a precision correction and a longer learning pilot. This failed pilot is preserved under `runs/scale-32m-pilot01`; its loss reduction is not presented as a successful decision model.

## Completed 32M development experiments

| Local run | What was tested | Recorded result |
| --- | --- | --- |
| `scale-32m-pilot02` | Longer 3,000-update workflow pilot | 23.83% final validation versus 24.61% majority baseline; best observed validation was 27.34%. |
| `scale-32m-fit01` | Can the setup memorize 32 examples? | 32/32 training examples fitted, but only 11/64 separate validation examples correct: **17.19%**. Memorization did not establish generalization. |
| `scale-32m-primitives01` | Learn simpler one-step foundations | Best development checkpoint: **180/256, 70.31%**, at update 1,200. Final update 1,500: 174/256, **67.97%**. Majority baseline: 75/256, **29.30%**. |

These are different experiments and curricula, so their percentages are not a shared benchmark. The primitive result demonstrates learning on that narrow development distribution. It does not establish multi-step competence, arbitrary language understanding, or the usefulness of a 4B model. The memorization and primitive experiments remain separate from the frozen v4 audit.

## Reproduce the structured growth procedure

Start with the pinned environment and our existing local `runs/goalpolicy-v4` checkpoint. [V4 reproduction](V4.md#reproduce-on-mac) describes its earlier training. The public repository does not ship trained weights. These commands mirror the recorded stage sizes, data settings, and checkpoint choices in new directories; they do not promise identical weights. Earlier stages used the optimizer implementation captured in their own source snapshots, while current code uses sequential updates with the same factored update rule. Floating-point execution and later code changes can affect results; each historical run's `protocol.json` and `source/` are the exact local record.

```sh
.venv/bin/python -m winward_scale.structured_train \
  --output runs/scale-replay-19m --init-from runs/goalpolicy-v4 --factor 2 \
  --steps 500 --batch-size 32 --train-cases 4096 --valid-cases 256 \
  --learning-rate 0.00003 --eval-every 100 --save-every 100 --seed 710000001

.venv/bin/python -m winward_scale.structured_train \
  --output runs/scale-replay-171m \
  --init-from runs/scale-replay-19m/checkpoints/step-00000100 --factor 3 \
  --steps 400 --batch-size 32 --train-cases 8192 --valid-cases 512 \
  --learning-rate 0.00001 --eval-every 100 --save-every 100 --seed 710000002

.venv/bin/python -m winward_scale.structured_train \
  --output runs/scale-replay-4b-probe \
  --init-from runs/scale-replay-171m/checkpoints/step-00000100 --factor 5 \
  --purpose capacity_probe --steps 8 --batch-size 4 \
  --train-cases 4096 --valid-cases 256 --validation-batch-size 4 \
  --learning-rate 0.000002 --eval-every 8 --save-every 8 --log-every 1 \
  --seed 710000003

.venv/bin/python -m winward_scale.structured_train \
  --output runs/scale-replay-4b-main --init-from runs/scale-replay-4b-probe \
  --factor 1 --continue-optimizer --steps 128 --batch-size 32 \
  --train-cases 8192 --valid-cases 512 --validation-batch-size 8 \
  --learning-rate 0.000002 --eval-every 32 --save-every 32 --log-every 8 \
  --seed 710000004
```

Each output must be unused. Historical development selection chose steps 100, 100, 8, and 96; selection for a new experiment must use its own development results. A parameter count or completed command is not a quality gate. The historical final audit lives locally at `runs/scale-structured-4b-train01/final-audit.json`; its public summary preserves the result and hashes. Its seed **810000001 is consumed** and must not be reused as a fresh test.

For a new frozen candidate, declare the complete audit before generating final cases, then execute it once with identical settings. The example below mirrors the historical step choice but uses a new output and seed. Verify that the seed is unused. The audit requires the exact original v4 baseline hash recorded by the runner; a newly retrained v4 checkpoint cannot silently substitute for it.

```sh
.venv/bin/python -m winward_scale.structured_audit \
  --run runs/scale-replay-4b-main/checkpoints/step-00000096 \
  --output runs/scale-replay-4b-main/final-audit.json \
  --cases 512 --rollouts 40 --batch-size 8 --seed 810000002 --declare

.venv/bin/python -m winward_scale.structured_audit \
  --run runs/scale-replay-4b-main/checkpoints/step-00000096 \
  --output runs/scale-replay-4b-main/final-audit.json \
  --cases 512 --rollouts 40 --batch-size 8 --seed 810000002 --run-audit
```

The audit seals sources and weights, reserves a seed across output paths, and consumes the audit before model loading or case generation. Failure does not permit a rerun on the same final seed. It measures bounded synthetic retention and never automatically promotes a policy. The next objective is to improve decision quality and resource cost against smaller policies and search; the current results do not justify another increase in size by themselves.

## Reproduce a byte learning pilot

Use the existing pinned Mac environment. For example, after checking the current run reports:

```sh
.venv/bin/python -m winward_scale.train \
  --output runs/scale-my-primitive-pilot --preset 32m --purpose pilot \
  --curriculum primitives --steps 1500 --batch-size 32 --max-seq-len 256 \
  --train-cases 8192 --valid-cases 256 --learning-rate 0.0003 \
  --eval-every 300 --save-every 300 --seed 510000003
```

Use an unused output directory. The larger stages need separately measured batch sizes and memory guidelines; changing `--preset` alone does not constitute a successful scaling experiment.

Implementation references: [MLX Adafactor](https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Adafactor.html), [Adafactor paper](https://arxiv.org/abs/1804.04235), and [MLX memory-limit semantics](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_memory_limit.html).
