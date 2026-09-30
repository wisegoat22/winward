# Winward 4.25B structured decision policy

**Status, 1 October 2026:** built and trained locally; final synthetic audit complete; **not promoted**. The larger policy did not improve on our original v4 baseline. Its intended use is local research into choosing a software agent's next action from supplied facts, goals, candidate actions, permissions, and costs.

## Identity and interface

The dense policy has **4,251,056,641 registered parameters**, 6 layers, width 7,680, 240 attention heads, and feed-forward expansion 4. It receives up to 12 candidate slots, each with **505 numeric features**, plus validity and eligibility masks, and returns candidate scores. A forward pass selects one action. Exact training teachers search declared consequences up to five steps; the model itself does not perform a guaranteed five-step search.

Stored weights contain 4,250,856,961 BF16 values and 199,680 FP32 normalization values, about **8.5 GB / 7.919 GiB**. Weights are local and excluded from Git. The selected checkpoint is:

```text
runs/scale-structured-4b-train01/checkpoints/step-00000096/model.safetensors
SHA-256: 509f15c2169590d1925b485c843a969aa4394ac75060e8d0c75473d03a4dda20
```

Open [Try 4B](http://127.0.0.1:8765/try?model=4b) on the development Mac with the local server running. V4 remains the default. The policy does not directly understand free-form English: the existing optional Qwen reader converts text into a form the user reviews. Qwen weights and outputs were not used to train this policy. The interface suggests actions and executes no tools.

## Own-weight lineage and training

Our original v1 policy began from random initialization; continued local training produced v4. Width expansion and full-parameter training then followed **4.86M → 19.16M → 170.73M → 4,251.06M**. This is an expansion of our learned policy, not a new 4B language model pretrained from random initialization. Expansion adds trainable capacity; it does not establish new skills.

| Growth run | Updates executed | Selected step inherited by the next stage/audit |
| --- | ---: | ---: |
| `scale-structured-19m-01` | 500 | 100 |
| `scale-structured-171m-01` | 400 | 100 |
| `scale-structured-4b-probe01` | 8 | 8 |
| `scale-structured-4b-train01` | 128 | 96 |

There were **136 executed updates at 4.25B size**; the selected checkpoint contains **104** of them. Its cumulative growth-stage counter of 304 includes the two smaller selected stages. The main run used 8,192 synthetic training cases, 512 development cases, batch size 32, and learning rate 0.000002. Its sequential Adafactor optimizer uses FP32 factored statistics and update math, BF16 stored matrices, and FP32 norms. On the **48 GiB Apple M5 Max**, MLX 0.32.3 measured **538.4 seconds** for the main run and **21.113 GiB peak allocation**; this excludes earlier training.

A complete stored-value comparison against the main run's identity-initialized parent found **2,705,102,694 changed values (63.63%)**, changes in all **38 matrices**, and changes in 77/78 total tensors. The scalar output bias was unchanged. This verifies weight updates, not that every value learned something useful. [Training stages](../reports/scale/structured-stages.json) · [Weight-change proof](../reports/scale/4b-weight-change-proof.json).

## Sealed final evaluation

After development selection, checkpoint hashes, original v4 weights, source snapshots, settings, and seed **810000001** were sealed. Fresh final seed domains were distinct from training/development and did not reuse earlier v1–v4 final cases. Evaluation used 512 synthetic structured decisions and 40 rollouts. Models received public features and supplied eligibility only; exact labels were withheld from forward inputs. Rollouts followed model choices without oracle correction.

| Measure on the same final cases | 4.25B policy | Original v4 |
| --- | ---: | ---: |
| Optimal next-action agreement | 476/512, **92.97%** | 479/512, **93.55%** |
| Graph decisions | 330/358, 92.18% | 331/358, 92.46% |
| Belief/uncertainty decisions | 146/154, 94.81% | 148/154, 96.10% |
| Invalid/ineligible choices | 0 | 0 |
| Mean expected completion, unfinished reachable cases | **96.748%** | **96.748%** |
| Warmed batch-8 inference, amortized per case | **5.59 ms** | **0.127 ms** |

Completion averages over **31 of 40** rollout cases; five initially verified cases and four with zero reference success are excluded from that denominator. Belief probabilities are integrated over outcomes, so this is expected success, not a fraction of 31 observed binary trials. The exact reference reached 100% on the reachable subset. On 30 identical cases with equal positive expected success, mean candidate-minus-v4 declared action cost was zero.

The paired next-action counts were 2 correct only for 4.25B, 5 only for v4, 474 correct for both, and 31 incorrect for both. The larger policy passed **4/7 narrow retention checks**: zero invalid choices and matched rollout success overall and in each domain. It failed next-action retention overall and in both domains. These checks are separate from the historical v4 twelve-check audit. [Public audit summary](../reports/scale/4b-final-audit-summary.json).

## Limits and next work

These controlled simulations do not establish arbitrary-world decision making, broad English understanding, autonomous coding reliability, or real-world cost savings. Permissions, finish conditions, and action effects are supplied rules; the model does not infer its own world model. The training objective prioritizes verified success, then declared costs, but does not guarantee optimal choices.

Timing comes from one fixed-order Mac run, with one warmup batch per model and loading/warmup reported separately. Amortized batch timing is not interactive UI latency and excludes Qwen reading. No confidence interval, statistical superiority, or improvement from parameter count is claimed. Original v4's known historical failures remain.

The next useful milestone is improved held-out decision quality and completion per unit of compute: investigate the observed errors on new development cases, compare against smaller policies and exact search, and reserve a new final audit only after another candidate is frozen. Increasing parameter count alone does not meet that goal. [Scaling methods and reproduction](SCALING.md).
