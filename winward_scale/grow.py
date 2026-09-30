"""Width-growth experiment using only our own structured policy weights.

This changes capacity, not initial skill, input vocabulary, or planning depth.
It does not load Qwen or any external model. The source architecture and frozen
experiments remain untouched. Uniform residual replication preserves LayerNorm
statistics; whole attention heads are repeated so their head dimension/scaling
does not change. Every destination tensor belongs to the ordinary dense forward
path and is independently trainable, rather than tied or unused padding.

With noise=0 the transformation is exact in real arithmetic. Floating point
reductions and dtype conversion can change logits; measure retention before use.
For a full-float32 GPU proof, launch with MLX_ENABLE_TF32=0. MLX's default M5
matrix kernels may use reduced precision even when arrays are float32; see
https://ml-explore.github.io/mlx/build/html/usage/precision.html . The function
does not change process precision settings or other running experiments.
Optional noise lies in the replicated-input nullspace, breaking weight symmetry
while still preserving the mathematical forward function. This is an experiment,
not a guarantee that training will exploit the extra capacity or improve quality.

The standard MLX constructors build lazy random arrays. This function replaces
*all* their parameter leaves, checked strictly, before evaluating the destination.
It does not explicitly evaluate those temporary random parameters. That avoids
an intentional full random-model allocation, but is NOT a measured guarantee of
large-model peak memory. Only tiny and factor-two growth have been exercised.

References: Net2Net https://arxiv.org/abs/1511.05641 and lossless transformer
expansion/symmetry breaking, LEMON https://arxiv.org/abs/2310.07999 . The explicit
mapping below is specialized to the existing Winward/MLX architecture.
"""
from dataclasses import replace
import math

import mlx.core as mx
from mlx.utils import tree_flatten

from agent_training.model import GoalPolicy, PolicyConfig
from agent_training.model_v3 import UnifiedPolicy


def _check_factor(factor):
    if type(factor) is not int or factor < 1:
        raise ValueError("factor must be a positive integer")


def parameter_count(config: PolicyConfig, factor: int = 1) -> int:
    """Count without constructing a model or allocating any weight arrays."""
    _check_factor(factor)
    for name in ("input_dim", "width", "layers", "heads", "expansion"):
        if type(getattr(config, name)) is not int or getattr(config, name) < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config.width % config.heads:
        raise ValueError("width must be divisible by heads")
    d = config.width * factor
    hidden = d * config.expansion
    # Four bias-free attention projections; two biased FF projections; two
    # affine LayerNorms per block; input, final affine norm, and scalar output.
    block = 4 * d * d + 2 * d * hidden + hidden + 5 * d
    return config.layers * block + (config.input_dim + 4) * d + 1


def _expanded_linear(weight, factor, *, output_copies, noise, key):
    old_inputs = weight.shape[1]
    expanded = mx.tile(weight, (output_copies, factor)) / factor
    if noise and factor > 1:
        # Each row receives perturbations whose sum over the duplicate input
        # coordinates is zero. Thus delta @ tile(x, factor) == 0, mathematically.
        perturbation = mx.random.normal(
            shape=(expanded.shape[0], factor, old_inputs), key=key)
        perturbation = perturbation - mx.mean(perturbation, axis=1, keepdims=True)
        scale = noise * mx.sqrt(mx.mean(mx.square(expanded)))
        expanded = expanded + scale * perturbation.reshape(expanded.shape)
    return expanded


def grow_policy(source_model, factor, dtype="float32", noise=0):
    """Return an independent, wider GoalPolicy/UnifiedPolicy with mapped weights.

    ``noise`` is the relative RMS scale of a zero-sum perturbation to expanded
    linear weights; no noise is added to the input projection, biases, or norms.
    Noise uses fixed per-leaf random keys so repeated transformations are stable.
    This mapping intentionally keeps layer count, FF expansion ratio, head
    dimension, LayerNorm epsilon, feature schema, and masking logic unchanged.
    Caller owns memory budgeting, evaluating the returned lazy arrays, measuring
    numerical drift, and checking training/validation before any promotion.
    """
    _check_factor(factor)
    if type(source_model) not in (GoalPolicy, UnifiedPolicy):
        raise ValueError("source must be an unmodified GoalPolicy or UnifiedPolicy")
    if dtype not in ("float32", "bfloat16", "float16"):
        raise ValueError("dtype must be float32, bfloat16, or float16")
    if isinstance(noise, bool) or not isinstance(noise, (int, float)) or not math.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative")
    parameter_count(source_model.config, factor)
    config = replace(source_model.config, width=source_model.config.width * factor,
                     heads=source_model.config.heads * factor)
    destination = type(source_model)(config)
    # Standard architecture epsilon is part of function preservation. Preserve
    # it even if the caller changed an existing LayerNorm's non-array setting.
    destination.norm.eps = source_model.norm.eps
    for old, new in zip(source_model.blocks, destination.blocks):
        new.norm1.eps, new.norm2.eps = old.norm1.eps, old.norm2.eps
    mapped = []
    for index, (name, original) in enumerate(tree_flatten(source_model.parameters())):
        value = original.astype(mx.float32)
        if name == "input_projection.weight":
            value = mx.tile(value, (factor, 1))
        elif name == "output.weight":
            value = _expanded_linear(value, factor, output_copies=1, noise=noise,
                                     key=mx.random.key(index))
        elif name == "output.bias":
            pass
        elif value.ndim == 2:
            value = _expanded_linear(value, factor, output_copies=factor,
                                     noise=noise, key=mx.random.key(index))
        elif value.ndim == 1:
            value = mx.tile(value, factor)
        else:
            raise ValueError(f"Unexpected source parameter: {name}")
        mapped.append((name, value.astype(getattr(mx, dtype))))
    destination.load_weights(mapped, strict=True)
    destination.train(source_model.training)
    return destination
