"""Memory-bounded growth of our own structured decision transformer.

Parameter names and operations match the original Winward set transformer.
This is not a text model and does not import external model weights. Wider
weights are independent dense arrays. More capacity alone establishes no skill.

The constructor creates lazy zero placeholders, not a random multi-billion-
parameter model. Load a complete checkpoint or use grow_structured before a
forward pass. Growth evaluates and installs one tensor at a time. Matrix mapping
and optional zero-sum perturbations use bounded row chunks in FP32, immediately
stored in the destination dtype. No full FP32 destination matrix/model is built.
Concatenating chunks temporarily needs another copy of one destination tensor;
callers must still budget source + destination + this temporary and training.

BF16 conversion/forward arithmetic is lossy: verify decision retention separately
from the full-FP32 mathematical expansion proof in grow.py. FP32 LayerNorm
parameters/statistics are cast back to the activation dtype after normalization.
"""
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import checkpoint
from mlx.utils import tree_flatten, tree_unflatten

from agent_training.model import GoalPolicy
from agent_training.model_v3 import UnifiedPolicy
from .grow import parameter_count


@dataclass(frozen=True)
class StructuredConfig:
    input_dim: int = 505
    width: int = 256
    layers: int = 6
    heads: int = 8
    expansion: int = 4
    dtype: str = "bfloat16"
    norm_dtype: str = "float32"
    gradient_checkpointing: bool = True
    norm_eps: float = 1e-5

    def __post_init__(self):
        parameter_count(self)
        if self.dtype not in ("bfloat16", "float32"):
            raise ValueError("dtype must be bfloat16 or float32")
        if self.norm_dtype != "float32":
            raise ValueError("LayerNorm parameters must remain float32")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be boolean")
        if isinstance(self.norm_eps, bool) or not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive and finite")

    def parameter_count(self):
        return parameter_count(self)

    def to_dict(self):
        return asdict(self)


class Linear(nn.Module):
    def __init__(self, inputs, outputs, dtype, *, bias=True):
        super().__init__()
        self.weight = mx.zeros((outputs, inputs), dtype=dtype)
        if bias:
            self.bias = mx.zeros((outputs,), dtype=dtype)

    def __call__(self, x):
        result = x.astype(self.weight.dtype) @ self.weight.T
        return result + self.bias if "bias" in self else result


class LayerNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = mx.ones((width,), dtype=mx.float32)
        self.bias = mx.zeros((width,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, x):
        return mx.fast.layer_norm(x.astype(mx.float32), self.weight, self.bias,
                                  self.eps).astype(x.dtype)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        dtype, d = getattr(mx, config.dtype), config.width
        self.num_heads = config.heads
        self.query_proj = Linear(d, d, dtype, bias=False)
        self.key_proj = Linear(d, d, dtype, bias=False)
        self.value_proj = Linear(d, d, dtype, bias=False)
        self.out_proj = Linear(d, d, dtype, bias=False)

    def __call__(self, x, mask):
        def split(value):
            return value.reshape(*value.shape[:-1], self.num_heads, -1).transpose(0, 2, 1, 3)
        q, k, v = (split(projection(x)) for projection in
                   (self.query_proj, self.key_proj, self.value_proj))
        attended = mx.fast.scaled_dot_product_attention(q, k, v,
                     scale=q.shape[-1] ** -.5, mask=mask)
        return self.out_proj(attended.transpose(0, 2, 1, 3).flatten(-2, -1))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        dtype, d = getattr(mx, config.dtype), config.width
        self.norm1 = LayerNorm(d, config.norm_eps)
        self.attention = Attention(config)
        self.norm2 = LayerNorm(d, config.norm_eps)
        self.up = Linear(d, d * config.expansion, dtype)
        self.down = Linear(d * config.expansion, d, dtype)

    def __call__(self, x, mask):
        x = x + self.attention(self.norm1(x), mask)
        return x + self.down(nn.gelu(self.up(self.norm2(x))))


class StructuredPolicy(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or StructuredConfig()
        dtype = getattr(mx, self.config.dtype)
        self.input_projection = Linear(self.config.input_dim, self.config.width, dtype)
        self.blocks = [Block(self.config) for _ in range(self.config.layers)]
        self.norm = LayerNorm(self.config.width, self.config.norm_eps)
        self.output = Linear(self.config.width, 1, dtype)
        self._ready = False

    def __call__(self, features, valid, eligible):
        if not self._ready:
            raise RuntimeError("Load complete structured weights or call grow_structured first")
        if features.ndim != 3 or features.shape[-1] != self.config.input_dim:
            raise ValueError("features must be [batch, candidates, configured input_dim]")
        if valid.shape != features.shape[:2] or eligible.shape != features.shape[:2]:
            raise ValueError("valid and eligible must match the candidate dimensions")
        if valid.dtype != mx.bool_ or eligible.dtype != mx.bool_:
            raise ValueError("valid and eligible must be boolean")
        x = self.input_projection(features.astype(self.input_projection.weight.dtype))
        mask = mx.where(valid[:, None, None, :], mx.array(0., dtype=x.dtype),
                        mx.array(-1e9, dtype=x.dtype))
        for block in self.blocks:
            fn = checkpoint(block) if self.training and self.config.gradient_checkpointing else block
            x = fn(x, mask)
        logits = self.output(self.norm(x)).squeeze(-1).astype(mx.float32)
        return mx.where(eligible, logits, mx.array(-1e9, dtype=mx.float32))

    def count_parameters(self):
        return sum(int(value.size) for _, value in tree_flatten(self.parameters()))

    def load_weights(self, file_or_weights, strict=True):
        super().load_weights(file_or_weights, strict=strict)
        if strict:
            for name, value in tree_flatten(self.parameters()):
                expected = mx.float32 if _is_norm(name) else getattr(mx, self.config.dtype)
                if value.dtype != expected:
                    raise ValueError(f"Checkpoint dtype for {name} does not match the config")
            self._ready = True
        return self


def _is_norm(name):
    return name.startswith("norm.") or ".norm1." in name or ".norm2." in name


def _map_matrix(original, output_copies, input_copies, dtype, noise, seed, chunk_rows):
    rows, cols = original.shape
    noise_scale = 0.
    if noise and input_copies > 1:
        rms = mx.sqrt(mx.mean(mx.square(original.astype(mx.float32))))
        noise_scale = float(rms.item()) * noise / input_copies
    pieces = []
    for start in range(0, rows * output_copies, chunk_rows):
        count = min(chunk_rows, rows * output_copies - start)
        indices = mx.arange(start, start + count) % rows
        base = mx.take(original, indices, axis=0).astype(mx.float32)
        piece = mx.tile(base, (1, input_copies)) / input_copies
        if noise_scale:
            perturbation = mx.random.normal(shape=(count, input_copies, cols),
                                             key=mx.random.key((seed + start) % 2**32))
            perturbation -= mx.mean(perturbation, axis=1, keepdims=True)
            piece = piece + noise_scale * perturbation.reshape(piece.shape)
        piece = piece.astype(dtype)
        mx.eval(piece)
        pieces.append(piece)
    result = mx.concatenate(pieces, axis=0) if len(pieces) > 1 else pieces[0]
    mx.eval(result)
    return result


def grow_structured(source_model, factor, *, noise=.01, dtype="bfloat16",
                    gradient_checkpointing=True, seed=0, row_chunk_size=128):
    """Grow own weights, storing each mapped tensor before constructing the next.

    Use successive factors 2, 3, 5 for 19.16M, 170.73M, 4.251B parameters.
    Neither this sequence nor a successful allocation proves training quality.
    ``noise`` has the relative RMS meaning in grow.py; casting to BF16 makes
    both replication and its nullspace perturbation only approximately preserving.
    """
    if type(source_model) not in (GoalPolicy, UnifiedPolicy, StructuredPolicy):
        raise ValueError("source must be our unmodified structured policy")
    parameter_count(source_model.config, factor)
    if isinstance(source_model, StructuredPolicy) and not source_model._ready:
        raise ValueError("source policy has no loaded weights")
    if isinstance(noise, bool) or not isinstance(noise, (float, int)) or not math.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative")
    if type(seed) is not int or seed < 0 or type(row_chunk_size) is not int or row_chunk_size <= 0:
        raise ValueError("seed must be nonnegative and row_chunk_size must be positive integers")
    eps = source_model.norm.eps
    if any(block.norm1.eps != eps or block.norm2.eps != eps for block in source_model.blocks):
        raise ValueError("source LayerNorm epsilons must be uniform for this saved config")
    old = source_model.config
    config = StructuredConfig(input_dim=old.input_dim, width=old.width * factor,
               layers=old.layers, heads=old.heads * factor, expansion=old.expansion,
               dtype=dtype, gradient_checkpointing=gradient_checkpointing, norm_eps=eps)
    destination = StructuredPolicy(config)
    expected = {name: value.shape for name, value in tree_flatten(destination.parameters())}
    source = dict(tree_flatten(source_model.parameters()))
    if source.keys() != expected.keys():
        raise ValueError("Source parameter names differ from the compatible architecture")
    for index, (name, original) in enumerate(source.items()):
        target_dtype = mx.float32 if _is_norm(name) else getattr(mx, dtype)
        if original.ndim == 2:
            outputs = 1 if name == "output.weight" else factor
            inputs = 1 if name == "input_projection.weight" else factor
            value = _map_matrix(original, outputs, inputs, target_dtype, noise,
                                (seed + 104729 * index) % 2**32, row_chunk_size)
        elif original.ndim == 1:
            value = mx.tile(original.astype(target_dtype), 1 if name == "output.bias" else factor)
            mx.eval(value)
        else:
            raise ValueError(f"Unexpected source parameter rank: {name}")
        if value.shape != expected[name]:
            raise ValueError(f"Mapped parameter shape differs: {name}")
        destination.update(tree_unflatten([(name, value)]))
        del value
        # Released chunk buffers should not accumulate over every mapped matrix.
        mx.clear_cache()
    destination._ready = True
    destination.train(source_model.training)
    return destination


def save_structured(model, directory):
    """Save a complete model/config into a new directory; never overwrite a run."""
    if not isinstance(model, StructuredPolicy) or not model._ready:
        raise ValueError("Only a fully initialized structured policy can be saved")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model.save_weights(str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(model.config.to_dict(), indent=2) + "\n")
    return directory


def load_structured(directory):
    directory = Path(directory)
    config = StructuredConfig(**json.loads((directory / "config.json").read_text()))
    model = StructuredPolicy(config)
    model.load_weights(str(directory / "model.safetensors"))
    model.eval()
    return model
