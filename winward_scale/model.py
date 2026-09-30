"""Random-initialized text decoder for the separate Winward scaling experiment.

No pretrained weights or frozen GoalPolicy sources are used here. Preset counts
describe capacity, not trained capability. Large presets should only be created
after the caller has checked memory and installed its resource limits.
"""
from dataclasses import dataclass, replace
import math
from typing import ClassVar

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import checkpoint


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 512
    intermediate_size: int = 1536
    num_layers: int = 10
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    vocab_size: int = 260
    dtype: str = "bfloat16"
    norm_dtype: str = "float32"
    gradient_checkpointing: bool = True
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    initializer_std: float = 0.02
    name: str = "custom"

    PRESETS: ClassVar[dict[str, tuple[int, int, int, int, int]]] = {
        "32m": (512, 1536, 10, 8, 4),
        "135m": (1024, 2816, 12, 16, 4),
        "500m": (1536, 4096, 20, 24, 8),
        "1b": (2048, 5632, 22, 32, 8),
        "4b": (3072, 8192, 40, 24, 8),
    }

    def __post_init__(self):
        for field in ("hidden_size", "intermediate_size", "num_layers",
                      "num_attention_heads", "num_key_value_heads", "vocab_size"):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must divide evenly into attention heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("query heads must be a multiple of key/value heads")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        if self.dtype not in ("bfloat16", "float32"):
            raise ValueError("dtype must be bfloat16 or float32")
        if self.norm_dtype not in ("bfloat16", "float32"):
            raise ValueError("norm_dtype must be bfloat16 or float32")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be boolean")
        for field in ("rms_norm_eps", "rope_theta", "initializer_std"):
            value = getattr(self, field)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def preset(cls, name: str, **overrides) -> "ModelConfig":
        if name not in cls.PRESETS:
            raise ValueError(f"Unknown model size: {name}")
        h, f, layers, q, kv = cls.PRESETS[name]
        return replace(cls(hidden_size=h, intermediate_size=f, num_layers=layers,
                           num_attention_heads=q, num_key_value_heads=kv,
                           name=name), **overrides)

    def parameter_count(self) -> int:
        h = self.hidden_size
        kv = self.num_key_value_heads * self.head_dim
        # One shared embedding/output matrix; four attention projections, three
        # SwiGLU projections, two per-layer norms, and one final norm.
        per_layer = 2 * h * h + 2 * h * kv + 3 * h * self.intermediate_size + 2 * h
        return self.vocab_size * h + self.num_layers * per_layer + h


def _normal(shape, std: float, dtype):
    # Generate in the final dtype and finish one matrix at a time. In particular,
    # do not build/evaluate a complete float32 model before casting it to BF16.
    value = mx.random.normal(shape=shape, scale=std, dtype=dtype)
    mx.eval(value)
    return value


class Linear(nn.Module):
    """Bias-free linear layer with initialization in its stored dtype."""
    def __init__(self, in_dims: int, out_dims: int, std: float, dtype):
        super().__init__()
        self.weight = _normal((out_dims, in_dims), std, dtype)

    def __call__(self, x):
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, dims: int, std: float, dtype):
        super().__init__()
        self.weight = _normal((vocab_size, dims), std, dtype)

    def __call__(self, tokens):
        return self.weight[tokens]

    def as_linear(self, x):
        return x @ self.weight.T


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float, dtype):
        super().__init__()
        self.weight = mx.ones((dims,), dtype=dtype)
        self.eps = eps

    def __call__(self, x):
        # FP32 norm weights must not promote every downstream activation and
        # matrix multiplication to FP32. Their gradients remain full precision.
        return mx.fast.rms_norm(x, self.weight, self.eps).astype(x.dtype)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, dtype):
        super().__init__()
        h, d = config.hidden_size, config.head_dim
        self.head_dim = d
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.q_proj = Linear(h, h, config.initializer_std, dtype)
        self.k_proj = Linear(h, self.num_kv_heads * d, config.initializer_std, dtype)
        self.v_proj = Linear(h, self.num_kv_heads * d, config.initializer_std, dtype)
        residual_std = config.initializer_std / math.sqrt(2 * config.num_layers)
        self.o_proj = Linear(h, h, residual_std, dtype)
        self.rope = nn.RoPE(d, traditional=False, base=config.rope_theta)

    def __call__(self, x):
        b, t, _ = x.shape
        q = self.q_proj(x).reshape(b, t, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, t, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, t, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q, k = self.rope(q), self.rope(k)
        # Keep grouped key/value heads compact; MLX handles the grouping.
        y = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim ** -0.5, mask="causal")
        return self.o_proj(y.transpose(0, 2, 1, 3).reshape(b, t, -1))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig, dtype):
        super().__init__()
        h, f = config.hidden_size, config.intermediate_size
        self.gate_proj = Linear(h, f, config.initializer_std, dtype)
        self.up_proj = Linear(h, f, config.initializer_std, dtype)
        residual_std = config.initializer_std / math.sqrt(2 * config.num_layers)
        self.down_proj = Linear(f, h, residual_std, dtype)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, dtype):
        super().__init__()
        # Unit-scale BF16 norm weights can round away useful small updates.
        # These vectors are tiny relative to the matrix parameters.
        norm_dtype = getattr(mx, config.norm_dtype)
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps, norm_dtype)
        self.attention = Attention(config, dtype)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps, norm_dtype)
        self.feed_forward = FeedForward(config, dtype)

    def __call__(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.ffn_norm(x))


class Decoder(nn.Module):
    """Causal text decoder. Every stored parameter participates in prediction."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        dtype = getattr(mx, config.dtype)
        self.embedding = Embedding(config.vocab_size, config.hidden_size,
                                   config.initializer_std, dtype)
        self.layers = [DecoderLayer(config, dtype) for _ in range(config.num_layers)]
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, getattr(mx, config.norm_dtype))
        mx.eval(self.parameters())

    def __call__(self, tokens):
        if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] == 0:
            raise ValueError("tokens must have nonempty [batch, sequence] shape")
        if not mx.issubdtype(tokens.dtype, mx.integer):
            raise ValueError("tokens must contain integer token IDs")
        x = self.embedding(tokens)
        for layer in self.layers:
            # This helper explicitly checkpoints both inputs AND layer parameters.
            fn = checkpoint(layer) if self.training and self.config.gradient_checkpointing else layer
            x = fn(x)
        return self.embedding.as_linear(self.norm(x))
