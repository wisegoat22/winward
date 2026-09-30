"""GoalPolicy: a set transformer with entirely random initial weights.

No pretrained model, tokenizer, text embeddings, or Qwen code/weights are loaded.
No positional encoding: permuting available actions permutes their scores.
"""
from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .features import FEATURE_DIM


@dataclass
class PolicyConfig:
    input_dim: int = FEATURE_DIM
    width: int = 256
    layers: int = 6
    heads: int = 8
    expansion: int = 4

    def to_dict(self):
        return asdict(self)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.width
        self.norm1 = nn.LayerNorm(d)
        self.attention = nn.MultiHeadAttention(d, config.heads)
        self.norm2 = nn.LayerNorm(d)
        self.up = nn.Linear(d, d * config.expansion)
        self.down = nn.Linear(d * config.expansion, d)

    def __call__(self, x, mask):
        h = self.norm1(x)
        x = x + self.attention(h, h, h, mask=mask)
        return x + self.down(nn.gelu(self.up(self.norm2(x))))


class GoalPolicy(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or PolicyConfig()
        self.input_projection = nn.Linear(self.config.input_dim, self.config.width)
        self.blocks = [Block(self.config) for _ in range(self.config.layers)]
        self.norm = nn.LayerNorm(self.config.width)
        self.output = nn.Linear(self.config.width, 1)

    def __call__(self, features, valid, eligible):
        mask = mx.where(valid[:, None, None, :], mx.array(0.0), mx.array(-1e9))
        x = self.input_projection(features)
        for block in self.blocks:
            x = block(x, mask)
        logits = self.output(self.norm(x)).squeeze(-1)
        return mx.where(eligible, logits, mx.array(-1e9))

    def count_parameters(self):
        return sum(int(value.size) for _, value in tree_flatten(self.parameters()))


def loss_fn(model, x, valid, eligible, targets):
    logits = model(x, valid, eligible)
    return -mx.mean(mx.sum(targets * (logits - mx.logsumexp(logits, axis=-1, keepdims=True)), axis=-1))
