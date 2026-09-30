"""Adafactor updates with bounded temporary storage for local full training.

The ordinary MLX tree update builds every FP32 update lazily before evaluation.
Even factored moments can therefore coexist with several full-size temporaries.
Here each update is evaluated and installed immediately, then its consumed
gradient leaf is released. A training step still advances the optimizer's global
step and scheduled values exactly once, using MLX's ordinary implementation.

Unlike the base optimizer, update() CONSUMES its gradient dictionary/list tree:
leaves become None. Callers needing gradients later must explicitly copy them
first, at the cost of retaining that storage. Gradients are fully evaluated
before changing any model parameter, including checkpointed backward passes.
"""
import mlx.core as mx
from mlx.utils import tree_flatten

from .train import FP32Adafactor


def _parent(tree, name):
    parts = name.split(".")
    current = tree
    for part in parts[:-1]:
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current, int(parts[-1]) if isinstance(current, list) else parts[-1]


class SequentialAdafactor(FP32Adafactor):
    """Same numerical update/state as FP32Adafactor; eagerly consume each leaf."""

    def apply_single(self, gradient, parameter, state):
        updated = super().apply_single(gradient, parameter, state)
        mx.eval(updated, state)
        return updated

    def update(self, model, gradients):
        # Do not retain a flattened list containing all old parameters or
        # gradients: it would keep the memory alive after replacing tree leaves.
        names = [name for name, _ in tree_flatten(gradients)]
        for name in names:
            gp, gk = _parent(gradients, name)
            pp, pk = _parent(model, name)
            if gp[gk].shape != pp[pk].shape:
                raise ValueError(f"Gradient shape differs for {name}")
        mx.eval(gradients)
        # Public init preserves existing moments. Applying an empty gradient
        # tree lets MLX own its normal scheduling and one global step increment.
        self.init(model.trainable_parameters())
        self.apply_gradients({}, {})
        mx.eval(self.state["step"])
        for name in names:
            gp, gk = _parent(gradients, name)
            pp, pk = _parent(model, name)
            sp, sk = _parent(self.state, name)
            updated = self.apply_single(gp[gk], pp[pk], sp[sk])
            pp[pk] = updated
            gp[gk] = None
