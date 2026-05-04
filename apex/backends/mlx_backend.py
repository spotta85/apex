"""MLX-backed expert compute.

Same backend interface as `FakeBackend`, but everything that touches
tensors runs through MLX (`mlx.core`). On Apple Silicon this lets the
operations use unified memory and optionally the GPU without explicit
transfer steps.

Math is HF GLU-style FFN (gate/up/down projections stored as
(out_features, in_features), multiplied against the transpose):

    out = (act(h @ gate_proj.T) * (h @ up_proj.T)) @ down_proj.T

Step 26A made `act` configurable: SiLU for Mixtral / Qwen / older
Gemma checkpoints (`hidden_act: silu`); tanh-approximation GELU for
Gemma 4 (`hidden_activation: gelu_pytorch_tanh`). MLX exposes both:
`mx.sigmoid(x) * x` is SiLU, `mlx.nn.gelu_approx(x)` is the tanh-GELU
(verified bit-equivalent to our numpy `apex.ops.gelu_tanh` to fp32
precision).

`prepare_expert` converts each of the three projections in the
ExpertWeights bundle to mx.array exactly once; the cache then holds
the bundle in MLX-native form so every subsequent hit is free.
"""

import mlx.core as mx
import mlx.nn
import numpy as np

from apex.types import ExpertWeights


# MLX-native lookup so MLXBackend.run_expert stays a single matmul
# pair + activation call. `mx.sigmoid(x) * x` is the SiLU formula
# without a `mx.silu` symbol (the MLX version this repo targets
# doesn't ship one). `mlx.nn.gelu_approx` is the tanh approximation,
# matching `apex.ops.gelu_tanh` to fp32 last-bit.
_MLX_ACTIVATIONS = {
    "silu": lambda x: mx.sigmoid(x) * x,
    "swish": lambda x: mx.sigmoid(x) * x,
    "gelu_pytorch_tanh": mlx.nn.gelu_approx,
    "gelu_new": mlx.nn.gelu_approx,
}


class MLXBackend:
    name = "mlx"

    def __init__(self, *, activation: str = "silu"):
        # Same construction-time resolution as FakeBackend so a typo
        # in `text_config.hidden_activation` fails loud at load, not
        # silently as gibberish at decode time.
        if activation not in _MLX_ACTIVATIONS:
            raise ValueError(
                f"unknown activation {activation!r}; "
                f"available: {sorted(_MLX_ACTIVATIONS)}"
            )
        self._activation_name = activation
        self._activation = _MLX_ACTIVATIONS[activation]

    def prepare_expert(self, expert_weights):
        # Convert each projection to mx.array once; cache holds the
        # whole MLX-native bundle so every cache hit hands back
        # ready-to-compute tensors.
        return ExpertWeights(
            gate_proj=mx.array(expert_weights.gate_proj),
            up_proj=mx.array(expert_weights.up_proj),
            down_proj=mx.array(expert_weights.down_proj),
        )

    def prepare_hidden(self, hidden_state):
        # Initial hidden is np.ones(...) in the runtime; convert to
        # mx.array so subsequent matmuls / combines stay in MLX.
        return mx.array(hidden_state)

    def run_expert(self, hidden_state, expert_weights):
        # HF-oriented GLU FFN — same algebra as FakeBackend, MLX
        # primitives. Inputs are already mx.array (hidden via
        # prepare_hidden, bundle via prepare_expert).
        gate = mx.matmul(hidden_state, expert_weights.gate_proj.T)
        up = mx.matmul(hidden_state, expert_weights.up_proj.T)
        # MLX is lazy; run_moe_layer combines outputs without forcing
        # a numpy round trip.
        return mx.matmul(self._activation(gate) * up, expert_weights.down_proj.T)

    def to_numpy(self, arr):
        # Boundary conversion for callers / tests / numpy routers.
        # mx.eval forces the lazy compute graph to materialize, then
        # np.array does the host copy.
        if isinstance(arr, mx.array):
            mx.eval(arr)
            return np.array(arr)
        # Already ndarray-like — be permissive.
        return np.asarray(arr)
