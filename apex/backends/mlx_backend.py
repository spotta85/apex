"""MLX-backed expert compute.

Same backend interface as `FakeBackend`, but everything that touches
tensors runs through MLX (`mlx.core`). On Apple Silicon this lets the
operations use unified memory and optionally the GPU without explicit
transfer steps.

Step 4 changed `run_expert` from a single matmul to the standard
SwiGLU FFN, computed via mx.matmul / mx.sigmoid:

    out = (silu(h @ gate_proj) * (h @ up_proj)) @ down_proj

`prepare_expert` converts each of the three projections in the
ExpertWeights bundle to mx.array exactly once; the cache then holds
the bundle in MLX-native form so every subsequent hit is free.
"""

import mlx.core as mx
import numpy as np

from apex.types import ExpertWeights


class MLXBackend:
    name = "mlx"

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
        # SwiGLU FFN: same algebra as FakeBackend, MLX primitives.
        # Inputs are already mx.array (hidden via prepare_hidden,
        # bundle via prepare_expert).
        gate = mx.matmul(hidden_state, expert_weights.gate_proj)
        up = mx.matmul(hidden_state, expert_weights.up_proj)
        # silu via mx.sigmoid (mx.silu / mx.fast.silu aren't in this
        # MLX version). MLX is lazy — the result here is unevaluated;
        # combine_outputs forces evaluation cheaply via the addition.
        gate_silu = gate * mx.sigmoid(gate)
        return mx.matmul(gate_silu * up, expert_weights.down_proj)

    def combine_outputs(self, outputs):
        # Uniform average; identical algebra to FakeBackend. mx.array
        # supports `+` via __radd__ (Python's sum() works starting
        # from int 0) and `/` with a Python int.
        return sum(outputs) / len(outputs)

    def to_numpy(self, arr):
        # Boundary conversion for callers / tests / numpy routers.
        # mx.eval forces the lazy compute graph to materialize, then
        # np.array does the host copy.
        if isinstance(arr, mx.array):
            mx.eval(arr)
            return np.array(arr)
        # Already ndarray-like — be permissive.
        return np.asarray(arr)
