"""Fake compute backend (NumPy).

A real backend would call into MLX, llama.cpp, a CUDA kernel, etc.
For the MVP we just do NumPy matmul / silu so the runtime can move
data through the pipeline and produce real arrays.

Step 4 changed `run_expert` from a single matmul to the standard
SwiGLU FFN:

    out = (silu(h @ gate_proj) * (h @ up_proj)) @ down_proj

Backend interface:
  - name                          string, used in trace lines
  - prepare_expert(bundle)        cache-side conversion (passthrough here)
  - prepare_hidden(arr)           initial hidden conversion (passthrough)
  - run_expert(hidden, expert)    SwiGLU FFN over an ExpertWeights bundle
  - combine_outputs(list)         average across selected experts
  - to_numpy(arr)                 boundary conversion to ndarray (passthrough)
"""

import numpy as np


class FakeBackend:
    name = "numpy"

    def prepare_expert(self, expert_weights):
        # Cache-side conversion. NumPy is the lingua franca; nothing to
        # transform. Returning the same object is the signal the manager
        # uses to skip the "prepared expert for backend: ..." trace line
        # (so non-MLX demos stay byte-identical).
        return expert_weights

    def prepare_hidden(self, hidden_state):
        # Same passthrough rationale — runtime stores hidden as ndarray.
        return hidden_state

    def run_expert(self, hidden_state, expert_weights):
        # SwiGLU FFN, the standard MoE expert shape:
        #   out = (silu(h @ W_gate) * (h @ W_up)) @ W_down
        # Shapes:
        #   h:        (batch, hidden)
        #   W_gate:   (hidden, intermediate)
        #   W_up:     (hidden, intermediate)
        #   W_down:   (intermediate, hidden)
        #   out:      (batch, hidden)  -- same as input, so layers chain
        gate = hidden_state @ expert_weights.gate_proj
        up = hidden_state @ expert_weights.up_proj
        # silu(x) = x * sigmoid(x) = x / (1 + exp(-x)).
        # Compact form; numerically OK for our small fake values.
        gate_silu = gate / (1.0 + np.exp(-gate))
        return (gate_silu * up) @ expert_weights.down_proj

    def combine_outputs(self, outputs):
        # Uniform mean across the per-expert outputs. Real MoE uses
        # router-weighted combination; uniform is enough to demonstrate
        # the loading/caching machinery.
        return sum(outputs) / len(outputs)

    def to_numpy(self, arr):
        # Boundary conversion for callers / tests. NumPy is already
        # ndarray, so this is identity (np.asarray covers np.memmap too).
        return np.asarray(arr)
