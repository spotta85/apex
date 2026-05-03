"""Fake compute backend (NumPy).

A real backend would call into MLX, llama.cpp, a CUDA kernel, etc.
For the MVP we just do NumPy matmul / silu so the runtime can move
data through the pipeline and produce real arrays.

Step 4 made `run_expert` a SwiGLU FFN; the HF-orientation refactor
flipped the projection shapes so APEX matches every public Transformer
checkpoint (Mixtral, Qwen MoE, Gemma 4, etc.) without per-load
transposes:

    out = (silu(h @ gate_proj.T) * (h @ up_proj.T)) @ down_proj.T

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
        # SwiGLU FFN with HF (PyTorch) weight orientation: each
        # projection's weight is stored as (out_features, in_features),
        # so projecting with the input requires multiplying against the
        # transpose. This is what every public HF MoE checkpoint
        # (Mixtral, Qwen MoE, Gemma 4) lays out, so APEX needs no
        # per-load transpose to consume real weights.
        # Shapes:
        #   h:        (batch, hidden)
        #   gate_proj: (intermediate, hidden)   -> h @ gate_proj.T -> (batch, intermediate)
        #   up_proj:   (intermediate, hidden)   -> h @ up_proj.T   -> (batch, intermediate)
        #   down_proj: (hidden, intermediate)   -> ... @ down_proj.T -> (batch, hidden)
        gate = hidden_state @ expert_weights.gate_proj.T
        up = hidden_state @ expert_weights.up_proj.T
        # silu(x) = x * sigmoid(x) = x / (1 + exp(-x)).
        # Compact form; numerically OK for our small fake values.
        gate_silu = gate / (1.0 + np.exp(-gate))
        return (gate_silu * up) @ expert_weights.down_proj.T

    def combine_outputs(self, outputs):
        # Uniform mean across the per-expert outputs. Real MoE uses
        # router-weighted combination; uniform is enough to demonstrate
        # the loading/caching machinery.
        return sum(outputs) / len(outputs)

    def to_numpy(self, arr):
        # Boundary conversion for callers / tests. NumPy is already
        # ndarray, so this is identity (np.asarray covers np.memmap too).
        return np.asarray(arr)
