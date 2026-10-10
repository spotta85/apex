"""NumPy compute backend (class name `FakeBackend` is historical).

Despite the name this is not a stub: it is the dense-fp32 NumPy
backend `MoEModel` uses for `backend="numpy"` on real checkpoints.
`apex.backends.mlx_backend.MLXBackend` is the Apple-Silicon (MLX)
counterpart; a CUDA or llama.cpp backend would follow the same
interface. It does plain NumPy matmul + a configurable gate
activation, so the runtime can move data through the pipeline and
produce real arrays.

Step 4 made `run_expert` a SwiGLU FFN; the HF-orientation refactor
flipped the projection shapes so APEX matches every public Transformer
checkpoint (Mixtral, Qwen MoE, Gemma 4, etc.) without per-load
transposes:

    out = (act(h @ gate_proj.T) * (h @ up_proj.T)) @ down_proj.T

Step 26A made `act` per-checkpoint. Mixtral / Qwen MoE / older Gemma
ship `hidden_act: silu` ⇒ SiLU. Gemma 4 ships
`hidden_activation: gelu_pytorch_tanh` ⇒ tanh-approximation GELU. The
backend reads the choice once at construction so per-call dispatch
stays a single attribute lookup. Defaulting to "silu" is the right
choice for synthetic / Mixtral / Qwen checkpoints; Gemma 4 callers
must pass `gelu_pytorch_tanh` (`MoEModel` does, from the config).

Backend interface:
  - name                          string, used in trace lines
  - prepare_expert(bundle)        cache-side conversion (passthrough here)
  - prepare_hidden(arr)           initial hidden conversion (passthrough)
  - run_expert(hidden, expert)    GLU FFN with the configured activation
  - to_numpy(arr)                 boundary conversion to ndarray (passthrough)
"""

import numpy as np

from apex.ops import get_activation


class FakeBackend:
    name = "numpy"

    def __init__(self, *, activation: str = "silu"):
        # Resolve the activation once. `get_activation` raises on an
        # unknown name so a typo in `text_config.hidden_activation`
        # surfaces here, not as silent gibberish at decode time.
        self._activation_name = activation
        self._activation = get_activation(activation)

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
        # GLU-style FFN with HF (PyTorch) weight orientation: each
        # projection's weight is stored as (out_features, in_features),
        # so projecting with the input requires multiplying against the
        # transpose. This is what every public HF MoE checkpoint
        # (Mixtral, Qwen MoE, Gemma 4) lays out, so APEX needs no
        # per-load transpose to consume real weights.
        # Shapes:
        #   h:         (batch, hidden)
        #   gate_proj: (intermediate, hidden)   -> h @ gate_proj.T -> (batch, intermediate)
        #   up_proj:   (intermediate, hidden)   -> h @ up_proj.T   -> (batch, intermediate)
        #   down_proj: (hidden, intermediate)   -> ... @ down_proj.T -> (batch, hidden)
        gate = hidden_state @ expert_weights.gate_proj.T
        up = hidden_state @ expert_weights.up_proj.T
        # Activation chosen at construction (silu for SwiGLU, gelu_tanh
        # for Gemma 4's GeluGLU). Same algebra otherwise.
        return (self._activation(gate) * up) @ expert_weights.down_proj.T

    def to_numpy(self, arr):
        # Boundary conversion for callers / tests. NumPy is already
        # ndarray, so this is identity (np.asarray covers np.memmap too).
        return np.asarray(arr)
