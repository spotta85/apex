"""Backend-aware gate router that runs the routing matmul on MLX.

Same constructor shape as `SafetensorsGateRouter` (path + layout + top_k)
so `make_mlx_runtime` swaps it in with one line. Differences:

  - Gate weights are loaded from safetensors as numpy, then immediately
    converted to mx.array and cached. After the first per-layer call
    every routing matmul is mx.matmul on already-resident gate weights.
  - Hidden state input is accepted as either np.ndarray or mx.array
    (we wrap with mx.array if needed). The backend-native runtime hands
    this router an mx.array, so the conversion is a cheap no-op there.
  - Top-k is computed via mx.argsort, not numpy. Only the final list-of-
    Python-ints is materialized via np.array(idx) at the end (cache
    keys + JSON serialization need plain ints).

For other on-disk formats (GGUF + MLX, npy + MLX, etc.) future work
would add a parallel adapter or extract a generic GateLoader interface.
For this sprint it's safetensors-only — that's what `demo_mlx_backend.py`
uses.
"""

import mlx.core as mx
import numpy as np
from safetensors import safe_open


class MLXGateRouter:
    # See `apex.router.FakeRouter.name` for what this is for.
    name = "mlx"

    def __init__(self, path: str, layout, top_k: int = 2):
        self.path = path
        self.layout = layout
        self.top_k = top_k
        # Gate weights cached as mx.array per layer after first load.
        self.gates: dict = {}

    def route(self, layer_id: int, hidden_state):
        # Accept either ndarray or mx.array — runtime always hands us
        # an mx.array under MLXBackend, but tests / mixed setups might
        # pass ndarray. mx.array(...) of an mx.array is essentially a
        # no-op; of an ndarray it's a one-time host->device transfer.
        if not isinstance(hidden_state, mx.array):
            hidden_state = mx.array(hidden_state)

        gate = self._load_gate(layer_id)
        # scores shape: (batch=1, num_experts)
        scores = mx.matmul(hidden_state, gate)

        # Top-k by score, descending. mx.argsort returns ascending; take
        # the last top_k, reverse with [::-1] for descending. The
        # mx.eval forces evaluation so the next conversion is cheap;
        # then np.array(idx) gives us a small numpy array we can iterate
        # to produce plain Python ints (cache keys / JSON expect ints).
        sorted_idx = mx.argsort(scores[0])
        topk = sorted_idx[-self.top_k :][::-1]
        mx.eval(topk)
        return [int(i) for i in np.array(topk)]

    def _load_gate(self, layer_id: int):
        if layer_id not in self.gates:
            name = self.layout.gate_weight(layer_id)
            with safe_open(self.path, framework="np") as f:
                np_gate = f.get_tensor(name)
            # Convert once and cache as mx.array — every subsequent
            # routing call reuses this without touching disk or doing
            # a numpy->mx transfer.
            self.gates[layer_id] = mx.array(np_gate)
        return self.gates[layer_id]
