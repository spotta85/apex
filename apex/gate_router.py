"""Gate-style router that picks experts from learned gate scores.

For each layer, a gate matrix of shape `(hidden_size, num_experts)` is
loaded from disk. Routing math (scoring + top-k) lives in the shared
`BaseNumpyGateRouter`; this subclass only knows how to fetch a per-
layer gate matrix from a directory of `.npy` files.

Why a per-layer gate file: it parallels how real MoE checkpoints are
organized — each layer has its own gate weights stored alongside its
expert weights. Loading is lazy and cached so repeated calls in the
same run only hit disk once per layer.
"""

import numpy as np
from pathlib import Path

from apex._numpy_gate_router_base import BaseNumpyGateRouter


class GateRouter(BaseNumpyGateRouter):
    def __init__(self, root, top_k: int = 2):
        super().__init__(top_k=top_k)
        # Same root layout as ExpertStore: <root>/layer_<L>/{gate,expert_*}.npy
        self.root = Path(root)

    def _load_gate(self, layer_id: int):
        if layer_id not in self.gates:
            path = self.root / f"layer_{layer_id}" / "gate.npy"
            # Let np.load's FileNotFoundError propagate — the path is
            # included in the message, which is clear enough for the
            # caller to figure out what went wrong.
            self.gates[layer_id] = np.load(path)
        return self.gates[layer_id]
