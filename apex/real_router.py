"""Router for real sharded safetensors MoE checkpoints.

`MultiLayerRealRouter` is the Runtime-facing adapter for production
checkpoint routers. It lazily loads each layer's BF16 router weight,
caches it as fp32, and returns both top-k expert IDs and softmax
weights so Runtime can use router-weighted MoE combination.
"""

from pathlib import Path

import numpy as np

from apex.checkpoint_index import read_safetensors_index
from apex.moe_layer import route
from apex.safetensors_loader import load_tensor


class MultiLayerRealRouter:
    """Lazy per-layer router for HF-style sharded safetensors checkpoints."""

    name = "numpy"

    def __init__(self, path, layout, top_k: int):
        self.path = Path(path)
        self.layout = layout
        self.top_k = top_k
        # Router weights are small relative to experts; cache each layer
        # after first use so repeated Runtime runs avoid disk reads.
        self._weights = {}
        self._weight_map = read_safetensors_index(self.path)

    def _load_for_layer(self, layer_id: int):
        if layer_id not in self._weights:
            name = self.layout.gate_weight(layer_id)
            self._weights[layer_id] = load_tensor(
                self.path / self._weight_map[name], name,
            )
        return self._weights[layer_id]

    def shape_for_layer(self, layer_id: int):
        """Return (num_experts, hidden_dim) for a layer's router weight."""
        return self._load_for_layer(layer_id).shape

    def route(self, layer_id: int, hidden_state):
        router_weight = self._load_for_layer(layer_id)
        # Real checkpoint matmuls can produce transient floating-point
        # warnings while still yielding finite logits and outputs; the
        # demos/tests assert finiteness after the full layer.
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            expert_ids, weights = route(
                np.asarray(hidden_state), router_weight, top_k=self.top_k,
            )
        return expert_ids[0].tolist(), weights[0].tolist()
