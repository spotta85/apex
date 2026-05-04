"""Router for real sharded safetensors MoE checkpoints.

`MultiLayerRealRouter` is the Runtime-facing adapter for production
checkpoint routers. It lazily loads each layer's BF16 router weight,
caches it as fp32, and returns both top-k expert IDs and softmax
weights so Runtime can use router-weighted MoE combination.

Step 24A added support for the optional Gemma 4 stabilizer tensors
`router.scale` (per-feature, multiplied into hidden BEFORE projection)
and `router.per_expert_scale` (per-expert, multiplied into logits
BEFORE top-k). Both are loaded eagerly on first router use for a layer
and skipped silently when the layout doesn't expose them — keeps
Mixtral / Qwen / synthetic routers byte-identical to the pre-step-24
path.
"""

from pathlib import Path

import numpy as np

from apex.checkpoint_index import read_safetensors_index
from apex.dequant import dequantize
from apex.moe_layer import route
from apex.safetensors_loader import load_tensor
from apex.tensor_types import QuantizedTensor


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
        # Per-layer cached scale vectors (None when the layout doesn't
        # expose a router stabilizer tensor for that layer). Keyed by
        # layer_id so the absence of one tensor doesn't disable the other.
        self._hidden_scales = {}
        self._per_expert_scales = {}
        self._weight_map = read_safetensors_index(self.path)

    def _load_for_layer(self, layer_id: int):
        if layer_id not in self._weights:
            self._weights[layer_id] = self._load_router_weight(layer_id)
            self._hidden_scales[layer_id] = self._maybe_load_scale(
                self.layout.router_scale(layer_id),
            )
            self._per_expert_scales[layer_id] = self._maybe_load_scale(
                self.layout.router_per_expert_scale(layer_id),
            )
        return self._weights[layer_id]

    def _maybe_load_scale(self, name):
        """Return the bf16 → fp32 scale tensor or None if the name is missing.

        Layouts that don't carry a stabilizer return `None` from the
        accessor; we also tolerate the rare case where the accessor
        returns a name that isn't in the safetensors index (defensive
        — shouldn't happen on a well-formed checkpoint).
        """
        if name is None or name not in self._weight_map:
            return None
        return self._load_one(name)

    def _load_router_weight(self, layer_id: int):
        if hasattr(self.layout, "router_tensors"):
            names = self.layout.router_tensors(layer_id)
            data = self._load_one(names["weight"])
            tensor = QuantizedTensor(
                data=data,
                dtype="AFFINE_UINT8_U32",
                shape=tuple(data.shape),
                weight_scale=self._load_one(names["scales"]),
                weight_scale_2=self._load_one(names["biases"]),
            )
            return dequantize(tensor, name=names["weight"])

        name = self.layout.gate_weight(layer_id)
        return self._load_one(name)

    def _load_one(self, name: str):
        return load_tensor(self.path / self._weight_map[name], name)

    def shape_for_layer(self, layer_id: int):
        """Return (num_experts, hidden_dim) for a layer's router weight."""
        return self._load_for_layer(layer_id).shape

    def weight_for_layer(self, layer_id: int):
        """Return a layer's dequantized router projection."""
        return self._load_for_layer(layer_id)

    def has_router_scales(self, layer_id: int) -> bool:
        """True iff this layer carries `router.scale` / `per_expert_scale`.

        Used by tests + the audit script to confirm the stabilizers are
        actually being loaded on a Gemma 4 checkpoint (a regression
        where the layout silently stops exposing them would be
        invisible without this).
        """
        # Ensure the layer is loaded so the cache slots are populated.
        self._load_for_layer(layer_id)
        return (
            self._hidden_scales.get(layer_id) is not None
            or self._per_expert_scales.get(layer_id) is not None
        )

    def route(self, layer_id: int, hidden_state, *, rms_eps: float = 1e-6):
        router_weight = self._load_for_layer(layer_id)
        # Real checkpoint matmuls can produce transient floating-point
        # warnings while still yielding finite logits and outputs; the
        # demos/tests assert finiteness after the full layer.
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            expert_ids, weights = route(
                np.asarray(hidden_state),
                router_weight,
                top_k=self.top_k,
                # Both default to None when the layer / layout doesn't
                # carry the stabilizer; route() then skips the RMS-norm
                # / per-expert-scale post-softmax (step 26D).
                hidden_scale=self._hidden_scales.get(layer_id),
                per_expert_scale=self._per_expert_scales.get(layer_id),
                rms_eps=rms_eps,
            )
        return expert_ids[0].tolist(), weights[0].tolist()
