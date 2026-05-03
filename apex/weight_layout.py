"""Tensor-name conventions for fake MoE weights.

This is the seam between the storage format (safetensors / GGUF / npy)
and the storage adapter. The adapter asks the layout "what's the
tensor name for layer L's gate?" or "what's the tensor name for layer
L expert E's down projection?" and gets back a string it can pass to
`safe_open.get_tensor(...)` (or whatever the underlying API uses).

Real MoE checkpoints use different naming schemes (`model.layers.0.
block_sparse_moe.gate.weight` vs `transformer.h.0.moe.router.weight`
vs ...), so future iterations will add MixtralLayout, QwenMoELayout,
etc. that implement the same five methods.

After step 4, each expert is FFN-style (gate_proj / up_proj / down_proj)
rather than a single matrix. The layout exposes each projection by
name; `expert_tensors(layer, expert)` is a convenience that returns
all three at once as `{"gate_proj": <name>, "up_proj": <name>,
"down_proj": <name>}`.
"""


class FakeMoELayout:
    """Layout used by scripts/make_fake_*_moe.py setup scripts.

    Tensor names look like:
      layers.<L>.gate.weight
      layers.<L>.experts.<E>.gate_proj.weight
      layers.<L>.experts.<E>.up_proj.weight
      layers.<L>.experts.<E>.down_proj.weight
    """

    def gate_weight(self, layer_id: int) -> str:
        return f"layers.{layer_id}.gate.weight"

    def expert_gate_proj(self, layer_id: int, expert_id: int) -> str:
        return f"layers.{layer_id}.experts.{expert_id}.gate_proj.weight"

    def expert_up_proj(self, layer_id: int, expert_id: int) -> str:
        return f"layers.{layer_id}.experts.{expert_id}.up_proj.weight"

    def expert_down_proj(self, layer_id: int, expert_id: int) -> str:
        return f"layers.{layer_id}.experts.{expert_id}.down_proj.weight"

    def expert_tensors(self, layer_id: int, expert_id: int) -> dict:
        """Return all three projection tensor names for one expert.

        The keys ("gate_proj" / "up_proj" / "down_proj") are stable
        across layouts — only the values (full tensor names) change
        with the underlying naming scheme. Stores can therefore
        construct an ExpertWeights bundle uniformly:

            names = layout.expert_tensors(layer, expert)
            ExpertWeights(
                gate_proj=load(names["gate_proj"]),
                up_proj=load(names["up_proj"]),
                down_proj=load(names["down_proj"]),
            )
        """
        return {
            "gate_proj": self.expert_gate_proj(layer_id, expert_id),
            "up_proj": self.expert_up_proj(layer_id, expert_id),
            "down_proj": self.expert_down_proj(layer_id, expert_id),
        }
