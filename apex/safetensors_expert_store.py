"""Expert store backed by a single .safetensors file.

Same `load(layer_id, expert_id)` interface as ExpertStore /
MMapExpertStore / GGUFExpertStore. Tensor names come from the layout
(see apex.weight_layout) — letting future code support real-model
tensor naming (Mixtral / Qwen / Gemma) by swapping the layout, not
the store.

Returns an `ExpertWeights` bundle holding the three SwiGLU FFN
projections. We use `framework="np"` so loaded tensors are numpy
arrays; no torch dependency required.
"""

from safetensors import safe_open

from apex.types import ExpertWeights


class SafetensorsExpertStore:
    def __init__(self, path: str, layout):
        # Path to the .safetensors file containing all layers/experts.
        self.path = path
        # Layout maps (layer, expert) -> per-projection tensor names.
        self.layout = layout

    def load(self, layer_id: int, expert_id: int) -> ExpertWeights:
        # Resolve all three projection names through the layout up
        # front so the safe_open block stays a tight loop. The store
        # itself stays format-agnostic about how layers/experts/projs
        # are labeled — that's the layout's job.
        names = self.layout.expert_tensors(layer_id, expert_id)
        with safe_open(self.path, framework="np") as f:
            return ExpertWeights(
                gate_proj=f.get_tensor(names["gate_proj"]),
                up_proj=f.get_tensor(names["up_proj"]),
                down_proj=f.get_tensor(names["down_proj"]),
            )
