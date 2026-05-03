"""Expert store backed by a single .gguf file.

Same `load(layer_id, expert_id)` interface as the other stores.
Returns an `ExpertWeights` bundle holding the three SwiGLU FFN
projections; tensors come from the .gguf via `GGUFTensorIndex`.

GGUFReader uses mmap internally, so the bundle holds np.memmap views
— same lazy-page-in profile as `MMapExpertStore`.
"""

from apex.gguf_tensor_index import GGUFTensorIndex
from apex.types import ExpertWeights


class GGUFExpertStore:
    def __init__(self, path: str, layout):
        # One index per store; opens the GGUF file (mmap) once and
        # serves all subsequent loads from the same handle.
        self.index = GGUFTensorIndex(path)
        self.layout = layout

    def load(self, layer_id: int, expert_id: int) -> ExpertWeights:
        names = self.layout.expert_tensors(layer_id, expert_id)
        return ExpertWeights(
            gate_proj=self.index.tensor(names["gate_proj"]),
            up_proj=self.index.tensor(names["up_proj"]),
            down_proj=self.index.tensor(names["down_proj"]),
        )
