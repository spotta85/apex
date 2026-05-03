"""Gate router backed by a .gguf file.

Routing math (scoring + top-k) lives in the shared
`BaseNumpyGateRouter`. This subclass owns its own GGUFTensorIndex
(opens the .gguf file via mmap) and resolves per-layer gate names via
the supplied layout.

Loaded tensors are mmap-backed (GGUFReader uses mmap), which mirrors
the laptop-friendly access pattern llama.cpp uses.
"""

from apex._numpy_gate_router_base import BaseNumpyGateRouter
from apex.gguf_tensor_index import GGUFTensorIndex


class GGUFGateRouter(BaseNumpyGateRouter):
    def __init__(self, path: str, layout, top_k: int = 2):
        super().__init__(top_k=top_k)
        self.index = GGUFTensorIndex(path)
        self.layout = layout

    def _load_gate(self, layer_id: int):
        if layer_id not in self.gates:
            name = self.layout.gate_weight(layer_id)
            self.gates[layer_id] = self.index.tensor(name)
        return self.gates[layer_id]
