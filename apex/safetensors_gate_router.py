"""Gate router backed by a .safetensors file.

Routing math (scoring + top-k) lives in the shared
`BaseNumpyGateRouter`. This subclass only knows how to fetch a per-
layer gate matrix from the .safetensors file using the layout-supplied
tensor name.

Caches loaded gate matrices in memory after the first read so repeat
calls in the same run don't re-open the file.
"""

from safetensors import safe_open

from apex._numpy_gate_router_base import BaseNumpyGateRouter


class SafetensorsGateRouter(BaseNumpyGateRouter):
    def __init__(self, path: str, layout, top_k: int = 2):
        super().__init__(top_k=top_k)
        # Path to the .safetensors file (same file as the experts in
        # the typical case — checkpoints bundle everything together).
        self.path = path
        self.layout = layout

    def _load_gate(self, layer_id: int):
        if layer_id not in self.gates:
            name = self.layout.gate_weight(layer_id)
            with safe_open(self.path, framework="np") as f:
                self.gates[layer_id] = f.get_tensor(name)
        return self.gates[layer_id]
