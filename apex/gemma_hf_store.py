"""Load stacked lowercase Gemma MoE expert tensors.

The mlx-community Gemma checkpoint stores all routed experts for a
layer in one stacked tensor:

    language_model.model.layers.L.experts.switch_glu.gate_proj.weight

Axis 0 is expert id. This store slices that axis before returning a
QuantizedTensor bundle, so the cache still stores one logical expert
per `(layer, expert)` key instead of a whole layer of experts.
"""

from pathlib import Path

from safetensors import safe_open

from apex.checkpoint_index import read_safetensors_index
from apex.tensor_types import QuantizedTensor
from apex.types import ExpertWeights


class GemmaHFQuantizedExpertStore:
    """Store adapter for stacked U32-packed Gemma HF/MLX expert tensors."""

    def __init__(self, path, layout):
        self.path = Path(path)
        self.layout = layout
        self._weight_map = read_safetensors_index(self.path) if self.path.is_dir() else None

    def load(self, layer_id: int, expert_id: int) -> ExpertWeights:
        names = self.layout.expert_tensors(layer_id, expert_id)
        return ExpertWeights(
            gate_proj=self._load_quantized(names["gate_proj"], expert_id),
            up_proj=self._load_quantized(names["up_proj"], expert_id),
            down_proj=self._load_quantized(names["down_proj"], expert_id),
        )

    def _load_quantized(self, weight_name: str, expert_id: int) -> QuantizedTensor:
        prefix = weight_name[: -len(".weight")]
        data = self._load_expert_slice(weight_name, expert_id)
        scales = self._load_expert_slice(f"{prefix}.scales", expert_id)
        return QuantizedTensor(
            data=data,
            dtype="NVFP4_U32",
            shape=tuple(data.shape),
            weight_scale=scales,
        )

    def _load_expert_slice(self, name: str, expert_id: int):
        """Load one expert slice from a stacked safetensors tensor."""
        with safe_open(str(self._resolve_shard(name)), framework="numpy") as f:
            return f.get_slice(name)[expert_id].copy()

    def _resolve_shard(self, name: str) -> Path:
        if self._weight_map is None:
            return self.path
        if name not in self._weight_map:
            raise KeyError(f"tensor {name!r} not in safetensors index of {self.path}")
        return self.path / self._weight_map[name]
