"""Load NVFP4-quantized expert weights from a safetensors checkpoint.

Same shape as ``SafetensorsExpertStore`` — `load(layer, expert)`
returns an ``ExpertWeights`` bundle — but the projections inside the
bundle are ``QuantizedTensor`` instances rather than raw ndarrays.
Run them through ``apex.dequant.dequantize_bundle`` to get a plain
ExpertWeights ready for the backend.

Two input shapes are supported:

  - single .safetensors file (everything's in one file)
  - directory containing model.safetensors.index.json (sharded)

Sharded mode follows the index's weight_map to find which shard owns
each tensor — this is what real HF checkpoints (Mixtral / Qwen MoE /
Gemma 4 NVFP4 / etc.) need.

For each expert weight we look up four tensors:

  - <prefix>.weight           the packed quantized data (uint8)
  - <prefix>.weight_scale     per-block scales (typically F8_E4M3)
  - <prefix>.weight_scale_2   scalar tensor-wide scale (F32)
  - <prefix>.input_scale      activation-quant scale (F32, optional)

`<prefix>` is the layout's expert_*_proj name with `.weight` stripped.
Sidecars that are absent are silently skipped — this lets the same
store handle un-quantized tensors (where only `<prefix>.weight`
exists) by returning a QuantizedTensor with empty scales, which the
dequantizer will then reject loudly.
"""

from pathlib import Path

from apex.checkpoint_index import read_safetensors_index
from apex.safetensors_loader import load_tensor
from apex.tensor_types import QuantizedTensor
from apex.types import ExpertWeights


_WEIGHT_SUFFIX = ".weight"


class QuantizedSafetensorsExpertStore:
    """Sharded-aware loader that returns expert bundles of QuantizedTensors.

    Constructor takes the same shape as `SafetensorsExpertStore`
    (path + layout) plus `quant_algo` (the string written into each
    QuantizedTensor.dtype, dispatched on by `apex.dequant.dequantize`).
    Default `"NVFP4"` matches the only quant scheme APEX currently
    decodes.
    """

    def __init__(self, path, layout, quant_algo: str = "NVFP4"):
        self.path = Path(path)
        self.layout = layout
        self.quant_algo = quant_algo
        # If we're pointed at a directory, build the shard index once
        # so per-tensor lookups are O(1). Single-file mode skips this.
        if self.path.is_dir():
            self._weight_map = read_safetensors_index(self.path)
        else:
            self._weight_map = None

    def load(self, layer_id: int, expert_id: int) -> ExpertWeights:
        """Load one expert as a bundle of QuantizedTensor projections."""
        names = self.layout.expert_tensors(layer_id, expert_id)
        return ExpertWeights(
            gate_proj=self._load_quantized(names["gate_proj"]),
            up_proj=self._load_quantized(names["up_proj"]),
            down_proj=self._load_quantized(names["down_proj"]),
        )

    # --- helpers ------------------------------------------------------------

    def _load_quantized(self, weight_name: str) -> QuantizedTensor:
        if not weight_name.endswith(_WEIGHT_SUFFIX):
            # Layouts always emit names ending in `.weight`; if a future
            # layout breaks that contract we want to fail loudly.
            raise ValueError(
                f"unexpected weight name (no '.weight' suffix): {weight_name!r}"
            )
        prefix = weight_name[: -len(_WEIGHT_SUFFIX)]
        data = self._load_one(weight_name)
        return QuantizedTensor(
            data=data,
            dtype=self.quant_algo,
            shape=tuple(data.shape),
            input_scale=self._maybe_load(f"{prefix}.input_scale"),
            weight_scale=self._maybe_load(f"{prefix}.weight_scale"),
            weight_scale_2=self._maybe_load(f"{prefix}.weight_scale_2"),
        )

    def _maybe_load(self, name: str):
        """Try to load a sidecar; return None if it isn't in the checkpoint."""
        try:
            return self._load_one(name)
        except KeyError:
            return None

    def _load_one(self, name: str):
        """Load one tensor by name; route to the right shard if sharded."""
        return load_tensor(self._resolve_shard(name), name)

    def _resolve_shard(self, name: str) -> Path:
        if self._weight_map is None:
            # Single-file mode — caller pointed at one file, tensor is
            # either there or not (KeyError surfaces from load_tensor).
            return self.path
        if name not in self._weight_map:
            raise KeyError(
                f"tensor {name!r} not in safetensors index of {self.path}"
            )
        return self.path / self._weight_map[name]
