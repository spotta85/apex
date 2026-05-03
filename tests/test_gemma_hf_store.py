"""Tests for the stacked lowercase Gemma HF/MLX expert store."""

import json

import numpy as np

from apex.dequant import NVFP4_GROUP_SIZE, dequantize_bundle
from apex.gemma_hf_store import GemmaHFQuantizedExpertStore
from apex.tensor_types import QuantizedTensor
from apex.weight_layout import GemmaHFMoELayout
from tests._synthetic_nvfp4 import E4M3_ONE, write_safetensors


def _pack_u32_nibbles(codes):
    codes = np.asarray(codes, dtype=np.uint32)
    return sum(codes[i::8] << (4 * i) for i in range(8)).astype(np.uint32)


def _stacked_items(layout, *, layer_id=0, num_experts=2, rows=2, cols=16):
    items = []
    for name in layout.expert_tensors(layer_id, 0).values():
        packed = np.zeros((num_experts, rows, cols // 8), dtype=np.uint32)
        scales = np.full(
            (num_experts, rows, cols // NVFP4_GROUP_SIZE),
            E4M3_ONE,
            dtype=np.uint8,
        )
        # Expert 0 dequantizes to all ones; expert 1 to all twos.
        packed[0, :, :] = _pack_u32_nibbles([0x2] * cols)
        packed[1, :, :] = _pack_u32_nibbles([0x4] * cols)
        prefix = name[: -len(".weight")]
        items.append((name, "U32", list(packed.shape), packed.tobytes()))
        items.append((f"{prefix}.scales", "U8", list(scales.shape), scales.tobytes()))
    return items


def test_load_slices_one_expert_from_stacked_tensors(tmp_path):
    layout = GemmaHFMoELayout()
    shard = tmp_path / "model-00001-of-00001.safetensors"
    items = _stacked_items(layout)
    write_safetensors(shard, items)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": {
            name: shard.name for name, *_ in items
        }})
    )

    bundle = GemmaHFQuantizedExpertStore(tmp_path, layout).load(0, 1)

    assert isinstance(bundle.gate_proj, QuantizedTensor)
    assert bundle.gate_proj.dtype == "NVFP4_U32"
    assert bundle.gate_proj.data.shape == (2, 2)
    assert bundle.gate_proj.weight_scale.shape == (2, 1)

    dequantized = dequantize_bundle(bundle)
    np.testing.assert_array_equal(
        dequantized.gate_proj,
        np.full((2, 16), 2.0, dtype=np.float32),
    )
