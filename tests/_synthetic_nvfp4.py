"""Shared helpers for tests that build hand-crafted NVFP4 checkpoints."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from apex.dequant import NVFP4_GROUP_SIZE


# F8_E4M3 byte 0x38 decodes to exactly 1.0.
E4M3_ONE = 0x38

# Two FP4 one-values (code 0x2) packed into one uint8 byte.
FP4_PACKED_ONE = 0x22

REAL_PATH_ENV = "REAL_MOE_PATH"
_REAL_PATH_CANDIDATES = (
    Path("/Users/spotta/Desktop/models/Gemma-4-26B-A4B-NVFP4"),
    Path("/Users/spotta/Desktop/models/gemma4/Gemma-4-26B-A4B-NVFP4"),
    Path("/Users/spotta/Desktop/models/gemma4/gemma-4-26b-a4b-it-nvfp4"),
)
DEFAULT_REAL_PATH = str(
    next((path for path in _REAL_PATH_CANDIDATES if path.is_dir()), _REAL_PATH_CANDIDATES[0])
)


def write_safetensors(path: Path, items):
    """Write a multi-tensor safetensors file with hand-built bytes."""
    header = {}
    cursor = 0
    payload = bytearray()
    for name, dtype, shape, raw in items:
        size = len(raw)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + size],
        }
        payload.extend(raw)
        cursor += size
    header_json = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(len(header_json).to_bytes(8, "little"))
        f.write(header_json)
        f.write(bytes(payload))


def build_expert_items(layout, layer_id, expert_id, hidden, intermediate):
    """Return safetensors items for one all-ones NVFP4 expert."""
    items = []
    names = layout.expert_tensors(layer_id, expert_id)
    for proj_key, weight_name in names.items():
        if proj_key == "down_proj":
            rows, cols = hidden, intermediate
        else:
            rows, cols = intermediate, hidden
        prefix = weight_name[: -len(".weight")]
        packed = np.full((rows, cols // 2), FP4_PACKED_ONE, dtype=np.uint8)
        scales = np.full(
            (rows, cols // NVFP4_GROUP_SIZE), E4M3_ONE, dtype=np.uint8,
        )
        items.extend([
            (weight_name, "U8", list(packed.shape), packed.tobytes()),
            (
                f"{prefix}.weight_scale",
                "F8_E4M3",
                list(scales.shape),
                scales.tobytes(),
            ),
            (f"{prefix}.weight_scale_2", "F32", [], np.float32(1.0).tobytes()),
            (f"{prefix}.input_scale", "F32", [], np.float32(1.0).tobytes()),
        ])
    return items


def build_one_layer_checkpoint(path, layout, num_experts, hidden, intermediate):
    """Build a single-file NVFP4 checkpoint with one layer."""
    items = []
    for expert_id in range(num_experts):
        items.extend(build_expert_items(layout, 0, expert_id, hidden, intermediate))
    write_safetensors(path, items)


def build_multi_layer_checkpoint(
    path, layout, num_layers, num_experts, hidden, intermediate,
):
    """Build a single-file NVFP4 checkpoint with multiple layers."""
    items = []
    for layer_id in range(num_layers):
        for expert_id in range(num_experts):
            items.extend(
                build_expert_items(layout, layer_id, expert_id, hidden, intermediate)
            )
    write_safetensors(path, items)


def real_checkpoint_path():
    """Return Path to the real Gemma 4 NVFP4 checkpoint, or None if absent."""
    candidate = os.environ.get(REAL_PATH_ENV, DEFAULT_REAL_PATH)
    path = Path(candidate)
    return path if path.is_dir() else None


def skip_if_no_real_checkpoint():
    """Decorator: skip a test if the real-Gemma checkpoint is not on disk.

    Also tags the test `real_checkpoint`, so the fast synthetic suite is
    `pytest -m "not real_checkpoint"`. Selecting by marker rather than
    by name matters: several "real"-named files hold synthetic tests
    that a `-k "not real"` filter would silently drop.
    """
    skip = pytest.mark.skipif(
        real_checkpoint_path() is None,
        reason=(
            f"real Gemma 4 NVFP4 checkpoint not found at "
            f"{DEFAULT_REAL_PATH} (set ${REAL_PATH_ENV} to override)"
        ),
    )

    def decorate(test):
        return pytest.mark.real_checkpoint(skip(test))

    return decorate


def real_layout_and_quant_store(path):
    """Return the supported Gemma layout + quantized store for `path`."""
    from apex.checkpoint_index import read_safetensors_index
    from apex.gemma_hf_store import GemmaHFQuantizedExpertStore
    from apex.quantized_safetensors_store import QuantizedSafetensorsExpertStore
    from apex.weight_layout import get_layout

    weight_map = read_safetensors_index(path)
    for layout_name, store_cls in [
        ("gemma-moe", QuantizedSafetensorsExpertStore),
        ("gemma-hf-moe", GemmaHFQuantizedExpertStore),
    ]:
        layout = get_layout(layout_name)
        if layout.gate_weight(0) in weight_map:
            return layout, store_cls(path, layout=layout)
    raise AssertionError(f"no supported Gemma MoE layout matched {path}")
