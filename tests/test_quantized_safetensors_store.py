"""Tests for apex/quantized_safetensors_store.py.

Builds a small "fake-NVFP4" expert checkpoint by hand (matching the
real Gemma 4 NVFP4 layout: per-weight `.weight` + `.weight_scale` +
`.weight_scale_2` + `.input_scale` sidecars), loads it through the
QuantizedSafetensorsExpertStore, and asserts the bundle's projections
are QuantizedTensors with the right sidecars discovered.

Then runs them through dequantize_bundle and SwiGLU on a real backend
to confirm the load → dequant → run path works end-to-end on
hand-built inputs.
"""

import json

import numpy as np
import pytest

from apex.backends.fake_backend import FakeBackend
from apex.dequant import NVFP4_GROUP_SIZE, dequantize_bundle
from apex.quantized_safetensors_store import QuantizedSafetensorsExpertStore
from apex.tensor_types import QuantizedTensor
from apex.types import ExpertWeights
from apex.weight_layout import FakeMoELayout


# --- helpers ---------------------------------------------------------------

def _e4m3_one() -> int:
    return 0x38  # F8_E4M3 byte that decodes to exactly 1.0


def _nvfp4_packed_ones(rows: int, cols: int) -> np.ndarray:
    """Pack a matrix of all-1.0 FP4 values (code 0x2) into uint8 bytes."""
    assert cols % 2 == 0
    # Both nibbles = 0x2 → packed byte = 0x22.
    return np.full((rows, cols // 2), 0x22, dtype=np.uint8)


def _write_safetensors(path, tensors_meta_and_bytes):
    """Write a multi-tensor safetensors file with hand-built bytes.

    Same shape as test_safetensors_loader's helper; duplicated here to
    keep test files independent.
    """
    header = {}
    cursor = 0
    payload = bytearray()
    for name, dtype, shape, raw in tensors_meta_and_bytes:
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


def _build_fake_nvfp4_expert(path, layout, layer_id, expert_id, hidden, intermediate):
    """Write one NVFP4 expert (3 projections + their sidecars) to a file.

    All FP4 codes are 1.0 (0x2), all block scales are 1.0 (E4M3 0x38),
    weight_scale_2 is 1.0 — so dequantization yields all-ones matrices.

    Shapes follow HF orientation: gate_proj/up_proj are (intermediate,
    hidden); down_proj is (hidden, intermediate). After dequant the
    backend multiplies against `.T`.
    """
    names = layout.expert_tensors(layer_id, expert_id)
    items = []
    for proj_key, weight_name in names.items():
        if proj_key == "down_proj":
            rows, cols = hidden, intermediate
        else:
            rows, cols = intermediate, hidden
        prefix = weight_name[: -len(".weight")]
        packed = _nvfp4_packed_ones(rows, cols)
        scales = np.full(
            (rows, cols // NVFP4_GROUP_SIZE), _e4m3_one(), dtype=np.uint8,
        )
        items.append((weight_name, "U8", list(packed.shape), packed.tobytes()))
        items.append(
            (f"{prefix}.weight_scale", "F8_E4M3", list(scales.shape), scales.tobytes()),
        )
        items.append(
            (f"{prefix}.weight_scale_2", "F32", [], np.float32(1.0).tobytes()),
        )
        items.append(
            (f"{prefix}.input_scale", "F32", [], np.float32(1.0).tobytes()),
        )
    _write_safetensors(path, items)


# --- single-file mode ------------------------------------------------------

def test_load_returns_bundle_of_quantized_tensors(tmp_path):
    layout = FakeMoELayout()
    path = tmp_path / "fake_nvfp4.safetensors"
    _build_fake_nvfp4_expert(path, layout, 0, 1, hidden=16, intermediate=32)

    store = QuantizedSafetensorsExpertStore(str(path), layout)
    bundle = store.load(0, 1)

    assert isinstance(bundle, ExpertWeights)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        qt = getattr(bundle, proj)
        assert isinstance(qt, QuantizedTensor), proj
        assert qt.dtype == "NVFP4"
        # Sidecars discovered via prefix matching.
        assert qt.weight_scale is not None
        assert qt.weight_scale_2 is not None
        assert qt.input_scale is not None


def test_load_skips_missing_sidecars(tmp_path):
    # Only the .weight is present; sidecars absent. Bundle still loads;
    # missing sidecars surface as None on the QuantizedTensor (the
    # dequantizer is the layer that turns this into a clear error).
    layout = FakeMoELayout()
    path = tmp_path / "no_sidecars.safetensors"
    name = layout.expert_gate_proj(0, 0)
    packed = _nvfp4_packed_ones(8, 16)
    _write_safetensors(
        path, [(name, "U8", list(packed.shape), packed.tobytes())],
    )

    # We still need up/down for load to not fail the whole bundle.
    # Add empty placeholders for them so the store can return.
    items = [(name, "U8", list(packed.shape), packed.tobytes())]
    for proj_name in (layout.expert_up_proj(0, 0), layout.expert_down_proj(0, 0)):
        items.append((proj_name, "U8", [8, 8], np.zeros((8, 8), np.uint8).tobytes()))
    _write_safetensors(path, items)

    bundle = QuantizedSafetensorsExpertStore(str(path), layout).load(0, 0)
    assert bundle.gate_proj.weight_scale is None
    assert bundle.gate_proj.weight_scale_2 is None
    assert bundle.gate_proj.input_scale is None


# --- end-to-end load → dequant → SwiGLU -----------------------------------

def test_load_dequant_then_swiglu_runs(tmp_path):
    # Hand-built all-ones expert. Dequantizes to ones. SwiGLU runs and
    # produces the expected shape (1, hidden).
    layout = FakeMoELayout()
    path = tmp_path / "ones_expert.safetensors"
    hidden, intermediate = 16, 32
    _build_fake_nvfp4_expert(
        path, layout, 0, 0, hidden=hidden, intermediate=intermediate,
    )

    store = QuantizedSafetensorsExpertStore(str(path), layout)
    quantized_bundle = store.load(0, 0)
    bundle = dequantize_bundle(quantized_bundle)

    # All projections should be ndarray ones now.
    for proj in ("gate_proj", "up_proj", "down_proj"):
        arr = getattr(bundle, proj)
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32

    # Hidden of all ones; SwiGLU on ones-everywhere produces a known
    # finite output of shape (1, hidden).
    hidden_state = np.ones((1, hidden), dtype=np.float32)
    out = FakeBackend().run_expert(hidden_state, bundle)
    assert out.shape == (1, hidden)
    assert np.all(np.isfinite(out))


# --- sharded mode ----------------------------------------------------------

def test_sharded_directory_load(tmp_path):
    layout = FakeMoELayout()
    # Put gate_proj's tensors in shard A, up_proj + down_proj in shard B.
    hidden, intermediate = 16, 32
    names = layout.expert_tensors(0, 0)

    shard_a_items = []
    shard_b_items = []
    for proj_key, weight_name in names.items():
        # HF orientation: gate/up are (intermediate, hidden); down is reversed.
        if proj_key == "down_proj":
            rows, cols = hidden, intermediate
        else:
            rows, cols = intermediate, hidden
        prefix = weight_name[: -len(".weight")]
        packed = _nvfp4_packed_ones(rows, cols)
        scales = np.full(
            (rows, cols // NVFP4_GROUP_SIZE), _e4m3_one(), dtype=np.uint8,
        )
        target = shard_a_items if proj_key == "gate_proj" else shard_b_items
        target.extend([
            (weight_name, "U8", list(packed.shape), packed.tobytes()),
            (f"{prefix}.weight_scale", "F8_E4M3", list(scales.shape), scales.tobytes()),
            (f"{prefix}.weight_scale_2", "F32", [], np.float32(1.0).tobytes()),
            (f"{prefix}.input_scale", "F32", [], np.float32(1.0).tobytes()),
        ])

    shard_a = tmp_path / "shard-a.safetensors"
    shard_b = tmp_path / "shard-b.safetensors"
    _write_safetensors(shard_a, shard_a_items)
    _write_safetensors(shard_b, shard_b_items)

    weight_map = {n: shard_a.name for n, _, _, _ in shard_a_items}
    weight_map.update({n: shard_b.name for n, _, _, _ in shard_b_items})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 0},
        "weight_map": weight_map,
    }))

    store = QuantizedSafetensorsExpertStore(str(tmp_path), layout)
    bundle = store.load(0, 0)
    # Each projection found regardless of which shard it lives in.
    for proj in ("gate_proj", "up_proj", "down_proj"):
        qt = getattr(bundle, proj)
        assert isinstance(qt, QuantizedTensor)
        assert qt.weight_scale is not None
        assert qt.weight_scale_2 is not None
