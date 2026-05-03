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

from apex.backends.fake_backend import FakeBackend
from apex.dequant import dequantize_bundle
from apex.quantized_safetensors_store import QuantizedSafetensorsExpertStore
from apex.tensor_types import QuantizedTensor
from apex.types import ExpertWeights
from apex.weight_layout import FakeMoELayout
from tests._synthetic_nvfp4 import (
    FP4_PACKED_ONE,
    build_expert_items,
    write_safetensors,
)


# --- single-file mode ------------------------------------------------------

def test_load_returns_bundle_of_quantized_tensors(tmp_path):
    layout = FakeMoELayout()
    path = tmp_path / "fake_nvfp4.safetensors"
    write_safetensors(path, build_expert_items(layout, 0, 1, hidden=16, intermediate=32))

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
    # Gate has only the .weight; sidecars absent. Bundle still loads;
    # missing sidecars surface as None on the QuantizedTensor.
    layout = FakeMoELayout()
    path = tmp_path / "no_sidecars.safetensors"
    name = layout.expert_gate_proj(0, 0)
    packed = np.full((8, 8), FP4_PACKED_ONE, dtype=np.uint8)

    # We still need up/down for load to not fail the whole bundle.
    # Add empty placeholders for them so the store can return.
    items = [(name, "U8", list(packed.shape), packed.tobytes())]
    for proj_name in (layout.expert_up_proj(0, 0), layout.expert_down_proj(0, 0)):
        items.append((proj_name, "U8", [8, 8], np.zeros((8, 8), np.uint8).tobytes()))
    write_safetensors(path, items)

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
    write_safetensors(
        path, build_expert_items(
            layout, 0, 0, hidden=hidden, intermediate=intermediate,
        ),
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
    gate_prefix = names["gate_proj"][: -len(".weight")]
    for item in build_expert_items(layout, 0, 0, hidden, intermediate):
        target = shard_a_items if item[0].startswith(gate_prefix) else shard_b_items
        target.append(item)

    shard_a = tmp_path / "shard-a.safetensors"
    shard_b = tmp_path / "shard-b.safetensors"
    write_safetensors(shard_a, shard_a_items)
    write_safetensors(shard_b, shard_b_items)

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
