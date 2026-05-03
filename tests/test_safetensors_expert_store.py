"""Tests for SafetensorsExpertStore: load ExpertWeights bundles by layout name."""

import numpy as np
import pytest
from safetensors.numpy import save_file

from apex.safetensors_expert_store import SafetensorsExpertStore
from apex.types import ExpertWeights
from apex.weight_layout import FakeMoELayout
from tests._world import write_safetensors_world


def test_loads_expert_as_bundle(tmp_path):
    # Each expert has three projections; load() should return them
    # bundled into an ExpertWeights with HF-orientation SwiGLU shapes
    # (gate/up are (intermediate, hidden); down is reversed).
    path = write_safetensors_world(tmp_path / "x.safetensors")

    bundle = SafetensorsExpertStore(str(path), FakeMoELayout()).load(0, 1)

    assert isinstance(bundle, ExpertWeights)
    assert bundle.gate_proj.shape == (16, 8)
    assert bundle.up_proj.shape == (16, 8)
    assert bundle.down_proj.shape == (8, 16)
    assert bundle.gate_proj.dtype == np.float32


def test_uses_layout_for_tensor_naming(tmp_path):
    # If we hand the store a custom layout, the store must ask the
    # layout for tensor names — not hardcode FakeMoELayout's pattern.
    class WeirdLayout:
        def expert_tensors(self, layer_id, expert_id):
            return {
                "gate_proj": f"weird/{layer_id}/{expert_id}/g",
                "up_proj": f"weird/{layer_id}/{expert_id}/u",
                "down_proj": f"weird/{layer_id}/{expert_id}/d",
            }

    path = tmp_path / "x.safetensors"
    save_file(
        {
            "weird/0/1/g": np.full((16, 8), 9.0, dtype="float32"),
            "weird/0/1/u": np.full((16, 8), 9.0, dtype="float32"),
            "weird/0/1/d": np.full((8, 16), 9.0, dtype="float32"),
        },
        str(path),
    )

    bundle = SafetensorsExpertStore(str(path), WeirdLayout()).load(0, 1)
    assert bundle.gate_proj.shape == (16, 8)


def test_missing_projection_raises_clear_error(tmp_path):
    # Asking for an expert whose tensors aren't in the file should
    # surface an error from safe_open.
    path = tmp_path / "x.safetensors"
    save_file(
        {"layers.0.experts.0.gate_proj.weight": np.zeros((1, 1), dtype="float32")},
        str(path),
    )
    store = SafetensorsExpertStore(str(path), FakeMoELayout())

    with pytest.raises(Exception):
        store.load(0, 99)


def test_works_with_cache_and_manager(tmp_path):
    # End-to-end: ExpertManager + ExpertCache should treat the bundle
    # as one cached unit; second request is a hit (same instance).
    from apex.expert_cache import ExpertCache
    from apex.expert_manager import ExpertManager

    path = write_safetensors_world(tmp_path / "x.safetensors")
    cache = ExpertCache(max_memory_mb=1, verbose=False)
    manager = ExpertManager(SafetensorsExpertStore(str(path), FakeMoELayout()), cache)

    first = manager.ensure_loaded(0, 1)
    second = manager.ensure_loaded(0, 1)

    assert isinstance(first, ExpertWeights)
    assert first is second  # cached
    # Cache size = sum of 3 projections' nbytes
    expected_bytes = (
        first.gate_proj.nbytes + first.up_proj.nbytes + first.down_proj.nbytes
    )
    assert cache.current_bytes == expected_bytes
