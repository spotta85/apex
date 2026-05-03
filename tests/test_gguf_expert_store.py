"""Tests for GGUFExpertStore: load ExpertWeights bundles by layout name."""

import numpy as np
import pytest

from apex.gguf_expert_store import GGUFExpertStore
from apex.types import ExpertWeights
from apex.weight_layout import FakeMoELayout
from tests._world import write_gguf_world


def test_loads_expert_as_bundle(tmp_path):
    # HF-orientation SwiGLU shapes — gate/up are (intermediate, hidden),
    # down is (hidden, intermediate).
    path = write_gguf_world(tmp_path / "x.gguf")

    bundle = GGUFExpertStore(str(path), FakeMoELayout()).load(0, 1)

    assert isinstance(bundle, ExpertWeights)
    assert bundle.gate_proj.shape == (16, 8)
    assert bundle.up_proj.shape == (16, 8)
    assert bundle.down_proj.shape == (8, 16)
    assert bundle.gate_proj.dtype == np.float32


def test_uses_layout_for_tensor_naming(tmp_path):
    # Custom layout -> store follows it; no hardcoded pattern.
    from gguf import GGUFWriter

    class WeirdLayout:
        def expert_tensors(self, layer_id, expert_id):
            return {
                "gate_proj": f"weird/{layer_id}/{expert_id}/g",
                "up_proj": f"weird/{layer_id}/{expert_id}/u",
                "down_proj": f"weird/{layer_id}/{expert_id}/d",
            }

    path = tmp_path / "x.gguf"
    writer = GGUFWriter(str(path), "probe")
    writer.add_tensor("weird/0/1/g", np.full((16, 8), 9.0, dtype="float32"))
    writer.add_tensor("weird/0/1/u", np.full((16, 8), 9.0, dtype="float32"))
    writer.add_tensor("weird/0/1/d", np.full((8, 16), 9.0, dtype="float32"))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    bundle = GGUFExpertStore(str(path), WeirdLayout()).load(0, 1)
    assert bundle.gate_proj.shape == (16, 8)


def test_missing_projection_raises_key_error(tmp_path):
    from gguf import GGUFWriter

    path = tmp_path / "x.gguf"
    writer = GGUFWriter(str(path), "probe")
    writer.add_tensor(
        "layers.0.experts.0.gate_proj.weight", np.zeros((1, 1), dtype="float32")
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    store = GGUFExpertStore(str(path), FakeMoELayout())
    with pytest.raises(KeyError) as excinfo:
        store.load(0, 99)
    # The missing tensor name should be in the error message.
    assert "experts.99" in str(excinfo.value)


def test_works_with_cache_and_manager(tmp_path):
    # ExpertManager + cache treats the bundle as one cached unit;
    # second request is a hit (same instance).
    from apex.expert_cache import ExpertCache
    from apex.expert_manager import ExpertManager

    path = write_gguf_world(tmp_path / "x.gguf")
    cache = ExpertCache(max_memory_mb=1, verbose=False)
    manager = ExpertManager(GGUFExpertStore(str(path), FakeMoELayout()), cache)

    first = manager.ensure_loaded(0, 1)
    second = manager.ensure_loaded(0, 1)

    assert isinstance(first, ExpertWeights)
    assert first is second
    expected_bytes = (
        first.gate_proj.nbytes + first.up_proj.nbytes + first.down_proj.nbytes
    )
    assert cache.current_bytes == expected_bytes


def test_returns_mmap_backed_projections(tmp_path):
    # GGUF gives us mmap for free; each projection in the bundle
    # should be an np.memmap (lazy page-in).
    path = write_gguf_world(tmp_path / "x.gguf")
    bundle = GGUFExpertStore(str(path), FakeMoELayout()).load(0, 0)
    assert isinstance(bundle.gate_proj, np.memmap)
    assert isinstance(bundle.up_proj, np.memmap)
    assert isinstance(bundle.down_proj, np.memmap)
