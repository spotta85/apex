"""Tests for GGUFGateRouter: gate-score routing from a .gguf file."""

import numpy as np
import pytest
from gguf import GGUFWriter

from apex.gguf_gate_router import GGUFGateRouter
from apex.weight_layout import FakeMoELayout


def write_fake_gguf(path, tensors):
    writer = GGUFWriter(str(path), "probe")
    for name, arr in tensors.items():
        writer.add_tensor(name, arr)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_returns_top_k_experts_by_score(tmp_path):
    # Same shape contract as the npy / safetensors gate routers:
    # column with highest score wins, then second-highest.
    gate = np.array(
        [
            [1.0, 5.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype="float32",
    )
    path = tmp_path / "x.gguf"
    write_fake_gguf(path, {"layers.0.gate.weight": gate})

    router = GGUFGateRouter(str(path), FakeMoELayout(), top_k=2)
    hidden = np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32")
    selected, weights = router.route(0, hidden)
    assert selected == [1, 2]
    assert weights is None


def test_returns_python_ints(tmp_path):
    # Cache keys / JSON serialization expect plain Python ints.
    gate = np.eye(4, 4, dtype="float32")
    path = tmp_path / "x.gguf"
    write_fake_gguf(path, {"layers.0.gate.weight": gate})

    router = GGUFGateRouter(str(path), FakeMoELayout(), top_k=2)
    hidden = np.ones((1, 4), dtype="float32")
    selected, _ = router.route(0, hidden)
    for expert_id in selected:
        assert type(expert_id) is int


def test_lazy_loads_and_caches_gate(tmp_path):
    # First call hits the index; second call should use the in-memory
    # cached gate. We verify by checking that two calls return equal
    # results without exception (file isn't re-read either way; mmap
    # makes underlying changes invisible until remap).
    gate = np.array([[1.0, 2.0, 3.0]], dtype="float32")
    path = tmp_path / "x.gguf"
    write_fake_gguf(path, {"layers.0.gate.weight": gate})

    router = GGUFGateRouter(str(path), FakeMoELayout(), top_k=1)
    hidden = np.array([[1.0]], dtype="float32")
    a, _ = router.route(0, hidden)
    b, _ = router.route(0, hidden)

    assert a == b == [2]  # column 2 has the highest weight (3.0)
    # The gate should now be in the per-router cache (one entry).
    assert 0 in router.gates


def test_different_hidden_chooses_different_experts(tmp_path):
    gate = np.eye(4, 4, dtype="float32")
    path = tmp_path / "x.gguf"
    write_fake_gguf(path, {"layers.0.gate.weight": gate})

    router = GGUFGateRouter(str(path), FakeMoELayout(), top_k=1)
    a = np.array([[1, 0, 0, 0]], dtype="float32")
    b = np.array([[0, 0, 1, 0]], dtype="float32")
    selected_a, _ = router.route(0, a)
    selected_b, _ = router.route(0, b)
    assert selected_a == [0]
    assert selected_b == [2]


def test_missing_gate_raises(tmp_path):
    # Asking for a layer whose gate isn't in the file should error.
    path = tmp_path / "x.gguf"
    write_fake_gguf(
        path, {"layers.0.gate.weight": np.zeros((1, 1), dtype="float32")}
    )

    router = GGUFGateRouter(str(path), FakeMoELayout(), top_k=1)
    hidden = np.array([[1.0]], dtype="float32")
    with pytest.raises(KeyError):
        router.route(99, hidden)
