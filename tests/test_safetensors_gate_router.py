"""Tests for SafetensorsGateRouter: gate-score routing from a .safetensors file."""

import numpy as np
import pytest
from safetensors.numpy import save_file

from apex.safetensors_gate_router import SafetensorsGateRouter
from apex.weight_layout import FakeMoELayout


def make_safetensors_file(tmp_path, tensors):
    path = tmp_path / "fake.safetensors"
    save_file(tensors, str(path))
    return path


def test_returns_top_k_experts_by_score(tmp_path):
    # Construct a gate where column 1 dominates, column 2 is second.
    gate = np.array(
        [
            [1.0, 5.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype="float32",
    )
    path = make_safetensors_file(tmp_path, {"layers.0.gate.weight": gate})
    router = SafetensorsGateRouter(str(path), FakeMoELayout(), top_k=2)

    hidden = np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32")
    selected, weights = router.route(0, hidden)

    # Highest-scoring two columns: 1 (score 5) then 2 (score 2).
    assert selected == [1, 2]
    assert weights is None


def test_returns_python_ints(tmp_path):
    # Cache keys / JSON serialization expect plain Python ints.
    gate = np.eye(4, 4, dtype="float32")
    path = make_safetensors_file(tmp_path, {"layers.0.gate.weight": gate})
    router = SafetensorsGateRouter(str(path), FakeMoELayout(), top_k=2)

    hidden = np.ones((1, 4), dtype="float32")
    selected, _ = router.route(0, hidden)
    for expert_id in selected:
        assert type(expert_id) is int


def test_lazy_loads_and_caches_gate(tmp_path):
    # First route() call reads the file; subsequent calls reuse the
    # in-memory copy. Confirm by overwriting the file after the first
    # call and seeing that route() returns the original result.
    path = make_safetensors_file(
        tmp_path,
        {"layers.0.gate.weight": np.array([[1.0, 0.0]], dtype="float32")},
    )
    router = SafetensorsGateRouter(str(path), FakeMoELayout(), top_k=1)

    hidden = np.array([[1.0]], dtype="float32")
    first, _ = router.route(0, hidden)

    # Overwrite the file so a re-read would prefer the other expert.
    save_file(
        {"layers.0.gate.weight": np.array([[0.0, 1.0]], dtype="float32")},
        str(path),
    )
    second, _ = router.route(0, hidden)

    assert first == second  # cached gate is still the original


def test_different_hidden_chooses_different_experts(tmp_path):
    # Each input dimension routes to its matching expert.
    gate = np.eye(4, 4, dtype="float32")
    path = make_safetensors_file(tmp_path, {"layers.0.gate.weight": gate})
    router = SafetensorsGateRouter(str(path), FakeMoELayout(), top_k=1)

    a = np.array([[1, 0, 0, 0]], dtype="float32")
    b = np.array([[0, 0, 1, 0]], dtype="float32")
    selected_a, _ = router.route(0, a)
    selected_b, _ = router.route(0, b)
    assert selected_a == [0]
    assert selected_b == [2]


def test_missing_gate_raises(tmp_path):
    # Asking for a layer whose gate isn't in the file should error.
    path = make_safetensors_file(
        tmp_path,
        {"layers.0.gate.weight": np.zeros((1, 1), dtype="float32")},
    )
    router = SafetensorsGateRouter(str(path), FakeMoELayout(), top_k=1)
    hidden = np.array([[1.0]], dtype="float32")

    with pytest.raises(Exception):
        router.route(99, hidden)
