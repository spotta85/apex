"""Tests for MLXGateRouter.

Routing math runs through MLX (mx.matmul + mx.argsort), gate weights
are cached as mx.array. Output (top-k expert IDs) must match the
SafetensorsGateRouter's numpy result for the same gate weights.
"""

import numpy as np
import pytest
from safetensors.numpy import save_file

mx = pytest.importorskip("mlx.core")

from apex.mlx_gate_router import MLXGateRouter
from apex.safetensors_gate_router import SafetensorsGateRouter
from apex.weight_layout import FakeMoELayout


def write_gate(tmp_path, gate, layer_id=0):
    path = tmp_path / "x.safetensors"
    save_file({f"layers.{layer_id}.gate.weight": gate.astype("float32")}, str(path))
    return path


def test_top_k_matches_safetensors_router(tmp_path):
    # Same gate, same hidden -> same selected experts.
    rng = np.random.default_rng(42)
    gate = rng.standard_normal((8, 4)).astype("float32")
    path = write_gate(tmp_path, gate)
    hidden = np.ones((1, 8), dtype="float32")

    np_router = SafetensorsGateRouter(str(path), FakeMoELayout(), top_k=2)
    mlx_router = MLXGateRouter(str(path), FakeMoELayout(), top_k=2)

    assert mlx_router.route(0, hidden) == np_router.route(0, hidden)


def test_gate_cached_as_mx_array(tmp_path):
    # First route() call should populate the per-layer cache with an
    # mx.array, not the raw numpy gate.
    gate = np.eye(4, 4, dtype="float32")
    path = write_gate(tmp_path, gate)

    router = MLXGateRouter(str(path), FakeMoELayout(), top_k=1)
    router.route(0, np.ones((1, 4), dtype="float32"))

    assert 0 in router.gates
    assert isinstance(router.gates[0], mx.array)


def test_accepts_mx_array_hidden(tmp_path):
    # Under MLX runtime the hidden state IS an mx.array. The router
    # should consume it directly, without forcing a numpy round trip.
    gate = np.eye(4, 4, dtype="float32")
    path = write_gate(tmp_path, gate)

    router = MLXGateRouter(str(path), FakeMoELayout(), top_k=1)
    hidden_mx = mx.array(np.array([[1, 0, 0, 0]], dtype="float32"))
    assert router.route(0, hidden_mx) == [0]


def test_accepts_ndarray_hidden(tmp_path):
    # Numpy hidden also works — keeps the router callable from the
    # numpy runtime path or from tests.
    gate = np.eye(4, 4, dtype="float32")
    path = write_gate(tmp_path, gate)

    router = MLXGateRouter(str(path), FakeMoELayout(), top_k=1)
    hidden_np = np.array([[0, 0, 1, 0]], dtype="float32")
    assert router.route(0, hidden_np) == [2]


def test_returns_python_ints(tmp_path):
    # Cache keys / JSON serialization expect plain Python ints.
    gate = np.eye(4, 4, dtype="float32")
    path = write_gate(tmp_path, gate)

    router = MLXGateRouter(str(path), FakeMoELayout(), top_k=2)
    for expert_id in router.route(0, np.ones((1, 4), dtype="float32")):
        assert type(expert_id) is int


def test_router_has_name_mlx():
    # Runtime prints "router backend: <name>" — MLXGateRouter declares
    # "mlx" so the trace says so.
    assert MLXGateRouter.name == "mlx"
