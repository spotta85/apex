"""Tests for backend.prepare_hidden / combine_outputs / to_numpy.

These three methods make the runtime backend-array-type-agnostic. With
them in place, MLXBackend can keep `hidden` as mx.array across layers
while FakeBackend keeps it as ndarray, and the runtime doesn't have to
branch on backend type.
"""

import numpy as np
import pytest

from apex.backends.fake_backend import FakeBackend

mx = pytest.importorskip("mlx.core")
from apex.backends.mlx_backend import MLXBackend
from tests._world import write_npy_world


def test_fake_prepare_hidden_is_passthrough():
    backend = FakeBackend()
    arr = np.ones((1, 8), dtype="float32")
    out = backend.prepare_hidden(arr)
    # NumPy backend doesn't transform — should be the same object.
    assert out is arr


def test_mlx_prepare_hidden_returns_mx_array():
    backend = MLXBackend()
    arr = np.ones((1, 8), dtype="float32")
    out = backend.prepare_hidden(arr)
    assert isinstance(out, mx.array)


def test_fake_combine_outputs_averages_ndarrays():
    backend = FakeBackend()
    a = np.full((1, 4), 2.0, dtype="float32")
    b = np.full((1, 4), 4.0, dtype="float32")
    avg = backend.combine_outputs([a, b])
    np.testing.assert_array_equal(avg, np.full((1, 4), 3.0, dtype="float32"))


def test_mlx_combine_outputs_averages_mx_arrays():
    backend = MLXBackend()
    a = mx.array(np.full((1, 4), 2.0, dtype="float32"))
    b = mx.array(np.full((1, 4), 4.0, dtype="float32"))
    avg = backend.combine_outputs([a, b])
    # Result should still be an mx.array (no implicit numpy conversion).
    assert isinstance(avg, mx.array)
    np.testing.assert_allclose(np.array(avg), np.full((1, 4), 3.0, dtype="float32"))


def test_fake_to_numpy_is_passthrough():
    backend = FakeBackend()
    arr = np.ones((1, 8), dtype="float32")
    out = backend.to_numpy(arr)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, arr)


def test_mlx_to_numpy_converts():
    backend = MLXBackend()
    arr = mx.array(np.full((1, 4), 5.0, dtype="float32"))
    out = backend.to_numpy(arr)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, np.full((1, 4), 5.0, dtype="float32"))


def test_runtime_hidden_stays_mx_array_under_mlx(tmp_path):
    # Drop a recording backend in front of MLXBackend that captures
    # what type each run_expert call sees. Under MLXBackend the runtime
    # should hand mx.array hidden every time, and the bundle's
    # projections should also be mx.array (prepare_expert converted
    # them once on cache miss). An ndarray sneaking through either
    # would mean a round trip leaked in.
    from apex.expert_store import ExpertStore
    from apex.expert_cache import ExpertCache
    from apex.expert_manager import ExpertManager
    from apex.router import FakeRouter
    from apex.runtime import Runtime

    # Shared writer matches what ExpertStore loads now (3 projections
    # per expert as an ExpertWeights bundle); FakeRouter ignores gates.
    write_npy_world(tmp_path, with_gates=False)

    inner = MLXBackend()
    seen_hidden_types = []
    seen_expert_types = []
    # Track projection type separately so we can keep the original
    # spirit of the test: no ndarray leaks into MLX expert math.
    seen_projection_types = []

    class RecordingBackend:
        name = inner.name

        def prepare_expert(self, x):
            return inner.prepare_expert(x)

        def prepare_hidden(self, x):
            return inner.prepare_hidden(x)

        def run_expert(self, hidden, expert):
            seen_hidden_types.append(type(hidden).__name__)
            seen_expert_types.append(type(expert).__name__)
            seen_projection_types.append(type(expert.gate_proj).__name__)
            return inner.run_expert(hidden, expert)

        def combine_outputs(self, outs):
            return inner.combine_outputs(outs)

        def to_numpy(self, x):
            return inner.to_numpy(x)

    backend = RecordingBackend()
    cache = ExpertCache(max_memory_mb=1, verbose=False)
    manager = ExpertManager(ExpertStore(tmp_path), cache, backend=backend)
    runtime = Runtime(router=FakeRouter(), expert_manager=manager, backend=backend)
    runtime.run()

    # Hidden state is still a bare mx.array between layers (no bundle
    # wrapping); experts arrive as ExpertWeights with mx.array projections.
    assert seen_hidden_types == ["array", "array", "array", "array"]
    assert seen_expert_types == ["ExpertWeights"] * 4
    assert seen_projection_types == ["array"] * 4
